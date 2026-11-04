#!/usr/bin/env python3
"""Independent, read-only audit of a streamed task-profile derivative.

The oracle imports neither the profile generator nor its window/motion helpers.
Sparse range queries enumerate maximal stable windows; significant-step groups
locate motions. Stored events, fields, scope and fixed-timing equations are then
checked against original rows. This does not establish visual grasp semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

EPS = 1e-9
TARGET = "evaluation.gripper_target"
MASK = "evaluation.gripper_rule_applied"
PHASE = "evaluation.gripper_phase"
SOURCE = "evaluation.gripper_source"
ANNOTATION = "annotation.gripper_target"
VALID = "annotation.gripper_target_valid"
EXTRA = (TARGET, MASK, PHASE, SOURCE, ANNOTATION, VALID)
WORKSPACE = Path(__file__).resolve().parents[1]
_PARALLEL_CATALOG = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def matching_bands_oracle(profile, margin):
    """Group anchors separated by blockers, then clip each padded envelope.

    This oracle works from anchor adjacency rather than the generator's
    complement-of-blockers construction and imports no runtime band helper.
    """
    if not profile["automatic_use"]:
        return []
    blockers = [b for key in ("free_bands", "interval_vision_bands", "open_bands")
                for b in profile.get(key, [])]
    anchors = sorted(profile["closed_bands"], key=lambda b: (b["lower"], b["upper"], b["id"]))
    groups = []
    for band in anchors:
        if groups and not any(groups[-1][-1]["upper"] < b["upper"]
                              and b["lower"] < band["lower"] for b in blockers):
            groups[-1].append(band)
        else:
            groups.append([band])
    output = []
    for group in groups:
        low, high = group[0]["lower"], max(b["upper"] for b in group)
        left = max([0.] + [math.nextafter(b["upper"], math.inf) for b in blockers if b["upper"] < low])
        right = min([1.] + [math.nextafter(b["lower"], -math.inf) for b in blockers if b["lower"] > high])
        lo, hi = max(left, low-margin), min(right, high+margin)
        output.append(dict(id=f"matching_{len(output):02d}", lower=lo, upper=hi, center=(lo+hi)/2,
                           basis_closed_band_ids=[b["id"] for b in group], pointwise_semantics_verified=False))
    return output


class Audit:
    def __init__(self, failure_limit=200):
        self.checks = Counter()
        self.failures = []
        self.failure_count = 0
        self.failure_limit = failure_limit

    def check(self, condition, kind, detail=None):
        self.checks[kind] += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < self.failure_limit:
                self.failures.append(dict(check=kind, detail=detail))
        return bool(condition)

    def report(self):
        return dict(status="FAIL" if self.failure_count else "PASS",
                    checks=dict(self.checks), failure_count=self.failure_count,
                    failures=self.failures,
                    failures_truncated=self.failure_count > len(self.failures))

    def merge(self, report):
        self.checks.update(report["checks"])
        self.failure_count += report["failure_count"]
        self.failures.extend(report["failures"][:max(0, self.failure_limit - len(self.failures))])


def maximal_windows(times, values, band, gap, span):
    """Independent O(n log n) oracle; no shared monotone deque implementation."""
    n = len(values)
    if not n:
        return []
    lower, upper = [list(values)], [list(values)]
    width = 1
    while width * 2 <= n:
        lower.append([min(lower[-1][i], lower[-1][i + width])
                      for i in range(n - width * 2 + 1)])
        upper.append([max(upper[-1][i], upper[-1][i + width])
                      for i in range(n - width * 2 + 1)])
        width *= 2

    def stable(a, b):
        level = (b - a + 1).bit_length() - 1
        last = b - (1 << level) + 1
        return (max(upper[level][a], upper[level][last])
                - min(lower[level][a], lower[level][last]) <= span + EPS)

    inside = [band["lower"] <= value <= band["upper"] for value in values]
    stop = list(range(n))
    for i in range(n - 2, -1, -1):
        if inside[i] and inside[i + 1] and 0 < times[i + 1] - times[i] <= gap + EPS:
            stop[i] = stop[i + 1]
    result = []
    for start in range(n):
        if not inside[start]:
            continue
        left, right = start, stop[start]
        while left < right:
            midpoint = (left + right + 1) // 2
            if stable(start, midpoint):
                left = midpoint
            else:
                right = midpoint - 1
        if (start and inside[start - 1]
                and 0 < times[start] - times[start - 1] <= gap + EPS
                and stable(start - 1, left)):
            continue
        result.append((start, left))
    return result


def motions(values, times, start, end, config, threshold=None):
    """Group significant increments first, independently of generator state."""
    significant = [(i, 1 if values[i] > values[i - 1] else -1)
                   for i in range(start + 1, end + 1)
                   if abs(values[i] - values[i - 1]) > config["motion_noise"]]
    groups = []
    for index, sign in significant:
        if (groups and groups[-1][0][1] == sign
                and times[index - 1] - times[groups[-1][-1][0]] <= config["motion_gap_s"] + EPS):
            groups[-1].append((index, sign))
        else:
            groups.append([(index, sign)])
    minimum = config["motion_min_delta"] if threshold is None else threshold
    return [dict(start=g[0][0] - 1, end=g[-1][0], direction=g[0][1])
            for g in groups if abs(values[g[-1][0]] - values[g[0][0] - 1]) >= minimum]


def candidates_oracle(times, values, profile, catalog):
    blocks, start = [], 0
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > catalog["maximum_sample_gap_s"] + EPS:
            blocks.append((start, index - 1))
            start = index
    blocks.append((start, len(times) - 1))
    boundaries = {i: block for block in blocks for i in range(block[0], block[1] + 1)}
    output = []
    for band, needs_vision in ([(b, False) for b in profile["closed_bands"]]
                               + [(b, True) for b in profile.get("interval_vision_bands", [])]):
        groups = []
        for a, b in maximal_windows(times, values, band, catalog["maximum_sample_gap_s"],
                                    catalog["maximum_plateau_range"]):
            block = boundaries[a]
            if times[b] - times[a] + EPS < catalog["minimum_hold_s"]:
                output.append(dict(closed_band_id=band["id"], start=a, end=b, block_start=block[0],
                    block_end=block[1], matched=False, matched_windows=[],
                    qualification_reason="insufficient_holding_duration"))
                continue
            confirm = next(i for i in range(a, b + 1)
                           if times[i] - times[a] + EPS >= catalog["minimum_hold_s"])
            window = dict(start=a, end=b, confirmation_frame=confirm, confirmation_time=times[confirm],
                          duration_s=times[b] - times[a], observed_lower=min(values[a:b + 1]),
                          observed_upper=max(values[a:b + 1]))
            if groups and boundaries[groups[-1][0]["start"]] == block and a <= groups[-1][-1]["end"] + 1:
                groups[-1].append(window)
            else:
                groups.append([window])
        for group in groups:
            a, b = group[0]["start"], max(w["end"] for w in group)
            block = boundaries[a]
            output.append(dict(closed_band_id=band["id"], start=a, end=b, block_start=block[0],
                block_end=block[1], matched=not needs_vision, matched_windows=group,
                qualification_reason="requires_interval_vision" if needs_vision else "task_profile_numeric_match"))
    return sorted(output, key=lambda c: (c["start"], c["end"], c["closed_band_id"]))


def fit_oracle(times, values, candidate, profile, catalog, config):
    """Derive event boundaries, rejection reason and fixed-time target equations."""
    if not candidate["matched"]:
        return None, candidate["qualification_reason"]
    a, b = candidate["start"], candidate["end"]
    band = next(x for x in profile["closed_bands"] if x["id"] == candidate["closed_band_id"])
    observed = motions(values, times, candidate["block_start"], candidate["block_end"], config)
    closing = sorted((m for m in observed if m["direction"] < 0 and m["start"] < a <= m["end"] <= b
                      and times[m["end"]] - times[a] <= config["boundary_tolerance_s"] + EPS),
                     key=lambda m: (m["end"], -m["start"]))
    if closing:
        cs, contact = closing[0]["start"], closing[0]["end"]
    elif a == candidate["block_start"]:
        cs, contact = None, a
    else:
        return None, "missing_aligned_closing_boundary"
    references = {o["id"]: o for o in profile["open_bands"]}
    reference_ids = band.get("open_band_ids", [o["id"] for o in references.values() if o["lower"] > band["upper"]])
    openings = [(m, reference_ids, None) for m in observed if m["direction"] > 0]
    for rule in profile.get("release_rules", []):
        if rule["closed_band_id"] == band["id"]:
            openings.extend((m, rule["open_band_ids"], rule) for m in
                motions(values, times, candidate["block_start"], candidate["block_end"], config, rule["motion_min_delta"])
                if m["direction"] > 0)
    eligible = [(m, min(references[i]["lower"] for i in ids), rule) for m, ids, rule in openings
                if contact <= m["start"] <= b < m["end"]
                and values[m["end"]] + EPS >= min(references[i]["lower"] for i in ids)]
    opening = min(eligible, key=lambda x: (x[0]["start"], x[0]["end"]), default=None)
    hold_end = opening[0]["start"] if opening else b
    if times[hold_end] - times[contact] + EPS < catalog["minimum_hold_s"]:
        return None, "insufficient_holding_duration_after_boundary_alignment"
    confirm = next(i for i in range(contact, hold_end + 1) if times[i] - times[contact] + EPS >= catalog["minimum_hold_s"])
    confirmed = values[contact:confirm + 1]
    if (not all(band["lower"] <= value <= band["upper"] for value in confirmed)
            or max(confirmed) - min(confirmed) > catalog["maximum_plateau_range"] + EPS):
        return None, "unstable_confirmation_after_boundary_alignment"
    if not opening and b < candidate["block_end"] and values[b + 1] > config["exit_tolerance"]:
        return None, "unresolved_release_boundary"
    for motion in ([closing[0]] if closing else []) + ([opening[0]] if opening else []):
        if abs(values[motion["end"]] - values[motion["start"]]) / (times[motion["end"]] - times[motion["start"]]) > config["maximum_motion_speed"]:
            return None, "unreliable_motion_speed"
    if cs is not None and values[cs] - values[contact] < config["motion_min_delta"]:
        return None, "insufficient_closing_excursion"
    start, end = cs if cs is not None else contact, opening[0]["end"] if opening else hold_end
    curve, phase = list(values), ["unchanged"] * len(values)
    if cs is not None:
        beginning, connection, running = values[cs], values[contact], 0.
        for i in range(cs, contact + 1):
            running = max(running, min(1., max(0., (beginning - values[i]) / (beginning - connection))))
            curve[i], phase[i] = beginning * (1 - running), "closing"
    for i in range(contact, hold_end + 1):
        curve[i], phase[i] = 0., "hold"
    if opening:
        rs, re = opening[0]["start"], opening[0]["end"]
        beginning, endpoint, running = values[rs], values[re], 0.
        for i in range(rs, re + 1):
            running = max(running, min(1., max(0., (values[i] - beginning) / (endpoint - beginning))))
            curve[i], phase[i] = endpoint * running, "release"
    else:
        rs = re = None
    rule = opening[2] if opening else None
    return dict(curve=curve, phases=phase, modified_start=start, modified_end=end,
        closing_start=cs, contact_frame=contact, hold_start=contact, hold_end=hold_end,
        zero_frame=contact, release_start=rs, release_end=re, confirmation_frame=confirm,
        confirmation_time=times[confirm], initial_or_cropped_hold=cs is None,
        open_reference_lower=opening[1] if opening else min(references[i]["lower"] for i in reference_ids),
        release_rule_id=rule["id"] if rule else None,
        release_motion_min_delta=(rule["motion_min_delta"] if rule else config["motion_min_delta"]) if opening else None,
        cropped_start_by_timestamp_gap=cs is None and candidate["block_start"] > 0,
        cropped_end_by_timestamp_gap=not opening and end == candidate["block_end"] and end < len(values) - 1,
        changed_frames=sum(abs(curve[i] - values[i]) > 1e-12 for i in range(start, end + 1))), None


def fit_hold_first_oracle(times, values, candidate, config):
    """Independently reconstruct the qualified hold and its outer transitions."""
    if not candidate["matched"]:
        return None, candidate["qualification_reason"]
    a, b = candidate["start"], candidate["end"]
    threshold = 2 * config["motion_noise"] + EPS
    observed = motions(values, times, candidate["block_start"], candidate["block_end"], config, threshold)
    closing = min((m for m in observed if m["direction"] == -1 and m["start"] < a <= m["end"] <= b),
                  key=lambda m: (m["end"], -m["start"]), default=None)
    opening = min((m for m in observed if m["direction"] == 1 and a <= m["start"] <= b < m["end"]),
                  key=lambda m: (m["start"], m["end"]), default=None)
    cs, re = closing["start"] if closing else None, opening["end"] if opening else None
    issues = []
    if cs is not None:
        delta = values[cs] - values[a]
        if delta <= EPS:
            issues.append("closing:nonpositive_excursion"); cs = None
        elif delta / (times[a] - times[cs]) > config["maximum_motion_speed"]:
            issues.append("closing:unreliable_motion_speed"); cs = None
    if re is not None:
        delta = values[re] - values[b]
        if delta <= EPS:
            issues.append("release:nonpositive_excursion"); re = None
        elif delta / (times[re] - times[b]) > config["maximum_motion_speed"]:
            issues.append("release:unreliable_motion_speed"); re = None
    if cs is None and a > candidate["block_start"] and not any(s.startswith("closing:") for s in issues):
        issues.append("closing:no_observed_boundary")
    if re is None and b < candidate["block_end"] and not any(s.startswith("release:") for s in issues):
        issues.append("release:no_observed_boundary")
    start, end = a if cs is None else cs, b if re is None else re
    curve, phase = values.copy(), ["unchanged"] * len(values)
    if cs is not None:
        fractions = [min(1., max(0., (values[cs] - x) / (values[cs] - values[a]))) for x in values[cs:a + 1]]
        running = 0.
        for i, fraction in enumerate(fractions, cs):
            running = max(running, fraction)
            curve[i], phase[i] = values[cs] * (1. - running), "closing"
    if re is not None:
        fractions = [min(1., max(0., (x - values[b]) / (values[re] - values[b]))) for x in values[b:re + 1]]
        running = 0.
        for i, fraction in enumerate(fractions, b):
            running = max(running, fraction)
            curve[i], phase[i] = values[re] * running, "release"
    curve[a:b + 1], phase[a:b + 1] = [0.] * (b-a+1), ["hold"] * (b-a+1)
    confirm = next(i for i in range(a, b + 1) if times[i] - times[a] + EPS >= config["minimum_hold_s"])
    return dict(curve=curve, phases=phase, modified_start=start, modified_end=end,
        closing_start=cs, contact_frame=a, hold_start=a, hold_end=b, zero_frame=a,
        release_start=b if re is not None else None, release_end=re,
        confirmation_frame=confirm, confirmation_time=times[confirm],
        initial_or_cropped_hold=a == candidate["block_start"],
        open_reference_lower=None, release_rule_id=None,
        release_motion_min_delta=threshold if re is not None else None,
        release_reference_gate_applied=False, transition_issues=issues,
        observed_closing_end=closing["end"] if cs is not None else None,
        observed_release_start=opening["start"] if re is not None else None,
        cropped_start_by_timestamp_gap=a == candidate["block_start"] and candidate["block_start"] > 0,
        cropped_end_by_timestamp_gap=b == candidate["block_end"] and b < len(values) - 1), None


def source_excluded(catalog, episode):
    return (episode["source_id"] in catalog.get("excluded_source_ids", [])
            or any(item["dataset_root"] == episode["dataset_root"]
                   and item["episode_index"] == episode["episode_index"]
                   for item in catalog.get("excluded_episode_identities", [])))


def verify_episode(episode, arrays, metadata, candidates, events, catalog, audit):
    """Check one original episode and its saved derivative/audit records."""
    check, identity = audit.check, episode["episode_id"]
    times, state, task = episode["timestamps"], episode["state"], episode["task"]
    n = len(times)
    if not check(n > 0 and len(state) == n and all(len(row) == 20 for row in state)
                 and all(math.isfinite(t) for t in times)
                 and all(b > a for a, b in zip(times, times[1:])), "original_episode_grid", identity):
        return Counter()
    original = [[r[9], r[19]] for r in state]
    if not check(all(len(arrays[k]) == n and all(len(r) == 2 for r in arrays[k])
                     for k in (TARGET, MASK, PHASE, SOURCE)), "evaluation_array_shapes", identity):
        return Counter()
    check(metadata.get("evaluation_only") is True and metadata.get("grasp_semantics_verified") is False
          and metadata.get("training_ready") is False and metadata.get("model_requests") == 0,
          "episode_no_visual_or_training_promotion", identity)
    version = metadata.get("rule_version")
    check(version in ("task-profile-rule-v2", "task-profile-rule-v3-hold-first"), "supported_rule_version", identity)
    hold_first = version == "task-profile-rule-v3-hold-first"
    if hold_first and episode["selected"]:
        check(metadata.get("hold_policy") == "stable_task_band_only"
              and metadata.get("overlap_policy") == "minimum_with_qualified_zero_holds", "hold_first_policy", identity)
    profiles = next(t["sides"] for t in catalog["tasks"] if t["task"] == task)
    selected = any(p["automatic_use"] for p in profiles.values())
    check(episode["selected"] is selected, "episode_selected_task_scope", identity)
    actual = {(c["side"], c["closed_band_id"], c["start"], c["end"]): c for c in candidates}
    event_map = {e["candidate_id"]: e for e in events}
    check(len(actual) == len(candidates) and len(event_map) == len(events)
          and set(event_map) == {c["candidate_id"] for c in candidates},
          "candidate_event_bijection", identity)
    expected_candidates, proposals = {}, []
    expected_curve = [list(r) for r in original]
    expected_mask = [[False, False] for _ in range(n)]
    expected_phase = [["unchanged", "unchanged"] for _ in range(n)]
    for col, side in enumerate(("left", "right")):
        coverage = metadata["coverage"][side]
        valid = side in episode["valid_sides"]
        profile = profiles.get(side)
        reason = ("invalid_side" if not valid else "scope_not_selected" if not selected
                  else "source_alignment_uncertain" if source_excluded(catalog, episode)
                  else "missing_side_profile" if profile is None
                  else "profile_not_automatic" if not profile["automatic_use"] and not profile.get("interval_vision_bands")
                  else None)
        if reason:
            check(coverage.get("reason") == reason and coverage.get("complete_numeric_scan") is False,
                  "side_fallback_scope", [identity, side, reason])
            continue
        config = metadata["config"]
        check(config["minimum_hold_s"] == catalog["minimum_hold_s"], "hold_config_matches_catalog", identity)
        check(config["closing_policy"] == config["release_policy"] == "fixed_timing",
              "fixed_timing_config", identity)
        profile = profile if profile["automatic_use"] else dict(profile, closed_bands=[])
        if catalog.get("matching_policy"):
            policy = catalog["matching_policy"]
            expected_bands = profile.get("matching_bands", [])
            check(coverage.get("matching_policy") == policy and coverage.get("matching_bands") == [
                {k:b[k] for k in ("id", "lower", "upper", "basis_closed_band_ids")} for b in expected_bands],
                "runtime_generalized_matching_policy", [identity, side])
            profile = dict(profile, closed_bands=expected_bands)
        values = [row[col] for row in original]
        found = candidates_oracle(times, values, profile, catalog)
        check(coverage.get("complete_numeric_scan") is True
              and coverage.get("candidates") == len(found)
              and coverage.get("matched_windows") == sum(len(c["matched_windows"]) for c in found)
              and coverage.get("matched_intervals") == sum(c["matched"] for c in found),
              "side_complete_numeric_inventory", [identity, side])
        check(coverage.get("complete_review") is False and coverage.get("reviewed_frames") == 0
              and coverage.get("target_valid_frames") == 0 and coverage.get("accepted_cycles") == 0,
              "side_no_fake_visual_coverage", [identity, side])
        for candidate in found:
            key = (side, candidate["closed_band_id"], candidate["start"], candidate["end"])
            expected_candidates[key] = candidate
            fit, refusal = (fit_hold_first_oracle(times, values, candidate, config) if hold_first else
                            fit_oracle(times, values, candidate, profile, catalog, config))
            proposals.append((key, fit, refusal, col))
    check(set(expected_candidates) == set(actual), "exhaustive_candidate_inventory", identity)
    conflicts = set()
    for i, (_, one, _, col) in enumerate(proposals):
        if hold_first or one is None:
            continue
        for j, (_, two, _, other_col) in enumerate(proposals[:i]):
            if two is None or other_col != col:
                continue
            a, b = max(one["modified_start"], two["modified_start"]), min(one["modified_end"], two["modified_end"])
            if a <= b and any(abs(one["curve"][k] - two["curve"][k]) > 1e-12 for k in range(a, b + 1)):
                conflicts.update((i, j))
    applied, rejected = 0, 0
    for i, (key, fit, refusal, col) in enumerate(proposals):
        candidate = expected_candidates[key]
        if key not in actual:
            continue
        saved = actual[key]
        detail = [identity, *key]
        check(all(saved.get(k) == v for k, v in candidate.items()), "candidate_numerical_witness", detail)
        wanted_id = f"{identity}:{key[0]}:profile:{key[1]}:{key[2]}:{key[3]}"
        check(saved["candidate_id"] == wanted_id, "candidate_identity", detail)
        event = event_map.get(saved["candidate_id"])
        if event is None:
            continue
        check(event.get("source") == "task_profile_rule" and event.get("reviewed") is False
              and event.get("evaluation_only") is True and event.get("grasp_semantics_verified") is False,
              "event_provenance", detail)
        check(all(event.get(k) == v for k, v in candidate.items()), "event_candidate_witness", detail)
        refusal = "overlapping_incompatible_profile_cycles" if i in conflicts else refusal
        if refusal:
            rejected += 1
            check(event.get("status") == "unchanged" and event.get("reason") == refusal
                  and event.get("modified_start") is None and event.get("modified_end") is None
                  and event.get("changed_frames") == 0, "independent_rejection_reason", [*detail, refusal])
            continue
        applied += 1
        check(event.get("status") == "applied" and event.get("reason") == "task_profile_rule_evaluation",
              "independent_event_applied", detail)
        for field, value in fit.items():
            if field not in ("curve", "phases"):
                check(event.get(field) == value, "fixed_timing_event_boundaries", [*detail, field])
        check(event.get("closing_policy") == event.get("release_policy") == "fixed_timing",
              "fixed_timing_event_policies", detail)
        for frame in range(fit["modified_start"], fit["modified_end"] + 1):
            value = fit["curve"][frame]
            if (not hold_first or not expected_mask[frame][col] or value < expected_curve[frame][col]
                    or value == expected_curve[frame][col] and fit["phases"][frame] == "hold"):
                expected_curve[frame][col] = value
                expected_phase[frame][col] = fit["phases"][frame]
            expected_mask[frame][col] = True
    if hold_first:
        for key, fit, _, col in proposals:
            if fit is None or key not in actual:
                continue
            event = event_map.get(actual[key]["candidate_id"], {})
            check(event.get("changed_frames") == sum(abs(expected_curve[i][col] - original[i][col]) > 1e-12
                  for i in range(fit["modified_start"], fit["modified_end"] + 1)), "event_composed_changed_frames", [identity, key])
            a, b = key[2], key[3]
            check(all(arrays[TARGET][i][col] == 0. and arrays[MASK][i][col]
                      for i in range(a, b + 1)), "every_qualified_hold_zero", [identity, key])
    check(arrays[MASK] == expected_mask, "mask_equals_all_applied_intervals", identity)
    check(arrays[PHASE] == expected_phase, "phase_matches_fixed_timing", identity)
    check(all(type(flag) is bool for row in arrays[MASK] for flag in row), "mask_boolean", identity)
    check(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
              for row in arrays[TARGET] for value in row), "evaluation_normalized", identity)
    check(all(abs(arrays[TARGET][i][col] - expected_curve[i][col]) <= 1e-12
              for i in range(n) for col in range(2)), "independent_complete_curve", identity)
    check(arrays[SOURCE] == [["task_profile_rule" if flag else "" for flag in row] for row in expected_mask],
          "source_mask_consistency", identity)
    check(all(arrays[TARGET][i][col] == original[i][col] for i in range(n) for col in range(2)
              if not expected_mask[i][col]), "unapplied_and_invalid_original", identity)
    changed = sum(any(abs(expected_curve[i][col] - original[i][col]) > 1e-12 for col in range(2)) for i in range(n))
    return Counter(episodes=1, frames=n, selected_episodes=int(selected), selected_frames=n * selected,
                   changed_episodes=int(changed > 0), changed_frames=changed,
                   candidates=len(expected_candidates), applied_intervals=applied, rejected_intervals=rejected,
                   applied_frames=sum(any(row) for row in expected_mask))


def verify_original_columns(original, generated, audit, detail):
    """Arrow preserves nested values, nulls, column ordering and field metadata."""
    import pyarrow as pa
    wanted = original.column_names + [k for k in EXTRA if k not in original.column_names]
    audit.check(not set((TARGET, MASK, PHASE, SOURCE)) & set(original.column_names), "no_existing_evaluation_collision", detail)
    if not audit.check(generated.column_names == wanted, "exact_column_schema_order", detail):
        return False
    audit.check(generated.num_rows == original.num_rows, "all_rows_preserved", detail)
    audit.check(generated.select(original.column_names).equals(original, check_metadata=True), "every_original_column_identical", detail)
    for key, typ in ((TARGET, pa.float64()), (MASK, pa.bool_()), (PHASE, pa.string()),
                     (SOURCE, pa.string()), (ANNOTATION, pa.float32()), (VALID, pa.bool_())):
        if key not in original.column_names:
            audit.check(generated.schema.field(key).type == pa.list_(typ, 2), "new_column_dtype", [detail, key])
    if ANNOTATION not in original.column_names:
        expected = pa.array([[r[9], r[19]] for r in original["observation.state"].to_pylist()], type=pa.list_(pa.float32(), 2))
        audit.check(generated[ANNOTATION].combine_chunks().equals(expected), "annotation_original_fallback", detail)
    if VALID not in original.column_names:
        audit.check(generated[VALID].to_pylist() == [[False, False]] * original.num_rows, "annotation_valid_false", detail)
    return generated.num_rows == original.num_rows


def below(root, relative):
    """Data and audit files must be real files inside the declared directory."""
    root = Path(root).resolve()
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("unsafe relative path: " + str(relative))
    candidate = root / part
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root and p.is_relative_to(root)):
        raise ValueError("symlink in data/audit path: " + str(candidate))
    candidate.resolve().relative_to(root)
    return candidate


def json_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_catalog(catalog, audit):
    enabled = {t["task"] for t in catalog["tasks"] if any(p["automatic_use"] for p in t["sides"].values())}
    audit.check(catalog["minimum_hold_s"] == .6 and catalog["maximum_sample_gap_s"] == .15
                and catalog["maximum_plateau_range"] == .02, "frozen_hold_policy")
    count = 0
    policy = catalog.get("matching_policy")
    if policy is not None:
        margin = policy.get("expansion_margin")
        audit.check(set(policy) == {"mode", "expansion_margin", "pointwise_semantics_verified"}
                    and policy.get("mode") == "merge_expand_excluding_free_v1"
                    and policy.get("pointwise_semantics_verified") is False
                    and type(margin) in (int, float) and math.isfinite(margin) and 0 <= margin <= .05,
                    "declared_generalization_not_new_visual_evidence")
    excluded = set(catalog.get("excluded_source_ids", []))
    for task in catalog["tasks"]:
        for side, profile in task["sides"].items():
            if policy is not None:
                expected = matching_bands_oracle(profile, policy["expansion_margin"])
                audit.check(profile.get("matching_bands", []) == expected,
                            "independent_merged_padded_ranges", [task["task"], side])
                blockers = [b for key in ("free_bands", "interval_vision_bands", "open_bands")
                            for b in profile.get(key, [])]
                for band in profile.get("matching_bands", []):
                    audit.check(not any(max(band["lower"], b["lower"]) <= min(band["upper"], b["upper"])
                                        for b in blockers), "matching_range_excludes_free_and_conflicts", [task["task"], side, band["id"]])
            spans = [(e["source_id"], max(0., h["lower"] - catalog["evidence_noise"]),
                      min(1., h["upper"] + catalog["evidence_noise"]))
                     for e in profile.get("semantic_evidence", []) if e["numeric_calibration_eligible"]
                     for h in e["held_intervals"] if h["qualifies_hold"]]
            for band in profile["closed_bands"]:
                count += 1
                detail = [task["task"], side, band["id"]]
                audit.check(len(set(band["source_ids"])) >= 2 and not excluded.intersection(band["source_ids"]),
                            "closed_band_independent_nonexcluded_sources", detail)
                audit.check(not any(max(band["lower"], f["lower"]) <= min(band["upper"], f["upper"])
                                    for f in profile.get("free_bands", [])), "closed_band_no_free_overlap", detail)
                cuts = sorted({band["lower"], band["upper"]} | {x for _, lo, hi in spans for x in (lo, hi)
                                                                               if band["lower"] <= x <= band["upper"]})
                for value in cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]:
                    supporters = {source for source, lo, hi in spans if lo <= value <= hi and source not in excluded}
                    audit.check(len(supporters) >= 2, "every_closed_value_two_source_support", [*detail, value])
    return enabled, count


def _source_task(meta, catalog):
    labels = meta.get("source_tasks", meta.get("tasks"))
    if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
        raise ValueError("source metadata does not have a unique task")
    label = labels[0]
    if label in catalog.get("ambiguous_aliases", {}):
        raise ValueError("source metadata uses an ambiguous task alias")
    names = {t["task"] for t in catalog["tasks"] if label in [t["task"], *t.get("aliases", [])]}
    if len(names) != 1:
        raise ValueError("source metadata does not resolve to one catalog task")
    return names.pop()


def _valid_sides(info, mask):
    field = mask.get("field_mask", {}).get("observation.state", False)
    elements = mask.get("element_mask", {}).get("observation.state", [])
    if not field or len(elements) != 20 or "observation.state" not in info["features"]:
        raise ValueError("invalid original state mask")
    return [side for side, index in (("left", 9), ("right", 19)) if elements[index]]


def selection_contract(manifest):
    """Full scope is unsampled; partial diagnostics must disclose selection."""
    if manifest.get("complete_source_scope") is True:
        return manifest.get("sampling") is False and manifest.get("selection_uses_band_matching") is False
    sampling = manifest.get("sampling")
    return (manifest.get("complete_source_scope") is False
            and (sampling is True or isinstance(sampling, str) and bool(sampling.strip()))
            and type(manifest.get("selection_uses_band_matching")) is bool)


def verify_shard(run, dataset, shard, record, source_rows, meta_rows, catalog, audit):
    import pyarrow.parquet as pq
    root = Path(dataset["dataset_root"]).resolve()
    identity = [dataset["dataset_id"], shard["path"]]
    source = below(root, shard["path"])
    relative_output = Path("staging") / (dataset.get("release_directory", dataset["release"]) + "-gripcmd") / dataset["subset"] / shard["path"]
    destination = below(run, relative_output)
    episode_path = below(run, Path("audit") / dataset["dataset_id"] / "episodes" / shard["path"])
    event_path = below(run, Path("audit") / dataset["dataset_id"] / "events" / shard["path"])
    audit.check(record["source_path"] == str(source) and record["output_path"] == str(destination)
                and record["episode_audit_path"] == str(episode_path) and record["event_audit_path"] == str(event_path),
                "shard_exact_paths", identity)
    stat_before = source.stat()
    for path, expected, kind in ((source, shard["sha256"], "original_source_sha256"),
                                  (destination, record["output_sha256"], "output_sha256"),
                                  (episode_path, record["episode_audit_sha256"], "episode_audit_sha256"),
                                  (event_path, record["event_audit_sha256"], "event_audit_sha256")):
        audit.check(digest(path) == expected, kind, identity)
    audit.check(record["source_sha256"] == shard["sha256"] and record["status"] == "complete",
                "shard_source_binding", identity)
    original, generated = pq.read_table(source, use_threads=False), pq.read_table(destination, use_threads=False)
    audit.check(original.num_rows == shard["rows"] == record["rows"], "manifest_shard_rows", identity)
    if not verify_original_columns(original, generated, audit, identity):
        return {}, []
    episode_audits = pq.read_table(episode_path, use_threads=False).to_pylist()
    event_audits = pq.read_table(event_path, use_threads=False).to_pylist()
    saved = {r["episode_index"]: r for r in episode_audits}
    expected_indices = set(shard["episodes"])
    audit.check(len(saved) == len(episode_audits) == record["episodes"]
                and set(saved) == expected_indices, "episode_audit_complete_inventory", identity)
    ids = original["episode_index"].to_pylist()
    breaks = [0] + [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]] + [len(ids)]
    groups = [ids[i] for i in breaks[:-1]]
    audit.check(len(set(groups)) == len(groups) and set(groups) == expected_indices,
                "source_episode_groups", identity)
    audit.check(len(event_audits) == record["events"] and all(e["episode_index"] in expected_indices for e in event_audits),
                "event_audit_inventory", identity)
    events_by_episode = {}
    for row in event_audits:
        events_by_episode.setdefault(row["episode_index"], []).append(row)
    counts, episode_reports = {}, []
    for a, b in zip(breaks[:-1], breaks[1:]):
        index = ids[a]
        if index not in source_rows or index not in meta_rows or index not in saved:
            audit.check(False, "missing_bound_episode", [*identity, index])
            continue
        ledger, meta, entry = source_rows[index], meta_rows[index], saved[index]
        task = _source_task(meta, catalog)
        episode_id = f"{root.parent.name}/{root.name}/{index}"
        common = dict(dataset_id=dataset["dataset_id"], dataset_root=str(root), source_data_path=shard["path"],
                      source_sha256=shard["sha256"], episode_index=index, episode_id=episode_id,
                      task=task, source_id=ledger["source_id"])
        audit.check(all(entry.get(k) == value for k, value in common.items()), "episode_identity_binding", [*identity, index])
        audit.check(entry["source_path"] == ledger["source_path"] and Path(ledger["source_path"]).parent.name == ledger["source_id"],
                    "original_recording_ledger_binding", episode_id)
        audit.check(entry["row_start"] == a and entry["row_count"] == b - a == meta["length"]
                    and ledger["shard_row_start"] == a and ledger["shard_row_end_exclusive"] == b,
                    "episode_row_offsets", episode_id)
        audit.check(entry["dataset_from_index"] == meta["dataset_from_index"] == ledger["dataset_from_index"]
                    and entry["dataset_to_index"] == meta["dataset_to_index"] == ledger["dataset_to_index"],
                    "episode_global_offsets", episode_id)
        part = original.slice(a, b - a)
        audit.check(part["frame_index"].to_pylist() == list(range(b - a))
                    and part["index"].to_pylist() == list(range(meta["dataset_from_index"], meta["dataset_to_index"])),
                    "frame_and_global_indices", episode_id)
        audit.check(entry["valid_sides"] == dataset["valid_sides"] == ledger["valid_sides"], "episode_side_binding", episode_id)
        audit.check(entry["selected"] == ledger["selected"] and task == ledger["task"], "frozen_scope_ledger_binding", episode_id)
        episode = dict(episode_id=episode_id, episode_index=index, dataset_root=str(root), source_id=ledger["source_id"],
                       task=task, selected=entry["selected"], valid_sides=dataset["valid_sides"],
                       timestamps=part["timestamp"].to_pylist(), state=part["observation.state"].to_pylist())
        arrays = {key: generated.slice(a, b - a)[key].to_pylist() for key in (TARGET, MASK, PHASE, SOURCE)}
        candidates, events = [], []
        for row in events_by_episode.get(index, []):
            audit.check(all(row.get(k) == value for k, value in common.items()), "event_source_identity_binding", episode_id)
            candidate, event = json.loads(row["candidate_json"]), json.loads(row["event_json"])
            audit.check(all(row.get(k) == event.get(k) for k in ("side", "candidate_id", "status", "reason", "start", "end", "modified_start", "modified_end")),
                        "flattened_event_matches_json", episode_id)
            candidates.append(candidate); events.append(event)
        failed_before = audit.failure_count
        result = verify_episode(episode, arrays, json.loads(entry["result_metadata_json"]), candidates, events, catalog, audit)
        for key in ("candidates", "applied_intervals", "rejected_intervals", "applied_frames", "changed_frames"):
            audit.check(entry[key] == result.get(key, 0), "episode_recorded_totals", [episode_id, key])
        task_count = counts.setdefault(task, Counter())
        task_count.update({k: v for k, v in result.items() if k != "applied_frames"})
        episode_reports.append(dict(episode_id=episode_id, task=task, selected=entry["selected"],
                                    failures=audit.failure_count - failed_before, **result))
    for task, actual in counts.items():
        audit.check(dict(actual) == record["tasks"].get(task), "shard_task_totals", [*identity, task])
    audit.check(set(counts) == set(record["tasks"]), "shard_task_inventory", identity)
    stat_after = source.stat()
    audit.check(digest(source) == shard["sha256"]
                and (stat_before.st_size, stat_before.st_mtime_ns) == (stat_after.st_size, stat_after.st_mtime_ns)
                and stat_after.st_size == shard["bytes"] and stat_after.st_mtime_ns == shard["mtime_ns"],
                "source_rehashed_and_stat_unchanged", identity)
    return counts, episode_reports


def _initialize_audit_worker(catalog):
    global _PARALLEL_CATALOG
    _PARALLEL_CATALOG = catalog
    import pyarrow as pa
    pa.set_cpu_count(1)


def _audit_shard_job(job, catalog=None):
    """A worker owns its counters; only compact per-shard results cross IPC."""
    local = Audit()
    dataset, shard = job["dataset"], job["shard"]
    try:
        counts, rows = verify_shard(job["run"], dataset, shard, job["record"], job["sources"],
                                   job["metadata"], catalog if catalog is not None else _PARALLEL_CATALOG, local)
    except (KeyError, ValueError, TypeError, OSError, IndexError, StopIteration, ZeroDivisionError) as exc:
        local.check(False, "invalid_shard_contract", [dataset["dataset_id"], shard["path"], type(exc).__name__, str(exc)])
        counts, rows = {}, []
    return counts, rows, local.report()


def audit_bulk(run, manifest_path, output, *, workers=1):
    import pyarrow.parquet as pq
    run, manifest_path = Path(run).resolve(strict=True), Path(manifest_path).resolve(strict=True)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    output = Path(output).absolute()
    allowed = (WORKSPACE / "runs").resolve()
    if (not output.resolve().is_relative_to(allowed) or output.resolve() == allowed
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("audit output must be a new real directory inside workspace runs")
    if output.resolve().is_relative_to(run) or run.is_relative_to(output.resolve()):
        raise ValueError("audit output must be separate from the evaluated run")
    output.mkdir(parents=True, exist_ok=False)
    started, audit = time.monotonic(), Audit()
    totals, by_task, by_dataset = Counter(), {}, {}
    source_files, verified_episodes, bands, enabled = 0, 0, 0, set()
    try:
        manifest, frozen = read(manifest_path), read(run / "manifest.json")
        config, summary, catalog = read(run / "config.json"), read(run / "summary.json"), read(run / "task_profiles.json")
        audit.check(manifest == frozen and digest(manifest_path) == config["manifest_sha256"], "manifest_snapshot_exact")
        audit.check(digest(run / "task_profiles.json") == manifest["catalog_sha256"] == config["catalog_sha256"], "catalog_snapshot_exact")
        audit.check(summary["status"] == "complete" and not summary.get("failures"), "run_completed")
        audit.check(config["evaluation_only"] is True and config["grasp_semantics_verified"] is False
                    and config["training_ready"] is False and config["model_requests"] == 0
                    and summary["training_annotations_promoted"] is False, "run_evaluation_only")
        audit.check(not config["publication_ready"] and not config["media_independent"], "bulk_is_data_staging")
        enabled, bands = verify_catalog(catalog, audit)
        audit.check(set(manifest["selected_tasks"]) == enabled and len(manifest["selected_tasks"]) == len(enabled)
                    and config["selected_tasks"] == manifest["selected_tasks"], "all_enabled_tasks_selected")
        audit.check(selection_contract(manifest), "full_or_disclosed_partial_selection")
        implementation = read(run / "implementation.json")
        actual_implementation = hashlib.sha256(b"".join(name.encode() + below(run, Path("implementation/close_gripper") / name).read_bytes()
                                                       for name in implementation["files"])).hexdigest()
        audit.check(actual_implementation == implementation["sha256"] == summary["implementation_sha256"]
                    and summary["implementation_changed_during_run"] is False, "implementation_snapshot_integrity")
        shards = list(json_lines(run / "shards.jsonl"))
        records = {(s["dataset_id"], s["source_relative_path"]): s for s in shards}
        expected = {(d["dataset_id"], s["path"]) for d in manifest["datasets"] for s in d["shards"]}
        audit.check(set(records) == expected and len(records) == len(shards) == len(expected)
                    and summary["completed_shards"] == summary["expected_shards"] == len(expected),
                    "all_source_shards_recorded_once")
        executor = (ProcessPoolExecutor(max_workers=workers, initializer=_initialize_audit_worker, initargs=(catalog,))
                    if workers > 1 else nullcontext(None))
        with gzip.open(output / "episodes.jsonl.gz", "wt", encoding="utf-8") as episode_stream, \
                (output / "shards.jsonl").open("x", encoding="utf-8") as shard_stream, executor as pool:
            def accept(dataset, shard, result):
                nonlocal source_files, verified_episodes
                counts, rows, local_report = result
                audit.merge(local_report)
                dsid = dataset["dataset_id"]
                key = (dsid, shard["path"])
                ds_counts = by_dataset[dsid]
                source_files += 1
                for task, counters in counts.items():
                    by_task.setdefault(task, Counter()).update(counters)
                    totals.update(counters); ds_counts.update(counters)
                ds_counts["shards"] += 1
                for row in rows:
                    episode_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                verified_episodes += len(rows)
                line = dict(dataset_id=dsid, path=shard["path"], episodes=len(rows), failures=local_report["failure_count"],
                            source_sha256=shard["sha256"], output_sha256=records[key]["output_sha256"])
                shard_stream.write(json.dumps(line, ensure_ascii=False) + "\n"); shard_stream.flush()
                print(json.dumps(dict(event="audit_shard", completed=source_files, total=len(expected),
                                      failures=audit.failure_count, elapsed_s=round(time.monotonic()-started, 2)), ensure_ascii=False), flush=True)

            for dataset in manifest["datasets"]:
                dsid, root = dataset["dataset_id"], Path(dataset["dataset_root"]).resolve(strict=True)
                ds_counts = by_dataset.setdefault(dsid, Counter())
                audit.check((root.parent.name, root.name) == (dataset.get("release_directory", dataset["release"]), dataset["subset"]),
                            "dataset_release_identity", dsid)
                metadata, listed_meta = {}, set()
                for item in dataset["metadata_files"]:
                    path = below(root, item["path"])
                    snapshot = below(run, Path("input_metadata") / dsid / item["path"])
                    audit.check(digest(path) == item["sha256"] == digest(snapshot), "original_metadata_and_snapshot_sha", [dsid, item["path"]])
                    if item["path"].startswith("meta/episodes/"):
                        listed_meta.add(item["path"])
                        for row in pq.read_table(path, use_threads=False).to_pylist():
                            index = row["episode_index"]
                            audit.check(index not in metadata, "unique_original_episode_metadata", [dsid, index])
                            metadata[index] = row
                audit.check(listed_meta == {p.relative_to(root).as_posix() for p in (root / "meta/episodes").glob("chunk-*/*.parquet")},
                            "complete_original_episode_metadata", dsid)
                info, mask = read(root / "meta/info.json"), read(root / "dataset_configs/masks/mask.json")
                audit.check(_valid_sides(info, mask) == dataset["valid_sides"], "valid_sides_from_original_mask", dsid)
                ledger_path = below(run, Path("input_metadata") / dsid / "source_episodes.jsonl.gz")
                audit.check(digest(ledger_path) == dataset["source_episodes_sha256"], "frozen_episode_ledger_sha", dsid)
                ledger_rows = list(json_lines(ledger_path))
                ledger = {r["episode_index"]: r for r in ledger_rows}
                audit.check(len(ledger) == len(ledger_rows) and set(ledger) == set(metadata), "ledger_all_original_episodes", dsid)
                actual_data = {p.relative_to(root).as_posix() for p in (root / "data").glob("chunk-*/*.parquet")}
                listed_data = {s["path"] for s in dataset["shards"]}
                if manifest["complete_source_scope"]:
                    audit.check(actual_data == listed_data and set(metadata) == {i for s in dataset["shards"] for i in s["episodes"]},
                                "complete_dataset_scope", dsid)
                pending = {}
                compact_dataset = {k: dataset[k] for k in ("dataset_id", "dataset_root", "release", "subset", "valid_sides")}
                if "release_directory" in dataset:
                    compact_dataset["release_directory"] = dataset["release_directory"]
                for shard in dataset["shards"]:
                    key = (dsid, shard["path"])
                    if key not in records:
                        continue
                    job = dict(run=run, dataset=compact_dataset, shard=shard, record=records[key],
                               sources={i: ledger[i] for i in shard["episodes"] if i in ledger},
                               metadata={i: metadata[i] for i in shard["episodes"] if i in metadata})
                    if pool is None:
                        accept(dataset, shard, _audit_shard_job(job, catalog))
                        continue
                    pending[pool.submit(_audit_shard_job, job)] = shard
                    if len(pending) >= workers:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            accept(dataset, pending.pop(future), future.result())
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        accept(dataset, pending.pop(future), future.result())
        for task, counts in by_task.items():
            audit.check(dict(counts) == summary["tasks"].get(task), "summary_task_totals", task)
        audit.check(set(by_task) == set(summary["tasks"]), "summary_task_scope")
        for key, value in totals.items():
            audit.check(summary.get(key) == value, "summary_total", key)
        declared_datasets = {d["dataset_id"]: d for d in summary["datasets"]}
        audit.check(set(declared_datasets) == set(by_dataset), "summary_dataset_inventory")
        for dsid, counts in by_dataset.items():
            audit.check(all(declared_datasets.get(dsid, {}).get(k) == v for k, v in counts.items()), "summary_dataset_totals", dsid)
    except (KeyError, ValueError, TypeError, OSError, IndexError, StopIteration, ZeroDivisionError) as exc:
        audit.check(False, "invalid_run_contract", [type(exc).__name__, str(exc)])
    report = dict(audit.report(), run=str(run), manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                  script_sha256=digest(__file__), elapsed_s=time.monotonic()-started,
                  workers=workers,
                  verified_source_shards=source_files, verified_episodes=verified_episodes,
                  enabled_tasks=len(enabled), closed_bands=bands, totals=dict(totals),
                  complete_source_scope=read(manifest_path).get("complete_source_scope"),
                  sampling=read(manifest_path).get("sampling"),
                  selection_uses_band_matching=read(manifest_path).get("selection_uses_band_matching"),
                  independent_effect_estimate=False,
                  independent_oracle="range-min/max binary search, significant-step grouping, fixed-timing equations; no generator imports",
                  delivery_media_audit="not_in_bulk_scope",
                  limitations=["Checks numerical generation and source preservation, not per-episode visual grasp truth or training benefit.",
                               "Original recording IDs are bound to the frozen source ledger; raw MCAP semantics are not reread."])
    with (output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional original scope manifest; default: RUN/manifest.json")
    parser.add_argument("--output", required=True, type=Path, help="Fresh workspace runs directory")
    parser.add_argument("--workers", type=int, default=1, help="Independent shard processes, 1 to 4")
    args = parser.parse_args(argv)
    report = audit_bulk(args.run, args.manifest or args.run / "manifest.json", args.output, workers=args.workers)
    print(json.dumps({k: report[k] for k in ("status", "failure_count", "verified_source_shards", "verified_episodes")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
