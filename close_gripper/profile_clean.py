"""Whole-episode task-profile rule evaluation without visual-label promotion.

Only ``rule_target`` / ``preview_target`` contain generated commands. Training
targets remain the observations and invalid. Optional per-frame visual states
restrict numerical candidates without promoting model review to ground truth.
"""

from collections import deque
import copy

from .contracts import SIDE_INDICES, validate_episode
from .cycles import _motions, cycle_config
from .task_profiles import resolve_profile, validate_catalog
from .profile_bands import matching_closed_bands


RULE_VERSION = "task-profile-rule-v3-hold-first"
RULE_SOURCE = "task_profile_rule"
REVIEWED_RULE_SOURCE = "task_profile_rule_reviewed"
_EPS = 1e-9


def _blocks(times, maximum_gap):
    start = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > maximum_gap + _EPS:
            yield start, i - 1
            start = i
    yield start, len(times) - 1


def _stable_windows(values, band, start, end, tolerance):
    """All maximal in-band range-bounded windows, including overlapping ones."""
    low, high, left = deque(), deque(), start
    for right in range(start, end + 1):
        value = values[right]
        if not band["lower"] <= value <= band["upper"]:
            if left < right:
                yield left, right - 1
            low.clear()
            high.clear()
            left = right + 1
            continue
        while low and values[low[-1]] >= value:
            low.pop()
        while high and values[high[-1]] <= value:
            high.pop()
        low.append(right)
        high.append(right)
        previous = left
        while values[high[0]] - values[low[0]] > tolerance + _EPS:
            if low[0] == left:
                low.popleft()
            if high[0] == left:
                high.popleft()
            left += 1
        if left != previous:
            yield previous, right - 1
    if left <= end:
        yield left, end


def _candidate(episode, side, band, block, windows, matched, reason=None):
    times = episode["timestamps"]
    start, end = windows[0]["start"], max(w["end"] for w in windows)
    result = dict(candidate_id=f"{episode['episode_id']}:{side}:profile:{band['id']}:{start}:{end}",
        side=side, closed_band_id=band["id"], start=start, end=end,
        hold_start=start, hold_end=end, start_time=times[start], end_time=times[end],
        duration_s=times[end] - times[start], block_start=block[0], block_end=block[1],
        trigger_kind="task_profile_interval", matched=matched, matched_windows=windows if matched else [],
        qualification_reason=reason or "task_profile_numeric_match", evaluation_only=True,
        grasp_semantics_verified=False)
    if matched:
        result.update(confirmation_frame=windows[0]["confirmation_frame"],
                      confirmation_time=windows[0]["confirmation_time"])
    return result


def _candidates(episode, side, profile, catalog):
    values = [s[SIDE_INDICES[side]] for s in episode["state"]]
    times = episode["timestamps"]
    candidates = []
    for block in _blocks(times, catalog["maximum_sample_gap_s"]):
        for band in matching_closed_bands(profile) + profile.get("interval_vision_bands", []):
            requires_vision = band in profile.get("interval_vision_bands", [])
            matched, short = [], []
            for a, b in _stable_windows(values, band, *block, catalog["maximum_plateau_range"]):
                window = dict(start=a, end=b, duration_s=times[b] - times[a],
                              observed_lower=min(values[a:b + 1]), observed_upper=max(values[a:b + 1]))
                if window["duration_s"] + _EPS < catalog["minimum_hold_s"]:
                    short.append(window)
                    continue
                confirmation = next(i for i in range(a, b + 1)
                                    if times[i] - times[a] + _EPS >= catalog["minimum_hold_s"])
                window.update(confirmation_frame=confirmation, confirmation_time=times[confirmation])
                matched.append(window)
            groups = []
            for window in matched:
                # Every merged frame is covered by a qualifying witness. A
                # witness, rather than this union, has range <= 0.02.
                if groups and window["start"] <= groups[-1][-1]["end"] + 1:
                    groups[-1].append(window)
                else:
                    groups.append([window])
            for group in groups:
                candidate = _candidate(episode, side, band, block, group, True)
                if requires_vision:
                    candidate.update(matched=False, qualification_reason="requires_interval_vision",
                                     interval_vision_band_id=band["id"])
                candidates.append(candidate)
            candidates.extend(_candidate(episode, side, band, block, [window], False,
                                         "insufficient_holding_duration") for window in short)
    return sorted(candidates, key=lambda c: (c["start"], c["end"], c["closed_band_id"]))


def _reviewed_candidates(episode, numeric, states, catalog):
    """Intersect each numerical witness with observed, contiguous held frames.

    The original timestamp block stays attached to every fragment so motion
    discovery can still see the surrounding closing/release. Rejected fragments
    account for every original matched frame without a qualified held witness.
    """
    times = episode["timestamps"]
    refined = []
    for original in numeric:
        if not original["matched"]:
            refined.append(original)
            continue
        side = original["side"]
        values = [row[SIDE_INDICES[side]] for row in episode["state"]]
        held, start = [], None
        for i in range(original["start"], original["end"] + 2):
            state = states[i] if i <= original["end"] else None
            if state == "held" and start is None:
                start = i
            elif state != "held" and start is not None:
                held.append((start, i - 1))
                start = None
        windows = {}
        for witness in original["matched_windows"]:
            for x, y in held:
                a, b = max(x, witness["start"]), min(y, witness["end"])
                if a > b or times[b] - times[a] + _EPS < catalog["minimum_hold_s"]:
                    continue
                confirmation = next(i for i in range(a, b + 1)
                                    if times[i] - times[a] + _EPS >= catalog["minimum_hold_s"])
                windows[a, b] = dict(start=a, end=b, duration_s=times[b] - times[a],
                    observed_lower=min(values[a:b + 1]), observed_upper=max(values[a:b + 1]),
                    confirmation_frame=confirmation, confirmation_time=times[confirmation])
        groups = []
        for window in sorted(windows.values(), key=lambda w: (w["start"], w["end"])):
            if groups and window["start"] <= max(w["end"] for w in groups[-1]) + 1:
                groups[-1].append(window)
            else:
                groups.append([window])
        covered = set()
        band = dict(id=original["closed_band_id"])
        block = original["block_start"], original["block_end"]
        for group in groups:
            candidate = _candidate(episode, side, band, block, group, True,
                                   "reviewed_held_stable_match")
            candidate.update(numeric_candidate_id=original["candidate_id"], semantic_gate_applied=True,
                             numeric_hold_start=original["start"], numeric_hold_end=original["end"])
            refined.append(candidate)
            covered.update(range(candidate["start"], candidate["end"] + 1))
        a = original["start"]
        while a <= original["end"]:
            if a in covered:
                a += 1
                continue
            state, b = states[a], a
            while b < original["end"] and b + 1 not in covered and states[b + 1] == state:
                b += 1
            reason = {"free": "reviewed_free", "uncertain": "reviewed_uncertain",
                      "held": "insufficient_reviewed_holding_duration"}[state]
            window = dict(start=a, end=b, duration_s=times[b] - times[a],
                          observed_lower=min(values[a:b + 1]), observed_upper=max(values[a:b + 1]))
            rejected = _candidate(episode, side, band, block, [window], False, reason)
            rejected.update(numeric_candidate_id=original["candidate_id"], semantic_gate_applied=True,
                            reviewed_state=state, numeric_hold_start=original["start"],
                            numeric_hold_end=original["end"])
            if state == "held":
                rejected["unqualified_windows"] = [
                    dict(start=x, end=y, duration_s=times[y] - times[x], reason=reason)
                    for witness in original["matched_windows"]
                    for x, y in [(max(a, witness["start"]), min(b, witness["end"]))] if x <= y]
            refined.append(rejected)
            a = b + 1
    return sorted(refined, key=lambda c: (c["start"], c["end"], c["closed_band_id"]))


def _gate_transitions(episode, candidate, fit, states):
    """Keep each fixed-time transition whole, or retain its original observations."""
    values = [row[SIDE_INDICES[candidate["side"]]] for row in episode["state"]]
    a, b = candidate["start"], candidate["end"]
    held_start, held_end = a, b
    while held_start > candidate["block_start"] and states[held_start - 1] == "held":
        held_start -= 1
    while held_end < candidate["block_end"] and states[held_end + 1] == "held":
        held_end += 1
    for kind, x, y in (("closing", fit["closing_start"], a - 1),
                       ("release", b + 1, fit["release_end"])):
        if x is None or y is None:
            continue
        indices = range(x, y + 1)
        reason = ("reviewed_uncertain_boundary" if any(states[i] == "uncertain" for i in indices) else
                  "crosses_independent_held_cycle" if any(states[i] == "held" and
                      not held_start <= i <= held_end for i in indices) else
                  "zero_on_reviewed_free_frame" if any(states[i] == "free" and
                      fit["curve"][i] == 0.0 for i in indices) else None)
        if reason is None:
            continue
        for i in indices:
            fit["curve"][i], fit["phases"][i] = values[i], "unchanged"
        fit["transition_issues"].append(kind + ":" + reason)
        if kind == "closing":
            fit.update(start=a, closing_start=None, observed_closing_end=None)
        else:
            fit.update(end=b, release_start=None, release_end=None,
                       observed_release_start=None, release_motion_min_delta=None)
    return fit


def _fit(episode, candidate, profile, cfg, motions, maximum_plateau_range, release_motions=None):
    """Zero a qualified hold, then fit observed transitions around its anchors.

    The stable band's first/last frames are immutable zero anchors. Motion
    detection only supplies the outer transition endpoints; it cannot veto or
    shorten the hold. The existing monotonic fixed-time formulas are retained.
    """
    values = [s[SIDE_INDICES[candidate["side"]]] for s in episode["state"]]
    times = episode["timestamps"]
    a, b = candidate["start"], candidate["end"]
    issues = []
    closings = [m for m in motions if m["direction"] < 0 and m["start"] < a <= m["end"] <= b]
    closing = min(closings, key=lambda m: (m["end"], -m["start"]), default=None)
    openings = [m for m in motions if m["direction"] > 0 and a <= m["start"] <= b < m["end"]]
    opening = min(openings, key=lambda m: (m["start"], m["end"]), default=None)
    cs = closing["start"] if closing else None
    re = opening["end"] if opening else None
    for kind, start, end in (("closing", cs, a), ("release", b, re)):
        if start is None or end is None:
            continue
        delta = values[start] - values[end] if kind == "closing" else values[end] - values[start]
        reason = ("nonpositive_excursion" if delta <= _EPS else
                  "unreliable_motion_speed" if delta / (times[end] - times[start]) > cfg["maximum_motion_speed"] else None)
        if reason:
            issues.append(kind + ":" + reason)
            if kind == "closing":
                cs = None
            else:
                re = None
    if cs is None and a > candidate["block_start"] and not any(x.startswith("closing:") for x in issues):
        issues.append("closing:no_observed_boundary")
    if re is None and b < candidate["block_end"] and not any(x.startswith("release:") for x in issues):
        issues.append("release:no_observed_boundary")

    curve, phases = list(values), ["unchanged"] * len(values)
    start, end = cs if cs is not None else a, re if re is not None else b
    if cs is not None:
        beginning, connection = values[cs], values[a]
        phase = 0.0
        for i in range(cs, a + 1):
            phase = max(phase, min(1.0, max(0.0, (beginning - values[i]) / (beginning - connection))))
            curve[i], phases[i] = beginning * (1.0 - phase), "closing"
    if re is not None:
        beginning, endpoint = values[b], values[re]
        phase = 0.0
        for i in range(b, re + 1):
            phase = max(phase, min(1.0, max(0.0, (values[i] - beginning) / (endpoint - beginning))))
            curve[i], phases[i] = endpoint * phase, "release"
    for i in range(a, b + 1):
        curve[i], phases[i] = 0.0, "hold"
    return dict(curve=curve, phases=phases, start=start, end=end, closing_start=cs,
        hold_start=a, contact_frame=a, hold_end=b, release_start=b if re is not None else None, release_end=re,
        confirmation_frame=candidate["confirmation_frame"], confirmation_time=candidate["confirmation_time"],
        initial_or_cropped_hold=a == candidate["block_start"],
        open_reference_lower=None, release_rule_id=None,
        release_motion_min_delta=2 * cfg["motion_noise"] + _EPS if re is not None else None,
        release_reference_gate_applied=False, transition_issues=issues,
        observed_closing_end=closing["end"] if cs is not None else None,
        observed_release_start=opening["start"] if re is not None else None,
        cropped_start_by_timestamp_gap=a == candidate["block_start"] and candidate["block_start"] > 0,
        cropped_end_by_timestamp_gap=b == candidate["block_end"] and b < len(values) - 1), None


def generate_profile_targets(episode, catalog, config=None, *, reviewed_states=None):
    """Evaluate every eligible interval, leaving training annotations unchanged.

    ``rule_applied_mask`` covers successful fits including their same-value
    endpoints. ``valid`` remains false. Unmatched/rejected ranges keep their
    original values in every curve. Timestamp gaps break both discovery and
    fitting; they are never treated as observed holding time. Optional explicit
    per-frame ``reviewed_states`` restrict holds and transitions to visible
    evidence; omission retains the historical numerical-only behavior.
    """
    validate_catalog(catalog)
    try:
        validate_episode(episode)
    except OverflowError as exc:
        raise ValueError("episode numbers must be finite") from exc
    cfg = cycle_config(config)
    if config is not None and "minimum_hold_s" in config and config["minimum_hold_s"] != catalog["minimum_hold_s"]:
        raise ValueError("minimum_hold_s must agree with the task catalog")
    cfg["minimum_hold_s"] = catalog["minimum_hold_s"]
    times, size = episode["timestamps"], len(episode["timestamps"])
    if reviewed_states is not None:
        if (not isinstance(reviewed_states, dict) or set(reviewed_states) != set(episode["valid_sides"])
                or any(not isinstance(states, list) or len(states) != size
                       or any(not isinstance(state, str) or state not in ("held", "free", "uncertain")
                              for state in states) for states in reviewed_states.values())):
            raise ValueError("reviewed_states must supply one held/free/uncertain state per frame of every valid side")
    rule_source = REVIEWED_RULE_SOURCE if reviewed_states is not None else RULE_SOURCE
    original = [[s[9], s[19]] for s in episode["state"]]
    rule = copy.deepcopy(original)
    mask = [[False, False] for _ in range(size)]
    phase = [["unchanged", "unchanged"] for _ in range(size)]
    source = [["", ""] for _ in range(size)]
    candidates, numeric_candidates, events, coverage, side_status = [], [], [], {}, {}
    proposals = []
    for side, state_index in SIDE_INDICES.items():
        col = 0 if side == "left" else 1
        coverage[side] = dict(valid_side=side in episode["valid_sides"], complete_numeric_scan=False,
            complete_review=False, reviewed_candidates=0, reviewed_frames=0, review_window_frames=0,
            target_valid_frames=0, candidates=0, matched_intervals=0, matched_windows=0,
            accepted_cycles=0, preview_cycles=0, rejected_cycles=0, applied_frames=0, changed_frames=0,
            coverage_definition="Numerical rule scan and evaluation fits only; no per-episode visual judgments.")
        if side not in episode["valid_sides"]:
            coverage[side]["reason"] = side_status[side] = "invalid_side"
            continue
        resolved = resolve_profile(catalog, episode, side)
        if resolved["status"] != "resolved":
            coverage[side]["reason"] = resolved["reason"]
            side_status[side] = "profile_rejected"
            continue
        profile = resolved["profile"]
        coverage[side].update(task=resolved["task"], profile_status=profile["status"],
                              automatic_use=profile["automatic_use"])
        if not profile["automatic_use"] and not profile.get("interval_vision_bands"):
            coverage[side]["reason"] = "profile_not_automatic"
            side_status[side] = "profile_rejected"
            continue
        scanned_profile = profile if profile["automatic_use"] else dict(profile, closed_bands=[])
        if catalog.get("matching_policy"):
            coverage[side].update(matching_policy=catalog["matching_policy"],
                matching_bands=[{k: b[k] for k in ("id", "lower", "upper", "basis_closed_band_ids")}
                                for b in matching_closed_bands(scanned_profile)])
        found = _candidates(episode, side, scanned_profile, catalog)
        if reviewed_states is not None:
            numeric_candidates.extend(copy.deepcopy(found))
            numeric_count, numeric_matched = len(found), sum(c["matched"] for c in found)
            found = _reviewed_candidates(episode, found, reviewed_states[side], catalog)
            coverage[side].update(semantic_gate_applied=True, complete_review=True,
                reviewed_frames=size, reviewed_candidates=numeric_count,
                numeric_candidates=numeric_count, numeric_matched_intervals=numeric_matched,
                reviewed_state_counts={state: reviewed_states[side].count(state)
                                       for state in ("held", "free", "uncertain")},
                coverage_definition="Full numerical scan restricted by supplied per-frame visual states; model review is not human validation.")
        candidates.extend(found)
        blocks = list(_blocks(times, catalog["maximum_sample_gap_s"]))
        values = [s[state_index] for s in episode["state"]]
        motions = {}
        for release_rule in profile.get("release_rules", []):
            if cfg["motion_noise"] != release_rule["motion_noise"]:
                raise ValueError("task-local release rule requires unchanged motion_noise")
        for a, b in blocks:
            # Noise distinguishes observed movement from jitter; neither the
            # task's opening reference nor a 0.04 excursion gates a hold.
            transition_cfg = dict(cfg, motion_min_delta=2 * cfg["motion_noise"] + _EPS)
            local = _motions(values[a:b + 1], times[a:b + 1], transition_cfg)
            motions[a] = [dict(m, start=m["start"] + a, end=m["end"] + a) for m in local]
        coverage[side].update(complete_numeric_scan=True, candidates=len(found),
            matched_intervals=sum(c["matched"] for c in found),
            matched_windows=sum(len(c["matched_windows"]) for c in found),
            timestamp_gap_boundaries=[[b, b + 1] for _, b in blocks[:-1]])
        for candidate in found:
            event = dict(candidate, status="unchanged", reason=candidate["qualification_reason"],
                         changed_frames=0, reviewed=reviewed_states is not None, source=rule_source,
                         modified_start=None, modified_end=None, closing_start=None,
                         contact_frame=None, release_start=None, release_end=None)
            events.append(event)
            if not candidate["matched"]:
                continue
            fitted, reason = _fit(episode, candidate, profile, cfg, motions[candidate["block_start"]],
                catalog["maximum_plateau_range"])
            if fitted is None:
                event["reason"] = reason
            else:
                if reviewed_states is not None:
                    fitted = _gate_transitions(episode, candidate, fitted, reviewed_states[side])
                proposals.append((event, fitted, col))
        side_status[side] = "numeric_scan_complete"

    for event, fit, col in proposals:
        for frame in range(fit["start"], fit["end"] + 1):
            # The lower envelope connects overlapping transition proposals;
            # a qualified zero hold always takes priority over a transition.
            value = fit["curve"][frame]
            if (not mask[frame][col] or value < rule[frame][col]
                    or value == rule[frame][col] and fit["phases"][frame] == "hold"):
                rule[frame][col] = value
                phase[frame][col] = fit["phases"][frame]
            mask[frame][col] = True
            source[frame][col] = rule_source
        event.update({k: v for k, v in fit.items() if k not in ("curve", "phases", "start", "end")})
        event.update(status="applied", reason="task_profile_rule_evaluation",
            modified_start=fit["start"], modified_end=fit["end"],
            zero_frame=fit["contact_frame"], pregrasp_status="not_modified",
            closing_policy="fixed_timing", release_policy="fixed_timing")
    for event, fit, col in proposals:
        event["changed_frames"] = sum(abs(rule[k][col] - original[k][col]) > 1e-12
                                      for k in range(fit["start"], fit["end"] + 1))
    for side in SIDE_INDICES:
        col = 0 if side == "left" else 1
        selected = [e for e in events if e["side"] == side]
        applied = sum(e["status"] == "applied" for e in selected)
        coverage[side].update(preview_cycles=applied, rejected_cycles=len(selected) - applied,
            applied_frames=sum(row[col] for row in mask),
            changed_frames=sum(abs(row[col] - original[i][col]) > 1e-12 for i, row in enumerate(rule)))
        if coverage[side]["complete_numeric_scan"]:
            coverage[side]["reason"] = ("rule_targets_generated" if applied else
                "requires_interval_vision" if any(e["reason"] == "requires_interval_vision" for e in selected) else
                "no_qualified_reviewed_held_window" if coverage[side].get("numeric_matched_intervals") else
                "matched_intervals_rejected_at_boundaries" if coverage[side]["matched_intervals"] else
                "insufficient_holding_duration" if selected else "outside_closed_bands")
            side_status[side] = "rule_applied" if applied else "numerically_unchanged"
    result = dict(workflow=RULE_SOURCE, evidence_type=RULE_SOURCE, rule_source=rule_source,
        rule_version=RULE_VERSION, evaluation_only=True, grasp_semantics_verified=False,
        hold_policy="stable_task_band_only", overlap_policy="minimum_with_qualified_zero_holds",
        training_ready=False, model_requests=0, catalog_version=catalog["version"],
        target=copy.deepcopy(original), valid=[[False, False] for _ in range(size)],
        phase=[["unchanged", "unchanged"] for _ in range(size)], source=[["", ""] for _ in range(size)],
        preview_target=copy.deepcopy(rule), rule_target=rule, rule_applied_mask=mask,
        preview_phase=phase, preview_source=source, candidates=candidates, events=events,
        coverage=coverage, side_status=side_status, config=cfg,
        closing_comparisons=[], release_comparisons=[])
    if reviewed_states is not None:
        result.update(rule_version=RULE_VERSION + "-reviewed-v1", semantic_gate_applied=True,
                      hold_policy="reviewed_held_stable_task_band", numeric_candidates=numeric_candidates)
    return result
