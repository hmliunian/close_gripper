"""Whole-episode task-profile rule evaluation without visual-label promotion.

Only ``rule_target`` / ``preview_target`` contain generated commands. Training
targets remain the observations and invalid. The numerical rule cannot establish
that an individual episode contains a grasp, a release, or a no-grasp action.
"""

from collections import deque
import copy

from .contracts import SIDE_INDICES, validate_episode
from .cycles import _motions, cycle_config
from .task_profiles import resolve_profile, validate_catalog


RULE_VERSION = "task-profile-rule-v1"
RULE_SOURCE = "task_profile_rule"
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
        for band in profile["closed_bands"]:
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
            candidates.extend(_candidate(episode, side, band, block, group, True) for group in groups)
            candidates.extend(_candidate(episode, side, band, block, [window], False,
                                         "insufficient_holding_duration") for window in short)
    return sorted(candidates, key=lambda c: (c["start"], c["end"], c["closed_band_id"]))


def _fit(episode, candidate, profile, cfg, motions, maximum_plateau_range):
    """Use the existing cycle motion boundaries and fixed-timing phase formulas.

    The visual cycle generator deliberately cannot be called here: its decision
    contract is not numerical evidence. No model/human decision is fabricated.
    """
    values = [s[SIDE_INDICES[candidate["side"]]] for s in episode["state"]]
    times = episode["timestamps"]
    a, b = candidate["start"], candidate["end"]
    band = next(p for p in profile["closed_bands"] if p["id"] == candidate["closed_band_id"])
    closings = [m for m in motions if m["direction"] < 0 and m["start"] < a <= m["end"] <= b
                and times[m["end"]] - times[a] <= cfg["boundary_tolerance_s"] + _EPS]
    closing = min(closings, key=lambda m: (m["end"], -m["start"]), default=None)
    if closing is not None:
        cs, contact = closing["start"], closing["end"]
    elif a == candidate["block_start"]:
        cs, contact = None, a
    else:
        return None, "missing_aligned_closing_boundary"

    open_threshold = min(o["lower"] for o in profile["open_bands"] if o["lower"] > band["upper"])
    openings = [m for m in motions if m["direction"] > 0 and contact <= m["start"] <= b < m["end"]
                and values[m["end"]] >= open_threshold]
    opening = min(openings, key=lambda m: (m["start"], m["end"]), default=None)
    hold_end = opening["start"] if opening else b
    if times[hold_end] - times[contact] + _EPS < cfg["minimum_hold_s"]:
        return None, "insufficient_holding_duration_after_boundary_alignment"
    confirmation = next(i for i in range(contact, hold_end + 1)
                        if times[i] - times[contact] + _EPS >= cfg["minimum_hold_s"])
    confirmation_values = values[contact:confirmation + 1]
    if (not all(band["lower"] <= value <= band["upper"] for value in confirmation_values)
            or max(confirmation_values) - min(confirmation_values) > maximum_plateau_range + _EPS):
        return None, "unstable_confirmation_after_boundary_alignment"
    if not opening and b < candidate["block_end"] and values[b + 1] > cfg["exit_tolerance"]:
        return None, "unresolved_release_boundary"
    for motion in (closing, opening):
        if motion and abs(values[motion["end"]] - values[motion["start"]]) / (
                times[motion["end"]] - times[motion["start"]]) > cfg["maximum_motion_speed"]:
            return None, "unreliable_motion_speed"

    curve, phases = list(values), ["unchanged"] * len(values)
    start, end = cs if cs is not None else contact, opening["end"] if opening else hold_end
    if cs is not None:
        beginning, connection = values[cs], values[contact]
        if beginning - connection < cfg["motion_min_delta"]:
            return None, "insufficient_closing_excursion"
        phase = 0.0
        for i in range(cs, contact + 1):
            phase = max(phase, min(1.0, max(0.0, (beginning - values[i]) / (beginning - connection))))
            curve[i], phases[i] = beginning * (1.0 - phase), "closing"
    for i in range(contact, hold_end + 1):
        curve[i], phases[i] = 0.0, "hold"
    if opening:
        rs, re = opening["start"], opening["end"]
        beginning, endpoint = values[rs], values[re]
        phase = 0.0
        for i in range(rs, re + 1):
            phase = max(phase, min(1.0, max(0.0, (values[i] - beginning) / (endpoint - beginning))))
            curve[i], phases[i] = endpoint * phase, "release"
    return dict(curve=curve, phases=phases, start=start, end=end, closing_start=cs,
        hold_start=contact, contact_frame=contact, hold_end=hold_end,
        release_start=opening["start"] if opening else None,
        release_end=opening["end"] if opening else None,
        confirmation_frame=confirmation, confirmation_time=times[confirmation],
        initial_or_cropped_hold=cs is None, open_reference_lower=open_threshold,
        cropped_start_by_timestamp_gap=cs is None and candidate["block_start"] > 0,
        cropped_end_by_timestamp_gap=not opening and end == candidate["block_end"] and end < len(values) - 1), None


def generate_profile_targets(episode, catalog, config=None):
    """Evaluate every eligible interval, leaving training annotations unchanged.

    ``rule_applied_mask`` covers successful fits including their same-value
    endpoints. ``valid`` remains false. Unmatched/rejected ranges keep their
    original values in every curve. Timestamp gaps break both discovery and
    fitting; they are never treated as observed holding time.
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
    original = [[s[9], s[19]] for s in episode["state"]]
    rule = copy.deepcopy(original)
    mask = [[False, False] for _ in range(size)]
    phase = [["unchanged", "unchanged"] for _ in range(size)]
    source = [["", ""] for _ in range(size)]
    candidates, events, coverage, side_status = [], [], {}, {}
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
        if not profile["automatic_use"]:
            coverage[side]["reason"] = "profile_not_automatic"
            side_status[side] = "profile_rejected"
            continue
        found = _candidates(episode, side, profile, catalog)
        candidates.extend(found)
        blocks = list(_blocks(times, catalog["maximum_sample_gap_s"]))
        values = [s[state_index] for s in episode["state"]]
        motions = {}
        for a, b in blocks:
            local = _motions(values[a:b + 1], times[a:b + 1], cfg)
            motions[a] = [dict(m, start=m["start"] + a, end=m["end"] + a) for m in local]
        coverage[side].update(complete_numeric_scan=True, candidates=len(found),
            matched_intervals=sum(c["matched"] for c in found),
            matched_windows=sum(len(c["matched_windows"]) for c in found),
            timestamp_gap_boundaries=[[b, b + 1] for _, b in blocks[:-1]])
        for candidate in found:
            event = dict(candidate, status="unchanged", reason=candidate["qualification_reason"],
                         changed_frames=0, reviewed=False, source=RULE_SOURCE,
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
                proposals.append((event, fitted, col))
        side_status[side] = "numeric_scan_complete"

    conflicts = set()
    for i, (event, fit, col) in enumerate(proposals):
        for j, (_, other, other_col) in enumerate(proposals[:i]):
            if col != other_col:
                continue
            a, b = max(fit["start"], other["start"]), min(fit["end"], other["end"])
            if a <= b and any(abs(fit["curve"][k] - other["curve"][k]) > 1e-12 for k in range(a, b + 1)):
                conflicts.update((i, j))
    for i, (event, fit, col) in enumerate(proposals):
        if i in conflicts:
            event["reason"] = "overlapping_incompatible_profile_cycles"
            continue
        for frame in range(fit["start"], fit["end"] + 1):
            rule[frame][col] = fit["curve"][frame]
            mask[frame][col] = True
            phase[frame][col] = fit["phases"][frame]
            source[frame][col] = RULE_SOURCE
        event.update({k: v for k, v in fit.items() if k not in ("curve", "phases", "start", "end")})
        event.update(status="applied", reason="task_profile_rule_evaluation",
            modified_start=fit["start"], modified_end=fit["end"],
            changed_frames=sum(abs(fit["curve"][k] - original[k][col]) > 1e-12
                               for k in range(fit["start"], fit["end"] + 1)),
            zero_frame=fit["contact_frame"], pregrasp_status="not_modified",
            closing_policy="fixed_timing", release_policy="fixed_timing")
    for side in SIDE_INDICES:
        col = 0 if side == "left" else 1
        selected = [e for e in events if e["side"] == side]
        applied = sum(e["status"] == "applied" for e in selected)
        coverage[side].update(preview_cycles=applied, rejected_cycles=len(selected) - applied,
            applied_frames=sum(row[col] for row in mask),
            changed_frames=sum(abs(row[col] - original[i][col]) > 1e-12 for i, row in enumerate(rule)))
        if coverage[side]["complete_numeric_scan"]:
            coverage[side]["reason"] = ("rule_targets_generated" if applied else
                "matched_intervals_rejected_at_boundaries" if coverage[side]["matched_intervals"] else
                "insufficient_holding_duration" if selected else "outside_closed_bands")
            side_status[side] = "rule_applied" if applied else "numerically_unchanged"
    return dict(workflow=RULE_SOURCE, evidence_type=RULE_SOURCE, rule_source=RULE_SOURCE,
        rule_version=RULE_VERSION, evaluation_only=True, grasp_semantics_verified=False,
        training_ready=False, model_requests=0, catalog_version=catalog["version"],
        target=copy.deepcopy(original), valid=[[False, False] for _ in range(size)],
        phase=[["unchanged", "unchanged"] for _ in range(size)], source=[["", ""] for _ in range(size)],
        preview_target=copy.deepcopy(rule), rule_target=rule, rule_applied_mask=mask,
        preview_phase=phase, preview_source=source, candidates=candidates, events=events,
        coverage=coverage, side_status=side_status, config=cfg,
        closing_comparisons=[], release_comparisons=[])
