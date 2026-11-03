"""Pure numerical candidate detection and conservative, independently reviewed labels."""

from statistics import median

from .contracts import SIDE_CAMERAS, SIDE_INDICES, validate_decision, validate_episode, validated_config


def _camera_names(episode, side):
    cameras = episode.get("cameras", {})
    names = []
    for name in SIDE_CAMERAS[side]:
        for key in (name, "observation.images." + name):
            value = cameras.get(key)
            if isinstance(value, dict) and value.get("valid", True):
                names.append(key)
                break
    return names


def detect_candidates(episode, config=None):
    """Find increases independently on each valid side, including one-frame pulses."""
    validate_episode(episode)
    cfg = validated_config(config)
    times = episode["timestamps"]
    candidates = []
    for side in episode["valid_sides"]:
        widths = [row[SIDE_INDICES[side]] * cfg["width_mm"] for row in episode["state"]]
        i = 0
        while i < len(widths) - 1:
            if widths[i + 1] <= widths[i]:
                i += 1
                continue
            start, peak = i, i + 1
            j = i + 2
            while j < len(widths):
                if widths[j] > widths[peak]:
                    peak = j
                elif widths[peak] - widths[j] > cfg["opening_noise_mm"] or times[j] - times[peak] > cfg["opening_gap_s"] + 1e-9:
                    break
                j += 1
            net = widths[peak] - widths[start]
            duration = times[peak] - times[start]
            speed = net / duration
            if net >= cfg["opening_min_mm"] and speed >= cfg["opening_min_speed_mm_s"]:
                cameras = _camera_names(episode, side)
                candidates.append({"candidate_id": f"{episode['episode_id']}:{side}:{start}:{peak}",
                                   "side": side, "start": start, "end": peak,
                                   "start_time": times[start], "end_time": times[peak],
                                   "net_opening_mm": net, "speed_mm_s": speed,
                                   "cameras": cameras,
                                   "camera_status": "available" if cameras else "missing_same_side_camera"})
            i = max(peak, j - 1)
    return sorted(candidates, key=lambda item: (item["start"], item["side"]))


def _estimate(widths, times, contact, opening_start, cfg):
    """Fit only the contiguous descending segment preceding the reviewed contact."""
    if contact <= 0:
        return None, "episode_starts_holding_or_missing_descent"
    platform_end = contact
    while platform_end + 1 <= opening_start and times[platform_end + 1] - times[contact] <= cfg["platform_window_s"] + 1e-9:
        platform_end += 1
    platform = widths[contact:platform_end + 1]
    if len(platform) < 2:
        return None, "insufficient_platform"
    c = median(platform)
    if max(platform) - min(platform) > cfg["platform_tolerance_mm"] or abs(widths[contact] - c) > cfg["platform_tolerance_mm"]:
        return None, "unstable_platform_or_contact_mismatch"
    if c <= cfg["zero_tolerance_mm"]:
        return {"contact_frame": contact, "platform_mm": c, "connection_mm": widths[contact]}, "near_zero_platform"
    start = contact
    while start > 0 and times[contact] - times[start - 1] <= cfg["closing_lookback_s"] + 1e-9:
        speed = (widths[start - 1] - widths[start]) / (times[start] - times[start - 1])
        if speed < cfg["closing_min_speed_mm_s"]:
            break
        start -= 1
    if start == contact:
        return None, "missing_valid_precontact_descent"
    diff_speed = median([(widths[i - 1] - widths[i]) / (times[i] - times[i - 1]) for i in range(start + 1, contact + 1)])
    fit_speed = median([(widths[i] - widths[j]) / (times[j] - times[i])
                        for i in range(start, contact) for j in range(i + 1, contact + 1)])
    if not cfg["closing_min_speed_mm_s"] <= fit_speed <= cfg["closing_max_speed_mm_s"]:
        return None, "unreliable_closing_speed"
    if abs(diff_speed - fit_speed) / fit_speed > cfg["speed_disagreement_ratio"]:
        return None, "speed_estimators_disagree"
    return {"contact_frame": contact, "closing_start": start, "platform_mm": c,
            "connection_mm": widths[contact], "speed_mm_s": fit_speed,
            "difference_median_mm_s": diff_speed, "robust_fit_mm_s": fit_speed,
            "speed_sample_intervals": contact - start,
            "contact_time": times[contact], "zero_time": times[contact] + widths[contact] / fit_speed}, None


def _release_comparison(candidate, decision, widths, times, base, cfg):
    start, end = candidate["start"], decision.get("release_end")
    if end is None or end <= start or end < candidate["end"]:
        return None, "missing_release_boundary"
    original_start, original_end = widths[start], widths[end]
    if original_end - original_start < cfg["opening_min_mm"]:
        return None, "invalid_release_end"
    segment = widths[start:end + 1]
    if any(b < a - cfg["opening_noise_mm"] for a, b in zip(segment, segment[1:])):
        return None, "nonmonotonic_release_requires_review"
    fixed = list(base)
    for i in range(start, end + 1):
        phase = (widths[i] - original_start) / (original_end - original_start)
        fixed[i] = max(0.0, min(original_end, phase * original_end))
    speed = median([(widths[i] - widths[i - 1]) / (times[i] - times[i - 1])
                    for i in range(start + 1, end + 1) if widths[i] > widths[i - 1]])
    reach_time = times[start] + original_end / speed
    # No extra time is assumed to be semantically available beyond the reviewed release end.
    original_speed = None
    status = "insufficient_confirmed_release_time"
    if reach_time <= times[end] + 1e-9:
        original_speed = list(base)
        for i in range(start, end + 1):
            original_speed[i] = min(original_end, max(0, speed * (times[i] - times[start])))
        status = "preview_only"
    return {"candidate_id": candidate["candidate_id"], "side": candidate["side"],
            "start": start, "end": end, "fixed_timing": [x / cfg["width_mm"] for x in fixed],
            "original_speed": None if original_speed is None else [x / cfg["width_mm"] for x in original_speed],
            "original_speed_status": status, "opening_speed_mm_s": speed,
            "original_speed_zero_to_endpoint_time": reach_time,
            "extra_time_needed_s": max(0.0, reach_time - times[end]),
            "fixed_timing_changes_speed": True}, None


def generate_targets(episode, candidates, decisions, config=None):
    """Return independent two-column targets and review previews without input mutation.

    Default releases are previews only. Synthetic review never marks labels valid for
    training. An interior hold endpoint with no continuous exit is conservatively rejected.
    """
    validate_episode(episode)
    cfg = validated_config(config)
    times = episode["timestamps"]
    original = [[row[SIDE_INDICES[side]] for side in ("left", "right")] for row in episode["state"]]
    target, preview = [row[:] for row in original], [row[:] for row in original]
    valid = [[False, False] for _ in original]
    events, comparisons = [], []
    by_id = {}
    duplicate_ids = set()
    for decision in decisions:
        key = decision.get("candidate_id") if isinstance(decision, dict) else None
        if key in by_id:
            duplicate_ids.add(key)
        by_id[key] = decision
    candidate_lookup = {c.get("candidate_id"): c for c in candidates}
    checked_decisions = []
    for key, decision in by_id.items():
        if key in duplicate_ids or key not in candidate_lookup:
            continue
        try:
            validate_decision(decision, episode, candidate_lookup[key])
        except (ValueError, TypeError, KeyError):
            continue
        checked_decisions.append(decision)
    claimed = {side: set() for side in SIDE_INDICES}
    for candidate in candidates:
        event = dict(candidate)
        event.update(status="unchanged", reason="missing_visual_decision", changed_frames=0)
        events.append(event)
        side = candidate.get("side")
        if side not in episode["valid_sides"]:
            event["reason"] = "invalid_side"
            continue
        if (type(candidate.get("start")) is not int or type(candidate.get("end")) is not int
                or not 0 <= candidate["start"] < candidate["end"] < len(times)):
            event["reason"] = "invalid_candidate_bounds"
            continue
        if not _camera_names(episode, side):
            event["reason"] = "missing_same_side_camera"
            continue
        decision = by_id.get(candidate["candidate_id"])
        if decision is None:
            continue
        if candidate["candidate_id"] in duplicate_ids:
            event["reason"] = "duplicate_visual_decisions"
            continue
        try:
            validate_decision(decision, episode, candidate)
        except (ValueError, TypeError, KeyError) as exc:
            event.update(reason="invalid_visual_decision", detail=str(exc))
            continue
        event["decision"] = dict(decision)
        category = decision["category"]
        if category in ("no_grasp", "uncertain"):
            event["reason"] = category
            continue
        if decision["slip"] or decision["grasp_before"] != "held":
            event["reason"] = "slip_or_conflicting_grasp_evidence"
            continue
        if category == "suspected_misopen" and any(decision[k] != "held" for k in ("grasp_during", "grasp_after")):
            event["reason"] = "continuous_grasp_not_established"
            continue
        if category == "normal_release" and decision["grasp_after"] != "not_held":
            event["reason"] = "release_not_established"
            continue
        evidence = decision["evidence_frames"]
        if not any(i <= candidate["start"] for i in evidence) or not any(i >= candidate["end"] for i in evidence):
            event["reason"] = "missing_before_after_evidence"
            continue
        contact, hold_end = decision.get("hold_start"), decision.get("hold_end")
        if contact is None or hold_end is None or contact >= candidate["start"] or hold_end < candidate["start"]:
            event["reason"] = "invalid_holding_boundary"
            continue
        if category == "normal_release" and hold_end != candidate["start"]:
            event["reason"] = "hold_release_boundary_mismatch"
            continue
        if category == "suspected_misopen" and hold_end < candidate["end"]:
            event["reason"] = "holding_interval_does_not_cover_event"
            continue
        col = 0 if side == "left" else 1
        widths = [row[col] * cfg["width_mm"] for row in original]
        estimate, reason = _estimate(widths, times, contact, candidate["start"], cfg)
        if estimate:
            event.update(estimate)
        if reason:
            event["reason"] = reason
            continue
        allowed_end = hold_end - 1 if category == "normal_release" else hold_end
        if estimate["zero_time"] > times[allowed_end] + 1e-9:
            event["reason"] = "insufficient_holding_time"
            continue
        # A reviewed independent grasp/release is a hard boundary regardless of curve sign.
        crossing = any(d is not decision and d.get("side") == side and d.get("category") == "normal_release"
                       and d.get("hold_end") is not None and contact < d["hold_end"] < hold_end for d in checked_decisions)
        if crossing:
            event["reason"] = "crosses_other_release_boundary"
            continue
        if any(d is not decision and d["side"] == side and d.get("hold_start") is not None
               and contact < d["hold_start"] <= hold_end for d in checked_decisions):
            event["reason"] = "crosses_next_independent_grasp"
            continue
        curve = list(widths)
        for i in range(contact + 1, hold_end + 1):
            value = widths[contact] - estimate["speed_mm_s"] * (times[i] - times[contact])
            curve[i] = 0.0 if value <= 1e-9 else value
        event["zero_frame"] = next(i for i in range(contact + 1, hold_end + 1) if curve[i] <= 1e-9)
        event["limit_frame"] = hold_end
        comparison = None
        if category == "normal_release":
            comparison, reason = _release_comparison(candidate, decision, widths, times, curve, cfg)
            if reason:
                event["reason"] = reason
                continue
            curve = [x * cfg["width_mm"] for x in comparison["fixed_timing"]]
            region_end = decision["release_end"]
            status, reason = "preview", "release_policy_pending"
        else:
            if hold_end < len(times) - 1 and widths[hold_end + 1] > cfg["zero_tolerance_mm"]:
                event["reason"] = "unconfirmed_exit_boundary"
                continue
            region_end = hold_end
            if decision["confirmed"] and decision["review_source"] == "human":
                status, reason = "applied", "confirmed_misopen"
            else:
                status, reason = "preview", "synthetic_evidence" if decision["review_source"] == "synthetic" else "requires_human_review"
        modified = set(range(contact + 1, region_end + 1))
        if claimed[side] & modified:
            event["reason"] = "overlapping_event_requires_review"
            continue
        claimed[side].update(modified)
        if comparison:
            comparisons.append(comparison)
        for i in modified:
            preview[i][col] = curve[i] / cfg["width_mm"]
            if status == "applied":
                target[i][col] = preview[i][col]
                valid[i][col] = True
        event.update(status=status, reason=reason,
                     changed_frames=sum(abs(curve[i] / cfg["width_mm"] - original[i][col]) > 1e-12 for i in modified),
                     modified_start=contact + 1, modified_end=region_end)
    triggered_sides = {c.get("side") for c in candidates}
    return {"target": target, "valid": valid, "preview_target": preview, "events": events,
            "release_comparisons": comparisons,
            "side_status": {side: ("invalid" if side not in episode["valid_sides"] else "triggered" if side in triggered_sides else "not_triggered") for side in SIDE_INDICES},
            "config": cfg}
