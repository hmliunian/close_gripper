"""Whole-episode gripper commands, with numerical boundaries and visual provenance.

The old opening-event implementation remains in ``core`` for reproducibility.
Here ``valid`` means an automatically checked command can be consumed by a loader;
it does not mean that a model decision has become a human reference annotation.
All opening values, differences, and rates in this module are normalized (0--1).
"""

from collections import Counter
import math
from statistics import median

from .contracts import SIDE_CAMERAS, SIDE_INDICES, validate_episode


CYCLE_DEFAULTS = {
    "config_version": "full-cycle-v3",
    "motion_min_delta": 0.04,
    "motion_noise": 0.003,
    "motion_gap_s": 0.2,
    "boundary_tolerance_s": 0.5,
    "platform_window_s": 0.2,
    "platform_tolerance": 0.03,
    "minimum_hold_s": 0.2,
    "maximum_motion_speed": 20.0,
    "context_before_s": 3.0,
    "context_after_s": 0.5,
    "pregrasp_plateau_tolerance": 0.01,
    "pregrasp_min_plateau_frames": 2,
    "exit_tolerance": 0.005,
    "auto_accept_model": True,
    "closing_policy": "fixed_timing",
    "release_policy": "fixed_timing",
}


def cycle_config(config=None):
    """Return only known numerical settings; transport credentials never persist."""
    result = dict(CYCLE_DEFAULTS)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("cycle configuration must be an object")
        result.update({k: config[k] for k in CYCLE_DEFAULTS if k in config})
    for key, default in CYCLE_DEFAULTS.items():
        value = result[key]
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError("configuration must be boolean: " + key)
        elif isinstance(default, (int, float)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("configuration must be positive and finite: " + key)
    if type(result["pregrasp_min_plateau_frames"]) is not int:
        raise ValueError("pregrasp_min_plateau_frames must be an integer")
    if result["motion_noise"] >= result["motion_min_delta"]:
        raise ValueError("motion noise must be below minimum motion")
    for key in ("closing_policy", "release_policy"):
        if result[key] != "fixed_timing":
            raise ValueError("full-cycle policies must be fixed_timing")
    if not isinstance(result["config_version"], str) or not result["config_version"]:
        raise ValueError("config_version must be nonempty text")
    return result


def _cameras(episode, side):
    found = []
    for name in SIDE_CAMERAS[side]:
        for key in (name, "observation.images." + name):
            camera = episode.get("cameras", {}).get(key)
            if isinstance(camera, dict) and camera.get("valid", True):
                found.append(key)
                break
    return found


def _motions(values, times, cfg):
    """Significant monotone excursions; neutral drift cannot move the start back."""
    motions = []
    direction, start, end = 0, None, None

    def finish():
        if start is not None and abs(values[end] - values[start]) >= cfg["motion_min_delta"]:
            motions.append(dict(direction=direction, start=start, end=end))

    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        sign = 1 if delta > cfg["motion_noise"] else -1 if delta < -cfg["motion_noise"] else 0
        if not sign:
            if direction and times[i] - times[end] > cfg["motion_gap_s"] + 1e-9:
                finish()
                direction, start, end = 0, None, None
            continue
        if sign != direction or (end is not None and times[i - 1] - times[end] > cfg["motion_gap_s"] + 1e-9):
            finish()
            direction, start = sign, i - 1
        end = i
    finish()
    return motions


def _intervals(flags, value=False):
    result, start = [], None
    for i, flag in enumerate(flags + [not value]):
        if flag == value and start is None:
            start = i
        elif flag != value and start is not None:
            result.append([start, i - 1])
            start = None
    return result


def detect_cycles(episode, config=None):
    """Propose every closing cycle and cover remaining time with observation windows.

    Motion is a review cue, not proof of grasping. ``initial_hold`` and ``static``
    windows intentionally also cover empty/open grippers; vision must decide.
    """
    validate_episode(episode)
    cfg = cycle_config(config)
    times = episode["timestamps"]
    size = len(times)
    candidates = []
    for side in episode["valid_sides"]:
        values = [row[SIDE_INDICES[side]] for row in episode["state"]]
        motions = _motions(values, times, cfg)
        closings = [m for m in motions if m["direction"] < 0]
        openings = [m for m in motions if m["direction"] > 0]
        side_candidates, covered = [], [False] * size
        for ordinal, closing in enumerate(closings):
            following_close = closings[ordinal + 1]["start"] if ordinal + 1 < len(closings) else size
            opening = next((m for m in openings if closing["end"] <= m["start"] < following_close), None)
            a = closing["start"]
            while a > 0 and times[closing["start"]] - times[a - 1] <= cfg["context_before_s"] + 1e-9:
                a -= 1
            b = opening["end"] if opening else following_close - 1
            if opening:
                while b + 1 < min(size, following_close) and times[b + 1] - times[opening["end"]] <= cfg["context_after_s"] + 1e-9:
                    b += 1
            b = max(b, closing["end"])
            side_candidates.append(dict(start=a, end=b, closing_start=closing["start"],
                contact_frame=closing["end"], release_start=opening["start"] if opening else None,
                release_end=opening["end"] if opening else None, trigger_kind="closure"))
            covered[a:b + 1] = [True] * (b - a + 1)
        for a, b in _intervals(covered):
            if a == b and side_candidates:
                adjacent = next((c for c in side_candidates if c["end"] == a - 1), None)
                if adjacent is not None:
                    adjacent["end"] = b
                    continue
                adjacent = next((c for c in side_candidates if c["start"] == b + 1), None)
                if adjacent is not None:
                    adjacent["start"] = a
                    continue
            opening = next((m for m in openings if a <= m["start"] < m["end"] <= b), None)
            side_candidates.append(dict(start=a, end=b, closing_start=None, contact_frame=None,
                release_start=opening["start"] if opening else None,
                release_end=opening["end"] if opening else None,
                trigger_kind="initial_hold" if a == 0 else "static"))
        cameras = _cameras(episode, side)
        for c in sorted(side_candidates, key=lambda c: (c["start"], c["closing_start"] or -1)):
            anchor = c["closing_start"] if c["closing_start"] is not None else c["start"]
            c.update(candidate_id=f"{episode['episode_id']}:{side}:cycle:{c['trigger_kind']}:{anchor}:{c['end']}",
                side=side, pregrasp_start=c["start"] if c["closing_start"] is not None else None,
                start_time=times[c["start"]], end_time=times[c["end"]],
                cameras=cameras, camera_status="available" if cameras else "missing_same_side_camera")
            candidates.append(c)
    return sorted(candidates, key=lambda c: (c["closing_start"] if c["closing_start"] is not None else c["start"],
                                             c["side"], c["candidate_id"]))


def _validate(candidate, decision, episode):
    size = len(episode["timestamps"])
    if not isinstance(candidate, dict) or candidate.get("side") not in episode["valid_sides"]:
        raise ValueError("invalid_candidate_side")
    a, b = candidate.get("start"), candidate.get("end")
    if type(a) is not int or type(b) is not int or not 0 <= a <= b < size:
        raise ValueError("invalid_candidate_bounds")
    if not isinstance(decision, dict) or decision.get("candidate_id") != candidate.get("candidate_id") or decision.get("side") != candidate["side"]:
        raise ValueError("decision_identity_mismatch")
    if decision.get("category") not in ("grasp_release", "grasp_hold", "no_grasp", "uncertain"):
        raise ValueError("invalid_category")
    if decision.get("review_source") not in ("model", "human", "synthetic"):
        raise ValueError("invalid_review_source")
    if "confirmed" in decision and type(decision["confirmed"]) is not bool:
        raise ValueError("confirmed_must_be_boolean")
    if decision["review_source"] == "model" and decision.get("confirmed"):
        raise ValueError("model_cannot_self_confirm")
    if decision["review_source"] == "human" and (not decision.get("confirmed") or
            any(not isinstance(decision.get(k), str) or not decision[k].strip()
                for k in ("reviewer", "reviewed_at", "review_note"))):
        raise ValueError("missing_human_provenance")
    for key in ("continuous_grasp", "slip", "pregrasp_clear"):
        if type(decision.get(key)) is not bool:
            raise ValueError("invalid_boolean_" + key)
    for key in ("hold_start", "hold_end", "release_end", "pregrasp_start"):
        value = decision.get(key)
        if value is not None and (type(value) is not int or not a <= value <= b):
            raise ValueError("boundary_outside_review_" + key)
    evidence = decision.get("evidence_frames")
    if (not isinstance(evidence, list) or not evidence or
            any(type(i) is not int or not a <= i <= b for i in evidence)):
        raise ValueError("missing_or_invalid_evidence")
    if not isinstance(decision.get("reason", ""), str):
        raise ValueError("reason_must_be_text")
    if decision["category"] in ("grasp_release", "grasp_hold"):
        hs, he = decision.get("hold_start"), decision.get("hold_end")
        if hs is None or he is None or hs > he:
            raise ValueError("invalid_holding_interval")
        if not any(hs <= i <= he for i in evidence):
            raise ValueError("missing_holding_evidence")
        if decision["category"] == "grasp_release":
            re = decision.get("release_end")
            if re is None or re <= he or not any(i >= re for i in evidence):
                raise ValueError("missing_release_evidence")
        else:
            if decision.get("release_end") is not None:
                raise ValueError("contradictory_holding_release_boundary")
            if not any(i >= he for i in evidence):
                raise ValueError("missing_holding_end_evidence")
        if decision["pregrasp_clear"]:
            ps = decision.get("pregrasp_start")
            if ps is None or ps >= hs or not any(ps <= i < hs for i in evidence):
                raise ValueError("missing_pregrasp_evidence")
        elif decision.get("pregrasp_start") is not None:
            raise ValueError("contradictory_pregrasp_evidence")
    elif (any(decision.get(k) is not None for k in ("hold_start", "hold_end", "release_end", "pregrasp_start"))
          or decision["continuous_grasp"] or decision["pregrasp_clear"]):
        raise ValueError("contradictory_non_grasp_evidence")


def _proposal(episode, candidate, d, cfg, motions):
    """Generate a complete local command or return an explicit rejection reason."""
    if d["category"] in ("no_grasp", "uncertain"):
        return None, d["category"]
    if d["slip"] or not d["continuous_grasp"]:
        return None, "slip_or_discontinuous_grasp"
    times = episode["timestamps"]
    values = [s[SIDE_INDICES[candidate["side"]]] for s in episode["state"]]
    hs, he = d["hold_start"], d["hold_end"]
    tol = cfg["boundary_tolerance_s"]
    closings = [m for m in motions if m["direction"] < 0 and
                candidate["start"] <= m["start"] < m["end"] <= candidate["end"] and
                abs(times[m["end"]] - times[hs]) <= tol + 1e-9]
    proposed = candidate.get("closing_start")
    if proposed is not None:
        closings = [m for m in closings if m["start"] == proposed]
    closing = min(closings, key=lambda m: (abs(times[m["end"]] - times[hs]), -m["end"]), default=None)
    if closing:
        cs, contact = closing["start"], closing["end"]
    elif candidate["start"] == 0 and hs == 0 and 0 in d["evidence_frames"]:
        cs, contact = None, 0
    else:
        return None, "missing_aligned_closing_boundary"
    opening = None
    if d["category"] == "grasp_release":
        openings = [m for m in motions if m["direction"] > 0 and
                    contact < m["start"] < m["end"] <= candidate["end"] and
                    abs(times[m["start"]] - times[he]) <= tol + 1e-9 and
                    abs(times[m["end"]] - times[d["release_end"]]) <= tol + 1e-9]
        opening = min(openings, key=lambda m: (abs(times[m["start"]] - times[he]) +
                    abs(times[m["end"]] - times[d["release_end"]])), default=None)
        if opening is None:
            return None, "missing_aligned_release_boundary"
        hold_end, end = opening["start"], opening["end"]
        if not any(i >= end for i in d["evidence_frames"]):
            return None, "missing_numeric_release_end_evidence"
    else:
        hold_end = end = he
        if end + 1 < len(values) and values[end + 1] > cfg["exit_tolerance"]:
            return None, "unconfirmed_exit_boundary"
    if times[hold_end] - times[contact] < cfg["minimum_hold_s"] - 1e-9:
        return None, "insufficient_holding_duration"
    platform_end = contact
    while platform_end + 1 <= hold_end and times[platform_end + 1] - times[contact] <= cfg["platform_window_s"] + 1e-9:
        platform_end += 1
    platform = values[contact:platform_end + 1]
    if len(platform) < 2 or max(platform) - min(platform) > cfg["platform_tolerance"]:
        return None, "unstable_contact_platform"
    for motion in (closing, opening):
        if motion and abs(values[motion["end"]] - values[motion["start"]]) / (times[motion["end"]] - times[motion["start"]]) > cfg["maximum_motion_speed"]:
            return None, "unreliable_motion_speed"
    curve, phases = list(values), ["unchanged"] * len(values)
    start = contact if cs is None else cs
    opening_target = values[cs] if cs is not None else None
    if cs is not None and d["pregrasp_clear"]:
        ps = d["pregrasp_start"]
        if ps > cs:
            return None, "pregrasp_boundary_after_closing"
        highest = max(values[ps:cs + 1])
        peaks = [i for i in range(ps, cs + 1) if highest - values[i] <= cfg["pregrasp_plateau_tolerance"]]
        if len(peaks) >= cfg["pregrasp_min_plateau_frames"]:
            # At the episode boundary there is no earlier target to join.
            # A confirmed free prefix may use the supported open plateau even
            # when the initial opening was lower than that later plateau.
            opening_target = values[peaks[0]]
            start = 0 if ps == 0 else peaks[0]
            for i in range(start, cs + 1):
                curve[i], phases[i] = opening_target, "pregrasp"
    if cs is not None:
        beginning, connection = values[cs], values[contact]
        if beginning - connection < cfg["motion_min_delta"]:
            return None, "insufficient_closing_excursion"
        phase = 0.0
        for i in range(cs, contact + 1):
            phase = max(phase, min(1.0, max(0.0, (beginning - values[i]) / (beginning - connection))))
            curve[i], phases[i] = opening_target * (1.0 - phase), "closing"
        curve[contact] = 0.0
    for i in range(contact, hold_end + 1):
        curve[i], phases[i] = 0.0, "hold"
    if opening:
        rs, re = opening["start"], opening["end"]
        beginning, endpoint = values[rs], values[re]
        phase = 0.0
        for i in range(rs, re + 1):
            phase = max(phase, min(1.0, max(0.0, (values[i] - beginning) / (endpoint - beginning))))
            curve[i], phases[i] = endpoint * phase, "release"
        curve[re] = endpoint
    source = d["review_source"]
    accepted = source == "human" or (source == "model" and cfg["auto_accept_model"])
    return dict(curve=curve, phases=phases, start=start, end=end, contact_frame=contact,
                closing_start=cs, hold_end=hold_end, release_start=opening["start"] if opening else None,
                release_end=opening["end"] if opening else None, source=source, accepted=accepted,
                platform_opening=median(platform), opening_target=opening_target,
                pregrasp_status=("cleaned" if start == d.get("pregrasp_start") else "partially_cleaned")
                if cs is not None and start < cs else "not_modified"), None


def generate_cycle_targets(episode, candidates, decisions, config=None):
    """Return aligned commands with per-frame validity, source, phase, and coverage."""
    validate_episode(episode)
    cfg = cycle_config(config)
    size = len(episode["timestamps"])
    original = [[s[9], s[19]] for s in episode["state"]]
    target, preview = [r[:] for r in original], [r[:] for r in original]
    valid = [[False, False] for _ in original]
    phases = [["unchanged", "unchanged"] for _ in original]
    preview_phases = [r[:] for r in phases]
    sources = [["", ""] for _ in original]
    preview_sources = [r[:] for r in sources]
    by_id, counts = {}, Counter()
    for d in decisions:
        if isinstance(d, dict) and isinstance(d.get("candidate_id"), str):
            by_id[d["candidate_id"]] = d
            counts[d["candidate_id"]] += 1
    candidate_counts = Counter(c.get("candidate_id") for c in candidates if isinstance(c, dict))
    reviewed = {side: [False] * size for side in episode["valid_sides"]}
    motion_by_side = {side: _motions([r[SIDE_INDICES[side]] for r in episode["state"]], episode["timestamps"], cfg)
                      for side in episode["valid_sides"]}
    events, proposals, checked = [], [], []
    for candidate in candidates:
        event = dict(candidate) if isinstance(candidate, dict) else {}
        event.update(status="unchanged", reason="missing_visual_decision", changed_frames=0, reviewed=False)
        events.append(event)
        key = event.get("candidate_id")
        if counts[key] > 1 or candidate_counts[key] > 1:
            event["reason"] = "duplicate_candidate_or_decision"
            continue
        d = by_id.get(key)
        if d is None:
            continue
        try:
            _validate(candidate, d, episode)
        except (ValueError, TypeError, KeyError) as exc:
            event.update(reason="invalid_visual_decision", detail=str(exc))
            continue
        event["decision"] = dict(d)
        side = candidate["side"]
        if not _cameras(episode, side):
            event["reason"] = "missing_same_side_camera"
            continue
        event["reviewed"] = True
        reviewed[side][candidate["start"]:candidate["end"] + 1] = [True] * (candidate["end"] - candidate["start"] + 1)
        checked.append((candidate, d))
        proposal, reason = _proposal(episode, candidate, d, cfg, motion_by_side[side])
        if proposal is None:
            event["reason"] = reason
        else:
            proposals.append((event, proposal, d))
    # Decide conflicts before applying anything, so input ordering cannot choose a winner.
    conflicts = set()
    checked_by_id = {c["candidate_id"]: d for c, d in checked}
    for index, (event, p, d) in enumerate(proposals):
        # A later free-space claim cannot erase an earlier unreviewed or uncertain
        # grasp cue. Preserve preparation if that is the only affected phase;
        # conflicts inside the grasp itself reject the entire proposed command.
        for other in candidates:
            if (not isinstance(other, dict) or other.get("side") != event["side"] or
                    other.get("candidate_id") == event["candidate_id"]):
                continue
            other_decision = checked_by_id.get(other.get("candidate_id"))
            if other_decision and other_decision["category"] != "uncertain":
                continue
            a = other.get("closing_start")
            if a is None:
                a = other.get("start")
            b = other.get("release_end")
            if b is None:
                b = other.get("end")
            if type(a) is not int or type(b) is not int or not 0 <= a <= b < size:
                continue
            if p["start"] <= b and a <= p["end"]:
                cs = p["closing_start"]
                if cs is not None and b <= cs:
                    values = [r[SIDE_INDICES[event["side"]]] for r in episode["state"]]
                    for i in range(p["start"], cs):
                        p["curve"][i], p["phases"][i] = values[i], "unchanged"
                    scale = values[cs] / p["opening_target"]
                    for i in range(cs, p["contact_frame"] + 1):
                        p["curve"][i] *= scale
                    p.update(start=cs, opening_target=values[cs], pregrasp_status="unresolved_context_preserved")
                else:
                    conflicts.add(index)
        for other_candidate, other in checked:
            if other is d or other["side"] != event["side"] or other["category"] not in ("grasp_release", "grasp_hold"):
                continue
            a = other.get("hold_start")
            b = other.get("release_end") if other["category"] == "grasp_release" else other.get("hold_end")
            if a is not None and b is not None and p["start"] <= b and a <= p["end"]:
                # An already-open release endpoint can be the next preparation's
                # unchanged anchor. No interior part of the earlier cycle is shared.
                shared_open_endpoint = (p["start"] == b and other["category"] == "grasp_release" and
                    abs(p["curve"][b] - episode["state"][b][SIDE_INDICES[event["side"]]]) <= 1e-12)
                if not shared_open_endpoint:
                    conflicts.add(index)
        for other_index in range(index):
            other_event, q, _ = proposals[other_index]
            if other_event["side"] != event["side"]:
                continue
            a, b = max(p["start"], q["start"]), min(p["end"], q["end"])
            if a < b or (a == b and abs(p["curve"][a] - q["curve"][a]) > 1e-12):
                conflicts.update((index, other_index))
    for index, (event, p, d) in enumerate(proposals):
        if index in conflicts:
            event["reason"] = "overlapping_independent_grasp_or_release"
            continue
        col = 0 if event["side"] == "left" else 1
        for i in range(p["start"], p["end"] + 1):
            preview[i][col] = p["curve"][i]
            preview_phases[i][col] = p["phases"][i]
            preview_sources[i][col] = p["source"]
            if p["accepted"]:
                target[i][col] = p["curve"][i]
                valid[i][col] = True
                phases[i][col] = p["phases"][i]
                sources[i][col] = p["source"]
        event.update(status="applied" if p["accepted"] else "preview",
                     reason="automatic_visual_numeric_cycle" if p["source"] == "model" and p["accepted"] else
                            "human_reviewed_cycle" if p["accepted"] else "synthetic_evidence" if p["source"] == "synthetic" else "model_review_only",
                     source=p["source"], modified_start=p["start"], modified_end=p["end"],
                     changed_frames=sum(abs(p["curve"][i] - original[i][col]) > 1e-12 for i in range(p["start"], p["end"] + 1)),
                     contact_frame=p["contact_frame"], closing_start=p["closing_start"],
                     hold_end=p["hold_end"], release_start=p["release_start"], release_end=p["release_end"],
                     platform_opening=p["platform_opening"], opening_target=p["opening_target"],
                     pregrasp_status=p["pregrasp_status"],
                     zero_frame=p["contact_frame"], closing_policy="fixed_timing", release_policy="fixed_timing")
    coverage = {}
    for side, state_index in SIDE_INDICES.items():
        col = 0 if side == "left" else 1
        selected = [e for e in events if e.get("side") == side]
        flags = reviewed.get(side, [False] * size)
        unreviewed_count = sum(not e["reviewed"] for e in selected)
        coverage[side] = dict(valid_side=side in episode["valid_sides"], candidates=len(selected),
            reviewed_candidates=sum(e["reviewed"] for e in selected),
            accepted_cycles=sum(e["status"] == "applied" for e in selected),
            preview_cycles=sum(e["status"] == "preview" for e in selected),
            rejected_cycles=sum(e["reviewed"] and e["status"] == "unchanged" and e["reason"] != "no_grasp" for e in selected),
            no_grasp_candidates=sum(e["reason"] == "no_grasp" for e in selected),
            unreviewed_candidates=unreviewed_count,
            review_window_frames=sum(flags), reviewed_frames=sum(flags), unreviewed_intervals=_intervals(flags),
            coverage_definition=("review_window_frames and legacy reviewed_frames count the union of input context windows "
                "for candidates with a valid structured judgment; they are not per-frame visual truth or cleaned-label coverage. "
                "complete_review requires every candidate to have such a judgment and the windows to cover the episode. "
                "A no_grasp judgment concerns its candidate, not all earlier grasps visible in its context."),
            target_valid_frames=sum(row[col] for row in valid),
            changed_frames=sum(abs(row[col] - episode["state"][i][state_index]) > 1e-12 for i, row in enumerate(target)),
            complete_review=side in episode["valid_sides"] and unreviewed_count == 0 and all(flags))
    return dict(target=target, preview_target=preview, valid=valid, phase=phases, source=sources,
                preview_phase=preview_phases, preview_source=preview_sources,
                events=events, config=cfg, coverage=coverage, closing_comparisons=[], release_comparisons=[],
                side_status={s: "invalid" if s not in episode["valid_sides"] else "reviewed" if coverage[s]["complete_review"] else "incomplete_review" for s in SIDE_INDICES})
