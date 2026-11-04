"""Build task calibration from a complete census and explicit visual reviews.

No image judgment is invented here. A numerical match is not a grasp label,
and this tool never modifies observations, fits curves or publishes data.
"""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.pipeline import create_run
from close_gripper.cycles import CYCLE_DEFAULTS, _motions
from close_gripper.task_profiles import CALIBRATION_CONTRACT, SCHEMA, UNITS, validate_catalog


PADDING = 0.01
MODE_REVIEW_SCHEMA = "close_gripper.mode_reviews.v1"
EVIDENCE_NOISE = CYCLE_DEFAULTS["motion_noise"]
_EPS = 1e-9


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def overlaps(first, second):
    return max(first["lower"], second["lower"]) <= min(first["upper"], second["upper"])


def merge_bands(bands, prefix):
    """Union only intersecting measured bands; never fill an unobserved gap."""
    merged = []
    for original in sorted(bands, key=lambda b: (b["lower"], b["upper"])):
        band = copy.deepcopy(original)
        if merged and overlaps(merged[-1], band):
            previous = merged[-1]
            previous["upper"] = max(previous["upper"], band["upper"])
            for key in ("case_ids", "source_ids", "numeric_mode_ids", "calibration_centers"):
                previous[key] = sorted(set(previous[key] + band[key]))
            previous["basis"] = sorted(set(previous["basis"] + band["basis"]))
        else:
            merged.append(band)
    for i, band in enumerate(merged):
        band["id"] = f"{prefix}_{i:02d}"
        band["center"] = median(band["calibration_centers"])
        band["reviewed_source_count"] = len(band["source_ids"])
    return merged


def closed_band(review, mapping, modes):
    for key in ("task", "side", "source_id", "dataset_root", "episode_index"):
        if review[key] != mapping[key]:
            raise ValueError("visual review and numerical mapping identity disagree")
    if (review["plateau_start"], review["plateau_end"]) != (mapping["v1_start"], mapping["v1_end"]):
        raise ValueError("visual review and numerical mapping interval disagree")
    # A partially intersecting window could be an adjacent, different event.
    containing = [w for w in mapping["v2_overlapping_windows"] if w["contains_v1_window"]]
    if not containing:
        raise ValueError("reviewed interval has no containing numerical census window")
    assignments = {(w["numeric_role"], w["mode_id"]) for w in containing if w["mode_id"] is not None}
    low, high = max(0.0, review["minimum"] - PADDING), min(1.0, review["maximum"] + PADDING)
    mode_ids, basis = [], ["visually_supported_sample_range"]
    if len(assignments) == 1:
        role, mode_id = next(iter(assignments))
        mode = modes[(role, mode_id)]
        low, high = min(low, mode["lower"]), max(high, mode["upper"])
        mode_ids = [role + ":" + mode_id]
        basis.append("source_equal_weight_mode_p05_p95_plus_0.01")
    elif len(assignments) > 1:
        basis.append("multiple_mode_assignments_sample_only")
    else:
        basis.append("no_population_mode_sample_only")
    return dict(lower=low, upper=high, calibration_centers=[review["center"]],
                case_ids=[review["case_id"]], source_ids=[review["source_id"]],
                numeric_mode_ids=mode_ids, basis=basis)


def build_profile(stat, membership, reviews, mappings):
    modes = {(role, m["id"]): m for role, group in membership["roles"].items() for m in group["modes"]}
    supported, negative, free, explicit_free = [], [], [], []
    for r in reviews:
        if r["review_source"] != "model" or r["review_method"] != "codex_native_vision" or r["confirmed"]:
            raise ValueError("this calibration requires explicit non-human native vision provenance")
        if r["visual_grasp_status"] not in ("supported", "not_supported", "uncertain"):
            raise ValueError("invalid visual review status")
        if any(not finite_number(r[k]) for k in ("minimum", "center", "maximum", "duration_s")):
            raise ValueError("calibration values must be finite numbers")
        if not 0 <= r["minimum"] <= r["center"] <= r["maximum"] <= 1:
            raise ValueError("calibration range must contain its center in normalized units")
        if r["duration_s"] + 1e-9 < 0.6 or r["maximum"] - r["minimum"] > 0.02 + 1e-9:
            raise ValueError("calibration interval does not meet the frozen stability rule")
        panels = {p["index"]: p for p in r["panels"]}
        if not set(r["holding_evidence_frames"] + r["open_evidence_frames"]) <= set(panels):
            raise ValueError("review cites an unviewed frame")
        if any(not r["plateau_start"] <= i <= r["plateau_end"] for i in r["holding_evidence_frames"]):
            raise ValueError("holding calibration frame lies outside the calibrated interval")
        for index in r.get("explicit_free_within_plateau", []):
            if index not in panels or not r["plateau_start"] <= index <= r["plateau_end"]:
                raise ValueError("explicit free evidence must be viewed and within the calibrated interval")
            value = panels[index]["opening"]
            explicit_free.append(dict(case_id=r["case_id"], frame=index,
                                      lower=max(0.0, value-PADDING), upper=min(1.0, value+PADDING)))
        if r["closed_mode_support"]:
            if r["visual_grasp_status"] != "supported" or not r["holding_evidence_frames"]:
                raise ValueError("closed mode lacks positive visual evidence")
            supported.append(r)
        if r["visual_grasp_status"] == "not_supported" or r.get("mixed_hold_free", False):
            negative.append(r)
        if r["open_mode_support"]:
            for index in r["open_evidence_frames"]:
                free.append(dict(case_id=r["case_id"], source_id=r["source_id"], frame=index,
                                 opening=panels[index]["opening"]))
    proposed = merge_bands([closed_band(r, mappings[r["sample_id"]], modes) for r in supported], "closed")
    closed, rejected = [], []
    for band in proposed:
        conflicts = [r["case_id"] for r in negative
                     if overlaps(band, dict(lower=r["minimum"], upper=r["maximum"]))]
        conflicts += [r["case_id"] for r in explicit_free if overlaps(band, r)]
        conflicts = sorted(set(conflicts))
        if conflicts:
            rejected.append(dict(band, rejection_reason="same_opening_has_non_grasp_or_mixed_evidence",
                                 counterexample_case_ids=conflicts))
        else:
            closed.append(band)
    # Free frames can be empty-but-closed. Keep them in evidence; only higher
    # observed openings provide an open anchor. Never extend a moving frame to
    # an entire numerical upper mode or silently force the target to one.
    open_candidates = []
    highest_closed = max((b["upper"] for b in closed), default=None)
    if highest_closed is not None:
        for f in free:
            lo, hi = max(0.0, f["opening"] - PADDING), min(1.0, f["opening"] + PADDING)
            if lo <= highest_closed:
                continue
            open_candidates.append(dict(lower=lo, upper=hi, calibration_centers=[f["opening"]],
                case_ids=[f["case_id"]], source_ids=[f["source_id"]], numeric_mode_ids=[],
                basis=["local_visual_free_open_frame_plus_0.01_not_a_population_mode"]))
    opened = merge_bands(open_candidates, "open")
    reasons = []
    if not supported:
        reasons.append("no_supported_holding_sample")
    if rejected:
        reasons.append("conflicting_closed_bands_excluded")
    if not closed:
        reasons.append("no_unambiguous_closed_band")
    if not opened:
        reasons.append("no_separate_higher_visual_open_anchor")
    automatic = bool(closed and opened)
    status = ("visual_reviewed" if automatic else "unsupported"
              if reviews and all(r["visual_grasp_status"] == "not_supported" for r in reviews) else "needs_review")
    selected = {mid for b in closed for mid in b["numeric_mode_ids"]}
    counts = dict(episodes=stat["episode_count"], unique_sources=stat["unique_source_count"],
        frames=stat["frame_count"], stable_windows=stat["stable_0_6s_count"],
        reviewed_cases=len(reviews), reviewed_sources=len({r["source_id"] for r in reviews}),
        supported_cases=len(supported), not_supported_cases=sum(r["visual_grasp_status"] == "not_supported" for r in reviews),
        uncertain_cases=sum(r["visual_grasp_status"] == "uncertain" for r in reviews),
        mixed_hold_free_cases=sum(bool(r.get("mixed_hold_free")) for r in reviews),
        explicit_free_frames_within_plateau=len(explicit_free),
        observed_free_frames=len(free), selected_numeric_modes=len(selected),
        total_numeric_modes=len(modes), rejected_closed_bands=len(rejected))
    return dict(status=status, automatic_use=automatic, automatic_use_scope="numeric_interval_matching_only",
        closed_bands=closed, open_bands=opened, rejected_closed_bands=rejected, reasons=reasons,
        counts=counts, uncalibrated_numeric_mode_ids=sorted(role + ":" + mid for role, mid in modes
                                                          if role + ":" + mid not in selected),
        provenance=dict(review_source="model", review_method="codex_native_vision", confirmed=False,
                        case_ids=[r["case_id"] for r in reviews],
                        independent_semantic_validation=False, full_dataset_visually_reviewed=False))


def _mode_record(record, mapping, modes):
    """Check identity and exact observations before deriving any semantic range.

    Numeric role ``mixed`` is census topology, not a visual classification.
    Sparse visual observations are model evidence, never human ground truth.
    """
    r = copy.deepcopy(record)
    required = ("case_id", "sample_id", "task", "side", "source_id", "dataset_root", "episode_index",
                "numeric_role", "numeric_mode_id", "plateau_start", "plateau_end", "duration_s",
                "minimum", "center", "maximum")
    if any(key not in r or key not in mapping for key in required if key != "case_id"):
        raise ValueError("mode review and mapping require explicit identity and interval fields")
    if any(not isinstance(r.get(k), str) or not r[k].strip()
           for k in ("case_id", "sample_id", "task", "source_id", "dataset_root")):
        raise ValueError("mode evidence requires nonempty original identity")
    if r["side"] not in ("left", "right") or type(r["episode_index"]) is not int or r["episode_index"] < 0:
        raise ValueError("mode evidence requires valid side and original episode index")
    for key in ("task", "side", "source_id", "dataset_root", "episode_index",
                "numeric_role", "numeric_mode_id"):
        if r.get(key) != mapping.get(key):
            raise ValueError("mode review and numerical mapping identity disagree: " + key)
    for key in ("plateau_start", "plateau_end", "duration_s", "minimum", "center", "maximum"):
        if r.get(key) != mapping.get(key):
            raise ValueError("mode review and numerical mapping interval disagree: " + key)
    if r["numeric_role"] not in ("lower", "upper", "static", "mixed"):
        raise ValueError("unknown numerical role")
    if r["numeric_mode_id"] is not None and (r["numeric_role"], r["numeric_mode_id"]) not in modes:
        raise ValueError("unknown numerical mode identity")
    if (r.get("review_source") != "model" or r.get("review_method") != "codex_native_vision"
            or r.get("confirmed") is not False):
        raise ValueError("mode calibration requires non-human native vision provenance")
    if type(r.get("source_alignment_uncertain", False)) is not bool:
        raise ValueError("source_alignment_uncertain must be boolean")
    if any(not finite_number(r[k]) for k in ("duration_s", "minimum", "center", "maximum")):
        raise ValueError("mode calibration values must be finite")
    if (not 0 <= r["minimum"] <= r["center"] <= r["maximum"] <= 1
            or r["duration_s"] + _EPS < .6 or r["maximum"] - r["minimum"] > .02 + _EPS):
        raise ValueError("mode calibration interval violates stability policy")
    a, b = r["plateau_start"], r["plateau_end"]
    if type(a) is not int or type(b) is not int or not 0 <= a < b:
        raise ValueError("mode calibration requires inclusive frame indices")
    panels = {}
    for panel in r.get("panels", []):
        i, value = panel.get("index"), panel.get("opening")
        if type(i) is not int or i < 0 or not finite_number(value) or not 0 <= value <= 1:
            raise ValueError("invalid viewed frame opening")
        if i in panels and abs(panels[i]["opening"] - value) > _EPS:
            raise ValueError("stereo panels disagree on opening")
        camera = panel.get("camera")
        if camera is not None and "cam_" + r["side"] + "_wrist_" not in camera:
            raise ValueError("visual evidence uses the wrong side camera")
        panels[i] = panel
    states = {}
    for frame in r.get("frame_states", []):
        i, state = frame.get("index"), frame.get("state")
        if type(i) is not int or i not in panels or i in states:
            raise ValueError("frame state must identify one viewed frame")
        if state not in ("held", "free", "uncertain"):
            raise ValueError("unknown frame semantic state")
        if "opening" in frame and (not finite_number(frame["opening"])
                                    or abs(frame["opening"] - panels[i]["opening"]) > _EPS):
            raise ValueError("frame semantic opening differs from observed opening")
        frame["opening"] = panels[i]["opening"]
        states[i] = state
    if not {a, (a + b) // 2, b} <= set(states):
        raise ValueError("stable platform requires start, middle and end visual states")
    uncertain = r.get("boundary_uncertain", [])
    if not isinstance(uncertain, list) or any(type(i) is not int or i not in panels for i in uncertain):
        raise ValueError("uncertain boundaries must cite viewed frames")
    inside = {states[i] for i in states if a <= i <= b}
    semantics = ("mixed" if {"held", "free"} <= inside else
                 "held" if inside == {"held"} else "free" if inside == {"free"} else "uncertain")
    if "interval_semantics" in r and r["interval_semantics"] != semantics:
        raise ValueError("declared interval semantics disagree with viewed frame states")
    r["interval_semantics"] = semantics
    trace = r.get("trace")
    if trace is not None:
        times, values = trace.get("timestamps"), trace.get("openings")
        if (not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values)
                or len(values) <= max(panels) or any(not finite_number(t) for t in times)
                or any(not finite_number(v) or not 0 <= v <= 1 for v in values)
                or any(y <= x for x, y in zip(times, times[1:]))):
            raise ValueError("invalid full-episode numerical trace")
        if (abs(times[b] - times[a] - r["duration_s"]) > _EPS
                or abs(min(values[a:b + 1]) - r["minimum"]) > _EPS
                or abs(max(values[a:b + 1]) - r["maximum"]) > _EPS
                or any(y - x > .15 + _EPS for x, y in zip(times[a:b], times[a + 1:b + 1]))):
            raise ValueError("mode trace and stable interval metadata disagree")
        for i, panel in panels.items():
            if abs(values[i] - panel["opening"]) > _EPS:
                raise ValueError("panel and numerical trace openings disagree")
            if "timestamp" in panel and (not finite_number(panel["timestamp"])
                    or abs(times[i] - panel["timestamp"]) > _EPS):
                raise ValueError("panel and numerical trace timestamps disagree")
    r["_panels"], r["_states"] = panels, states
    return r


def _semantic_spans(r, state):
    """Only explicitly supported subsegments, with no invented mixed boundary."""
    a, b = r["plateau_start"], r["plateau_end"]
    key = state + "_intervals"
    supplied = r.get(key)
    if supplied is None:
        supplied = [dict(start=a, end=b)] if r["interval_semantics"] == state else []
    if not isinstance(supplied, list):
        raise ValueError(key + " must be a list")
    result = []
    for span in supplied:
        x, y = span.get("start"), span.get("end")
        if (type(x) is not int or type(y) is not int or not a <= x <= y <= b
                or r["_states"].get(x) != state or r["_states"].get(y) != state):
            raise ValueError("semantic interval endpoints must be viewed with the declared state")
        if any(s != state for i, s in r["_states"].items() if x <= i <= y):
            raise ValueError("semantic interval crosses conflicting or uncertain visual evidence")
        if any(x <= i <= y for i in r.get("boundary_uncertain", [])):
            raise ValueError("semantic interval crosses an uncertain boundary")
        if r.get("trace") is not None:
            times, values = r["trace"]["timestamps"], r["trace"]["openings"]
            duration, low, high = times[y] - times[x], min(values[x:y + 1]), max(values[x:y + 1])
            if any(v - u > .15 + _EPS for u, v in zip(times[x:y], times[x + 1:y + 1])):
                raise ValueError("semantic interval crosses a timestamp gap")
        elif (x, y) == (a, b):
            duration, low, high = r["duration_s"], r["minimum"], r["maximum"]
        else:
            raise ValueError("partial semantic interval requires the actual numerical trace")
        if high - low > .02 + _EPS:
            raise ValueError("semantic interval is not numerically stable")
        result.append(dict(start=x, end=y, duration_s=duration, lower=low, upper=high,
                           qualifies_hold=state == "held" and duration + _EPS >= .6))
    return result


def _evidence_band(r, low, high, basis):
    mode = r["numeric_role"] + ":" + r["numeric_mode_id"] if r["numeric_mode_id"] is not None else None
    return dict(lower=max(0., low - EVIDENCE_NOISE), upper=min(1., high + EVIDENCE_NOISE),
        calibration_centers=[(low + high) / 2], case_ids=[r["case_id"]], source_ids=[r["source_id"]],
        numeric_mode_ids=[mode] if mode else [], basis=[basis])


def _slice_evidence_band(lower, upper, evidence):
    """Recompute support from original evidence that intersects this slice."""
    contributing = [b for b in evidence if max(lower, b["lower"]) <= min(upper, b["upper"])]
    if not contributing:
        raise ValueError("a calibrated sub-band requires actual held evidence")
    result = dict(lower=lower, upper=upper)
    for key in ("case_ids", "source_ids", "numeric_mode_ids", "calibration_centers", "basis"):
        result[key] = sorted({v for b in contributing for v in b[key]})
    result["center"] = min(upper, max(lower, median(result["calibration_centers"])))
    result["reviewed_source_count"] = len(result["source_ids"])
    return result


def _subtract_conflict_intervals(band, conflicts):
    """Subtract inclusive conflicts using exact adjacent floating-point values."""
    slices = [(band["lower"], band["upper"])]
    for conflict in conflicts:
        remainder = []
        for lower, upper in slices:
            if upper < conflict["lower"] or lower > conflict["upper"]:
                remainder.append((lower, upper))
                continue
            if lower < conflict["lower"]:
                remainder.append((lower, math.nextafter(conflict["lower"], -math.inf)))
            if upper > conflict["upper"]:
                remainder.append((math.nextafter(conflict["upper"], math.inf), upper))
        slices = remainder
    return slices


def _two_source_intersections(evidence):
    """Every retained value has overlapping evidence from distinct recordings."""
    intersections = []
    for index, first in enumerate(evidence):
        for second in evidence[index + 1:]:
            if first["source_ids"] == second["source_ids"] or not overlaps(first, second):
                continue
            low, high = max(first["lower"], second["lower"]), min(first["upper"], second["upper"])
            intersection = _slice_evidence_band(low, high, [first, second])
            intersection["basis"] = sorted(set(intersection["basis"] + ["pointwise_two_distinct_source_intersection"]))
            intersections.append(intersection)
    result = merge_bands(intersections, "two_source")
    for band in result:
        band["center"] = min(band["upper"], max(band["lower"], band["center"]))
    return result


def _release_rules(records, closed, opened):
    """Learn only band-local release thresholds from same-record held -> free."""
    evidence, rejected = defaultdict(list), []
    discovery = dict(CYCLE_DEFAULTS, motion_min_delta=2 * EVIDENCE_NOISE + _EPS)
    open_by_id = {b["id"]: b for b in opened}
    for r in records:
        if r.get("source_alignment_uncertain", False):
            if r.get("release_pairs"):
                rejected.append(dict(case_id=r["case_id"], reason="source_alignment_uncertain"))
            continue
        for pair in r.get("release_pairs", []):
            a, b = pair.get("held_frame"), pair.get("free_frame")
            if (type(a) is not int or type(b) is not int or a >= b
                    or r["_states"].get(a) != "held" or r["_states"].get(b) != "free"):
                raise ValueError("release pair requires ordered, viewed held and free frames")
            if r.get("trace") is None:
                rejected.append(dict(case_id=r["case_id"], reason="release_pair_missing_trace"))
                continue
            times, values = r["trace"]["timestamps"], r["trace"]["openings"]
            if any(y - x > .15 + _EPS for x, y in zip(times[a:b], times[a + 1:b + 1])):
                rejected.append(dict(case_id=r["case_id"], reason="release_pair_timestamp_gap"))
                continue
            motions = _motions(values[a:b + 1], times[a:b + 1], discovery)
            if len(motions) != 1 or motions[0]["direction"] != 1:
                rejected.append(dict(case_id=r["case_id"], reason="release_pair_not_one_opening_motion"))
                continue
            motion = motions[0]
            start, end = a + motion["start"], a + motion["end"]
            band = next((c for c in closed if c["lower"] <= values[a] <= c["upper"]
                         and c["lower"] <= values[start] <= c["upper"]), None)
            refs = [] if band is None else [i for i in band["open_band_ids"]
                if open_by_id[i]["lower"] <= values[b] <= open_by_id[i]["upper"]
                and values[end] + _EPS >= open_by_id[i]["lower"]]
            if not refs:
                rejected.append(dict(case_id=r["case_id"], reason="release_pair_outside_usable_band_anchors"))
                continue
            evidence[band["id"]].append(dict(case_id=r["case_id"], source_id=r["source_id"],
                dataset_root=r["dataset_root"], episode_index=r["episode_index"],
                held_frame=a, free_frame=b, held_opening=values[a], free_opening=values[b],
                motion_start=start, motion_end=end, start_opening=values[start], end_opening=values[end],
                observed_excursion=values[end] - values[start], open_band_ids=refs,
                review_source="model", confirmed=False))
    rules = []
    for band_id, items in sorted(evidence.items()):
        observed = min(e["observed_excursion"] for e in items)
        delta = max(2 * EVIDENCE_NOISE + _EPS, min(CYCLE_DEFAULTS["motion_min_delta"], observed - EVIDENCE_NOISE))
        rules.append(dict(id="release_" + band_id, closed_band_id=band_id,
            open_band_ids=sorted({i for e in items for i in e["open_band_ids"]}),
            motion_min_delta=delta, motion_noise=EVIDENCE_NOISE,
            threshold_policy="max_2noise_min_default_observed_excursion_minus_noise",
            evidence=items, reviewed_source_count=len({e["source_id"] for e in items}),
            evaluation_only=True, preserves_fixed_timing=True))
    return rules, rejected


def build_mode_profile(stat, membership, reviews, mappings):
    """All-role review with local observed semantics and independent support."""
    modes = {(role, m["id"]): m for role, group in membership["roles"].items() for m in group["modes"]}
    records = [_mode_record(r, mappings[r["sample_id"]], modes) for r in reviews]
    excluded_sources = {r["source_id"] for r in records if r.get("source_alignment_uncertain", False)}
    for r in records:
        r["source_alignment_uncertain"] = r["source_id"] in excluded_sources
    positives, free, observed, span_evidence = [], [], [], []
    for r in records:
        held_spans, free_spans = _semantic_spans(r, "held"), _semantic_spans(r, "free")
        span_evidence.append(dict(case_id=r["case_id"], sample_id=r["sample_id"], source_id=r["source_id"],
            numeric_role=r["numeric_role"], numeric_mode_id=r["numeric_mode_id"],
            interval_semantics=r["interval_semantics"], frame_states=r["frame_states"],
            held_intervals=held_spans, free_intervals=free_spans,
            boundary_uncertain=r.get("boundary_uncertain", []),
            source_alignment_uncertain=r["source_alignment_uncertain"],
            numeric_calibration_eligible=not r["source_alignment_uncertain"]))
        if r["source_alignment_uncertain"]:
            continue
        for span in held_spans:
            if span["qualifies_hold"]:
                positives.append(_evidence_band(r, span["lower"], span["upper"], "reviewed_held_span_plus_motion_noise"))
        for span in free_spans:
            free.append(_evidence_band(r, span["lower"], span["upper"], "reviewed_free_span_plus_motion_noise"))
        # Individual free observations are real counterexamples, including an
        # empty fully closed gripper. A mixed label alone supplies no range.
        for i, state in r["_states"].items():
            value = r["_panels"][i]["opening"]
            if state == "free":
                free.append(_evidence_band(r, value, value, "reviewed_free_frame_plus_motion_noise"))
            if state == "held":
                observed.append(_evidence_band(r, value, value, "reviewed_held_frame_plus_motion_noise"))
    proposed = merge_bands(positives, "closed")
    independent_support = _two_source_intersections(positives)
    free_bands = merge_bands(free, "free")
    conflict_ranges = []
    for h in observed + positives:
        for f in free:
            if not overlaps(h, f):
                continue
            low, high = max(h["lower"], f["lower"]), min(h["upper"], f["upper"])
            intersection = copy.deepcopy(h)
            intersection.update(lower=low, upper=high,
                calibration_centers=[min(high, max(low, c)) for c in h["calibration_centers"]],
                basis=sorted(set(h["basis"] + ["actual_held_free_band_intersection"])))
            conflict_ranges.append(intersection)
    interval_vision = merge_bands(conflict_ranges, "interval_vision")
    for b in interval_vision:
        b["reason"] = "same_opening_has_held_and_free_evidence"
        b["counterexample_case_ids"] = sorted({c for f in free_bands if overlaps(b, f) for c in f["case_ids"]})
    closed, rejected, fragment_index = [], [], 0
    for band in proposed:
        conflicts = [b for b in interval_vision if overlaps(band, b)]
        for conflict in conflicts:
            rejected.append(dict(_slice_evidence_band(max(band["lower"], conflict["lower"]),
                min(band["upper"], conflict["upper"]), positives), id=band["id"] + "_" + conflict["id"],
                parent_band_id=band["id"], rejection_reason="requires_interval_vision",
                counterexample_case_ids=conflict["counterexample_case_ids"]))
        for low, high in _subtract_conflict_intervals(band, conflicts):
            remainder = dict(lower=low, upper=high)
            supported = [dict(lower=max(low, supported["lower"]), upper=min(high, supported["upper"]))
                         for supported in independent_support if overlaps(remainder, supported)]
            ranges = [(b["lower"], b["upper"], True) for b in supported]
            ranges += [(a, b, False) for a, b in _subtract_conflict_intervals(remainder, supported)]
            for a, b, usable in sorted(ranges):
                fragment = _slice_evidence_band(a, b, positives)
                fragment.update(id=f"closed_{fragment_index:02d}", parent_band_id=band["id"],
                    excluded_interval_vision_ids=[b["id"] for b in conflicts])
                fragment_index += 1
                if usable:
                    fragment["minimum_pointwise_source_count"] = 2
                    fragment["basis"] = sorted(set(fragment["basis"] + ["pointwise_two_distinct_source_intersection"]))
                    closed.append(fragment)
                else:
                    rejected.append(dict(fragment, rejection_reason="insufficient_independent_holding_sources",
                                         counterexample_case_ids=[]))
    # Preserve all observed free states. Open anchors are separately bound to
    # the closed mode they can release; never use a global highest closed mode.
    def linked(f, c):
        return f["lower"] > c["upper"] and bool(set(f["source_ids"]).intersection(c["source_ids"]))

    open_candidates = [f for f in free_bands if any(linked(f, c) for c in closed)]
    opened = merge_bands(open_candidates, "open")
    usable = []
    for band in closed:
        refs = [o["id"] for o in opened if linked(o, band)]
        if refs:
            usable.append(dict(band, open_band_ids=refs))
        else:
            rejected.append(dict(band, rejection_reason="no_separate_higher_visual_open_anchor"))
    closed = usable
    rules, rule_rejections = _release_rules(records, closed, opened)
    mode_reviews = []
    for role, mode_id in sorted(modes):
        rs = [r for r in records if (r["numeric_role"], r["numeric_mode_id"]) == (role, mode_id)]
        case_ids = {r["case_id"] for r in rs}
        platform_states = {s for r in rs for i, s in r["_states"].items()
                           if r["plateau_start"] <= i <= r["plateau_end"]}
        has_held, has_free = "held" in platform_states, "free" in platform_states
        conflict = any(overlaps(dict(lower=r["minimum"], upper=r["maximum"]), b)
                       for r in rs if not r["source_alignment_uncertain"] for b in interval_vision)
        state = ("unreviewed" if not rs else "numeric_source_uncertain"
                 if all(r["source_alignment_uncertain"] for r in rs) else "requires_interval_vision" if conflict else
                 "held_and_free_separable" if has_held and has_free else "held_supported" if has_held else
                 "free_supported" if has_free else "uncertain")
        mode_reviews.append(dict(numeric_role=role, numeric_mode_id=mode_id, state=state,
            reviewed_case_ids=sorted(case_ids), reviewed_source_count=len({r["source_id"] for r in rs}),
            numeric_source_uncertain=any(r["source_alignment_uncertain"] for r in rs),
            excluded_case_ids=sorted(r["case_id"] for r in rs if r["source_alignment_uncertain"]),
            distribution_reference=copy.deepcopy(modes[(role, mode_id)])))
    automatic = bool(closed)
    reasons = sorted({b["rejection_reason"] for b in rejected})
    if not proposed:
        reasons.append("no_supported_stable_holding_interval")
    if any(m["state"] == "unreviewed" for m in mode_reviews):
        reasons.append("unreviewed_numeric_modes")
    if excluded_sources:
        reasons.append("source_alignment_uncertain")
    selected = {mid for b in closed for mid in b["numeric_mode_ids"]}
    counts = dict(episodes=stat["episode_count"], unique_sources=stat["unique_source_count"],
        frames=stat["frame_count"], stable_windows=stat["stable_0_6s_count"], reviewed_cases=len(records),
        reviewed_sources=len({r["source_id"] for r in records}),
        numeric_calibration_sources=len({r["source_id"] for r in records} - excluded_sources),
        excluded_alignment_sources=len(excluded_sources),
        excluded_alignment_cases=sum(r["source_alignment_uncertain"] for r in records),
        supported_cases=sum(e["numeric_calibration_eligible"] and any(s["qualifies_hold"]
                            for s in e["held_intervals"]) for e in span_evidence),
        not_supported_cases=sum(r["interval_semantics"] == "free" for r in records),
        uncertain_cases=sum(r["interval_semantics"] == "uncertain" for r in records),
        mixed_hold_free_cases=sum(r["interval_semantics"] == "mixed" for r in records),
        observed_free_frames=sum(list(r["_states"].values()).count("free") for r in records),
        total_numeric_modes=len(modes), selected_numeric_modes=len(selected),
        reviewed_numeric_modes=sum(bool(m["reviewed_case_ids"]) for m in mode_reviews),
        reviewed_unassigned_cases=sum(r["numeric_mode_id"] is None for r in records),
        rejected_closed_bands=len(rejected), interval_vision_bands=len(interval_vision))
    return dict(calibration_contract=CALIBRATION_CONTRACT, status="visual_reviewed" if automatic else
        "unsupported" if records and not excluded_sources and all(r["interval_semantics"] == "free"
            for r in records) else "needs_review",
        automatic_use=automatic, automatic_use_scope="numeric_interval_matching_only", minimum_positive_sources=2,
        positive_source_support_policy="every_value_covered_by_two_distinct_sources",
        evidence_noise=EVIDENCE_NOISE, closed_bands=closed, open_bands=opened, free_bands=free_bands,
        held_candidate_bands=proposed, rejected_closed_bands=rejected, interval_vision_bands=interval_vision,
        semantic_evidence=span_evidence, mode_reviews=mode_reviews, release_rules=rules,
        release_rule_rejections=rule_rejections, reasons=reasons, counts=counts,
        uncalibrated_numeric_mode_ids=sorted(role + ":" + mid for role, mid in modes if role + ":" + mid not in selected),
        provenance=dict(review_source="model", review_method="codex_native_vision", confirmed=False,
            case_ids=[r["case_id"] for r in records], independent_semantic_validation=False,
            full_dataset_visually_reviewed=False,
            all_numeric_mode_representatives_reviewed=all(m["reviewed_case_ids"] for m in mode_reviews)))


def build_catalog(statistics, membership, mappings, scope, reviews, version):
    all_modes = reviews.get("schema") == MODE_REVIEW_SCHEMA
    if reviews.get("example_only"):
        raise ValueError("schema examples are not calibration evidence")
    expected_parameters = dict(minimum_hold_s=0.6, maximum_sample_gap_s=0.15, plateau_peak_to_peak=0.02)
    if any(statistics.get("parameters", {}).get(key) != value for key, value in expected_parameters.items()):
        raise ValueError("census stability parameters differ from this calibration policy")
    stats = {(s["task"], s["side"]): s for s in statistics["profiles"]}
    modes = {(p["task"], p["side"]): p for p in membership["profiles"]}
    review_groups = defaultdict(list)
    case_ids = set()
    for r in reviews["records"]:
        if r["case_id"] in case_ids:
            raise ValueError("duplicate visual review case")
        case_ids.add(r["case_id"])
        review_groups[(r["task"], r["side"])].append(r)
    if all_modes:
        excluded_sources = {r["source_id"] for r in reviews["records"]
                            if r.get("source_alignment_uncertain") is True}
        review_groups = {key: [dict(r, source_alignment_uncertain=True)
                              if r["source_id"] in excluded_sources else r for r in records]
                         for key, records in review_groups.items()}
    expected = {(s["task"], s["side"]): s["episodes"] for s in scope["task_side_counts"]}
    if set(stats) != set(expected) or set(modes) != set(expected) or set(review_groups) != set(expected):
        raise ValueError("census, metadata, visual review and mode task-side coverage disagree")
    mapping = {r["sample_id"]: r for r in mappings["samples"]}
    if len(mapping) != len(mappings["samples"]) or len({r["sample_id"] for r in reviews["records"]}) != len(case_ids):
        raise ValueError("duplicate numerical or visual sample identity")
    tasks = {}
    ambiguous = scope["ambiguous_aliases"]
    for (name, side), stat in sorted(stats.items()):
        if stat["episode_count"] != expected[(name, side)]:
            raise ValueError("metadata and numerical episode counts disagree")
        if name not in tasks:
            tasks[name] = dict(task=name, aliases=[a for a in scope["aliases"][name] if a not in ambiguous], sides={})
        builder = build_mode_profile if all_modes else build_profile
        tasks[name]["sides"][side] = builder(stat, modes[(name, side)], review_groups[(name, side)], mapping)
    catalog = dict(schema=SCHEMA, version=version, units=UNITS,
        minimum_hold_s=0.6, maximum_sample_gap_s=0.15, maximum_plateau_range=0.02,
        confirmation_policy="timestamp_elapsed_then_backfill_to_original_interval_start",
        automatic_use_scope="numeric_interval_matching_only",
        generated_targets=False, fitting_method_changed=False,
        scope=dict(tasks=len(tasks), task_sides=len(stats),
                   release_episode_rows=sum(s["episodes"] for s in scope["sources"]),
                   release_frames=sum(s["frames"] for s in scope["sources"]),
                   reviewed_cases=len(case_ids), review_source="model"),
        ambiguous_aliases=ambiguous, tasks=list(tasks.values()))
    if all_modes:
        catalog.update(calibration_contract=CALIBRATION_CONTRACT, minimum_positive_sources=2,
                       positive_source_support_policy="every_value_covered_by_two_distinct_sources",
                       evidence_noise=EVIDENCE_NOISE, excluded_source_ids=sorted(excluded_sources),
                       excluded_episode_identities=[dict(source_id=source_id, dataset_root=root, episode_index=index)
                           for source_id, root, index in sorted({(r["source_id"], r["dataset_root"], r["episode_index"])
                               for r in reviews["records"] if r["source_id"] in excluded_sources})])
    validate_catalog(catalog)
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--mappings", type=Path,
                        help="Explicit all-mode sample mapping; legacy census mapping is the default")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", default="task-profiles-20261103-v1")
    args = parser.parse_args(argv)
    inputs = dict(statistics=args.census / "task_side_statistics.json",
                  membership=args.census / "mode_membership.json",
                  mappings=args.mappings or args.census / "review_sample_mode_mapping.json",
                  scope=args.scope, reviews=args.reviews)
    catalog = build_catalog(**{key: read_json(path) for key, path in inputs.items()}, version=args.version)
    catalog["input_provenance"] = {key: dict(path=str(path), sha256=digest(path)) for key, path in inputs.items()}
    validate_catalog(catalog)
    output = create_run(args.output, input_paths=[args.census, args.scope, args.reviews])
    write_json(output / "task_gripper_profiles.json", catalog)
    rows = []
    for task in catalog["tasks"]:
        for side, profile in task["sides"].items():
            rows.append(dict(task=task["task"], side=side, status=profile["status"],
                automatic_use=profile["automatic_use"],
                closed_bands=[{k: b[k] for k in ("lower", "center", "upper")} for b in profile["closed_bands"]],
                open_bands=[{k: b[k] for k in ("lower", "center", "upper")} for b in profile["open_bands"]],
                reasons=profile["reasons"], **profile["counts"]))
    write_json(output / "profile_summary.json", dict(scope=catalog["scope"],
        statuses=dict(Counter(r["status"] for r in rows)),
        automatic_profiles=sum(r["automatic_use"] for r in rows), profiles=rows))
    with (output / "profile_summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    print(json.dumps(dict(output=str(output), profiles=len(rows),
                         automatic_profiles=sum(r["automatic_use"] for r in rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
