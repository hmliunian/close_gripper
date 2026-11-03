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
from close_gripper.task_profiles import SCHEMA, UNITS, validate_catalog


PADDING = 0.01


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


def build_catalog(statistics, membership, mappings, scope, reviews, version):
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
        tasks[name]["sides"][side] = build_profile(stat, modes[(name, side)], review_groups[(name, side)], mapping)
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
    validate_catalog(catalog)
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", default="task-profiles-20261103-v1")
    args = parser.parse_args(argv)
    inputs = dict(statistics=args.census / "task_side_statistics.json",
                  membership=args.census / "mode_membership.json",
                  mappings=args.census / "review_sample_mode_mapping.json",
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
