"""Build task configs from dense main-view observations and evaluate the same cohort."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.contracts import SIDE_INDICES
from close_gripper.head_review import validate_head_review
from close_gripper.pipeline import create_run, dump, episode_fingerprint
from close_gripper.profile_bands import build_matching_catalog
from close_gripper.profile_clean import _blocks, _stable_windows
from close_gripper.profile_run import run_profile_clean
from close_gripper.task_profiles import CALIBRATION_CONTRACT, SCHEMA, UNITS, validate_catalog
from build_task_profiles import build_mode_profile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_review(inputs, reviews, row):
    folder = inputs / row["directory"]
    ep = read(folder / "episode.json")
    if episode_fingerprint(ep) != row["episode_sha256"]:
        raise ValueError("prepared episode hash differs")
    review_path = reviews / (row["directory"] + ".json")
    review = read(review_path)
    available_frames = read(folder / "frames.json")
    sheets = {s["path"]: s for s in read(folder / "sheets.json")}
    frames_by_path = {f["path"]: f for f in available_frames}
    viewed, artifact_hashes = set(), {}
    for name in review.get("viewed_sheets", []):
        if name not in sheets or sha(name) != sheets[name]["sha256"]:
            raise ValueError("viewed contact sheet is not the frozen main evidence")
        sheet = sheets[name]
        artifact_hashes[name] = sheet["sha256"]
        viewed.update((sheet["camera"], i) for i in sheet["frame_indices"])
    for name in review.get("viewed_extra_frames", []):
        if name not in frames_by_path:
            raise ValueError("extra viewed frame is not in the main evidence manifest")
        frame = frames_by_path[name]
        viewed.add((frame["camera"], frame["frame_index"]))
    frames = [f for f in available_frames if (f["camera"], f["frame_index"]) in viewed]
    for f in frames:
        if sha(f["path"]) != f["sha256"]:
            raise ValueError("main evidence image changed")
        artifact_hashes[f["path"]] = f["sha256"]
    validate_head_review(ep, review, frames)
    review["focus_intervals"] = read(folder / "focus_intervals.json")
    review["review_file_sha256"] = sha(review_path)
    review["viewed_artifact_hashes"] = artifact_hashes
    return ep, review, frames


def build_profiles(items, version):
    grouped = defaultdict(list)
    for row, ep, review, frames in items:
        for side in ep["valid_sides"]:
            grouped[(row["task"], side)].append((row, ep, review, frames))
    tasks, calibration_records, task_summary = {}, [], []
    for (task, side), entries in grouped.items():
        records, mappings, observations = [], {}, []
        state_counts = Counter()
        for row, ep, review, frames in entries:
            value = review["sides"][side]
            states = {i:s["state"] for s in value["segments"] for i in range(s["start"], s["end"]+1)}
            state_counts.update(states.values())
            times = ep["timestamps"]
            openings = [s[SIDE_INDICES[side]] for s in ep["state"]]
            # Preserve all actual observations, even a task with no calibratable plateau.
            for s in value["segments"]:
                vals = openings[s["start"]:s["end"]+1]
                observations.append(dict(episode_id=ep["episode_id"], source_id=row["source_id"],
                    **s, lower=min(vals), upper=max(vals), duration_s=times[s["end"]]-times[s["start"]]))
            panels = [dict(index=f["frame_index"], timestamp=f["timestamp"], camera=f["camera"],
                opening=openings[f["frame_index"]], path=f["path"], sha256=f["sha256"]) for f in frames]
            windows = [(a,b) for lo,hi in _blocks(times, .15)
                for a,b in _stable_windows(openings, dict(lower=0, upper=1), lo, hi, .02)
                if times[b] - times[a] >= .6 - 1e-9]
            window_indices = {i for a,b in windows for i in range(a,b+1)}
            outside_indices = set(range(len(times))) - window_indices
            for ordinal, (a,b) in enumerate(windows):
                name = f"{row['directory']}:{side}:{a}-{b}"
                vals = openings[a:b+1]
                # The full semantic review stays intact. Calibration records
                # carry their own window plus each outside observation once,
                # avoiding a quadratic duplication of all episode free frames.
                panel_indices = set(range(a,b+1)) | (outside_indices if ordinal == 0 else set())
                record = dict(case_id=name, sample_id=name, task=task, side=side,
                    source_id=row["source_id"], dataset_root=ep["identity"]["dataset_root"],
                    episode_index=row["episode_index"], numeric_role="static", numeric_mode_id=None,
                    plateau_start=a, plateau_end=b, duration_s=times[b]-times[a],
                    minimum=min(vals), center=median(vals), maximum=max(vals), camera_group="head",
                    review_source="model", review_method="codex_native_vision", confirmed=False,
                    panels=[p for p in panels if p["index"] in panel_indices],
                    frame_states=[dict(index=i, state=states[i]) for i in sorted(panel_indices)],
                    boundary_uncertain=[i for i in sorted(panel_indices) if states[i]=="uncertain"],
                    trace=dict(timestamps=times, openings=openings),
                    held_intervals=[], free_intervals=[], review_file_sha256=review["review_file_sha256"])
                for s in value["segments"]:
                    x,y = max(a,s["start"]), min(b,s["end"])
                    if x<=y and s["state"] in ("held", "free"):
                        record[s["state"] + "_intervals"].append(dict(start=x,end=y))
                keys = ("sample_id", "task", "side", "source_id", "dataset_root", "episode_index",
                    "numeric_role", "numeric_mode_id", "plateau_start", "plateau_end", "duration_s",
                    "minimum", "center", "maximum")
                mappings[name] = {k:record[k] for k in keys}
                records.append(record)
        stat = dict(episode_count=len(entries), unique_source_count=len({row["source_id"] for row,*_ in entries}),
            frame_count=sum(len(ep["timestamps"]) for _,ep,*_ in entries), stable_0_6s_count=len(records))
        profile = build_mode_profile(stat, dict(roles={}), records, mappings)
        profile["provenance"].update(camera_group="head", camera_keys=["cam_head_left", "cam_head_right"],
            calibration_scope="selected_twenty_episodes_only", all_numeric_mode_representatives_reviewed=False)
        profile["observed_segments"] = observations
        profile["observed_state_frame_counts"] = dict(state_counts)
        if state_counts["held"] == 0:
            profile["reasons"] = sorted(set(profile["reasons"] + ["no_confirmed_holding_in_main_view"]))
        tasks.setdefault(task, dict(task=task, aliases=[], sides={}))["sides"][side] = profile
        calibration_records.extend(records)
        task_summary.append(dict(task=task, side=side, episodes=len(entries), state_frames=dict(state_counts),
            automatic_use=profile["automatic_use"], closed_bands=profile["closed_bands"],
            held_candidate_bands=profile["held_candidate_bands"], reasons=profile["reasons"]))
    catalog = dict(schema=SCHEMA, version=version, units=UNITS, calibration_contract=CALIBRATION_CONTRACT,
        minimum_hold_s=.6, maximum_sample_gap_s=.15, maximum_plateau_range=.02,
        minimum_positive_sources=2, evidence_noise=.003, excluded_source_ids=[], excluded_episode_identities=[],
        tasks=list(tasks.values()), ambiguous_aliases={},
        vision=dict(camera_group="head", wrist_evidence_used=False, method="codex_native_vision",
                    confirmed=False, priority_shapes=["full_valley", "descending_half", "ascending_half"],
                    shape_is_semantic_proof=False),
        scope=dict(episodes=len(items), sources=len({x[0]["source_id"] for x in items}),
            tasks=len(tasks), full_dataset_calibration=False, heldout_evaluation=False,
            description="Configs derived solely from the selected main-view recordings; same-cohort curve review"))
    validate_catalog(catalog)
    return catalog, calibration_records, task_summary


def run(inputs, reviews, config_output, output):
    inputs, reviews = Path(inputs).resolve(strict=True), Path(reviews).resolve(strict=True)
    manifest = read(inputs / "manifest.json")
    if manifest.get("status") != "complete" or len(manifest["episodes"]) != 20:
        raise ValueError("requires the complete selected twenty-episode cohort")
    selected = manifest["episodes"]
    counts = Counter(row["task"] for row in selected)
    if len(counts) != 10 or set(counts.values()) != {2} or len({row["source_id"] for row in selected}) != 20:
        raise ValueError("requires ten tasks with two independent original recordings each")
    items = [(row, *load_review(inputs, reviews, row)) for row in manifest["episodes"]]
    config_dir = create_run(config_output, [inputs, reviews])
    calibration, records, stats = build_profiles(items, "head-main20-calibration-20261105-v1")
    catalog = build_matching_catalog(calibration, margin=.01, version="head-main20-matching-20261105-v1")
    dump(config_dir / "calibration_catalog.json", calibration)
    dump(config_dir / "task_profiles.json", catalog)
    dump(config_dir / "calibration_records.json", dict(records=records))
    dump(config_dir / "visual_reviews.json", dict(records=[x[2] for x in items]))
    dump(config_dir / "task_summary.json", stats)
    dump(config_dir / "provenance.json", dict(inputs=str(inputs), reviews=str(reviews),
        input_manifest_sha256=sha(inputs / "manifest.json"),
        build_script_sha256=sha(__file__), calibration_builder_sha256=sha(Path(__file__).with_name("build_task_profiles.py")),
        native_visual_episodes=20, native_visual_episode_sides=sum(len(x[1]["valid_sides"]) for x in items),
        external_model_requests=0, actual_viewed_images=sum(len(x[3]) for x in items),
        review_sources={x[0]["directory"]:x[2]["review_file_sha256"] for x in items}))
    summary = run_profile_clean([x[1] for x in items], output, catalog, workers=4,
        input_paths=[inputs, reviews, config_dir],
        visual_reviews={x[1]["episode_id"]:x[2] for x in items},
        evidence_frames={x[1]["episode_id"]:x[3] for x in items})
    print(json.dumps(dict(config=str(config_dir / "task_profiles.json"), output=str(output),
        tasks=len(catalog["tasks"]), automatic_task_sides=sum(s["automatic_use"] for t in catalog["tasks"] for s in t["sides"].values()),
        episodes=len(summary["episodes"]), source="new_native_main_view_model_observations"), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.inputs, args.reviews, args.config_output, args.output)
