"""Read every release gripper trace; emit numerical plateaus and review samples.

This is a calibration census, not grasp recognition or target generation.
All datasets are read-only and an existing output directory is refused.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RELEASES = {
    "priority": "PushCart-CollectSite-20261029-priority-en-flu-tool-v6-idletrim1-accepted-durclean",
    "remaining": "PushCart-CollectSite-20261029-remaining-en-flu-tool-v6-idletrim1-accepted-durclean",
    "drawer": "PushCart-CollectSite-scope139-20261026-b39c35c5-remaining-en-flu-tool-v4-idletrim3-drawer-accepted-durclean",
}
PARAMS = dict(minimum_hold_s=0.6, minimum_reported_plateau_s=0.2,
              plateau_peak_to_peak=0.02, maximum_sample_gap_s=0.15,
              interval_enumeration="inclusion_maximal_stable_windows_may_overlap",
              motion_context_s=1.5, minimum_context_excursion=0.04,
              mode_bin_width=0.005, mode_halfwidth_bins=4,
              mode_minimum_source_fraction=0.01, mode_band_padding=0.01,
              semantics="Numerical stability and motion context only; no visual or human grasp truth.")


def write_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                    (float(x) for x in np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))


def plateaus(t, v):
    """All inclusion-maximal stable windows, including overlapping boundaries.

    A short prefix must not consume the start of a later >=0.6s window.
    Monotone deques give O(n) enumeration. Windows are not distinct grasp events.
    """
    low_queue, high_queue = deque(), deque()
    left, previous = 0, None
    minimum = PARAMS["minimum_reported_plateau_s"]
    for right, x in enumerate(v):
        gap = right > 0 and not (0 < t[right] - t[right - 1]
                                <= PARAMS["maximum_sample_gap_s"] + 1e-9)
        invalid = not np.isfinite(x) or not np.isfinite(t[right])
        if gap or invalid:
            if previous is not None and t[previous[1]] - t[previous[0]] + 1e-9 >= minimum:
                yield previous
            low_queue.clear(); high_queue.clear()
            left, previous = right, None
            if invalid:
                left = right + 1
                continue
        while low_queue and v[low_queue[-1]] >= x:
            low_queue.pop()
        while high_queue and v[high_queue[-1]] <= x:
            high_queue.pop()
        low_queue.append(right); high_queue.append(right)
        while v[high_queue[0]] - v[low_queue[0]] > PARAMS["plateau_peak_to_peak"] + 1e-9:
            if low_queue[0] == left: low_queue.popleft()
            if high_queue[0] == left: high_queue.popleft()
            left += 1
        if previous is not None and left > previous[0]:
            if t[previous[1]] - t[previous[0]] + 1e-9 >= minimum:
                yield previous
        previous = (left, right, float(v[low_queue[0]]), float(v[high_queue[0]]))
    if previous is not None and t[previous[1]] - t[previous[0]] + 1e-9 >= minimum:
        yield previous


def survey_file(job):
    file_path, dataset_root, release, subset, sides, metadata = job
    p = Path(file_path)
    before = p.stat()
    raw = p.read_bytes()
    table = pq.read_table(pa.BufferReader(raw), columns=[
        "episode_index", "frame_index", "timestamp", "timestamp_ns", "observation.state"], use_threads=False)
    state = table["observation.state"].combine_chunks().values.to_numpy().reshape(-1, 20)
    ids = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    times = table["timestamp"].to_numpy()
    ns = table["timestamp_ns"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    episodes, spans = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ep_id = int(ids[a])
        meta = metadata[str(ep_id)]
        t, frame, stamp = times[a:b], frames[a:b], ns[a:b]
        if len(t) != meta["length"] or not np.array_equal(frame, np.arange(len(t))):
            raise ValueError(f"Episode incomplete/duplicated across file: {p} episode {ep_id}")
        if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError(f"Invalid timestamps: {p} episode {ep_id}")
        common = dict(task=meta["task"], release=release, subset=subset,
                      dataset_root=dataset_root, episode_index=ep_id,
                      source_path=meta["source"], source_id=Path(meta["source"]).parent.name,
                      frames=len(t))
        for side, column in sides:
            v = state[a:b, column]
            if not np.isfinite(v).all() or np.any(v < -1e-6) or np.any(v > 1 + 1e-6):
                raise ValueError(f"Invalid opening: {p} episode {ep_id} {side}")
            base = dict(common, side=side)
            ep = dict(base, duration_s=float(t[-1] - t[0]), opening_quantiles=quantiles(v),
                      initial_opening=float(v[0]), final_opening=float(v[-1]),
                      timestamp_gaps=int(np.sum(np.diff(t) > PARAMS["maximum_sample_gap_s"] + 1e-9)),
                      plateau_count=0, stable_0_6s_count=0)
            for start, end, lo, hi in plateaus(t, v):
                center = float(np.median(v[start:end + 1]))
                left = int(np.searchsorted(t, t[start] - PARAMS["motion_context_s"], side="left"))
                right = min(len(t) - 1, int(np.searchsorted(t, t[end] + PARAMS["motion_context_s"], side="right")) - 1)
                pre, post = v[left:start + 1], v[end:right + 1]
                pre_max, post_max, pre_min, post_min = map(float, (max(pre), max(post), min(pre), min(post)))
                exc = PARAMS["minimum_context_excursion"]
                before_close = pre_max - center >= exc
                after_open = post_max - center >= exc
                before_open = center - pre_min >= exc
                after_close = center - post_min >= exc
                low_evidence, high_evidence = before_close or after_open, before_open or after_close
                role = ("lower" if low_evidence and not high_evidence else
                        "upper" if high_evidence and not low_evidence else
                        "mixed" if low_evidence and high_evidence else "static")
                duration = float(t[end] - t[start])
                eligible = duration + 1e-9 >= PARAMS["minimum_hold_s"]
                pre_frame = left + int(np.argmax(pre)) if before_close else left
                post_frame = end + int(np.argmax(post)) if after_open else right
                mid = (start + end) // 2
                spans.append(dict(base, plateau_start=start, plateau_end=end,
                    start_time=float(t[start]), end_time=float(t[end]),
                    start_timestamp_ns=int(stamp[start]), end_timestamp_ns=int(stamp[end]),
                    duration_s=duration, center=center, minimum=lo, maximum=hi,
                    stable_0_6s=eligible, numeric_role=role,
                    incoming_closing=bool(before_close), outgoing_opening=bool(after_open),
                    incoming_opening=bool(before_open), outgoing_closing=bool(after_close),
                    pre_context_start=left, post_context_end=right,
                    closing_start=pre_frame if before_close else None,
                    opening_end=post_frame if after_open else None,
                    pre_max=pre_max, post_max=post_max, pre_min=pre_min, post_min=post_min,
                    sample_frames=[int(pre_frame), int(mid), int(end), int(post_frame)],
                    touches_start=start == 0, touches_end=end == len(t) - 1))
                ep["plateau_count"] += 1
                ep["stable_0_6s_count"] += int(eligible)
            episodes.append(ep)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during read: {p}")
    audit = dict(path=str(p), rows=len(ids), episode_rows=len(boundaries) - 1,
                 bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                 mtime_ns=before.st_mtime_ns, status="ok")
    return episodes, spans, audit


def mode_groups(rows, population_sources):
    """Bounded-width density groups; weight a recording once within each mode."""
    remaining = list(rows)
    modes = []
    bin_width = PARAMS["mode_bin_width"]
    half = PARAMS["mode_halfwidth_bins"]
    minimum = max(1, min(3, len(population_sources)),
                  int(np.ceil(len(population_sources) * PARAMS["mode_minimum_source_fraction"])))
    while remaining:
        bins = defaultdict(set)
        for r in remaining:
            bins[int(round(r["center"] / bin_width))].add(r["source_path"])
        windows = []
        for center in sorted(bins):
            members = set().union(*(bins.get(i, set()) for i in range(center - half, center + half + 1)))
            windows.append((len(members), -center, center))
        count, _, peak = max(windows)
        if count < minimum:
            break
        group = [r for r in remaining if abs(int(round(r["center"] / bin_width)) - peak) <= half]
        remaining = [r for r in remaining if abs(int(round(r["center"] / bin_width)) - peak) > half]
        by_source = defaultdict(list)
        for r in group:
            by_source[r["source_path"]].append(r["center"])
        source_centers = [float(np.median(v)) for v in by_source.values()]
        q = quantiles(source_centers)
        padding = PARAMS["mode_band_padding"]
        modes.append(dict(id=f"mode_{len(modes):02d}", center=q["p50"],
            lower=max(0., q["p05"] - padding), upper=min(1., q["p95"] + padding),
            source_count=len(by_source), platform_count=len(group),
            population_source_fraction=len(by_source) / len(population_sources),
            source_center_quantiles=q, _rows=group))
    return modes, remaining


def compact_mode(mode):
    return {k: v for k, v in mode.items() if not k.startswith("_")}


def summarize(episodes, spans, task_ids):
    episode_groups, span_groups = defaultdict(list), defaultdict(list)
    for r in episodes: episode_groups[(r["task"], r["side"])].append(r)
    for r in spans: span_groups[(r["task"], r["side"])].append(r)
    profiles, samples = [], []
    for (task, side), eps in sorted(episode_groups.items()):
        all_spans = span_groups[(task, side)]
        sources = {e["source_path"] for e in eps}
        stable = [r for r in all_spans if r["stable_0_6s"]]
        lower = [r for r in stable if r["numeric_role"] == "lower"]
        upper = [r for r in stable if r["numeric_role"] == "upper"]
        close_modes, close_residual = mode_groups(lower, sources)
        open_modes, open_residual = mode_groups(upper, sources)
        by_source = defaultdict(list)
        for e in eps: by_source[e["source_path"]].append(e)
        source_p95 = [float(np.median([e["opening_quantiles"]["p95"] for e in values]))
                      for values in by_source.values()]
        profile = dict(task=task, task_index=task_ids[task], side=side,
            episode_count=len(eps), unique_source_count=len(sources),
            frame_count=sum(e["frames"] for e in eps),
            episode_count_with_stable_platform=sum(e["stable_0_6s_count"] > 0 for e in eps),
            all_plateau_count=len(all_spans), stable_0_6s_count=len(stable),
            numeric_role_counts=dict(Counter(r["numeric_role"] for r in stable)),
            sources_with_lower_candidates=len({r["source_path"] for r in lower}),
            sources_with_upper_candidates=len({r["source_path"] for r in upper}),
            closed_candidate_quantiles=quantiles([r["center"] for r in lower]),
            open_candidate_quantiles=quantiles([r["center"] for r in upper]),
            source_equal_weight_episode_p95_quantiles=quantiles(source_p95),
            closed_modes=[compact_mode(m) for m in close_modes],
            open_modes=[compact_mode(m) for m in open_modes],
            closed_mode_unassigned_platforms=len(close_residual),
            open_mode_unassigned_platforms=len(open_residual),
            status="numeric_only", visual_reviewed=False)
        profiles.append(profile)
        used_sources = set()
        choices = []
        # Prioritize a dominant lower mode, a different mode/source, and an upper contrast.
        pools = [("dominant_lower", close_modes[0]["_rows"], close_modes[0]["center"])] if close_modes else []
        if len(close_modes) > 1:
            pools.append(("secondary_lower", close_modes[1]["_rows"], close_modes[1]["center"]))
        elif close_modes:
            pools.append(("repeat_lower", close_modes[0]["_rows"], close_modes[0]["center"]))
        if open_modes:
            pools.append(("upper_contrast", open_modes[0]["_rows"], open_modes[0]["center"]))
        pools.extend([("fallback_stable", stable, None), ("fallback_short", all_spans, None)])
        for label, pool, center in pools:
            available = [r for r in pool if r["source_path"] not in used_sources]
            if not available:
                continue
            r = min(available, key=lambda r: (
                abs(r["center"] - center) if center is not None else not r["stable_0_6s"],
                -int(r["incoming_closing"]) - int(r["outgoing_opening"]),
                -min(r["duration_s"], 3.), hashlib.sha256(r["source_path"].encode()).hexdigest()))
            sample = dict(r, task_index=task_ids[task], sample_role=label,
                          sample_id=f"task_{task_ids[task]:03d}_{side}_{len(choices):02d}")
            choices.append(sample); used_sources.add(r["source_path"])
            if len(choices) >= 3:
                break
        if len(choices) < 2:
            for e in sorted(eps, key=lambda e: e["source_id"]):
                if e["source_path"] in used_sources:
                    continue
                # Whole-episode fallback is explicitly not a stable lower candidate.
                end = e["frames"] - 1
                choices.append(dict(e, task_index=task_ids[task], sample_role="whole_episode_no_platform",
                    sample_id=f"task_{task_ids[task]:03d}_{side}_{len(choices):02d}",
                    plateau_start=0, plateau_end=end, center=e["opening_quantiles"]["p50"],
                    minimum=e["opening_quantiles"]["min"], maximum=e["opening_quantiles"]["max"],
                    stable_0_6s=False, numeric_role="unresolved", closing_start=None, opening_end=None,
                    pre_context_start=0, post_context_end=end,
                    sample_frames=[0, end // 3, 2 * end // 3, end]))
                used_sources.add(e["source_path"])
                if len(choices) >= 2: break
        samples.extend(choices)
    return profiles, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/gpfs/jiuquyun/datasets/PRETRAIN_DATA"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if not out.is_relative_to(workspace / "runs") or out == workspace / "runs":
        raise ValueError("Output must be a new workspace runs directory")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "parameters.json", PARAMS)
    write_json(out / "provenance.json", dict(command=sys.argv, python=sys.executable,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_data_changed=False, model_calls=0, evidence_type="full_population_numerical_census"))
    start = time.time()
    jobs, datasets, inventory = [], [], defaultdict(lambda: dict(episodes=0, sources=set(), sides=set()))
    for release, name in RELEASES.items():
        root = args.data_root / name
        mapping_file = root / ".orion/evidence/publication_source_mapping.json"
        mappings = defaultdict(dict)
        for row in json.loads(mapping_file.read_text()):
            if row["episode_index"] is not None:
                mappings[row["dataset"]][row["episode_index"]] = row["source"]
        for info_path in sorted(root.glob("*/meta/info.json")):
            subset = info_path.parent.parent
            info = json.loads(info_path.read_text())
            mask = json.loads((subset / "dataset_configs/masks/mask.json").read_text())
            element = mask["element_mask"]["observation.state"]
            sides = [(s, i) for s, i in (("left", 9), ("right", 19)) if element[i]]
            per_file = defaultdict(dict)
            for epfile in sorted((subset / "meta/episodes").rglob("*.parquet")):
                rows = pq.read_table(epfile, columns=["episode_index", "length", "source_tasks",
                           "data/chunk_index", "data/file_index"], use_threads=False).to_pylist()
                for row in rows:
                    task = " | ".join(row["source_tasks"])
                    source = mappings[subset.name][row["episode_index"]]
                    path = subset / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
                    per_file[str(path)][str(row["episode_index"])] = dict(task=task, source=source, length=row["length"])
                    inventory[task]["episodes"] += 1
                    inventory[task]["sources"].add(source)
                    inventory[task]["sides"].update(s for s, _ in sides)
            assert sum(len(v) for v in per_file.values()) == info["total_episodes"]
            assert set(per_file) == {str(p) for p in (subset / "data").rglob("*.parquet")}
            datasets.append(dict(release=release, name=name, subset=subset.name,
                dataset_root=str(subset), episodes=info["total_episodes"], frames=info["total_frames"],
                files=len(per_file), valid_sides=[s for s, _ in sides],
                info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
                source_mapping_sha256=hashlib.sha256(mapping_file.read_bytes()).hexdigest()))
            jobs.extend((p, str(subset), release, subset.name, sides, meta) for p, meta in sorted(per_file.items()))
    task_ids = {task: i for i, task in enumerate(sorted(inventory))}
    write_json(out / "task_inventory.json", dict(tasks=[dict(task=k, task_index=task_ids[k],
        episodes=v["episodes"], unique_sources=len(v["sources"]), sides=sorted(v["sides"]))
        for k, v in sorted(inventory.items())], datasets=datasets))
    print(f"INVENTORY tasks={len(inventory)} files={len(jobs)} episodes={sum(v['episodes'] for v in inventory.values())}", flush=True)
    all_episodes, all_spans, files = [], [], []
    with gzip.open(out / "episodes.jsonl.gz", "wt", encoding="utf-8") as epout, \
            gzip.open(out / "plateaus.jsonl.gz", "wt", encoding="utf-8") as plout:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for ordinal, (episodes, spans, audit) in enumerate(pool.map(survey_file, jobs), 1):
                for row in episodes: epout.write(json.dumps(row, ensure_ascii=False) + "\n")
                for row in spans: plout.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_episodes.extend(episodes); all_spans.extend(spans); files.append(audit)
                if ordinal % 50 == 0 or ordinal == len(jobs):
                    print(f"SCAN {ordinal}/{len(jobs)} files episode_sides={len(all_episodes)} plateaus={len(all_spans)} elapsed={time.time()-start:.1f}s", flush=True)
    write_json(out / "files.json", files)
    print("AGGREGATING modes and review samples", flush=True)
    profiles, samples = summarize(all_episodes, all_spans, task_ids)
    write_json(out / "task_side_statistics.json", dict(parameters=PARAMS, profiles=profiles))
    write_json(out / "review_samples.json", dict(evidence_type="numerically_selected_unreviewed_samples", samples=samples))
    counts = dict(tasks=len(inventory), task_sides=len(profiles), files=len(files),
        episodes=sum(f["episode_rows"] for f in files), frames=sum(f["rows"] for f in files),
        unique_sources=len({e["source_path"] for e in all_episodes}),
        episode_sides=len(all_episodes), platforms=len(all_spans),
        stable_0_6s_platforms=sum(r["stable_0_6s"] for r in all_spans),
        review_samples=len(samples), source_data_changed=False, model_calls=0,
        elapsed_s=time.time()-start, status="complete_numerical_census",
        limitations=["Plateaus are not grasp labels.", "Numerical modes require visual semantic review.",
                     "Same source is weighted once within each mode; different crops remain separately counted episodes.",
                     "Inclusion-maximal stable windows can overlap and are not separate grasp events."])
    assert counts["episodes"] == sum(d["episodes"] for d in datasets)
    assert counts["frames"] == sum(d["frames"] for d in datasets)
    write_json(out / "summary.json", counts)
    with (out / "task_summary.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["task_index", "task", "side", "episode_count", "unique_source_count", "stable_0_6s_count",
                  "sources_with_lower_candidates", "sources_with_upper_candidates", "closed_modes", "open_modes"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in profiles:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list) else row[k] for k in fields})
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
