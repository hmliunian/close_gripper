"""Shard-wise task-profile evaluation with unchanged source records.

Only evaluation columns contain rule-generated curves. Added annotation columns
are explicit original-value/false-validity fallbacks, never inferred labels.
This prepares data-only staging under runs/; it does not publish or copy media.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

from . import pipeline
from .contracts import SIDE_INDICES, validate_episode
from .cycles import cycle_config
from .dataset import valid_sides_and_cameras
from .profile_clean import RULE_VERSION, generate_profile_targets
from .task_profiles import CALIBRATION_CONTRACT, _task_labels, validate_catalog


MANIFEST_SCHEMA = "close_gripper.profile_bulk_manifest.v1"
BULK_SCHEMA = "close_gripper.profile_bulk.v1"
EVALUATION_COLUMNS = (
    "evaluation.gripper_target", "evaluation.gripper_rule_applied",
    "evaluation.gripper_phase", "evaluation.gripper_source",
)
ANNOTATION_COLUMNS = ("annotation.gripper_target", "annotation.gripper_target_valid")
CURVE_KEYS = {"target", "valid", "phase", "source", "preview_target", "rule_target",
              "rule_applied_mask", "preview_phase", "preview_source"}
_WORKER = None


class ProfileBulkError(ValueError):
    """Incomplete/invalid bulk evaluation; a failed run is not a completed run."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _checked_bytes(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProfileBulkError("input requires its frozen SHA256: " + str(path))
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ProfileBulkError("input SHA256 differs from manifest: " + str(path))
    return raw


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ProfileBulkError("relative input/output path is required")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ProfileBulkError("paths must be relative without ..")
    return path


def _component(value):
    path = _relative(value)
    if len(path.parts) != 1:
        raise ProfileBulkError("dataset identifiers must be single path components")
    return value


def _source_path(root, relative):
    path = (root / _relative(relative)).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ProfileBulkError("input file escapes its source dataset")
    return path


def _release_directory(dataset):
    return _component(dataset.get("release_directory", dataset["release"]))


def _task_index(catalog):
    return {name: task["task"] for task in catalog["tasks"]
            for name in [task["task"], *task.get("aliases", [])]}


def _task_name(identity, index, ambiguous):
    label, origin = _task_labels(identity)
    if label is None:
        raise ProfileBulkError(origin)
    if label in ambiguous:
        raise ProfileBulkError("ambiguous_task_alias")
    if label not in index:
        raise ProfileBulkError("unknown_task")
    return index[label]


def _project_catalog(catalog, task_name):
    """Retain all names/aliases/exclusions; only the dispatched task is usable.

    Other task identities remain as disabled lightweight placeholders so the
    complete ambiguous-alias contract remains valid. Dispatch separately checks
    the authoritative original task before this projection is ever evaluated.
    """
    result = {k: copy.deepcopy(v) for k, v in catalog.items() if k not in ("tasks", "input_provenance")}
    tasks = []
    for task in catalog["tasks"]:
        entry = dict(task=task["task"], aliases=copy.deepcopy(task.get("aliases", [])), sides={})
        for side, profile in task["sides"].items():
            if task["task"] == task_name:
                keep = {"calibration_contract", "status", "automatic_use", "minimum_positive_sources",
                        "positive_source_support_policy", "evidence_noise", "closed_bands", "open_bands",
                        "free_bands", "interval_vision_bands", "release_rules", "counts", "provenance"}
                p = {k: copy.deepcopy(v) for k, v in profile.items() if k in keep}
                p["provenance"] = {k: v for k, v in p["provenance"].items() if k != "case_ids"}
            else:
                p = dict(status="needs_review", automatic_use=False, counts={}, provenance={},
                         closed_bands=[], open_bands=[])
                if catalog.get("calibration_contract") == CALIBRATION_CONTRACT:
                    p.update(calibration_contract=CALIBRATION_CONTRACT, minimum_positive_sources=2,
                             evidence_noise=.003, free_bands=[], interval_vision_bands=[], release_rules=[])
            entry["sides"][side] = p
        tasks.append(entry)
    result["tasks"] = tasks
    validate_catalog(result)
    return result


def _identity(root, info_sha, mask_sha, meta, data_path, source):
    fields = {k: v for k, v in meta.items() if k.startswith(("source_", "original_", "crop_"))}
    return dict(dataset_root=str(root), release=root.parent.name, subset=root.name,
        episode_index=meta["episode_index"], data_path=str(data_path),
        dataset_from_index=meta["dataset_from_index"], dataset_to_index=meta["dataset_to_index"],
        tasks=meta.get("tasks", []), source_metadata=fields,
        source_id=source["source_id"], source_episode_identity_verified=False,
        source_identity_binding="frozen_scope_ledger_and_original_metadata",
        info_sha256=info_sha, mask_sha256=mask_sha)


def _validate_manifest(manifest, catalog, catalog_sha):
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ProfileBulkError("unsupported bulk manifest schema")
    if manifest.get("catalog_sha256") != catalog_sha:
        raise ProfileBulkError("catalog SHA256 differs from frozen selection")
    if type(manifest.get("complete_source_scope")) is not bool:
        raise ProfileBulkError("manifest must explicitly declare complete_source_scope")
    enabled = {t["task"] for t in catalog["tasks"] if any(p["automatic_use"] for p in t["sides"].values())}
    selected = manifest.get("selected_tasks")
    if not isinstance(selected, list) or len(set(selected)) != len(selected) or set(selected) != enabled:
        raise ProfileBulkError("selected task scope must equal the frozen catalog's enabled tasks")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ProfileBulkError("manifest requires source datasets")
    roots, ids, destinations = set(), set(), set()
    for ds in datasets:
        identifier = _component(ds["dataset_id"])
        root = Path(ds["dataset_root"]).resolve(strict=True)
        release, subset = _release_directory(ds), _component(ds["subset"])
        if (root.parent.name, root.name) != (release, subset):
            raise ProfileBulkError("release/subset differs from source root identity")
        if root in roots or identifier in ids or (release, subset) in destinations:
            raise ProfileBulkError("duplicate source or destination dataset")
        roots.add(root); ids.add(identifier); destinations.add((release, subset))
        if not isinstance(ds.get("shards"), list) or not ds["shards"]:
            raise ProfileBulkError("dataset requires explicit source shards")
        paths, episodes = set(), set()
        for shard in ds["shards"]:
            relative = _relative(shard["path"])
            if relative.parts[0] != "data" or relative.suffix != ".parquet" or str(relative) in paths:
                raise ProfileBulkError("invalid or duplicate source data shard")
            paths.add(str(relative))
            numbers = shard.get("episodes")
            if (not isinstance(numbers, list) or not numbers or any(type(i) is not int or i < 0 for i in numbers)
                    or len(set(numbers)) != len(numbers) or episodes.intersection(numbers)
                    or type(shard.get("rows")) is not int or shard["rows"] <= 0):
                raise ProfileBulkError("invalid or repeated shard episode scope")
            episodes.update(numbers)
    return sorted(str(root) for root in roots)


def _snapshot_manifest_sidecars(manifest, origin, directory):
    """Keep original manifest-relative dependencies usable for direct replay."""
    entries = [(d["source_episodes_path"], d["source_episodes_sha256"]) for d in manifest["datasets"]]
    for prefix in ("episode_ledger", "video_reference_index"):
        if prefix + "_path" in manifest:
            entries.append((manifest[prefix + "_path"], manifest[prefix + "_sha256"]))
    seen = {}
    for relative, checksum in entries:
        path = _relative(relative)
        if path.parts[0] in {"implementation", "staging", "audit", "input_metadata"}:
            raise ProfileBulkError("manifest sidecar overlaps reserved output directories")
        if str(path) in seen:
            if seen[str(path)] != checksum:
                raise ProfileBulkError("manifest sidecar hash declarations disagree")
            continue
        source = (origin / path).resolve(strict=True)
        if not source.is_relative_to(origin.resolve()):
            raise ProfileBulkError("manifest sidecar escapes its frozen directory")
        raw = _checked_bytes(source, checksum)
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        seen[str(path)] = checksum


def _dataset_jobs(ds, manifest, manifest_parent, directory, catalog):
    """Read and hash each small metadata file once; keep shard jobs bounded."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(ds["dataset_root"]).resolve(strict=True)
    snapshots = directory / "input_metadata" / ds["dataset_id"]
    metadata, seen_files, info, mask = {}, set(), None, None
    for item in ds["metadata_files"]:
        relative = _relative(item["path"])
        if str(relative) in seen_files:
            raise ProfileBulkError("duplicate metadata snapshot")
        seen_files.add(str(relative))
        raw = _checked_bytes(_source_path(root, str(relative)), item["sha256"])
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        if str(relative) == "meta/info.json":
            info = json.loads(raw)
            if item["sha256"] != ds["info_sha256"]:
                raise ProfileBulkError("info hash declarations disagree")
        elif str(relative) == "dataset_configs/masks/mask.json":
            mask = json.loads(raw)
            if item["sha256"] != ds["mask_sha256"]:
                raise ProfileBulkError("mask hash declarations disagree")
        elif relative.parts[:2] == ("meta", "episodes") and relative.suffix == ".parquet":
            for row in pq.read_table(pa.BufferReader(raw), use_threads=False).to_pylist():
                i = row.get("episode_index")
                if type(i) is not int or i < 0 or i in metadata:
                    raise ProfileBulkError("episode metadata identity missing or duplicated")
                metadata[i] = row
    if info is None or mask is None or not metadata or info.get("codebase_version") != "v3.0":
        raise ProfileBulkError("complete LeRobot v3 info/mask/episode metadata is required")
    actual_metadata = {p.relative_to(root).as_posix() for p in (root / "meta/episodes").glob("chunk-*/*.parquet")}
    listed_metadata = {p for p in seen_files if p.startswith("meta/episodes/") and p.endswith(".parquet")}
    if actual_metadata != listed_metadata:
        raise ProfileBulkError("manifest omitted or added an episode metadata file")
    sides, _ = valid_sides_and_cameras(info, mask)
    source_path = (manifest_parent / _relative(ds["source_episodes_path"])).resolve(strict=True)
    if not source_path.is_relative_to(manifest_parent.resolve()):
        raise ProfileBulkError("source ledger escapes frozen manifest directory")
    raw = _checked_bytes(source_path, ds["source_episodes_sha256"])
    with (snapshots / "source_episodes.jsonl.gz").open("xb") as stream:
        stream.write(raw)
    ledger = {}
    for line in gzip.decompress(raw).splitlines():
        source = json.loads(line)
        i = source["episode_index"]
        if type(i) is not int or i not in metadata or i in ledger:
            raise ProfileBulkError("source ledger differs from original episode metadata")
        if source.get("dataset_root", str(root)) != str(root):
            raise ProfileBulkError("source ledger dataset identity differs")
        if not isinstance(source.get("source_id"), str) or not source["source_id"].strip():
            raise ProfileBulkError("scope ledger requires a nonempty original source identity")
        ledger[i] = source
    if set(ledger) != set(metadata):
        raise ProfileBulkError("source ledger must cover every original metadata episode")
    index, selected = _task_index(catalog), set(manifest["selected_tasks"])
    expected_shards = defaultdict(list)
    for i, meta in metadata.items():
        try:
            for key in ("data/chunk_index", "data/file_index", "length", "dataset_from_index", "dataset_to_index"):
                if type(meta[key]) is not int or meta[key] < 0:
                    raise ValueError("invalid integer metadata")
            relative = str(_relative(info["data_path"].format(chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileBulkError("invalid original data location or episode bounds") from exc
        source = ledger[i]
        identity = _identity(root, ds["info_sha256"], ds["mask_sha256"], meta, root / relative, source)
        task = _task_name(identity, index, catalog.get("ambiguous_aliases", {}))
        if source.get("task") != task or type(source.get("selected")) is not bool or source["selected"] != (task in selected):
            raise ProfileBulkError("task or selected scope disagrees with original metadata")
        if any(key in source and source[key] != meta[key] for key in ("dataset_from_index", "dataset_to_index")):
            raise ProfileBulkError("source ledger episode bounds disagree with original metadata")
        if source.get("data_path", relative) != relative:
            raise ProfileBulkError("source ledger data location disagrees with original metadata")
        for key in ("source_path",):
            if meta.get(key) and meta[key] != source.get(key):
                raise ProfileBulkError("original source path disagrees with frozen source ledger")
        raw_path = source.get("source_path")
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).parent.name != source["source_id"]:
            raise ProfileBulkError("source path does not bind the stated original recording")
        if source.get("valid_sides", sides) != sides:
            raise ProfileBulkError("source ledger sides disagree with mask")
        expected_shards[relative].append(i)
    listed = {shard["path"]: shard for shard in ds["shards"]}
    if manifest["complete_source_scope"] and set(listed) != set(expected_shards):
        raise ProfileBulkError("full scope manifest omitted a source data shard")
    if manifest["complete_source_scope"]:
        physical = {p.relative_to(root).as_posix() for p in (root / "data").glob("chunk-*/*.parquet")}
        if physical != set(listed):
            raise ProfileBulkError("full scope manifest differs from source data files")
    for relative, shard in listed.items():
        if relative not in expected_shards or set(shard["episodes"]) != set(expected_shards[relative]):
            raise ProfileBulkError("shard episode scope disagrees with original metadata")
        if shard["rows"] != sum(metadata[i]["length"] for i in shard["episodes"]):
            raise ProfileBulkError("shard frame count differs from episode metadata")
        yield dict(dataset_id=ds["dataset_id"], dataset_root=str(root), release=_release_directory(ds), subset=ds["subset"],
            output_root=str(directory / "staging" / (_release_directory(ds) + "-gripcmd") / ds["subset"]),
            directory=str(directory), shard=shard, valid_sides=sides, info_sha256=ds["info_sha256"],
            mask_sha256=ds["mask_sha256"], metadata=[metadata[i] for i in shard["episodes"]],
            sources={i: ledger[i] for i in shard["episodes"]})


def _unchanged_result(episode):
    original = [[s[9], s[19]] for s in episode["state"]]
    flags = [[False, False] for _ in original]
    phase = [["unchanged", "unchanged"] for _ in original]
    source = [["", ""] for _ in original]
    return dict(workflow="task_profile_rule", evidence_type="task_profile_rule", rule_version=RULE_VERSION,
        evaluation_only=True, grasp_semantics_verified=False, training_ready=False, model_requests=0,
        target=original, valid=flags, preview_target=copy.deepcopy(original), rule_applied_mask=flags,
        preview_phase=phase, preview_source=source, candidates=[], events=[],
        coverage={side: dict(valid_side=side in episode["valid_sides"], complete_numeric_scan=False,
            reason="scope_not_selected" if side in episode["valid_sides"] else "invalid_side",
            applied_frames=0, changed_frames=0, pregrasp_status="not_modified") for side in SIDE_INDICES},
        side_status={side: "scope_not_selected" if side in episode["valid_sides"] else "invalid_side" for side in SIDE_INDICES})


def _initialize_worker(catalog, numerical):
    global _WORKER
    import pyarrow as pa
    pa.set_cpu_count(1)
    _WORKER = dict(catalog=catalog, numerical=numerical, projected={})


def _write_parquet(table, path, *, original=None):
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if path.exists():
        raise FileExistsError("derived shard already exists")
    with temporary.open("xb") as stream:
        pq.write_table(table, stream, compression="zstd")
    reread = pq.read_table(temporary, use_threads=False)
    if not reread.equals(table, check_metadata=True):
        raise ProfileBulkError("Parquet roundtrip changed the derived records")
    if original is not None and not reread.select(original.column_names).equals(original, check_metadata=True):
        raise ProfileBulkError("derived Parquet changed original fields, order or rows")
    checksum = _sha(temporary)
    temporary.rename(path)
    return checksum


def _audit_schemas(pa):
    common = [("dataset_id", pa.string()), ("dataset_root", pa.string()), ("source_data_path", pa.string()),
              ("source_sha256", pa.string()), ("episode_index", pa.int64()), ("episode_id", pa.string()),
              ("task", pa.string()), ("source_id", pa.string())]
    episode = pa.schema(common + [("source_path", pa.string()), ("selected", pa.bool_()),
        ("row_start", pa.int64()), ("row_count", pa.int64()), ("valid_sides", pa.list_(pa.field("element", pa.string()))),
        ("dataset_from_index", pa.int64()), ("dataset_to_index", pa.int64()),
        ("candidates", pa.int64()), ("applied_intervals", pa.int64()), ("rejected_intervals", pa.int64()),
        ("applied_frames", pa.int64()), ("changed_frames", pa.int64()),
        ("pregrasp_status", pa.string()), ("result_metadata_json", pa.string())])
    event = pa.schema(common + [("side", pa.string()), ("candidate_id", pa.string()),
        ("status", pa.string()), ("reason", pa.string()), ("start", pa.int64()), ("end", pa.int64()),
        ("modified_start", pa.int64()), ("modified_end", pa.int64()),
        ("candidate_json", pa.string()), ("event_json", pa.string())])
    return episode, event


def _process_shard(job):
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    catalog, numerical, projected = (_WORKER[k] for k in ("catalog", "numerical", "projected"))
    root, shard = Path(job["dataset_root"]), job["shard"]
    source_path = _source_path(root, shard["path"])
    raw = _checked_bytes(source_path, shard["sha256"])
    table = pq.read_table(pa.BufferReader(raw), use_threads=False)
    del raw
    if table.num_rows != shard["rows"] or len(table.column_names) != len(set(table.column_names)):
        raise ProfileBulkError("source shard row count or field names are invalid")
    required = {"observation.state", "timestamp", "episode_index", "frame_index", "index"}
    if not required <= set(table.column_names) or set(EVALUATION_COLUMNS).intersection(table.column_names):
        raise ProfileBulkError("source must have original numeric fields and no existing evaluation output")
    episode_ids = table["episode_index"].to_pylist()
    bounds = [0] + [i for i in range(1, len(episode_ids)) if episode_ids[i] != episode_ids[i-1]] + [len(episode_ids)]
    groups = [episode_ids[a] for a in bounds[:-1]]
    if len(groups) != len(set(groups)) or set(groups) != set(shard["episodes"]):
        raise ProfileBulkError("source episodes are missing, duplicated or split across a shard")
    metadata = {m["episode_index"]: m for m in job["metadata"]}
    task_index = _task_index(catalog)
    targets, masks, phases, sources, originals = [], [], [], [], []
    episode_rows, event_rows, task_totals = [], [], {}
    side_reasons, event_reasons = Counter(), Counter()
    for a, b in zip(bounds[:-1], bounds[1:]):
        index = episode_ids[a]
        meta, source = metadata[index], job["sources"][index]
        part = table.slice(a, b-a)
        frames, indices = part["frame_index"].to_pylist(), part["index"].to_pylist()
        if (b-a != meta["length"] or frames != list(range(b-a))
                or indices != list(range(meta["dataset_from_index"], meta["dataset_to_index"]))):
            raise ProfileBulkError("episode rows differ from original length/frame/global-index metadata")
        if source.get("shard_row_start", a) != a or source.get("shard_row_end_exclusive", b) != b:
            raise ProfileBulkError("source ledger shard bounds disagree with actual row order")
        identity = _identity(root, job["info_sha256"], job["mask_sha256"], meta, source_path, source)
        if "timestamp_ns" in table.column_names:
            identity.update(first_timestamp_ns=part["timestamp_ns"][0].as_py(), last_timestamp_ns=part["timestamp_ns"][-1].as_py())
        episode = dict(episode_id=f"{root.parent.name}/{root.name}/{index}", identity=identity,
            timestamps=part["timestamp"].to_pylist(), state=part["observation.state"].to_pylist(),
            valid_sides=list(job["valid_sides"]), cameras={}, camera_issues={}, frame_indices=frames, global_indices=indices)
        validate_episode(episode)
        task = _task_name(identity, task_index, catalog.get("ambiguous_aliases", {}))
        if task != source["task"]:
            raise ProfileBulkError("episode task differs from dispatched scope")
        original = [[s[9], s[19]] for s in episode["state"]]
        immutable = copy.deepcopy((episode["state"], episode["timestamps"]))
        if source["selected"]:
            if task not in projected:
                projected[task] = _project_catalog(catalog, task)
            result = generate_profile_targets(episode, projected[task], numerical)
        else:
            result = _unchanged_result(episode)
        if (episode["state"], episode["timestamps"]) != immutable:
            raise ProfileBulkError("rule processing mutated its observation input")
        if result["target"] != original or any(any(row) for row in result["valid"]):
            raise ProfileBulkError("rule processing attempted to promote training labels")
        curve, applied = result["preview_target"], result["rule_applied_mask"]
        if any(len(v) != b-a for v in (curve, applied, result["preview_phase"], result["preview_source"])):
            raise ProfileBulkError("generated evaluation arrays have wrong frame counts")
        for i, (values, flags) in enumerate(zip(curve, applied)):
            if len(values) != 2 or len(flags) != 2 or any(type(f) is not bool for f in flags):
                raise ProfileBulkError("evaluation target/mask has invalid shape")
            for col, side in enumerate(SIDE_INDICES):
                if (side not in episode["valid_sides"] or not flags[col]) and values[col] != original[i][col]:
                    raise ProfileBulkError("rule modified an invalid side or unapplied frame")
        candidates, events = result["candidates"], result["events"]
        if len(candidates) != len(events) or any(c["candidate_id"] != e["candidate_id"] for c, e in zip(candidates, events)):
            raise ProfileBulkError("candidate/event audit is incomplete")
        changed = sum(any(abs(v[col] - original[i][col]) > 1e-12 for col in (0, 1)) for i, v in enumerate(curve))
        applied_count = sum(e["status"] == "applied" for e in events)
        common = dict(dataset_id=job["dataset_id"], dataset_root=str(root), source_data_path=shard["path"],
            source_sha256=shard["sha256"], episode_index=index, episode_id=episode["episode_id"], task=task, source_id=source["source_id"])
        result_meta = {k: v for k, v in result.items() if k not in CURVE_KEYS | {"candidates", "events"}}
        result_meta["pregrasp_policy"] = "unchanged_missing_per_episode_visual_evidence"
        episode_rows.append(dict(common, source_path=source["source_path"], selected=source["selected"],
            row_start=a, row_count=b-a, valid_sides=episode["valid_sides"], dataset_from_index=meta["dataset_from_index"],
            dataset_to_index=meta["dataset_to_index"], candidates=len(candidates), applied_intervals=applied_count,
            rejected_intervals=len(events)-applied_count, applied_frames=sum(any(row) for row in applied), changed_frames=changed,
            pregrasp_status="not_modified_missing_episode_visual_evidence" if source["selected"] else "scope_not_selected",
            result_metadata_json=_json(result_meta)))
        for candidate, event in zip(candidates, events):
            event_rows.append(dict(common, **{k: event.get(k) for k in ("side", "candidate_id", "status", "reason", "start", "end", "modified_start", "modified_end")},
                candidate_json=_json(candidate), event_json=_json(event)))
            if event["status"] != "applied":
                event_reasons[event["reason"]] += 1
        for side in episode["valid_sides"]:
            side_reasons[result["coverage"][side]["reason"]] += 1
        totals = task_totals.setdefault(task, Counter())
        totals.update(episodes=1, frames=b-a, selected_episodes=int(source["selected"]), selected_frames=(b-a)*int(source["selected"]),
            changed_episodes=int(changed > 0), changed_frames=changed, applied_intervals=applied_count,
            rejected_intervals=len(events)-applied_count, candidates=len(candidates))
        targets.extend(curve); masks.extend(applied); phases.extend(result["preview_phase"]); sources.extend(result["preview_source"]); originals.extend(original)
    output = table
    for name, values, typ in zip(EVALUATION_COLUMNS, (targets, masks, phases, sources),
            (pa.float64(), pa.bool_(), pa.string(), pa.string())):
        output = output.append_column(name, pa.array(values, type=pa.list_(pa.field("element", typ), 2)))
    added_annotations = []
    for name, values, typ in zip(ANNOTATION_COLUMNS, (originals, [[False, False]] * table.num_rows), (pa.float32(), pa.bool_())):
        if name not in table.column_names:
            output = output.append_column(name, pa.array(values, type=pa.list_(pa.field("element", typ), 2)))
            added_annotations.append(name)
    target_path = Path(job["output_root"]) / _relative(shard["path"])
    data_sha = _write_parquet(output, target_path, original=table)
    episode_schema, event_schema = _audit_schemas(pa)
    audit_relative = _relative(shard["path"])
    audit_root = Path(job["directory"]) / "audit" / job["dataset_id"]
    episode_path, event_path = audit_root / "episodes" / audit_relative, audit_root / "events" / audit_relative
    episode_sha = _write_parquet(pa.Table.from_pylist(episode_rows, schema=episode_schema), episode_path)
    event_sha = _write_parquet(pa.Table.from_pylist(event_rows, schema=event_schema), event_path)
    # Rehash after processing; stat equality alone does not prove source content.
    if _sha(source_path) != shard["sha256"]:
        raise ProfileBulkError("source content changed while the shard was evaluated")
    return dict(status="complete", dataset_id=job["dataset_id"], source_root=str(root),
        output_root=job["output_root"], source_path=str(source_path), source_relative_path=shard["path"],
        output_path=str(target_path), source_sha256=shard["sha256"], output_sha256=data_sha,
        rows=table.num_rows, episodes=len(episode_rows), tasks={k: dict(v) for k, v in task_totals.items()},
        episode_audit_path=str(episode_path), episode_audit_sha256=episode_sha,
        event_audit_path=str(event_path), event_audit_sha256=event_sha, events=len(event_rows),
        annotation_columns_added=added_annotations, annotation_original_columns_preserved=True,
        original_columns=table.column_names, original_columns_arrow_equal=True,
        side_reasons=dict(side_reasons), event_rejection_reasons=dict(event_reasons),
        duration_s=time.monotonic()-started)


def run_profile_dataset(manifest_path, catalog_path, output, *, workers=4):
    """Evaluate frozen complete shards; failed runs remain explicitly incomplete."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    manifest_path, catalog_path = Path(manifest_path).resolve(strict=True), Path(catalog_path).resolve(strict=True)
    manifest_raw, catalog_raw = manifest_path.read_bytes(), catalog_path.read_bytes()
    manifest, catalog = json.loads(manifest_raw), json.loads(catalog_raw)
    validate_catalog(catalog)
    catalog_sha = hashlib.sha256(catalog_raw).hexdigest()
    roots = _validate_manifest(manifest, catalog, catalog_sha)
    directory = pipeline.create_run(output, [*roots, manifest_path, catalog_path])
    for name, raw in (("manifest.json", manifest_raw), ("task_profiles.json", catalog_raw)):
        with (directory / name).open("xb") as stream:
            stream.write(raw)
    numerical = cycle_config()
    numerical["minimum_hold_s"] = catalog["minimum_hold_s"]
    config = dict(schema=BULK_SCHEMA, workflow="task_profile_rule", rule_version=RULE_VERSION,
        evaluation_only=True, training_ready=False, grasp_semantics_verified=False, model_requests=0,
        numerical=numerical, catalog_sha256=catalog_sha, catalog_version=catalog["version"],
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), manifest_origin=str(manifest_path),
        complete_source_scope=manifest["complete_source_scope"], selected_tasks=manifest["selected_tasks"], workers=workers,
        evaluation_columns=dict(zip(EVALUATION_COLUMNS, ("float64[2]", "bool[2]", "string[2]", "string[2]"))),
        annotation_fallback_columns=dict(zip(ANNOTATION_COLUMNS, ("float32[2] original observations", "bool[2] false"))),
        existing_annotation_columns="preserve unchanged", pregrasp_policy="unchanged_missing_per_episode_visual_evidence",
        publication_ready=False, media_independent=False, staging_kind="data_only_source_overlay")
    _save(directory / "config.json", config)
    package = Path(__file__).resolve().parent
    implementation = directory / "implementation" / "close_gripper"
    implementation.mkdir(parents=True)
    implementation_files = sorted(package.glob("*.py"))
    implementation_sha = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in implementation_files)).hexdigest()
    for path in implementation_files:
        shutil.copyfile(path, implementation / path.name)
    _save(directory / "implementation.json", dict(sha256=implementation_sha, files=[p.name for p in implementation_files]))
    expected_shards = sum(len(d["shards"]) for d in manifest["datasets"])
    total_tasks, totals, by_dataset, failures = {}, Counter(), {}, []
    side_reasons, event_reasons = Counter(), Counter()
    started = time.monotonic()
    completed = 0

    def jobs():
        for ds in manifest["datasets"]:
            yield from _dataset_jobs(ds, manifest, directory, directory, catalog)

    with (directory / "shards.jsonl").open("x", encoding="utf-8") as shard_log, \
            (directory / "progress.jsonl").open("x", encoding="utf-8") as progress_log:
        def accept(result):
            nonlocal completed
            completed += 1
            shard_log.write(_json(result) + "\n"); shard_log.flush()
            ds_counts = by_dataset.setdefault(result["dataset_id"], Counter())
            ds_counts["shards"] += 1
            for task, counts in result["tasks"].items():
                total_tasks.setdefault(task, Counter()).update(counts)
                totals.update(counts); ds_counts.update(counts)
                progress = dict(event="task_progress", task=task, **dict(total_tasks[task]))
                progress_log.write(_json(progress) + "\n")
                print(_json(progress), flush=True)
            side_reasons.update(result["side_reasons"]); event_reasons.update(result["event_rejection_reasons"])
            progress = dict(event="shard_complete", dataset_id=result["dataset_id"], path=result["source_relative_path"],
                completed_shards=completed, expected_shards=expected_shards, elapsed_s=time.monotonic()-started,
                **dict(totals))
            progress_log.write(_json(progress) + "\n"); progress_log.flush()
            print(_json(progress), flush=True)

        try:
            _snapshot_manifest_sidecars(manifest, manifest_path.parent, directory)
            if workers == 1:
                _initialize_worker(catalog, numerical)
                for job in jobs():
                    accept(_process_shard(job))
            else:
                iterator, pending, exhausted = iter(jobs()), {}, False
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                        initializer=_initialize_worker, initargs=(catalog, numerical)) as pool:
                    while pending or not exhausted:
                        while not exhausted and not failures and len(pending) < workers:
                            try:
                                job = next(iterator)
                            except StopIteration:
                                exhausted = True
                                break
                            pending[pool.submit(_process_shard, job)] = (job["dataset_id"], job["shard"]["path"])
                        if not pending:
                            break
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            identity = pending.pop(future)
                            try:
                                accept(future.result())
                            except Exception as exc:
                                failures.append(dict(dataset_id=identity[0], shard=identity[1], error_type=type(exc).__name__, reason=str(exc)))
                        if failures:
                            exhausted = True
        except Exception as exc:
            failures.append(dict(error_type=type(exc).__name__, reason=str(exc)))
    current_sha = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in implementation_files)).hexdigest()
    if current_sha != implementation_sha:
        failures.append(dict(error_type="ImplementationChanged", reason="implementation changed during evaluation"))
    if completed != expected_shards and not failures:
        failures.append(dict(error_type="IncompleteScope", reason="not every frozen source shard completed"))
    datasets = [dict(dataset_id=d["dataset_id"], source_root=d["dataset_root"],
        output_root=str(directory / "staging" / (_release_directory(d) + "-gripcmd") / d["subset"]),
        relative_output_root=str(Path("staging") / (_release_directory(d) + "-gripcmd") / d["subset"]),
        **dict(by_dataset.get(d["dataset_id"], {}))) for d in manifest["datasets"]]
    summary = dict(config, status="failed" if failures else "complete", completed_shards=completed,
        expected_shards=expected_shards, failures=failures, elapsed_s=time.monotonic()-started,
        datasets=datasets, tasks={k: dict(v) for k, v in total_tasks.items()}, **dict(totals),
        side_reasons=dict(side_reasons), event_rejection_reasons=dict(event_reasons),
        implementation_sha256=implementation_sha, implementation_changed_during_run=current_sha != implementation_sha,
        original_columns_verified_for_completed_shards=True, training_annotations_promoted=False,
        source_content_rehashed_after_each_completed_shard=True, shard_audit="shards.jsonl")
    _save(directory / "summary.json", summary)
    if failures:
        raise ProfileBulkError("bulk evaluation incomplete; see " + str(directory / "summary.json"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        summary = run_profile_dataset(args.manifest, args.catalog, args.output, workers=args.workers)
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(_json({k: summary[k] for k in ("status", "completed_shards", "episodes", "frames", "changed_episodes", "changed_frames")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
