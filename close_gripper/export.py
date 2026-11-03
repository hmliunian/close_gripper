"""Derive a bounded LeRobot sample with new targets and unchanged source columns.

This prepares reviewable data under runs/. It never publishes into PRETRAIN_DATA.
Selected episodes retain their source indices; the explicit metadata extension
allows the patched ORION loader to validate this sparse, episode-indexed view.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from . import pipeline
from .dataset import load_episode, valid_sides_and_cameras

TARGET_KEY = "annotation.gripper_target"
VALID_KEY = "annotation.gripper_target_valid"
EXPORT_SCHEMA = "close_gripper_targets_v1"


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file(root, relative):
    relative = _safe_relative(relative)
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("source metadata path escapes dataset root")
    if not path.is_file():
        raise ValueError("source metadata references a missing file")
    return path


def _safe_relative(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("metadata paths must be relative and cannot contain ..")
    return path


def _destination_file(directory, relative):
    path = directory / _safe_relative(relative)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("destination path escapes the output directory")
    return path


def _target_snapshots(items):
    snapshots = []
    for item in items:
        path = Path(item).resolve() if isinstance(item, (str, Path)) else None
        content = path.read_bytes() if path else (json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        snapshots.append(dict(value=json.loads(content), content=content, original_path=path))
    return snapshots


def _binding_mode(artifact):
    if "episode_sha256" in artifact:
        return "episode_fingerprint"
    provenance = artifact.get("provenance") or {}
    if isinstance(provenance, dict) and str(provenance.get("kind", "")).startswith("controlled_"):
        return "controlled_fixture_metadata_only"
    return "legacy_metadata_only"


def _target_artifacts(items, root):
    artifacts = {}
    for item in items:
        value = _read(item) if isinstance(item, (str, Path)) else copy.deepcopy(item)
        if not isinstance(value, dict):
            raise ValueError("target artifact must be an object")
        identity = value.get("identity", {})
        episode = identity.get("episode_index")
        if type(episode) is not int or episode < 0:
            raise ValueError("target identity needs an episode_index")
        source_root = identity.get("dataset_root")
        if not isinstance(source_root, str) or Path(source_root).resolve() != root:
            raise ValueError("target dataset_root does not match the source")
        if episode in artifacts:
            raise ValueError("duplicate episode target artifact")
        if TARGET_KEY not in value or VALID_KEY not in value:
            raise ValueError("export requires target and validity fields; previews are not labels")
        artifacts[episode] = value
    return artifacts


def _validate_labels(episode, artifact):
    original = [[row[9], row[19]] for row in episode["state"]]
    if artifact is None:
        return original, [[False, False] for _ in original]
    if artifact.get("episode_id") != episode["episode_id"]:
        raise ValueError("target episode_id differs from the source")
    if artifact.get("timestamps") != episode["timestamps"]:
        raise ValueError("target timestamps differ from the source")
    if artifact.get("frame_indices") != episode["frame_indices"]:
        raise ValueError("target frame indices differ from the source")
    for key in ("info_sha256", "mask_sha256", "dataset_from_index", "dataset_to_index"):
        if artifact["identity"].get(key) != episode["identity"].get(key):
            raise ValueError("target source identity has changed: " + key)
    method = str(artifact.get("method_version", "")).lower().replace("_", "-")
    full_cycle = method.startswith("full-cycle") or "annotation.gripper_phase" in artifact or artifact.get("workflow") == "full_cycle"
    if full_cycle and "episode_sha256" not in artifact:
        raise ValueError("full-cycle targets require episode_sha256")
    if "episode_sha256" in artifact and artifact["episode_sha256"] != pipeline.episode_fingerprint(episode):
        raise ValueError("target episode_sha256 differs from the current source episode")
    targets, valid = artifact[TARGET_KEY], artifact[VALID_KEY]
    if not isinstance(targets, list) or not isinstance(valid, list) or len(targets) != len(original) or len(valid) != len(original):
        raise ValueError("target frame count differs from the source")
    result = []
    for source, values, flags in zip(original, targets, valid):
        if not isinstance(values, list) or not isinstance(flags, list) or len(values) != 2 or len(flags) != 2:
            raise ValueError("target and validity rows must each have two columns")
        copied = list(source)
        for col, side in enumerate(("left", "right")):
            if type(flags[col]) is not bool:
                raise ValueError("target validity must be boolean")
            if flags[col]:
                if side not in episode["valid_sides"]:
                    raise ValueError("target enables an unavailable gripper side")
                value = values[col]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("valid target must be a finite opening in [0,1]")
                copied[col] = float(value)
        result.append(copied)
    return result, valid


def _statistics(rows):
    result = []
    for col in range(2):
        values = [float(row[col]) for row in rows]
        if not values:
            result.append(dict(count=0, min=None, max=None, mean=None, std=None))
            continue
        mean = sum(values) / len(values)
        result.append(dict(count=len(values), min=min(values), max=max(values), mean=mean,
                           std=math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))))
    return result


def _label_sources(snapshots, directory=None):
    sources = []
    for snapshot in snapshots:
        path, value = snapshot["original_path"], snapshot["value"]
        companion = _read(path.parent / "result.json") if path and (path.parent / "result.json").is_file() else {}
        artifact_path = str(path) if path else None
        if directory is not None:
            relative = Path("labels") / f"episode_{value['identity']['episode_index']}.json"
            destination = _destination_file(directory, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(snapshot["content"])
            artifact_path = relative.as_posix()
        sources.append(dict(episode_index=value.get("identity", {}).get("episode_index"),
                            artifact_path=artifact_path,
                            artifact_sha256=hashlib.sha256(snapshot["content"]).hexdigest(),
                            original_artifact_path=str(path) if path else None,
                            episode_sha256=value.get("episode_sha256"), source_binding=_binding_mode(value),
                            evidence_type=value.get("evidence_type", companion.get("evidence_type", "unspecified")),
                            method_version=value.get("method_version", companion.get("method_version")),
                            source_run=value.get("source_run", str(path.parent.parent) if path else None),
                            review_coverage=companion.get("coverage"),
                            provenance=value.get("provenance")))
    return sources


def derive_subset(dataset_root, episode_targets, output, *, episode_indices=None, video_mode="copy"):
    """Create a new, explicit sample derivative and return its summary.

    ``episode_targets`` contains paths to targets.json or their decoded objects.
    ``episode_indices`` can additionally select episodes without new labels;
    these retain their source grippers and receive false target validity.
    No sampling, model invocation or inference takes place in this function.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if video_mode != "copy":
        raise ValueError("video_mode must be copy; exported files are independent")
    episode_targets = list(episode_targets)
    root = Path(dataset_root).resolve(strict=True)
    info = _read(root / "meta/info.json")
    if TARGET_KEY in info.get("features", {}) or VALID_KEY in info.get("features", {}):
        raise ValueError("source already contains derived gripper fields")
    snapshots = _target_snapshots(episode_targets)
    artifacts = _target_artifacts([snapshot["value"] for snapshot in snapshots], root)
    selected = sorted(artifacts) if episode_indices is None else list(episode_indices)
    if not selected or any(type(index) is not int or index < 0 for index in selected) or len(selected) != len(set(selected)):
        raise ValueError("select unique nonnegative episode indices")
    if not set(artifacts).issubset(selected):
        raise ValueError("target artifact episode was not selected for export")
    selected_set = set(selected)
    episodes, labels = {}, {}
    for index in selected:
        episode = load_episode(root, index)
        episodes[index] = episode
        labels[index] = _validate_labels(episode, artifacts.get(index))

    metadata = []
    metadata_tables = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        filtered = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(selected, type=table["episode_index"].type)))
        if filtered.num_rows:
            metadata_tables.append((path.relative_to(root), filtered))
            metadata.extend(filtered.to_pylist())
    if len(metadata) != len(selected) or {row["episode_index"] for row in metadata} != selected_set:
        raise ValueError("selected episode catalog is missing or duplicated")
    videos = set()
    for row in metadata:
        for key, feature in info["features"].items():
            if feature.get("dtype") != "video":
                continue
            prefix = f"videos/{key}/"
            if row.get(prefix + "file_index") is None or row.get(prefix + "chunk_index") is None:
                continue
            relative = info["video_path"].format(video_key=key, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"])
            _source_file(root, relative)
            videos.add(relative)
    source_paths = {root / "meta/info.json", root / "dataset_configs/masks/mask.json"}
    source_paths.update(Path(episode["identity"]["data_path"]) for episode in episodes.values())
    before = {str(path): _digest(path) for path in source_paths}
    directory = pipeline.create_run(output, [root])

    # Copy support metadata, never the parent's publication-complete marker.
    for name in ("tasks.parquet", "joint_schema.json"):
        path = root / "meta" / name
        if path.is_file():
            (directory / "meta").mkdir(exist_ok=True)
            shutil.copy2(path, directory / "meta" / name)
    shutil.copytree(root / "dataset_configs", directory / "dataset_configs")
    for relative, table in metadata_tables:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)

    original_columns_verified = 0
    data_files = []
    tasks = defaultdict(lambda: dict(episodes=0, changed_episodes=0, passthrough_episodes=0,
                                     frames=0, changed_frames=0, valid_side_frames=0))
    values_all, flags_all = [], []
    for source_path in sorted({Path(episode["identity"]["data_path"]) for episode in episodes.values()}):
        source = pq.read_table(source_path)
        if TARGET_KEY in source.column_names or VALID_KEY in source.column_names:
            raise ValueError("source parquet already contains derived gripper fields")
        selected_rows = source.filter(pc.is_in(source["episode_index"], value_set=pa.array(selected, type=source["episode_index"].type)))
        values, flags = [], []
        for row in selected_rows.select(["episode_index", "frame_index"]).to_pylist():
            target, valid = labels[row["episode_index"]]
            values.append(target[row["frame_index"]])
            flags.append(valid[row["frame_index"]])
        derived = selected_rows.append_column(TARGET_KEY, pa.array(values, type=pa.list_(pa.float32(), 2)))
        derived = derived.append_column(VALID_KEY, pa.array(flags, type=pa.list_(pa.bool_(), 2)))
        destination = directory / source_path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(derived, destination)
        actual = pq.read_table(destination)
        if not actual.select(source.column_names).equals(selected_rows, check_metadata=True):
            raise ValueError("derived data changed a source column")
        original_columns_verified += len(source.column_names)
        # Statistics describe the stored float32 values, not Python pre-rounding.
        values_all.extend(actual[TARGET_KEY].to_pylist())
        flags_all.extend(actual[VALID_KEY].to_pylist())
        data_files.append(destination.relative_to(directory).as_posix())

    (directory / "videos").mkdir(exist_ok=True)
    for relative in sorted(videos):
        source = _source_file(root, relative)
        destination = _destination_file(directory, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for index, episode in episodes.items():
        target, valid = labels[index]
        changed = sum(any(abs(target[i][col] - row[(9, 19)[col]]) > 1e-7 for col in range(2))
                      for i, row in enumerate(episode["state"]))
        for task in episode["identity"].get("tasks", []) or ["unknown"]:
            counts = tasks[str(task)]
            counts["episodes"] += 1
            counts["changed_episodes"] += bool(changed)
            counts["passthrough_episodes"] += not changed
            counts["frames"] += len(target)
            counts["changed_frames"] += changed
            counts["valid_side_frames"] += sum(sum(row) for row in valid)

    derived_info = copy.deepcopy(info)
    derived_info["features"][TARGET_KEY] = dict(dtype="float32", shape=[2], names=["left", "right"])
    derived_info["features"][VALID_KEY] = dict(dtype="bool", shape=[2], names=["left", "right"])
    derived_info.update(total_episodes=len(selected), total_frames=len(values_all), total_data_files=len(data_files),
                        total_videos=len(videos), total_chunks=len({str(Path(name).parent) for name in data_files}))
    # Source episode IDs and global row IDs are intentionally preserved.
    # A sparse sample cannot truthfully advertise the parent's contiguous split.
    # ORION derives its configured split from the selected episode catalog.
    if len(selected) != info.get("total_episodes"):
        derived_info["splits"] = {}
    extensions = derived_info.setdefault("ngad_extensions", {})
    extensions["gripper_targets"] = dict(schema=EXPORT_SCHEMA, source_root=str(root), scope="selected_episodes",
                                          episode_indices=sorted(selected), index_layout="preserved_source_indices",
                                          values_key=TARGET_KEY, valid_key=VALID_KEY, units="normalized_0_1",
                                          side_order=["left", "right"], statistics_path="meta/gripper_target_stats.json",
                                          label_sources_path="meta/gripper_label_sources.json",
                                          publication_ready=False)
    _json(directory / "meta/info.json", derived_info)
    stats = dict(schema=EXPORT_SCHEMA, target=_statistics(values_all),
                 valid_side_frames=[sum(row[col] for row in flags_all) for col in range(2)],
                 side_order=["left", "right"], units="normalized_0_1", tasks=dict(tasks))
    _json(directory / "meta/gripper_target_stats.json", stats)
    label_sources = _label_sources(snapshots, directory)
    _json(directory / "meta/gripper_label_sources.json", label_sources)
    original_mask = _read(root / "dataset_configs/masks/mask.json")
    norm_source = root.parent / "dataset_configs/normalization/pushcart.json"
    training_config = None
    if norm_source.is_file():
        norm_destination = directory / "dataset_configs/normalization/pushcart.json"
        norm_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(norm_source, norm_destination)
        config = {"dataset": dict(normalization_stats_path=str(norm_destination),
                   dataset_dirs=[dict(name=original_mask["dataset"], path=str(directory),
                                      mask_and_mapping_path=str(directory / "dataset_configs/masks/mask.json"),
                                      parquet_row_addressing="episode_indexed")],
                   gripper_target_source="annotation", tactile={"enabled": False},
                   timeline=dict(rgb_rate_hz=10, action_steps_per_rgb_frame=2, tactile_steps_per_rgb_frame=8,
                                 source_phase_offsets=[0], randomize_source_phase=False, frame_ranges=[[0, 16]]),
                   validation_split=0, video={"required_camera_indices": []})}
        training_config = directory / "dataset_configs/gripper_training.yaml"
        # JSON is a YAML subset and avoids making export depend on PyYAML.
        _json(training_config, config)
    if any(_digest(path) != digest for path, digest in before.items()):
        raise ValueError("source changed during derivation")
    summary = dict(status="complete", scope="selected_episodes", publication_ready=False,
                   source_root=str(root), output_root=str(directory), episode_indices=sorted(selected),
                   episodes=len(selected), frames=len(values_all), video_files=len(videos), data_files=data_files,
                   source_columns_verified=original_columns_verified, source_sha256=before,
                   targets=TARGET_KEY, validity=VALID_KEY, tasks=dict(tasks),
                   valid_side_frames=stats["valid_side_frames"],
                   training_config=str(training_config) if training_config else None,
                   label_sources=label_sources,
                   episode_fingerprints_verified=[index for index, artifact in artifacts.items() if "episode_sha256" in artifact],
                   normalization_reused_from=str(norm_source) if norm_source.is_file() else None,
                   loader_required="ORION gripper target patch; episode_indexed addressing")
    _json(directory / "derivation_summary.json", summary)
    return summary


def prepare_release_manifest(release_roots, output, *, episode_targets=()):
    """Inventory complete releases using metadata only; do not derive their data.

    Label coverage is checked against catalog length/source metadata signatures.
    Exact Parquet-row identity and timestamps are checked later by derive_subset.
    An artifact is not a claim that its entire episode was visually reviewed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    roots = [Path(path).resolve(strict=True) for path in release_roots]
    if not roots or len(roots) != len(set(roots)) or len({path.name for path in roots}) != len(roots):
        raise ValueError("release roots must be nonempty, unique and have distinct names")
    targets_by_root = defaultdict(list)
    for item in episode_targets:
        value = _read(item) if isinstance(item, (str, Path)) else copy.deepcopy(item)
        source_root = value.get("identity", {}).get("dataset_root")
        if not isinstance(source_root, str):
            raise ValueError("target needs source dataset_root")
        targets_by_root[str(Path(source_root).resolve())].append(item)
    selected_roots = set()
    releases = []
    inventories = []
    for release in roots:
        subsets = []
        for info_path in sorted(release.glob("*/meta/info.json")):
            subset = info_path.parents[1]
            selected_roots.add(str(subset))
            info, mask = _read(info_path), _read(subset / "dataset_configs/masks/mask.json")
            valid_sides, _ = valid_sides_and_cameras(info, mask)
            if info.get("codebase_version") != "v3.0":
                raise ValueError("release planning only supports LeRobot v3.0")
            info_digest, mask_digest = _digest(info_path), _digest(subset / "dataset_configs/masks/mask.json")
            artifacts = _target_artifacts(targets_by_root.get(str(subset), []), subset)
            metadata_paths = sorted((subset / "meta/episodes").glob("chunk-*/*.parquet"))
            catalog = []
            inventory_tables = []
            for path in metadata_paths:
                table = pq.read_table(path, columns=["episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks"])
                catalog.extend(table.to_pylist())
                inventory_tables.append(table)
            if not catalog or len({row["episode_index"] for row in catalog}) != len(catalog):
                raise ValueError("release catalog is empty or has duplicate episode indices")
            frame_count = sum(row["length"] for row in catalog)
            if len(catalog) != info["total_episodes"] or frame_count != info["total_frames"]:
                raise ValueError("release metadata counts differ from info.json")
            if not set(artifacts).issubset({row["episode_index"] for row in catalog}):
                raise ValueError("label artifact episode is missing from release catalog")
            tasks = defaultdict(lambda: dict(episodes=0, frames=0, artifact_episodes=0,
                                            artifact_frames=0, valid_target_frames=0, valid_side_frames=0))
            coverage = dict(artifact_episodes=len(artifacts), artifact_frames=0,
                            valid_target_frames=0, valid_side_frames=[0, 0],
                            total_episodes=len(catalog), total_frames=frame_count,
                            valid_source_side_frames=frame_count * len(valid_sides))
            for row in catalog:
                artifact = artifacts.get(row["episode_index"])
                valid_frames, valid_side_count = 0, 0
                if artifact is not None:
                    identity = artifact["identity"]
                    for key, expected in (("info_sha256", info_digest), ("mask_sha256", mask_digest),
                                          ("dataset_from_index", row["dataset_from_index"]),
                                          ("dataset_to_index", row["dataset_to_index"])):
                        if identity.get(key) != expected:
                            raise ValueError("release target metadata identity differs: " + key)
                    values, flags = artifact[TARGET_KEY], artifact[VALID_KEY]
                    if len(values) != row["length"] or len(flags) != row["length"]:
                        raise ValueError("release target count differs from catalog length")
                    for target, valid in zip(values, flags):
                        if len(target) != 2 or len(valid) != 2 or any(type(flag) is not bool for flag in valid):
                            raise ValueError("release target rows require two values and boolean flags")
                        for side, col in (("left", 0), ("right", 1)):
                            if valid[col] and (side not in valid_sides or isinstance(target[col], bool)
                                or not isinstance(target[col], (int, float)) or not math.isfinite(target[col]) or not 0 <= target[col] <= 1):
                                raise ValueError("release target enables unavailable side or invalid value")
                            coverage["valid_side_frames"][col] += int(valid[col])
                        valid_frames += int(any(valid))
                        valid_side_count += sum(valid)
                    coverage["artifact_frames"] += row["length"]
                    coverage["valid_target_frames"] += valid_frames
                for task in row["tasks"] or ["unknown"]:
                    count = tasks[task]
                    count["episodes"] += 1
                    count["frames"] += row["length"]
                    if artifact is not None:
                        count["artifact_episodes"] += 1
                        count["artifact_frames"] += row["length"]
                        count["valid_target_frames"] += valid_frames
                        count["valid_side_frames"] += valid_side_count
            coverage["artifact_episode_fraction"] = len(artifacts) / len(catalog)
            coverage["artifact_frame_fraction"] = coverage["artifact_frames"] / frame_count
            coverage["valid_target_frame_fraction"] = coverage["valid_target_frames"] / frame_count
            coverage["valid_target_side_fraction"] = sum(coverage["valid_side_frames"]) / coverage["valid_source_side_frames"] if valid_sides else 0.
            relative = Path("inventories") / release.name / (subset.name + ".parquet")
            inventories.append((relative, pa.concat_tables(inventory_tables)))
            subsets.append(dict(name=subset.name, source_root=str(subset), total_episodes=len(catalog), total_frames=frame_count,
                                valid_sides=valid_sides, source_fps=info["fps"], source_info_sha256=info_digest,
                                source_mask_sha256=mask_digest, episode_inventory=relative.as_posix(),
                                expected_staging_relative=(Path("staging") / release.name / subset.name).as_posix(),
                                derivation_status="not_generated", coverage=coverage, tasks=dict(tasks),
                                label_sources=_label_sources(_target_snapshots(targets_by_root.get(str(subset), []))),
                                metadata_sources=[str(path) for path in metadata_paths],
                                normalization_source=str(release / "dataset_configs/normalization/pushcart.json")))
        if not subsets:
            raise ValueError("release has no direct subset/meta/info.json roots")
        releases.append(dict(name=release.name, source_root=str(release), subsets=subsets,
                             total_episodes=sum(row["total_episodes"] for row in subsets),
                             total_frames=sum(row["total_frames"] for row in subsets)))
    if set(targets_by_root) - selected_roots:
        raise ValueError("target artifacts contain a subset outside the selected releases")
    directory = pipeline.create_run(output, roots)
    for relative, table in inventories:
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
    for release in releases:
        for subset in release["subsets"]:
            subset["expected_staging_root"] = str(directory / subset["expected_staging_relative"])
    plan = dict(schema="close_gripper_release_plan_v1", status="prepared_not_derived", publication_ready=False,
                source_reads="info, masks and episode catalog only; no source data/video/model reads",
                source_data_row_validation="pending derivation; metadata identity checked only",
                coverage_semantics="Artifact counts show supplied target files, not complete visual review; valid-target counts are per frame and side.",
                releases=releases, release_count=len(releases), subset_count=sum(len(row["subsets"]) for row in releases),
                total_episodes=sum(row["total_episodes"] for row in releases),
                total_frames=sum(row["total_frames"] for row in releases),
                required_fields={TARGET_KEY: dict(dtype="float32", shape=[2], side_order=["left", "right"], units="normalized_0_1"),
                                 VALID_KEY: dict(dtype="bool", shape=[2], side_order=["left", "right"])},
                source_preservation=["all original columns and rows", "indices and timestamps", "tasks and video paths/offsets", "masks and pixel masks"],
                required_metadata=["info.features and exact counts", "episode/task inventories", "new target statistics and validity coverage",
                                   "per-task changed/passthrough counts", "source and label provenance", "ORION target-enabled YAML and compatible normalization"],
                pending=["full episode processing and source-row validation", "human reference and algorithm-effect acceptance",
                         "all inventoried subsets derived and checked", "actual loader checks on all staged subsets",
                         "user confirmation before formal publication"],
                task_count_semantics="Each task membership receives the episode/frame count; multi-task episodes may contribute to multiple task rows.")
    _json(directory / "release_plan.json", plan)
    return plan


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    subset = commands.add_parser("subset", help="Derive only explicitly selected episodes")
    subset.add_argument("--dataset-root", required=True)
    subset.add_argument("--targets", action="append", default=[])
    subset.add_argument("--episode", action="append", type=int)
    subset.add_argument("--output", required=True)
    plan = commands.add_parser("plan", help="Read metadata and prepare full-release inventories only")
    plan.add_argument("--release-root", action="append", required=True)
    plan.add_argument("--targets", action="append", default=[])
    plan.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "subset":
        result = derive_subset(args.dataset_root, args.targets, args.output, episode_indices=args.episode)
        print(json.dumps({key: result[key] for key in ("status", "episodes", "frames", "output_root")}, ensure_ascii=False))
    else:
        result = prepare_release_manifest(args.release_root, args.output, episode_targets=args.targets)
        print(json.dumps({key: result[key] for key in ("status", "release_count", "subset_count", "total_episodes", "total_frames")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
