#!/usr/bin/env python3
"""Independently verify metadata, training isolation and copied video bytes.

Numerical correctness belongs to verify_profile_dataset.py. This audit binds
the delivered Parquets to that bulk output and checks every media copy against
its original source. Source-reference mode is explicitly non-independent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_profile_dataset import Audit, below, digest, json_lines, read

WORKSPACE = Path(__file__).resolve().parents[1]
FEATURES = {
    "evaluation.gripper_target": ("float64", [2]),
    "evaluation.gripper_rule_applied": ("bool", [2]),
    "evaluation.gripper_phase": ("string", [2]),
    "evaluation.gripper_source": ("string", [2]),
    "annotation.gripper_target": ("float32", [2]),
    "annotation.gripper_target_valid": ("bool", [2]),
}


def file_stat(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def verify_copy(record, expected_source, expected_destination, *, video=False):
    """Check both physical files, not merely the preparer's digest journal."""
    audit = Audit()
    source, destination = Path(expected_source), Path(expected_destination)
    detail = str(destination)
    try:
        if not audit.check(record["source"] == str(source) and record["destination"] == str(destination),
                           "copy_exact_identity", detail):
            return audit.report()
        if not audit.check(source.is_file() and destination.is_file() and not source.is_symlink()
                           and not destination.is_symlink(), "copy_physical_files", detail):
            return audit.report()
        before, copied = file_stat(source), file_stat(destination)
        audit.check((before["device"], before["inode"]) != (copied["device"], copied["inode"]),
                    "copy_independent_inode", detail)
        audit.check(digest(source) == record["sha256"] == digest(destination), "copy_source_and_target_sha256", detail)
        audit.check(before == file_stat(source) and copied == file_stat(destination)
                    and before["bytes"] == copied["bytes"] == record["bytes"], "copy_source_and_target_stable", detail)
        if video:
            audit.check(before == record["source_stat"], "video_original_frozen_stat", detail)
    except (KeyError, ValueError, OSError) as exc:
        audit.check(False, "invalid_copy_record", [detail, str(exc)])
    return audit.report()


def verify_info_and_training(source, target, dataset, bulk, mode, audit):
    original, delivered = read(source / "meta/info.json"), read(target / "meta/info.json")
    restored = copy.deepcopy(delivered)
    for key, (dtype, shape) in FEATURES.items():
        feature = restored["features"].pop(key, None)
        audit.check(feature == dict(dtype=dtype, shape=shape, names=["left", "right"]), "new_feature_contract", [str(target), key])
    extensions = restored.get("ngad_extensions", {})
    annotations = extensions.pop("gripper_targets", {})
    evaluation = extensions.pop("task_profile_evaluation", {})
    if not extensions and "ngad_extensions" not in original:
        restored.pop("ngad_extensions", None)
    audit.check(restored == original, "all_original_info_fields_preserved", str(target))
    audit.check(annotations.get("scope") == "complete_dataset"
                and annotations.get("index_layout") == "preserved_source_indices"
                and annotations.get("values_key") == "annotation.gripper_target"
                and annotations.get("valid_key") == "annotation.gripper_target_valid"
                and annotations.get("new_training_labels") is False and annotations.get("publication_ready") is False,
                "annotation_fallback_declared", str(target))
    audit.check(evaluation.get("source_root") == str(source) and evaluation.get("bulk_run") == str(bulk)
                and evaluation.get("catalog_sha256") == digest(bulk / "task_profiles.json")
                and evaluation.get("evaluation_only") is True and evaluation.get("grasp_semantics_verified") is False
                and evaluation.get("video_mode") == mode and evaluation.get("media_independent") is (mode == "copy")
                and evaluation.get("publication_ready") is False, "evaluation_metadata_provenance", str(target))
    config = read(target / "dataset_configs/gripper_training.yaml")["dataset"]
    roots = config.get("dataset_dirs", [])
    audit.check(config.get("gripper_target_source") == "observation" and len(roots) == 1
                and roots[0].get("path") == str(target)
                and roots[0].get("mask_and_mapping_path") == str(target / "dataset_configs/masks/mask.json")
                and roots[0].get("parquet_row_addressing") == "episode_indexed"
                and config.get("normalization_stats_path") == str(target / "dataset_configs/normalization/pushcart.json"),
                "training_default_observation_isolation", str(target))
    audit.check(digest(source / "dataset_configs/masks/mask.json") == digest(target / "dataset_configs/masks/mask.json"),
                "mask_contract_unchanged", str(target))
    audit.check(read(target / "derivation_summary.json") == dataset
                and dataset.get("valid_side_frames") == [0, 0] and dataset.get("label_sources") == [],
                "dataset_summary_no_new_training_labels", str(target))
    return original


def audit_delivery(delivery, output, *, workers=4):
    import pyarrow.parquet as pq
    delivery, output = Path(delivery).resolve(strict=True), Path(output).absolute()
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    allowed = (WORKSPACE / "runs").resolve()
    if (not output.resolve().is_relative_to(allowed) or output.resolve() == allowed
            or any(p.is_symlink() for p in (output, *output.parents))
            or output.resolve().is_relative_to(delivery) or delivery.is_relative_to(output.resolve())):
        raise ValueError("output must be a fresh, separate workspace runs directory")
    output.mkdir(parents=True, exist_ok=False)
    started, audit = time.monotonic(), Audit()
    media_count = media_bytes = support_count = 0
    mode, bulk = None, None
    try:
        index = read(delivery / "delivery_index.json")
        bulk = Path(index["bulk_run"]).resolve(strict=True)
        summary = read(bulk / "summary.json")
        mode = index["video_mode"]
        audit.check(index["status"] == summary["status"] == "complete" and mode in ("copy", "source-reference"), "complete_delivery")
        audit.check(index.get("evaluation_only") is True and index.get("new_training_labels") is False
                    and index.get("publication_ready") is False and index.get("model_requests") == 0
                    and index.get("source_writes") == 0 and index.get("media_independent") is (mode == "copy"),
                    "delivery_scope")
        audit.check(index["bulk_summary_sha256"] == digest(bulk / "summary.json")
                    and index["catalog_sha256"] == digest(bulk / "task_profiles.json"), "bulk_output_binding")
        audit.check(index["implementation_sha256"] == digest(delivery / "prepare_profile_staging.py"), "staging_implementation_snapshot")
        datasets = {d["source_root"]: d for d in index["datasets"]}
        original_datasets = {d["source_root"]: d for d in summary["datasets"]}
        audit.check(len(datasets) == len(index["datasets"]) and set(datasets) == set(original_datasets), "all_datasets_delivered_once")
        expected_copies, media_sets, target_by_source = {}, {}, {}
        for source_name, dataset in datasets.items():
            source = Path(source_name).resolve(strict=True)
            target = below(delivery, Path("staging") / (source.parent.name + "-gripcmd") / source.name)
            target_by_source[source_name] = target
            numerical = Path(original_datasets[source_name]["output_root"]).resolve(strict=True)
            numerical.relative_to(bulk / "staging")
            audit.check(dataset["output_root"] == str(target) and dataset["video_mode"] == mode, "dataset_exact_destination", source_name)
            info = verify_info_and_training(source, target, dataset, bulk, mode, audit)
            source_data = {p.relative_to(source).as_posix() for p in (source / "data").glob("chunk-*/*.parquet")}
            target_data = {p.relative_to(target).as_posix() for p in (target / "data").glob("chunk-*/*.parquet")}
            numerical_data = {p.relative_to(numerical).as_posix() for p in (numerical / "data").glob("chunk-*/*.parquet")}
            audit.check(source_data == target_data == numerical_data and len(source_data) == info["total_data_files"],
                        "complete_derived_data_file_inventory", source_name)
            for relative in numerical_data:
                expected_copies[str(below(target, relative))] = str(below(numerical, relative))
            for base in ("meta", "dataset_configs"):
                for path in (source / base).rglob("*"):
                    if path.is_file() and path != source / "meta/info.json":
                        expected_copies[str(below(target, path.relative_to(source)))] = str(path)
            expected_copies[str(target / "dataset_configs/normalization/pushcart.json")] = str(source.parent / "dataset_configs/normalization/pushcart.json")
            videos = {p.relative_to(source).as_posix() for p in (source / "videos").rglob("*.mp4")}
            media_sets[source_name] = videos
            audit.check(len(videos) == info["total_videos"], "complete_source_video_inventory", source_name)
            if mode == "copy":
                copied_videos = {p.relative_to(target).as_posix() for p in (target / "videos").rglob("*.mp4")}
                audit.check(not (target / "videos").is_symlink() and copied_videos == videos,
                            "complete_physical_video_inventory", source_name)
            else:
                audit.check((target / "videos").is_symlink()
                            and (target / "videos").resolve(strict=True) == source / "videos",
                            "explicit_exact_media_reference", source_name)
            rows = [r for p in (source / "meta/episodes").glob("chunk-*/*.parquet")
                    for r in pq.read_table(p, use_threads=False).to_pylist()]
            audit.check(len(rows) == info["total_episodes"] and sum(r["length"] for r in rows) == info["total_frames"],
                        "complete_episode_frame_metadata_counts", source_name)
            for row in rows:
                for camera, feature in info["features"].items():
                    if feature["dtype"] != "video":
                        continue
                    prefix = f"videos/{camera}/"
                    relative = info["video_path"].format(video_key=camera, chunk_index=row[prefix + "chunk_index"], file_index=row[prefix + "file_index"])
                    audit.check(relative in videos and 0 <= row[prefix + "from_timestamp"] < row[prefix + "to_timestamp"],
                                "episode_video_reference_and_offset", [source_name, row["episode_index"], camera])
        copied = read(delivery / "copied_files.json")
        audit.check(len(copied) == len(expected_copies) == index["copied_files"]
                    and {r["destination"] for r in copied} == set(expected_copies), "all_support_copies_recorded_once")
        for record in copied:
            expected = expected_copies.get(record["destination"])
            if expected is not None:
                audit.merge(verify_copy(record, expected, record["destination"]))
                support_count += 1
        audit.check(sum(r["bytes"] for r in copied) == index["copied_bytes"], "support_byte_total")
        seen, by_dataset = set(), {name: [0, 0] for name in datasets}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch = []
            def flush():
                nonlocal media_count, media_bytes
                for result, source_name, record in batch:
                    audit.merge(result.result())
                    media_count += 1; media_bytes += record["bytes"]
                    by_dataset[source_name][0] += 1; by_dataset[source_name][1] += record["bytes"]
                batch.clear()
            for record in json_lines(below(delivery, index["video_journal"])):
                source_name = next((name for name in datasets if Path(record["source"]).is_relative_to(Path(name) / "videos")), None)
                if not audit.check(mode == "copy" and source_name is not None, "journal_declared_source", record.get("source")):
                    continue
                relative = Path(record["source"]).relative_to(source_name).as_posix()
                destination = below(target_by_source[source_name], relative)
                key = (source_name, relative)
                audit.check(key not in seen and relative in media_sets[source_name], "video_journal_exact_inventory", str(destination))
                seen.add(key)
                batch.append((pool.submit(verify_copy, record, Path(source_name) / relative, destination, video=True), source_name, record))
                if len(batch) >= workers * 32:
                    flush()
                    if media_count % (workers * 4096) == 0:
                        print(json.dumps(dict(event="audit_videos", count=media_count, bytes=media_bytes, failures=audit.failure_count)), flush=True)
            flush()
        expected_videos = {(name, path) for name, paths in media_sets.items() for path in paths} if mode == "copy" else set()
        audit.check(seen == expected_videos and media_count == len(expected_videos) == index["video_files_copied"], "every_video_verified_once")
        audit.check(media_bytes == index["video_bytes_copied"], "video_byte_total")
        for name, counts in by_dataset.items():
            audit.check(counts == [datasets[name]["video_files_copied"], datasets[name]["video_bytes_copied"]], "dataset_video_totals", name)
    except (KeyError, ValueError, OSError, TypeError) as exc:
        audit.check(False, "invalid_delivery_contract", [type(exc).__name__, str(exc)])
    result = dict(audit.report(), delivery=str(delivery), script_sha256=digest(__file__), workers=workers,
                  bulk_run=str(bulk) if bulk is not None else None,
                  delivery_index_sha256=digest(delivery / "delivery_index.json") if (delivery / "delivery_index.json").is_file() else None,
                  elapsed_s=time.monotonic()-started, video_mode=mode, copied_videos_verified=media_count,
                  video_bytes_verified=media_bytes, support_files_verified=support_count,
                  limitations=["Checks independent copies, metadata and training configuration; numerical curves use the separate bulk audit.",
                               "Does not claim video decoding, grasp accuracy, formal publication or training effects."])
    with (output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = audit_delivery(args.delivery, args.output, workers=args.workers)
    print(json.dumps({k: result[k] for k in ("status", "failure_count", "copied_videos_verified", "video_bytes_verified")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
