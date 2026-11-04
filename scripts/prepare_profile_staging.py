"""Prepare complete-index datasets from a finished task-profile bulk run.

New numerical Parquets and support metadata are independent copies. Videos are
copied and verified by default; an explicit reference mode is available for
local preparation. No annotation validity is promoted or source written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.pipeline import create_run


FEATURES = {
    "evaluation.gripper_target": {"dtype": "float64", "shape": [2], "names": ["left", "right"]},
    "evaluation.gripper_rule_applied": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
    "evaluation.gripper_phase": {"dtype": "string", "shape": [2], "names": ["left", "right"]},
    "evaluation.gripper_source": {"dtype": "string", "shape": [2], "names": ["left", "right"]},
    "annotation.gripper_target": {"dtype": "float32", "shape": [2], "names": ["left", "right"]},
    "annotation.gripper_target_valid": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("support files must be real files and destinations must be new")
    before = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != before or digest(source) != before:
        raise ValueError("copy content or its source changed")
    return {"source": str(source), "destination": str(destination), "sha256": before,
            "bytes": source.stat().st_size}


def _stat(path):
    value = Path(path).stat()
    return {"device": value.st_dev, "inode": value.st_ino,
            "bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def copy_video(pair):
    source, destination = pair
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("video copy requires a physical source and new destination")
    before = _stat(source)
    h = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(2 ** 20), b""):
            h.update(block)
            dst.write(block)
    expected = h.hexdigest()
    after = _stat(source)
    if before != after or digest(destination) != expected:
        raise ValueError("source video changed or copied bytes differ")
    shutil.copystat(source, destination)
    if (_stat(destination)["device"], _stat(destination)["inode"]) == (before["device"], before["inode"]):
        raise ValueError("video copy must not share its source inode")
    return {"source": str(source), "destination": str(destination), "sha256": expected,
            "bytes": before["bytes"], "source_stat": before}


def prepare(bulk_run, output, *, video_mode="copy"):
    if video_mode not in ("copy", "source-reference"):
        raise ValueError("video_mode must be copy or source-reference")
    bulk = Path(bulk_run).resolve(strict=True)
    summary = read(bulk / "summary.json")
    if summary.get("status") != "complete" or summary.get("evaluation_only") is not True:
        raise ValueError("requires a complete evaluation-only bulk run")
    catalog = read(bulk / "task_profiles.json")
    datasets = summary["datasets"]
    if not datasets or len({d["source_root"] for d in datasets}) != len(datasets):
        raise ValueError("requires distinct dataset inputs")
    directory = create_run(output, [bulk, *[d["source_root"] for d in datasets]])
    implementation = Path(__file__).read_bytes()
    with (directory / "prepare_profile_staging.py").open("xb") as stream:
        stream.write(implementation)
    copied, results = [], []
    video_count, video_bytes = 0, 0
    journal = directory / "video_files.jsonl.gz"
    with gzip.open(journal, "xt", encoding="utf-8"):
        pass
    for dataset in datasets:
        source = Path(dataset["source_root"]).resolve(strict=True)
        numerical = Path(dataset["output_root"]).resolve(strict=True)
        if not numerical.is_relative_to(bulk / "staging"):
            raise ValueError("bulk output escapes its staging directory")
        relative = Path("staging") / (source.parent.name + "-gripcmd") / source.name
        target = directory / relative
        target.mkdir(parents=True)
        source_info = read(source / "meta/info.json")
        info = copy.deepcopy(source_info)
        for key, feature in FEATURES.items():
            if key in info["features"]:
                raise ValueError("this staging contract requires the original unannotated source")
            info["features"][key] = feature
        files = sorted((numerical / "data").glob("chunk-*/*.parquet"))
        originals = sorted((source / "data").glob("chunk-*/*.parquet"))
        if (len(files) != source_info["total_data_files"]
                or {p.relative_to(numerical) for p in files} != {p.relative_to(source) for p in originals}):
            raise ValueError("bulk data file inventory differs from the complete source dataset")
        for path in files:
            if not path.resolve(strict=True).is_relative_to(numerical):
                raise ValueError("bulk data path escapes its numerical staging directory")
            copied.append(copy_verified(path, target / path.relative_to(numerical)))
        for base in ("meta", "dataset_configs"):
            for path in sorted((source / base).rglob("*")):
                if path.is_file() and path != source / "meta/info.json":
                    copied.append(copy_verified(path, target / path.relative_to(source)))
        videos = source / "videos"
        if not videos.is_dir() or videos.is_symlink():
            raise ValueError("source videos must be the declared physical source directory")
        dataset_video_count, dataset_video_bytes = 0, 0
        if video_mode == "source-reference":
            (target / "videos").symlink_to(videos, target_is_directory=True)
            if (target / "videos").resolve(strict=True) != videos.resolve(strict=True):
                raise ValueError("video source reference changed")
        else:
            paths = sorted(videos.rglob("*.mp4"))
            if len(paths) != source_info["total_videos"]:
                raise ValueError("physical video inventory differs from the source info")
            with ThreadPoolExecutor(max_workers=4) as pool:
                for offset in range(0, len(paths), 64):
                    batch = paths[offset:offset + 64]
                    for path in batch:
                        if not path.resolve(strict=True).is_relative_to(videos):
                            raise ValueError("video file escapes its declared source directory")
                    pairs = [(p, target / p.relative_to(source)) for p in batch]
                    with gzip.open(journal, "at", encoding="utf-8") as video_log:
                        for record in pool.map(copy_video, pairs):
                            video_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                            dataset_video_count += 1
                            dataset_video_bytes += record["bytes"]
                    if offset % 4096 == 0:
                        print(json.dumps({"dataset": source.name, "videos_copied": dataset_video_count,
                                          "videos_total": len(paths), "source_root": str(source),
                                          "videos_copied_total": video_count + dataset_video_count}, ensure_ascii=False), flush=True)
            video_count += dataset_video_count
            video_bytes += dataset_video_bytes
        extension = info.setdefault("ngad_extensions", {})
        if "gripper_targets" in extension or "task_profile_evaluation" in extension:
            raise ValueError("source already has a gripper derivation extension")
        extension["gripper_targets"] = {
            "schema": "close_gripper_targets_v1", "scope": "complete_dataset",
            "index_layout": "preserved_source_indices", "values_key": "annotation.gripper_target",
            "valid_key": "annotation.gripper_target_valid", "units": "normalized_0_1",
            "side_order": ["left", "right"], "publication_ready": False,
            "new_training_labels": False,
        }
        extension["task_profile_evaluation"] = {
            "schema": "close_gripper.profile_bulk_staging.v1", "source_root": str(source),
            "bulk_run": str(bulk), "catalog_sha256": digest(bulk / "task_profiles.json"),
            "catalog_version": catalog["version"], "evaluation_only": True,
            "values_key": "evaluation.gripper_target", "applied_key": "evaluation.gripper_rule_applied",
            "grasp_semantics_verified": False, "media_independent": video_mode == "copy",
            "video_mode": video_mode, "video_source": str(videos),
            "publication_ready": False,
        }
        save(target / "meta/info.json", info)
        normalization = source.parent / "dataset_configs/normalization/pushcart.json"
        norm_target = target / "dataset_configs/normalization/pushcart.json"
        if not normalization.is_file():
            raise ValueError("source normalization required for actual loader verification")
        copied.append(copy_verified(normalization, norm_target))
        mask_path = target / "dataset_configs/masks/mask.json"
        mask = read(mask_path)
        # ORION validates the original mask contract; custom annotation fields
        # are selected by the explicitly patched loader, not new mask keys.
        if digest(mask_path) != digest(source / "dataset_configs/masks/mask.json"):
            raise ValueError("source mask was not preserved")
        config = {"dataset": {
            "normalization_stats_path": str(norm_target),
            "dataset_dirs": [{"name": mask["dataset"], "path": str(target),
                              "mask_and_mapping_path": str(mask_path),
                              "parquet_row_addressing": "episode_indexed"}],
            "gripper_target_source": "observation", "tactile": {"enabled": False},
            "timeline": {"rgb_rate_hz": 10, "action_steps_per_rgb_frame": 2,
                         "tactile_steps_per_rgb_frame": 8, "source_phase_offsets": [0],
                         "randomize_source_phase": False, "frame_ranges": [[0, 16]]},
            "validation_split": 0, "video": {"required_camera_indices": []},
        }}
        save(target / "dataset_configs/gripper_training.yaml", config)
        item = {"source_root": str(source), "output_root": str(target), "relative_output_root": str(relative),
                "episodes": source_info["total_episodes"], "frames": source_info["total_frames"],
                "data_files": len(files), "source_mask_sha256": digest(mask_path),
                "source_info_sha256": digest(source / "meta/info.json"),
                "selected_episodes": dataset["selected_episodes"], "selected_frames": dataset["selected_frames"],
                "changed_episodes": dataset["changed_episodes"], "changed_frames": dataset["changed_frames"],
                "training_config": str(target / "dataset_configs/gripper_training.yaml"),
                "valid_side_frames": [0, 0], "label_sources": [],
                "evaluation_only": True, "media_independent": video_mode == "copy", "publication_ready": False,
                "video_mode": video_mode, "video_files_copied": dataset_video_count,
                "video_bytes_copied": dataset_video_bytes,
                "video_source": str(videos), "video_reference": str(target / "videos")}
        save(target / "derivation_summary.json", item)
        results.append(item)
        print(json.dumps({"dataset": source.name, "episodes": item["episodes"],
                          "changed_episodes": item["changed_episodes"], "media": video_mode}, ensure_ascii=False), flush=True)
    save(directory / "copied_files.json", copied)
    if Path(__file__).read_bytes() != implementation:
        raise ValueError("staging implementation changed during preparation")
    report = {"schema": "close_gripper.profile_bulk_delivery.v1", "status": "complete",
              "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
              "bulk_run": str(bulk), "bulk_summary_sha256": digest(bulk / "summary.json"),
              "catalog_sha256": digest(bulk / "task_profiles.json"), "datasets": results,
              "evaluation_only": True, "new_training_labels": False,
              "media_independent": video_mode == "copy", "publication_ready": False, "source_writes": 0,
              "video_mode": video_mode, "video_files_copied": video_count,
              "video_bytes_copied": video_bytes, "video_journal": "video_files.jsonl.gz",
              "copied_files": len(copied), "copied_bytes": sum(r["bytes"] for r in copied),
              "model_requests": 0}
    save(directory / "delivery_index.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-mode", choices=("copy", "source-reference"), default="copy")
    args = parser.parse_args()
    result = prepare(args.bulk_run, args.output, video_mode=args.video_mode)
    print(json.dumps({"status": result["status"], "datasets": len(result["datasets"]),
                      "publication_ready": False}, ensure_ascii=False))
