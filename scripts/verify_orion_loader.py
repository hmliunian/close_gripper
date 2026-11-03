#!/usr/bin/env python3
"""Prepare a reviewable ORION patch and verify its real DataLoader on samples.

The external checkout is read-only. Only its dataloader package is copied into a
fresh workspace run; dirty tracked changes are preserved and separately recorded.
Run with a Python environment containing torch, pyarrow, PyYAML and video codecs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from close_gripper.pipeline import create_run
from close_gripper.export import TARGET_KEY, VALID_KEY


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("ORION source differs from the reviewed patch context: " + repr(old[:100]))
    return text.replace(old, new, 1)


def patched_sources(original):
    """Exact contexts fail closed when the upstream implementation changes."""
    result = dict(original)
    config = result["pretrain/config.py"]
    config = replace_once(config, "    tactile_enabled: bool = True\n", "    tactile_enabled: bool = True\n    gripper_target_source: str = \"observation\"\n")
    config = replace_once(config, '            "tactile",\n', '            "tactile",\n            "gripper_target_source",\n')
    config = replace_once(config, '        roots = tuple(DatasetRootConfig.from_mapping(root) for root in roots_value)\n', '''        gripper_target_source = value.get("gripper_target_source", "observation")
        if gripper_target_source not in ("observation", "annotation"):
            raise ValueError("gripper_target_source must be observation or annotation.")
        roots = tuple(DatasetRootConfig.from_mapping(root) for root in roots_value)
''')
    config = replace_once(config, '            tactile_enabled=_parse_tactile_section(value.get("tactile")),\n', '            tactile_enabled=_parse_tactile_section(value.get("tactile")),\n            gripper_target_source=gripper_target_source,\n')
    config = replace_once(config, '            "tactile_enabled": self.tactile_enabled,\n', '            "tactile_enabled": self.tactile_enabled,\n            "gripper_target_source": self.gripper_target_source,\n')
    result["pretrain/config.py"] = config

    table = result["pretrain/table.py"]
    addition = '''        value_keys.extend(key for key in (
            "annotation.gripper_target", "annotation.gripper_target_valid"
        ) if field_mask.get(key, False))
'''
    table = replace_once(table, '        value_keys = [STATE_KEY] if field_mask[STATE_KEY] else []\n', '        value_keys = [STATE_KEY] if field_mask[STATE_KEY] else []\n' + addition)
    table = replace_once(table, '            canonical_columns.append(STATE_KEY)\n', '''            canonical_columns.append(STATE_KEY)
        canonical_columns.extend(key for key in (
            "annotation.gripper_target", "annotation.gripper_target_valid"
        ) if field_mask.get(key, False))
''')
    result["pretrain/table.py"] = table

    canonical = result["pretrain/canonical.py"]
    canonical = replace_once(canonical, 'from .backends import create_storage_backends\n', 'from .backends import create_storage_backends\nfrom .gripper_targets import interpolate_gripper_target_pairs, validate_uniform_source_timestamps\n')
    canonical = replace_once(canonical, '        tactile_enabled: bool = True,\n        rot6d_pinned: bool = True,\n', '        tactile_enabled: bool = True,\n        rot6d_pinned: bool = True,\n        gripper_target_source: str = "observation",\n')
    canonical = replace_once(canonical, '        self.tactile_enabled = bool(tactile_enabled)\n', '''        self.tactile_enabled = bool(tactile_enabled)
        if gripper_target_source not in ("observation", "annotation"):
            raise ValueError("gripper_target_source must be observation or annotation.")
        if gripper_target_source == "annotation" and normalization_stats_path is None:
            raise ValueError("Gripper action annotations require canonical state/action mode.")
        self.gripper_target_source = gripper_target_source
''')
    canonical = replace_once(canonical, '        # mask contract 是字段是否存在的唯一事实来源，不按数据集名称猜测。\n', '''        if self.gripper_target_source == "annotation":
            # The new labels are already normalized. Do not normalize them twice
            # under a physical-unit statistics file from a different dataset.
            if not (
                torch.equal(self._tcp_transform.gripper_open_value, torch.ones(2))
                and torch.equal(self._tcp_transform.gripper_closed_value, torch.zeros(2))
            ):
                raise ValueError("Normalized gripper annotations require closed=0/open=1 stats.")
        # mask contract 是字段是否存在的唯一事实来源，不按数据集名称猜测。
''')
    canonical = replace_once(canonical, '            source_fps = float(info.get("fps", 0.0))\n', '''            if self.gripper_target_source == "annotation":
                for key, dtype in (("annotation.gripper_target", "float32"),
                                   ("annotation.gripper_target_valid", "bool")):
                    feature = info.get("features", {}).get(key, {})
                    if feature.get("dtype") != dtype or feature.get("shape") != [2]:
                        raise ValueError(f"{root} requires {key}: {dtype}[2].")
            source_fps = float(info.get("fps", 0.0))
''')
    canonical = replace_once(canonical, '''            expected_episode_start = 0
            for episode in episodes:
                if episode["dataset_from_index"] != expected_episode_start:
''', '''            extension = info.get("ngad_extensions", {}).get("gripper_targets", {})
            source_index_sample = (
                extension.get("schema") == "close_gripper_targets_v1"
                and extension.get("scope") == "selected_episodes"
                and extension.get("index_layout") == "preserved_source_indices"
            )
            if source_index_sample:
                if configured["parquet_row_addressing"] != "episode_indexed":
                    raise ValueError("Source-index sample requires episode_indexed addressing.")
                if sorted(extension.get("episode_indices", [])) != sorted(
                    episode["episode_index"] for episode in episodes
                ):
                    raise ValueError("Selected episode manifest differs from the catalog.")
            expected_episode_start = 0
            for episode in episodes:
                bad_offset = (
                    episode["dataset_from_index"] < expected_episode_start
                    if source_index_sample
                    else episode["dataset_from_index"] != expected_episode_start
                )
                if bad_offset:
''')
    canonical = replace_once(canonical, '            if expected_episode_start != table_record["num_frames"]:\n', '            if not source_index_sample and expected_episode_start != table_record["num_frames"]:\n')
    canonical = replace_once(canonical, '        # 落盘 Action 永远不是监督输入；模型 Action 统一由 absolute State 重建。\n', '        # Pose actions are rebuilt from State; explicit gripper annotations are optional.\n')
    canonical = replace_once(canonical, '            CANONICAL_ACTION_KEY: False,\n', '''            CANONICAL_ACTION_KEY: False,
            "annotation.gripper_target": self.gripper_target_source == "annotation",
            "annotation.gripper_target_valid": self.gripper_target_source == "annotation",
''')
    canonical = replace_once(canonical, '''        state_element_mask = meta["state_element_mask"].clone()
        action_element_mask = meta["action_element_mask"].clone()
''', '''        # Keep physical observations and the physical anchor separate from labels.
        action_state_grid = absolute_state_grid
        if self.gripper_target_source == "annotation":
            targets, valid = interpolate_gripper_target_pairs(
                [rows[int(frame)]["annotation.gripper_target"] for frame in state_lower.tolist()],
                [rows[int(frame)]["annotation.gripper_target"] for frame in state_upper.tolist()],
                [rows[int(frame)]["annotation.gripper_target_valid"] for frame in state_lower.tolist()],
                [rows[int(frame)]["annotation.gripper_target_valid"] for frame in state_upper.tolist()],
                state_fraction.tolist(),
            )
            action_state_grid = absolute_state_grid.clone()
            action_state_grid[..., 9] = torch.where(
                torch.tensor(valid, dtype=torch.bool),
                torch.tensor(targets, dtype=absolute_state_grid.dtype),
                absolute_state_grid[..., 9],
            )
        state_element_mask = meta["state_element_mask"].clone()
        action_element_mask = meta["action_element_mask"].clone()
''')
    canonical = replace_once(canonical, '''            absolute_state_grid[0],
            absolute_state_grid[1:],
            action_element_mask,
''', '''            absolute_state_grid[0],
            action_state_grid[1:],
            action_element_mask,
''')
    canonical = replace_once(canonical, '        del source_indices, timestamp_start\n', '''        if self.gripper_target_source == "annotation":
            validate_uniform_source_timestamps(
                source_indices.tolist(), actual_timestamps.tolist(),
                float(timestamp_start), self._root_meta[episode["root_index"]]["source_fps"],
            )
''')
    result["pretrain/canonical.py"] = canonical
    result["pretrain/gripper_targets.py"] = (WORKSPACE / "close_gripper/training.py").read_text(encoding="utf-8")
    return result


def prepare_loader(source, directory):
    source = Path(source).resolve(strict=True)
    git_base = ["git", "--no-optional-locks", "-c", f"safe.directory={source}", "-C", str(source)]
    def git(*args):
        return subprocess.check_output([*git_base, *args], text=True)
    version = dict(source_root=str(source), commit=git("rev-parse", "HEAD").strip(),
                   dirty_status=git("status", "--short"), dirty_stat=git("diff", "--stat"),
                   deployment="isolated_workspace_copy_only")
    dirty_diff = git("diff", "--", "src/dataloader")
    package = source / "src/dataloader"
    original_bytes = {str(p.relative_to(package)): p.read_bytes() for p in package.rglob("*.py")}
    destination = directory / "loader/src/dataloader"
    shutil.copytree(package, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    names = ("pretrain/config.py", "pretrain/table.py", "pretrain/canonical.py")
    original = {name: original_bytes[name].decode("utf-8") for name in names}
    modified = patched_sources(original)
    patch = []
    for name, contents in modified.items():
        (destination / name).write_text(contents, encoding="utf-8")
        patch.extend(difflib.unified_diff(original.get(name, "").splitlines(keepends=True), contents.splitlines(keepends=True),
                     fromfile="a/src/dataloader/" + name if name in original else "/dev/null",
                     tofile="b/src/dataloader/" + name))
    if any((package / name).read_bytes() != data for name, data in original_bytes.items()):
        raise ValueError("External loader source changed during isolated preparation")
    (directory / "orion_gripper_targets.patch").write_text("".join(patch), encoding="utf-8")
    (directory / "upstream_worktree.diff").write_text(dirty_diff, encoding="utf-8")
    version.update(source_files_byte_checked=len(original_bytes), source_unchanged=True,
                   patched_files=list(modified), inherited_dirty_diff="upstream_worktree.diff",
                   patch_base="recorded commit plus inherited upstream_worktree.diff")
    version["patch_sha256"] = hashlib.sha256("".join(patch).encode("utf-8")).hexdigest()
    version["local_verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    version["local_training_sha256"] = hashlib.sha256((WORKSPACE / "close_gripper/training.py").read_bytes()).hexdigest()
    write_json(directory / "source_version.json", version)
    return version


def verify_dataset(directory, derived_root, ordinal):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import Subset
    from dataloader import make_dataloader
    from dataloader.orion_pretrain import get_dataloader

    root = Path(derived_root).resolve(strict=True)
    derivation = json.loads((root / "derivation_summary.json").read_text())
    base_config = json.loads((root / "dataset_configs/gripper_training.yaml").read_text())
    # Four RGB frames suffice to test exact / half-step times and episode tails.
    base_config["dataset"]["timeline"]["frame_ranges"] = [[0, 4]]
    loaders = []
    for mode in ("observation", "annotation"):
        config = copy.deepcopy(base_config)
        config["dataset"]["gripper_target_source"] = mode
        path = directory / f"dataset_{ordinal}_{mode}.yaml"
        write_json(path, config)
        loaders.append(get_dataloader(data_dir=str(path), batch_size=1, num_workers=0,
                                      shuffle=False, pin_memory=False))
    baseline, target = [loader.dataset for loader in loaders]
    tables = {}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", TARGET_KEY, VALID_KEY]).to_pylist():
            tables.setdefault(row["episode_index"], {})[row["frame_index"]] = row
    indices, offset = [], 0
    for episode in target._episodes:
        rows = tables[episode["episode_index"]]
        source_fps = target._root_meta[episode["root_index"]]["source_fps"]
        source_times = [rows[index]["timestamp"] for index in range(episode["length"])]
        # Independent oracle reads every stored timestamp. It does not call the
        # loader's grid helper or infer source time from frame numbers alone.
        if any(not math.isfinite(value) for value in source_times) or any(right <= left for left, right in zip(source_times, source_times[1:])):
            raise AssertionError("Stored source timestamps are not finite and strictly increasing")
        if any(abs(value - (source_times[0] + index / source_fps)) > 1e-6 for index, value in enumerate(source_times)):
            raise AssertionError("Stored source timestamps do not match the declared uniform grid")
        candidates = {0, episode["rgb_target_length"] - 1}
        flagged = [index for index, row in rows.items() if any(row[VALID_KEY])]
        if flagged:
            candidates.update((flagged[0], max(0, flagged[0] - 1), flagged[-1]))
            for index in flagged:
                if index + 1 in rows and rows[index][VALID_KEY] != rows[index + 1][VALID_KEY]:
                    candidates.add(index)
                    break
        indices.extend(offset + index for index in sorted(candidates) if index < episode["rgb_target_length"])
        offset += episode["rgb_target_length"]
    def selected_loader(dataset):
        factory = make_dataloader(lambda **_: Subset(dataset, indices))
        return factory(data_dir=None, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
    cases, changed, checked, valid_updates, invalid_fallback = [], 0, 0, 0, 0
    for before, after in zip(selected_loader(baseline), selected_loader(target)):
        if not torch.equal(before["state"], after["state"]):
            raise AssertionError("Physical state changed when labels were enabled")
        other = [index for index in range(after["action"].shape[-1]) if index not in (9, 19)]
        if not torch.equal(before["action"][..., other], after["action"][..., other]):
            raise AssertionError("Non-gripper action dimensions changed")
        for key in ("video", "action_timestamps", "source_frame_indices", "action_valid", "camera_mask"):
            if not torch.equal(before[key], after[key]):
                raise AssertionError("Label configuration changed " + key)
        info = after["data_info"]
        episode_index = int(info["episode_index"][0])
        anchor = int(info["anchor_rgb_index"][0])
        assert float(info["action_rate_hz"][0]) == 20
        assert float(info["source_fps"][0]) == 10
        rows = tables[episode_index]
        times = after["action_timestamps"][0].reshape(-1).tolist()
        source_times = [rows[index]["timestamp"] for index in range(len(rows))]
        live = after["action_valid"][0].reshape(-1).tolist()
        actual = after["action"][0].reshape(-1, after["action"].shape[-1])[:, [9, 19]].numpy()
        original = before["action"][0].reshape(-1, before["action"].shape[-1])[:, [9, 19]].numpy()
        expected = original.copy()
        for step, (action_time, enabled) in enumerate(zip(times, live)):
            if step and abs(times[step] - times[step - 1] - .05) > 1e-8:
                raise AssertionError("Action timestamps are not a 20 Hz grid")
            if not enabled:
                if np.any(actual[step] != 0):
                    raise AssertionError("Episode tail padding is not masked")
                continue
            upper = bisect_left(source_times, action_time)
            exact = [index for index in (upper - 1, upper) if 0 <= index < len(source_times) and abs(source_times[index] - action_time) <= 1e-6]
            if exact:
                lower = upper = min(exact, key=lambda index: abs(source_times[index] - action_time))
                fraction = 0.
            else:
                lower = upper - 1
                if lower < 0 or upper >= len(source_times):
                    raise AssertionError("Valid action escaped the stored episode time range")
                fraction = (action_time - source_times[lower]) / (source_times[upper] - source_times[lower])
            if lower not in rows or upper not in rows:
                raise AssertionError("Valid action escaped its episode")
            for side in range(2):
                ok = rows[lower][VALID_KEY][side] and (fraction == 0 or rows[upper][VALID_KEY][side])
                if ok:
                    expected[step, side] = (1 - fraction) * rows[lower][TARGET_KEY][side] + fraction * rows[upper][TARGET_KEY][side]
                    valid_updates += 1
                else:
                    invalid_fallback += 1
                checked += 1
        if not np.allclose(actual, expected, atol=2e-6, rtol=0):
            raise AssertionError("20 Hz action differs from independently evaluated stored labels")
        count = int(np.count_nonzero(np.abs(actual - original) > 1e-7))
        changed += count
        cases.append(dict(episode_index=episode_index, anchor_rgb_index=anchor, changed_action_values=count,
                          action_timestamps=times, action_valid=live,
                          old_grippers=original.tolist(), new_grippers=actual.tolist()))
    return dict(dataset_root=str(root), status="passed", samples=len(cases), checked_gripper_values=checked,
                changed_action_values=changed, valid_label_values=valid_updates, fallback_values=invalid_fallback,
                physical_state_unchanged=True, other_action_dimensions_unchanged=True,
                video_timestamps_masks_unchanged=True, action_rate_hz=20,
                source_timestamps_checked="every stored source row; oracle brackets by actual timestamp",
                label_sources=derivation.get("label_sources", []),
                controlled_test_fixture=(root / "controlled_target_fixture.json").is_file(),
                effect_evidence="loader consumption only; no algorithm or training-effect conclusion",
                loader="dataloader.orion_pretrain.get_dataloader / dataloader.make_dataloader", cases=cases)


def verify_nonuniform_rejection(directory, derived_root):
    """A fresh two-row fixture must fail through the actual public DataLoader."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataloader.orion_pretrain import get_dataloader

    source = Path(derived_root).resolve(strict=True)
    fixture = directory / "nonuniform_timestamp_fixture"
    fixture.mkdir()
    shutil.copytree(source / "dataset_configs", fixture / "dataset_configs")
    shutil.copytree(source / "videos", fixture / "videos")
    (fixture / "meta/episodes/chunk-000").mkdir(parents=True)
    shutil.copy2(source / "meta/tasks.parquet", fixture / "meta/tasks.parquet")
    info = json.loads((source / "meta/info.json").read_text())
    metadata = pq.read_table(sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))[0]).slice(0, 1)
    row = metadata.to_pylist()[0]
    for column, value in (("length", 2), ("dataset_to_index", row["dataset_from_index"] + 2)):
        metadata = metadata.set_column(metadata.column_names.index(column), column,
                                       pa.array([value], type=metadata.schema.field(column).type))
    pq.write_table(metadata, fixture / "meta/episodes/chunk-000/file-000.parquet")
    relative = info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
    table = pq.read_table(source / relative, filters=[("episode_index", "=", row["episode_index"])]).slice(0, 2)
    if table.num_rows != 2:
        raise AssertionError("Timestamp regression needs at least two source rows")
    table = table.set_column(table.column_names.index("timestamp"), "timestamp",
                             pa.array([0., .2], type=table.schema.field("timestamp").type))
    destination = fixture / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    info.update(total_episodes=1, total_frames=2, total_data_files=1)
    info["ngad_extensions"]["gripper_targets"]["episode_indices"] = [row["episode_index"]]
    info["ngad_extensions"]["gripper_targets"]["regression_fixture"] = "controlled_nonuniform_timestamps"
    write_json(fixture / "meta/info.json", info)
    config = json.loads((source / "dataset_configs/gripper_training.yaml").read_text())
    config["dataset"]["timeline"]["frame_ranges"] = [[0, 1]]
    config["dataset"]["gripper_target_source"] = "annotation"
    config["dataset"]["normalization_stats_path"] = str(fixture / "dataset_configs/normalization/pushcart.json")
    config["dataset"]["dataset_dirs"][0].update(path=str(fixture), mask_and_mapping_path=str(fixture / "dataset_configs/masks/mask.json"))
    config_path = fixture / "dataset_configs/nonuniform.yaml"
    write_json(config_path, config)
    loader = get_dataloader(data_dir=str(config_path), batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
    try:
        next(iter(loader))
    except ValueError as error:
        if "annotation source timestamps do not match the declared uniform grid" not in str(error):
            raise
        return dict(status="passed", source_timestamps=[0., .2], declared_source_fps=10,
                    rejected_by="actual dataloader.orion_pretrain.get_dataloader",
                    fixture=str(fixture), error=str(error))
    raise AssertionError("The actual annotation loader accepted nonuniform [0, .2] at 10 Hz")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/gpfs/jiuquyun/projects/youzi/orion-0")
    parser.add_argument("--output", required=True, help="Fresh workspace runs/ directory")
    parser.add_argument("--derived-root", action="append", default=[])
    args = parser.parse_args(argv)
    directory = create_run(args.output, [args.source, *args.derived_root])
    version = prepare_loader(args.source, directory)
    sys.path.insert(0, str(directory / "loader/src"))
    results = [verify_dataset(directory, root, index) for index, root in enumerate(args.derived_root)]
    timestamp_regression = verify_nonuniform_rejection(directory, args.derived_root[0]) if args.derived_root else None
    report = dict(status="passed" if results else "prepared_not_executed", source=version, datasets=results,
                  nonuniform_timestamp_regression=timestamp_regression,
                  deployment="isolated_workspace_copy_only", formal_publication=False)
    write_json(directory / "verification.json", report)
    print(json.dumps({"output": str(directory), "status": report["status"], "datasets": len(results),
                      "samples": sum(row["samples"] for row in results),
                      "changed_action_values": sum(row["changed_action_values"] for row in results)}))


if __name__ == "__main__":
    main()
