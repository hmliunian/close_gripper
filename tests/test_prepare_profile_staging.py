"""Staging preserves source metadata and explicitly declares shared video reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline
from scripts.prepare_profile_staging import FEATURES, prepare


class ProfileStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source = self.workspace / "sources/release/subset"
        self.bulk = self.workspace / "runs/bulk"
        self.numerical = self.bulk / "staging/release-gripcmd/subset"
        for root in (self.source, self.numerical):
            (root / "data/chunk-000").mkdir(parents=True)
        (self.source / "data/chunk-000/file-000.parquet").write_bytes(b"original observation parquet")
        (self.numerical / "data/chunk-000/file-000.parquet").write_bytes(b"reviewed bulk parquet with new fields")
        self.info = {"total_data_files": 1, "total_episodes": 2, "total_frames": 15,
                     "total_videos": 1, "splits": {"train": "0:2"}, "video_path": "videos/{video_key}/file.mp4",
                     "features": {"observation.state": {"dtype": "float32", "shape": [20]}}}
        self.write(self.source / "meta/info.json", self.info)
        self.write(self.source / "dataset_configs/masks/mask.json", {"dataset": "subset", "field_mask": {"observation.state": True}})
        self.write(self.source.parent / "dataset_configs/normalization/pushcart.json", {"normalized": True})
        (self.source / "videos").mkdir()
        (self.source / "videos/a.mp4").write_bytes(b"video fixture; never copied or decoded here")
        self.write(self.source / "meta/episodes/chunk-000/file-000.parquet", {"fixture": "metadata bytes retained"})
        self.write(self.bulk / "task_profiles.json", {"version": "fixture-version"})
        self.summary = {"status": "complete", "evaluation_only": True, "datasets": [
            {"source_root": str(self.source), "output_root": str(self.numerical),
             "selected_episodes": 1, "selected_frames": 9, "changed_episodes": 1, "changed_frames": 7}]}
        self.write(self.bulk / "summary.json", self.summary)
        self.output = self.workspace / "runs/delivery"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_source_preserved_full_indices_and_explicit_media_dependency(self):
        before = {str(p): p.read_bytes() for p in self.source.parent.rglob("*") if p.is_file()}
        report = prepare(self.bulk, self.output, video_mode="source-reference")
        target = self.output / "staging/release-gripcmd/subset"
        info = json.loads((target / "meta/info.json").read_text())
        for key, value in self.info.items():
            if key != "features":
                self.assertEqual(info[key], value)
        self.assertEqual(info["features"]["observation.state"], self.info["features"]["observation.state"])
        self.assertTrue(set(FEATURES).issubset(info["features"]))
        self.assertEqual((target / "data/chunk-000/file-000.parquet").read_bytes(),
                         (self.numerical / "data/chunk-000/file-000.parquet").read_bytes())
        self.assertFalse((target / "data/chunk-000/file-000.parquet").is_symlink())
        self.assertEqual((target / "dataset_configs/masks/mask.json").read_bytes(),
                         (self.source / "dataset_configs/masks/mask.json").read_bytes())
        self.assertEqual((target / "videos").resolve(), self.source / "videos")
        self.assertFalse(report["media_independent"])
        self.assertFalse(report["publication_ready"])
        config = json.loads((target / "dataset_configs/gripper_training.yaml").read_text())
        self.assertEqual(config["dataset"]["gripper_target_source"], "observation")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.source.parent.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            prepare(self.bulk, self.output, video_mode="source-reference")

    def test_default_video_copy_has_independent_bytes_and_digest(self):
        report = prepare(self.bulk, self.output)
        target = self.output / "staging/release-gripcmd/subset/videos/a.mp4"
        original = self.source / "videos/a.mp4"
        self.assertFalse(target.is_symlink())
        self.assertNotEqual(target.stat().st_ino, original.stat().st_ino)
        self.assertEqual(target.read_bytes(), original.read_bytes())
        self.assertTrue(report["media_independent"])
        self.assertEqual(report["video_files_copied"], 1)
        self.assertEqual(report["video_bytes_copied"], original.stat().st_size)

    def test_incomplete_bulk_and_escaped_numerical_root_are_rejected(self):
        self.summary["status"] = "running"
        self.write(self.bulk / "summary.json", self.summary)
        with self.assertRaises(ValueError):
            prepare(self.bulk, self.output)
        self.assertFalse(self.output.exists())
        self.summary["status"] = "complete"
        self.summary["datasets"][0]["output_root"] = str(self.source)
        self.write(self.bulk / "summary.json", self.summary)
        with self.assertRaises(ValueError):
            prepare(self.bulk, self.output)

    def test_missing_data_shard_cannot_be_advertised_as_complete(self):
        (self.numerical / "data/chunk-000/file-000.parquet").unlink()
        with self.assertRaises(ValueError):
            prepare(self.bulk, self.output)


if __name__ == "__main__":
    unittest.main()
