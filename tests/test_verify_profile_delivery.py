"""A second reader catches missing media, corrupt copies and training promotion."""
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import verify_profile_delivery as verifier
from scripts.prepare_profile_staging import prepare


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow unavailable")
class DeliveryAuditTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        spec = importlib.util.spec_from_file_location("delivery_source_fixture", Path(__file__).with_name("test_prepare_profile_staging.py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.fixture = module.ProfileStagingTests(methodName="runTest")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.patch = patch.object(verifier, "WORKSPACE", f.workspace)
        self.patch.start(); self.addCleanup(self.patch.stop)
        f.info["features"]["cam"] = dict(dtype="video", shape=[256, 256, 3])
        f.write(f.source / "meta/info.json", f.info)
        video = f.source / "videos/cam/file.mp4"
        video.parent.mkdir()
        (f.source / "videos/a.mp4").rename(video)
        rows = [dict(episode_index=0, length=7, **{"videos/cam/chunk_index": 0, "videos/cam/file_index": 0,
                                                "videos/cam/from_timestamp": 0., "videos/cam/to_timestamp": .7}),
                dict(episode_index=1, length=8, **{"videos/cam/chunk_index": 0, "videos/cam/file_index": 0,
                                                "videos/cam/from_timestamp": .7, "videos/cam/to_timestamp": 1.5})]
        pq.write_table(pa.Table.from_pylist(rows), f.source / "meta/episodes/chunk-000/file-000.parquet")
        self.target = f.output / "staging/release-gripcmd/subset"

    def run_prepare(self, mode="copy"):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.fixture.bulk, self.fixture.output, video_mode=mode)

    def verify(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return verifier.audit_delivery(self.fixture.output, self.fixture.workspace / "runs/audit", workers=2)

    def test_complete_metadata_and_independent_video_copy_pass(self):
        self.run_prepare()
        result = self.verify()
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(result["copied_videos_verified"], 1)

    def test_explicit_source_reference_is_not_reported_as_independent_copy(self):
        self.run_prepare("source-reference")
        result = self.verify()
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(result["video_mode"], "source-reference")
        self.assertEqual(result["copied_videos_verified"], 0)

    def test_changed_video_bytes_are_rejected(self):
        self.run_prepare()
        (self.target / "videos/cam/file.mp4").write_bytes(b"corrupt video bytes")
        result = self.verify()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("copy_source_and_target_sha256", {r["check"] for r in result["failures"]})

    def test_missing_journal_entry_cannot_hide_a_video(self):
        self.run_prepare()
        with gzip.open(self.fixture.output / "video_files.jsonl.gz", "wt"):
            pass
        result = self.verify()
        self.assertIn("every_video_verified_once", {r["check"] for r in result["failures"]})

    def test_removed_video_cannot_pass_inventory_check(self):
        self.run_prepare()
        (self.target / "videos/cam/file.mp4").unlink()
        result = self.verify()
        self.assertIn("complete_physical_video_inventory", {r["check"] for r in result["failures"]})

    def test_evaluation_cannot_become_default_training_source(self):
        self.run_prepare()
        path = self.target / "dataset_configs/gripper_training.yaml"
        config = json.loads(path.read_text()); config["dataset"]["gripper_target_source"] = "evaluation"
        path.write_text(json.dumps(config))
        result = self.verify()
        self.assertIn("training_default_observation_isolation", {r["check"] for r in result["failures"]})


if __name__ == "__main__":
    unittest.main()
