"""Offline complete-episode orchestration, loader sidecar, and request accounting."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from close_gripper import clean, pipeline
from close_gripper.cli import main
from close_gripper.cycles import detect_cycles
from close_gripper.dataset import DatasetError, extract_selected_frames
from test_cycle_vision import PNG, fixture_decision, fixture_episode, response


class CleanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manifest_dir = self.root / "manifest"
        (self.manifest_dir / "episodes").mkdir(parents=True)
        self.manifest = self.manifest_dir / "manifest.json"
        self.ep = fixture_episode()

    def save_manifest(self, episode=None):
        episode = episode or self.ep
        pipeline.dump(self.manifest_dir / "episodes/one.json", episode)
        row = dict(episode_cache="episodes/one.json", episode_id=episode["episode_id"],
                   episode_sha256=pipeline.episode_fingerprint(episode),
                   candidate_id="legacy-selected-event-must-not-limit-clean", side="left")
        pipeline.dump(self.manifest, {"episodes": [row]})
        return row

    def selected_frames(self, ep, side, indices, output, config):
        output = Path(output)
        output.mkdir()
        path = output / "fixture.png"
        path.write_bytes(PNG)
        camera = next(k for k in ep["cameras"] if f"{side}_wrist" in k)
        return [dict(side=side, camera=camera, frame_index=i, timestamp=ep["timestamps"][i],
                     video_timestamp=ep["timestamps"][i] + ep["cameras"][camera]["from_timestamp"],
                     path=str(path), sha256=hashlib.sha256(PNG).hexdigest()) for i in indices]

    def run_offline(self, name="offline", reviews=None):
        with patch.object(clean, "write_plot", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            return clean.run_clean([self.ep], self.workspace / "runs" / name, reviews=reviews)

    def test_no_model_preserves_targets_and_marks_review_coverage_incomplete(self):
        snapshot = copy.deepcopy(self.ep)
        with patch.object(clean, "BoundedCycleClient", side_effect=AssertionError("client constructed")), patch(
                "close_gripper.dataset.extract_selected_frames", side_effect=AssertionError("video decoded")):
            summary = self.run_offline()
        directory = self.workspace / "runs/offline"
        target = json.loads(next(directory.glob("*/targets.json")).read_text())
        self.assertEqual(target["annotation.gripper_target"], [[s[9], s[19]] for s in self.ep["state"]])
        self.assertFalse(any(any(row) for row in target["annotation.gripper_target_valid"]))
        self.assertEqual(summary["actual_requests"], 0)
        self.assertFalse(summary["episodes"][0]["coverage"]["right"]["complete_review"])
        self.assertEqual(self.ep, snapshot)

    def test_successful_model_release_writes_command_sidecar_without_becoming_human_truth(self):
        snapshot = copy.deepcopy(self.ep)
        candidate = detect_cycles(self.ep)[0]
        with patch.object(clean, "write_plot", return_value=None), patch(
                "close_gripper.dataset.extract_selected_frames", side_effect=self.selected_frames), patch(
                "close_gripper.cycle_vision.CycleQwenClient._credential", return_value="sk-clean-fixture"), patch(
                "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(response(fixture_decision(candidate))).encode())) as network, \
                contextlib.redirect_stdout(io.StringIO()):
            summary = clean.run_clean([self.ep], self.workspace / "runs/model", {"max_requests": 1}, call_model=True)
        output = self.workspace / "runs/model"
        episode_dir = output / summary["episodes"][0]["directory"]
        target = json.loads((episode_dir / "targets.json").read_text())
        self.assertEqual(target["annotation.gripper_target"][5][1], 0)
        self.assertTrue(target["annotation.gripper_target_valid"][5][1])
        self.assertEqual(target["annotation.gripper_target_source"][5][1], "model")
        self.assertEqual(target["annotation.gripper_phase"][5][1], "hold")
        self.assertEqual(target["annotation.gripper_phase"][11][1], "release")
        self.assertEqual(target["timestamps"], self.ep["timestamps"])
        self.assertEqual(target["frame_indices"], self.ep["frame_indices"])
        self.assertEqual([r[0] for r in target["annotation.gripper_target"]], [r[9] for r in self.ep["state"]])
        self.assertEqual(json.loads((episode_dir / "episode.json").read_text()), snapshot)
        self.assertEqual(self.ep, snapshot)
        self.assertFalse(summary["human_reference"])
        self.assertEqual(summary["actual_requests"], 1)
        self.assertEqual(summary["reported_token_usage"]["total_tokens"], 18)
        self.assertEqual(network.call_count, 1)
        self.assertTrue((output / "review.html").is_file())
        persisted = "".join(p.read_text() for p in output.rglob("*.json"))
        self.assertNotIn("sk-clean-fixture", persisted)

    def test_supplied_model_review_remains_model_and_requires_matching_snapshot(self):
        candidate = detect_cycles(self.ep)[0]
        d = dict(fixture_decision(candidate), review_source="model", confirmed=False)
        envelope = {"episode_sha256": pipeline.episode_fingerprint(self.ep), "decisions": [d]}
        summary = self.run_offline("review", {self.ep["episode_id"]: envelope})
        output = self.workspace / "runs/review"
        data = json.loads(next(output.glob("*/targets.json")).read_text())
        self.assertEqual(data["annotation.gripper_target_source"][5][1], "model")
        self.assertEqual(summary["actual_requests"], 0)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_offline("stale", {self.ep["episode_id"]: dict(envelope, episode_sha256="stale")})

    def test_manifest_ignores_legacy_selected_event_but_binds_entire_episode(self):
        row = self.save_manifest()
        self.assertEqual(clean.load_clean_manifest(self.manifest), [self.ep])
        changed = copy.deepcopy(self.ep)
        changed["state"][0][0] = .456
        pipeline.dump(self.manifest_dir / row["episode_cache"], changed)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            clean.load_clean_manifest(self.manifest)

    def test_manifest_rejects_parent_escape_absolute_cache_and_symlink(self):
        row = self.save_manifest()
        outside = self.root / "outside.json"
        pipeline.dump(outside, self.ep)
        inner = self.manifest_dir / "episodes/one.json"
        (self.manifest_dir / "episodes/outside-link.json").symlink_to(outside)
        (self.manifest_dir / "episodes/inside-link.json").symlink_to(inner)
        for cache in ("../outside.json", str(outside), str(inner), "episodes/outside-link.json", "episodes/inside-link.json"):
            pipeline.dump(self.manifest, [dict(row, episode_cache=cache)])
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                clean.load_clean_manifest(self.manifest)

    def test_manifest_identity_mismatch_is_rejected(self):
        row = self.save_manifest()
        pipeline.dump(self.manifest, [dict(row, episode_id="another-episode")])
        with self.assertRaisesRegex(ValueError, "identity"):
            clean.load_clean_manifest(self.manifest)

    def test_duplicate_or_empty_episodes_and_invalid_workers_fail_before_output(self):
        for episodes in ([], [self.ep, self.ep]):
            with self.assertRaises(ValueError):
                clean.run_clean(episodes, self.workspace / "runs/invalid")
        for workers in (0, 5, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                clean.run_clean([self.ep], self.workspace / "runs/invalid", workers=workers)
        self.assertFalse((self.workspace / "runs/invalid").exists())

    def test_clean_config_rejects_embedded_authentication_objects(self):
        for override in ({"api_key": "sk-fake"}, {"transport": {"Authorization": "Bearer sk-fake"}},
                         {"headers": {"authorization": "Bearer sk-fake"}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                clean.clean_config(override)
        self.assertEqual(clean.clean_config({"api_key_file": "/fixture/key"})["api_key_file"], "/fixture/key")

    def test_cli_clean_offline_processes_full_frozen_episode(self):
        self.save_manifest()
        output = self.workspace / "runs/cli"
        with patch.object(clean, "write_plot", return_value=None), patch.object(
                clean, "BoundedCycleClient", side_effect=AssertionError("implicit model")), \
                contextlib.redirect_stdout(io.StringIO()):
            status = main(["clean", "--manifest", str(self.manifest), "--output", str(output)])
        self.assertEqual(status, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["workflow"], "full_cycle")
        self.assertEqual(summary["evaluation_scope"], "all_detected_cycles_and_remaining_observation_windows")

    def test_selected_frames_use_only_requested_indices_with_camera_offset(self):
        indices = [0, 3, 5, 10, 15]
        with patch("close_gripper.dataset._extract_requests", side_effect=lambda ep, side, requests, out, cfg: requests):
            frames = extract_selected_frames(self.ep, "right", indices, self.root / "frames", {})
        self.assertEqual([f["frame_index"] for f in frames], indices)
        self.assertTrue(all(f["side"] == "right" for f in frames))
        self.assertEqual([f["source_frame_index"] for f in frames], indices)
        self.assertEqual([f["video_timestamp"] for f in frames], [.7 + i / 10 for i in indices])
        for bad in ([3, 0], [0, 0], [-1, 0], [0, True], [], [0, len(self.ep["state"])]):
            with self.subTest(bad=bad), self.assertRaises(DatasetError):
                extract_selected_frames(self.ep, "right", bad, self.root / "frames", {})


class CleanBudgetTests(unittest.TestCase):
    def test_one_global_budget_bounds_parallel_episode_requests(self):
        config = clean.clean_config({"max_requests": 3})
        client = clean.BoundedCycleClient(config)
        barrier = threading.Barrier(3)
        calls, configurations, lock = [], [], threading.Lock()

        class FakeClient:
            authentication_blocked = False

            def __init__(self, cfg):
                configurations.append(cfg)

            def annotate(self, episode, candidate, frames):
                with lock:
                    calls.append(candidate["candidate_id"])
                barrier.wait(timeout=5)
                return dict(status="ok", attempts=1, usage={}, decision={})

        with patch.object(clean, "CycleQwenClient", FakeClient), ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda i: client.annotate({}, {"candidate_id": str(i), "side": "right"}, []), range(12)))
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(r["attempts"] for r in results), 3)
        self.assertEqual(sum(r.get("error") == "request_budget_exhausted" for r in results), 9)
        self.assertTrue(all(cfg["max_requests"] == 1 and cfg["max_retries"] == 0 for cfg in configurations))

    def test_auth_denial_stops_requests_not_already_in_flight(self):
        client = clean.BoundedCycleClient(clean.clean_config({"max_requests": 20}))
        barrier = threading.Barrier(4)
        calls, lock = [], threading.Lock()

        class FakeClient:
            def __init__(self, cfg):
                self.authentication_blocked = False

            def annotate(self, episode, candidate, frames):
                with lock:
                    calls.append(candidate["candidate_id"])
                barrier.wait(timeout=5)
                self.authentication_blocked = True
                return dict(status="error", attempts=1, error="http_401", usage={}, decision={})

        with patch.object(clean, "CycleQwenClient", FakeClient), ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: client.annotate({}, {"candidate_id": str(i), "side": "right"}, []), range(12)))
        self.assertEqual(len(calls), 4)
        self.assertTrue(client.authentication_blocked)
        self.assertEqual(sum(r.get("error") == "authentication_blocked" for r in results), 8)
        self.assertEqual(sum(r["attempts"] for r in results), 4)


if __name__ == "__main__":
    unittest.main()
