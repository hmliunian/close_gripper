"""Saved-evidence replay is offline, identity-bound and separately accounted."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import clean, cycles, pipeline
from close_gripper.cycle_vision import parse_cycle_decision
from scripts import replay_clean_run as replay
from scripts.verify_clean_run import verify_run
from test_cycle_vision import PNG, fixture_decision, fixture_episode, response


class ReplayCleanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.inputs = self.workspace / "inputs"
        (self.inputs / "episodes").mkdir(parents=True)
        first, second = fixture_episode(), fixture_episode()
        second["episode_id"] = "failure-fixture"
        self.episodes = [first, second]
        rows = []
        for index, episode in enumerate(self.episodes):
            relative = f"episodes/{index}.json"
            pipeline.dump(self.inputs / relative, episode)
            rows.append(dict(episode_id=episode["episode_id"], episode_cache=relative,
                             episode_sha256=pipeline.episode_fingerprint(episode)))
        self.manifest = self.inputs / "manifest.json"
        pipeline.dump(self.manifest, rows)
        self.source = self.workspace / "runs/source"
        with patch.object(clean, "write_plot", return_value=None), patch.object(clean, "BoundedCycleClient") as client, \
             patch("close_gripper.dataset.extract_selected_frames", side_effect=self.frames), contextlib.redirect_stdout(io.StringIO()):
            client.return_value.annotate.side_effect = self.annotation
            self.source_summary = clean.run_clean(self.episodes, self.source,
                {"config_version": "full-cycle-v2", "max_requests": 2}, call_model=True)

    def frames(self, episode, side, indices, output, config):
        output = Path(output)
        output.mkdir()
        camera = next(iter(episode["cameras"]))
        result = []
        for index in indices:
            path = output / f"frame_{index:06d}.png"
            path.write_bytes(PNG)
            result.append(dict(path=str(path), sha256=hashlib.sha256(PNG).hexdigest(), side=side,
                camera=camera, frame_index=index, source_frame_index=episode["frame_indices"][index],
                timestamp=episode["timestamps"][index], video_timestamp=.7 + episode["timestamps"][index],
                video_path=episode["cameras"][camera]["path"]))
        return result

    def annotation(self, episode, candidate, frames):
        if episode["episode_id"] == "failure-fixture":
            return dict(status="error", attempts=1, error="synthetic_service_failure", usage={},
                        raw_response={"error": "synthetic_service_failure", "usage": {}}, request_hash="a" * 64)
        raw = response(fixture_decision(candidate))
        return dict(status="ok", attempts=1, usage=raw["usage"], raw_response=raw,
                    request_hash="b" * 64, decision=parse_cycle_decision(raw, episode, candidate, frames))

    def output(self, name="replay"):
        return self.workspace / "runs" / name

    def replay(self, name="replay"):
        with patch.object(replay, "write_plot", return_value=None), \
             patch.object(clean, "BoundedCycleClient", side_effect=AssertionError("model constructed")), \
             patch("close_gripper.dataset.extract_selected_frames", side_effect=AssertionError("video decoded")), \
             patch("close_gripper.vision.urlopen", side_effect=AssertionError("network called")):
            return replay.replay_clean_run(self.source, self.manifest, self.output(name))

    def test_replay_preserves_evidence_and_failure_reasons_with_zero_new_requests(self):
        before = {str(path): replay.digest_file(path) for path in self.source.rglob("*") if path.is_file()}
        with patch.object(cycles, "generate_cycle_targets", wraps=cycles.generate_cycle_targets) as numeric:
            summary = self.replay()
        self.assertEqual(numeric.call_count, 2)
        self.assertEqual(summary["actual_requests"], 0)
        self.assertEqual(summary["new_model_requests"], 0)
        self.assertEqual(summary["reused_actual_requests"], 2)
        self.assertEqual(summary["reported_token_usage"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(summary["reused_reported_token_usage"]["total_tokens"], 18)
        self.assertTrue(summary["repeated_evaluation_same_inputs"])
        self.assertEqual(summary["evidence_type"], "reused_real_vlm")
        self.assertEqual(summary["config_changes"], {"config_version": {"source": "full-cycle-v2", "replay": cycles.cycle_config()["config_version"]}})
        self.assertFalse(summary["implementation_changed_during_run"])
        self.assertEqual(before, {str(path): replay.digest_file(path) for path in self.source.rglob("*") if path.is_file()})
        for row in summary["episodes"]:
            old, new = self.source / row["directory"], self.output() / row["directory"]
            for name in ("episode.json", "decisions.json", "candidates.json", "review_template.json"):
                self.assertEqual((old / name).read_bytes(), (new / name).read_bytes())
            for name in ("model_responses.json", "frames.json", "result.json", "targets.json"):
                self.assertEqual((old / name).read_bytes(), (new / ("source_" + name)).read_bytes())
            self.assertEqual(json.loads((new / "model_responses.json").read_text()), [])
            source_result, new_result = [json.loads((base / "result.json").read_text()) for base in (old, new)]
            self.assertEqual(new_result["pipeline_issues"], source_result["pipeline_issues"])
            self.assertEqual(new_result["evidence_type"], "reused_real_vlm")
            target = json.loads((new / "targets.json").read_text())
            self.assertEqual(target["annotation.gripper_target"], new_result["target"])
            original_frames, frames = [json.loads((base / "frames.json").read_text()) for base in (old, new)]
            for original, frame in zip(original_frames, frames):
                self.assertEqual({k: v for k, v in original.items() if k != "path"}, {k: v for k, v in frame.items() if k != "path"})
                self.assertTrue(Path(frame["path"]).is_relative_to(new))
                self.assertEqual(replay.digest_file(frame["path"]), frame["sha256"])
                self.assertNotEqual(Path(frame["path"]).stat().st_ino, Path(original["path"]).stat().st_ino)
            self.assertTrue((new / "review.html").is_file())
        audit = verify_run(self.output(), self.manifest)
        self.assertEqual(audit["status"], "passed", audit["errors"][:5])
        self.assertEqual(audit["actual_requests"], 0)
        self.assertEqual(audit["reused_request_attempts"], 2)
        self.assertEqual(audit["reused_reported_token_usage"]["total_tokens"], 18)
        self.assertEqual(clean.load_clean_manifest(self.output() / "source_inputs/manifest.json"), self.episodes)

    def test_incomplete_source_is_rejected_before_creating_output(self):
        pipeline.dump(self.source / "summary.json", dict(self.source_summary, status="running"))
        with self.assertRaisesRegex(ValueError, "completed"):
            self.replay()
        self.assertFalse(self.output().exists())

    def test_changed_frozen_state_is_rejected(self):
        changed = copy.deepcopy(self.episodes[0])
        changed["state"][0][19] = .2
        pipeline.dump(self.inputs / "episodes/0.json", changed)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.replay()
        self.assertFalse(self.output().exists())

    def test_changed_candidates_require_new_visual_review(self):
        detect = cycles.detect_cycles
        def changed(*args, **kwargs):
            result = detect(*args, **kwargs)
            result[0]["start"] += 1
            return result
        with patch.object(cycles, "detect_cycles", side_effect=changed), self.assertRaisesRegex(ValueError, "candidates changed"):
            self.replay()
        self.assertFalse(self.output().exists())

    def test_tampered_config_or_image_fails_source_audit(self):
        config = json.loads((self.source / "config.json").read_text())
        pipeline.dump(self.source / "config.json", dict(config, model="changed-model"))
        with self.assertRaisesRegex(ValueError, "source run audit failed"):
            self.replay()
        pipeline.dump(self.source / "config.json", config)
        image = next(self.source.glob("*/frames_*/*.png"))
        image.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "source run audit failed"):
            self.replay()
        self.assertFalse(self.output().exists())

    def test_missing_implementation_snapshot_is_not_inferred(self):
        (self.source / "implementation.json").rename(self.source / "old_implementation.json")
        with self.assertRaisesRegex(ValueError, "source implementation snapshot"):
            self.replay()
        self.assertFalse(self.output().exists())

    def test_existing_or_overlapping_output_is_rejected(self):
        self.replay()
        with self.assertRaises(FileExistsError):
            self.replay()
        with patch.object(replay, "write_plot", return_value=None), self.assertRaisesRegex(ValueError, "overlap"):
            replay.replay_clean_run(self.source, self.manifest, self.source / "replay")


if __name__ == "__main__":
    unittest.main()
