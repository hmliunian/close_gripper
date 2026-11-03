"""Video presentation preserves numerical artifacts and reuses source MP4s."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import cycles, pipeline
from scripts import render_video_review as render
from test_cycle_vision import PNG, fixture_episode


class RenderVideoReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source = self.workspace / "runs/source"
        self.episode_dir = self.source / "episode"
        (self.episode_dir / "frames_000").mkdir(parents=True)
        video = self.workspace / "source.mp4"
        video.write_bytes(b"fixture video; codec decoding belongs to browser validation")
        episode = fixture_episode()
        for eye in ("left", "right"):
            episode["cameras"][f"observation.images.cam_right_wrist_{eye}"] = dict(path=str(video), from_timestamp=.7, to_timestamp=3.)
        episode["cameras"]["observation.images.cam_left_wrist_left"] = dict(path="/missing-invalid-side.mp4", from_timestamp=0., to_timestamp=3.)
        candidates = cycles.detect_cycles(episode)
        result = cycles.generate_cycle_targets(episode, candidates, [])
        result.update(workflow="full_cycle", pipeline_issues=[], evidence_type="no_visual_review")
        image = self.episode_dir / "frames_000/one.png"
        image.write_bytes(PNG)
        frames = [dict(path=str(image), sha256=hashlib.sha256(PNG).hexdigest(), side="right", frame_index=0,
                       source_frame_index=0, camera="observation.images.cam_right_wrist_left", timestamp=0., video_timestamp=.7)]
        target = dict(episode_id=episode["episode_id"], episode_sha256=pipeline.episode_fingerprint(episode),
                      **{"annotation.gripper_target": result["target"], "annotation.gripper_target_valid": result["valid"]})
        for name, value in (("episode", episode), ("result", result), ("candidates", candidates), ("targets", target),
                            ("frames", frames), ("decisions", []), ("model_responses", [])):
            pipeline.dump(self.episode_dir / (name + ".json"), value)
        pipeline.dump(self.source / "summary.json", dict(status="complete", episodes=[dict(episode_id=episode["episode_id"], directory="episode")]))

    def output(self):
        return self.workspace / "runs/video_review"

    def test_preparation_and_final_html_preserve_json_images_and_video_offsets(self):
        before = {str(path): render.sha256(path) for path in self.source.rglob("*") if path.is_file()}
        with patch.object(cycles, "generate_cycle_targets", side_effect=AssertionError("numerical generation")), \
             patch("close_gripper.vision.urlopen", side_effect=AssertionError("network")):
            summary = render.prepare_video_review(self.source, self.output(), prepare_only=True)
            self.assertEqual(summary["status"], "prepared")
            self.assertFalse((self.output() / "review.html").exists())
            complete = render.finalize_video_review(self.output())
        self.assertEqual(complete["status"], "complete")
        self.assertTrue(complete["presentation_only"])
        self.assertEqual((complete["video_references"], complete["unique_source_videos"], complete["video_bytes_copied"]), (2, 1, 0))
        self.assertEqual((complete["model_requests"], complete["algorithm_executions"]), (0, 0))
        self.assertFalse((self.output() / "summary.json").exists())
        self.assertTrue((self.output() / "source_summary.json").is_file())
        new = self.output() / "episode"
        for path in self.episode_dir.glob("*.json"):
            self.assertEqual(path.read_bytes(), (new / path.name).read_bytes())
        self.assertEqual((new / "frames_000/one.png").read_bytes(), PNG)
        self.assertNotEqual((new / "frames_000/one.png").stat().st_ino, (self.episode_dir / "frames_000/one.png").stat().st_ino)
        display = render.read_json(self.output() / "display_manifest.json")["episodes"][0]
        self.assertEqual({v["side"] for v in display["videos"]}, {"right"})
        self.assertEqual([v["from_timestamp"] for v in display["videos"]], [.7, .7])
        for path in (new / "review.html", self.output() / "review.html"):
            self.assertIn(display["videos"][0]["url"], path.read_text())
        self.assertEqual(before, {str(path): render.sha256(path) for path in self.source.rglob("*") if path.is_file()})

    def test_prepared_targets_and_source_images_cannot_change_silently(self):
        render.prepare_video_review(self.source, self.output(), prepare_only=True)
        (self.output() / "episode/targets.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            render.finalize_video_review(self.output())

    def test_fingerprint_or_invalid_video_interval_is_rejected(self):
        target = render.read_json(self.episode_dir / "targets.json")
        target["episode_sha256"] = "stale"
        pipeline.dump(self.episode_dir / "targets.json", target)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            render.prepare_video_review(self.source, self.output())
        self.assertFalse(self.output().exists())
        episode = render.read_json(self.episode_dir / "episode.json")
        episode["cameras"]["observation.images.cam_right_wrist_left"]["from_timestamp"] = -1
        target["episode_sha256"] = pipeline.episode_fingerprint(episode)
        pipeline.dump(self.episode_dir / "episode.json", episode)
        pipeline.dump(self.episode_dir / "targets.json", target)
        with self.assertRaisesRegex(ValueError, "video interval"):
            render.prepare_video_review(self.source, self.output())

    def test_new_run_and_existing_preparation_are_required(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            render.prepare_video_review(self.source, self.source / "nested")
        render.prepare_video_review(self.source, self.output(), prepare_only=True)
        with self.assertRaises(FileExistsError):
            render.prepare_video_review(self.source, self.output(), prepare_only=True)
        render.finalize_video_review(self.output())
        with self.assertRaisesRegex(ValueError, "newly prepared"):
            render.finalize_video_review(self.output())


if __name__ == "__main__":
    unittest.main()
