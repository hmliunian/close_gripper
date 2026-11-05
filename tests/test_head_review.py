"""Main-camera evidence identity, dense coverage and evaluation-only delivery."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline, profile_run
from close_gripper.head_review import HEAD_CAMERAS, REVIEW_SCHEMA, validate_head_review
from test_cycle_vision import PNG
from test_profile_run import catalog, episode as numeric_episode


def head_fixture(*, arms=("right",), image_dir=None):
    episode = numeric_episode()
    episode.update(vision_camera_group="head", valid_sides=list(arms), cameras={
        camera: dict(path=f"/synthetic/head_{eye}.mp4", from_timestamp=offset, to_timestamp=6.)
        for camera, eye, offset in zip(HEAD_CAMERAS, ("left", "right"), (1.7, 3.2))})
    size = len(episode["timestamps"])
    review = dict(schema=REVIEW_SCHEMA, episode_id=episode["episode_id"],
                  episode_sha256=pipeline.episode_fingerprint(episode), camera_group="head",
                  review_source="model", review_method="codex_native_vision", confirmed=False,
                  sides={side: dict(reviewed_frame_indices=list(range(size)),
                                    segments=[dict(start=0, end=size-1, state="held",
                                                   note="合成契约夹具，不作为视觉真值")])
                         for side in arms})
    frames = []
    for camera in HEAD_CAMERAS:
        eye = camera.rsplit("_", 1)[-1]
        for i, timestamp in enumerate(episode["timestamps"]):
            path = Path(image_dir or "/synthetic/images") / f"{eye}_{i:03d}.png"
            if image_dir is not None:
                path.write_bytes(PNG)
            frames.append(dict(camera=camera, side="head", frame_index=i, timestamp=timestamp,
                               video_timestamp=episode["cameras"][camera]["from_timestamp"] + timestamp,
                               sha256=hashlib.sha256(PNG).hexdigest(), path=str(path)))
    return episode, review, frames


class HeadReviewContracts(unittest.TestCase):
    def test_each_eye_keeps_its_offset_and_both_arms_use_shared_head_evidence(self):
        episode, review, frames = head_fixture(arms=("left", "right"))
        before = copy.deepcopy((episode, review, frames))
        self.assertEqual(validate_head_review(episode, review, frames), review)
        self.assertEqual((episode, review, frames), before)
        self.assertNotEqual(frames[0]["video_timestamp"], frames[9]["video_timestamp"])

    def test_review_cannot_be_reused_after_input_or_episode_identity_changes(self):
        for change in ("width", "identity", "offset", "digest"):
            episode, review, frames = head_fixture()
            if change == "width":
                episode["state"][2][19] += .001
            elif change == "identity":
                review["episode_id"] = "another/published/crop"
            elif change == "offset":
                episode["cameras"][HEAD_CAMERAS[0]]["from_timestamp"] += .1
            else:
                review["episode_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "identity or input hash"):
                validate_head_review(episode, review, frames)

    def test_wrist_or_wrong_eye_time_cannot_supply_main_camera_evidence(self):
        for change in ("unmarked", "mixed_cameras", "wrist_frame", "unavailable_eye",
                       "episode_timestamp", "other_eye_offset", "duplicate_frame"):
            episode, review, frames = head_fixture()
            if change == "unmarked":
                episode.pop("vision_camera_group")
            elif change == "mixed_cameras":
                episode["cameras"]["observation.images.cam_right_wrist_left"] = copy.deepcopy(
                    episode["cameras"][HEAD_CAMERAS[0]])
            elif change == "wrist_frame":
                frames[0]["camera"] = "observation.images.cam_right_wrist_left"
            elif change == "unavailable_eye":
                episode["cameras"].pop(HEAD_CAMERAS[1])
            elif change == "episode_timestamp":
                frames[0]["timestamp"] = .1
            elif change == "other_eye_offset":
                frames[9]["video_timestamp"] = frames[0]["video_timestamp"]
            else:
                frames.append(copy.deepcopy(frames[0]))
            review["episode_sha256"] = pipeline.episode_fingerprint(episode)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_head_review(episode, review, frames)

    def test_dense_coverage_requires_every_frame_and_every_effective_arm(self):
        for change in ("missing_arm", "head_as_arm", "missing_review_index", "duplicate_review_index",
                       "missing_images", "gap", "overlap", "truncated_tail"):
            episode, review, frames = head_fixture(arms=("left", "right"))
            right = review["sides"]["right"]
            if change == "missing_arm":
                review["sides"].pop("left")
            elif change == "head_as_arm":
                review["sides"]["head"] = review["sides"].pop("left")
            elif change == "missing_review_index":
                right["reviewed_frame_indices"].pop(3)
            elif change == "duplicate_review_index":
                right["reviewed_frame_indices"].append(3)
            elif change == "missing_images":
                frames = [f for f in frames if f["frame_index"] != 3]
            elif change in ("gap", "overlap"):
                right["segments"] = [dict(start=0, end=3, state="free", note="合成空夹"),
                                     dict(start=5 if change == "gap" else 3, end=8,
                                          state="held", note="合成夹持")]
            else:
                right["segments"][0]["end"] = 7
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_head_review(episode, review, frames)

    def test_no_human_claim_unknown_semantics_or_malformed_image_digest(self):
        for change in ("human", "confirmed", "method", "state", "empty_note", "digest"):
            episode, review, frames = head_fixture()
            if change == "human":
                review["review_source"] = "human"
            elif change == "confirmed":
                review["confirmed"] = True
            elif change == "method":
                review["review_method"] = "numeric_curve_inference"
            elif change == "digest":
                frames[0]["sha256"] = "unverified"
            else:
                segment = review["sides"]["right"]["segments"][0]
                segment["state" if change == "state" else "note"] = "possibly_held" if change == "state" else ""
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_head_review(episode, review, frames)


class HeadReviewRunContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.images = self.workspace / "evidence"
        self.images.mkdir()
        patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_real_rule_generation_preserves_review_images_and_never_promotes_training_labels(self):
        episode, review, frames = head_fixture(image_dir=self.images)
        before = copy.deepcopy((episode, review, frames))
        original_images = {p.name: p.read_bytes() for p in self.images.iterdir()}
        output = self.workspace / "runs/head"
        summary = profile_run.run_profile_clean([episode], output, catalog(), workers=1,
            visual_reviews={episode["episode_id"]: review}, evidence_frames={episode["episode_id"]: frames})
        child = output / summary["episodes"][0]["directory"]
        saved = json.loads((child / "result.json").read_text())
        targets = json.loads((child / "targets.json").read_text())
        copied = json.loads((child / "frames.json").read_text())
        self.assertEqual(saved["visual_review"], review)
        self.assertEqual(targets["annotation.gripper_target"], [[.7, .2]] * 9)
        self.assertEqual(targets["annotation.gripper_target_valid"], [[False, False]] * 9)
        self.assertEqual(targets["evaluation.gripper_target"], [[.7, 0.]] * 9)
        self.assertTrue(summary["evaluation_only"])
        self.assertFalse(summary["grasp_semantics_verified"])
        self.assertEqual(summary["actual_requests"], 0)
        self.assertEqual(len(copied), 18)
        for source, copied_frame in zip(frames, copied):
            self.assertTrue(Path(copied_frame["path"]).is_relative_to(child))
            self.assertEqual(Path(copied_frame["path"]).read_bytes(), Path(source["path"]).read_bytes())
            self.assertEqual(copied_frame["sha256"], source["sha256"])
        self.assertEqual((episode, review, frames), before)
        self.assertEqual({p.name: p.read_bytes() for p in self.images.iterdir()}, original_images)

    def test_run_rejects_partial_review_cohorts_before_writing_output(self):
        episode, review, frames = head_fixture(image_dir=self.images)
        for reviews, images in ((None, {episode["episode_id"]: frames}),
                                 ({episode["episode_id"]: review}, None),
                                 ({}, {episode["episode_id"]: frames}),
                                 ({episode["episode_id"]: review}, {"different_episode": frames})):
            output = self.workspace / "runs/incomplete"
            with self.subTest(reviews=reviews is None, images=images is None), self.assertRaisesRegex(
                    ValueError, "whole input cohort"):
                profile_run.run_profile_clean([episode], output, catalog(), workers=1,
                                              visual_reviews=reviews, evidence_frames=images)
            self.assertFalse(output.exists())

    def test_missing_or_tampered_image_cannot_produce_a_completed_run(self):
        episode, review, frames = head_fixture(image_dir=self.images)
        for failure in ("missing", "wrong_digest"):
            bad = copy.deepcopy(frames)
            if failure == "missing":
                bad[0]["path"] = str(self.images / "missing.png")
            else:
                bad[0]["sha256"] = "0" * 64
            output = self.workspace / f"runs/{failure}"
            with self.subTest(failure=failure), self.assertRaises((ValueError, FileNotFoundError)):
                profile_run.run_profile_clean([episode], output, catalog(), workers=1,
                    visual_reviews={episode["episode_id"]: review}, evidence_frames={episode["episode_id"]: bad})
            self.assertFalse((output / "summary.json").exists())
        self.assertEqual(len(list(self.images.iterdir())), len(frames))


if __name__ == "__main__":
    unittest.main()
