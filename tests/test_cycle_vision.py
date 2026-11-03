"""Full-cycle prompt/transport contracts using only synthetic bytes and responses."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from close_gripper.cycle_vision import (
    CycleQwenClient, PROMPT_VERSION, SYSTEM_PROMPT, build_cycle_request,
    cycle_frame_indices, parse_cycle_decision,
)
from close_gripper.cycles import detect_cycles
from close_gripper.vision import VisionValidationError, request_fingerprint


VALUES = [.9, .9, .9, .7, .5, .3, .3, .3, .3, .3, .3, .5, .7, .9, .9, .9]
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9hQAAAAASUVORK5CYII=")


def fixture_episode():
    state = [[.123] * 20 for _ in VALUES]
    for row, value in zip(state, VALUES):
        row[19] = value
    camera = "observation.images.cam_right_wrist_left"
    return dict(episode_id="clean-vision-fixture", timestamps=[i / 10 for i in range(len(VALUES))],
                state=state, valid_sides=["right"], frame_indices=list(range(len(VALUES))),
                cameras={camera: {"valid": True, "from_timestamp": .7, "to_timestamp": 3., "path": "/fixture.mp4"}},
                identity={"task": "synthetic grasp fixture", "source": "test"})


def fixture_decision(candidate):
    return dict(candidate_id=candidate["candidate_id"], side=candidate["side"],
                category="grasp_release", hold_start=5, hold_end=10, release_end=13,
                pregrasp_start=0, pregrasp_clear=True, continuous_grasp=True, slip=False,
                evidence_frames=[0, 5, 10, 13, 15], reason="Synthetic visual fixture")


def response(decision):
    return dict(model="offline-cycle-model", usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                choices=[{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}])


class CycleVisionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.episode = fixture_episode()
        self.candidate = detect_cycles(self.episode)[0]
        self.path = self.root / "frame.png"
        self.path.write_bytes(PNG)
        self.frames = self.make_frames(range(len(VALUES)))
        self.decision = fixture_decision(self.candidate)

    def make_frames(self, indices):
        camera = next(iter(self.episode["cameras"]))
        return [dict(side="right", camera=camera, frame_index=i,
                     timestamp=self.episode["timestamps"][i], video_timestamp=.7 + self.episode["timestamps"][i],
                     path=str(self.path), sha256=hashlib.sha256(PNG).hexdigest()) for i in indices]

    def test_payload_uses_normalized_opening_and_complete_cycle_prompt(self):
        payload = build_cycle_request(self.episode, self.candidate, self.frames,
                                      {"api_key_file": "/not-read-key", "api_key": "sk-fake-never-send"})
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_PROMPT)
        context = json.loads(payload["messages"][1]["content"][0]["text"])
        self.assertEqual(context["prompt_version"], PROMPT_VERSION)
        self.assertEqual(context["units"]["opening"], "normalized_0_1")
        self.assertEqual([p["opening"] for p in context["curve"]], VALUES)
        self.assertNotIn("width_mm", json.dumps(context))
        self.assertNotIn("sk-fake-never-send", json.dumps(payload))
        self.assertNotIn("not-read-key", json.dumps(payload))
        self.assertEqual(context["image_refs"][0]["video_timestamp"], .7)
        self.assertEqual(context["image_sampling"]["submitted_frame_indices"], list(range(len(VALUES))))

    def test_sampling_keeps_original_transition_frames_and_reports_sparse_context(self):
        ep = fixture_episode()
        ep["timestamps"] = [i / 10 for i in range(120)]
        ep["state"] = [[.2] * 20 for _ in ep["timestamps"]]
        c = dict(self.candidate, start=0, end=119, pregrasp_start=0,
                 closing_start=20, contact_frame=24, release_start=90, release_end=94)
        indices = cycle_frame_indices(ep, c, {"steady_image_interval_s": .5, "transition_context_s": .3})
        self.assertTrue(set(range(17, 28)).issubset(indices))
        self.assertTrue(set(range(87, 98)).issubset(indices))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 119)
        self.assertLess(len(indices), len(ep["timestamps"]))
        self.assertLessEqual(max(b - a for a, b in zip(indices, indices[1:])), 5)

    def test_image_budget_rejects_instead_of_discarding_boundary_frames(self):
        for cfg in ({"max_images": 2}, {"max_images": 20, "camera_mode": "stereo"}):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(VisionValidationError, "image_limit"):
                cycle_frame_indices(self.episode, self.candidate, cfg)
        with self.assertRaisesRegex(VisionValidationError, "sampling"):
            cycle_frame_indices(self.episode, self.candidate, {"steady_image_interval_s": 0})

    def test_response_boundaries_and_evidence_must_have_been_submitted(self):
        sparse = self.make_frames([0, 5, 10, 13, 15])
        parsed = parse_cycle_decision(response(self.decision), self.episode, self.candidate, sparse)
        self.assertEqual(parsed["category"], "grasp_release")
        for change in ({"evidence_frames": [0, 5, 9, 13]}, {"hold_start": 6},
                       {"release_end": True}, {"evidence_frames": [0, 5, 13, 13]}):
            with self.subTest(change=change), self.assertRaises(VisionValidationError):
                parse_cycle_decision(response(dict(self.decision, **change)), self.episode, self.candidate, sparse)

    def test_wrong_arm_images_or_model_identity_are_rejected(self):
        wrong = copy.deepcopy(self.frames)
        wrong[0]["camera"] = "observation.images.cam_left_wrist_left"
        with self.assertRaisesRegex(VisionValidationError, "camera_side"):
            build_cycle_request(self.episode, self.candidate, wrong, {})
        with self.assertRaisesRegex(VisionValidationError, "identity"):
            parse_cycle_decision(response(dict(self.decision, side="left")), self.episode, self.candidate, self.frames)

    def test_missing_release_after_evidence_or_slip_does_not_pass(self):
        for change in ({"evidence_frames": [0, 5, 10]}, {"release_end": 10},
                       {"slip": True}, {"continuous_grasp": False}):
            with self.subTest(change=change), self.assertRaises(VisionValidationError):
                parse_cycle_decision(response(dict(self.decision, **change)), self.episode, self.candidate, self.frames)

    def test_uncertain_response_requires_actual_evidence_without_claiming_grasp(self):
        d = dict(self.decision, category="uncertain", hold_start=None, hold_end=None,
                 release_end=None, pregrasp_start=None, pregrasp_clear=False,
                 continuous_grasp=False, evidence_frames=[5], reason="Occluded synthetic fixture")
        parsed = parse_cycle_decision(response(d), self.episode, self.candidate, self.frames)
        self.assertEqual(parsed["category"], "uncertain")
        with self.assertRaises(VisionValidationError):
            parse_cycle_decision(response(dict(d, evidence_frames=[])), self.episode, self.candidate, self.frames)

    def test_model_cannot_promote_its_output_to_a_human_reference(self):
        d = dict(self.decision, review_source="human", confirmed=True, reviewer="invented")
        parsed = parse_cycle_decision(response(d), self.episode, self.candidate, self.frames)
        self.assertEqual(parsed["review_source"], "model")
        self.assertFalse(parsed["confirmed"])
        self.assertNotIn("reviewer", parsed)

    def test_cycle_client_runs_subclass_payload_and_parser_hooks(self):
        client = CycleQwenClient({"max_requests": 1})
        with patch.object(client, "_credential", return_value="sk-fake-cycle-test"), patch(
                "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(response(self.decision)).encode())) as call:
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prompt_version"], PROMPT_VERSION)
        self.assertEqual(result["decision"]["category"], "grasp_release")
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(sent["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(result["request_hash"], request_fingerprint(sent, client.base_url))
        self.assertNotIn("sk-fake-cycle-test", json.dumps(sent))
        self.assertNotIn("sk-fake-cycle-test", json.dumps(result))
        self.assertEqual(call.call_args.args[0].get_header("Authorization"), "Bearer sk-fake-cycle-test")

    def test_image_hash_failure_happens_before_credentials_or_network(self):
        self.frames[0]["sha256"] = "wrong"
        client = CycleQwenClient({})
        with patch.object(client, "_credential", side_effect=AssertionError("credentials read")), patch(
                "close_gripper.vision.urlopen", side_effect=AssertionError("network accessed")):
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["error"], "image_hash_mismatch")
        self.assertEqual(result["attempts"], 0)
        self.assertIn("pregrasp_clear", result["decision"])
        self.assertNotIn("grasp_before", result["decision"])

    def test_transport_error_returns_new_uncertain_schema_and_redacts_secrets(self):
        client = CycleQwenClient({})
        error = HTTPError("https://test.invalid", 401, "sk-fake-cycle-test", {}, io.BytesIO(b"do not record this body"))
        with patch.object(client, "_credential", return_value="sk-fake-cycle-test"), patch(
                "close_gripper.vision.urlopen", side_effect=error):
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["error"], "http_401")
        self.assertEqual(result["decision"]["category"], "uncertain")
        self.assertIn("continuous_grasp", result["decision"])
        self.assertEqual(result["decision"]["evidence_frames"], [])
        self.assertTrue(client.authentication_blocked)
        self.assertNotIn("sk-fake-cycle-test", json.dumps(result))
        self.assertNotIn("do not record this body", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
