"""Offline vision contract tests: no real credentials, service, or dataset."""

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from close_gripper.vision import (
    QwenClient, VisionValidationError, build_request, parse_decision, request_fingerprint,
)


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = "observation.images.cam_left_wrist_left"
        self.episode = {"episode_id": "crop-a", "timestamps": [i / 10 for i in range(20)],
                        "state": [[0.2] * 20 for _ in range(20)], "valid_sides": ["left"],
                        "cameras": {self.camera: {"from_timestamp": 0.5}},
                        "identity": {"source": "original-a", "crop_start": 2.0}}
        self.candidate = {"candidate_id": "left-8", "side": "left", "start": 8, "end": 10}
        data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9hQAAAAASUVORK5CYII=")
        path = self.root / "frame.png"
        path.write_bytes(data)
        self.frames = [{"side": "left", "camera": self.camera, "frame_index": i,
                        "timestamp": i / 10, "video_timestamp": i / 10 + 0.5,
                        "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
                       for i in range(3, 15)]
        self.decision = {"candidate_id": "left-8", "side": "left", "category": "normal_release",
                         "grasp_before": "held", "grasp_during": "held", "grasp_after": "not_held",
                         "hold_start": 3, "hold_end": 8, "evidence_frames": [3, 8, 11],
                         "slip": False, "release_end": 11, "confirmed": True,
                         "review_source": "human", "reason": "Released onto the table."}

    def response(self, decision=None):
        return {"id": "test-request", "model": "test-model", "usage": {"total_tokens": 18},
                "choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(self.decision if decision is None else decision)}}]}

    def test_same_side_labels_and_offset_are_sent(self):
        request = build_request(self.episode, self.candidate, self.frames, {})
        content = request["messages"][1]["content"]
        context = json.loads(content[0]["text"])
        self.assertEqual(context["curve"][0]["width_mm"], 19)
        self.assertEqual(context["image_refs"][0]["video_timestamp"], 0.8)
        self.assertEqual(len([item for item in content if item["type"] == "image_url"]), len(self.frames))
        self.assertNotIn("api_key", json.dumps(request))

    def test_wrong_camera_or_offset_rejected(self):
        for key, value in (("camera", "observation.images.cam_right_wrist_left"),
                           ("video_timestamp", 0.3), ("side", "right"), ("sha256", "wrong")):
            frames = copy.deepcopy(self.frames)
            frames[0][key] = value
            with self.subTest(key=key), self.assertRaises(VisionValidationError):
                build_request(self.episode, self.candidate, frames, {})

    def test_caps_reject_without_downsampling(self):
        with self.assertRaisesRegex(VisionValidationError, "image_limit"):
            build_request(self.episode, self.candidate, self.frames, {"max_images": 3})

    def test_fingerprint_tracks_crop_images_model_prompt_and_generation(self):
        original = build_request(self.episode, self.candidate, self.frames, {})
        digest = request_fingerprint(original)
        for field, value in (("model", "new-model"), ("temperature", 0.5)):
            request = copy.deepcopy(original)
            request[field] = value
            self.assertNotEqual(digest, request_fingerprint(request))
        different = copy.deepcopy(self.episode)
        different["identity"]["crop_start"] = 3.0
        self.assertNotEqual(digest, request_fingerprint(build_request(different, self.candidate, self.frames, {})))
        request = copy.deepcopy(original)
        request["messages"][0]["content"] += "another prompt"
        self.assertNotEqual(digest, request_fingerprint(request))
        request = copy.deepcopy(original)
        request["messages"][1]["content"][2]["image_url"]["url"] += "changed"
        self.assertNotEqual(digest, request_fingerprint(request))

    def test_model_cannot_promote_itself_to_confirmed_human(self):
        decision = parse_decision(self.response(), self.candidate, 20, self.frames)
        self.assertFalse(decision["confirmed"])
        self.assertEqual(decision["review_source"], "model")

    def test_missing_unseen_or_wrong_side_evidence_rejected(self):
        variants = [{"evidence_frames": [3, 8]}, {"evidence_frames": [3, 19]}, {"side": "right"},
                    {"hold_start": 0}, {"hold_start": True}, {"slip": "false"},
                    {"grasp_after": "held"}, {"hold_end": 10}, {"release_end": 9}]
        for delta in variants:
            value = dict(self.decision, **delta)
            with self.subTest(delta=delta), self.assertRaises(VisionValidationError):
                parse_decision(value, self.candidate, 20, self.frames)

    def test_misopen_requires_continuous_grasp(self):
        value = dict(self.decision, category="suspected_misopen", grasp_after="held", hold_end=14, release_end=None)
        self.assertEqual(parse_decision(value, self.candidate, 20, self.frames)["category"], "suspected_misopen")
        for delta in ({"grasp_during": "uncertain"}, {"slip": True}, {"hold_end": 8}):
            with self.assertRaises(VisionValidationError):
                parse_decision(dict(value, **delta), self.candidate, 20, self.frames)

    def test_malformed_and_truncated_response_rejected(self):
        with self.assertRaises(VisionValidationError):
            parse_decision("not JSON", self.candidate, 20, self.frames)
        response = self.response()
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(VisionValidationError):
            parse_decision(response, self.candidate, 20, self.frames)

    def test_client_init_is_offline_and_does_not_read_key(self):
        with patch("pathlib.Path.read_text", side_effect=AssertionError("credential read")), patch(
                "close_gripper.vision.urlopen", side_effect=AssertionError("network")):
            QwenClient({"api_key_file": "missing-file"})

    def test_labelled_single_line_credentials_and_json(self):
        path = self.root / "fake-credential.txt"
        for text in ("sk-fake-only", "api_key=sk-fake-only", 'key: "sk-fake-only"',
                     "测试密钥：'sk-fake-only'", '{"api_key": "sk-fake-only"}'):
            path.write_text(text)
            self.assertEqual(QwenClient({"api_key_file": str(path)})._credential(), "sk-fake-only")
        for text in ("sk-fake-only sk-another-fake", "label\nsk-fake-only", "not-a-key"):
            path.write_text(text)
            with self.assertRaises(VisionValidationError):
                QwenClient({"api_key_file": str(path)})._credential()

    def test_default_workspace_file_beats_implicit_environment(self):
        fake_default = self.root / "qwen_api.txt"
        fake_default.write_text("测试密钥：sk-default-fake")
        with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", fake_default), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": "sk-env-fake"}, clear=True):
            self.assertEqual(QwenClient({})._credential(), "sk-default-fake")

    def test_dotted_key_is_preserved_in_full_authorization_and_absent_from_body(self):
        key = "sk-FAKE.segment-two.segment-three.final_" + "x" * 75
        self.assertEqual(len(key), 115)
        fake_default = self.root / "qwen_api.txt"
        for text in (key, f"api_key={key}", f'key: "{key}"', f"测试密钥 {key}", json.dumps({"api_key": key})):
            fake_default.write_text(text)
            client = QwenClient({})
            with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", fake_default), patch(
                    "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(self.response()).encode())) as call:
                result = client.annotate(self.episode, self.candidate, self.frames)
            self.assertEqual(result["status"], "ok")
            request = call.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer " + key)
            self.assertNotIn(key.encode(), request.data)
            self.assertNotIn(key, json.dumps(result))

    def test_invalid_key_suffixes_and_multiple_keys_never_truncate(self):
        fake = self.root / "invalid-key.txt"
        for text in ("sk-valid.suffix!", "sk-valid/suffix", "sk-valid+suffix", "sk-valid trailing",
                     "key=sk-valid!", "sk-first sk-second", "key sk-first sk-second", "key=sk-first,sk-second",
                     "sk-first\nsk-second", 'key="sk-valid.suffix\''):
            with self.subTest(text=text):
                fake.write_text(text)
                with self.assertRaises(VisionValidationError):
                    QwenClient({"api_key_file": str(fake)})._credential()

    def test_explicit_file_and_named_environment_precedence(self):
        explicit = self.root / "explicit.txt"
        explicit.write_text("sk-explicit-fake")
        with patch.dict(os.environ, {"CUSTOM_QWEN_KEY": "sk-env-fake"}, clear=True), patch(
                "close_gripper.vision.DEFAULT_API_KEY_FILE") as default:
            self.assertEqual(QwenClient({"api_key_file": str(explicit), "api_key_env": "CUSTOM_QWEN_KEY"})._credential(),
                             "sk-explicit-fake")
            self.assertEqual(QwenClient({"api_key_env": "CUSTOM_QWEN_KEY"})._credential(), "sk-env-fake")
            default.exists.assert_not_called()

    def test_missing_default_file_allows_environment_fallback_only(self):
        missing = self.root / "absent.txt"
        with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", missing), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": "sk-env-fake"}, clear=True):
            self.assertEqual(QwenClient({})._credential(), "sk-env-fake")
            with self.assertRaisesRegex(VisionValidationError, "credential_unavailable"):
                QwenClient({"api_key_file": str(missing)})._credential()
            with self.assertRaisesRegex(VisionValidationError, "credential_unavailable"):
                QwenClient({"api_key_env": "MISSING_QWEN_KEY"})._credential()
        with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", missing), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(VisionValidationError, "credential_unavailable"):
                QwenClient({})._credential()

    def test_invalid_default_file_does_not_fall_back_to_environment(self):
        invalid = self.root / "qwen_api.txt"
        invalid.write_text("not a credential")
        with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", invalid), patch.dict(
                os.environ, {"DASHSCOPE_API_KEY": "sk-env-fake"}, clear=True):
            with self.assertRaises(VisionValidationError):
                QwenClient({})._credential()

    def test_annotate_default_key_never_enters_result_or_config(self):
        fake_default = self.root / "qwen_api.txt"
        fake_default.write_text("key=sk-default-fake")
        client = QwenClient({})
        response = self.response()
        # Exercise redaction as well as actual default-source authentication.
        response["diagnostic"] = "sk-default-fake"
        with patch("close_gripper.vision.DEFAULT_API_KEY_FILE", fake_default), patch(
                "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(call.call_args.args[0].get_header("Authorization"), "Bearer sk-default-fake")
        self.assertNotIn("sk-default-fake", json.dumps(result))
        self.assertEqual(client.config, {})

    def test_client_records_usage_and_enforces_budget(self):
        client = QwenClient({"max_requests": 1})
        with patch.object(client, "_credential", return_value="sk-fake-test"), patch(
                "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(self.response()).encode())) as call:
            result = client.annotate(self.episode, self.candidate, self.frames)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["usage"]["total_tokens"], 18)
            self.assertEqual(result["model"], "test-model")
            second = client.annotate(self.episode, self.candidate, self.frames)
            self.assertEqual(second["error"], "request_budget_exhausted")
            self.assertEqual(call.call_count, 1)

    def test_network_errors_do_not_leak_service_body_or_key(self):
        client = QwenClient({})
        error = HTTPError("https://test.invalid", 401, "sk-fake-test", {}, io.BytesIO(b"secret echoed body"))
        with patch.object(client, "_credential", return_value="sk-fake-test"), patch(
                "close_gripper.vision.urlopen", side_effect=error) as call:
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["error"], "http_401")
        self.assertNotIn("sk-fake-test", json.dumps(result))
        self.assertNotIn("secret echoed body", json.dumps(result))
        self.assertEqual(call.call_count, 1)

    def test_retry_budget_includes_failed_calls(self):
        client = QwenClient({"max_requests": 1, "max_retries": 2})
        error = HTTPError("https://test.invalid", 503, "unavailable", {}, io.BytesIO(b"ignored"))
        with patch.object(client, "_credential", return_value="sk-fake-test"), patch(
                "close_gripper.vision.urlopen", side_effect=error) as call, patch("close_gripper.vision.time.sleep"):
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["error"], "request_budget_exhausted")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(call.call_count, 1)

    def test_authentication_denial_blocks_later_uploads_and_credential_reads(self):
        for status_code in (401, 403):
            with self.subTest(status_code=status_code):
                client = QwenClient({"max_requests": 4, "max_retries": 2})
                error = HTTPError("https://test.invalid", status_code, "denied", {}, io.BytesIO(b"ignored"))
                with patch.object(client, "_credential", return_value="sk-fake-test") as credential, patch(
                        "close_gripper.vision.urlopen", side_effect=error) as call:
                    first = client.annotate(self.episode, self.candidate, self.frames)
                    with patch("close_gripper.vision.build_request", side_effect=AssertionError("image bytes read")):
                        second = client.annotate(self.episode, self.candidate, self.frames)
                self.assertEqual(first["error"], f"http_{status_code}")
                self.assertEqual(first["attempts"], 1)
                self.assertEqual(second["error"], "authentication_blocked")
                self.assertEqual(second["attempts"], 0)
                self.assertEqual(second["decision"]["category"], "uncertain")
                self.assertIsNone(second["request_hash"])
                self.assertEqual(client.request_count, 1)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(credential.call_count, 1)

    def test_invalid_response_preserves_raw_and_falls_back(self):
        client = QwenClient({})
        response = self.response(dict(self.decision, evidence_frames=[0]))
        with patch.object(client, "_credential", return_value="sk-fake-test"), patch(
                "close_gripper.vision.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            result = client.annotate(self.episode, self.candidate, self.frames)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["decision"]["category"], "uncertain")
        self.assertEqual(result["raw_response"], response)
        self.assertEqual(result["usage"]["total_tokens"], 18)

    def test_credential_only_goes_to_https_qwen_endpoint(self):
        for url in ("http://dashscope.aliyuncs.com/compatible-mode/v1", "https://example.com/compatible-mode/v1",
                    "https://user@dashscope.aliyuncs.com/compatible-mode/v1"):
            with self.assertRaises(VisionValidationError):
                QwenClient({"base_url": url})


if __name__ == "__main__":
    unittest.main()
