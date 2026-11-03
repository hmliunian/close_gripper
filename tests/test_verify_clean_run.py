"""Independent saved-run audits catch corruption without invoking the producer."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_clean_run", Path(__file__).resolve().parents[1] / "scripts/verify_clean_run.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class CleanRunAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        self.run = self.workspace / "runs/source"
        self.directory = self.run / "episode"
        self.directory.mkdir(parents=True)
        self.input = self.workspace / "inputs"
        self.input.mkdir()
        self.camera = "observation.images.cam_right_wrist_left"
        self.episode = dict(episode_id="audit/0", timestamps=[0., .1, .2, .3], frame_indices=[0, 1, 2, 3],
                            global_indices=[10, 11, 12, 13], state=[[0.] * 19 + [v] for v in (.8, .4, .4, .8)],
                            valid_sides=["right"], identity=dict(tasks=["fixture"], dataset_root="/read-only/source"),
                            cameras={self.camera: dict(path="/read-only/source/video.mp4", from_timestamp=1., to_timestamp=1.4)})
        self.candidate = dict(candidate_id="candidate", side="right", start=0, end=3, closing_start=0,
                              contact_frame=1, release_start=2, release_end=3)
        self.decision = dict(candidate_id="candidate", side="right", category="grasp_release", hold_start=1,
                             hold_end=2, release_end=3, pregrasp_start=None, pregrasp_clear=False,
                             continuous_grasp=True, slip=False, evidence_frames=[0, 1, 2, 3], review_source="model",
                             confirmed=False, reason="fixture only", cameras=[self.camera])
        self.event = dict(self.candidate, status="applied", reason="automatic_visual_numeric_cycle", reviewed=True,
                          decision=copy.deepcopy(self.decision), modified_start=0, modified_end=3,
                          source="model", changed_frames=2, hold_end=2)
        self.coverage = {}
        for side in ("left", "right"):
            active = side == "right"
            self.coverage[side] = dict(valid_side=active, candidates=int(active), reviewed_candidates=int(active),
                accepted_cycles=int(active), preview_cycles=0, rejected_cycles=0, no_grasp_candidates=0,
                unreviewed_candidates=0, reviewed_frames=4 if active else 0, review_window_frames=4 if active else 0,
                unreviewed_intervals=[] if active else [[0, 3]], target_valid_frames=4 if active else 0,
                changed_frames=2 if active else 0, complete_review=active)
        self.result = dict(target=[[0., v] for v in (.8, 0., 0., .8)], valid=[[False, True] for _ in range(4)],
                           source=[["", "model"] for _ in range(4)],
                           phase=[["unchanged", p] for p in ("closing", "hold", "release", "release")],
                           events=[self.event], coverage=self.coverage, side_status=dict(left="invalid", right="reviewed"),
                           config=dict(config_version="fixture-v1"), evidence_type="real_vlm")
        for key in ("target", "source", "phase"):
            self.result["preview_" + key] = copy.deepcopy(self.result[key])
        self.targets = dict(episode_id=self.episode["episode_id"], identity=self.episode["identity"],
                            timestamps=self.episode["timestamps"], frame_indices=self.episode["frame_indices"],
                            episode_sha256=digest(self.episode), evidence_type="real_vlm", method_version="fixture-v1")
        for name, key in verify.FIELDS.items():
            self.targets[key] = copy.deepcopy(self.result[name])
        self.frames = []
        (self.directory / "frames_000").mkdir()
        for i in range(4):
            path = self.directory / "frames_000" / f"{i}.jpg"
            path.write_bytes(b"image fixture" + bytes([i]))
            self.frames.append(dict(side="right", camera=self.camera, frame_index=i, source_frame_index=i,
                timestamp=i / 10, video_timestamp=1 + i / 10, video_path="/read-only/source/video.mp4", path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        usage = dict(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        raw = dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps(self.decision)))], usage=usage)
        self.response = dict(status="ok", attempts=1, candidate_id="candidate", decision=copy.deepcopy(self.decision),
                             raw_response=raw, usage=usage, request_hash="a" * 64, submitted_frame_indices=[0, 1, 2, 3], evidence_images=4)
        self.config = dict(max_requests=2, config_version="fixture-v1")
        self.summary = dict(status="complete", workflow="full_cycle", episodes=[dict(episode_id="audit/0", directory="episode",
                            changed_frames=2, coverage=self.coverage, side_status=self.result["side_status"])],
                            tasks={'["fixture"]':dict(episodes=1, changed_episodes=1, changed_frames=2)},
                            actual_requests=1, usage_reported_requests=1, reported_token_usage=usage,
                            config_sha256=hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
                            implementation_changed_during_run=False)
        implementation = self.run / "implementation"
        implementation.mkdir()
        (implementation / "sample.py").write_text("VALUE = 1\n")
        value = hashlib.sha256(b"sample.pyVALUE = 1\n").hexdigest()
        self.summary["implementation_sha256"] = value
        self.save(self.run / "implementation.json", dict(files=["sample.py"], sha256=value))
        self.save(self.input / "episode.json", self.episode)
        self.manifest = self.input / "manifest.json"
        self.save(self.manifest, dict(episodes=[dict(episode_id="audit/0", episode_cache="episode.json", episode_sha256=digest(self.episode))]))
        self.write()

    def save(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False))

    def write(self):
        for name, value in (("episode", self.episode), ("targets", self.targets), ("result", self.result),
                            ("candidates", [self.candidate]), ("decisions", [self.decision]), ("frames", self.frames),
                            ("model_responses", [self.response]),
                            ("review_template", dict(episode_sha256=digest(self.episode), decisions=[self.decision]))):
            self.save(self.directory / (name + ".json"), value)
        self.save(self.run / "summary.json", self.summary)
        self.save(self.run / "config.json", self.config)

    def report(self):
        return verify.verify_run(self.run, self.manifest)

    def assertProblem(self, code):
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertIn(code, {e["check"] for e in report["errors"]})

    def test_complete_artifact_audit_uses_no_producer_and_writes_fresh_output(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        report = self.report()
        self.assertEqual(report["errors"], [])
        self.assertEqual((report["changed_frames"], report["valid_side_frames"], report["actual_requests"]), (2, [0, 4], 1))
        self.assertEqual(report["input_snapshots_checked"], 1)
        self.assertEqual(report["implementation"]["status"], "checked")
        self.assertIn("does not mean every frame", report["coverage_definition"])
        output = verify.write_audit(report, self.workspace / "runs/audit", self.workspace)
        self.assertTrue((output / "verification.json").is_file())
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            verify.write_audit(report, output, self.workspace)
        with self.assertRaises(ValueError):
            verify.write_audit(report, self.run / "overwrite", self.workspace)

    def test_invalid_target_and_unavailable_side_changes_are_rejected(self):
        self.result["target"][1][0] = .3
        self.targets["annotation.gripper_target"][1][0] = .3
        self.write()
        self.assertProblem("invalid_target_preserves_original")
        self.assertProblem("unavailable_side_preserved")

    def test_consistently_rehashed_observation_tampering_still_differs_from_original_snapshot(self):
        self.episode["state"][0][2] = 12.
        self.targets["episode_sha256"] = digest(self.episode)
        self.write()
        self.assertProblem("original_observations_match_input_snapshot")

    def test_target_result_disagreement_and_wrong_phase_fail(self):
        self.targets["annotation.gripper_target"][1][1] = .5
        self.result["phase"][1][1] = "closing"
        self.write()
        self.assertProblem("targets_result_target")
        self.assertProblem("valid_source_phase_interval")

    def test_model_cannot_be_promoted_to_human_review(self):
        self.decision["review_source"] = "human"
        self.event["decision"] = copy.deepcopy(self.decision)
        self.write()
        self.assertProblem("human_review_provenance")
        self.assertProblem("event_source")

    def test_image_bytes_camera_offset_and_unsubmitted_evidence_are_checked(self):
        Path(self.frames[0]["path"]).write_bytes(b"tampered")
        self.frames[1]["camera"] = "observation.images.cam_left_wrist_left"
        self.frames[2]["video_timestamp"] = 9.
        self.response["submitted_frame_indices"] = [0, 1, 3]
        self.write()
        self.assertProblem("image_sha256")
        self.assertProblem("image_same_side_camera")
        self.assertProblem("image_video_offset")
        self.assertProblem("evidence_was_submitted")

    def test_request_usage_and_task_counters_are_independently_recomputed(self):
        self.summary["actual_requests"] = 2
        self.summary["reported_token_usage"] = dict(prompt_tokens=10, completion_tokens=4, total_tokens=99)
        self.summary["tasks"]['["fixture"]']["changed_frames"] = 99
        self.write()
        for code in ("summary_actual_requests", "summary_reported_token_usage", "summary_tasks"):
            self.assertProblem(code)

    def test_failed_network_attempts_do_not_become_service_responses_or_known_zero_usage(self):
        self.response.update(status="error", error="network_error", raw_response=None, usage={})
        self.event.update(status="unchanged", reason="missing_visual_decision", reviewed=False, changed_frames=0)
        self.event.pop("decision")
        self.result["target"] = [[0., s[19]] for s in self.episode["state"]]
        self.result["valid"] = [[False, False] for _ in range(4)]
        self.result["source"] = [["", ""] for _ in range(4)]
        self.result["phase"] = [["unchanged", "unchanged"] for _ in range(4)]
        for name, key in verify.FIELDS.items():
            self.targets[key] = copy.deepcopy(self.result[name])
        self.coverage["right"].update(reviewed_candidates=0, accepted_cycles=0, unreviewed_candidates=1,
            reviewed_frames=0, review_window_frames=0, unreviewed_intervals=[[0, 3]], target_valid_frames=0,
            changed_frames=0, complete_review=False)
        self.result["side_status"]["right"] = "incomplete_review"
        self.summary["episodes"][0]["changed_frames"] = 0
        self.summary.update(usage_reported_requests=0, reported_token_usage=dict.fromkeys(verify.TOKENS, 0),
                            tasks={'["fixture"]':dict(episodes=1, changed_episodes=0, changed_frames=0)})
        self.write()
        self.save(self.directory / "decisions.json", [])
        self.save(self.directory / "review_template.json", dict(episode_sha256=digest(self.episode), decisions=[]))
        report = self.report()
        self.assertEqual(report["errors"], [])
        self.assertEqual((report["request_attempts"], report["service_responses"], report["usage_reported_requests"]), (1, 0, 0))
        self.assertEqual(report["attempts_without_reported_usage"], 1)
        self.assertEqual(report["attempts_without_saved_service_response"], 1)
        self.assertEqual(report["request_failures"], {"network_error": 1})
        self.assertIn("unknown", report["request_count_definition"])

    def test_implementation_tampering_and_missing_historical_snapshot_are_distinguished(self):
        (self.run / "implementation/sample.py").write_text("VALUE = 2\n")
        self.assertProblem("implementation_snapshot_sha256")
        (self.run / "implementation.json").unlink()
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["implementation"]["status"], "not_available")
        self.assertTrue(any("historical run" in text for text in report["limitations"]))

    def test_missing_candidate_judgment_is_not_hidden_by_other_reviewed_context(self):
        candidate = dict(self.candidate, candidate_id="unreviewed")
        event = dict(candidate, status="unchanged", reason="missing_visual_decision", changed_frames=0, reviewed=False)
        self.result["events"].append(event)
        self.coverage["right"].update(candidates=2, unreviewed_candidates=1)
        self.write()
        self.save(self.directory / "candidates.json", [self.candidate, candidate])
        self.assertProblem("coverage_complete_review")

    def test_manifest_parent_traversal_is_rejected_before_reading(self):
        self.save(self.manifest, dict(episodes=[dict(episode_cache="../foreign.json")]))
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.report()

    def replay_fixture(self):
        replay = self.workspace / "runs/replay"
        shutil.copytree(self.run, replay)
        directory = replay / "episode"
        filenames = ("episode.json", "candidates.json", "decisions.json", "review_template.json",
                     "model_responses.json", "frames.json", "targets.json", "result.json")
        hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest() for name in filenames}
        self.save(directory / "reused_evidence.json", dict(schema="fixture_reuse", source_run=str(self.run),
                  source_episode_directory="episode", episode_sha256=digest(self.episode), source_files_sha256=hashes))
        shutil.copyfile(self.directory / "model_responses.json", directory / "source_model_responses.json")
        shutil.copyfile(self.directory / "frames.json", directory / "source_frames.json")
        self.save(directory / "model_responses.json", [])
        frames = copy.deepcopy(self.frames)
        for frame in frames:
            frame["path"] = str(replay / Path(frame["path"]).relative_to(self.run))
        self.save(directory / "frames.json", frames)
        targets, result = copy.deepcopy(self.targets), copy.deepcopy(self.result)
        targets.update(evidence_type="reused_real_vlm", method_version="fixture-v2")
        result.update(evidence_type="reused_real_vlm", config=dict(config_version="fixture-v2"))
        self.save(directory / "targets.json", targets)
        self.save(directory / "result.json", result)
        config = dict(self.config, config_version="fixture-v2")
        summary = dict(self.summary, source_run=str(self.run), repeated_evaluation_same_inputs=True,
                       new_model_requests=0, actual_requests=0, usage_reported_requests=0,
                       reported_token_usage=dict.fromkeys(verify.TOKENS, 0), reused_actual_requests=1,
                       reused_reported_token_usage=self.response["usage"],
                       source_implementation_sha256=self.summary["implementation_sha256"],
                       config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest())
        self.save(replay / "config.json", config)
        self.save(replay / "summary.json", summary)
        return replay

    def test_reused_evidence_keeps_model_labels_without_counting_old_usage_as_new_calls(self):
        replay = self.replay_fixture()
        report = verify.verify_run(replay, self.manifest)
        self.assertEqual(report["errors"], [])
        self.assertEqual((report["request_attempts"], report["service_responses"], report["reported_token_usage"]["total_tokens"]), (0, 0, 0))
        self.assertEqual((report["reused_request_attempts"], report["reused_service_responses"], report["reused_reported_token_usage"]["total_tokens"]), (1, 1, 14))
        self.assertEqual(report["valid_side_frames"], [0, 4])
        self.assertTrue(any("not an independent evaluation" in value for value in report["limitations"]))
        with self.assertRaises(ValueError):
            verify.write_audit(report, self.run / "nested", self.workspace)

    def test_reused_decisions_and_original_response_bytes_cannot_be_silently_rewritten(self):
        replay = self.replay_fixture()
        directory = replay / "episode"
        decision = dict(self.decision, reason="edited after seeing the new curve")
        self.save(directory / "decisions.json", [decision])
        self.save(directory / "source_model_responses.json", [])
        report = verify.verify_run(replay, self.manifest)
        codes = {e["check"] for e in report["errors"]}
        self.assertEqual(report["status"], "failed")
        self.assertIn("reused_decisions_unchanged", codes)
        self.assertIn("reuse_copied_file_sha256", codes)
        self.assertIn("reuse_responses_unchanged", codes)


if __name__ == "__main__":
    unittest.main()
