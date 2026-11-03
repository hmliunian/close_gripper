"""Offline end-to-end artifact, filesystem isolation, and local training contracts."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline
from close_gripper.cli import main
from close_gripper.contracts import SIDE_CAMERAS
from close_gripper.demo import examples
from close_gripper.training import apply_gripper_targets


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def new_run(self, name="case"):
        return pipeline.create_run(self.workspace / "runs" / name)

    def fixture(self):
        ep, ds = examples()[0]
        ep = copy.deepcopy(ep)
        # Explicit synthetic camera metadata; no images or visual truth are fabricated.
        ep["cameras"] = {name: {"valid": True, "synthetic": True} for names in SIDE_CAMERAS.values() for name in names}
        return ep, ds

    def test_create_run_refuses_reuse(self):
        run = self.new_run()
        sentinel = run / "preserved.txt"
        sentinel.write_text("original")
        with self.assertRaises(FileExistsError):
            pipeline.create_run(run)
        self.assertEqual(sentinel.read_text(), "original")

    def test_create_run_refuses_escape_and_runs_root(self):
        for path in (self.root / "outside", self.workspace / "runs", self.workspace / "runs/../../escape"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                pipeline.create_run(path)

    def test_create_run_refuses_symlinked_parent(self):
        runs = self.workspace / "runs"
        runs.mkdir()
        source = self.root / "dataset"
        source.mkdir()
        (runs / "link").symlink_to(source, target_is_directory=True)
        with self.assertRaises(ValueError):
            pipeline.create_run(runs / "link" / "new")
        self.assertEqual(list(source.iterdir()), [])

    def test_create_run_refuses_symlinked_runs_directory(self):
        source = self.root / "dataset"
        source.mkdir()
        (self.workspace / "runs").symlink_to(source, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.new_run()

    def test_create_run_refuses_input_overlap_in_either_direction(self):
        prospective = self.workspace / "runs" / "new"
        for source in (prospective, self.workspace, prospective / "input.json"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                pipeline.create_run(prospective, [source])

    def test_process_without_decision_preserves_every_target(self):
        ep, _ = self.fixture()
        snapshot = copy.deepcopy(ep)
        run = self.new_run()
        record = pipeline.process_episode(ep, run, pipeline.default_config())
        self.assertEqual(ep, snapshot)
        self.assertEqual(record["result"]["target"], [[s[9], s[19]] for s in ep["state"]])
        self.assertFalse(any(any(row) for row in record["result"]["valid"]))
        self.assertEqual(record["result"]["evidence_type"], "no_visual_review")
        self.assertEqual(record["result"]["events"][0]["reason"], "missing_visual_decision")
        artifact = json.loads(next(run.glob("*/targets.json")).read_text())
        self.assertEqual(artifact["timestamps"], ep["timestamps"])
        self.assertEqual(artifact["identity"], ep["identity"])

    def test_synthetic_review_changes_preview_only(self):
        ep, ds = self.fixture()
        run = self.new_run()
        record = pipeline.process_episode(ep, run, pipeline.default_config(), decisions=ds)
        res = record["result"]
        self.assertNotEqual(res["preview_target"], res["target"])
        self.assertFalse(any(any(row) for row in res["valid"]))
        self.assertEqual(res["events"][0]["reason"], "synthetic_evidence")
        files = {p.name for p in next(run.iterdir()).iterdir()}
        self.assertTrue({"episode.json", "candidates.json", "frames.json", "model_responses.json", "decisions.json", "targets.json", "result.json"}.issubset(files))
        self.assertEqual(json.loads(next(run.glob("*/decisions.json")).read_text()), ds)

    def test_finish_run_writes_self_contained_report_and_summary(self):
        ep, ds = self.fixture()
        run = self.new_run()
        record = pipeline.process_episode(ep, run, pipeline.default_config(), decisions=ds)
        with patch.object(pipeline, "write_plot", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            summary = pipeline.finish_run(run, [record], pipeline.default_config())
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["real_vlm_requests"], 0)
        self.assertEqual(summary["tasks"][ep["identity"]["task"]]["accepted_side_frames"], 0)
        page = (run / "review.html").read_text()
        self.assertIn("<svg", page)
        self.assertIn("synthetic", page)
        self.assertNotIn("__DATA__", page)
        self.assertTrue((run / "config.json").exists())
        self.assertEqual(json.loads((run / "summary.json").read_text()), summary)

    def test_demo_cli_completes_without_network_or_dataset(self):
        output = self.workspace / "runs" / "demo"
        with patch.object(pipeline, "write_plot", return_value=None), patch("socket.create_connection", side_effect=AssertionError("offline test must not access network")), contextlib.redirect_stdout(io.StringIO()):
            code = main(["demo", "--output", str(output)])
        self.assertEqual(code, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["episodes"], 5)
        results = [json.loads(p.read_text()) for p in output.glob("*/result.json")]
        self.assertTrue(any(result["preview_target"] != result["target"] for result in results))
        self.assertFalse(any(any(row) for result in results for row in result["valid"]))

    def test_config_omits_credentials(self):
        config = pipeline.public_config({"api_key": "never-log", "api_key_file": "/secret/key", "token": "never-log", "password": "never-log", "max_tokens": 4000})
        self.assertEqual(config, {"max_tokens": 4000})

    def test_config_recursive_redaction_preserves_only_public_fields(self):
        original = {"nested": {"api_key": "do-not-persist", "settings": [
            {"password": "do-not-persist", "max_tokens": 123, "temperature": 0},
            {"client_secret": "do-not-persist", "model": "fixture"}]},
            "max_tokens": 4000}
        snapshot = copy.deepcopy(original)
        public = pipeline.public_config(original)
        self.assertEqual(public, {"nested": {"settings": [{"max_tokens": 123, "temperature": 0}, {"model": "fixture"}]}, "max_tokens": 4000})
        self.assertEqual(original, snapshot)

    def test_inline_credentials_are_rejected(self):
        for key in ("api_key", "secret", "password", "access_token"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                pipeline.default_config({key: "do-not-persist"})
        self.assertEqual(pipeline.default_config({"api_key_file": "/fixture/credential"})["api_key_file"], "/fixture/credential")

    def test_fingerprint_is_stable_and_covers_source_data(self):
        ep, _ = self.fixture()
        digest = pipeline.episode_fingerprint(ep)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, pipeline.episode_fingerprint(dict(reversed(list(ep.items())))))
        mutations = [("state", lambda v: v[0].__setitem__(9, .123)),
                     ("timestamps", lambda v: v.__setitem__(0, .001)),
                     ("identity", lambda v: v.update(source="another-source")),
                     ("cameras", lambda v: v.update(extra_camera={"valid": False}))]
        for key, mutate in mutations:
            changed = copy.deepcopy(ep)
            mutate(changed[key])
            with self.subTest(key=key):
                self.assertNotEqual(digest, pipeline.episode_fingerprint(changed))

    def test_stale_review_snapshot_is_rejected(self):
        ep, ds = self.fixture()
        envelope = {"episode_sha256": pipeline.episode_fingerprint(ep), "decisions": ds}
        self.assertEqual(pipeline.reviewed_decisions(ep, envelope), ds)
        ep["state"][0][9] = .123
        with self.assertRaises(ValueError):
            pipeline.reviewed_decisions(ep, envelope)
        for bad in ([], {}, {"episode_sha256": "stale", "decisions": ds}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pipeline.reviewed_decisions(ep, bad)

    def test_confirmed_human_review_requires_individual_audit_fields(self):
        ep, ds = self.fixture()
        human = dict(ds[0], review_source="human", confirmed=True,
                     reviewer="fixture reviewer", reviewed_at="2026-11-03T10:00:00+08:00", review_note="Reviewed before/during/after fixture")
        envelope = dict(episode_sha256=pipeline.episode_fingerprint(ep), decisions=[human])
        self.assertEqual(pipeline.reviewed_decisions(ep, envelope), [human])
        for key in ("reviewer", "reviewed_at", "review_note"):
            for value in (None, "", " ", 123):
                changed = copy.deepcopy(envelope)
                changed["decisions"][0][key] = value
                # Envelope metadata must not supply missing per-decision attribution.
                changed[key] = "envelope value does not confer approval"
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    pipeline.reviewed_decisions(ep, changed)

    def test_optional_review_metadata_is_preserved_without_conferring_confirmation(self):
        ep, ds = self.fixture()
        pending = dict(ds[0], review_source="model", confirmed=False, extra_review_metadata={"future_version": 2})
        envelope = dict(episode_sha256=pipeline.episode_fingerprint(ep), decisions=[pending], future_envelope_metadata="ignored")
        result = pipeline.reviewed_decisions(ep, envelope)
        self.assertEqual(result, [pending])
        record = pipeline.process_episode(ep, self.new_run(), pipeline.default_config(), decisions=result)
        self.assertFalse(any(any(row) for row in record["result"]["valid"]))

    def test_unknown_review_source_cannot_become_a_training_target(self):
        ep, ds = self.fixture()
        unknown = dict(ds[0], review_source="unknown-reviewer-format", confirmed=True, reviewer="someone")
        envelope = dict(episode_sha256=pipeline.episode_fingerprint(ep), decisions=[unknown])
        record = pipeline.process_episode(ep, self.new_run(), pipeline.default_config(), decisions=pipeline.reviewed_decisions(ep, envelope))
        self.assertEqual(record["result"]["events"][0]["reason"], "invalid_visual_decision")
        self.assertFalse(any(any(row) for row in record["result"]["valid"]))

    def test_malformed_review_entries_fail_with_value_error(self):
        ep, _ = self.fixture()
        fingerprint = pipeline.episode_fingerprint(ep)
        for decisions in (None, {}, [None], ["unexpected"], [2]):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                pipeline.reviewed_decisions(ep, {"episode_sha256": fingerprint, "decisions": decisions})


class TrainingContractTests(unittest.TestCase):
    def test_action_adapter_preserves_observation_and_other_dimensions(self):
        observations = [[float(i) for i in range(20)] for _ in range(3)]
        snapshot = copy.deepcopy(observations)
        target = [[.1, .2], [.3, .4], [.5, .6]]
        valid = [[True, False], [False, True], [False, False]]
        actions = apply_gripper_targets(observations, target, valid)
        self.assertEqual(observations, snapshot)
        for frame in range(3):
            for dimension in range(20):
                expected = .1 if (frame, dimension) == (0, 9) else .4 if (frame, dimension) == (1, 19) else snapshot[frame][dimension]
                self.assertEqual(actions[frame][dimension], expected)
        self.assertIsNot(actions, observations)
        self.assertTrue(all(a is not b for a, b in zip(actions, observations)))

    def test_invalid_targets_only_fall_back_when_flag_false(self):
        action = [[0.] * 20]
        self.assertEqual(apply_gripper_targets(action, [[float("nan"), 10]], [[False, False]]), action)
        for value in (float("nan"), float("inf"), -1, 2, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_gripper_targets(action, [[value, 0]], [[True, False]])

    def test_dimensions_and_boolean_contract(self):
        cases = [([[0] * 19], [[0, 0]], [[True, False]]),
                 ([[0] * 20], [[0]], [[True, False]]),
                 ([[0] * 20], [[0, 0]], [[1, False]]),
                 ([[0] * 20], [], [])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                apply_gripper_targets(*args)


if __name__ == "__main__":
    unittest.main()
