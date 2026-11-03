"""Offline batch orchestration and accounting contracts; no actual Qwen calls."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from close_gripper import batch, pipeline
from close_gripper.core import detect_candidates
from close_gripper.demo import examples


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.manifest_dir = self.root / "frozen-manifest"
        (self.manifest_dir / "episodes").mkdir(parents=True)
        self.dataset = self.root / "release" / "subset"
        self.dataset.mkdir(parents=True)
        self.manifest = self.manifest_dir / "manifest.json"
        self.patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def entries(self, count=4):
        items = []
        for i in range(count):
            ep, _ = examples()[2]  # Same opening index on BOTH sides.
            ep = copy.deepcopy(ep)
            ep["episode_id"] = f"batch-fixture-{i}"
            side = "right" if i % 2 == 0 else "left"
            candidates = detect_candidates(ep)
            chosen = next(c for c in candidates if c["side"] == side)
            cache = Path("episodes") / f"{i}.json"
            pipeline.dump(self.manifest_dir / cache, ep)
            items.append(dict(episode_cache=cache.as_posix(), episode_sha256=pipeline.episode_fingerprint(ep),
                              episode_id=ep["episode_id"], candidate_id=chosen["candidate_id"], side=side,
                              candidate_start=chosen["start"], candidate_end=chosen["end"],
                              dataset=str(self.dataset), task=["cup" if i % 2 == 0 else "fold"], source_tasks=["fixture-task"]))
        pipeline.dump(self.manifest, items)
        return items

    def fake_client(self, barrier=None, error=None):
        calls, instances, lock = [], [], threading.Lock()

        class FakeClient:
            def __init__(self, config):
                self.config = dict(config)
                self.request_count = 0
                with lock:
                    instances.append(self)

            def annotate(self, ep, candidate, frames):
                self.request_count += 1
                if self.request_count > self.config["max_requests"]:
                    raise AssertionError("test detected an extra request")
                if self.config["candidate_ids"] != [candidate["candidate_id"]]:
                    raise AssertionError("wrong selected candidate")
                with lock:
                    calls.append((ep["episode_id"], candidate["candidate_id"], candidate["side"]))
                if barrier:
                    barrier.wait(timeout=10)
                decision = dict(candidate_id=candidate["candidate_id"], side=candidate["side"],
                                category="uncertain", grasp_before="uncertain", grasp_during="uncertain",
                                grasp_after="uncertain", hold_start=None, hold_end=None, release_end=None,
                                evidence_frames=[], slip=False, confirmed=False, review_source="model", reason="offline test")
                return dict(status="error" if error else "ok", decision=decision, attempts=1,
                            error=error, raw_response=None if error == "http_401" else {"fixture": True},
                            usage={} if error == "http_401" else {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                            model="offline-fixture")

        return FakeClient, calls, instances

    def run_fixture(self, output_name="batch", count=4, workers=4, error=None, barrier=None):
        entries = self.entries(count)
        output = self.workspace / "runs" / output_name
        fake, calls, instances = self.fake_client(barrier=barrier, error=error)
        with patch("close_gripper.vision.QwenClient", fake), \
                patch("close_gripper.dataset.extract_frames", return_value=[]), \
                patch.object(batch, "finish_run") as finish, \
                patch("socket.create_connection", side_effect=AssertionError("offline test attempted network")), \
                contextlib.redirect_stdout(io.StringIO()):
            summary = batch.run_batch(self.manifest, output, {"max_requests": 999, "max_retries": 2, "candidate_ids": ["ignored"], "candidate_starts": [0]}, workers=workers, call_model=True)
        return entries, output, summary, calls, instances, finish

    def test_load_case_validates_hash_and_exact_bilateral_candidate_identity(self):
        entry = self.entries(1)[0]
        ep, candidate = batch.load_case(entry, self.manifest_dir, pipeline.default_config())
        self.assertEqual(candidate["candidate_id"], entry["candidate_id"])
        self.assertEqual(candidate["side"], "right")
        all_candidates = detect_candidates(ep)
        self.assertEqual(all_candidates[0]["side"], "left")
        self.assertEqual(all_candidates[0]["start"], candidate["start"])
        for mutation in ({"episode_sha256": "stale"}, {"episode_id": "different"},
                         {"candidate_id": "missing-candidate"}, {"side": "left"}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                batch.load_case(dict(entry, **mutation), self.manifest_dir, pipeline.default_config())
        ep["state"][0][9] = .123
        pipeline.dump(self.manifest_dir / entry["episode_cache"], ep)
        with self.assertRaises(ValueError):
            batch.load_case(entry, self.manifest_dir, pipeline.default_config())

    def test_load_case_refuses_cache_escape_and_symlink_escape(self):
        entry = self.entries(1)[0]
        outside = self.root / "outside.json"
        outside.write_bytes((self.manifest_dir / entry["episode_cache"]).read_bytes())
        symlink = self.manifest_dir / "episodes" / "escape.json"
        symlink.symlink_to(outside)
        for cache in ("../outside.json", str(outside), "episodes/escape.json"):
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                batch.load_case(dict(entry, episode_cache=cache), self.manifest_dir, pipeline.default_config())

    def test_four_workers_each_make_one_selected_request_with_no_retry(self):
        entries, output, summary, calls, instances, finish = self.run_fixture(barrier=threading.Barrier(4))
        self.assertEqual(summary["cases"], 4)
        self.assertEqual(summary["actual_requests"], 4)
        self.assertEqual(summary["statuses"], {"model_ok": 4})
        self.assertEqual(summary["categories"], {"uncertain": 4})
        self.assertEqual(summary["accepted_changed_frames"], 0)
        self.assertEqual(summary["reported_token_usage"], {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52})
        self.assertEqual(set(calls), {(e["episode_id"], e["candidate_id"], e["side"]) for e in entries})
        self.assertEqual(len(instances), 4)
        self.assertEqual(len({id(instance) for instance in instances}), 4)
        self.assertTrue(all(c.request_count == 1 and c.config["max_requests"] == 1 and c.config["max_retries"] == 0 for c in instances))
        self.assertEqual(finish.call_count, 4)
        self.assertTrue(all(call.kwargs["quiet"] for call in finish.call_args_list))
        rows = json.loads((output / "case_results.json").read_text())
        self.assertEqual([row["index"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(len(list(output.glob("cases/*/case_result.json"))), 4)
        self.assertTrue((output / "batch_review.html").exists())
        self.assertEqual(json.loads((output / "progress.json").read_text())["completed"], 4)
        for target in output.glob("cases/*/*/targets.json"):
            self.assertFalse(any(any(flags) for flags in json.loads(target.read_text())["annotation.gripper_target_valid"]))
        for entry in entries:
            batch.load_case(entry, self.manifest_dir, pipeline.default_config())  # Source snapshots remain unchanged.

    def test_parse_error_usage_is_counted_without_false_uncertain_category(self):
        _, output, summary, _, _, _ = self.run_fixture(count=1, workers=1, error="missing_before_after_evidence")
        self.assertEqual(summary["actual_requests"], 1)
        self.assertEqual(summary["statuses"], {"model_error": 1})
        self.assertEqual(summary["categories"], {})
        self.assertEqual(summary["reported_token_usage"]["total_tokens"], 13)
        self.assertEqual(summary["server_responses"], 1)
        row = json.loads((output / "case_results.json").read_text())[0]
        self.assertIsNone(row["category"])
        self.assertEqual(row["reason"], "missing_before_after_evidence")

    def test_authentication_failure_stops_queued_cases_without_extra_requests(self):
        _, output, summary, calls, instances, _ = self.run_fixture(count=5, workers=1, error="http_401")
        self.assertEqual(summary["cases"], 5)
        self.assertEqual(summary["actual_requests"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(instances), 1)
        self.assertEqual(summary["statuses"], {"model_error": 1, "no_model": 4})
        rows = json.loads((output / "case_results.json").read_text())
        self.assertTrue(all(row["reason"] == "batch_authentication_blocked" for row in rows[1:]))

    def test_offline_batch_never_constructs_client_or_extracts_images(self):
        self.entries(2)
        output = self.workspace / "runs" / "offline"
        with patch("close_gripper.vision.QwenClient", side_effect=AssertionError("no client in offline mode")), \
                patch("close_gripper.dataset.extract_frames", side_effect=AssertionError("no video in offline mode")), \
                patch.object(batch, "finish_run"), contextlib.redirect_stdout(io.StringIO()):
            result = batch.run_batch(self.manifest, output, workers=2, call_model=False)
        self.assertEqual(result["actual_requests"], 0)
        self.assertEqual(result["statuses"], {"no_model": 2})
        self.assertEqual(result["categories"], {})
        self.assertEqual(json.loads((output / "execution.json").read_text())["max_network_requests"], 0)

    def test_bad_snapshot_is_an_explicit_case_error_without_a_request(self):
        entries = self.entries(1)
        entries[0]["episode_sha256"] = "stale"
        pipeline.dump(self.manifest, entries)
        output = self.workspace / "runs" / "bad-snapshot"
        with patch("close_gripper.vision.QwenClient", side_effect=AssertionError("snapshot rejected before client")), contextlib.redirect_stdout(io.StringIO()):
            result = batch.run_batch(self.manifest, output, workers=1, call_model=True)
        self.assertEqual(result["statuses"], {"error": 1})
        self.assertEqual(result["actual_requests"], 0)
        self.assertEqual(result["reasons"], {"case_error_ValueError": 1})

    def test_duplicate_episodes_empty_manifest_and_invalid_workers_are_rejected(self):
        entries = self.entries(1)
        for workers in (0, 5, True, 2.5, None):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                batch.run_batch(self.manifest, self.workspace / "runs" / "invalid", workers=workers)
        for content in ([], {}, entries * 2):
            pipeline.dump(self.manifest, content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                batch.run_batch(self.manifest, self.workspace / "runs" / "invalid")
        self.assertFalse((self.workspace / "runs" / "invalid").exists())


class AggregateTests(unittest.TestCase):
    def row(self, **changes):
        row = dict(status="model_ok", side="right", task=["cup"], release="r", subset="s",
                   category="uncertain", reason="uncertain", request_attempts=1, server_response=True,
                   usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                   preview_changed_frames=0, accepted_changed_frames=0, detected_candidates=5, input_images=41)
        row.update(changes)
        return row

    def test_totals_separate_parse_success_preview_and_valid_targets(self):
        rows = [self.row(category="normal_release", reason="release_policy_pending", preview_changed_frames=12),
                self.row(status="model_error", category="uncertain", reason="bad_json", side="left"),
                self.row(status="error", category=None, request_attempts=0, server_response=False, usage={}, input_images=0, detected_candidates=0)]
        summary = batch.aggregate_rows(rows)
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["categories"], {"normal_release": 1})
        self.assertEqual(summary["actual_requests"], 2)
        self.assertEqual(summary["reported_token_usage"]["total_tokens"], 26)
        self.assertEqual(summary["preview_cases"], 1)
        self.assertEqual(summary["preview_changed_frames"], 12)
        self.assertEqual(summary["accepted_changed_frames"], 0)
        self.assertEqual(summary["total_detected_candidates"], 10)
        self.assertFalse(summary["human_reference"])
        self.assertEqual(summary["evaluation_scope"], "one_selected_opening_per_episode")

    def test_missing_or_malformed_usage_does_not_break_batch_reporting(self):
        rows = [self.row(usage=None), self.row(usage="unknown"), self.row(usage={}),
                self.row(usage={"prompt_tokens": -2, "completion_tokens": True, "total_tokens": "3"})]
        summary = batch.aggregate_rows(rows)
        self.assertEqual(summary["reported_token_usage"], {})
        self.assertEqual(summary["actual_requests"], 4)


if __name__ == "__main__":
    unittest.main()
