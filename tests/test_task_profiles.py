"""Offline profile contracts; fixtures do not claim real visual calibration."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from close_gripper.task_profiles import (
    SCHEMA, UNITS, match_hold_interval, resolve_profile, validate_catalog,
)


def band(identifier, low, high, center=None):
    return dict(id=identifier, lower=low, center=(low + high) / 2 if center is None else center, upper=high)


def catalog():
    return dict(schema=SCHEMA, version="offline_fixture_v1", units=UNITS,
                minimum_hold_s=.6, maximum_sample_gap_s=.15, maximum_plateau_range=.02,
                tasks=[dict(task="拿起杯子", aliases=["Pick up the cup."], sides={
                    "right": dict(status="visual_reviewed", automatic_use=True,
                        closed_bands=[band("cup", .18, .22)], open_bands=[band("open", .8, 1.)],
                        counts={"episodes": 20, "visual_examples": 2},
                        provenance={"review_source": "model", "evidence_type": "synthetic_fixture"})})])


def episode(values=None, times=None):
    values = [.2] * 8 if values is None else values
    states = [[0.] * 20 for _ in values]
    for row, value in zip(states, values):
        row[19] = value
    return dict(episode_id="fixture", timestamps=[i / 10 for i in range(len(values))] if times is None else times,
                state=states, valid_sides=["right"], cameras={},
                identity={"source_metadata": {"source_tasks": ["拿起杯子"]}, "tasks": ["Pick up the cup."]})


def all_mode_catalog():
    c = catalog()
    c.update(calibration_contract="all_modes_semantic_review_v2", minimum_positive_sources=2, evidence_noise=.003)
    p = c["tasks"][0]["sides"]["right"]
    p.update(calibration_contract=c["calibration_contract"], minimum_positive_sources=2, evidence_noise=.003,
             interval_vision_bands=[], free_bands=[band("free", .8, 1.)], release_rules=[])
    p["closed_bands"][0].update(source_ids=["a", "b"], reviewed_source_count=2, open_band_ids=["open"])
    p["open_bands"][0].update(source_ids=["a"])
    return c


class TaskProfileContracts(unittest.TestCase):
    def test_exact_time_threshold_uses_seven_timestamps_and_backfills_start(self):
        ep = episode([.8] * 3 + [.2] * 8)
        r = match_hold_interval(catalog(), ep, "right", 3, 9)
        self.assertTrue(r["matched"])
        self.assertEqual((r["hold_start"], r["hold_end"], r["confirmation_frame"]), (3, 9, 9))
        self.assertAlmostEqual(r["duration_s"], .6)
        self.assertFalse(r["grasp_semantics_verified"])
        self.assertFalse(r["generates_targets"])

    def test_six_frames_are_only_half_a_second(self):
        r = match_hold_interval(catalog(), episode(), "right", 0, 5)
        self.assertEqual(r["reason"], "insufficient_holding_duration")
        self.assertFalse(r["matched"])

    def test_late_confirmation_does_not_move_the_interval_start(self):
        ep = episode([.2] * 12)
        r = match_hold_interval(catalog(), ep, "right", 1, 11)
        self.assertEqual(r["hold_start"], 1)
        self.assertEqual(r["confirmation_frame"], 7)
        self.assertAlmostEqual(r["confirmation_time"], .7)

    def test_missing_sample_cannot_be_counted_as_observed_hold_time(self):
        ep = episode([.2] * 7, [0, .1, .2, .4, .5, .6, .7])
        r = match_hold_interval(catalog(), ep, "right", 0, 6)
        self.assertEqual(r["reason"], "timestamp_gap")

    def test_irregular_but_contiguous_times_are_not_replaced_with_frame_count(self):
        ep = episode([.2] * 6, [0, .12, .24, .36, .48, .6])
        r = match_hold_interval(catalog(), ep, "right", 0, 5)
        self.assertTrue(r["matched"])
        self.assertEqual(r["confirmation_frame"], 5)

    def test_slow_traversal_of_a_wide_band_is_not_a_stable_plateau(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [band("wide", .1, .4)]
        r = match_hold_interval(c, episode([.1, .13, .16, .19, .22, .25, .28]), "right", 0, 6)
        self.assertEqual(r["reason"], "unstable_plateau")

    def test_disjoint_closed_modes_do_not_bridge_the_gap(self):
        c = catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"] = [band("thin", .1, .12), band("thick", .2, .22)]
        r = match_hold_interval(c, episode([.16] * 7), "right", 0, 6)
        self.assertEqual(r["reason"], "outside_single_closed_band")

    def test_interval_cannot_switch_between_two_nearby_modes(self):
        c = catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"] = [band("a", .19, .20), band("b", .21, .22)]
        r = match_hold_interval(c, episode([.20, .20, .20, .21, .21, .21, .21]), "right", 0, 6)
        self.assertEqual(r["reason"], "outside_single_closed_band")

    def test_each_supported_mode_can_match_independently(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [band("thin", .1, .12), band("thick", .2, .22)]
        for value, mode in ((.11, "thin"), (.21, "thick")):
            with self.subTest(value=value):
                r = match_hold_interval(c, episode([value] * 7), "right", 0, 6)
                self.assertEqual(r["closed_band_id"], mode)

    def test_invalid_side_placeholders_never_match_even_zero_closed_band(self):
        c = catalog()
        right = c["tasks"][0]["sides"]["right"]
        c["tasks"][0]["sides"]["left"] = copy.deepcopy(right)
        c["tasks"][0]["sides"]["left"]["closed_bands"] = [band("zero", 0, .01)]
        r = match_hold_interval(c, episode(), "left", 0, 6)
        self.assertEqual(r["reason"], "invalid_side")

    def test_valid_near_zero_side_is_allowed(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [band("zero", 0, .01)]
        r = match_hold_interval(c, episode([0.] * 7), "right", 0, 6)
        self.assertTrue(r["matched"])

    def test_missing_side_profile_does_not_borrow_other_side(self):
        ep = episode()
        ep["valid_sides"] = ["left", "right"]
        r = resolve_profile(catalog(), ep, "left")
        self.assertEqual(r["reason"], "missing_side_profile")

    def test_alias_collision_is_rejected_before_any_task_is_resolved(self):
        c = catalog()
        other = copy.deepcopy(c["tasks"][0])
        other["task"] = "另一个任务"
        c["tasks"].append(other)
        with self.assertRaisesRegex(ValueError, "collision"):
            resolve_profile(c, episode(), "right")

    def test_source_task_is_authoritative_and_unknown_source_does_not_fallback(self):
        ep = episode()
        ep["identity"]["source_metadata"]["source_tasks"] = ["unknown"]
        r = resolve_profile(catalog(), ep, "right")
        self.assertEqual(r["reason"], "unknown_task")
        self.assertEqual(r["identity_source"], "source_metadata.source_tasks")

    def test_ambiguous_english_alias_is_explicitly_rejected_but_source_label_resolves(self):
        c = catalog()
        other = copy.deepcopy(c["tasks"][0])
        other.update(task="拿起另一个杯子", aliases=[])
        c["tasks"].append(other)
        c["ambiguous_aliases"] = {"Pick up the vessel.": ["拿起杯子", "拿起另一个杯子"]}
        ep = episode()
        ep["identity"]["tasks"] = ["Pick up the vessel."]
        self.assertEqual(resolve_profile(c, ep, "right")["task"], "拿起杯子")
        del ep["identity"]["source_metadata"]
        r = resolve_profile(c, ep, "right")
        self.assertEqual(r["reason"], "ambiguous_task_alias")
        self.assertEqual(r["candidate_tasks"], ["拿起杯子", "拿起另一个杯子"])

    def test_declared_ambiguous_alias_cannot_remain_resolvable(self):
        c = catalog()
        other = copy.deepcopy(c["tasks"][0])
        other.update(task="拿起另一个杯子", aliases=[])
        c["tasks"].append(other)
        c["ambiguous_aliases"] = {"Pick up the cup.": ["拿起杯子", "拿起另一个杯子"]}
        with self.assertRaisesRegex(ValueError, "must not also resolve"):
            validate_catalog(c)

    def test_catalogued_english_alias_is_available_without_source_metadata(self):
        ep = episode()
        del ep["identity"]["source_metadata"]
        r = resolve_profile(catalog(), ep, "right")
        self.assertEqual(r["task"], "拿起杯子")
        self.assertEqual(r["identity_source"], "tasks")
        ep["identity"]["tasks"] = ["pick up the cup"]
        self.assertEqual(resolve_profile(catalog(), ep, "right")["reason"], "unknown_task")

    def test_multiple_source_tasks_are_not_reduced_to_first(self):
        ep = episode()
        ep["identity"]["source_metadata"]["source_tasks"] = ["拿起杯子", "Pick up the cup."]
        self.assertEqual(resolve_profile(catalog(), ep, "right")["reason"], "multiple_task_identity")

    def test_present_empty_source_task_does_not_fallback_to_english(self):
        ep = episode()
        ep["identity"]["source_metadata"]["source_tasks"] = []
        self.assertEqual(resolve_profile(catalog(), ep, "right")["reason"], "invalid_task_identity")

    def test_malformed_source_metadata_never_falls_back_to_english(self):
        for value in (None, [], "invalid", True):
            ep = episode()
            ep["identity"]["source_metadata"] = value
            with self.subTest(value=value):
                self.assertEqual(resolve_profile(catalog(), ep, "right")["reason"], "invalid_source_metadata")

    def test_numeric_or_review_required_profile_is_inspectable_but_not_automatic(self):
        for status in ("numeric_only", "needs_review", "unsupported", "visual_reviewed"):
            c = catalog()
            p = c["tasks"][0]["sides"]["right"]
            p.update(status=status, automatic_use=False)
            with self.subTest(status=status):
                self.assertEqual(resolve_profile(c, episode(), "right")["status"], "resolved")
                self.assertEqual(match_hold_interval(c, episode(), "right", 0, 6)["reason"], "profile_not_automatic")

    def test_numeric_status_cannot_be_silently_promoted_to_automatic(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["status"] = "numeric_only"
        with self.assertRaisesRegex(ValueError, "visual_reviewed"):
            validate_catalog(c)

    def test_automatic_requires_explicit_visual_review_source(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["provenance"] = {"review_source": "numerical"}
        with self.assertRaisesRegex(ValueError, "review source"):
            validate_catalog(c)

    def test_ambiguous_open_closed_bands_must_stay_disabled(self):
        c = catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["open_bands"] = [band("ambiguous", .20, .9)]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_catalog(c)
        p.update(status="needs_review", automatic_use=False)
        validate_catalog(c)
        self.assertEqual(match_hold_interval(c, episode(), "right", 0, 6)["reason"], "profile_not_automatic")

    def test_overlapping_closed_modes_and_shared_endpoints_are_rejected(self):
        for lower in (.21, .22):
            c = catalog()
            c["tasks"][0]["sides"]["right"]["closed_bands"].append(band("second", lower, .3))
            with self.subTest(lower=lower), self.assertRaisesRegex(ValueError, "disjoint"):
                validate_catalog(c)

    def test_automatic_profile_needs_a_larger_observed_opening(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["open_bands"] = [band("wrong", .01, .1)]
        with self.assertRaisesRegex(ValueError, "higher observed"):
            validate_catalog(c)

    def test_invalid_catalog_numbers_are_rejected(self):
        for value in (True, None, "0.2", {}, [], float("nan"), float("inf"), 10 ** 400, -.1, 1.1):
            c = catalog()
            c["tasks"][0]["sides"]["right"]["closed_bands"][0]["center"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_catalog(c)
        for key in ("minimum_hold_s", "maximum_sample_gap_s", "maximum_plateau_range"):
            for value in (True, None, "0.6", {}, [], 0, -.1, float("nan"), float("inf"), 10 ** 400):
                c = catalog(); c[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_catalog(c)

    def test_invalid_episode_values_and_timestamps_are_rejected(self):
        for value in (True, None, "0.2", {}, [], float("nan"), float("inf"), 10 ** 400, -.1, 1.1):
            ep = episode(); ep["state"][2][19] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                match_hold_interval(catalog(), ep, "right", 0, 6)
        ep = episode(); ep["timestamps"][2] = ep["timestamps"][1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            match_hold_interval(catalog(), ep, "right", 0, 6)

    def test_non_numeric_or_overflowing_timestamps_are_explicitly_rejected(self):
        for value in (True, None, "0.2", {}, [], float("nan"), float("inf"), 10 ** 400):
            ep = episode()
            ep["timestamps"][2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                match_hold_interval(catalog(), ep, "right", 0, 6)

    def test_invalid_interval_bounds_do_not_silently_slice(self):
        for a, b in ((-1, 6), (3, 2), (0, 99), (True, 6), (0, 6.0), (None, 6), (0, "6"), ({}, 6)):
            with self.subTest(start=a, end=b), self.assertRaisesRegex(ValueError, "frame indices"):
                match_hold_interval(catalog(), episode(), "right", a, b)

    def test_inputs_are_unchanged_and_returned_profile_is_detached(self):
        c, ep = catalog(), episode()
        snapshot = copy.deepcopy((c, ep))
        match_hold_interval(c, ep, "right", 0, 6)
        result = resolve_profile(c, ep, "right")
        result["profile"]["closed_bands"][0]["center"] = .99
        self.assertEqual((c, ep), snapshot)

    def test_all_mode_source_denominators_cannot_count_duplicate_cases(self):
        validate_catalog(all_mode_catalog())
        for sources, count in ((["a"], 2), (["a", "a"], 2), (["a", "b"], 3)):
            c = all_mode_catalog()
            c["tasks"][0]["sides"]["right"]["closed_bands"][0].update(
                source_ids=sources, reviewed_source_count=count)
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, "independent holding"):
                validate_catalog(c)

    def test_all_mode_rejects_closed_free_overlap_and_unlinked_open_references(self):
        for kind in ("free_overlap", "interval_vision_overlap", "unknown_open", "unlinked_source"):
            c = all_mode_catalog()
            p = c["tasks"][0]["sides"]["right"]
            if kind == "free_overlap":
                p["free_bands"].append(band("empty_closed", .19, .21))
            elif kind == "interval_vision_overlap":
                p["interval_vision_bands"].append(band("ambiguous", .19, .21))
            elif kind == "unknown_open":
                p["closed_bands"][0]["open_band_ids"] = ["unknown"]
            else:
                p["open_bands"][0]["source_ids"] = ["unrelated"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_catalog(c)

    def test_same_value_semantic_conflict_is_an_explicit_match_rejection(self):
        c = all_mode_catalog()
        p = c["tasks"][0]["sides"]["right"]
        p.update(automatic_use=False, status="needs_review", closed_bands=[],
                 interval_vision_bands=[band("ambiguous", .19, .21)])
        r = match_hold_interval(c, episode(), "right", 0, 6)
        self.assertEqual(r["reason"], "requires_interval_vision")
        self.assertEqual(r["interval_vision_band_ids"], ["ambiguous"])
        self.assertFalse(r["matched"])

    def test_extensions_require_the_versioned_contract_without_changing_legacy_behavior(self):
        c = catalog()
        validate_catalog(c)
        c["tasks"][0]["sides"]["right"]["release_rules"] = []
        with self.assertRaisesRegex(ValueError, "all-mode contract"):
            validate_catalog(c)
        c = all_mode_catalog()
        c["minimum_hold_s"] = .2
        with self.assertRaisesRegex(ValueError, "frozen policy"):
            validate_catalog(c)

    def test_read_only_cli_validation_inspection_and_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, ep_path = root / "catalog.json", root / "episode.json"
            path.write_text(json.dumps(catalog()), encoding="utf-8")
            ep_path.write_text(json.dumps(episode()), encoding="utf-8")
            snapshots = {p.name: p.read_bytes() for p in root.iterdir()}
            cases = [(["validate", str(path)], 0, "valid"),
                     (["inspect", str(path), "--task", "拿起杯子", "--side", "right"], 0, "resolved"),
                     (["inspect", str(path), "--task", "unknown", "--side", "right"], 2, "rejected"),
                     (["match", str(path), "--episode", str(ep_path), "--side", "right", "--start", "0", "--end", "6"], 0, "matched")]
            for args, code, status in cases:
                with self.subTest(args=args):
                    proc = subprocess.run([sys.executable, "-B", "-m", "close_gripper.task_profiles", *args],
                                          capture_output=True, text=True, check=False)
                    self.assertEqual(proc.returncode, code, proc.stderr)
                    self.assertEqual(json.loads(proc.stdout)["status"], status)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, snapshots)


if __name__ == "__main__":
    unittest.main()
