"""Calibration-builder regressions using synthetic, self-contained evidence."""

import copy
import importlib.util
from pathlib import Path
import unittest


def review(tag="positive", center=.2, free_opening=.9):
    return dict(
        case_id=tag, sample_id="sample-" + tag, task="夹取测试块", side="right",
        source_id="source-" + tag, dataset_root="/synthetic/calibration", episode_index=0,
        plateau_start=2, plateau_end=8, duration_s=.6,
        minimum=center, center=center, maximum=center,
        review_source="model", review_method="codex_native_vision", confirmed=False,
        visual_grasp_status="supported", closed_mode_support=True, open_mode_support=True,
        holding_evidence_frames=[2, 8], open_evidence_frames=[0],
        panels=[dict(index=0, opening=free_opening)] +
               [dict(index=i, opening=center) for i in (2, 5, 8)],
    )


def mapping(record, mode_id="grip"):
    return dict(
        sample_id=record["sample_id"],
        **{key: record[key] for key in ("task", "side", "source_id", "dataset_root", "episode_index")},
        v1_start=record["plateau_start"], v1_end=record["plateau_end"],
        v2_overlapping_windows=[dict(contains_v1_window=True, numeric_role="lower", mode_id=mode_id)],
    )


def fixture():
    record = review()
    stat = dict(task=record["task"], side="right", episode_count=3, unique_source_count=3,
                frame_count=27, stable_0_6s_count=3)
    membership = dict(task=record["task"], side="right", roles={
        "lower": {"modes": [dict(id="grip", lower=.18, upper=.22)]},
        "upper": {"modes": [dict(id="upper", lower=.5, upper=1.)]},
    })
    return stat, membership, [record], {record["sample_id"]: mapping(record)}


def catalog_fixture():
    stat, membership, records, mappings = fixture()
    task = stat["task"]
    return dict(
        statistics=dict(parameters=dict(minimum_hold_s=.6, maximum_sample_gap_s=.15,
                                        plateau_peak_to_peak=.02), profiles=[stat]),
        membership=dict(profiles=[membership]), mappings=dict(samples=list(mappings.values())),
        scope=dict(task_side_counts=[dict(task=task, side="right", episodes=3)],
                   sources=[dict(episodes=3, frames=27)],
                   aliases={task: ["Pick up the test block."]}, ambiguous_aliases={}),
        reviews=dict(records=records), version="synthetic-builder-fixture-v1",
    )


def mode_review(tag="a", role="lower", center=.164, free=.201):
    values = [center] * 9 + [center + .015, center + .035, free]
    indices = [0, 4, 8, 11]
    return dict(case_id="case-" + tag, sample_id="sample-" + tag, task="夹取测试块", side="right",
        source_id="source-" + tag, dataset_root="/synthetic/mode", episode_index=0,
        numeric_role=role, numeric_mode_id="grip", plateau_start=0, plateau_end=8,
        minimum=center, center=center, maximum=center, duration_s=.8,
        review_source="model", review_method="codex_native_vision", confirmed=False,
        panels=[dict(index=i, timestamp=i / 10, opening=values[i], camera="cam_right_wrist_left") for i in indices],
        frame_states=[dict(index=i, state="free" if i == 11 else "held", opening=values[i]) for i in indices],
        release_pairs=[dict(held_frame=8, free_frame=11)],
        trace=dict(timestamps=[i / 10 for i in range(len(values))], openings=values))


def mode_fixture(records=None):
    args = catalog_fixture()
    records = records or [mode_review("a"), mode_review("b", role="mixed")]
    args["membership"]["profiles"][0]["roles"] = {
        role: dict(modes=[dict(id="grip", lower=.1, upper=.3)])
        for role in ("lower", "upper", "static", "mixed")}
    keys = ("sample_id", "task", "side", "source_id", "dataset_root", "episode_index", "numeric_role",
            "numeric_mode_id", "plateau_start", "plateau_end", "minimum", "center", "maximum", "duration_s")
    args["mappings"] = dict(samples=[{k: r[k] for k in keys} for r in records])
    args["reviews"] = dict(schema="close_gripper.mode_reviews.v1", records=records)
    return args


class BuildTaskProfilesContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/build_task_profiles.py"
        spec = importlib.util.spec_from_file_location("build_task_profiles_contract", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_same_opening_free_or_mixed_evidence_blocks_positive_calibration(self):
        for kind in ("explicit_free", "mixed", "not_supported"):
            stat, modes, records, mappings = fixture()
            counterexample = review("counterexample")
            counterexample["open_mode_support"] = False
            counterexample["open_evidence_frames"] = []
            if kind == "mixed":
                counterexample["mixed_hold_free"] = True
            else:
                counterexample.update(closed_mode_support=False, holding_evidence_frames=[],
                                      visual_grasp_status="uncertain" if kind == "explicit_free" else "not_supported")
                if kind == "explicit_free":
                    counterexample["explicit_free_within_plateau"] = [5]
            records.append(counterexample)
            mappings[counterexample["sample_id"]] = mapping(counterexample)
            with self.subTest(kind=kind):
                result = self.module.build_profile(stat, modes, records, mappings)
                self.assertFalse(result["automatic_use"])
                self.assertEqual(result["closed_bands"], [])
                self.assertEqual(result["rejected_closed_bands"][0]["counterexample_case_ids"],
                                 ["counterexample"])
                self.assertIn("conflicting_closed_bands_excluded", result["reasons"])

    def test_empty_closed_free_frame_is_evidence_but_not_an_open_anchor(self):
        stat, modes, records, mappings = fixture()
        records[0]["panels"][0]["opening"] = 0.
        result = self.module.build_profile(stat, modes, records, mappings)
        self.assertEqual(len(result["closed_bands"]), 1)
        self.assertEqual(result["counts"]["observed_free_frames"], 1)
        self.assertEqual(result["open_bands"], [])
        self.assertFalse(result["automatic_use"])
        self.assertIn("no_separate_higher_visual_open_anchor", result["reasons"])

    def test_single_open_frame_stays_local_instead_of_selecting_upper_mode(self):
        stat, modes, records, mappings = fixture()
        records[0]["panels"][0]["opening"] = .6
        result = self.module.build_profile(stat, modes, records, mappings)
        self.assertTrue(result["automatic_use"])
        self.assertEqual(len(result["open_bands"]), 1)
        opened = result["open_bands"][0]
        self.assertAlmostEqual(opened["lower"], .59)
        self.assertAlmostEqual(opened["center"], .6)
        self.assertAlmostEqual(opened["upper"], .61)
        self.assertEqual(opened["numeric_mode_ids"], [])
        self.assertIn("upper:upper", result["uncalibrated_numeric_mode_ids"])

    def test_partial_window_cannot_supply_a_nearby_different_mode(self):
        stat, modes, records, mappings = fixture()
        modes["roles"]["lower"]["modes"].append(dict(id="adjacent", lower=.4, upper=.6))
        windows = mappings[records[0]["sample_id"]]["v2_overlapping_windows"]
        windows.insert(0, dict(contains_v1_window=False, numeric_role="lower", mode_id="adjacent"))
        result = self.module.build_profile(stat, modes, records, mappings)
        self.assertEqual(result["closed_bands"][0]["numeric_mode_ids"], ["lower:grip"])
        self.assertEqual((result["closed_bands"][0]["lower"], result["closed_bands"][0]["upper"]),
                         (.18, .22))
        windows[1]["contains_v1_window"] = False
        with self.assertRaisesRegex(ValueError, "no containing"):
            self.module.build_profile(stat, modes, records, mappings)

    def test_population_mode_assignment_requires_matching_source_and_task_identity(self):
        changes = dict(task="另一任务", side="left", source_id="another-source",
                       dataset_root="/another/source", episode_index=99)
        for key, value in changes.items():
            stat, modes, records, mappings = fixture()
            mappings[records[0]["sample_id"]][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity disagree"):
                self.module.build_profile(stat, modes, records, mappings)
        stat, modes, records, mappings = fixture()
        mappings[records[0]["sample_id"]]["v1_start"] = 3
        with self.assertRaisesRegex(ValueError, "interval disagree"):
            self.module.build_profile(stat, modes, records, mappings)

    def test_merging_overlapping_bands_preserves_an_unobserved_gap(self):
        stat, modes, _, _ = fixture()
        modes["roles"]["lower"]["modes"] = [
            dict(id="first", lower=.18, upper=.22),
            dict(id="overlap", lower=.21, upper=.25),
            dict(id="separate", lower=.30, upper=.32),
        ]
        records = [review("first", .2), review("overlap", .23), review("separate", .31)]
        mappings = {r["sample_id"]: mapping(r, r["case_id"]) for r in records}
        result = self.module.build_profile(stat, modes, records, mappings)
        self.assertTrue(result["automatic_use"])
        self.assertEqual([(b["lower"], b["upper"]) for b in result["closed_bands"]],
                         [(.18, .25), (.30, .32)])
        self.assertEqual(result["closed_bands"][0]["case_ids"], ["first", "overlap"])
        self.assertFalse(any(b["lower"] <= .27 <= b["upper"] for b in result["closed_bands"]))

    def test_catalog_rejects_missing_task_side_coverage_from_any_input(self):
        cases = [("scope", "task_side_counts"), ("statistics", "profiles"),
                 ("membership", "profiles"), ("reviews", "records")]
        for component, key in cases:
            args = catalog_fixture()
            args[component][key] = []
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, "coverage disagree"):
                self.module.build_catalog(**args)
        args = catalog_fixture()
        args["scope"]["task_side_counts"].append(dict(task="遗漏的任务", side="right", episodes=1))
        with self.assertRaisesRegex(ValueError, "coverage disagree"):
            self.module.build_catalog(**args)

    def test_catalog_keeps_frozen_policy_and_does_not_mutate_evidence(self):
        args = catalog_fixture()
        original = copy.deepcopy(args)
        result = self.module.build_catalog(**args)
        self.assertEqual(args, original)
        self.assertEqual(result["minimum_hold_s"], .6)
        self.assertEqual(result["maximum_sample_gap_s"], .15)
        self.assertFalse(result["generated_targets"])
        self.assertFalse(result["fitting_method_changed"])
        self.assertTrue(result["tasks"][0]["sides"]["right"]["automatic_use"])
        args["statistics"]["parameters"]["minimum_hold_s"] = .2
        with self.assertRaisesRegex(ValueError, "stability parameters"):
            self.module.build_catalog(**args)

    def test_all_mode_closed_band_uses_observed_values_and_distinct_sources_across_roles(self):
        args = mode_fixture()
        before = copy.deepcopy(args)
        catalog = self.module.build_catalog(**args)
        result = catalog["tasks"][0]["sides"]["right"]
        band, = result["closed_bands"]
        self.assertEqual(args, before)
        self.assertTrue(result["automatic_use"])
        self.assertEqual(band["reviewed_source_count"], 2)
        self.assertEqual(band["numeric_mode_ids"], ["lower:grip", "mixed:grip"])
        self.assertAlmostEqual(band["lower"], .161)
        self.assertAlmostEqual(band["upper"], .167)
        self.assertEqual(catalog["calibration_contract"], "all_modes_semantic_review_v2")
        self.assertFalse(result["provenance"]["full_dataset_visually_reviewed"])
        self.assertEqual(result["counts"]["reviewed_numeric_modes"], 2)
        # A second case from the original recording is not independent support.
        args["reviews"]["records"][1]["source_id"] = "source-a"
        args["mappings"]["samples"][1]["source_id"] = "source-a"
        rejected = self.module.build_catalog(**args)["tasks"][0]["sides"]["right"]
        self.assertFalse(rejected["automatic_use"])
        self.assertIn("insufficient_independent_holding_sources", rejected["reasons"])

    def test_mixed_platform_separates_held_subspan_from_small_normal_release(self):
        r = mode_review("mixed")
        r.update(plateau_end=9, duration_s=.9, maximum=.179, held_intervals=[dict(start=0, end=8)])
        r["panels"].append(dict(index=9, timestamp=.9, opening=.179))
        r["frame_states"].append(dict(index=9, state="free", opening=.179))
        args = mode_fixture([mode_review(), r])
        result = self.module.build_catalog(**args)["tasks"][0]["sides"]["right"]
        self.assertTrue(result["automatic_use"])
        self.assertEqual(result["interval_vision_bands"], [])
        self.assertEqual(result["semantic_evidence"][1]["interval_semantics"], "mixed")
        self.assertAlmostEqual(result["closed_bands"][0]["upper"], .167)
        self.assertTrue(any(o["lower"] <= .179 <= o["upper"] for o in result["open_bands"]))

    def test_two_sources_only_enable_their_pointwise_overlap_not_single_source_edges(self):
        records = [mode_review("a", center=.160), mode_review("b", center=.164)]
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        band, = profile["closed_bands"]
        self.assertAlmostEqual(band["lower"], .161)
        self.assertAlmostEqual(band["upper"], .163)
        self.assertEqual(band["minimum_pointwise_source_count"], 2)
        self.assertEqual(profile["positive_source_support_policy"], "every_value_covered_by_two_distinct_sources")
        edges = [b for b in profile["rejected_closed_bands"]
                 if b["rejection_reason"] == "insufficient_independent_holding_sources"]
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(b["reviewed_source_count"] == 1 for b in edges))
        self.assertTrue(all(not self.module.overlaps(band, edge) for edge in edges))
        # Two overlapping intervals of the same recording provide only one source.
        records[1]["source_id"] = records[0]["source_id"]
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        self.assertFalse(profile["automatic_use"])
        self.assertEqual(profile["closed_bands"], [])

    def test_source_alignment_uncertain_preserves_visual_review_but_excludes_all_calibration(self):
        records = [mode_review("a"), mode_review("b", role="static"), mode_review("duplicate", role="mixed")]
        records[1]["source_alignment_uncertain"] = True
        records[2]["source_id"] = records[1]["source_id"]
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        self.assertFalse(profile["automatic_use"])
        self.assertEqual(profile["closed_bands"], [])
        self.assertEqual(profile["release_rules"], [])
        self.assertEqual(profile["counts"]["excluded_alignment_cases"], 2)
        self.assertEqual(profile["counts"]["numeric_calibration_sources"], 1)
        self.assertEqual(profile["semantic_evidence"][1]["interval_semantics"], "held")
        self.assertFalse(profile["semantic_evidence"][2]["numeric_calibration_eligible"])
        self.assertTrue(all("source-b" not in b["source_ids"] for b in profile["free_bands"]))
        self.assertEqual({m["state"] for m in profile["mode_reviews"] if m["excluded_case_ids"]},
                         {"numeric_source_uncertain"})
        # Misaligned free evidence must neither create a false conflict nor an open anchor.
        records = [mode_review("a"), mode_review("b"), mode_review("bad", center=.164, free=.8)]
        records[2]["source_alignment_uncertain"] = True
        records[2]["release_pairs"] = []
        for frame in records[2]["frame_states"]:
            frame["state"] = "free"
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        self.assertTrue(profile["automatic_use"])
        self.assertEqual(profile["interval_vision_bands"], [])
        self.assertTrue(all("source-bad" not in b["source_ids"] for b in profile["open_bands"]))

    def test_actual_same_value_free_observation_requires_interval_vision(self):
        r = mode_review("free")
        for frame in r["frame_states"]:
            frame["state"] = "free"
        r["release_pairs"] = []
        result = self.module.build_catalog(**mode_fixture([mode_review(), mode_review("b"), r]))
        profile = result["tasks"][0]["sides"]["right"]
        self.assertFalse(profile["automatic_use"])
        self.assertEqual(profile["closed_bands"], [])
        self.assertEqual(profile["rejected_closed_bands"][0]["rejection_reason"], "requires_interval_vision")
        self.assertEqual(profile["interval_vision_bands"][0]["counterexample_case_ids"], ["case-free"])

    def test_partial_free_conflict_preserves_independently_supported_held_remainder(self):
        free = mode_review("free", center=.155, free=.4)
        free["release_pairs"] = []
        for frame in free["frame_states"]:
            frame["state"] = "free"
        records = [mode_review("low", center=.158), mode_review("high-a", center=.164),
                   mode_review("high-b", center=.164), free]
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        conflict, = profile["interval_vision_bands"]
        self.assertAlmostEqual(conflict["lower"], .155)
        self.assertAlmostEqual(conflict["upper"], .158)
        band, = profile["closed_bands"]
        self.assertGreater(band["lower"], conflict["upper"])
        self.assertLessEqual(band["lower"], .164)
        self.assertGreaterEqual(band["upper"], .164)
        self.assertLessEqual(band["lower"], band["center"])
        self.assertLessEqual(band["center"], band["upper"])
        self.assertFalse(any(self.module.overlaps(band, b) for b in profile["free_bands"]))
        self.assertTrue(any(r["closed_band_id"] == band["id"] for r in profile["release_rules"]))
        rejected = [b for b in profile["rejected_closed_bands"] if b["rejection_reason"] == "requires_interval_vision"]
        self.assertTrue(rejected)
        self.assertTrue(all(b["upper"] <= .158 for b in rejected))

    def test_conflict_subtraction_recounts_sources_instead_of_inheriting_parent_support(self):
        free = mode_review("free", center=.155, free=.4)
        free["release_pairs"] = []
        for frame in free["frame_states"]:
            frame["state"] = "free"
        # The two low recordings end at .158. The higher source intersects
        # those original bands, but is alone after the free conflict is cut.
        records = [mode_review("low-a", center=.155), mode_review("low-b", center=.155),
                   mode_review("high", center=.160), free]
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        self.assertFalse(profile["automatic_use"])
        remainder = [b for b in profile["rejected_closed_bands"]
                     if b["rejection_reason"] == "insufficient_independent_holding_sources"]
        self.assertEqual(len(remainder), 1)
        self.assertEqual(remainder[0]["source_ids"], ["source-high"])
        self.assertEqual(remainder[0]["case_ids"], ["case-high"])
        self.assertEqual(remainder[0]["reviewed_source_count"], 1)
        self.assertLessEqual(remainder[0]["lower"], remainder[0]["center"])
        self.assertLessEqual(remainder[0]["center"], remainder[0]["upper"])

    def test_mode_state_uses_platform_semantics_without_context_contamination(self):
        profile = self.module.build_catalog(**mode_fixture())["tasks"][0]["sides"]["right"]
        self.assertTrue(all(m["state"] == "held_supported" for m in profile["mode_reviews"]
                            if m["reviewed_case_ids"]))
        # This free upper platform has a held pre-frame at a different opening.
        upper = mode_review("upper", role="upper")
        values = [.164, .179, .199] + [.201] * 9
        upper.update(plateau_start=3, plateau_end=11, minimum=.201, center=.201, maximum=.201,
                     panels=[dict(index=i, opening=values[i]) for i in (0, 3, 7, 11)],
                     frame_states=[dict(index=i, state="held" if i == 0 else "free") for i in (0, 3, 7, 11)],
                     release_pairs=[dict(held_frame=0, free_frame=3)],
                     trace=dict(timestamps=[i / 10 for i in range(12)], openings=values))
        free = mode_review("free", role="static")
        free["release_pairs"] = []
        for frame in free["frame_states"]:
            frame["state"] = "free"
        profile = self.module.build_catalog(**mode_fixture([mode_review(), mode_review("b"), upper, free]))["tasks"][0]["sides"]["right"]
        self.assertTrue(profile["interval_vision_bands"])
        self.assertEqual(next(m["state"] for m in profile["mode_reviews"] if m["numeric_role"] == "upper"),
                         "free_supported")

    def test_uncertain_boundary_cannot_be_bridged_by_a_declared_held_interval(self):
        r = mode_review("uncertain")
        r["frame_states"][1]["state"] = "uncertain"
        r["held_intervals"] = [dict(start=0, end=8)]
        with self.assertRaisesRegex(ValueError, "conflicting or uncertain"):
            self.module.build_catalog(**mode_fixture([mode_review(), r]))

    def test_small_release_rule_requires_same_episode_visual_pair_and_preserves_noise(self):
        result = self.module.build_catalog(**mode_fixture())["tasks"][0]["sides"]["right"]
        rule, = result["release_rules"]
        self.assertAlmostEqual(rule["motion_min_delta"], .032)
        self.assertEqual(rule["motion_noise"], .003)
        self.assertEqual(rule["reviewed_source_count"], 2)
        self.assertTrue(rule["preserves_fixed_timing"])
        self.assertEqual({(e["motion_start"], e["motion_end"]) for e in rule["evidence"]}, {(8, 10)})
        args = mode_fixture()
        for r in args["reviews"]["records"]:
            r["release_pairs"] = []
        result = self.module.build_catalog(**args)["tasks"][0]["sides"]["right"]
        self.assertTrue(result["automatic_use"])
        self.assertEqual(result["release_rules"], [])
        args["reviews"]["records"][0]["release_pairs"] = [dict(held_frame=8, free_frame=4)]
        with self.assertRaisesRegex(ValueError, "ordered, viewed"):
            self.module.build_catalog(**args)

    def test_mode_review_requires_exact_identity_and_actual_frame_opening(self):
        for change in ("source", "value", "unviewed", "trace"):
            args = mode_fixture()
            r = args["reviews"]["records"][0]
            if change == "source":
                args["mappings"]["samples"][0]["source_id"] = "another-source"
            elif change == "value":
                r["frame_states"][0]["opening"] = .2
            elif change == "unviewed":
                r["frame_states"].pop(1)
            else:
                r["trace"]["openings"][3] = .5
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.module.build_catalog(**args)

    def test_single_free_frame_does_not_inflate_to_population_upper_mode(self):
        args = mode_fixture()
        profile = self.module.build_catalog(**args)["tasks"][0]["sides"]["right"]
        opened, = profile["open_bands"]
        self.assertAlmostEqual(opened["lower"], .198)
        self.assertAlmostEqual(opened["upper"], .204)
        self.assertEqual(profile["closed_bands"][0]["open_band_ids"], [opened["id"]])

    def test_no_mode_assignment_can_still_supply_initial_holding_evidence(self):
        records = [mode_review("a", role="static"), mode_review("b", role="static")]
        for r in records:
            r["numeric_mode_id"] = None
        profile = self.module.build_catalog(**mode_fixture(records))["tasks"][0]["sides"]["right"]
        self.assertTrue(profile["automatic_use"])
        self.assertEqual(profile["counts"]["reviewed_unassigned_cases"], 2)
        self.assertEqual(profile["closed_bands"][0]["numeric_mode_ids"], [])


if __name__ == "__main__":
    unittest.main()
