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


if __name__ == "__main__":
    unittest.main()
