"""Offline business-contract tests; these are synthetic evidence, never VLM accuracy."""

import copy
import unittest

from close_gripper.contracts import SIDE_CAMERAS, validate_decision, validate_episode
from close_gripper.core import detect_candidates, generate_targets


def episode(widths, side="right", other=None):
    states = [[0.0] * 20 for _ in widths]
    for i, width in enumerate(widths):
        states[i][9 if side == "left" else 19] = width / 95
        if other is not None:
            states[i][19 if side == "left" else 9] = other[i] / 95
    return {"episode_id": "fixture", "timestamps": [i / 10 for i in range(len(widths))],
            "state": states, "valid_sides": [side] if other is None else ["left", "right"],
            "cameras": {name: {"valid": True} for names in SIDE_CAMERAS.values() for name in names},
            "identity": {"fixture": True}}


MISOPEN = [80, 70, 60, 50, 40, 30, 30, 30, 30, 30, 45, 30, 30, 30, 30, 30]
RELEASE = [80, 70, 60, 50, 40, 30, 30, 30, 30, 30, 30, 40, 50, 60, 70, 80, 80]


def decision(candidate, size, **changes):
    result = dict(candidate_id=candidate["candidate_id"], side=candidate["side"],
                  category="suspected_misopen", grasp_before="held", grasp_during="held", grasp_after="held",
                  hold_start=5, hold_end=size - 1, evidence_frames=[5, candidate["start"], candidate["end"], size - 1],
                  confirmed=True, review_source="human", slip=False, release_end=None, reason="reviewed fixture")
    result.update(changes)
    return result


class CoreTests(unittest.TestCase):
    def run_case(self, widths=MISOPEN, config=None, **changes):
        ep = episode(widths)
        candidates = detect_candidates(ep)
        d = decision(candidates[0], len(widths), **changes)
        return ep, candidates, d, generate_targets(ep, candidates, [d], config)

    def test_one_frame_pulse_triggers(self):
        ep = episode([30, 30, 35, 30, 30])
        candidates = detect_candidates(ep)
        self.assertEqual([(c["start"], c["end"]) for c in candidates], [(1, 2)])

    def test_both_sides_trigger_independently(self):
        ep = episode(MISOPEN, other=MISOPEN)
        candidates = detect_candidates(ep)
        self.assertEqual({c["side"] for c in candidates}, {"left", "right"})
        for c in candidates:
            self.assertTrue(all(f"{c['side']}_wrist" in name for name in c["cameras"]))

    def test_invalid_side_placeholders_never_trigger(self):
        ep = episode(MISOPEN, other=MISOPEN)
        ep["valid_sides"] = ["right"]
        self.assertEqual({c["side"] for c in detect_candidates(ep)}, {"right"})

    def test_noise_does_not_trigger(self):
        self.assertEqual(detect_candidates(episode([30, 30.1, 29.9, 30.2, 30])), [])

    def test_slow_ramp_rejected_by_speed(self):
        self.assertEqual(detect_candidates(episode([30 + i * .1 for i in range(50)])), [])

    def test_original_descent_and_other_side_preserved(self):
        ep = episode(MISOPEN)
        snapshot = copy.deepcopy(ep)
        candidates = detect_candidates(ep)
        result = generate_targets(ep, candidates, [decision(candidates[0], len(MISOPEN))], {"closing_policy": "original_speed"})
        self.assertEqual(ep, snapshot)
        self.assertEqual([r[1] for r in result["target"][:6]], [r[19] for r in ep["state"][:6]])
        self.assertEqual([r[0] for r in result["target"]], [0] * len(MISOPEN))
        self.assertEqual(result["events"][0]["status"], "applied")
        self.assertAlmostEqual(result["target"][6][1] * 95, 20)
        self.assertAlmostEqual(result["target"][7][1] * 95, 10)
        self.assertTrue(all(row[1] == 0 for row in result["target"][8:]))
        self.assertEqual(result["events"][0]["zero_frame"], 8)
        self.assertTrue(all(not r[0] for r in result["valid"]))

    def test_left_correction_does_not_touch_right(self):
        ep = episode(MISOPEN, side="left", other=[24] * len(MISOPEN))
        c = detect_candidates(ep)
        result = generate_targets(ep, c, [decision(c[0], len(MISOPEN))])
        self.assertEqual([r[1] for r in result["target"]], [24 / 95] * len(MISOPEN))
        self.assertEqual(result["target"][10][0], 0)

    def test_connection_uses_actual_not_platform_median(self):
        widths = list(MISOPEN)
        widths[5:9] = [31, 30, 30, 30]
        _, _, _, result = self.run_case(widths, config={"closing_policy": "original_speed"})
        event = result["events"][0]
        self.assertEqual(event["connection_mm"], 31)
        self.assertEqual(event["platform_mm"], 30)
        self.assertAlmostEqual(result["target"][6][1] * 95, 21)

    def test_normal_release_only_preview_no_training_validity(self):
        ep, _, _, result = self.run_case(RELEASE, category="normal_release", grasp_after="not_held", hold_end=10, release_end=15)
        self.assertEqual(result["target"], [[r[9], r[19]] for r in ep["state"]])
        self.assertFalse(any(any(row) for row in result["valid"]))
        self.assertEqual(result["preview_target"][10][1], 0)
        self.assertAlmostEqual(result["preview_target"][15][1] * 95, 80)
        self.assertGreater(result["preview_target"][11][1], 0)
        comparison = result["release_comparisons"][0]
        self.assertIsNone(comparison["original_speed"])
        self.assertAlmostEqual(comparison["extra_time_needed_s"], .3)
        self.assertEqual(result["events"][0]["reason"], "release_policy_pending")

    def test_no_grasp_uncertainty_and_slip_unchanged(self):
        for changes in ({"category": "no_grasp"}, {"category": "uncertain"}, {"slip": True}, {"grasp_after": "uncertain"}):
            with self.subTest(changes=changes):
                ep, _, _, result = self.run_case(**changes)
                self.assertEqual(result["target"], [[r[9], r[19]] for r in ep["state"]])
                self.assertEqual(result["preview_target"], result["target"])

    def test_model_cannot_self_confirm(self):
        _, _, _, result = self.run_case(review_source="model")
        self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")

    def test_model_and_synthetic_are_previews(self):
        for source in ("model", "synthetic"):
            _, _, _, result = self.run_case(review_source=source, confirmed=False)
            self.assertEqual(result["events"][0]["status"], "preview")
            self.assertFalse(any(any(row) for row in result["valid"]))
            self.assertEqual(result["preview_target"][10][1], 0)

    def test_insufficient_hold_rolls_back_whole_event(self):
        widths = [80, 70, 60, 50, 40, 30, 30, 40, 50, 60]
        _, _, _, result = self.run_case(widths, config={"closing_policy": "original_speed"}, category="normal_release", grasp_after="not_held", hold_end=6, release_end=9)
        self.assertEqual(result["events"][0]["reason"], "insufficient_holding_time")
        self.assertEqual(result["target"], result["preview_target"])

    def test_missing_same_side_camera_keeps_values(self):
        ep = episode(MISOPEN)
        ep["cameras"] = {name: {"valid": True} for name in SIDE_CAMERAS["left"]}
        c = detect_candidates(ep)
        result = generate_targets(ep, c, [decision(c[0], len(MISOPEN))])
        self.assertEqual(result["events"][0]["reason"], "missing_same_side_camera")

    def test_near_zero_plateau_still_triggers(self):
        widths = [50, 40, 30, 20, 10, .5, .5, .5, .5, .5, 8, .5, .5, .5, .5, .5]
        ep, candidates, _, result = self.run_case(widths)
        self.assertEqual(len(candidates), 1)
        event = result["events"][0]
        self.assertEqual(event["reason"], "confirmed_misopen")
        self.assertTrue(event["no_closure_extension"])
        self.assertNotIn("speed_mm_s", event)
        self.assertEqual([r[1] for r in result["target"][:6]], [r[19] for r in ep["state"][:6]])
        self.assertTrue(all(r[1] == 0 for r in result["target"][6:]))
        self.assertTrue(all(not r[0] for r in result["valid"]))

    def test_zero_hold_needs_no_fabricated_closing_speed(self):
        widths = [0] * 10 + [8] + [0] * 5
        _, _, _, result = self.run_case(widths, hold_start=0)
        event = result["events"][0]
        self.assertEqual(event["status"], "applied")
        self.assertTrue(event["no_closure_extension"])
        self.assertEqual(event["zero_frame"], 0)
        self.assertNotIn("speed_mm_s", event)
        self.assertTrue(all(row[1] == 0 for row in result["target"]))

    def test_near_zero_unconfirmed_and_synthetic_remain_preview_only(self):
        widths = [0] * 10 + [8] + [0] * 5
        for source in ("human", "model", "synthetic"):
            ep, _, _, result = self.run_case(widths, hold_start=0, review_source=source, confirmed=False)
            with self.subTest(source=source):
                self.assertEqual(result["events"][0]["status"], "preview")
                self.assertFalse(any(any(row) for row in result["valid"]))
                self.assertEqual(result["target"], [[r[9], r[19]] for r in ep["state"]])
                self.assertEqual(result["preview_target"][10][1], 0)

    def test_near_zero_normal_release_is_unchanged(self):
        widths = [50, 40, 30, 20, 10, .5, .5, .5, .5, .5, .5, 20, 40, 60, 80, 80]
        ep, _, _, result = self.run_case(widths, category="normal_release", grasp_after="not_held", hold_end=10, release_end=14)
        self.assertEqual(result["events"][0]["reason"], "near_zero_platform")
        self.assertEqual(result["target"], [[r[9], r[19]] for r in ep["state"]])
        self.assertEqual(result["preview_target"], result["target"])

    def test_near_zero_requires_continuous_grasp_and_valid_exit(self):
        widths = [0] * 10 + [8] + [0] * 3 + [30, 30]
        for changes, reason in (({"hold_end": 13}, "unconfirmed_exit_boundary"),
                                ({"grasp_after": "uncertain"}, "continuous_grasp_not_established")):
            _, _, _, result = self.run_case(widths, hold_start=0, **changes)
            with self.subTest(changes=changes):
                self.assertEqual(result["events"][0]["reason"], reason)
                self.assertEqual(result["target"], result["preview_target"])

    def test_near_zero_median_cannot_hide_nonzero_contact(self):
        widths = [50, 40, 30, 20, 10, 1.5, .5, .5, .5, .5, 8, .5, .5, .5, .5, .5]
        _, _, _, result = self.run_case(widths)
        self.assertEqual(result["events"][0]["reason"], "near_zero_connection_mismatch")
        self.assertEqual(result["target"], result["preview_target"])

    def test_initial_hold_without_descent_not_fabricated(self):
        widths = [30] * 10 + [45] + [30] * 5
        _, _, _, result = self.run_case(widths, hold_start=0)
        self.assertEqual(result["events"][0]["reason"], "episode_starts_holding_or_missing_descent")

    def test_no_opening_is_not_triggered(self):
        ep = episode([80, 60, 40, 30, 30, 30, 30])
        result = generate_targets(ep, detect_candidates(ep), [])
        self.assertEqual(result["side_status"]["right"], "not_triggered")
        self.assertEqual(result["events"], [])

    def test_interior_hold_cannot_jump_back_to_original(self):
        _, _, _, result = self.run_case(hold_end=13)
        self.assertEqual(result["events"][0]["reason"], "unconfirmed_exit_boundary")

    def test_duplicate_decisions_rejected(self):
        ep, c, d, _ = self.run_case()
        result = generate_targets(ep, c, [d, dict(d)])
        self.assertEqual(result["events"][0]["reason"], "duplicate_visual_decisions")

    def test_before_and_after_evidence_required(self):
        _, _, _, result = self.run_case(evidence_frames=[5])
        self.assertEqual(result["events"][0]["reason"], "missing_before_after_evidence")

    def test_invalid_times_and_widths_rejected(self):
        for invalid in (float("nan"), -1, 2):
            ep = episode(MISOPEN)
            ep["state"][0][19] = invalid
            with self.assertRaises(ValueError):
                validate_episode(ep)
        ep = episode(MISOPEN)
        ep["timestamps"][2] = ep["timestamps"][1]
        with self.assertRaises(ValueError):
            validate_episode(ep)

    def test_model_confidence_is_not_an_authorization(self):
        _, _, _, result = self.run_case(review_source="model", confirmed=False, confidence=1.0)
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_contract_rejects_wrong_side_and_string_booleans(self):
        ep, c, d, _ = self.run_case()
        for changes in ({"side": "left"}, {"confirmed": "true"}, {"hold_start": -1}, {"release_end": 100}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_decision(dict(d, **changes), ep, c[0])

    def test_configuration_does_not_enable_unapproved_release_policy(self):
        with self.assertRaises(ValueError):
            detect_candidates(episode(RELEASE), {"release_policy": "fixed_timing"})

    def test_persisted_numeric_config_cannot_contain_credentials(self):
        ep, c, d, _ = self.run_case()
        result = generate_targets(ep, c, [d], {"api_key": "secret", "api_key_file": "/sensitive/path", "transport": {"Authorization": "secret"}})
        self.assertNotIn("api_key", result["config"])
        self.assertNotIn("api_key_file", result["config"])
        self.assertNotIn("transport", result["config"])

    def test_each_independent_grasp_estimates_its_own_platform(self):
        second = [80, 70, 60, 50, 40, 20, 20, 20, 20, 20, 20, 40, 60, 80, 80]
        widths = RELEASE + second
        ep = episode(widths)
        c = detect_candidates(ep)
        self.assertEqual(len(c), 2)
        decisions = [decision(c[0], len(widths), category="normal_release", grasp_after="not_held", hold_end=c[0]["start"], release_end=c[0]["end"]),
                     decision(c[1], len(widths), category="normal_release", grasp_after="not_held", hold_start=len(RELEASE)+5, hold_end=c[1]["start"], release_end=c[1]["end"])]
        result = generate_targets(ep, c, decisions)
        self.assertEqual([e["status"] for e in result["events"]], ["preview", "preview"])
        self.assertEqual([e["platform_mm"] for e in result["events"]], [30, 20])
        self.assertEqual(result["preview_target"][len(RELEASE)][1], 80 / 95)

    def test_continuous_hold_cannot_cross_independent_next_grasp(self):
        widths = MISOPEN + [80, 70, 60, 50, 40, 20, 20, 20, 20, 20, 20, 40, 60, 80, 80]
        ep = episode(widths)
        c = detect_candidates(ep)
        # The increase to 80 between independent grasps is also a candidate.
        first, last = c[0], c[-1]
        decisions = [decision(first, len(widths)),
                     decision(last, len(widths), hold_start=len(MISOPEN)+5, category="normal_release", grasp_after="not_held", hold_end=last["start"], release_end=last["end"])]
        result = generate_targets(ep, c, decisions)
        self.assertEqual(result["events"][0]["reason"], "crosses_other_release_boundary")
        # Without a second release, its independently reviewed hold start still bounds the first hold.
        decisions[1].update(category="suspected_misopen", grasp_after="held", release_end=None, hold_end=len(widths)-1)
        result = generate_targets(ep, c, decisions)
        self.assertEqual(result["events"][0]["reason"], "crosses_next_independent_grasp")

    def test_invalid_other_decision_cannot_crash_boundary_check(self):
        ep, c, d, _ = self.run_case()
        malformed = dict(d, candidate_id="nonexistent", hold_end="bad", category="normal_release")
        result = generate_targets(ep, c, [d, malformed])
        self.assertEqual(result["events"][0]["status"], "applied")

    def test_default_closure_maps_original_endpoints_on_same_time_grid(self):
        ep, _, _, result = self.run_case()
        event = result["events"][0]
        self.assertEqual(event["closing_policy"], "fixed_timing")
        self.assertTrue(event["closing_speed_changed"])
        self.assertEqual(event["modified_start"], 1)
        self.assertEqual(event["zero_frame"], 5)
        self.assertEqual(event["zero_time"], ep["timestamps"][5])
        self.assertEqual(len(result["target"]), len(ep["timestamps"]))
        self.assertEqual(result["target"][0][1], ep["state"][0][19])
        for i in range(6):
            phase = (80 - MISOPEN[i]) / (80 - 30)
            self.assertAlmostEqual(result["target"][i][1] * 95, 80 * (1 - phase))
        self.assertTrue(all(row[1] == 0 for row in result["target"][5:]))
        self.assertEqual([row[0] for row in result["target"]], [row[9] for row in ep["state"]])

    def test_fixed_closure_and_release_apply_symmetric_endpoint_mapping(self):
        widths = [80, 70, 60, 50, 40, 30, 30, 30, 30, 30, 30, 40, 50, 60, 70, 80]
        _, _, _, result = self.run_case(widths, category="normal_release", grasp_after="not_held", hold_end=10, release_end=15)
        preview = [row[1] * 95 for row in result["preview_target"]]
        for closing, opening in zip(preview[:6], reversed(preview[10:16])):
            self.assertAlmostEqual(closing, opening)
        self.assertEqual(result["events"][0]["status"], "preview")
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_fixed_closure_preserves_observed_phase_instead_of_inventing_smoothness(self):
        widths = [80, 79, 60, 58, 40, 30] + MISOPEN[6:]
        _, _, _, result = self.run_case(widths)
        mapped = [row[1] * 95 for row in result["target"][:6]]
        for i in range(1, 6):
            self.assertAlmostEqual((mapped[i - 1] - mapped[i]) / (widths[i - 1] - widths[i]), 80 / 50)
        self.assertEqual(result["events"][0]["zero_frame"], 5)

    def test_fixed_closure_succeeds_when_old_extension_would_overrun_hold(self):
        widths = [80, 70, 60, 50, 40, 30, 30, 40, 50, 60]
        _, _, _, result = self.run_case(widths, category="normal_release", grasp_after="not_held", hold_end=6, release_end=9)
        event = result["events"][0]
        self.assertEqual(event["status"], "preview")
        self.assertEqual(event["zero_frame"], 5)
        self.assertEqual(result["preview_target"][5][1], 0)
        comparison = result["closing_comparisons"][0]
        self.assertIsNone(comparison["original_speed"])
        self.assertEqual(comparison["original_speed_status"], "insufficient_holding_time")
        self.assertAlmostEqual(comparison["original_speed_zero_time"], .8)
        self.assertAlmostEqual(comparison["extra_time_needed_s"], .3)

    def test_closure_comparisons_share_one_release_transition(self):
        _, _, _, result = self.run_case(RELEASE, category="normal_release", grasp_after="not_held", hold_end=10, release_end=15)
        comparison = result["closing_comparisons"][0]
        release = result["release_comparisons"][0]
        self.assertEqual(comparison["start"], 0)
        self.assertEqual(comparison["end"], 5)
        self.assertEqual(comparison["display_end"], 15)
        self.assertEqual(comparison["fixed_timing"][10:16], release["fixed_timing"][10:16])
        self.assertEqual(comparison["original_speed"][10:16], release["fixed_timing"][10:16])
        self.assertEqual(comparison["fixed_timing"][5], 0)
        self.assertAlmostEqual(comparison["original_speed"][5] * 95, 30)
        self.assertEqual(comparison["original_speed"][8], 0)

    def test_fixed_closure_preserves_every_frame_before_detected_closing_start(self):
        widths = [80, 80] + MISOPEN
        ep, _, _, result = self.run_case(widths, hold_start=7)
        self.assertEqual(result["events"][0]["closing_start"], 2)
        self.assertEqual(result["target"][:3], [[row[9], row[19]] for row in ep["state"][:3]])
        self.assertEqual(result["target"][7][1], 0)
        self.assertTrue(all(not any(row) for row in result["valid"][:3]))

    def test_fixed_closure_bilateral_targets_and_independent_comparisons(self):
        ep = episode(MISOPEN, other=MISOPEN)
        candidates = detect_candidates(ep)
        result = generate_targets(ep, candidates, [decision(c, len(MISOPEN)) for c in candidates])
        self.assertEqual({c["side"] for c in result["closing_comparisons"]}, {"left", "right"})
        self.assertEqual(result["target"][5], [0, 0])
        self.assertTrue(all(all(row) for row in result["valid"][1:]))

    def test_fixed_closure_unconfirmed_cannot_modify_training_targets(self):
        ep, _, _, result = self.run_case(confirmed=False, review_source="model")
        self.assertEqual(result["events"][0]["status"], "preview")
        self.assertEqual(result["preview_target"][5][1], 0)
        self.assertEqual(result["target"], [[row[9], row[19]] for row in ep["state"]])
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_invalid_closing_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(config={"closing_policy": "unknown-fit"})

    def test_fixed_policy_still_rejects_unreliable_descent(self):
        _, _, _, result = self.run_case(config={"closing_max_speed_mm_s": 50})
        self.assertEqual(result["events"][0]["reason"], "unreliable_closing_speed")
        self.assertEqual(result["target"], result["preview_target"])
        self.assertEqual(result["closing_comparisons"], [])

    def test_rejected_release_still_protects_its_full_interval_from_later_fit(self):
        widths = [90, 60, 30, 30, 30, 50, 70, 90, 80, 70, 60, 50, 40, 40, 40, 40, 40, 55, 40, 40]
        ep = episode(widths)
        candidates = detect_candidates(ep)
        self.assertEqual([(c["start"], c["end"]) for c in candidates], [(4, 7), (16, 17)])
        decisions = [decision(candidates[0], len(widths), category="normal_release", grasp_after="not_held", hold_start=2, hold_end=4, release_end=10),
                     decision(candidates[1], len(widths), hold_start=12, hold_end=19)]
        result = generate_targets(ep, candidates, decisions)
        self.assertEqual(result["events"][0]["reason"], "nonmonotonic_release_requires_review")
        self.assertEqual(result["events"][1]["reason"], "overlaps_other_release_interval")
        self.assertEqual(result["target"], [[r[9], r[19]] for r in ep["state"]])
        self.assertEqual(result["preview_target"], result["target"])
        self.assertFalse(any(any(row) for row in result["valid"]))
        self.assertEqual(result["closing_comparisons"], [])
        # The old extension only modifies frames after contact 12, outside the
        # preceding declared release [4,10], so that independent case is allowed.
        original = generate_targets(ep, candidates, decisions, {"closing_policy": "original_speed"})
        self.assertEqual(original["events"][1]["status"], "applied")
        self.assertEqual(original["target"][:13], [[r[9], r[19]] for r in ep["state"][:13]])

    def test_own_release_end_cannot_reach_next_reviewed_grasp(self):
        widths = RELEASE + MISOPEN
        ep = episode(widths)
        candidates = detect_candidates(ep)
        self.assertEqual(len(candidates), 2)
        for next_contact in (14, 15, 16):
            decisions = [decision(candidates[0], len(widths), category="normal_release", grasp_after="not_held", hold_end=10, release_end=15),
                         decision(candidates[1], len(widths), hold_start=next_contact)]
            result = generate_targets(ep, candidates, decisions)
            with self.subTest(next_contact=next_contact):
                if next_contact <= 15:
                    self.assertEqual(result["events"][0]["reason"], "crosses_next_independent_grasp")
                    self.assertEqual(result["events"][0]["status"], "unchanged")
                    self.assertFalse(result["release_comparisons"])
                else:
                    self.assertEqual(result["events"][0]["reason"], "release_policy_pending")
                    self.assertEqual(result["events"][0]["status"], "preview")

    def test_other_arm_release_interval_does_not_block_a_same_side_fit(self):
        widths = [90, 60, 30, 30, 30, 50, 70, 90, 80, 70, 60, 50, 40, 40, 40, 40, 40, 55, 40, 40]
        ep = episode(widths, other=widths)
        candidates = detect_candidates(ep)
        first = next(c for c in candidates if c["side"] == "left" and c["start"] == 4)
        second = next(c for c in candidates if c["side"] == "right" and c["start"] == 16)
        decisions = [decision(first, len(widths), category="normal_release", grasp_after="not_held", hold_start=2, hold_end=4, release_end=10),
                     decision(second, len(widths), hold_start=12)]
        result = generate_targets(ep, candidates, decisions)
        current = next(e for e in result["events"] if e["candidate_id"] == second["candidate_id"])
        self.assertEqual(current["status"], "applied")
        self.assertEqual(result["target"][12][1], 0)
        self.assertEqual([r[0] for r in result["target"]], [r[9] for r in ep["state"]])


if __name__ == "__main__":
    unittest.main()
