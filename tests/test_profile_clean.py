"""Rule-evaluation contracts; synthetic fixtures are not grasp ground truth."""

import copy
import unittest

from close_gripper.profile_clean import generate_profile_targets
from close_gripper.task_profiles import SCHEMA, UNITS


def catalog():
    profile = dict(status="visual_reviewed", automatic_use=True,
        closed_bands=[dict(id="hold", lower=.18, center=.2, upper=.22)],
        open_bands=[dict(id="open", lower=.75, center=.8, upper=1.)], counts={},
        provenance=dict(review_source="model", confirmed=False))
    return dict(schema=SCHEMA, version="synthetic-profile-rule", units=UNITS,
        minimum_hold_s=.6, maximum_sample_gap_s=.15, maximum_plateau_range=.02,
        tasks=[dict(task="测试夹取", aliases=[], sides={"right": profile})])


def episode(values, times=None, left=None):
    state = [[i * .01] * 20 for i in range(len(values))]
    for i, value in enumerate(values):
        state[i][19], state[i][9] = value, left[i] if left is not None else 0.
    return dict(episode_id="synthetic-rule", state=state, cameras={},
        timestamps=times if times is not None else [i / 10 for i in range(len(values))],
        valid_sides=["left", "right"] if left is not None else ["right"],
        identity=dict(source_metadata=dict(source_tasks=["测试夹取"])))


def applied(result):
    return [e for e in result["events"] if e["status"] == "applied"]


class ProfileRuleContracts(unittest.TestCase):
    def test_full_cycle_preserves_motion_times_and_actual_open_endpoint(self):
        values = [.8, .8, .6, .4, .2] + [.2] * 8 + [.4, .6, .8, .8]
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        self.assertEqual((event["closing_start"], event["contact_frame"], event["hold_end"], event["release_end"]),
                         (1, 4, 12, 15))
        curve = [row[1] for row in result["rule_target"]]
        self.assertAlmostEqual(curve[2], .8 * (.6 - .2) / (.8 - .2))
        self.assertAlmostEqual(curve[3], .8 * (.4 - .2) / (.8 - .2))
        self.assertEqual(curve[4:13], [0.] * 9)
        self.assertAlmostEqual(curve[13], .8 * (.4 - .2) / (.8 - .2))
        self.assertEqual((curve[0], curve[1], curve[15], curve[16]), (.8, .8, .8, .8))
        self.assertEqual(event["confirmation_frame"], 10)
        self.assertEqual(event["hold_start"], 4)

    def test_initial_holding_and_normal_release_need_no_invented_closing(self):
        values = [.2] * 9 + [.4, .8]
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        self.assertIsNone(event["closing_start"])
        self.assertEqual((event["hold_start"], event["release_start"], event["release_end"]), (0, 8, 10))
        self.assertTrue(event["initial_or_cropped_hold"])
        self.assertEqual([row[1] for row in result["rule_target"][:9]], [0.] * 9)

    def test_episode_ending_while_holding_is_processed_through_last_frame(self):
        values = [.8, .6, .2] + [.2] * 8
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        self.assertIsNone(event["release_end"])
        self.assertEqual(event["modified_end"], len(values) - 1)
        self.assertEqual(result["rule_target"][-1][1], 0.)

    def test_six_tenths_is_elapsed_time_and_confirmation_backfills_start(self):
        result = generate_profile_targets(episode([.2] * 7), catalog())
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["confirmation_frame"]), (0, 6))
        self.assertEqual([row[1] for row in result["rule_target"]], [0.] * 7)
        short = generate_profile_targets(episode([.2] * 6), catalog())
        self.assertFalse(applied(short))
        self.assertEqual(short["events"][0]["reason"], "insufficient_holding_duration")

    def test_all_repeated_cycles_and_closed_modes_are_processed(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"].append(
            dict(id="wide", lower=.48, center=.5, upper=.52))
        values = [.8, .6, .2] + [.2] * 7 + [.4, .8, .8, .65, .5] + [.5] * 7 + [.65, .8]
        result = generate_profile_targets(episode(values), c)
        events = applied(result)
        self.assertEqual(len(events), 2)
        self.assertEqual([e["closed_band_id"] for e in events], ["hold", "wide"])
        for event in events:
            self.assertTrue(all(result["rule_target"][i][1] == 0
                                for i in range(event["hold_start"], event["hold_end"] + 1)))
        self.assertEqual(result["rule_target"][-1][1], .8)

    def test_overlapping_maximal_matches_are_deduplicated_with_witnesses_retained(self):
        values = [.19] * 7 + [.205] * 7 + [.22] * 7
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        windows = event["matched_windows"]
        self.assertEqual([(w["start"], w["end"]) for w in windows], [(0, 13), (7, 20)])
        self.assertEqual((event["hold_start"], event["hold_end"]), (0, 20))
        self.assertEqual([row[1] for row in result["rule_target"]], [0.] * len(values))
        self.assertTrue(all(w["observed_upper"] - w["observed_lower"] <= .02 + 1e-9 for w in windows))

    def test_short_prefix_does_not_hide_a_later_overlapping_valid_window(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [dict(id="zero", lower=0., center=.02, upper=.04)]
        values = [0.] + [.015] * 3 + [.03] * 5
        result = generate_profile_targets(episode(values), c)
        qualified = [c for c in result["candidates"] if c["matched"]]
        self.assertEqual([(c["start"], c["end"]) for c in qualified], [(1, 8)])
        # Numerical qualification does not fabricate a missing closing edge.
        self.assertEqual(result["events"][-1]["reason"], "missing_aligned_closing_boundary")

    def test_timestamp_gap_cannot_combine_two_short_holds(self):
        times = [i / 10 for i in range(6)] + [1. + i / 10 for i in range(6)]
        ep = episode([.2] * 12, times)
        result = generate_profile_targets(ep, catalog())
        self.assertFalse(applied(result))
        self.assertEqual(result["rule_target"], result["target"])
        self.assertEqual(result["coverage"]["right"]["timestamp_gap_boundaries"], [[5, 6]])

    def test_gap_crops_holding_without_fitting_an_unobserved_transition(self):
        values = [.8] + [.2] * 7
        times = [0.] + [1. + i / 10 for i in range(7)]
        result = generate_profile_targets(episode(values, times), catalog())
        event, = applied(result)
        self.assertTrue(event["cropped_start_by_timestamp_gap"])
        self.assertIsNone(event["closing_start"])
        self.assertEqual(event["modified_start"], 1)
        self.assertEqual(result["rule_target"][0][1], .8)
        self.assertFalse(result["rule_applied_mask"][0][1])

    def test_gap_before_release_keeps_unobserved_release_original(self):
        values = [.2] * 7 + [.4, .8]
        times = [i / 10 for i in range(7)] + [1.3, 1.4]
        result = generate_profile_targets(episode(values, times), catalog())
        event, = applied(result)
        self.assertTrue(event["cropped_end_by_timestamp_gap"])
        self.assertIsNone(event["release_start"])
        self.assertEqual([row[1] for row in result["rule_target"][-2:]], [.4, .8])

    def test_side_masks_and_training_annotations_remain_isolated(self):
        ep = episode([.2] * 7)
        ep["annotation.gripper_target"] = "existing external metadata"
        before = copy.deepcopy(ep)
        result = generate_profile_targets(ep, catalog())
        self.assertEqual(ep, before)
        self.assertEqual(result["target"], [[0., .2] for _ in range(7)])
        self.assertFalse(any(any(row) for row in result["valid"]))
        self.assertTrue(all(not row[0] and row[1] for row in result["rule_applied_mask"]))
        self.assertTrue(all(row == ["", "task_profile_rule"] for row in result["preview_source"]))
        self.assertTrue(result["evaluation_only"])
        self.assertFalse(result["grasp_semantics_verified"])
        self.assertFalse(result["training_ready"])
        self.assertTrue(all("decision" not in e and not e["reviewed"] for e in result["events"]))

    def test_both_sides_are_independently_matched(self):
        c = catalog()
        c["tasks"][0]["sides"]["left"] = copy.deepcopy(c["tasks"][0]["sides"]["right"])
        result = generate_profile_targets(episode([.8] * 7, left=[.2] * 7), c)
        self.assertEqual({e["side"] for e in applied(result)}, {"left"})
        self.assertEqual(result["rule_target"], [[0., .8] for _ in range(7)])
        self.assertEqual(result["coverage"]["right"]["reason"], "outside_closed_bands")

    def test_no_match_unknown_task_and_disabled_profile_do_not_modify(self):
        c = catalog()
        for kind in ("outside", "unknown", "disabled"):
            cfg, ep = copy.deepcopy(c), episode([.8 if kind == "outside" else .2] * 7)
            if kind == "unknown":
                ep["identity"]["source_metadata"]["source_tasks"] = ["unknown"]
            elif kind == "disabled":
                cfg["tasks"][0]["sides"]["right"].update(status="needs_review", automatic_use=False)
            with self.subTest(kind=kind):
                result = generate_profile_targets(ep, cfg)
                self.assertEqual(result["rule_target"], result["target"])
                self.assertFalse(any(any(row) for row in result["rule_applied_mask"]))
                self.assertEqual(result["coverage"]["right"]["reason"],
                                 dict(outside="outside_closed_bands", unknown="unknown_task",
                                      disabled="profile_not_automatic")[kind])

    def test_actual_closing_end_is_not_moved_to_a_tolerated_earlier_band_sample(self):
        values = [.8, .5, .21, .2] + [.2] * 6
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        self.assertEqual(event["start"], 2)
        self.assertEqual((event["contact_frame"], event["confirmation_frame"]), (3, 9))
        self.assertGreater(result["rule_target"][2][1], 0.)
        self.assertEqual(result["rule_target"][3][1], 0.)

    def test_release_alignment_rechecks_minimum_holding_duration(self):
        result = generate_profile_targets(episode([.2] * 6 + [.21, .8]), catalog())
        self.assertFalse(applied(result))
        self.assertEqual(result["events"][0]["reason"], "insufficient_holding_duration_after_boundary_alignment")

    def test_aligned_confirmation_must_itself_be_stable_across_merged_windows(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [
            dict(id="drift", lower=.20, center=.225, upper=.25)]
        values = [.8, .5, .22, .215] + [.23] * 5 + [.24] * 7
        result = generate_profile_targets(episode(values), c)
        candidate, = [p for p in result["candidates"] if p["matched"]]
        self.assertEqual([(w["start"], w["end"]) for w in candidate["matched_windows"]],
                         [(2, 8), (4, 15)])
        # The actual closing end is frame 3, and 3..9 spans 0.025, although
        # each original overlapping witness was stable. Do not move contact.
        self.assertFalse(applied(result))
        self.assertEqual(result["events"][0]["reason"], "unstable_confirmation_after_boundary_alignment")
        self.assertEqual(result["rule_target"], result["target"])

    def test_larger_release_endpoint_is_preserved_above_local_open_anchor(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["open_bands"] = [dict(id="local", lower=.79, center=.8, upper=.81)]
        result = generate_profile_targets(episode([.2] * 8 + [.5, .95]), c)
        event, = applied(result)
        self.assertEqual(event["release_end"], 9)
        self.assertEqual(result["rule_target"][-1][1], .95)

    def test_unresolved_exit_and_unreliable_speed_preserve_observations(self):
        scenarios = [
            (episode([.2] * 8 + [.23, .23]), "unresolved_release_boundary"),
            (episode([.8] + [.2] * 8, [0.] + [.01 + i / 10 for i in range(8)]), "unreliable_motion_speed"),
        ]
        for ep, reason in scenarios:
            with self.subTest(reason=reason):
                result = generate_profile_targets(ep, catalog())
                self.assertFalse(applied(result))
                self.assertEqual(result["rule_target"], result["target"])
                self.assertEqual(result["events"][0]["reason"], reason)

    def test_invalid_data_or_minimum_hold_override_are_rejected(self):
        for value in (float("nan"), True, -.1, 1.1):
            ep = episode([.2] * 7)
            ep["state"][2][19] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate_profile_targets(ep, catalog())
        with self.assertRaisesRegex(ValueError, "agree with the task catalog"):
            generate_profile_targets(episode([.2] * 7), catalog(), dict(minimum_hold_s=.2))


if __name__ == "__main__":
    unittest.main()
