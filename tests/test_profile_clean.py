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


def local_release_catalog():
    c = catalog()
    c.update(calibration_contract="all_modes_semantic_review_v2", minimum_positive_sources=2, evidence_noise=.003)
    p = c["tasks"][0]["sides"]["right"]
    p.update(calibration_contract=c["calibration_contract"], minimum_positive_sources=2, evidence_noise=.003,
        closed_bands=[dict(id="hold", lower=.161, center=.164, upper=.167,
                          source_ids=["a", "b"], reviewed_source_count=2, open_band_ids=["open"])],
        open_bands=[dict(id="open", lower=.198, center=.201, upper=.204, source_ids=["a"])],
        free_bands=[dict(id="free", lower=.198, center=.201, upper=.204)], interval_vision_bands=[],
        release_rules=[dict(id="local", closed_band_id="hold", open_band_ids=["open"],
            motion_min_delta=.032, motion_noise=.003,
            threshold_policy="max_2noise_min_default_observed_excursion_minus_noise",
            evaluation_only=True, preserves_fixed_timing=True, reviewed_source_count=1,
            evidence=[dict(case_id="viewed-a", source_id="a", dataset_root="/synthetic", episode_index=0,
                held_frame=8, free_frame=11, motion_start=8, motion_end=10,
                held_opening=.164, free_opening=.201, start_opening=.164, end_opening=.199,
                observed_excursion=.035, open_band_ids=["open"], review_source="model", confirmed=False)])])
    return c


class ProfileRuleContracts(unittest.TestCase):
    def test_reviewed_half_valleys_stop_at_visible_brush_hold_boundaries(self):
        # Main-view brush reviews end before the numeric plateau: ambiguity
        # covers insertion/release while width remains in the holding band.
        for numeric_end, held_end, uncertain_end, size in ((29, 26, 34, 46), (23, 20, 28, 37)):
            with self.subTest(held_end=held_end):
                values = [.2] * (numeric_end + 1) + [.4, .6, .8]
                values += [.8] * (size - len(values))
                states = (["held"] * (held_end + 1) + ["uncertain"] * (uncertain_end - held_end)
                          + ["free"] * (size - uncertain_end - 1))
                ep, c = episode(values), catalog()
                numeric = generate_profile_targets(ep, c)
                result = generate_profile_targets(ep, c, reviewed_states={"right": states})
                event, = applied(result)
                self.assertEqual((event["hold_start"], event["hold_end"], event["modified_end"]),
                                 (0, held_end, held_end))
                self.assertIsNone(event["release_end"])
                self.assertIn("release:no_observed_boundary", event["transition_issues"])
                self.assertEqual([row[1] for row in result["rule_target"][:held_end + 1]],
                                 [0.] * (held_end + 1))
                for i in range(held_end + 1, size):
                    self.assertEqual(result["rule_target"][i][1], values[i])
                    self.assertFalse(result["rule_applied_mask"][i][1])
                self.assertEqual(result["numeric_candidates"], numeric["candidates"])
                self.assertEqual((event["block_start"], event["block_end"]), (0, size - 1))
                removed, = [e for e in result["events"] if e["reason"] == "reviewed_uncertain"]
                self.assertEqual((removed["start"], removed["end"]), (held_end + 1, numeric_end))
                self.assertEqual(removed["numeric_candidate_id"], event["numeric_candidate_id"])

    def test_reviewed_intersections_need_six_tenths_after_unknown_frames_split_hold(self):
        values = [.2] * 13
        states = ["held"] * 6 + ["uncertain"] + ["held"] * 6
        result = generate_profile_targets(episode(values), catalog(), reviewed_states={"right": states})
        self.assertFalse(applied(result))
        self.assertEqual(result["rule_target"], result["target"])
        self.assertFalse(any(any(row) for row in result["rule_applied_mask"]))
        self.assertEqual([(e["start"], e["end"], e["reason"]) for e in result["events"]],
                         [(0, 5, "insufficient_reviewed_holding_duration"),
                          (6, 6, "reviewed_uncertain"),
                          (7, 12, "insufficient_reviewed_holding_duration")])
        self.assertEqual(result["coverage"]["right"]["reason"], "no_qualified_reviewed_held_window")
        self.assertEqual(result["coverage"]["right"]["numeric_matched_intervals"], 1)

    def test_reviewed_hold_union_cannot_replace_a_qualified_stable_witness(self):
        # The 0.7 s reviewed hold crosses two drift modes; each stable
        # intersection lasts only 0.5 s, even though the original union qualified.
        values = [.19] * 3 + [.205] * 4 + [.22] * 3
        states = ["free"] + ["held"] * 8 + ["free"]
        result = generate_profile_targets(episode(values), catalog(), reviewed_states={"right": states})
        numeric, = result["numeric_candidates"]
        self.assertEqual([(w["start"], w["end"]) for w in numeric["matched_windows"]], [(0, 6), (3, 9)])
        self.assertFalse(applied(result))
        self.assertEqual(result["rule_target"], result["target"])
        rejected, = [e for e in result["events"] if e.get("reviewed_state") == "held"]
        self.assertEqual([(w["start"], w["end"]) for w in rejected["unqualified_windows"]],
                         [(1, 6), (3, 8)])
        self.assertTrue(all(w["duration_s"] < .6 for w in rejected["unqualified_windows"]))
        self.assertEqual(sum(e["end"] - e["start"] + 1 for e in result["events"]), len(values))

    def test_overlapping_reviewed_witnesses_merge_and_cover_each_claimed_hold_frame(self):
        values = [.19] * 7 + [.205] * 7 + [.22] * 7
        states = ["uncertain"] + ["held"] * 19 + ["free"]
        result = generate_profile_targets(episode(values), catalog(), reviewed_states={"right": states})
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["hold_end"]), (1, 19))
        self.assertEqual([(w["start"], w["end"]) for w in event["matched_windows"]], [(1, 13), (7, 19)])
        for i in range(1, 20):
            self.assertTrue(any(w["start"] <= i <= w["end"] and w["duration_s"] >= .6
                                for w in event["matched_windows"]))
            self.assertEqual(result["rule_target"][i][1], 0.)
        for i in (0, 20):
            self.assertEqual(result["rule_target"][i][1], values[i])
            self.assertFalse(result["rule_applied_mask"][i][1])

    def test_visible_full_cycle_preserves_fixed_time_mapping_and_marks_reviewed_source(self):
        values = [.8, .8, .6, .4, .2] + [.2] * 8 + [.4, .6, .8, .8]
        states = ["free"] * 4 + ["held"] * 9 + ["free"] * 4
        ep, c = episode(values), catalog()
        before = copy.deepcopy((ep, c, states))
        numeric = generate_profile_targets(ep, c)
        result = generate_profile_targets(ep, c, reviewed_states={"right": states})
        event, = applied(result)
        self.assertEqual((event["closing_start"], event["hold_start"], event["hold_end"], event["release_end"]),
                         (1, 4, 12, 15))
        for field in ("rule_target", "rule_applied_mask", "preview_phase", "target", "valid"):
            self.assertEqual(result[field], numeric[field])
        self.assertAlmostEqual(result["rule_target"][2][1], .8 * (.6 - .2) / (.8 - .2))
        self.assertAlmostEqual(result["rule_target"][13][1], .8 * (.4 - .2) / (.8 - .2))
        self.assertEqual(result["workflow"], "task_profile_rule")
        self.assertEqual(result["rule_source"], "task_profile_rule_reviewed")
        self.assertEqual(event["source"], result["rule_source"])
        self.assertEqual(result["preview_source"][5][1], result["rule_source"])
        self.assertEqual(result["hold_policy"], "reviewed_held_stable_task_band")
        self.assertNotEqual(result["rule_version"], numeric["rule_version"])
        self.assertFalse(result["grasp_semantics_verified"])
        self.assertFalse(result["training_ready"])
        self.assertEqual((ep, c, states), before)
        self.assertNotIn("numeric_candidates", numeric)
        self.assertNotIn("semantic_gate_applied", numeric)
        self.assertEqual(numeric, generate_profile_targets(ep, c, reviewed_states=None))

    def test_unknown_frame_rejects_whole_transition_without_rejecting_hold(self):
        values = [.8, .8, .6, .4, .2] + [.2] * 8 + [.4, .6, .8, .8]
        states = ["free"] * 4 + ["held"] * 9 + ["free"] * 4
        states[2] = states[14] = "uncertain"
        result = generate_profile_targets(episode(values), catalog(), reviewed_states={"right": states})
        event, = applied(result)
        self.assertEqual((event["modified_start"], event["modified_end"]), (4, 12))
        self.assertIsNone(event["closing_start"])
        self.assertIsNone(event["release_end"])
        self.assertIn("closing:reviewed_uncertain_boundary", event["transition_issues"])
        self.assertIn("release:reviewed_uncertain_boundary", event["transition_issues"])
        for i, value in enumerate(values):
            self.assertEqual(result["rule_target"][i][1], 0. if 4 <= i <= 12 else value)
            self.assertEqual(result["rule_applied_mask"][i][1], 4 <= i <= 12)

    def test_reviewed_transition_cannot_cross_an_independent_held_cycle(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"].append(
            dict(id="next", lower=.24, center=.25, upper=.26))
        values = [.2] * 8 + [.21, .22, .23, .24, .25] + [.25] * 8
        states = ["held"] * 10 + ["free"] + ["held"] * 10
        result = generate_profile_targets(episode(values), c, reviewed_states={"right": states})
        first, second = applied(result)
        self.assertEqual((first["hold_start"], first["hold_end"]), (0, 9))
        self.assertEqual((second["hold_start"], second["hold_end"]), (11, 20))
        self.assertIn("release:crosses_independent_held_cycle", first["transition_issues"])
        self.assertIsNone(first["release_end"])
        self.assertEqual(result["rule_target"][10][1], values[10])
        self.assertFalse(result["rule_applied_mask"][10][1])

    def test_transition_cannot_extend_zero_to_visible_free_frames(self):
        values = [.2] * 6 + [.21, .22, .219, .3, .8]
        states = ["held"] * 8 + ["free"] * 3
        result = generate_profile_targets(episode(values), catalog(), reviewed_states={"right": states})
        event, = applied(result)
        self.assertEqual(event["hold_end"], 7)
        self.assertIn("release:zero_on_reviewed_free_frame", event["transition_issues"])
        self.assertEqual(result["rule_target"][8:], [[0., v] for v in values[8:]])
        self.assertTrue(all(not row[1] for row in result["rule_applied_mask"][8:]))

    def test_reviewed_states_are_specific_to_each_actual_arm(self):
        c = catalog()
        c["tasks"][0]["sides"]["left"] = copy.deepcopy(c["tasks"][0]["sides"]["right"])
        ep = episode([.2] * 7, left=[.2] * 7)
        result = generate_profile_targets(ep, c, reviewed_states={"left": ["held"] * 7,
                                                                 "right": ["uncertain"] * 7})
        self.assertEqual({e["side"] for e in applied(result)}, {"left"})
        self.assertEqual(result["rule_target"], [[0., .2] for _ in range(7)])
        self.assertEqual(result["rule_applied_mask"], [[True, False] for _ in range(7)])

    def test_malformed_per_frame_reviewed_states_are_rejected(self):
        cases = [[], {}, {"left": ["held"] * 7}, {"right": ["held"] * 6},
                 {"right": "held"}, {"right": tuple(["held"] * 7)},
                 {"right": ["held"] * 6 + ["unknown"]}, {"right": ["held"] * 6 + [None]},
                 {"right": ["held"] * 6 + [True]},
                 {"right": ["held"] * 7, "left": ["held"] * 7}]
        for states in cases:
            with self.subTest(states=states), self.assertRaisesRegex(ValueError, "reviewed_states"):
                generate_profile_targets(episode([.2] * 7), catalog(), reviewed_states=states)

    def test_known_misaligned_source_is_rejected_by_source_or_exact_release_identity(self):
        c = catalog()
        c.update(excluded_source_ids=["bad-source"], excluded_episode_identities=[
            dict(source_id="bad-source", dataset_root="/release/a", episode_index=12)])
        for by_source in (True, False):
            ep = episode([.2] * 8)
            if by_source:
                ep["identity"]["source_metadata"]["source_episode_id"] = "bad-source"
            else:
                ep["identity"].update(dataset_root="/release/a", episode_index=12)
            result = generate_profile_targets(ep, c)
            self.assertEqual(result["coverage"]["right"]["reason"], "source_alignment_uncertain")
            self.assertEqual(applied(result), [])
            self.assertFalse(any(any(row) for row in result["rule_applied_mask"]))
        ep["identity"]["dataset_root"] = "/release/other"
        self.assertTrue(applied(generate_profile_targets(ep, c)))

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
        event, = applied(result)
        self.assertIsNone(event["closing_start"])
        self.assertIn("closing:no_observed_boundary", event["transition_issues"])
        self.assertEqual([row[1] for row in result["rule_target"]][1:], [0.] * 8)

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

    def test_qualified_hold_anchor_is_not_shortened_by_closing_alignment(self):
        values = [.8, .5, .21, .2] + [.2] * 6
        result = generate_profile_targets(episode(values), catalog())
        event, = applied(result)
        self.assertEqual(event["start"], 2)
        self.assertEqual((event["contact_frame"], event["confirmation_frame"]), (2, 8))
        self.assertEqual(event["observed_closing_end"], 3)
        self.assertEqual(result["rule_target"][2][1], 0.)
        self.assertEqual(result["rule_target"][3][1], 0.)

    def test_release_alignment_cannot_veto_a_qualified_six_tenths_hold(self):
        result = generate_profile_targets(episode([.2] * 6 + [.21, .8]), catalog())
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["hold_end"], event["release_start"]), (0, 6, 6))
        self.assertEqual([row[1] for row in result["rule_target"][:7]], [0.] * 7)

    def test_qualified_overlapping_windows_survive_transition_alignment(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"] = [
            dict(id="drift", lower=.20, center=.225, upper=.25)]
        values = [.8, .5, .22, .215] + [.23] * 5 + [.24] * 7
        result = generate_profile_targets(episode(values), c)
        candidate, = [p for p in result["candidates"] if p["matched"]]
        self.assertEqual([(w["start"], w["end"]) for w in candidate["matched_windows"]],
                         [(2, 8), (4, 15)])
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["hold_end"]), (2, 15))
        self.assertEqual([row[1] for row in result["rule_target"][2:]], [0.] * 14)

    def test_larger_release_endpoint_is_preserved_above_local_open_anchor(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["open_bands"] = [dict(id="local", lower=.79, center=.8, upper=.81)]
        result = generate_profile_targets(episode([.2] * 8 + [.5, .95]), c)
        event, = applied(result)
        self.assertEqual(event["release_end"], 9)
        self.assertEqual(result["rule_target"][-1][1], .95)

    def test_short_release_is_fitted_and_unreliable_transition_does_not_cancel_hold(self):
        result = generate_profile_targets(episode([.2] * 8 + [.23, .23]), catalog())
        event, = applied(result)
        self.assertEqual((event["release_start"], event["release_end"]), (7, 8))
        self.assertEqual([row[1] for row in result["rule_target"][:8]], [0.] * 8)
        self.assertEqual(result["rule_target"][-1][1], .23)
        scenarios = [
            (episode([.8] + [.2] * 8, [0.] + [.01 + i / 10 for i in range(8)]), "closing:unreliable_motion_speed"),
        ]
        for ep, reason in scenarios:
            with self.subTest(reason=reason):
                result = generate_profile_targets(ep, catalog())
                event, = applied(result)
                self.assertIn(reason, event["transition_issues"])
                self.assertEqual(result["rule_target"][0][1], .8)
                self.assertEqual([row[1] for row in result["rule_target"][1:]], [0.] * 8)

    def test_invalid_data_or_minimum_hold_override_are_rejected(self):
        for value in (float("nan"), True, -.1, 1.1):
            ep = episode([.2] * 7)
            ep["state"][2][19] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate_profile_targets(ep, catalog())
        with self.assertRaisesRegex(ValueError, "agree with the task catalog"):
            generate_profile_targets(episode([.2] * 7), catalog(), dict(minimum_hold_s=.2))

    def test_evidence_bound_small_release_keeps_original_fixed_times(self):
        values = [.164] * 9 + [.179, .199, .201, .201]
        c = local_release_catalog()
        original = copy.deepcopy(c)
        result = generate_profile_targets(episode(values), c)
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["release_start"], event["release_end"]), (0, 8, 10))
        self.assertIsNone(event["release_rule_id"])
        self.assertAlmostEqual(event["release_motion_min_delta"], .006000001)
        self.assertAlmostEqual(result["rule_target"][9][1], .199 * (.179 - .164) / (.199 - .164))
        self.assertEqual(result["rule_target"][10:], [[0., v] for v in values[10:]])
        self.assertEqual(result["config"]["motion_min_delta"], .04)
        self.assertEqual(result["config"]["motion_noise"], .003)
        self.assertEqual(c, original)
        c["tasks"][0]["sides"]["right"]["release_rules"] = []
        without_local_rule = generate_profile_targets(episode(values), c)
        self.assertEqual(without_local_rule["rule_target"], result["rule_target"])

    def test_small_transitions_work_without_borrowing_other_mode_evidence(self):
        c = local_release_catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"].append(dict(id="other", lower=.101, center=.104, upper=.107,
                                     source_ids=["a", "b"], reviewed_source_count=2, open_band_ids=["other_open"]))
        p["open_bands"].append(dict(id="other_open", lower=.138, center=.141, upper=.144, source_ids=["a"]))
        p["free_bands"].append(dict(id="other_free", lower=.138, center=.141, upper=.144))
        result = generate_profile_targets(episode([.104] * 9 + [.12, .139, .141]), c)
        event, = applied(result)
        self.assertIsNone(event["release_rule_id"])
        self.assertEqual(event["release_end"], 10)
        result = generate_profile_targets(episode([.199, .179] + [.164] * 9), c)
        event, = applied(result)
        self.assertEqual(event["closing_start"], 0)
        self.assertEqual([row[1] for row in result["rule_target"][2:]], [0.] * 9)

    def test_release_endpoint_can_be_below_reference_without_changing_noise(self):
        values = [.164] * 9 + [.179, .195, .197]
        result = generate_profile_targets(episode(values), local_release_catalog())
        event, = applied(result)
        self.assertEqual(event["release_end"], 10)
        self.assertFalse(event["release_reference_gate_applied"])
        self.assertEqual(result["rule_target"][-1][1], .197)
        with self.assertRaisesRegex(ValueError, "unchanged motion_noise"):
            generate_profile_targets(episode([.164] * 8), local_release_catalog(), dict(motion_noise=.001))

    def test_3093_shaped_hold_maps_to_zero_and_preserves_release_times(self):
        c = catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"] = [dict(id="ladle", lower=.153, center=.156, upper=.1601)]
        p["open_bands"] = [dict(id="reference", lower=.7848, center=.79, upper=.80)]
        values = [.156] * 42 + [.179, .516, .694, .735, .744, .746, .738]
        result = generate_profile_targets(episode(values), c)
        event, = applied(result)
        self.assertEqual((event["hold_start"], event["hold_end"], event["release_start"], event["release_end"]), (0, 41, 41, 46))
        self.assertEqual(result["rule_target"][18][1], 0.)
        self.assertEqual(result["rule_target"][46:], [[0., v] for v in values[46:]])
        self.assertAlmostEqual(result["rule_target"][43][1], .744 * (.516 - .156) / (.744 - .156))

    def test_unobserved_release_keeps_qualified_hold_and_records_transition_issue(self):
        result = generate_profile_targets(episode([.2] * 8 + [.23, .23]), catalog(), dict(motion_noise=.035))
        event, = applied(result)
        self.assertEqual([row[1] for row in result["rule_target"][:8]], [0.] * 8)
        self.assertIsNone(event["release_end"])
        self.assertIn("release:no_observed_boundary", event["transition_issues"])

    def test_short_noisy_or_outside_windows_do_not_become_holds(self):
        for values in ([.2] * 6, [.1] * 20, [.18, .22] * 10):
            result = generate_profile_targets(episode(values), catalog())
            self.assertFalse(applied(result))
            self.assertEqual(result["rule_target"], result["target"])

    def test_overlapping_transition_never_overwrites_another_qualified_hold(self):
        c = catalog()
        c["tasks"][0]["sides"]["right"]["closed_bands"].append(
            dict(id="next", lower=.24, center=.25, upper=.26))
        values = [.2] * 8 + [.21, .22, .23, .24, .25] + [.25] * 8
        result = generate_profile_targets(episode(values), c)
        self.assertEqual(len(applied(result)), 2)
        for candidate in result["candidates"]:
            if candidate["matched"]:
                for i in range(candidate["start"], candidate["end"] + 1):
                    self.assertEqual(result["rule_target"][i][1], 0.)
                    self.assertEqual(result["preview_phase"][i][1], "hold")
                    self.assertTrue(result["rule_applied_mask"][i][1])
        self.assertGreater(result["rule_target"][10][1], 0.)

    def test_same_value_semantic_conflict_is_explicit_and_never_generates_a_curve(self):
        c = local_release_catalog()
        p = c["tasks"][0]["sides"]["right"]
        p.update(automatic_use=False, status="needs_review", closed_bands=[], release_rules=[],
                 interval_vision_bands=[dict(id="ambiguous", lower=.161, center=.164, upper=.167)])
        result = generate_profile_targets(episode([.164] * 8), c)
        self.assertFalse(applied(result))
        self.assertEqual(result["rule_target"], result["target"])
        self.assertEqual(result["coverage"]["right"]["reason"], "requires_interval_vision")
        event, = result["events"]
        self.assertEqual(event["qualification_reason"], "requires_interval_vision")
        self.assertEqual(event["interval_vision_band_id"], "ambiguous")
        self.assertFalse(event["matched"])

    def test_forged_local_release_evidence_is_rejected_before_processing(self):
        changes = [dict(motion_noise=.002), dict(motion_min_delta=.01), dict(reviewed_source_count=2)]
        for changeset in changes:
            c = local_release_catalog()
            c["tasks"][0]["sides"]["right"]["release_rules"][0].update(changeset)
            with self.subTest(changes=changeset), self.assertRaises(ValueError):
                generate_profile_targets(episode([.164] * 8), c)
        c = local_release_catalog()
        evidence = c["tasks"][0]["sides"]["right"]["release_rules"][0]["evidence"][0]
        evidence["free_frame"] = 5
        with self.assertRaisesRegex(ValueError, "ordered held-to-free"):
            generate_profile_targets(episode([.164] * 8), c)


if __name__ == "__main__":
    unittest.main()
