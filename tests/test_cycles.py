"""Complete-command contracts with synthetic evidence, not VLM accuracy claims."""

import copy
import math
import unittest

from close_gripper.contracts import SIDE_CAMERAS
from close_gripper.cycles import cycle_config, detect_cycles, generate_cycle_targets


def episode(values, left=None):
    states = [[0.0] * 20 for _ in values]
    for i, row in enumerate(states):
        row[19] = values[i]
        if left is not None:
            row[9] = left[i]
    return dict(episode_id="cycle_fixture", timestamps=[i / 10 for i in range(len(values))],
                state=states, valid_sides=["right"] if left is None else ["left", "right"],
                cameras={name: {"valid": True} for names in SIDE_CAMERAS.values() for name in names},
                identity={"source": "fixture"})


RELEASE = [.9, .9, .9, .7, .5, .3, .3, .3, .3, .3, .3, .5, .7, .9, .9, .9]


def decision(candidate, **changes):
    d = dict(candidate_id=candidate["candidate_id"], side=candidate["side"],
             category="grasp_release", hold_start=5, hold_end=10, release_end=13,
             pregrasp_start=0, pregrasp_clear=True, continuous_grasp=True, slip=False,
             evidence_frames=[0, 5, 10, 13, 15], review_source="model", confirmed=False,
             reason="Offline fixture portraying a visually supported full grasp/release")
    d.update(changes)
    return d


def closing(candidates, side="right", start=None):
    return next(c for c in candidates if c["side"] == side and c["trigger_kind"] == "closure"
                and (start is None or c["closing_start"] == start))


class CycleTests(unittest.TestCase):
    def run_case(self, values=RELEASE, config=None, **changes):
        ep = episode(values)
        candidates = detect_cycles(ep, config)
        d = decision(closing(candidates), **changes)
        return ep, candidates, d, generate_cycle_targets(ep, candidates, [d], config)

    def test_full_release_is_automatic_with_separate_model_provenance(self):
        ep, candidates, _, result = self.run_case()
        event = next(e for e in result["events"] if e["status"] == "applied")
        self.assertEqual(event["contact_frame"], 5)
        self.assertEqual(event["release_start"], 10)
        self.assertEqual(event["release_end"], 13)
        self.assertEqual(event["reason"], "automatic_visual_numeric_cycle")
        self.assertTrue(all(result["target"][i][1] == 0 for i in range(5, 11)))
        self.assertEqual([round(result["target"][i][1], 6) for i in range(2, 6)], [.9, .6, .3, 0])
        self.assertEqual([round(result["target"][i][1], 6) for i in range(10, 14)], [0, .3, .6, .9])
        self.assertTrue(all(result["valid"][i][1] for i in range(14)))
        self.assertTrue(all(result["source"][i][1] == "model" for i in range(14)))
        self.assertEqual(result["target"][13:], [[s[9], s[19]] for s in ep["state"][13:]])
        self.assertEqual(result["coverage"]["right"]["accepted_cycles"], 1)
        self.assertFalse(any(row[0] for row in result["valid"]))

    def test_input_identity_values_times_and_decisions_are_unchanged(self):
        ep = episode(RELEASE)
        candidates = detect_cycles(ep)
        d = decision(closing(candidates))
        snapshot = copy.deepcopy((ep, candidates, [d]))
        generate_cycle_targets(ep, candidates, [d])
        self.assertEqual((ep, candidates, [d]), snapshot)

    def test_pregrasp_jitter_is_flattened_only_for_supported_grasp(self):
        values = [.9, .7, .9, .9, .6, .3, .3, .3, .3, .3, .3, .5, .7, .9, .9, .9]
        ep = episode(values)
        candidates = detect_cycles(ep)
        actual = closing(candidates, start=3)
        ds = [decision(actual)]
        for c in candidates:
            if c is not actual:
                ds.append(decision(c, category="no_grasp", hold_start=None, hold_end=None,
                                   release_end=None, pregrasp_start=None, pregrasp_clear=False,
                                   continuous_grasp=False, evidence_frames=[c["start"], c["end"]]))
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual([row[1] for row in result["target"][:4]], [.9] * 4)
        self.assertEqual(result["phase"][1][1], "pregrasp")
        self.assertEqual(result["target"][5][1], 0)
        self.assertEqual(result["coverage"]["right"]["accepted_cycles"], 1)

    def prefix_preparation_case(self):
        # Rounded source shape of pilot episode 1025; the false close is separate
        # from the visually supported grasp at frames 15--18.
        values = [.9015] * 7 + [.8999, .7990, .7276, .7123, .7323, .9182,
                               .9817, .9951, .9988, .9797, .7514] + [.4485] * 15
        ep = episode(values)
        candidates = detect_cycles(ep)
        false_close = closing(candidates, start=7)
        grasp = closing(candidates, start=15)
        held = decision(grasp, category="grasp_hold", hold_start=18, hold_end=32,
                        release_end=None, evidence_frames=[0, 10, 15, 18, 25, 32])
        negative = decision(false_close, category="no_grasp", hold_start=None, hold_end=None,
                            release_end=None, pregrasp_start=None, pregrasp_clear=False,
                            continuous_grasp=False, evidence_frames=[0, 7, 10, 15])
        return ep, candidates, held, negative

    def test_episode_start_free_preparation_cleans_the_entire_supported_prefix(self):
        ep, candidates, held, negative = self.prefix_preparation_case()
        original = copy.deepcopy(ep)
        result = generate_cycle_targets(ep, candidates, [negative, held])
        event = next(e for e in result["events"] if e["status"] == "applied")
        self.assertEqual(event["modified_start"], 0)
        self.assertEqual(event["pregrasp_status"], "cleaned")
        self.assertEqual((event["closing_start"], event["contact_frame"]), (15, 18))
        self.assertEqual(event["opening_target"], .9951)
        self.assertEqual([r[1] for r in result["target"][:16]], [.9951] * 16)
        self.assertEqual([r[1] for r in result["target"][18:]], [0.] * 15)
        self.assertEqual([r[1] for r in result["phase"][:15]], ["pregrasp"] * 15)
        self.assertTrue(all(row[1] for row in result["valid"]))
        self.assertTrue(all(row[0] == 0. for row in result["target"]))
        self.assertEqual(result["events"][0]["reason"], "no_grasp")
        self.assertEqual(ep, original)

    def test_prefix_extension_keeps_unreviewed_and_uncertain_previous_motion_protected(self):
        for review_previous in (False, True):
            ep, candidates, held, negative = self.prefix_preparation_case()
            decisions = [held]
            if review_previous:
                decisions.append(dict(negative, category="uncertain"))
            result = generate_cycle_targets(ep, candidates, decisions)
            with self.subTest(review_previous=review_previous):
                event = next(e for e in result["events"] if e["candidate_id"] == held["candidate_id"])
                self.assertEqual(event["modified_start"], 15)
                self.assertEqual(event["pregrasp_status"], "unresolved_context_preserved")
                self.assertEqual(result["target"][:15], [[s[9], s[19]] for s in ep["state"][:15]])
                self.assertFalse(any(row[1] for row in result["valid"][:15]))
                self.assertEqual(result["target"][18][1], 0.)

    def test_interior_preparation_preserves_its_entry_and_reports_partial_cleaning(self):
        values = [.5] * 5 + [.8, .8, .6, .8, .95, .95, .7, .3] + [.3] * 8
        ep = episode(values)
        candidates = detect_cycles(ep)
        grasp = closing(candidates, start=10)
        held = decision(grasp, category="grasp_hold", hold_start=12, hold_end=20,
                        release_end=None, pregrasp_start=5, evidence_frames=[5, 10, 12, 20])
        decisions = [held]
        for candidate in candidates:
            if candidate is not grasp:
                decisions.append(decision(candidate, category="no_grasp", hold_start=None, hold_end=None,
                    release_end=None, pregrasp_start=None, pregrasp_clear=False, continuous_grasp=False,
                    evidence_frames=[candidate["start"], candidate["end"]]))
        result = generate_cycle_targets(ep, candidates, decisions)
        event = next(e for e in result["events"] if e["status"] == "applied")
        self.assertEqual(event["modified_start"], 9)
        self.assertEqual(event["pregrasp_status"], "partially_cleaned")
        self.assertEqual(result["target"][:9], [[s[9], s[19]] for s in ep["state"][:9]])
        self.assertEqual(result["target"][9][1], ep["state"][9][19])
        self.assertEqual(result["target"][12][1], 0.)

    def test_no_pregrasp_free_evidence_preserves_preparation(self):
        values = [.9, .88, .9] + RELEASE[3:]
        ep, _, _, result = self.run_case(values, pregrasp_clear=False, pregrasp_start=None)
        self.assertEqual(result["target"][:2], [[s[9], s[19]] for s in ep["state"][:2]])
        self.assertFalse(any(any(row) for row in result["valid"][:2]))

    def test_visual_hold_boundary_can_follow_numeric_contact_by_one_frame(self):
        _, _, _, result = self.run_case(hold_start=6, evidence_frames=[0, 6, 10, 13, 15])
        event = next(e for e in result["events"] if e["status"] == "applied")
        self.assertEqual(event["decision"]["hold_start"], 6)
        self.assertEqual(event["contact_frame"], 5)
        self.assertEqual(result["target"][5][1], 0)

    def test_tiny_positive_drift_does_not_shift_actual_release_start(self):
        values = RELEASE.copy()
        values[6:11] = [.3, .3002, .3003, .3004, .3005]
        _, _, _, result = self.run_case(values)
        event = next(e for e in result["events"] if e["status"] == "applied")
        self.assertEqual(event["release_start"], 10)

    def test_closing_without_later_opening_can_generate_end_hold(self):
        values = RELEASE[:11]
        _, _, _, result = self.run_case(values, category="grasp_hold", release_end=None,
                                       hold_end=10, evidence_frames=[0, 5, 10])
        self.assertEqual(result["events"][0]["status"], "applied")
        self.assertEqual(result["target"][-1][1], 0)
        self.assertEqual(result["phase"][-1][1], "hold")

    def test_initially_held_object_can_release_without_fabricated_closure(self):
        values = [.3] * 6 + [.5, .7, .9, .9]
        ep = episode(values)
        c, = detect_cycles(ep)
        self.assertEqual(c["trigger_kind"], "initial_hold")
        d = decision(c, hold_start=0, hold_end=5, release_end=8,
                     pregrasp_start=None, pregrasp_clear=False, evidence_frames=[0, 5, 8, 9])
        result = generate_cycle_targets(ep, [c], [d])
        event = result["events"][0]
        self.assertEqual(event["status"], "applied")
        self.assertIsNone(event["closing_start"])
        self.assertEqual([r[1] for r in result["target"][:6]], [0] * 6)
        self.assertEqual(result["target"][-1][1], .9)

    def test_initially_held_until_end_is_supported_from_frame_zero(self):
        ep = episode([.3] * 8)
        c, = detect_cycles(ep)
        d = decision(c, category="grasp_hold", hold_start=0, hold_end=7, release_end=None,
                     pregrasp_start=None, pregrasp_clear=False, evidence_frames=[0, 7])
        result = generate_cycle_targets(ep, [c], [d])
        self.assertTrue(all(r[1] == 0 for r in result["target"]))
        self.assertEqual(result["events"][0]["status"], "applied")

    def test_flat_gripper_is_an_observation_request_not_automatic_no_grasp(self):
        ep = episode([0] * 8)
        c, = detect_cycles(ep)
        result = generate_cycle_targets(ep, [c], [])
        self.assertEqual(result["events"][0]["reason"], "missing_visual_decision")
        self.assertFalse(result["coverage"]["right"]["complete_review"])
        self.assertEqual(result["coverage"]["right"]["unreviewed_intervals"], [[0, 7]])
        self.assertFalse(any(any(r) for r in result["valid"]))

    def test_whole_episode_no_grasp_preserves_every_value(self):
        ep = episode(RELEASE)
        candidates = detect_cycles(ep)
        ds = [decision(c, category="no_grasp", hold_start=None, hold_end=None,
                       release_end=None, pregrasp_clear=False, pregrasp_start=None,
                       continuous_grasp=False, evidence_frames=[c["start"], c["end"]]) for c in candidates]
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])
        self.assertEqual(result["preview_target"], result["target"])
        self.assertTrue(result["coverage"]["right"]["complete_review"])
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_detection_covers_every_valid_frame_including_gaps(self):
        ep = episode([.9] * 60 + RELEASE + [.9] * 50, left=[0] * 126)
        candidates = detect_cycles(ep)
        for side in ep["valid_sides"]:
            seen = set()
            for c in candidates:
                if c["side"] == side:
                    seen.update(range(c["start"], c["end"] + 1))
            self.assertEqual(seen, set(range(len(ep["state"]))))

    def test_two_independent_grasps_get_separate_platforms_and_curves(self):
        first = RELEASE
        second = [.9, .9, .9, .6, .3, .1, .1, .1, .1, .1, .1, .5, .7, .9, .9, .9]
        ep = episode(first + second)
        candidates = detect_cycles(ep)
        ds = []
        for offset in (0, len(first)):
            c = closing(candidates, start=offset + 2)
            ds.append(decision(c, hold_start=offset + 5, hold_end=offset + 10,
                release_end=offset + 13, pregrasp_start=offset,
                evidence_frames=[offset, offset + 5, offset + 10, offset + 13, offset + 15]))
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual([e["platform_opening"] for e in result["events"] if e["status"] == "applied"], [.3, .1])
        self.assertEqual(result["target"][5][1], 0)
        self.assertEqual(result["target"][21][1], 0)
        self.assertEqual(result["target"][16][1], .9)

    def test_simultaneous_bilateral_cycles_stay_independent(self):
        ep = episode(RELEASE, left=RELEASE)
        candidates = detect_cycles(ep)
        ds = [decision(c) for c in candidates if c["trigger_kind"] == "closure"]
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual(result["target"][5], [0, 0])
        self.assertEqual(result["valid"][5], [True, True])
        self.assertEqual(result["source"][5], ["model", "model"])

    def test_invalid_left_placeholder_never_gets_a_candidate_or_label(self):
        ep = episode(RELEASE)
        ep["state"][0][9] = -12
        candidates = detect_cycles(ep)
        self.assertEqual({c["side"] for c in candidates}, {"right"})
        result = generate_cycle_targets(ep, candidates, [decision(closing(candidates))])
        self.assertEqual(result["target"][0][0], -12)
        self.assertEqual(result["side_status"]["left"], "invalid")

    def test_synthetic_and_explicit_review_only_are_previews(self):
        for source, cfg in (("synthetic", {}), ("model", {"auto_accept_model": False})):
            ep, _, _, result = self.run_case(config=cfg, review_source=source)
            with self.subTest(source=source):
                self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])
                self.assertEqual(result["preview_target"][5][1], 0)
                self.assertFalse(any(any(r) for r in result["valid"]))
                self.assertEqual(result["preview_source"][5][1], source)
                self.assertEqual(result["source"][5][1], "")

    def test_uncertain_slip_and_discontinuous_evidence_cannot_modify(self):
        for changes in ({"category": "uncertain", "hold_start": None, "hold_end": None,
                         "release_end": None, "pregrasp_start": None, "pregrasp_clear": False,
                         "continuous_grasp": False}, {"slip": True}, {"continuous_grasp": False}):
            ep, _, _, result = self.run_case(**changes)
            self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])
            self.assertEqual(result["preview_target"], result["target"])

    def test_missing_same_side_camera_cannot_use_opposite_arm(self):
        ep = episode(RELEASE)
        ep["cameras"] = {c: {"valid": True} for c in SIDE_CAMERAS["left"]}
        candidates = detect_cycles(ep)
        result = generate_cycle_targets(ep, candidates, [decision(closing(candidates))])
        self.assertEqual(result["events"][0]["reason"], "missing_same_side_camera")
        self.assertFalse(any(any(r) for r in result["valid"]))

    def test_model_cannot_self_confirm_or_forge_an_unrecognized_source(self):
        for changes in ({"confirmed": True}, {"review_source": "automatic-human"}):
            _, _, _, result = self.run_case(**changes)
            self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")

    def test_human_provenance_is_required_for_human_source(self):
        _, _, _, missing = self.run_case(review_source="human", confirmed=True)
        self.assertEqual(missing["events"][0]["reason"], "invalid_visual_decision")
        _, _, _, accepted = self.run_case(review_source="human", confirmed=True,
                                         reviewer="fixture-reviewer", reviewed_at="2026-11-03",
                                         review_note="Offline fixture; not actual human reference")
        self.assertEqual(accepted["source"][5][1], "human")

    def test_boundaries_or_evidence_outside_window_are_rejected(self):
        for changes in ({"hold_start": -1}, {"evidence_frames": [0, 5, 99]},
                        {"release_end": 99}, {"pregrasp_clear": "true"},
                        {"evidence_frames": [0, 5, 10]}):
            _, _, _, result = self.run_case(**changes)
            self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")
            self.assertFalse(any(any(r) for r in result["valid"]))

    def test_unsupported_interior_holding_exit_is_not_a_step_back_to_nonzero(self):
        _, _, _, result = self.run_case(category="grasp_hold", release_end=None,
                                       hold_end=9, evidence_frames=[0, 5, 9])
        self.assertEqual(result["events"][0]["reason"], "unconfirmed_exit_boundary")
        self.assertFalse(any(any(r) for r in result["valid"]))

    def test_duplicate_decisions_are_not_resolved_by_input_order(self):
        ep, candidates, d, _ = self.run_case()
        result = generate_cycle_targets(ep, candidates, [d, dict(d)])
        self.assertEqual(result["events"][0]["reason"], "duplicate_candidate_or_decision")
        self.assertFalse(any(any(r) for r in result["valid"]))

    def test_too_distant_visual_boundary_is_not_forced_to_numeric_candidate(self):
        _, _, _, result = self.run_case(hold_start=10, hold_end=10,
                                       evidence_frames=[0, 10, 13, 15],
                                       config={"boundary_tolerance_s": .2})
        self.assertEqual(result["events"][0]["reason"], "missing_aligned_closing_boundary")

    def test_config_keeps_credentials_out_and_rejects_bad_units(self):
        config = cycle_config({"api_key": "do-not-persist", "transport": {"Authorization": "secret"}})
        self.assertNotIn("api_key", config)
        self.assertNotIn("transport", config)
        for override in ({"motion_min_delta": math.nan}, {"boundary_tolerance_s": 0},
                         {"motion_noise": .2}, {"auto_accept_model": "true"},
                         {"release_policy": "review"}, {"pregrasp_min_plateau_frames": 1.5}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                cycle_config(override)

    def test_rejected_second_cycle_still_protects_its_declared_semantic_interval(self):
        ep = episode(RELEASE + RELEASE)
        candidates = detect_cycles(ep)
        first = closing(candidates, start=2)
        second = closing(candidates, start=18)
        ds = [decision(first), decision(second, hold_start=8, hold_end=26, release_end=29,
                 pregrasp_start=0, evidence_frames=[0, 8, 26, 29, 31])]
        for ordered in (ds, list(reversed(ds))):
            result = generate_cycle_targets(ep, candidates, ordered)
            self.assertEqual(result["events"][0]["reason"], "overlapping_independent_grasp_or_release")
            self.assertEqual(result["events"][1]["reason"], "missing_aligned_closing_boundary")
            self.assertFalse(any(any(r) for r in result["valid"]))

    def test_adjacent_cycles_may_share_an_unchanged_open_endpoint(self):
        ep = episode(RELEASE + RELEASE)
        candidates = detect_cycles(ep)
        first = closing(candidates, start=2)
        second = closing(candidates, start=18)
        ds = [decision(first), decision(second, hold_start=21, hold_end=26, release_end=29,
                 pregrasp_start=13, evidence_frames=[13, 21, 26, 29, 31])]
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual(result["coverage"]["right"]["accepted_cycles"], 2)
        self.assertEqual(result["target"][13][1], .9)

    def test_independent_event_on_other_arm_does_not_block_a_supported_cycle(self):
        ep = episode(RELEASE + RELEASE, left=RELEASE + RELEASE)
        candidates = detect_cycles(ep)
        first = closing(candidates, side="right", start=2)
        second = closing(candidates, side="left", start=18)
        ds = [decision(first), decision(second, hold_start=8, hold_end=26, release_end=29,
                 pregrasp_start=0, evidence_frames=[0, 8, 26, 29, 31])]
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual(result["coverage"]["right"]["accepted_cycles"], 1)
        self.assertEqual(result["coverage"]["left"]["accepted_cycles"], 0)
        self.assertEqual(result["target"][5][1], 0)
        self.assertEqual(result["target"][5][0], .3)

    def test_model_confidence_cannot_bypass_missing_holding_evidence(self):
        _, _, _, result = self.run_case(evidence_frames=[0, 13, 15], confidence=1.0)
        self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_unreviewed_earlier_motion_preserves_preparation_without_blocking_later_grasp(self):
        values = [.9, .7, .9, .9, .6, .3, .3, .3, .3, .3, .3, .5, .7, .9, .9, .9]
        ep = episode(values)
        candidates = detect_cycles(ep)
        actual = closing(candidates, start=3)
        result = generate_cycle_targets(ep, candidates, [decision(actual)])
        self.assertEqual([r[1] for r in result["target"][:3]], values[:3])
        self.assertEqual(result["target"][5][1], 0)
        event = next(e for e in result["events"] if e["candidate_id"] == actual["candidate_id"])
        self.assertEqual(event["pregrasp_status"], "unresolved_context_preserved")
        self.assertEqual(event["modified_start"], 3)

    def test_uncertain_earlier_motion_is_not_erased_by_later_pregrasp_claim(self):
        values = [.9, .7, .9, .9, .6, .3, .3, .3, .3, .3, .3, .5, .7, .9, .9, .9]
        ep = episode(values)
        candidates = detect_cycles(ep)
        actual = closing(candidates, start=3)
        uncertain = closing(candidates, start=0)
        ds = [decision(actual), decision(uncertain, category="uncertain", hold_start=None,
            hold_end=None, release_end=None, pregrasp_start=None, pregrasp_clear=False,
            continuous_grasp=False, evidence_frames=[uncertain["start"], uncertain["end"]])]
        result = generate_cycle_targets(ep, candidates, ds)
        self.assertEqual([r[1] for r in result["target"][:3]], values[:3])
        self.assertEqual(result["target"][5][1], 0)

    def test_one_frame_gap_is_added_to_neighbouring_review_window(self):
        values = RELEASE + [.9] * 6
        ep = episode(values)
        candidates = detect_cycles(ep)
        # Release 13 + 0.5 s context ends at 18, leaving 19..21. Configure
        # context to leave exactly one final frame instead.
        candidates = detect_cycles(ep, {"context_after_s": .7})
        self.assertTrue(all(c["end"] > c["start"] for c in candidates))
        self.assertEqual(max(c["end"] for c in candidates), len(values) - 1)

    def test_reviewed_context_overlap_does_not_complete_a_missing_candidate_review(self):
        ep = episode(RELEASE + RELEASE)
        candidates = detect_cycles(ep)
        second = closing(candidates, start=18)
        d = decision(second, category="no_grasp", hold_start=None, hold_end=None,
                     release_end=None, pregrasp_start=None, pregrasp_clear=False,
                     continuous_grasp=False, evidence_frames=[second["start"], second["end"]])
        result = generate_cycle_targets(ep, candidates, [d])
        coverage = result["coverage"]["right"]
        self.assertEqual(coverage["review_window_frames"], len(ep["timestamps"]))
        self.assertEqual(coverage["reviewed_frames"], coverage["review_window_frames"])
        self.assertEqual(coverage["unreviewed_intervals"], [])
        self.assertEqual(coverage["unreviewed_candidates"], 1)
        self.assertFalse(coverage["complete_review"])
        self.assertEqual(result["side_status"]["right"], "incomplete_review")
        self.assertIn("not per-frame visual truth", coverage["coverage_definition"])
        self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])

    def test_imported_negative_or_uncertain_decision_cannot_claim_holding_boundaries(self):
        for category in ("no_grasp", "uncertain"):
            for source in ("model", "human"):
                provenance = {} if source == "model" else dict(confirmed=True, reviewer="fixture",
                    reviewed_at="2026-11-03", review_note="Synthetic imported review fixture")
                ep, _, _, result = self.run_case(category=category, review_source=source, **provenance)
                with self.subTest(category=category, source=source):
                    self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")
                    self.assertEqual(result["events"][0]["detail"], "contradictory_non_grasp_evidence")
                    self.assertFalse(result["events"][0]["reviewed"])
                    self.assertFalse(result["coverage"]["right"]["complete_review"])
                    self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])

    def test_imported_continuing_hold_cannot_simultaneously_declare_a_release(self):
        ep, _, _, result = self.run_case(category="grasp_hold", hold_end=15, release_end=13,
                                       evidence_frames=[0, 5, 13, 15])
        self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")
        self.assertEqual(result["events"][0]["detail"], "contradictory_holding_release_boundary")
        self.assertEqual(result["target"], [[s[9], s[19]] for s in ep["state"]])
        self.assertFalse(any(any(row) for row in result["valid"]))

    def test_imported_unclear_preparation_cannot_carry_a_free_preparation_boundary(self):
        _, _, _, result = self.run_case(pregrasp_clear=False, pregrasp_start=0)
        self.assertEqual(result["events"][0]["reason"], "invalid_visual_decision")
        self.assertEqual(result["events"][0]["detail"], "contradictory_pregrasp_evidence")
        self.assertFalse(any(any(row) for row in result["valid"]))


if __name__ == "__main__":
    unittest.main()
