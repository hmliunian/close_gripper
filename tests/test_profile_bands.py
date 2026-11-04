"""Generalization preserves calibration, free gaps, timing and task boundaries."""
import copy
import math
import unittest

from close_gripper.profile_bands import build_matching_catalog, derive_matching_bands
from close_gripper.profile_clean import generate_profile_targets
from close_gripper.task_profiles import match_hold_interval, validate_catalog
from test_task_profiles import all_mode_catalog, band, episode


class MatchingBandContracts(unittest.TestCase):
    def expanded(self, catalog=None, margin=.01):
        return build_matching_catalog(catalog or all_mode_catalog(), margin=margin, version="generalized_fixture")

    def test_padding_adds_supported_runtime_range_without_rewriting_calibration(self):
        old = all_mode_catalog()
        before = copy.deepcopy(old)
        new = self.expanded(old)
        self.assertEqual(old, before)
        p = new["tasks"][0]["sides"]["right"]
        self.assertEqual(p["closed_bands"], old["tasks"][0]["sides"]["right"]["closed_bands"])
        self.assertAlmostEqual(p["matching_bands"][0]["lower"], .17)
        self.assertAlmostEqual(p["matching_bands"][0]["upper"], .23)
        self.assertFalse(match_hold_interval(old, episode([.175]*8), "right", 0, 7)["matched"])
        self.assertTrue(match_hold_interval(new, episode([.175]*8), "right", 0, 7)["matched"])
        r = generate_profile_targets(episode([.175]*8), new)
        self.assertTrue(all(row[1] == 0 for row in r["preview_target"]))
        self.assertTrue(all(not any(row) for row in r["valid"]))
        self.assertFalse(p["matching_bands"][0]["pointwise_semantics_verified"])

    def two_anchors(self):
        c = all_mode_catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"] = [dict(p["closed_bands"][0], id=name, lower=lo, upper=hi, center=(lo+hi)/2)
                             for name,lo,hi in [("thin", .10,.12), ("thick", .20,.22)]]
        return c

    def test_merge_accepts_stable_gap_between_same_task_anchors(self):
        c = self.expanded(self.two_anchors())
        p = c["tasks"][0]["sides"]["right"]
        self.assertEqual(len(p["matching_bands"]), 1)
        self.assertEqual(p["matching_bands"][0]["basis_closed_band_ids"], ["thin", "thick"])
        self.assertTrue(match_hold_interval(c, episode([.16]*8), "right", 0, 7)["matched"])
        self.assertEqual(match_hold_interval(c, episode([.16]*6), "right", 0, 5)["reason"], "insufficient_holding_duration")

    def test_free_interval_prevents_merging_and_padding_cannot_jump_across(self):
        c = self.two_anchors()
        p = c["tasks"][0]["sides"]["right"]
        p["free_bands"].append(band("gap", .125, .195))
        c = self.expanded(c)
        bands = c["tasks"][0]["sides"]["right"]["matching_bands"]
        self.assertEqual(len(bands), 2)
        self.assertEqual(bands[0]["upper"], math.nextafter(.125, -math.inf))
        self.assertEqual(bands[1]["lower"], math.nextafter(.195, math.inf))
        for value in [.125, .16, .195, .8, 1.]:
            self.assertFalse(match_hold_interval(c, episode([value]*8), "right", 0, 7)["matched"])

    def test_conflict_range_stays_excluded_and_cannot_be_forged_away(self):
        c = self.two_anchors()
        c["tasks"][0]["sides"]["right"]["interval_vision_bands"].append(band("uncertain", .13, .19))
        c = self.expanded(c)
        self.assertEqual(match_hold_interval(c, episode([.16]*8), "right", 0, 7)["reason"], "requires_interval_vision")
        c["tasks"][0]["sides"]["right"]["matching_bands"][0]["upper"] = .18
        with self.assertRaisesRegex(ValueError, "matching bands"):
            validate_catalog(c)

    def test_clipped_zero_boundary_is_finite_and_disabled_sides_stay_disabled(self):
        c = all_mode_catalog()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"][0].update(lower=.002, center=.004, upper=.006)
        c = self.expanded(c)
        self.assertEqual(c["tasks"][0]["sides"]["right"]["matching_bands"][0]["lower"], 0)
        disabled = copy.deepcopy(p); disabled["automatic_use"] = False
        self.assertEqual(derive_matching_bands(disabled, .01), [])

    def test_9596_shape_matches_after_merging_without_relaxing_stability(self):
        c = self.two_anchors()
        p = c["tasks"][0]["sides"]["right"]
        p["closed_bands"] = [dict(p["closed_bands"][0], id=str(i), lower=a, upper=b, center=(a+b)/2)
                            for i,(a,b) in enumerate([(.02577,.02981),(.06842,.06991),(.12763,.12966),(.16837,.17107)])]
        values=[.126659,.126488,.126454,.125838,.123752,.123171,.11904,.118597,.113828]
        self.assertFalse(match_hold_interval(c, episode(values), "right", 0, 8)["matched"])
        new = self.expanded(c)
        self.assertTrue(match_hold_interval(new, episode(values), "right", 0, 8)["matched"])
        self.assertEqual(match_hold_interval(new, episode([.04,.06,.08,.1,.12,.14,.16]), "right", 0, 6)["reason"], "unstable_plateau")
        self.assertEqual(match_hold_interval(new, episode(values), "left", 0, 8)["reason"], "invalid_side")

    def test_undeclared_fake_evidence_repeated_padding_and_bad_margins_rejected(self):
        c = self.expanded()
        with self.assertRaisesRegex(ValueError, "already expanded"):
            self.expanded(c)
        for margin in [True, -.01, .1, float("nan"), float("inf")]:
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.expanded(margin=margin)
        del c["matching_policy"]
        with self.assertRaisesRegex(ValueError, "explicit catalog"):
            validate_catalog(c)
        c = self.expanded(); c["matching_policy"]["pointwise_semantics_verified"] = True
        with self.assertRaisesRegex(ValueError, "pointwise"):
            validate_catalog(c)
