"""Calibration census regressions: time windows and independent-source weight."""
import importlib.util
from pathlib import Path
import unittest

try:
    import numpy as np
    import pyarrow  # noqa: F401
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy/pyarrow census dependencies unavailable")
class SurveyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/survey_task_grippers.py"
        spec = importlib.util.spec_from_file_location("survey_task_grippers_contract", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_confirmation_uses_elapsed_time_not_sample_count(self):
        rows = list(self.module.plateaus(np.arange(7) / 10, np.full(7, .2)))
        self.assertEqual([(r[0], r[1]) for r in rows], [(0, 6)])
        self.assertGreaterEqual(.6 + 1e-9, self.module.PARAMS["minimum_hold_s"])
        short = list(self.module.plateaus(np.arange(6) / 10, np.full(6, .2)))
        self.assertEqual(short[0][:2], (0, 5))
        self.assertLess(.5 + 1e-9, self.module.PARAMS["minimum_hold_s"])

    def test_short_prefix_does_not_hide_overlapping_six_tenths_window(self):
        # A greedy 0..3 prefix used to hide the legitimate 1..8 stable span.
        values = np.array([0.] + [.015] * 3 + [.03] * 5)
        rows = list(self.module.plateaus(np.arange(9) / 10, values))
        self.assertEqual([(r[0], r[1]) for r in rows], [(0, 3), (1, 8)])

    def test_timestamp_gap_breaks_constant_plateau(self):
        times = np.array([0., .1, .2, .5, .6, .7])
        rows = list(self.module.plateaus(times, np.full(6, .2)))
        self.assertEqual([(r[0], r[1]) for r in rows], [(0, 2), (3, 5)])

    def test_nonfinite_value_and_amplitude_jump_break_windows(self):
        rows = list(self.module.plateaus(np.arange(14) / 10, np.array([.2] * 7 + [.6] * 7)))
        self.assertEqual([(r[0], r[1]) for r in rows], [(0, 6), (7, 13)])
        rows = list(self.module.plateaus(np.arange(7) / 10, np.array([.2, .2, np.nan, .2, .2, .2, .2])))
        self.assertEqual([(r[0], r[1]) for r in rows], [(3, 6)])

    def test_enumeration_matches_exhaustive_maximal_window_oracle(self):
        rng = np.random.default_rng(7)
        for _ in range(80):
            values = rng.choice([0., .01, .015, .025, .04], 12)
            times = np.cumsum(rng.choice([.1, .1, .1, .3], 12))
            valid = []
            for a in range(len(times)):
                for b in range(a, len(times)):
                    if (np.ptp(values[a:b + 1]) <= .02 + 1e-9
                            and np.all(np.diff(times[a:b + 1]) <= .15 + 1e-9)
                            and times[b] - times[a] >= .2 - 1e-9):
                        valid.append((a, b))
            maximal = {x for x in valid if not any(y != x and y[0] <= x[0] and y[1] >= x[1] for y in valid)}
            got = {(r[0], r[1]) for r in self.module.plateaus(times, values)}
            self.assertEqual(got, maximal)

    def test_source_weight_and_separated_modes(self):
        rows = ([dict(center=.2, source_path="recording-a")] * 100
                + [dict(center=.2, source_path="recording-b"),
                   dict(center=.8, source_path="recording-c"),
                   dict(center=.8, source_path="recording-d")])
        # Four sources require three per mode under production defaults. Add
        # one source at each level to test repeated windows cannot inflate support.
        rows += [dict(center=.2, source_path="recording-e"), dict(center=.8, source_path="recording-f")]
        sources = {r["source_path"] for r in rows}
        modes, remaining = self.module.mode_groups(rows, sources)
        self.assertEqual(len(modes), 2)
        self.assertEqual(sorted(m["source_count"] for m in modes), [3, 3])
        self.assertEqual(sorted(m["center"] for m in modes), [.2, .8])
        self.assertFalse(remaining)
        self.assertLess(modes[0]["upper"], modes[1]["lower"])


if __name__ == "__main__":
    unittest.main()
