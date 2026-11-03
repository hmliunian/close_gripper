"""Target interpolation respects invalid intervals and exact source frames."""
import math
import unittest

from close_gripper.training import interpolate_gripper_target_pairs, validate_uniform_source_timestamps


class GripperInterpolationTests(unittest.TestCase):
    def test_actual_source_timestamps_must_match_the_declared_grid(self):
        with self.assertRaisesRegex(ValueError, "uniform grid"):
            validate_uniform_source_timestamps([0, 1], [0., .2], 0., 10.)
        with self.assertRaisesRegex(ValueError, "uniform grid"):
            validate_uniform_source_timestamps([7, 11], [.7, 1.2], 0., 10.)
        validate_uniform_source_timestamps([0, 1], [0., .1 + 5e-8], 0., 10.)
        validate_uniform_source_timestamps([7, 11], [2.7, 3.1], 2., 10.)
        start = 1_700_000_000.
        validate_uniform_source_timestamps([0, 1, 11], [start, start + .1, start + 1.1], start, 10.)

    def test_source_time_validation_rejects_nonfinite_unordered_or_malformed_input(self):
        for args in (([], [], 0., 10.), ([0], [math.nan], 0., 10.),
                     ([0, 1], [0., 0.], 0., 10.), ([1, 0], [.1, 0.], 0., 10.),
                     ([True], [0.], 0., 10.), ([0], [0.], 0., 0.), ([0], [0.], math.inf, 10.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_uniform_source_timestamps(*args)

    def test_valid_sides_interpolate_independently(self):
        value, valid = interpolate_gripper_target_pairs([[1., .2]], [[0., .6]], [[True, True]], [[True, True]], [.5])
        self.assertEqual(value, [[.5, .4]])
        self.assertEqual(valid, [[True, True]])

    def test_invalid_endpoint_does_not_contaminate_valid_endpoint(self):
        values, flags = interpolate_gripper_target_pairs(
            [[0., .2]] * 3, [[float("nan"), .8]] * 3,
            [[True, False]] * 3, [[False, True]] * 3, [0., .5, 1.])
        self.assertEqual(flags, [[True, False], [False, False], [False, True]])
        self.assertEqual(values, [[0., 0.], [0., 0.], [0., .8]])

    def test_invalid_intervals_are_not_blended_with_measured_fallback(self):
        values, flags = interpolate_gripper_target_pairs(
            [[0., 0.]], [[.6, .8]], [[True, True]], [[False, False]], [.5])
        self.assertEqual(flags, [[False, False]])
        self.assertEqual(values, [[0., 0.]])

    def test_bad_shapes_flags_fractions_and_valid_values_fail(self):
        cases = [([[0, 0]], [], [[True, True]], [[True, True]], [.5]),
                 ([[0]], [[1, 1]], [[True, True]], [[True, True]], [.5]),
                 ([[0, 0]], [[1, 1]], [[1, True]], [[True, True]], [.5]),
                 ([[0, 0]], [[1, 1]], [[True, True]], [[True, True]], [math.nan]),
                 ([[0, 0]], [[1, 1]], [[True, True]], [[True, True]], [1.1]),
                 ([[0, 0]], [[1, 1]], [[True, True]], [[True, True]], [True]),
                 ([[0, math.inf]], [[1, 1]], [[True, True]], [[True, True]], [.5]),
                 ([[0, 0]], [[1, -.1]], [[True, True]], [[True, True]], [.5])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                interpolate_gripper_target_pairs(*args)


if __name__ == "__main__":
    unittest.main()
