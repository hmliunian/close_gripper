"""Explicit, local loader integration contract; observations are never replaced."""
import math


def validate_uniform_source_timestamps(indices, timestamps, timestamp_start, source_fps):
    """Require actual source times to match the grid used for action interpolation.

    A 1 microsecond absolute tolerance admits ordinary float serialization noise
    and Unix-epoch float64 rounding, without admitting a missing 10 Hz frame.
    """
    if not indices or len(indices) != len(timestamps):
        raise ValueError("annotation timestamps require equally sized nonempty indices and times")
    for value in (timestamp_start, source_fps):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("annotation timestamp origin and source fps must be finite")
    if source_fps <= 0:
        raise ValueError("annotation source fps must be positive")
    previous_index, previous_time = -1, -math.inf
    for index, timestamp in zip(indices, timestamps):
        if type(index) is not int or index < 0 or index <= previous_index:
            raise ValueError("annotation source indices must be increasing nonnegative integers")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp <= previous_time:
            raise ValueError("annotation source timestamps must be finite and strictly increasing")
        if not math.isclose(timestamp, timestamp_start + index / source_fps, rel_tol=0., abs_tol=1e-6):
            raise ValueError("annotation source timestamps do not match the declared uniform grid")
        previous_index, previous_time = index, timestamp


def interpolate_gripper_target_pairs(lower, upper, lower_valid, upper_valid, fractions):
    """Interpolate targets on an already identity-checked single-episode grid.

    Exact source timestamps use that row's validity alone. Interior timestamps
    require both bracketing targets to be valid on the same side; otherwise the
    returned flag is false and the caller must use its original action value.
    The caller owns episode boundaries and source timestamp/index validation.
    """
    size = len(fractions)
    if any(len(rows) != size for rows in (lower, upper, lower_valid, upper_valid)):
        raise ValueError("interpolation input frame counts differ")
    values, flags = [], []
    for left, right, left_flags, right_flags, fraction in zip(lower, upper, lower_valid, upper_valid, fractions):
        if any(len(row) != 2 for row in (left, right, left_flags, right_flags)):
            raise ValueError("gripper interpolation requires two-column rows")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("interpolation fraction must be finite and in [0,1]")
        row, accepted = [0., 0.], [False, False]
        for col in range(2):
            if type(left_flags[col]) is not bool or type(right_flags[col]) is not bool:
                raise ValueError("target validity must be boolean")
            for value, valid in ((left[col], left_flags[col]), (right[col], right_flags[col])):
                if valid and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("valid gripper target must be finite and in [0,1]")
            if fraction == 0 and left_flags[col]:
                row[col], accepted[col] = float(left[col]), True
            elif fraction == 1 and right_flags[col]:
                row[col], accepted[col] = float(right[col]), True
            elif 0 < fraction < 1 and left_flags[col] and right_flags[col]:
                row[col], accepted[col] = float(left[col]) + fraction * (float(right[col]) - float(left[col])), True
        values.append(row)
        flags.append(accepted)
    return values, flags


def apply_gripper_targets(actions, target, valid):
    """Return copied 20-D actions with valid left/right targets at 9/19.

    Call on the same timestamp grid *after* constructing physical observations.
    Resampling to a different training grid is the external loader's responsibility.
    """
    if len(actions) != len(target) or len(actions) != len(valid):
        raise ValueError("action/target/valid frame counts differ")
    result = [list(row) for row in actions]
    for row, values, flags in zip(result, target, valid):
        if len(row) != 20 or len(values) != 2 or len(flags) != 2:
            raise ValueError("expected 20-D actions and two-sided targets")
        for col, index in enumerate((9, 19)):
            if not isinstance(flags[col], bool):
                raise ValueError("validity must be boolean")
            if flags[col]:
                value = values[col]
                if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid gripper target")
                row[index] = value
    return result
