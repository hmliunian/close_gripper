"""Explicit, local loader integration contract; observations are never replaced."""
import math


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
