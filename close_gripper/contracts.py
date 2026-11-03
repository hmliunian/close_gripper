"""Dependency-free input contracts. All frame references are episode-local indices."""

import math

SIDE_INDICES = {"left": 9, "right": 19}
SIDE_CAMERAS = {
    "left": ("cam_left_wrist_left", "cam_left_wrist_right"),
    "right": ("cam_right_wrist_left", "cam_right_wrist_right"),
}

# Experimental settings, not calibrated production thresholds.
DEFAULT_CONFIG = {
    "config_version": "experimental-v1",
    "width_mm": 95.0,
    "opening_min_mm": 3.0,
    "opening_min_speed_mm_s": 5.0,
    "opening_noise_mm": 0.25,
    "opening_gap_s": 0.2,
    "zero_tolerance_mm": 1.0,
    "platform_window_s": 0.3,
    "platform_tolerance_mm": 2.0,
    "closing_lookback_s": 1.5,
    "closing_min_speed_mm_s": 5.0,
    "closing_max_speed_mm_s": 1500.0,
    "speed_disagreement_ratio": 0.5,
    "closing_policy": "fixed_timing",
    "release_policy": "review",
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_episode(episode):
    """Raise ValueError before processing malformed input; never repair the time grid."""
    if not isinstance(episode, dict) or not str(episode.get("episode_id", "")):
        raise ValueError("episode_id is required")
    times, states = episode.get("timestamps"), episode.get("state")
    if not isinstance(times, list) or not times or not isinstance(states, list) or len(times) != len(states):
        raise ValueError("timestamps and state must have equal nonzero lengths")
    if any(not _number(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be finite, nonnegative and strictly increasing")
    sides = episode.get("valid_sides")
    if not isinstance(sides, list) or len(set(sides)) != len(sides) or any(s not in SIDE_INDICES for s in sides):
        raise ValueError("valid_sides must explicitly list unique valid sides")
    for row in states:
        if not isinstance(row, list) or len(row) != 20:
            raise ValueError("each state must have exactly 20 elements")
        for side in sides:
            value = row[SIDE_INDICES[side]]
            if not _number(value) or not 0 <= value <= 1:
                raise ValueError("valid gripper widths must be finite normalized values in [0,1]")
    if not isinstance(episode.get("cameras", {}), dict) or not isinstance(episode.get("identity", {}), dict):
        raise ValueError("cameras and identity must be objects")


def validate_decision(decision, episode, candidate=None):
    """Validate provenance and evidence bounds; syntactic validity is not visual truth."""
    if not isinstance(decision, dict):
        raise ValueError("decision must be an object")
    if decision.get("side") not in episode["valid_sides"]:
        raise ValueError("decision references invalid side")
    if candidate and (decision.get("candidate_id") != candidate["candidate_id"] or decision["side"] != candidate["side"]):
        raise ValueError("decision candidate identity or side mismatch")
    if decision.get("category") not in ("normal_release", "suspected_misopen", "no_grasp", "uncertain"):
        raise ValueError("invalid event category")
    for key in ("grasp_before", "grasp_during", "grasp_after"):
        if decision.get(key) not in ("held", "not_held", "uncertain"):
            raise ValueError("invalid grasp state")
    if type(decision.get("confirmed")) is not bool or type(decision.get("slip")) is not bool:
        raise ValueError("confirmed and slip must be booleans")
    if decision.get("review_source") not in ("human", "model", "synthetic"):
        raise ValueError("review_source is required")
    if decision["confirmed"] and decision["review_source"] == "model":
        raise ValueError("model output cannot confirm its own interpretation")
    size = len(episode["timestamps"])
    for key in ("hold_start", "hold_end", "release_end"):
        value = decision.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < size):
            raise ValueError("invalid boundary frame: " + key)
    a, b = decision.get("hold_start"), decision.get("hold_end")
    if (a is None) != (b is None) or (a is not None and a > b):
        raise ValueError("invalid holding interval")
    evidence = decision.get("evidence_frames")
    if not isinstance(evidence, list) or any(type(i) is not int or not 0 <= i < size for i in evidence):
        raise ValueError("evidence_frames must contain valid frame indices")
    if decision.get("confirmed") and not evidence:
        raise ValueError("confirmed decision requires evidence frames")
    if not isinstance(decision.get("reason", ""), str):
        raise ValueError("reason must be text")


def validated_config(config=None):
    result = dict(DEFAULT_CONFIG)
    if config:
        # The numerical result is persisted. Never propagate transport options,
        # arbitrary nested objects, credential paths, or unknown secrets here.
        result.update({key: config[key] for key in DEFAULT_CONFIG if key in config})
    for key in DEFAULT_CONFIG:
        if isinstance(DEFAULT_CONFIG[key], (float, int)):
            if not _number(result[key]) or result[key] <= 0:
                raise ValueError("configuration must be positive and finite: " + key)
    if result["width_mm"] != 95.0:
        raise ValueError("dataset normalization is fixed at 95 mm")
    if result["release_policy"] != "review":
        raise ValueError("release_policy must remain review pending sample adjudication")
    if result["closing_policy"] not in ("fixed_timing", "original_speed"):
        raise ValueError("closing_policy must be fixed_timing or original_speed")
    return result
