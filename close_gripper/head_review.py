"""Identity and coverage contracts for native main-view semantic observations."""
import math

from .contracts import validate_episode
from .pipeline import episode_fingerprint

HEAD_CAMERAS = ("observation.images.cam_head_left", "observation.images.cam_head_right")
REVIEW_SCHEMA = "close_gripper.head_visual_review.v1"


def validate_head_review(episode, review, frames):
    """Validate provenance/coverage; this does not certify visual accuracy."""
    validate_episode(episode)
    if episode.get("vision_camera_group") != "head" or set(episode["cameras"]) - set(HEAD_CAMERAS):
        raise ValueError("main review requires an explicitly head-only episode")
    if (review.get("schema") != REVIEW_SCHEMA or review.get("episode_id") != episode["episode_id"]
            or review.get("episode_sha256") != episode_fingerprint(episode)):
        raise ValueError("main review identity or input hash differs")
    if (review.get("camera_group") != "head" or review.get("review_source") != "model"
            or review.get("review_method") != "codex_native_vision" or review.get("confirmed") is not False):
        raise ValueError("main review requires non-human native vision provenance")
    size = len(episode["timestamps"])
    seen, identities = set(), set()
    for frame in frames:
        camera, i = frame.get("camera"), frame.get("frame_index")
        if camera not in HEAD_CAMERAS or camera not in episode["cameras"]:
            raise ValueError("main evidence cannot contain wrist or unavailable cameras")
        if type(i) is not int or not 0 <= i < size or (camera, i) in identities:
            raise ValueError("invalid or duplicate main evidence frame")
        identities.add((camera, i))
        timestamp = episode["timestamps"][i]
        expected = episode["cameras"][camera]["from_timestamp"] + timestamp
        if any(type(frame.get(k)) not in (int, float) or not math.isfinite(frame[k])
               or abs(frame[k] - v) > 1e-6 for k, v in
               (("timestamp", timestamp), ("video_timestamp", expected))):
            raise ValueError("main evidence timing differs from metadata")
        digest = frame.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("main evidence requires an image digest")
        seen.add(i)
    sides = review.get("sides", {})
    if set(sides) != set(episode["valid_sides"]):
        raise ValueError("main review must explicitly cover each effective arm")
    for value in sides.values():
        indices = value.get("reviewed_frame_indices", [])
        if (any(type(i) is not int for i in indices) or sorted(indices) != list(range(size))
                or not set(indices) <= seen):
            raise ValueError("dense main review requires actual evidence for every frame")
        cursor = 0
        for segment in value.get("segments", []):
            a, b = segment.get("start"), segment.get("end")
            if (type(a) is not int or type(b) is not int or a != cursor or not a <= b < size
                    or segment.get("state") not in ("held", "free", "uncertain")
                    or not isinstance(segment.get("note"), str) or not segment["note"].strip()):
                raise ValueError("main segments must cover every frame exactly once with an observation")
            cursor = b + 1
        if cursor != size:
            raise ValueError("main segments do not cover the full episode")
    return review
