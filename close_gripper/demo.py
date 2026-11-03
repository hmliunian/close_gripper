"""Deterministic synthetic examples, explicitly not visual ground truth."""


def examples():
    from .core import detect_candidates
    cases = []
    # Closing: 90 -> 70 -> 50 -> 30 mm, then object-blocked plateau.
    # Brief opening during a synthetic, continuously confirmed hold.
    a = [90, 90, 70, 50, 30] + [30] * 10 + [42, 30] + [30] * 14
    # A full grasp/release cycle; release timing is not approved for training.
    b = [90, 90, 70, 50, 30] + [30] * 14 + [45, 60, 75, 90] + [90] * 8
    for name, left, right, category in [
        ("left_misopen", a, [65] * len(a), "suspected_misopen"),
        ("right_release", [0] * len(b), b, "normal_release"),
        ("bilateral_misopen", a, a, "suspected_misopen"),
        ("no_grasp", a, [65] * len(a), "no_grasp"),
        ("untriggered_hold", [30] * len(a), [65] * len(a), "uncertain"),
    ]:
        state = [[0.] * 20 for _ in left]
        for i, row in enumerate(state):
            row[9], row[19] = left[i] / 95, right[i] / 95
        episode = dict(episode_id=name, timestamps=[i / 10 for i in range(len(left))],
                       state=state, valid_sides=["left", "right"],
                       cameras={f"observation.images.cam_{side}_wrist_left":
                                {"valid": True, "synthetic": True} for side in ("left", "right")},
                       identity={"source": "synthetic", "task": name},
                       frame_indices=list(range(len(left))))
        decisions = []
        for candidate in detect_candidates(episode):
            release = category == "normal_release"
            decisions.append(dict(candidate_id=candidate["candidate_id"], side=candidate["side"],
                category=category, grasp_before="held" if category != "no_grasp" else "not_held",
                grasp_during="held" if not release else "uncertain",
                grasp_after="not_held" if release or category == "no_grasp" else "held",
                hold_start=4, hold_end=18 if release else len(left)-1,
                release_end=22 if release else None, evidence_frames=[4, candidate["start"], candidate["end"]],
                confirmed=True, review_source="synthetic", slip=False,
                reason="Synthetic contract demonstration; not human or model evidence"))
        cases.append((episode, decisions))
    return cases
