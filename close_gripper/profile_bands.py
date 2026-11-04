"""Generalized matching ranges, kept separate from calibrated evidence bands."""
import copy
import hashlib
import json
import math

POLICY_MODE = "merge_expand_excluding_free_v1"


def validate_matching_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != {
            "mode", "expansion_margin", "pointwise_semantics_verified"}
            or policy["mode"] != POLICY_MODE
            or policy["pointwise_semantics_verified"] is not False):
        raise ValueError("unsupported matching policy or false pointwise evidence claim")
    margin = policy["expansion_margin"]
    if (type(margin) not in (int, float) or not math.isfinite(margin)
            or not 0 <= margin <= .05):
        raise ValueError("matching expansion margin must be finite and between 0 and 0.05")


def derive_matching_bands(profile, margin):
    """Merge calibrated anchors inside each free/conflict-free component.

    Closed-set blockers split the number line before any merging or padding.
    Thus expansion cannot jump over even a very narrow known free interval.
    New ranges carry anchor ids, never a claim of pointwise visual support.
    """
    if not profile["automatic_use"]:
        return []
    blockers = sorted((b["lower"], b["upper"])
                      for key in ("free_bands", "interval_vision_bands", "open_bands")
                      for b in profile.get(key, []))
    merged = []
    for lo, hi in blockers:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    components, start = [], 0.
    for lo, hi in merged:
        end = math.nextafter(lo, -math.inf)
        if start <= end:
            components.append((start, end))
        start = math.nextafter(hi, math.inf)
    if start <= 1:
        components.append((start, 1.))
    output = []
    for left, right in components:
        anchors = sorted((b for b in profile["closed_bands"]
                          if left <= b["lower"] <= b["upper"] <= right),
                         key=lambda b: (b["lower"], b["upper"], b["id"]))
        if not anchors:
            continue
        low = max(left, min(b["lower"] for b in anchors) - margin)
        high = min(right, max(b["upper"] for b in anchors) + margin)
        output.append(dict(id=f"matching_{len(output):02d}", lower=low, upper=high,
                           center=(low + high) / 2,
                           basis_closed_band_ids=[b["id"] for b in anchors],
                           pointwise_semantics_verified=False))
    if sum(len(b["basis_closed_band_ids"]) for b in output) != len(profile["closed_bands"]):
        raise ValueError("calibrated anchor intersects a blocked matching range")
    return output


def matching_closed_bands(profile):
    return profile.get("matching_bands", profile["closed_bands"])


def build_matching_catalog(catalog, *, margin, version):
    """Derive a separate catalog; all calibration and source evidence survives."""
    from .task_profiles import validate_catalog
    validate_catalog(catalog)
    if "matching_policy" in catalog:
        raise ValueError("derive matching ranges from the original calibration, not an already expanded catalog")
    policy = dict(mode=POLICY_MODE, expansion_margin=margin, pointwise_semantics_verified=False)
    validate_matching_policy(policy)
    result = copy.deepcopy(catalog)
    result.update(version=version, matching_policy=policy,
                  matching_basis=dict(catalog_version=catalog["version"],
                      catalog_semantic_sha256=hashlib.sha256(json.dumps(
                          catalog, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()))
    for task in result["tasks"]:
        for profile in task["sides"].values():
            profile["matching_bands"] = derive_matching_bands(profile, margin)
    validate_catalog(result)
    return result
