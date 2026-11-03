"""Task-specific opening profiles and pure numerical interval checks.

A matched interval is numerical evidence, not a per-episode grasp annotation.
This module neither calls a visual model nor changes observations/targets; the
existing full-cycle evidence and fitting contracts remain separate.
"""

import copy
import math
from statistics import median

from .contracts import SIDE_INDICES, validate_episode


SCHEMA = "close_gripper.task_profiles.v1"
UNITS = "normalized_opening_0_1"
PROFILE_STATUSES = {"numeric_only", "visual_reviewed", "needs_review", "unsupported"}
_EPS = 1e-9


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # JSON integers may be too large to convert to a finite machine float.
        return False


def _name(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _bands(profile, key):
    bands = profile.get(key)
    if not isinstance(bands, list):
        raise ValueError(key + " must be a list")
    identifiers = set()
    for band in bands:
        if not isinstance(band, dict) or not _name(band.get("id")):
            raise ValueError("each band requires an id")
        if band["id"] in identifiers:
            raise ValueError("duplicate band id in " + key)
        identifiers.add(band["id"])
        values = [band.get(k) for k in ("lower", "center", "upper")]
        if any(not _number(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("band values must be finite normalized openings")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError("band must satisfy lower <= center <= upper")
    return bands


def _overlap(first, second):
    # Bands are closed sets. A shared endpoint is ambiguous as well.
    return max(first["lower"], second["lower"]) <= min(first["upper"], second["upper"])


def validate_catalog(catalog):
    """Validate the catalog without reading files or mutating the input.

    Ambiguous measured bands may be retained in disabled profiles. Automatic
    matching requires explicit visual-review provenance and disjoint open/closed
    modes, but still does not certify an individual episode's grasp semantics.
    """
    if not isinstance(catalog, dict) or catalog.get("schema") != SCHEMA:
        raise ValueError("unsupported task profile schema")
    if not _name(catalog.get("version")) or catalog.get("units") != UNITS:
        raise ValueError("catalog version and normalized opening units are required")
    for key in ("minimum_hold_s", "maximum_sample_gap_s", "maximum_plateau_range"):
        value = catalog.get(key)
        if not _number(value) or value <= 0:
            raise ValueError(key + " must be positive and finite")
    if catalog["maximum_plateau_range"] > 1:
        raise ValueError("maximum_plateau_range must not exceed one")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("catalog requires a nonempty task list")
    names = {}
    for task in tasks:
        if not isinstance(task, dict) or not _name(task.get("task")):
            raise ValueError("canonical task name is required")
        aliases = task.get("aliases", [])
        if not isinstance(aliases, list) or any(not _name(a) for a in aliases):
            raise ValueError("task aliases must be a list of nonempty names")
        for name in [task["task"], *aliases]:
            if name in names:
                raise ValueError("task or alias collision: " + name)
            names[name] = task["task"]
        sides = task.get("sides")
        if not isinstance(sides, dict) or not sides or set(sides) - set(SIDE_INDICES):
            raise ValueError("task must contain explicit left/right side profiles")
        for profile in sides.values():
            if (not isinstance(profile, dict) or not isinstance(profile.get("status"), str)
                    or profile["status"] not in PROFILE_STATUSES):
                raise ValueError("invalid task profile status")
            if type(profile.get("automatic_use")) is not bool:
                raise ValueError("automatic_use must be an explicit boolean")
            if not isinstance(profile.get("provenance"), dict) or not isinstance(profile.get("counts"), dict):
                raise ValueError("profile provenance and counts must be objects")
            if any(type(v) is not int or v < 0 for v in profile["counts"].values()):
                raise ValueError("profile counts must be nonnegative integers")
            closed, opened = _bands(profile, "closed_bands"), _bands(profile, "open_bands")
            if not profile["automatic_use"]:
                continue
            if profile["status"] != "visual_reviewed":
                raise ValueError("automatic profiles require visual_reviewed status")
            if profile["provenance"].get("review_source") not in ("model", "human"):
                raise ValueError("automatic profiles require explicit visual review source")
            if not closed or not opened:
                raise ValueError("automatic profiles require measured closed and open bands")
            for bands in (closed, opened):
                for i, band in enumerate(bands):
                    if any(_overlap(band, other) for other in bands[i + 1:]):
                        raise ValueError("automatic profile modes must be disjoint")
            if any(_overlap(c, o) for c in closed for o in opened):
                raise ValueError("automatic closed and open bands must be disjoint")
            if any(not any(o["lower"] > c["upper"] for o in opened) for c in closed):
                raise ValueError("each closed band needs a higher observed open band")
    ambiguous = catalog.get("ambiguous_aliases", {})
    if not isinstance(ambiguous, dict):
        raise ValueError("ambiguous_aliases must be an object")
    canonical = {t["task"] for t in tasks}
    for alias, candidates in ambiguous.items():
        if not _name(alias) or alias in names:
            raise ValueError("ambiguous alias must not also resolve to a task")
        if (not isinstance(candidates, list) or len(candidates) < 2
                or any(not _name(c) or c not in canonical for c in candidates)
                or len(candidates) != len(set(candidates))):
            raise ValueError("ambiguous alias requires at least two distinct canonical tasks")


def _reject(reason, **extra):
    return dict(status="rejected", reason=reason, **extra)


def _task_labels(identity):
    """Prefer source labels; a present but malformed source is never bypassed."""
    source = identity.get("source_metadata")
    if "source_metadata" in identity and not isinstance(source, dict):
        return None, "invalid_source_metadata"
    if source is not None and "source_tasks" in source:
        labels, origin = source["source_tasks"], "source_metadata.source_tasks"
    elif "source_tasks" in identity:
        labels, origin = identity["source_tasks"], "source_tasks"
    elif "tasks" in identity:
        labels, origin = identity["tasks"], "tasks"
    elif "task" in identity:
        labels, origin = [identity["task"]], "task"
    else:
        return None, "missing_task_identity"
    if not isinstance(labels, list) or not labels or any(not _name(x) for x in labels):
        return None, "invalid_task_identity"
    if len(labels) != 1:
        return None, "multiple_task_identity"
    return labels[0], origin


def resolve_profile(catalog, episode, side):
    """Resolve one valid side with exact canonical names or catalogued aliases.

    Resolution permits inspection of disabled profiles. Use
    :func:`match_hold_interval` to apply their explicit automatic-use policy.
    The returned profile is copied so a caller cannot mutate the catalog.
    """
    validate_catalog(catalog)
    if not isinstance(side, str) or side not in SIDE_INDICES:
        return _reject("unknown_side", side=side)
    if not isinstance(episode, dict):
        return _reject("invalid_episode_identity", side=side)
    sides = episode.get("valid_sides")
    if (not isinstance(sides, list) or any(not isinstance(s, str) or s not in SIDE_INDICES for s in sides)
            or len(sides) != len(set(sides))):
        return _reject("invalid_valid_sides", side=side)
    if side not in sides:
        return _reject("invalid_side", side=side)
    identity = episode.get("identity")
    if not isinstance(identity, dict):
        return _reject("missing_task_identity", side=side)
    label, origin = _task_labels(identity)
    if label is None:
        return _reject(origin, side=side)
    if label in catalog.get("ambiguous_aliases", {}):
        return _reject("ambiguous_task_alias", side=side, task_label=label, identity_source=origin,
                       candidate_tasks=list(catalog["ambiguous_aliases"][label]))
    task = next((t for t in catalog["tasks"] if label in [t["task"], *t.get("aliases", [])]), None)
    if task is None:
        return _reject("unknown_task", side=side, task_label=label, identity_source=origin)
    if side not in task["sides"]:
        return _reject("missing_side_profile", task=task["task"], side=side, identity_source=origin)
    return dict(status="resolved", task=task["task"], side=side, task_label=label,
                identity_source=origin, catalog_version=catalog["version"],
                profile=copy.deepcopy(task["sides"][side]))


def match_hold_interval(catalog, episode, side, start, end):
    """Check one inclusive interval against a task profile and time thresholds.

    All samples must belong to the same closed mode, remain stable and have no
    excessive timestamp gaps. The confirmation timestamp does not shift the
    hold start: offline callers can retain ``hold_start=start``. No trajectory
    is rewritten and no visual/no-grasp claim is made by this numerical check.
    """
    resolved = resolve_profile(catalog, episode, side)
    base = dict(matched=False, grasp_semantics_verified=False, generates_targets=False,
                hold_start=start, hold_end=end, side=side)
    if resolved["status"] != "resolved":
        return dict(base, **{k: v for k, v in resolved.items() if k != "side"})
    profile = resolved["profile"]
    base.update(task=resolved["task"], catalog_version=catalog["version"],
                profile_status=profile["status"], review_source=profile["provenance"].get("review_source"))
    if not profile["automatic_use"]:
        return _reject("profile_not_automatic", **base)
    try:
        validate_episode(episode)
    except OverflowError as exc:
        raise ValueError("episode numbers must be finite") from exc
    times = episode["timestamps"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(times):
        raise ValueError("hold interval requires valid inclusive frame indices")
    values = [row[SIDE_INDICES[side]] for row in episode["state"][start:end + 1]]
    duration = times[end] - times[start]
    gaps = [b - a for a, b in zip(times[start:end], times[start + 1:end + 1])]
    low, high = min(values), max(values)
    base.update(duration_s=duration, minimum_hold_s=catalog["minimum_hold_s"],
                observed_lower=low, observed_center=median(values), observed_upper=high,
                maximum_observed_gap_s=max(gaps, default=0.0))
    if duration + _EPS < catalog["minimum_hold_s"]:
        return _reject("insufficient_holding_duration", **base)
    if max(gaps, default=0.0) > catalog["maximum_sample_gap_s"] + _EPS:
        return _reject("timestamp_gap", **base)
    if high - low > catalog["maximum_plateau_range"] + _EPS:
        return _reject("unstable_plateau", **base)
    matching = [b for b in profile["closed_bands"] if b["lower"] <= low and high <= b["upper"]]
    if len(matching) != 1:
        return _reject("outside_single_closed_band", **base)
    confirmation = next(i for i in range(start, end + 1)
                        if times[i] - times[start] + _EPS >= catalog["minimum_hold_s"])
    base.update(matched=True, status="matched", reason="task_profile_numeric_match",
                closed_band_id=matching[0]["id"], confirmation_frame=confirmation,
                confirmation_time=times[confirmation])
    return base


def main(argv=None):
    """Read-only catalog validation, inspection and interval matching CLI."""
    import argparse
    from collections import Counter
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "inspect", "match"):
        command = commands.add_parser(name)
        command.add_argument("catalog", type=Path)
        if name == "inspect":
            command.add_argument("--task", required=True)
        if name in ("inspect", "match"):
            command.add_argument("--side", choices=tuple(SIDE_INDICES), required=True)
        if name == "match":
            command.add_argument("--episode", type=Path, required=True)
            command.add_argument("--start", type=int, required=True)
            command.add_argument("--end", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        validate_catalog(catalog)
        if args.command == "validate":
            profiles = [p for t in catalog["tasks"] for p in t["sides"].values()]
            result = dict(status="valid", schema=catalog["schema"], version=catalog["version"],
                          task_count=len(catalog["tasks"]), side_profile_count=len(profiles),
                          automatic_profile_count=sum(p["automatic_use"] for p in profiles),
                          status_counts=dict(Counter(p["status"] for p in profiles)))
        elif args.command == "inspect":
            episode = dict(valid_sides=[args.side], identity={"tasks": [args.task]})
            result = resolve_profile(catalog, episode, args.side)
        else:
            episode = json.loads(args.episode.read_text(encoding="utf-8"))
            result = match_hold_interval(catalog, episode, args.side, args.start, args.end)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if result["status"] == "rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
