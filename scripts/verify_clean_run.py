#!/usr/bin/env python3
"""Independently audit saved full-cycle artifacts without rerunning the algorithm.

Only JSON, saved evidence images and an optional frozen-input manifest are read.
No business-package imports, source datasets, video decoders or model calls are
used. A passing audit is artifact consistency, not visual truth or model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SIDES = (("left", 9), ("right", 19))
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
FIELDS = {"target": "annotation.gripper_target", "valid": "annotation.gripper_target_valid",
          "source": "annotation.gripper_target_source", "phase": "annotation.gripper_phase"}
COVERAGE_DEFINITION = (
    "complete_review means every candidate has a structured judgment and their input context windows "
    "cover the episode. It does not mean every frame was viewed, cleaned, or independently verified; "
    "no_grasp concerns its candidate rather than every event visible in that context.")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_child(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path must be relative without parent traversal")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its root")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root):
        raise ValueError("artifact path traverses a symlink")
    return path


def close(a, b):
    return (type(a) in (float, int) and type(b) in (float, int) and
            math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9)


def gaps(flags):
    ranges = []
    for i, flag in enumerate(flags):
        if not flag:
            if ranges and ranges[-1][1] == i - 1:
                ranges[-1][1] = i
            else:
                ranges.append([i, i])
    return ranges


class Audit:
    def __init__(self):
        self.errors, self.limitations, self.files = [], [], {}
        self.checks = Counter()

    def check(self, passed, code, location):
        self.checks[code] += 1
        if not passed:
            self.errors.append(dict(check=code, location=location))
        return bool(passed)

    def limit(self, message):
        if message not in self.limitations:
            self.limitations.append(message)

    def read(self, path):
        data = Path(path).read_bytes()
        self.files[str(Path(path).resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def hash_file(self, path):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.files[str(Path(path).resolve())] = digest
        return digest


def input_snapshots(manifest, audit):
    if manifest is None:
        audit.limit("No input manifest supplied: preservation against the pre-run observation snapshot is unverified.")
        return {}
    path = Path(manifest).resolve(strict=True)
    data = audit.read(path)
    entries = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("input manifest needs an episode list")
    snapshots = {}
    for row in entries:
        if "episode_cache" not in row:
            audit.limit("Manifest entries without episode_cache cannot establish unchanged pre-run observations.")
            continue
        episode = audit.read(safe_child(path.parent, row["episode_cache"]))
        identifier = episode["episode_id"]
        audit.check(row.get("episode_id", identifier) == identifier, "manifest_episode_identity", identifier)
        audit.check(row.get("episode_sha256") == fingerprint(episode), "manifest_input_fingerprint", identifier)
        audit.check(identifier not in snapshots, "manifest_unique_episode", identifier)
        snapshots[identifier] = episode
    return snapshots


def check_implementation(root, summary, audit):
    metadata = root / "implementation.json"
    if not metadata.is_file():
        audit.limit("implementation.json is absent: the historical run's implementation snapshot SHA cannot be verified.")
        return dict(status="not_available", files=0)
    record = audit.read(metadata)
    names = record.get("files")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or Path(n).name != n or not n.endswith(".py") for n in names):
        raise ValueError("invalid implementation file inventory")
    audit.check(len(names) == len(set(names)), "implementation_unique_files", "implementation.json")
    actual = sorted(p.name for p in (root / "implementation").glob("*.py"))
    audit.check(sorted(names) == actual, "implementation_exact_inventory", "implementation")
    digest = hashlib.sha256()
    for name in sorted(names):
        file = safe_child(root / "implementation", name)
        audit.hash_file(file)
        digest.update(name.encode())
        digest.update(file.read_bytes())
    value = digest.hexdigest()
    audit.check(value == record.get("sha256"), "implementation_snapshot_sha256", "implementation.json")
    audit.check(value == summary.get("implementation_sha256"), "implementation_summary_sha256", "summary.json")
    audit.check(summary.get("implementation_changed_during_run") is False,
                "implementation_not_reported_changed", "summary.json")
    return dict(status="checked", sha256=value, files=len(names))


def check_decision(decision, candidate, audit, location):
    audit.check(decision.get("candidate_id") == candidate["candidate_id"] and
                decision.get("side") == candidate["side"], "decision_identity", location)
    source = decision.get("review_source")
    audit.check(source in ("model", "human", "synthetic"), "decision_source", location)
    if source == "model":
        audit.check(decision.get("confirmed", False) is False and not any(decision.get(k) for k in
                    ("reviewer", "reviewed_at", "review_note")), "model_not_human", location)
    elif source == "human":
        audit.check(decision.get("confirmed") is True and all(isinstance(decision.get(k), str) and decision[k].strip()
                    for k in ("reviewer", "reviewed_at", "review_note")), "human_review_provenance", location)
    evidence = decision.get("evidence_frames")
    audit.check(isinstance(evidence, list) and bool(evidence) and all(type(i) is int and
                candidate["start"] <= i <= candidate["end"] for i in evidence), "decision_evidence_window", location)
    category = decision.get("category")
    audit.check(category in ("grasp_hold", "grasp_release", "no_grasp", "uncertain"), "decision_category", location)
    if category in ("no_grasp", "uncertain"):
        audit.check(all(decision.get(k) is None for k in ("hold_start", "hold_end", "release_end", "pregrasp_start"))
                    and decision.get("continuous_grasp") is False and decision.get("pregrasp_clear") is False,
                    "negative_decision_consistency", location)


def check_evidence(directory, episode, candidates, responses, frames, decisions, audit):
    size = len(episode["timestamps"])
    by_id = {c["candidate_id"]: c for c in candidates}
    groups = {}
    for ordinal, candidate in enumerate(candidates):
        groups[f"frames_{ordinal:03d}"] = candidate
    images = {c["candidate_id"]: [] for c in candidates}
    for frame in frames:
        location = str(frame.get("path", "missing_image_path"))
        path = Path(location)
        if not path.is_absolute():
            path = directory / path
        if not audit.check(path.resolve().is_relative_to(directory.resolve()), "image_inside_episode", location):
            continue
        file = safe_child(directory, path.relative_to(directory).as_posix())
        audit.check(audit.hash_file(file) == frame.get("sha256"), "image_sha256", location)
        candidate = groups.get(path.parent.name)
        if not audit.check(candidate is not None, "image_candidate_group", location):
            continue
        images[candidate["candidate_id"]].append(frame)
        side, camera, index = frame.get("side"), frame.get("camera"), frame.get("frame_index")
        allowed = {f"observation.images.cam_{candidate['side']}_wrist_{eye}" for eye in ("left", "right")}
        audit.check(side == candidate["side"] and camera in allowed and camera in episode["cameras"],
                    "image_same_side_camera", location)
        if not audit.check(type(index) is int and 0 <= index < size and candidate["start"] <= index <= candidate["end"],
                           "image_frame_window", location):
            continue
        audit.check(frame.get("source_frame_index") == episode.get("frame_indices", list(range(size)))[index],
                    "image_source_frame_index", location)
        audit.check(close(frame.get("timestamp"), episode["timestamps"][index]), "image_episode_timestamp", location)
        spec = episode["cameras"].get(camera)
        if spec:
            expected = spec["from_timestamp"] + episode["timestamps"][index]
            audit.check(close(frame.get("video_timestamp"), expected) and
                        spec["from_timestamp"] <= expected < spec["to_timestamp"], "image_video_offset", location)
            audit.check(frame.get("video_path") == spec["path"], "image_source_video", location)
    totals = dict(attempts=0, service_responses=0, successful_model_decisions=0,
                  attempts_without_saved_service_response=0, attempts_without_reported_usage=0,
                  usage_reported_requests=0, usage={k: 0 for k in TOKENS}, failures={})
    response_ids = set()
    for response in responses:
        identifier = response.get("candidate_id")
        location = str(directory.name) + "/response/" + str(identifier)
        audit.check(identifier not in response_ids, "one_response_per_candidate", location)
        response_ids.add(identifier)
        candidate = by_id.get(identifier)
        if not audit.check(candidate is not None, "response_candidate_identity", location):
            continue
        submitted = response.get("submitted_frame_indices")
        actual = images[identifier]
        audit.check(submitted == sorted({f["frame_index"] for f in actual}) and bool(submitted),
                    "response_submitted_images", location)
        audit.check(response.get("evidence_images") == len(actual), "response_image_count", location)
        attempts = response.get("attempts", 0)
        if audit.check(type(attempts) is int and 0 <= attempts <= 1, "response_attempt_count", location):
            totals["attempts"] += attempts
        usage = response.get("usage", {})
        audit.check(isinstance(usage, dict), "response_usage_object", location)
        if isinstance(usage, dict) and any(key in usage for key in TOKENS):
            totals["usage_reported_requests"] += 1
            for key in TOKENS:
                if key in usage and audit.check(type(usage[key]) is int and usage[key] >= 0, "response_usage_integer", location):
                    totals["usage"][key] += usage[key]
        raw = response.get("raw_response")
        if isinstance(raw, dict):
            totals["service_responses"] += 1
            audit.check(raw.get("usage", {}) == usage, "usage_matches_raw_response", location)
        elif type(attempts) is int and attempts > 0:
            totals["attempts_without_saved_service_response"] += attempts
        if type(attempts) is int and attempts > 0 and not (isinstance(usage, dict) and any(k in usage for k in TOKENS)):
            totals["attempts_without_reported_usage"] += attempts
        if response.get("status") != "ok":
            reason = str(response.get("error") or "unknown_error")
            totals["failures"][reason] = totals["failures"].get(reason, 0) + 1
        if response.get("status") == "ok":
            totals["successful_model_decisions"] += 1
            d = response.get("decision", {})
            audit.check(attempts == 1 and d.get("review_source") == "model" and d.get("confirmed", False) is False,
                        "response_model_provenance", location)
            audit.check(d in decisions, "response_decision_saved", location)
            check_decision(d, candidate, audit, location)
            evidence = d.get("evidence_frames", [])
            audit.check(all(i in (submitted or []) for i in evidence), "evidence_was_submitted", location)
            audit.check(set(d.get("cameras", [])) == {f["camera"] for f in actual}, "decision_submitted_cameras", location)
            if isinstance(raw, dict) and "choices" in raw:
                choice = raw["choices"][0]
                content = choice["message"]["content"]
                raw_decision = json.loads(content) if isinstance(content, str) else content
                semantic = ("candidate_id", "side", "category", "hold_start", "hold_end", "release_end",
                            "pregrasp_start", "pregrasp_clear", "continuous_grasp", "slip")
                audit.check(all(raw_decision.get(k) == d.get(k) for k in semantic) and
                            sorted(raw_decision.get("evidence_frames", [])) == d.get("evidence_frames") and
                            raw_decision.get("reason", "")[:2000] == d.get("reason"),
                            "decision_matches_raw_response", location)
                audit.check(choice.get("finish_reason") in (None, "stop"), "response_complete", location)
            else:
                audit.limit("A successful model decision lacks the raw service response; remote-response provenance is incomplete.")
        if attempts:
            request_hash = response.get("request_hash")
            audit.check(isinstance(request_hash, str) and len(request_hash) == 64 and all(c in "0123456789abcdef" for c in request_hash),
                        "request_hash_present", location)
    return totals


def check_reused_evidence(root, directory, data, audit):
    """Bind reused model observations to their original run without counting calls twice."""
    episode, result = data["episode"], data["result"]
    metadata = directory / "reused_evidence.json"
    if result.get("evidence_type") != "reused_real_vlm":
        audit.check(not metadata.exists(), "reuse_requires_explicit_evidence_type", directory.name)
        return None
    record = audit.read(metadata)
    source = Path(record["source_run"]).resolve(strict=True)
    audit.check(source != root, "reuse_source_is_distinct", directory.name)
    old_directory = safe_child(source, record["source_episode_directory"])
    hashes = record["source_files_sha256"]
    required = {"episode.json", "candidates.json", "decisions.json", "review_template.json",
                "model_responses.json", "frames.json", "targets.json", "result.json"}
    audit.check(isinstance(hashes, dict) and required <= hashes.keys(), "reuse_source_file_inventory", directory.name)
    for name, digest in hashes.items():
        audit.check(audit.hash_file(safe_child(old_directory, name)) == digest, "reuse_original_file_sha256", name)
    original = {name: audit.read(safe_child(old_directory, name + ".json")) for name in
                ("episode", "candidates", "decisions", "review_template", "model_responses", "frames", "targets", "result")}
    for name in ("episode", "candidates", "decisions", "review_template"):
        audit.check(original[name] == data[name], "reused_" + name + "_unchanged", directory.name)
    digest = fingerprint(episode)
    audit.check(record.get("episode_sha256") == digest == original["targets"].get("episode_sha256"),
                "reuse_input_fingerprint", directory.name)
    audit.check(original["result"].get("evidence_type") == "real_vlm", "reuse_original_real_vlm", directory.name)
    audit.check(result.get("pipeline_issues", []) == original["result"].get("pipeline_issues", []),
                "reuse_preserves_failed_issues", directory.name)
    for destination, source_name in (("source_model_responses.json", "model_responses.json"), ("source_frames.json", "frames.json")):
        audit.check(audit.hash_file(directory / destination) == hashes.get(source_name), "reuse_copied_file_sha256", destination)
    copied = audit.read(directory / "source_model_responses.json")
    audit.check(copied == original["model_responses"], "reuse_responses_unchanged", directory.name)
    audit.check(audit.read(directory / "source_frames.json") == original["frames"], "reuse_original_frames_unchanged", directory.name)
    strip_paths = lambda frames: [{k: v for k, v in frame.items() if k != "path"} for frame in frames]
    audit.check(strip_paths(data["frames"]) == strip_paths(original["frames"]), "reuse_image_metadata_unchanged", directory.name)
    audit.check(data["model_responses"] == [], "reuse_has_no_new_model_responses", directory.name)
    counts = check_evidence(directory, episode, data["candidates"], copied, data["frames"], data["decisions"], audit)
    counts["source_run"] = str(source)
    counts["source_episode_directory"] = record["source_episode_directory"]
    return counts


def check_episode(root, row, snapshot, audit):
    directory = safe_child(root, row["directory"])
    data = {name: audit.read(safe_child(directory, name + ".json")) for name in
            ("episode", "targets", "result", "candidates", "decisions", "frames", "model_responses", "review_template")}
    episode, targets, result = data["episode"], data["targets"], data["result"]
    identifier = episode["episode_id"]
    size = len(episode["timestamps"])
    audit.check(row["episode_id"] == identifier == targets.get("episode_id"), "episode_identity", identifier)
    audit.check(size > 0 and len(episode["state"]) == size and all(len(s) == 20 for s in episode["state"]),
                "observation_shape", identifier)
    audit.check(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in episode["timestamps"]) and
                all(a < b for a, b in zip(episode["timestamps"], episode["timestamps"][1:])), "episode_time_order", identifier)
    digest = fingerprint(episode)
    audit.check(targets.get("episode_sha256") == digest == data["review_template"].get("episode_sha256"),
                "run_input_fingerprint", identifier)
    audit.check(data["review_template"].get("decisions") == data["decisions"], "review_template_decisions", identifier)
    if snapshot is None:
        audit.limit("Some episodes lack a matching pre-run input snapshot; their saved fingerprint is only internally cross-checked.")
    else:
        audit.check(episode == snapshot, "original_observations_match_input_snapshot", identifier)
    for key in ("identity", "timestamps", "frame_indices"):
        expected = episode.get(key, list(range(size)) if key == "frame_indices" else None)
        audit.check(targets.get(key) == expected, "target_" + key, identifier)
    for name, field in FIELDS.items():
        values = result[name]
        if not audit.check(isinstance(values, list) and len(values) == size and all(isinstance(r, list) and len(r) == 2 for r in values),
                           "two_column_" + name, identifier):
            raise ValueError("malformed target arrays")
        audit.check(targets.get(field) == values, "targets_result_" + name, identifier)
    audit.check(targets.get("method_version") == result["config"].get("config_version") and
                targets.get("evidence_type") == result.get("evidence_type"), "target_method_provenance", identifier)
    for key in ("preview_target", "preview_source", "preview_phase"):
        audit.check(len(result[key]) == size and all(len(r) == 2 for r in result[key]), "two_column_" + key, identifier)
    candidates, events = data["candidates"], result["events"]
    ids = [c["candidate_id"] for c in candidates]
    audit.check(len(ids) == len(set(ids)) and ids == [e.get("candidate_id") for e in events], "candidate_event_inventory", identifier)
    by_id = {c["candidate_id"]: c for c in candidates}
    owners = [[[] for _ in SIDES] for _ in range(size)]
    for event in events:
        location = event["candidate_id"]
        candidate = by_id[location]
        audit.check(type(candidate["start"]) is int and type(candidate["end"]) is int and
                    0 <= candidate["start"] <= candidate["end"] < size and candidate["side"] in episode["valid_sides"],
                    "candidate_window_side", location)
        audit.check(all(event.get(k) == candidate[k] for k in ("start", "end", "side")), "event_candidate_context", location)
        audit.check(type(event.get("reviewed")) is bool, "event_review_flag", location)
        if event.get("reviewed"):
            decision = event.get("decision", {})
            audit.check(decision in data["decisions"], "event_decision_saved", location)
            check_decision(decision, candidate, audit, location)
        if event.get("status") in ("applied", "preview"):
            a, b = event["modified_start"], event["modified_end"]
            if not audit.check(type(a) is int and type(b) is int and candidate["start"] <= a <= b <= candidate["end"],
                               "event_modified_interval", location):
                continue
            decision = event.get("decision", {})
            audit.check(event.get("reviewed") and decision.get("category") in ("grasp_hold", "grasp_release") and
                        decision.get("continuous_grasp") is True and decision.get("slip") is False,
                        "applied_grasp_evidence", location)
            audit.check(event.get("source") == decision.get("review_source"), "event_source", location)
            col = 0 if event["side"] == "left" else 1
            array = result["target"] if event["status"] == "applied" else result["preview_target"]
            changed = sum(abs(array[i][col] - episode["state"][i][SIDES[col][1]]) > 1e-12 for i in range(a, b + 1))
            audit.check(event.get("changed_frames") == changed, "event_changed_frames", location)
            if event["status"] == "applied":
                audit.check(event.get("source") in ("model", "human"), "no_synthetic_training_labels", location)
                contact, hold_end = event.get("contact_frame"), event.get("hold_end")
                audit.check(type(contact) is int and type(hold_end) is int and a <= contact <= hold_end <= b,
                            "applied_hold_boundaries", location)
                for i in range(a, b + 1):
                    phase = ("release" if event.get("release_start") is not None and i >= event["release_start"] else
                             "hold" if i >= event["contact_frame"] else
                             "closing" if event.get("closing_start") is not None and i >= event["closing_start"] else "pregrasp")
                    owners[i][col].append((event["source"], phase))
    changed_frames, valid_frames, source_counts = 0, [0, 0], Counter()
    for i in range(size):
        changed = False
        for col, (side, state_index) in enumerate(SIDES):
            valid, target = result["valid"][i][col], result["target"][i][col]
            location = f"{identifier}:{side}:{i}"
            audit.check(type(valid) is bool, "target_valid_boolean", location)
            audit.check(type(target) in (float, int) and math.isfinite(target) and 0 <= target <= 1, "target_range", location)
            audit.check(bool(owners[i][col]) == valid, "valid_matches_applied_intervals", location)
            altered = abs(target - episode["state"][i][state_index]) > 1e-12
            changed |= altered
            if not valid:
                audit.check(target == episode["state"][i][state_index], "invalid_target_preserves_original", location)
                audit.check(result["phase"][i][col] == "unchanged" and result["source"][i][col] == "", "invalid_metadata_unchanged", location)
            else:
                valid_frames[col] += 1
                source_counts[result["source"][i][col]] += 1
                audit.check((result["source"][i][col], result["phase"][i][col]) in owners[i][col], "valid_source_phase_interval", location)
                audit.check(result["preview_target"][i][col] == target, "applied_preview_alignment", location)
                if result["phase"][i][col] == "hold":
                    audit.check(target == 0, "holding_target_zero", location)
            if side not in episode["valid_sides"]:
                audit.check(not valid and not altered, "unavailable_side_preserved", location)
                audit.check(result["preview_target"][i][col] == episode["state"][i][state_index],
                            "unavailable_side_preview_preserved", location)
        changed_frames += changed
    recomputed_coverage = {}
    for col, (side, state_index) in enumerate(SIDES):
        selected = [e for e in events if e["side"] == side]
        windows = [False] * size
        for event in selected:
            if event["reviewed"]:
                for i in range(event["start"], event["end"] + 1):
                    windows[i] = True
        missing = sum(not e["reviewed"] for e in selected)
        expected = dict(valid_side=side in episode["valid_sides"], candidates=len(selected),
                        reviewed_candidates=len(selected) - missing, unreviewed_candidates=missing,
                        accepted_cycles=sum(e["status"] == "applied" for e in selected),
                        preview_cycles=sum(e["status"] == "preview" for e in selected),
                        rejected_cycles=sum(e["reviewed"] and e["status"] == "unchanged" and e["reason"] != "no_grasp" for e in selected),
                        no_grasp_candidates=sum(e["reason"] == "no_grasp" for e in selected),
                        reviewed_frames=sum(windows), unreviewed_intervals=gaps(windows),
                        target_valid_frames=valid_frames[col],
                        changed_frames=sum(abs(result["target"][i][col] - episode["state"][i][state_index]) > 1e-12 for i in range(size)),
                        complete_review=side in episode["valid_sides"] and missing == 0 and all(windows))
        recorded = result["coverage"][side]
        for key, value in expected.items():
            audit.check(recorded.get(key) == value, "coverage_" + key, f"{identifier}:{side}")
        if "review_window_frames" in recorded:
            audit.check(recorded["review_window_frames"] == sum(windows), "coverage_review_window_frames", f"{identifier}:{side}")
        expected["review_window_frames"] = expected["reviewed_frames"]
        recomputed_coverage[side] = expected
        status = "invalid" if side not in episode["valid_sides"] else "reviewed" if expected["complete_review"] else "incomplete_review"
        audit.check(result["side_status"][side] == status, "side_status", f"{identifier}:{side}")
    audit.check(row.get("changed_frames") == changed_frames, "episode_summary_changed_frames", identifier)
    audit.check(row.get("coverage") == result["coverage"] and row.get("side_status") == result["side_status"],
                "episode_summary_coverage", identifier)
    counts = check_evidence(directory, episode, candidates, data["model_responses"], data["frames"], data["decisions"], audit)
    reused = check_reused_evidence(root, directory, data, audit)
    return dict(episode_id=identifier, frames=size, changed_frames=changed_frames, valid_side_frames=valid_frames,
                source_side_frames=dict(source_counts), coverage=recomputed_coverage, input_snapshot_checked=snapshot is not None,
                task_key=json.dumps(episode["identity"].get("tasks", episode["identity"].get("task", "unknown")), ensure_ascii=False),
                reused_evidence=reused, **counts)


def verify_run(run, manifest=None):
    audit = Audit()
    root = Path(run).resolve(strict=True)
    summary, config = audit.read(root / "summary.json"), audit.read(root / "config.json")
    audit.check(summary.get("status") == "complete" and summary.get("workflow") == "full_cycle", "completed_full_cycle_run", str(root))
    audit.check(summary.get("config_sha256") == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "config_sha256", "config.json")
    snapshots = input_snapshots(manifest, audit)
    implementation = check_implementation(root, summary, audit)
    episodes, tasks = [], {}
    rows = summary["episodes"]
    audit.check(len({r["episode_id"] for r in rows}) == len(rows) and len({r["directory"] for r in rows}) == len(rows),
                "unique_run_episodes", "summary.json")
    for row in rows:
        try:
            result = check_episode(root, row, snapshots.get(row["episode_id"]), audit)
            episodes.append(result)
            task = tasks.setdefault(result["task_key"], dict(episodes=0, changed_episodes=0, changed_frames=0))
            task["episodes"] += 1
            task["changed_episodes"] += bool(result["changed_frames"])
            task["changed_frames"] += result["changed_frames"]
        except (KeyError, TypeError, IndexError, ValueError, OSError) as exc:
            audit.errors.append(dict(check="readable_episode_artifacts", location=row.get("directory"), error_type=type(exc).__name__))
    actual_requests = sum(e["attempts"] for e in episodes)
    known_usage = sum(e["usage_reported_requests"] for e in episodes)
    usage = {k: sum(e["usage"][k] for e in episodes) for k in TOKENS}
    for key, value in (("actual_requests", actual_requests), ("usage_reported_requests", known_usage),
                       ("reported_token_usage", usage), ("tasks", tasks)):
        audit.check(summary.get(key) == value, "summary_" + key, "summary.json")
    audit.check(actual_requests <= config["max_requests"], "global_request_budget", "summary.json")
    audit.limit("No source dataset or source video was reread: checks bind saved observations and image bytes, not original recording contents or decoded-image semantics.")
    audit.limit("Request counts and usage are recomputed from saved responses; service billing and request payload hashes are not independently reconstructed.")
    audit.limit("Structured decisions and source labels are checked for consistency, not grasp correctness, human-reference accuracy, or training benefit.")
    request_counts = {key: sum(e[key] for e in episodes) for key in
                      ("service_responses", "successful_model_decisions", "attempts_without_saved_service_response", "attempts_without_reported_usage")}
    failures = Counter()
    for episode in episodes:
        failures.update(episode["failures"])
    reused = [e["reused_evidence"] for e in episodes if e["reused_evidence"] is not None]
    reused_requests = sum(e["attempts"] for e in reused)
    reused_usage = {k: sum(e["usage"][k] for e in reused) for k in TOKENS}
    source_runs = sorted({e["source_run"] for e in reused})
    if reused or summary.get("repeated_evaluation_same_inputs"):
        audit.check(len(reused) == len(episodes) == len(rows), "reuse_covers_all_episodes", "summary.json")
        audit.check(summary.get("repeated_evaluation_same_inputs") is True and summary.get("new_model_requests") == 0 and
                    actual_requests == 0, "reuse_not_independent_or_new_calls", "summary.json")
        audit.check(summary.get("reused_actual_requests") == reused_requests and summary.get("reused_reported_token_usage") == reused_usage,
                    "reuse_summary_requests_usage", "summary.json")
        audit.check(len(source_runs) == 1 and summary.get("source_run") == source_runs[0], "reuse_summary_source_run", "summary.json")
        for source_name in source_runs:
            source = Path(source_name)
            source_summary = audit.read(source / "summary.json")
            source_config = audit.read(source / "config.json")
            source_implementation = check_implementation(source, source_summary, audit)
            audit.check(source_implementation["status"] == "checked", "reuse_original_implementation_available", source_name)
            audit.check(source_summary.get("config_sha256") == hashlib.sha256(json.dumps(source_config, sort_keys=True).encode()).hexdigest(),
                        "reuse_source_config_sha256", source_name)
            audit.check(summary.get("source_implementation_sha256") == source_summary.get("implementation_sha256"),
                        "reuse_summary_source_implementation", "summary.json")
            audit.check({k: v for k, v in source_config.items() if k != "config_version"} ==
                        {k: v for k, v in config.items() if k != "config_version"}, "reuse_only_method_version_changed", "config.json")
            audit.check({e["episode_id"] for e in episodes} == {e["episode_id"] for e in source_summary["episodes"]},
                        "reuse_complete_source_episode_inventory", source_name)
            audit.check(source_summary.get("actual_requests") == reused_requests and
                        source_summary.get("reported_token_usage") == reused_usage,
                        "reuse_counts_match_original_run", source_name)
        audit.limit("Replayed numerical targets reuse the same model observations and input samples; their differences are not an independent evaluation.")
    return dict(schema="close_gripper_run_audit_v1", status="failed" if audit.errors else "passed", run=str(root),
                scope="saved_full_cycle_artifact_consistency", coverage_definition=COVERAGE_DEFINITION,
                implementation=implementation, input_snapshots_checked=sum(e["input_snapshot_checked"] for e in episodes),
                episodes=episodes, tasks=tasks, actual_requests=actual_requests, usage_reported_requests=known_usage,
                request_attempts=actual_requests, request_failures=dict(failures), **request_counts,
                request_count_definition="actual_requests is the legacy name for request_attempts; attempts do not prove service receipt. Missing usage is unknown, not zero billed tokens.",
                reused_request_attempts=reused_requests, reused_service_responses=sum(e["service_responses"] for e in reused),
                reused_usage_reported_requests=sum(e["usage_reported_requests"] for e in reused),
                reused_reported_token_usage=reused_usage, reused_source_runs=source_runs,
                reported_token_usage=usage, changed_frames=sum(e["changed_frames"] for e in episodes),
                valid_side_frames=[sum(e["valid_side_frames"][col] for e in episodes) for col in range(2)],
                checks=dict(audit.checks), errors=audit.errors, limitations=audit.limitations,
                input_files_sha256=audit.files, model_calls=0, source_data_reads=False,
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def write_audit(report, output, workspace=WORKSPACE):
    path = Path(output).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit output must not traverse symlinks")
    path = path.resolve()
    runs = (Path(workspace) / "runs").resolve()
    if path == runs or not path.is_relative_to(runs):
        raise ValueError("audit output must be a fresh directory under workspace runs/")
    protected = [Path(report["run"]), *(Path(p) for p in report.get("reused_source_runs", [])),
                 *(Path(p) for p in report["input_files_sha256"])]
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError("audit output overlaps an input")
    path.mkdir(parents=True, exist_ok=False)
    (path / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", help="Frozen input manifest with episode_cache and episode_sha256")
    parser.add_argument("--output", required=True, help="New directory under workspace runs/")
    args = parser.parse_args(argv)
    report = verify_run(args.run, args.manifest)
    output = write_audit(report, args.output)
    print(json.dumps(dict(status=report["status"], output=str(output), episodes=len(report["episodes"]),
                         requests=report["actual_requests"], changed_frames=report["changed_frames"],
                         errors=len(report["errors"]), implementation=report["implementation"]["status"])))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
