#!/usr/bin/env python3
"""Replay saved real-VLM decisions with the current numeric implementation.

This entry point makes no model requests. It requires a completed, auditable
full-cycle run and its frozen input manifest. Configuration changes are limited
to the current method version, and every detected candidate must remain exact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from close_gripper import clean, cycles, pipeline
from close_gripper.contracts import validate_episode
from close_gripper.report import write_plot, write_report
from scripts.verify_clean_run import verify_run

SOURCE_FILES = ("episode.json", "candidates.json", "decisions.json", "review_template.json",
                "model_responses.json", "frames.json", "targets.json", "result.json")
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_child(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("replay artifact paths must be relative without ..")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("replay artifact path escapes its root or traverses a symlink")
    return path


def checked_read(path, hashes):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != hashes.get(str(Path(path).resolve())):
        raise ValueError("source artifact changed after its audit: " + Path(path).name)
    return json.loads(raw)


def checked_copy(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest_file(destination) != expected:
        raise ValueError("copied evidence differs from its audited SHA: " + source.name)


def usage_totals(responses):
    total = dict(actual_requests=0, usage_reported_requests=0,
                 reported_token_usage={key: 0 for key in TOKENS})
    for response in responses:
        total["actual_requests"] += response.get("attempts", 0)
        usage = response.get("usage", {})
        if isinstance(usage, dict) and any(key in usage for key in TOKENS):
            total["usage_reported_requests"] += 1
            for key in TOKENS:
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    total["reported_token_usage"][key] += value
    return total


def replay_clean_run(source_run, manifest, output):
    source = Path(source_run).resolve(strict=True)
    manifest = Path(manifest).resolve(strict=True)
    source_summary = json.loads((source / "summary.json").read_text())
    if source_summary.get("status") != "complete" or source_summary.get("workflow") != "full_cycle":
        raise ValueError("replay requires a completed full-cycle source run")
    manifest_value = json.loads(manifest.read_text())
    entries = manifest_value.get("episodes") if isinstance(manifest_value, dict) else manifest_value
    if not isinstance(entries, list) or not entries or any(not isinstance(row, dict) or "episode_cache" not in row for row in entries):
        raise ValueError("replay requires frozen episode_cache inputs for every manifest row")
    input_episodes = clean.load_clean_manifest(manifest)
    identifiers = [episode["episode_id"] for episode in input_episodes]
    if len(identifiers) != len(set(identifiers)) or identifiers != [row["episode_id"] for row in source_summary["episodes"]]:
        raise ValueError("source run and frozen input episode inventory/order differ")
    audit = verify_run(source, manifest)
    if audit["status"] != "passed":
        checks = sorted({row["check"] for row in audit["errors"]})
        raise ValueError("source run audit failed: " + ", ".join(checks))
    if audit["implementation"]["status"] != "checked":
        raise ValueError("replay requires the source implementation snapshot")
    hashes = audit["input_files_sha256"]
    source_summary = checked_read(source / "summary.json", hashes)
    source_config = checked_read(source / "config.json", hashes)
    config = clean.clean_config(dict(source_config, config_version=cycles.cycle_config()["config_version"]))
    public = pipeline.public_config(config)
    if {k: v for k, v in public.items() if k != "config_version"} != {k: v for k, v in source_config.items() if k != "config_version"}:
        raise ValueError("numeric replay permits only the config_version configuration change")
    changes = ({"config_version": {"source": source_config.get("config_version"), "replay": public["config_version"]}}
               if public["config_version"] != source_config.get("config_version") else {})
    prepared = []
    for row, frozen in zip(source_summary["episodes"], input_episodes):
        old = safe_child(source, row["directory"])
        data = {name: checked_read(safe_child(old, name), hashes) for name in SOURCE_FILES}
        episode = data["episode.json"]
        validate_episode(episode)
        if episode != frozen or pipeline.episode_fingerprint(episode) != data["targets.json"].get("episode_sha256"):
            raise ValueError("replay source episode differs from the frozen input fingerprint")
        if data["result.json"].get("evidence_type") != "real_vlm":
            raise ValueError("replay requires original real_vlm evidence, not another replay")
        candidates = cycles.detect_cycles(episode, config)
        if pipeline.episode_fingerprint(candidates) != pipeline.episode_fingerprint(data["candidates.json"]):
            raise ValueError("detected candidates changed; new visual review is required")
        prepared.append(dict(row=row, old=old, data=data, candidates=candidates))

    protected = [source, manifest, *(Path(p) for p in hashes)]
    directory = pipeline.create_run(output, protected)
    package = Path(clean.__file__).resolve().parent
    current_files = sorted(package.glob("*.py"))
    implementation = directory / "implementation"
    implementation.mkdir()
    for path in current_files:
        shutil.copy2(path, implementation / path.name)
    current_digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in current_files)).hexdigest()
    script_digest = digest_file(Path(__file__))
    shutil.copy2(Path(__file__), directory / "replay_script.py")
    pipeline.dump(directory / "implementation.json", dict(sha256=current_digest, files=[p.name for p in current_files],
                                                         replay_entrypoint_sha256=script_digest))
    pipeline.dump(directory / "source_audit.json", audit)
    for name in ("summary.json", "config.json", "implementation.json"):
        checked_copy(source / name, directory / ("source_" + name), hashes[str((source / name).resolve())])
    source_impl = checked_read(source / "implementation.json", hashes)
    for name in source_impl["files"]:
        path = safe_child(source / "implementation", name)
        checked_copy(path, directory / "source_implementation" / name, hashes[str(path.resolve())])
    inputs = directory / "source_inputs"
    checked_copy(manifest, inputs / "manifest.json", hashes[str(manifest)])
    for row in entries:
        path = safe_child(manifest.parent, row["episode_cache"])
        checked_copy(path, safe_child(inputs, row["episode_cache"]), hashes[str(path.resolve())])
    records = []
    for item in prepared:
        row, old, data = item["row"], item["old"], item["data"]
        episode, candidates = data["episode.json"], item["candidates"]
        target_dir = safe_child(directory, row["directory"])
        target_dir.mkdir()
        saved = {}
        for name in SOURCE_FILES:
            destination_name = name if name in ("episode.json", "candidates.json", "decisions.json", "review_template.json") else "source_" + name
            path = safe_child(old, name)
            checked_copy(path, target_dir / destination_name, hashes[str(path.resolve())])
            saved[name] = destination_name
        frames, copied = [], set()
        for frame in data["frames.json"]:
            old_path = Path(frame["path"])
            if not old_path.is_absolute():
                old_path = old / old_path
            relative = old_path.relative_to(old).as_posix()
            old_path = safe_child(old, relative)
            new_path = safe_child(target_dir, relative)
            if relative not in copied:
                checked_copy(old_path, new_path, frame["sha256"])
                copied.add(relative)
            frames.append(dict(frame, path=str(new_path)))
        decisions = data["decisions.json"]
        result = cycles.generate_cycle_targets(copy.deepcopy(episode), copy.deepcopy(candidates), copy.deepcopy(decisions), config)
        issues = copy.deepcopy(data["result.json"].get("pipeline_issues", []))
        result.update(workflow="full_cycle", pipeline_issues=issues, evidence_type="reused_real_vlm",
                      repeated_evaluation_same_inputs=True, new_model_requests=0, source_evidence_run=str(source))
        by_id = {issue["candidate_id"]: issue for issue in issues}
        for event in result["events"]:
            if event["candidate_id"] in by_id:
                event["pipeline_issue"] = by_id[event["candidate_id"]]
        reused = usage_totals(data["model_responses.json"])
        evidence = dict(schema="close_gripper_reused_evidence_v1", source_run=str(source),
                        source_episode_directory=row["directory"], episode_sha256=pipeline.episode_fingerprint(episode),
                        source_files_sha256={name: hashes[str((old / name).resolve())] for name in SOURCE_FILES},
                        copied_source_files=saved, source_implementation_sha256=source_summary["implementation_sha256"],
                        source_config_sha256=source_summary["config_sha256"], config_changes=changes,
                        repeated_evaluation_same_inputs=True, new_model_requests=0,
                        reused_actual_requests=reused["actual_requests"], reused_usage_reported_requests=reused["usage_reported_requests"],
                        reused_reported_token_usage=reused["reported_token_usage"])
        pipeline.dump(target_dir / "reused_evidence.json", evidence)
        pipeline.dump(target_dir / "model_responses.json", [])
        pipeline.dump(target_dir / "frames.json", frames)
        pipeline.dump(target_dir / "result.json", result)
        pipeline.dump(target_dir / "targets.json", dict(episode_id=episode["episode_id"], identity=episode["identity"],
            episode_sha256=pipeline.episode_fingerprint(episode), evidence_type="reused_real_vlm",
            method_version=config["config_version"], source_run=str(directory), source_evidence_run=str(source),
            timestamps=episode["timestamps"], frame_indices=episode.get("frame_indices", list(range(len(episode["timestamps"])))),
            **{"annotation.gripper_target": result["target"], "annotation.gripper_target_valid": result["valid"],
               "annotation.gripper_target_source": result["source"], "annotation.gripper_phase": result["phase"]}))
        record = dict(episode=episode, candidates=candidates, result=result, frames=frames,
                      directory=target_dir.name, responses=[])
        records.append(record)
        write_report(target_dir, [record])
        write_plot(target_dir, [record])
    if any(digest_file(path) != expected for path, expected in hashes.items()):
        raise ValueError("a source artifact changed during replay")
    provenance = dict(schema="close_gripper_replay_run_v1", source_run=str(source),
                      source_input_manifest=str(manifest), source_input_manifest_sha256=hashes[str(manifest)],
                      source_config_sha256=source_summary["config_sha256"], source_implementation_sha256=source_summary["implementation_sha256"],
                      implementation_sha256=current_digest, replay_entrypoint_sha256=script_digest, config_changes=changes,
                      source_root_files_sha256={name: hashes[str((source / name).resolve())] for name in ("summary.json", "config.json", "implementation.json")},
                      repeated_evaluation_same_inputs=True, evidence_type="reused_real_vlm", new_model_requests=0,
                      reused_actual_requests=source_summary["actual_requests"],
                      reused_usage_reported_requests=source_summary["usage_reported_requests"],
                      reused_reported_token_usage=source_summary["reported_token_usage"],
                      source_artifacts_unchanged=True, candidate_inventory_identical=True,
                      effect_evidence="Same saved visual evidence evaluated again; no new model or independent human evaluation")
    pipeline.dump(directory / "reused_evidence.json", provenance)
    summary = clean.finish_clean_run(directory, records, config)
    summary.update(provenance)
    if summary["implementation_changed_during_run"]:
        summary["status"] = "failed_implementation_changed"
    pipeline.dump(directory / "summary.json", summary)
    if summary["status"] != "complete":
        raise ValueError("implementation changed during numeric replay")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--manifest", required=True, help="Original frozen cached-input manifest")
    parser.add_argument("--output", required=True, help="Fresh workspace runs/ directory")
    args = parser.parse_args(argv)
    summary = replay_clean_run(args.source_run, args.manifest, args.output)
    print(json.dumps(dict(status=summary["status"], episodes=len(summary["episodes"]), actual_requests=summary["actual_requests"],
                          reused_actual_requests=summary["reused_actual_requests"], review=str(Path(args.output).resolve() / "review.html"))))


if __name__ == "__main__":
    main()
