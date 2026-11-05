"""Run task-profile curve evaluation with immutable observations and provenance."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from .contracts import validate_episode
from .cycles import cycle_config
from .pipeline import create_run, dump, episode_fingerprint
from .profile_clean import RULE_VERSION, generate_profile_targets
from .report import write_report
from .task_profiles import validate_catalog
from .head_review import validate_head_review


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def run_profile_clean(episodes, output, catalog, *, config=None, workers=4, input_paths=(),
                      visual_reviews=None, evidence_frames=None):
    """Evaluate full curves, preserving the training target/validity contract.

    The actual rule curve is saved as ``evaluation.gripper_target``. Existing
    annotation targets stay original and invalid until separately reviewed.
    """
    validate_catalog(catalog)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    if not episodes or len({ep["episode_id"] for ep in episodes}) != len(episodes):
        raise ValueError("distinct nonempty episode inputs are required")
    for episode in episodes:
        validate_episode(episode)
    if visual_reviews is not None or evidence_frames is not None:
        expected = {ep["episode_id"] for ep in episodes}
        if (not isinstance(visual_reviews, dict) or not isinstance(evidence_frames, dict)
                or set(visual_reviews) != expected or set(evidence_frames) != expected):
            raise ValueError("visual reviews and evidence must match the whole input cohort")
        for episode in episodes:
            identifier = episode["episode_id"]
            validate_head_review(episode, visual_reviews[identifier], evidence_frames[identifier])
    numerical = cycle_config(config)
    numerical["minimum_hold_s"] = catalog["minimum_hold_s"]
    fingerprints = {ep["episode_id"]: episode_fingerprint(ep) for ep in episodes}
    sources = [ep["identity"]["dataset_root"] for ep in episodes if "dataset_root" in ep["identity"]]
    directory = create_run(output, [*sources, *input_paths])
    package = Path(__file__).resolve().parent
    implementation = directory / "implementation"
    implementation.mkdir()
    files = sorted(package.glob("*.py"))
    for source in files:
        shutil.copy2(source, implementation / source.name)
    implementation_sha = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()
    dump(directory / "implementation.json", dict(sha256=implementation_sha, files=[p.name for p in files]))
    dump(directory / "task_profiles.json", catalog)
    cfg = dict(workflow="task_profile_rule", rule_version=RULE_VERSION, evaluation_only=True, model_requests=0,
               catalog_version=catalog["version"], catalog_semantic_sha256=_digest(catalog),
               minimum_hold_s=catalog["minimum_hold_s"], numerical=numerical)
    if visual_reviews is not None:
        cfg.update(visual_camera_group="head", native_visual_review_episodes=len(visual_reviews),
                   review_method="codex_native_vision", external_model_requests=0,
                   visual_review_source="model", human_confirmed=False,
                   rule_version=RULE_VERSION + "-reviewed-v1")
    dump(directory / "config.json", cfg)

    def process(episode):
        identifier = episode["episode_id"]
        if visual_reviews is None:
            result = generate_profile_targets(episode, catalog, numerical)
        else:
            states = {side: [segment["state"] for segment in value["segments"]
                            for _ in range(segment["start"], segment["end"] + 1)]
                      for side, value in visual_reviews[identifier]["sides"].items()}
            result = generate_profile_targets(episode, catalog, numerical, reviewed_states=states)
            for side, values in states.items():
                col = 0 if side == "left" else 1
                for i, state in enumerate(values):
                    if state == "uncertain" and (result["rule_applied_mask"][i][col]
                            or result["preview_target"][i][col] != episode["state"][i][9 if col == 0 else 19]):
                        raise ValueError("main-view unknown frame was modified")
                    if result["preview_phase"][i][col] == "hold" and state != "held":
                        raise ValueError("main-view zero hold extends outside observed holding")
        if episode_fingerprint(episode) != fingerprints[identifier]:
            raise ValueError("rule processing modified its observation input")
        original = [[row[9], row[19]] for row in episode["state"]]
        if result["target"] != original or any(any(row) for row in result["valid"]):
            raise ValueError("profile evaluation cannot promote training annotations")
        candidate_list = result.get("candidates", [])
        child = directory / hashlib.sha256(identifier.encode()).hexdigest()[:12]
        child.mkdir(exist_ok=False)
        frames = []
        if visual_reviews is not None:
            result["visual_review"] = visual_reviews[identifier]
            result["visual_review_scope"] = "dense_main_camera_model_observations_separate_from_numeric_rule"
            image_dir = child / "images"
            image_dir.mkdir()
            for frame in evidence_frames[identifier]:
                source_image = Path(frame["path"])
                raw = source_image.read_bytes()
                if hashlib.sha256(raw).hexdigest() != frame["sha256"]:
                    raise ValueError("reviewed main image checksum differs")
                destination = image_dir / (frame["camera"].rsplit("_", 1)[-1] + f"_{frame['frame_index']:06d}.jpg")
                destination.write_bytes(raw)
                frames.append(dict(frame, path=str(destination)))
        dump(child / "episode.json", episode)
        dump(child / "result.json", result)
        dump(child / "candidates.json", candidate_list)
        dump(child / "frames.json", frames)
        dump(child / "model_responses.json", [])
        dump(child / "decisions.json", [])
        dump(child / "targets.json", dict(episode_id=identifier, identity=episode["identity"],
            episode_sha256=fingerprints[identifier], evidence_type="task_profile_rule",
            evaluation_only=True, grasp_semantics_verified=False,
            method_version=result["rule_version"], catalog_version=catalog["version"],
            source_run=str(directory), timestamps=episode["timestamps"],
            frame_indices=episode.get("frame_indices", list(range(len(original)))),
            **{"annotation.gripper_target": result["target"],
               "annotation.gripper_target_valid": result["valid"],
               "annotation.gripper_target_source": result["source"],
               "annotation.gripper_phase": result["phase"],
               "evaluation.gripper_target": result["preview_target"],
               "evaluation.gripper_rule_applied": result["rule_applied_mask"],
               "evaluation.gripper_phase": result["preview_phase"],
               "evaluation.gripper_source": result["preview_source"]}))
        record = dict(episode=episode, result=result, candidates=candidate_list, frames=frames,
                      responses=[], directory=child.name)
        write_report(child, [record])
        changed = sum(any(abs(row[col] - original[i][col]) > 1e-12 for col in (0, 1))
                      for i, row in enumerate(result["preview_target"]))
        print(json.dumps(dict(episode=identifier, workflow="task_profile_rule", changed_frames=changed,
            applied_intervals=sum(e.get("status") == "applied" for e in result["events"])), ensure_ascii=False), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(process, episodes))
    write_report(directory, records)
    tasks, entries, reasons = {}, [], Counter()
    side_outcomes, unchanged_side_reasons = Counter(), Counter()
    for record in records:
        episode, result = record["episode"], record["result"]
        original = [[row[9], row[19]] for row in episode["state"]]
        changed = sum(any(abs(row[col] - original[i][col]) > 1e-12 for col in (0, 1))
                      for i, row in enumerate(result["preview_target"]))
        identity = episode["identity"]
        source_metadata = identity.get("source_metadata", {})
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        labels = source_metadata.get("source_tasks", identity.get("source_tasks", identity.get("tasks", [])))
        task_name = " | ".join(labels) if isinstance(labels, list) else str(labels)
        totals = tasks.setdefault(task_name, dict(episodes=0, applied_episodes=0, changed_episodes=0, changed_frames=0,
                                                 applied_intervals=0, rejected_intervals=0))
        applied = sum(e.get("status") == "applied" for e in result["events"])
        rejected = len(result["events"]) - applied
        totals["episodes"] += 1
        totals["applied_episodes"] += bool(applied)
        totals["changed_episodes"] += bool(changed)
        totals["changed_frames"] += changed
        totals["applied_intervals"] += applied
        totals["rejected_intervals"] += rejected
        reasons.update(e.get("reason", "unspecified") for e in result["events"] if e.get("status") != "applied")
        for side in episode["valid_sides"]:
            outcome = result.get("side_status", {}).get(side, "unspecified")
            side_outcomes[outcome] += 1
            if outcome != "rule_applied":
                unchanged_side_reasons[result.get("coverage", {}).get(side, {}).get("reason", outcome)] += 1
        entries.append(dict(episode_id=episode["episode_id"], directory=record["directory"], task=task_name,
            changed_frames=changed, applied_intervals=applied, rejected_intervals=rejected,
            coverage=result.get("coverage", {}), side_status=result.get("side_status", {})))
    current_sha = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()
    summary = dict(status="complete", workflow="task_profile_rule", evaluation_only=True,
        grasp_semantics_verified=False, episodes=entries, tasks=tasks, actual_requests=0,
        usage_reported_requests=0, reported_token_usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        source_data_changed=False, observation_fingerprints_preserved=True, training_target_unchanged=True,
        review="review.html", implementation_sha256=implementation_sha,
        implementation_changed_during_run=current_sha != implementation_sha,
        config_sha256=_digest(cfg), catalog_version=catalog["version"], catalog_semantic_sha256=_digest(catalog),
        changed_episodes=sum(bool(e["changed_frames"]) for e in entries),
        applied_episodes=sum(bool(e["applied_intervals"]) for e in entries),
        changed_frames=sum(e["changed_frames"] for e in entries),
        applied_intervals=sum(e["applied_intervals"] for e in entries), rejection_reasons=dict(reasons),
        side_outcomes=dict(side_outcomes), unchanged_side_reasons=dict(unchanged_side_reasons),
        evaluation_scope="all_valid_sides_and_task_profile_matching_intervals",
        generated_curve_field="evaluation.gripper_target", generated_mask_field="evaluation.gripper_rule_applied")
    if summary["implementation_changed_during_run"]:
        raise ValueError("implementation changed during this evaluation")
    if visual_reviews is not None:
        summary.update(native_visual_review_episodes=len(visual_reviews),
                       native_visual_review_source="model", visual_camera_group="head",
                       rule_version=records[0]["result"]["rule_version"],
                       unknown_visual_frames_preserved=True, heldout_evaluation=False)
    dump(directory / "summary.json", summary)
    return summary
