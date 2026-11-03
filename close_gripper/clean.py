"""Complete-episode command generation with explicit evidence and coverage."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading

from .contracts import validate_episode
from .cycles import cycle_config, detect_cycles, generate_cycle_targets
from .cycle_vision import CycleQwenClient, cycle_frame_indices, uncertain_cycle
from .pipeline import create_run, dump, episode_fingerprint, public_config, reviewed_decisions
from .report import write_report, write_plot

_PLOT_LOCK = threading.Lock()


def load_clean_manifest(path):
    """Load frozen full episodes; legacy selected-event fields are not reused."""
    path = Path(path).resolve(strict=True)
    entries = json.loads(path.read_text())
    if isinstance(entries, dict):
        entries = entries.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a nonempty episode list")
    episodes = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("manifest rows must be objects")
        if "episode_cache" in row:
            relative = row["episode_cache"]
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("cached episode paths must be relative to the manifest")
            unresolved = path.parent / relative
            if any(p.is_symlink() for p in (unresolved, *unresolved.parents) if p != path.parent):
                raise ValueError("cached episode paths must not traverse symlinks")
            cached = unresolved.resolve(strict=True)
            if not cached.is_relative_to(path.parent):
                raise ValueError("cached episodes must be inside the manifest directory")
            episode = json.loads(cached.read_text())
            if row.get("episode_sha256") != episode_fingerprint(episode):
                raise ValueError("cached episode fingerprint mismatch")
        else:
            from .dataset import load_episode
            episode = load_episode(row["dataset"], row["episode_index"])
        if row.get("episode_id", episode["episode_id"]) != episode["episode_id"]:
            raise ValueError("manifest episode identity mismatch")
        episodes.append(episode)
    return episodes


def clean_config(overrides=None):
    runtime_keys = {"calibrated", "max_images", "steady_image_interval_s", "transition_context_s",
                    "max_requests", "max_retries", "max_tokens", "timeout_s", "image_max_size_px",
                    "camera_mode", "model", "base_url", "api_key_file", "api_key_env", "temperature",
                    "max_image_bytes", "max_total_image_bytes"}
    allowed = set(cycle_config()) | runtime_keys
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides) - allowed):
        raise ValueError("unsupported clean configuration; use a credential file or environment variable")
    config = dict(cycle_config(overrides), calibrated=False, max_images=128,
                  steady_image_interval_s=.5, transition_context_s=.3,
                  max_requests=128, max_retries=0, max_tokens=4000, timeout_s=120,
                  image_max_size_px=512, camera_mode="primary",
                  model="qwen3.8-max-2026-09-02",
                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    config.update(overrides or {})
    cycle_config(config)
    if type(config["max_requests"]) is not int or config["max_requests"] < 1 or config["max_retries"] != 0:
        raise ValueError("full-cycle runs require a positive request budget and zero retries")
    return config


class BoundedCycleClient:
    """Share one absolute budget across episode workers; stop on auth rejection."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.reserved = 0
        self.authentication_blocked = False

    def annotate(self, episode, candidate, frames):
        with self.lock:
            error = ("authentication_blocked" if self.authentication_blocked else
                     "request_budget_exhausted" if self.reserved >= self.config["max_requests"] else None)
            if error:
                return dict(status="error", error=error, attempts=0, usage={},
                            decision=uncertain_cycle(candidate, error))
            self.reserved += 1
        client = CycleQwenClient(dict(self.config, max_requests=1, max_retries=0))
        result = client.annotate(episode, candidate, frames)
        if client.authentication_blocked:
            with self.lock:
                self.authentication_blocked = True
        return result


def process_clean_episode(episode, run_dir, config, decisions=None, client=None):
    validate_episode(episode)
    candidates = detect_cycles(episode, config)
    directory = Path(run_dir) / hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:12]
    directory.mkdir(exist_ok=False)
    dump(directory / "episode.json", episode)
    dump(directory / "candidates.json", candidates)
    decisions = list(decisions or [])
    supplied = {d.get("candidate_id") for d in decisions if isinstance(d, dict)}
    frames, responses, issues = [], [], []
    for ordinal, candidate in enumerate(candidates):
        if candidate["candidate_id"] in supplied or client is None:
            continue
        if candidate["start"] == candidate["end"]:
            issues.append(dict(candidate_id=candidate["candidate_id"], reason="insufficient_visual_context"))
            continue
        if candidate.get("camera_status") != "available":
            issues.append(dict(candidate_id=candidate["candidate_id"], reason="missing_same_side_camera"))
            continue
        try:
            from .dataset import extract_selected_frames
            indices = cycle_frame_indices(episode, candidate, config)
            extracted = extract_selected_frames(episode, candidate["side"], indices,
                                                directory / f"frames_{ordinal:03d}", config)
            frames.extend(extracted)
            response = client.annotate(episode, candidate, extracted)
            response.update(candidate_id=candidate["candidate_id"],
                            submitted_frame_indices=indices, evidence_images=len(extracted))
            responses.append(response)
            if response.get("status") == "ok":
                decisions.append(response["decision"])
            else:
                issues.append(dict(candidate_id=candidate["candidate_id"], reason=response.get("error", "model_failure")))
        except (ValueError, OSError, RuntimeError) as exc:
            from .vision import VisionValidationError
            reason = str(exc) if isinstance(exc, VisionValidationError) else "frame_or_model_error"
            issues.append(dict(candidate_id=candidate["candidate_id"], reason=reason, error_type=type(exc).__name__))
    result = generate_cycle_targets(episode, candidates, decisions, config)
    result.update(workflow="full_cycle", pipeline_issues=issues,
                  evidence_type="real_vlm" if client else "supplied_review" if supplied else "no_visual_review")
    by_id = {i["candidate_id"]: i for i in issues}
    for event in result["events"]:
        if event["candidate_id"] in by_id:
            event["pipeline_issue"] = by_id[event["candidate_id"]]
    record = dict(episode=episode, candidates=candidates, result=result, frames=frames,
                  directory=directory.name, responses=responses)
    dump(directory / "frames.json", frames)
    dump(directory / "model_responses.json", responses)
    dump(directory / "decisions.json", decisions)
    dump(directory / "review_template.json", dict(episode_sha256=episode_fingerprint(episode), decisions=decisions))
    dump(directory / "result.json", result)
    dump(directory / "targets.json", dict(episode_id=episode["episode_id"], identity=episode["identity"],
        episode_sha256=episode_fingerprint(episode), evidence_type=result["evidence_type"],
        method_version=config["config_version"], source_run=str(Path(run_dir).resolve()),
        timestamps=episode["timestamps"], frame_indices=episode.get("frame_indices", list(range(len(episode["timestamps"])))),
        **{"annotation.gripper_target": result["target"], "annotation.gripper_target_valid": result["valid"],
           "annotation.gripper_target_source": result["source"], "annotation.gripper_phase": result["phase"]}))
    write_report(directory, [record])
    with _PLOT_LOCK:
        write_plot(directory, [record])
    print(json.dumps(dict(episode=episode["episode_id"], candidates=len(candidates),
                          applied=sum(e["status"] == "applied" for e in result["events"]),
                          requests=sum(r.get("attempts", 0) for r in responses)), ensure_ascii=False), flush=True)
    return record


def finish_clean_run(directory, records, config):
    tasks, episodes = {}, []
    usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    requests = known_usage = 0
    for record in records:
        ep, result = record["episode"], record["result"]
        changed = sum(any(abs(row[col] - ep["state"][i][(9, 19)[col]]) > 1e-12 for col in (0, 1))
                      for i, row in enumerate(result["target"]))
        key = json.dumps(ep["identity"].get("tasks", ep["identity"].get("task", "unknown")), ensure_ascii=False)
        task = tasks.setdefault(key, dict(episodes=0, changed_episodes=0, changed_frames=0))
        task["episodes"] += 1
        task["changed_episodes"] += bool(changed)
        task["changed_frames"] += changed
        episodes.append(dict(episode_id=ep["episode_id"], directory=record["directory"], changed_frames=changed,
                             coverage=result["coverage"], side_status=result["side_status"]))
        for response in record["responses"]:
            requests += response.get("attempts", 0)
            reported = response.get("usage")
            if isinstance(reported, dict) and any(k in reported for k in usage):
                known_usage += 1
                for name in usage:
                    value = reported.get(name)
                    if type(value) is int and value >= 0:
                        usage[name] += value
    cfg = public_config(config)
    dump(directory / "config.json", cfg)
    write_report(directory, records)
    # Each episode already has a readable plot; avoid an unbounded tall PNG.
    summary = dict(status="complete", workflow="full_cycle", episodes=episodes, tasks=tasks,
                   actual_requests=requests, usage_reported_requests=known_usage, reported_token_usage=usage,
                   human_reference=False, calibrated=config.get("calibrated", False),
                   evaluation_scope="all_detected_cycles_and_remaining_observation_windows",
                   review="review.html", source_data_changed=False,
                   config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    package = Path(__file__).resolve().parent
    current_digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(package.glob("*.py")))).hexdigest()
    recorded = json.loads((directory / "implementation.json").read_text()) if (directory / "implementation.json").is_file() else {}
    summary["implementation_sha256"] = recorded.get("sha256", current_digest)
    summary["implementation_changed_during_run"] = summary["implementation_sha256"] != current_digest
    dump(directory / "summary.json", summary)
    return summary


def run_clean(episodes, output, config=None, *, call_model=False, workers=1, reviews=None):
    config = clean_config(config)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    if not episodes or len({ep["episode_id"] for ep in episodes}) != len(episodes):
        raise ValueError("nonempty distinct episode inputs are required")
    for episode in episodes:
        validate_episode(episode)
    sources = [ep["identity"]["dataset_root"] for ep in episodes if "dataset_root" in ep["identity"]]
    directory = create_run(output, sources)
    package = Path(__file__).resolve().parent
    implementation = directory / "implementation"
    implementation.mkdir()
    files = sorted(package.glob("*.py"))
    for source in files:
        shutil.copy2(source, implementation / source.name)
    digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()
    dump(directory / "implementation.json", dict(sha256=digest, files=[p.name for p in files]))
    client = BoundedCycleClient(config) if call_model else None
    reviews = reviews or {}
    decisions = {ep["episode_id"]: reviewed_decisions(ep, reviews[ep["episode_id"]])
                 for ep in episodes if ep["episode_id"] in reviews}
    def process(ep):
        return process_clean_episode(ep, directory, config, decisions.get(ep["episode_id"]), client)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(process, episodes))
    return finish_clean_run(directory, records, config)
