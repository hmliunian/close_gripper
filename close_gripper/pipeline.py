"""Local, bounded runs with separate observations, evidence and target artifacts."""
import hashlib
import json
from pathlib import Path
from .contracts import DEFAULT_CONFIG, validate_episode
from .core import detect_candidates, generate_targets
from .report import write_report, write_plot

WORKSPACE = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def create_run(output, input_paths=()):
    path = Path(output).absolute()
    allowed = WORKSPACE / "runs"
    # A symlink in runs or any child could otherwise redirect writes into input data.
    if allowed.is_symlink() or any(p.is_symlink() for p in (path, *path.parents) if p != WORKSPACE):
        raise ValueError("run path must not traverse symlinks")
    resolved = path.resolve()
    if not resolved.is_relative_to(allowed.resolve()) or resolved == allowed.resolve():
        raise ValueError("output must be a new directory under this workspace's runs/")
    for source in input_paths:
        source = Path(source).resolve()
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise ValueError("input and output must not overlap")
    if path.exists():
        raise FileExistsError("run already exists; choose a new run id")
    path.mkdir(parents=True, exist_ok=False)
    return path


def public_config(config):
    if isinstance(config, dict):
        return {k: public_config(v) for k, v in config.items() if not any(s in k.lower() for s in ("key", "token", "secret", "password")) or k == "max_tokens"}
    if isinstance(config, list):
        return [public_config(v) for v in config]
    return config


def episode_fingerprint(episode):
    return hashlib.sha256(json.dumps(episode, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reviewed_decisions(episode, envelope):
    if not isinstance(envelope, dict) or envelope.get("episode_sha256") != episode_fingerprint(episode):
        raise ValueError("review must be bound to the exact episode fingerprint")
    decisions = envelope.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("review decisions must be a list")
    for d in decisions:
        if not isinstance(d, dict):
            raise ValueError("each review decision must be an object")
        if d.get("review_source") == "human" and d.get("confirmed"):
            if not all(isinstance(d.get(k), str) and d[k].strip() for k in ("reviewer", "reviewed_at", "review_note")):
                raise ValueError("confirmed human reviews require reviewer, reviewed_at and review_note")
    return decisions


def process_episode(episode, run_dir, config, decisions=None, client=None, candidate_limit=None):
    validate_episode(episode)
    candidates = detect_candidates(episode, config)
    eid = hashlib.sha256(str(episode["episode_id"]).encode()).hexdigest()[:12]
    directory = run_dir / eid
    directory.mkdir(exist_ok=False)
    dump(directory / "episode.json", episode)
    dump(directory / "candidates.json", candidates)
    supplied = decisions is not None
    decisions = list(decisions or [])
    frames, requests, issues = [], [], []
    requested = 0
    for ordinal, candidate in enumerate(candidates):
        if client is None:
            continue
        selection = config.get("candidate_starts")
        selected_ids = config.get("candidate_ids")
        if selected_ids is not None and candidate["candidate_id"] not in selected_ids:
            issues.append({"candidate_id":candidate["candidate_id"], "reason":"not_selected_for_visual_review"})
            continue
        if selection is not None and candidate["start"] not in selection:
            issues.append({"candidate_id":candidate["candidate_id"], "reason":"not_selected_for_visual_review"})
            continue
        if candidate_limit is not None and requested >= candidate_limit:
            issues.append({"candidate_id":candidate["candidate_id"], "reason":"request_budget_not_checked"})
            continue
        from .dataset import extract_frames
        frame_dir = directory / f"frames_{ordinal:03d}"
        frame_dir.mkdir()
        try:
            extracted = extract_frames(episode, candidate, frame_dir, config)
            frames.extend(extracted)
            requested += 1
            response = client.annotate(episode, candidate, extracted)
            requests.append(response)
            if response.get("status") == "ok" and response.get("decision"):
                decisions.append(response["decision"])
            else:
                issues.append({"candidate_id":candidate["candidate_id"], "reason":response.get("error", "model_failure")})
        except (ValueError, OSError, RuntimeError) as exc:
            # Do not persist arbitrary exception text from model/IO adapters.
            issues.append({"candidate_id":candidate["candidate_id"], "reason":"frame_or_model_error", "error_type":type(exc).__name__})
    result = generate_targets(episode, candidates, decisions, config)
    issues_by_id = {issue["candidate_id"]: issue for issue in issues}
    for event in result["events"]:
        if event["candidate_id"] in issues_by_id:
            event["pipeline_issue"] = issues_by_id[event["candidate_id"]]
            if event.get("reason") == "missing_visual_decision":
                event["reason"] = issues_by_id[event["candidate_id"]]["reason"]
    result["pipeline_issues"] = issues
    result["evidence_type"] = "supplied_review" if supplied else "real_vlm" if client else "no_visual_review"
    record = dict(episode=episode, result=result, candidates=candidates, frames=frames)
    dump(directory / "frames.json", frames)
    dump(directory / "model_responses.json", requests)
    dump(directory / "decisions.json", decisions)
    dump(directory / "review_template.json", dict(episode_sha256=episode_fingerprint(episode), decisions=decisions))
    dump(directory / "targets.json", dict(episode_id=episode["episode_id"], identity=episode["identity"],
        timestamps=episode["timestamps"], frame_indices=episode.get("frame_indices", list(range(len(episode["timestamps"])))),
        **{"annotation.gripper_target":result["target"], "annotation.gripper_target_valid":result["valid"]}))
    dump(directory / "result.json", result)
    return record


def finish_run(run_dir, records, config, quiet=False):
    cfg = public_config(config)
    dump(run_dir / "config.json", cfg)
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    tasks = {}
    side_counts = {s: dict(candidates=0, applied=0, preview=0, unchanged=0, not_triggered_episodes=0, invalid_episodes=0) for s in ("left", "right")}
    for record in records:
        ep, res = record["episode"], record["result"]
        task = str(ep["identity"].get("task", ep["identity"].get("tasks", "unknown")))
        counts = tasks.setdefault(task, dict(episodes=0, candidates=0, changed_frames=0, accepted_side_frames=0))
        counts["episodes"] += 1
        counts["candidates"] += len(record["candidates"])
        counts["changed_frames"] += sum(any(abs(row[c]-ep["state"][i][(9,19)[c]])>1e-12 for c in (0,1)) for i,row in enumerate(res["target"]))
        counts["accepted_side_frames"] += sum(sum(row) for row in res["valid"])
        for event in res["events"]:
            side_counts[event["side"]]["candidates"] += 1
            side_counts[event["side"]][event["status"]] += 1
        for side, status in res["side_status"].items():
            if status in ("not_triggered", "invalid"):
                side_counts[side][status + "_episodes"] += 1
    write_report(run_dir, records)
    plot = write_plot(run_dir, records)
    summary = dict(status="complete", config_sha256=digest, episodes=len(records), tasks=tasks, sides=side_counts,
                   release_policy="review_only", real_vlm_requests=sum(sum(response.get("attempts", 0) for response in json.loads((run_dir / hashlib.sha256(str(r["episode"]["episode_id"]).encode()).hexdigest()[:12] / "model_responses.json").read_text())) for r in records),
                   review="review.html", plot=plot.name if plot else None)
    summary["implementation_sha256"] = hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted((WORKSPACE / "close_gripper").glob("*.py")))).hexdigest()
    dump(run_dir / "summary.json", summary)
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False))
    return summary


def default_config(overrides=None):
    if overrides and any(k in overrides for k in ("api_key", "secret", "password", "access_token")):
        raise ValueError("use api_key_file or an environment variable, not inline credentials")
    config = dict(DEFAULT_CONFIG)
    config.update(dict(config_version="experimental-v1", calibrated=False,
        context_before_s=3., context_after_s=1., max_images=96, max_requests=3,
        model="qwen3.8-max-2026-09-02", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens=4000, timeout_s=120, max_retries=0))
    config.update(overrides or {})
    return config
