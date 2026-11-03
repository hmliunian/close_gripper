"""Bounded real-case evaluation with one explicitly selected event per episode."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

from .core import detect_candidates
from .pipeline import (create_run, default_config, dump, episode_fingerprint,
                       finish_run, process_episode, public_config)


def load_case(entry, manifest_dir, config):
    """Validate the frozen snapshot and exact event identity before any model call."""
    cache = (Path(manifest_dir) / entry["episode_cache"]).resolve()
    if not cache.is_relative_to(Path(manifest_dir).resolve()):
        raise ValueError("episode cache must remain within the manifest directory")
    episode = json.loads(cache.read_text())
    if episode_fingerprint(episode) != entry["episode_sha256"]:
        raise ValueError("episode snapshot fingerprint mismatch")
    if episode["episode_id"] != entry["episode_id"]:
        raise ValueError("episode identity mismatch")
    candidates = detect_candidates(episode, config)
    selected = [c for c in candidates if c["candidate_id"] == entry["candidate_id"]]
    if len(selected) != 1 or selected[0]["side"] != entry["side"]:
        raise ValueError("selected candidate identity mismatch")
    return episode, selected[0]


def aggregate_rows(rows):
    usage = Counter()
    usage_known = 0
    for row in rows:
        reported = row.get("usage")
        reported = reported if isinstance(reported, dict) else {}
        known = False
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = reported.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] += value
                known = True
        usage_known += known
    return dict(cases=len(rows), tasks=len({task for r in rows for task in r.get("task", [])}),
        task_groups=len({tuple(r.get("task", [])) for r in rows}),
        statuses=dict(Counter(r["status"] for r in rows)),
        sides=dict(Counter(r["side"] for r in rows)),
        releases=dict(Counter(r.get("release", "unknown") for r in rows)),
        subsets=dict(Counter(r.get("release", "") + "/" + r.get("subset", "unknown") for r in rows)),
        task_counts=dict(Counter(" / ".join(r.get("task", [])) for r in rows)),
        categories=dict(Counter(r["category"] for r in rows if r["status"] == "model_ok")),
        reasons=dict(Counter(r["reason"] for r in rows)),
        actual_requests=sum(r.get("request_attempts", 0) for r in rows),
        server_responses=sum(bool(r.get("server_response")) for r in rows),
        server_responses_definition="responses_with_parsed_json_body",
        usage_reported_cases=usage_known, reported_token_usage=dict(usage),
        input_images=sum(r.get("input_images", 0) for r in rows),
        preview_cases=sum(r.get("preview_changed_frames", 0) > 0 for r in rows),
        preview_changed_frames=sum(r.get("preview_changed_frames", 0) for r in rows),
        accepted_changed_frames=sum(r.get("accepted_changed_frames", 0) for r in rows),
        total_detected_candidates=sum(r.get("detected_candidates", 0) for r in rows),
        human_reference=False, evaluation_scope="one_selected_opening_per_episode")


def run_batch(manifest_path, output, config=None, workers=4, call_model=False):
    from .vision import QwenClient
    from .batch_report import write_batch_report
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be 1 to 4")
    path = Path(manifest_path).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty list")
    if len({e["episode_id"] for e in entries}) != len(entries):
        raise ValueError("batch requires distinct published episodes")
    cfg = default_config(config)
    # A case is one model request, not unlimited opening checks or retries.
    cfg.update(max_requests=1, max_retries=0)
    cfg.pop("candidate_starts", None)
    cfg.pop("candidate_ids", None)
    directory = create_run(output, [e["dataset"] for e in entries])
    (directory / "cases").mkdir()
    dump(directory / "manifest.json", entries)
    dump(directory / "config.json", public_config(cfg))
    dump(directory / "execution.json", dict(workers=workers, planned_cases=len(entries),
        max_network_requests=len(entries) if call_model else 0, retries=0,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), call_model=call_model))
    stopped = threading.Event()
    # Matplotlib has process-global state; serialize rendering.
    render_lock = threading.Lock()

    def work(index, entry):
        started = time.monotonic()
        case_dir = directory / "cases" / f"{index:03d}"
        case_dir.mkdir()
        root = Path(entry["dataset"])
        row = dict(index=index, episode_id=entry["episode_id"], task=entry.get("task", []),
            source_tasks=entry.get("source_tasks", []), subset=root.name, release=root.parent.name,
            side=entry["side"], candidate_id=entry["candidate_id"],
            candidate_start=entry.get("candidate_start"), candidate_end=entry.get("candidate_end"),
            status="error", category=None, reason="not_started", preview_changed_frames=0,
            accepted_changed_frames=0, request_attempts=0, usage={}, server_response=False,
            input_images=0, detected_candidates=0, episode_dir=case_dir.relative_to(directory).as_posix())
        try:
            if stopped.is_set():
                row.update(status="no_model", reason="batch_authentication_blocked")
                return row
            episode, candidate = load_case(entry, path.parent, cfg)
            case_cfg = dict(cfg, candidate_ids=[candidate["candidate_id"]])
            client = QwenClient(case_cfg) if call_model else None
            record = process_episode(episode, case_dir, case_cfg, client=client, candidate_limit=1)
            artifact = case_dir / hashlib.sha256(episode["episode_id"].encode()).hexdigest()[:12]
            responses = json.loads((artifact / "model_responses.json").read_text())
            response = responses[0] if responses else {}
            if any(r.get("error") in ("http_401", "http_403", "authentication_blocked") for r in responses):
                stopped.set()
            chosen = next(e for e in record["result"]["events"] if e["candidate_id"] == candidate["candidate_id"])
            status = "model_ok" if response.get("status") == "ok" else "model_error" if response else "no_model"
            decision = response.get("decision", {}) if response.get("status") == "ok" else {}
            row.update(status=status, category=decision.get("category"), reason=chosen["reason"],
                preview_changed_frames=chosen.get("changed_frames", 0) if chosen["status"] in ("preview", "applied") else 0,
                accepted_changed_frames=chosen.get("changed_frames", 0) if chosen["status"] == "applied" else 0,
                request_attempts=sum(r.get("attempts", 0) for r in responses),
                usage=response.get("usage", {}), server_response=response.get("raw_response") is not None,
                model=response.get("model"), model_error=response.get("error"),
                input_images=len(record["frames"]), detected_candidates=len(record["candidates"]))
            if any(any(flags) for flags in record["result"]["valid"]):
                raise ValueError("model-only batch must not produce accepted training targets")
            with render_lock:
                # Keep progress output to one compact JSON line per completed case.
                finish_run(case_dir, [record], case_cfg, quiet=True)
        except Exception as exc:
            row.update(status="error", reason="case_error_" + type(exc).__name__)
        finally:
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            dump(case_dir / "case_result.json", row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, entry) for i, entry in enumerate(entries, 1)]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            dump(directory / "progress.json", dict(completed=len(rows), planned=len(entries),
                status_counts=dict(Counter(r["status"] for r in rows)), requests=sum(r["request_attempts"] for r in rows)))
            print(json.dumps({k:row[k] for k in ("index", "status", "side", "category", "reason", "elapsed_s")}, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: r["index"])
    summary = aggregate_rows(rows)
    dump(directory / "case_results.json", rows)
    write_batch_report(directory, rows)
    summary["report"] = "batch_review.html"
    summary["status"] = "complete"
    dump(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
