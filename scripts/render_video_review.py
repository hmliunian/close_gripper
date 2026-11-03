#!/usr/bin/env python3
"""Create a separate video review without regenerating labels or model evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from close_gripper import pipeline, report
from close_gripper.review_server import MEDIA_SCHEMA, safe_child, source_stat

RENDER_SCHEMA = "close_gripper_video_review_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def copy_artifact(source, destination, source_root, output, inventory, expected=None):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("saved image or source artifact SHA differs: " + source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    inventory[source.relative_to(source_root).as_posix()] = dict(
        sha256=digest, destination=destination.relative_to(output).as_posix(), size_bytes=len(raw))


def video_entries(episode, media):
    videos = []
    for side in ("left", "right"):
        if side not in episode.get("valid_sides", []):
            continue
        for eye in ("left", "right"):
            camera = f"observation.images.cam_{side}_wrist_{eye}"
            spec = episode.get("cameras", {}).get(camera)
            if not isinstance(spec, dict) or spec.get("valid", True) is not True:
                continue
            start, end = spec.get("from_timestamp"), spec.get("to_timestamp")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
                raise ValueError("invalid episode video interval")
            path = Path(spec["path"]).resolve(strict=True)
            if path.suffix.lower() != ".mp4" or not path.is_file():
                raise ValueError("episode video must be a readable MP4")
            stat = source_stat(path)
            identity = hashlib.sha256(json.dumps(dict(source_path=str(path), **stat), sort_keys=True).encode()).hexdigest()
            url = f"/media/{identity}.mp4"
            media[url] = dict(source_path=str(path), **stat)
            videos.append(dict(side=side, camera=camera, url=url, from_timestamp=start, to_timestamp=end))
    return videos


def verify_preservation(root, inventory, media, source):
    for relative, entry in inventory.items():
        original = safe_child(source, relative)
        copied = safe_child(root, entry["destination"])
        if sha256(original) != entry["sha256"] or sha256(copied) != entry["sha256"]:
            raise ValueError("source or copied review artifact changed: " + relative)
    for entry in media.values():
        if source_stat(entry["source_path"]) != {key: entry[key] for key in ("size_bytes", "mtime_ns", "device", "inode")}:
            raise ValueError("source video changed during review preparation")


def prepare_video_review(source_run, output, *, prepare_only=False):
    source = Path(source_run).absolute()
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise ValueError("source run must not traverse symlinks")
    source = source.resolve(strict=True)
    summary = read_json(source / "summary.json")
    if summary.get("status") != "complete" or not isinstance(summary.get("episodes"), list) or not summary["episodes"]:
        raise ValueError("video review requires a complete source run")
    rows = summary["episodes"]
    if len({row["episode_id"] for row in rows}) != len(rows) or len({row["directory"] for row in rows}) != len(rows):
        raise ValueError("source episode inventory must be unique")
    prepared, media = [], {}
    for row in rows:
        old = safe_child(source, row["directory"])
        episode, targets = read_json(old / "episode.json"), read_json(old / "targets.json")
        if row["episode_id"] != episode["episode_id"] or targets.get("episode_sha256") != pipeline.episode_fingerprint(episode):
            raise ValueError("source episode/target fingerprint mismatch")
        prepared.append(dict(directory=row["directory"], videos=video_entries(episode, media)))
    root = pipeline.create_run(output, [source, *(entry["source_path"] for entry in media.values())])
    inventory = {}
    for name in ("summary.json", "config.json", "implementation.json", "reused_evidence.json"):
        path = source / name
        if path.is_file():
            copy_artifact(safe_child(source, name), root / ("source_" + name), source, root, inventory)
    image_count = 0
    for item in prepared:
        old, new = safe_child(source, item["directory"]), safe_child(root, item["directory"])
        new.mkdir()
        for path in sorted(old.glob("*.json")):
            path = safe_child(old, path.name)
            copy_artifact(path, new / path.name, source, root, inventory)
        display_frames, copied = [], set()
        for frame in read_json(new / "frames.json"):
            path = Path(frame["path"])
            if not path.is_absolute():
                path = old / path
            relative = path.relative_to(old).as_posix()
            path = safe_child(old, relative)
            destination = safe_child(new, relative)
            if relative not in copied:
                copy_artifact(path, destination, source, root, inventory, frame["sha256"])
                copied.add(relative)
                image_count += 1
            display_frames.append(dict(frame, path=relative))
        write_json(new / "display_frames.json", display_frames)
    write_json(root / "media_manifest.json", dict(schema=MEDIA_SCHEMA, source_run=str(source),
        url_identity="SHA256 of canonical source path and stat; not a video content hash", media=media))
    write_json(root / "display_manifest.json", dict(schema=RENDER_SCHEMA, episodes=prepared))
    write_json(root / "source_files.json", dict(source_run=str(source), files=inventory))
    verify_preservation(root, inventory, media, source)
    render_summary = dict(schema=RENDER_SCHEMA, status="prepared", presentation_only=True,
        source_run=str(source), source_summary_sha256=inventory["summary.json"]["sha256"],
        source_implementation_sha256=summary.get("implementation_sha256"),
        episodes=len(rows), video_references=sum(len(item["videos"]) for item in prepared), unique_source_videos=len(media),
        copied_evidence_images=image_count, copied_source_files=len(inventory), source_artifacts_unchanged=True,
        video_bytes_copied=0, model_requests=0, algorithm_executions=0,
        note="Presentation of saved labels and existing source videos only; no new model, numerical cleaning, or training result",
        media_manifest_sha256=sha256(root / "media_manifest.json"),
        display_manifest_sha256=sha256(root / "display_manifest.json"),
        source_inventory_sha256=sha256(root / "source_files.json"))
    write_json(root / "render_summary.json", render_summary)
    return render_summary if prepare_only else finalize_video_review(root)


def finalize_video_review(output):
    root = Path(output).absolute()
    allowed = (pipeline.WORKSPACE / "runs").resolve()
    if not root.resolve().is_relative_to(allowed) or root.resolve() == allowed or any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("video review must be inside this workspace's runs directory")
    root = root.resolve(strict=True)
    summary = read_json(root / "render_summary.json")
    if summary.get("schema") != RENDER_SCHEMA or summary.get("status") != "prepared":
        raise ValueError("finalize requires a newly prepared video review")
    for name, key in (("media_manifest.json", "media_manifest_sha256"), ("display_manifest.json", "display_manifest_sha256"), ("source_files.json", "source_inventory_sha256")):
        if sha256(root / name) != summary[key]:
            raise ValueError("prepared video review manifest changed")
    media = read_json(root / "media_manifest.json")["media"]
    inventory = read_json(root / "source_files.json")["files"]
    source = Path(summary["source_run"])
    verify_preservation(root, inventory, media, source)
    renderer_hash = sha256(Path(report.__file__))
    records = []
    for item in read_json(root / "display_manifest.json")["episodes"]:
        directory = safe_child(root, item["directory"])
        frames = read_json(directory / "display_frames.json")
        for frame in frames:
            frame["path"] = str(safe_child(directory, frame["path"]))
            if sha256(frame["path"]) != frame["sha256"]:
                raise ValueError("display image changed")
        record = dict(episode=read_json(directory / "episode.json"), result=read_json(directory / "result.json"),
                      candidates=read_json(directory / "candidates.json"), frames=frames, videos=item["videos"], directory=item["directory"])
        records.append(record)
        report.write_report(directory, [record])
    report.write_report(root, records)
    verify_preservation(root, inventory, media, source)
    if renderer_hash != sha256(Path(report.__file__)):
        raise ValueError("report renderer changed during HTML generation")
    summary.update(status="complete", review="review.html", renderer_sha256=renderer_hash,
                   render_tool_sha256=sha256(Path(__file__)), source_artifacts_unchanged=True)
    write_json(root / "render_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run")
    parser.add_argument("--output")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize", metavar="PREPARED_RUN")
    args = parser.parse_args(argv)
    if args.finalize:
        if args.source_run or args.output or args.prepare_only:
            parser.error("--finalize is a separate operation")
        summary = finalize_video_review(args.finalize)
    else:
        if not args.source_run or not args.output:
            parser.error("--source-run and --output are required")
        summary = prepare_video_review(args.source_run, args.output, prepare_only=args.prepare_only)
    print(json.dumps({key: summary[key] for key in ("status", "episodes", "video_references", "unique_source_videos", "copied_evidence_images", "model_requests", "algorithm_executions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
