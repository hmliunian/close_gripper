"""Read-only LeRobot v3 episode and same-wrist image access.

Heavy dependencies are imported only inside their respective entry points.
All frame indices in this API refer to positions in the returned episode.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .contracts import SIDE_CAMERAS as WRIST_CAMERAS, SIDE_INDICES

SIDE_CAMERAS = {
    side: tuple(f"observation.images.{camera}" for camera in cameras)
    for side, cameras in WRIST_CAMERAS.items()
}


class DatasetError(ValueError):
    """Missing, inconsistent or unsafe dataset input."""


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetError(f"{name} must be a finite number")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"{name} must be a nonnegative integer")
    return value


def _under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DatasetError("metadata path escapes dataset root")
    return path


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot read required metadata: {path.name}") from exc


def valid_sides_and_cameras(info: dict, mask: dict) -> tuple[list[str], list[str]]:
    """Honor feature presence plus field and element validity, never placeholders."""
    fields = mask.get("field_mask", {})
    elements = mask.get("element_mask", {}).get("observation.state", [])
    features = info.get("features", {})
    if not isinstance(fields, dict) or not isinstance(elements, list):
        raise DatasetError("invalid validity mask structure")
    if "observation.state" not in features or fields.get("observation.state") is not True:
        return [], []
    sides = [side for side, index in SIDE_INDICES.items()
             if len(elements) > index and elements[index] is True]
    cameras = [key for side in sides for key in SIDE_CAMERAS[side]
               if key in features and fields.get(key) is True]
    return sides, cameras


def _episode_metadata(root: Path, episode_index: int, pq) -> dict:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise DatasetError("LeRobot v3 episode metadata is missing")
    found = []
    for file in files:
        table = pq.read_table(file, filters=[("episode_index", "=", episode_index)])
        found.extend(table.to_pylist())
    if len(found) != 1:
        raise DatasetError(f"expected exactly one episode metadata row, found {len(found)}")
    return found[0]


def load_episode(root: Path | str, episode_index: int) -> dict:
    """Load exactly one episode without sorting, shifting, or rewriting its rows.

    A missing mask fails closed. Unavailable cameras are excluded and reported;
    their valid numerical side still participates in trend detection.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError("load_episode requires the optional pyarrow dependency") from exc
    episode_index = _integer(episode_index, "episode_index")
    root = Path(root).resolve(strict=True)
    info = _read_json(root / "meta/info.json")
    mask = _read_json(root / "dataset_configs/masks/mask.json")
    if info.get("codebase_version") != "v3.0":
        raise DatasetError("only LeRobot v3.0 is supported")
    sides, camera_keys = valid_sides_and_cameras(info, mask)
    meta = _episode_metadata(root, episode_index, pq)
    try:
        data_path = _under(root, info["data_path"].format(
            chunk_index=_integer(meta["data/chunk_index"], "data/chunk_index"),
            file_index=_integer(meta["data/file_index"], "data/file_index")))
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError("invalid v3 data location") from exc
    table = pq.read_table(data_path, filters=[("episode_index", "=", episode_index)])
    required = {"observation.state", "timestamp", "episode_index", "frame_index", "index"}
    if not required.issubset(table.column_names):
        raise DatasetError("required episode columns are missing")
    rows = table.to_pylist()
    if not rows or len(rows) != meta.get("length"):
        raise DatasetError("episode length differs from metadata; cross-shard episodes are unsupported")
    frames = [row["frame_index"] for row in rows]
    if frames != list(range(len(rows))):
        raise DatasetError("episode frame indices must be consecutive from zero")
    indices = [row["index"] for row in rows]
    if indices != list(range(meta["dataset_from_index"], meta["dataset_to_index"])):
        raise DatasetError("episode global index range differs from metadata")
    timestamps = [_finite(row["timestamp"], "timestamp") for row in rows]
    if timestamps[0] < 0 or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise DatasetError("timestamps must be nonnegative and strictly increasing")
    state = [row["observation.state"] for row in rows]
    if any(not isinstance(row, list) or len(row) != 20 for row in state):
        raise DatasetError("observation.state must have exactly 20 elements")
    for row in state:
        for side in sides:
            value = _finite(row[SIDE_INDICES[side]], f"{side} gripper")
            if not 0 <= value <= 1:
                raise DatasetError("valid gripper values must lie in [0,1]")
    cameras, issues = {}, {}
    for key in camera_keys:
        prefix = f"videos/{key}/"
        try:
            path = _under(root, info["video_path"].format(
                video_key=key, chunk_index=_integer(meta[prefix + "chunk_index"], "video chunk"),
                file_index=_integer(meta[prefix + "file_index"], "video file")))
            offset = _finite(meta[prefix + "from_timestamp"], "video offset")
            end = _finite(meta[prefix + "to_timestamp"], "video end")
            if offset < 0 or end <= offset or offset + timestamps[-1] >= end + 1e-5:
                raise DatasetError("camera timestamp interval does not cover episode")
            if not path.is_file():
                raise DatasetError("camera video is missing")
            cameras[key] = {"path": str(path), "from_timestamp": offset, "to_timestamp": end}
        except (KeyError, TypeError, ValueError) as exc:
            issues[key] = str(exc) if isinstance(exc, DatasetError) else "camera metadata is missing or invalid"
    for side in sides:
        for key in SIDE_CAMERAS[side]:
            if key not in camera_keys:
                issues[key] = "camera absent or masked invalid"
    source_fields = {key: value for key, value in meta.items()
                     if key.startswith(("source_", "original_", "crop_"))}
    identity = {
        "dataset_root": str(root), "release": root.parent.name, "subset": root.name,
        "episode_index": episode_index, "data_path": str(data_path),
        "dataset_from_index": meta["dataset_from_index"], "dataset_to_index": meta["dataset_to_index"],
        "tasks": meta.get("tasks", []), "source_metadata": source_fields,
        "source_episode_identity_available": any(k in source_fields for k in (
            "source_episode_id", "source_episode_index", "original_episode_id", "source_path")),
        "info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256((root / "dataset_configs/masks/mask.json").read_bytes()).hexdigest(),
    }
    if "timestamp_ns" in table.column_names:
        identity["first_timestamp_ns"] = rows[0]["timestamp_ns"]
        identity["last_timestamp_ns"] = rows[-1]["timestamp_ns"]
    return {"episode_id": f"{root.parent.name}/{root.name}/{episode_index}",
            "timestamps": timestamps, "state": state, "valid_sides": sides,
            "cameras": cameras, "camera_issues": issues, "identity": identity,
            "frame_indices": frames, "global_indices": indices}


def frame_requests(episode: dict, candidate: dict, config: dict) -> list[dict]:
    """Build dense 10 Hz boundary requests with separate same-wrist offsets."""
    side = candidate["side"]
    if side not in SIDE_CAMERAS or side not in episode["valid_sides"]:
        raise DatasetError("candidate belongs to an invalid gripper side")
    times = episode["timestamps"]
    start, end = candidate["start"], candidate["end"]
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(times)):
        raise DatasetError("candidate boundaries are outside episode")
    before = _finite(config.get("context_before_s", 0.5), "context_before_s")
    after = _finite(config.get("context_after_s", 0.5), "context_after_s")
    lookback = _finite(config.get("lookback_s", 2.0), "lookback_s")
    if min(before, after, lookback) < 0:
        raise DatasetError("frame context durations must be nonnegative")
    begin_time = times[start] - max(before, lookback)
    if candidate.get("hold_start") is not None:
        hold = _integer(candidate["hold_start"], "hold_start")
        if hold > start:
            raise DatasetError("hold_start must precede opening")
        begin_time = min(begin_time, times[hold])
    chosen = [i for i, time in enumerate(times)
              if begin_time - 1e-6 <= time <= times[end] + after + 1e-6]
    requests = []
    for camera in SIDE_CAMERAS[side]:
        spec = episode.get("cameras", {}).get(camera)
        if not spec:
            continue
        offset = _finite(spec["from_timestamp"], "camera offset")
        for index in chosen:
            video_time = offset + times[index]
            if video_time < offset or video_time >= spec.get("to_timestamp", float("inf")):
                raise DatasetError("requested frame crosses episode video boundary")
            requests.append({"side": side, "camera": camera, "frame_index": index,
                             "source_frame_index": episode.get("frame_indices", list(range(len(times))))[index],
                             "timestamp": times[index], "video_timestamp": video_time,
                             "video_path": spec["path"]})
    if not requests:
        raise DatasetError("no valid same-wrist camera is available")
    return requests


def extract_frames(episode: dict, candidate: dict, output_dir: Path | str, config: dict) -> list[dict]:
    """Decode same-wrist frames; require full temporal coverage from either eye.

    JPEG files are newly created, never overwritten. Returned image hashes cover
    the actual bytes supplied to the vision adapter.
    """
    requests = frame_requests(episode, candidate, config)
    output_dir = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (output_dir, *output_dir.parents)):
        raise DatasetError("frame output must not traverse a symlink")
    resolved = output_dir.resolve()
    source_root = episode.get("identity", {}).get("dataset_root")
    if source_root and resolved.is_relative_to(Path(source_root).resolve()):
        raise DatasetError("frame output overlaps source dataset")
    for request in requests:
        if resolved == Path(request["video_path"]).resolve():
            raise DatasetError("frame output overlaps source video")
    try:
        import cv2
    except ImportError as exc:
        raise DatasetError("extract_frames requires the optional opencv-python-headless dependency") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, failures = [], []
    camera_mode = config.get("camera_mode", "primary")
    if camera_mode not in ("primary", "stereo"):
        raise DatasetError("camera_mode must be primary or stereo")
    expected = {request["frame_index"] for request in requests}
    max_size = _integer(config.get("image_max_size_px", 512), "image_max_size_px")
    if max_size < 32:
        raise DatasetError("image_max_size_px must be at least 32")
    for camera in SIDE_CAMERAS[candidate["side"]]:
        group = [request for request in requests if request["camera"] == camera]
        if not group:
            continue
        cap = cv2.VideoCapture(group[0]["video_path"])
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                failures.append(camera)
                continue
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            for request in group:
                number = int(math.floor(request["video_timestamp"] * fps + 0.5))
                if number >= count:
                    failures.append(camera)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                ok, frame = cap.read()
                actual_number = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if not ok or actual_number != number:
                    failures.append(camera)
                    continue
                height, width = frame.shape[:2]
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    failures.append(camera)
                    continue
                image_bytes = buffer.tobytes()
                file = output_dir / f"{camera}_frame-{request['frame_index']:06d}.jpg"
                with file.open("xb") as handle:
                    handle.write(image_bytes)
                frames.append({**request, "path": str(file.resolve()),
                               "decoded_video_timestamp": number / fps,
                               "sha256": hashlib.sha256(image_bytes).hexdigest()})
        finally:
            cap.release()
        if camera_mode == "primary" and {frame["frame_index"] for frame in frames} == expected:
            break
    if {frame["frame_index"] for frame in frames} != expected:
        raise DatasetError("same-wrist video decoding failed to cover requested time window")
    return frames
