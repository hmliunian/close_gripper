"""Freeze a small main-camera review cohort and decode its original 10 Hz frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.dataset import _episode_metadata, _under
from close_gripper.cycles import detect_cycles
from close_gripper.pipeline import create_run, dump, episode_fingerprint

TASKS = ["拿起刷子", "拿起杯子", "拿起魔方", "拿起海绵擦", "拿起芒果",
         "插入缝隙刷", "揭开锅盖", "盖上锅盖", "橱柜门打开", "叠衣服"]
HEAD_CAMERAS = ["observation.images.cam_head_left", "observation.images.cam_head_right"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(manifest_path, output):
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
    source = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(source.read_text())
    rows = []
    for task in TASKS:
        pair = [r for r in manifest["episodes"] if r["task"] == task]
        if len(pair) != 2 or len({r["source_id"] for r in pair}) != 2:
            raise ValueError("exactly two independent sources required: " + task)
        rows.extend(pair)
    out = create_run(output, [source, *[r["dataset_root"] for r in rows]])
    dump(out / "selection.json", dict(schema="close_gripper.head_selection.v1", tasks=TASKS,
        source_manifest=str(source), source_manifest_sha256=sha(source),
        selection_policy="ten declared tasks; both frozen independent-source episodes; no visual/outcome filtering",
        visual_camera_group="head", episode_count=len(rows), episodes=rows))

    def process(item):
        ordinal, row = item
        cached = source.parent / row["episode_cache"]
        if sha(cached) != row["episode_cache_file_sha256"]:
            raise ValueError("cached input checksum mismatch")
        ep = json.loads(cached.read_text())
        root = Path(row["dataset_root"])
        info = json.loads((root / "meta/info.json").read_text())
        mask = json.loads((root / "dataset_configs/masks/mask.json").read_text())
        meta = _episode_metadata(root, row["episode_index"], pq)
        identity = ep.get("identity", {})
        if (ep.get("episode_id") != row["episode_id"] or identity.get("dataset_root") != str(root)
                or identity.get("episode_index") != row["episode_index"]):
            raise ValueError("cached episode identity differs from selected source")
        data_path = _under(root, info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]))
        original = pq.read_table(data_path, filters=[("episode_index", "=", row["episode_index"])],
            columns=["observation.state", "timestamp", "frame_index", "index"]).to_pydict()
        if (ep["state"] != original["observation.state"] or ep["timestamps"] != original["timestamp"]
                or ep.get("frame_indices") != original["frame_index"]
                or ep.get("global_indices") != original["index"]):
            raise ValueError("cached observations or indices differ from source Parquet")
        ep["cameras"], ep["camera_issues"] = {}, {}
        ep["vision_camera_group"] = "head"
        for camera in HEAD_CAMERAS:
            if camera not in info["features"] or mask["field_mask"].get(camera) is not True:
                raise ValueError("head camera absent or invalid")
            p = "videos/" + camera + "/"
            path = _under(root, info["video_path"].format(video_key=camera,
                chunk_index=meta[p + "chunk_index"], file_index=meta[p + "file_index"]))
            a, b = float(meta[p + "from_timestamp"]), float(meta[p + "to_timestamp"])
            if not path.is_file() or not 0 <= a < b or a + ep["timestamps"][-1] >= b + 1e-5:
                raise ValueError("head video does not cover episode")
            ep["cameras"][camera] = dict(path=str(path), from_timestamp=a, to_timestamp=b)
        directory = out / f"episode_{ordinal:02d}"
        directory.mkdir()
        dump(directory / "episode.json", ep)
        numeric = detect_cycles(ep, {"motion_min_delta": .006000001})
        for c in numeric:
            c["cameras"] = HEAD_CAMERAS
            c["camera_status"] = "available_shared_head_view"
            c["shape_hint"] = ("full_valley" if c["closing_start"] is not None and c["release_start"] is not None
                else "descending_half" if c["closing_start"] is not None
                else "ascending_half" if c["release_start"] is not None else "static_context")
        dump(directory / "focus_intervals.json", numeric)
        frames, sheets = [], []
        for camera, spec in ep["cameras"].items():
            eye = camera.rsplit("_", 1)[-1]
            image_dir = directory / eye
            image_dir.mkdir()
            capture = cv2.VideoCapture(spec["path"])
            if not capture.isOpened():
                raise ValueError("cannot decode head camera")
            images = []
            fps = capture.get(cv2.CAP_PROP_FPS)
            for index, t in enumerate(ep["timestamps"]):
                requested = spec["from_timestamp"] + t
                video_frame = round(requested * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                ok, pixels = capture.read()
                pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not ok or abs(pts - requested) > .051:
                    raise ValueError(f"failed or misaligned frame {camera}:{index}: {pts}/{requested}")
                image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                dest = image_dir / f"frame_{index:04d}.jpg"
                image.save(dest, quality=95)
                images.append(image)
                frames.append(dict(camera=camera, camera_group="head", frame_index=index,
                    timestamp=t, video_timestamp=requested, decoded_timestamp=pts,
                    source_video_frame=video_frame, path=str(dest), sha256=sha(dest)))
            capture.release()
            # Dense native-size panels retain every 10 Hz frame for semantic review.
            for start in range(0, len(images), 36):
                indices = list(range(start, min(start + 36, len(images))))
                canvas = Image.new("RGB", (6 * 260, ((len(indices) + 5) // 6) * 282 + 28), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text((5, 7), f"episode_{ordinal:02d} / {row['episode_index']} / head_{eye} / frames {indices[0]}-{indices[-1]}", fill="black")
                for j, index in enumerate(indices):
                    x, y = (j % 6) * 260, 28 + (j // 6) * 282
                    canvas.paste(images[index], (x, y))
                    draw.text((x + 3, y + 258), f"f{index:03d} t={ep['timestamps'][index]:.1f}", fill="black")
                dest = directory / f"sheet_{eye}_{start:04d}.jpg"
                canvas.save(dest, quality=94)
                sheets.append(dict(camera=camera, path=str(dest), frame_indices=indices, sha256=sha(dest)))
        dump(directory / "frames.json", frames)
        dump(directory / "sheets.json", sheets)
        record = dict(directory=directory.name, task=row["task"], episode_index=row["episode_index"],
            episode_id=ep["episode_id"], source_id=row["source_id"], valid_sides=ep["valid_sides"],
            frames=len(ep["timestamps"]), source_episode_cache_sha256=sha(cached),
            episode_sha256=episode_fingerprint(ep), head_image_count=len(frames),
            max_decode_alignment_error_s=max(abs(f["decoded_timestamp"]-f["video_timestamp"]) for f in frames),
            shape_counts={name:sum(c["shape_hint"]==name for c in numeric) for name in
                ("full_valley", "descending_half", "ascending_half", "static_context")})
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(process, enumerate(rows)))
    dump(out / "manifest.json", dict(schema="close_gripper.head_inputs.v1", status="complete",
        camera_group="head", episodes=entries, total_frames=sum(r["frames"] for r in entries),
        total_images=sum(r["head_image_count"] for r in entries), source_data_changed=False))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.manifest, args.output))
