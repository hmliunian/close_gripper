import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from close_gripper.dataset import (
    DatasetError, SIDE_CAMERAS, extract_frames, frame_requests,
    load_episode, valid_sides_and_cameras,
)


def fixture_episode():
    return {"episode_id": "fixture", "timestamps": [i / 10 for i in range(10)],
            "state": [[0.0] * 20 for _ in range(10)], "valid_sides": ["left", "right"],
            "frame_indices": list(range(10)),
            "cameras": {camera: {"path": f"/{camera}.mp4", "from_timestamp": 0.5,
                                  "to_timestamp": 1.5}
                        for cameras in SIDE_CAMERAS.values() for camera in cameras},
            "identity": {}}


class FrameRequestTests(unittest.TestCase):
    def test_side_and_offset_and_dense_boundaries(self):
        episode = fixture_episode()
        episode["cameras"][SIDE_CAMERAS["left"][1]]["from_timestamp"] = 0.6
        episode["cameras"][SIDE_CAMERAS["left"][1]]["to_timestamp"] = 1.6
        frames = frame_requests(episode, {"side": "left", "start": 4, "end": 5},
                                {"context_before_s": 0.1, "context_after_s": 0.2, "lookback_s": 0})
        self.assertEqual({f["frame_index"] for f in frames}, {3, 4, 5, 6, 7})
        self.assertEqual({f["camera"] for f in frames}, set(SIDE_CAMERAS["left"]))
        for frame in frames:
            offset = episode["cameras"][frame["camera"]]["from_timestamp"]
            self.assertAlmostEqual(frame["video_timestamp"], offset + frame["timestamp"])

    def test_invalid_or_missing_same_side_never_uses_other_arm(self):
        episode = fixture_episode()
        episode["valid_sides"] = ["right"]
        with self.assertRaises(DatasetError):
            frame_requests(episode, {"side": "left", "start": 1, "end": 2}, {})
        episode["valid_sides"] = ["left", "right"]
        episode["cameras"] = {k: v for k, v in episode["cameras"].items() if "right_wrist" in k}
        with self.assertRaisesRegex(DatasetError, "same-wrist"):
            frame_requests(episode, {"side": "left", "start": 1, "end": 2}, {})

    def test_mask_field_element_and_feature_presence(self):
        features = {"observation.state": {}}
        features.update({key: {} for keys in SIDE_CAMERAS.values() for key in keys})
        info = {"features": features}
        mask = {"field_mask": {key: True for key in features},
                "element_mask": {"observation.state": [True] * 20}}
        mask["element_mask"]["observation.state"][9] = False
        mask["field_mask"][SIDE_CAMERAS["right"][0]] = False
        sides, cameras = valid_sides_and_cameras(info, mask)
        self.assertEqual(sides, ["right"])
        self.assertEqual(cameras, [SIDE_CAMERAS["right"][1]])
        mask["field_mask"]["observation.state"] = False
        self.assertEqual(valid_sides_and_cameras(info, mask), ([], []))
        self.assertEqual(valid_sides_and_cameras(info, {}), ([], []))

    def test_episode_video_end_is_exclusive(self):
        episode = fixture_episode()
        episode["cameras"][SIDE_CAMERAS["left"][0]]["to_timestamp"] = 0.8
        with self.assertRaisesRegex(DatasetError, "boundary"):
            frame_requests(episode, {"side": "left", "start": 4, "end": 5}, {})

    def test_output_overlap_and_symlink_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = fixture_episode()
            episode["identity"]["dataset_root"] = str(root)
            candidate = {"side": "left", "start": 2, "end": 3}
            with self.assertRaisesRegex(DatasetError, "overlaps"):
                extract_frames(episode, candidate, root / "output", {})
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(DatasetError, "symlink"):
                extract_frames(episode, candidate, link, {})
            episode["identity"] = {}
            with self.assertRaisesRegex(DatasetError, "symlink"):
                extract_frames(episode, candidate, link / "nested" / "frames", {})


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable")
class EpisodeReadTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ("meta/episodes/chunk-000", "dataset_configs/masks", "data/chunk-000", "videos"):
            (self.root / directory).mkdir(parents=True)
        self.info = {"codebase_version": "v3.0", "fps": 10,
                     "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                     "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                     "features": {"observation.state": {}, **{key: {} for keys in SIDE_CAMERAS.values() for key in keys}}}
        (self.root / "meta/info.json").write_text(json.dumps(self.info))
        self.mask = {"field_mask": {key: True for key in self.info["features"]},
                     "element_mask": {"observation.state": [True] * 20}}
        (self.root / "dataset_configs/masks/mask.json").write_text(json.dumps(self.mask))
        self.metadata = []
        self.rows = []
        for episode_index in (0, 1):
            meta = {"episode_index": episode_index, "length": 3,
                    "dataset_from_index": episode_index * 3, "dataset_to_index": episode_index * 3 + 3,
                    "data/chunk_index": 0, "data/file_index": 0, "tasks": ["fixture"]}
            for keys in SIDE_CAMERAS.values():
                for key in keys:
                    prefix = f"videos/{key}/"
                    meta.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                                 prefix + "from_timestamp": 0.5 + episode_index,
                                 prefix + "to_timestamp": 0.8 + episode_index})
                    path = self.root / self.info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"not decoded in read test")
            self.metadata.append(meta)
            for frame in range(3):
                state = [0.0] * 20
                state[9] = 0.2 + frame / 10
                state[19] = episode_index / 2
                self.rows.append({"episode_index": episode_index, "frame_index": frame,
                                  "index": episode_index * 3 + frame, "timestamp": frame / 10,
                                  "observation.state": state})
        self._write_tables()

    def _write_tables(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(self.metadata), self.root / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist(self.rows), self.root / "data/chunk-000/file-000.parquet")

    def test_shared_shard_exact_episode_and_input_unchanged(self):
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.root.rglob("*") if path.is_file()}
        episode = load_episode(self.root, 1)
        self.assertEqual(episode["timestamps"], [0.0, 0.1, 0.2])
        self.assertEqual(episode["global_indices"], [3, 4, 5])
        self.assertEqual([row[19] for row in episode["state"]], [0.5] * 3)
        self.assertEqual(episode["cameras"][SIDE_CAMERAS["left"][0]]["from_timestamp"], 1.5)
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.root.rglob("*") if path.is_file()})

    def test_missing_mask_rejected(self):
        (self.root / "dataset_configs/masks/mask.json").unlink()
        with self.assertRaises(DatasetError):
            load_episode(self.root, 0)

    def test_missing_camera_keeps_numerical_side_and_reports_reason(self):
        camera = SIDE_CAMERAS["left"][0]
        path = self.root / self.info["video_path"].format(video_key=camera, chunk_index=0, file_index=0)
        path.unlink()
        episode = load_episode(self.root, 0)
        self.assertIn("left", episode["valid_sides"])
        self.assertNotIn(camera, episode["cameras"])
        self.assertIn("missing", episode["camera_issues"][camera])

    def test_nonmonotonic_timestamp_rejected_without_sorting(self):
        self.rows[2]["timestamp"] = 0.05
        self._write_tables()
        with self.assertRaisesRegex(DatasetError, "strictly increasing"):
            load_episode(self.root, 0)

    def test_length_or_index_mismatch_rejected(self):
        self.metadata[0]["length"] = 4
        self._write_tables()
        with self.assertRaisesRegex(DatasetError, "length"):
            load_episode(self.root, 0)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "optional opencv is unavailable")
class DecodeTests(unittest.TestCase):
    def test_fallback_uses_only_other_eye_of_same_wrist(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, length in (("short", 1), ("complete", 20)):
                writer = cv2.VideoWriter(str(root / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
                if not writer.isOpened():
                    self.skipTest("MJPG codec unavailable")
                for i in range(length):
                    writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
                writer.release()
            episode = fixture_episode()
            episode["cameras"][SIDE_CAMERAS["left"][0]]["path"] = str(root / "short.avi")
            episode["cameras"][SIDE_CAMERAS["left"][1]]["path"] = str(root / "complete.avi")
            frames = extract_frames(episode, {"side": "left", "start": 2, "end": 3}, root / "frames",
                                    {"context_before_s": 0, "context_after_s": 0, "lookback_s": 0})
            self.assertEqual(len(frames), 2)
            self.assertEqual({f["camera"] for f in frames}, {SIDE_CAMERAS["left"][1]})

    def test_primary_side_offset_real_decode_and_no_overwrite(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            if not writer.isOpened():
                self.skipTest("MJPG codec unavailable")
            for i in range(20):
                writer.write(np.full((32, 32, 3), i * 10, dtype=np.uint8))
            writer.release()
            episode = fixture_episode()
            for value in episode["cameras"].values():
                value["path"] = str(video)
            candidate = {"side": "left", "start": 2, "end": 3}
            config = {"context_before_s": 0, "context_after_s": 0, "lookback_s": 0}
            frames = extract_frames(episode, candidate, root / "frames", config)
            self.assertEqual(len(frames), 2)
            self.assertEqual({f["camera"] for f in frames}, {SIDE_CAMERAS["left"][0]})
            self.assertAlmostEqual(frames[0]["decoded_video_timestamp"], 0.7)
            decoded = cv2.imread(frames[0]["path"])
            self.assertAlmostEqual(float(decoded.mean()), 70, delta=2)
            self.assertEqual(frames[0]["sha256"], hashlib.sha256(Path(frames[0]["path"]).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                extract_frames(episode, candidate, root / "frames", config)


if __name__ == "__main__":
    unittest.main()
