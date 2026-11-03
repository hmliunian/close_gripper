"""Independent derived columns, source identity binding and sample metadata."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline
from close_gripper.dataset import load_episode
from close_gripper.export import TARGET_KEY, VALID_KEY, derive_subset, prepare_release_manifest, _destination_file


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is unavailable")
class ExportTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.source = self.workspace / "source/release/subset"
        self.patcher = patch.object(pipeline, "WORKSPACE", self.workspace)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        for path in ("meta/episodes/chunk-000", "data/chunk-000", "dataset_configs/masks"):
            (self.source / path).mkdir(parents=True)
        camera = "observation.images.cam_right_wrist_left"
        self.info = dict(codebase_version="v3.0", fps=10, total_episodes=3, total_frames=12,
                         total_tasks=1, total_videos=1, total_data_files=1, total_chunks=1,
                         splits={"train": "0:3"}, features={"observation.state": dict(dtype="float32", shape=[20]),
                                                           camera: dict(dtype="video", shape=[256, 256, 3])},
                         data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                         video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        (self.source / "meta/info.json").write_text(json.dumps(self.info))
        self.mask = dict(dataset="subset", field_mapping={},
                         field_mask={"observation.state": True, camera: True},
                         element_mask={"observation.state": [True] * 20},
                         image_pixel_mask=dict(path="pixels.npz"))
        (self.source / "dataset_configs/masks/mask.json").write_text(json.dumps(self.mask))
        (self.source / "dataset_configs/masks/pixels.npz").write_bytes(b"mask fixture")
        self.video = self.source / self.info["video_path"].format(video_key=camera, chunk_index=0, file_index=0)
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"shared video bytes; decoding belongs to loader integration")
        rows, metas = [], []
        for index in range(3):
            meta = dict(episode_index=index, length=4, dataset_from_index=index * 4, dataset_to_index=index * 4 + 4,
                        tasks=["fixture"], **{"data/chunk_index": 0, "data/file_index": 0})
            prefix = f"videos/{camera}/"
            meta.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                         prefix + "from_timestamp": index * .5, prefix + "to_timestamp": index * .5 + .4})
            metas.append(meta)
            for frame in range(4):
                state = [0.] * 20
                state[9], state[19] = .2 + .1 * frame, .6
                rows.append(dict(episode_index=index, frame_index=frame, index=index * 4 + frame,
                                 timestamp=frame / 10, timestamp_ns=1000000000000000000 + (index * 4 + frame) * 100000000,
                                 task_index=0, **{"observation.state": state}))
        table = pa.Table.from_pylist(rows)
        state_col = table.column_names.index("observation.state")
        table = table.set_column(state_col, "observation.state", pa.array([r["observation.state"] for r in rows], type=pa.list_(pa.float32(), 20)))
        self.data_path = self.source / "data/chunk-000/file-000.parquet"
        pq.write_table(table, self.data_path)
        pq.write_table(pa.Table.from_pylist(metas), self.source / "meta/episodes/chunk-000/file-000.parquet")
        pq.write_table(pa.Table.from_pylist([dict(task_index=0, task="fixture")]), self.source / "meta/tasks.parquet")
        (self.source / ".orion").mkdir()
        (self.source / ".orion/COMPLETE.json").write_text('{"publication_ready":true}')

    def artifact(self, index=1):
        episode = load_episode(self.source, index)
        return dict(episode_id=episode["episode_id"], identity=episode["identity"],
                    timestamps=episode["timestamps"], frame_indices=episode["frame_indices"],
                    **{TARGET_KEY: [[.1, 0.]] * 4, VALID_KEY: [[False, True]] * 4})

    def output(self, name="export"):
        return self.workspace / "runs" / name

    def test_selected_rows_keep_every_old_column_and_sparse_source_indices(self):
        import pyarrow.parquet as pq
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob("*") if p.is_file()}
        summary = derive_subset(self.source, [self.artifact()], self.output(), episode_indices=[1, 2])
        original = pq.read_table(self.data_path, filters=[("episode_index", "in", [1, 2])])
        result = pq.read_table(self.output() / self.data_path.relative_to(self.source))
        self.assertTrue(result.select(original.column_names).equals(original, check_metadata=True))
        self.assertEqual(result["index"].to_pylist(), list(range(4, 12)))
        self.assertEqual(result[TARGET_KEY].to_pylist()[:4], [[row[9], 0.] for row in original["observation.state"].to_pylist()[:4]])
        self.assertEqual(result[TARGET_KEY].to_pylist()[4:], [[row[9], row[19]] for row in original["observation.state"].to_pylist()[4:]])
        self.assertEqual(result[VALID_KEY].to_pylist(), [[False, True]] * 4 + [[False, False]] * 4)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob("*") if p.is_file()})
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["frames"], 8)
        self.assertFalse(summary["publication_ready"])
        self.assertFalse((self.output() / ".orion/COMPLETE.json").exists())
        self.assertEqual(summary["tasks"]["fixture"]["changed_episodes"], 1)
        self.assertEqual(summary["tasks"]["fixture"]["passthrough_episodes"], 1)

    def test_metadata_masks_statistics_and_video_copy_match_selected_data(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        derive_subset(self.source, [self.artifact()], self.output())
        info = json.loads((self.output() / "meta/info.json").read_text())
        self.assertEqual((info["total_episodes"], info["total_frames"], info["total_data_files"], info["total_videos"]), (1, 4, 1, 1))
        self.assertEqual(info["features"][TARGET_KEY]["shape"], [2])
        self.assertEqual(info["ngad_extensions"]["gripper_targets"]["episode_indices"], [1])
        meta = pq.read_table(self.output() / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        self.assertEqual(meta[0]["dataset_from_index"], 4)
        self.assertEqual(meta[0]["videos/observation.images.cam_right_wrist_left/from_timestamp"], .5)
        stats = json.loads((self.output() / "meta/gripper_target_stats.json").read_text())
        self.assertEqual(stats["valid_side_frames"], [0, 4])
        self.assertEqual(stats["target"][1]["mean"], 0.)
        self.assertEqual((self.output() / "dataset_configs/masks/mask.json").read_bytes(), (self.source / "dataset_configs/masks/mask.json").read_bytes())
        video = self.output() / self.video.relative_to(self.source)
        self.assertEqual(video.read_bytes(), self.video.read_bytes())
        self.assertNotEqual(video.stat().st_ino, self.video.stat().st_ino)
        schema = pq.read_schema(self.output() / self.data_path.relative_to(self.source))
        self.assertEqual(schema.field(TARGET_KEY).type, pa.list_(pa.float32(), 2))
        self.assertEqual(schema.field(VALID_KEY).type, pa.list_(pa.bool_(), 2))

    def test_wrong_root_time_frame_or_source_signature_is_rejected(self):
        edits = [lambda a: a["identity"].update(dataset_root="/tmp/not-this-source"),
                 lambda a: a["timestamps"].__setitem__(1, .15),
                 lambda a: a["frame_indices"].__setitem__(0, 100),
                 lambda a: a["identity"].update(info_sha256="stale"),
                 lambda a: a.update(episode_id="wrong")]
        for ordinal, edit in enumerate(edits):
            artifact = self.artifact()
            edit(artifact)
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                derive_subset(self.source, [artifact], self.output(str(ordinal)))
            self.assertFalse(self.output(str(ordinal)).exists())

    def test_preview_is_not_exported_as_a_valid_label(self):
        artifact = self.artifact()
        artifact["preview_target"] = artifact.pop(TARGET_KEY)
        with self.assertRaisesRegex(ValueError, "previews"):
            derive_subset(self.source, [artifact], self.output())

    def test_full_cycle_fingerprint_rejects_changed_state_with_same_metadata_and_time(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        artifact = self.artifact()
        before = load_episode(self.source, 1)
        artifact.update(method_version="full-cycle-v1", episode_sha256=pipeline.episode_fingerprint(before))
        table = pq.read_table(self.data_path)
        states = table["observation.state"].to_pylist()
        states[4][19] = .25
        table = table.set_column(table.column_names.index("observation.state"), "observation.state",
                                 pa.array(states, type=table.schema.field("observation.state").type))
        pq.write_table(table, self.data_path)
        current = load_episode(self.source, 1)
        self.assertEqual(before["identity"], current["identity"])
        self.assertEqual(before["timestamps"], current["timestamps"])
        with self.assertRaisesRegex(ValueError, "episode_sha256"):
            derive_subset(self.source, [artifact], self.output())
        self.assertFalse(self.output().exists())

    def test_full_cycle_requires_digest_and_any_supplied_digest_is_checked(self):
        for ordinal, change in enumerate((dict(method_version="full-cycle-v1"),
                                         {"annotation.gripper_phase": [["hold", "hold"]] * 4},
                                         dict(episode_sha256="stale"))):
            artifact = self.artifact()
            artifact.update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "episode_sha256"):
                derive_subset(self.source, [artifact], self.output(str(ordinal)))
        artifact = self.artifact()
        artifact["provenance"] = dict(kind="controlled_loader_test_fixture")
        result = derive_subset(self.source, [artifact], self.output("controlled"))
        self.assertEqual(result["label_sources"][0]["source_binding"], "controlled_fixture_metadata_only")
        result = derive_subset(self.source, [self.artifact()], self.output("legacy"))
        self.assertEqual(result["label_sources"][0]["source_binding"], "legacy_metadata_only")

    def test_full_artifact_sidecar_is_portable_and_sha_bound(self):
        artifact = self.artifact()
        artifact.update(method_version="full-cycle-v1", episode_sha256=pipeline.episode_fingerprint(load_episode(self.source, 1)))
        artifact["annotation.gripper_phase"] = [["untouched", "hold"]] * 4
        artifact["annotation.gripper_target_source"] = [["original", "real_vlm"]] * 4
        original = self.workspace / "targets.json"
        raw = (json.dumps(artifact, ensure_ascii=False, indent=4) + "\n").encode()
        original.write_bytes(raw)
        result = derive_subset(self.source, [original], self.output())
        source = result["label_sources"][0]
        self.assertFalse(Path(source["artifact_path"]).is_absolute())
        sidecar = self.output() / source["artifact_path"]
        self.assertEqual(sidecar.read_bytes(), raw)
        self.assertEqual(source["artifact_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(source["episode_sha256"], artifact["episode_sha256"])
        self.assertEqual(result["episode_fingerprints_verified"], [1])
        self.assertEqual(json.loads(sidecar.read_text()), artifact)

    def test_source_resolving_inside_root_still_cannot_supply_escaping_video_path(self):
        relative = self.info["video_path"]
        for ordinal, bad in enumerate(("../subset/" + relative, str(self.source / relative))):
            self.info["video_path"] = bad
            (self.source / "meta/info.json").write_text(json.dumps(self.info))
            # load_episode resolves these back into the source, but export must
            # reject their unsafe lexical path before constructing a destination.
            artifact = self.artifact()
            with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "relative"):
                derive_subset(self.source, [artifact], self.output(str(ordinal)))
            self.assertFalse(self.output(str(ordinal)).exists())

    def test_destination_resolution_rejects_existing_output_symlink(self):
        output = self.output()
        output.mkdir(parents=True)
        (output / "escape").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            _destination_file(output, "escape/file.mp4")

    def test_missing_labels_preserve_selected_episode(self):
        import pyarrow.parquet as pq
        derive_subset(self.source, [], self.output(), episode_indices=[2])
        result = pq.read_table(self.output() / self.data_path.relative_to(self.source))
        self.assertEqual(result[VALID_KEY].to_pylist(), [[False, False]] * 4)
        self.assertEqual(result[TARGET_KEY].to_pylist(), [[s[9], s[19]] for s in result["observation.state"].to_pylist()])

    def test_validity_flags_and_values_are_checked(self):
        for ordinal, invalid in enumerate((True, float("nan"), -1., 1.1)):
            artifact = self.artifact()
            artifact[TARGET_KEY][0][1] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                derive_subset(self.source, [artifact], self.output(str(ordinal)))
        artifact = self.artifact()
        artifact[VALID_KEY][0][1] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            derive_subset(self.source, [artifact], self.output())

    def test_invalid_side_cannot_gain_labels(self):
        self.mask["element_mask"]["observation.state"][9] = False
        (self.source / "dataset_configs/masks/mask.json").write_text(json.dumps(self.mask))
        artifact = self.artifact()
        artifact[VALID_KEY] = [[True, True]] * 4
        with self.assertRaisesRegex(ValueError, "unavailable"):
            derive_subset(self.source, [artifact], self.output())

    def test_export_refuses_duplicate_selection_existing_output_and_source_overlap(self):
        artifact = self.artifact()
        for selected in ([], [1, 1], [True], [2]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                derive_subset(self.source, [artifact], self.output(), episode_indices=selected)
        with self.assertRaises(ValueError):
            derive_subset(self.source, [artifact], self.source / "output")
        derive_subset(self.source, [artifact], self.output())
        with self.assertRaises(FileExistsError):
            derive_subset(self.source, [artifact], self.output())

    def test_symlink_output_is_rejected(self):
        (self.workspace / "runs").mkdir()
        (self.workspace / "runs/link").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            derive_subset(self.source, [self.artifact()], self.workspace / "runs/link/export")

    def test_release_plan_reads_metadata_only_and_counts_coverage_denominators(self):
        import pyarrow.parquet as pq
        artifact = self.artifact()
        original_read = pq.read_table
        reads = []
        def metadata_only(path, *args, **kwargs):
            reads.append(str(path))
            if "/data/" in str(path):
                raise AssertionError("planning must not read source data")
            return original_read(path, *args, **kwargs)
        with patch.object(pq, "read_table", side_effect=metadata_only), patch("close_gripper.export.load_episode", side_effect=AssertionError("not metadata-only")):
            plan = prepare_release_manifest([self.source.parent], self.output(), episode_targets=[artifact])
        self.assertEqual((plan["release_count"], plan["subset_count"], plan["total_episodes"], plan["total_frames"]), (1, 1, 3, 12))
        self.assertEqual(plan["status"], "prepared_not_derived")
        self.assertFalse(plan["publication_ready"])
        subset = plan["releases"][0]["subsets"][0]
        self.assertFalse(Path(subset["expected_staging_root"]).exists())
        self.assertEqual(subset["coverage"]["valid_side_frames"], [0, 4])
        self.assertEqual(subset["coverage"]["artifact_episode_fraction"], 1 / 3)
        self.assertEqual(subset["coverage"]["valid_target_side_fraction"], 1 / 6)
        self.assertEqual(subset["tasks"]["fixture"]["episodes"], 3)
        self.assertEqual(pq.read_table(self.output() / subset["episode_inventory"]).num_rows, 3)
        self.assertTrue(reads)

    def test_release_plan_rejects_stale_label_metadata_and_unknown_roots(self):
        artifact = self.artifact()
        artifact["identity"]["dataset_to_index"] += 1
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_release_manifest([self.source.parent], self.output(), episode_targets=[artifact])
        self.assertFalse(self.output().exists())
        artifact = self.artifact()
        artifact["identity"]["dataset_root"] = str(self.workspace / "another_release/subset")
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_release_manifest([self.source.parent], self.output(), episode_targets=[artifact])

    def test_release_plan_without_artifacts_does_not_claim_processed_episodes(self):
        plan = prepare_release_manifest([self.source.parent], self.output())
        subset = plan["releases"][0]["subsets"][0]
        self.assertEqual(subset["coverage"]["artifact_episodes"], 0)
        self.assertEqual(subset["coverage"]["valid_target_frames"], 0)
        self.assertEqual(subset["derivation_status"], "not_generated")


if __name__ == "__main__":
    unittest.main()
