"""Complete-shard evaluation must preserve source rows and fail honestly."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline, profile_bulk
from close_gripper.profile_clean import generate_profile_targets
from close_gripper.task_profiles import SCHEMA, UNITS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def catalog():
    active = dict(status="visual_reviewed", automatic_use=True, provenance=dict(review_source="model"), counts={},
        closed_bands=[dict(id="hold", lower=.18, center=.2, upper=.22)],
        open_bands=[dict(id="open", lower=.79, center=.8, upper=.81)])
    disabled = dict(active, status="needs_review", automatic_use=False)
    return dict(schema=SCHEMA, units=UNITS, version="bulk-fixture", minimum_hold_s=.6,
        maximum_sample_gap_s=.15, maximum_plateau_range=.02, excluded_source_ids=["source-3"],
        ambiguous_aliases={"shared": ["夹持", "待复核"]}, scope=dict(tasks=3),
        tasks=[dict(task="夹持", aliases=["Grip."], sides={"right": active}),
               dict(task="双臂", aliases=["Fold."], sides={"left":copy.deepcopy(active), "right":copy.deepcopy(active)}),
               dict(task="待复核", aliases=["Review."], sides={"right": disabled})])


def synthetic_episode(task="夹持", source="source-0"):
    states = [[.7] * 20 for _ in range(9)]
    for row in states:
        row[19] = .2
    return dict(episode_id="fixture/0", state=states, timestamps=[i/10 for i in range(9)],
        valid_sides=["right"], cameras={}, identity=dict(source_id=source, source_metadata=dict(source_tasks=[task])))


class ProjectionContracts(unittest.TestCase):
    def test_projection_preserves_task_alias_conflict_and_exclusion_outcomes(self):
        full = catalog()
        projected = profile_bulk._project_catalog(full, "夹持")
        before = copy.deepcopy(full)
        for kind in ("canonical", "alias", "ambiguous", "multiple", "source_excluded"):
            ep = synthetic_episode()
            if kind == "alias":
                ep["identity"] = dict(tasks=["Grip."])
            elif kind == "ambiguous":
                ep["identity"] = dict(tasks=["shared"])
            elif kind == "multiple":
                ep["identity"]["source_metadata"]["source_tasks"] = ["夹持", "待复核"]
            elif kind == "source_excluded":
                ep["identity"]["source_id"] = "source-3"
            with self.subTest(kind=kind):
                self.assertEqual(generate_profile_targets(ep, projected), generate_profile_targets(ep, full))
        self.assertEqual(full, before)
        self.assertEqual(projected["scope"], full["scope"])
        self.assertEqual(projected["ambiguous_aliases"], full["ambiguous_aliases"])


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow unavailable")
class BulkDatasetContracts(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq = pa, pq
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.source = self.workspace / "source/release-long/subset"
        self.scope = self.workspace / "scope"
        self.scope.mkdir()
        for relative in ("meta/episodes/chunk-000", "dataset_configs/masks", "data/chunk-000"):
            (self.source / relative).mkdir(parents=True)
        self.catalog = catalog()
        self.catalog_path = self.scope / "catalog.json"
        dump(self.catalog_path, self.catalog)
        info = dict(codebase_version="v3.0", fps=10,
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", features={"observation.state": {}})
        mask = dict(field_mask={"observation.state":True}, element_mask={"observation.state":[True]*20})
        dump(self.source / "meta/info.json", info)
        dump(self.source / "dataset_configs/masks/mask.json", mask)
        self.metadata, self.ledger, self.rows = [], [], [[], []]
        global_index = 0
        for index, task in enumerate(("夹持", "待复核", "双臂", "夹持")):
            values = ([.8, .8, .6, .4, .2] + [.2]*8 + [.4, .6, .8, .8]) if index == 0 else [.2]*9
            shard = index // 2
            self.metadata.append(dict(episode_index=index, length=len(values), dataset_from_index=global_index,
                dataset_to_index=global_index+len(values), **{"data/chunk_index":0, "data/file_index":shard},
                tasks=[dict(夹持="Grip.", 双臂="Fold.", 待复核="Review.")[task]], source_tasks=[task]))
            self.ledger.append(dict(episode_index=index, dataset_root=str(self.source), task=task,
                source_id=f"source-{index}", source_path=f"/raw/source-{index}/episode.mcap",
                valid_sides=["left", "right"], selected=task != "待复核"))
            for frame, value in enumerate(values):
                state = [.0]*20
                state[0], state[9], state[19] = frame+index, .2 if index == 2 else .7, value
                self.rows[shard].append(dict(episode_index=index, frame_index=frame, index=global_index+frame,
                    timestamp=frame/10, timestamp_ns=1_700_000_000_000_000_000+(global_index+frame)*100_000_000,
                    **{"observation.state":state, "action":[float(index)]*20, "extra":"原字段"+str(frame)}))
            global_index += len(values)
        self.schema = pa.schema([("episode_index",pa.int64()),("frame_index",pa.int64()),("index",pa.int64()),
            ("timestamp",pa.float64()),("timestamp_ns",pa.int64()),("observation.state",pa.list_(pa.float32(),20)),
            ("action",pa.list_(pa.float32(),20)),("extra",pa.string())], metadata={b"fixture":b"unchanged"})
        self._write_sources()
        self._write_manifest()
        self.patch = patch.object(pipeline, "WORKSPACE", self.workspace)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def _write_sources(self):
        self.pq.write_table(self.pa.Table.from_pylist(self.metadata), self.source / "meta/episodes/chunk-000/file-000.parquet")
        for i, rows in enumerate(self.rows):
            self.pq.write_table(self.pa.Table.from_pylist(rows,schema=self.schema), self.source / f"data/chunk-000/file-{i:03d}.parquet")

    def _write_manifest(self):
        ledger = self.scope / "source.jsonl.gz"
        with gzip.open(ledger,"wt",encoding="utf-8") as stream:
            for row in self.ledger:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
        metadata_paths = ["meta/info.json","dataset_configs/masks/mask.json","meta/episodes/chunk-000/file-000.parquet"]
        self.manifest = dict(schema=profile_bulk.MANIFEST_SCHEMA, complete_source_scope=True,
            catalog_sha256=digest(self.catalog_path), selected_tasks=["夹持","双臂"],
            datasets=[dict(dataset_id="short__subset",dataset_root=str(self.source),release="short",
                release_directory="release-long",subset="subset",info_sha256=digest(self.source/metadata_paths[0]),
                mask_sha256=digest(self.source/metadata_paths[1]),metadata_files=[dict(path=p,sha256=digest(self.source/p)) for p in metadata_paths],
                source_episodes_path=ledger.name,source_episodes_sha256=digest(ledger),
                shards=[dict(path=f"data/chunk-000/file-{i:03d}.parquet",sha256=digest(self.source/f"data/chunk-000/file-{i:03d}.parquet"),
                    rows=len(rows),episodes=sorted({r['episode_index'] for r in rows})) for i,rows in enumerate(self.rows)])])
        self.manifest_path = self.scope / "manifest.json"
        dump(self.manifest_path,self.manifest)

    def run_bulk(self,name="bulk",workers=1):
        self.output = self.workspace / "runs" / name
        return profile_bulk.run_profile_dataset(self.manifest_path,self.catalog_path,self.output,workers=workers)

    def audit_rows(self):
        paths=sorted((self.output / "audit/short__subset/episodes/data/chunk-000").glob("*.parquet"))
        return [r for p in paths for r in self.pq.read_table(p).to_pylist()]

    def test_complete_mixed_shards_keep_original_rows_and_separate_evaluation_from_training(self):
        before={str(p):digest(p) for p in self.source.rglob("*") if p.is_file()}
        summary=self.run_bulk()
        self.assertEqual(summary["status"],"complete")
        self.assertEqual((summary["completed_shards"],summary["episodes"],summary["selected_episodes"]),(2,4,3))
        self.assertFalse(summary["publication_ready"])
        self.assertFalse(summary["training_annotations_promoted"])
        audits={r["episode_index"]:r for r in self.audit_rows()}
        self.assertGreater(audits[0]["changed_frames"],0)
        self.assertEqual(audits[1]["changed_frames"],0)
        self.assertFalse(audits[1]["selected"])
        self.assertEqual(audits[3]["changed_frames"],0)
        rejected=json.loads(audits[3]["result_metadata_json"])
        self.assertEqual(rejected["coverage"]["right"]["reason"],"source_alignment_uncertain")
        for item in self.manifest['datasets'][0]['shards']:
            original=self.pq.read_table(self.source/item['path'])
            saved=self.pq.read_table(self.output/'staging/release-long-gripcmd/subset'/item['path'])
            self.assertEqual(saved.column_names[:len(original.column_names)],original.column_names)
            self.assertTrue(saved.select(original.column_names).equals(original,check_metadata=True))
            state=original['observation.state'].to_pylist()
            self.assertEqual(saved['annotation.gripper_target'].to_pylist(),[[r[9],r[19]] for r in state])
            self.assertEqual(saved['annotation.gripper_target_valid'].to_pylist(),[[False,False]]*len(state))
        self.assertEqual(before,{str(p):digest(p) for p in self.source.rglob('*') if p.is_file()})
        events=[r for p in (self.output/'audit/short__subset/events').rglob('*.parquet') for r in self.pq.read_table(p).to_pylist()]
        self.assertEqual(len(events),sum(r['candidates'] for r in audits.values()))
        self.assertTrue(all(json.loads(r['candidate_json'])['candidate_id']==json.loads(r['event_json'])['candidate_id'] for r in events))
        self.assertFalse(list(self.output.rglob('*.png'))+list(self.output.rglob('*.jpg')))

    def test_existing_annotation_columns_are_preserved_even_when_valid(self):
        for i in range(2):
            p=self.source/f'data/chunk-000/file-{i:03d}.parquet'
            table=self.pq.read_table(p)
            table=table.append_column('annotation.gripper_target',self.pa.array([[.8,.9]]*table.num_rows,type=self.pa.list_(self.pa.float32(),2)))
            table=table.append_column('annotation.gripper_target_valid',self.pa.array([[True,True]]*table.num_rows,type=self.pa.list_(self.pa.bool_(),2)))
            self.pq.write_table(table,p)
        self._write_manifest()
        self.run_bulk()
        for item in self.manifest['datasets'][0]['shards']:
            original=self.pq.read_table(self.source/item['path'])
            saved=self.pq.read_table(self.output/'staging/release-long-gripcmd/subset'/item['path'])
            self.assertTrue(saved.select(original.column_names).equals(original,check_metadata=True))

    def test_saved_manifest_replays_using_frozen_relative_sidecars(self):
        global_ledger=self.scope/'all_episodes.jsonl.gz'
        global_ledger.write_bytes((self.scope/'source.jsonl.gz').read_bytes())
        video_index=self.scope/'video_references.jsonl.gz'
        video_index.write_bytes(gzip.compress(b''))
        self.manifest.update(episode_ledger_path=global_ledger.name,episode_ledger_sha256=digest(global_ledger),
            video_reference_index_path=video_index.name,video_reference_index_sha256=digest(video_index))
        dump(self.manifest_path,self.manifest)
        self.run_bulk('first')
        first=self.output
        for name in ('source.jsonl.gz',global_ledger.name,video_index.name):
            self.assertEqual((first/name).read_bytes(),(self.scope/name).read_bytes())
            (self.scope/name).write_bytes(b'origin changed after frozen snapshot')
        second=self.workspace/'runs/replay'
        summary=profile_bulk.run_profile_dataset(first/'manifest.json',first/'task_profiles.json',second,workers=1)
        self.assertEqual(summary['status'],'complete')
        for item in self.manifest['datasets'][0]['shards']:
            relative=Path('staging/release-long-gripcmd/subset')/item['path']
            self.assertTrue(self.pq.read_table(first/relative).equals(self.pq.read_table(second/relative),check_metadata=True))

    def test_partial_scope_is_explicit_and_full_scope_cannot_omit_a_shard(self):
        self.manifest['datasets'][0]['shards']=self.manifest['datasets'][0]['shards'][:1]
        dump(self.manifest_path,self.manifest)
        with self.assertRaisesRegex(profile_bulk.ProfileBulkError,'incomplete'):
            self.run_bulk('missing')
        failed=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(failed['status'],'failed')
        self.assertIn('omitted',failed['failures'][0]['reason'])
        self.manifest['complete_source_scope']=False
        dump(self.manifest_path,self.manifest)
        result=self.run_bulk('pilot')
        self.assertEqual(result['episodes'],2)
        self.assertFalse(result['complete_source_scope'])

    def test_source_hash_change_fails_without_a_successful_shard(self):
        p=self.source/'data/chunk-000/file-000.parquet'
        p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk()
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['status'],'failed')
        self.assertEqual(summary['completed_shards'],0)
        self.assertIn('SHA256',summary['failures'][0]['reason'])
        self.assertFalse(list((self.output/'staging').rglob('*.parquet')))

    def test_scope_ledger_cannot_turn_an_unselected_task_into_selected(self):
        self.ledger[1]['selected']=True
        self._write_manifest()
        with self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk()
        self.assertEqual(json.loads((self.output/'summary.json').read_text())['status'],'failed')

    def test_source_ledger_row_binding_is_checked_against_actual_shard(self):
        self.ledger[0]['shard_row_start']=1
        self._write_manifest()
        with self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk()
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['completed_shards'],0)
        self.assertIn('shard bounds',summary['failures'][0]['reason'])

    def test_split_or_reordered_episode_rows_are_rejected_not_sorted(self):
        self.rows[0][0],self.rows[0][1]=self.rows[0][1],self.rows[0][0]
        self._write_sources();self._write_manifest()
        with self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk()
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertEqual(summary['completed_shards'],0)
        self.assertIn('frame/global-index',summary['failures'][0]['reason'])

    def test_generator_may_not_promote_rule_results_to_training_labels(self):
        def invalid(*args):
            result=generate_profile_targets(*args)
            result['valid'][0][1]=True
            return result
        with patch.object(profile_bulk,'generate_profile_targets',side_effect=invalid),self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk()
        summary=json.loads((self.output/'summary.json').read_text())
        self.assertIn('promote training',summary['failures'][0]['reason'])

    def test_invalid_side_never_changes_and_missing_timestamp_is_not_repaired(self):
        mask=json.loads((self.source/'dataset_configs/masks/mask.json').read_text())
        mask['element_mask']['observation.state'][9]=False
        dump(self.source/'dataset_configs/masks/mask.json',mask)
        for r in self.ledger:r['valid_sides']=['right']
        self._write_manifest();self.run_bulk()
        for p in (self.output/'staging').rglob('*.parquet'):
            saved=self.pq.read_table(p)
            self.assertTrue(all(not flags[0] for flags in saved['evaluation.gripper_rule_applied'].to_pylist()))
        self.rows[0][3]['timestamp']=None
        self._write_sources();self._write_manifest()
        with self.assertRaises(profile_bulk.ProfileBulkError):
            self.run_bulk('missing-time')

    def test_output_reuse_outside_runs_and_symlink_are_refused(self):
        self.run_bulk()
        with self.assertRaises(FileExistsError):self.run_bulk()
        with self.assertRaisesRegex(ValueError,'runs'):
            profile_bulk.run_profile_dataset(self.manifest_path,self.catalog_path,self.workspace/'outside',workers=1)
        link=self.workspace/'runs/link'
        link.symlink_to(self.source,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink'):
            profile_bulk.run_profile_dataset(self.manifest_path,self.catalog_path,link/'bad',workers=1)

    def test_two_worker_processing_keeps_bounded_complete_shards(self):
        summary=self.run_bulk(workers=2)
        self.assertEqual(summary['status'],'complete')
        self.assertEqual(summary['completed_shards'],2)
        self.assertEqual(len(self.audit_rows()),4)


if __name__=='__main__':
    unittest.main()
