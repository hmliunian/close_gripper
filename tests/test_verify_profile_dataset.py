"""Independent-oracle regressions and corrupt saved-output rejection."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_tool = load("independent_profile_dataset_audit", ROOT / "scripts/verify_profile_dataset.py")


class WindowOracleTests(unittest.TestCase):
    def test_matching_oracle_keeps_known_free_gap_and_limits_padding(self):
        profile = dict(automatic_use=True,
            closed_bands=[dict(id="a", lower=.10, upper=.12),dict(id="b", lower=.20, upper=.22)],
            free_bands=[dict(lower=.125, upper=.195)], open_bands=[], interval_vision_bands=[])
        bands = audit_tool.matching_bands_oracle(profile, .01)
        self.assertEqual(len(bands), 2)
        self.assertLess(bands[0]["upper"], .125)
        self.assertGreater(bands[1]["lower"], .195)
        self.assertAlmostEqual(bands[0]["lower"], .09)
        profile["free_bands"] = []
        merged = audit_tool.matching_bands_oracle(profile, .01)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["basis_closed_band_ids"], ["a", "b"])

    def test_partial_diagnostic_selection_is_disclosed_not_mislabeled_as_full(self):
        partial = dict(complete_source_scope=False, sampling="first shards plus known boundary diagnostics",
                       selection_uses_band_matching=False)
        self.assertTrue(audit_tool.selection_contract(partial))
        self.assertFalse(audit_tool.selection_contract(dict(partial, complete_source_scope=True)))
        self.assertFalse(audit_tool.selection_contract(dict(partial, sampling="")))
        full = dict(complete_source_scope=True, sampling=False, selection_uses_band_matching=False)
        self.assertTrue(audit_tool.selection_contract(full))
        self.assertFalse(audit_tool.selection_contract(dict(full, sampling=True)))
        self.assertFalse(audit_tool.selection_contract(dict(full, selection_uses_band_matching=True)))

    def test_sparse_queries_equal_exhaustive_definition_with_gaps_and_overlaps(self):
        rng = random.Random(117)
        for _ in range(40):
            n = rng.randrange(1, 35)
            values = [rng.choice([.19, .20, .21, .225, .5]) for _ in range(n)]
            times = [0.]
            for _ in range(n - 1):
                times.append(times[-1] + rng.choice([.1, .1, .3]))
            band = dict(lower=.18, upper=.23)
            eligible = []
            for a in range(n):
                for b in range(a, n):
                    segment = values[a:b + 1]
                    if (all(.18 <= v <= .23 for v in segment)
                            and max(segment) - min(segment) <= .02 + 1e-9
                            and all(times[i] - times[i - 1] <= .15 + 1e-9 for i in range(a + 1, b + 1))):
                        eligible.append((a, b))
            maximal = [x for x in eligible if not any(y != x and y[0] <= x[0] <= x[1] <= y[1] for y in eligible)]
            self.assertEqual(audit_tool.maximal_windows(times, values, band, .15, .02), maximal)

    def test_neutral_drift_does_not_move_significant_motion_start(self):
        config = dict(motion_noise=.003, motion_min_delta=.04, motion_gap_s=.2)
        result = audit_tool.motions([.5, .501, .502, .4, .3, .301, .302, .303, .6],
                                   [i / 10 for i in range(9)], 0, 8, config)
        self.assertEqual(result, [dict(start=2, end=4, direction=-1), dict(start=7, end=8, direction=1)])

    def test_short_and_gapped_hold_cannot_qualify_by_count(self):
        profile = dict(closed_bands=[dict(id="held", lower=.19, upper=.21)], interval_vision_bands=[])
        catalog = dict(minimum_hold_s=.6, maximum_sample_gap_s=.15, maximum_plateau_range=.02)
        short = audit_tool.candidates_oracle([i / 10 for i in range(6)], [.2] * 6, profile, catalog)
        self.assertFalse(short[0]["matched"])
        gap = audit_tool.candidates_oracle([0, .1, .2, .5, .6, .7, .8], [.2] * 7, profile, catalog)
        self.assertEqual(len(gap), 2)
        self.assertTrue(all(not c["matched"] for c in gap))


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow unavailable")
class SavedDerivativeAuditTests(unittest.TestCase):
    def setUp(self):
        # The fixture writer supplies actual saved Parquet, not audit answers.
        # All acceptance decisions still come from the independent oracle.
        module = load("profile_bulk_audit_fixture", ROOT / "tests/test_profile_bulk.py")
        self.fixture = module.BulkDatasetContracts(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.catalog["evidence_noise"] = .003
        for task in f.catalog["tasks"]:
            for profile in task["sides"].values():
                for band in profile["closed_bands"]:
                    band["source_ids"] = ["reference-a", "reference-b"]
                profile["semantic_evidence"] = [dict(source_id=source, numeric_calibration_eligible=True,
                    held_intervals=[dict(lower=.18, upper=.22, qualifies_hold=True)])
                    for source in ("reference-a", "reference-b")]
        module.dump(f.catalog_path, f.catalog)
        for ledger, meta in zip(f.ledger, f.metadata):
            shard = meta["data/file_index"]
            start = next(i for i, row in enumerate(f.rows[shard]) if row["episode_index"] == meta["episode_index"])
            ledger.update(shard_row_start=start, shard_row_end_exclusive=start + meta["length"],
                          dataset_from_index=meta["dataset_from_index"], dataset_to_index=meta["dataset_to_index"])
        f._write_manifest()
        f.manifest.update(sampling=False, selection_uses_band_matching=False)
        f.manifest["datasets"][0]["valid_sides"] = ["left", "right"]
        for item in f.manifest["datasets"][0]["shards"]:
            stat = (f.source / item["path"]).stat()
            item.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        module.dump(f.manifest_path, f.manifest)
        self.workspace_patch = patch.object(audit_tool, "WORKSPACE", f.workspace)
        self.workspace_patch.start(); self.addCleanup(self.workspace_patch.stop)
        with contextlib.redirect_stdout(io.StringIO()):
            f.run_bulk()

    def verify(self, suffix="audit", workers=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit_tool.audit_bulk(self.fixture.output, self.fixture.output / "manifest.json",
                                         self.fixture.workspace / "runs" / suffix, workers=workers)

    def rewrite_shards(self, rows):
        (self.fixture.output / "shards.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def alter_output(self, mutate):
        rows = list(audit_tool.json_lines(self.fixture.output / "shards.jsonl"))
        record = rows[0]
        path = Path(record["output_path"])
        table = self.fixture.pq.read_table(path)
        table = mutate(table)
        self.fixture.pq.write_table(table, path)
        record["output_sha256"] = audit_tool.digest(path)
        self.rewrite_shards(rows)

    def test_complete_real_parquet_fixture_passes_without_generator_replay(self):
        result = self.verify()
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(result["verified_episodes"], 4)
        self.assertEqual(result["verified_source_shards"], 2)

    def test_expanded_catalog_runs_through_bulk_projection_and_independent_oracle(self):
        from close_gripper.profile_bands import build_matching_catalog
        f = self.fixture
        f.catalog.update(calibration_contract="all_modes_semantic_review_v2", minimum_positive_sources=2)
        for task in f.catalog["tasks"]:
            for profile in task["sides"].values():
                profile.update(calibration_contract="all_modes_semantic_review_v2", minimum_positive_sources=2,
                               evidence_noise=.003, free_bands=copy.deepcopy(profile["open_bands"]),
                               interval_vision_bands=[], release_rules=[])
                for band in profile["closed_bands"]:
                    band.update(reviewed_source_count=2, open_band_ids=[profile["open_bands"][0]["id"]])
                for band in profile["open_bands"]:
                    band["source_ids"] = ["reference-a"]
        f.catalog = build_matching_catalog(f.catalog, margin=.01, version="expanded_fixture")
        f.catalog_path.write_text(json.dumps(f.catalog,ensure_ascii=False))
        f.manifest["catalog_sha256"] = audit_tool.digest(f.catalog_path)
        f.manifest_path.write_text(json.dumps(f.manifest,ensure_ascii=False))
        with contextlib.redirect_stdout(io.StringIO()):
            f.run_bulk("expanded")
        result = self.verify("expanded_audit")
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertGreater(result["checks"]["runtime_generalized_matching_policy"], 0)
        self.assertGreater(result["checks"]["independent_merged_padded_ranges"], 0)
        corrupted = copy.deepcopy(f.catalog)
        profile = corrupted["tasks"][0]["sides"]["right"]
        profile["matching_bands"][0]["upper"] = .95
        audit = audit_tool.Audit()
        audit_tool.verify_catalog(corrupted,audit)
        self.assertIn("matching_range_excludes_free_and_conflicts", {e["check"] for e in audit.failures})

    def test_parallel_shards_have_identical_checks_and_totals(self):
        sequential = self.verify("sequential")
        parallel = self.verify("parallel", workers=2)
        for key in ("status", "checks", "failure_count", "totals", "verified_episodes", "verified_source_shards"):
            self.assertEqual(parallel[key], sequential[key], key)

    def test_missing_shard_fails_even_if_the_remaining_digest_is_valid(self):
        rows = list(audit_tool.json_lines(self.fixture.output / "shards.jsonl"))
        self.rewrite_shards(rows[:1])
        result = self.verify()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("all_source_shards_recorded_once", [f["check"] for f in result["failures"]])

    def test_original_column_change_is_detected_with_updated_output_hash(self):
        def mutate(table):
            values = table["extra"].to_pylist(); values[0] = "corrupt original"
            return table.set_column(table.column_names.index("extra"), "extra", self.fixture.pa.array(values))
        self.alter_output(mutate)
        result = self.verify()
        self.assertIn("every_original_column_identical", [f["check"] for f in result["failures"]])

    def test_out_of_scope_curve_edit_is_detected_even_without_mask(self):
        def mutate(table):
            values = table[audit_tool.TARGET].to_pylist()
            index = table["episode_index"].to_pylist().index(1)
            values[index][1] = 0.
            field = table.schema.field(audit_tool.TARGET)
            return table.set_column(table.column_names.index(audit_tool.TARGET), field, self.fixture.pa.array(values, type=field.type))
        self.alter_output(mutate)
        result = self.verify()
        self.assertIn("independent_complete_curve", [f["check"] for f in result["failures"]])

    def test_qualified_hold_cannot_remain_nonzero_even_with_updated_artifact_hash(self):
        def mutate(table):
            values = table[audit_tool.TARGET].to_pylist()
            values[5][1] = .2
            field = table.schema.field(audit_tool.TARGET)
            return table.set_column(table.column_names.index(audit_tool.TARGET), field,
                                    self.fixture.pa.array(values, type=field.type))
        self.alter_output(mutate)
        result = self.verify()
        self.assertIn("every_qualified_hold_zero", [f["check"] for f in result["failures"]])

    def test_frame_shift_and_training_promotion_are_both_detected(self):
        def mutate(table):
            curve = table[audit_tool.TARGET].to_pylist()
            curve = curve[1:] + curve[:1]
            field = table.schema.field(audit_tool.TARGET)
            table = table.set_column(table.column_names.index(audit_tool.TARGET), field, self.fixture.pa.array(curve, type=field.type))
            valid = table[audit_tool.VALID].to_pylist(); valid[0][1] = True
            field = table.schema.field(audit_tool.VALID)
            return table.set_column(table.column_names.index(audit_tool.VALID), field, self.fixture.pa.array(valid, type=field.type))
        self.alter_output(mutate)
        result = self.verify()
        failed = {f["check"] for f in result["failures"]}
        self.assertTrue({"independent_complete_curve", "annotation_valid_false"} <= failed)


if __name__ == "__main__":
    unittest.main()
