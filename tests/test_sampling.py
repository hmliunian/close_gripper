"""Real temporary Parquet fixtures for deterministic, source-aware sample manifests."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from close_gripper.sampling import build_manifest

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


@unittest.skipIf(pa is None, "optional pyarrow is not installed")
class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, name, rows):
        root = self.root / name
        directory = root / "meta/episodes/chunk-000"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / "file-000.parquet")
        return root

    def test_round_robin_covers_tasks_and_subsets(self):
        first = self.dataset("subset-a", [dict(episode_index=i, length=30, tasks=["fold" if i < 4 else "cup"]) for i in range(8)])
        second = self.dataset("subset-b", [dict(episode_index=i, length=30, tasks=["drawer"]) for i in range(4)])
        rows = build_manifest([first, second], limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual({tuple(r["task"]) for r in rows}, {("fold",), ("cup",), ("drawer",)})
        self.assertEqual({r["dataset"] for r in rows}, {str(first), str(second)})

    def test_limit_validation(self):
        root = self.dataset("sample", [dict(episode_index=0, length=30, tasks=["cup"])])
        for limit in (0, -1, 1.2, True, "3", None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                build_manifest([root], limit)
        self.assertEqual(len(build_manifest([root], 100)), 1)

    def test_deterministic_input_order_and_duplicate_roots(self):
        roots = [self.dataset(f"subset-{k}", [dict(episode_index=i, length=30, tasks=["fold" if i % 2 else "cup"]) for i in range(10)]) for k in range(2)]
        expected = build_manifest(roots, limit=7)
        self.assertEqual(expected, build_manifest(list(reversed(roots)), limit=7))
        self.assertEqual(expected, build_manifest(roots + roots, limit=7))
        self.assertEqual(expected, build_manifest(roots, limit=7))

    def test_unknown_original_identity_remains_unassigned(self):
        root = self.dataset("sample", [dict(episode_index=i, length=30, tasks=["cup"]) for i in range(3)])
        rows = build_manifest([root])
        self.assertTrue(all(row["split"] == "unassigned" and row["source_identity"] == {} for row in rows))
        self.assertTrue(all(row["source_group"] == "unresolved_source_identity" for row in rows))
        self.assertTrue(all(row["semantic_category"] == "needs_visual_review" and row["review_status"] == "not_reviewed" and row["semantic_coverage_verified"] is False for row in rows))

    def test_null_and_empty_identity_fields_do_not_create_a_false_split(self):
        root = self.dataset("sample", [dict(episode_index=i, length=30, tasks=["cup"], source_episode_id=source) for i, source in enumerate((None, "", " "))])
        self.assertTrue(all(row["split"] == "unassigned" for row in build_manifest([root])))

    def test_crops_of_same_original_stay_in_one_split(self):
        first = self.dataset("crop-a", [dict(episode_index=1, length=30, tasks=["cup"], source_episode_id="shared-source")])
        second = self.dataset("crop-b", [dict(episode_index=91, length=20, tasks=["cup"], source_episode_id="shared-source")])
        rows = build_manifest([first, second])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_group"], rows[1]["source_group"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertIn(rows[0]["split"], ("calibration", "holdout"))

    def test_manifest_reads_do_not_modify_parquet(self):
        root = self.dataset("sample", [dict(episode_index=i, length=30, tasks=["cup"], source_episode_id=f"source-{i}") for i in range(6)])
        file = next(root.glob("meta/episodes/chunk-*/*.parquet"))
        original = hashlib.sha256(file.read_bytes()).hexdigest()
        rows = build_manifest([root], 4)
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), original)
        self.assertTrue(all(row["source_metadata_file"] == str(file) for row in rows))


if __name__ == "__main__":
    unittest.main()
