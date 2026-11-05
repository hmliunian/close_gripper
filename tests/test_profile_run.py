"""Evaluation artifacts must never become training labels by accident."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper import pipeline, profile_run
from close_gripper.task_profiles import SCHEMA, UNITS


def episode():
    states = [[0.0] * 20 for _ in range(9)]
    for i, row in enumerate(states):
        row[0], row[9], row[19] = 10 + i, .7, .2
    return dict(episode_id="sample/evaluation", timestamps=[i / 10 for i in range(9)],
        state=states, valid_sides=["right"], cameras={},
        identity=dict(source_metadata=dict(source_tasks=["测试拿杯子"])))


def catalog():
    return dict(schema=SCHEMA, units=UNITS, version="test-profiles-v1", minimum_hold_s=.6,
        maximum_sample_gap_s=.15, maximum_plateau_range=.02,
        tasks=[dict(task="测试拿杯子", aliases=[], sides={"right":dict(status="visual_reviewed",
            automatic_use=True, provenance=dict(review_source="model"), counts={},
            closed_bands=[dict(id="hold",lower=.18,center=.2,upper=.22)],
            open_bands=[dict(id="open",lower=.95,center=1.,upper=1.)])})])


def generated(ep, profiles, config):
    original = [[r[9], r[19]] for r in ep["state"]]
    preview = [[r[9], 0.] for r in ep["state"]]
    return dict(workflow="task_profile_rule", rule_version="test-profile-generator-v1", target=original, preview_target=preview,
        valid=[[False,False] for _ in original], rule_applied_mask=[[False,True] for _ in original],
        phase=[["unchanged","unchanged"] for _ in original], source=[["",""] for _ in original],
        preview_phase=[["unchanged","hold"] for _ in original],
        preview_source=[["","task_profile_rule"] for _ in original],
        events=[dict(status="applied",reason="task_profile_rule",side="right",start=0,end=8)],
        candidates=[], coverage={},side_status={})


class ProfileRunContracts(unittest.TestCase):
    def test_evaluation_curve_and_training_columns_remain_separate(self):
        ep, profiles = episode(), catalog()
        before = copy.deepcopy(ep)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            with patch.object(pipeline, "WORKSPACE", workspace), patch.object(
                    profile_run, "generate_profile_targets", side_effect=generated):
                output = workspace / "runs/evaluation"
                summary = profile_run.run_profile_clean([ep], output, profiles, workers=1)
                self.assertEqual(ep, before)
                self.assertEqual(summary["changed_frames"], 9)
                self.assertTrue(summary["training_target_unchanged"])
                child = output / summary["episodes"][0]["directory"]
                saved_ep = json.loads((child / "episode.json").read_text())
                targets = json.loads((child / "targets.json").read_text())
                self.assertEqual(saved_ep, before)
                self.assertEqual(targets["annotation.gripper_target"], [[.7,.2]] * 9)
                self.assertEqual(targets["annotation.gripper_target_valid"], [[False,False]] * 9)
                self.assertEqual(targets["evaluation.gripper_target"], [[.7,0.]] * 9)
                self.assertEqual(targets["evaluation.gripper_rule_applied"], [[False,True]] * 9)
                self.assertEqual(targets["method_version"], "test-profile-generator-v1")
                self.assertTrue(targets["evaluation_only"])
                self.assertFalse(targets["grasp_semantics_verified"])
                self.assertEqual(summary["actual_requests"], 0)
                self.assertTrue((output / "review.html").is_file())
                with self.assertRaises(FileExistsError):
                    profile_run.run_profile_clean([ep], output, profiles, workers=1)

    def test_runner_rejects_algorithm_promoting_a_rule_into_training_validity(self):
        ep = episode()
        bad = generated(ep,catalog(),{})
        bad["valid"][0][1] = True
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            with patch.object(pipeline,"WORKSPACE",workspace), patch.object(
                    profile_run,"generate_profile_targets",return_value=bad):
                with self.assertRaisesRegex(ValueError,"promote training"):
                    profile_run.run_profile_clean([ep],workspace / "runs/bad",catalog(),workers=1)


if __name__ == "__main__":
    unittest.main()
