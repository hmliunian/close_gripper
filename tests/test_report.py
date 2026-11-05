import json
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from close_gripper.core import detect_candidates, generate_targets
from close_gripper.demo import examples
from close_gripper.report import write_report


class ReportTests(unittest.TestCase):
    def video_record(self):
        episode, decisions = examples()[1]
        camera = "observation.images.cam_right_wrist_left"
        episode["cameras"][camera].update(from_timestamp=12.7, to_timestamp=16.0)
        candidates = detect_candidates(episode)
        return dict(episode=episode, candidates=candidates,
                    result=generate_targets(episode, candidates, decisions),
                    videos=[dict(side="right", camera=camera, url="/media/abc123.mp4",
                                 from_timestamp=12.7, to_timestamp=16.0)])

    def head_record(self):
        record = self.video_record()
        episode = record["episode"]
        episode.update(vision_camera_group="head", valid_sides=["right"], cameras={})
        record["videos"] = []
        for eye, offset in (("left", 12.7), ("right", 12.9)):
            camera = f"observation.images.cam_head_{eye}"
            episode["cameras"][camera] = dict(from_timestamp=offset, to_timestamp=16.2)
            record["videos"].append(dict(side="head", camera=camera, url=f"/media/head_{eye}.mp4",
                                         from_timestamp=offset, to_timestamp=16.2))
        record["result"]["visual_review"] = dict(camera_group="head", review_source="model", confirmed=False,
            sides=dict(right=dict(segments=[dict(start=0, end=1, state="free", note="夹爪尚未接触"),
                                            dict(start=2, end=4, state="held", note="物体随手移动")],
                                  reviewed_frame_indices=[0, 1, 2, 3, 4])),
            focus_intervals=[dict(side="right", start=0, end=4, shape_hint="descending_half")])
        return record

    def test_head_stereo_keeps_one_view_group_and_arm_labels_separate(self):
        record = self.head_record()
        before = copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "task_profiles.json").write_text('{"tasks": []}')
            page = write_report(tmp, [record]).read_text()
        value = json.loads(page.split("const records=", 1)[1].split(";let r, timer;", 1)[0])[0]
        self.assertEqual([v["side"] for v in value["videos"]], ["head", "head"])
        self.assertEqual([v["from_timestamp"] for v in value["videos"]], [12.7, 12.9])
        self.assertEqual(value["valid_sides"], ["right"])
        self.assertEqual(value["visual_review"], record["result"]["visual_review"])
        self.assertEqual(value["target"], record["result"]["target"])
        self.assertIn('id="taskConfigDownload" href="task_profiles.json" download', page)
        self.assertEqual(record, before)

    def test_head_camera_requires_explicit_group_and_cannot_impersonate_an_arm(self):
        for mutation in ("missing_group", "head_is_right_arm", "wrist_in_head_group", "wrong_eye_offset"):
            record = self.head_record()
            if mutation == "missing_group":
                record["episode"].pop("vision_camera_group")
            elif mutation == "head_is_right_arm":
                record["videos"][0]["side"] = "right"
            elif mutation == "wrist_in_head_group":
                record["videos"][0]["camera"] = "observation.images.cam_right_wrist_left"
            else:
                record["videos"][1]["from_timestamp"] = 12.7
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    write_report(tmp, [record])

    def test_main_view_segments_reject_false_provenance_or_wrong_frame_identity(self):
        for mutation in ("human_claim", "invalid_side", "overlap", "outside_episode", "unknown_state", "bad_focus"):
            record = self.head_record()
            review = record["result"]["visual_review"]
            if mutation == "human_claim":
                review["confirmed"] = True
            elif mutation == "invalid_side":
                review["sides"]["left"] = review["sides"].pop("right")
            elif mutation == "bad_focus":
                review["focus_intervals"][0]["side"] = "head"
            else:
                segment = review["sides"]["right"]["segments"][1]
                if mutation == "overlap":
                    segment["start"] = 1
                elif mutation == "outside_episode":
                    segment["end"] = len(record["episode"]["timestamps"])
                else:
                    segment["state"] = "maybe"
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    write_report(tmp, [record])

    def test_video_preserves_the_episode_offset_and_numeric_curves(self):
        record = self.video_record()
        snapshot = copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as tmp:
            page = write_report(tmp, [record]).read_text()
        value = json.loads(page.split("const records=", 1)[1].split(";let r, timer;", 1)[0])[0]
        self.assertEqual(value["videos"][0]["from_timestamp"], 12.7)
        self.assertEqual(value["videos"][0]["url"], "/media/abc123.mp4")
        self.assertEqual(value["target"], record["result"]["target"])
        self.assertEqual(value["original"], [[s[9], s[19]] for s in record["episode"]["state"]])
        self.assertEqual(record, snapshot)

    def test_video_from_the_wrong_arm_or_time_cannot_be_shown_as_corresponding(self):
        for changes in ({"side": "left"}, {"from_timestamp": 12.8},
                        {"url": "/media/../outside.mp4"}, {"url": "https://example.com/a.mp4"}):
            record = self.video_record()
            record["videos"][0].update(changes)
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    write_report(tmp, [record])

    def test_task_rule_curve_is_preserved_without_claiming_verified_targets(self):
        record = self.video_record()
        result = record["result"]
        original = [[row[9], row[19]] for row in record["episode"]["state"]]
        preview = copy.deepcopy(original)
        preview[5][1] = 0.
        mask = [[False, i == 5] for i in range(len(original))]
        result.update(workflow="task_profile_rule", target=original,
                      preview_target=preview, valid=[[False, False] for _ in original],
                      rule_applied_mask=mask, evaluation_only=True, grasp_semantics_verified=False,
                      preview_phase=[["unchanged", "hold"] for _ in original],
                      preview_source=[["none", "task_profile_rule"] for _ in original])
        record["episode"]["identity"]["source_metadata"] = {"source_tasks": ["拿杯子"]}
        snapshot = copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as tmp:
            page = write_report(tmp, [record]).read_text()
        value = json.loads(page.split("const records=", 1)[1].split(";let r, timer;", 1)[0])[0]
        self.assertEqual(value["task"], "拿杯子")
        self.assertEqual(value["preview"], preview)
        self.assertEqual(value["target"], original)
        self.assertEqual(value["rule_applied_mask"], mask)
        self.assertEqual(value["phase"], result["preview_phase"])
        self.assertTrue(value["evaluation_only"])
        self.assertFalse(value["grasp_semantics_verified"])
        self.assertIn("按任务开度规则生成，待审阅", page)
        self.assertNotIn("绿色为通过视觉证据与数值规则校验的目标", page)
        self.assertEqual(record, snapshot)

    def test_closing_comparison_is_available_under_both_policies(self):
        episode, decisions = examples()[1]
        candidates = detect_candidates(episode)
        for policy in ("fixed_timing", "original_speed"):
            result = generate_targets(episode, candidates, decisions, {"closing_policy": policy})
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                path = write_report(tmp, [dict(episode=episode, candidates=candidates, result=result)])
                script = path.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
                serialized = script.split("const records=", 1)[1].split(";let r, timer;", 1)[0]
                record = json.loads(serialized)[0]
                self.assertEqual(record["closing_policy"], policy)
                comp = record["closing_comparisons"][0]
                self.assertNotEqual(comp["fixed_timing"], comp["original_speed"])
                self.assertEqual(comp["fixed_timing"][comp["end"]], 0.)

    @unittest.skipUnless(shutil.which("node"), "Node required for generated JavaScript syntax check")
    def test_generated_review_script_is_valid_and_does_not_allow_html_injection(self):
        episode, decisions = examples()[1]
        # Dataset-provided text must remain text, even when embedded inside script.
        episode["identity"]["task"] = '</script><script>alert("unexpected")</script>'
        candidates = detect_candidates(episode)
        result = generate_targets(episode, candidates, decisions)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(tmp, [dict(episode=episode, candidates=candidates, result=result)])
            page = path.read_text()
            self.assertEqual(page.count("<script>"), 1)
            self.assertEqual(page.count("</script>"), 1)
            script = page.split("<script>", 1)[1].split("</script>", 1)[0]
            js = Path(tmp) / "report.js"
            js.write_text(script)
            completed = subprocess.run([shutil.which("node"), "--check", str(js)], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
