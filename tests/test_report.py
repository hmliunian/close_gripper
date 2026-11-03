import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from close_gripper.core import detect_candidates, generate_targets
from close_gripper.demo import examples
from close_gripper.report import write_report


class ReportTests(unittest.TestCase):
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
