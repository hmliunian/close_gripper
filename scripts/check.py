"""Offline checks only: local links, syntax, and temporary-fixture contracts."""
import ast
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
errors = []
for name in ("AGENTS.md", "PLAN.md", "BOUNDARIES.md", "README.md", "VALIDATION.md"):
    doc = root / name
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding="utf-8")):
        if target.startswith(("https://", "http://", "#")):
            continue
        local = target.split("#", 1)[0]
        if local and not (doc.parent / local).exists():
            errors.append(f"broken link: {name}: {target}")
for directory in ("close_gripper", "tests", "scripts", "research"):
    for source in (root / directory).rglob("*.py"):
        try:
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except SyntaxError:
            errors.append(f"syntax error: {source.relative_to(root)}")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print("PASS: documentation links and Python syntax", flush=True)
raise SystemExit(subprocess.call([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=root))
