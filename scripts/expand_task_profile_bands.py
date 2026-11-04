#!/usr/bin/env python3
"""Save merged/padded matching bands while preserving the original calibration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from close_gripper.pipeline import create_run, dump
from close_gripper.profile_bands import build_matching_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=.01)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.catalog.read_bytes()
    old = json.loads(raw)
    new = build_matching_catalog(old, margin=args.margin, version=args.version)
    directory = create_run(args.output, [args.catalog])
    shutil.copyfile(args.catalog, directory / "calibration_catalog.json")
    dump(directory / "task_profiles.json", new)
    rows = []
    for task in new["tasks"]:
        for side, profile in task["sides"].items():
            rows.append(dict(task=task["task"], side=side, enabled=profile["automatic_use"],
                calibrated_bands=len(profile["closed_bands"]), matching_bands=len(profile["matching_bands"]),
                old_ranges=json.dumps([[b["lower"], b["upper"]] for b in profile["closed_bands"]]),
                new_ranges=json.dumps([[b["lower"], b["upper"]] for b in profile["matching_bands"]])))
    with (directory / "band_changes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status="complete", version=new["version"], matching_policy=new["matching_policy"],
        input_catalog_sha256=hashlib.sha256(raw).hexdigest(),
        output_catalog_sha256=hashlib.sha256((directory / "task_profiles.json").read_bytes()).hexdigest(),
        calibrated_bands=sum(r["calibrated_bands"] for r in rows),
        matching_bands=sum(r["matching_bands"] for r in rows),
        enabled_sides=sum(r["enabled"] for r in rows),
        enabled_tasks=sum(any(p["automatic_use"] for p in t["sides"].values()) for t in new["tasks"]),
        calibration_preserved=True, source_selection_changed=False, new_visual_judgments=0,
        generalized_ranges_are_not_new_visual_evidence=True)
    dump(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
