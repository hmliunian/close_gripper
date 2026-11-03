"""Deterministic task/subset stratification without inventing semantic labels."""
from collections import defaultdict, deque
import hashlib
from pathlib import Path


def build_manifest(roots, limit=48):
    import pyarrow.parquet as pq
    if type(limit) is not int or limit <= 0:
        raise ValueError("sample limit must be positive")
    groups = defaultdict(list)
    for root in sorted({str(Path(r).resolve(strict=True)) for r in roots}):
        files = sorted((Path(root) / "meta/episodes").glob("chunk-*/*.parquet"))
        if not files:
            raise ValueError("episode metadata missing")
        for file in files:
            for row in pq.read_table(file).to_pylist():
                task = tuple(row.get("tasks", []))
                source = {k:v for k,v in row.items() if k in ("source_episode_id", "source_episode_index", "source_path", "original_episode_id") and v is not None and (not isinstance(v, str) or v.strip())}
                # Without original episode mapping, a random holdout could leak crops.
                group = str(sorted(source.items())) if source else "unresolved_source_identity"
                split = ("holdout" if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0 else "calibration") if source else "unassigned"
                groups[(root,task)].append(dict(dataset=root, episode_index=row["episode_index"],
                    task=list(task), frames=row["length"], source_identity=source,
                    source_group=group, split=split, semantic_category="needs_visual_review",
                    review_status="not_reviewed", source_metadata_file=str(file),
                    semantic_coverage_verified=False))
    queues = []
    for group in sorted(groups):
        # Hash order avoids always selecting the first temporal block.
        items = groups[group]
        items.sort(key=lambda x: hashlib.sha256(f'{x["dataset"]}:{x["episode_index"]}'.encode()).hexdigest())
        queues.append(deque(items))
    chosen = []
    while queues and len(chosen) < limit:
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.popleft())
        queues = [q for q in queues if q]
    return chosen
