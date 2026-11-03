"""Bounded, deterministic paired gripper survey. Source MCAPs are read-only."""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path('/gpfs/jiuquyun/datasets/ks3-raw/discover/PushCart-CollectSite/20261023-20261028')
OUT = Path(__file__).resolve().parent
TOPICS = {'/arm/right/gripper/joint_command': 'leader',
          '/arm/right/gripper/joint_state': 'follower'}
SEED = 'gripper-amplitude-20261102-v1'


def rank(path):
    return hashlib.sha256((SEED+str(path)).encode()).hexdigest()


def describe(values):
    a = np.asarray(values, dtype=float)
    return {k: float(v) for k,v in zip(('min','p05','p50','p95','max'), np.quantile(a,[0,.05,.5,.95,1]))}


def review_status(path):
    sidecar=path.parent/(path.parent.name+'.json')
    return json.loads(sidecar.read_text()).get('task.review.status','missing') if sidecar.exists() else 'missing'


def select_files(accepted_only=False):
    manifest = []
    for task in sorted(ROOT.iterdir()):
        files = sorted((task/'2026-10-26').glob('**/recording_*.mcap'))
        if not files or task.name == 'xxx':
            continue
        total_files=len(files)
        if accepted_only:
            files=[f for f in files if review_status(f)=='Accepted']
        groups = collections.defaultdict(list)
        for f in files:
            groups[f.parts[-3]].append(f)
        for g in groups.values():
            g.sort(key=rank)
        selected = []
        target = 24 if task.name == '叠衣服' else 32
        while len(selected) < min(target, len(files)):
            for cart in sorted(groups):
                if groups[cart] and len(selected) < target:
                    selected.append(groups[cart].pop(0))
        manifest.append(dict(task=task.name, total_files=total_files, eligible_files=len(files), selected=[
            dict(path=str(f), cart=f.parts[-3], source_id=f.parent.name,
                 review_status=review_status(f),
                 # Alternate assignment is fixed before reading any signal.
                 split='calibration' if i%2 == 0 else 'heldout',
                 size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
            for i,f in enumerate(selected)]))
    return manifest


def read_pair(path):
    data = {name: [] for name in TOPICS.values()}
    decoders = {}
    fixes = []
    with open(path,'rb') as stream:
        reader = make_reader(stream)
        for schema,ch,message in reader.iter_messages(topics=list(TOPICS)):
            if schema.id not in decoders:
                source = schema.data.decode()
                adjusted = source.replace('\nInterfaceValue[]','\ncontrol_msgs/InterfaceValue[]').rstrip().rstrip('=').rstrip()
                patched = dataclasses.replace(schema, name=schema.name.replace('/msg/','/'), data=adjusted.encode())
                decoder = DecoderFactory().decoder_for(ch.message_encoding, patched)
                if decoder is None:
                    raise ValueError('unsupported schema '+schema.name)
                decoders[schema.id] = decoder
                fixes.append(dict(original_name=schema.name, decoded_name=patched.name,
                                  schema_sha256=hashlib.sha256(schema.data).hexdigest()))
            msg = decoders[schema.id](message.data)
            kind = TOPICS[ch.topic]
            if kind == 'leader':
                iv = msg.interface_values[msg.joint_names.index('gripper')]
                position = iv.values[iv.interface_names.index('position')]
            else:
                position = msg.position[msg.name.index('gripper')]
            stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            data[kind].append((stamp,message.log_time,float(position)))
    arrays,quality = {},{}
    for kind, rows in data.items():
        if len(rows)<20:
            raise ValueError(f'{kind}: too few messages ({len(rows)})')
        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        logs = np.asarray([r[1] for r in rows], dtype=np.int64)
        pos = np.asarray([r[2] for r in rows], dtype=np.float64)
        if not np.isfinite(pos).all() or np.any(ts<=0):
            raise ValueError('nonfinite values or invalid header timestamp')
        quality[kind] = dict(messages=len(rows), nonincreasing_header=int((np.diff(ts)<=0).sum()),
                             log_minus_header_ms=describe((logs-ts)/1e6))
        order = np.argsort(ts, kind='stable')
        # Sort by header clock; retain last logged value at each duplicate stamp.
        ts,logs,pos = ts[order],logs[order],pos[order]
        keep = np.r_[np.diff(ts)!=0,True]
        arrays[kind+'_ns'] = ts[keep]
        arrays[kind+'_log_ns'] = logs[keep]
        arrays[kind+'_m'] = pos[keep]
        quality[kind]['duplicate_headers_removed'] = int((~keep).sum())
        quality[kind]['max_gap_s'] = float(np.max(np.diff(ts[keep]))/1e9)
    return arrays,quality,fixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--accepted-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'traces').mkdir(exist_ok=True)
    manifest = select_files(args.accepted_only)
    prefix='accepted_' if args.accepted_only else ''
    (OUT/(prefix+'manifest.json')).write_text(json.dumps(dict(seed=SEED,accepted_only=args.accepted_only,groups=manifest),ensure_ascii=False,indent=2)+'\n')
    if args.manifest_only:
        print('selected',sum(len(g['selected']) for g in manifest),'files',
              round(sum(f['size'] for g in manifest for f in g['selected'])/2**30,3),'GiB')
        return
    results=[]
    for group in manifest:
        for i,source in enumerate(group['selected']):
            record = dict(task=group['task'], **source)
            cache = OUT/'traces'/(source['source_id']+'.npz')
            meta = cache.with_suffix('.json')
            try:
                if cache.exists() and meta.exists():
                    old = json.loads(meta.read_text())
                    if old['size'] != source['size'] or old['mtime_ns'] != source['mtime_ns']:
                        raise ValueError('source changed; do not silently reuse cache')
                    arrays = dict(np.load(cache))
                    quality,fixes = old['quality'],old['schema_fixes']
                else:
                    arrays,quality,fixes = read_pair(source['path'])
                    np.savez_compressed(cache, **arrays)
                    meta.write_text(json.dumps(dict(**source,quality=quality,schema_fixes=fixes),ensure_ascii=False,indent=2)+'\n')
                record.update(status='ok', cache=str(cache.relative_to(OUT)), quality=quality)
                for kind in ('leader','follower'):
                    values = arrays[kind+'_m']*1000
                    stats = describe(values)
                    stats.update(q01=float(np.quantile(values,.01)),q99=float(np.quantile(values,.99)),
                                 first=float(values[0]),last=float(values[-1]),
                                 amplitude=float(values.max()-values.min()))
                    record[kind+'_mm'] = stats
            except Exception as exc:
                record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            results.append(record)
            if (i+1)%8==0 or record['status']=='error' or i+1==len(group['selected']):
                print(group['task'],i+1,'/',len(group['selected']),record['status'],flush=True)
        (OUT/(prefix+'episodes.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary=[]
    for group in manifest:
        rows = [r for r in results if r['task']==group['task'] and r['status']=='ok']
        row = dict(task=group['task'], total_files=group['total_files'], sampled=len(group['selected']), usable=len(rows))
        if rows:
            for kind in ('leader','follower'):
                row[kind] = {k:describe([r[kind+'_mm'][k] for r in rows]) for k in ('min','max','q01','q99','amplitude')}
                row[kind]['episodes_min_le_1mm'] = sum(r[kind+'_mm']['min']<=1 for r in rows)
                row[kind]['episodes_max_ge_90mm'] = sum(r[kind+'_mm']['max']>=90 for r in rows)
        summary.append(row)
    (OUT/(prefix+'ranges.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/(prefix+'episode_ranges.csv')).open('w') as f:
        fields=['task','cart','source_id','split','status','leader_min_mm','leader_max_mm','leader_q01_mm','leader_q99_mm','follower_min_mm','follower_max_mm','follower_q01_mm','follower_q99_mm']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            row={k:r[k] for k in fields[:5]}
            if r['status']=='ok':
                for kind in ('leader','follower'):
                    for stat in ('min','max','q01','q99'):
                        row[f'{kind}_{stat}_mm']=r[kind+'_mm'][stat]
            writer.writerow(row)
    print('DONE',len(results),'episodes;',sum(r['status']=='ok' for r in results),'usable',flush=True)


if __name__=='__main__':
    main()
