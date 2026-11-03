"""Read-only feasibility audit. Writes evidence here, never to source datasets.

Run with the existing orion-data-pipeline virtualenv (numpy, pyarrow, mcap_ros2).
This is a diagnostic, not a grasp classifier or a production relabeling tool.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


OUT = Path(__file__).resolve().parent
BASE = Path('/gpfs/jiuquyun/datasets/PRETRAIN_DATA')
FIGS = Path('/gpfs/jiuquyun/projects/yiyang/pushcart-gripper-figs-20261102')
RELEASES = [
    'PushCart-CollectSite-20261029-priority-en-flu-tool-v6-idletrim1-accepted-durclean',
    'PushCart-CollectSite-20261029-remaining-en-flu-tool-v6-idletrim1-accepted-durclean',
    'PushCart-CollectSite-scope139-20261026-b39c35c5-remaining-en-flu-tool-v4-idletrim3-drawer-accepted-durclean',
]
TOPICS = {
    '/arm/right/gripper/joint_command': 'cmd',
    '/arm/right/gripper/joint_state': 'state',
}


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), min=float(a.min()), p10=float(np.quantile(a, .1)),
                p50=float(np.median(a)), p90=float(np.quantile(a, .9)), max=float(a.max()))


def first_below(v, threshold):
    ix = np.flatnonzero(v < threshold)
    return int(ix[0]) if len(ix) else None


def cached_audit():
    result = {}
    for filename in ('ros_grip.json', 'ros_fold.json'):
        records = json.loads((FIGS / filename).read_text())
        rows, closing, opening = [], [], []
        for e in records:
            c, s = np.asarray(e['cmd']), np.asarray(e['state'])
            g, y = c[:, 1] / .095, s[:, 1] / .095
            row = dict(task=e['task'], path=e['path'],
                       duration_s=float(c[-1, 0]-c[0, 0]),
                       command_hz=1/float(np.median(np.diff(c[:, 0]))),
                       follower_hz=1/float(np.median(np.diff(s[:, 0]))),
                       command_min=float(g.min()), command_max=float(g.max()),
                       follower_min=float(y.min()), follower_max=float(y.max()))
            if filename == 'ros_grip.json':
                # Precisely reproduce plot.py's per-episode-max threshold convention,
                # while excluding missing crossings instead of treating argmax as 0.
                scale = c[:, 1].max()
                i, j = first_below(c[:, 1]/scale, .9), first_below(s[:, 1]/scale, .9)
                k = first_below(c[:, 1]/scale, .1)
                row['plot_convention_close_90_10_ms'] = float((c[k, 0]-c[i, 0])*1000) if i is not None and k is not None and k >= i else None
                row['plot_convention_lag_ms'] = float((s[j, 0]-c[i, 0])*1000) if i is not None and j is not None else None
            else:
                # Only complete absolute 0.9<->0.1 transitions. Partial openings
                # are excluded; these durations are NOT representative of all cycles.
                last_high = last_low = None
                used_high = used_low = -1
                for i, val in enumerate(g):
                    if val >= .9:
                        if last_low is not None and last_low > used_low:
                            crossing = last_low+1
                            if crossing <= i:
                                opening.append(float((c[i, 0]-c[crossing, 0])*1000))
                            used_low = i
                        last_high = i
                    if val <= .1:
                        if last_high is not None and last_high > used_high:
                            crossing = last_high+1
                            if crossing <= i:
                                closing.append(float((c[i, 0]-c[crossing, 0])*1000))
                            used_high = i
                        last_low = i
            rows.append(row)
        group = dict(episodes=len(rows), rows=rows)
        if filename == 'ros_grip.json':
            group['close_90_10_ms'] = summary([r['plot_convention_close_90_10_ms'] for r in rows if r['plot_convention_close_90_10_ms'] is not None])
            group['same_threshold_log_time_lag_ms'] = summary([r['plot_convention_lag_ms'] for r in rows if r['plot_convention_lag_ms'] is not None])
            group['follower_never_below_half'] = sum(r['follower_min'] > .5 for r in rows)
            group['command_never_below_tenth'] = sum(r['command_min'] > .1 for r in rows)
            group['command_never_above_half'] = sum(r['command_max'] < .5 for r in rows)
        else:
            group['complete_absolute_close_90_10_ms'] = summary(closing)
            group['complete_absolute_open_10_90_ms'] = summary(opening)
        result[filename] = group
    return result


def raw_timing_audit(records):
    rows = []
    for e in records:
        decoders, observations = {}, {'cmd': [], 'state': []}
        schema_changes = []
        with open(e['path'], 'rb') as stream:
            for schema, channel, message in make_reader(stream).iter_messages(topics=list(TOPICS)):
                if schema.id not in decoders:
                    # Compatibility fixes from existing extract.py, plus public
                    # DecoderFactory name normalization. Only in-memory copies change.
                    text = schema.data.decode().replace('\nInterfaceValue[]', '\ncontrol_msgs/InterfaceValue[]').rstrip().rstrip('=').rstrip()
                    adjusted = dataclasses.replace(schema, name=schema.name.replace('/msg/', '/'), data=text.encode())
                    decoders[schema.id] = DecoderFactory().decoder_for(channel.message_encoding, adjusted)
                    schema_changes.append(dict(original_name=schema.name, decoded_name=adjusted.name,
                                               original_schema_sha256=hashlib.sha256(schema.data).hexdigest(),
                                               repaired_schema_sha256=hashlib.sha256(adjusted.data).hexdigest()))
                msg = decoders[schema.id](message.data)
                kind = TOPICS[channel.topic]
                if kind == 'cmd':
                    i = msg.joint_names.index('gripper')
                    iv = msg.interface_values[i]
                    value = iv.values[iv.interface_names.index('position')]
                else:
                    value = msg.position[msg.name.index('gripper')]
                stamp = msg.header.stamp.sec*10**9 + msg.header.stamp.nanosec
                observations[kind].append((message.log_time, stamp, value))
        stats = {}
        for kind, data in observations.items():
            log = np.asarray([r[0] for r in data], dtype=np.int64)
            header = np.asarray([r[1] for r in data], dtype=np.int64)
            stats[kind] = dict(samples=len(data), log_minus_header_ms=summary((log-header)/1e6),
                               nonincreasing_header_count=int(np.sum(np.diff(header) <= 0)))
        c, s = observations['cmd'], observations['state']
        scale = max(r[2] for r in c)
        i = first_below(np.asarray([r[2] for r in c])/scale, .9)
        j = first_below(np.asarray([r[2] for r in s])/scale, .9)
        row = dict(task=e['task'], path=e['path'], streams=stats, schema_changes=schema_changes)
        if i is not None and j is not None:
            row['threshold_lag_log_ms'] = (s[j][0]-c[i][0])/1e6
            row['threshold_lag_header_ms'] = (s[j][1]-c[i][1])/1e6
        rows.append(row)
        print('raw timing:', e['task'], Path(e['path']).parent.name[:8], flush=True)
    return dict(episodes=len(rows), rows=rows,
                threshold_lag_log_ms=summary([r['threshold_lag_log_ms'] for r in rows if 'threshold_lag_log_ms' in r]),
                threshold_lag_header_ms=summary([r['threshold_lag_header_ms'] for r in rows if 'threshold_lag_header_ms' in r]))


def release_audit():
    releases = []
    for name in RELEASES:
        release = dict(name=name, roots=[])
        for info_path in sorted((BASE/name).glob('*/meta/info.json')):
            root = info_path.parent.parent
            info = json.loads(info_path.read_text())
            paths = sorted(root.glob('data/**/*.parquet'))
            selected = sorted({0, len(paths)//2, len(paths)-1})
            tasks = {r['task_index']: r.get('source_text', r['task']) for r in pq.read_table(root/'meta/tasks.parquet').to_pylist()}
            episodes = []
            for file_index in selected:
                table = pq.read_table(paths[file_index], columns=['observation.state', 'episode_index', 'task_index', 'timestamp'])
                states = table['observation.state'].combine_chunks().values.to_numpy().reshape(-1, 20)
                episode = table['episode_index'].to_numpy()
                task = table['task_index'].to_numpy()
                time = table['timestamp'].to_numpy()
                bounds = np.r_[0, np.flatnonzero(np.diff(episode) != 0)+1, len(episode)]
                for a, b in zip(bounds[:-1], bounds[1:]):
                    g = states[a:b, 19]
                    episodes.append(dict(episode_index=int(episode[a]), task=tasks[int(task[a])],
                                         frames=int(b-a), minimum=float(g.min()), maximum=float(g.max()),
                                         range=float(np.ptp(g)), start=float(g[0]), end=float(g[-1]),
                                         median_dt_s=float(np.median(np.diff(time[a:b]))) if b-a > 1 else None,
                                         max_frame_delta=float(np.max(np.abs(np.diff(g)))) if b-a > 1 else 0.))
            mask = json.loads((root/'dataset_configs/masks/mask.json').read_text())
            release['roots'].append(dict(name=root.name, info={k: info[k] for k in ('codebase_version', 'fps', 'total_episodes', 'total_frames', 'total_tasks', 'total_data_files')},
                info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
                parquet_bytes=sum(p.stat().st_size for p in paths),
                source_features=list(info['features']), available_gripper_indices=[i for i in (9,19) if mask['element_mask']['observation.state'][i]],
                sampled_files=[str(paths[i].relative_to(root)) for i in selected],
                sample=dict(episodes=len(episodes), frames=sum(e['frames'] for e in episodes),
                            range_le_005=sum(e['range'] <= .05 for e in episodes),
                            starts_below_half=sum(e['start'] < .5 for e in episodes),
                            never_below_half=sum(e['minimum'] > .5 for e in episodes),
                            range_distribution=summary([e['range'] for e in episodes]), rows=episodes)))
            print('release sample:', name, root.name, len(episodes), flush=True)
        release['total_episodes'] = sum(r['info']['total_episodes'] for r in release['roots'])
        release['total_frames'] = sum(r['info']['total_frames'] for r in release['roots'])
        releases.append(release)
    return releases


def main():
    result = dict(scope='Research only; cached examples are selected, release sample uses first/middle/last shards. No validation accuracy is claimed.',
                  packages={name: importlib.metadata.version(name) for name in ('numpy','scipy','pyarrow','mcap','mcap-ros2-support')},
                  cached=cached_audit(), releases=release_audit())
    result['raw_timing'] = raw_timing_audit(json.loads((FIGS/'ros_grip.json').read_text()))
    result['total_episodes_across_releases_not_deduplicated'] = sum(r['total_episodes'] for r in result['releases'])
    result['total_frames_across_releases_not_deduplicated'] = sum(r['total_frames'] for r in result['releases'])
    (OUT/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('wrote', OUT/'audit.json')


if __name__ == '__main__':
    main()
