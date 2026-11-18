#!/usr/bin/env python3
"""Recompute 2D A-only positions for saved cross-platform object events."""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from prepare_inputs import digest
from src.localization import Bearing, HorizontalFrame, LocalizationConfig, localize, filter_track
from src.localization_plot import render

ROOT = Path(__file__).resolve().parent
SEQ_KEYS = ['scenario_id', 'data_version', 'platform_id', 'sequence_id']
EVENT_KEYS = ['scenario_id', 'data_version', 'object_id', 'event_id']
POINT_FIELDS = EVENT_KEYS+['time_ns', 'time_seconds', 'east_m', 'north_m', 'longitude', 'latitude',
    'input_platform_count', 'used_platform_count', 'platform_ids', 'sequence_ids', 'time_span_seconds',
    'fit_status', 'crossing_angle_degrees', 'condition_number', 'rms_bearing_residual_degrees',
    'maximum_bearing_residual_degrees', 'covariance_ee_m2', 'covariance_en_m2', 'covariance_nn_m2',
    'major_error_proxy_m', 'minor_error_proxy_m']
REJECT_FIELDS = EVENT_KEYS+['time_ns', 'platform_ids', 'sequence_ids', 'reason']
OBS_FIELDS = EVENT_KEYS+['platform_id', 'sequence_id', 'time_ns', 'observer_longitude', 'observer_latitude',
    'observer_east_m', 'observer_north_m', 'doa_degrees', 'used', 'range_m',
    'bearing_residual_degrees', 'effective_sigma_degrees']
TRACK_FIELDS = EVENT_KEYS[:3]+['segment_id', 'time_ns', 'time_seconds', 'east_m', 'north_m',
    'longitude', 'latitude', 'measurement_east_m', 'measurement_north_m',
    'velocity_east_m_per_second', 'velocity_north_m_per_second', 'event_count', 'event_ids',
    'filter_status', 'innovation_squared', 'filter_covariance_ee_m2', 'filter_covariance_en_m2',
    'filter_covariance_nn_m2']


def read_bearings(path, frames):
    observations = {}
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            key = tuple(r[k] for k in SEQ_KEYS)
            lon, lat, doa, error = (float(r[k]) for k in ('observer_longitude', 'observer_latitude',
                                                        'doa_degrees', 'doa_standard_error_degrees'))
            start, end, t = (int(r[k]) for k in ('start_ns', 'end_ns', 'time_ns'))
            if (key in observations or not all(key) or key[:2] not in frames or t != (start+end)//2 or
                start < 0 or end < start or not all(math.isfinite(v) for v in (lon, lat, doa, error)) or
                not -180 <= lon <= 180 or not -89 <= lat <= 89 or not 0 <= doa <= 360 or error <= 0):
                raise ValueError('Invalid/duplicate bearing observation')
            frame = frames[key[:2]]
            # observer_height is deliberately neither parsed nor supplied to the solver.
            observations[key] = Bearing(key, t, start, end, frame.project(lon, lat),
                frame.direction(lon, lat, doa), error, lon, lat, doa)
    return observations


def checked_inputs(root, fusion_dir, split):
    meta_path = fusion_dir/'fusion_run.json'
    meta = json.loads(meta_path.read_text())
    paths = {
        'fusion_run': meta_path,
        'events': fusion_dir/'cross_platform_events.csv',
        'links': fusion_dir/'sequence_object_links.csv',
        'objects': fusion_dir/'fused_objects.csv',
        'bearings': root/f'output/grouped/{split}/bearing_observations.csv',
        'positions': root/f'output/group_inputs/{split}_platform_positions.csv',
    }
    hashes = {k: digest(p) for k, p in paths.items()}
    for key, filename in [('events', 'cross_platform_events.csv'), ('links', 'sequence_object_links.csv'), ('objects', 'fused_objects.csv')]:
        if hashes[key] != meta['files'][filename]['sha256']:raise ValueError(f'Changed fusion artifact: {filename}')
    for key, name in [('bearings', 'bearing_source_sha256'), ('positions', 'positions_source_sha256')]:
        if hashes[key] != meta[name]:raise ValueError(f'Changed localization input: {key}')
    # The original A-only reference frame is coordinate metadata, not a B location.
    frame_path = root/f'output/fused/{split}/fusion_run.json'
    baseline = json.loads(frame_path.read_text())
    if meta.get('B_used') and digest(frame_path) != meta['baseline_run_sha256']:
        raise ValueError('Fusion reference-frame source changed')
    if baseline['B_used']:raise ValueError('Reference frame must come from A-only processing')
    for key in ('bearing_source_sha256', 'positions_source_sha256'):
        if baseline[key] != meta[key]:raise ValueError('Frame/observation mismatch')
    frames = {(r['scenario_id'], r['data_version']): HorizontalFrame(r['origin_longitude'], r['origin_latitude']) for r in baseline['frames']}
    hashes['reference_frame_run'] = digest(frame_path)
    paths['reference_frame_run'] = frame_path
    return meta, paths, hashes, frames


def write_rows(path, fields, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader();w.writerows(rows)


def read_rows(path):
    with Path(path).open(newline='') as f:
        yield from csv.DictReader(f)


def run(root, fusion_dir, output, split, cfg):
    cfg.validate()
    root, fusion_dir, output = map(lambda p: Path(p).resolve(), (root, fusion_dir, output))
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    started = time.monotonic()
    upstream, paths, hashes, frames = checked_inputs(root, fusion_dir, split)
    observations = read_bearings(paths['bearings'], frames)
    links = {}
    for r in read_rows(paths['links']):
        key = tuple(r[k] for k in SEQ_KEYS)
        if key in links or key not in observations:raise ValueError('Invalid fusion sequence key')
        links[key] = r
    if links.keys() != observations.keys():raise ValueError('Incomplete fusion links')
    objects = {}
    for r in read_rows(paths['objects']):
        key = tuple(r[k] for k in EVENT_KEYS[:3])
        if key in objects or not key[2]:raise ValueError('Invalid object key')
        objects[key] = r
    points, rejected, observation_rows = [], [], []
    seen, owned = set(), set()
    event_counts = Counter()
    for r in read_rows(paths['events']):
        scene = r['scenario_id'], r['data_version']
        key = scene+(r['event_id'],)
        if key in seen or scene not in frames or scene[1] != split:raise ValueError('Duplicate/mixed event scope')
        seen.add(key)
        platforms, sequences = r['platform_ids'].split('|'), r['sequence_ids'].split('|')
        if len(platforms) != len(set(platforms)) or len(platforms) != len(sequences) or len(platforms) != int(r['platform_count']):
            raise ValueError('Invalid event platform membership')
        obs = []
        for p, s in zip(platforms, sequences):
            k = scene+(p, s)
            if k not in links or k in owned:raise ValueError('Missing/reused event sequence')
            if links[k]['event_id'] != r['event_id'] or links[k]['object_id'] != r['object_id']:
                raise ValueError('Event/sequence identity mismatch')
            owned.add(k);obs.append(observations[k])
        prefix = {k: r[k] for k in EVENT_KEYS}
        event_counts['all'] += 1
        if not r['object_id']:
            result, reason = None, 'unassigned_object_identity'
        else:
            if scene+(r['object_id'],) not in objects:raise ValueError('Unknown event object')
            event_counts['identified'] += 1
            result, reason = localize(obs, cfg)
        if result is None:
            rejected.append(prefix|{k:r[k] for k in ('time_ns', 'platform_ids', 'sequence_ids')}|{'reason':reason})
            continue
        selected = result['selected']
        x, y = result['east_m'], result['north_m']
        lon, lat = frames[scene].inverse(x, y)
        t = result['time_ns']
        row = prefix|{k: result[k] for k in ('time_ns', 'east_m', 'north_m', 'time_span_seconds', 'fit_status',
            'crossing_angle_degrees', 'condition_number', 'rms_bearing_residual_degrees',
            'maximum_bearing_residual_degrees', 'major_error_proxy_m', 'minor_error_proxy_m')}
        row.update(time_seconds=t/1e9, longitude=lon, latitude=lat, input_platform_count=len(obs),
            used_platform_count=len(selected), platform_ids='|'.join(o.platform for o in selected),
            sequence_ids='|'.join(o.key[3] for o in selected))
        row.update(zip(('covariance_ee_m2', 'covariance_en_m2', 'covariance_nn_m2'), result['covariance']))
        points.append(row|{'covariance':result['covariance']})
        used = {o.key for o in selected}
        for o in obs:
            dist, _, residual, sigma = result['metrics'][o.key]
            observation_rows.append(prefix|dict(platform_id=o.platform, sequence_id=o.key[3], time_ns=o.time,
                observer_longitude=o.longitude, observer_latitude=o.latitude, observer_east_m=o.xy[0],
                observer_north_m=o.xy[1], doa_degrees=o.doa, used=o.key in used, range_m=dist,
                bearing_residual_degrees=math.degrees(residual), effective_sigma_degrees=math.degrees(sigma)))
        if len(seen)%10000 == 0:print(json.dumps({'split':split, 'localized_events':len(points), 'processed_events':len(seen)}), flush=True)
    expected = {key for key, r in links.items() if r['event_id']}
    if owned != expected or len(seen) != upstream['event_count']:raise ValueError('Incomplete event coverage')
    by_object = defaultdict(list)
    for p in points:by_object[tuple(p[k] for k in EVENT_KEYS[:3])].append(p)
    tracks, track_rejected, summaries = [], [], []
    for key, obj in sorted(objects.items()):
        ps = sorted(by_object.get(key, []), key=lambda p:(p['time_ns'], p['event_id']))
        accepted, refused = filter_track(ps, cfg)
        for r in accepted:
            lon, lat = frames[key[:2]].inverse(r['east_m'], r['north_m'])
            row = dict(zip(EVENT_KEYS[:3], key))|{k: r[k] for k in ('segment_id', 'time_ns', 'east_m', 'north_m',
                'measurement_east_m', 'measurement_north_m', 'velocity_east_m_per_second',
                'velocity_north_m_per_second', 'event_count', 'event_ids', 'filter_status', 'innovation_squared')}
            row.update(time_seconds=r['time_ns']/1e9, longitude=lon, latitude=lat)
            row.update(zip(('filter_covariance_ee_m2', 'filter_covariance_en_m2', 'filter_covariance_nn_m2'), r['filter_covariance']))
            tracks.append(row)
        track_rejected.extend(dict(zip(EVENT_KEYS[:3], key))|{k:r[k] for k in ('time_ns', 'segment_id', 'event_ids', 'innovation_squared', 'reason')} for r in refused)
        summaries.append(dict(zip(EVENT_KEYS[:3], key))|dict(associated_sequence_count=int(obj['sequence_count']),
            identified_event_count=int(obj['event_count']), localized_point_count=len(ps), track_point_count=len(accepted),
            track_segment_count=len({r['segment_id'] for r in accepted}), track_rejected_bin_count=len(refused),
            start_seconds=min((p['time_ns']/1e9 for p in ps), default=''),
            end_seconds=max((p['time_ns']/1e9 for p in ps), default=''),
            localization_status='localized' if ps else 'no_usable_multiplatform_event'))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f'.{output.name}.partial-{os.getpid()}')
    partial.mkdir()
    write_rows(partial/'localized_points.csv', POINT_FIELDS, ({k:p[k] for k in POINT_FIELDS} for p in points))
    write_rows(partial/'rejected_localizations.csv', REJECT_FIELDS, rejected)
    write_rows(partial/'localization_observations.csv', OBS_FIELDS, observation_rows)
    write_rows(partial/'track_points.csv', TRACK_FIELDS, tracks)
    write_rows(partial/'track_rejections.csv', EVENT_KEYS[:3]+['time_ns','segment_id','event_ids','innovation_squared','reason'], track_rejected)
    write_rows(partial/'localization_objects.csv', list(summaries[0]) if summaries else EVENT_KEYS[:3], summaries)
    report = {'config':asdict(cfg), 'split':split, 'dimension':2, 'height_used':False, 'height_estimated':False,
        'A_truth_columns_used':[], 'B_coordinates_used_in_localization':False,
        'upstream_association_B_assisted':bool(upstream['B_used']), 'TOA_common_clock_assumed':True,
        'clock_offset_estimated':False,
        'frames':[dict(scenario_id=k[0], data_version=k[1], origin_longitude=f.longitude, origin_latitude=f.latitude)
                  for k, f in sorted(frames.items())],
        'input_files':{k:{'path':str(paths[k]), 'sha256':hashes[k]} for k in paths},
        'input_event_count':len(seen), 'identified_event_count':event_counts['identified'],
        'localized_point_count':len(points), 'rejected_event_count':len(rejected),
        'rejection_reasons':dict(Counter(r['reason'] for r in rejected)),
        'used_platform_counts':dict(Counter(p['used_platform_count'] for p in points)),
        'fit_status_counts':dict(Counter(p['fit_status'] for p in points)),
        'associated_object_count':len(objects), 'localized_object_count':len(by_object),
        'track_point_count':len(tracks), 'track_segment_count':sum(r['track_segment_count'] for r in summaries),
        'track_rejected_bin_count':len(track_rejected),
        'limitations':[
            'Same upstream B-assisted identities; localization is not an independent A-only accuracy evaluation.',
            'Existing consistent A events only; no new pairing or identity repair is performed.',
            'Short-window static target approximation with motion uncertainty; no unknown clock correction.',
            'Projected horizontal rays approximate the local WGS84 surface; all input heights are ignored.',
            'Error ellipses/covariances are model proxies, not calibrated confidence intervals.',
            'Two bearings fit exactly and cannot independently detect a wrong object association.',
            'Filter output is only at accepted measurement times; gaps are not filled or extrapolated.'
        ],
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ('localize_objects.py','src/localization.py','src/localization_plot.py')},
    }
    render(partial, frames, points, tracks, summaries, paths['positions'], report)
    report['elapsed_seconds'] = round(time.monotonic()-started, 2)
    report['files'] = {p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(partial.iterdir())}
    (partial/'localization_run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    partial.rename(output)
    print(json.dumps({k:report[k] for k in ('split','localized_point_count','used_platform_counts','rejection_reasons',
        'localized_object_count','track_point_count','track_segment_count','elapsed_seconds')}, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root', type=Path, default=ROOT)
    p.add_argument('--fusion-root', type=Path, help='Default: PIPELINE_ROOT/output/fused_b_assisted_guarded')
    p.add_argument('--output-dir', type=Path, default=ROOT/'output/localized_2d')
    p.add_argument('--config', type=Path, default=ROOT/'config/localization_config.json')
    p.add_argument('--split', choices=('train','test','both'), default='both')
    a = p.parse_args()
    cfg = LocalizationConfig(**json.loads(a.config.read_text()))
    fusion = a.fusion_root or a.pipeline_root/'output/fused_b_assisted_guarded'
    for split in (('train','test') if a.split=='both' else (a.split,)):
        run(a.pipeline_root, fusion/split, a.output_dir/split, split, cfg)


if __name__=='__main__':main()
