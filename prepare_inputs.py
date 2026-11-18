#!/usr/bin/env python3
"""Validate the complete formal datasets through the streaming observation adapter."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from src.dataio import InputConfig, NS_PER_SECOND, iter_a_batches, read_b_tracks

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    'train': ('B04赛题初赛A榜训练数据/初赛数据_A榜_A类传感器训练数据.csv',
              'B04赛题初赛A榜训练数据/初赛数据_A榜_B类传感器训练数据.csv'),
    'test': ('B04赛题初赛A榜测试数据/初赛数据_A榜_A类传感器测试数据.csv',
             'B04赛题初赛A榜测试数据/初赛数据_A榜_B类传感器测试数据.csv')}
A_HEADER = ['scenario_id', 'data_version', 'pulse_id', 'datetime', 'toa', 'toa_raw', 'toa_input_unit',
            'param1', 'param2', 'param3', 'doa', 'sensor_longitude', 'sensor_latitude', 'sensor_height',
            'platform_id', 'time_seconds', 'timestamp_ns', 'recorded_timestamp_ns']
B_HEADER = ['scenario_id', 'data_version', 'track_id', 'datetime', 'time_ns', 'time_seconds', 'timestamp_ns',
            'target_longitude', 'target_latitude', 'target_height', 'target_type', 'target_class']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def a_row(pulse, unit):
    return [pulse.scenario, pulse.version, pulse.pulse_id, pulse.datetime_raw, pulse.toa, pulse.toa_raw, unit,
            pulse.rf, pulse.pw, pulse.pa, pulse.doa, *pulse.position, pulse.platform,
            pulse.toa / NS_PER_SECOND, pulse.timestamp_ns, pulse.recorded_timestamp_ns]


def prepare_a(source, output, split, config, batch_size=20_000, id_check='ordered', sample_rows=1000, write_normalized=False):
    source = Path(source)
    before = source.stat()
    started, count, batch_count = time.monotonic(), 0, 0
    platforms, scopes, ranges = Counter(), Counter(), {}
    difference_min, difference_max, maximum_batch = None, None, 0
    field_ranges = {k: [float('inf'), -float('inf')] for k in ('rf_mhz', 'pw_us', 'pa_dbm')}
    sample = output / f'{split}_A_sample.csv'
    full = output / f'{split}_A_normalized.csv'
    with sample.with_suffix('.csv.partial').open('w', encoding='utf-8', newline='') as sample_stream:
        writer = csv.writer(sample_stream)
        writer.writerow(A_HEADER)
        full_stream = full.with_suffix('.csv.partial').open('w', encoding='utf-8', newline='') if write_normalized else None
        try:
            full_writer = csv.writer(full_stream) if full_stream else None
            if full_writer:
                full_writer.writerow(A_HEADER)
            for batch in iter_a_batches(source, config, batch_size, id_check):
                batch_count += 1
                maximum_batch = max(maximum_batch, len(batch))
                for pulse in batch:
                    if pulse.version != split:
                        raise ValueError(f'Expected {split} input, got {pulse.version}')
                    count += 1
                    platforms[pulse.platform] += 1
                    scopes[pulse.scenario, pulse.version] += 1
                    key = pulse.scope
                    if key not in ranges:
                        ranges[key] = [pulse.toa, pulse.toa]
                    else:
                        ranges[key][0] = min(ranges[key][0], pulse.toa)
                        ranges[key][1] = max(ranges[key][1], pulse.toa)
                    error = pulse.timestamp_ns - pulse.recorded_timestamp_ns
                    difference_min = min(difference_min, error) if difference_min is not None else error
                    difference_max = max(difference_max, error) if difference_max is not None else error
                    for name, value in zip(field_ranges, (pulse.rf, pulse.pw, pulse.pa)):
                        field_ranges[name][0] = min(field_ranges[name][0], value)
                        field_ranges[name][1] = max(field_ranges[name][1], value)
                    if count <= sample_rows or full_writer:
                        row = a_row(pulse, config.a_toa_unit)
                        if count <= sample_rows:
                            writer.writerow(row)
                        if full_writer:
                            full_writer.writerow(row)
                if batch_count % max(1, 1_000_000 // batch_size) == 0:
                    print(json.dumps({'split': split, 'validated_A_rows': count}, ensure_ascii=False), flush=True)
        finally:
            if full_stream:
                full_stream.close()
    if (before.st_size, before.st_mtime_ns) != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError('Input file changed during validation')
    sample.with_suffix('.csv.partial').replace(sample)
    if full_stream:
        full.with_suffix('.csv.partial').replace(full)
    return {'source': str(source), 'source_bytes': before.st_size, 'source_sha256': digest(source),
            'input_rows': count, 'platform_count': len(platforms), 'platform_pulse_counts': dict(platforms),
            'scene_versions': [dict(scenario_id=k[0], data_version=k[1], pulses=v) for k, v in sorted(scopes.items())],
            'platform_time_ranges_ns': [dict(scenario_id=k[0], data_version=k[1], platform_id=k[2], min_ns=v[0], max_ns=v[1]) for k, v in sorted(ranges.items())],
            'recorded_datetime_difference_ns_min_max': [difference_min, difference_max], 'feature_ranges': field_ranges,
            'batches': batch_count, 'maximum_batch_rows': maximum_batch,
            'ID_validation': 'Strictly increasing numeric IDs per scene/version (therefore unique)' if id_check == 'ordered' else 'Exact scene/version/pulse uniqueness via temporary SQLite',
            'truth_columns_read': [], 'sample_file': str(sample), 'sample_rows': min(count, sample_rows),
            'full_normalized_file': str(full) if full_stream else None, 'elapsed_seconds': round(time.monotonic() - started, 2)}


def prepare_b(source, output, split, config):
    tracks = read_b_tracks(source, config)
    rows, summaries = [], []
    for key, track in sorted(tracks.items()):
        if key[1] != split:
            raise ValueError('Wrong B data_version')
        gaps = Counter(b.time_ns - a.time_ns for a, b in zip(track.observations, track.observations[1:]))
        for obs in track.observations:
            rows.append([*key, obs.datetime_raw, obs.time_ns, obs.time_ns / NS_PER_SECOND, obs.timestamp_ns,
                         *obs.position, obs.target_type, obs.target_class])
        summaries.append({'scenario_id': key[0], 'data_version': key[1], 'track_id': key[2],
            'observations': len(track.observations), 'first_time_ns': track.times_ns[0], 'last_time_ns': track.times_ns[-1],
            'gap_counts_seconds': {str(k / NS_PER_SECOND): v for k, v in sorted(gaps.items())},
            'source_was_reordered': track.reordered,
            'target_type_values': sorted({o.target_type for o in track.observations}),
            'public_class_values': sorted({o.target_class for o in track.observations}),
            'long_gap_interpolation_blocked': sum(v for k, v in gaps.items() if k > config.maximum_b_interpolation_gap_seconds * NS_PER_SECOND)})
    path = output / f'{split}_B_normalized.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(B_HEADER)
        writer.writerows(rows)
    return {'source': str(source), 'source_sha256': digest(source), 'rows': len(rows), 'tracks': summaries,
            'normalized_file': str(path), 'target_id_read': False, 'synthetic_cadence_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/input_units.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/prepared')
    parser.add_argument('--split', choices=('both', 'train', 'test'), default='both')
    parser.add_argument('--a-file', type=Path)
    parser.add_argument('--b-file', type=Path)
    parser.add_argument('--batch-size', type=int, default=20_000)
    parser.add_argument('--id-check', choices=('ordered', 'sqlite'), default='ordered')
    parser.add_argument('--sample-rows', type=int, default=1000)
    parser.add_argument('--write-normalized', action='store_true', help='Also write complete normalized A CSVs (large)')
    args = parser.parse_args()
    if args.split == 'both' and (args.a_file or args.b_file):
        parser.error('File overrides require --split train or --split test')
    if args.sample_rows < 0 or args.batch_size <= 0:
        parser.error('Invalid sample or batch size')
    cfg_dict = json.loads(args.config.read_text(encoding='utf-8'))
    config = InputConfig.from_dict(cfg_dict)
    output = args.output_dir.resolve()
    sources = {}
    for split in ('train', 'test') if args.split == 'both' else (args.split,):
        paths = args.a_file or args.data_root / DEFAULT_FILES[split][0], args.b_file or args.data_root / DEFAULT_FILES[split][1]
        for path in paths:
            if not path.is_file():
                parser.error(f'Missing input: {path}')
            if output == path.resolve().parent:
                parser.error('Output must be separate from raw input directories')
        sources[split] = paths
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'input_config': cfg_dict, 'internal_toa_units_per_second': NS_PER_SECOND,
              'time_reference': 'Configured public scene-local origins, checked against A datetime. Naive timestamps are not silently labeled UTC.',
              'clock_bias_correction': 'Not estimated or applied. Actual B timestamps and their offsets are retained.', 'datasets': {}}
    for split, (a_source, b_source) in sources.items():
        b_report = prepare_b(b_source, output, split, config)
        a_report = prepare_a(a_source, output, split, config, args.batch_size, args.id_check, args.sample_rows, args.write_normalized)
        report['datasets'][split] = {'A': a_report, 'B': b_report}
        (output / 'input_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'split': split, 'A_rows': a_report['input_rows'], 'A_platforms': a_report['platform_count'],
                          'B_rows': b_report['rows'], 'B_tracks': len(b_report['tracks'])}, ensure_ascii=False), flush=True)
    report['status'] = 'passed'
    (output / 'input_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'validation': str(output / 'input_validation.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
