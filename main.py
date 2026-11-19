#!/usr/bin/env python3
"""Run the frozen zgy P1A pipeline on one A/B test scene in Docker."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.dataio import parse_timestamp, time_to_ns

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT.parent
A_NAME = Path('B04赛题初赛A榜测试数据/初赛数据_A榜_A类传感器测试数据.csv')
B_NAME = Path('B04赛题初赛A榜测试数据/初赛数据_A榜_B类传感器测试数据.csv')
STAGES = ['estimate_pri.py', 'group_sequences.py', 'fuse_platforms.py', 'assist_fusion.py',
          'guard_b_assistance.py', 'localize_objects.py', 'match_tracks.py', 'classify_targets.py',
          'prepare_state_features.py', 'prepare_sequence_states.py', 'augment_state_features.py',
          'predict_target_state.py', 'export_submission.py']


def first_row(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, next(reader, None)


def inputs(root, explicit_a, explicit_b):
    if explicit_a and explicit_b:
        a, b = Path(explicit_a).resolve(), Path(explicit_b).resolve()
        if not a.is_file() or not b.is_file():
            raise FileNotFoundError('A/B input path does not exist')
        return a, b
    if bool(explicit_a) != bool(explicit_b):
        raise ValueError('Specify both --a-file and --b-file')
    candidates = {}
    for p in Path(root).rglob('*.csv'):
        try:
            header, row = first_row(p)
        except (OSError, UnicodeError, StopIteration):
            continue
        if not row or row.get('data_version') != 'test':
            continue
        h = set(header)
        if {'pulse_id', 'doa', 'platform_id', 'toa'} <= h:
            kind = 'A'
        elif {'track_id', 'datetime'} <= h and (('target_latitude' in h) or ('latitude' in h)):
            kind = 'B'
        else:
            continue
        candidates.setdefault((row['scenario_id'], row['data_version'], kind), []).append(p)
    scopes = {(s,v) for s,v,k in candidates}
    pairs = [(s,v) for s,v in scopes if len(candidates.get((s,v,'A'),[])) == 1 and len(candidates.get((s,v,'B'),[])) == 1]
    if len(pairs) != 1:
        raise ValueError(f'Expected exactly one test A/B pair in {root}; found {candidates}')
    s,v = pairs[0]
    return candidates[s,v,'A'][0].resolve(), candidates[s,v,'B'][0].resolve()


def configure(a, b):
    ah, ar = first_row(a)
    bh, br = first_row(b)
    if not ar or not br or ar['scenario_id'] != br['scenario_id'] or ar['data_version'] != br['data_version']:
        raise ValueError('A/B scenario and version must match')
    if ar['data_version'] != 'test':
        raise ValueError('This package runs the test split only')
    for source, dest in [(a, RAW_ROOT/A_NAME), (b, RAW_ROOT/B_NAME)]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if dest.resolve() != source:
                raise FileExistsError(f'Input staging path already points elsewhere: {dest}')
        else:
            dest.symlink_to(source)
    config_path = ROOT/'config/input_units.json'
    config = json.loads(config_path.read_text())
    key = (ar['scenario_id'], ar['data_version'])
    if not any((o['scenario_id'], o['data_version']) == key for o in config['origins']):
        stamp_ns, resolution_ns, kind = parse_timestamp(ar['datetime'])
        origin_guess_ns = stamp_ns - time_to_ns(ar['toa'], 'us')
        # Round to the A timestamp's recorded resolution; the reader checks
        # the inferred origin against every A observation afterwards.
        origin_ns = ((origin_guess_ns + resolution_ns // 2) // resolution_ns) * resolution_ns
        whole, fraction = divmod(origin_ns, 1_000_000_000)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc if kind == 'aware_UTC' else None)
        origin = epoch + timedelta(seconds=whole)
        stamp_text = origin.strftime('%Y-%m-%d %H:%M:%S')
        if fraction:
            stamp_text += '.' + f'{fraction:09d}'.rstrip('0')
        if kind == 'aware_UTC':
            stamp_text += 'Z'
        config['origins'].append({'scenario_id': key[0], 'data_version': key[1], 'datetime': stamp_text})
        config['origin_source'] = 'Inferred from first A datetime minus TOA; each row checked against the inferred origin.'
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    return key


def run_stage(script, extra=()):
    cmd = [sys.executable, str(ROOT/script), *extra]
    print('RUN', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=Path(os.getenv('DATA_ROOT', '/input')))
    p.add_argument('--out-dir', type=Path, default=Path(os.getenv('OUTPUT_DIR', '/output')))
    p.add_argument('--a-file', type=Path, default=os.getenv('A_FILE'))
    p.add_argument('--b-file', type=Path, default=os.getenv('B_FILE'))
    p.add_argument('--check-inputs-only', action='store_true')
    args = p.parse_args()
    a, b = inputs(args.input_root, args.a_file, args.b_file)
    scope = configure(a, b)
    print(json.dumps({'scenario_id': scope[0], 'data_version': scope[1], 'A': str(a), 'B': str(b)}, ensure_ascii=False), flush=True)
    if args.check_inputs_only:
        return
    for stage in STAGES:
        extra = ['--split', 'test']
        if stage == 'prepare_sequence_states.py':
            pass
        elif stage == 'export_submission.py':
            extra += ['--output-dir', str(ROOT/'output/submission')]
        run_stage(stage, extra)
    run_stage('verify_submission.py', ['--directory', str(ROOT/'output/submission')])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'output/submission').glob('*.csv'):
        shutil.copy2(path, args.out_dir/path.name)
    print(json.dumps({'output': str(args.out_dir), 'scenario_id': scope[0], 'data_version': scope[1]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
