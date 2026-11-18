#!/usr/bin/env python3
"""Extract platform-local stable/jittered PRI sequences; retain unknown pulses."""
import argparse
import csv
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from prepare_inputs import DEFAULT_FILES, digest
from src.dataio import InputConfig, iter_a_batches
from src.pri import PRIConfig, WindowedPRI

ROOT = Path(__file__).resolve().parent
PULSE_FIELDS = ['scenario_id', 'data_version', 'platform_id', 'pulse_id', 'sequence_id',
    'pri_model', 'pri_center_ns', 'pri_center_us', 'pri_p05_us', 'pri_p95_us', 'confidence',
    'previous_pulse_id', 'observed_interval_ns', 'interval_multiple', 'status']
SEQUENCE_FIELDS = ['scenario_id', 'data_version', 'platform_id', 'sequence_id', 'pri_model',
    'pulse_count', 'start_toa_ns', 'end_toa_ns', 'pri_center_ns', 'pri_center_us', 'pri_p05_us',
    'pri_p95_us', 'direct_interval_fraction', 'confidence', 'rf_mean', 'pw_mean', 'pa_mean',
    'doa_mean_degrees', 'doa_std_degrees', 'max_relative_interval_deviation', 'basis']


def extract_file(source, output, input_config, pri_config, batch_size=20_000,
                 id_check='ordered', expected_split=None, progress=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    pri_config.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f'.{output.name}.partial-{os.getpid()}')
    partial.mkdir()  # On failure, retain partial results for diagnosis, never publish as complete.
    before, start = source.stat(), time.monotonic()
    counts, platforms = Counter(), Counter()
    with (partial / 'pulse_pri.csv').open('w', encoding='utf-8', newline='') as pulse_stream, \
         (partial / 'pri_sequences.csv').open('w', encoding='utf-8', newline='') as sequence_stream:
        pw = csv.DictWriter(pulse_stream, PULSE_FIELDS)
        sw = csv.DictWriter(sequence_stream, SEQUENCE_FIELDS)
        pw.writeheader()
        sw.writeheader()

        def write_pulse(item):
            p = item.pulse
            row = dict(zip(PULSE_FIELDS[:4], p.key))
            row.update(item.assignment or {'pri_model': 'unknown', 'status': 'unknown'})
            pw.writerow(row)
            counts[row['pri_model']] += 1
            platforms[p.scope + (row['pri_model'],)] += 1

        runner = WindowedPRI(pri_config, sw.writerow, write_pulse)
        next_progress = 1_000_000
        for batch in iter_a_batches(source, input_config, batch_size, id_check):
            for pulse in batch:
                if expected_split and pulse.version != expected_split:
                    raise ValueError(f'Expected {expected_split}, got {pulse.version}')
                runner.feed(pulse)
            if progress and runner.input_count >= next_progress:
                print(json.dumps({'source': source.name, 'input_pulses': runner.input_count,
                    'written_pulses': runner.output_count, 'elapsed_seconds': round(time.monotonic() - start, 1)},
                    ensure_ascii=False), flush=True)
                next_progress += 1_000_000
        runner.finish()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source changed during PRI extraction')
    report = {'source': str(source), 'source_bytes': before.st_size, 'source_sha256': digest(source),
        'input_config': asdict(input_config) | {'origins': [
            {'scenario_id': k[0], 'data_version': k[1], 'timestamp_ns': v[0], 'clock_kind': v[1]}
            for k, v in sorted(input_config.origins.items())]},
        'pri_config': asdict(pri_config), 'input_pulses': runner.input_count,
        'output_pulses': runner.output_count, 'model_pulse_counts': dict(counts),
        'supported_pulse_fraction': 1 - counts['unknown'] / runner.input_count,
        'platform_model_counts': [dict(zip(('scenario_id', 'data_version', 'platform_id', 'pri_model'), k), pulses=v)
                                  for k, v in sorted(platforms.items())],
        'sequence_statistics': dict(runner.extractor.statistics), 'windows': runner.windows,
        'maximum_pending_pulses': runner.max_pending, 'batch_size': batch_size, 'id_check': id_check,
        'truth_columns_read': [], 'B_used': False, 'confidence_is_calibrated_probability': False,
        'sequence_is_target_identity': False, 'sequence_is_true_CPI': False,
        'unknown_policy': 'Retain every pulse with blank sequence and PRI fields',
        'algorithm_source_sha256': {name: digest(ROOT / name) for name in
            ('estimate_pri.py', 'src/pri.py', 'src/dataio.py')},
        'files': {name: {'sha256': digest(partial / name), 'bytes': (partial / name).stat().st_size}
                  for name in ('pulse_pri.csv', 'pri_sequences.csv')},
        'elapsed_seconds': round(time.monotonic() - start, 2)}
    # asdict contains a tuple-key origin mapping; replace it above before serialization.
    (partial / 'pri_run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    partial.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent)
    parser.add_argument('--split', choices=('train', 'test', 'both'), default='both')
    parser.add_argument('--a-file', type=Path)
    parser.add_argument('--input-config', type=Path, default=ROOT / 'config/input_units.json')
    parser.add_argument('--pri-config', type=Path, default=ROOT / 'config/pri_config.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/pri')
    parser.add_argument('--batch-size', type=int, default=20_000)
    parser.add_argument('--id-check', choices=('ordered', 'sqlite'), default='ordered')
    args = parser.parse_args()
    if args.a_file and args.split == 'both':
        parser.error('--a-file requires a single split')
    ic = InputConfig.from_dict(json.loads(args.input_config.read_text(encoding='utf-8')))
    pc = PRIConfig(**json.loads(args.pri_config.read_text(encoding='utf-8')))
    for split in (('train', 'test') if args.split == 'both' else (args.split,)):
        source = args.a_file or args.data_root / DEFAULT_FILES[split][0]
        report = extract_file(source, args.output_dir / split, ic, pc, args.batch_size, args.id_check, split)
        print(json.dumps({k: report[k] for k in ('input_pulses', 'model_pulse_counts',
              'supported_pulse_fraction', 'elapsed_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
