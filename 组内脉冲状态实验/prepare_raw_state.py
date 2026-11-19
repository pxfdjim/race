"""Align the frozen PRI members with original pulse measurements for state training."""

import argparse
import csv
import json
import math
import mmap
import struct
import sys
from array import array
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory


SUMMARY_DIM = 43
PULSE_FIELDS = ('relative_rf', 'relative_pw', 'absolute_pa', 'relative_pa', 'normalized_gap',
                'fractional_position', 'has_previous_pulse')
PULSE = struct.Struct('<7f')
RECORD = struct.Struct(f'<{SUMMARY_DIM}f5I')
FOLD_NAMES = ('development', 'selection', 'check_300_320', 'check_400_420',
              'diagnostic_420_500', 'instance_holdout')


def add_repository_paths(root):
    sys.path[:0] = [str(root), str(root / '状态两阶段实验'), str(root / 'CPI时序状态实验')]


def summary_scaling(rows):
    count = len(rows)
    means = [math.fsum(column) / count for column in zip(*rows)]
    scale = [max(1e-6, math.sqrt(math.fsum((value - mean) ** 2 for value in column) / count))
             for column, mean in zip(zip(*rows), means, strict=True)]
    return means, scale


def prepare(root, output):
    from compare_class_state import (SEQ, check_hash, classifier_input, csv_write, read_json, rows, seq_key,
                                     state_features, state_statistics, write_json)
    from compare_cpi_context import partition
    from cpi_context import TIME_BLOCKS_SECONDS
    from prepare_inputs import DEFAULT_FILES, digest

    if output.exists():
        raise FileExistsError(output)
    old_dir = root / 'CPI时序状态实验/results'
    baseline_file = old_dir / 'joint_class_state_validation.csv'
    old_report = read_json(old_dir / 'comparison.json')
    manifest_dir = root / '跨时片段关联实验/models'
    manifest = read_json(manifest_dir / 'fit_manifest.json')
    excluded = set(manifest['excluded_target_instances'])
    if manifest['fit_range_ns'] != [0, 300_000_000_000] or excluded != set(old_report['excluded_fit_target_instances']):
        raise ValueError('Reference folds differ')
    for file, sha in manifest['files'].items():
        check_hash(manifest_dir / file, sha)
    names, windows, keys, state_truth, targets = state_features(root, 'train', True)
    if len(names) != SUMMARY_DIM:
        raise ValueError('The frozen state summary schema changed')
    groups = partition(windows, keys, state_truth, targets, excluded)
    all_keys = list(chain.from_iterable(groups[n] for n in FOLD_NAMES))
    if len(all_keys) != len(set(all_keys)):
        raise ValueError('Fold assignments overlap')
    waveform_rows = classifier_input(root, 'train', set(all_keys))
    reference = {seq_key(r): r for r in rows(baseline_file)}
    expected_refs = set(chain.from_iterable(groups[n] for n in FOLD_NAMES[1:]))
    if reference.keys() != expected_refs:
        raise ValueError('Reference predictions have different evaluation keys')
    for fold in FOLD_NAMES[1:]:
        values = [reference[k] for k in groups[fold]]
        truth = [state_truth[keys[k]] for k in groups[fold]]
        if any(r['fold'] != fold or tuple(int(r[f'true_state_{s}_pulses']) for s in range(3)) != t
               for r, t in zip(values, truth, strict=True)):
            raise ValueError(f'Reference truth or fold differs: {fold}')
        if state_statistics((int(r['three_way_target_state']) for r in values), truth) != old_report['results'][fold]['original_three_way']:
            raise ValueError(f'Reference baseline metrics differ: {fold}')
    means, scales = summary_scaling([windows[keys[k]].features for k in groups['development']])
    metadata = {}
    size = 0
    for i, k in enumerate(all_keys):
        w = windows[keys[k]]
        r = waveform_rows[k]
        first, last = int(r['start_toa_ns']), int(r['end_toa_ns'])
        if (w.first, w.last, w.count) != (first, last, int(r['pulse_count'])):
            raise ValueError(f'PRI and state boundaries/count differ: {k}')
        metadata[k] = (i, size, w.count, first, last, float(r['pri_center_ns']),
                       float(r['rf_mean']), float(r['pw_mean']), float(r['pa_mean']))
        size += w.count
    raw = root.parent / DEFAULT_FILES['train'][0]
    pri = root / 'output/pri/train/pulse_pri.csv'
    saved_pri = read_json(pri.with_name('pri_run.json'))
    if saved_pri['input_config']['a_toa_unit'] != 'us':
        raise ValueError('Raw pulse preparation requires microsecond TOA')
    check_hash(pri, saved_pri['files'][pri.name]['sha256'])
    check_hash(raw, saved_pri['source_sha256'])
    print(json.dumps({'phase': 'features_verified', 'sequences': len(all_keys), 'pulses': size}), flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.state-raw-', dir=output.parent) as directory:
        stage = Path(directory) / 'prepare'
        stage.mkdir()
        raw_stream = stage / 'pulses.bin'
        pulse_counts, max_bin_count = align_pulses(raw, pri, raw_stream, metadata, size)
        if any(tuple(pulse_counts[3 * i:3 * i + 3]) != state_truth[keys[k]]
               for k, (i, *_) in metadata.items()):
            raise ValueError('Raw and cached state labels disagree')
        label_counts = {}
        for fold in FOLD_NAMES:
            with (stage / f'{fold}.bin').open('wb') as stream:
                for k in groups[fold]:
                    _, offset, count, *_ = metadata[k]
                    wkey = keys[k]
                    x = [max(-8., min(8., (v - m) / s))
                         for v, m, s in zip(windows[wkey].features, means, scales, strict=True)]
                    stream.write(RECORD.pack(*x, offset, count, *state_truth[wkey]))
            label_counts[fold] = dict(zip(('0', '1', '2'), map(sum, zip(*(state_truth[keys[k]] for k in groups[fold])))))
        fields = ('fold', *SEQ, 'pulse_count', 'object_id', 'target_class', 'three_way_target_state',
                  'true_state_0_pulses', 'true_state_1_pulses', 'true_state_2_pulses')
        csv_write(stage / 'evaluation_index.csv', fields, ([reference[k][f] for f in fields]
                  for fold in FOLD_NAMES[1:] for k in groups[fold]))
        write_json(stage / 'prepared.json', {'summary_features': names, 'pulse_features': PULSE_FIELDS,
            'summary_mean': means, 'summary_scale': scales, 'pulses': size, 'max_pulses_in_sequence': max_bin_count,
            'fold_sequences': {f: len(groups[f]) for f in FOLD_NAMES}, 'pulse_label_counts': label_counts,
            'time_blocks_seconds': TIME_BLOCKS_SECONDS, 'excluded_fit_target_instances': sorted(excluded),
            'raw_A_sha256': saved_pri['source_sha256'], 'pulse_PRI_sha256': saved_pri['files'][pri.name]['sha256'],
            'reference_predictions_sha256': digest(baseline_file),
            'prepared_files_sha256': {p.name: digest(p) for p in stage.iterdir()}})
        stage.rename(output)
    print(json.dumps({'phase': 'prepared', 'label_counts': label_counts}), flush=True)


def align_pulses(raw, pri, output, sequences, total_pulses):
    from src.dataio import time_to_ns

    count = len(sequences)
    seen = array('I', [0]) * count
    truth = array('I', [0]) * (count * 3)
    preceding_toa = array('Q', [0]) * count
    preceding_id = [''] * count
    common = ('scenario_id', 'data_version', 'platform_id', 'pulse_id')
    with raw.open(encoding='utf-8-sig', newline='') as raw_stream, pri.open(newline='') as pri_stream, output.open('w+b') as f:
        raw_rows, pri_rows = csv.reader(raw_stream), csv.reader(pri_stream)
        ah, ph = next(raw_rows), next(pri_rows)
        a0, a1, a2, ap = [ah.index(k) for k in common]
        p0, p1, p2, pp = [ph.index(k) for k in common]
        af, aw, aa, at, ast = [ah.index(k) for k in ('param1', 'param2', 'param3', 'toa', 'target_state')]
        seq, pprev, pdelta = [ph.index(k) for k in ('sequence_id', 'previous_pulse_id', 'observed_interval_ns')]
        f.truncate(total_pulses * PULSE.size)
        with mmap.mmap(f.fileno(), 0) as destination:
            for n, (ar, pr) in enumerate(zip(raw_rows, pri_rows, strict=True), 1):
                pid = pr[pp]
                if (ar[a0], ar[a1], ar[a2], ar[ap]) != (pr[p0], pr[p1], pr[p2], pid):
                    raise ValueError(f'PRI pulse and source disagree at row {n}')
                if n % 4_000_000 == 0:
                    print(json.dumps({'phase': 'pulse_stream', 'source_rows_read': n}), flush=True)
                metadata = sequences.get((pr[p0], pr[p1], pr[p2], pr[seq]))
                if metadata is None:
                    continue
                i, offset, length, first, last, center, rf_mean, pw_mean, pa_mean = metadata
                pos = seen[i]
                t = time_to_ns(ar[at], 'us')
                if pos >= length or not first <= t <= last:
                    raise ValueError(f'Pulse falls outside its sequence: {pid}')
                if pos:
                    dt = t - preceding_toa[i]
                    if dt <= 0 or int(pr[pdelta]) != dt or pr[pprev] != preceding_id[i]:
                        raise ValueError(f'Pulse predecessor differs from original PRI result: {pid}')
                    gap = math.log(max(0.01, dt / center))
                else:
                    if pr[pdelta] or pr[pprev] or first != t:
                        raise ValueError(f'Sequence start differs from PRI result: {pid}')
                    gap = 0.
                if pos + 1 == length and last != t:
                    raise ValueError(f'Sequence end differs from PRI result: {pid}')
                pa = float(ar[aa])
                values = ((float(ar[af]) - rf_mean) / 2, (float(ar[aw]) - pw_mean) / .2,
                          (pa - 35) / 8, (pa - pa_mean) / 2, gap)
                PULSE.pack_into(destination, (offset + pos) * PULSE.size,
                                *(max(-8., min(8., value)) for value in values),
                                (t - first) / max(1, last - first), float(pos > 0))
                preceding_toa[i], preceding_id[i] = t, pid
                seen[i] += 1
                s = int(ar[ast])
                if not 0 <= s < 3:
                    raise ValueError(f'Invalid state truth: {pid}')
                truth[3 * i + s] += 1
    if any(seen[i] != total for i, _, total, *_ in sequences.values()):
        raise ValueError('Raw pulse members and sequence counts differ')
    return truth, max(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    add_repository_paths(root)
    prepare(root, args.output_dir.resolve())


if __name__ == '__main__':
    main()
