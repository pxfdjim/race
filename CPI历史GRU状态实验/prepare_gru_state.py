"""Assemble observation-only CPI histories on frozen state comparison folds."""

import argparse
import math
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from struct import Struct
from tempfile import TemporaryDirectory


FOLDS = ('development', 'selection', 'check_300_320', 'check_400_420',
         'diagnostic_420_500', 'instance_holdout')
FEATURES = ('mean_rf', 'mean_pw', 'mean_pa', 'log_pulse_count', 'log_span_seconds', 'log_pri_us',
            'relative_pri_width', 'direct_interval_fraction', 'log_observed_density', 'jittered',
            'max_relative_interval_deviation', 'log_gap_to_previous_cpi', 'previous_cpi_present')
STEPS = 4
RECORD = Struct(f'<4I{STEPS * len(FEATURES)}f')


def add_paths(root):
    sys.path[:0] = [str(root), *(str(root / n) for n in ('状态两阶段实验', 'CPI时序状态实验', '组内脉冲状态实验'))]


def waveform_features(row):
    period = float(row['pri_center_ns'])
    span = max(1, int(row['end_toa_ns']) - int(row['start_toa_ns']))
    n = int(row['pulse_count'])
    width = (float(row['pri_p95_us']) - float(row['pri_p05_us'])) / (period / 1000)
    if row['pri_model'] not in ('stable', 'jittered'):
        raise ValueError('Unsupported PRI model for a short sequence')
    return (float(row['rf_mean']), float(row['pw_mean']), float(row['pa_mean']), math.log(n),
            math.log(span / 1e9), math.log(period / 1000), width, float(row['direct_interval_fraction']),
            math.log1p((n - 1) * period / span), float(row['pri_model'] == 'jittered'),
            float(row['max_relative_interval_deviation']))


def make_histories(partitions, keys, windows, confirmed, maximum_gap_ns):
    """Build one bounded, past-only window for every center CPI."""
    by_group = defaultdict(list)
    result = {}
    for fold, members in partitions.items():
        for seq in members:
            wkey = keys[seq]
            kind, source_bin = wkey[3:5]
            source, sep, suffix = source_bin.rpartition('~')
            if not sep or not source or suffix != seq[-1]:
                raise ValueError(f'State window has a different CPI identity: {seq}')
            scope = seq[:3]
            if kind == 'object' or (kind == 'local' and scope + (source,) in confirmed):
                by_group[fold, scope, kind, source].append(seq)
            else:
                result[seq] = ((seq, 0),)
    for members in by_group.values():
        members.sort(key=lambda s: (windows[keys[s]].first, windows[keys[s]].last, s))
        past = []
        for seq in members:
            current = windows[keys[seq]]
            if past:
                gap = current.first - windows[keys[past[-1][0]]].last
                if not 0 <= gap <= maximum_gap_ns:
                    past.clear()
            past.append((seq, gap if past else 0))
            last = past[-STEPS:]
            result[seq] = ((last[0][0], 0), *last[1:])
            if len(past) == STEPS:
                past.pop(0)
    if len(result) != sum(map(len, partitions.values())):
        raise ValueError('History assembly dropped or duplicated a center CPI')
    return result


def prepare(root, output):
    from compare_class_state import (SEQ, check_hash, classifier_input, csv_write, read_json, rows, seq_key,
                                     state_features, state_statistics, write_json)
    from compare_cpi_context import confirmed_groups, partition
    from cpi_context import MAX_GAP_NS, TIME_BLOCKS_SECONDS
    from prepare_inputs import digest
    from prepare_raw_state import summary_scaling

    if output.exists():
        raise FileExistsError(output)
    source = root / '组内脉冲状态实验/results'
    reference_meta = read_json(source / 'comparison.json')
    reference_path = source / 'joint_class_state_validation.csv'
    check_hash(reference_path, reference_meta['joint_predictions_sha256'])
    reference = {seq_key(r): r for r in rows(reference_path)}
    _, windows, keys, labels, owners = state_features(root, 'train', True)
    excluded = set(reference_meta['prepared_summary']['excluded_fit_target_instances'])
    partitions = partition(windows, keys, labels, owners, excluded)
    if set(chain.from_iterable(partitions[f] for f in FOLDS[1:])) != reference.keys():
        raise ValueError('Saved comparison and new experiment use different folds')
    for fold in FOLDS[1:]:
        samples = partitions[fold]
        targets = [labels[keys[k]] for k in samples]
        if any(reference[k]['fold'] != fold or tuple(int(reference[k][f'true_state_{i}_pulses']) for i in range(3)) != y
               for k, y in zip(samples, targets, strict=True)):
            raise ValueError(f'Saved sequence truth differs for {fold}')
        before = state_statistics((int(reference[k]['three_way_target_state']) for k in samples), targets)
        if before != reference_meta['results'][fold]['current_state']:
            raise ValueError(f'Baseline confusion differs for {fold}')
    histories = make_histories(partitions, keys, windows, confirmed_groups(root, 'train'), MAX_GAP_NS)
    observations = classifier_input(root, 'train', histories.keys())
    features = {}
    for k, row in observations.items():
        w = windows[keys[k]]
        if (w.count, w.first, w.last) != tuple(int(row[f]) for f in ('pulse_count', 'start_toa_ns', 'end_toa_ns')):
            raise ValueError(f'PRI summary and state window have different boundaries: {k}')
        features[k] = waveform_features(row)
    means, scales = summary_scaling([features[k] for k in partitions['development']])
    normalized = {k: tuple(max(-8., min(8., (v - m) / scale))
                  for v, m, scale in zip(x, means, scales, strict=True)) for k, x in features.items()}
    maximum_gap_log = math.log1p(MAX_GAP_NS / 1e6)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gru-state-data-', dir=output.parent) as directory:
        stage = Path(directory) / 'prepared'
        stage.mkdir()
        counts = {}
        for fold in FOLDS:
            samples = partitions[fold]
            with (stage / f'{fold}.bin').open('wb') as stream:
                for k in samples:
                    history = histories[k]
                    x = []
                    for i, (seq, gap) in enumerate(history):
                        x.extend((*normalized[seq], math.log1p(gap / 1e6) / maximum_gap_log, float(i > 0)))
                    x.extend([0.] * ((STEPS - len(history)) * len(FEATURES)))
                    stream.write(RECORD.pack(len(history), *labels[keys[k]], *x))
            counts[fold] = {'sequences': len(samples), 'pulses': sum(windows[keys[k]].count for k in samples),
                            'sequences_with_history': sum(len(histories[k]) > 1 for k in samples)}
        fields = ('fold', *SEQ, 'object_id', 'target_class', 'pulse_count', 'three_way_target_state',
                  'true_state_0_pulses', 'true_state_1_pulses', 'true_state_2_pulses')
        csv_write(stage / 'evaluation_index.csv', (*fields, 'history_cpis'),
                  ([*(reference[k][f] for f in fields), len(histories[k]) - 1]
                   for fold in FOLDS[1:] for k in partitions[fold]))
        write_json(stage / 'prepared.json', {'features': FEATURES, 'means': means, 'scales': scales,
            'folds': counts, 'maximum_gap_ns': MAX_GAP_NS, 'max_steps': STEPS,
            'time_blocks_seconds': TIME_BLOCKS_SECONDS, 'excluded_fit_instances': sorted(excluded),
            'reference_predictions_sha256': digest(reference_path),
            'files_sha256': {p.name: digest(p) for p in stage.iterdir()}})
        stage.rename(output)
    print({'prepared_folds': counts}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    add_paths(root)
    prepare(root, args.output_dir.resolve())


if __name__ == '__main__':
    main()
