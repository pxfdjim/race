#!/usr/bin/env python3
"""Evaluate sequence source consistency using separate TRAIN labels, never fit PRI."""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.dataio import time_to_ns


def evaluate(source, predictions, unit='us'):
    totals, assigned, models, by_sequence, last = Counter(), Counter(), Counter(), defaultdict(Counter), {}
    edges = correct_edges = rows = supported = 0
    platform_counts = defaultdict(Counter)
    with Path(source).open(encoding='utf-8-sig', newline='') as truth, \
         Path(predictions).open(encoding='utf-8-sig', newline='') as predicted:
        a, b = csv.DictReader(truth), csv.DictReader(predicted)
        if 'target_id' not in a.fieldnames or 'target_class' not in a.fieldnames:
            raise ValueError('Evaluation requires labeled training data')
        for raw, pred in itertools.zip_longest(a, b):
            if raw is None or pred is None:
                raise ValueError('Prediction/input row counts differ')
            key = tuple(raw[k] for k in ('scenario_id', 'data_version', 'platform_id', 'pulse_id'))
            if key != tuple(pred[k] for k in ('scenario_id', 'data_version', 'platform_id', 'pulse_id')):
                raise ValueError('Prediction/input key order differs')
            target, klass = raw['target_id'].strip(), raw['target_class'].strip()
            if not target or not klass:
                raise ValueError('Truth label missing: do not evaluate unlabeled test data')
            rows += 1
            totals[klass] += 1
            platform_counts[key[2]]['input'] += 1
            if not pred['sequence_id']:
                if pred['status'] != 'unknown' or pred['pri_center_ns']:
                    raise ValueError('Unassigned pulse must retain unknown status and blank PRI')
                continue
            if pred['status'] != 'sequence_supported' or pred['pri_model'] not in ('stable', 'jittered'):
                raise ValueError('Invalid supported pulse status/model')
            supported += 1
            assigned[klass] += 1
            models[klass, pred['pri_model']] += 1
            platform_counts[key[2]]['supported'] += 1
            seq = key[:3] + (pred['sequence_id'],)
            by_sequence[seq][target] += 1
            toa = time_to_ns(raw['toa'], unit)
            if seq in last:
                prior_id, prior_time, prior_target = last[seq]
                if pred['previous_pulse_id'] != prior_id or int(pred['observed_interval_ns']) != toa - prior_time:
                    raise ValueError('Stored sequence predecessor or interval is inconsistent with source TOA')
                if toa <= prior_time or int(pred['interval_multiple']) < 1:
                    raise ValueError('Nonpositive sequence interval')
                edges += 1
                correct_edges += target == prior_target
            elif pred['previous_pulse_id'] or pred['observed_interval_ns'] or pred['interval_multiple']:
                raise ValueError('First sequence pulse has a predecessor')
            last[seq] = key[-1], toa, target
    majority = sum(max(counts.values()) for counts in by_sequence.values())
    return {'input_pulses': rows, 'supported_pulses': supported, 'unknown_pulses': rows - supported,
        'supported_fraction': supported / rows if rows else None, 'sequence_count': len(by_sequence),
        'weighted_sequence_purity': majority / supported if supported else None,
        'fully_pure_sequences': sum(len(c) == 1 for c in by_sequence.values()),
        'mixed_sequences': sum(len(c) > 1 for c in by_sequence.values()),
        'majority_disagreement_pulses': supported - majority,
        'evaluated_edges': edges, 'same_source_edges': correct_edges,
        'same_source_edge_fraction': correct_edges / edges if edges else None,
        'class_coverage': {k: {'input': totals[k], 'supported': assigned[k],
             'coverage': assigned[k] / totals[k],
             'stable': models[k, 'stable'], 'jittered': models[k, 'jittered']} for k in sorted(totals)},
        'platform_coverage': {k: dict(v) | {'coverage': v['supported'] / v['input']}
                              for k, v in sorted(platform_counts.items())},
        'interpretation': 'Source identity consistency and coverage, not physical PRI ground-truth accuracy. '
                          'Training data does not supply true emission PRI or CPI IDs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth-file', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--toa-unit', choices=('ns', 'us', 'ms', 's'), default='us')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.truth_file, args.predictions, args.toa_unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
