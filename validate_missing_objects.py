#!/usr/bin/env python3
"""Evaluate the frozen residual-object rules after hiding entire object IDs."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from recover_objects import SECOND, build_plan, evaluate, load_events, load_sequences, records, sequence_key
from recover_objects_v2 import assess_fragments, b_fragment_scores, joint_fragments
from src.dataio import InputConfig


def assignments(sequences, truth):
    counts = defaultdict(Counter)
    for s in sequences:
        if s.object_id:
            counts[s.object_id].update(truth[s.key])
    return {obj: labels.most_common(1)[0][0] for obj, labels in counts.items()}


def exact_b_support(fragments, used):
    assess_fragments(fragments, used)
    by_track = defaultdict(list)
    for row in fragments:
        if row['clear'] and not row['track_already_used']:
            by_track[row['track']].append(row)
    return {k: [key for r in rows for key in r['members']]
            for k, rows in by_track.items() if len(rows) >= 2
            and len({r['group'][2] for r in rows}) == len(rows)}


def guarded_single_platform(fragments, used, by_key):
    """Retain a single sustained fragment if its bearing has a clear alternative gap."""
    by_track = defaultdict(list)
    for row in fragments:
        margin = row['margin_degrees']
        if (row['track'] in used or row['pulses'] < 4096 or row['median_degrees'] > .5
                or margin is None or margin < 1. or row['seconds'] < 20):
            continue
        bins = sorted({by_key[key].time // SECOND for key in row['members']})
        midpoint = bins[len(bins) // 2]
        halves = [defaultdict(list), defaultdict(list)]
        members = []
        for key in row['members']:
            s = by_key[key]
            sec = s.time // SECOND
            error = row['member_errors'].get(key)
            good = error is not None and error <= 1.
            halves[int(sec >= midpoint)][sec].append((s.count, good))
            if good:
                members.append(key)
        supported = [sum(sum(n for n, good in waveforms if good) >= .9 * sum(n for n, _ in waveforms)
                         for waveforms in half.values()) for half in halves]
        if min(supported) >= 10 and sum(by_key[k].count for k in members) >= .95 * row['pulses']:
            by_track[row['track']].append(members)
    return {track: groups[0] for track, groups in by_track.items() if len(groups) == 1}


def group_by_id(edits):
    groups = defaultdict(list)
    for key, oid in edits.items():
        groups[oid].append(key)
    return groups


def measure(groups, truth, target, target_total, available_truth):
    rows = []
    for group_id, keys in sorted(groups.items()):
        counts = Counter()
        for key in keys:
            counts.update(truth[key])
        total = sum(counts.values())
        n = counts[target]
        dominant = counts.most_common(1)[0][0]
        rows.append({'id': group_id, 'pulses': total, 'truth_counts': dict(counts),
                     'platforms': sorted({key[2] for key in keys}), 'target_pulses': n,
                     'precision_for_withheld_target': n / total,
                     'recall_for_withheld_target': n / target_total,
                     'recall_for_available_target': n / available_truth if available_truth else 0.,
                     'dominant_truth': dominant})
    top = max((r for r in rows if r['target_pulses']),
              key=lambda r: r['target_pulses'] / (target_total + r['pulses'] - r['target_pulses']), default=None)
    return {'candidate_count': len(rows), 'best': top, 'groups': rows,
            'target_majority_fragment_count': sum(r['dominant_truth'] == target for r in rows),
            'recovered_known_target': bool(top and top['precision_for_withheld_target'] >= .95
                                           and top['recall_for_withheld_target'] >= .50)}


def clean_group_opportunity(sequences, truth, target):
    groups = defaultdict(list)
    occupied = {s.group_key for s in sequences if s.group and s.object_id}
    for s in sequences:
        if s.group and s.group_key not in occupied:
            groups[s.group_key].append(s)
    eligible = {key: items for key, items in groups.items() if sum(s.count for s in items) >= 512
                and len({s.time // SECOND for s in items}) >= 4}
    return {key: sum(truth[s.key][target] for s in items) for key, items in eligible.items()}


def run(args):
    paths = args.intermediates
    sequences = load_sequences(paths, 'train')
    events = list(load_events(paths, 'train'))
    truth = defaultdict(Counter)
    labels = defaultdict(Counter)
    for r in records(paths / 'group_validation/train_sequence_truth.csv'):
        count = int(r['pulse_count'])
        truth[sequence_key(r)][r['target_id']] += count
        labels[r['target_id']][r['target_class']] += count
    for s in sequences:
        if sum(truth[s.key].values()) != s.count:
            raise ValueError(f'Truth cache count mismatch for {s.key}')
    dominant = assignments(sequences, truth)
    links = {r['object_id']: r['track_id'] for r in records(
        paths / 'ab_matched_2d/train/object_track_links.csv') if r['track_id']}
    units = InputConfig.from_dict(json.loads((Path(__file__).parent / 'config/input_units.json').read_text()))
    b_truth = {r['track_id']: r['target_id'] for r in records(args.train_b)}
    cases = []
    for target, class_counts in sorted(labels.items()):
        removed = {oid for oid, label in dominant.items() if label == target}
        masked = [replace(s, object_id='') if s.object_id in removed else s for s in sequences]
        used = {track for oid, track in links.items() if oid not in removed}
        target_total = sum(class_counts.values())
        available_truth = sum(truth[s.key][target] for s in masked if not s.object_id)
        eligible = clean_group_opportunity(masked, truth, target)
        plan = build_plan(masked, events)
        fragments = b_fragment_scores(masked, paths, 'train', args.train_b, units)
        joint = joint_fragments(fragments, masked, used)
        modes = {'repeated_A_events': group_by_id(plan.multiplatform),
                 'single_platform_split': group_by_id(plan.multiplatform | plan.standalone),
                 'strict_B_fragment': exact_b_support(fragments, used),
                 'joint_B_waveform': {tid: [s.key for s in group['members']] for tid, group in joint.items()},
                 'guarded_single_platform': guarded_single_platform(fragments, used, {s.key: s for s in masked})}
        reports = {name: measure(groups, truth, target, target_total, available_truth)
                   for name, groups in modes.items()}
        for name, result in reports.items():
            edits = {key: f'VAL{i:03d}' for i, keys in enumerate(modes[name].values()) for key in keys}
            result['global_pair_f1'] = evaluate(masked, truth, edits)['pair_f1']
            if name in ('strict_B_fragment', 'joint_B_waveform', 'guarded_single_platform'):
                for group in result['groups']:
                    group['curve_truth_id_for_analysis'] = b_truth.get(group['id'])
        case = {'target_id': target, 'target_class': class_counts.most_common(1)[0][0],
                'removed_objects': sorted(removed), 'removed_links': {k: links[k] for k in sorted(removed & links.keys())},
                'total_target_pulses': target_total, 'unassigned_target_pulses_after_mask': available_truth,
                'eligible_clean_local_group_target_pulses': sum(eligible.values()),
                'eligible_clean_local_group_target_platforms': sorted({k[2] for k, n in eligible.items() if n}),
                'target_pulses_in_groups_with_B_candidates': sum(eligible.get(r['group'], 0) for r in fragments),
                'masked_baseline_pair_f1': evaluate(masked, truth, {})['pair_f1'], 'methods': reports}
        cases.append(case)
        print(target, case['target_class'], available_truth, sum(eligible.values()),
              {key: (r['candidate_count'], r['best']['target_pulses'] if r['best'] else 0,
                     r['recovered_known_target']) for key, r in reports.items()}, flush=True)
    count = {}
    for key in cases[0]['methods']:
        rs = [case['methods'][key] for case in cases]
        count[key] = {'cases_with_qualified_recovery': sum(r['recovered_known_target'] for r in rs),
                      'cases_with_candidates': sum(r['candidate_count'] > 0 for r in rs),
                      'emitted_candidates': sum(r['candidate_count'] for r in rs)}
    result = {'evaluation': 'One target per case. Remove all baseline objects dominated by its training '
                            'target ID and release their B links. Inference sees frozen PRI and local groups, and '
                            'public B positions only. Training truth chooses masks and scores assignments. '
                            'This simulates whole-object loss downstream of local grouping.',
              'qualified_recovery': 'One new object has >=95% precision for the withheld target and '
                                    'recovers >=50% of all of its PRI-supported pulses.',
              'case_count': len(cases), 'summary': count, 'cases': cases}
    controls = {}
    for split, seqs, path in (('train', sequences, args.train_b), ('test', None, args.test_b)):
        if path is None:
            continue
        if seqs is None:
            seqs = load_sequences(paths, split)
        lookup = {s.key: s for s in seqs}
        used = {r['track_id'] for r in records(paths / 'ab_matched_2d' / split / 'object_track_links.csv')
                if r['track_id']}
        fragments = b_fragment_scores(seqs, paths, split, path, units)
        controls[split] = {track: {'pulses': sum(lookup[k].count for k in keys),
                                    'platforms': sorted({k[2] for k in keys})}
                           for track, keys in guarded_single_platform(fragments, used, lookup).items()}
    result['guarded_single_platform_unmasked'] = controls
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', type=Path, required=True)
    parser.add_argument('--train-b', type=Path, required=True)
    parser.add_argument('--test-b', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
