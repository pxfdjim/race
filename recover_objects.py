#!/usr/bin/env python3
"""Replay object recovery on PRI sequences, then optionally export candidate CSVs."""

import argparse
import csv
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, zip_longest
from pathlib import Path


SECOND = 1_000_000_000
MODES = ('refill', 'multiplatform', 'standalone')


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def sequence_key(row):
    return tuple(row[k] for k in ('scenario_id', 'data_version', 'platform_id', 'sequence_id'))


@dataclass(slots=True)
class Sequence:
    key: tuple
    group: str
    object_id: str
    count: int
    time: int
    rf: float
    pw: float
    pri: float
    model: str
    bearing: float

    @property
    def group_key(self):
        return self.key[:3] + (self.group,)


def load_sequences(intermediates, split):
    fusion = intermediates / 'fused_b_assisted_guarded' / split
    pri = intermediates / 'pri' / split
    links = {sequence_key(row): (row['local_cluster_id'], row['object_id'])
             for row in records(fusion / 'sequence_object_links.csv')}
    result = []
    for row in records(pri / 'pri_sequences.csv'):
        key = sequence_key(row)
        group, object_id = links.pop(key)
        midpoint = (int(row['start_toa_ns']) + int(row['end_toa_ns'])) // 2
        result.append(Sequence(key, group, object_id, int(row['pulse_count']), midpoint,
                               float(row['rf_mean']), float(row['pw_mean']),
                               float(row['pri_center_us']), row['pri_model'],
                               float(row['doa_mean_degrees'])))
    if links:
        raise ValueError('Some fusion sequence IDs are absent from the PRI table')
    return result


def load_events(intermediates, split):
    path = intermediates / 'fused' / split / 'cross_platform_events.csv'
    for row in records(path):
        platforms = row['platform_ids'].split('|')
        sequences = row['sequence_ids'].split('|')
        scope = (row['scenario_id'], row['data_version'])
        keys = [scope + (platform, seq) for platform, seq in zip(platforms, sequences, strict=True)]
        yield int(row['time_ns']), keys


def compatible(a, b):
    # Separate operating modes and compare only nearby times on the same sensor.
    pri_tolerance = .008 if a.model == 'stable' else .05
    bearing_delta = abs((a.bearing - b.bearing + 180.) % 360. - 180.)
    return (a.model == b.model and abs(a.rf - b.rf) <= 80.
            and abs(a.pw - b.pw) <= .30
            and abs(a.pri - b.pri) <= pri_tolerance * min(a.pri, b.pri)
            and bearing_delta <= 2. + .25 * abs(a.time - b.time) / SECOND)


def recovered_identity(seq, anchors, times):
    left = bisect_left(times, seq.time - 8 * SECOND)
    right = bisect_right(times, seq.time + 8 * SECOND)
    # One vote per object and half-second bin limits repeated pulse trains.
    votes = defaultdict(set)
    for anchor in anchors[left:right]:
        if compatible(seq, anchor):
            votes[anchor.object_id].add(anchor.time // (SECOND // 2))
    if len(votes) != 1:
        return None
    object_id, bins = next(iter(votes.items()))
    return object_id if len(bins) >= 2 else None


def supported_components(groups, events):
    key_to_group = {seq.key: key for key, items in groups.items() for seq in items}
    evidence = defaultdict(set)
    for instant, keys in events:
        present = sorted({key_to_group[key] for key in keys if key in key_to_group})
        for a, b in combinations(present, 2):
            evidence[a, b].add(instant // SECOND)
    neighbors = defaultdict(set)
    for (a, b), bins in evidence.items():
        if len(bins) >= 3:
            neighbors[a].add(b)
            neighbors[b].add(a)
    components = []
    visited = set()
    for head in sorted(neighbors):
        if head in visited:
            continue
        pending, component = [head], set()
        while pending:
            group = pending.pop()
            if group not in component:
                component.add(group)
                pending.extend(neighbors[group] - component)
        visited.update(component)
        # Every pair must agree and each member must come from a different sensor.
        if (len({g[2] for g in component}) == len(component)
                and all(neighbors[g] >= component - {g} for g in component)):
            components.append(sorted(component))
    return components


@dataclass
class Plan:
    refill: dict = field(default_factory=dict)
    multiplatform: dict = field(default_factory=dict)
    standalone: dict = field(default_factory=dict)

    def edits(self, mode):
        result = dict(self.refill)
        if mode in ('multiplatform', 'standalone'):
            result.update(self.multiplatform)
        if mode == 'standalone':
            result.update(self.standalone)
        return result


def build_plan(sequences, events):
    grouped = defaultdict(list)
    for seq in sequences:
        if seq.group:
            grouped[seq.group_key].append(seq)
    plan, candidates = Plan(), {}
    for key, items in grouped.items():
        residual = [s for s in items if not s.object_id]
        if not residual:
            continue
        anchors = sorted((s for s in items if s.object_id), key=lambda s: s.time)
        if anchors:
            times = [s.time for s in anchors]
            for seq in residual:
                object_id = recovered_identity(seq, anchors, times)
                if object_id:
                    plan.refill[seq.key] = object_id
        elif (sum(s.count for s in residual) >= 512
              and len({s.time // SECOND for s in residual}) >= 4
              and max(s.time for s in residual) - min(s.time for s in residual) >= 3 * SECOND):
            candidates[key] = residual

    used = {s.object_id for s in sequences if s.object_id}
    components = supported_components(candidates, events)
    merged = {key for component in components for key in component}
    singletons = [[key] for key in sorted(candidates.keys() - merged)]
    for i, component in enumerate(components + singletons, 1):
        object_id = f'OR{i:05d}'
        if object_id in used:
            raise ValueError(f'Recovery ID collides with an existing object: {object_id}')
        edits = plan.multiplatform if len(component) > 1 else plan.standalone
        for key in component:
            for seq in candidates[key]:
                edits[seq.key] = object_id
    return plan


def identity_metrics(cells, truth_counts):
    def pairs(n):
        return n * (n - 1) // 2

    assigned = sum(cells.values())
    total = sum(truth_counts.values())
    objects = Counter()
    for (object_id, _), count in cells.items():
        objects[object_id] += count
    same = sum(pairs(n) for n in cells.values())
    predicted_pairs = sum(pairs(n) for n in objects.values())
    true_pairs = sum(pairs(n) for n in truth_counts.values())
    return {'pulses': total, 'assigned': assigned, 'coverage': assigned / total,
            'objects': len(objects), 'pair_precision': same / predicted_pairs if predicted_pairs else 0.,
            'pair_recall': same / true_pairs if true_pairs else 0.,
            'pair_f1': 2 * same / (predicted_pairs + true_pairs) if true_pairs else 0.}


def evaluate(seqs, truth, edits):
    cells, totals = Counter(), Counter()
    recovery = defaultdict(Counter)
    base_truth = defaultdict(Counter)
    for seq in seqs:
        labels = truth[seq.key]
        if seq.count != sum(labels.values()):
            raise ValueError(f'Truth count differs for {seq.key}')
        totals.update(labels)
        oid = edits.get(seq.key, seq.object_id)
        if seq.object_id:
            base_truth[seq.object_id].update(labels)
        if oid:
            for label, count in labels.items():
                cells[oid, label] += count
        if seq.key in edits:
            recovery[oid].update(labels)
    dominant = {counts.most_common(1)[0][0] for counts in base_truth.values()}
    report = identity_metrics(cells, totals)
    report['recovered_to_existing'] = sum(sum(c.values()) for k, c in recovery.items() if k in base_truth)
    report['recovered_to_existing_agree_with_object_majority'] = sum(
        recovery[k][base_truth[k].most_common(1)[0][0]] for k in recovery.keys() & base_truth.keys())
    report['new_objects'] = [
        {'object_id': oid, 'pulses': sum(counts.values()), 'truth_counts': counts,
         'dominant_truth_already_represented': counts.most_common(1)[0][0] in dominant}
        for oid, counts in sorted(recovery.items()) if oid not in base_truth]
    return report


def validate(intermediates, sequences, events):
    truth_path = intermediates / 'group_validation/train_sequence_truth.csv'
    truth = defaultdict(Counter)
    for r in records(truth_path):
        truth[sequence_key(r)][r['target_id']] += int(r['pulse_count'])
    # Frozen upstream artifacts come from the entire train scene. Recompute only
    # the recovery decisions independently inside each disjoint time block.
    stop = max(s.time for s in sequences)
    bounds = (('earlier_60_percent', 0, int(stop * .6)), ('later_40_percent', int(stop * .6), stop + 1),
              ('full_train', 0, stop + 1))
    result = {'evaluation': 'Frozen upstream train outputs. Exact pulse-label counts for all PRI sequences. '
                            'True same-target pairs with unassigned pulses count against recall.'}
    for name, start, end in bounds:
        block = [s for s in sequences if start <= s.time < end]
        plan = build_plan(block, ((time, keys) for time, keys in events if start <= time < end))
        result[name] = {'baseline': evaluate(block, truth, {})}
        for mode in MODES:
            result[name][mode] = evaluate(block, truth, plan.edits(mode))
    return result


def export(intermediates, baseline, output, edits):
    if output.resolve() == baseline.resolve() or output.exists():
        raise ValueError('Use a new directory for candidate results')
    pulse_path = baseline / 'P1A_test_pulse_predictions.csv'
    link_path = baseline / 'P1A_test_object_track_links.csv'
    fusion_path = intermediates / 'fused_b_assisted_guarded/test/pulse_object_predictions.csv'
    output.mkdir(parents=True)
    changed = Counter()
    recovered_classes = Counter()
    old_objects = set()
    existing_links = list(records(link_path))
    with pulse_path.open(newline='') as pf, fusion_path.open(newline='') as ff:
        pulse_reader, fusion_reader = csv.reader(pf), csv.reader(ff)
        header, fusion_header = next(pulse_reader), next(fusion_reader)
        scenario, version, platform, pulse, sequence, group, oid = (
            fusion_header.index(k) for k in ('scenario_id', 'data_version', 'platform_id', 'pulse_id',
                                             'sequence_id', 'local_cluster_id', 'object_id'))
        count = blank_before = 0
        with (output / pulse_path.name).open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(header)
            for p, f in zip_longest(pulse_reader, fusion_reader):
                if p is None or f is None or p[:5] != [f[i] for i in (scenario, version, pulse, group, oid)]:
                    raise ValueError('Baseline and fusion pulse identities or memberships differ')
                count += 1
                if p[4]:
                    old_objects.add(p[4])
                else:
                    blank_before += 1
                    new_id = edits.get(tuple(f[i] for i in (scenario, version, platform, sequence)))
                    if new_id:
                        p[4] = new_id
                        changed[new_id] += 1
                        recovered_classes[p[5]] += 1
                writer.writerow(p)
    if old_objects != {r['object_id'] for r in existing_links}:
        raise ValueError('Existing links do not cover the baseline objects')
    with (output / link_path.name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(existing_links[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(existing_links)
        for oid in sorted(changed.keys() - old_objects):
            writer.writerow(dict(scenario_id=existing_links[0]['scenario_id'],
                                 data_version=existing_links[0]['data_version'],
                                 object_id=oid, track_id='', link_status='no_match'))
    return {'rows': count, 'blank_object_before': blank_before,
            'blank_object_after': blank_before - sum(changed.values()),
            'changed_pulses_by_object': changed, 'changed_pulses_by_class': recovered_classes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'test'), required=True)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--mode', choices=MODES, default='refill')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if bool(args.output) != bool(args.baseline) or (args.output and args.split != 'test'):
        parser.error('Use --baseline and --output together on the test split')
    seqs = load_sequences(args.intermediates, args.split)
    events = list(load_events(args.intermediates, args.split))
    if args.split == 'train':
        report = validate(args.intermediates, seqs, events)
    else:
        plan = build_plan(seqs, events)
        report = {name: {'sequences': len(edits), 'pulses': sum(s.count for s in seqs if s.key in edits),
                         'objects': len(set(edits.values()))}
                  for name, edits in vars(plan).items()}
        if args.output:
            report['export'] = export(args.intermediates, args.baseline, args.output, plan.edits(args.mode))
            report['export']['mode'] = args.mode
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
