#!/usr/bin/env python3
"""Recover unresolved jittered pulses bracketed by the same local group and object."""
import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PLATFORM = '104002'
PULSE_NAME = 'P1A_test_pulse_predictions.csv'
LINK_NAME = 'P1A_test_object_track_links.csv'


def read_pulses(raw, upstream, prediction=None, previous=None):
    previous = previous or {}
    indexed, unresolved, types = defaultdict(list), [], defaultdict(Counter)
    with raw.open(encoding='utf-8-sig') as file, upstream.open() as frozen:
        names = next(file).strip().split(',')
        indices = [names.index(k) for k in ('pulse_id', 'platform_id', 'toa', 'param1', 'param2', 'doa')]
        truth = names.index('target_id')
        cols = next(frozen).strip().split(',')
        a, g, o = [cols.index(k) for k in ('pulse_id', 'local_cluster_id', 'object_id')]

        def ingest(raw_line, source, candidate=None):
            fields = raw_line.strip().split(',')
            if candidate is not None and candidate[4]:
                types[candidate[4]][candidate[5]] += 1
            if fields[indices[1]] != PLATFORM or not 35365 <= float(fields[indices[3]]) <= 35445:
                return
            row = source.strip().split(',')
            pid, _, toa, rf, pw, doa = (fields[i] for i in indices)
            if row[a] != pid or (candidate is not None and candidate[2] != pid):
                raise ValueError('Raw and prediction pulse keys differ')
            data = (int(pid), round(float(toa) * 1000), float(rf), float(pw), float(doa), int(fields[truth] or 0))
            if row[g] and row[o]:
                indexed[row[g], row[o]].append(data)
            elif (candidate is not None and not row[g] and not row[o] and candidate[5] == 'R06'
                  and (not any(candidate[3:5]) or tuple(candidate[3:5]) == previous.get(int(pid)))):
                unresolved.append(data)

        if prediction is None:
            for line, source in zip(file, frozen, strict=True):
                ingest(line, source)
        else:
            with prediction.open(newline='') as p:
                rows = csv.reader(p)
                header = next(rows)
                if header[2:7] != ['pulse_id', 'local_cluster_id', 'object_id', 'target_class', 'target_state']:
                    raise ValueError('Unexpected prediction columns')
                for line, source, row in zip(file, frozen, rows, strict=True):
                    ingest(line, source, row)
    refs = {key: np.asarray(rows, dtype=np.float64).reshape(-1, 6) for key, rows in indexed.items()}
    return refs, np.asarray(unresolved, dtype=np.float64).reshape(-1, 6), types


def closest_witness(queries, anchors, sign):
    center = queries[:, 1] + sign * 400_000
    times = anchors[:, 1]
    left = np.searchsorted(times, center - 65_000)
    right = np.searchsorted(times, center + 65_000, 'right')
    result = {}
    for i in np.flatnonzero(right > left):
        choices = np.arange(left[i], right[i])
        nearby = anchors[choices]
        angle = abs((nearby[:, 4] - queries[i, 4] + 180) % 360 - 180)
        available = (abs(nearby[:, 2] - queries[i, 2]) < 3) & (abs(nearby[:, 3] - queries[i, 3]) < .24) & (angle < 5)
        if available.any():
            indices = choices[available]
            result[i] = indices[np.argmin(abs(times[indices] - center[i]))]
    return result


def recover(queries, refs):
    candidates = defaultdict(list)
    for key, pulses in refs.items():
        anchors = pulses[np.argsort(pulses[:, 1], kind='stable')]
        before = closest_witness(queries, anchors, -1)
        after = closest_witness(queries, anchors, 1)
        for i in before.keys() & after.keys():
            first, second = anchors[before[i]], anchors[after[i]]
            if 745_000 <= second[1] - first[1] <= 855_000:
                candidates[i].append((key, int(first[5]), int(second[5])))
    return {i: matches[0] for i, matches in candidates.items() if len(matches) == 1}, sum(
        len(matches) > 1 for matches in candidates.values())


def recover_run(queries, refs, occupied, missing):
    """Find uniquely bracketed runs of consecutive missing pulses."""
    left = np.searchsorted(queries[:, 1], queries[:, 1] + 350_000)
    right = np.searchsorted(queries[:, 1], queries[:, 1] + 450_000, 'right')
    successors = {}
    for i in np.flatnonzero(right > left):
        if i in occupied:
            continue
        possible = np.arange(left[i], right[i])
        others = queries[possible]
        da = abs((others[:, 4] - queries[i, 4] + 180) % 360 - 180)
        close = (abs(others[:, 2] - queries[i, 2]) < 3) & (abs(others[:, 3] - queries[i, 3]) < .24) & (da < 5)
        successors[i] = tuple(j for j in possible[close] if j not in occupied)
    paths = [(i,) for i in successors]
    for _ in range(missing - 1):
        paths = [p + (j,) for p in paths for j in successors.get(p[-1], ())]

    proposals = defaultdict(list)
    expected_span = (missing + 1) * 400_000
    for key, pulses in refs.items():
        anchors = pulses[np.argsort(pulses[:, 1], kind='stable')]
        before = closest_witness(queries, anchors, -1)
        after = closest_witness(queries, anchors, 1)
        for path in paths:
            i, j = path[0], path[-1]
            if i not in before or j not in after:
                continue
            first, last = anchors[before[i]], anchors[after[j]]
            if (abs(last[1] - first[1] - expected_span) <= 55_000
                    and abs(first[2] - last[2]) < 3 and abs(first[3] - last[3]) < .24):
                for member in path:
                    proposals[member].append((path, key, int(first[5]), int(last[5])))
    result = {}
    for i, options in proposals.items():
        if len(options) == 1:
            path, *choice = options[0]
            if all(len(proposals[member]) == 1 for member in path):
                result[i] = tuple(choice)
    return result


def recover_pairs(queries, refs, singles):
    return recover_run(queries, refs, singles, 2)


def masked_replay(refs, missing=1):
    samples, memberships, retained = [], [], {}
    for key, pulses in refs.items():
        if missing == 1:
            held_out = pulses[:, 0].astype('int64') % 11 == 1
        else:
            pulses = pulses[np.argsort(pulses[:, 1], kind='stable')]
            index = np.arange(len(pulses)) % 41
            held_out = (7 <= index) & (index < 7 + missing)
        samples.append(pulses[held_out])
        memberships.extend([key] * int(held_out.sum()))
        retained[key] = pulses[~held_out]
    queries = np.vstack(samples)
    if missing > 1:
        order = np.argsort(queries[:, 1], kind='stable')
        queries = queries[order]
        memberships = [memberships[i] for i in order]
    recovered, ambiguous = recover(queries, retained)
    if missing > 1:
        occupied = recovered.copy()
        for length in range(2, missing + 1):
            recovered = recover_run(queries, retained, occupied, length)
            occupied.update(recovered)
    counts = Counter(queried=len(queries), selected=len(recovered), ambiguous=ambiguous)
    for i, (choice, first, second) in recovered.items():
        counts['same_original_group'] += choice[0] == memberships[i][0]
        counts['same_original_object'] += choice[1] == memberships[i][1]
        if queries[i][5]:
            counts['both_witnesses_same_true_target'] += int(first == second == queries[i][5])
            counts['wrong_true_target_witness'] += int(first != queries[i][5] or second != queries[i][5])
    return counts


def test_edits(refs, queries, types):
    ordered = queries[np.argsort(queries[:, 1], kind='stable')]
    singles, ambiguous = recover(ordered, refs)
    pairs = recover_pairs(ordered, refs, singles)
    triples = recover_run(ordered, refs, singles | pairs, 3)
    quadruples = recover_run(ordered, refs, singles | pairs | triples, 4)
    votes = defaultdict(Counter)
    for (group, obj), rows in refs.items():
        votes[group][obj] += len(rows)
    edits = {}
    for i, ((group, obj), _, _) in (singles | pairs | triples | quadruples).items():
        prior = votes[group]
        ranked = types[obj].most_common(1)
        if (ranked and ranked[0][0] == 'R06' and len(refs[group, obj]) >= 96
                and prior[obj] >= .75 * sum(prior.values())):
            edits[int(ordered[i, 0])] = (group, obj)
    return edits, {'queries': len(queries), 'unique_single_pulse_support': len(singles),
                   'unique_two_pulse_support': len(pairs), 'unique_three_pulse_support': len(triples),
                   'unique_four_pulse_support': len(quadruples), 'ambiguous_single_pulse_support': ambiguous,
                   'accepted': len(edits)}


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_edits(directory, edits, delta, previous):
    pulse = directory / PULSE_NAME
    with (directory / LINK_NAME).open(newline='') as links:
        link_rows = list(csv.DictReader(links))
    known_objects = {row['object_id'] for row in link_rows}
    if previous.items() - edits.items() or not edits or any(obj not in known_objects for _, obj in edits.values()):
        raise ValueError('Empty or conflicting edits, or unknown object')
    before_hash = digest(pulse)
    changed, blanks, rows_seen = Counter(), 0, 0
    with tempfile.TemporaryDirectory(prefix='.r06-gap-', dir=directory.parent) as temp:
        staged, journal = Path(temp) / PULSE_NAME, Path(temp) / 'delta.csv'
        with pulse.open(newline='') as source, staged.open('w', newline='') as target, journal.open('w', newline='') as diffs:
            input_rows = csv.reader(source)
            output_rows = csv.writer(target, lineterminator='\n')
            edit_rows = csv.writer(diffs, lineterminator='\n')
            output_rows.writerow(next(input_rows))
            edit_rows.writerow(['pulse_id', 'local_cluster_id', 'object_id'])
            for row in input_rows:
                rows_seen += 1
                pid = int(row[2])
                blanks += not row[4]
                if pid in edits:
                    old_assignment = tuple(row[3:5])
                    if old_assignment != previous.get(pid, ('', '')) or row[5] != 'R06':
                        raise ValueError(f'Unexpected R06 assignment for pulse {pid}')
                    row[3], row[4] = edits[pid]
                    changed[edits[pid]] += 1
                    edit_rows.writerow([pid, row[3], row[4]])
                output_rows.writerow(row)
        if sum(changed.values()) != len(edits):
            raise ValueError('Some proposed edits were absent from candidate input')
        after_hash = digest(staged)
        journal.replace(delta)
        staged.replace(pulse)
    return {'rows': rows_seen, 'blank_before': blanks + len(previous), 'blank_after': blanks + len(previous) - len(edits),
            'by_group_object': {f'{group}/{obj}': count for (group, obj), count in sorted(changed.items())},
            'previous_pulse_sha256': before_hash, 'pulse_sha256': after_hash, 'links_sha256': digest(directory / LINK_NAME),
            'delta_file': str(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', type=Path, required=True)
    parser.add_argument('--train-a', type=Path, required=True)
    parser.add_argument('--test-a', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--delta', type=Path)
    args = parser.parse_args()
    base = args.intermediates / 'fused_b_assisted_guarded'
    previous = {}
    if args.delta and args.delta.exists():
        with args.delta.open(newline='') as existing:
            previous = {int(row['pulse_id']): (row['local_cluster_id'], row['object_id']) for row in csv.DictReader(existing)}
    train, _, _ = read_pulses(args.train_a, base / 'train/pulse_object_predictions.csv')
    train_runs = {length: masked_replay(train, missing=length) for length in range(1, 5)}
    print('TRAIN', train_runs, flush=True)
    refs, queries, types = read_pulses(args.test_a, base / 'test/pulse_object_predictions.csv', args.candidate / PULSE_NAME, previous)
    shadow = {length: masked_replay(refs, missing=length) for length in range(1, 5)}
    edits, test = test_edits(refs, queries, types)
    print('TEST', shadow, test, flush=True)
    report = json.loads(args.report.read_text()) if args.report.exists() else {}
    result = {'platform': PLATFORM, 'scope': 'Narrow-band jittered observations, bracketed by same-group and '
              'same-object raw pulses. TRAIN target_id used only to evaluate withhold-and-recover. '
              'Test shadow comparison uses frozen assignments as a consistency reference.',
              'train_masked': train_runs[1], 'train_masked_two_pulses': train_runs[2],
              'train_masked_three_pulses': train_runs[3], 'train_masked_four_pulses': train_runs[4],
              'test_masked_consistency': shadow[1], 'test_masked_two_pulse_consistency': shadow[2],
              'test_masked_three_pulse_consistency': shadow[3], 'test_masked_four_pulse_consistency': shadow[4],
              'test_blank_replay': test}
    previous_probe = report.get('r06_gap_recovery', {}).get('cross_platform_timing_probe')
    if previous_probe:
        result['cross_platform_timing_probe'] = previous_probe
    if args.delta:
        result['candidate'] = write_edits(args.candidate, edits, args.delta, previous)
    report['r06_gap_recovery'] = result
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
