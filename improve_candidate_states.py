#!/usr/bin/env python3
"""Evaluate and apply the CPI-history state model using PXF sequence memberships."""
import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from prepare_inputs import digest
from recover_objects import build_plan, load_sequences, records, sequence_key
from src.state_io import StateWindow, state_metrics


PULSE_NAME = 'P1A_test_pulse_predictions.csv'
LINK_NAME = 'P1A_test_object_track_links.csv'
MAIN_FOLDS = ('diagnostic_420_500', 'instance_holdout')


def confirmed_groups(root, split):
    return {tuple(row[name] for name in ('scenario_id', 'data_version', 'platform_id', 'local_cluster_id'))
            for row in records(root / f'output/grouped/{split}/local_groups.csv')
            if row['group_status'] == 'confirmed'}


def native_model(root, temp):
    experiment = root / 'CPI历史GRU状态实验'
    sys.path.insert(0, str(experiment))
    from prepare_gru_state import FEATURES, RECORD, STEPS, make_histories, waveform_features

    training = experiment / 'results/comparison.json'
    archive = json.loads(training.read_text())
    model = experiment / 'results/history_gru.model'
    validation = experiment / 'results/joint_class_state_validation.csv'
    selection = json.loads((root / '完整流程新提交/state_neural_comparison.json').read_text())
    if (digest(model) != archive['fit']['files_sha256'][model.name]
            or digest(validation) != selection['source_sha256'] or not selection['passes']):
        raise ValueError('GRU model or validation table differs from the selected experiment')
    profile = archive['prepared']
    if tuple(profile['features']) != FEATURES or profile['max_steps'] != STEPS:
        raise ValueError('GRU feature schema differs from the trained model')
    binary = temp / 'gru_state'
    subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', str(experiment / 'gru_state.cpp'), '-o', str(binary)],
                   check=True)
    return binary, model, validation, selection['chosen_in_selection'], profile, (RECORD, STEPS, make_histories, waveform_features)


def score(root, split, membership, partitions, trained, temp):
    binary, model, _, _, profile, helpers = trained
    record, steps, make_histories, waveform_features = helpers
    keys, windows, normalized = {}, {}, {}
    for r in records(root / f'output/pri/{split}/pri_sequences.csv'):
        seq = sequence_key(r)
        if seq not in membership:
            continue
        group, obj = membership[seq]
        category = 'object' if obj else 'local' if group else 'signature'
        source = obj or group or 'unknown'
        key = seq[:3] + (category, f'{source}~{seq[-1]}', 0)
        keys[seq] = key
        windows[key] = StateWindow(key, int(r['pulse_count']), int(r['start_toa_ns']), int(r['end_toa_ns']), ())
        normalized[seq] = tuple(max(-8., min(8., (x - m) / s)) for x, m, s in zip(
            waveform_features(r), profile['means'], profile['scales'], strict=True))
    if keys.keys() != membership.keys():
        raise ValueError('Some candidate CPI keys have no PRI record')
    max_gap = profile['maximum_gap_ns']
    histories = make_histories(partitions, keys, windows, confirmed_groups(root, split), max_gap)
    data, result = temp / 'observations.bin', temp / 'scores.bin'
    log_gap = math.log1p(max_gap / 1e6)
    with data.open('wb') as out:
        for seq, key in keys.items():
            history = histories[seq]
            values = [item for i, (k, gap) in enumerate(history)
                      for item in (*normalized[k], math.log1p(gap / 1e6) / log_gap, float(i > 0))]
            values.extend([0.] * ((steps - len(history)) * (len(normalized[seq]) + 2)))
            out.write(record.pack(len(history), windows[key].count, 0, 0, *values))
    subprocess.run([str(binary), 'predict', str(model), str(data), str(result)], check=True,
                   env=os.environ | {'OMP_NUM_THREADS': '4'})
    probabilities = dict(zip(keys, struct.iter_unpack('<3f', result.read_bytes()), strict=True))
    if any(abs(sum(row) - 1.) > 1e-5 or any(not 0 <= v <= 1 for v in row) for row in probabilities.values()):
        raise ValueError('Non-probability GRU output')
    return probabilities, {seq: len(history) for seq, history in histories.items()}


def hybrid(states, scores, history_lengths, profile):
    selected = {}
    for seq, base in states.items():
        scores_for_seq = scores[seq]
        prediction = max(range(3), key=scores_for_seq.__getitem__)
        if (base in (0, 2) and prediction in (0, 2) and prediction != base
                and scores_for_seq[prediction] >= profile['threshold']
                and (not profile['history_required'] or history_lengths[seq] > 1)):
            selected[seq] = prediction
    return selected


def shared_events(root, split, membership, fold_by_key=None):
    events = []
    for row in records(root / f'output/fused/{split}/cross_platform_events.csv'):
        if (float(row['association_cost']) > .30 or float(row['geometry_error_proxy_m']) > 400
                or float(row['crossing_angle_degrees']) < 15):
            continue
        scope = row['scenario_id'], row['data_version']
        keys = [scope + (p, s) for p, s in zip(row['platform_ids'].split('|'), row['sequence_ids'].split('|'), strict=True)
                if scope + (p, s) in membership]
        ids = {membership[k][1] for k in keys}
        if len(keys) > 1 and len(ids) == 1 and '' not in ids:
            if fold_by_key is None or len({fold_by_key[k] for k in keys}) == 1:
                events.append(tuple(keys))
    return events


def co_view_edits(events, baseline, scores, edits, floor, minimum, opposing_limit):
    adjusted = dict(edits)
    for event in events:
        trusted = Counter(baseline[k] for k in event if baseline[k] in (0, 2) and scores[k][baseline[k]] >= floor)
        if len(trusted) != 1:
            continue
        (label, votes), = trusted.items()
        if votes < minimum:
            continue
        for k in event:
            predicted = edits.get(k, baseline[k])
            if predicted in (0, 2) and predicted != label and scores[k][predicted] < opposing_limit:
                if label == baseline[k]:
                    adjusted.pop(k, None)
                else:
                    adjusted[k] = label
    return adjusted


def validate(root, trained, temp):
    _, _, archive, profile, _, _ = trained
    folds, truth, states = defaultdict(list), {}, {}
    stored = {}
    for row in records(archive):
        key = sequence_key(row)
        folds[row['fold']].append(key)
        truth[key] = tuple(int(row[f'true_state_{i}_pulses']) for i in range(3))
        states[key] = int(row['three_way_target_state'])
        stored[key] = tuple(float(row[f'history_gru_score_{i}']) for i in range(3))
    seq = {s.key: s for s in load_sequences(root / 'output', 'train') if s.key in states}
    original = {k: (s.group, s.object_id) for k, s in seq.items()}
    updated = dict(original)
    for members in folds.values():
        refilled = build_plan([seq[k] for k in members], ()).refill
        for k, obj in refilled.items():
            updated[k] = original[k][0], obj

    old_scores, old_history = score(root, 'train', original, folds, trained, temp)
    deviation = max(abs(x - y) for k, v in old_scores.items() for x, y in zip(v, stored[k], strict=True))
    if deviation > 1e-5:
        raise ValueError(f'Unable to replay saved GRU scores exactly (max difference {deviation:g})')
    new_scores, new_history = score(root, 'train', updated, folds, trained, temp)
    old_edits = hybrid(states, old_scores, old_history, profile)
    new_edits = hybrid(states, new_scores, new_history, profile)
    report = {'maximum_saved_gru_score_difference': deviation, 'same_group_training_refills': sum(
        original[k] != updated[k] for k in original), 'profile': profile, 'folds': {}}

    def summarize(members, edits):
        predictions = [tuple(float(i == edits.get(k, states[k])) for i in range(3)) for k in members]
        result = state_metrics(members, predictions, truth)
        result['changed_pulses'] = sum(sum(truth[k]) for k in members if k in edits)
        return result

    events = shared_events(root, 'train', updated, {k: f for f, members in folds.items() for k in members})
    def selection_score(edits):
        result = summarize(folds['selection'], edits)
        return result['pulse_accuracy'], result['macro_f1_present_states']

    options = {}
    for floor in (.85, .90, .95):
        for minimum in (1, 2):
            for oppose in (.70, .85, 1.):
                params = floor, minimum, oppose
                adjusted = co_view_edits(events, states, new_scores, new_edits, *params)
                options[params] = selection_score(adjusted)
    chosen = max(options, key=options.__getitem__)
    co_view = co_view_edits(events, states, new_scores, new_edits, *chosen)
    for fold, members in folds.items():
        report['folds'][fold] = {'forest': summarize(members, {}), 'original_gru_context': summarize(members, old_edits),
                                 'pxf_refilled_gru_context': summarize(members, new_edits),
                                 'pxf_gru_and_co_view': summarize(members, co_view)}
    report['co_view_profile'] = {'minimum_evidence': chosen[0], 'minimum_anchors': chosen[1], 'opposing_limit': chosen[2]}
    report['co_view_improves_both_main_folds'] = all(
        all(report['folds'][fold]['pxf_gru_and_co_view'][metric] > report['folds'][fold]['pxf_refilled_gru_context'][metric]
            for metric in ('pulse_accuracy', 'macro_f1_present_states')) for fold in MAIN_FOLDS)
    passed = all(all(v['pxf_refilled_gru_context'][metric] > v['forest'][metric]
                     for metric in ('pulse_accuracy', 'macro_f1_present_states'))
                 for v in (report['folds'][fold] for fold in MAIN_FOLDS))
    report['passes_both_main_folds'] = passed
    return report


def joined_pulses(candidate, upstream):
    with candidate.open(newline='') as a, upstream.open(newline='') as b:
        left, right = csv.reader(a), csv.reader(b)
        header, fields = next(left), next(right)
        if header != ['scenario_id', 'data_version', 'pulse_id', 'local_cluster_id',
                      'object_id', 'target_class', 'target_state']:
            raise ValueError('Unexpected candidate CSV schema')
        akey = [fields.index(f) for f in header[:3]]
        skey = [fields.index(f) for f in ('scenario_id', 'data_version', 'platform_id', 'sequence_id')]
        for row, base in zip(left, right, strict=True):
            if row[:3] != [base[i] for i in akey]:
                raise ValueError('Candidate and upstream pulse keys differ')
            yield row, tuple(base[i] for i in skey)


def test_context(candidate, upstream):
    per_seq = defaultdict(Counter)
    for row, key in joined_pulses(candidate, upstream):
        if key[-1]:
            per_seq[key][(row[3], row[4], int(row[6]))] += 1
    membership, states, mixed = {}, {}, set()
    for key, values in per_seq.items():
        best, _ = values.most_common(1)[0]
        membership[key] = best[:2]
        states[key] = best[2]
        if len(values) > 1:
            mixed.add(key)
    return membership, states, mixed


def audit_events(events, states, edits):
    count = Counter()
    for keys in events:
        count['events'] += 1
        count['before_disagreement'] += len({states[k] for k in keys}) > 1
        count['after_disagreement'] += len({edits.get(k, states[k]) for k in keys}) > 1
    return count


def export(candidate, upstream, edits, states, delta):
    if delta.exists():
        raise FileExistsError(delta)
    before = digest(candidate)
    if not edits:
        raise ValueError('No state changes selected')
    changed, kinds, after_counts = Counter(), Counter(), Counter()
    header = ['scenario_id', 'data_version', 'pulse_id', 'local_cluster_id', 'object_id', 'target_class', 'target_state']
    with tempfile.TemporaryDirectory(prefix='.pxf-state-', dir=candidate.parent) as d:
        staged, journal = Path(d) / PULSE_NAME, Path(d) / 'sequence_delta.csv'
        undo = hashlib.sha256()
        undo.update((','.join(header) + '\n').encode())
        with staged.open('w', newline='') as out:
            writer = csv.writer(out, lineterminator='\n')
            writer.writerow(header)
            for row, key in joined_pulses(candidate, upstream):
                undo.update((','.join(row) + '\n').encode())
                if key in edits:
                    state = int(row[6])
                    if state != states[key] or state == 1 or edits[key] == 1:
                        raise ValueError('State update touches an incompatible value')
                    kinds[f'{state}->{edits[key]}'] += 1
                    row[6] = str(edits[key])
                    changed[key] += 1
                after_counts[row[6]] += 1
                writer.writerow(row)
        if undo.hexdigest() != before or changed.keys() != edits.keys():
            raise ValueError('Candidate input changed or some state changes could not be applied')
        with journal.open('w', newline='') as out:
            writer = csv.writer(out, lineterminator='\n')
            writer.writerow(['scenario_id', 'data_version', 'platform_id', 'sequence_id', 'state_before', 'state_after', 'pulse_count'])
            writer.writerows([*k, states[k], edits[k], n] for k, n in sorted(changed.items()))
        after = digest(staged)
        journal.replace(delta)
        staged.replace(candidate)
    return {'previous_pulse_sha256': before, 'pulse_sha256': after, 'links_sha256': digest(candidate.with_name(LINK_NAME)),
            'rows': sum(after_counts.values()), 'changed_states': kinds, 'state_counts': after_counts,
            'changed_sequences': len(changed), 'changed_pulses': sum(changed.values()), 'delta': str(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zgy', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--delta', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='pxf-gru-') as scratch:
        temp = Path(scratch)
        trained = native_model(args.zgy, temp)
        result = validate(args.zgy, trained, temp)
        print('validation', {f: [round(result['folds'][f][m]['pulse_accuracy'], 6) for m in (
            'forest', 'original_gru_context', 'pxf_refilled_gru_context', 'pxf_gru_and_co_view')]
            for f in MAIN_FOLDS}, result['co_view_profile'], flush=True)
        candidate = args.candidate / PULSE_NAME
        upstream = args.zgy / 'output/fused_b_assisted_guarded/test/pulse_object_predictions.csv'
        membership, states, mixed = test_context(candidate, upstream)
        predictions, histories = score(args.zgy, 'test', membership, {'test': list(membership)}, trained, temp)
        edits = hybrid(states, predictions, histories, trained[3])
        events = shared_events(args.zgy, 'test', membership)
        result['test_event_consistency_before_co_view'] = audit_events(events, states, edits)
        if result['co_view_improves_both_main_folds']:
            profile = result['co_view_profile']
            edits = co_view_edits(events, states, predictions, edits, profile['minimum_evidence'],
                                  profile['minimum_anchors'], profile['opposing_limit'])
        edits = {k: p for k, p in edits.items() if k not in mixed}
        result['test_event_consistency'] = audit_events(events, states, edits)
        result['mixed_test_sequences'] = len(mixed)
        print('test', len(edits), dict(result['test_event_consistency']), flush=True)
        if args.delta and result['passes_both_main_folds']:
            result['candidate'] = export(candidate, upstream, edits, states, args.delta)
    report = json.loads(args.report.read_text()) if args.report.exists() else {}
    report['pxf_state_history'] = result
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
