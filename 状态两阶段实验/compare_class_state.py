#!/usr/bin/env python3
"""Compare independent class heads and hierarchical state on existing zgy folds."""

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from hierarchical_state import BinaryForest, ForestConfig, decide_state, joint_scores


ROOT = Path(__file__).resolve().parents[1]
SEQ = ('scenario_id', 'data_version', 'platform_id', 'sequence_id')
STATES = ('0', '1', '2')
CLASSES = tuple(f'R{i:02d}' for i in range(1, 9))
WINDOWS = {'development': (0, 300), 'selection': (320, 400), 'time_holdout': (420, 500)}
FOLLOW_THRESHOLDS = (.10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70)
COARSE_THRESHOLDS = (.15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
MAX_ACCURACY_LOSS = .005


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def csv_write(path, fields, data):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(data)


def seq_key(row):
    return tuple(row[n] for n in SEQ)


def check_hash(path, expected):
    from prepare_inputs import digest

    if digest(path) != expected:
        raise ValueError(f'Artifact differs from the saved run: {path}')


def state_features(root, split, with_truth):
    from src.state_io import read_truth, read_windows, window_key

    directory = root / f'output/calibrated_sequence_state_features/{split}'
    meta = read_json(directory / 'state_features_run.json')
    if meta['labels_used_in_features']:
        raise ValueError('Cannot use state features derived from labels')
    feature_path = directory / 'window_features.csv'
    check_hash(feature_path, meta['files'][feature_path.name]['sha256'])
    windows = read_windows(feature_path, meta['features'])
    state_keys = {}
    for row in rows(feature_path):
        k = seq_key(row)
        if k in state_keys:
            raise ValueError(f'Duplicate state sequence: {k}')
        state_keys[k] = window_key(row)
    if set(state_keys.values()) != windows.keys():
        raise ValueError('State cache has inconsistent window coverage')
    truth, targets = None, None
    if with_truth:
        path = directory / 'window_truth.csv'
        check_hash(path, meta['files'][path.name]['sha256'])
        truth, targets = read_truth(path, windows)
    return meta['features'], windows, state_keys, truth, targets


def classifier_input(root, split, chosen):
    path = root / f'output/pri/{split}/pri_sequences.csv'
    run = read_json(path.with_name('pri_run.json'))
    check_hash(path, run['files'][path.name]['sha256'])
    result = {seq_key(r): r for r in rows(path) if chosen is None or seq_key(r) in chosen}
    if chosen is not None and result.keys() != set(chosen):
        raise ValueError('Missing waveform features for a state sequence')
    return result


def neural_class_scores(root, records):
    """Use the previously fitted CPI MLP with its original sequence inputs."""
    native = root / '类别网络实验'
    model = native / 'results/sequence_mlp.model'
    with tempfile.TemporaryDirectory(prefix='state-class-mlp-') as tmp:
        tmp = Path(tmp)
        exe, data, scores = (tmp / n for n in ('predict_mlp', 'input.bin', 'scores.bin'))
        subprocess.run(['g++', '-std=c++17', '-O3', '-fopenmp', str(native / 'class_mlp.cpp'), '-o', str(exe)], check=True)
        pack = struct.Struct('<9f').pack
        with data.open('wb') as output:
            for idx, row in enumerate(records.values()):
                pri = float(row['pri_center_us'])
                width = (float(row['pri_p95_us']) - float(row['pri_p05_us'])) / pri
                output.write(pack(float(row['rf_mean']), float(row['pw_mean']), pri, width,
                                  {'stable': 1, 'jittered': 2}[row['pri_model']], 0, 1, idx, 0))
        subprocess.run([str(exe), 'predict', str(model), str(data), str(scores)], check=True,
                       env=os.environ | {'OMP_NUM_THREADS': '8'})
        return dict(zip(records, struct.iter_unpack('<8f', scores.read_bytes()), strict=True))


def statistical_class_scores(model, observations, memberships):
    from src.target_classification import combine, pool_objects

    independent = {k: model.predict(float(r['rf_mean']), float(r['pw_mean']), float(r['pri_center_us']), r['pri_model'])
                   for k, r in observations.items()}
    pooled = pool_objects(((k[:2] + (memberships[k],), (int(r['start_toa_ns']) + int(r['end_toa_ns'])) // 2,
                            independent[k], int(r['pulse_count'])) for k, r in observations.items()), model.config)
    combined = {k: combine(p, pooled.get(k[:2] + (memberships[k],)), model.config)[0]
                for k, p in independent.items()}
    return independent, combined


def folds(windows, keys, targets, exclude, truth):
    result = defaultdict(list)
    for k, window_key in keys.items():
        labels = targets[window_key]
        if len(labels) != 1:
            continue
        if exclude.intersection(labels):
            result['instance_holdout'].append(k)
            continue
        w = windows[window_key]
        for name, (a, b) in WINDOWS.items():
            if w.first < a * 1e9 or w.last > b * 1e9 or (w.last == b * 1e9 and name != 'time_holdout'):
                continue
            y = truth[window_key]
            if name != 'development' or max(y) >= .95 * sum(y):
                result[name].append(k)
            break
    if not all(result[n] for n in (*WINDOWS, 'instance_holdout')):
        raise ValueError('One or more class/state folds are empty')
    return result


def fit_hierarchy(names, windows, keys, truth, development):
    lookup = [keys[k] for k in development]
    x = [windows[k].features for k in lookup]
    y = [max(range(3), key=truth[k].__getitem__) for k in lookup]
    weights = [min(256, windows[k].count) for k in lookup]
    follow = BinaryForest(ForestConfig(), names).fit(x, [int(s > 0) for s in y], weights)
    print('Fitted search/tracking head', flush=True)
    track = [i for i, state in enumerate(y) if state > 0]
    coarse = BinaryForest(ForestConfig(seed=20261119), names).fit(
        [x[i] for i in track], [int(y[i] == 1) for i in track], [weights[i] for i in track])
    print('Fitted coarse/fine head', flush=True)
    return follow, coarse, dict(Counter(str(y_) for y_ in y))


def summarize(confusion, order):
    n = sum(map(sum, confusion))
    results = {}
    for i, name in enumerate(order):
        support = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        true_positive = confusion[i][i]
        results[name] = {'support_pulses': support, 'predicted_pulses': predicted,
                         'precision': true_positive / predicted if predicted else 0.,
                         'recall': true_positive / support if support else 0.,
                         'f1': 2 * true_positive / (support + predicted) if support + predicted else 0.}
    f1 = [r['f1'] for r in results.values() if r['support_pulses']]
    return {'pulses': n, 'accuracy': sum(c[i] for i, c in enumerate(confusion)) / n if n else None,
            'macro_f1': sum(f1) / len(f1) if f1 else None, 'by_label': results, 'confusion': confusion}


def state_statistics(predicted, truth):
    confusion = [[0] * 3 for _ in range(3)]
    for answer, counts in zip(predicted, truth, strict=True):
        for i, n in enumerate(counts):
            confusion[i][answer] += n
    tracking = [[confusion[0][0], sum(confusion[0][1:])],
                [confusion[1][0] + confusion[2][0], sum(sum(row[1:]) for row in confusion[1:])]]
    return summarize(confusion, STATES) | {'tracking_binary': summarize(tracking, ('search', 'tracking'))}


def oracle_tracking_stage(scores, truth, coarse_threshold):
    cm = [[0, 0], [0, 0]]
    for (_, coarse), counts in zip(scores, truth, strict=True):
        col = 0 if coarse >= coarse_threshold else 1
        cm[0][col] += counts[1]
        cm[1][col] += counts[2]
    return summarize(cm, ('1', '2'))


def coarse_routing(hierarchy, oracle):
    routed = hierarchy['confusion'][1]
    return {'true_coarse_pulses': sum(routed), 'coarse_sent_to_search': routed[0],
            'coarse_sent_to_tracking': routed[1] + routed[2], 'coarse_correct_after_stage_two': routed[1],
            'coarse_marked_by_stage_two_given_true_tracking': oracle['confusion'][0][0]}


def tune(scores, truth, base):
    """Select thresholds only on the designated earlier selection partition."""
    acceptable, fallback = [], []
    for f in FOLLOW_THRESHOLDS:
        for c in COARSE_THRESHOLDS:
            metrics = state_statistics((decide_state(a, b, f, c) for a, b in scores), truth)
            coarse = metrics['by_label']['1']
            item = {'follow_threshold': f, 'coarse_threshold': c, 'metrics': metrics}
            fallback.append(item)
            if (metrics['accuracy'] >= base['accuracy'] - MAX_ACCURACY_LOSS
                    and coarse['precision'] >= base['by_label']['1']['precision']
                    and coarse['predicted_pulses']):
                acceptable.append(item)
    if acceptable:
        selected = max(acceptable, key=lambda x: (x['metrics']['by_label']['1']['f1'], x['metrics']['accuracy']))
    else:
        selected = max(fallback, key=lambda x: (x['metrics']['accuracy'], x['metrics']['by_label']['1']['f1']))
    return selected | {'guard_met_on_selection': bool(acceptable), 'tried': len(fallback),
                       'eligible': len(acceptable), 'max_selection_accuracy_loss': MAX_ACCURACY_LOSS}


def class_metrics(scores, truth, keys):
    confusion = [[0] * 8 for _ in CLASSES]
    for k in keys:
        pred = max(range(8), key=scores[k].__getitem__)
        for label, n in truth[k].items():
            confusion[CLASSES.index(label)][pred] += n
    return summarize(confusion, CLASSES)


def run(root, output):
    from prepare_inputs import digest
    from src.state_model import StateForest
    from src.target_classification import WaveformClassModel

    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    reference = root / '跨时片段关联实验/models'
    ref = read_json(reference / 'fit_manifest.json')
    if ref['fit_range_ns'] != [0, 300_000_000_000] or not ref['fit_end_exclusive']:
        raise ValueError('Comparison requires the existing development-only reference models')
    for n, checksum in ref['files'].items():
        check_hash(reference / n, checksum)
    class_model = WaveformClassModel.from_dict(read_json(reference / 'target_class_model.json'))
    baseline_state = StateForest.from_dict(read_json(reference / 'target_state_model.json')['sequence_model'])
    names, windows, keys, state_truth, targets = state_features(root, 'train', True)
    if names != baseline_state.features:
        raise ValueError('Baseline and hierarchy need the same feature cache')
    exclude = set(ref['excluded_target_instances'])
    partitions = folds(windows, keys, targets, exclude, state_truth)
    follow, coarse, fit_counts = fit_hierarchy(names, windows, keys, state_truth, partitions['development'])
    views = ('selection', 'time_holdout', 'instance_holdout')
    selected_keys = set(k for view in views for k in partitions[view])
    q = {k: (follow.score(windows[keys[k]].features), coarse.score(windows[keys[k]].features)) for k in selected_keys}
    base = {k: baseline_state.predict(windows[keys[k]].features) for k in selected_keys}
    base_class = {k: max(range(3), key=s.__getitem__) for k, s in base.items()}
    selection_keys = partitions['selection']
    reference_selection = state_statistics((base_class[k] for k in selection_keys), (state_truth[keys[k]] for k in selection_keys))
    selected = tune([q[k] for k in selection_keys], [state_truth[keys[k]] for k in selection_keys], reference_selection)
    threshold1, threshold2 = (selected[n] for n in ('follow_threshold', 'coarse_threshold'))
    print(json.dumps({'validation_follow_threshold': threshold1, 'validation_coarse_threshold': threshold2,
                      'selection_guard_met': selected['guard_met_on_selection']}), flush=True)

    labels_path = root / 'output/group_validation/train_sequence_truth.csv'
    class_truth = defaultdict(Counter)
    for r in rows(labels_path):
        k = seq_key(r)
        if k in selected_keys:
            if r['target_id'] not in targets[keys[k]]:
                raise ValueError(f'Class/state identity truth differs: {k}')
            class_truth[k][r['target_class']] += int(r['pulse_count'])
    measurements = classifier_input(root, 'train', selected_keys)
    class_links = {seq_key(r): r['object_id'] for r in rows(root / 'output/fused_b_assisted_guarded/train/sequence_object_links.csv')
                   if seq_key(r) in selected_keys}
    if class_links.keys() != selected_keys or any(sum(class_truth[k].values()) != windows[keys[k]].count for k in selected_keys):
        raise ValueError('Sequence class, state and membership inputs disagree')
    mlp_scores = neural_class_scores(root, measurements)
    output_rows = []
    results = {}
    for view in views:
        kk = partitions[view]
        truth_counts = [state_truth[keys[k]] for k in kk]
        new_state = {k: decide_state(*q[k], threshold1, threshold2) for k in kk}
        ind, contextual = statistical_class_scores(class_model, {k: measurements[k] for k in kk}, class_links)
        before = state_statistics((base_class[k] for k in kk), truth_counts)
        after = state_statistics((new_state[k] for k in kk), truth_counts)
        oracle = oracle_tracking_stage((q[k] for k in kk), truth_counts, threshold2)
        results[view] = {'state_three_way': before, 'state_hierarchy': after,
                         'oracle_true_tracking_only': oracle, 'coarse_routing': coarse_routing(after, oracle),
                         'class_statistical_waveform': class_metrics(ind, class_truth, kk),
                         'class_statistical_context': class_metrics(contextual, class_truth, kk),
                         'class_sequence_mlp': class_metrics(mlp_scores, class_truth, kk)}
        for k in kk:
            ct = class_truth[k]
            output_rows.append([view, *k, windows[keys[k]].count, class_links[k],
                                max(ct, key=ct.__getitem__), CLASSES[max(range(8), key=contextual[k].__getitem__)],
                                CLASSES[max(range(8), key=mlp_scores[k].__getitem__)], *state_truth[keys[k]],
                                base_class[k], new_state[k], *q[k], *joint_scores(*q[k])])
        print(json.dumps({'split': view, 'three_way_accuracy': before['accuracy'], 'hierarchy_accuracy': after['accuracy'],
                          'three_way_coarse': before['by_label']['1'], 'hierarchy_coarse': after['by_label']['1']}), flush=True)

    def passes(view):
        b, a = results[view]['state_three_way'], results[view]['state_hierarchy']
        return (a['accuracy'] >= b['accuracy'] - MAX_ACCURACY_LOSS and
                a['by_label']['1']['precision'] >= b['by_label']['1']['precision'] and
                a['by_label']['1']['recall'] > b['by_label']['1']['recall'])

    stage = output.with_name(f'.{output.name}.partial-{os.getpid()}')
    stage.mkdir(parents=True)
    write_json(stage / 'hierarchical_state_model.json', {
        'format': 'ZGY_hierarchical_state_v1', 'follow_model': follow.to_dict(), 'coarse_model': coarse.to_dict(),
        'follow_threshold': threshold1, 'coarse_threshold': threshold2,
        'fit_range_ns': ref['fit_range_ns'], 'excluded_target_instances': sorted(exclude),
        'score_interpretation': 'Evidence from two supervised heads, not calibrated probabilities.',
        'unknown_PRI_policy': 'Use the existing window-state branch.'})
    csv_write(stage / 'joint_validation_sequences.csv',
              ('split', *SEQ, 'pulse_count', 'object_id', 'target_class_truth', 'target_class_statistical',
               'target_class_mlp', 'state_0_truth_pulses', 'state_1_truth_pulses', 'state_2_truth_pulses',
               'baseline_target_state', 'hierarchical_target_state', 'tracking_score', 'coarse_given_tracking_score',
               'evidence_0', 'evidence_1', 'evidence_2'), output_rows)
    manifest = {'scope': 'PRI-supported single-source records with two separately supervised targets.',
                'original_time_windows_seconds': WINDOWS, 'excluded_target_instances': sorted(exclude),
                'fit_state_pure_sequence_counts': fit_counts, 'selection': selected, 'results': results,
                'candidate_acceptance_by_fold': {view: passes(view) for view in views[1:]},
                'candidate_passed': selected['guard_met_on_selection'] and all(passes(view) for view in views[1:]),
                'features': names, 'state_uses_predicted_class': False,
                'notes': ['The same frozen full-scene memberships supply both state models.',
                          'The statistical class reference uses development-only weights; the neural branch reuses the prior development CPI MLP.',
                          'No labels from the later or unseen-instance folds enter either state fit or threshold selection.',
                          'These slices are within one known scene; scores do not represent an online result.'],
                'model_sha256': digest(stage / 'hierarchical_state_model.json'),
                'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(stage / 'comparison.json', manifest)
    stage.rename(output)
    print(json.dumps({'candidate_passed': manifest['candidate_passed'], 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.pipeline_root.resolve()
    sys.path.insert(0, str(root))
    run(root, args.output_dir.resolve())


if __name__ == '__main__':
    main()
