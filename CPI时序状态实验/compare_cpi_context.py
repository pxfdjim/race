#!/usr/bin/env python3
"""Hold the class and tracking heads fixed while testing ordered CPI features."""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from cpi_context import CONTEXT_FEATURES, MAX_GAP_NS, NEIGHBORS, TIME_BLOCKS_SECONDS, ordered_context, time_block


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ('scenario_id', 'data_version', 'platform_id')
TEST_FOLDS = ('check_300_320', 'check_400_420', 'diagnostic_420_500', 'instance_holdout')


def conditional_ap(scores, labels):
    totals = defaultdict(lambda: [0, 0])
    positive = count = 0
    for q, (_, coarse, fine) in zip(scores, labels, strict=True):
        if coarse or fine:
            slot = totals[q]
            slot[0] += coarse
            slot[1] += fine
            positive += coarse
            count += coarse + fine
    area, tp, fp = 0., 0, 0
    if positive:
        for _, (coarse, fine) in sorted(totals.items(), reverse=True):
            tp += coarse
            fp += fine
            area += (tp / (tp + fp)) * (coarse / positive)
    return {'true_tracking_pulses': count, 'true_coarse_pulses': positive,
            'coarse_fraction': positive / count if count else None,
            'average_precision': area if positive else None}


def support_by_true_state(observations):
    counts = [[0, 0] for _ in range(3)]
    for has_neighbor, truth_counts in observations:
        for state, n in enumerate(truth_counts):
            counts[state][0] += n
            counts[state][1] += n * bool(has_neighbor)
    return {str(i): {'total_pulses': n, 'with_neighbor_pulses': supported,
                     'coverage': supported / n if n else None}
            for i, (n, supported) in enumerate(counts)}


def partition(windows, keys, truth, targets, excluded):
    result = defaultdict(list)
    for sk, wk in keys.items():
        owners = targets[wk]
        if len(owners) != 1:
            continue
        if excluded.intersection(owners):
            result['instance_holdout'].append(sk)
            continue
        tag = time_block(windows[wk])
        counts = truth[wk]
        if tag and (tag != 'development' or max(counts) >= .95 * sum(counts)):
            result[tag].append(sk)
    if not all(result[k] for k in ('development', 'selection', *TEST_FOLDS)):
        raise ValueError('A configured validation fold has no supported sequence')
    return result


def confirmed_groups(root, split):
    from compare_class_state import rows

    return {tuple(r[n] for n in (*SCOPE, 'local_cluster_id'))
            for r in rows(root / f'output/grouped/{split}/local_groups.csv') if r['group_status'] == 'confirmed'}


def thresholds(follow, coarse, truth, cutoff, reference):
    from compare_class_state import COARSE_THRESHOLDS, MAX_ACCURACY_LOSS, state_statistics
    from hierarchical_state import decide_state

    candidates = []
    for t in COARSE_THRESHOLDS:
        m = state_statistics((decide_state(a, b, cutoff, t) for a, b in zip(follow, coarse, strict=True)), truth)
        candidates.append((t, m))
    eligible = [item for item in candidates if item[1]['accuracy'] >= reference['accuracy'] - MAX_ACCURACY_LOSS
                and item[1]['by_label']['1']['precision'] >= reference['by_label']['1']['precision']
                and item[1]['by_label']['1']['predicted_pulses']]
    if eligible:
        choice = max(eligible, key=lambda e: (e[1]['by_label']['1']['f1'], e[1]['accuracy']))
    else:
        choice = max(candidates, key=lambda e: (e[1]['accuracy'], e[1]['by_label']['1']['f1']))
    return {'threshold': choice[0], 'selection_metrics': choice[1], 'guard_met': bool(eligible),
            'eligible_thresholds': len(eligible), 'tested_thresholds': len(candidates)}


def run(root, output):
    from compare_class_state import (CLASSES, SEQ, MAX_ACCURACY_LOSS, check_hash, classifier_input, class_metrics,
                                     csv_write, read_json, rows, seq_key, state_features, state_statistics,
                                     statistical_class_scores, write_json)
    from hierarchical_state import BinaryForest, decide_state
    from prepare_inputs import digest
    from src.state_model import StateForest
    from src.target_classification import WaveformClassModel

    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    before = root / '状态两阶段实验/results'
    baseline_bundle = before / 'hierarchical_state_model.json'
    check_hash(baseline_bundle, read_json(before / 'comparison.json')['model_sha256'])
    source = read_json(baseline_bundle)
    follow_head = BinaryForest.from_dict(source['follow_model'])
    old_head = BinaryForest.from_dict(source['coarse_model'])
    follow_threshold = source['follow_threshold']
    excluded = set(source['excluded_target_instances'])
    names, windows, keys, truth, targets = state_features(root, 'train', True)
    if tuple(names) != follow_head.features or follow_head.features != old_head.features:
        raise ValueError('State head features differ from the fixed reference cache')
    categories = partition(windows, keys, truth, targets, excluded)
    context = ordered_context(windows, names, confirmed_groups(root, 'train'), partition_by_time=True)
    fit_keys = [keys[k] for k in categories['development'] if truth[keys[k]][0] < max(truth[keys[k]][1:])]
    fit_labels = [int(truth[k][1] >= truth[k][2]) for k in fit_keys]
    fit_weights = [min(windows[k].count, 256) for k in fit_keys]
    print(json.dumps({'coarse_fine_fit_sequences': len(fit_keys), 'coarse_fit': sum(fit_labels),
                      'additional_features': len(CONTEXT_FEATURES)}), flush=True)
    new_head = BinaryForest(old_head.config, names + list(CONTEXT_FEATURES)).fit(
        [windows[k].features + context[k] for k in fit_keys], fit_labels, fit_weights)
    print('Fitted state head with ordered short-sequence context', flush=True)

    reference = root / '跨时片段关联实验/models'
    fit_meta = read_json(reference / 'fit_manifest.json')
    if fit_meta['fit_range_ns'] != source['fit_range_ns'] or set(fit_meta['excluded_target_instances']) != excluded:
        raise ValueError('Fixed baseline files use different validation partitions')
    for name, sha in fit_meta['files'].items():
        check_hash(reference / name, sha)
    three_way = StateForest.from_dict(read_json(reference / 'target_state_model.json')['sequence_model'])
    class_head = WaveformClassModel.from_dict(read_json(reference / 'target_class_model.json'))
    evaluation = set(k for f in ('selection', *TEST_FOLDS) for k in categories[f])
    scores = {}
    for k in evaluation:
        wk = keys[k]
        x = windows[wk].features
        scores[k] = (follow_head.score(x), old_head.score(x), new_head.score(x + context[wk]),
                     max(range(3), key=three_way.predict(x).__getitem__))

    def evaluate_vectors(name):
        seqs = categories[name]
        return zip(*(scores[k] for k in seqs)), [truth[keys[k]] for k in seqs]

    vectors, y_select = evaluate_vectors('selection')
    f_select, old_select, new_select, base_select = vectors
    guard_ref = state_statistics(base_select, y_select)
    picked = {name: thresholds(f_select, vals, y_select, follow_threshold, guard_ref)
              for name, vals in [('original', old_select), ('cpi_context', new_select)]}
    print(json.dumps({'selection': {k: {'threshold': v['threshold'], 'guard_met': v['guard_met']}
                                   for k, v in picked.items()}}), flush=True)

    cls_truth = defaultdict(Counter)
    for r in rows(root / 'output/group_validation/train_sequence_truth.csv'):
        sk = seq_key(r)
        if sk in evaluation:
            if r['target_id'] not in targets[keys[sk]]:
                raise ValueError('Type and state truth sources differ')
            cls_truth[sk][r['target_class']] += int(r['pulse_count'])
    if any(sum(cls_truth[k].values()) != windows[keys[k]].count for k in evaluation):
        raise ValueError('Class and state truth pulse counts differ')
    measurements = classifier_input(root, 'train', evaluation)
    fusion_dir = root / 'output/fused_b_assisted_guarded/train'
    links = fusion_dir / 'sequence_object_links.csv'
    check_hash(links, read_json(fusion_dir / 'fusion_run.json')['files'][links.name]['sha256'])
    memberships = {seq_key(r): r['object_id'] for r in rows(links) if seq_key(r) in evaluation}
    if memberships.keys() != evaluation:
        raise ValueError('Missing fixed fusion memberships')

    results, output_rows = {}, []
    for fold in ('selection', *TEST_FOLDS):
        seqs = categories[fold]
        vectors, y = evaluate_vectors(fold)
        f, old, new, baseline = vectors
        x0 = tuple(decide_state(p, q, follow_threshold, picked['original']['threshold']) for p, q in zip(f, old, strict=True))
        x1 = tuple(decide_state(p, q, follow_threshold, picked['cpi_context']['threshold']) for p, q in zip(f, new, strict=True))
        _, cls = statistical_class_scores(class_head, {k: measurements[k] for k in seqs}, memberships)
        results[fold] = {
            'sequences': len(seqs), 'eligible_pulses': sum(sum(v) for v in y),
            'pulses_with_any_neighbor': sum(windows[keys[k]].count for k in seqs if sum(context[keys[k]][-3:-1]) > 0),
            'conditional_original': conditional_ap(old, y), 'conditional_cpi_context': conditional_ap(new, y),
            'original_three_way': state_statistics(baseline, y),
            'original_second_stage': state_statistics(x0, y), 'cpi_context_second_stage': state_statistics(x1, y),
            'class_fixed': class_metrics(cls, cls_truth, seqs),
            'neighbor_support_by_true_state': support_by_true_state(
                (sum(context[keys[k]][-3:-1]) > 0, c) for k, c in zip(seqs, y, strict=True)),
        }
        for k, previous, updated, truth_counts in zip(seqs, x0, x1, y, strict=True):
            p, old_q, updated_q, three = scores[k]
            c = CLASSES[max(range(8), key=cls[k].__getitem__)]
            output_rows.append([fold, *k, windows[keys[k]].count, memberships[k], c, *truth_counts, three,
                                previous, updated, p, old_q, updated_q, *(int(v) for v in context[keys[k]][-3:-1])])
        r = results[fold]
        print(json.dumps({'fold': fold, 'ap_old': r['conditional_original']['average_precision'],
                          'ap_new': r['conditional_cpi_context']['average_precision'],
                          'new_coarse': r['cpi_context_second_stage']['by_label']['1']}, ensure_ascii=False), flush=True)

    def pass_fold(r):
        baseline = r['original_three_way']
        candidate = r['cpi_context_second_stage']
        return (r['conditional_cpi_context']['average_precision'] > r['conditional_original']['average_precision']
                and candidate['accuracy'] >= baseline['accuracy'] - MAX_ACCURACY_LOSS
                and candidate['by_label']['1']['precision'] >= baseline['by_label']['1']['precision']
                and candidate['by_label']['1']['recall'] > baseline['by_label']['1']['recall'])

    checks = {k: pass_fold(results[k]) for k in TEST_FOLDS if results[k]['conditional_original']['true_coarse_pulses']}
    stage = output.with_name(f'.{output.name}.partial-{os.getpid()}')
    stage.mkdir(parents=True)
    model = stage / 'cpi_temporal_state_model.json'
    write_json(model, {'format': 'ZGY_ordered_CPI_state_v1', 'follow_model': source['follow_model'],
                       'coarse_model': new_head.to_dict(), 'follow_threshold': follow_threshold,
                       'coarse_threshold': picked['cpi_context']['threshold'], 'neighbors_per_side': NEIGHBORS,
                       'maximum_neighbor_gap_ns': MAX_GAP_NS, 'development_only': True,
                       'score_interpretation': 'Uncalibrated state evidence from the two heads.'})
    csv_write(stage / 'joint_class_state_validation.csv', ('fold', *SEQ, 'pulse_count', 'object_id', 'target_class',
              'true_state_0_pulses', 'true_state_1_pulses', 'true_state_2_pulses', 'three_way_target_state',
              'original_target_state', 'cpi_context_target_state', 'fixed_tracking_score', 'original_coarse_score',
              'cpi_context_coarse_score', 'previous_cpis', 'next_cpis'), output_rows)
    manifest = {'time_blocks_seconds': TIME_BLOCKS_SECONDS, 'excluded_fit_target_instances': sorted(excluded),
                'coarse_fine_fit_counts': {'coarse': sum(fit_labels), 'fine': len(fit_labels) - sum(fit_labels)},
                'state_reference_sha256': digest(baseline_bundle), 'model_sha256': digest(model),
                'extra_features': CONTEXT_FEATURES, 'fixed_tracking_threshold': follow_threshold,
                'selection': picked, 'results': results, 'passed_by_fold': checks,
                'candidate_passed': bool(checks) and all(checks.values()) and picked['cpi_context']['guard_met'],
                'notes': ['All metrics are pulse-weighted on complete PRI-supported single-source sequences.',
                          'Added neighbor context does not cross the defined blocks. Existing 43 features retain the original full-scene offline context.',
                          '300–320 s and 400–420 s were excluded from these state fits and threshold selection. They are from the same scene.',
                          '420–500 s has been examined before and remains a diagnostic only.',
                          'The class head and memberships are frozen; type predictions never enter state features.',
                          'The no-PRI branch remains the existing native window model.'],
                'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(stage / 'comparison.json', manifest)
    stage.rename(output)


def preview(root, model_path, output):
    from compare_class_state import CLASSES, SEQ, check_hash, csv_write, read_json, rows, seq_key, state_features, write_json
    from hierarchical_state import BinaryForest, decide_state
    from prepare_inputs import digest

    if output.exists():
        raise FileExistsError(output)
    model = read_json(model_path)
    if model['format'] != 'ZGY_ordered_CPI_state_v1' or model['neighbors_per_side'] != NEIGHBORS or model[
            'maximum_neighbor_gap_ns'] != MAX_GAP_NS:
        raise ValueError('Unsupported temporal state model')
    follow = BinaryForest.from_dict(model['follow_model'])
    coarse = BinaryForest.from_dict(model['coarse_model'])
    names, windows, keys, _, _ = state_features(root, 'test', False)
    if tuple(names) != follow.features or tuple(names) + CONTEXT_FEATURES != coarse.features:
        raise ValueError('Test feature schema differs from the saved model')
    context = ordered_context(windows, names, confirmed_groups(root, 'test'), partition_by_time=False)
    cls_path = root / 'output/classified/test/sequence_class_predictions.csv'
    check_hash(cls_path, read_json(cls_path.with_name('classification_run.json'))['files'][cls_path.name]['sha256'])
    labels = {seq_key(r): r for r in rows(cls_path)}
    if labels.keys() != keys.keys():
        raise ValueError('Test class and state sequence keys differ')
    results = []
    counts = Counter()
    for sk, wk in keys.items():
        win, cls = windows[wk], labels[sk]
        if win.count != int(cls['pulse_count']) or cls['target_class'] not in CLASSES:
            raise ValueError(f'Test class/state pulse count or label differs: {sk}')
        v = context[wk]
        a, b = follow.score(win.features), coarse.score(win.features + v)
        pred = decide_state(a, b, model['follow_threshold'], model['coarse_threshold'])
        counts[str(pred)] += win.count
        results.append([*sk, win.count, cls['object_id'], cls['target_class'], pred, a, b, *(int(x) for x in v[-3:-1])])
    stage = output.with_name(f'.{output.name}.partial-{os.getpid()}')
    stage.mkdir(parents=True)
    csv_write(stage / 'test_sequence_class_state.csv', (*SEQ, 'pulse_count', 'object_id', 'target_class',
              'target_state', 'tracking_score', 'coarse_given_tracking_score', 'previous_cpis', 'next_cpis'), results)
    write_json(stage / 'preview.json', {'sequences': len(results), 'supported_pulse_counts_by_predicted_state': counts,
                                       'model_sha256': digest(model_path),
                                       'scope': 'Test sequence preview; unlabeled, excludes no-PRI records.'})
    stage.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preview-model', type=Path)
    args = p.parse_args()
    root = args.pipeline_root.resolve()
    sys.path[:0] = [str(root), str(root / '状态两阶段实验')]
    if args.preview_model:
        preview(root, args.preview_model.resolve(), args.output_dir.resolve())
    else:
        run(root, args.output_dir.resolve())


if __name__ == '__main__':
    main()
