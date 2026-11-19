"""Compare the saved state networks and export joint sequence predictions."""

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from fit_raw_state import ARMS, digest
from prepare_raw_state import FOLD_NAMES, add_repository_paths


MULTIPLIERS = (.25, .35, .5, .65, .8, 1., 1.2, 1.5, 2., 2.5, 3., 4.)
ACCEPTANCE_FOLDS = ('diagnostic_420_500', 'instance_holdout')


def decide(scores, multiplier):
    return max(range(3), key=(scores[0], scores[1] * multiplier, scores[2]).__getitem__)


def load_scores(path, count):
    scores = list(struct.iter_unpack('<3f', path.read_bytes()))
    if len(scores) != count or any(not all(math.isfinite(p) and 0 <= p <= 1 for p in row)
                                   or abs(sum(row) - 1) > 1e-5 for row in scores):
        raise ValueError(f'Invalid three-state scores: {path}')
    return scores


def select_multiplier(scores, truth, baseline):
    from compare_class_state import MAX_ACCURACY_LOSS, state_statistics

    candidates = [(m, state_statistics((decide(row, m) for row in scores), truth)) for m in MULTIPLIERS]
    eligible = [(m, r) for m, r in candidates
                if r['accuracy'] >= baseline['accuracy'] - MAX_ACCURACY_LOSS
                and r['by_label']['1']['precision'] >= baseline['by_label']['1']['precision']
                and r['by_label']['1']['predicted_pulses']]
    if eligible:
        picked = max(eligible, key=lambda pair: (pair[1]['by_label']['1']['f1'], pair[1]['accuracy']))
    else:
        picked = max(candidates, key=lambda pair: (pair[1]['accuracy'], pair[1]['by_label']['1']['f1']))
    return {'multiplier': picked[0], 'selection_metrics': picked[1], 'guard_met': bool(eligible),
            'grid_metrics': [{'multiplier': m, 'accuracy': r['accuracy'], 'coarse': r['by_label']['1']}
                             for m, r in candidates]}


def run(root, prepared, fitted, output):
    from compare_class_state import MAX_ACCURACY_LOSS, SEQ, read_json, rows, state_statistics, write_json
    from compare_cpi_context import conditional_ap

    if output.exists():
        raise FileExistsError(output)
    meta_path = prepared / 'prepared.json'
    meta, fit_meta = read_json(meta_path), read_json(fitted / 'fit.json')
    if digest(meta_path) != fit_meta['prepared_manifest_sha256']:
        raise ValueError('Fits used a different prepared test set')
    index = prepared / 'evaluation_index.csv'
    if digest(index) != meta['prepared_files_sha256'][index.name]:
        raise ValueError('Model and evaluation row order differ')
    for name, sha in fit_meta['files_sha256'].items():
        if digest(fitted / name) != sha:
            raise ValueError(f'Fit artifact changed: {name}')
    original = read_json(root / 'CPI时序状态实验/results/comparison.json')['results']
    by_fold, labels, scores = defaultdict(list), {}, {}
    for row in rows(index):
        by_fold[row['fold']].append(row)
    for fold in FOLD_NAMES[1:]:
        entries = by_fold[fold]
        n = len(entries)
        if n != meta['fold_sequences'][fold]:
            raise ValueError(f'Alignment changed for {fold}')
        labels[fold] = [tuple(int(r[f'true_state_{state}_pulses']) for state in range(3)) for r in entries]
        scores[fold] = {arm: load_scores(fitted / f'{arm}_{fold}.bin', n) for arm in ARMS}
    selection = {arm: select_multiplier(scores['selection'][arm], labels['selection'],
                                        original['selection']['original_three_way']) for arm in ARMS}

    report, predictions = {}, {}
    for fold in FOLD_NAMES[1:]:
        entries, truth = by_fold[fold], labels[fold]
        ref = state_statistics((int(r['three_way_target_state']) for r in entries), truth)
        if ref != original[fold]['original_three_way']:
            raise ValueError(f'Baseline does not reproduce for {fold}')
        report[fold] = {'sequences': len(entries), 'eligible_pulses': ref['pulses'], 'current_state': ref,
                        'class_reference': original[fold]['class_fixed']}
        predictions[fold] = {}
        for arm, values in scores[fold].items():
            picked = selection[arm]['multiplier']
            pred = [decide(row, picked) for row in values]
            predictions[fold][arm] = pred
            conditional_scores = [p[1] / (p[1] + p[2]) if p[1] + p[2] else 0. for p in values]
            report[fold][arm] = {'selected': state_statistics(pred, truth),
                                 'argmax': state_statistics((decide(row, 1.) for row in values), truth),
                                 'conditional_coarse': conditional_ap(conditional_scores, truth)}
        print(json.dumps({'fold': fold, 'state_1': {arm: block['by_label']['1'] for arm, block in (
              ('reference', ref), *((arm, report[fold][arm]['selected']) for arm in ARMS))}}), flush=True)

    def pass_fold(fold):
        r = report[fold]
        ref, control, candidate = r['current_state'], r['summary_mlp']['selected'], r['raw_pulse_mlp']['selected']
        ref_coarse, control_coarse, new_coarse = (m['by_label']['1'] for m in (ref, control, candidate))
        return (candidate['accuracy'] >= ref['accuracy'] - MAX_ACCURACY_LOSS
                and new_coarse['precision'] >= ref_coarse['precision']
                and new_coarse['recall'] > ref_coarse['recall']
                and new_coarse['f1'] > control_coarse['f1'])

    passed = {fold: pass_fold(fold) for fold in ACCEPTANCE_FOLDS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.raw-state-results-', dir=output.parent) as temp:
        stage = Path(temp) / 'results'
        stage.mkdir()
        combined = stage / 'joint_class_state_validation.csv'
        with combined.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            initial = ('fold', *SEQ, 'pulse_count', 'object_id', 'target_class', 'true_state_0_pulses',
                       'true_state_1_pulses', 'true_state_2_pulses', 'three_way_target_state')
            writer.writerow([*initial, *(f'{arm}_target_state' for arm in ARMS),
                             *(f'{arm}_score_{s}' for arm in ARMS for s in range(3))])
            for fold in FOLD_NAMES[1:]:
                for i, row in enumerate(by_fold[fold]):
                    writer.writerow([*(row[f] for f in initial), *(predictions[fold][arm][i] for arm in ARMS),
                                     *(f'{p:.8g}' for arm in ARMS for p in scores[fold][arm][i])])
        for arm in ARMS:
            for extension in ('model', 'log'):
                copy2(fitted / f'{arm}.{extension}', stage)
        write_json(stage / 'comparison.json', {
            'scope': 'Offline estimates for complete PRI-supported single-source sequences.',
            'prepared_summary': meta, 'fit': fit_meta, 'selection_only_tuning': selection, 'results': report,
            'shared_training': {'summary_features': 43, 'pulse_features': 7, 'pulse_embedding': 8,
                                'time_bands': 4, 'common_hidden_width': 32, 'per_sequence_loss_weight_cap': 256,
                                'loss_class_weight': 'sqrt(largest class count / class count), development only'},
            'passes_held_out_folds': passed,
            'candidate_passed': selection['raw_pulse_mlp']['guard_met'] and all(passed.values()),
            'class_scope': 'Reuses the reference statistical class predictions on the same sequences.',
            'limitations': ['All slices are in the same training scene. The reference features include offline context.',
                            'State metrics are weighted by actual pulse truth counts. Predictions are per sequence.',
                            'Network softmax values are uncalibrated evidence.'],
            'joint_predictions_sha256': digest(combined),
        })
        stage.rename(output)
    print(json.dumps({'passed_held_out_folds': passed,
                      'passed_selection_guard': selection['raw_pulse_mlp']['guard_met']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    add_repository_paths(root)
    run(root, args.prepared_dir.resolve(), args.fit_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
