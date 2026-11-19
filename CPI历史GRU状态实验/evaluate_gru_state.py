"""Compare the current state model, one-step GRU and past-CPI GRU."""

import argparse
from collections import defaultdict
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from fit_gru_state import ARMS, SOURCE, digest
from prepare_gru_state import FOLDS, add_paths


def evaluate(root, prepared, fitted, output):
    from compare_class_state import MAX_ACCURACY_LOSS, SEQ, csv_write, read_json, rows, state_statistics, write_json
    from compare_cpi_context import conditional_ap, support_by_true_state
    from compare_raw_state import decide, load_scores, select_multiplier

    if output.exists():
        raise FileExistsError(output)
    meta_path = prepared / 'prepared.json'
    meta, manifest = read_json(meta_path), read_json(fitted / 'fit.json')
    if manifest['prepared_sha256'] != digest(meta_path) or manifest['native_sha256'] != digest(SOURCE):
        raise ValueError('Fitted model and evaluation inputs have different provenance')
    path = prepared / 'evaluation_index.csv'
    if digest(path) != meta['files_sha256'][path.name]:
        raise ValueError('Evaluation row order changed')
    for name, expected in manifest['files_sha256'].items():
        if digest(fitted / name) != expected:
            raise ValueError(f'Fitted artifact changed: {name}')
    reference = read_json(root / '组内脉冲状态实验/results/comparison.json')['results']
    records = defaultdict(list)
    for row in rows(path):
        records[row['fold']].append(row)
    report, predictions = {}, {}
    for fold in FOLDS[1:]:
        subset = records[fold]
        has_history = [int(r['history_cpis']) > 0 for r in subset]
        if (len(subset), sum(has_history)) != (meta['folds'][fold]['sequences'], meta['folds'][fold]['sequences_with_history']):
            raise ValueError(f'Missing evaluation centers or history: {fold}')
        counts = [tuple(int(r[f'true_state_{j}_pulses']) for j in range(3)) for r in subset]
        old_state = [int(r['three_way_target_state']) for r in subset]
        baseline = state_statistics(old_state, counts)
        if baseline != reference[fold]['current_state']:
            raise ValueError(f'Baseline does not reproduce: {fold}')
        all_scores = {a: load_scores(fitted / f'{a}_{fold}.bin', len(subset)) for a in ARMS}
        if fold == 'selection':
            selection = {a: select_multiplier(z, counts, baseline) for a, z in all_scores.items()}
        predictions[fold] = {}
        entries = {'current_state': baseline, 'class_reference': reference[fold]['class_reference'],
                   'history_coverage_by_state': support_by_true_state(zip(has_history, counts, strict=True))}
        for arm, scores in all_scores.items():
            pred = [decide(z, selection[arm]['multiplier']) for z in scores]
            predictions[fold][arm] = pred, scores
            p_coarse_given_tracking = [s[1] / (s[1] + s[2]) if s[1] + s[2] else 0. for s in scores]
            entries[arm] = {'all': state_statistics(pred, counts),
                            'only_cpis_with_history': state_statistics((p for p, seen in zip(pred, has_history) if seen),
                                                                      (y for y, seen in zip(counts, has_history) if seen)),
                            'conditional_tracking': conditional_ap(p_coarse_given_tracking, counts)}
        report[fold] = entries
        print({'fold': fold, 'coarse': {name: entries[name]['all']['by_label']['1'] for name in ARMS}}, flush=True)

    def passed(fold):
        r = report[fold]
        before, current, past = r['current_state'], r['current_cpi']['all'], r['history_gru']['all']
        b, c, g = (v['by_label']['1'] for v in (before, current, past))
        return (past['accuracy'] >= before['accuracy'] - MAX_ACCURACY_LOSS and g['precision'] >= b['precision']
                and g['recall'] > b['recall'] and g['f1'] > c['f1'])

    checks = {fold: passed(fold) for fold in ('diagnostic_420_500', 'instance_holdout')}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gru-results-', dir=output.parent) as folder:
        stage = Path(folder) / 'results'
        stage.mkdir()
        initial = ('fold', *SEQ, 'object_id', 'target_class', 'pulse_count', 'history_cpis',
                   'true_state_0_pulses', 'true_state_1_pulses', 'true_state_2_pulses', 'three_way_target_state')
        joint = stage / 'joint_class_state_validation.csv'
        csv_write(joint, (*initial, *(f'{a}_target_state' for a in ARMS), *(f'{a}_score_{i}' for a in ARMS for i in range(3))),
                  ([*(row[k] for k in initial), *(predictions[fold][a][0][i] for a in ARMS),
                    *(f'{value:.8g}' for a in ARMS for value in predictions[fold][a][1][i])]
                   for fold in FOLDS[1:] for i, row in enumerate(records[fold])))
        for arm in ARMS:
            for suffix in ('model', 'log'):
                copy2(fitted / f'{arm}.{suffix}', stage)
        write_json(stage / 'comparison.json', {'prepared': meta, 'fit': manifest, 'selection_only': selection,
            'results': report, 'passes_holdout': checks, 'candidate_passed': all(checks.values()) and selection['history_gru']['guard_met'],
            'network': {'kind': 'GRU', 'hidden_size': 16, 'predicted_label_per_center_cpi': True,
                        'training': 'pulse-count soft labels capped at 256, inverse square-root class weighting'},
            'scope': 'Complete PRI-supported sequences in the same frozen offline scene.',
            'joint_predictions_sha256': digest(joint)})
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.pipeline_root.resolve()
    add_paths(root)
    evaluate(root, args.prepared_dir.resolve(), args.fit_dir.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
