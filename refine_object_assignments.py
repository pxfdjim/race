#!/usr/bin/env python3
"""Compare object-level recovery using frozen A/B proposals and independent A evidence."""
import argparse
import csv
import json
import math
import shutil
import tempfile
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from prepare_inputs import digest
from recover_objects import SECOND, build_plan, evaluate, load_sequences, records, sequence_key
from src.target_classification import ClassConfig, WaveformClassModel, top_class


ROOT = Path(__file__).resolve().parent
ANCHORS = {'guarded_A_identity_anchor', 'guarded_multiplatform_B_seed', 'guarded_B_refill_with_radio_support'}
WINDOWS = {'development_0_150': (0, 150), 'holdout_150_300': (150, 300),
           'holdout_300_end': (300, math.inf), 'whole_scene': (0, math.inf)}


def read_context(root, split):
    guarded = root / 'fused_b_assisted_guarded' / split
    assist = root / 'fused_b_assisted' / split
    sequences = load_sequences(root, split)
    basis = {sequence_key(r): r['fusion_basis'] for r in records(guarded / 'sequence_object_links.csv')}
    proposals = {sequence_key(r): r['object_id'] for r in records(assist / 'sequence_object_links.csv')
                 if r['fusion_basis'] == 'b_bearing_unique' and r['object_id']}
    geometry = {}
    for r in records(assist / 'sequence_B_evidence.csv'):
        key = sequence_key(r)
        if key in proposals and r['bearing_residual_degrees']:
            geometry[key] = (proposals[key], float(r['bearing_residual_degrees']))
    return sequences, basis, geometry


def final_b_links(root, split):
    aligned = {r['object_id']: r['track_id'] for r in records(
        root / 'ab_matched_2d' / split / 'object_track_links.csv') if r['link_status'] == 'matched'}
    auxiliary = {r['object_id']: r['track_id'] for r in records(
        root / 'fused_b_assisted_guarded' / split / 'object_B_priors.csv')}
    return {oid: track for oid, track in aligned.items() if auxiliary.get(oid) == track}


def train_waveform(sequences, truth_classes, config):
    clean = ((s.rf, s.pw, s.pri, s.model, next(iter(truth_classes[s.key])), s.count)
             for s in sequences if s.time < 150 * SECOND and len(truth_classes[s.key]) == 1)
    return WaveformClassModel(config).fit(clean)


def independent_classes(sequences, basis, model):
    known = {}
    bins = defaultdict(Counter)
    for s in sequences:
        label, score, margin = top_class(model.predict(s.rf, s.pw, s.pri, s.model))
        if score < .90 or margin < .50:
            continue
        known[s.key] = label
        if s.object_id and basis[s.key] in ANCHORS:
            bins[s.object_id, s.time // SECOND][label] += min(s.count, 64)
    votes = defaultdict(Counter)
    for (oid, _), v in bins.items():
        label, count = v.most_common(1)[0]
        if count >= .85 * sum(v.values()):
            votes[oid][label] += 1
    classes = {}
    for oid, v in votes.items():
        label, count = v.most_common(1)[0]
        if count >= 3 and count >= .85 * sum(v.values()):
            classes[oid] = label
    return known, classes


def waveform_bin(s):
    return (s.model, math.floor(s.rf / 20), math.floor(s.pw / .20), math.floor(math.log(s.pri) / .025))


def reference_waveforms(sequences, basis):
    index = defaultdict(lambda: defaultdict(set))
    neighbors = defaultdict(lambda: defaultdict(set))
    for s in sequences:
        if s.object_id and basis[s.key] in ANCHORS:
            time = s.time // SECOND
            index[waveform_bin(s)][s.object_id].add(time)
            if s.group:
                neighbors[s.group_key][s.object_id].add(time)
    return index, {g: {oid: sorted(seconds) for oid, seconds in choices.items()}
                   for g, choices in neighbors.items()}


def radio_support(s, oid, radio):
    model, rf, pw, pri = waveform_bin(s)
    found, other = set(), set()
    for f in range(rf - 1, rf + 2):
        for w in range(pw - 1, pw + 2):
            delta = 0 if model == 'stable' else 1
            for p in range(pri - delta, pri + delta + 1):
                for candidate, seconds in radio.get((model, f, w, p), {}).items():
                    (found if candidate == oid else other).update(seconds)
    return len(found) >= 3, len(found) >= max(3, 2 * len(other))


def recent_support(s, local, oid):
    target, total = 0, 0
    second = s.time // SECOND
    for candidate, ts in local.get(s.group_key, {}).items():
        n = bisect_right(ts, second + 8) - bisect_left(ts, second - 8)
        total += n
        if candidate == oid:
            target = n
    return target >= 3 and target >= .85 * total


def candidate_arms(sequences, basis, geometry, model, aliases, confirmed):
    radio, local = reference_waveforms(sequences, basis)
    own_class, object_class = independent_classes(sequences, basis, model)
    arms = {name: {} for name in ('unique_bearing', 'waveform_supported', 'class_agreement',
                                 'object_evidence', 'object_evidence_with_corrections')}
    for s in sequences:
        if s.key not in geometry:
            continue
        oid, angle = geometry[s.key]
        if s.object_id == oid:
            continue
        own, suggested = own_class.get(s.key), object_class.get(oid)
        conflict = own and suggested and own != suggested
        type_agrees = bool(own and own == suggested)
        mode_exists, mode_distinct = radio_support(s, oid, radio)
        stitched = aliases.get(s.group_key, s.group)
        group_agrees = recent_support(s, local, oid)
        if stitched != s.group:
            ts = local.get(s.key[:3] + (stitched,), {}).get(oid, [])
            second = s.time // SECOND
            group_agrees |= bisect_right(ts, second + 4) - bisect_left(ts, second - 4) >= 3
        if not s.object_id:
            if angle <= .75:
                arms['unique_bearing'][s.key] = oid
            if angle <= 1.25 and mode_exists and not conflict:
                arms['waveform_supported'][s.key] = oid
            if angle <= 1.25 and type_agrees:
                arms['class_agreement'][s.key] = oid
            if not conflict and ((angle <= .9 and mode_distinct) or
                                 (angle <= 1.25 and group_agrees) or
                                 (angle <= .75 and type_agrees and mode_exists)):
                arms['object_evidence'][s.key] = oid
        elif (basis[s.key] != 'guarded_multiplatform_B_seed' and angle <= .75
              and type_agrees and object_class.get(s.object_id) not in (None, own)
              and (group_agrees or mode_distinct)):
            arms['object_evidence_with_corrections'][s.key] = oid
    arms['object_evidence_with_corrections'].update(arms['object_evidence'])
    arms['same_group_refill'] = build_plan(sequences, ()).refill
    arms['refill_and_object_evidence'] = arms['same_group_refill'] | arms['object_evidence']
    arms['waveform_confirmed_link'] = {key: oid for key, oid in arms['waveform_supported'].items() if oid in confirmed}
    arms['object_evidence_confirmed_link'] = {key: oid for key, oid in arms['object_evidence'].items() if oid in confirmed}
    arms['confirmed_type_and_waveform'] = {key: oid for key, oid in arms['waveform_confirmed_link'].items()
                                         if arms['class_agreement'].get(key) == oid}
    arms['same_group_plus_confirmed'] = arms['confirmed_type_and_waveform'] | arms['same_group_refill']
    return arms


def edit_counts(sequences, truth, edits):
    members = defaultdict(Counter)
    for s in sequences:
        if s.object_id:
            members[s.object_id].update(truth[s.key])
    identity = {oid: counts.most_common(1)[0][0] for oid, counts in members.items()}
    counts = dict.fromkeys(('filled_correct', 'filled_wrong', 'moved_to_correct', 'moved_from_correct',
                           'moved_wrong_to_wrong', 'moved_same_identity'), 0)
    by_true_target = defaultdict(Counter)
    for s in sequences:
        if s.key not in edits:
            continue
        new = identity.get(edits[s.key])
        old = identity.get(s.object_id)
        for actual, n in truth[s.key].items():
            status = ('filled_correct' if new == actual else 'filled_wrong') if not s.object_id else (
                'moved_same_identity' if new == old else 'moved_to_correct' if new == actual else
                'moved_from_correct' if old == actual else 'moved_wrong_to_wrong')
            counts[status] += n
            by_true_target[actual][status] += n
    return counts, by_true_target


def stitch_aliases(path, sequences):
    report = json.loads(path.read_text())
    rows = report['results']['whole_scene']['stitch_cross_platform_overlap_guard']['group_remapping']
    scope = {s.key[:2] for s in sequences}
    if len(scope) != 1:
        raise ValueError('Stitch report needs a single scene/version')
    prefix = next(iter(scope))
    return {prefix + (r['platform'], r['old_group']): r['new_group'] for r in rows}


def compare(root, stitch_report, b_truth_path):
    sequences, basis, proposals = read_context(root, 'train')
    truth, classes = defaultdict(Counter), defaultdict(set)
    for r in records(root / 'group_validation/train_sequence_truth.csv'):
        key = sequence_key(r)
        truth[key][r['target_id']] += int(r['pulse_count'])
        classes[key].add(r['target_class'])
    it = ROOT / 'models/target_class/target_class_model.json'
    classifier = train_waveform(sequences, classes, ClassConfig(**json.loads(it.read_text())['config']))
    aliases = stitch_aliases(stitch_report, sequences)
    confirmed = final_b_links(root, 'train')
    by_track = defaultdict(set)
    for row in records(b_truth_path):
        by_track[row['track_id']].add(row['target_id'])
    known_target = {oid: next(iter(by_track[track])) for oid, track in confirmed.items() if len(by_track[track]) == 1}
    results = {}
    for window, (start, end) in WINDOWS.items():
        block = [s for s in sequences if start * SECOND <= s.time < end * SECOND]
        print(f'{window}: {len(block)} sequences', flush=True)
        predictions = {'baseline': {}} | candidate_arms(block, basis, proposals, classifier, aliases, confirmed)
        results[window] = {}
        for name, edits in predictions.items():
            report = evaluate(block, truth, edits)
            report['edits'], report['edits_by_true_target'] = edit_counts(block, truth, edits)
            linked = {'correct': 0, 'wrong': 0, 'without_confirmed_link': 0}
            for key, oid in edits.items():
                labels = truth[key]
                if oid in known_target:
                    right = labels[known_target[oid]]
                    linked['correct'] += right
                    linked['wrong'] += sum(labels.values()) - right
                else:
                    linked['without_confirmed_link'] += sum(labels.values())
            report['edits_against_final_b_track_truth'] = linked
            results[window][name] = report
        incremental = {k: oid for k, oid in predictions['confirmed_type_and_waveform'].items()
                       if k not in predictions['same_group_refill']}
        results[window]['same_group_plus_confirmed']['new_beyond_same_group_by_b_track_truth'] = {
            'correct': sum(truth[key][known_target[oid]] for key, oid in incremental.items()),
            'wrong': sum(sum(truth[key].values()) - truth[key][known_target[oid]]
                         for key, oid in incremental.items()),
        }
    return {'scope': 'PRI-supported TRAIN pulses. Each recovery window uses only its own reference sequences. '
            'A/B proposals and fragment grouping are frozen offline artifacts from the whole TRAIN scene. '
            'The independent waveform classifier is fitted on the first 150 seconds. '
            'Truth pairs including blank object IDs count against pair recall. '
            'Edit counts use the baseline object majority truth in the evaluation window. '
            'The separate A–B edit audit compares with targets on official training B tracks. '
            'Track truth is used for evaluation only.', 'results': results}


def export_candidate(root, baseline, output, arm):
    sequences, basis, geometry = read_context(root, 'test')
    model_path = ROOT / 'models/target_class/target_class_model.json'
    classifier = WaveformClassModel.from_dict(json.loads(model_path.read_text()))
    confirmed = final_b_links(root, 'test')
    edits = candidate_arms(sequences, basis, geometry, classifier, {}, confirmed)[arm]
    pulse_name, link_name = 'P1A_test_pulse_predictions.csv', 'P1A_test_object_track_links.csv'
    linked = {r['object_id']: r['track_id'] for r in records(baseline / link_name) if r['link_status'] == 'matched'}
    if any(linked.get(oid) != confirmed.get(oid) or oid not in linked for oid in edits.values()):
        raise ValueError('Selected edit points to a different or absent baseline B link')
    if output.exists():
        raise FileExistsError(output)
    fusion = root / 'fused_b_assisted_guarded/test/pulse_object_predictions.csv'
    by_object, by_class = Counter(), Counter()
    count = blank = 0
    with tempfile.TemporaryDirectory(prefix='.object-candidate-', dir=output.parent) as tmp:
        staged = Path(tmp) / 'output'
        staged.mkdir()
        with (baseline / pulse_name).open(newline='') as f, fusion.open(newline='') as g, \
             (staged / pulse_name).open('w', newline='') as destination:
            previous, upstream = csv.reader(f), csv.reader(g)
            header, columns = next(previous), next(upstream)
            expected = ['scenario_id', 'data_version', 'pulse_id', 'local_cluster_id',
                        'object_id', 'target_class', 'target_state']
            if header != expected:
                raise ValueError('Unexpected candidate format')
            a = [columns.index(k) for k in expected[:4]]
            b = [columns.index(k) for k in ('scenario_id', 'data_version', 'platform_id', 'sequence_id')]
            old_object = columns.index('object_id')
            writer = csv.writer(destination, lineterminator='\n')
            writer.writerow(header)
            for row, source in zip(previous, upstream, strict=True):
                if row[:4] != [source[i] for i in a] or (source[old_object] and row[4] != source[old_object]):
                    raise ValueError(f'Previous candidate and upstream differ on pulse row {count + 1}')
                count += 1
                if not row[4]:
                    blank += 1
                    oid = edits.get(tuple(source[i] for i in b))
                    if oid:
                        row[4] = oid
                        by_object[oid] += 1
                        by_class[row[5]] += 1
                writer.writerow(row)
        shutil.copyfile(baseline / link_name, staged / link_name)
        file_hashes = {name: digest(staged / name) for name in (pulse_name, link_name)}
        if file_hashes[link_name] != digest(baseline / link_name):
            raise ValueError('Link table changed')
        staged.rename(output)
    return {'arm': arm, 'rows': count, 'blank_before': blank, 'blank_after': blank - sum(by_object.values()),
            'changes_by_existing_object': by_object, 'changes_by_previous_predicted_type': by_class,
            'verified_only_blank_object_ids_changed': True, 'hashes': file_hashes}


def refill_conflict_replay(root, raw_b, split):
    from recover_missing_tracks import load_context
    from src.b_assisted import AssistanceConfig, bearing_candidates
    from src.dataio import InputConfig

    config = InputConfig.from_dict(json.loads((ROOT / 'config/input_units.json').read_text()))
    sequences, observations, _, _, priors, _ = load_context(root, split, raw_b, config)
    by_track = {p.key[-1]: p for p in priors}
    links = final_b_links(root, split)
    basis = {sequence_key(r): r['fusion_basis'] for r in records(
        root / 'fused_b_assisted_guarded' / split / 'sequence_object_links.csv')}
    model = json.loads((ROOT / 'models/target_class/target_class_model.json').read_text())
    truth, classes = defaultdict(Counter), defaultdict(set)
    if split == 'train':
        for r in records(root / 'group_validation/train_sequence_truth.csv'):
            key = sequence_key(r)
            truth[key][r['target_id']] += int(r['pulse_count'])
            classes[key].add(r['target_class'])
        classifier = train_waveform(sequences, classes, ClassConfig(**model['config']))
        track_truth = {r['track_id']: r['target_id'] for r in records(raw_b)}
        windows = WINDOWS
    else:
        classifier = WaveformClassModel.from_dict(model)
        windows = {'whole_scene': WINDOWS['whole_scene']}
    report, rejected = {}, {}
    bearing_config = AssistanceConfig()
    for window, (lo, hi) in windows.items():
        part = [s for s in sequences if lo * SECOND <= s.time < hi * SECOND]
        original = build_plan(part, ()).refill
        independent, objects = independent_classes(part, basis, classifier)
        rejected = {}
        for key, oid in original.items():
            own, assigned = independent.get(key), objects.get(oid)
            if not own or not assigned or own == assigned or oid not in links:
                continue
            prior = by_track[links[oid]]
            match = bearing_candidates(observations[key], [prior], bearing_config).get(prior.key)
            # Repeated, independent waveform evidence and a discrepant bearing are both required.
            if match and match['residual'] >= 1.0:
                rejected[key] = oid, own, assigned
        counts, evidence = Counter(), Counter()
        for s in part:
            if s.key not in rejected:
                continue
            oid, own, assigned = rejected[s.key]
            evidence[f'{oid}/{own}/{assigned}'] += s.count
            if split == 'train':
                correct = truth[s.key][track_truth[links[oid]]]
                counts['correct_removed'] += correct
                counts['wrong_removed'] += s.count - correct
        report[window] = {'rejected_by_object_and_waveform_type': evidence}
        if split == 'train':
            corrected = {k: oid for k, oid in original.items() if k not in rejected}
            report[window].update(counts=counts, before=evaluate(part, truth, original),
                                  after=evaluate(part, truth, corrected))
    return report, rejected


def undo_refill_conflicts(root, candidate, delta, rejected):
    if delta.exists():
        raise FileExistsError(delta)
    pulse = candidate / 'P1A_test_pulse_predictions.csv'
    upstream = root / 'fused_b_assisted_guarded/test/pulse_object_predictions.csv'
    before = digest(pulse)
    counts, rows = Counter(), 0
    with tempfile.TemporaryDirectory(prefix='.refill-correction-', dir=candidate.parent) as scratch:
        staged, audit = Path(scratch) / pulse.name, Path(scratch) / delta.name
        with pulse.open(newline='') as a, upstream.open(newline='') as b, \
             staged.open('w', newline='') as c, audit.open('w', newline='') as d:
            current, old = csv.reader(a), csv.reader(b)
            header, base_header = next(current), next(old)
            keys = [base_header.index(k) for k in ('scenario_id', 'data_version', 'platform_id', 'sequence_id')]
            pulse_keys = [base_header.index(k) for k in header[:3]]
            object_col = base_header.index('object_id')
            writer, change = csv.writer(c, lineterminator='\n'), csv.writer(d, lineterminator='\n')
            writer.writerow(header)
            change.writerow(['pulse_id', 'object_id_before'])
            for row, previous in zip(current, old, strict=True):
                rows += 1
                if row[:3] != [previous[i] for i in pulse_keys]:
                    raise ValueError('Unaligned pulse identities')
                key = tuple(previous[i] for i in keys)
                if key in rejected:
                    oid = rejected[key][0]
                    if previous[object_col] or row[4] != oid:
                        raise ValueError('Correction touches a non-refill assignment')
                    change.writerow([row[2], row[4]])
                    counts[oid] += 1
                    row[4] = ''
                writer.writerow(row)
        if not counts:
            raise ValueError('No conflicting assignments found in current candidate')
        after = digest(staged)
        audit.replace(delta)
        staged.replace(pulse)
    return {'rows': rows, 'reverted_assignments': counts, 'delta': str(delta),
            'previous_pulse_sha256': before, 'pulse_sha256': after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', type=Path, required=True)
    parser.add_argument('--stitch-report', type=Path)
    parser.add_argument('--train-b', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--export-baseline', type=Path)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--conflict-candidate', type=Path, help='Review high-confidence contradictions in early same-group refills')
    parser.add_argument('--test-b', type=Path)
    parser.add_argument('--conflict-delta', type=Path)
    parser.add_argument('--arm', default='confirmed_type_and_waveform',
                        choices=('confirmed_type_and_waveform', 'object_evidence_confirmed_link', 'waveform_confirmed_link'))
    args = parser.parse_args()
    if args.conflict_candidate:
        if not args.train_b or not args.test_b:
            parser.error('Refill audit needs --train-b and --test-b')
        report = json.loads(args.report.read_text()) if args.report.exists() else {}
        train, _ = refill_conflict_replay(args.intermediates, args.train_b, 'train')
        test, rejected = refill_conflict_replay(args.intermediates, args.test_b, 'test')
        result = {'training': train, 'test_review': test}
        holdout = (train[name] for name in ('holdout_150_300', 'holdout_300_end'))
        if args.conflict_delta and all(not v['counts']['correct_removed'] and v['counts']['wrong_removed'] > 0
                                       and v['after']['pair_f1'] >= v['before']['pair_f1'] for v in holdout):
            result['candidate'] = undo_refill_conflicts(args.intermediates, args.conflict_candidate,
                                                       args.conflict_delta, rejected)
        report['refill_conflict_review'] = result
        print('refill conflict review', test, result.get('candidate'), flush=True)
    elif args.export_baseline or args.export_dir:
        if not args.export_baseline or not args.export_dir:
            parser.error('--export-baseline and --export-dir must be given together')
        report = json.loads(args.report.read_text())
        report['test_candidate'] = export_candidate(args.intermediates, args.export_baseline, args.export_dir, args.arm)
    else:
        if not args.stitch_report or not args.train_b:
            parser.error('Training comparison requires --stitch-report and --train-b')
        report = compare(args.intermediates, args.stitch_report, args.train_b)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
