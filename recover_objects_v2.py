#!/usr/bin/env python3
"""Compare evidence rules and a learned scorer for recovering blank object IDs."""

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from math import atan2, cos, degrees, hypot, log1p, radians, sin, sqrt
from pathlib import Path
from statistics import median

import numpy as np

from recover_objects import (SECOND, build_plan, evaluate, export,
                             load_sequences, records, sequence_key)
from src.b_assisted import AssistanceConfig, BearingPrior
from src.dataio import InputConfig, read_b_tracks
from src.fusion import LocalFrame


def angular_difference(a, b):
    return abs((a - b + 180.) % 360. - 180.)


def candidate_rows(sequences):
    """One example per residual sequence and plausible existing object."""
    indexed = defaultdict(list)
    for q in sequences:
        if q.object_id:
            indexed[q.key[:3] + (int(q.rf // 100),)].append(q)
    times = {}
    for k, items in indexed.items():
        items.sort(key=lambda q: q.time)
        times[k] = [q.time for q in items]
    candidates = {}
    for s in sequences:
        if s.object_id or not s.group:
            continue
        votes = defaultdict(dict)
        same_group = defaultdict(set)
        for bucket in range(int(s.rf // 100) - 2, int(s.rf // 100) + 3):
            k = s.key[:3] + (bucket,)
            if k not in times:
                continue
            ts = times[k]
            first = bisect_left(ts, s.time - 8 * SECOND)
            last = bisect_right(ts, s.time + 8 * SECOND)
            for q in indexed[k][first:last]:
                if q.model != s.model:
                    continue
                df, dw = abs(s.rf - q.rf), abs(s.pw - q.pw)
                dp = abs(s.pri - q.pri) / min(s.pri, q.pri)
                dt = abs(s.time - q.time) / SECOND
                da = angular_difference(s.bearing, q.bearing)
                same = s.group == q.group
                if (df > 160 or dw > .55 or dp > (.035 if s.model == 'stable' else .10)
                        or da > 4. + .2 * dt):
                    continue
                scale = .008 if s.model == 'stable' else .05
                cost = sqrt((df / 80) ** 2 + (dw / .30) ** 2 + (dp / scale) ** 2
                            + (da / (2 + .25 * dt)) ** 2 + (dt / 8) ** 2)
                bin_id = q.time // (SECOND // 2)
                if same:
                    same_group[q.object_id].add(bin_id)
                sample = (cost, dt, df / 80, dw / .30, dp / scale, da, int(same))
                if bin_id not in votes[q.object_id] or sample < votes[q.object_id][bin_id]:
                    votes[q.object_id][bin_id] = sample
        features = {}
        for oid, bins in votes.items():
            if len(bins) < 2:
                continue
            ordered = sorted(bins.values())
            best = ordered[0]
            features[oid] = (*best, median(r[0] for r in ordered[:3]),
                             log1p(len(bins)), log1p(len(same_group[oid])),
                             log1p(len(votes) - 1), int(s.model != 'stable'))
        if features:
            candidates[s.key] = features
    return candidates


def dominant_targets(sequences, truth):
    labels = defaultdict(Counter)
    for s in sequences:
        if s.object_id:
            labels[s.object_id].update(truth[s.key])
    return {oid: counts.most_common(1)[0][0] for oid, counts in labels.items()}


def logistic(values):
    return 1. / (1. + np.exp(-np.clip(values, -35., 35.)))


def fit_scorer(sequences, candidates, truth):
    x, y, weights = [], [], []
    dominant = dominant_targets(sequences, truth)
    for s in sequences:
        for oid, features in candidates.get(s.key, {}).items():
            x.append(features)
            y.append(truth[s.key][dominant[oid]] / s.count)
            weights.append(min(s.count, 256))
    x, y, weights = map(np.asarray, (x, y, weights))
    if len(x) == 0 or not 0 < np.average(y, weights=weights) < 1:
        raise ValueError('Scorer requires positive and negative training examples')
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-5)
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    weights = weights / weights.sum()
    coefs = np.zeros(design.shape[1])
    penalty = np.diag([0.] + [.005] * x.shape[1])
    for _ in range(24):
        p = logistic(design @ coefs)
        gradient = design.T @ (weights * (p - y)) + penalty @ coefs
        hessian = design.T @ ((weights * p * (1 - p))[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefs -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return center, scale, coefs, {'examples': len(x), 'positive_weight_fraction': float(weights @ y)}


def score_candidates(candidates, model):
    center, scale, coefs, _ = model
    result = {}
    for key, options in candidates.items():
        ids = list(options)
        x = np.asarray(list(options.values()))
        scores = logistic(coefs[0] + ((x - center) / scale) @ coefs[1:])
        result[key] = sorted(zip(scores, ids), reverse=True)
    return result


def accept_scores(scores, threshold, margin):
    return {key: ranked[0][1] for key, ranked in scores.items()
            if ranked[0][0] >= threshold and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= margin)}


def choose_rule(candidates, same_only):
    edits = {}
    for key, options in candidates.items():
        good = []
        for oid, f in options.items():
            best_cost, _, _, _, _, _, same, top_median, bins, same_bins, *_ = f
            if same_only:
                if same and same_bins >= log1p(3) and best_cost <= 1.20 and top_median <= 1.60:
                    good.append(oid)
            elif bins >= log1p(3) and best_cost <= .95 and top_median <= 1.25:
                good.append(oid)
        if len(good) == 1:
            edits[key] = good[0]
    return edits


def b_fragment_scores(sequences, intermediates, split, b_path, input_units):
    groups, occupied = defaultdict(list), set()
    for s in sequences:
        if s.object_id and s.group:
            occupied.add(s.group_key)
        elif s.group:
            groups[s.group_key].append(s)
    groups = {key: items for key, items in groups.items() if key not in occupied
              and sum(s.count for s in items) >= 512 and len({s.time // SECOND for s in items}) >= 4}
    anchors = {s.key for items in groups.values() for s in items}
    positions = {sequence_key(r): tuple(float(r[k]) for k in
                 ('observer_longitude', 'observer_latitude', 'observer_height'))
                 for r in records(intermediates / 'grouped' / split / 'bearing_observations.csv')
                 if sequence_key(r) in anchors}
    cfg = AssistanceConfig()
    priors = [BearingPrior(t, cfg) for t in read_b_tracks(b_path, input_units).values()]
    result = []
    for key, items in sorted(groups.items()):
        by_track = defaultdict(lambda: defaultdict(list))
        member_errors = defaultdict(dict)
        seconds = set()
        for s in items:
            sec = s.time // SECOND
            seconds.add(sec)
            lon, lat, h = positions[s.key]
            x, y = radians(lon), radians(lat)
            origin = LocalFrame.ecef(lon, lat, h)
            east = (-sin(x), cos(x), 0.)
            north = (-sin(y) * cos(x), -sin(y) * sin(x), cos(y))
            for prior in priors:
                p = prior.position(s.time)
                if p is None:
                    continue
                delta = [b - a for a, b in zip(origin, p)]
                dx, dy = (sum(a * b for a, b in zip(delta, axis)) for axis in (east, north))
                error = angular_difference(degrees(atan2(dx, dy)), s.bearing)
                if 1. < hypot(dx, dy) <= cfg.maximum_range_m:
                    by_track[prior.key[2]][sec].append(error)
                    member_errors[prior.key[2]][s.key] = error
        scores = []
        for tid, times in by_track.items():
            medians = [median(v) for v in times.values()]
            if len(medians) >= max(4, .75 * len(seconds)):
                scores.append((median(medians), -sum(e <= 3 for e in medians) / len(medians), tid))
        scores.sort()
        if scores:
            best, within, tid = scores[0]
            margin = scores[1][0] - best if len(scores) > 1 else None
            result.append({'group': key, 'track': tid, 'pulses': sum(s.count for s in items),
                           'seconds': len(seconds), 'median_degrees': best, 'within_3deg': -within,
                           'margin_degrees': margin,
                           'candidates': [{'track': t, 'median_degrees': d, 'within_3deg': -w}
                                          for d, w, t in scores[:5]],
                           'member_errors': member_errors[tid],
                           'members': [s.key for s in items]})
    return result


def assess_fragments(fragments, used_tracks, truth=None, b_truth=None):
    by_track = defaultdict(list)
    for row in fragments:
        row['clear'] = (row['median_degrees'] <= 2. and row['within_3deg'] >= .8
                        and row['margin_degrees'] is not None and row['margin_degrees'] >= .75)
        row['track_already_used'] = row['track'] in used_tracks
        if truth is not None:
            label = b_truth.get(row['track'])
            labels = Counter()
            for key in row['members']:
                labels.update(truth[key])
            row['agree_with_B_truth'] = labels[label] / row['pulses'] if label else None
            row['A_truth'] = labels
        if row['clear'] and not row['track_already_used']:
            by_track[row['track']].append(row)
    good = {track: items for track, items in by_track.items()
            if len({r['group'][2] for r in items}) == len(items) >= 2}
    return {'passing_fragment_count': sum(len(x) for x in by_track.values()),
            'potential_new_track_count': len(good),
            'potential_new_object_pulses': sum(r['pulses'] for items in good.values() for r in items),
            'fragments': [{k: v for k, v in r.items() if k not in ('members', 'member_errors')}
                          for r in fragments]}


def fingerprint(items):
    histograms = (Counter(), Counter(), Counter())
    total = 0
    for s in items:
        weight = min(s.count, 256)
        total += weight
        for hist, value in zip(histograms, (round(s.rf / 100), round(s.pw / .25),
                                            (s.model, round(s.pri / 16))), strict=True):
            hist[value] += weight
    return [{k: n / total for k, n in hist.items()} for hist in histograms]


def fingerprint_overlap(a, b):
    return [sum(min(left[k], right[k]) for k in left.keys() & right.keys())
            for left, right in zip(a, b, strict=True)]


def joint_fragments(fragments, sequences, used_tracks):
    """Use a shared B curve to compare A fragments; identity of B remains open."""
    by_key = {s.key: s for s in sequences}
    proposals = defaultdict(list)
    for row in fragments:
        if row['track'] in used_tracks or row['median_degrees'] > .35:
            continue
        compatible = [by_key[k] for k, error in row['member_errors'].items() if error <= 1.]
        seconds = {s.time // SECOND for s in compatible}
        pulses = sum(s.count for s in compatible)
        if len(seconds) < 4 or pulses < .75 * row['pulses']:
            continue
        proposals[row['track']].append((row['group'], compatible, seconds, fingerprint(compatible)))
    results = {}
    for track, pool in sorted(proposals.items()):
        partners, synchrony = defaultdict(set), set()
        comparisons = []
        for i, (a, _, sa, fa) in enumerate(pool):
            for j in range(i + 1, len(pool)):
                b, _, sb, fb = pool[j]
                if a[2] == b[2]:
                    continue
                overlap = fingerprint_overlap(fa, fb)
                common_seconds = len(sa & sb)
                comparisons.append({'fragment_a': a, 'fragment_b': b, 'overlap': overlap,
                                    'common_seconds': common_seconds})
                if min(overlap) >= .25:
                    partners[i].add(j)
                    partners[j].add(i)
                    if common_seconds >= 3:
                        synchrony.add((i, j))
        components, seen = [], set()
        for i in sorted(partners):
            if i in seen:
                continue
            pending, component = [i], set()
            while pending:
                j = pending.pop()
                if j not in component:
                    component.add(j)
                    pending.extend(partners[j] - component)
            seen.update(component)
            if any({a, b} <= component for a, b in synchrony):
                components.append(component)
        # Ambiguous disconnected clusters or simultaneous fragments at one node
        # are not resolved by assigning them a common identifier.
        if len(components) != 1:
            continue
        component = components[0]
        selected = [pool[i] for i in sorted(component)]
        if any(a[0][2] == b[0][2] and a[2] & b[2]
               for i, a in enumerate(selected) for b in selected[i + 1:]):
            continue
        results[track] = {'groups': [group for group, *_ in selected],
                          'members': [s for _, items, _, _ in selected for s in items],
                          'comparisons': comparisons}
    return results


def joint_summary(results, truth=None, b_truth=None):
    summary = {}
    for track, result in results.items():
        columns = {k: v for k, v in result.items() if k != 'members'}
        columns['pulses'] = sum(s.count for s in result['members'])
        if truth is not None:
            labels = Counter()
            for s in result['members']:
                labels.update(truth[s.key])
            columns['truth_counts'] = dict(labels)
            columns['agree_with_B_truth'] = labels[b_truth[track]] / columns['pulses']
        summary[track] = columns
    return summary


def run(args):
    paths = args.intermediates
    train = load_sequences(paths, 'train')
    test = load_sequences(paths, 'test')
    truth = defaultdict(Counter)
    for row in records(paths / 'group_validation/train_sequence_truth.csv'):
        truth[sequence_key(row)][row['target_id']] += int(row['pulse_count'])
    cuts = [0, int(max(q.time for q in train) * .60), int(max(q.time for q in train) * .80),
            max(q.time for q in train) + 1]
    blocks = [[s for s in train if a <= s.time < b] for a, b in zip(cuts, cuts[1:])]
    lo, mid, hi = blocks
    features = [candidate_rows(block) for block in blocks]
    model = fit_scorer(lo, features[0], truth)
    old = [build_plan(block, ()).refill for block in blocks]
    reports = {}
    restrictions = (.88, .92, .95, .97, .985)
    thresholds = [(p, m) for p in restrictions for m in (.10, .20)]
    mid_scores = score_candidates(features[1], model)
    tuning = []
    for p, m in thresholds:
        result = evaluate(mid, truth, accept_scores(mid_scores, p, m))
        tuning.append({'p': p, 'margin': m, 'metrics': result})
    # Require nearly the same pair precision as the existing rule during selection.
    mid_old = evaluate(mid, truth, old[1])
    eligible = [t for t in tuning if t['metrics']['pair_precision'] >= mid_old['pair_precision'] - .002]
    chosen = max(eligible or tuning, key=lambda t: t['metrics']['pair_f1'])
    p, margin = chosen['p'], chosen['margin']
    test_features = candidate_rows(test)
    test_options = {'prior_refill': build_plan(test, ()).refill,
                    'strict_same_group': choose_rule(test_features, True),
                    'wider_group_rule': choose_rule(test_features, False),
                    'learned': accept_scores(score_candidates(test_features, model), p, margin)}
    for name, block, data, prior in zip(('fit_60', 'tune_20', 'holdout_20'), blocks, features, old, strict=True):
        variants = {'baseline': {}, 'prior_refill': prior,
                    'strict_same_group': choose_rule(data, True), 'wider_group_rule': choose_rule(data, False),
                    'learned': accept_scores(score_candidates(data, model), p, margin)}
        reports[name] = {nm: evaluate(block, truth, edits) for nm, edits in variants.items()}
        print(name, {nm: round(r['pair_f1'], 6) for nm, r in reports[name].items()}, flush=True)
    config = InputConfig.from_dict(json.loads((Path(__file__).parent / 'config/input_units.json').read_text()))
    b_reports, joint_reports = {}, {}
    for split, seqs, path in (('train', train, args.train_b), ('test', test, args.test_b)):
        link_path = paths / 'ab_matched_2d' / split / 'object_track_links.csv'
        used = {row['track_id'] for row in records(link_path) if row['track_id']}
        fragments = b_fragment_scores(seqs, paths, split, path, config)
        labels = {r['track_id']: r['target_id'] for r in records(path)} if split == 'train' else None
        b_reports[split] = assess_fragments(fragments, used, truth if labels is not None else None, labels)
        print('B fragments', split, {k: v for k, v in b_reports[split].items() if k != 'fragments'}, flush=True)
        found = joint_fragments(fragments, seqs, used)
        joint_reports[split] = joint_summary(found, truth if labels is not None else None, labels)
        if split == 'train':
            masked = {}
            for track in sorted(used):
                withheld = joint_fragments(fragments, seqs, used - {track})
                if track in withheld:
                    masked[track] = withheld[track]
            joint_reports['train_occupied_track_masking'] = joint_summary(masked, truth, labels)
        else:
            combined = dict(test_options['prior_refill'])
            taken = {s.object_id for s in seqs if s.object_id}
            for i, group in enumerate(found.values(), 1):
                oid = f'OR{i:05d}'
                if oid in taken:
                    raise ValueError(f'Duplicate object ID: {oid}')
                for s in group['members']:
                    combined[s.key] = oid
            test_options['joint_new'] = combined
    report = {'model_fit': model[3], 'threshold_selection': chosen, 'threshold_grid': tuning,
              'train': reports, 'fragment_B_check': b_reports,
              'joint_fragment_check': joint_reports,
              'test_candidates': {name: {'sequences': len(edits),
                  'pulses': sum(q.count for q in test if q.key in edits)} for name, edits in test_options.items()}}
    if args.export_variant:
        args.output.mkdir(parents=True, exist_ok=True)
        report['export'] = export(paths, args.baseline, args.output / args.export_variant,
                                  test_options[args.export_variant])
        report['export']['variant'] = args.export_variant
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / '对照结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('intermediates', 'train-b', 'test-b', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--export-variant', choices=('strict_same_group', 'wider_group_rule', 'learned', 'joint_new'))
    args = p.parse_args()
    if bool(args.baseline) != bool(args.export_variant):
        p.error('Use --baseline together with --export-variant')
    run(args)


if __name__ == '__main__':
    main()
