#!/usr/bin/env python3
"""Compare changes to platform-local grouping on fixed PRI observations."""
import argparse
import bisect
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

from group_sequences import read_sequences
from prepare_group_inputs import ObserverPaths
from src.local_grouping import GroupConfig, associate, overlap_conflict
from src.pri import angle_delta


SECOND = 1_000_000_000
ROOT = Path(__file__).resolve().parent
WINDOWS = {'development': (0, 150), 'holdout_mid': (150, 300),
           'holdout_late': (300, 500), 'whole_scene': (None, None)}


@dataclass
class Fragment:
    key: tuple
    pulses: int
    start: int
    end: int
    rf: float
    pw: float
    pri: float
    bearings: dict
    compact_sequences: list


def summarize(groups):
    fragments = {}
    for group in groups:
        if group.count < 3 or group.pulse_count < 32:
            continue
        bins = defaultdict(list)
        for s in group.members:
            bins[s.time // SECOND].append(s.doa)
        bearings = {t: (a[0] + median(angle_delta(x, a[0]) for x in a)) % 360
                    for t, a in bins.items()}
        fragments[group.scope + (group.name,)] = Fragment(
            group.scope + (group.name,), group.pulse_count, group.first, group.last,
            median(s.rf for s in group.members), median(s.pw for s in group.members),
            median(s.pri for s in group.members), bearings,
            sorted((s for s in group.members if (s.count - 1) * s.pri * 1000 >= .8 * (s.end - s.start)),
                   key=lambda s: s.start))
    return fragments


def time_and_bearing(a, b):
    common = sorted(a.bearings.keys() & b.bearings.keys())
    errors = sorted(abs(angle_delta(a.bearings[t], b.bearings[t])) for t in common)
    near = median(errors) if errors else None
    upper = errors[math.floor(.8 * (len(errors) - 1))] if errors else None
    span = common[-1] - common[0] if common else 0
    return len(errors), near, upper, span


def endpoint_error(a, b):
    if b.start < a.start:
        a, b = b, a
    gap = (b.start - a.end) / SECOND
    if not 0 <= gap <= 12:
        return None
    left, right = sorted(a.bearings)[-5:], sorted(b.bearings)[:5]
    if len(left) < 3 or len(right) < 3 or left[-1] - left[0] > 10 or right[-1] - right[0] > 10:
        return None
    mid = (left[-1] + right[0]) / 2

    def project(fragment, times, origin):
        values = [angle_delta(fragment.bearings[t], fragment.bearings[origin]) for t in times]
        slope = median((y - x) / (u - t) for (t, x), (u, y) in itertools.combinations(zip(times, values), 2))
        if abs(slope) > 3:
            return None
        return fragment.bearings[origin] + median(v + slope * (mid - t) for t, v in zip(times, values))

    x, y = project(a, left, left[-1]), project(b, right, right[0])
    if x is None or y is None:
        return None
    return abs(angle_delta(x, y)), gap


def cross_platform_evidence(path, links, fragments):
    votes = defaultdict(lambda: defaultdict(set))
    with path.open(newline='') as source:
        for row in csv.DictReader(source):
            if float(row['association_cost']) > .30 or float(row['geometry_error_proxy_m']) > 400:
                continue
            scope = row['scenario_id'], row['data_version']
            stamp = int(row['time_ns']) // SECOND
            members = []
            for platform, sid in zip(row['platform_ids'].split('|'), row['sequence_ids'].split('|'), strict=True):
                key = scope + (platform, sid)
                if key in links:
                    group = key[:3] + (links[key]['local_cluster_id'],)
                    if group in fragments:
                        members.append(group)
            for a, b in itertools.permutations(members, 2):
                votes[a][b].add(stamp)
    totals = defaultdict(set)
    for local, anchors in votes.items():
        for remote, stamps in anchors.items():
            totals[local, remote[:3]].update(stamps)
    return votes, totals


def supported_times(a, b, maximum_gap=12):
    ordered = sorted(b)
    count = 0
    for stamp in a:
        index = bisect.bisect_left(ordered, stamp)
        neighbors = ordered[max(0, index - 1):index + 1]
        count += any(abs(stamp - other) <= maximum_gap for other in neighbors)
    return count


def peer_support(a, b, votes, totals):
    peers = {}
    for anchor in votes[a].keys() & votes[b].keys():
        first, second = votes[a][anchor], votes[b][anchor]
        if min(len(first), len(second)) < 4:
            continue
        count = min(supported_times(first, second), supported_times(second, first))
        if count < 4:
            continue
        coverage = min(len(first) / len(totals[a, anchor[:3]]), len(second) / len(totals[b, anchor[:3]]))
        if coverage >= .35:
            peers[anchor[2]] = max(peers.get(anchor[2], (0., 0)), (coverage, count))
    return peers


def candidates(fragments, votes, totals):
    by_scope = defaultdict(list)
    for fragment in fragments.values():
        by_scope[fragment.key[:3]].append(fragment)
    local, cross, conflicts = [], [], set()
    for items in by_scope.values():
        for a, b in itertools.combinations(items, 2):
            x, y = sorted((a.key, b.key))
            n, near, upper, span = time_and_bearing(a, b)
            drf, dpw = abs(a.rf - b.rf), abs(a.pw - b.pw)
            ratio = max(a.pri, b.pri) / min(a.pri, b.pri)
            if drf > 190 or dpw > 1.3 or ratio > 3 or (n >= 5 and upper > 3):
                conflicts.add((x, y))
                continue
            radio = drf / 90 + dpw / .9 + math.log(ratio) / math.log(2.5)
            if drf <= 90 and dpw <= .9 and ratio <= 2.5:
                end = endpoint_error(a, b)
                if n >= 5 and span >= 4 and near <= .7 and upper <= 1.3:
                    local.append((5 + min(n, 20) / 20 - near - .15 * radio, x, y, 'repeated_bearing'))
                elif end and end[0] <= .8:
                    local.append((4 - end[0] - end[1] / 24 - .15 * radio, x, y, 'endpoint_continuation'))
            if n >= 5 and (near > 1.4 or upper > 2.4):
                continue
            peers = peer_support(x, y, votes, totals)
            strong_peer = any(coverage >= .68 and count >= 7 for coverage, count in peers.values())
            if strong_peer or len(peers) >= 2:
                cross.append((8 + len(peers) + max(c for c, _ in peers.values()) - .15 * radio,
                              x, y, 'cross_platform_bridge'))
    return local, cross, conflicts


def repeated_overlap_conflict(a, b, config):
    if a.end <= b.start or b.end <= a.start:
        return False
    first, second = a.compact_sequences, b.compact_sequences
    affected_first, affected_second, seconds = set(), set(), set()
    cursor = 0
    for i, x in enumerate(first):
        while cursor < len(second) and second[cursor].end <= x.start:
            cursor += 1
        j = cursor
        while j < len(second) and second[j].start < x.end:
            y = second[j]
            if overlap_conflict(x, y, config):
                affected_first.add(i)
                affected_second.add(j)
                seconds.add(max(x.start, y.start) // SECOND)
                if len(seconds) >= 3 and (len(affected_first) >= .05 * len(first)
                                          or len(affected_second) >= .05 * len(second)):
                    return True
            j += 1
    return False


def join_fragments(edges, conflicts, fragments, config=None):
    components = {key: {key} for key in fragments}
    rejected, accepted, checked = [], [], {}
    for score, a, b, reason in sorted(edges, key=lambda edge: (-edge[0], edge[1], edge[2])):
        left, right = components[a], components[b]
        if left is right or any(tuple(sorted((x, y))) in conflicts for x in left for y in right):
            continue
        if config is not None:
            pairs = [tuple(sorted((x, y))) for x in left for y in right]
            for pair in pairs:
                if pair not in checked:
                    checked[pair] = repeated_overlap_conflict(*(fragments[k] for k in pair), config)
            if any(checked[pair] for pair in pairs):
                rejected.append({'platform': a[2], 'groups': [a[3], b[3]], 'basis': reason})
                continue
        members = left | right
        for member in members:
            components[member] = members
        accepted.append({'platform': a[2], 'groups': [a[3], b[3]], 'basis': reason, 'score': round(score, 4)})
    mapping = {}
    for key, members in components.items():
        primary = min(members, key=lambda member: (-fragments[member].pulses, member))
        if key != primary:
            mapping[key] = primary[-1]
    return mapping, accepted, rejected


def map_links(links, mapping):
    return {key: mapping.get(key[:3] + (link['local_cluster_id'],), link['local_cluster_id'])
            for key, link in links.items()}


def evaluate(predictions, labeled_rows, metrics, eligible=None, scope_size=3):
    cells = Counter()
    total = 0
    for key, target, count in labeled_rows:
        if eligible is not None and key not in eligible:
            continue
        total += count
        group = predictions[key]
        if group:
            cells[key[:scope_size], group, target] += count
    report = metrics(cells)
    pairs = lambda n: n * (n - 1) // 2
    by_group, by_target = Counter(), Counter()
    for (scope, group, target), count in cells.items():
        by_group[scope, group] += count
        by_target[scope, target] += count
    true_pairs = sum(pairs(n) for n in cells.values())
    report.update(eligible_pulses=total, coverage=report['assigned_pulses_evaluated'] / total,
                  false_merge_pairs=sum(pairs(n) for n in by_group.values()) - true_pairs,
                  false_split_pairs=sum(pairs(n) for n in by_target.values()) - true_pairs)
    del report['groups_per_true_target']
    return report


def read_truth(output, keys):
    label_path = output / 'group_validation/train_sequence_truth.csv'
    fields = 'scenario_id data_version platform_id sequence_id'.split()
    labeled_rows = []
    with label_path.open(newline='') as source:
        for row in csv.DictReader(source):
            key = tuple(row[field] for field in fields)
            if key in keys:
                labeled_rows.append((key, row['target_id'], int(row['pulse_count'])))
    if {key for key, _, _ in labeled_rows} != keys:
        raise ValueError('Some selected sequences have no truth labels')
    return labeled_rows


def run_window(output, window):
    sys.path.append(str(ROOT.parent / '代码'))
    from evaluate_local_groups import metrics

    start, end = WINDOWS[window]
    config = GroupConfig(**json.loads((ROOT / 'config/group_config.json').read_text()))
    positions = ObserverPaths(output / 'group_inputs/train_platform_positions.csv')
    sequences = read_sequences(output / 'pri/train/pri_sequences.csv', positions, start, end)
    groups, links = associate(sequences, config)
    print(f'{window}: original grouping complete ({len(sequences)} sequences)', flush=True)
    fragments = summarize(groups)
    votes, totals = cross_platform_evidence(output / 'fused/train/cross_platform_events.csv', links, fragments)
    local, cross, conflicts = candidates(fragments, votes, totals)
    predictions = {'baseline': map_links(links, {})}
    edits, rejected, merge_maps = {}, {}, {}
    for name, edges, guard in [('stitch_local', local, None), ('stitch_cross_platform', cross, None),
                               ('stitch_combined', local + cross, None),
                               ('stitch_cross_platform_overlap_guard', cross, config),
                               ('stitch_combined_overlap_guard', local + cross, config)]:
        mapping, edits[name], rejected[name] = join_fragments(edges, conflicts, fragments, guard)
        predictions[name] = map_links(links, mapping)
        merge_maps[name] = [dict(platform=key[2], old_group=key[3], new_group=value)
                            for key, value in sorted(mapping.items())]
    wider = {
        'wider_bearing': replace(config, bearing_base_gate_degrees=config.bearing_base_gate_degrees * 1.5,
                                 maximum_bearing_gate_degrees=config.maximum_bearing_gate_degrees * 1.5,
                                 gap_gate_growth_degrees_per_second=config.gap_gate_growth_degrees_per_second * 1.5),
        'wider_bearing_and_waveform': replace(config, bearing_base_gate_degrees=config.bearing_base_gate_degrees * 1.5,
                                             maximum_bearing_gate_degrees=config.maximum_bearing_gate_degrees * 1.5,
                                             rf_family_gate_mhz=config.rf_family_gate_mhz * 1.5,
                                             pw_family_gate_us=config.pw_family_gate_us * 1.5,
                                             maximum_pri_ratio=config.maximum_pri_ratio * 1.5,
                                             maximum_cost=config.maximum_cost * 1.25),
    }
    for name, variant in wider.items():
        _, alternative = associate(sequences, variant)
        predictions[name] = map_links(alternative, {})

    labeled_rows = read_truth(output, set(links))
    if sum(n for _, _, n in labeled_rows) != sum(s.count for s in sequences):
        raise ValueError('Truth and observation pulse counts differ')
    common = {key for key in links if all(arm[key] for arm in predictions.values())}
    results = {}
    for name, arm in predictions.items():
        results[name] = {'all': evaluate(arm, labeled_rows, metrics),
                         'common_assignments': evaluate(arm, labeled_rows, metrics, common),
                         'accepted_merges': edits.get(name, []), 'rejected_for_overlap': rejected.get(name, []),
                         'group_remapping': merge_maps.get(name, [])}
    print(f'{window}: evaluation complete', flush=True)
    return results


def replay_fusion(output, b_path, group_remapping):
    from fuse_platforms import load_observations
    from src.b_assisted import AssistanceConfig, assist
    from src.dataio import InputConfig, read_b_tracks
    from src.fusion import Event, FusionConfig, assign_single_observations, associate_events
    from src.guarded_assistance import GuardConfig, guard

    def load_config(name):
        return json.loads((ROOT / 'config' / name).read_text())

    observations, _ = load_observations(output / 'pri/train/pri_sequences.csv',
                                        output / 'grouped/train/bearing_observations.csv',
                                        output / 'group_inputs/train_platform_positions.csv')
    mapping = {(row['platform'], row['old_group']): row['new_group'] for row in group_remapping}
    scenes = {o.sequence.scope[:2] for o in observations}
    if len(scenes) != 1:
        raise ValueError('Fusion replay requires one scene/version')
    changed = 0
    for o in observations:
        group = mapping.get((o.platform, o.local_group))
        if group:
            o.local_group = group
            changed += o.sequence.count
    by_key = {o.key: o for o in observations}
    events = []
    with (output / 'fused/train/cross_platform_events.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            scope = row['scenario_id'], row['data_version']
            members = tuple(by_key[scope + (p, sid)] for p, sid in zip(
                row['platform_ids'].split('|'), row['sequence_ids'].split('|'), strict=True))
            geom = {'east_m': float(row['coarse_east_m']), 'north_m': float(row['coarse_north_m']),
                    'error_m': float(row['geometry_error_proxy_m'])}
            events.append(Event(members, int(row['time_ns']), geom, float(row['association_cost']), row['event_id']))
    fusion = FusionConfig(**load_config('fusion_config.json'))
    fusion.validate()
    objects = associate_events(events, fusion)
    baseline = {o.key: {'object_id': e.object_id, 'event_id': e.event_id}
                for e in events for o in e.members}
    assign_single_observations(observations, objects, baseline, fusion)
    print(f'Fusion replay ({changed} pulses remapped): A-only complete', flush=True)
    tracks = read_b_tracks(b_path, InputConfig.from_dict(load_config('input_units.json')))
    proposals, _, diagnostics, _ = assist(observations, tracks, baseline,
                                           AssistanceConfig(**load_config('b_assisted_config.json')))
    seeds = {row['key'] for row in diagnostics if row['seed']}
    protected, _ = guard(observations, baseline, proposals, seeds, GuardConfig(**load_config('b_guard_config.json')))
    print(f'Fusion replay ({changed} pulses remapped): B-assisted and guarded complete', flush=True)
    return {name: {key: link['object_id'] for key, link in links.items()}
            for name, links in [('a_only', baseline), ('b_assisted', proposals), ('guarded', protected)]}


def assess_fusion(output, b_path, group_remapping):
    sys.path.append(str(ROOT.parent / '代码'))
    from evaluate_local_groups import metrics

    variants = {'baseline': [], 'stitched': group_remapping}
    with ProcessPoolExecutor(max_workers=len(variants)) as pool:
        futures = {name: pool.submit(replay_fusion, output, b_path, mapping) for name, mapping in variants.items()}
        predictions = {name: future.result() for name, future in futures.items()}
    keys = set(predictions['baseline']['a_only'])
    labeled_rows = read_truth(output, keys)
    directories = {'a_only': 'fused', 'b_assisted': 'fused_b_assisted', 'guarded': 'fused_b_assisted_guarded'}
    report = {}
    for stage, dirname in directories.items():
        path = output / dirname / 'train/sequence_object_links.csv'
        fields = 'scenario_id data_version platform_id sequence_id'.split()
        with path.open(newline='') as source:
            cached = {tuple(r[k] for k in fields): r['object_id'] for r in csv.DictReader(source)}
        baseline = predictions['baseline'][stage]
        if cached != baseline:
            raise ValueError(f'Replayed baseline does not reproduce saved {stage} identities')
        common = {key for key in keys if all(p[stage][key] for p in predictions.values())}
        report[stage] = {name: {'all': evaluate(arms[stage], labeled_rows, metrics, scope_size=2),
                                'common_assignments': evaluate(arms[stage], labeled_rows, metrics, common, scope_size=2)}
                         for name, arms in predictions.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intermediates', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--window', choices=tuple(WINDOWS), action='append')
    parser.add_argument('--replay-fusion', metavar='ARM', help='Replay whole-scene fusion for an arm in the existing report')
    parser.add_argument('--b-file', type=Path)
    args = parser.parse_args()
    if args.replay_fusion:
        if not args.b_file or args.window:
            parser.error('--replay-fusion requires --b-file and uses the whole-scene result')
        report = json.loads(args.report.read_text())
        mapping = report['results']['whole_scene'][args.replay_fusion]['group_remapping']
        report['fusion_replay'] = {'arm': args.replay_fusion, 'cached_baseline_object_ids_reproduced': True,
                                   'results': assess_fusion(args.intermediates.resolve(), args.b_file, mapping)}
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        return
    windows = args.window or list(WINDOWS)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {window: pool.submit(run_window, args.intermediates.resolve(), window) for window in windows}
        results = {window: future.result() for window, future in futures.items()}
    report = {'evaluation_scope': 'Labeled TRAIN pulses with frozen PRI sequences. Grouping is restarted '
              'at the beginning of each bounded window. A-only cross-platform events are precomputed '
              'without local grouping, with evidence restricted to selected sequences inside each window. '
              'Sequence truth is loaded only after candidate group assignments.',
              'time_windows_seconds': {window: WINDOWS[window] for window in windows}, 'results': results}
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
