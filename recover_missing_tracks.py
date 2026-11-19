#!/usr/bin/env python3
"""Offline recovery of residual object IDs and A-only position checks on P1A."""

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from statistics import median

from recover_objects import SECOND, build_plan, export, load_sequences, records, sequence_key
from recover_objects_v2 import b_fragment_scores, joint_fragments
from src.b_assisted import AssistanceConfig, BearingPrior, bearing_candidates
from src.dataio import InputConfig, read_b_tracks
from src.fusion import FusionConfig, geometry
from fuse_platforms import load_observations
from validate_missing_objects import assignments, guarded_single_platform, measure

HERE = Path(__file__).resolve().parent
MAX_B_GAP = 3 * SECOND


def track_families(tracks, frame):
    """Group near-overlapping B references as competing explanations, not labels."""
    compatible = set()
    for a, b in combinations(sorted(tracks), 2):
        group_a, group_b = tracks[a], tracks[b]
        errors = []
        for obs in group_a.observations:
            other = group_b.position(obs.time_ns, MAX_B_GAP)
            if other is not None:
                x, y = frame.project((*obs.position[:2], 0.))
                xx, yy = frame.project((*other[:2], 0.))
                errors.append(math.hypot(x-xx, y-yy))
        if len(errors) < .8 * len(group_a.observations):
            continue
        errors.sort()
        if median(errors) <= 500 and errors[math.ceil(.9*len(errors))-1] <= 800:
            compatible.add((a,b))
    families = []
    for track in sorted(tracks):
        matches = [f for f in families if all(tuple(sorted((track, t))) in compatible for t in f)]
        if len(matches) == 1:
            matches[0].append(track)
        else:
            families.append([track])
    return {t:tuple(f) for f in families for t in f}


def b_links(intermediates, split):
    path = intermediates / 'ab_matched_2d' / split / 'object_track_links.csv'
    return {r['object_id']:r['track_id'] for r in records(path) if r['track_id']}


def load_context(paths, split, b_path, config):
    seqs = load_sequences(paths, split)
    obs, frames = load_observations(paths/'pri'/split/'pri_sequences.csv',
                                    paths/'grouped'/split/'bearing_observations.csv',
                                    paths/'group_inputs'/f'{split}_platform_positions.csv')
    frame, = frames.values()
    tracks = {k[2]:t for k,t in read_b_tracks(b_path, config).items()}
    families = track_families(tracks,frame)
    priors = [BearingPrior(t,AssistanceConfig()) for t in tracks.values()]
    return seqs, {o.key:o for o in obs}, tracks, families, priors, frame


def signature(s):
    return s.model, round(s.rf/70), round(s.pw/.2), round(math.log(s.pri)/.11)


def build_profile(keys, lookup):
    supports = defaultdict(set)
    for key in keys:
        s = lookup[key]
        supports[signature(s)].add((s.key[2],s.time//SECOND))
    return {motif for motif,bins in supports.items() if len(bins) >= 2}


def fits_profile(s, profile):
    model, rf, pw, pri = signature(s)
    # The exact motif or a single adjacent bin is accepted. A candidate cannot
    # simultaneously rely on broad tolerances in every waveform coordinate.
    return ((model,rf,pw,pri) in profile or any(motif in profile for motif in (
            (model,rf-1,pw,pri),(model,rf+1,pw,pri),(model,rf,pw-1,pri),(model,rf,pw+1,pri),
            (model,rf,pw,pri-1),(model,rf,pw,pri+1))))


def eligible_seeds(fragments, sequences, used, family, protected):
    lookup = {s.key:s for s in sequences}
    joint = joint_fragments(fragments,sequences,used)
    single = guarded_single_platform(fragments,used,lookup)
    seeds, provenance = {}, {}
    for mode, options in [('joint', {t:[s.key for s in r['members']] for t,r in joint.items()}),
                          ('single',single)]:
        for track,keys in options.items():
            group = family[track]
            if group in seeds:
                continue
            seeds[group] = set(keys) - protected.keys()
            provenance[group] = mode
    return seeds, provenance


def extend(sequences, obs, priors, family, seeds, protected):
    profiles = {f:build_profile(keys,{s.key:s for s in sequences}) for f,keys in seeds.items()}
    seed_keys = set().union(*seeds.values())
    candidates = defaultdict(list)
    cfg = AssistanceConfig()
    for s in sequences:
        if s.object_id or not s.group or s.key in protected or s.key in seed_keys:
            continue
        options = {f for f,profile in profiles.items() if fits_profile(s,profile)}
        if not options:
            continue
        distances = {}
        for k,v in bearing_candidates(obs[s.key],priors,cfg).items():
            f = family[k[2]]
            distances[f] = min(distances.get(f,math.inf),v['residual'])
        ranked = sorted((error,f) for f,error in distances.items())
        if len(ranked) < 2:
            continue
        error, f = ranked[0]
        if f in options and error <= .75 and ranked[1][0] - error >= .30:
            candidates[f,s.group_key].append(s)
    enriched = {f:set(keys) for f,keys in seeds.items()}
    for (f,_), items in candidates.items():
        segment = []
        for s in sorted(items,key=lambda s:s.time):
            if segment and s.time - segment[-1].time > 6 * SECOND:
                if len({q.time//SECOND for q in segment}) >= 3:
                    enriched[f].update(q.key for q in segment)
                segment = []
            segment.append(s)
        if len({q.time//SECOND for q in segment}) >= 3:
            enriched[f].update(q.key for q in segment)
    return enriched


def infer(seqs, obs, priors, family, links, paths, split, b_path, config, refill=None):
    used = {tid for t in links.values() for tid in family[t]}
    protected = refill if refill is not None else build_plan(seqs,()).refill
    fragments = b_fragment_scores(seqs,paths,split,b_path,config)
    seeds, provenance = eligible_seeds(fragments,seqs,used,family,protected)
    enriched = extend(seqs,obs,priors,family,seeds,protected)
    return protected,seeds,enriched,provenance


def family_distance(east, north, stamp, tracks, family, frame):
    distances = []
    for track in family:
        pos = tracks[track].position(stamp,MAX_B_GAP)
        if pos is not None:
            x,y = frame.project((*pos[:2],0.))
            distances.append(math.hypot(x-east,north-y))
    return min(distances) if distances else None


def cross_platform_repair(sequences, obs, tracks, priors, families, seeds, expanded, protected, frame):
    """Recover short residual fragments when independent A bearings agree twice."""
    lookup = {s.key:s for s in sequences}
    profiles = {f:build_profile(keys,lookup) for f,keys in seeds.items()}
    locked = set().union(protected,*expanded.values())
    by_family = {}
    for family, keys in expanded.items():
        by_platform = defaultdict(list)
        for key in keys:
            by_platform[obs[key].platform].append(obs[key])
        by_family[family] = {}
        for platform, observations in by_platform.items():
            observations.sort(key=lambda o:o.time)
            by_family[family][platform] = ([o.time for o in observations],observations)
    votes = defaultdict(dict)
    ac = AssistanceConfig()
    gc = replace(FusionConfig(),minimum_crossing_angle_degrees=10.)
    for seq in sequences:
        if seq.object_id or not seq.group or seq.key in locked:
            continue
        eligible = {f for f,profile in profiles.items() if fits_profile(seq,profile)}
        if not eligible:
            continue
        candidates = {}
        for k,r in bearing_candidates(obs[seq.key],priors,ac).items():
            f = families[k[2]]
            candidates[f] = min(r['residual'],candidates.get(f,math.inf))
        options = sorted((error,f) for f,error in candidates.items())
        if len(options) < 2:
            continue
        error,family = options[0]
        if family not in eligible or error > .75 or options[1][0]-error < .30:
            continue
        a = obs[seq.key]
        for platform,(times,anchors) in by_family[family].items():
            if platform == a.platform:
                continue
            left = bisect_left(times,a.time-SECOND)
            right = bisect_right(times,a.time+SECOND)
            for b in anchors[left:right]:
                other = lookup[b.key]
                if (seq.model != other.model or abs(seq.rf-other.rf)>100. or abs(seq.pw-other.pw)>.45
                        or abs(seq.pri-other.pri)>.28*min(seq.pri,other.pri)):
                    continue
                pos = geometry((a,b),gc)
                if pos is None or pos['error_m'] > 600.:
                    continue
                t = (a.time+b.time)//2
                distance = family_distance(pos['east_m'],pos['north_m'],t,tracks,family,frame)
                if distance is not None and distance <= 250.:
                    key = family,seq.group_key,platform
                    votes[key][seq.key] = a.time
    repaired = {family:set(keys) for family,keys in expanded.items()}
    evidence_windows = defaultdict(set)
    for (family,group,platform), evidence in votes.items():
        chunk = []
        for key,when in sorted(evidence.items(),key=lambda e:e[1]):
            if chunk and when-chunk[-1][1] > 2*SECOND:
                enough = {t//(SECOND//2) for _,t in chunk}
                if len(enough)>=2 and chunk[-1][1]-chunk[0][1]>=SECOND//4:
                    repaired[family].update(k for k,_ in chunk)
                    evidence_windows[family].update(enough)
                chunk = []
            chunk.append((key,when))
        enough = {t//(SECOND//2) for _,t in chunk}
        if len(enough)>=2 and chunk[-1][1]-chunk[0][1]>=SECOND//4:
            repaired[family].update(k for k,_ in chunk)
            evidence_windows[family].update(enough)
    return repaired,evidence_windows


def positions(keys, obs, tracks, family, frame):
    windows = defaultdict(lambda:defaultdict(list))
    for key in keys:
        o = obs[key]
        windows[o.time//SECOND][o.platform].append(o)
    cfg = replace(FusionConfig(),minimum_crossing_angle_degrees=10.)
    fixed = {}

    def make_point(fitted, stamp, platform_count):
        return {'time_seconds':stamp/SECOND,'east_km':fitted['east_m']/1000,
                'north_km':fitted['north_m']/1000,'platform_count':platform_count,
                'uncertainty_proxy_m':fitted['error_m'],
                'nearest_family_B_distance_m':family_distance(fitted['east_m'],fitted['north_m'],stamp,tracks,family,frame)}

    for second,platforms in sorted(windows.items()):
        if len(platforms) < 2:
            continue
        # One observed A direction per platform, chosen as the angular medoid.
        # No B coordinates participate in the position computation.
        selected = []
        for items in platforms.values():
            x = sum(o.direction[0] for o in items)
            y = sum(o.direction[1] for o in items)
            selected.append(max(items,key=lambda o:x*o.direction[0] + y*o.direction[1]))
        fitted = geometry(selected,cfg)
        if fitted is None or fitted['error_m'] > 600.:
            continue
        stamp = round(median(o.time for o in selected))
        fixed[second] = make_point(fitted,stamp,len(selected))
    platforms = defaultdict(list)
    for key in keys:
        platforms[obs[key].platform].append(obs[key])
    sliding = {}
    for aa,bb in combinations(platforms.values(),2):
        bb = sorted(bb,key=lambda o:o.time)
        times = [o.time for o in bb]
        for a in aa:
            for b in bb[bisect_left(times,a.time-SECOND):bisect_right(times,a.time+SECOND)]:
                stamp = (a.time+b.time)//2
                sec = stamp//SECOND
                dt = abs(a.time-b.time)
                if sec in fixed or (sec in sliding and dt>=sliding[sec][0]):
                    continue
                fitted = geometry((a,b),cfg)
                if fitted is not None and fitted['error_m']<=600.:
                    sliding[sec] = (dt,make_point(fitted,stamp,2))
    points = fixed | {k:v for k,(_,v) in sliding.items()}
    return [points[k] for k in sorted(points)], sum(len(v)>1 for v in windows.values())


def validate(paths,b_path,config,targets):
    seqs,obs,tracks,families,priors,frame = load_context(paths,'train',b_path,config)
    truth = defaultdict(Counter)
    totals = Counter()
    for row in records(paths/'group_validation/train_sequence_truth.csv'):
        n = int(row['pulse_count'])
        truth[sequence_key(row)][row['target_id']] += n
        totals[row['target_id']] += n
    dominant = assignments(seqs,truth)
    links = b_links(paths,'train')
    result = {}
    for target in targets:
        removed = {o for o,t in dominant.items() if t==target}
        kept = {o:t for o,t in links.items() if o not in removed}
        masked = [replace(s,object_id='') if s.object_id in removed else s for s in seqs]
        keep,seeds,ext,_ = infer(masked,obs,priors,families,kept,paths,'train',b_path,config)
        repaired,_ = cross_platform_repair(masked,obs,tracks,priors,families,seeds,ext,keep,frame)
        n_available = sum(truth[s.key][target] for s in masked if not s.object_id)
        result[target] = {name:measure({'+'.join(f):keys for f,keys in run.items()},truth,target,totals[target],n_available)
                          for name,run in [('seed_only',seeds),('expanded',ext),('short_cross_platform',repaired)]}
        print(target,{label:(r['candidate_count'],r['best']['target_pulses'] if r['best'] else 0)
                      for label,r in result[target].items()},flush=True)
    return {'targets':result,'note':'Frozen train upstream with whole-target object IDs removed. '
                                     'Prediction uses A observations and public B positions only.'}


def run_test(args,config):
    seqs,obs,tracks,families,priors,frame = load_context(args.intermediates,'test',args.b_file,config)
    links = b_links(args.intermediates,'test')
    refill,seeds,ext,provenance = infer(seqs,obs,priors,families,links,args.intermediates,'test',args.b_file,config)
    repaired,windows = cross_platform_repair(seqs,obs,tracks,priors,families,seeds,ext,refill,frame)
    lookup = {s.key:s for s in seqs}
    edits, trajectories, report = dict(refill),[],{}
    for i,(family,keys) in enumerate(repaired.items(),1):
        oid = f'OR{i:05d}'
        for key in keys:
            if key in edits:
                raise ValueError(f'Recovery overlap for {key}')
            edits[key] = oid
        points, common = positions(keys,obs,tracks,family,frame)
        trajectories.extend({'object_id':oid,**r} for r in points)
        errors = [r['nearest_family_B_distance_m'] for r in points if r['nearest_family_B_distance_m'] is not None]
        report[oid] = {'B_comparison_family':family,'seed_mode':provenance[family],
                       'seed_pulses':sum(lookup[k].count for k in seeds[family]),
                       'expanded_pulses':sum(lookup[k].count for k in keys),
                       'short_cross_platform_pulses':sum(lookup[k].count for k in keys-ext[family]),
                       'short_cross_platform_windows':sorted(t/2 for t in windows[family]),
                       'pulses_by_platform':{p:sum(lookup[k].count for k in keys if k[2]==p)
                                             for p in sorted({k[2] for k in keys})},
                       'seconds_with_multiplatform_support':common,'A_position_points':len(points),
                       'A_position_to_nearest_family_B_median_m':median(errors) if errors else None}
    print(json.dumps(report,ensure_ascii=False),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'新增A定位点.csv').open('w',newline='') as stream:
        cols = ['object_id','time_seconds','east_km','north_km','platform_count',
                'uncertainty_proxy_m','nearest_family_B_distance_m']
        writer = csv.DictWriter(stream,fieldnames=cols)
        writer.writeheader()
        writer.writerows(trajectories)
    return {'families':sorted(set(families.values())), 'new_objects':report,
            'export':export(args.intermediates,args.baseline,args.output/'候选CSV',edits)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intermediates',type=Path,required=True)
    p.add_argument('--b-file',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate',nargs='+',metavar='TARGET')
    args = p.parse_args()
    config = InputConfig.from_dict(json.loads((HERE/'config/input_units.json').read_text()))
    if args.validate:
        report = validate(args.intermediates,args.b_file,config,args.validate)
        name = '整目标对照.json'
    else:
        if args.baseline is None:
            p.error('--baseline is required to export test CSVs')
        report = run_test(args,config)
        name = '补回结果.json'
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
