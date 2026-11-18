"""Offline B trajectory priors for A identity association, without truth columns.

B positions constrain bearings; they never become A triangulation outputs.
The number of B tracks is not a requested number of A objects.
"""
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .fusion import LocalFrame


@dataclass(frozen=True)
class AssistanceConfig:
    bearing_base_gate_degrees: float = 1.25
    bearing_error_multiplier: float = 3.
    b_position_margin_m: float = 150.
    maximum_b_gap_seconds: float = 3.
    smoothing_window_seconds: float = 6.
    maximum_range_m: float = 180000.
    maximum_bearing_gate_degrees: float = 5.
    seed_ambiguity_margin: float = .20
    context_ambiguity_margin: float = .15
    seed_maximum_cost: float = .6
    minimum_anchor_sequences: int = 5
    minimum_anchor_pulses: int = 128
    local_context_seconds: float = 8.
    local_context_dominance: float = .85
    baseline_mapping_dominance: float = .90
    waveform_rf_gate_mhz: float = 250.
    waveform_pw_gate_us: float = 1.5
    waveform_maximum_pri_ratio: float = 3.
    seed_requires_multiplatform: bool = True

    def validate(self):
        for key, value in asdict(self).items():
            if key == 'seed_requires_multiplatform':
                if not isinstance(value,bool):raise ValueError('seed_requires_multiplatform must be boolean')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('minimum_anchor_sequences', 'minimum_anchor_pulses'):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key} must be an integer')
        if not .5 < self.baseline_mapping_dominance <= 1 or not .5 < self.local_context_dominance <= 1:
            raise ValueError('Dominance must be in (.5,1]')
        if self.waveform_maximum_pri_ratio <= 1:
            raise ValueError('PRI ratio must exceed one')


class BearingPrior:
    """Smooth only within contiguous observed B spans; never extrapolate."""
    def __init__(self, track, cfg):
        self.key, self.times = track.key, track.times_ns
        self.max_gap = round(cfg.maximum_b_gap_seconds * 1e9)
        raw = [LocalFrame.ecef(*o.position) for o in track.observations]
        span = round(cfg.smoothing_window_seconds * 1e9)
        self.xyz = []
        # Centered local linear regression reduces B observation noise. Bounds
        # stop at long missing spans; a window cannot smooth across such gaps.
        begin = 0
        while begin < len(raw):
            stop = begin + 1
            while stop < len(raw) and self.times[stop] - self.times[stop-1] <= self.max_gap:
                stop += 1
            for i in range(begin, stop):
                left = max(begin, bisect.bisect_left(self.times, self.times[i] - span//2))
                right = min(stop, bisect.bisect_right(self.times, self.times[i] + span//2))
                xs = [(self.times[j] - self.times[i])/1e9 for j in range(left, right)]
                mx = sum(xs)/len(xs)
                var = sum((x-mx)**2 for x in xs)
                result = []
                for axis in range(3):
                    ys = [raw[j][axis] for j in range(left, right)]
                    my = sum(ys)/len(ys)
                    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var if var > 1e-12 else 0.
                    result.append(my-slope*mx)
                self.xyz.append(tuple(result))
            begin = stop

    def position(self, time):
        j = bisect.bisect_left(self.times, time)
        if j < len(self.times) and self.times[j] == time:
            return self.xyz[j]
        if j == 0 or j == len(self.times) or self.times[j]-self.times[j-1] > self.max_gap:
            return None
        f = (time-self.times[j-1])/(self.times[j]-self.times[j-1])
        return tuple(a+f*(b-a) for a,b in zip(self.xyz[j-1],self.xyz[j]))


def bearing_candidates(observation, priors, cfg):
    s = observation.sequence
    lon,lat = map(math.radians,s.position[:2])
    east = (-math.sin(lon),math.cos(lon),0.)
    north = (-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat))
    origin = LocalFrame.ecef(*s.position)
    result = {}
    for prior in priors:
        if prior.key[:2] != s.scope[:2]:
            continue
        point = prior.position(s.time)
        if point is None:
            continue
        delta = tuple(b-a for a,b in zip(origin,point))
        dx,dy = (sum(a*b for a,b in zip(delta,axis)) for axis in (east,north))
        distance = math.hypot(dx,dy)
        if distance <= 1. or distance > cfg.maximum_range_m:
            continue
        angle = math.degrees(math.atan2(dx,dy)) % 360
        residual = abs((angle-s.doa+180)%360-180)
        gate = min(cfg.maximum_bearing_gate_degrees,cfg.bearing_base_gate_degrees +
                   cfg.bearing_error_multiplier*s.error + math.degrees(math.atan2(cfg.b_position_margin_m,distance)))
        # Keep the runner-up outside the gate, too: one passing candidate need
        # not be unique evidence when another lies just outside the boundary.
        result[prior.key] = {'cost': (residual/gate)**2, 'residual': residual, 'gate': gate}
    return result


def unique_choice(scores, margin, maximum=1.):
    ranked = sorted((cost,key) for key,cost in scores.items())
    if ranked and ranked[0][0] <= maximum and (len(ranked)==1 or ranked[1][0]-ranked[0][0] >= margin):
        return ranked[0][1],ranked[0][0]
    return None


def waveform_compatible(a,b,cfg):
    return (abs(a.rf-b.rf) <= cfg.waveform_rf_gate_mhz and abs(a.pw-b.pw) <= cfg.waveform_pw_gate_us
            and max(a.pri,b.pri)/min(a.pri,b.pri) <= cfg.waveform_maximum_pri_ratio)


def assist(observations, tracks, baseline, cfg, on_progress=None):
    """Use confident B bearing anchors and frozen local context; retain A fallback.

    Seeds never propagate recursively. An unused B track creates no A object.
    The baseline is evidence only; mixed groups may have multiple identities.
    """
    cfg.validate()
    by_scene = defaultdict(list)
    for track in tracks.values():
        if len(track.observations) >= 3:
            by_scene[track.key[:2]].append(BearingPrior(track,cfg))
    candidates, seeds = {}, {}
    event_members=defaultdict(list)
    for i,o in enumerate(observations):
        cs = bearing_candidates(o,by_scene[o.sequence.scope[:2]],cfg)
        candidates[o.key] = cs
        choice = unique_choice({k:v['cost'] for k,v in cs.items()}, cfg.seed_ambiguity_margin,cfg.seed_maximum_cost)
        if choice and not cfg.seed_requires_multiplatform:
            seeds[o.key] = choice
        event=baseline[o.key].get('event_id','')
        if event:event_members[o.sequence.scope[:2]+(event,)].append(o)
        if on_progress and i and i % 40000 == 0:
            on_progress({'stage':'B_bearing_candidates','sequences':i,'seeds':len(seeds)})
    for items in event_members.values():
        if len({o.platform for o in items})<2:continue
        common=set.intersection(*(set(candidates[o.key]) for o in items))
        scores={k:sum(candidates[o.key][k]['cost'] for o in items)/len(items) for k in common}
        picked=unique_choice(scores,cfg.seed_ambiguity_margin,cfg.seed_maximum_cost)
        if picked and max(candidates[o.key][picked[0]]['cost'] for o in items)<=1.:
            for o in items:seeds[o.key]=picked
    evidence, evidence_pulses = Counter(),Counter()
    for o in observations:
        if o.key in seeds:
            k = seeds[o.key][0]; evidence[k] += 1; evidence_pulses[k] += o.sequence.count
    active = {k for k,n in evidence.items() if n >= cfg.minimum_anchor_sequences and evidence_pulses[k] >= cfg.minimum_anchor_pulses}
    seeds = {k:v for k,v in seeds.items() if v[0] in active}
    anchor_objects = {k:f'OB{i:05d}' for i,k in enumerate(sorted(active),1)}
    # Waveform family evidence is learned only from geometrically anchored A
    # observations, not from a B class or physical target ID. Median centers
    # resist occasional spurious intersections and reject unrelated emitters.
    from statistics import median
    families=defaultdict(list)
    for o in observations:
        if o.key in seeds:families[seeds[o.key][0]].append(o.sequence)
    profiles={k:(median(s.rf for s in seqs),median(s.pw for s in seqs),median(s.pri for s in seqs))
              for k,seqs in families.items()}
    def radio_cost(s,k):
        rf,pw,pri=profiles[k]
        if (abs(s.rf-rf)>cfg.waveform_rf_gate_mhz or abs(s.pw-pw)>cfg.waveform_pw_gate_us or
            max(s.pri,pri)/min(s.pri,pri)>cfg.waveform_maximum_pri_ratio):return None
        return .3*(abs(s.rf-rf)/cfg.waveform_rf_gate_mhz)**2+.2*(abs(s.pw-pw)/cfg.waveform_pw_gate_us)**2
    # A spurious geometric event must not bypass the anchor's waveform check.
    # Build once from the original anchors, then freeze this filtered seed set.
    observation_by_key={o.key:o for o in observations}
    seeds={key:choice for key,choice in seeds.items()
           if radio_cost(observation_by_key[key].sequence,choice[0]) is not None}
    evidence=Counter(v[0] for v in seeds.values())
    evidence_pulses=Counter()
    for key,(bkey,_) in seeds.items():evidence_pulses[bkey]+=observation_by_key[key].sequence.count
    active={k for k,n in evidence.items() if n>=cfg.minimum_anchor_sequences and evidence_pulses[k]>=cfg.minimum_anchor_pulses}
    seeds={k:v for k,v in seeds.items() if v[0] in active}
    anchor_objects={k:f'OB{i:05d}' for i,k in enumerate(sorted(active),1)}
    # All future context uses these original seeds, never newly inferred links.
    local, local_times = defaultdict(list),{}
    votes = defaultdict(Counter)
    for o in observations:
        if o.key not in seeds:
            continue
        bkey,_ = seeds[o.key]
        if o.local_group:
            local[o.sequence.scope+(o.local_group,)].append(o)
        old = baseline[o.key]['object_id']
        if old:
            votes[o.sequence.scope[:2]+(old,)][bkey] += o.sequence.count
    for k,items in local.items():
        items.sort(key=lambda o:o.time);local_times[k] = [o.time for o in items]
    baseline_map = {}
    for k,v in votes.items():
        best,n = v.most_common(1)[0]
        if n/sum(v.values()) >= cfg.baseline_mapping_dominance:
            baseline_map[k] = best
    links,diagnostics = {},[]
    for o in observations:
        s=o.sequence;cs=candidates[o.key];choice=seeds.get(o.key)
        basis='b_bearing_seed' if choice else ''
        context = Counter()
        local_key=s.scope+(o.local_group,)
        if o.local_group and local_key in local:
            items,ts=local[local_key],local_times[local_key]
            radius=cfg.local_context_seconds*1e9
            left,right=bisect.bisect_left(ts,s.time-radius),bisect.bisect_right(ts,s.time+radius)
            # One vote per temporal bin avoids counting arbitrary PRI splitting
            # as independent identity evidence.
            bins={}
            for near in items[left:right]:
                if not waveform_compatible(s,near.sequence,cfg):continue
                key=seeds[near.key][0];b=(key,near.time//200_000_000)
                weight=min(near.sequence.count,128)/(1+abs(near.time-s.time)/1e9)
                bins[b]=max(bins.get(b,0.),weight)
            for (key,_),weight in bins.items():context[key]+=weight
        if not choice:
            scores={k:v['cost']+radio_cost(s,k) for k,v in cs.items() if k in active and radio_cost(s,k) is not None}
            # Geometry remains a hard gate. Context only breaks close ties.
            if context:
                scale=sum(context.values())
                scores={k:v-.5*context[k]/scale for k,v in scores.items()}
            picked=unique_choice(scores,cfg.context_ambiguity_margin if context else cfg.seed_ambiguity_margin,
                                 1. if context else cfg.seed_maximum_cost)
            if picked and cs[picked[0]]['cost'] <= 1.:
                choice=picked;basis='b_bearing_with_context' if context else 'b_bearing_unique'
        if not choice and context:
            key,weight=context.most_common(1)[0]
            # Fill a missing B time sample only with strongly consistent local
            # anchors, and never overrule an available incompatible bearing.
            if weight/sum(context.values()) >= cfg.local_context_dominance and (key not in cs) and radio_cost(s,key) is not None:
                choice=(key,0.);basis='b_local_gap_context'
        old=baseline[o.key]['object_id'];mapped=baseline_map.get(s.scope[:2]+(old,)) if old else None
        if not choice and mapped and (mapped not in cs or cs[mapped]['cost'] <= 1.) and radio_cost(s,mapped) is not None:
            choice=(mapped,cs[mapped]['cost'] if mapped in cs else 0.)
            basis='b_anchor_A_object_context'
        if choice:
            key,cost=choice;oid=anchor_objects[key]
        else:
            key=None;cost=None
            # Unanchored A objects survive; do not force them into a B track.
            oid='OA'+old[1:] if old and mapped is None else ''
            basis='A_only_fallback' if oid else 'unmatched_B_ambiguous_or_missing'
        links[o.key]={'object_id':oid,'event_id':'','basis':basis,'cost':cost}
        diagnostics.append({'key':o.key,'track_id':key[2] if key else '',
            'residual':cs[key]['residual'] if key in cs else None,'gate':cs[key]['gate'] if key in cs else None,
            'seed':o.key in seeds,'baseline_object_id':old,'basis':basis})
    stats={'input_B_tracks':len(tracks),'B_tracks_with_A_anchors':len(active),
        'seed_sequences':len(seeds),'baseline_objects_mapped_to_B':len(baseline_map),
        'basis_sequence_counts':dict(Counter(v['basis'] for v in links.values())),
        'count_constraint_used':False,'recursive_context_used':False}
    return links,anchor_objects,diagnostics,stats
