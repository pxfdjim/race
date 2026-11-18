"""Conservative B refill: preserve A identity evidence and require radio support."""
import bisect
import math
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass


@dataclass(frozen=True)
class GuardConfig:
    maximum_support_gap_seconds: float = 4.
    rf_support_gate_mhz: float = 10.
    pw_support_gate_us: float = .3
    stable_relative_pri_gate: float = .008
    jitter_relative_pri_gate: float = .05
    baseline_mapping_dominance: float = .9

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{key} must be positive and finite')
        if not .5<self.baseline_mapping_dominance<=1:raise ValueError('Invalid mapping dominance')


def guard(observations,baseline,proposals,seed_keys,cfg):
    cfg.validate()
    support=defaultdict(list);votes=defaultdict(Counter)
    for o in observations:
        if o.key not in seed_keys or not proposals[o.key]['object_id']:continue
        oid=proposals[o.key]['object_id'];scene=o.sequence.scope[:2]
        support[scene+(oid,)].append(o)
        old=baseline[o.key]['object_id']
        if old:votes[scene+(old,)][oid]+=o.sequence.count
    times={}
    for key,items in support.items():
        items.sort(key=lambda o:o.time);times[key]=[o.time for o in items]
    mapping={}
    for key,v in votes.items():
        oid,n=v.most_common(1)[0]
        if n/sum(v.values())>=cfg.baseline_mapping_dominance:mapping[key]=oid
    links={};counts=Counter()
    for o in observations:
        s=o.sequence;p=proposals[o.key];old=baseline[o.key]['object_id'];oid=p['object_id'];cost=p['cost']
        if o.key in seed_keys and oid:
            basis='guarded_multiplatform_B_seed'
        elif old:
            # An identity from the high-precision A baseline is retained. B
            # anchors can name/merge its fragments, but a weak bearing candidate
            # cannot move its observations into a competing identity.
            oid=mapping.get(s.scope[:2]+(old,),'OA'+old[1:]);cost=None
            basis='guarded_A_identity_anchor' if oid.startswith('OB') else 'guarded_A_only_fallback'
        elif oid and s.scope[:2]+(oid,) in support:
            key=s.scope[:2]+(oid,);items,ts=support[key],times[key]
            radius=cfg.maximum_support_gap_seconds*1e9
            left,right=bisect.bisect_left(ts,s.time-radius),bisect.bisect_right(ts,s.time+radius)
            compatible=False
            for near in items[left:right]:
                a=near.sequence
                pri_gate=cfg.stable_relative_pri_gate if s.model=='stable' else cfg.jitter_relative_pri_gate
                if (a.model==s.model and abs(a.rf-s.rf)<=cfg.rf_support_gate_mhz and
                    abs(a.pw-s.pw)<=cfg.pw_support_gate_us and abs(a.pri-s.pri)/min(a.pri,s.pri)<=pri_gate):
                    compatible=True;break
            if compatible:basis='guarded_B_refill_with_radio_support'
            else:oid='';cost=None;basis='unmatched_no_nearby_radio_anchor'
        else:
            oid='';cost=None;basis='unmatched_B_ambiguous_or_missing'
        links[o.key]={'object_id':oid,'event_id':'','basis':basis,'cost':cost}
        counts[basis]+=s.count
    return links,{'basis_sequence_pulses':dict(counts),'baseline_objects_with_B_identity_anchor':len(mapping),
                  'baseline_assigned_sequences_preserved':all(links[k]['object_id'] for k,v in baseline.items() if v['object_id'])}
