"""Platform-local sequence association with motion, waveform and overlap constraints."""
from __future__ import annotations

import bisect
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field

from .pri import angle_delta


@dataclass(frozen=True)
class GroupConfig:
    maximum_gap_seconds: float = 60.
    history_seconds: float = 8.
    maximum_history_sequences: int = 128
    bearing_base_gate_degrees: float = .8
    bearing_error_factor: float = 2.5
    gap_gate_growth_degrees_per_second: float = .12
    maximum_bearing_gate_degrees: float = 15.
    maximum_bearing_rate_degrees_per_second: float = 3.
    observer_reference_range_m: float = 10000.
    observer_gate_fraction: float = .2
    rf_family_gate_mhz: float = 250.
    pw_family_gate_us: float = 1.5
    maximum_pri_ratio: float = 3.
    overlap_rf_gate_mhz: float = 10.
    overlap_pw_gate_us: float = .3
    overlap_relative_pri_tolerance: float = .004
    overlap_phase_tolerance_us: float = 2.
    significant_overlap_seconds: float = .002
    ambiguity_margin: float = .15
    maximum_cost: float = 3.5
    confirmed_min_sequences: int = 3
    confirmed_min_pulses: int = 32
    unknown_maximum_gap_seconds: float = 1.
    unknown_bearing_gate_degrees: float = 4.
    unknown_rf_gate_mhz: float = 10.
    unknown_pw_gate_us: float = .3
    unknown_ambiguity_margin: float = .3

    def validate(self):
        for k,v in asdict(self).items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{k} must be positive and finite')
        for k in ('maximum_history_sequences','confirmed_min_sequences','confirmed_min_pulses'):
            if not isinstance(getattr(self,k),int):
                raise ValueError(f'{k} must be an integer')
        if self.maximum_pri_ratio<=1 or self.maximum_history_sequences<2:
            raise ValueError('Invalid PRI ratio or history length')


@dataclass(slots=True)
class Sequence:
    scope: tuple
    sequence_id: str
    start: int
    end: int
    count: int
    model: str
    pri: float
    rf: float
    pw: float
    pa: float
    doa: float
    doa_std: float
    position: tuple

    @property
    def key(self):
        return self.scope+(self.sequence_id,)

    @property
    def time(self):
        return (self.start+self.end)//2

    @property
    def error(self):
        return max(.08,self.doa_std/math.sqrt(self.count))


def distance_m(a,b):
    lat=math.radians((a[1]+b[1])*.5)
    return math.hypot(math.radians(a[0]-b[0])*math.cos(lat)*6371000.,
                      math.radians(a[1]-b[1])*6371000.)


def overlap_conflict(a,b,cfg):
    overlap=(min(a.end,b.end)-max(a.start,b.start))/1e9
    short=(min(a.end-a.start,b.end-b.start))/1e9
    if overlap<=min(cfg.significant_overlap_seconds,short*.3):
        return False
    # Sparse graph paths can span several CPIs. Their bounding intervals alone
    # do not establish simultaneous transmission of their summarized modes.
    if min((a.count-1)*a.pri*1000/max(1,a.end-a.start),
           (b.count-1)*b.pri*1000/max(1,b.end-b.start))<.8:
        return False
    if (abs(a.rf-b.rf)>cfg.overlap_rf_gate_mhz or abs(a.pw-b.pw)>cfg.overlap_pw_gate_us or
        a.model!=b.model or abs(a.pri-b.pri)/min(a.pri,b.pri)>cfg.overlap_relative_pri_tolerance):
        return True
    if a.model=='stable':
        period=(a.pri+b.pri)*.5
        delta=(a.start-b.start)/1000.
        residual=abs(delta-round(delta/period)*period)
        if residual>cfg.overlap_phase_tolerance_us:
            return True
    return False


def predict(history,time,config):
    """Fit local bearing trend; observer motion remains in the measured bearing."""
    last=history[-1]
    recent=[s for s in history if s.time>=last.time-config.history_seconds*1e9]
    origin=last.doa
    xs=[(s.time-last.time)/1e9 for s in recent]
    ys=[angle_delta(s.doa,origin) for s in recent]
    ws=[min(s.count,64)/(1+s.doa_std**2) for s in recent]
    sw=sum(ws)
    mx=sum(x*w for x,w in zip(xs,ws))/sw
    my=sum(y*w for y,w in zip(ys,ws))/sw
    variance=sum(w*(x-mx)**2 for x,w in zip(xs,ws))
    slope=sum(w*(x-mx)*(y-my) for x,y,w in zip(xs,ys,ws))/variance if variance>.01 else 0.
    slope=max(-config.maximum_bearing_rate_degrees_per_second,min(config.maximum_bearing_rate_degrees_per_second,slope))
    intercept=my-slope*mx
    rms=math.sqrt(sum(w*(y-intercept-slope*x)**2 for x,y,w in zip(xs,ys,ws))/sw)
    delta=(time-last.time)/1e9
    return (origin+intercept+slope*delta)%360.,rms,last.error


@dataclass
class LocalGroup:
    number: int
    scope: tuple
    first: int
    last: int
    count: int = 0
    pulse_count: int = 0
    rf: float = 0.
    pw: float = 0.
    log_pri: float = 0.
    pa: float = 0.
    history: deque = field(default_factory=deque)
    members: list = field(default_factory=list)
    models: Counter = field(default_factory=Counter)
    fit_rows: deque = field(default_factory=deque)
    fit_sums: list = field(default_factory=lambda:[0.]*6)
    unwrapped_angle: float = 0.
    fit_angle: float = 0.
    fit_rate: float = 0.
    fit_rms: float = 0.

    @property
    def name(self):
        return f'L{self.number:05d}'

    def add(self,s,cfg):
        self.unwrapped_angle=(self.unwrapped_angle+angle_delta(s.doa,self.history[-1].doa)) if self.history else s.doa
        self.count+=1
        self.pulse_count+=s.count
        self.first=min(self.first,s.start)
        self.last=max(self.last,s.end)
        alpha=1/min(self.count,128)
        self.rf+=alpha*(s.rf-self.rf)
        self.pw+=alpha*(s.pw-self.pw)
        self.pa+=alpha*(s.pa-self.pa)
        self.log_pri+=alpha*(math.log(s.pri)-self.log_pri)
        self.models[s.model]+=s.count
        self.history.append(s)
        while len(self.history)>cfg.maximum_history_sequences:
            self.history.popleft()
        self.members.append(s)
        x,y,w=s.time/1e9,self.unwrapped_angle,min(s.count,64)/(1+s.doa_std**2)
        row=(w,w*x,w*y,w*x*x,w*x*y,w*y*y)
        self.fit_rows.append((x,row))
        for i,v in enumerate(row):
            self.fit_sums[i]+=v
        while len(self.fit_rows)>cfg.maximum_history_sequences or x-self.fit_rows[0][0]>cfg.history_seconds:
            _,old=self.fit_rows.popleft()
            for i,v in enumerate(old):
                self.fit_sums[i]-=v
        sw,sx,sy,sxx,sxy,syy=self.fit_sums
        variance=sxx-sx*sx/sw
        slope=(sxy-sx*sy/sw)/variance if variance>.01 else 0.
        slope=max(-cfg.maximum_bearing_rate_degrees_per_second,min(cfg.maximum_bearing_rate_degrees_per_second,slope))
        intercept=sy/sw-slope*sx/sw
        self.fit_angle=(intercept+slope*x)%360.
        self.fit_rate=slope
        residual=syy-2*intercept*sy-2*slope*sxy+intercept*intercept*sw+2*intercept*slope*sx+slope*slope*sxx
        self.fit_rms=math.sqrt(max(0.,residual/sw))

    def cost(self,s,cfg):
        gap=max(0.,(s.start-self.last)/1e9)
        if s.scope!=self.scope or gap>cfg.maximum_gap_seconds:
            return None
        if (abs(s.rf-self.rf)>cfg.rf_family_gate_mhz or abs(s.pw-self.pw)>cfg.pw_family_gate_us or
            abs(math.log(s.pri)-self.log_pri)>math.log(cfg.maximum_pri_ratio)):
            return None
        predicted=(self.fit_angle+self.fit_rate*(s.time-self.history[-1].time)/1e9)%360.
        rms,error=self.fit_rms,self.history[-1].error
        elapsed=max(0.,(s.time-self.history[-1].time)/1e9)
        motion=math.degrees(math.atan2(distance_m(self.history[-1].position,s.position),cfg.observer_reference_range_m))
        gate=min(cfg.maximum_bearing_gate_degrees,cfg.bearing_base_gate_degrees+
            cfg.bearing_error_factor*math.sqrt(s.error**2+error**2+min(rms,2.)**2)+
            cfg.gap_gate_growth_degrees_per_second*elapsed+cfg.observer_gate_fraction*motion)
        da=abs(angle_delta(s.doa,predicted))
        if da>gate:
            return None
        if any(overlap_conflict(old,s,cfg) for old in reversed(self.history) if old.end>s.start):
            return None
        radio=.25*(abs(s.rf-self.rf)/80.)**2+.12*((s.pw-self.pw)/.6)**2
        pri=.15*((math.log(s.pri)-self.log_pri)/.7)**2
        pa=.05*min(abs(s.pa-self.pa)/15.,2.)
        return 2*(da/gate)**2+radio+pri+pa+.01*gap


def deferred_cost(s,g,items,times,cfg):
    if (abs(s.rf-g.rf)>cfg.rf_family_gate_mhz or abs(s.pw-g.pw)>cfg.pw_family_gate_us or
        abs(math.log(s.pri)-g.log_pri)>math.log(cfg.maximum_pri_ratio)):
        return None
    j=bisect.bisect_left(times,s.time)
    neighbors=items[max(0,j-1):min(len(items),j+1)]
    if not neighbors or min(abs(s.time-p.time) for p in neighbors)/1e9>cfg.maximum_gap_seconds:
        return None
    if len(neighbors)==2:
        left,right=neighbors
        f=(s.time-left.time)/max(1,right.time-left.time)
        angle=(left.doa+f*angle_delta(right.doa,left.doa))%360.
    else:
        angle=neighbors[0].doa
    gap=min(abs(s.time-p.time) for p in neighbors)/1e9
    error=max(p.error for p in neighbors)
    gate=min(cfg.maximum_bearing_gate_degrees,cfg.bearing_base_gate_degrees+
        cfg.bearing_error_factor*math.hypot(s.error,error)+cfg.gap_gate_growth_degrees_per_second*gap)
    da=abs(angle_delta(s.doa,angle))
    if da>gate:
        return None
    bound=max((s.end-s.start)/2,600_000_000)
    nearby=items[bisect.bisect_left(times,s.time-bound):bisect.bisect_right(times,s.time+bound)]
    if any(overlap_conflict(p,s,cfg) for p in nearby):
        return None
    radio=.25*(abs(s.rf-g.rf)/80.)**2+.12*((s.pw-g.pw)/.6)**2
    pri=.15*((math.log(s.pri)-g.log_pri)/.7)**2
    local=items[bisect.bisect_left(times,s.time-3e9):bisect.bisect_right(times,s.time+3e9)]
    mode=min((abs(p.rf-s.rf)/20+abs(p.pw-s.pw)/.3+abs(math.log(p.pri/s.pri))/.05+
             (p.model!=s.model)*2 for p in local),default=4.)
    return 2*(da/gate)**2+radio+pri+.15*min(mode,4.)+.01*gap


def associate(sequences,config,on_progress=None):
    config.validate()
    by_scope={}
    for s in sequences:
        by_scope.setdefault(s.scope,[]).append(s)
    groups,links=[],{}
    for scope,items in sorted(by_scope.items()):
        active=[]
        scope_groups=[]
        deferred=[]
        for index,s in enumerate(sorted(items,key=lambda s:(s.time,s.start,s.sequence_id)),1):
            active=[g for g in active if (s.start-g.last)/1e9<=config.maximum_gap_seconds]
            candidates=[]
            for g in active:
                cost=g.cost(s,config)
                if cost is not None and cost<=config.maximum_cost:
                    candidates.append((cost,g.number,g))
            candidates.sort(key=lambda t:(t[0],t[1]))
            if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=config.ambiguity_margin):
                cost,_,group=candidates[0]
                reason='temporal_waveform_association'
            elif candidates:
                deferred.append(s)
                continue
            else:
                group=LocalGroup(len(scope_groups)+1,scope,s.start,s.end)
                scope_groups.append(group)
                active.append(group)
                cost=None
                reason='new_tracklet'
            group.add(s,config)
            links[s.key]={'local_cluster_id':group.name,'association_cost':cost,'association_basis':reason}
            if on_progress and index%10000==0:
                on_progress({'platform':scope,'associated_sequences':index,'groups':len(scope_groups),'active':len(active)})
        views={g.number:sorted(g.members,key=lambda s:s.time) for g in scope_groups}
        times={k:[s.time for s in v] for k,v in views.items()}
        for s in deferred:
            ranked=[]
            for g in scope_groups:
                cost=deferred_cost(s,g,views[g.number],times[g.number],config)
                if cost is not None and cost<=config.maximum_cost:
                    ranked.append((cost,g.number,g))
            ranked.sort(key=lambda r:(r[0],r[1]))
            if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>=config.ambiguity_margin*.5):
                cost,_,g=ranked[0]
                g.count+=1
                g.pulse_count+=s.count
                g.first=min(g.first,s.start)
                g.last=max(g.last,s.end)
                g.members.append(s)
                g.models[s.model]+=s.count
                links[s.key]={'local_cluster_id':g.name,'association_cost':cost,'association_basis':'deferred_context_association'}
            else:
                links[s.key]={'local_cluster_id':'','association_cost':ranked[0][0] if ranked else None,
                              'association_basis':'ambiguous_unassigned_sequence'}
        groups.extend(scope_groups)
    return groups,links


class PulseAttacher:
    """Recover unknown PRI pulses only near directly observed compatible sequences."""
    def __init__(self,sequences,links,config):
        self.config,self.links=config,links
        self.items,self.times={},{}
        for s in sequences:
            if links[s.key]['local_cluster_id']:
                self.items.setdefault(s.scope,[]).append(s)
        for scope,items in self.items.items():
            items.sort(key=lambda s:s.time)
            self.times[scope]=[s.time for s in items]
        self.span=max(((s.end-s.start)/2e9 for s in sequences),default=0.)

    def attach(self,scope,time,rf,pw,pa,doa):
        cfg=self.config
        if scope not in self.items:
            return '',None,'no_sequence_support'
        items,times=self.items[scope],self.times[scope]
        bound=(cfg.unknown_maximum_gap_seconds+self.span)*1e9
        left=bisect.bisect_left(times,time-bound)
        right=bisect.bisect_right(times,time+bound)
        costs={}
        for s in items[left:right]:
            gap=max(s.start-time,time-s.end,0)/1e9
            drf,dpw=abs(rf-s.rf),abs(pw-s.pw)
            da=abs(angle_delta(doa,s.doa))
            if (gap>cfg.unknown_maximum_gap_seconds or drf>cfg.unknown_rf_gate_mhz or
                dpw>cfg.unknown_pw_gate_us or da>cfg.unknown_bearing_gate_degrees):
                continue
            score=((da/cfg.unknown_bearing_gate_degrees)**2+(drf/cfg.unknown_rf_gate_mhz)**2+
                (dpw/cfg.unknown_pw_gate_us)**2+.05*min(abs(pa-s.pa)/20.,2.)+.2*gap)
            group=self.links[s.key]['local_cluster_id']
            costs[group]=min(costs.get(group,float('inf')),score)
        ranked=sorted((v,k) for k,v in costs.items())
        if not ranked:
            return '',None,'no_sequence_support'
        if len(ranked)>1 and ranked[1][0]-ranked[0][0]<cfg.unknown_ambiguity_margin:
            return '',ranked[0][0],'ambiguous'
        return ranked[0][1],ranked[0][0],'nearby_sequence_support'
