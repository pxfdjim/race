"""A-only multi-observer event association and persistent object tracking.

Geometry is a coarse horizontal consistency check, not the final localization.
No truth IDs, classes, states or B observations enter these functions.
"""
from __future__ import annotations
import bisect
import itertools
import math
from collections import Counter,defaultdict,deque
from dataclasses import dataclass,asdict,field
from statistics import median

from .pri import angle_delta
from .local_grouping import overlap_conflict,GroupConfig

C_M_PER_US=299.792458


@dataclass(frozen=True)
class FusionConfig:
    maximum_event_time_difference_seconds: float=.08
    minimum_temporal_overlap_fraction: float=.45
    rf_pair_gate_mhz: float=3.
    pw_pair_gate_us: float=.12
    stable_relative_pri_gate: float=.003
    jitter_relative_pri_gate: float=.035
    minimum_crossing_angle_degrees: float=7.
    maximum_range_m: float=180000.
    bearing_error_floor_degrees: float=.15
    geometry_residual_gate_degrees: float=1.2
    stable_tdoa_tolerance_us: float=5.
    jitter_tdoa_tolerance_us: float=25.
    maximum_neighbors_per_platform: int=3
    maximum_object_gap_seconds: float=60.
    object_position_gate_m: float=700.
    object_error_multiplier: float=2.
    object_maximum_speed_m_per_second: float=180.
    object_history_seconds: float=8.
    object_rf_family_gate_mhz: float=250.
    object_pw_family_gate_us: float=1.5
    object_maximum_pri_ratio: float=3.
    object_ambiguity_margin: float=.12
    single_maximum_gap_seconds: float=12.
    single_bearing_gate_degrees: float=2.
    single_ambiguity_margin: float=.18

    def validate(self):
        for k,v in asdict(self).items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{k} must be positive and finite')
        if not isinstance(self.maximum_neighbors_per_platform,int):
            raise ValueError('Neighbor limit must be integer')
        if self.minimum_temporal_overlap_fraction>1 or self.object_maximum_pri_ratio<=1:
            raise ValueError('Invalid fraction or PRI ratio')
        if self.minimum_crossing_angle_degrees>=90:
            raise ValueError('Crossing angle must be below 90 degrees')


class LocalFrame:
    """WGS84 ECEF projected onto a common local East/North tangent plane."""
    def __init__(self,longitude,latitude):
        self.longitude,self.latitude=longitude,latitude
        lon,lat=math.radians(longitude),math.radians(latitude)
        self.origin=self.ecef(longitude,latitude,0.)
        self.east=(-math.sin(lon),math.cos(lon),0.)
        self.north=(-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat))

    @staticmethod
    def ecef(longitude,latitude,height):
        lon,lat=math.radians(longitude),math.radians(latitude)
        a,e2=6378137.,6.6943799901413165e-3
        n=a/math.sqrt(1-e2*math.sin(lat)**2)
        return ((n+height)*math.cos(lat)*math.cos(lon),(n+height)*math.cos(lat)*math.sin(lon),
                (n*(1-e2)+height)*math.sin(lat))

    def project(self,position):
        p=self.ecef(*position)
        delta=tuple(a-b for a,b in zip(p,self.origin))
        return tuple(sum(x*y for x,y in zip(delta,axis)) for axis in (self.east,self.north))

    def direction(self,position,doa):
        lon,lat=map(math.radians,position[:2])
        east=(-math.sin(lon),math.cos(lon),0.)
        north=(-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat))
        a=math.radians(doa)
        ray=tuple(math.sin(a)*e+math.cos(a)*n for e,n in zip(east,north))
        xy=tuple(sum(x*y for x,y in zip(ray,axis)) for axis in (self.east,self.north))
        norm=math.hypot(*xy)
        return xy[0]/norm,xy[1]/norm


@dataclass(slots=True)
class FusionObservation:
    sequence: object
    local_group: str
    xy: tuple
    direction: tuple

    @property
    def key(self):return self.sequence.key
    @property
    def time(self):return self.sequence.time
    @property
    def platform(self):return self.sequence.scope[2]


def crossing(a,b):
    d=abs(a.direction[0]*b.direction[0]+a.direction[1]*b.direction[1])
    return math.degrees(math.acos(min(1.,d)))


def geometry(observations,cfg):
    best=max((crossing(a,b) for a,b in itertools.combinations(observations,2)),default=0.)
    if best<cfg.minimum_crossing_angle_degrees:return None
    aa=ab=bb=ar=br=0.
    for o in observations:
        # Fit angular means; this first fit uses a common nominal distance.
        nx,ny=-o.direction[1],o.direction[0]
        w=1/max(o.sequence.error,cfg.bearing_error_floor_degrees)**2
        dot=nx*o.xy[0]+ny*o.xy[1]
        aa+=w*nx*nx;ab+=w*nx*ny;bb+=w*ny*ny
        ar+=w*nx*dot;br+=w*ny*dot
    det=aa*bb-ab*ab
    if det<=1e-12:return None
    x,y=(ar*bb-br*ab)/det,(br*aa-ar*ab)/det
    ranges,residuals,errors=[],[],[]
    for o in observations:
        dx,dy=x-o.xy[0],y-o.xy[1]
        dist=math.hypot(dx,dy)
        forward=dx*o.direction[0]+dy*o.direction[1]
        residual=abs(math.degrees(math.atan2(o.direction[0]*dy-o.direction[1]*dx,forward)))
        gate=max(cfg.geometry_residual_gate_degrees,3*o.sequence.error)
        if forward<=0 or dist>cfg.maximum_range_m or residual>gate:return None
        ranges.append(dist);residuals.append(residual)
        errors.append(dist*math.radians(max(o.sequence.error,cfg.bearing_error_floor_degrees)))
    uncertainty=max(30.,math.hypot(*sorted(errors)[-2:])/max(.05,math.sin(math.radians(best))))
    return {'east_m':x,'north_m':y,'error_m':uncertainty,'crossing_degrees':best,
            'ranges_m':ranges,'maximum_bearing_residual_degrees':max(residuals)}


def waveform_cost(a,b,cfg):
    a,b=a.sequence,b.sequence
    if a.scope[:2]!=b.scope[:2] or a.scope[2]==b.scope[2] or a.model!=b.model:return None
    if abs(a.time-b.time)>cfg.maximum_event_time_difference_seconds*1e9:return None
    overlap=max(0,min(a.end,b.end)-max(a.start,b.start))
    if overlap/max(1,min(a.end-a.start,b.end-b.start))<cfg.minimum_temporal_overlap_fraction:return None
    drf,dpw=abs(a.rf-b.rf),abs(a.pw-b.pw)
    rel=abs(a.pri-b.pri)/min(a.pri,b.pri)
    gate=cfg.stable_relative_pri_gate if a.model=='stable' else cfg.jitter_relative_pri_gate
    if drf>cfg.rf_pair_gate_mhz or dpw>cfg.pw_pair_gate_us or rel>gate:return None
    return .3*(drf/cfg.rf_pair_gate_mhz)**2+.3*(dpw/cfg.pw_pair_gate_us)**2+.3*(rel/gate)**2


def tdoa_residual(a,b,ranges,cfg,error_m):
    """Check differential propagation against observed sequence start phase.

    Uses documented common TOA clock; it does not estimate unknown clock offsets.
    Slant distance is unknown, so horizontal ranges are an approximation.
    """
    left,right=a.sequence,b.sequence
    observed=(right.start-left.start)/1000.
    predicted=(ranges[1]-ranges[0])/C_M_PER_US
    period=(left.pri+right.pri)*.5
    delta=observed-predicted
    delta=abs(delta-round(delta/period)*period)
    base=cfg.stable_tdoa_tolerance_us if left.model=='stable' else cfg.jitter_tdoa_tolerance_us
    # Partial jittered sequences do not share an exact periodic phase.
    if left.model=='jittered' and abs(observed)>period:
        base+=.06*period
    tolerance=base+min(8.,2*error_m/C_M_PER_US)
    return delta/tolerance if delta<=tolerance else None


@dataclass(slots=True)
class Event:
    members: tuple
    time: int
    geometry: dict
    cost: float
    event_id: str=''
    object_id: str=''

    @property
    def rf(self):return sum(o.sequence.rf for o in self.members)/len(self.members)
    @property
    def pw(self):return sum(o.sequence.pw for o in self.members)/len(self.members)
    @property
    def pri(self):return sum(o.sequence.pri for o in self.members)/len(self.members)


def build_events(observations,cfg,on_progress=None):
    """Enumerate bounded pairwise-compatible cliques; never union graph components."""
    cfg.validate()
    observations=sorted(observations,key=lambda o:(o.time,o.key))
    times=[o.time for o in observations]
    edges={};neighbors=defaultdict(list)
    for i,a in enumerate(observations):
        if on_progress and i and i%20000==0:on_progress({'stage':'candidate_pairs','sequences':i,'edges':len(edges)})
        stop=bisect.bisect_right(times,a.time+cfg.maximum_event_time_difference_seconds*1e9,i+1)
        for j in range(i+1,stop):
            b=observations[j]
            cost=waveform_cost(a,b,cfg)
            if cost is None:continue
            g=geometry((a,b),cfg)
            if g is None:continue
            phase=tdoa_residual(a,b,g['ranges_m'],cfg,g['error_m'])
            if phase is None:continue
            score=cost+.35*phase**2
            edges[i,j]=score
            neighbors[i].append((score,j));neighbors[j].append((score,i))
    candidates={}
    for i,a in enumerate(observations):
        if on_progress and i and i%20000==0:on_progress({'stage':'candidate_cliques','sequences':i,'cliques':len(candidates)})
        by_platform=defaultdict(list)
        for cost,j in sorted(neighbors[i]):
            by_platform[observations[j].platform].append(j)
        choices=[[-1]+v[:cfg.maximum_neighbors_per_platform] for _,v in sorted(by_platform.items())]
        for combo in itertools.product(*choices):
            indices=tuple(sorted((i,)+tuple(j for j in combo if j>=0)))
            if len(indices)<2 or indices in candidates:continue
            pairs=list(itertools.combinations(indices,2))
            if any(pair not in edges for pair in pairs):continue
            members=tuple(observations[j] for j in indices)
            g=geometry(members,cfg)
            if g is None:continue
            valid=True
            for k,l in itertools.combinations(range(len(members)),2):
                if tdoa_residual(members[k],members[l],(g['ranges_m'][k],g['ranges_m'][l]),cfg,g['error_m']) is None:
                    valid=False;break
            if not valid:continue
            score=sum(edges[p] for p in pairs)/len(pairs)+.1*g['maximum_bearing_residual_degrees']**2
            candidates[indices]=Event(members,round(median(o.time for o in members)),g,score)
    # More independent platforms first, then lower cost; immutable ownership.
    ranked=sorted(candidates.values(),key=lambda e:(-len(e.members),e.cost,e.time,tuple(o.key for o in e.members)))
    used=set();selected=[]
    for e in ranked:
        if any(o.key in used for o in e.members):continue
        used.update(o.key for o in e.members)
        selected.append(e)
    selected.sort(key=lambda e:(e.time,tuple(o.key for o in e.members)))
    for i,e in enumerate(selected,1):e.event_id=f'E{i:07d}'
    return selected,{'candidate_edges':len(edges),'candidate_cliques':len(candidates),
                     'selected_events':len(selected),'event_assigned_sequences':len(used)}


@dataclass
class FusedObject:
    object_id: str
    events: list=field(default_factory=list)
    recent: deque=field(default_factory=deque)
    rf: float=0.
    pw: float=0.
    log_pri: float=0.
    local_groups: Counter=field(default_factory=Counter)

    def add(self,e,cfg):
        n=len(self.events)+1
        alpha=1/min(n,128)
        self.rf+=alpha*(e.rf-self.rf);self.pw+=alpha*(e.pw-self.pw)
        self.log_pri+=alpha*(math.log(e.pri)-self.log_pri)
        self.events.append(e);self.recent.append(e)
        while e.time-self.recent[0].time>cfg.object_history_seconds*1e9:self.recent.popleft()
        for o in e.members:
            if o.local_group:self.local_groups[o.sequence.scope+(o.local_group,)]+=o.sequence.count
        e.object_id=self.object_id

    def predict(self,time,cfg):
        last=self.events[-1]
        # Limit fit density to one strongest event per 0.2 s.
        bins={}
        for e in self.recent:
            k=e.time//200_000_000
            if k not in bins or e.geometry['error_m']<bins[k].geometry['error_m']:bins[k]=e
        events=list(bins.values())
        weights=[1/max(e.geometry['error_m'],50.)**2 for e in events]
        sw=sum(weights)
        xs=[(e.time-last.time)/1e9 for e in events]
        mx=sum(w*x for w,x in zip(weights,xs))/sw
        variance=sum(w*(x-mx)**2 for w,x in zip(weights,xs))
        result=[]
        for axis in ('east_m','north_m'):
            my=sum(w*e.geometry[axis] for w,e in zip(weights,events))/sw
            slope=sum(w*(x-mx)*(e.geometry[axis]-my) for w,x,e in zip(weights,xs,events))/variance if variance>1e-6 else 0.
            slope=max(-cfg.object_maximum_speed_m_per_second,min(cfg.object_maximum_speed_m_per_second,slope))
            result.append(my+slope*((time-last.time)/1e9-mx))
        return tuple(result)

    def cost(self,e,cfg):
        gap=max(0,(e.time-self.events[-1].time)/1e9)
        if gap>cfg.maximum_object_gap_seconds:return None
        if (abs(e.rf-self.rf)>cfg.object_rf_family_gate_mhz or abs(e.pw-self.pw)>cfg.object_pw_family_gate_us or
            abs(math.log(e.pri)-self.log_pri)>math.log(cfg.object_maximum_pri_ratio)):return None
        predicted=self.predict(e.time,cfg)
        gate=(cfg.object_position_gate_m+cfg.object_error_multiplier*min(2000.,e.geometry['error_m']+
            self.events[-1].geometry['error_m'])+cfg.object_maximum_speed_m_per_second*gap)
        distance=math.hypot(predicted[0]-e.geometry['east_m'],predicted[1]-e.geometry['north_m'])
        if distance>gate:return None
        for old in reversed(self.recent):
            if e.time-old.time>1e9:break
            if any(a.platform==b.platform and overlap_conflict(a.sequence,b.sequence,GroupConfig())
                   for a in old.members for b in e.members):return None
        radio=.1*(abs(e.rf-self.rf)/100.)**2+.05*((e.pw-self.pw)/.6)**2
        pri=.04*((math.log(e.pri)-self.log_pri)/.7)**2
        hint=any(self.local_groups[o.sequence.scope+(o.local_group,)] for o in e.members if o.local_group)
        return 2*(distance/gate)**2+radio+pri-(.04 if hint else 0.)


def associate_events(events,cfg,on_progress=None):
    objects=[];deferred=[]
    for index,e in enumerate(events):
        if on_progress and index and index%5000==0:on_progress({'stage':'event_identity','events':index,'objects':len(objects)})
        candidates=[]
        for obj in objects:
            cost=obj.cost(e,cfg)
            if cost is not None:candidates.append((cost,obj.object_id,obj))
        candidates.sort(key=lambda x:(x[0],x[1]))
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=cfg.object_ambiguity_margin):
            obj=candidates[0][2]
        elif candidates:
            # A doubtful event must not seed another competing object; otherwise
            # one ambiguity can recursively generate thousands of tiny objects.
            e.object_id='';deferred.append(e);continue
        else:
            obj=FusedObject(f'O{len(objects)+1:05d}');objects.append(obj)
        obj.add(e,cfg)
    views={o.object_id:sorted(o.events,key=lambda e:e.time) for o in objects}
    times={k:[e.time for e in v] for k,v in views.items()}
    for e in deferred:
        ranked=[]
        for obj in objects:
            if (abs(e.rf-obj.rf)>cfg.object_rf_family_gate_mhz or abs(e.pw-obj.pw)>cfg.object_pw_family_gate_us or
                abs(math.log(e.pri)-obj.log_pri)>math.log(cfg.object_maximum_pri_ratio)):continue
            items,ts=views[obj.object_id],times[obj.object_id]
            j=bisect.bisect_left(ts,e.time);near=items[max(0,j-1):min(len(items),j+1)]
            gap=min(abs(p.time-e.time) for p in near)/1e9
            if gap>cfg.maximum_object_gap_seconds:continue
            if len(near)==2 and near[1].time-near[0].time<=2*cfg.maximum_object_gap_seconds*1e9:
                a,b=near;f=(e.time-a.time)/max(1,b.time-a.time)
                xy=tuple(a.geometry[k]+f*(b.geometry[k]-a.geometry[k]) for k in ('east_m','north_m'))
            else:
                p=min(near,key=lambda p:abs(p.time-e.time));xy=p.geometry['east_m'],p.geometry['north_m']
            gate=cfg.object_position_gate_m+cfg.object_error_multiplier*min(2000.,e.geometry['error_m']+min(p.geometry['error_m'] for p in near))+cfg.object_maximum_speed_m_per_second*gap
            distance=math.hypot(xy[0]-e.geometry['east_m'],xy[1]-e.geometry['north_m'])
            if distance>gate:continue
            radio=.1*(abs(e.rf-obj.rf)/100.)**2+.05*((e.pw-obj.pw)/.6)**2
            pri=.04*((math.log(e.pri)-obj.log_pri)/.7)**2
            ranked.append((2*(distance/gate)**2+radio+pri,obj.object_id,obj))
        ranked.sort(key=lambda r:(r[0],r[1]))
        if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>=cfg.object_ambiguity_margin*.5):
            obj=ranked[0][2];e.object_id=obj.object_id;obj.events.append(e)
            for o in e.members:
                if o.local_group:obj.local_groups[o.sequence.scope+(o.local_group,)]+=o.sequence.count
    for obj in objects:obj.events.sort(key=lambda e:(e.time,e.event_id))
    return objects


def assign_single_observations(observations,objects,links,cfg,on_progress=None):
    # Offline context: interpolate between observed events, never across a long
    # unobserved gap; bounded extrapolation uses the closest actual event.
    views={o.object_id:[e.time for e in o.events] for o in objects}
    for index,obs in enumerate(observations):
        if on_progress and index and index%20000==0:on_progress({'stage':'single_context','sequences':index,'objects':len(objects)})
        if obs.key in links:continue
        s=obs.sequence;ranked=[]
        for obj in objects:
            if (abs(s.rf-obj.rf)>cfg.object_rf_family_gate_mhz or abs(s.pw-obj.pw)>cfg.object_pw_family_gate_us or
                abs(math.log(s.pri)-obj.log_pri)>math.log(cfg.object_maximum_pri_ratio)):continue
            times=views[obj.object_id];j=bisect.bisect_left(times,obs.time)
            near=obj.events[max(0,j-1):min(len(times),j+1)]
            gap=min(abs(e.time-obs.time) for e in near)/1e9
            if gap>cfg.single_maximum_gap_seconds:continue
            if len(near)==2 and (near[1].time-near[0].time)<=2*cfg.single_maximum_gap_seconds*1e9:
                a,b=near;f=(obs.time-a.time)/max(1,b.time-a.time)
                xy=tuple(a.geometry[k]+f*(b.geometry[k]-a.geometry[k]) for k in ('east_m','north_m'))
            else:
                e=min(near,key=lambda e:abs(e.time-obs.time));xy=e.geometry['east_m'],e.geometry['north_m']
            dx,dy=xy[0]-obs.xy[0],xy[1]-obs.xy[1]
            forward=dx*obs.direction[0]+dy*obs.direction[1]
            if forward<=0:continue
            residual=abs(math.degrees(math.atan2(obs.direction[0]*dy-obs.direction[1]*dx,forward)))
            gate=cfg.single_bearing_gate_degrees+min(2.,gap*.08)+2*s.error
            if residual>gate:continue
            if any(a.platform==obs.platform and overlap_conflict(a.sequence,s,GroupConfig()) for e in near for a in e.members):continue
            hint=obj.local_groups[s.scope+(obs.local_group,)] if obs.local_group else 0
            score=(residual/gate)**2+.15*(abs(s.rf-obj.rf)/100.)**2+.08*((s.pw-obj.pw)/.6)**2-(.08 if hint else 0.)
            ranked.append((score,obj.object_id))
        ranked.sort()
        if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>=cfg.single_ambiguity_margin):
            links[obs.key]={'object_id':ranked[0][1],'event_id':'','basis':'single_platform_context','cost':ranked[0][0]}
        else:
            links[obs.key]={'object_id':'','event_id':'','basis':'unmatched_ambiguous' if ranked else 'unmatched_no_support','cost':None}


def fuse_observations(observations,cfg,on_progress=None):
    by_scene=defaultdict(list)
    for o in observations:by_scene[o.sequence.scope[:2]].append(o)
    all_objects,all_events,links,stats=[],[],{},{}
    for scene,items in sorted(by_scene.items()):
        events,counts=build_events(items,cfg,on_progress)
        if on_progress:on_progress({'stage':'events_built',**counts})
        objects=associate_events(events,cfg,on_progress)
        for e in events:
            for o in e.members:
                links[o.key]={'object_id':e.object_id,'event_id':e.event_id,
                    'basis':'multiplatform_event' if e.object_id else 'unmatched_event_ambiguous','cost':e.cost}
        assign_single_observations(items,objects,links,cfg,on_progress)
        all_objects.extend((scene,o) for o in objects)
        all_events.extend((scene,e) for e in events)
        stats['/'.join(scene)]=counts
    return all_objects,all_events,links,stats
