"""Conservative 2D A--B trajectory matching, without identity/class/height labels."""
from __future__ import annotations
import bisect
import math
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class MatchConfig:
    maximum_b_gap_seconds: float = 3.
    maximum_a_gap_seconds: float = 5.
    minimum_comparison_points: int = 8
    minimum_supported_duration_seconds: float = 5.
    minimum_temporal_coverage: float = .5
    inlier_distance_m: float = 500.
    minimum_spatial_inlier_fraction: float = .8
    maximum_median_distance_m: float = 350.
    maximum_p90_distance_m: float = 700.
    motion_window_seconds: float = 6.
    minimum_motion_span_seconds: float = 3.
    minimum_motion_displacement_m: float = 120.
    minimum_motion_comparisons: int = 5
    minimum_motion_comparison_fraction: float = .15
    maximum_median_heading_difference_degrees: float = 45.
    maximum_median_speed_difference_m_per_second: float = 60.
    minimum_score_margin: float = .04
    minimum_relative_score_margin: float = .15

    def validate(self):
        for k,v in asdict(self).items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError(f'{k} must be positive and finite')
        for k in ('minimum_comparison_points','minimum_motion_comparisons'):
            if not isinstance(getattr(self,k),int) or getattr(self,k)<2:raise ValueError(f'{k} must be an integer >= 2')
        for k in ('minimum_temporal_coverage','minimum_spatial_inlier_fraction','minimum_relative_score_margin','minimum_motion_comparison_fraction'):
            if getattr(self,k)>1:raise ValueError(f'{k} must be <= 1')
        if self.maximum_median_heading_difference_degrees>180 or self.minimum_motion_span_seconds>self.motion_window_seconds:
            raise ValueError('Invalid motion window/heading threshold')


@dataclass(frozen=True, slots=True)
class TrackPoint:
    time: int
    xy: tuple
    segment: str


class BTrajectory:
    def __init__(self,key,points,maximum_gap_seconds):
        self.key=key
        self.points=sorted(points,key=lambda p:p.time)
        self.times=[p.time for p in self.points]
        if not self.points or len(set(self.times))!=len(self.times):raise ValueError('Empty or duplicate B timestamps')
        self.maximum_gap_ns=round(maximum_gap_seconds*1e9)
        segment=0;self.segments=[]
        for i,p in enumerate(self.points):
            if i and p.time-self.points[i-1].time>self.maximum_gap_ns:segment+=1
            self.segments.append(str(segment))
        # Ignore caller-supplied segment labels; B gaps define interpolation spans.
        self.points=[TrackPoint(p.time,p.xy,s) for p,s in zip(self.points,self.segments)]

    def sample(self,t):
        j=bisect.bisect_left(self.times,t)
        if j<len(self.times) and self.times[j]==t:return self.points[j]
        if j==0 or j==len(self.times):return None
        left,right=self.points[j-1],self.points[j]
        if left.segment!=right.segment:return None
        f=(t-left.time)/(right.time-left.time)
        return TrackPoint(t,tuple(a+f*(b-a) for a,b in zip(left.xy,right.xy)),left.segment)


def motions(points,cfg):
    """Local linear motion only inside one continuous segment; no track gap fit."""
    groups={}
    for p in points:groups.setdefault(p.segment,[]).append(p)
    result={}
    half=cfg.motion_window_seconds*.5e9
    for ps in groups.values():
        ps=sorted(ps,key=lambda p:p.time);times=[p.time for p in ps]
        for p in ps:
            lo=bisect.bisect_left(times,p.time-half);hi=bisect.bisect_right(times,p.time+half)
            nearby=ps[lo:hi]
            if len(nearby)<3:continue
            span=(nearby[-1].time-nearby[0].time)/1e9
            if span<cfg.minimum_motion_span_seconds:continue
            ts=[(q.time-p.time)/1e9 for q in nearby];mt=sum(ts)/len(ts)
            variance=sum((t-mt)**2 for t in ts)
            if variance<=0:continue
            velocities=[]
            for axis in (0,1):
                mean=sum(q.xy[axis] for q in nearby)/len(nearby)
                velocities.append(sum((t-mt)*(q.xy[axis]-mean) for t,q in zip(ts,nearby))/variance)
            if math.hypot(*velocities)*span<cfg.minimum_motion_displacement_m:continue
            result[p.time]=tuple(velocities)
    return result


def b_motion(trajectory,t,vectors):
    j=bisect.bisect_left(trajectory.times,t)
    if j<len(trajectory.times) and trajectory.times[j]==t:return vectors.get(t)
    if j==0 or j==len(trajectory.times):return None
    a,b=trajectory.points[j-1],trajectory.points[j]
    va,vb=vectors.get(a.time),vectors.get(b.time)
    if a.segment!=b.segment or va is None or vb is None:return None
    f=(t-a.time)/(b.time-a.time)
    return tuple(x+f*(y-x) for x,y in zip(va,vb))


def percentile(values,fraction):
    values=sorted(values)
    index=(len(values)-1)*fraction;lo=int(index);hi=min(lo+1,len(values)-1)
    return values[lo]+(index-lo)*(values[hi]-values[lo])


def compare(a_points,b,cfg,a_vectors=None,b_vectors=None):
    a_points=sorted(a_points,key=lambda p:p.time)
    av=motions(a_points,cfg) if a_vectors is None else a_vectors
    bv=motions(b.points,cfg) if b_vectors is None else b_vectors
    paired=[];distances=[];headings=[];speeds=[];duration=0.
    for p in a_points:
        q=b.sample(p.time)
        if q is None:continue
        distance=math.dist(p.xy,q.xy)
        paired.append((p,q,distance));distances.append(distance)
        va,vb=av.get(p.time),b_motion(b,p.time,bv)
        if va is not None and vb is not None:
            na,nb=math.hypot(*va),math.hypot(*vb)
            if min(na,nb)>1e-8:
                angle=math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(va,vb))/(na*nb)))))
                headings.append(angle);speeds.append(abs(na-nb))
    for (a0,b0,_),(a1,b1,_) in zip(paired,paired[1:]):
        dt=(a1.time-a0.time)/1e9
        if a0.segment==a1.segment and b0.segment==b1.segment and 0<dt<=cfg.maximum_a_gap_seconds:
            duration+=dt
    n=len(paired);coverage=n/len(a_points) if a_points else 0.
    med=median(distances) if n else None
    p90=percentile(distances,.9) if n else None
    fraction=sum(d<=cfg.inlier_distance_m for d in distances)/n if n else 0.
    heading=median(headings) if headings else None
    speed=median(speeds) if speeds else None
    temporal=(n>=cfg.minimum_comparison_points and duration>=cfg.minimum_supported_duration_seconds and coverage>=cfg.minimum_temporal_coverage)
    reasons=[]
    if n<cfg.minimum_comparison_points:reasons.append('too_few_comparison_points')
    if duration<cfg.minimum_supported_duration_seconds:reasons.append('too_short_supported_duration')
    if coverage<cfg.minimum_temporal_coverage:reasons.append('insufficient_temporal_coverage')
    if med is not None and med>cfg.maximum_median_distance_m:reasons.append('median_distance_gate')
    if p90 is not None and p90>cfg.maximum_p90_distance_m:reasons.append('p90_distance_gate')
    if fraction<cfg.minimum_spatial_inlier_fraction:reasons.append('spatial_inlier_fraction_gate')
    motion_usable=(len(headings)>=cfg.minimum_motion_comparisons and
                   len(headings)>=cfg.minimum_motion_comparison_fraction*n)
    if motion_usable and heading>cfg.maximum_median_heading_difference_degrees:reasons.append('heading_gate')
    if motion_usable and speed>cfg.maximum_median_speed_difference_m_per_second:reasons.append('speed_gate')
    score=None
    if n:
        score=.65*(med/cfg.maximum_median_distance_m)**2+.20*(p90/cfg.maximum_p90_distance_m)**2
        score+=.10*(1-fraction)+.05*(1-coverage)
        if motion_usable:
            score+=.10*(heading/cfg.maximum_median_heading_difference_degrees)**2
            score+=.05*(speed/cfg.maximum_median_speed_difference_m_per_second)**2
    return {'track_id':b.key[2],'a_point_count':len(a_points),'comparison_count':n,
            'supported_duration_seconds':duration,'temporal_coverage':coverage,'spatial_inlier_fraction':fraction,
            'median_distance_m':med,'p90_distance_m':p90,'rms_distance_m':math.sqrt(sum(d*d for d in distances)/n) if n else None,
            'motion_comparison_count':len(headings),'median_heading_difference_degrees':heading,
            'median_speed_difference_m_per_second':speed,'motion_used':motion_usable,
            'first_comparison_seconds':paired[0][0].time/1e9 if n else None,
            'last_comparison_seconds':paired[-1][0].time/1e9 if n else None,
            'score':score,'temporal_eligible':temporal,'eligible':not reasons,'rejection_reasons':'|'.join(reasons)}


def choose(candidates,cfg):
    """Choose one B ID per A object, retaining close competitors even if gated.

    Independent A objects can share a B ID (upstream fragments). B identity
    numbers and former B anchors have no scoring or disambiguation role.
    """
    ranked=sorted((c for c in candidates if c['temporal_eligible'] and c['score'] is not None),key=lambda c:(c['score'],c['track_id']))
    if not ranked:
        return {'track_id':'','status':'unmatched','reason':'insufficient_temporal_support','best_track_id':'',
                'best_score':None,'runner_up_track_id':'','score_margin':None,'relative_score_margin':None}
    best=ranked[0];second=ranked[1] if len(ranked)>1 else None
    margin=second['score']-best['score'] if second else None
    relative=margin/max(second['score'],1e-12) if second else None
    result={'track_id':'','status':'unmatched','reason':best['rejection_reasons'],'best_track_id':best['track_id'],
            'best_score':best['score'],'runner_up_track_id':second['track_id'] if second else '',
            'score_margin':margin,'relative_score_margin':relative}
    if not best['eligible']:return result
    if second and (margin<cfg.minimum_score_margin or relative<cfg.minimum_relative_score_margin):
        return result|{'status':'ambiguous','reason':'close_competing_B_tracks'}
    return result|{'track_id':best['track_id'],'status':'matched','reason':'trajectory_geometry_supported'}


def match(objects,tracks,cfg):
    cfg.validate();pairs={};decisions={}
    b_vectors={k:motions(b.points,cfg) for k,b in tracks.items()}
    for key,points in sorted(objects.items()):
        av=motions(points,cfg)
        candidates=[]
        for bkey,b in sorted(tracks.items()):
            if key[:2]!=bkey[:2]:continue
            candidate=compare(points,b,cfg,av,b_vectors[bkey])
            pairs[key+(bkey[2],)]=candidate;candidates.append(candidate)
        decisions[key]=choose(candidates,cfg)
    return decisions,pairs
