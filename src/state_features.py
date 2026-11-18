"""Observable illumination and local time features, with labels kept separate."""
from __future__ import annotations
import bisect
import math
from dataclasses import asdict,dataclass,field


@dataclass(frozen=True)
class StateFeatureConfig:
    bin_seconds: float=.05
    context_half_windows_seconds: tuple=(.25,1.,3.)

    def validate(self):
        if not math.isfinite(self.bin_seconds) or self.bin_seconds<=0:raise ValueError('Invalid state bin width')
        if not self.context_half_windows_seconds or any(not math.isfinite(x) or x<self.bin_seconds for x in self.context_half_windows_seconds):
            raise ValueError('Invalid state context windows')


BASE_FEATURES=('log_pulse_count','observed_span_fraction','log_PRI_normalized_occupancy','PRI_supported_fraction',
    'log_sequence_count','within_bin_gap_cv','log_max_gap_over_PRI','log_previous_gap_seconds','log_next_gap_seconds',
    'PA_std_db','PA_range_db','PA_mean_position_in_range')
CONTEXT_FEATURES=('observed_bin_fraction','log_pulses_per_second','PA_std_db','PA_bin_mean_std_db','PA_mean_change_db')
KEY_FIELDS=('scenario_id','data_version','platform_id','source_kind','source_id','bin_index')


def feature_names(cfg):
    return list(BASE_FEATURES)+[f'context_{i}_{name}' for i,_ in enumerate(cfg.context_half_windows_seconds) for name in CONTEXT_FEATURES]


def source_key(scope,object_id,local_id,rf=None,pw=None,doa=None):
    if object_id:return scope+('object',object_id)
    if local_id:return scope+('local',local_id)
    if rf is None or pw is None or doa is None:raise ValueError('Ungrouped pulse needs observable fallback key')
    if not all(math.isfinite(x) for x in (rf,pw,doa)):raise ValueError('Invalid fallback measurements')
    return scope+('waveform',f'{math.floor(rf/250.)}:{round(pw/.5)}:{math.floor((doa%360.)/5.)}')


@dataclass(slots=True)
class BinStats:
    key: tuple
    count: int=0
    first: int=0
    last: int=0
    sum_pa: float=0.
    sum_pa2: float=0.
    min_pa: float=float('inf')
    max_pa: float=-float('inf')
    known_pri_count: int=0
    sum_pri_us: float=0.
    gap_sum: float=0.
    gap_sum2: float=0.
    gap_max: float=0.
    sequences: set=field(default_factory=set)

    def add(self,t,pa,sequence_id='',pri_us=None):
        if not math.isfinite(pa) or t<0 or (self.count and t<self.last):raise ValueError('Invalid/unordered pulse in state source')
        if self.count:
            gap=(t-self.last)/1e9;self.gap_sum+=gap;self.gap_sum2+=gap*gap;self.gap_max=max(self.gap_max,gap)
        else:self.first=t
        self.last=t;self.count+=1;self.sum_pa+=pa;self.sum_pa2+=pa*pa
        self.min_pa=min(self.min_pa,pa);self.max_pa=max(self.max_pa,pa)
        if sequence_id:self.sequences.add(sequence_id)
        if pri_us is not None:
            if not math.isfinite(pri_us) or pri_us<=0:raise ValueError('Invalid PRI state normalization')
            self.known_pri_count+=1;self.sum_pri_us+=pri_us

    @property
    def mean_pa(self):return self.sum_pa/self.count

    @property
    def pa_std(self):return math.sqrt(max(0.,self.sum_pa2/self.count-self.mean_pa**2))


def features_for_group(stats,cfg):
    """Offline centered contexts, no observations/labels synthesized in gaps."""
    cfg.validate();stats=sorted(stats,key=lambda s:s.key[-1]);indices=[s.key[-1] for s in stats]
    if len(set(indices))!=len(indices):raise ValueError('Duplicate state source/time bin')
    rows=[]
    for j,s in enumerate(stats):
        mean_gap=s.gap_sum/(s.count-1) if s.count>1 else 0.
        gap_var=max(0.,s.gap_sum2/(s.count-1)-mean_gap**2) if s.count>1 else 0.
        gap_cv=math.sqrt(gap_var)/mean_gap if mean_gap>0 else 0.
        pri_s=s.sum_pri_us/s.known_pri_count/1e6 if s.known_pri_count else (mean_gap or .001)
        previous=max(0.,(s.first-stats[j-1].last)/1e9) if j else 10.
        following=max(0.,(stats[j+1].first-s.last)/1e9) if j+1<len(stats) else 10.
        span=(s.last-s.first)/1e9;pa_range=s.max_pa-s.min_pa
        features=[math.log1p(s.count),min(1.,span/cfg.bin_seconds),math.log1p(min(100.,s.sum_pri_us/1e6/cfg.bin_seconds)),
            s.known_pri_count/s.count,math.log1p(len(s.sequences)),min(100.,gap_cv),
            math.log1p(min(1e5,s.gap_max/max(pri_s,1e-9))),math.log1p(min(10.,previous)),math.log1p(min(10.,following)),
            s.pa_std,pa_range,(s.mean_pa-s.min_pa)/pa_range if pa_range>0 else .5]
        center=s.key[-1]
        for window in cfg.context_half_windows_seconds:
            radius=math.floor(window/cfg.bin_seconds)
            lo=bisect.bisect_left(indices,center-radius);hi=bisect.bisect_right(indices,center+radius)
            nearby=stats[lo:hi];n=sum(p.count for p in nearby);total=sum(p.sum_pa for p in nearby);squared=sum(p.sum_pa2 for p in nearby)
            mean=total/n;std=math.sqrt(max(0.,squared/n-mean*mean))
            bin_mean=sum(p.mean_pa for p in nearby)/len(nearby)
            bin_std=math.sqrt(sum((p.mean_pa-bin_mean)**2 for p in nearby)/len(nearby))
            before=[p for p in nearby if p.key[-1]<center];after=[p for p in nearby if p.key[-1]>center]
            change=(sum(p.mean_pa for p in after)/len(after)-sum(p.mean_pa for p in before)/len(before)) if before and after else 0.
            features.extend([len(nearby)/(2*radius+1),math.log1p(n/((2*radius+1)*cfg.bin_seconds)),std,bin_std,abs(change)])
        if len(features)!=len(feature_names(cfg)) or not all(math.isfinite(x) for x in features):raise ValueError('Invalid state feature vector')
        rows.append((s,tuple(features)))
    return rows
