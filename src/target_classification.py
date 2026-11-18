"""Label-supervised waveform densities and label-free temporal object pooling.

Only RF, PW, estimated PRI and PRI model enter the class density. Object IDs
index a pooling container, never an identity-to-class lookup. Scores are model
evidence weights, not calibrated probabilities.
"""
from __future__ import annotations
import math
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass

CLASSES=tuple(f'R{i:02d}' for i in range(1,9))
FEATURES=('rf_mhz','pw_us','pri_us','pri_model')


@dataclass(frozen=True)
class ClassConfig:
    rf_bin_mhz: float=5.
    pw_bin_us: float=.05
    log_pri_bin_width: float=.02
    smoothing: float=.5
    joint_weight: float=.95
    maximum_sequence_weight: int=64
    object_time_bin_seconds: float=.5
    minimum_object_bins: int=3
    minimum_object_score: float=.7
    minimum_object_margin: float=.2
    strong_sequence_score: float=.9
    strong_sequence_margin: float=.5
    low_score_threshold: float=.7

    def validate(self):
        for k,v in asdict(self).items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError(f'Invalid {k}')
        for k in ['maximum_sequence_weight','minimum_object_bins']:
            if not isinstance(getattr(self,k),int):raise ValueError(f'{k} must be integer')
        for k in ['joint_weight','minimum_object_score','minimum_object_margin','strong_sequence_score','strong_sequence_margin','low_score_threshold']:
            if getattr(self,k)>1:raise ValueError(f'{k} must be <= 1')


def feature_bins(rf,pw,pri,model,cfg):
    if not math.isfinite(rf) or not math.isfinite(pw) or rf<=0 or pw<0:raise ValueError('Invalid RF/PW')
    if pri is not None:
        if not math.isfinite(pri) or pri<=0 or model not in ('stable','jittered'):raise ValueError('Invalid PRI/model')
    return (math.floor(rf/cfg.rf_bin_mhz),round(pw/cfg.pw_bin_us),
            math.floor(math.log(pri)/cfg.log_pri_bin_width) if pri is not None else None,
            model if pri is not None else None)


def normalize(logs):
    top=max(logs);values=[math.exp(x-top) for x in logs];total=sum(values)
    return tuple(x/total for x in values)


def top_class(scores):
    order=sorted(range(len(CLASSES)),key=lambda i:(-scores[i],CLASSES[i]))
    return CLASSES[order[0]],scores[order[0]],scores[order[0]]-scores[order[1]]


class WaveformClassModel:
    def __init__(self,cfg,counts=None):
        cfg.validate();self.config=cfg;self.counts=counts or {};self.cache={}

    def fit(self,rows):
        """Rows contain (rf,pw,pri,model,class,weight); no IDs or position features."""
        counts={c:{'total':0.,'rf':Counter(),'pw':Counter(),'models':Counter(),
                   'mode_pw':defaultdict(Counter),'mode_pri':defaultdict(Counter),
                   'joint':defaultdict(Counter)} for c in CLASSES}
        for rf,pw,pri,mode,label,weight in rows:
            if label not in counts or not math.isfinite(weight) or weight<=0:raise ValueError('Invalid training label/weight')
            r,p,t,m=feature_bins(rf,pw,pri,mode,self.config)
            if t is None:raise ValueError('Fit requires supported PRI sequences')
            w=min(weight,self.config.maximum_sequence_weight);d=counts[label]
            d['total']+=w;d['rf'][str(r)]+=w;d['pw'][str(p)]+=w;d['models'][m]+=w
            d['mode_pw'][m][str(p)]+=w;d['mode_pri'][m][str(t)]+=w;d['joint'][m][f'{p}:{t}']+=w
        if any(d['total']<=0 for d in counts.values()):raise ValueError('Training must contain all eight classes')
        # Freeze plain dictionaries for deterministic JSON model storage.
        self.counts={c:{'total':d['total'],'rf':dict(d['rf']),'pw':dict(d['pw']),'models':dict(d['models']),
            'mode_pw':{m:dict(x) for m,x in d['mode_pw'].items()},
            'mode_pri':{m:dict(x) for m,x in d['mode_pri'].items()},
            'joint':{m:dict(x) for m,x in d['joint'].items()}} for c,d in counts.items()}
        self.cache={};return self

    @staticmethod
    def smooth(hist,index,alpha,kernel):
        weights=sum(w for _,w in kernel)
        return sum(w*(hist.get(str(index+delta),0.)+alpha) for delta,w in kernel)/weights

    def predict(self,rf,pw,pri=None,mode=None):
        key=feature_bins(rf,pw,pri,mode,self.config)
        if key in self.cache:return self.cache[key]
        if not self.counts:raise ValueError('Model not fitted')
        r,p,t,m=key;a=self.config.smoothing
        rf_kernel=((-1,.5),(0,1.),(1,.5));pw_kernel=((-2,.25),(-1,.5),(0,1.),(1,.5),(2,.25))
        pri_kernel=((-1,.5),(0,1.),(1,.5))
        logs=[]
        for c in CLASSES:
            d=self.counts[c];n=d['total']
            prf=self.smooth(d['rf'],r,a,rf_kernel)/(n+a*4096)
            if t is None:
                ppw=self.smooth(d['pw'],p,a,pw_kernel)/(n+a*256)
                likelihood=prf*ppw
            else:
                nm=d['models'].get(m,0.);pm=(nm+a)/(n+2*a)
                joint=d['joint'].get(m,{})
                total_weight=sum(w for _,w in pw_kernel)*sum(w for _,w in pri_kernel)
                pj=sum(wp*wt*(joint.get(f'{p+dp}:{t+dt}',0.)+a) for dp,wp in pw_kernel for dt,wt in pri_kernel)/total_weight
                pj/=nm+a*4096
                ppw=self.smooth(d['mode_pw'].get(m,{}),p,a,pw_kernel)/(nm+a*256)
                ppri=self.smooth(d['mode_pri'].get(m,{}),t,a,pri_kernel)/(nm+a*512)
                likelihood=prf*pm*(self.config.joint_weight*pj+(1-self.config.joint_weight)*ppw*ppri)
            logs.append(math.log(max(likelihood,1e-300)))
        result=normalize(logs);self.cache[key]=result
        return result

    def to_dict(self):
        return {'format':'B04_waveform_class_density_v1','classes':list(CLASSES),'features':list(FEATURES),
                'config':asdict(self.config),'counts':self.counts,'score_interpretation':'Uncalibrated class evidence weights'}

    @classmethod
    def from_dict(cls,data):
        if data['format']!='B04_waveform_class_density_v1' or data['classes']!=list(CLASSES) or data['features']!=list(FEATURES):
            raise ValueError('Unknown model format/features')
        model=cls(ClassConfig(**data['config']),data['counts'])
        if set(model.counts)!=set(CLASSES) or any(d['total']<=0 for d in model.counts.values()):raise ValueError('Invalid stored class model')
        return model


def pool_objects(rows,cfg):
    """Rows: (object_key,time_ns,scores,pulse_count). One vote per time bin.

    Bin means stop simultaneous platform copies/dense sequences from becoming
    independent evidence. Mean log evidence (not its sum) limits saturation.
    """
    bins={};period=round(cfg.object_time_bin_seconds*1e9)
    if period<=0:raise ValueError('Object bin too short')
    for key,t,scores,count in rows:
        if not key[-1]:continue
        target=bins.setdefault((key,t//period),[0.,[0.]*len(CLASSES)])
        w=min(count,cfg.maximum_sequence_weight);target[0]+=w
        for i,p in enumerate(scores):target[1][i]+=w*math.log(max(p,1e-6))
    summaries={}
    for (key,_),(w,totals) in bins.items():
        value=summaries.setdefault(key,[0,[0.]*len(CLASSES)])
        value[0]+=1
        for i,x in enumerate(totals):value[1][i]+=x/w
    result={}
    for key,(n,logs) in summaries.items():
        scores=normalize([x/n for x in logs]);label,score,margin=top_class(scores)
        reliable=(n>=cfg.minimum_object_bins and score>=cfg.minimum_object_score and margin>=cfg.minimum_object_margin)
        result[key]={'scores':scores,'target_class':label,'score':score,'margin':margin,'time_bin_count':n,'reliable':reliable}
    return result


def combine(scores,object_evidence,cfg):
    if object_evidence is None or not object_evidence['reliable']:return scores,'waveform_only'
    label,score,margin=top_class(scores)
    if label!=object_evidence['target_class'] and score>=cfg.strong_sequence_score and margin>=cfg.strong_sequence_margin:
        return scores,'strong_waveform_disagrees_with_object'
    pooled=normalize([math.log(max(x,1e-12))+math.log(max(y,1e-12)) for x,y in zip(scores,object_evidence['scores'])])
    return pooled,'waveform_and_object_context'
