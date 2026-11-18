"""Read PRI waveform summaries, frozen fusion memberships and separate train labels."""
import csv
import math
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
from .target_classification import CLASSES

SEQ_KEYS=['scenario_id','data_version','platform_id','sequence_id']
PULSE_KEYS=SEQ_KEYS[:3]+['pulse_id']


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


@dataclass(frozen=True,slots=True)
class ClassSequence:
    key: tuple
    start: int
    end: int
    count: int
    rf: float
    pw: float
    pri: float
    model: str

    @property
    def time(self):return (self.start+self.end)//2

    def predict(self,model):return model.predict(self.rf,self.pw,self.pri,self.model)


def read_sequences(path):
    sequences={}
    for r in read_rows(path):
        k=tuple(r[n] for n in SEQ_KEYS)
        s=ClassSequence(k,int(r['start_toa_ns']),int(r['end_toa_ns']),int(r['pulse_count']),
            float(r['rf_mean']),float(r['pw_mean']),float(r['pri_center_us']),r['pri_model'])
        if (k in sequences or not all(k) or s.start<0 or s.end<s.start or s.count<2 or
            not all(math.isfinite(x) for x in (s.rf,s.pw,s.pri)) or s.rf<=0 or s.pw<0 or s.pri<=0 or s.model not in ['stable','jittered']):
            raise ValueError('Invalid/duplicate PRI sequence')
        sequences[k]=s
    return sequences


def read_links(path,sequences):
    result={}
    for r in read_rows(path):
        k=tuple(r[n] for n in SEQ_KEYS)
        if k in result or k not in sequences:raise ValueError('Invalid fusion sequence key')
        result[k]=k[:2]+(r['object_id'],)
    if result.keys()!=sequences.keys():raise ValueError('Missing fusion sequence memberships')
    return result


def read_truth(path,sequences):
    classes=defaultdict(Counter);targets=defaultdict(Counter);mapping={}
    for r in read_rows(path):
        k=tuple(r[n] for n in SEQ_KEYS);label=r['target_class'];tid=r['target_id'];n=int(r['pulse_count'])
        if k not in sequences or label not in CLASSES or not tid or n<=0:raise ValueError('Invalid training truth row')
        if tid in mapping and mapping[tid]!=label:raise ValueError('Target instance has conflicting class truth')
        mapping[tid]=label;classes[k][label]+=n;targets[k][tid]+=n
    if classes.keys()!=sequences.keys() or any(sum(classes[k].values())!=s.count for k,s in sequences.items()):
        raise ValueError('Truth cache does not cover PRI sequences/counts exactly')
    return dict(classes),dict(targets),mapping
