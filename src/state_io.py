"""Separate readers for observable state windows and training-only truth."""
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from .state_features import KEY_FIELDS


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def window_key(row):return tuple(row[n] for n in KEY_FIELDS[:-1])+(int(row['bin_index']),)


@dataclass(frozen=True,slots=True)
class StateWindow:
    key: tuple
    count: int
    first: int
    last: int
    features: tuple


def read_windows(path,names):
    result={}
    for r in rows(path):
        k=window_key(r)
        w=StateWindow(k,int(r['pulse_count']),int(r['first_time_ns']),int(r['last_time_ns']),tuple(float(r[n]) for n in names))
        if k in result or not all(k[:-1]) or w.count<1 or w.first<0 or w.last<w.first or not all(math.isfinite(x) for x in w.features):raise ValueError('Invalid state feature window')
        result[k]=w
    return result


def read_truth(path,windows):
    labels={};targets={}
    for r in rows(path):
        k=window_key(r);counts=tuple(int(r[f'state_{s}_pulses']) for s in range(3));ids=json.loads(r['target_counts_json'])
        if k in labels or k not in windows or any(n<0 for n in counts) or sum(counts)!=windows[k].count or sum(ids.values())!=windows[k].count:
            raise ValueError('State truth cache mismatch')
        labels[k]=counts;targets[k]=ids
    if labels.keys()!=windows.keys():raise ValueError('Incomplete state truth')
    return labels,targets


def state_metrics(keys,probabilities,truth):
    confusion=[[0]*3 for _ in range(3)];correct=total=oracle=uncertain=0
    for k,ps in zip(keys,probabilities):
        pred=max(range(3),key=lambda i:ps[i]);counts=truth[k]
        total+=sum(counts);correct+=counts[pred];oracle+=max(counts)
        if max(ps)<.65:uncertain+=sum(counts)
        for state,n in enumerate(counts):confusion[state][pred]+=n
    per={}
    for state in range(3):
        support=sum(confusion[state]);predicted=sum(r[state] for r in confusion);tp=confusion[state][state]
        p=tp/predicted if predicted else 0.;r=tp/support if support else 0.
        per[str(state)]={'support_pulses':support,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
    present=[v for v in per.values() if v['support_pulses']]
    binary_correct=confusion[0][0]+sum(confusion[i][j] for i in [1,2] for j in [1,2])
    return {'window_count':len(keys),'supported_pulses':total,'pulse_accuracy':correct/total if total else None,
        'macro_f1_present_states':sum(v['f1'] for v in present)/len(present) if present else None,
        'binary_search_vs_tracking_accuracy':binary_correct/total if total else None,
        'window_resolution_oracle_accuracy':oracle/total if total else None,
        'low_evidence_pulse_fraction':uncertain/total if total else 0.,'per_state':per,'confusion_matrix_pulses':confusion}
