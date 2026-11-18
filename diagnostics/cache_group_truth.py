#!/usr/bin/env python3
"""Independent TRAIN-only sequence labels for evaluating grouping; never an input."""
import argparse
import csv
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path


def cache(source,predictions,output):
    if output.exists():
        raise FileExistsError(output)
    labels=defaultdict(Counter)
    n=0
    with source.open(encoding='utf-8-sig',newline='') as af,predictions.open(newline='') as pf:
        a,b=csv.reader(af),csv.reader(pf)
        ah,bh=next(a),next(b)
        keys=('scenario_id','data_version','platform_id','pulse_id')
        ai,bi=[ah.index(k) for k in keys],[bh.index(k) for k in keys]
        seq,tid,klass=bh.index('sequence_id'),ah.index('target_id'),ah.index('target_class')
        for raw,pred in itertools.zip_longest(a,b):
            if raw is None or pred is None or any(raw[i]!=pred[j] for i,j in zip(ai,bi)):
                raise ValueError('Source and PRI rows differ')
            if not raw[tid] or not raw[klass]:
                raise ValueError('Training labels required')
            if pred[seq]:
                labels[tuple(pred[i] for i in bi[:3])+(pred[seq],)][raw[tid],raw[klass]]+=1
            n+=1
            if n%3000000==0:
                print('truth_cache_rows',n,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as out:
        w=csv.writer(out)
        w.writerow(['scenario_id','data_version','platform_id','sequence_id','target_id','target_class','pulse_count'])
        for key,counts in sorted(labels.items()):
            for (target,klass),count in sorted(counts.items()):
                w.writerow([*key,target,klass,count])
    print(json.dumps({'input_pulses':n,'sequences':len(labels),'evaluation_only':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-file',type=Path,required=True)
    p.add_argument('--pri-predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cache(a.a_file,a.pri_predictions,a.output)


if __name__=='__main__':
    main()
