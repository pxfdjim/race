#!/usr/bin/env python3
"""Attach observable short-sequence morphology and separate exact state truth."""
import argparse
import bisect
import csv
import itertools
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

from prepare_inputs import DEFAULT_FILES,digest
from src.state_features import source_key,KEY_FIELDS
from src.state_io import read_windows

ROOT=Path(__file__).resolve().parent
SEQ_KEYS=['scenario_id','data_version','platform_id','sequence_id']
EXTRA=['sequence_log_pulses','sequence_log_span_seconds','sequence_density','sequence_direct_interval_fraction',
       'sequence_relative_PRI_width','sequence_local_PA_rank','sequence_PA_above_local_low','sequence_PA_below_local_high',
       'sequence_local_PA_range','sequence_log_count_ratio','sequence_log_span_ratio','native_context_available']


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def run(root,window_root,split,output):
    root,window_root,output=map(lambda p:Path(p).resolve(),(root,window_root,output))
    if output.exists():raise FileExistsError(output)
    directory=window_root/split;meta=json.loads((directory/'state_features_run.json').read_text())
    native=read_windows(directory/'window_features.csv',meta['features']);period=round(meta['config']['bin_seconds']*1e9)
    seq_path=root/f'output/pri/{split}/pri_sequences.csv';link_path=root/f'output/fused_b_assisted_guarded/{split}/sequence_object_links.csv'
    links={tuple(r[k] for k in SEQ_KEYS):r for r in read(link_path)}
    records={};groups=defaultdict(list)
    for r in read(seq_path):
        k=tuple(r[n] for n in SEQ_KEYS);l=links[k]
        group=source_key(k[:3],l['object_id'],l['local_cluster_id'],float(r['rf_mean']),float(r['pw_mean']),float(r['doa_mean_degrees']))
        start,end=int(r['start_toa_ns']),int(r['end_toa_ns']);t=(start+end)//2
        item={'key':k,'group':group,'time':t,'start':start,'end':end,'count':int(r['pulse_count']),
              'pri':float(r['pri_center_us']),'pa':float(r['pa_mean']),
              'direct':float(r['direct_interval_fraction']),
              'relative_width':(float(r['pri_p95_us'])-float(r['pri_p05_us']))/float(r['pri_center_us'])}
        records[k]=item;groups[group].append(item)
    features={}
    for group,items in groups.items():
        items.sort(key=lambda s:s['time']);times=[s['time'] for s in items]
        for s in items:
            lo=bisect.bisect_left(times,s['time']-20_000_000_000);hi=bisect.bisect_right(times,s['time']+20_000_000_000)
            nearby=items[lo:hi];amplitudes=sorted(n['pa'] for n in nearby)
            # Relative amplitude and sequence duration; no source/target ID values.
            low=amplitudes[int((len(amplitudes)-1)*.1)];high=amplitudes[int((len(amplitudes)-1)*.9)]
            rank=bisect.bisect_right(amplitudes,s['pa'])/len(amplitudes)
            count_med=median(n['count'] for n in nearby)
            span=(s['end']-s['start'])/1e9;span_med=median((n['end']-n['start'])/1e9 for n in nearby)
            base=native.get(group+(s['time']//period,))
            x=list(base.features) if base else [0.]*len(meta['features'])
            x.extend([math.log1p(s['count']),math.log1p(span),min(10.,(s['count']-1)*s['pri']/1e6/max(span,1e-9)),
                s['direct'],s['relative_width'],rank,s['pa']-low,high-s['pa'],high-low,
                math.log(max(1,s['count'])/max(1,count_med)),math.log(max(span,1e-6)/max(span_med,1e-6)),float(base is not None)])
            features[s['key']]=x
    truth=defaultdict(Counter);targets=defaultdict(Counter)
    if split=='train':
        raw=root.parent/DEFAULT_FILES[split][0];fused=root/f'output/fused_b_assisted_guarded/{split}/pulse_object_predictions.csv'
        with raw.open(encoding='utf-8-sig',newline='') as af,fused.open(newline='') as ff:
            a,f=csv.reader(af),csv.reader(ff);ah,fh=next(a),next(f)
            names=['scenario_id','data_version','platform_id','pulse_id'];ai,fi=([h.index(n) for n in names] for h in [ah,fh])
            st,tid=ah.index('target_state'),ah.index('target_id');si=fh.index('sequence_id');rows=0
            for ar,fr in itertools.zip_longest(a,f):
                if ar is None or fr is None or any(ar[i]!=fr[j] for i,j in zip(ai,fi)):raise ValueError('Sequence state source mismatch')
                if fr[si]:
                    k=tuple(fr[i] for i in fi[:3])+(fr[si],)
                    if ar[st] not in ['0','1','2']:raise ValueError('Invalid training state')
                    truth[k][ar[st]]+=1;targets[k][ar[tid]]+=1
                rows+=1
                if rows%3_000_000==0:print(json.dumps({'state_sequence_truth_pulses':rows}),flush=True)
        if any(sum(truth[k].values())!=s['count'] for k,s in records.items()):raise ValueError('Sequence state truth count mismatch')
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    names=meta['features']+EXTRA
    with (partial/'window_features.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(list(KEY_FIELDS)+['pulse_count','first_time_ns','last_time_ns','sequence_id']+names)
        for k,s in sorted(records.items()):w.writerow([*s['group'],s['time']//period,s['count'],s['start'],s['end'],k[3],*features[k]])
    # Sequence keys remain unique even when several centers share the same native bin.
    # The state reader's source_id is extended with sequence_id; temporal decoding
    # for this estimator is disabled, so no artificial sequence-to-sequence links.
    with (partial/'window_features.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:r['source_id']=r['source_id']+'~'+r['sequence_id']
    with (partial/'window_features.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    if split=='train':
        with (partial/'window_truth.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(list(KEY_FIELDS)+['state_0_pulses','state_1_pulses','state_2_pulses','target_counts_json'])
            for k,s in sorted(records.items()):
                group=s['group'][:-1]+(s['group'][-1]+'~'+k[3],)
                w.writerow([*group,s['time']//period,truth[k]['0'],truth[k]['1'],truth[k]['2'],json.dumps(dict(targets[k]),separators=(',',':'))])
    report={**meta,'features':names,'window_count':len(records),'input_pulses':sum(s['count'] for s in records.values()),
        'sequence_estimator':True,'maximum_context_half_window_seconds':20.,
        'source_files':{'sequences':{'path':str(seq_path),'sha256':digest(seq_path)},'links':{'path':str(link_path),'sha256':digest(link_path)}},
        'algorithm_source_sha256':{'prepare_sequence_states.py':digest(ROOT/'prepare_sequence_states.py')}}
    report['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in partial.iterdir()}
    (partial/'state_features_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({'split':split,'sequence_state_records':len(records)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--window-root',type=Path,default=ROOT/'output/state_features')
    p.add_argument('--split',choices=['train','test'],required=True);p.add_argument('--output-dir',type=Path,default=ROOT/'output/sequence_state_features')
    a=p.parse_args();run(a.pipeline_root,a.window_root,a.split,a.output_dir/a.split)


if __name__=='__main__':main()
