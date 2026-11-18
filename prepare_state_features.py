#!/usr/bin/env python3
"""Cache label-free state features and optional separate TRAIN state truth."""
import argparse
import csv
import itertools
import json
import os
import time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path

from prepare_inputs import DEFAULT_FILES,digest
from src.dataio import time_to_ns
from src.state_features import StateFeatureConfig,BinStats,KEY_FIELDS,feature_names,source_key,features_for_group
from src.target_classification_io import read_sequences

ROOT=Path(__file__).resolve().parent
PULSE_KEYS=['scenario_id','data_version','platform_id','pulse_id']


def run(root,split,output,cfg):
    root,output=Path(root).resolve(),Path(output).resolve();cfg.validate();started=time.monotonic()
    if output.exists():raise FileExistsError(f'Choose a new state feature directory: {output}')
    raw=root.parent/DEFAULT_FILES[split][0]
    fusion=root/f'output/fused_b_assisted_guarded/{split}/pulse_object_predictions.csv'
    seq=root/f'output/pri/{split}/pri_sequences.csv'
    pri_meta=json.loads((root/f'output/pri/{split}/pri_run.json').read_text())
    fusion_meta=json.loads((root/f'output/fused_b_assisted_guarded/{split}/fusion_run.json').read_text())
    hashes={n:digest(p) for n,p in [('raw_A',raw),('fusion_pulses',fusion),('PRI_sequences',seq)]}
    if hashes['raw_A']!=pri_meta['source_sha256'] or hashes['fusion_pulses']!=fusion_meta['files']['pulse_object_predictions.csv']['sha256'] or hashes['PRI_sequences']!=pri_meta['files']['pri_sequences.csv']['sha256']:
        raise ValueError('State feature source hash mismatch')
    sequences=read_sequences(seq);bins={};truth=defaultdict(Counter);targets=defaultdict(Counter);target_classes={}
    rows=0;by_state=Counter();fallback=Counter();period=round(cfg.bin_seconds*1e9)
    with raw.open(encoding='utf-8-sig',newline='') as af,fusion.open(newline='') as ff:
        a,f=csv.reader(af),csv.reader(ff);ah,fh=next(a),next(f)
        ai,fi=([h.index(k) for k in PULSE_KEYS] for h in [ah,fh])
        ti,pi,ri,wi,di=[ah.index(k) for k in ['toa','param3','param1','param2','doa']]
        si,oi,gi=[fh.index(k) for k in ['sequence_id','object_id','local_cluster_id']]
        label_indices=[ah.index(k) for k in ['target_state','target_id','target_class']] if split=='train' else None
        for ar,fr in itertools.zip_longest(a,f):
            if ar is None or fr is None or any(ar[i]!=fr[j] for i,j in zip(ai,fi)):raise ValueError('Raw/fused state input key/order/count mismatch')
            scope=tuple(fr[i] for i in fi[:3]);t=time_to_ns(ar[ti],'us')
            group=source_key(scope,fr[oi],fr[gi],float(ar[ri]),float(ar[wi]),float(ar[di]))
            key=group+(t//period,)
            stat=bins.get(key)
            if stat is None:stat=BinStats(key);bins[key]=stat
            pri=sequences[scope+(fr[si],)].pri if fr[si] else None
            stat.add(t,float(ar[pi]),fr[si],pri)
            fallback[group[-2]]+=1
            # Ground truth is stored separately and never enters feature updates.
            if label_indices is not None:
                state,tid,klass=(ar[i] for i in label_indices)
                if state not in ['0','1','2'] or not tid:raise ValueError('Missing training state/identity truth')
                truth[key][state]+=1;targets[key][tid]+=1;by_state[state]+=1
                if tid in target_classes and target_classes[tid]!=klass:raise ValueError('Conflicting target class truth')
                target_classes[tid]=klass
            rows+=1
            if rows%2_000_000==0:print(json.dumps({'split':split,'cached_pulses':rows,'state_windows':len(bins)}),flush=True)
    if rows!=pri_meta['input_pulses']:raise ValueError('State feature input count mismatch')
    groups=defaultdict(list)
    for key,s in bins.items():groups[key[:-1]].append(s)
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    fields=list(KEY_FIELDS)+['pulse_count','first_time_ns','last_time_ns']+feature_names(cfg)
    with (partial/'window_features.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        for _,stats in sorted(groups.items()):
            for s,features in features_for_group(stats,cfg):writer.writerow([*s.key,s.count,s.first,s.last,*features])
    if split=='train':
        with (partial/'window_truth.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(list(KEY_FIELDS)+['state_0_pulses','state_1_pulses','state_2_pulses','target_counts_json'])
            for k in sorted(bins):w.writerow([*k,truth[k]['0'],truth[k]['1'],truth[k]['2'],json.dumps(dict(targets[k]),sort_keys=True,separators=(',',':'))])
    report={'split':split,'config':asdict(cfg),'features':feature_names(cfg),'input_pulses':rows,'window_count':len(bins),'source_group_count':len(groups),
        'source_kind_pulse_counts':dict(fallback),'state_truth_counts':dict(by_state) if split=='train' else None,
        'training_target_class_mapping':target_classes if split=='train' else None,
        'labels_used_in_features':False,'absolute_time_used_as_feature':False,'absolute_PA_RF_PW_PRI_DOA_used_as_features':False,
        'B_columns_read':[],'source_IDs_used_only_for_local_context':True,'centered_offline_context':True,
        'maximum_context_half_window_seconds':max(cfg.context_half_windows_seconds),
        'input_files':{n:{'path':str(p),'sha256':hashes[n]} for n,p in [('raw_A',raw),('fusion_pulses',fusion),('PRI_sequences',seq)]},
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['prepare_state_features.py','src/state_features.py']},
        'elapsed_seconds':round(time.monotonic()-started,2)}
    report['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(partial.iterdir())}
    (partial/'state_features_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({k:report[k] for k in ['split','input_pulses','window_count','state_truth_counts','elapsed_seconds']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/state_features')
    p.add_argument('--split',choices=['train','test','both'],default='both')
    a=p.parse_args()
    for split in (['train','test'] if a.split=='both' else [a.split]):run(a.pipeline_root,split,a.output_dir/split,StateFeatureConfig())


if __name__=='__main__':main()
