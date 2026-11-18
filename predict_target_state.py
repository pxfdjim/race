#!/usr/bin/env python3
"""Predict legal per-pulse states with sequence estimates and window fallback."""
import argparse
import csv
import itertools
import json
import math
import os
import time
from collections import Counter,defaultdict
from pathlib import Path

from prepare_inputs import DEFAULT_FILES,digest
from src.dataio import time_to_ns
from src.state_features import KEY_FIELDS,source_key
from src.state_io import read_windows,rows,window_key
from src.state_model import StateForest,StateModelConfig,smooth_states

ROOT=Path(__file__).resolve().parent
PULSE_KEYS=['scenario_id','data_version','platform_id','pulse_id']
SEQ_KEYS=PULSE_KEYS[:3]+['sequence_id']


def predict_cache(directory,model):
    directory=Path(directory);meta=json.loads((directory/'state_features_run.json').read_text())
    if meta['features']!=model.features or meta['labels_used_in_features']:raise ValueError('State model/feature schema mismatch')
    if digest(directory/'window_features.csv')!=meta['files']['window_features.csv']['sha256']:raise ValueError('State features changed')
    windows=read_windows(directory/'window_features.csv',model.features);keys=list(windows)
    raw=[model.predict(windows[k].features) for k in keys]
    probabilities=smooth_states(keys,raw,model.transitions,model.config.temporal_strength)
    return meta,windows,dict(zip(keys,probabilities))


def run(root,feature_root,sequence_root,model_path,output,split,records_only=False,profile='accuracy'):
    root,feature_root,sequence_root,model_path,output=map(lambda p:Path(p).resolve(),(root,feature_root,sequence_root,model_path,output))
    if output.exists():raise FileExistsError(f'Choose a new state output directory: {output}')
    started=time.monotonic();bundle=json.loads(model_path.read_text())
    if bundle['format']!='B04_target_state_bundle_v1':raise ValueError('Unknown state bundle')
    wm=StateForest.from_dict(bundle['window_model']);sm=StateForest.from_dict(bundle['sequence_model'])
    if profile=='balanced':
        if 'balanced_config' not in bundle:raise ValueError('No balanced profile in state bundle')
        sm.config=StateModelConfig(**bundle['balanced_config'])
    wmeta,windows,wp=predict_cache(feature_root/split,wm)
    smeta,sequences,sp=predict_cache(sequence_root/split,sm)
    seq_index={};seq_rows=[]
    for r in rows(sequence_root/split/'window_features.csv'):
        k=window_key(r);sk=tuple(r[n] for n in SEQ_KEYS)
        if sk in seq_index:raise ValueError('Duplicate state sequence')
        seq_index[sk]=sp[k]
        seq_rows.append((sk,k,sequences[k],sp[k]))
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    def summary(p):
        state=max(range(3),key=lambda i:p[i]);uncertain=max(p)<.65
        tracking=p[1]+p[2];ambiguous=tracking>.5 and abs(p[1]-p[2])/tracking<.25
        return state,p[state],uncertain,ambiguous
    def pulse_summary(p):
        state,score,uncertain,ambiguous=summary(p)
        return state,f'{score:.8f}',uncertain,ambiguous
    sequence_pulse_outputs={k:pulse_summary(p) for k,p in seq_index.items()}
    window_pulse_outputs={k:pulse_summary(p) for k,p in wp.items()}
    fields=list(KEY_FIELDS)+['pulse_count','first_time_ns','last_time_ns','target_state','state_evidence_score','state_low_evidence','coarse_fine_ambiguous','evidence_0','evidence_1','evidence_2']
    with (partial/'window_state_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for k,s in sorted(windows.items()):w.writerow([*k,s.count,s.first,s.last,*summary(wp[k]),*wp[k]])
    with (partial/'sequence_state_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(SEQ_KEYS+['source_kind','source_id','bin_index','pulse_count','first_time_ns','last_time_ns','target_state','state_evidence_score','state_low_evidence','coarse_fine_ambiguous','evidence_0','evidence_1','evidence_2'])
        for sk,k,s,p in sorted(seq_rows):w.writerow([*sk,k[3],k[4].rsplit('~',1)[0],k[5],s.count,s.first,s.last,*summary(p),*p])
    pulse_report=None;timeline=defaultdict(lambda:[0,0,0]);source_paths={
        'window_features':feature_root/split/'window_features.csv','sequence_features':sequence_root/split/'window_features.csv','model':model_path}
    if not records_only:
        raw=root.parent/DEFAULT_FILES[split][0];fused=root/f'output/fused_b_assisted_guarded/{split}/pulse_object_predictions.csv'
        for n,p in [('raw_A',raw),('fusion_pulses',fused)]:
            if digest(p)!=wmeta['input_files'][n]['sha256']:raise ValueError('Raw/fused state source changed')
        source_paths.update(raw_A=raw,fusion_pulses=fused)
        labels=Counter();bases=Counter();count=low=ambiguous=0;period=round(wmeta['config']['bin_seconds']*1e9)
        with raw.open(encoding='utf-8-sig',newline='') as af,fused.open(newline='') as ff,(partial/'pulse_state_predictions.csv').open('w',newline='') as out:
            a,f=csv.reader(af),csv.reader(ff);ah,fh=next(a),next(f)
            ai,fi=([h.index(k) for k in PULSE_KEYS] for h in [ah,fh]);si,oi,gi=[fh.index(k) for k in ['sequence_id','object_id','local_cluster_id']]
            ti,ri,wi,di=[ah.index(k) for k in ['toa','param1','param2','doa']]
            w=csv.writer(out);w.writerow(PULSE_KEYS+['sequence_id','local_cluster_id','object_id','target_state','state_evidence_score','state_basis','state_low_evidence','coarse_fine_ambiguous'])
            for ar,fr in itertools.zip_longest(a,f):
                if ar is None or fr is None or any(ar[i]!=fr[j] for i,j in zip(ai,fi)):raise ValueError('State pulse source order/count/key mismatch')
                key=tuple(fr[i] for i in fi);t=time_to_ns(ar[ti],'us');index=t//period
                if fr[si]:result=sequence_pulse_outputs[key[:3]+(fr[si],)];basis='sequence_temporal_features'
                else:
                    group=source_key(key[:3],fr[oi],fr[gi],float(ar[ri]),float(ar[wi]),float(ar[di]))
                    result=window_pulse_outputs[group+(index,)];basis='unknown_PRI_window_fallback'
                state,score,uncertain,cf=result
                w.writerow([*key,fr[si],fr[gi],fr[oi],state,score,basis,uncertain,cf])
                labels[str(state)]+=1;bases[basis]+=1;count+=1;low+=uncertain;ambiguous+=cf
                if fr[oi]:timeline[key[:2]+(fr[oi],index)][state]+=1
                if count%2_000_000==0:print(json.dumps({'split':split,'state_predicted_pulses':count}),flush=True)
        if count!=wmeta['input_pulses']:raise ValueError('State pulse count mismatch')
        pulse_report={'input_output_pulses':count,'state_counts':dict(labels),'basis_counts':dict(bases),'low_evidence_pulses':low,
                      'coarse_fine_ambiguous_pulses':ambiguous}
    with (partial/'object_state_timeline.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scenario_id','data_version','object_id','bin_index','time_seconds','target_state','state_0_pulses','state_1_pulses','state_2_pulses','dominant_fraction'])
        for k,counts in sorted(timeline.items()):
            state=max(range(3),key=lambda i:counts[i]);w.writerow([*k,k[3]*wmeta['config']['bin_seconds'],state,*counts,max(counts)/sum(counts)])
    report={'split':split,'profile':profile,'states':{'0':'search','1':'coarse_tracking','2':'fine_tracking'},'pulse_export':pulse_report,
        'window_count':len(windows),'sequence_count':len(seq_index),'object_state_time_bins':len(timeline),
        'state_is_time_varying':True,'unobserved_times_filled':False,'truth_columns_read_for_inference':[],'B_columns_read':[],
        'observed_waveform_and_PA_calibration_used':any(n.startswith('observed_') for n in sm.features),
        'upstream_B_assisted_memberships':True,'complete_submission_generated':False,
        'feature_config':wmeta['config'],'input_files':{n:{'path':str(p),'sha256':digest(p)} for n,p in source_paths.items()},
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['predict_target_state.py','src/state_features.py','src/state_model.py','src/state_io.py']},
        'limitations':['Evidence weights are not calibrated correctness probabilities.',
            'Tracking state 1 vs 2 may remain ambiguous; all legal top-1 states are retained with flags.',
            'States inherit source memberships and short-sequence extraction errors.',
            'Offline centered context up to 20 seconds is used; no predictions are made in unobserved time.'],
        'elapsed_seconds':round(time.monotonic()-started,2)}
    report['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(partial.iterdir())}
    (partial/'state_prediction_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({'split':split,'pulse_export':pulse_report,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--feature-root',type=Path,default=ROOT/'output/state_features');p.add_argument('--sequence-root',type=Path)
    p.add_argument('--model',type=Path,default=ROOT/'models/target_state/target_state_model.json');p.add_argument('--output-dir',type=Path,default=ROOT/'output/state_predicted')
    p.add_argument('--split',choices=['train','test','both'],default='both');p.add_argument('--records-only',action='store_true')
    p.add_argument('--profile',choices=['accuracy','balanced'],default='accuracy')
    a=p.parse_args()
    bundle=json.loads(a.model.read_text())
    selected_features=bundle['sequence_model']['features']
    sequence_root=a.sequence_root or ROOT/('output/calibrated_sequence_state_features' if any(n.startswith('observed_') for n in selected_features) else 'output/sequence_state_features')
    for split in (['train','test'] if a.split=='both' else [a.split]):run(a.pipeline_root,a.feature_root,sequence_root,a.model,a.output_dir/split,split,a.records_only,a.profile)


if __name__=='__main__':main()
