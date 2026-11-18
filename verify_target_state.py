#!/usr/bin/env python3
"""Check all per-pulse state records, with optional separate TRAIN evaluation."""
import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from prepare_inputs import digest

PULSE_KEYS=['scenario_id','data_version','platform_id','pulse_id']
SEQ_KEYS=PULSE_KEYS[:3]+['sequence_id']


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def evidence(row):
    p=[float(row[f'evidence_{s}']) for s in range(3)];state=max(range(3),key=lambda s:p[s])
    if any(not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-7 or row['target_state']!=str(state):raise ValueError('Invalid state evidence/label')
    if abs(max(p)-float(row['state_evidence_score']))>1e-8:raise ValueError('Invalid state evidence score')


def verify(directory,evaluate_train=False):
    directory=Path(directory);meta=json.loads((directory/'state_prediction_run.json').read_text())
    if evaluate_train and meta['split']!='train':raise ValueError('Test state truth is unavailable')
    for n,info in meta['files'].items():
        if digest(directory/n)!=info['sha256']:raise ValueError(f'Changed state output: {n}')
    for n,info in meta['input_files'].items():
        if digest(info['path'])!=info['sha256']:raise ValueError(f'Changed state input: {n}')
    if meta['truth_columns_read_for_inference'] or meta['B_columns_read'] or meta['unobserved_times_filled']:raise ValueError('Unexpected state inference behavior')
    windows=set()
    for row in rows(directory/'window_state_predictions.csv'):
        k=tuple(row[n] for n in ['scenario_id','data_version','platform_id','source_kind','source_id','bin_index'])
        if k in windows:raise ValueError('Duplicate state window')
        windows.add(k);evidence(row)
    sequences={}
    for row in rows(directory/'sequence_state_predictions.csv'):
        k=tuple(row[n] for n in SEQ_KEYS)
        if k in sequences:raise ValueError('Duplicate state sequence')
        sequences[k]=row;evidence(row)
    if len(windows)!=meta['window_count'] or len(sequences)!=meta['sequence_count']:raise ValueError('State record count mismatch')
    result={'passed':True,'input_and_output_hashes_match':True,'legal_normalized_state_records':True,
            'verified_windows':len(windows),'verified_sequences':len(sequences),'unobserved_states_synthesized':False}
    if meta['pulse_export'] is not None:
        paths=meta['input_files'];count=low=ambiguous=correct=0;states=Counter();bases=Counter();confusion=[[0]*3 for _ in range(3)]
        with Path(paths['raw_A']['path']).open(encoding='utf-8-sig',newline='') as af,Path(paths['fusion_pulses']['path']).open(newline='') as ff,(directory/'pulse_state_predictions.csv').open(newline='') as pf:
            a,f,p=csv.reader(af),csv.reader(ff),csv.reader(pf);ah,fh,ph=next(a),next(f),next(p)
            ai,fi,pi=([h.index(n) for n in PULSE_KEYS] for h in [ah,fh,ph])
            fs,fo,fg=[fh.index(n) for n in ['sequence_id','object_id','local_cluster_id']]
            ps,po,pg,st,sc,bs,le,ca=[ph.index(n) for n in ['sequence_id','object_id','local_cluster_id','target_state','state_evidence_score','state_basis','state_low_evidence','coarse_fine_ambiguous']]
            truth_index=ah.index('target_state') if evaluate_train else None
            for ar,fr,pr in itertools.zip_longest(a,f,p):
                if ar is None or fr is None or pr is None:raise ValueError('State pulse count mismatch')
                if any(ar[i]!=fr[j] or ar[i]!=pr[k] for i,j,k in zip(ai,fi,pi)):raise ValueError('State pulse keys/order changed')
                if pr[ps]!=fr[fs] or pr[po]!=fr[fo] or pr[pg]!=fr[fg]:raise ValueError('State output modified source memberships')
                state=pr[st];score=float(pr[sc])
                if state not in ['0','1','2'] or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Illegal pulse state/evidence')
                if fr[fs]:
                    seq=sequences[tuple(fr[i] for i in fi[:3])+(fr[fs],)]
                    if state!=seq['target_state'] or abs(score-float(seq['state_evidence_score']))>1e-8:raise ValueError('Pulse/sequence state mismatch')
                elif pr[bs]!='unknown_PRI_window_fallback':raise ValueError('Unknown PRI state basis missing')
                states[state]+=1;bases[pr[bs]]+=1;low+=pr[le]=='True';ambiguous+=pr[ca]=='True';count+=1
                if evaluate_train:
                    true=ar[truth_index]
                    if true not in ['0','1','2']:raise ValueError('Training state truth missing')
                    confusion[int(true)][int(state)]+=1;correct+=state==true
                if count%3_000_000==0:print(json.dumps({'verified_state_pulses':count}),flush=True)
        pulse=meta['pulse_export']
        if count!=pulse['input_output_pulses'] or dict(states)!=pulse['state_counts'] or dict(bases)!=pulse['basis_counts'] or low!=pulse['low_evidence_pulses'] or ambiguous!=pulse['coarse_fine_ambiguous_pulses']:
            raise ValueError('State pulse summary mismatch')
        result.update(verified_pulses=count,all_raw_fused_state_keys_and_order_equal=True,all_memberships_unchanged=True,state_counts=dict(states))
        if evaluate_train:result['training_resubstitution_evaluation']={'pulse_accuracy':correct/count,'confusion_matrix_pulses':confusion,
            'interpretation':'Descriptive in-sample check after full training refit, not held-out accuracy.'}
    (directory/'state_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--evaluate-train',action='store_true')
    a=p.parse_args();verify(a.directory,a.evaluate_train)


if __name__=='__main__':main()
