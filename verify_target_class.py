#!/usr/bin/env python3
"""Check all class outputs and optionally evaluate refitted TRAIN pulse labels."""
import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from prepare_inputs import digest
from src.target_classification import CLASSES

KEYS=['scenario_id','data_version','platform_id','sequence_id']
PULSE_KEYS=KEYS[:3]+['pulse_id']


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def evidence(r):
    scores=[float(r[f'evidence_{c}']) for c in CLASSES]
    if not all(math.isfinite(x) and 0<=x<=1 for x in scores) or abs(sum(scores)-1)>1e-8:raise ValueError('Invalid class evidence')
    label=CLASSES[max(range(len(scores)),key=lambda i:scores[i])]
    if label!=r['target_class'] or abs(max(scores)-float(r['class_evidence_score']))>1e-8:raise ValueError('Label/evidence mismatch')


def verify(directory,evaluate_train=False):
    directory=Path(directory);report=json.loads((directory/'classification_run.json').read_text())
    if evaluate_train and report['split']!='train':raise ValueError('Test data has no class truth')
    for name,info in report['files'].items():
        if digest(directory/name)!=info['sha256']:raise ValueError(f'Changed class output: {name}')
    for name,info in report['input_files'].items():
        if digest(info['path'])!=info['sha256']:raise ValueError(f'Changed class input: {name}')
    if report['A_truth_columns_read'] or report['B_columns_read'] or report['state_predicted']:raise ValueError('Unexpected class inference features')
    sources=report['input_files']
    sequence_counts={tuple(r[k] for k in KEYS):int(r['pulse_count']) for r in rows(sources['sequences']['path'])}
    memberships={tuple(r[k] for k in KEYS):r['object_id'] for r in rows(sources['fusion_links']['path'])}
    objects={}
    for r in rows(directory/'object_class_predictions.csv'):
        key=r['scenario_id'],r['data_version'],r['object_id']
        if key in objects or not key[2] or r['target_class'] not in CLASSES:raise ValueError('Invalid object classification')
        evidence(r);objects[key]=r
    expected_objects={k[:2]+(oid,) for k,oid in memberships.items() if oid}
    if objects.keys()!=expected_objects:raise ValueError('Missing object class prediction')
    sequence_predictions={}
    for r in rows(directory/'sequence_class_predictions.csv'):
        key=tuple(r[k] for k in KEYS)
        if key in sequence_predictions or key not in sequence_counts or r['object_id']!=memberships[key] or int(r['pulse_count'])!=sequence_counts[key]:
            raise ValueError('Invalid sequence class key/count/identity')
        if r['target_class'] not in CLASSES or r['waveform_target_class'] not in CLASSES:raise ValueError('Illegal sequence class')
        evidence(r);sequence_predictions[key]=r
    if sequence_predictions.keys()!=sequence_counts.keys():raise ValueError('Missing class sequence')
    result={'passed':True,'input_and_output_hashes_match':True,'object_and_sequence_membership_complete':True,
            'all_labels_legal':True,'evidence_vectors_normalized':True,'verified_sequences':len(sequence_predictions),
            'verified_objects':len(objects),'state_predicted':False}
    if report['pulse_export'] is not None:
        n=low=correct=0;labels=Counter();bases=Counter();confusion={c:Counter() for c in CLASSES}
        with Path(sources['raw_A']['path']).open(encoding='utf-8-sig',newline='') as af, \
             Path(sources['fused_pulses']['path']).open(newline='') as ff, \
             (directory/'pulse_class_predictions.csv').open(newline='') as cf:
            a,f,c=csv.reader(af),csv.reader(ff),csv.reader(cf);ah,fh,ch=next(a),next(f),next(c)
            ai,fi,ci=([h.index(k) for k in PULSE_KEYS] for h in [ah,fh,ch])
            fs,fo,fg=[fh.index(k) for k in ['sequence_id','object_id','local_cluster_id']]
            cs,co,cg,cl,cc,cb,ce=[ch.index(k) for k in ['sequence_id','object_id','local_cluster_id','target_class','class_evidence_score','class_basis','class_low_evidence']]
            ti=ah.index('target_class') if evaluate_train else None
            for ar,fr,cr in itertools.zip_longest(a,f,c):
                if ar is None or fr is None or cr is None:raise ValueError('Pulse class count mismatch')
                if any(ar[i]!=fr[j] or ar[i]!=cr[k] for i,j,k in zip(ai,fi,ci)):raise ValueError('Pulse class keys/order mismatch')
                if cr[cs]!=fr[fs] or cr[co]!=fr[fo] or cr[cg]!=fr[fg]:raise ValueError('Pulse memberships modified')
                label=cr[cl];score=float(cr[cc])
                if label not in CLASSES or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid pulse class/score')
                if fr[fs]:
                    key=tuple(fr[i] for i in fi[:3])+(fr[fs],);seq=sequence_predictions[key]
                    if label!=seq['target_class'] or cr[cb]!=seq['class_basis'] or abs(score-float(seq['class_evidence_score']))>1e-8:
                        raise ValueError('Pulse/sequence class disagreement')
                elif not cr[cb].startswith('unknown_PRI_'):raise ValueError('Unknown PRI classification basis missing')
                labels[label]+=1;bases[cr[cb]]+=1;low+=cr[ce]=='True';n+=1
                if evaluate_train:
                    truth=ar[ti]
                    if truth not in CLASSES:raise ValueError('Training class truth missing')
                    confusion[truth][label]+=1;correct+=truth==label
                if n%3_000_000==0:print(json.dumps({'verified_pulse_classes':n}),flush=True)
        pulse=report['pulse_export']
        if n!=pulse['input_output_pulses'] or dict(labels)!=pulse['class_counts'] or dict(bases)!=pulse['class_basis_counts'] or low!=pulse['low_evidence_pulses']:
            raise ValueError('Class pulse summary mismatch')
        result.update(verified_pulses=n,all_raw_fused_class_keys_and_order_equal=True,all_pulse_memberships_unchanged=True,
                      class_counts=dict(labels),low_evidence_pulses=low)
        if evaluate_train:
            result['training_resubstitution_evaluation']={'pulse_accuracy':correct/n,'all_input_pulses':n,
                'confusion_matrix_pulses':{c:dict(v) for c,v in confusion.items()},
                'interpretation':'Model refitted on all training labels; descriptive in-sample evaluation, not held-out accuracy.'}
    (directory/'classification_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training_resubstitution_evaluation']},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--evaluate-train',action='store_true')
    a=p.parse_args();verify(a.directory,a.evaluate_train)


if __name__=='__main__':main()
