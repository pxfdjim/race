#!/usr/bin/env python3
"""Predict R01-R08 for waveform sequences, fused objects, and every A pulse."""
import argparse
import csv
import itertools
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from prepare_inputs import DEFAULT_FILES,digest
from src.target_classification import CLASSES,WaveformClassModel,combine,pool_objects,top_class
from src.target_classification_io import SEQ_KEYS,PULSE_KEYS,read_rows,read_sequences,read_links

ROOT=Path(__file__).resolve().parent
PULSE_FIELDS=PULSE_KEYS+['sequence_id','local_cluster_id','object_id','target_class','class_evidence_score','class_basis','class_low_evidence']


def write_rows(path,fields,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def export_pulses(raw_source,fused_source,destination,sequences,links,predictions,objects,model):
    labels=Counter();bases=Counter();object_counts=Counter();rows=low=0
    with Path(raw_source).open(encoding='utf-8-sig',newline='') as af,Path(fused_source).open(newline='') as ff,Path(destination).open('w',newline='') as out:
        ar,fr=csv.reader(af),csv.reader(ff);ah,fh=next(ar),next(fr)
        ai,fi=[ah.index(n) for n in PULSE_KEYS],[fh.index(n) for n in PULSE_KEYS]
        si,gi,oi=[fh.index(n) for n in ['sequence_id','local_cluster_id','object_id']]
        ri,wi=ah.index('param1'),ah.index('param2')
        writer=csv.writer(out);writer.writerow(PULSE_FIELDS)
        for a,f in itertools.zip_longest(ar,fr):
            if a is None or f is None or any(a[i]!=f[j] for i,j in zip(ai,fi)):raise ValueError('Raw/fused pulse keys, counts or order differ')
            key=tuple(f[i] for i in fi);object_key=key[:2]+(f[oi],)
            if f[si]:
                skey=key[:3]+(f[si],)
                if skey not in predictions or links[skey]!=object_key:raise ValueError('Pulse/sequence/object class source mismatch')
                scores,basis=predictions[skey]
            else:
                # Deliberately no labels, DOA, platform ID or B fields as features.
                local=model.predict(float(a[ri]),float(a[wi]))
                scores,basis=combine(local,objects.get(object_key),model.config)
                basis='unknown_PRI_'+basis
            label,score,_=top_class(scores)
            evidence_low=score<model.config.low_score_threshold
            writer.writerow([*key,f[si],f[gi],f[oi],label,f'{score:.8f}',basis,evidence_low])
            labels[label]+=1;bases[basis]+=1;low+=evidence_low
            if f[oi]:object_counts[object_key]+=1
            rows+=1
            if rows%2_000_000==0:print(json.dumps({'exported_pulses':rows,'low_evidence':low}),flush=True)
    return {'input_output_pulses':rows,'class_counts':dict(labels),'class_basis_counts':dict(bases),
            'low_evidence_pulses':low,'low_evidence_fraction':low/rows if rows else 0.},object_counts


def run(root,split,output,model_file,sequences_only=False):
    root,output,model_file=map(lambda p:Path(p).resolve(),(root,output,model_file));started=time.monotonic()
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    model=WaveformClassModel.from_dict(json.loads(model_file.read_text()));cfg=model.config
    seq_file=root/f'output/pri/{split}/pri_sequences.csv'
    fusion_dir=root/f'output/fused_b_assisted_guarded/{split}';link_file=fusion_dir/'sequence_object_links.csv'
    meta=json.loads((fusion_dir/'fusion_run.json').read_text());pri=json.loads((root/f'output/pri/{split}/pri_run.json').read_text())
    if digest(seq_file)!=meta['sequence_source_sha256'] or digest(seq_file)!=pri['files']['pri_sequences.csv']['sha256']:
        raise ValueError('Sequence source hash mismatch')
    if digest(link_file)!=meta['files']['sequence_object_links.csv']['sha256']:raise ValueError('Fusion links changed')
    sequences=read_sequences(seq_file);links=read_links(link_file,sequences)
    raw={k:s.predict(model) for k,s in sequences.items()}
    objects=pool_objects(((links[k],s.time,raw[k],s.count) for k,s in sequences.items()),cfg)
    predictions={k:combine(scores,objects.get(links[k]),cfg) for k,scores in raw.items()}
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    object_fields=['scenario_id','data_version','object_id','target_class','class_evidence_score','class_margin',
                   'class_time_bin_count','class_context_reliable']+[f'evidence_{c}' for c in CLASSES]
    write_rows(partial/'object_class_predictions.csv',object_fields,(
        dict(zip(['scenario_id','data_version','object_id'],k),target_class=v['target_class'],class_evidence_score=v['score'],
             class_margin=v['margin'],class_time_bin_count=v['time_bin_count'],class_context_reliable=v['reliable'])|
             dict(zip([f'evidence_{c}' for c in CLASSES],v['scores'])) for k,v in sorted(objects.items())))
    fields=SEQ_KEYS+['object_id','time_ns','pulse_count','waveform_target_class','waveform_evidence_score','target_class',
                     'class_evidence_score','class_margin','class_basis','class_low_evidence']+[f'evidence_{c}' for c in CLASSES]
    seq_rows=[]
    for k,s in sorted(sequences.items()):
        local,local_score,_=top_class(raw[k]);scores,basis=predictions[k];label,score,margin=top_class(scores)
        seq_rows.append(dict(zip(SEQ_KEYS,k),object_id=links[k][2],time_ns=s.time,pulse_count=s.count,
            waveform_target_class=local,waveform_evidence_score=local_score,target_class=label,class_evidence_score=score,
            class_margin=margin,class_basis=basis,class_low_evidence=score<cfg.low_score_threshold)|
            dict(zip([f'evidence_{c}' for c in CLASSES],scores)))
    write_rows(partial/'sequence_class_predictions.csv',fields,seq_rows)
    pulse_export=None;input_paths={'sequences':seq_file,'fusion_links':link_file,'model':model_file}
    if not sequences_only:
        raw_file=root.parent/DEFAULT_FILES[split][0];fused_file=fusion_dir/'pulse_object_predictions.csv'
        if digest(raw_file)!=pri['source_sha256'] or digest(fused_file)!=meta['files']['pulse_object_predictions.csv']['sha256']:
            raise ValueError('Raw/fused pulse source changed')
        pulse_export,_=export_pulses(raw_file,fused_file,partial/'pulse_class_predictions.csv',sequences,links,predictions,objects,model)
        if pulse_export['input_output_pulses']!=pri['input_pulses']:raise ValueError('Class output dropped input pulses')
        input_paths|={'raw_A':raw_file,'fused_pulses':fused_file}
    report={'split':split,'classes':list(CLASSES),'model_config':model.to_dict()['config'],'sequence_count':len(sequences),
        'object_count':len(objects),'sequence_class_counts':dict(Counter(r['target_class'] for r in seq_rows)),
        'object_class_counts':dict(Counter(v['target_class'] for v in objects.values())),
        'reliable_object_context_count':sum(v['reliable'] for v in objects.values()),
        'pulse_export':pulse_export,'features':model.to_dict()['features'],
        'A_truth_columns_read':[],'B_columns_read':[],'object_ids_used_only_to_pool_waveform_evidence':True,
        'upstream_association_B_assisted':meta['B_used'],'state_predicted':False,'complete_submission_generated':False,
        'input_files':{n:{'path':str(p),'sha256':digest(p)} for n,p in input_paths.items()},
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['classify_targets.py','src/target_classification.py','src/target_classification_io.py']},
        'limitations':['Scores are uncalibrated evidence, not guaranteed correctness.',
            'Single-CPI overlapping waveforms can be intrinsically ambiguous.',
            'Object pooling inherits upstream membership errors and B-assisted identity construction.',
            'Unknown PRI without a reliable object uses RF/PW only; all legal top-1 labels are retained with low-evidence flags.',
            'target_state and official submission export remain separate pending tasks.'],
        'elapsed_seconds':round(time.monotonic()-started,2)}
    report['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(partial.iterdir())}
    (partial/'classification_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({'split':split,'object_classes':{k[2]:v['target_class'] for k,v in objects.items()},'pulse_export':pulse_export,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--model',type=Path,default=ROOT/'models/target_class/target_class_model.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/classified')
    p.add_argument('--split',choices=['train','test','both'],default='both')
    p.add_argument('--sequences-only',action='store_true')
    a=p.parse_args()
    for split in (['train','test'] if a.split=='both' else [a.split]):run(a.pipeline_root,split,a.output_dir/split,a.model,a.sequences_only)


if __name__=='__main__':main()
