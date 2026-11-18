#!/usr/bin/env python3
"""Train waveform class densities with separated target/time evaluation."""
import argparse
import json
import os
import time
from collections import Counter,defaultdict
from dataclasses import asdict,replace
from pathlib import Path

from prepare_inputs import digest
from src.target_classification import CLASSES,ClassConfig,WaveformClassModel,combine,pool_objects,top_class
from src.target_classification_io import read_sequences,read_links,read_truth

ROOT=Path(__file__).resolve().parent


def fit(model,sequences,truth,keys):
    return model.fit((s.rf,s.pw,s.pri,s.model,truth[k].most_common(1)[0][0],s.count) for k in keys for s in [sequences[k]])


def metrics(predictions,truth):
    confusion={c:Counter() for c in CLASSES};n=correct=high_total=high_correct=0
    for k,(scores,_) in predictions.items():
        label,score,_=top_class(scores)
        count=sum(truth[k].values());n+=count;correct+=truth[k][label]
        for c,w in truth[k].items():confusion[c][label]+=w
        if score>=.7:high_total+=count;high_correct+=truth[k][label]
    per_class={}
    for c in CLASSES:
        support=sum(confusion[c].values());predicted=sum(row[c] for row in confusion.values());tp=confusion[c][c]
        precision=tp/predicted if predicted else 0.;recall=tp/support if support else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class[c]={'support_pulses':support,'precision':precision,'recall':recall,'f1':f1}
    present=[v for v in per_class.values() if v['support_pulses']]
    return {'sequence_count':len(predictions),'sequence_supported_pulses':n,
        'pulse_weighted_accuracy':correct/n if n else None,
        'macro_f1_present_classes':sum(v['f1'] for v in present)/len(present) if present else None,
        'high_evidence_pulse_coverage_at_0_7':high_total/n if n else 0.,
        'high_evidence_conditional_accuracy':high_correct/high_total if high_total else None,
        'per_class':per_class,'confusion_matrix_pulses':{c:dict(v) for c,v in confusion.items()}}


def evaluate(model,sequences,truth,keys,links):
    raw={k:(sequences[k].predict(model),'waveform_only') for k in keys}
    # Only evaluation-slice evidence is pooled, never training or future-slice labels.
    pooled=pool_objects(((links[k],sequences[k].time,raw[k][0],sequences[k].count) for k in keys),model.config)
    combined={k:combine(scores,pooled.get(links[k]),model.config) for k,(scores,_) in raw.items()}
    return {'waveform_only':metrics(raw,truth),'with_object_context':metrics(combined,truth),
            'object_context_source':'Only unlabeled waveform evidence from this evaluation slice; frozen B-assisted memberships.',
            'pooled_objects':len(pooled)}


def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(f'Choose a new model directory: {output}')
    started=time.monotonic()
    paths={'sequences':root/'output/pri/train/pri_sequences.csv',
           'truth':root/'output/group_validation/train_sequence_truth.csv',
           'links':root/'output/fused_b_assisted_guarded/train/sequence_object_links.csv'}
    pri=json.loads((root/'output/pri/train/pri_run.json').read_text())
    fusion=json.loads((root/'output/fused_b_assisted_guarded/train/fusion_run.json').read_text())
    if digest(paths['sequences'])!=pri['files']['pri_sequences.csv']['sha256'] or digest(paths['links'])!=fusion['files']['sequence_object_links.csv']['sha256']:
        raise ValueError('Training sequence/fusion artifacts changed')
    sequences=read_sequences(paths['sequences']);truth,targets,mapping=read_truth(paths['truth'],sequences)
    links=read_links(paths['links'],sequences)
    by_class=defaultdict(list)
    for tid,c in mapping.items():by_class[c].append(tid)
    held={sorted(ids)[-1] for c,ids in by_class.items() if len(ids)>1}
    pure=[k for k in sequences if len(targets[k])==1]
    nonheld=[k for k in pure if next(iter(targets[k])) not in held]
    development=[k for k in nonheld if sequences[k].end<300_000_000_000]
    selection=[k for k in nonheld if sequences[k].start>=320_000_000_000 and sequences[k].end<400_000_000_000]
    holdout=[k for k in nonheld if sequences[k].start>=420_000_000_000 and sequences[k].end<=500_000_000_000]
    unseen=[k for k in pure if next(iter(targets[k])) in held]
    for name,ks in [('development',development),('selection',selection),('time_holdout',holdout),('unseen_instances',unseen)]:
        if not ks:raise ValueError(f'Empty validation split: {name}')
    base=ClassConfig();configs=[base,replace(base,pw_bin_us=.1),replace(base,rf_bin_mhz=10.),replace(base,log_pri_bin_width=.04)]
    trials=[];best=None
    for index,cfg in enumerate(configs):
        model=fit(WaveformClassModel(cfg),sequences,truth,development)
        evaluation=evaluate(model,sequences,truth,selection,links)
        score=evaluation['waveform_only']['macro_f1_present_classes']
        trials.append({'candidate':index,'config':asdict(cfg),'selection_evaluation':evaluation})
        print(json.dumps({'candidate':index,'validation_macro_f1':score,'validation_with_context_accuracy':evaluation['with_object_context']['pulse_weighted_accuracy']}),flush=True)
        if best is None or score>best[0]:best=score,index,model
    _,index,selected=best
    time_eval=evaluate(selected,sequences,truth,holdout,links)
    instance_eval=evaluate(selected,sequences,truth,unseen,links)
    final=fit(WaveformClassModel(selected.config),sequences,truth,pure)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    model_path=partial/'target_class_model.json'
    model_path.write_text(json.dumps(final.to_dict(),sort_keys=True,separators=(',',':'))+'\n')
    counts=Counter(c for k in pure for c in [truth[k].most_common(1)[0][0]])
    report={'class_definitions':'R01-R08 detector model labels, not target IDs or behavior states',
        'features':list(final.to_dict()['features']),'excluded_features':['target_id','platform_id','object_id','B_track_id','DOA','position','PA','target_state'],
        'label_source':'Separate verified training sequence truth cache; only target_class supervises density fitting.',
        'input_files':{n:{'path':str(p),'sha256':digest(p)} for n,p in paths.items()},
        'training_A_source_sha256':pri['source_sha256'],
        'all_training_sequences':len(sequences),'fit_pure_sequences':len(pure),'excluded_mixed_source_sequences':len(sequences)-len(pure),
        'fit_sequence_class_counts':dict(counts),'held_out_target_instances':{t:mapping[t] for t in sorted(held)},
        'split_sequence_counts':{n:len(ks) for n,ks in [('development',development),('selection',selection),('time_holdout',holdout),('unseen_instances',unseen)]},
        'development_seconds':[0,300],'selection_seconds':[320,400],'time_holdout_seconds':[420,500],
        'held_instances_excluded_from_development_selection_and_time_holdout':True,
        'selection_criterion':'Maximum waveform-only pulse-weighted macro F1 on 320-400 seconds, non-held instances',
        'trials':trials,'selected_candidate':index,'selected_config':asdict(selected.config),
        'time_holdout_evaluation':time_eval,'unseen_instance_evaluation':instance_eval,
        'final_model_refit':'All pure training sequences after fixed validation; no test labels read.',
        'model_file':{'name':model_path.name,'sha256':digest(model_path),'bytes':model_path.stat().st_size},
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['train_target_class.py','src/target_classification.py','src/target_classification_io.py']},
        'limitations':['Time holdout is within one scene and mostly known instances.',
            'Unseen-instance check covers only the three classes with multiple labeled instances, not all eight classes.',
            'PRI summaries and frozen memberships were computed offline from the whole scene; this is not a causal end-to-end holdout.',
            'Object pooling uses B-assisted memberships; waveform-only results are reported separately.',
            'Scores are uncalibrated evidence weights; ambiguous CPIs may be intrinsically indistinguishable.'],
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    partial.rename(output)
    print(json.dumps({'selected_candidate':index,'time_holdout_raw_accuracy':time_eval['waveform_only']['pulse_weighted_accuracy'],
        'time_holdout_context_accuracy':time_eval['with_object_context']['pulse_weighted_accuracy'],
        'unseen_instance_context_accuracy':instance_eval['with_object_context']['pulse_weighted_accuracy'],'model':str(output/model_path.name)},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--output-dir',type=Path,default=ROOT/'models/target_class')
    a=p.parse_args();run(a.pipeline_root,a.output_dir)


if __name__=='__main__':main()
