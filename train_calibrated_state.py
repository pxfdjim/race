#!/usr/bin/env python3
"""Train target_state from observed illumination, with target/time holdouts."""
import argparse
import json
import os
import time
from dataclasses import asdict,replace
from pathlib import Path

from prepare_inputs import digest
from src.state_model import StateForest,StateModelConfig,learn_transitions,smooth_states
from src.state_io import read_windows,read_truth,state_metrics

ROOT=Path(__file__).resolve().parent


def fit(windows,truth,keys,names,cfg):
    x=[windows[k].features for k in keys];y=[max(range(3),key=lambda i:truth[k][i]) for k in keys]
    weights=[min(windows[k].count,256) for k in keys]
    model=StateForest(cfg,names).fit(x,y,weights);model.transitions=learn_transitions(keys,y,weights)
    return model


def evaluate(model,windows,truth,keys,raw=None):
    raw=raw if raw is not None else [model.predict(windows[k].features) for k in keys]
    smoothed=smooth_states(keys,raw,model.transitions,model.config.temporal_strength)
    return {'local_forest':state_metrics(keys,raw,truth),'with_temporal_context':state_metrics(keys,smoothed,truth)}


def run(root,feature_root,output):
    root,feature_root,output=map(lambda p:Path(p).resolve(),(root,feature_root,output));started=time.monotonic()
    if output.exists():raise FileExistsError(f'Choose a new state model directory: {output}')
    directory=feature_root/'train';meta=json.loads((directory/'state_features_run.json').read_text())
    for n in ['window_features.csv','window_truth.csv']:
        if digest(directory/n)!=meta['files'][n]['sha256']:raise ValueError('State feature/truth cache changed')
    if meta['labels_used_in_features']:raise ValueError('Labels leaked into state features')
    names=meta['features'];windows=read_windows(directory/'window_features.csv',names);truth,targets=read_truth(directory/'window_truth.csv',windows)
    mapping=meta['training_target_class_mapping'];by_class={}
    for tid,c in mapping.items():by_class.setdefault(c,[]).append(tid)
    held={sorted(v)[-1] for v in by_class.values() if len(v)>1}
    pure_source=[k for k in windows if len(targets[k])==1]
    nonheld=[k for k in pure_source if next(iter(targets[k])) not in held]
    is_clean=lambda k:max(truth[k])/sum(truth[k])>=.95
    development=[k for k in nonheld if windows[k].last<300_000_000_000 and is_clean(k)]
    selection=[k for k in nonheld if windows[k].first>=320_000_000_000 and windows[k].last<400_000_000_000]
    holdout=[k for k in nonheld if windows[k].first>=420_000_000_000 and windows[k].last<=500_000_000_000]
    unseen=[k for k in pure_source if next(iter(targets[k])) in held]
    if any(not ks for ks in [development,selection,holdout,unseen]):raise ValueError('Empty state validation split')
    trials=[];best=None;base=StateModelConfig()
    for depth in [8,12]:
        fitted=fit(windows,truth,development,names,replace(base,maximum_depth=depth))
        for prior in [0.,.5]:
            fitted.config=replace(fitted.config,prior_power=prior)
            raw=[fitted.predict(windows[k].features) for k in selection]
            for temporal in [0.,.2,.5]:
                fitted.config=replace(fitted.config,temporal_strength=temporal)
                evaluation=evaluate(fitted,windows,truth,selection,raw)
                score=evaluation['with_temporal_context']['macro_f1_present_states']
                candidate={'config':asdict(fitted.config),'selection_evaluation':evaluation};trials.append(candidate)
                print(json.dumps({'depth':depth,'prior':prior,'temporal':temporal,'validation_macro_f1':score,
                    'validation_accuracy':evaluation['with_temporal_context']['pulse_accuracy']}),flush=True)
                if best is None or score>best[0]:best=score,len(trials)-1,fitted.to_dict()
    _,selected_index,data=best;selected=StateForest.from_dict(data)
    time_evaluation=evaluate(selected,windows,truth,holdout)
    instance_evaluation=evaluate(selected,windows,truth,unseen)
    final_keys=[k for k in pure_source if is_clean(k)]
    final=fit(windows,truth,final_keys,names,selected.config)
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    model_path=partial/'target_state_model.json';model_path.write_text(json.dumps(final.to_dict(),separators=(',',':'))+'\n')
    report={'states':{'0':'search','1':'coarse_tracking','2':'fine_tracking'},'feature_config':meta['config'],'features':names,
        'excluded_predictor_inputs':['truth target_id','truth target_state','target_class','B track_id','numeric source_id','absolute time','absolute DOA','observer/target position'],
        'input_files':{n:{'path':str(directory/n),'sha256':digest(directory/n)} for n in ['window_features.csv','window_truth.csv','state_features_run.json']},
        'held_out_target_instances':{t:mapping[t] for t in sorted(held)},'held_instances_excluded_from_fit_and_parameter_selection':True,
        'development_seconds':[0,300],'selection_seconds':[320,400],'time_holdout_seconds':[420,500],
        'split_window_counts':{n:len(ks) for n,ks in [('development',development),('selection',selection),('time_holdout',holdout),('unseen_instances',unseen)]},
        'all_training_windows':len(windows),'fit_windows':len(final_keys),'excluded_mixed_target_windows':len(windows)-len(pure_source),
        'excluded_state_mixture_windows_for_fit':len(pure_source)-len(final_keys),
        'selection_criterion':'Maximum 3-state macro F1 on non-held-instance 320-400 s windows, weighted by their actual pulses',
        'trials':trials,'selected_candidate':selected_index,'selected_config':asdict(selected.config),
        'time_holdout_evaluation':time_evaluation,'unseen_instance_evaluation':instance_evaluation,
        'final_refit':'All clean single-target training windows after fixed selection; no test labels used.',
        'model_file':{'name':model_path.name,'sha256':digest(model_path),'bytes':model_path.stat().st_size},
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['train_calibrated_state.py','src/state_model.py','src/state_io.py']},
        'limitations':['State 1/2 may be observationally ambiguous when illumination patterns coincide.',
            'Feature windows are offline and centered up to 3 seconds; this is not a causal streaming predictor.',
            'Frozen source groupings inherit upstream errors and B-assisted identity construction.',
            'Time holdout remains one scene; unseen-instance evaluation covers three classes only.',
            'Evidence scores are not calibrated probabilities; no state is synthesized for unobserved time.'],
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({'selected_config':report['selected_config'],'time_holdout':time_evaluation['with_temporal_context'],
        'unseen_instance':instance_evaluation['with_temporal_context'],'model':str(output/model_path.name),'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--feature-root',type=Path,default=ROOT/'output/calibrated_sequence_state_features')
    p.add_argument('--output-dir',type=Path,default=ROOT/'models/target_state_calibrated')
    a=p.parse_args();run(a.pipeline_root,a.feature_root,a.output_dir)


if __name__=='__main__':main()
