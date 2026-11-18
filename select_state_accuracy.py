#!/usr/bin/env python3
"""Select the already declared state trials by the published accuracy metric."""
import argparse
import json
import os
from pathlib import Path
from src.state_model import StateModelConfig
from src.state_io import read_windows,read_truth
from train_calibrated_state import fit,evaluate
from prepare_inputs import digest

ROOT=Path(__file__).resolve().parent


def run(features,trial_dir,output):
    features,trial_dir,output=map(lambda p:Path(p).resolve(),(features,trial_dir,output))
    if output.exists():raise FileExistsError(output)
    report=json.loads((trial_dir/'training_report.json').read_text())
    selected=max(range(len(report['trials'])),key=lambda i:report['trials'][i]['selection_evaluation']['with_temporal_context']['pulse_accuracy'])
    cfg=StateModelConfig(**report['trials'][selected]['config'])
    directory=features/'train';meta=json.loads((directory/'state_features_run.json').read_text())
    windows=read_windows(directory/'window_features.csv',meta['features']);truth,targets=read_truth(directory/'window_truth.csv',windows)
    held=set(report['held_out_target_instances'])
    pure=[k for k in windows if len(targets[k])==1];nonheld=[k for k in pure if next(iter(targets[k])) not in held]
    clean=lambda k:max(truth[k])/sum(truth[k])>=.95
    development=[k for k in nonheld if windows[k].last<300_000_000_000 and clean(k)]
    selection=[k for k in nonheld if windows[k].first>=320_000_000_000 and windows[k].last<400_000_000_000]
    holdout=[k for k in nonheld if windows[k].first>=420_000_000_000 and windows[k].last<=500_000_000_000]
    unseen=[k for k in pure if next(iter(targets[k])) in held]
    model=fit(windows,truth,development,meta['features'],cfg)
    check=evaluate(model,windows,truth,selection)
    expected=report['trials'][selected]['selection_evaluation']['with_temporal_context']['pulse_accuracy']
    if abs(check['with_temporal_context']['pulse_accuracy']-expected)>1e-12:raise ValueError('State trial reproducibility mismatch')
    time_evaluation=evaluate(model,windows,truth,holdout);instance_evaluation=evaluate(model,windows,truth,unseen)
    final=fit(windows,truth,[k for k in pure if clean(k)],meta['features'],cfg)
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    path=partial/'target_state_model.json';path.write_text(json.dumps(final.to_dict(),separators=(',',':'))+'\n')
    report.update(selected_candidate=selected,selected_config=report['trials'][selected]['config'],
        selection_criterion='Maximum actual-pulse-weighted 3-state accuracy on 320-400 s non-held instances; macro F1 and minority recall remain reported.',
        time_holdout_evaluation=time_evaluation,unseen_instance_evaluation=instance_evaluation,
        model_file={'name':path.name,'sha256':digest(path),'bytes':path.stat().st_size},
        accuracy_selection_source={'path':str(trial_dir/'training_report.json'),'sha256':digest(trial_dir/'training_report.json')},
        algorithm_source_sha256={n:digest(ROOT/n) for n in ['select_state_accuracy.py','train_calibrated_state.py','src/state_model.py','src/state_io.py']})
    report['limitations'].append('Baseline and feature refinements were iteratively inspected on within-scene checks; this is not an untouched external test.')
    (partial/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({'selected_config':report['selected_config'],'time_holdout':time_evaluation['with_temporal_context'],
        'unseen_instance':instance_evaluation['with_temporal_context']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,default=ROOT/'output/calibrated_sequence_state_features')
    p.add_argument('--trial-dir',type=Path,default=ROOT/'models/target_state_calibrated');p.add_argument('--output-dir',type=Path,default=ROOT/'models/target_state_accuracy')
    a=p.parse_args();run(a.features,a.trial_dir,a.output_dir)


if __name__=='__main__':main()
