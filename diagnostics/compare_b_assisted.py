#!/usr/bin/env python3
"""Compare independently evaluated A-only and B-assisted results, never infer IDs."""
import argparse
import csv
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_fusion import evaluate_sequences
from prepare_inputs import digest


def compare_sequences(truth, baseline, assisted):
    names=('scenario_id','data_version','platform_id','sequence_id')
    def read(p):
        return {tuple(r[k] for k in names):r for r in csv.DictReader(Path(p).open())}
    old,new=read(baseline),read(assisted)
    selected=old.keys() & new.keys()
    per_target=defaultdict(Counter);transitions=Counter()
    for r in csv.DictReader(Path(truth).open()):
        key=tuple(r[k] for k in names)
        if key not in selected:continue
        n=int(r['pulse_count']);scope=key[:2]+(r['target_id'],)
        per_target[scope]['input']+=n
        a,b=bool(old[key]['object_id']),bool(new[key]['object_id'])
        per_target[scope]['baseline_assigned']+=n*a;per_target[scope]['assisted_assigned']+=n*b
        transitions[f'{"assigned" if a else "unassigned"}_to_{"assigned" if b else "unassigned"}']+=n
    return {'assignment_transition_sequence_pulses':dict(transitions),
        'per_target_sequence_coverage':[dict(scenario_id=k[0],data_version=k[1],target_id=k[2],**v,
            baseline_coverage=v['baseline_assigned']/v['input'],assisted_coverage=v['assisted_assigned']/v['input'])
            for k,v in sorted(per_target.items())]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,required=True)
    p.add_argument('--assisted-root',type=Path,default=ROOT/'output/fused_b_assisted_guarded')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('proposal','guarded'),default='guarded')
    a=p.parse_args();root=a.pipeline_root;candidate=a.assisted_root
    result={'stage':a.stage,'B_used':True,'A_or_B_target_identity_used_for_inference':False,
            'test_accuracy_evaluated':False,'metrics_condition_on_assigned_pulses':True,'splits':{}}
    truth=root/'output/group_validation/train_sequence_truth.csv'
    for split in ('train','test'):
        old=root/f'output/fused/{split}';new=candidate/split
        before=json.loads((old/'fusion_run.json').read_text());after=json.loads((new/'fusion_run.json').read_text())
        r={'A_only':{'objects':before['object_count'],**before['pulse_export']},
           'B_assisted':{'objects':after['object_count'],**after['pulse_export']}}
        r['coverage_change_percentage_points']=100*(r['B_assisted']['assignment_coverage']-r['A_only']['assignment_coverage'])
        if split=='train':
            r['A_only']['pulse_metrics']=json.loads((old/'pulse_evaluation.json').read_text())
            r['B_assisted']['pulse_metrics']=json.loads((new/'pulse_evaluation.json').read_text())
            r.update(compare_sequences(truth,old/'sequence_object_links.csv',new/'sequence_object_links.csv'))
        result['splits'][split]=r
    development=ROOT/'output/b_assisted_development/train'
    validation=ROOT/'output/b_assisted_validation/train'
    if development.exists() and validation.exists():
        result['proposal_development']={'seconds':[0,150],'metrics':json.loads((development/'sequence_evaluation.json').read_text()),
            'configuration':json.loads((development/'fusion_run.json').read_text())['config']}
        result['proposal_later_training_check']={'seconds':[300,450],
            'metrics':json.loads((validation/'sequence_evaluation.json').read_text())}
    result['validation_limitations']=('Development on early training observations; later-time descriptive check reuses '
        'the full-scene label-free upstream results and A-only baseline. Not an independent-scene or strict end-to-end holdout. '
        'All full-training metrics are descriptive; no test truth is used.')
    result['config_sha256']=digest(ROOT/('config/b_assisted_config.json' if a.stage=='proposal' else 'config/b_guard_config.json'))
    if a.stage=='guarded':
        result['validation_limitations']+=' The conservative guard was developed using full-training diagnostics; '
        result['validation_limitations']+='its full-training comparison is in-sample, not a held-out validation.'
        review=ROOT/'output/b_guard_parameter_review.json'
        if review.exists():result['guard_parameter_review']=json.loads(review.read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k in ('coverage_change_percentage_points',)} for s,r in result['splits'].items()}))


if __name__=='__main__':main()
