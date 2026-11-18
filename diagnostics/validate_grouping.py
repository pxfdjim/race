#!/usr/bin/env python3
"""Choose association gates on early TRAIN data; evaluate on a disjoint later window."""
import argparse
import csv
import json
import sys
from collections import Counter,defaultdict
from dataclasses import asdict,replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from group_sequences import read_sequences,run_grouping
from prepare_group_inputs import ObserverPaths
from src.local_grouping import GroupConfig,associate
from evaluate_local_groups import metrics,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences',type=Path,required=True)
    p.add_argument('--positions',type=Path,required=True)
    p.add_argument('--sequence-truth',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    seqs=read_sequences(a.sequences,ObserverPaths(a.positions),0,150)
    truth=defaultdict(Counter)
    with a.sequence_truth.open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            truth[key][r['target_id']]+=int(r['pulse_count'])
    base=GroupConfig()
    variants=[('base',base),('lower_ambiguity',replace(base,ambiguity_margin=.05)),
        ('long_gap',replace(base,ambiguity_margin=.05,maximum_gap_seconds=120.)),
        ('tight_bearing',replace(base,ambiguity_margin=.05,maximum_gap_seconds=120.,bearing_base_gate_degrees=.5)),
        ('wide_bearing',replace(base,ambiguity_margin=.05,maximum_gap_seconds=120.,bearing_base_gate_degrees=1.2)),
        ('pri_ratio',replace(base,ambiguity_margin=.05,maximum_gap_seconds=120.,maximum_pri_ratio=2.2))]
    results=[]
    for name,cfg in variants:
        groups,links=associate(seqs,cfg)
        cells=Counter()
        total=sum(s.count for s in seqs)
        for s in seqs:
            group=links[s.key]['local_cluster_id']
            if group:
                for target,n in truth[s.key].items():
                    cells[s.scope,group,target]+=n
        m=metrics(cells)
        m['assignment_coverage']=m['assigned_pulses_evaluated']/total
        score=m['bcubed_f1']*m['assignment_coverage']
        results.append({'name':name,'config':asdict(cfg),'metrics':m,'selection_score':score})
        print(json.dumps({'name':name,'groups':len(groups),'precision':m['bcubed_precision'],
            'recall':m['bcubed_recall'],'f1':m['bcubed_f1'],'coverage':m['assignment_coverage']}),flush=True)
    eligible=[r for r in results if r['metrics']['bcubed_precision']>=.99 and r['metrics']['assignment_coverage']>=.97]
    winner=max(eligible or results,key=lambda r:(r['selection_score'],r['metrics']['bcubed_precision']))
    cfg=GroupConfig(**winner['config'])
    run_grouping(a.sequences,a.positions,a.output_dir/'validation',cfg,start=300,end=450)
    heldout=evaluate(a.sequence_truth,a.output_dir/'validation/sequence_group_links.csv')
    report={'development_window_seconds':[0,150],'validation_window_seconds':[300,450],
        'selection_rule':'Among B-cubed precision >=0.99 and assignment coverage >=0.97, maximize B-cubed F1 * coverage; '
            'if none qualifies use the same score across all candidates. Ties prefer precision.',
        'development_runs':results,'selected_variant':winner['name'],'selected_config':asdict(cfg),
        'validation':heldout,'test_labels_used':False,
        'limitation':'Time-window holdout within one scene; not an independent-scene generalization benchmark.'}
    (a.output_dir/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output_dir/'selected_group_config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    print(json.dumps({'selected':winner['name'],'validation_precision':heldout['bcubed_precision'],
        'validation_recall':heldout['bcubed_recall'],'validation_f1':heldout['bcubed_f1'],
        'validation_coverage':heldout['assignment_coverage']}),flush=True)


if __name__=='__main__':
    main()
