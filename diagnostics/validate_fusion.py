#!/usr/bin/env python3
"""Develop A-only fusion gates in early TRAIN, then check later TRAIN observations."""
import argparse
import csv
import json
import sys
from dataclasses import asdict,replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fuse_platforms import load_observations,run_fusion,LINK_FIELDS
from evaluate_fusion import evaluate_sequences
from src.fusion import FusionConfig,build_events,associate_events,assign_single_observations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.pipeline_root;output=a.output_dir
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    paths=[root/'output/pri/train/pri_sequences.csv',root/'output/grouped/train/bearing_observations.csv',
           root/'output/group_inputs/train_platform_positions.csv']
    truth=root/'output/group_validation/train_sequence_truth.csv'
    observations,_=load_observations(*paths,start=0,end=150)
    base=FusionConfig()
    progress=lambda r:print(json.dumps(r),flush=True)
    events,stats=build_events(observations,base,progress)
    print(json.dumps(stats),flush=True)
    variants=[('tight_position',replace(base,object_position_gate_m=400.)),('base',base),
              ('wide_position',replace(base,object_position_gate_m=1100.))]
    results=[]
    for name,cfg in variants:
        objects=associate_events(events,cfg,progress)
        links={o.key:{'object_id':e.object_id,'event_id':e.event_id,
                'basis':'multiplatform_event' if e.object_id else 'unmatched_event_ambiguous','cost':e.cost}
               for e in events for o in e.members}
        assign_single_observations(observations,objects,links,cfg,progress)
        destination=output/f'{name}_sequence_object_links.csv'
        with destination.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(LINK_FIELDS)
            for o in sorted(observations,key=lambda o:o.key):
                l=links[o.key]
                w.writerow([*o.key,o.local_group,l['object_id'],l['event_id'],l['basis'],l['cost'] if l['cost'] is not None else ''])
        m=evaluate_sequences(truth,destination)
        score=m.get('bcubed_f1',0)*m['assignment_coverage']
        results.append({'name':name,'config':asdict(cfg),'metrics':m,'score':score})
        print(json.dumps({'variant':name,'objects':len(objects),'coverage':m['assignment_coverage'],
            'precision':m.get('bcubed_precision'),'recall':m.get('bcubed_recall'),'f1':m.get('bcubed_f1'),
            'event_pair_precision':m['cross_platform_pair_precision']}),flush=True)
    eligible=[r for r in results if r['metrics'].get('bcubed_precision',0)>=.98 and r['metrics']['assignment_coverage']>=.8]
    chosen=max(eligible or results,key=lambda r:(r['score'],r['metrics'].get('bcubed_precision',0),-r['config']['object_position_gate_m']))
    cfg=FusionConfig(**chosen['config'])
    run_fusion(*paths,output/'validation',cfg,start=300,end=450)
    validation=evaluate_sequences(truth,output/'validation/sequence_object_links.csv')
    report={'development_seconds':[0,150],'validation_seconds':[300,450],
        'selection_rule':'Among development B-cubed precision >=0.98 and sequence pulse coverage >=0.8, '
            'maximize B-cubed F1 times coverage. If none qualifies, use the same score across all. '
            'Ties prefer precision then the smaller position gate.',
        'development_candidates':results,'development_criteria_met':bool(eligible),
        'selected_variant':chosen['name'],'selected_config':asdict(cfg),
        'validation':validation,'B_used':False,'test_labels_used':False,
        'limitations':'Time-window check within one training scene; upstream label-free observations and local groups '
            'are reused from the full-scene offline pipeline. Not an independent-scene or strict end-to-end holdout.'}
    (output/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'selected_fusion_config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    print(json.dumps({'selected':chosen['name'],'validation':{k:v for k,v in validation.items() if k!='groups_per_true_target'}},indent=2),flush=True)


if __name__=='__main__':main()
