#!/usr/bin/env python3
"""Evaluate merges and splits with independent TRAIN sequence labels."""
import argparse
import csv
import json
import itertools
from collections import Counter,defaultdict
from pathlib import Path


def metrics(cells):
    by_group,by_target=Counter(),Counter()
    for (scope,group,target),n in cells.items():
        by_group[scope,group]+=n
        by_target[scope,target]+=n
    total=sum(cells.values())
    if not total:
        raise ValueError('No labeled assignments')
    precision=sum(n*n/by_group[scope,group] for (scope,group,target),n in cells.items())/total
    recall=sum(n*n/by_target[scope,target] for (scope,group,target),n in cells.items())/total
    gp=defaultdict(Counter)
    fragments=defaultdict(Counter)
    for (scope,group,target),n in cells.items():
        gp[scope,group][target]+=n
        fragments[scope,target][group]+=n
    pairs=lambda n:n*(n-1)//2
    tp=sum(pairs(n) for n in cells.values())
    pp=sum(pairs(n) for n in by_group.values())
    ap=sum(pairs(n) for n in by_target.values())
    pair_precision=tp/pp if pp else 0.
    pair_recall=tp/ap if ap else 0.
    return {'assigned_pulses_evaluated':total,'predicted_groups':len(by_group),'observed_true_targets':len(by_target),
        'weighted_group_purity':sum(max(v.values()) for v in gp.values())/total,
        'bcubed_precision':precision,'bcubed_recall':recall,'bcubed_f1':2*precision*recall/(precision+recall),
        'pairwise_precision':pair_precision,'pairwise_recall':pair_recall,
        'pairwise_f1':2*pair_precision*pair_recall/(pair_precision+pair_recall) if pair_precision+pair_recall else 0.,
        'mixed_groups':sum(len(v)>1 for v in gp.values()),
        'targets_split_into_multiple_significant_groups':sum(sum(n>=max(32,.01*sum(v.values())) for n in v.values())>1 for v in fragments.values()),
        'groups_per_true_target':[{'scope':list(scope),'target_id':target,'pulses':sum(v.values()),
            'groups':len(v),'significant_groups':sum(n>=max(32,.01*sum(v.values())) for n in v.values()),
            'dominant_group_fraction':max(v.values())/sum(v.values())} for (scope,target),v in sorted(fragments.items())]}


def evaluate(truth_file,links_file):
    links={}
    with Path(links_file).open(newline='') as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            if key in links:
                raise ValueError('Duplicate sequence assignment')
            links[key]=r['local_cluster_id']
    cells=Counter()
    covered=set()
    selected_pulses=0
    with Path(truth_file).open(newline='') as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            if key in links:
                selected_pulses+=int(r['pulse_count'])
                if links[key]:
                    cells[key[:3],links[key],r['target_id']]+=int(r['pulse_count'])
                covered.add(key)
    if covered!=links.keys():
        raise ValueError('Some output sequences have no evaluation labels')
    report=metrics(cells)
    report['selected_sequence_pulses']=selected_pulses
    report['unassigned_sequence_pulses']=selected_pulses-report['assigned_pulses_evaluated']
    report['assignment_coverage']=report['assigned_pulses_evaluated']/selected_pulses
    report['per_platform']={str(scope):metrics(Counter({k:v for k,v in cells.items() if k[0]==scope}))
                            for scope in sorted({k[0] for k in cells})}
    report['evaluation_scope']='PRI-supported pulses only. Unknown-PRI pulse attachment requires separate pulse evaluation.'
    return report


def evaluate_pulses(source,predictions):
    cells,inputs,assigned,classes,assigned_classes=Counter(),Counter(),Counter(),Counter(),Counter()
    targets=set()
    statuses=Counter()
    with Path(source).open(encoding='utf-8-sig',newline='') as af,Path(predictions).open(newline='') as pf:
        a,b=csv.reader(af),csv.reader(pf)
        ah,bh=next(a),next(b)
        names=('scenario_id','data_version','platform_id','pulse_id')
        ai,bi=[ah.index(k) for k in names],[bh.index(k) for k in names]
        tid,klass=ah.index('target_id'),ah.index('target_class')
        group,status=bh.index('local_cluster_id'),bh.index('group_assignment_status')
        for raw,pred in itertools.zip_longest(a,b):
            if raw is None or pred is None or any(raw[i]!=pred[j] for i,j in zip(ai,bi)):
                raise ValueError('Prediction/input keys or row counts differ')
            if not raw[tid] or not raw[klass]:
                raise ValueError('Labeled TRAIN data required')
            scope=tuple(raw[i] for i in ai[:3])
            targets.add(scope+(raw[tid],))
            inputs[scope]+=1
            classes[raw[klass]]+=1
            statuses[pred[status]]+=1
            if pred[group]:
                cells[scope,pred[group],raw[tid]]+=1
                assigned[scope]+=1
                assigned_classes[raw[klass]]+=1
    report=metrics(cells)
    n=sum(inputs.values())
    report.update(input_pulses=n,assigned_pulses=sum(assigned.values()),unassigned_pulses=n-sum(assigned.values()),
        assignment_coverage=sum(assigned.values())/n,observed_true_targets_in_input=len(targets),
        assignment_status_counts=dict(statuses),input_output_keys_and_order_exact=True,
        class_coverage={k:{'input':v,'assigned':assigned_classes[k],'coverage':assigned_classes[k]/v}
                        for k,v in sorted(classes.items())},
        platform_coverage=[dict(zip(names[:3],k),input=v,assigned=assigned[k],coverage=assigned[k]/v)
                           for k,v in sorted(inputs.items())],
        evaluation_scope='All input pulses including unknown-PRI attachment. Purity and B-cubed metrics '
                         'condition on assigned pulses; assignment coverage is reported separately.')
    report['per_platform']={str(scope):metrics(Counter({k:v for k,v in cells.items() if k[0]==scope}))
                            for scope in sorted(inputs) if assigned[scope]}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-truth',type=Path)
    p.add_argument('--links',type=Path)
    p.add_argument('--truth-file',type=Path)
    p.add_argument('--pulse-predictions',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.sequence_truth and a.links and not (a.truth_file or a.pulse_predictions):
        report=evaluate(a.sequence_truth,a.links)
    elif a.truth_file and a.pulse_predictions and not (a.sequence_truth or a.links):
        report=evaluate_pulses(a.truth_file,a.pulse_predictions)
    else:
        p.error('Provide either --sequence-truth/--links or --truth-file/--pulse-predictions')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_platform','groups_per_true_target')},indent=2))


if __name__=='__main__':
    main()
