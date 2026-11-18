#!/usr/bin/env python3
"""Independent identity evaluation across platforms, using TRAIN labels only here."""
import argparse
import csv
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
from evaluate_local_groups import metrics


def evaluate_sequences(truth_file,links_file):
    links={}
    with Path(links_file).open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            if key in links:raise ValueError('Duplicate fusion assignment')
            links[key]=r
    cells=Counter();events=defaultdict(lambda:defaultdict(Counter));seen=set();total=0
    for r in csv.DictReader(Path(truth_file).open()):
        key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
        if key not in links:continue
        n=int(r['pulse_count']);total+=n;seen.add(key);p=links[key]
        if p['object_id']:cells[key[:2],p['object_id'],r['target_id']]+=n
        if p['event_id']:events[key[:2]+(p['event_id'],)][key[2]][r['target_id']]+=n
    if seen!=links.keys():raise ValueError('Some sequences lack training truth')
    report=metrics(cells) if cells else {'assigned_pulses_evaluated':0}
    correct=pairs=0;mixed=0
    for platforms in events.values():
        ids=[v.most_common(1)[0][0] for v in platforms.values()]
        mixed+=len(set(ids))>1
        for a,b in itertools.combinations(ids,2):pairs+=1;correct+=a==b
    report.update(selected_sequence_pulses=total,assignment_coverage=sum(cells.values())/total if total else 0.,
        event_count=len(events),mixed_events_by_sequence_majority=mixed,evaluated_platform_pairs=pairs,
        same_target_platform_pairs=correct,cross_platform_pair_precision=correct/pairs if pairs else None,
        evaluation_scope='PRI-supported pulses, global scene/version identity across all platforms. '
                         'B-cubed conditions on assigned pulses; coverage separately. Event pairs use sequence-majority truth.')
    return report


def evaluate_pulses(truth_file,predictions):
    cells=Counter();status=Counter();platform=defaultdict(Counter);total=0
    with Path(truth_file).open(encoding='utf-8-sig',newline='') as af,Path(predictions).open(newline='') as pf:
        a,b=csv.reader(af),csv.reader(pf);ah,bh=next(a),next(b)
        names=('scenario_id','data_version','platform_id','pulse_id')
        ai,bi=[ah.index(k) for k in names],[bh.index(k) for k in names]
        tid,oid,si=ah.index('target_id'),bh.index('object_id'),bh.index('fusion_status')
        for raw,pred in itertools.zip_longest(a,b):
            if raw is None or pred is None or any(raw[i]!=pred[j] for i,j in zip(ai,bi)):raise ValueError('Input/prediction keys, counts or order differ')
            if not raw[tid]:raise ValueError('Labeled training data required')
            scope=tuple(raw[i] for i in ai[:2]);total+=1;status[pred[si]]+=1;platform[raw[ai[2]]]['input']+=1
            if pred[oid]:cells[scope,pred[oid],raw[tid]]+=1;platform[raw[ai[2]]]['assigned']+=1
    report=metrics(cells)
    report.update(input_pulses=total,assigned_pulses=sum(cells.values()),assignment_coverage=sum(cells.values())/total,
        status_counts=dict(status),platform_coverage={k:dict(v)|{'coverage':v['assigned']/v['input']} for k,v in sorted(platform.items())},
        input_output_keys_and_order_exact=True,evaluation_scope='All pulses, global scene/version identity across platforms. '
        'Purity/B-cubed condition on assigned pulses; coverage separately.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-truth',type=Path);p.add_argument('--links',type=Path)
    p.add_argument('--truth-file',type=Path);p.add_argument('--pulse-predictions',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.sequence_truth and a.links and not(a.truth_file or a.pulse_predictions):r=evaluate_sequences(a.sequence_truth,a.links)
    elif a.truth_file and a.pulse_predictions and not(a.sequence_truth or a.links):r=evaluate_pulses(a.truth_file,a.pulse_predictions)
    else:p.error('Provide sequence-truth/links or truth-file/pulse-predictions')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='groups_per_true_target'},indent=2))


if __name__=='__main__':main()
