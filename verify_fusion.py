#!/usr/bin/env python3
"""Independent membership and artifact checks for saved fusion results; no labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from prepare_inputs import digest


def verify(directory):
    directory=Path(directory);run=json.loads((directory/'fusion_run.json').read_text())
    for name,info in run['files'].items():
        if digest(directory/name)!=info['sha256']:raise ValueError(f'Output hash mismatch: {name}')
    objects={}
    with (directory/'fused_objects.csv').open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','object_id'))
            if key in objects:raise ValueError('Duplicate object')
            objects[key]=r
    links={};sequence_counts=Counter();event_counts=Counter();events={}
    with (directory/'sequence_object_links.csv').open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            if key in links:raise ValueError('Duplicate sequence')
            if r['object_id'] and key[:2]+(r['object_id'],) not in objects:raise ValueError('Object missing')
            links[key]=r
            if r['object_id']:sequence_counts[key[:2]+(r['object_id'],)]+=1
            if r['event_id']:
                event=key[:2]+(r['event_id'],)
                event_counts[event]+=1
                events.setdefault(event,[]).append((key[2],key[3],r['object_id']))
    if len(links)!=run['sequence_count'] or len(objects)!=run['object_count']:raise ValueError('Summary count mismatch')
    if any(sequence_counts[k]!=int(r['sequence_count']) for k,r in objects.items()):raise ValueError('Object sequence count mismatch')
    event_seen=set()
    with (directory/'cross_platform_events.csv').open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','event_id'))
            if key in event_seen:raise ValueError('Duplicate event')
            event_seen.add(key)
            members=events.get(key,[])
            if len(members)!=int(r['platform_count']) or len({v[0] for v in members})!=len(members):raise ValueError('Event platform uniqueness/count mismatch')
            if set(zip(r['platform_ids'].split('|'),r['sequence_ids'].split('|')))!={(v[0],v[1]) for v in members}:raise ValueError('Event membership mismatch')
            if any(v[2]!=r['object_id'] for v in members):raise ValueError('Event/object mismatch')
    if event_seen!=events.keys() or len(events)!=run['event_count']:raise ValueError('Incomplete events')
    report={'output_hashes_match':True,'object_and_sequence_keys_unique':True,'event_platforms_unique':True,
            'sequence_event_object_membership_exact':True,'labels_used':False}
    if run['pulse_export'] is not None:
        counts,statuses=Counter(),Counter();n=0
        with (directory/'pulse_object_predictions.csv').open() as f:
            reader=csv.reader(f);h=next(reader)
            ki=[h.index(k) for k in ('scenario_id','data_version','platform_id')]
            si,oi,ei,st=[h.index(k) for k in ('sequence_id','object_id','event_id','fusion_status')]
            for r in reader:
                scope=tuple(r[i] for i in ki)
                if r[si]:
                    l=links[scope+(r[si],)]
                    if l['object_id']!=r[oi] or l['event_id']!=r[ei]:raise ValueError('Pulse/sequence membership mismatch')
                elif r[ei]:raise ValueError('Unknown PRI pulse has direct event')
                if r[oi]:
                    if scope[:2]+(r[oi],) not in objects:raise ValueError('Pulse object missing')
                    counts[scope[:2]+(r[oi],)]+=1
                statuses[r[st]]+=1;n+=1
        if n!=run['pulse_export']['input_output_pulses'] or dict(statuses)!=run['pulse_export']['status_counts']:raise ValueError('Pulse count/status mismatch')
        if any(counts[k]!=int(r['total_assigned_pulses']) for k,r in objects.items()):raise ValueError('Object pulse count mismatch')
        report.update(verified_pulse_rows=n,pulse_sequence_object_membership_exact=True,
                      object_pulse_counts_exact=True,assigned_pulses=sum(counts.values()),status_counts=dict(statuses))
    (directory/'output_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fusion-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.fusion_dir),indent=2))


if __name__=='__main__':main()
