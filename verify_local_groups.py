#!/usr/bin/env python3
"""Verify saved group membership/counts independently, without ground-truth labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_inputs import digest


def verify(directory):
    directory=Path(directory)
    run=json.loads((directory/'grouping_run.json').read_text())
    for name,info in run['files'].items():
        if digest(directory/name)!=info['sha256']:
            raise ValueError(f'Output hash mismatch: {name}')
    groups={}
    with (directory/'local_groups.csv').open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','local_cluster_id'))
            if key in groups:
                raise ValueError('Duplicate group key')
            groups[key]=r
    links={}
    with (directory/'sequence_group_links.csv').open() as f:
        for r in csv.DictReader(f):
            scope=tuple(r[k] for k in ('scenario_id','data_version','platform_id'))
            key=scope+(r['sequence_id'],)
            if key in links or (r['local_cluster_id'] and scope+(r['local_cluster_id'],) not in groups):
                raise ValueError('Duplicate sequence or missing group')
            links[key]=r['local_cluster_id']
    observed=set()
    with (directory/'bearing_observations.csv').open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ('scenario_id','data_version','platform_id','sequence_id'))
            if key in observed or links[key]!=r['local_cluster_id'] or not int(r['start_ns'])<=int(r['time_ns'])<=int(r['end_ns']):
                raise ValueError('Bearing observation/sequence mismatch')
            observed.add(key)
    if observed!=links.keys() or len(links)!=run['sequence_count']:
        raise ValueError('Incomplete sequence coverage')
    report={'group_keys_unique':True,'sequence_keys_unique':True,'all_sequence_bearings_present':True,
            'saved_file_hashes_match':True,'labels_used':False}
    if run['pulse_export'] is not None:
        counts,statuses,rows=Counter(),Counter(),0
        with (directory/'local_predictions.csv').open() as f:
            reader=csv.reader(f)
            h=next(reader)
            si=[h.index(k) for k in ('scenario_id','data_version','platform_id')]
            seq,group,status=[h.index(k) for k in ('sequence_id','local_cluster_id','group_assignment_status')]
            for r in reader:
                scope=tuple(r[i] for i in si)
                if r[seq] and links[scope+(r[seq],)]!=r[group]:
                    raise ValueError('Pulse/sequence group mismatch')
                if r[group]:
                    if scope+(r[group],) not in groups:
                        raise ValueError('Pulse group missing')
                    counts[scope+(r[group],)]+=1
                statuses[r[status]]+=1
                rows+=1
        if rows!=run['pulse_export']['input_output_pulses'] or dict(statuses)!=run['pulse_export']['assignment_status_counts']:
            raise ValueError('Pulse output status/count mismatch')
        if any(counts[k]!=int(v['total_assigned_pulses']) for k,v in groups.items()):
            raise ValueError('Group member count mismatch')
        report.update(verified_pulse_rows=rows,pulse_sequence_membership_exact=True,
                      pulse_group_member_counts_exact=True,assignment_status_counts=dict(statuses))
    (directory/'output_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-dir',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.group_dir),indent=2))


if __name__=='__main__':
    main()
