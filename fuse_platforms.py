#!/usr/bin/env python3
"""Fuse asynchronous A observations across platforms without B or identity labels."""
import argparse
import bisect
import csv
import itertools
import json
import math
import os
import time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path

from prepare_group_inputs import ObserverPaths
from prepare_inputs import DEFAULT_FILES,digest
from group_sequences import read_sequences
from src.dataio import time_to_ns
from src.fusion import FusionConfig,LocalFrame,FusionObservation,fuse_observations

ROOT=Path(__file__).resolve().parent
SEQ_KEYS=['scenario_id','data_version','platform_id','sequence_id']
PULSE_KEYS=SEQ_KEYS[:3]+['pulse_id']
LINK_FIELDS=SEQ_KEYS+['local_cluster_id','object_id','event_id','fusion_basis','association_cost']
PULSE_FIELDS=PULSE_KEYS+['sequence_id','local_cluster_id','object_id','event_id','fusion_status']


def load_observations(sequence_file,bearings_file,positions_file,start=None,end=None):
    positions=ObserverPaths(positions_file)
    sequences=read_sequences(sequence_file,positions,start,end)
    bearings={}
    with Path(bearings_file).open() as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in SEQ_KEYS)
            if key in bearings:raise ValueError('Duplicate bearing key')
            bearings[key]=r
    by_scene=defaultdict(list)
    for s in sequences:by_scene[s.scope[:2]].append(s.position)
    frames={scene:LocalFrame(sum(p[0] for p in ps)/len(ps),sum(p[1] for p in ps)/len(ps)) for scene,ps in by_scene.items()}
    observations=[]
    for s in sequences:
        r=bearings.get(s.key)
        if r is None or int(r['time_ns'])!=s.time or int(r['start_ns'])!=s.start or int(r['end_ns'])!=s.end:
            raise ValueError('PRI and grouped bearing observations differ')
        if abs((float(r['doa_degrees'])-s.doa+180)%360-180)>1e-6:raise ValueError('Bearing mismatch')
        frame=frames[s.scope[:2]]
        observations.append(FusionObservation(s,r['local_cluster_id'],frame.project(s.position),frame.direction(s.position,s.doa)))
    return observations,frames


def export_pulses(raw_source,local_predictions,destination,observations,links,unit='us'):
    # Local groups are used only in a nearby time context, never mapped globally
    # to a single object: one mixed local group may contain several objects.
    support=defaultdict(list)
    for o in observations:
        if o.local_group and links[o.key]['object_id']:
            support[o.sequence.scope+(o.local_group,)].append(o)
    times={}
    for key,items in support.items():
        items.sort(key=lambda o:o.time);times[key]=[o.time for o in items]
    statuses,object_counts=Counter(),Counter()
    rows=0
    with Path(raw_source).open(encoding='utf-8-sig',newline='') as af, \
         Path(local_predictions).open(newline='') as pf,Path(destination).open('w',newline='') as out:
        raw,pred=csv.reader(af),csv.reader(pf)
        ah,ph=next(raw),next(pred)
        ai,pi=[ah.index(k) for k in PULSE_KEYS],[ph.index(k) for k in PULSE_KEYS]
        ti=ah.index('toa');si,gi=ph.index('sequence_id'),ph.index('local_cluster_id')
        writer=csv.writer(out);writer.writerow(PULSE_FIELDS)
        for a,p in itertools.zip_longest(raw,pred):
            if a is None or p is None or any(a[i]!=p[j] for i,j in zip(ai,pi)):
                raise ValueError('Raw/grouped pulse keys, count or order differ')
            key=tuple(p[i] for i in pi);event='';object_id=''
            if p[si]:
                link=links[key[:3]+(p[si],)]
                object_id,event,status=link['object_id'],link['event_id'],link['basis']
            else:
                status='unmatched_unknown_pri'
                group=key[:3]+(p[gi],)
                if p[gi] and group in support:
                    t=time_to_ns(a[ti],unit);items=support[group];ts=times[group]
                    left,right=bisect.bisect_left(ts,t-500_000_000),bisect.bisect_right(ts,t+500_000_000)
                    scores=Counter()
                    for o in items[left:right]:
                        gap=max(o.sequence.start-t,t-o.sequence.end,0)/1e9
                        if gap<=.25:
                            scores[links[o.key]['object_id']]+=min(o.sequence.count,128)/(1+gap*10)
                    ranked=scores.most_common()
                    if ranked and (len(ranked)==1 or ranked[0][1]>=3*ranked[1][1]):
                        object_id=ranked[0][0];status='unknown_pri_local_time_context'
            writer.writerow([*key,p[si],p[gi],object_id,event,status])
            statuses[status]+=1
            if object_id:object_counts[key[:2]+(object_id,)]+=1
            rows+=1
            if rows%3_000_000==0:print(json.dumps({'exported_pulses':rows,'status_counts':dict(statuses)}),flush=True)
    return {'input_output_pulses':rows,'assigned_pulses':sum(object_counts.values()),
            'assignment_coverage':sum(object_counts.values())/rows if rows else 0.,
            'status_counts':dict(statuses)},object_counts


def run_fusion(sequence_file,bearings_file,positions_file,output,config,raw_source=None,
               group_dir=None,pri_dir=None,start=None,end=None,unit='us'):
    config.validate();started=time.monotonic()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    if (start is not None and (not math.isfinite(start) or start<0)) or (end is not None and (not math.isfinite(end) or end<=0)) or (start is not None and end is not None and end<=start):
        raise ValueError('Invalid time filter')
    observations,frames=load_observations(sequence_file,bearings_file,positions_file,start,end)
    objects,events,links,stats=fuse_observations(observations,config,lambda r:print(json.dumps(r),flush=True))
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    with (partial/'sequence_object_links.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(LINK_FIELDS)
        for o in sorted(observations,key=lambda o:o.key):
            l=links[o.key]
            w.writerow([*o.key,o.local_group,l['object_id'],l['event_id'],l['basis'],l['cost'] if l['cost'] is not None else ''])
    with (partial/'cross_platform_events.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scenario_id','data_version','event_id','object_id','time_ns','platform_count',
            'platform_ids','sequence_ids','coarse_east_m','coarse_north_m','geometry_error_proxy_m',
            'crossing_angle_degrees','maximum_bearing_residual_degrees','association_cost'])
        for scene,e in events:
            g=e.geometry
            w.writerow([*scene,e.event_id,e.object_id,e.time,len(e.members),'|'.join(o.platform for o in e.members),
                '|'.join(o.sequence.sequence_id for o in e.members),g['east_m'],g['north_m'],g['error_m'],
                g['crossing_degrees'],g['maximum_bearing_residual_degrees'],e.cost])
    export,counts=None,Counter()
    if raw_source is not None:
        if start is not None or end is not None:raise ValueError('Pulse export cannot use time-filtered sequences')
        group_dir,pri_dir=Path(group_dir),Path(pri_dir)
        pri_meta=json.loads((pri_dir/'pri_run.json').read_text());group_meta=json.loads((group_dir/'grouping_run.json').read_text())
        position_meta=json.loads(Path(positions_file).with_suffix('.json').read_text())
        if unit!=pri_meta['input_config']['a_toa_unit']:raise ValueError('Source TOA unit mismatch')
        if digest(raw_source)!=pri_meta['source_sha256'] or position_meta['source_sha256']!=pri_meta['source_sha256']:
            raise ValueError('Raw/PRI/position source mismatch')
        if digest(sequence_file)!=pri_meta['files']['pri_sequences.csv']['sha256'] or digest(bearings_file)!=group_meta['files']['bearing_observations.csv']['sha256']:
            raise ValueError('Sequence/bearing artifacts changed')
        if digest(group_dir/'local_predictions.csv')!=group_meta['files']['local_predictions.csv']['sha256']:
            raise ValueError('Grouped pulse predictions changed')
        export,counts=export_pulses(raw_source,group_dir/'local_predictions.csv',partial/'pulse_object_predictions.csv',observations,links,unit)
        if export['input_output_pulses']!=pri_meta['input_pulses']:raise ValueError('Pulse output coverage mismatch')
    members=defaultdict(list)
    for o in observations:
        if links[o.key]['object_id']:members[o.sequence.scope[:2]+(links[o.key]['object_id'],)].append(o)
    with (partial/'fused_objects.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scenario_id','data_version','object_id','platform_count','platform_ids',
            'local_group_count','sequence_count','sequence_pulses','total_assigned_pulses','event_count',
            'start_ns','end_ns','object_status'])
        for scene,obj in objects:
            object_members=members[scene+(obj.object_id,)]
            platforms=sorted({o.platform for o in object_members})
            w.writerow([*scene,obj.object_id,len(platforms),'|'.join(platforms),
                len({o.sequence.scope+(o.local_group,) for o in object_members if o.local_group}),len(object_members),
                sum(o.sequence.count for o in object_members),counts.get(scene+(obj.object_id,),sum(o.sequence.count for o in object_members)),len(obj.events),
                min(o.sequence.start for o in object_members),max(o.sequence.end for o in object_members),
                'supported' if len(obj.events)>=3 else 'tentative'])
    local_members=Counter()
    for o in observations:
        oid=links[o.key]['object_id']
        if o.local_group and oid:local_members[o.sequence.scope+(o.local_group,oid)]+=o.sequence.count
    with (partial/'local_object_memberships.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(SEQ_KEYS[:3]+['local_cluster_id','object_id','sequence_pulses'])
        for key,n in sorted(local_members.items()):w.writerow([*key,n])
    report={'config':asdict(config),'sequence_count':len(observations),'sequence_pulses':sum(o.sequence.count for o in observations),
        'assigned_sequence_count':sum(bool(l['object_id']) for l in links.values()),
        'assigned_sequence_pulses':sum(o.sequence.count for o in observations if links[o.key]['object_id']),
        'object_count':len(objects),'event_count':len(events),
        'events_with_object_identity':sum(bool(e.object_id) for _,e in events),
        'ambiguous_events_without_object_identity':sum(not e.object_id for _,e in events),
        'event_platform_counts':dict(Counter(len(e.members) for _,e in events)),
        'scene_event_statistics':stats,'pulse_export':export,'time_filter_seconds':[start,end],
        'frames':[{'scenario_id':k[0],'data_version':k[1],'origin_longitude':v.longitude,'origin_latitude':v.latitude,'datum':'WGS84'} for k,v in sorted(frames.items())],
        'B_used':False,'A_truth_columns_used':[],'TOA_common_clock_assumed':True,
        'TOA_clock_offset_estimated':False,'target_altitude_estimated':False,
        'geometry_role':'Coarse horizontal association check only; error proxy is not a calibrated confidence interval.',
        'sequence_source':str(Path(sequence_file).resolve()),'sequence_source_sha256':digest(sequence_file),
        'bearing_source_sha256':digest(bearings_file),'positions_source_sha256':digest(positions_file),
        'algorithm_source_sha256':{name:digest(ROOT/name) for name in ('fuse_platforms.py','src/fusion.py','src/local_grouping.py','group_sequences.py')},
        'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in partial.iterdir()},
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'fusion_run.json').write_text(json.dumps(report,indent=2)+'\n');partial.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT.parent)
    p.add_argument('--split',choices=('train','test','both'),default='both')
    p.add_argument('--a-file',type=Path)
    p.add_argument('--pri-root',type=Path,default=ROOT/'output/pri')
    p.add_argument('--group-root',type=Path,default=ROOT/'output/grouped')
    p.add_argument('--position-root',type=Path,default=ROOT/'output/group_inputs')
    p.add_argument('--config',type=Path,default=ROOT/'config/fusion_config.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/fused')
    p.add_argument('--sequences-only',action='store_true')
    p.add_argument('--start-seconds',type=float)
    p.add_argument('--end-seconds',type=float)
    p.add_argument('--toa-unit',choices=('us','ns','ms','s'),default='us')
    a=p.parse_args()
    if a.a_file and a.split=='both':p.error('--a-file requires a single split')
    if not a.sequences_only and (a.start_seconds is not None or a.end_seconds is not None):p.error('Time filters require --sequences-only')
    cfg=FusionConfig(**json.loads(a.config.read_text()))
    for split in (('train','test') if a.split=='both' else (a.split,)):
        r=run_fusion(a.pri_root/split/'pri_sequences.csv',a.group_root/split/'bearing_observations.csv',
            a.position_root/f'{split}_platform_positions.csv',a.output_dir/split,cfg,
            None if a.sequences_only else a.a_file or a.data_root/DEFAULT_FILES[split][0],
            a.group_root/split,a.pri_root/split,a.start_seconds,a.end_seconds,a.toa_unit)
        print(json.dumps({k:r[k] for k in ('object_count','event_count','assigned_sequence_pulses','pulse_export','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
