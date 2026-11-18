#!/usr/bin/env python3
"""Associate PRI sequences into persistent platform-local groups, without A labels or B."""
import argparse
import csv
import itertools
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from prepare_group_inputs import ObserverPaths, sample_positions
from prepare_inputs import DEFAULT_FILES,digest
from src.dataio import time_to_ns
from src.local_grouping import GroupConfig,Sequence,PulseAttacher,associate

ROOT=Path(__file__).resolve().parent
KEYS=['scenario_id','data_version','platform_id','pulse_id']
LINK_FIELDS=KEYS[:3]+['sequence_id','local_cluster_id','association_cost','association_basis']
GROUP_FIELDS=KEYS[:3]+['local_cluster_id','group_status','sequence_count','sequence_pulses',
    'total_assigned_pulses','start_toa_ns','end_toa_ns','rf_mean_mhz','rf_min_mhz','rf_max_mhz',
    'pw_mean_us','pw_min_us','pw_max_us','pri_min_us','pri_max_us','stable_pulses','jittered_pulses']
PULSE_FIELDS=KEYS+['sequence_id','local_cluster_id','pri_model','pri_center_us','group_assignment_status',
    'association_cost','association_basis']


def read_sequences(path,positions,start=None,end=None):
    items,seen=[],set()
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            scope=tuple(r[k] for k in KEYS[:3])
            a,b=int(r['start_toa_ns']),int(r['end_toa_ns'])
            if (start is not None and a<start*1e9) or (end is not None and b>=end*1e9):
                continue
            s=Sequence(scope,r['sequence_id'],a,b,int(r['pulse_count']),r['pri_model'],
                float(r['pri_center_us']),float(r['rf_mean']),float(r['pw_mean']),float(r['pa_mean']),
                float(r['doa_mean_degrees'])%360.,float(r['doa_std_degrees']),positions.position(scope,(a+b)//2))
            if (s.key in seen or s.end<s.start or s.count<2 or s.pri<=0 or
                s.model not in ('stable','jittered') or not 0<=float(r['doa_mean_degrees'])<=360 or s.doa_std<0 or
                not all(math.isfinite(v) for v in (s.pri,s.rf,s.pw,s.pa,s.doa,s.doa_std))):
                raise ValueError('Invalid/duplicate sequence observation')
            seen.add(s.key)
            items.append(s)
    return items


def export_pulses(source,pri_predictions,destination,sequences,links,config,unit='us'):
    attacher=PulseAttacher(sequences,links,config)
    statuses,group_counts=Counter(),Counter()
    unknown_bounds={}
    total=0
    with Path(source).open(encoding='utf-8-sig',newline='') as af, \
         Path(pri_predictions).open(newline='') as pf, Path(destination).open('w',newline='') as out:
        a,p=csv.reader(af),csv.reader(pf)
        ah,ph=next(a),next(p)
        ai,pi=[ah.index(k) for k in KEYS],[ph.index(k) for k in KEYS]
        seq_i,model_i,center_i=[ph.index(k) for k in ('sequence_id','pri_model','pri_center_us')]
        def index(*names):
            return next(ah.index(k) for k in names if k in ah)
        toa_i,rf_i,pw_i,pa_i,doa_i=(index('toa'),index('param1','rf_mhz','RF'),
            index('param2','pw_us','PW'),index('param3','pa_dbm','PA'),index('doa','doa_degrees'))
        writer=csv.writer(out)
        writer.writerow(PULSE_FIELDS)
        for raw,pri in itertools.zip_longest(a,p):
            if raw is None or pri is None or any(raw[i]!=pri[j] for i,j in zip(ai,pi)):
                raise ValueError('Raw input and PRI rows differ in count, keys, or order')
            key=tuple(pri[i] for i in pi)
            if pri[seq_i]:
                link=links[key[:3]+(pri[seq_i],)]
                group,cost,basis=link['local_cluster_id'],link['association_cost'],link['association_basis']
                status='sequence_assigned' if group else 'unassigned_ambiguous_sequence'
            else:
                unknown_time=time_to_ns(raw[toa_i],unit)
                group,cost,basis=attacher.attach(key[:3],unknown_time,
                    float(raw[rf_i]),float(raw[pw_i]),float(raw[pa_i]),float(raw[doa_i]))
                status='unknown_pri_attached' if group else 'unassigned_'+basis
                if group:
                    bounds=unknown_bounds.setdefault(key[:3]+(group,),[unknown_time,unknown_time])
                    bounds[0],bounds[1]=min(bounds[0],unknown_time),max(bounds[1],unknown_time)
            writer.writerow([*key,pri[seq_i],group,pri[model_i],pri[center_i],status,
                             cost if cost is not None else '',basis])
            statuses[status]+=1
            if group:
                group_counts[key[:3]+(group,)]+=1
            total+=1
            if total%2_000_000==0:
                print(json.dumps({'exported_pulses':total,'statuses':dict(statuses)}),flush=True)
    return {'input_output_pulses':total,'assignment_status_counts':dict(statuses),
        'attached_unknown_time_ranges_ns':[dict(zip(KEYS[:3]+['local_cluster_id'],k),start_ns=v[0],end_ns=v[1])
            for k,v in sorted(unknown_bounds.items())]},group_counts


def run_grouping(sequence_file,position_file,output,config,source=None,pri_dir=None,
                 start=None,end=None,unit='us'):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    config.validate()
    if ((start is not None and (not math.isfinite(start) or start<0)) or
        (end is not None and (not math.isfinite(end) or end<=0)) or
        (start is not None and end is not None and end<=start)):
        raise ValueError('Invalid time filter')
    started=time.monotonic()
    position_file,sequence_file=Path(position_file),Path(sequence_file)
    positions=ObserverPaths(position_file)
    sequences=read_sequences(sequence_file,positions,start,end)
    groups,links=associate(sequences,config,lambda r:print(json.dumps(r),flush=True))
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}')
    partial.mkdir()
    with (partial/'sequence_group_links.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,LINK_FIELDS)
        w.writeheader()
        for key,link in sorted(links.items()):
            w.writerow(dict(zip(LINK_FIELDS[:4],key))|link)
    export,group_counts=None,Counter()
    if source is not None:
        if start is not None or end is not None:
            raise ValueError('Time-filtered association cannot export full pulse predictions')
        pri_dir=Path(pri_dir)
        meta=json.loads((pri_dir/'pri_run.json').read_text())
        if unit!=meta['input_config']['a_toa_unit']:
            raise ValueError('Source TOA unit differs from PRI run metadata')
        path_meta=json.loads(position_file.with_suffix('.json').read_text())
        if digest(source)!=meta['source_sha256'] or path_meta['source_sha256']!=meta['source_sha256']:
            raise ValueError('Raw/PRI/observer source hash mismatch')
        for name in ('pulse_pri.csv','pri_sequences.csv'):
            if digest(pri_dir/name)!=meta['files'][name]['sha256']:
                raise ValueError('PRI artifacts changed')
        export,group_counts=export_pulses(source,pri_dir/'pulse_pri.csv',partial/'local_predictions.csv',
            sequences,links,config,unit)
        if export['input_output_pulses']!=meta['input_pulses']:
            raise ValueError('Pulse coverage mismatch')
    with (partial/'local_groups.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,GROUP_FIELDS)
        w.writeheader()
        for g in groups:
            bounds=next((r for r in (export or {}).get('attached_unknown_time_ranges_ns',[])
                         if tuple(r[k] for k in KEYS[:3])==g.scope and r['local_cluster_id']==g.name),None)
            confirmed=g.count>=config.confirmed_min_sequences and g.pulse_count>=config.confirmed_min_pulses
            row=dict(zip(KEYS[:3],g.scope),local_cluster_id=g.name,
                group_status='confirmed' if confirmed else 'tentative',sequence_count=g.count,
                sequence_pulses=g.pulse_count,total_assigned_pulses=group_counts.get(g.scope+(g.name,),g.pulse_count),
                start_toa_ns=min(g.first,bounds['start_ns']) if bounds else g.first,
                end_toa_ns=max(g.last,bounds['end_ns']) if bounds else g.last,
                rf_mean_mhz=sum(s.rf*s.count for s in g.members)/g.pulse_count,
                rf_min_mhz=min(s.rf for s in g.members),rf_max_mhz=max(s.rf for s in g.members),
                pw_mean_us=sum(s.pw*s.count for s in g.members)/g.pulse_count,
                pw_min_us=min(s.pw for s in g.members),pw_max_us=max(s.pw for s in g.members),
                pri_min_us=min(s.pri for s in g.members),pri_max_us=max(s.pri for s in g.members),
                stable_pulses=g.models['stable'],jittered_pulses=g.models['jittered'])
            w.writerow(row)
    with (partial/'bearing_observations.csv').open('w',newline='') as out:
        w=csv.writer(out)
        w.writerow(KEYS[:3]+['local_cluster_id','sequence_id','time_ns','start_ns','end_ns','doa_degrees',
                            'doa_standard_error_degrees','observer_longitude','observer_latitude','observer_height'])
        for s in sorted(sequences,key=lambda s:(s.scope,s.time,s.sequence_id)):
            w.writerow([*s.scope,links[s.key]['local_cluster_id'],s.sequence_id,s.time,s.start,s.end,s.doa,s.error,*s.position])
    report={'config':asdict(config),'sequence_source':str(sequence_file.resolve()),
        'sequence_source_sha256':digest(sequence_file),'observer_source_sha256':digest(position_file),
        'sequence_count':len(sequences),'sequence_pulses':sum(s.count for s in sequences),
        'assigned_sequence_count':sum(bool(r['local_cluster_id']) for r in links.values()),
        'assigned_sequence_pulses':sum(g.pulse_count for g in groups),
        'group_count':len(groups),'platform_group_counts':[dict(zip(KEYS[:3],scope),groups=n)
            for scope,n in sorted(Counter(g.scope for g in groups).items())],
        'pulse_export':export,'time_filter_seconds':[start,end],
        'A_truth_fields_used':[],'B_used':False,'class_equals_identity_assumed':False,
        'group_ID_scope':KEYS[:3]+['local_cluster_id'],'output_is_ground_truth':False,
        'algorithm_source_sha256':{name:digest(ROOT/name) for name in
            ('group_sequences.py','src/local_grouping.py','prepare_group_inputs.py')},
        'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in partial.iterdir()},
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'grouping_run.json').write_text(json.dumps(report,indent=2)+'\n')
    partial.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT.parent)
    p.add_argument('--split',choices=('train','test','both'),default='both')
    p.add_argument('--a-file',type=Path)
    p.add_argument('--pri-root',type=Path,default=ROOT/'output/pri')
    p.add_argument('--position-root',type=Path,default=ROOT/'output/group_inputs')
    p.add_argument('--config',type=Path,default=ROOT/'config/group_config.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/grouped')
    p.add_argument('--toa-unit',choices=('us','ns','ms','s'),default='us')
    p.add_argument('--sequences-only',action='store_true')
    p.add_argument('--start-seconds',type=float)
    p.add_argument('--end-seconds',type=float)
    a=p.parse_args()
    if a.a_file and a.split=='both':
        p.error('--a-file requires a single split')
    if not a.sequences_only and (a.start_seconds is not None or a.end_seconds is not None):
        p.error('Time filters require --sequences-only')
    cfg=GroupConfig(**json.loads(a.config.read_text()))
    for split in (('train','test') if a.split=='both' else (a.split,)):
        source=a.a_file or a.data_root/DEFAULT_FILES[split][0]
        positions=a.position_root/f'{split}_platform_positions.csv'
        if not positions.exists():
            sample_positions(source,positions,a.toa_unit)
        r=run_grouping(a.pri_root/split/'pri_sequences.csv',positions,a.output_dir/split,cfg,
            None if a.sequences_only else source,a.pri_root/split,a.start_seconds,a.end_seconds,a.toa_unit)
        print(json.dumps({k:r[k] for k in ('group_count','platform_group_counts','pulse_export','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    main()
