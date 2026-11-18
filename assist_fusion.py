#!/usr/bin/env python3
"""Add B bearing priors to saved A-only fusion; preserve the baseline directory."""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from fuse_platforms import load_observations, export_pulses, LINK_FIELDS, SEQ_KEYS
from prepare_inputs import DEFAULT_FILES, digest
from src.dataio import InputConfig, read_b_tracks
from src.b_assisted import AssistanceConfig, assist

ROOT=Path(__file__).resolve().parent


def read_baseline(directory, observations):
    directory=Path(directory)
    meta=json.loads((directory/'fusion_run.json').read_text())
    if meta['B_used']:
        raise ValueError('Input must be the unchanged A-only baseline')
    for name in ('sequence_object_links.csv','cross_platform_events.csv'):
        if digest(directory/name)!=meta['files'][name]['sha256']:
            raise ValueError(f'Changed baseline artifact: {name}')
    keys={o.key for o in observations}
    links={}
    with (directory/'sequence_object_links.csv').open() as f:
        for r in csv.DictReader(f):
            k=tuple(r[n] for n in SEQ_KEYS)
            if k in keys:
                if k in links:raise ValueError('Duplicate baseline sequence')
                links[k]=r
    if links.keys()!=keys:raise ValueError('Missing baseline sequences')
    return links,meta


def run(root, split, output, cfg, input_config, b_file=None, start=None, end=None, sequences_only=False):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    started=time.monotonic();cfg.validate()
    if ((start is not None and (not math.isfinite(start) or start<0)) or
        (end is not None and (not math.isfinite(end) or end<=0)) or
        (start is not None and end is not None and end<=start)):
        raise ValueError('Invalid time filter')
    if (start is not None or end is not None) and not sequences_only:
        raise ValueError('Time filters require sequences-only')
    seq=root/f'output/pri/{split}/pri_sequences.csv'
    bearings=root/f'output/grouped/{split}/bearing_observations.csv'
    positions=root/f'output/group_inputs/{split}_platform_positions.csv'
    baseline_dir=root/f'output/fused/{split}'
    obs,frames=load_observations(seq,bearings,positions,start,end)
    baseline,base_meta=read_baseline(baseline_dir,obs)
    if digest(seq)!=base_meta['sequence_source_sha256'] or digest(bearings)!=base_meta['bearing_source_sha256'] or digest(positions)!=base_meta['positions_source_sha256']:
        raise ValueError('Baseline/source mismatch')
    b_file=Path(b_file) if b_file else root.parent/DEFAULT_FILES[split][1]
    units=InputConfig.from_dict(json.loads(Path(input_config).read_text()))
    tracks=read_b_tracks(b_file,units)
    links,anchors,diagnostics,stats=assist(obs,tracks,baseline,cfg,lambda r:print(json.dumps(r),flush=True))
    members=defaultdict(list)
    for o in obs:
        if links[o.key]['object_id']:members[o.sequence.scope[:2]+(links[o.key]['object_id'],)].append(o)
    # Retain an A-only geometric event only when its complete membership agrees.
    # B cannot turn a rejected crossing or a single bearing into a triangulation.
    retained=[];released=[]
    with (baseline_dir/'cross_platform_events.csv').open() as f:
        reader=csv.DictReader(f);event_fields=reader.fieldnames
        for r in reader:
            scene=r['scenario_id'],r['data_version']
            keys=[scene+(p,s) for p,s in zip(r['platform_ids'].split('|'),r['sequence_ids'].split('|'))]
            if any(k not in links for k in keys):continue
            ids={links[k]['object_id'] for k in keys}
            if len(ids)==1:
                r['object_id']=next(iter(ids));retained.append(r)
                for k in keys:links[k]['event_id']=r['event_id']
            else:released.append({k:r[k] for k in ('scenario_id','data_version','event_id')}|{'reason':'B_assisted_member_identity_disagreement'})
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    def write(name,fields,rows):
        with (partial/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write('sequence_object_links.csv',LINK_FIELDS,(
        dict(zip(SEQ_KEYS,o.key),local_cluster_id=o.local_group,object_id=links[o.key]['object_id'],
             event_id=links[o.key]['event_id'],fusion_basis=links[o.key]['basis'],
             association_cost=links[o.key]['cost'] if links[o.key]['cost'] is not None else '') for o in sorted(obs,key=lambda o:o.key)))
    write('cross_platform_events.csv',event_fields,retained)
    write('released_events.csv',['scenario_id','data_version','event_id','reason'],released)
    write('sequence_B_evidence.csv',SEQ_KEYS+['track_id','bearing_residual_degrees','bearing_gate_degrees',
        'initial_seed','baseline_object_id','association_basis'],(
        dict(zip(SEQ_KEYS,d['key']),track_id=d['track_id'],bearing_residual_degrees=d['residual'] if d['residual'] is not None else '',
             bearing_gate_degrees=d['gate'] if d['gate'] is not None else '',initial_seed=d['seed'],
             baseline_object_id=d['baseline_object_id'],association_basis=d['basis']) for d in diagnostics))
    write('object_B_priors.csv',['scenario_id','data_version','object_id','track_id','prior_status'],(
        dict(scenario_id=k[0],data_version=k[1],object_id=oid,track_id=k[2],prior_status='auxiliary_identity_anchor')
        for k,oid in sorted(anchors.items()) if k[:2]+(oid,) in members))
    pulse_export,counts=None,Counter()
    if not sequences_only:
        pri_meta=json.loads((root/f'output/pri/{split}/pri_run.json').read_text())
        group_meta=json.loads((root/f'output/grouped/{split}/grouping_run.json').read_text())
        raw=root.parent/DEFAULT_FILES[split][0]
        if digest(raw)!=pri_meta['source_sha256'] or units.a_toa_unit!=pri_meta['input_config']['a_toa_unit']:
            raise ValueError('Raw A/PRI units or source mismatch')
        predictions=root/f'output/grouped/{split}/local_predictions.csv'
        if digest(predictions)!=group_meta['files']['local_predictions.csv']['sha256']:
            raise ValueError('Changed local predictions')
        pulse_export,counts=export_pulses(raw,predictions,partial/'pulse_object_predictions.csv',obs,links,units.a_toa_unit)
        if pulse_export['input_output_pulses']!=pri_meta['input_pulses']:raise ValueError('Lost source pulses')
    event_counts=Counter((r['scenario_id'],r['data_version'],r['object_id']) for r in retained if r['object_id'])
    obj_rows=[];local_members=Counter()
    for key,items in sorted(members.items()):
        scene,version,oid=key;platforms=sorted({o.platform for o in items});n=sum(o.sequence.count for o in items)
        obj_rows.append(dict(scenario_id=scene,data_version=version,object_id=oid,platform_count=len(platforms),
            platform_ids='|'.join(platforms),local_group_count=len({o.sequence.scope+(o.local_group,) for o in items if o.local_group}),
            sequence_count=len(items),sequence_pulses=n,total_assigned_pulses=counts.get(key,n),event_count=event_counts[key],
            start_ns=min(o.sequence.start for o in items),end_ns=max(o.sequence.end for o in items),
            object_status='B_anchored' if oid.startswith('OB') else 'A_only_fallback'))
        for o in items:
            if o.local_group:local_members[o.sequence.scope+(o.local_group,oid)]+=o.sequence.count
    write('fused_objects.csv',['scenario_id','data_version','object_id','platform_count','platform_ids','local_group_count',
        'sequence_count','sequence_pulses','total_assigned_pulses','event_count','start_ns','end_ns','object_status'],obj_rows)
    fields=SEQ_KEYS[:3]+['local_cluster_id','object_id','sequence_pulses']
    write('local_object_memberships.csv',fields,(dict(zip(fields,k+(v,))) for k,v in sorted(local_members.items())))
    report={'config':asdict(cfg),'input_config':json.loads(Path(input_config).read_text()),'sequence_count':len(obs),
        'sequence_pulses':sum(o.sequence.count for o in obs),'assigned_sequence_count':sum(bool(l['object_id']) for l in links.values()),
        'assigned_sequence_pulses':sum(o.sequence.count for o in obs if links[o.key]['object_id']),
        'object_count':len(members),'event_count':len(retained),'released_baseline_event_count':len(released),
        'events_with_object_identity':sum(bool(r['object_id']) for r in retained),'pulse_export':pulse_export,
        'assistance_statistics':stats,'B_used':True,'A_truth_columns_used':[],'B_identity_truth_used':False,
        'B_class_or_type_used':False,'B_count_hard_constraint':False,'B_position_used_as_A_localization':False,
        'baseline_directory':str(baseline_dir),'baseline_run_sha256':digest(baseline_dir/'fusion_run.json'),
        'B_source':str(b_file.resolve()),'B_source_sha256':digest(b_file),
        'sequence_source_sha256':digest(seq),'bearing_source_sha256':digest(bearings),'positions_source_sha256':digest(positions),
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ('assist_fusion.py','src/b_assisted.py','src/dataio.py','fuse_platforms.py')},
        'time_filter_seconds':[start,end],
        'limitations':'B-assisted association, not independent A-only accuracy or final A localization. '
            'B false tracks and same-bearing ambiguities may remain. A-only global identity is a fallback, not a truth label.',
        'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in partial.iterdir()},
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'fusion_run.json').write_text(json.dumps(report,indent=2)+'\n');partial.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--split',choices=('train','test','both'),default='both')
    p.add_argument('--b-file',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'config/b_assisted_config.json')
    p.add_argument('--input-config',type=Path,default=ROOT/'config/input_units.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/fused_b_assisted')
    p.add_argument('--sequences-only',action='store_true')
    p.add_argument('--start-seconds',type=float);p.add_argument('--end-seconds',type=float)
    a=p.parse_args()
    if a.b_file and a.split=='both':p.error('--b-file requires a single split')
    if (a.start_seconds is not None or a.end_seconds is not None) and not a.sequences_only:p.error('Time filters require --sequences-only')
    cfg=AssistanceConfig(**json.loads(a.config.read_text()))
    for split in ('train','test') if a.split=='both' else (a.split,):
        r=run(a.pipeline_root,split,a.output_dir/split,cfg,a.input_config,a.b_file,
              a.start_seconds,a.end_seconds,a.sequences_only)
        print(json.dumps({k:r[k] for k in ('object_count','assigned_sequence_pulses','pulse_export','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
