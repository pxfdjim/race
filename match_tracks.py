#!/usr/bin/env python3
"""Match final horizontal A tracks to all supplied B tracks using trajectory data."""
import argparse
import csv
import json
import math
import os
import time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path

from prepare_inputs import DEFAULT_FILES,digest
from src.dataio import InputConfig,parse_timestamp
from src.localization import HorizontalFrame
from src.track_matching import BTrajectory,MatchConfig,TrackPoint,match
from src.track_matching_plot import render

ROOT=Path(__file__).resolve().parent
SCOPE=['scenario_id','data_version']
OBJECT=SCOPE+['object_id']
POINT_FIELDS=OBJECT+['segment_id','time_ns','time_seconds','track_id','object_match_status','point_status',
    'a_east_m','a_north_m','b_east_m','b_north_m','distance_m']


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def write_rows(path,fields,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def load_a(directory,split):
    directory=Path(directory)
    meta=json.loads((directory/'localization_run.json').read_text())
    if meta['dimension']!=2 or meta['height_used'] or meta['split']!=split:raise ValueError('Expected horizontal A localization results for selected split')
    files={n:directory/n for n in ['localization_run.json','track_points.csv','localization_objects.csv']}
    for n in ['track_points.csv','localization_objects.csv']:
        if digest(files[n])!=meta['files'][n]['sha256']:raise ValueError(f'A localization artifact changed: {n}')
    frames={(r['scenario_id'],r['data_version']):HorizontalFrame(r['origin_longitude'],r['origin_latitude']) for r in meta['frames']}
    objects={}
    for r in read_rows(files['localization_objects.csv']):
        key=tuple(r[k] for k in OBJECT)
        if key in objects or key[:2] not in frames or key[1]!=split:raise ValueError('Invalid A object key')
        objects[key]=[]
    for r in read_rows(files['track_points.csv']):
        key=tuple(r[k] for k in OBJECT)
        t=int(r['time_ns']);x,y=float(r['east_m']),float(r['north_m'])
        if (key not in objects or t<0 or not r['segment_id'] or not all(map(math.isfinite,(x,y))) or
            abs(t/1e9-float(r['time_seconds']))>1e-8):raise ValueError('Invalid A track point')
        objects[key].append(TrackPoint(t,(x,y),r['segment_id']))
    for key,points in objects.items():
        points.sort(key=lambda p:p.time)
        if len({p.time for p in points})!=len(points):raise ValueError('Duplicate A object/time')
    return meta,files,frames,objects


def load_b(path,frames,units,cfg):
    groups=defaultdict(list)
    for r in read_rows(path):
        scope=r['scenario_id'],r['data_version']
        if scope not in frames or not r['track_id']:raise ValueError('B scope does not match A')
        stamp,_,kind=parse_timestamp(r['datetime']);origin,origin_kind=units.origin(*scope)
        lon=float(r.get('target_longitude',r.get('target_lontitude','nan')));lat=float(r['target_latitude'])
        if kind!=origin_kind or stamp<origin or not all(map(math.isfinite,(lon,lat))) or not -180<=lon<=180 or not -89<=lat<=89:
            raise ValueError('Invalid B timestamp/position')
        groups[scope+(r['track_id'],)].append(TrackPoint(stamp-origin,frames[scope].project(lon,lat),''))
    if not groups:raise ValueError('No B tracks')
    return {k:BTrajectory(k,ps,cfg.maximum_b_gap_seconds) for k,ps in groups.items()}


def run(root,a_root,output,split,cfg,b_file=None):
    cfg.validate();started=time.monotonic()
    root,a_root,output=map(lambda p:Path(p).resolve(),(root,a_root,output))
    if output.exists():raise FileExistsError(f'Choose a new output directory: {output}')
    meta,a_files,frames,objects=load_a(a_root/split,split)
    units_path=root/'config/input_units.json';units_dict=json.loads(units_path.read_text())
    units=InputConfig.from_dict(units_dict)
    b_file=Path(b_file).resolve() if b_file else root.parent/DEFAULT_FILES[split][1]
    tracks=load_b(b_file,frames,units,cfg)
    # Frozen trajectory decisions: former B anchor IDs never enter match().
    decisions,pairs=match(objects,tracks,cfg)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    pair_fields=['track_id','a_point_count','comparison_count','supported_duration_seconds','temporal_coverage',
        'spatial_inlier_fraction','median_distance_m','p90_distance_m','rms_distance_m','motion_comparison_count',
        'median_heading_difference_degrees','median_speed_difference_m_per_second','motion_used',
        'first_comparison_seconds','last_comparison_seconds','score','temporal_eligible','eligible','rejection_reasons']
    write_rows(partial/'pair_candidates.csv',OBJECT+pair_fields,
        (dict(zip(OBJECT,k[:3]))|v for k,v in sorted(pairs.items())))
    write_rows(partial/'object_track_links.csv',OBJECT+['track_id','link_status'],
        (dict(zip(OBJECT,k),track_id=d['track_id'],link_status='matched' if d['status']=='matched' else 'unmatched') for k,d in sorted(decisions.items())))
    detail_rows=[]
    for k,d in sorted(decisions.items()):
        best=pairs.get(k+(d['best_track_id'],),{})
        detail_rows.append(dict(zip(OBJECT,k))|d|{n:best.get(n) for n in pair_fields if n!='track_id'})
    detail_fields=OBJECT+['track_id','status','reason','best_track_id','best_score','runner_up_track_id',
        'score_margin','relative_score_margin']+[n for n in pair_fields if n!='track_id']
    write_rows(partial/'object_match_details.csv',detail_fields,detail_rows)
    point_rows=[];point_statuses=Counter()
    for k,points in sorted(objects.items()):
        d=decisions[k];b=tracks.get(k[:2]+(d['track_id'],)) if d['track_id'] else None
        for p in points:
            q=b.sample(p.time) if b else None
            distance=math.dist(p.xy,q.xy) if q else None
            status=('matched' if distance<=cfg.inlier_distance_m else 'spatial_outlier') if q else ('B_gap_or_outside_span' if b else 'object_'+d['status'])
            point_rows.append(dict(zip(OBJECT,k),segment_id=p.segment,time_ns=p.time,time_seconds=p.time/1e9,
                track_id=d['track_id'],object_match_status=d['status'],point_status=status,
                a_east_m=p.xy[0],a_north_m=p.xy[1],b_east_m=q.xy[0] if q else None,
                b_north_m=q.xy[1] if q else None,distance_m=distance))
            point_statuses[status]+=1
    write_rows(partial/'matched_track_points.csv',POINT_FIELDS,point_rows)
    usage=Counter(k[:2]+(d['track_id'],) for k,d in decisions.items() if d['track_id'])
    write_rows(partial/'B_track_usage.csv',SCOPE+['track_id','observation_count','linked_A_object_count','status'],
        (dict(zip(SCOPE,k[:2]),track_id=k[2],observation_count=len(b.points),linked_A_object_count=usage[k],
              status='used' if usage[k] else 'unused') for k,b in sorted(tracks.items())))
    # Optional AFTER-the-fact audit only, deliberately outside the matching core.
    prior_path=root/f'output/fused_b_assisted_guarded/{split}/object_B_priors.csv'
    priors={tuple(r[k] for k in OBJECT):r['track_id'] for r in read_rows(prior_path)} if prior_path.exists() else {}
    audit=[]
    for k,d in sorted(decisions.items()):
        old=priors.get(k,'')
        comparison='no_prior' if not old else ('no_final_match' if not d['track_id'] else ('same' if old==d['track_id'] else 'different'))
        audit.append(dict(zip(OBJECT,k),prior_track_id=old,final_track_id=d['track_id'],final_status=d['status'],comparison=comparison))
    write_rows(partial/'auxiliary_prior_comparison.csv',OBJECT+['prior_track_id','final_track_id','final_status','comparison'],audit)
    input_paths={**a_files,'B_source':b_file,'input_units':units_path}
    report={'config':asdict(cfg),'split':split,'dimension':2,'height_used':False,
        'A_identity_truth_used':False,'B_identity_truth_used':False,'class_or_type_used':False,
        'former_B_anchors_used_to_decide':False,'upstream_association_B_assisted':meta['upstream_association_B_assisted'],
        'B_smoothing_applied':False,'B_extrapolation_allowed':False,'A_points_modified':False,
        'assignment_policy':'At most one B track per A object; multiple A fragments may link to the same B. Ambiguity remains unmatched.',
        'frames':meta['frames'],'A_object_count':len(objects),'B_track_count':len(tracks),
        'candidate_pair_count':len(pairs),'status_counts':dict(Counter(d['status'] for d in decisions.values())),
        'matched_object_count':sum(d['status']=='matched' for d in decisions.values()),
        'used_B_track_count':len(usage),'input_A_track_points':sum(map(len,objects.values())),
        'point_status_counts':dict(point_statuses),'auxiliary_prior_comparison':dict(Counter(r['comparison'] for r in audit)),
        'prior_annotation':{'path':str(prior_path),'sha256':digest(prior_path)} if prior_path.exists() else None,
        'input_files':{n:{'path':str(p),'sha256':digest(p)} for n,p in input_paths.items()},
        'input_units':units_dict,
        'algorithm_source_sha256':{n:digest(ROOT/n) for n in ['match_tracks.py','src/track_matching.py','src/track_matching_plot.py']},
        'limitations':[
            'A identities were constructed with B assistance; this is not an independent A-only accuracy evaluation.',
            'Position residuals compare two sensor estimates, not independent truth.',
            'Heuristic gates are not calibrated probabilities; no test truth or prior B-anchor IDs decide matching.',
            'One whole-object association is produced; mixed upstream objects may remain unmatched despite some valid fragments.',
            'B candidates may include false, duplicated, or anomalous tracks; some remain ambiguous or unused.'
        ]}
    render(partial,frames,objects,tracks,decisions,pairs)
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    report['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(partial.iterdir())}
    (partial/'track_match_run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    partial.rename(output)
    print(json.dumps({k:report[k] for k in ['split','A_object_count','B_track_count','status_counts','used_B_track_count','point_status_counts','elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--a-root',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'output/ab_matched_2d')
    p.add_argument('--split',choices=['train','test','both'],default='both')
    p.add_argument('--config',type=Path,default=ROOT/'config/track_matching_config.json')
    p.add_argument('--b-file',type=Path,help='Custom B CSV, for a single split only')
    a=p.parse_args()
    if a.split=='both' and a.b_file:p.error('--b-file requires a single split')
    cfg=MatchConfig(**json.loads(a.config.read_text()))
    for split in (['train','test'] if a.split=='both' else [a.split]):
        run(a.pipeline_root,a.a_root or a.pipeline_root/'output/localized_2d',a.output_dir/split,split,cfg,a.b_file)


if __name__=='__main__':main()
