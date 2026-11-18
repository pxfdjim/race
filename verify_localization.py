#!/usr/bin/env python3
"""Independent artifact, membership and forward-bearing checks for 2D outputs."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from prepare_inputs import digest
from src.localization import HorizontalFrame


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def key(row, names):return tuple(row[n] for n in names)


SCOPE = ['scenario_id','data_version']
OBJECT = SCOPE+['object_id']
EVENT = SCOPE+['event_id']
SEQUENCE = SCOPE+['platform_id','sequence_id']


def finite(row, fields):
    if not all(math.isfinite(float(row[n])) for n in fields):raise ValueError('Nonfinite output value')


def verify(directory):
    directory=Path(directory)
    report=json.loads((directory/'localization_run.json').read_text())
    for n,info in report['files'].items():
        if digest(directory/n)!=info['sha256']:raise ValueError(f'Changed output: {n}')
    for name,info in report['input_files'].items():
        if digest(info['path'])!=info['sha256']:raise ValueError(f'Changed input: {name}')
    cfg=report['config']
    if report['dimension']!=2 or report['height_used'] or report['height_estimated'] or report['B_coordinates_used_in_localization'] or report['A_truth_columns_used']:
        raise ValueError('Unexpected localization inputs/dimension')
    frames={key(r,SCOPE):HorizontalFrame(r['origin_longitude'],r['origin_latitude']) for r in report['frames']}
    inputs=report['input_files']
    events={key(r,EVENT):r for r in rows(inputs['events']['path'])}
    bearings={key(r,SEQUENCE):r for r in rows(inputs['bearings']['path'])}
    links={key(r,SEQUENCE):r for r in rows(inputs['links']['path'])}
    points={};counts=Counter()
    for r in rows(directory/'localized_points.csv'):
        k=key(r,EVENT);scene=k[:2]
        if k in points or k not in events:raise ValueError('Duplicate or unknown localized event')
        e=events[k]
        if not r['object_id'] or r['object_id']!=e['object_id']:raise ValueError('Wrong localization identity')
        finite(r,['time_seconds','east_m','north_m','longitude','latitude','condition_number',
                  'covariance_ee_m2','covariance_en_m2','covariance_nn_m2','major_error_proxy_m','minor_error_proxy_m'])
        if abs(float(r['time_seconds'])-int(r['time_ns'])/1e9)>1e-9:raise ValueError('Time unit mismatch')
        a,b,c=(float(r[n]) for n in ['covariance_ee_m2','covariance_en_m2','covariance_nn_m2'])
        if a<=0 or c<=0 or a*c-b*b<=0:raise ValueError('Nonpositive position covariance')
        if float(r['condition_number'])>cfg['maximum_condition_number'] or float(r['major_error_proxy_m'])>cfg['maximum_major_error_proxy_m']:
            raise ValueError('Invalid accepted geometry')
        platforms,sequences=r['platform_ids'].split('|'),r['sequence_ids'].split('|')
        if len(platforms)<2 or len(platforms)!=len(set(platforms)) or len(platforms)!=len(sequences) or len(platforms)!=int(r['used_platform_count']):
            raise ValueError('Not independent multi-platform geometry')
        original=set(zip(e['platform_ids'].split('|'),e['sequence_ids'].split('|')))
        if not set(zip(platforms,sequences))<=original or int(r['input_platform_count'])!=len(original):raise ValueError('Changed event members')
        original_times=[int(bearings[scene+(p,s)]['time_ns']) for p,s in original]
        if int(r['time_ns'])!=round(median(original_times)):raise ValueError('Wrong reference time')
        if max(original_times)-min(original_times)>cfg['maximum_time_span_seconds']*1e9+1:raise ValueError('Invalid async span')
        reconstructed=frames[scene].project(float(r['longitude']),float(r['latitude']))
        if math.dist(reconstructed,(float(r['east_m']),float(r['north_m'])))>.001:raise ValueError('Geographic conversion mismatch')
        points[k]=r;counts[key(r,OBJECT)]+=1
    rejected={}
    for r in rows(directory/'rejected_localizations.csv'):
        k=key(r,EVENT)
        if k in rejected or k not in events or r['object_id']!=events[k]['object_id'] or not r['reason']:
            raise ValueError('Invalid rejected event')
        rejected[k]=r
    if points.keys() & rejected.keys() or points.keys() | rejected.keys()!=events.keys():raise ValueError('Incomplete event disposition')
    diagnostic=defaultdict(dict)
    for r in rows(directory/'localization_observations.csv'):
        ekey=key(r,EVENT);skey=key(r,SEQUENCE)
        if ekey not in points or skey not in bearings or skey in diagnostic[ekey]:raise ValueError('Invalid observer diagnostic')
        p,src=points[ekey],bearings[skey]
        if key(r,OBJECT)!=key(p,OBJECT):raise ValueError('Diagnostic identity mismatch')
        if links[skey]['object_id']!=r['object_id'] or links[skey]['event_id']!=r['event_id']:raise ValueError('Observer identity mismatch')
        if any(r[a]!=src[b] for a,b in [('time_ns','time_ns'),('observer_longitude','observer_longitude'),('observer_latitude','observer_latitude')]):
            # CSV float spelling can differ without numeric value changing.
            if any(float(r[a])!=float(src[b]) for a,b in [('time_ns','time_ns'),('observer_longitude','observer_longitude'),('observer_latitude','observer_latitude')]):
                raise ValueError('Observer measurement changed')
        frame=frames[skey[:2]]
        xy=frame.project(float(src['observer_longitude']),float(src['observer_latitude']))
        direction=frame.direction(float(src['observer_longitude']),float(src['observer_latitude']),float(src['doa_degrees']))
        dx,dy=float(p['east_m'])-xy[0],float(p['north_m'])-xy[1]
        distance=math.hypot(dx,dy);forward=dx*direction[0]+dy*direction[1]
        residual=math.degrees(math.atan2(direction[0]*dy-direction[1]*dx,forward))
        if abs(residual-float(r['bearing_residual_degrees']))>1e-8 or abs(distance-float(r['range_m']))>1e-5:
            raise ValueError('Forward bearing mismatch')
        if r['used']=='True':
            gate=max(cfg['residual_gate_degrees'],cfg['residual_sigma_multiplier']*float(r['effective_sigma_degrees']))
            if forward<=0 or distance>cfg['maximum_range_m'] or abs(residual)>gate+1e-7:raise ValueError('Accepted bearing outside gate')
        elif r['used']!='False':raise ValueError('Invalid observer membership flag')
        diagnostic[ekey][skey]=r
    for k,p in points.items():
        selected={(r['platform_id'],r['sequence_id']) for r in diagnostic[k].values() if r['used']=='True'}
        if selected!=set(zip(p['platform_ids'].split('|'),p['sequence_ids'].split('|'))):raise ValueError('Accepted membership mismatch')
        if len(diagnostic[k])!=int(p['input_platform_count']):raise ValueError('Missing diagnostic observation')
    tracks=rows(directory/'track_points.csv');track_rejected=rows(directory/'track_rejections.csv')
    owned_points=set();times={};last_accepted={};segments=defaultdict(set);track_counts=Counter()
    bin_width=round(cfg['track_bin_seconds']*1e9)
    for r in tracks+track_rejected:
        obj=key(r,OBJECT);ids=r['event_ids'].split('|')
        members=[points.get(obj[:2]+(e,)) for e in ids]
        if not all(members) or any(key(p,OBJECT)!=obj for p in members):raise ValueError('Track references wrong object/event')
        if len({int(p['time_ns'])//bin_width for p in members})!=1:raise ValueError('Invalid track time bin')
        if int(r['time_ns'])!=round(median(int(p['time_ns']) for p in members)):raise ValueError('Wrong track measurement time')
        for p in members:
            k=key(p,EVENT)
            if k in owned_points:raise ValueError('Reused localization in track bins')
            owned_points.add(k)
    if owned_points!=points.keys():raise ValueError('Incomplete track bin disposition')
    for r in tracks:
        obj=key(r,OBJECT);t=int(r['time_ns']);segment=r['segment_id']
        finite(r,['east_m','north_m','longitude','latitude','velocity_east_m_per_second','velocity_north_m_per_second',
                  'filter_covariance_ee_m2','filter_covariance_en_m2','filter_covariance_nn_m2'])
        if obj in times and t<=times[obj]:raise ValueError('Unordered track points')
        old=last_accepted.get(obj)
        if old and segment==old[1] and t-old[0]>cfg['track_segment_gap_seconds']*1e9:raise ValueError('Track bridges unsupported gap')
        if old and segment!=old[1] and segment in segments[obj]:raise ValueError('Reused track segment')
        times[obj]=t;last_accepted[obj]=(t,segment);segments[obj].add(segment);track_counts[obj]+=1
    summaries=rows(directory/'localization_objects.csv')
    for r in summaries:
        obj=key(r,OBJECT)
        if counts[obj]!=int(r['localized_point_count']) or track_counts[obj]!=int(r['track_point_count']) or len(segments[obj])!=int(r['track_segment_count']):
            raise ValueError('Object summary mismatch')
    if len(points)!=report['localized_point_count'] or len(rejected)!=report['rejected_event_count'] or len(tracks)!=report['track_point_count']:
        raise ValueError('Run summary mismatch')
    result={'passed':True,'output_hashes_match':True,'input_hashes_match':True,'all_events_disposed_exactly_once':True,
        'localized_events':len(points),'rejected_events':len(rejected),'track_points':len(tracks),
        'multi_platform_and_object_membership_exact':True,'forward_bearings_recomputed':True,
        'geographic_roundtrip_checked':True,'track_bins_cover_localizations_once':True,'track_gap_bound_checked':True,
        'accuracy_or_truth_evaluated':False}
    (directory/'localization_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.directory),indent=2))


if __name__=='__main__':main()
