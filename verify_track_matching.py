#!/usr/bin/env python3
"""Independently check A--B links, interpolation, point distances and artifact hashes."""
import argparse
import bisect
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

from prepare_inputs import digest
from src.dataio import InputConfig,parse_timestamp
from src.localization import HorizontalFrame

SCOPE=['scenario_id','data_version']
OBJECT=SCOPE+['object_id']


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def key(r,names):return tuple(r[n] for n in names)


def close(a,b,tolerance=1e-6):
    if not math.isfinite(float(a)) or abs(float(a)-b)>tolerance:raise ValueError('Numeric comparison mismatch')


def verify(directory):
    directory=Path(directory);meta=json.loads((directory/'track_match_run.json').read_text());cfg=meta['config']
    for n,info in meta['files'].items():
        if digest(directory/n)!=info['sha256']:raise ValueError(f'Changed matching output: {n}')
    for n,info in meta['input_files'].items():
        if digest(info['path'])!=info['sha256']:raise ValueError(f'Changed matching input: {n}')
    if meta['height_used'] or meta['former_B_anchors_used_to_decide'] or meta['A_identity_truth_used'] or meta['B_identity_truth_used']:
        raise ValueError('Unexpected matching feature')
    frames={key(r,SCOPE):HorizontalFrame(r['origin_longitude'],r['origin_latitude']) for r in meta['frames']}
    units=InputConfig.from_dict(meta['input_units'])
    a={};objects={key(r,OBJECT) for r in read(meta['input_files']['localization_objects.csv']['path'])}
    for r in read(meta['input_files']['track_points.csv']['path']):
        k=key(r,OBJECT)+(int(r['time_ns']),)
        if k in a or k[:3] not in objects:raise ValueError('Invalid source A point')
        a[k]=r
    b=defaultdict(list)
    for r in read(meta['input_files']['B_source']['path']):
        scope=key(r,SCOPE);origin,kind=units.origin(*scope);stamp,_,clock=parse_timestamp(r['datetime'])
        if kind!=clock:raise ValueError('Clock mismatch')
        lon=float(r.get('target_longitude',r.get('target_lontitude','nan')))
        b[scope+(r['track_id'],)].append((stamp-origin,frames[scope].project(lon,float(r['target_latitude']))))
    times={};segments={}
    for k,ps in b.items():
        ps.sort();times[k]=[t for t,_ in ps]
        seg=0;chunks=[]
        for i,(t,_) in enumerate(ps):
            if i and t-ps[i-1][0]>cfg['maximum_b_gap_seconds']*1e9:seg+=1
            chunks.append(seg)
        segments[k]=chunks
    def sample(k,t):
        ts=times[k];ps=b[k];j=bisect.bisect_left(ts,t)
        if j<len(ts) and ts[j]==t:return ps[j][1],segments[k][j]
        if j==0 or j==len(ts) or ps[j][0]-ps[j-1][0]>cfg['maximum_b_gap_seconds']*1e9:return None
        f=(t-ps[j-1][0])/(ps[j][0]-ps[j-1][0])
        return tuple(x+f*(y-x) for x,y in zip(ps[j-1][1],ps[j][1])),segments[k][j]
    links={}
    for r in read(directory/'object_track_links.csv'):
        k=key(r,OBJECT)
        if k in links or k not in objects or r['link_status'] not in ['matched','unmatched']:raise ValueError('Invalid object link')
        if bool(r['track_id'])!=(r['link_status']=='matched'):raise ValueError('Status/link mismatch')
        if r['track_id'] and k[:2]+(r['track_id'],) not in b:raise ValueError('Unknown linked B')
        links[k]=r
    if links.keys()!=objects:raise ValueError('Missing A object disposition')
    pairs={};by_object=defaultdict(list)
    for r in read(directory/'pair_candidates.csv'):
        k=key(r,OBJECT);bk=k[:2]+(r['track_id'],);pk=k+(r['track_id'],)
        if pk in pairs or k not in objects or bk not in b:raise ValueError('Invalid candidate pair')
        paired=[]
        for ak,ar in sorted(a.items()):
            if ak[:3]!=k:continue
            sampled=sample(bk,ak[3])
            if sampled is not None:
                xy,seg=sampled;d=math.dist((float(ar['east_m']),float(ar['north_m'])),xy)
                paired.append((ak[3],ar['segment_id'],seg,d))
        distances=sorted(p[3] for p in paired);n=len(distances)
        if n!=int(r['comparison_count']):raise ValueError('Candidate comparison count mismatch')
        if n:
            close(r['median_distance_m'],median(distances))
            index=(n-1)*.9;i=int(index);p90=distances[i]+(index-i)*(distances[min(i+1,n-1)]-distances[i])
            close(r['p90_distance_m'],p90)
            close(r['rms_distance_m'],math.sqrt(sum(d*d for d in distances)/n))
            close(r['spatial_inlier_fraction'],sum(d<=cfg['inlier_distance_m'] for d in distances)/n)
        duration=0.
        for p,q in zip(paired,paired[1:]):
            dt=(q[0]-p[0])/1e9
            if p[1:3]==q[1:3] and 0<dt<=cfg['maximum_a_gap_seconds']:duration+=dt
        close(r['supported_duration_seconds'],duration)
        pairs[pk]=r;by_object[k].append(r)
    expected={(k+(bk[2],)) for k in objects for bk in b if k[:2]==bk[:2]}
    if pairs.keys()!=expected:raise ValueError('Incomplete A-B candidate matrix')
    details={key(r,OBJECT):r for r in read(directory/'object_match_details.csv')}
    if details.keys()!=objects:raise ValueError('Missing object details')
    for k,r in links.items():
        d=details[k]
        if r['track_id']!=d['track_id'] or (r['link_status']=='matched')!=(d['status']=='matched'):raise ValueError('Details/link mismatch')
        if not r['track_id']:continue
        p=pairs[k+(r['track_id'],)]
        if p['eligible']!='True' or p['temporal_eligible']!='True':raise ValueError('Selected gated-out pair')
        if (int(p['comparison_count'])<cfg['minimum_comparison_points'] or
            float(p['supported_duration_seconds'])<cfg['minimum_supported_duration_seconds'] or
            float(p['temporal_coverage'])<cfg['minimum_temporal_coverage'] or
            float(p['spatial_inlier_fraction'])<cfg['minimum_spatial_inlier_fraction'] or
            float(p['median_distance_m'])>cfg['maximum_median_distance_m'] or
            float(p['p90_distance_m'])>cfg['maximum_p90_distance_m']):raise ValueError('Accepted pair violates geometric support')
        if p['motion_used']=='True':
            if (int(p['motion_comparison_count'])<cfg['minimum_motion_comparisons'] or
                int(p['motion_comparison_count'])<cfg['minimum_motion_comparison_fraction']*int(p['comparison_count']) or
                float(p['median_heading_difference_degrees'])>cfg['maximum_median_heading_difference_degrees'] or
                float(p['median_speed_difference_m_per_second'])>cfg['maximum_median_speed_difference_m_per_second']):
                raise ValueError('Accepted pair violates motion support')
        ranked=sorted((v for v in by_object[k] if v['temporal_eligible']=='True'),key=lambda v:float(v['score']))
        if ranked[0]['track_id']!=r['track_id']:raise ValueError('Chosen link not best trajectory candidate')
        if len(ranked)>1:
            margin=float(ranked[1]['score'])-float(ranked[0]['score'])
            relative=margin/max(float(ranked[1]['score']),1e-12)
            if margin<cfg['minimum_score_margin'] or relative<cfg['minimum_relative_score_margin']:raise ValueError('Ambiguous match forced')
    seen=set();statuses=Counter()
    for r in read(directory/'matched_track_points.csv'):
        k=key(r,OBJECT)+(int(r['time_ns']),)
        if k in seen or k not in a:raise ValueError('Duplicate or new A point')
        seen.add(k);ar=a[k];d=details[k[:3]]
        close(r['a_east_m'],float(ar['east_m']));close(r['a_north_m'],float(ar['north_m']))
        if r['segment_id']!=ar['segment_id'] or r['track_id']!=d['track_id'] or r['object_match_status']!=d['status']:
            raise ValueError('Point identity changed')
        q=sample(k[:2]+(r['track_id'],),k[3]) if r['track_id'] else None
        if q is None:
            if any(r[n] for n in ['b_east_m','b_north_m','distance_m']):raise ValueError('Extrapolated or fabricated B point')
        else:
            close(r['b_east_m'],q[0][0]);close(r['b_north_m'],q[0][1])
            distance=math.dist((float(ar['east_m']),float(ar['north_m'])),q[0]);close(r['distance_m'],distance)
            expected_status='matched' if distance<=cfg['inlier_distance_m'] else 'spatial_outlier'
            if r['point_status']!=expected_status:raise ValueError('Point distance status mismatch')
        statuses[r['point_status']]+=1
    if seen!=a.keys() or dict(statuses)!=meta['point_status_counts']:raise ValueError('A point disposition/count mismatch')
    if Counter(d['status'] for d in details.values())!=meta['status_counts']:raise ValueError('Object status count mismatch')
    result={'passed':True,'input_and_output_hashes_match':True,'all_A_objects_and_points_retained':True,
        'complete_candidate_matrix':True,'B_interpolation_and_distances_independently_recomputed':True,
        'no_B_gap_bridging_or_extrapolation':True,'accepted_matches_pass_support_and_ambiguity_gates':True,
        'verified_candidate_pairs':len(pairs),'verified_A_points':len(a),'matched_objects':meta['matched_object_count'],
        'identity_accuracy_evaluated':False}
    (directory/'track_match_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.directory),indent=2))


if __name__=='__main__':main()
