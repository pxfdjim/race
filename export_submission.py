#!/usr/bin/env python3
"""Compose the two published CSV schemas from frozen class/state/match outputs.

No retraining, identity invention, unresolved-group merging or external upload.
Blank unresolved IDs are retained and explicitly audited: the PDF does not
define their portal validation behavior.
"""
import argparse
import csv
import itertools
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from prepare_inputs import DEFAULT_FILES,digest

ROOT=Path(__file__).resolve().parent
KEYS=['scenario_id','data_version','platform_id','pulse_id']
PULSE_FIELDS=['scenario_id','data_version','pulse_id','local_cluster_id','object_id','target_class','target_state']
LINK_FIELDS=['scenario_id','data_version','object_id','track_id','link_status']
CLASSES={f'R{i:02d}' for i in range(1,9)}


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def normalize_links(source,objects,b_tracks):
    normalized={};used={}
    for r in rows(source):
        k=tuple(r[n] for n in ['scenario_id','data_version','object_id'])
        if k in normalized or k not in objects:raise ValueError('Duplicate/unknown object link')
        status=r['link_status'];track=r['track_id']
        if status=='matched':
            bk=k[:2]+(track,)
            if not track or bk not in b_tracks:raise ValueError('Matched B track does not exist')
            if bk in used:raise ValueError(f'B track reused by multiple objects: {bk}; resolve trajectory matching first')
            used[bk]=k
        elif status in ['unmatched','no_match','ambiguous']:
            if track:raise ValueError('Unmatched object has a B track ID')
            status='no_match'
        else:raise ValueError(f'Unknown link status: {status}')
        normalized[k]=dict(zip(LINK_FIELDS,[*k,track,status]))
    if normalized.keys()!=objects:raise ValueError('Source links dropped an existing object')
    return normalized


def run(root,output,split='test'):
    root,output=Path(root).resolve(),Path(output).resolve();started=time.monotonic()
    if output.exists():raise FileExistsError(f'Choose a fresh submission directory: {output}')
    class_dir=root/f'output/classified/{split}';state_dir=root/f'output/state_predicted/{split}';match_dir=root/f'output/ab_matched_2d/{split}'
    paths={'raw_A':root.parent/DEFAULT_FILES[split][0],'raw_B':root.parent/DEFAULT_FILES[split][1],
        'classes':class_dir/'pulse_class_predictions.csv','states':state_dir/'pulse_state_predictions.csv',
        'links':match_dir/'object_track_links.csv',
        'objects':root/f'output/fused_b_assisted_guarded/{split}/fused_objects.csv'}
    class_meta=json.loads((class_dir/'classification_run.json').read_text())
    state_meta=json.loads((state_dir/'state_prediction_run.json').read_text())
    match_meta=json.loads((match_dir/'track_match_run.json').read_text())
    hashes={n:digest(p) for n,p in paths.items()}
    for n,m,file in [('classes',class_meta,'pulse_class_predictions.csv'),('states',state_meta,'pulse_state_predictions.csv'),('links',match_meta,'object_track_links.csv')]:
        if hashes[n]!=m['files'][file]['sha256']:raise ValueError(f'Changed prediction artifact: {n}')
    if hashes['raw_A']!=class_meta['input_files']['raw_A']['sha256'] or hashes['raw_A']!=state_meta['input_files']['raw_A']['sha256']:
        raise ValueError('Prediction/raw A source mismatch')
    if hashes['raw_B']!=match_meta['input_files']['B_source']['sha256']:raise ValueError('Matching/raw B source mismatch')
    before={n:(p.stat().st_size,p.stat().st_mtime_ns) for n,p in paths.items()}
    objects={tuple(r[n] for n in ['scenario_id','data_version','object_id']) for r in rows(paths['objects'])}
    b_tracks={tuple(r[n] for n in ['scenario_id','data_version','track_id']) for r in rows(paths['raw_B'])}
    links=normalize_links(paths['links'],objects,b_tracks)
    first=next(rows(paths['raw_A']),None)
    if first is None:raise ValueError('Empty A input')
    scope=first['scenario_id'],first['data_version']
    if scope[1]!=split or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) for v in scope):raise ValueError('Invalid submission scope')
    if any(k[:2]!=scope for k in objects):raise ValueError('Multiple scenes require separate exports')
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_name(f'.{output.name}.partial-{os.getpid()}');partial.mkdir()
    pulse_name=f'{scope[0]}_{scope[1]}_pulse_predictions.csv';link_name=f'{scope[0]}_{scope[1]}_object_track_links.csv'
    classes=Counter();states=Counter();object_counts=Counter();missing_local=missing_object=0;count=0;last_id=-1
    with paths['raw_A'].open(encoding='utf-8-sig',newline='') as af,paths['classes'].open(newline='') as cf,paths['states'].open(newline='') as sf,(partial/pulse_name).open('w',newline='') as out:
        a,c,s=csv.reader(af),csv.reader(cf),csv.reader(sf);ah,ch,sh=next(a),next(c),next(s)
        ai,ci,si=([h.index(n) for n in KEYS] for h in [ah,ch,sh])
        cg,co,cl=[ch.index(n) for n in ['local_cluster_id','object_id','target_class']]
        sg,so,st=[sh.index(n) for n in ['local_cluster_id','object_id','target_state']]
        w=csv.writer(out);w.writerow(PULSE_FIELDS)
        for ar,cr,sr in itertools.zip_longest(a,c,s):
            if ar is None or cr is None or sr is None:raise ValueError('Input/class/state pulse count mismatch')
            if any(ar[i]!=cr[j] or ar[i]!=sr[k] for i,j,k in zip(ai,ci,si)):raise ValueError('Pulse key/order mismatch')
            key=tuple(ar[i] for i in ai)
            if key[:2]!=scope:raise ValueError('Mixed submission scene/version')
            pulse_id=key[3]
            if not pulse_id.isascii() or not pulse_id.isdigit() or str(int(pulse_id))!=pulse_id or int(pulse_id)<=last_id:
                raise ValueError('Formal numeric pulse IDs must be globally unique and increasing in this input')
            last_id=int(pulse_id)
            if cr[cg]!=sr[sg] or cr[co]!=sr[so]:raise ValueError('Class/state source memberships disagree')
            local,oid,label,state=cr[cg],cr[co],cr[cl],sr[st]
            if label not in CLASSES or state not in ['0','1','2']:raise ValueError('Illegal class/state')
            if oid and scope+(oid,) not in links:raise ValueError('Pulse references object absent from links')
            w.writerow([*scope,pulse_id,local,oid,label,state])
            count+=1;classes[label]+=1;states[state]+=1;missing_local+=not bool(local);missing_object+=not bool(oid)
            if oid:object_counts[oid]+=1
            if count%2_000_000==0:print(json.dumps({'exported_submission_pulses':count}),flush=True)
    if count!=class_meta['pulse_export']['input_output_pulses'] or count!=state_meta['pulse_export']['input_output_pulses']:
        raise ValueError('Submission pulse coverage mismatch')
    for n,p in paths.items():
        if before[n]!=(p.stat().st_size,p.stat().st_mtime_ns):raise ValueError(f'Source changed during export: {n}')
    with (partial/link_name).open('w',newline='') as f:
        w=csv.DictWriter(f,LINK_FIELDS);w.writeheader();w.writerows(v for _,v in sorted(links.items()))
    quality_path=root/'models/target_state/training_report.json';quality=None
    if quality_path.exists():
        q=json.loads(quality_path.read_text())['sequence_time_holdout']['with_temporal_context']
        quality={'state_accuracy_in_iterated_scene_diagnostic':q['pulse_accuracy'],
                 'coarse_tracking_recall_in_that_diagnostic':q['per_state']['1']['recall'],
                 'not_test_accuracy':True}
    report={'scenario_id':scope[0],'data_version':scope[1],'input_output_pulses':count,'object_count':len(links),
        'matched_objects':sum(v['link_status']=='matched' for v in links.values()),
        'no_match_objects':sum(v['link_status']=='no_match' for v in links.values()),
        'class_counts':dict(classes),'state_counts':dict(states),'nonempty_object_pulse_counts':dict(object_counts),
        'blank_local_cluster_id_pulses':missing_local,'blank_object_id_pulses':missing_object,
        'unresolved_ids_preserved_without_new_clustering':True,
        'blank_ID_portal_acceptance_defined_in_PDF':False,'platform_upload_tested':False,'uploaded':False,
        'data_completeness_and_quality_are_separate':True,'quality_note':quality,
        'input_files':{n:{'path':str(p),'sha256':hashes[n]} for n,p in paths.items()},
        'algorithm_source_sha256':{'export_submission.py':digest(ROOT/'export_submission.py')},
        'files':{name:{'sha256':digest(partial/name),'bytes':(partial/name).stat().st_size} for name in [pulse_name,link_name]},
        'elapsed_seconds':round(time.monotonic()-started,2)}
    (partial/'submission_export.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');partial.rename(output)
    print(json.dumps({k:report[k] for k in ['input_output_pulses','object_count','matched_objects','no_match_objects','blank_local_cluster_id_pulses','blank_object_id_pulses','elapsed_seconds']},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--split',choices=['train','test'],default='test');p.add_argument('--output-dir',type=Path,default=ROOT/'提交结果')
    a=p.parse_args();run(a.pipeline_root,a.output_dir,a.split)


if __name__=='__main__':main()
