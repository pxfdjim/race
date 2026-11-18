#!/usr/bin/env python3
"""Independently validate the published submission schemas against all inputs."""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from prepare_inputs import digest

PULSE_FIELDS=['scenario_id','data_version','pulse_id','local_cluster_id','object_id','target_class','target_state']
LINK_FIELDS=['scenario_id','data_version','object_id','track_id','link_status']
KEYS=['scenario_id','data_version','platform_id','pulse_id']
CLASSES={f'R{i:02d}' for i in range(1,9)}


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def verify(directory):
    directory=Path(directory);meta=json.loads((directory/'submission_export.json').read_text())
    scope=meta['scenario_id'],meta['data_version'];prefix='_'.join(scope)
    for name,info in meta['files'].items():
        if digest(directory/name)!=info['sha256']:raise ValueError(f'Changed submission output: {name}')
    for name,info in meta['input_files'].items():
        if digest(info['path'])!=info['sha256']:raise ValueError(f'Changed submission input: {name}')
    paths={k:Path(v['path']) for k,v in meta['input_files'].items()}
    b_tracks={tuple(r[k] for k in ['scenario_id','data_version','track_id']) for r in rows(paths['raw_B'])}
    expected_objects={tuple(r[k] for k in ['scenario_id','data_version','object_id']) for r in rows(paths['objects'])}
    links={};used=set()
    link_path=directory/f'{prefix}_object_track_links.csv'
    with link_path.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=LINK_FIELDS:raise ValueError('Object link schema/order mismatch')
        for r in reader:
            k=r['scenario_id'],r['data_version'],r['object_id']
            if k in links or k not in expected_objects or not k[2]:raise ValueError('Unknown, blank or duplicated object link')
            if r['link_status']=='matched':
                bk=k[:2]+(r['track_id'],)
                if not r['track_id'] or bk not in b_tracks or bk in used:raise ValueError('B track absent or linked to multiple objects')
                used.add(bk)
            elif r['link_status']=='no_match':
                if r['track_id']:raise ValueError('no_match has nonblank track ID')
            else:raise ValueError('Illegal official link_status')
            links[k]=r
    if links.keys()!=expected_objects:raise ValueError('Existing object omitted from submission')
    classes=Counter();states=Counter();object_counts=Counter();count=blank_local=blank_object=0;last_id=-1
    with paths['raw_A'].open(encoding='utf-8-sig',newline='') as af,paths['classes'].open(newline='') as cf,paths['states'].open(newline='') as sf,(directory/f'{prefix}_pulse_predictions.csv').open(newline='') as pf:
        a,c,s,p=csv.reader(af),csv.reader(cf),csv.reader(sf),csv.reader(pf)
        ah,ch,sh,ph=next(a),next(c),next(s),next(p)
        if ph!=PULSE_FIELDS:raise ValueError('Pulse submission schema/order mismatch')
        ai,ci,si=([h.index(n) for n in KEYS] for h in [ah,ch,sh])
        cg,co,cl=[ch.index(n) for n in ['local_cluster_id','object_id','target_class']]
        sg,so,st=[sh.index(n) for n in ['local_cluster_id','object_id','target_state']]
        for ar,cr,sr,pr in itertools.zip_longest(a,c,s,p):
            if any(r is None for r in [ar,cr,sr,pr]):raise ValueError('Submission lost or added pulses')
            if len(pr)!=7 or any(ar[i]!=cr[j] or ar[i]!=sr[k] for i,j,k in zip(ai,ci,si)):raise ValueError('Source row mismatch')
            if pr[:2]!=list(scope) or pr[:2]!=[ar[ai[0]],ar[ai[1]]] or pr[2]!=ar[ai[3]]:raise ValueError('Submission pulse keys/order mismatch')
            if not pr[2].isascii() or not pr[2].isdigit() or str(int(pr[2]))!=pr[2] or int(pr[2])<=last_id:raise ValueError('Duplicate/noncanonical global pulse ID')
            last_id=int(pr[2])
            if pr[3]!=cr[cg] or pr[3]!=sr[sg] or pr[4]!=cr[co] or pr[4]!=sr[so]:raise ValueError('Submission modified group/object memberships')
            if pr[5]!=cr[cl] or pr[5] not in CLASSES or pr[6]!=sr[st] or pr[6] not in ['0','1','2']:
                raise ValueError('Class/state values changed or illegal')
            if pr[4] and scope+(pr[4],) not in links:raise ValueError('Pulse object absent from link table')
            classes[pr[5]]+=1;states[pr[6]]+=1;blank_local+=not bool(pr[3]);blank_object+=not bool(pr[4]);count+=1
            if pr[4]:object_counts[pr[4]]+=1
            if count%3_000_000==0:print(json.dumps({'verified_submission_pulses':count}),flush=True)
    if count!=meta['input_output_pulses'] or dict(classes)!=meta['class_counts'] or dict(states)!=meta['state_counts'] or blank_local!=meta['blank_local_cluster_id_pulses'] or blank_object!=meta['blank_object_id_pulses']:
        raise ValueError('Submission summary mismatch')
    report={'passes_explicit_PDF_checks':True,'verified_pulses':count,'verified_objects':len(links),
        'matched_objects':len(used),'no_match_objects':len(links)-len(used),
        'exact_official_headers':True,'every_raw_pulse_exactly_once':True,'all_existing_objects_retained':True,
        'B_track_ids_exist_and_unique_per_object':True,'class_and_state_unchanged':True,
        'memberships_unchanged':True,'input_and_output_hashes_match':True,
        'blank_local_cluster_id_pulses':blank_local,'blank_object_id_pulses':blank_object,
        'blank_ID_portal_acceptance_not_specified_in_PDF':True,'platform_upload_tested':False,'uploaded':False,
        'validation_is_not_a_quality_score':True}
    (directory/'submission_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();verify(a.directory)


if __name__=='__main__':main()
