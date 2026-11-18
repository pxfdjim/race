#!/usr/bin/env python3
"""Color saved A trajectories by predicted detector class; no labels or B input."""
import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from prepare_inputs import digest
from src.localization import HorizontalFrame
from src.localization_plot import chart

ROOT=Path(__file__).resolve().parent


def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def run(root,class_root,output,split):
    root,class_root,output=map(lambda p:Path(p).resolve(),(root,class_root,output))
    if (output/'target_classes.svg').exists() or (output/'target_classes.png').exists():raise FileExistsError('Class figure already exists')
    source=class_root/split/'object_class_predictions.csv';classes={(r['scenario_id'],r['data_version'],r['object_id']):r['target_class'] for r in read(source)}
    directory=root/f'output/localized_2d/{split}';meta=json.loads((directory/'localization_run.json').read_text())
    if len(meta['frames'])!=1:raise ValueError('Class overview requires one scene')
    origin=meta['frames'][0];frame=HorizontalFrame(origin['origin_longitude'],origin['origin_latitude'])
    tracks=[]
    for r in read(directory/'track_points.csv'):
        key=tuple(r[k] for k in ['scenario_id','data_version','object_id'])
        if key not in classes:raise ValueError('A track has no class prediction')
        tracks.append({'object_id':classes[key],'segment_id':key[2]+'/'+r['segment_id'],
            'time_ns':int(r['time_ns']),'east_m':float(r['east_m']),'north_m':float(r['north_m'])})
    names=sorted(set(classes.values()))
    svg=chart(frame,tracks,tracks,{},names,f'{origin["scenario_id"]} / {split}: A tracks by predicted target class','track')
    output.mkdir(parents=True,exist_ok=True);(output/'target_classes.svg').write_text(svg,encoding='utf-8')
    converter=shutil.which('convert')
    if converter:subprocess.run([converter,'-background','white','-density','120',str(output/'target_classes.svg'),str(output/'target_classes.png')],check=True)
    report={'object_class_source':str(source),'object_class_source_sha256':digest(source),
        'A_track_source_sha256':digest(directory/'track_points.csv'),'object_class_counts':dict(Counter(classes.values())),
        'labels_are_predictions':True,'same_class_can_have_multiple_instances':True,'object_segments_kept_separate':True,
        'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in output.glob('target_classes.*')}}
    (output/'target_class_plot.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['object_class_counts']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--class-root',type=Path,default=ROOT/'output/classified')
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();run(a.pipeline_root,a.class_root,a.output_dir or a.class_root/a.split,a.split)


if __name__=='__main__':main()
