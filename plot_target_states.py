#!/usr/bin/env python3
"""Plot time-varying predicted states; leave unobserved intervals gray."""
import argparse
import csv
import html
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from prepare_inputs import digest

ROOT=Path(__file__).resolve().parent


def run(directory):
    directory=Path(directory).resolve()
    if (directory/'target_state_timeline.svg').exists() or (directory/'target_state_timeline.png').exists():raise FileExistsError('State figure already exists')
    meta=json.loads((directory/'state_prediction_run.json').read_text());period=meta['feature_config']['bin_seconds']
    objects=defaultdict(list)
    with (directory/'object_state_timeline.csv').open() as f:
        for r in csv.DictReader(f):objects[r['object_id']].append((int(r['bin_index']),r['target_state']))
    height=190+len(objects)*48;left,top,pw=150,120,1100
    end=max((i+1 for ps in objects.values() for i,_ in ps),default=1)*period
    colors={'0':'#2563eb','1':'#ea580c','2':'#dc2626'}
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}">',
        '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#172554">',
        f'<text x="700" y="32" text-anchor="middle" font-size="23">{html.escape(meta["split"])}: predicted target_state over observed time</text>',
        '<text x="700" y="60" text-anchor="middle" font-size="14">Predictions, not truth; gray intervals have no object-associated observations</text>']
    for i,(s,text) in enumerate([('0','0 search'),('1','1 coarse tracking'),('2','2 fine tracking')]):
        x=200+i*320;svg.append(f'<rect x="{x}" y="79" width="20" height="14" fill="{colors[s]}"/><text x="{x+29}" y="91" font-size="15">{text}</text>')
    for j,(oid,ps) in enumerate(sorted(objects.items())):
        y=top+j*48;svg.append(f'<text x="{left-15}" y="{y+20}" text-anchor="end" font-size="15">{html.escape(oid)}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{pw}" height="28" fill="#e2e8f0"/>')
        ps.sort();runs=[]
        for index,state in ps:
            if runs and state==runs[-1][2] and index==runs[-1][1]+1:runs[-1][1]=index
            else:runs.append([index,index,state])
        for start,stop,state in runs:
            x=left+start*period/end*pw;width=(stop-start+1)*period/end*pw
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="28" fill="{colors[state]}"/>')
    y=top+len(objects)*48
    for i in range(7):
        x=left+i*pw/6;svg.append(f'<text x="{x}" y="{y+10}" text-anchor="middle" font-size="14">{i*end/6:.1f}</text>')
    svg.append(f'<text x="700" y="{y+39}" text-anchor="middle" font-size="16">Seconds from scene origin</text></g></svg>')
    (directory/'target_state_timeline.svg').write_text('\n'.join(svg),encoding='utf-8')
    converter=shutil.which('convert')
    if converter:subprocess.run([converter,'-background','white','-density','120',str(directory/'target_state_timeline.svg'),str(directory/'target_state_timeline.png')],check=True)
    record={'predictions_only':True,'gaps_not_filled':True,'timeline_source_sha256':digest(directory/'object_state_timeline.csv'),
        'object_count':len(objects),'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in directory.glob('target_state_timeline.*')}}
    (directory/'state_plot.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'state_timeline_objects':len(objects)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();run(a.directory)


if __name__=='__main__':main()
