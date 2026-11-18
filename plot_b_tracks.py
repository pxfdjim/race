#!/usr/bin/env python3
"""Plot every supplied B track in the same horizontal frame as the A estimates."""
import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from prepare_inputs import DEFAULT_FILES, digest
from src.dataio import InputConfig, parse_timestamp
from src.localization import HorizontalFrame
from src.localization_plot import COLORS

ROOT=Path(__file__).resolve().parent


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        yield from csv.DictReader(f)


def run(root, split, output):
    root, output=Path(root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    names=['b_tracks_2d.svg','b_tracks_2d.png','b_tracks_plot.json']
    if any((output/n).exists() for n in names):raise FileExistsError('B plot already exists; choose a new output directory')
    renderer=shutil.which('convert')
    if renderer is None:raise RuntimeError('PNG export requires ImageMagick convert')
    source=root.parent/DEFAULT_FILES[split][1]
    reference=root/f'output/localized_2d/{split}/localization_run.json'
    meta=json.loads(reference.read_text())
    if len(meta['frames'])!=1:raise ValueError('Select a single scene for this overview plot')
    origin=meta['frames'][0]
    scope=origin['scenario_id'],origin['data_version']
    frame=HorizontalFrame(origin['origin_longitude'],origin['origin_latitude'])
    units=InputConfig.from_dict(json.loads((root/'config/input_units.json').read_text()))
    scene_origin,kind=units.origin(*scope)
    tracks=defaultdict(list);seen=set()
    for r in read_rows(source):
        if (r['scenario_id'],r['data_version'])!=scope:raise ValueError('Mixed B scene/version')
        t,_,stamp_kind=parse_timestamp(r['datetime'])
        key=r['track_id'],t
        lon=float(r.get('target_longitude',r.get('target_lontitude','nan')))
        lat=float(r['target_latitude'])
        if (not r['track_id'] or key in seen or stamp_kind!=kind or t<scene_origin or
            not all(math.isfinite(v) for v in (lon,lat)) or not -180<=lon<=180 or not -89<=lat<=89):
            raise ValueError('Invalid or duplicate B observation')
        seen.add(key)
        # No height, target_id, class, type, smoothing or association selection.
        tracks[r['track_id']].append((t-scene_origin,frame.project(lon,lat)))
    for ps in tracks.values():ps.sort()
    platforms=defaultdict(list)
    position_file=root/f'output/group_inputs/{split}_platform_positions.csv'
    for r in read_rows(position_file):
        if (r['scenario_id'],r['data_version'])!=scope:continue
        platforms[r['platform_id']].append((int(r['toa_ns']),frame.project(float(r['longitude']),float(r['latitude']))))
    for ps in platforms.values():ps.sort()
    all_xy=[xy for ps in list(tracks.values())+list(platforms.values()) for _,xy in ps]
    if not tracks:raise ValueError('No B observations')
    left,top,pw,ph=100,135,1200,760
    xs,ys=zip(*all_xy);cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
    scale=max(1.,(max(xs)-min(xs))/pw,(max(ys)-min(ys))/ph)*1.12
    xlo,yhi=cx-pw*scale/2,cy+ph*scale/2
    def xy(p):return left+(p[0]-xlo)/scale,top+(yhi-p[1])/scale
    def coords(ps):return ' '.join(f'{x:.2f},{y:.2f}' for x,y in (xy(p) for p in ps))
    def esc(s):return html.escape(str(s),quote=True)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1080" viewBox="0 0 1400 1080" role="img" aria-label="All supplied B tracks">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#14213d">',
         f'<text x="700" y="35" text-anchor="middle" font-size="24">{esc(scope[0])} / {split}: all supplied B tracks</text>',
         f'<text x="700" y="64" text-anchor="middle" font-size="16">{len(tracks)} track IDs / {len(seen):,} raw observations; 2D; equal East/North scale</text>',
         f'<text x="700" y="90" text-anchor="middle" font-size="14">Same A origin WGS84: {frame.longitude:.6f}, {frame.latitude:.6f}; all heights ignored</text>',
         '<text x="700" y="113" text-anchor="middle" font-size="14">B observations may include false or anomalous tracks; these are not independent position ground truth</text>']
    for i in range(7):
        px=left+i*pw/6;py=top+i*ph/6
        svg += [f'<path d="M{px},{top} V{top+ph} M{left},{py} H{left+pw}" fill="none" stroke="#e2e8f0"/>',
                f'<text x="{px}" y="{top+ph+25}" text-anchor="middle" font-size="14">{(xlo+i*pw*scale/6)/1000:.1f}</text>',
                f'<text x="{left-12}" y="{py+5}" text-anchor="end" font-size="14">{(yhi-i*ph*scale/6)/1000:.1f}</text>']
    svg += [f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8"/>',
            '<text x="700" y="948" text-anchor="middle" font-size="17">East (km)</text>',
            '<text x="27" y="515" text-anchor="middle" font-size="17" transform="rotate(-90 27 515)">North (km)</text>']
    for platform,ps in sorted(platforms.items()):
        svg.append(f'<polyline points="{coords(p for _,p in ps)}" fill="none" stroke="#64748b" stroke-width="1.2" stroke-dasharray="5 4"/>')
        x,y=xy(ps[0][1])
        svg.append(f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="#334155"/><text x="{x+8}" y="{y-9}" font-size="14">{esc(platform)}</text>')
    track_ids=sorted(tracks,key=lambda t:(0,int(t)) if t.isascii() and t.isdigit() else (1,t))
    segment_counts={}
    for i,tid in enumerate(track_ids):
        ps=tracks[tid];color=COLORS[i%len(COLORS)];segments=[]
        for j,(t,p) in enumerate(ps):
            if j==0 or t-ps[j-1][0]>3_000_000_000:segments.append([])
            segments[-1].append(p)
        segment_counts[tid]=len(segments)
        svg.append(f'<g data-track="{esc(tid)}" stroke="{color}" fill="{color}">')
        for segment in segments:
            svg.append(f'<polyline points="{coords(segment)}" fill="none" stroke-width="1.1" opacity="0.45"/>')
        for _,p in ps:
            x,y=xy(p);svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" stroke="none" opacity="0.75"/>')
        svg.append('</g>')
        lx,ly=left+(i%5)*245,982+(i//5)*23
        svg.append(f'<text x="{lx}" y="{ly}" fill="{color}" font-size="15">B{esc(tid)} ({len(ps):,} points)</text>')
    svg.append('</g></svg>')
    content='\n'.join(svg);ET.fromstring(content)
    (output/names[0]).write_text(content,encoding='utf-8')
    subprocess.run([renderer,'-background','white','-density','120',str(output/names[0]),str(output/names[1])],check=True)
    report={'source':str(source),'source_sha256':digest(source),'reference_run_sha256':digest(reference),
        'split':split,'frame':origin,'dimension':2,'height_used':False,'all_B_tracks_included':True,
        'smoothing_applied':False,'track_count':len(tracks),'point_count':len(seen),
        'track_point_counts':{t:len(tracks[t]) for t in track_ids},'track_segment_counts':segment_counts,
        'line_maximum_gap_seconds':3,'axes_km':{'east_min':xlo/1000,'east_max':(xlo+pw*scale)/1000,
                                              'north_min':(yhi-ph*scale)/1000,'north_max':yhi/1000},
        'note':'B is the supplied sensor trajectory set, potentially containing false/anomalous tracks; not independent position ground truth.',
        'files':{n:{'sha256':digest(output/n),'bytes':(output/n).stat().st_size} for n in names[:2]}}
    (output/names[2]).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'split':split,'tracks':len(tracks),'points':len(seen),'png':str(output/names[1])},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();run(a.pipeline_root,a.split,a.output_dir or ROOT/f'output/localized_2d/{a.split}')


if __name__=='__main__':main()
