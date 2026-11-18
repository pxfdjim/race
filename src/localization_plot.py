"""Dependency-free, equal-scale SVG figures and self-contained HTML preview."""
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

COLORS = ['#2563eb','#dc2626','#059669','#9333ea','#ea580c','#0891b2','#db2777','#4f46e5',
          '#65a30d','#b45309','#0f766e','#7c3aed','#be123c','#0369a1','#a16207','#475569','#c026d3']


def esc(s):return html.escape(str(s), quote=True)


def read_rows(path):
    with Path(path).open(newline='') as f:
        yield from csv.DictReader(f)


def chart(frame, points, tracks, paths, names, title, kind, example=None):
    width, height = 1400, 1060
    left, top, pw, ph = 100, 120, 1200, 770
    colors = {n:COLORS[i%len(COLORS)] for i,n in enumerate(names)}
    values = [(p['east_m'],p['north_m']) for p in points]+[xy for v in paths.values() for xy in v]
    if example:
        values = [o.xy for o in example['selected']]+[(example['east_m'],example['north_m'])]
    if not values:values=[(-1000.,-1000.),(1000.,1000.)]
    xs,ys=zip(*values)
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
    scale=max(1.,(max(xs)-min(xs))/pw,(max(ys)-min(ys))/ph)*1.12
    xlo,xhi=cx-pw*scale/2,cx+pw*scale/2
    ylo,yhi=cy-ph*scale/2,cy+ph*scale/2
    def xy(x,y):return left+(x-xlo)/scale,top+(yhi-y)/scale
    def coords(seq):return ' '.join(f'{x:.2f},{y:.2f}' for x,y in (xy(*p) for p in seq))
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="sans-serif" fill="#14213d">',
         f'<text x="{width/2}" y="36" text-anchor="middle" font-size="23">{esc(title)}</text>',
         '<text x="700" y="65" text-anchor="middle" font-size="15">A bearings + moving observer positions; 2D only; equal East/North scale</text>',
         f'<text x="700" y="88" text-anchor="middle" font-size="14">Origin WGS84: {frame.longitude:.6f}, {frame.latitude:.6f}; B-assisted upstream identities</text>']
    for i in range(7):
        x=left+i*pw/6;y=top+i*ph/6
        out.extend([f'<path d="M{x},{top} V{top+ph}" stroke="#e2e8f0"/>',
                    f'<path d="M{left},{y} H{left+pw}" stroke="#e2e8f0"/>',
                    f'<text x="{x}" y="{top+ph+25}" text-anchor="middle" font-size="14">{(xlo+i*(xhi-xlo)/6)/1000:.1f}</text>',
                    f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="14">{(yhi-i*(yhi-ylo)/6)/1000:.1f}</text>'])
    out += [f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8"/>',
            f'<text x="700" y="{top+ph+52}" text-anchor="middle" font-size="17">East (km)</text>',
            f'<text x="26" y="{top+ph/2}" text-anchor="middle" font-size="17" transform="rotate(-90 26 {top+ph/2})">North (km)</text>']
    if example:
        p=example
        for o in p['selected']:
            dx,dy=p['east_m']-o.xy[0],p['north_m']-o.xy[1]
            length=math.hypot(dx,dy)*1.12
            end=(o.xy[0]+o.direction[0]*length,o.xy[1]+o.direction[1]*length)
            out.append(f'<polyline points="{coords([o.xy,end])}" fill="none" stroke="#475569" stroke-width="1.7"/>')
            xx,yy=xy(*o.xy)
            out.append(f'<rect x="{xx-4}" y="{yy-4}" width="8" height="8" fill="#111827"/>')
            out.append(f'<text x="{xx+8}" y="{yy-9}" font-size="15">{esc(o.platform)} t={o.time/1e9:.4f}s</text>')
        xx,yy=xy(p['east_m'],p['north_m'])
        out.append(f'<circle cx="{xx}" cy="{yy}" r="6" fill="#dc2626"/><text x="{xx+12}" y="{yy-12}" font-size="16">{esc(p["object_id"])} / {esc(p["event_id"])}</text>')
    else:
        for platform, positions in sorted(paths.items()):
            if not positions:continue
            out.append(f'<polyline points="{coords(positions)}" fill="none" stroke="#64748b" stroke-width="1.2" stroke-dasharray="5 4"/>')
            xx,yy=xy(*positions[0])
            out.append(f'<rect x="{xx-4}" y="{yy-4}" width="8" height="8" fill="#334155"/><text x="{xx+8}" y="{yy-9}" font-size="14">{esc(platform)}</text>')
        by_object=defaultdict(list)
        for p in (points if kind=='raw' else tracks):by_object[p['object_id']].append(p)
        for name in names:
            color=colors[name];rows=by_object[name]
            out.append(f'<g data-object="{esc(name)}" fill="{color}" stroke="{color}">')
            if kind=='raw':
                for p in rows:
                    xx,yy=xy(p['east_m'],p['north_m'])
                    out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="1.5" stroke="none" opacity="0.3"/>')
            else:
                segments=defaultdict(list)
                for p in rows:segments[p['segment_id']].append(p)
                for ps in segments.values():
                    ps.sort(key=lambda p:p['time_ns'])
                    out.append(f'<polyline points="{coords((p["east_m"],p["north_m"]) for p in ps)}" fill="none" stroke-width="2.2"/>')
                    for p in ps:
                        xx,yy=xy(p['east_m'],p['north_m'])
                        out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="1.7" stroke="none"/>')
            out.append('</g>')
        for i,n in enumerate(names):
            xx,yy=left+(i%5)*245,970+(i//5)*23
            out.append(f'<text x="{xx}" y="{yy}" fill="{colors[n]}" font-size="15">{esc(n)} ({len(by_object[n]):,} points)</text>')
    out += ['</g></svg>']
    return '\n'.join(out)


def render(output, frames, points, tracks, summaries, position_file, report):
    paths=defaultdict(lambda:defaultdict(list))
    for r in read_rows(position_file):
        scope=r['scenario_id'],r['data_version']
        paths[scope][r['platform_id']].append(frames[scope].project(float(r['longitude']),float(r['latitude'])))
    content=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B04 二维交叉定位</title>',
        '<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#172554}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #cbd5e1;text-align:left}label{display:inline-block;margin:6px 16px 6px 0}p{line-height:1.7}.note{background:#eff6ff;padding:14px}a{color:#1d4ed8}</style>',
        '<h1>B04 二维交叉定位与分段航迹</h1>',
        '<p class="note">位置由 A 的 DOA 与观测时刻的平台经纬度计算，忽略全部高度。沿用 B 辅助保护版对象归属；B 坐标没有参与本步定位。图中不是目标真值，也不是独立测试精度验证。两平台相交，多平台联合加权求解，不平均六组交点。</p>',
        f'<p>输入 {report["input_event_count"]:,} 个事件，其中 {report["identified_event_count"]:,} 个有对象身份；生成 {len(points):,} 个定位点、{len(tracks):,} 个滤波航迹点。协方差为模型误差参考，不是已验证的置信区间。</p>']
    for scope,frame in sorted(frames.items()):
        ps=[p for p in points if (p['scenario_id'],p['data_version'])==scope]
        ts=[p for p in tracks if (p['scenario_id'],p['data_version'])==scope]
        names=sorted({p['object_id'] for p in ps})
        suffix='' if len(frames)==1 else '_'+str(len(content))
        raw=chart(frame,ps,ts,paths[scope],names,f'{scope[0]} / {scope[1]}: cross-bearing positions','raw')
        smooth=chart(frame,ps,ts,paths[scope],names,f'{scope[0]} / {scope[1]}: segmented A tracks','track')
        (output/f'cross_localization{suffix}.svg').write_text(raw)
        (output/f'tracks_2d{suffix}.svg').write_text(smooth)
        content.append('<div>'+''.join(f'<label><input type="checkbox" checked data-toggle="{esc(n)}">{esc(n)}</label>' for n in names)+'</div>')
        content += ['<h2>1. 交叉定位点</h2><p>每点对应一个具有一致对象身份的 A 跨平台事件；灰色虚线为观测平台移动路径。</p>',raw,
                    '<h2>2. 分段航迹</h2><p>每0.5秒对定位点取稳健中值，再用二维匀速卡尔曼模型滤波。只输出接受观测时刻的点；距上一接受点超过5秒分段，不填补缺测段。不同段仍沿用同一对象编号。</p>',smooth]
        if ps:
            sample=min(ps,key=lambda p:(-p['used_platform_count'],p['major_error_proxy_m'],p['event_id']))
            keys=set(zip(sample['platform_ids'].split('|'),sample['sequence_ids'].split('|')))
            selected=[]
            # Read only the saved A bearing diagnostic rows for this example.
            from .localization import Bearing
            for r in read_rows(output/'localization_observations.csv'):
                if (r['scenario_id'],r['data_version'],r['event_id']) != scope+(sample['event_id'],):continue
                if (r['platform_id'],r['sequence_id']) not in keys:continue
                lon,lat,doa=float(r['observer_longitude']),float(r['observer_latitude']),float(r['doa_degrees'])
                t=int(r['time_ns'])
                selected.append(Bearing(scope+(r['platform_id'],r['sequence_id']),t,t,t,
                    frame.project(lon,lat),frame.direction(lon,lat,doa),float(r['effective_sigma_degrees'])))
            example=chart(frame,ps,ts,{},names,f'{scope[0]} / {scope[1]}: {len(selected)}-platform joint fit',
                          'example',example=sample|{'selected':selected})
            (output/f'bearing_example{suffix}.svg').write_text(example)
            content += ['<h2>3. 一次多平台交叉定位示例</h2><p>使用各平台自己的观测时刻及位置。图中的射线仅用于展示，交点通过联合角度残差最小化计算。</p>',example]
    fields=['object_id','identified_event_count','localized_point_count','track_point_count','track_segment_count','track_rejected_bin_count','start_seconds','end_seconds']
    content += ['<h2>对象与航迹统计</h2><table><thead><tr>'+''.join(f'<th>{esc(k)}</th>' for k in fields)+'</tr></thead><tbody>']
    for r in summaries:content.append('<tr>'+''.join(f'<td>{esc(round(r[k],3) if isinstance(r[k],float) else r[k])}</td>' for k in fields)+'</tr>')
    content += ['</tbody></table><p>仅单平台、身份未确定、退化几何与滤波拒绝情况保留在审计 CSV 中。已有 A 事件之外的补关联序列，本步没有新增配对。</p>',
        '<p><a href="localized_points.csv">定位点 CSV</a> · <a href="track_points.csv">航迹点 CSV</a> · <a href="rejected_localizations.csv">定位拒绝原因</a> · <a href="localization_run.json">运行参数与统计</a></p>',
        '<script>document.querySelectorAll("input[data-toggle]").forEach(x=>x.addEventListener("change",()=>{document.querySelectorAll("g[data-object]").forEach(g=>{if(g.getAttribute("data-object")===x.getAttribute("data-toggle"))g.style.display=x.checked?"":"none"})}));</script></html>']
    (output/'track_preview.html').write_text('\n'.join(content))
