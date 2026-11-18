"""Standalone SVG/PNG overlay of A tracks, selected B tracks and all B candidates."""
import html
import math
import shutil
import subprocess
from collections import defaultdict
from .localization_plot import COLORS


def render(output,frames,objects,tracks,decisions,pairs):
    for index,(scope,frame) in enumerate(sorted(frames.items())):
        current={k:ps for k,ps in objects.items() if k[:2]==scope}
        bs={k:b for k,b in tracks.items() if k[:2]==scope}
        values=[p.xy for ps in current.values() for p in ps]+[p.xy for b in bs.values() for p in b.points]
        if not values:values=[(-1000.,-1000.),(1000.,1000.)]
        xs,ys=zip(*values);cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
        left,top,pw,ph=100,135,1200,740
        scale=max(1.,(max(xs)-min(xs))/pw,(max(ys)-min(ys))/ph)*1.14
        xlo,yhi=cx-pw*scale/2,cy+ph*scale/2
        def xy(p):return left+(p[0]-xlo)/scale,top+(yhi-p[1])/scale
        def coords(ps):return ' '.join(f'{x:.2f},{y:.2f}' for x,y in (xy(p.xy) for p in ps))
        def esc(s):return html.escape(str(s),quote=True)
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1100" viewBox="0 0 1400 1100" role="img" aria-label="A B trajectory matching overlay">',
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#14213d">',
            f'<text x="700" y="35" text-anchor="middle" font-size="23">{esc(scope[0])} / {esc(scope[1])}: A-B trajectory matching</text>',
            '<text x="700" y="64" text-anchor="middle" font-size="16">Solid: A estimates; same-color dashed: selected B; light gray: all B candidates</text>',
            '<text x="700" y="89" text-anchor="middle" font-size="14">Unmatched A tracks are retained. Shared colors indicate the final link, not truth labels.</text>',
            f'<text x="700" y="112" text-anchor="middle" font-size="14">2D / equal scale / origin WGS84 {frame.longitude:.6f}, {frame.latitude:.6f}; upstream B-assisted identities</text>']
        for i in range(7):
            x,y=left+i*pw/6,top+i*ph/6
            svg.extend([f'<path d="M{x},{top} V{top+ph} M{left},{y} H{left+pw}" fill="none" stroke="#e2e8f0"/>',
                f'<text x="{x}" y="{top+ph+25}" text-anchor="middle" font-size="14">{(xlo+i*pw*scale/6)/1000:.1f}</text>',
                f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="14">{(yhi-i*ph*scale/6)/1000:.1f}</text>'])
        svg.extend([f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8"/>',
                    '<text x="700" y="932" text-anchor="middle" font-size="17">East (km)</text>',
                    '<text x="27" y="510" text-anchor="middle" font-size="17" transform="rotate(-90 27 510)">North (km)</text>'])
        def draw(ps,color,width,dash='',alpha=1.,dots=False):
            segments=defaultdict(list)
            for p in ps:segments[p.segment].append(p)
            for segment in segments.values():
                dash_attribute=f' stroke-dasharray="{dash}"' if dash else ''
                svg.append(f'<polyline points="{coords(segment)}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attribute} opacity="{alpha}"/>')
            if dots:
                for p in ps:
                    x,y=xy(p.xy);svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" fill="{color}" opacity="{alpha}"/>')
        for b in bs.values():draw(b.points,'#94a3b8',1.2,alpha=.3)
        for i,(key,ps) in enumerate(sorted(current.items())):
            color=COLORS[i%len(COLORS)];d=decisions[key]
            if d['track_id']:draw(bs[scope+(d['track_id'],)].points,color,2.,'7 5',.8)
            draw(ps,color,2.2,dots=True)
            text=f'{key[2]} -> B{d["track_id"]}' if d['track_id'] else f'{key[2]}: {d["status"]}'
            if d['track_id']:text+=f' ({pairs[key+(d["track_id"],)]["median_distance_m"]:.0f} m)'
            x,y=left+(i%4)*305,973+(i//4)*23
            svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="14">{esc(text)}</text>')
        svg.append('</g></svg>')
        name='matching_overlay'+('' if len(frames)==1 else f'_{index+1}')
        source=output/(name+'.svg');source.write_text('\n'.join(svg),encoding='utf-8')
        converter=shutil.which('convert')
        if converter:subprocess.run([converter,'-background','white','-density','120',str(source),str(output/(name+'.png'))],check=True)
