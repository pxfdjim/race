#!/usr/bin/env python3
"""Optionally rasterize saved SVG charts with an installed ImageMagick convert."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from prepare_inputs import digest


def export(directory):
    directory=Path(directory).resolve()
    command=shutil.which('convert')
    if not command:raise RuntimeError('PNG export requires ImageMagick; SVG and HTML do not require it.')
    svgs=sorted(directory.glob('*.svg'))
    if not svgs:raise ValueError('No localization SVG figures found')
    if any(p.with_suffix('.png').exists() for p in svgs):raise FileExistsError('PNG files already exist; choose a fresh result directory')
    report={'renderer':'ImageMagick convert','density':120,'files':[]}
    for svg in svgs:
        png=svg.with_suffix('.png')
        subprocess.run([command,'-background','white','-density','120',str(svg),str(png)],check=True)
        report['files'].append({'source':svg.name,'source_sha256':digest(svg),
                                'png':png.name,'png_sha256':digest(png),'bytes':png.stat().st_size})
    (directory/'png_export.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.directory),indent=2))


if __name__=='__main__':main()
