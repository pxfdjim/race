#!/usr/bin/env python3
"""Sample moving observer positions without reading truth columns."""
import argparse
import bisect
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_inputs import digest
from src.dataio import time_to_ns


class ObserverPaths:
    def __init__(self, path):
        self.points, self.times = {}, {}
        with Path(path).open(newline='') as f:
            for r in csv.DictReader(f):
                key = tuple(r[k] for k in ('scenario_id', 'data_version', 'platform_id'))
                self.points.setdefault(key, []).append((int(r['toa_ns']), tuple(float(r[k]) for k in ('longitude', 'latitude', 'height'))))
        for key, points in self.points.items():
            points.sort()
            self.times[key] = [r[0] for r in points]

    def position(self, scope, time):
        if scope not in self.points:
            raise ValueError(f'Observer path missing: {scope}')
        points, times = self.points[scope], self.times[scope]
        j = bisect.bisect_left(times, time)
        if j < len(times) and times[j] == time:
            return points[j][1]
        if j == 0 or j == len(times):
            raise ValueError('Sequence time lies outside observer observations')
        left, right = points[j-1], points[j]
        f = (time-left[0])/(right[0]-left[0])
        return tuple(a+f*(b-a) for a,b in zip(left[1],right[1]))


def sample_positions(source, output, unit='us', period_ns=100_000_000):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if period_ns <= 0:
        raise ValueError('Sampling period must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    last, written, counts, rows = {}, {}, Counter(), 0
    header = ['scenario_id','data_version','platform_id','toa_ns','longitude','latitude','height']
    with source.open(encoding='utf-8-sig',newline='') as f, output.with_suffix('.partial').open('w',newline='') as out:
        reader = csv.reader(f)
        columns = next(reader)
        indices = [columns.index(k) for k in ('scenario_id','data_version','platform_id','toa','sensor_latitude','sensor_height')]
        lon = columns.index('sensor_longitude' if 'sensor_longitude' in columns else 'sensor_lontitude')
        writer = csv.writer(out)
        writer.writerow(header)
        for r in reader:
            rows += 1
            scope = tuple(r[i] for i in indices[:3])
            time = time_to_ns(r[indices[3]],unit)
            if scope in last and time < last[scope][3]:
                raise ValueError('Observer input not time ordered')
            row = (*scope,time,r[lon],r[indices[4]],r[indices[5]])
            last[scope] = row
            if scope not in written or time-written[scope] >= period_ns:
                writer.writerow(row)
                counts[scope] += 1
                written[scope] = time
        for scope,row in sorted(last.items()):
            if written[scope] != row[3]:
                writer.writerow(row)
                counts[scope] += 1
    output.with_suffix('.partial').replace(output)
    report = {'source':str(source.resolve()),'source_sha256':digest(source),'input_rows':rows,
        'truth_columns_used':[],'sampling_period_ns':period_ns,
        'points':[dict(zip(header[:3],k),points=v) for k,v in sorted(counts.items())]}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--toa-unit',choices=('us','ns','ms','s'),default='us')
    a=p.parse_args()
    print(json.dumps(sample_positions(a.a_file,a.output,a.toa_unit),indent=2))


if __name__=='__main__':
    main()
