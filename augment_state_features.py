#!/usr/bin/env python3
"""Add observed waveform/PA calibration covariates, never identity/state labels."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from prepare_inputs import digest

ROOT=Path(__file__).resolve().parent
EXTRA=['observed_RF_ghz','observed_PW_us','observed_log_PRI_us','observed_PA_dbm']


def rows(path):
    with Path(path).open(newline='') as f:yield from csv.DictReader(f)


def run(root,source_root,output_root,split):
    root,source_root,output_root=map(lambda p:Path(p).resolve(),(root,source_root,output_root))
    source=source_root/split;output=output_root/split
    if output.exists():raise FileExistsError(output)
    seq=root/f'output/pri/{split}/pri_sequences.csv';keys=['scenario_id','data_version','platform_id','sequence_id']
    measurements={tuple(r[k] for k in keys):[float(r['rf_mean'])/1000.,float(r['pw_mean']),math.log(float(r['pri_center_us'])),float(r['pa_mean'])] for r in rows(seq)}
    meta=json.loads((source/'state_features_run.json').read_text())
    output.mkdir(parents=True)
    with (source/'window_features.csv').open() as f,(output/'window_features.csv').open('w',newline='') as out:
        reader=csv.DictReader(f);writer=csv.DictWriter(out,reader.fieldnames+EXTRA);writer.writeheader()
        for r in reader:
            k=tuple(r[n] for n in keys)
            writer.writerow(r|dict(zip(EXTRA,measurements[k])))
    if (source/'window_truth.csv').exists():shutil.copy2(source/'window_truth.csv',output/'window_truth.csv')
    meta['features']+=EXTRA;meta['calibration_features']=EXTRA
    meta['absolute_PA_RF_PW_PRI_DOA_used_as_features']='RF/PW/PRI/PA only; DOA and all IDs remain excluded'
    meta['calibration_note']='Waveform and amplitude alone do not define a state; these covariates calibrate the temporal/context classifier and can carry scenario correlations.'
    meta['algorithm_source_sha256']={'augment_state_features.py':digest(ROOT/'augment_state_features.py')}
    meta['files']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in output.iterdir()}
    (output/'state_features_run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'split':split,'calibrated_features':len(meta['features'])}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pipeline-root',type=Path,default=ROOT)
    p.add_argument('--source-root',type=Path,default=ROOT/'output/sequence_state_features')
    p.add_argument('--output-root',type=Path,default=ROOT/'output/calibrated_sequence_state_features')
    p.add_argument('--split',choices=['train','test','both'],default='both');a=p.parse_args()
    for split in (['train','test'] if a.split=='both' else [a.split]):run(a.pipeline_root,a.source_root,a.output_root,split)


if __name__=='__main__':main()
