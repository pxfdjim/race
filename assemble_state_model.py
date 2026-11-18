#!/usr/bin/env python3
"""Package the trained sequence estimator and native-window fallback together."""
import argparse
import json
from pathlib import Path
from prepare_inputs import digest
from src.state_model import StateForest

ROOT=Path(__file__).resolve().parent


def run(window_dir,sequence_dir,output,balanced_dir=None):
    window_dir,sequence_dir,output=map(lambda p:Path(p).resolve(),(window_dir,sequence_dir,output))
    if output.exists():raise FileExistsError(output)
    wm=json.loads((window_dir/'target_state_model.json').read_text());sm=json.loads((sequence_dir/'target_state_model.json').read_text())
    wr=json.loads((window_dir/'training_report.json').read_text());sr=json.loads((sequence_dir/'training_report.json').read_text())
    for directory,r in [(window_dir,wr),(sequence_dir,sr)]:
        if digest(directory/'target_state_model.json')!=r['model_file']['sha256']:raise ValueError('State model changed')
    StateForest.from_dict(wm);StateForest.from_dict(sm)
    output.mkdir(parents=True)
    bundle={'format':'B04_target_state_bundle_v1','states':['0','1','2'],'window_model':wm,'sequence_model':sm,
            'inference_policy':'Supported PRI pulses use sequence morphology and temporal context; unknown PRI uses native observed-window fallback.'}
    balanced_report=None
    if balanced_dir is not None:
        balanced_dir=Path(balanced_dir)
        balanced_model=json.loads((balanced_dir/'target_state_model.json').read_text())
        balanced_report=json.loads((balanced_dir/'training_report.json').read_text())
        if balanced_model['trees']!=sm['trees'] or balanced_model['features']!=sm['features']:
            raise ValueError('Balanced profile must share the same fitted forest')
        bundle['balanced_config']=balanced_model['config']
    (output/'target_state_model.json').write_text(json.dumps(bundle,separators=(',',':'))+'\n')
    (output/'window_training_report.json').write_text(json.dumps(wr,ensure_ascii=False,indent=2)+'\n')
    (output/'sequence_training_report.json').write_text(json.dumps(sr,ensure_ascii=False,indent=2)+'\n')
    summary={'states':{'0':'search','1':'coarse_tracking','2':'fine_tracking'},
        'primary_model':'sequence_model','fallback_model':'window_model',
        'sequence_time_holdout':sr['time_holdout_evaluation'],
        'sequence_unseen_instance':sr['unseen_instance_evaluation'],
        'window_time_holdout':wr['time_holdout_evaluation'],
        'window_unseen_instance':wr['unseen_instance_evaluation'],
        'validation_boundary':'Window baseline diagnostics were inspected before sequence feature refinement. These are iterated within-scene development checks, not an untouched external test.',
        'no_test_labels_used':True,'labels_and_source_ID_values_excluded_from_predictor_features':True,
        'model_file':{'name':'target_state_model.json','sha256':digest(output/'target_state_model.json'),'bytes':(output/'target_state_model.json').stat().st_size},
        'algorithm_source_sha256':{'assemble_state_model.py':digest(ROOT/'assemble_state_model.py')},
        'source_reports':{'window':str(window_dir/'training_report.json'),'sequence':str(sequence_dir/'training_report.json')}}
    if balanced_report is not None:
        summary['balanced_profile']={'config':balanced_report['selected_config'],
            'time_holdout':balanced_report['time_holdout_evaluation'],'unseen_instance':balanced_report['unseen_instance_evaluation']}
        (output/'balanced_sequence_training_report.json').write_text(json.dumps(balanced_report,ensure_ascii=False,indent=2)+'\n')
    (output/'training_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'model':str(output/'target_state_model.json'),'sequence_time_holdout':summary['sequence_time_holdout']['with_temporal_context']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--window-dir',type=Path,default=ROOT/'models/target_state')
    p.add_argument('--sequence-dir',type=Path,default=ROOT/'models/target_state_sequence')
    p.add_argument('--output-dir',type=Path,default=ROOT/'models/target_state_final')
    p.add_argument('--balanced-dir',type=Path)
    a=p.parse_args();run(a.window_dir,a.sequence_dir,a.output_dir,a.balanced_dir)


if __name__=='__main__':main()
