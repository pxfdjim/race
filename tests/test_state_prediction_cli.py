import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from prepare_inputs import digest
from predict_target_state import run
from verify_target_state import verify
from src.state_model import StateForest,StateModelConfig


def fixture(root):
    x=[(float(s),) for s in range(3) for _ in range(20)];y=[s for s in range(3) for _ in range(20)]
    cfg=StateModelConfig(trees=3,maximum_depth=3,samples_per_tree=90,features_per_split=1,minimum_leaf_samples=2)
    m=StateForest(cfg,['observable']).fit(x,y,[1]*len(y)).to_dict()
    model=root/'model.json';model.write_text(json.dumps({'format':'B04_target_state_bundle_v1','window_model':m,'sequence_model':m}))
    for kind in ['windows','sequences']:
        p=root/kind/'test';p.mkdir(parents=True)
        fields=['scenario_id','data_version','platform_id','source_kind','source_id','bin_index','pulse_count','first_time_ns','last_time_ns','sequence_id','observable']
        rows=[]
        for s in range(3):
            rows.append(dict(zip(fields,['s','test','p','object',('O' if kind=='windows' else f'O~S{s}'),s,20,s*50_000_000,s*50_000_000+10_000_000,f'S{s}',s])))
        with (p/'window_features.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
        meta={'features':['observable'],'labels_used_in_features':False,'config':{'bin_seconds':.05},
              'files':{'window_features.csv':{'sha256':digest(p/'window_features.csv')}}}
        (p/'state_features_run.json').write_text(json.dumps(meta))
    return model


class StatePredictionCliTests(unittest.TestCase):
    def test_records_only_pipeline_and_independent_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                meta=run(root,root/'windows',root/'sequences',model,root/'out','test',True)
                result=verify(root/'out')
            self.assertIsNone(meta['pulse_export']);self.assertEqual(meta['sequence_count'],3)
            self.assertTrue(result['passed'])
            with self.assertRaises(FileExistsError):run(root,root/'windows',root/'sequences',model,root/'out','test',True)

    def test_changed_feature_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=fixture(root)
            with (root/'windows/test/window_features.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'State features changed'):
                run(root,root/'windows',root/'sequences',model,root/'out','test',True)


if __name__=='__main__':unittest.main()
