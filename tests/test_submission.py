import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from prepare_inputs import DEFAULT_FILES,digest
from export_submission import run,normalize_links
from verify_submission import verify


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(base,duplicate_B=False,bad_state=False):
    root=base/'zgy';root.mkdir()
    raw=root.parent/DEFAULT_FILES['test'][0];bfile=root.parent/DEFAULT_FILES['test'][1]
    a=[dict(scenario_id='P1A',data_version='test',platform_id='p',pulse_id=str(i)) for i in [1,2,3]]
    classes=[dict(r,sequence_id='S'+str(i),local_cluster_id='' if i==2 else 'L'+str(i),object_id='' if i==2 else 'O'+str(i),target_class='R01') for i,r in enumerate(a,1)]
    states=[{k:v for k,v in r.items() if k!='target_class'}|{'target_state':'3' if bad_state else '0'} for r in classes]
    class_dir=root/'output/classified/test';state_dir=root/'output/state_predicted/test';match_dir=root/'output/ab_matched_2d/test'
    write(raw,a);write(bfile,[dict(scenario_id='P1A',data_version='test',track_id='10')])
    write(class_dir/'pulse_class_predictions.csv',classes);write(state_dir/'pulse_state_predictions.csv',states)
    links=[dict(scenario_id='P1A',data_version='test',object_id='O1',track_id='10',link_status='matched'),
           dict(scenario_id='P1A',data_version='test',object_id='O3',track_id='10' if duplicate_B else '',link_status='matched' if duplicate_B else 'unmatched')]
    write(match_dir/'object_track_links.csv',links)
    write(root/'output/fused_b_assisted_guarded/test/fused_objects.csv',[{k:r[k] for k in ['scenario_id','data_version','object_id']} for r in links])
    for directory,name,meta_name in [(class_dir,'pulse_class_predictions.csv','classification_run.json'),(state_dir,'pulse_state_predictions.csv','state_prediction_run.json')]:
        meta={'files':{name:{'sha256':digest(directory/name)}},'input_files':{'raw_A':{'sha256':digest(raw)}},'pulse_export':{'input_output_pulses':3}}
        (directory/meta_name).write_text(json.dumps(meta))
    (match_dir/'track_match_run.json').write_text(json.dumps({'files':{'object_track_links.csv':{'sha256':digest(match_dir/'object_track_links.csv')}},'input_files':{'B_source':{'sha256':digest(bfile)}}}))
    return root


class SubmissionTests(unittest.TestCase):
    def test_exact_headers_no_match_and_no_rows_or_objects_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=fixture(base)
            with contextlib.redirect_stdout(io.StringIO()):
                meta=run(root,base/'out');result=verify(base/'out')
            self.assertEqual(meta['input_output_pulses'],3);self.assertEqual(meta['object_count'],2)
            self.assertEqual(meta['blank_object_id_pulses'],1)
            self.assertTrue(result['passes_explicit_PDF_checks']);self.assertFalse(result['platform_upload_tested'])
            with (base/'out/P1A_test_object_track_links.csv').open() as f:links=list(csv.DictReader(f))
            self.assertEqual(links[1]['link_status'],'no_match');self.assertEqual(links[1]['track_id'],'')
            with self.assertRaises(FileExistsError):run(root,base/'out')

    def test_repeated_B_assignment_is_rejected_instead_of_silent_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=fixture(base,duplicate_B=True)
            with self.assertRaisesRegex(ValueError,'B track reused'):run(root,base/'out')

    def test_illegal_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=fixture(base,bad_state=True)
            with self.assertRaisesRegex(ValueError,'Illegal class/state'):run(root,base/'out')

    def test_changed_output_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=fixture(base)
            with contextlib.redirect_stdout(io.StringIO()):run(root,base/'out')
            with (base/'out/P1A_test_pulse_predictions.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed submission output'):verify(base/'out')

    def test_nonexistent_B_track_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'links.csv'
            write(path,[dict(scenario_id='s',data_version='test',object_id='O',track_id='missing',link_status='matched')])
            with self.assertRaisesRegex(ValueError,'does not exist'):normalize_links(path,{('s','test','O')},{('s','test','other')})


if __name__=='__main__':unittest.main()
