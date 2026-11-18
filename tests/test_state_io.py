import csv
import json
import tempfile
import unittest
from pathlib import Path
from src.state_io import read_windows,read_truth,state_metrics


class StateIOTests(unittest.TestCase):
    def test_truth_columns_do_not_change_observable_features(self):
        fields=['scenario_id','data_version','platform_id','source_kind','source_id','bin_index','pulse_count','first_time_ns','last_time_ns','observable','target_state','target_id']
        row=['s','test','p','object','O','0','10','0','40000000','1.25','2','SECRET']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'features.csv'
            with path.open('w',newline='') as f:w=csv.writer(f);w.writerow(fields);w.writerow(row)
            a=read_windows(path,['observable'])
            with path.open('w',newline='') as f:w=csv.writer(f);w.writerow(fields[:-2]);w.writerow(row[:-2])
            self.assertEqual(a,read_windows(path,['observable']))

    def test_state_metrics_expose_minority_failure(self):
        keys=[('a',),('b',)];p=[(.99,.005,.005)]*2;truth={keys[0]:(990,0,0),keys[1]:(0,10,0)}
        m=state_metrics(keys,p,truth)
        self.assertAlmostEqual(m['pulse_accuracy'],.99)
        self.assertEqual(m['per_state']['1']['recall'],0.)
        self.assertLess(m['macro_f1_present_states'],.5)

    def test_mixed_window_truth_is_evaluated_per_pulse(self):
        key=('a',);m=state_metrics([key],[(.1,.8,.1)],{key:(2,5,3)})
        self.assertEqual(m['supported_pulses'],10);self.assertEqual(m['pulse_accuracy'],.5)
        self.assertEqual(m['window_resolution_oracle_accuracy'],.5)


if __name__=='__main__':unittest.main()
