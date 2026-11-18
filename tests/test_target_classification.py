import json
import math
import tempfile
import csv
import unittest
from dataclasses import replace
from pathlib import Path

from src.target_classification import CLASSES,ClassConfig,WaveformClassModel,feature_bins,pool_objects,combine,top_class
from classify_targets import export_pulses


def model():
    rows=[]
    for i,label in enumerate(CLASSES):
        for k in range(20):
            rows.append((28000.+i*1000.+k%3,.5+i*.3,300.+i*50.,'stable',label,64))
    return WaveformClassModel(ClassConfig()).fit(rows)


class TargetClassificationTests(unittest.TestCase):
    def test_density_recovers_all_classes_and_roundtrips_JSON(self):
        m=model();loaded=WaveformClassModel.from_dict(json.loads(json.dumps(m.to_dict())))
        for i,label in enumerate(CLASSES):
            x=(28000.+i*1000.,.5+i*.3,300.+i*50.,'stable')
            p=m.predict(*x)
            self.assertEqual(top_class(p)[0],label);self.assertEqual(p,loaded.predict(*x))
            self.assertAlmostEqual(sum(p),1.)

    def test_unknown_PRI_has_RF_PW_prediction(self):
        p=model().predict(31000.,1.4)
        self.assertEqual(top_class(p)[0],'R04');self.assertTrue(all(math.isfinite(x) for x in p))

    def test_fit_requires_all_classes_and_valid_features(self):
        with self.assertRaises(ValueError):WaveformClassModel(ClassConfig()).fit([(28000.,.5,300.,'stable','R01',64)])
        for x in [(0.,1.,300.,'stable'),(28000.,-1.,300.,'stable'),(28000.,1.,-1.,'stable')]:
            with self.assertRaises(ValueError):model().predict(*x)

    def test_class_model_never_stores_identity_features(self):
        m=model().to_dict()
        self.assertEqual(m['features'],['rf_mhz','pw_us','pri_us','pri_model'])
        for forbidden in ['target_id','object_id','platform_id','track_id','target_state']:
            self.assertNotIn(forbidden,json.dumps(m))

    def test_multimode_same_class_supported(self):
        m=model();rows=[]
        for i,label in enumerate(CLASSES):
            for pw in [.5+i*.3,4.+i*.3]:rows.append((28000.+i*1000.,pw,300.+i*50.,'stable',label,100))
        m.fit(rows)
        self.assertEqual(top_class(m.predict(28000.,4.,300.,'stable'))[0],'R01')

    def test_time_bin_pooling_does_not_multiply_simultaneous_observer_copies(self):
        cfg=ClassConfig();p=(.8,.2,0.,0.,0.,0.,0.,0.)
        a=pool_objects([(('s','test','O1'),0,p,64)],cfg)
        b=pool_objects([(('s','test','O1'),0,p,64)]*100,cfg)
        self.assertEqual(a[('s','test','O1')]['time_bin_count'],1)
        for x,y in zip(a[('s','test','O1')]['scores'],b[('s','test','O1')]['scores']):self.assertAlmostEqual(x,y)

    def test_pooling_never_crosses_scene_version_or_object(self):
        cfg=ClassConfig();p=(.9,.1,0.,0.,0.,0.,0.,0.)
        rows=[(k,0,p,64) for k in [('s','test','O1'),('s','train','O1'),('s2','test','O1'),('s','test','O2')]]
        self.assertEqual(len(pool_objects(rows,cfg)),4)

    def test_strong_conflicting_waveform_survives_mixed_object(self):
        cfg=ClassConfig();e={'reliable':True,'target_class':'R01','scores':(.99,.01,0.,0.,0.,0.,0.,0.)}
        p=(.01,.99,0.,0.,0.,0.,0.,0.)
        result,basis=combine(p,e,cfg)
        self.assertEqual(result,p);self.assertEqual(basis,'strong_waveform_disagrees_with_object')

    def test_ambiguous_waveform_uses_reliable_object_evidence(self):
        cfg=ClassConfig();e={'reliable':True,'target_class':'R01','scores':(.99,.01,0.,0.,0.,0.,0.,0.)}
        p=(.45,.55,0.,0.,0.,0.,0.,0.)
        result,basis=combine(p,e,cfg)
        self.assertEqual(top_class(result)[0],'R01');self.assertEqual(basis,'waveform_and_object_context')

    def test_model_configuration_invalid_values(self):
        for cfg in [replace(ClassConfig(),rf_bin_mhz=0),replace(ClassConfig(),joint_weight=1.1),replace(ClassConfig(),minimum_object_bins=1.5)]:
            with self.assertRaises(ValueError):cfg.validate()

    def test_pulse_export_ignores_truth_and_preserves_unknown_PRI_rows(self):
        m=model()
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            raw=[dict(scenario_id='s',data_version='test',platform_id='p',pulse_id='1',param1='28000',param2='.5',target_id='hidden',target_class='R08',target_state='2'),
                 dict(scenario_id='s',data_version='test',platform_id='p',pulse_id='2',param1='31000',param2='1.4',target_id='hidden',target_class='R08',target_state='2')]
            fused=[dict(scenario_id='s',data_version='test',platform_id='p',pulse_id='1',sequence_id='S1',local_cluster_id='L1',object_id='O1'),
                   dict(scenario_id='s',data_version='test',platform_id='p',pulse_id='2',sequence_id='',local_cluster_id='',object_id='')]
            def write(name,rows):
                with (r/name).open('w',newline='') as f:
                    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
            write('A.csv',raw);write('F.csv',fused)
            key=('s','test','p','S1');links={key:('s','test','O1')};pred={key:(m.predict(28000.,.5,300.,'stable'),'waveform_only')}
            info,_=export_pulses(r/'A.csv',r/'F.csv',r/'one.csv',{},links,pred,{},m)
            self.assertEqual(info['input_output_pulses'],2)
            for row in raw:
                for n in ['target_id','target_class','target_state']:del row[n]
            write('A.csv',raw)
            export_pulses(r/'A.csv',r/'F.csv',r/'two.csv',{},links,pred,{},m)
            self.assertEqual((r/'one.csv').read_bytes(),(r/'two.csv').read_bytes())


if __name__=='__main__':unittest.main()
