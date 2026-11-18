import json
import math
import random
import unittest
from dataclasses import replace

from src.state_features import BinStats,StateFeatureConfig,feature_names,features_for_group,source_key
from src.state_model import StateForest,StateModelConfig,smooth_states,learn_transitions


class StateFeatureTests(unittest.TestCase):
    def test_features_ignore_PA_offset_and_absolute_time_with_equivalent_context(self):
        cfg=StateFeatureConfig()
        def build(offset,shift):
            result=[]
            for index in range(5):
                b=BinStats(('s','test','p','object','O',index+shift))
                for i in range(20):b.add((index+shift)*50_000_000+i*1_000_000,offset+(.2 if i%2 else -.2),'S'+str(index),1000.)
                result.append(b)
            return features_for_group(result,cfg)
        a,b=build(30.,0),build(90.,1000)
        for (_,x),(_,y) in zip(a,b):
            self.assertEqual(len(x),len(feature_names(cfg)))
            for u,v in zip(x,y):self.assertAlmostEqual(u,v,places=8)

    def test_source_membership_priority_and_fallback(self):
        scope=('s','test','p')
        self.assertEqual(source_key(scope,'O','L')[-2:],('object','O'))
        self.assertEqual(source_key(scope,'','L')[-2:],('local','L'))
        self.assertEqual(source_key(scope,'','',30000.,1.,359.)[-2],'waveform')
        with self.assertRaises(ValueError):source_key(scope,'','')

    def test_feature_does_not_create_unobserved_bins(self):
        a=BinStats(('s','test','p','object','O',0));a.add(0,30.)
        b=BinStats(('s','test','p','object','O',100));b.add(5_000_000_000,31.)
        result=features_for_group([a,b],StateFeatureConfig())
        self.assertEqual(len(result),2);self.assertTrue(all(math.isfinite(x) for _,f in result for x in f))

    def test_bad_pulse_order_and_configuration_rejected(self):
        b=BinStats(('s','test','p','object','O',0));b.add(20,30.)
        with self.assertRaises(ValueError):b.add(10,30.)
        with self.assertRaises(ValueError):replace(StateFeatureConfig(),bin_seconds=0).validate()


class StateModelTests(unittest.TestCase):
    def test_forest_learns_three_states_and_roundtrips(self):
        rng=random.Random(7);x=[];y=[]
        for state in range(3):
            for i in range(100):x.append((state*10.+rng.random(),rng.random()));y.append(state)
        cfg=StateModelConfig(trees=5,maximum_depth=5,samples_per_tree=600,features_per_split=2,minimum_leaf_samples=4)
        m=StateForest(cfg,['observable_a','observable_b']).fit(x,y,[1]*len(y))
        loaded=StateForest.from_dict(json.loads(json.dumps(m.to_dict())))
        for state in range(3):
            p=m.predict((state*10.+.4,.5));self.assertEqual(max(range(3),key=lambda i:p[i]),state)
            self.assertEqual(p,loaded.predict((state*10.+.4,.5)));self.assertAlmostEqual(sum(p),1.)

    def test_missing_training_state_is_rejected(self):
        with self.assertRaises(ValueError):StateForest(StateModelConfig(),['x']).fit([(0.,),(1.,)],[0,1],[1,1])

    def test_temporal_decoder_respects_gaps_and_sources(self):
        prefix=('s','test','p','object','O')
        keys=[prefix+(0,),prefix+(1,),prefix+(10,),('s','test','p2','object','O',0)]
        p=[(.99,.005,.005),(.4,.35,.25),(.01,.98,.01),(.01,.01,.98)]
        transition=[[.99,.005,.005],[.005,.99,.005],[.005,.005,.99]]
        result=smooth_states(keys,p,transition,.5)
        self.assertGreater(result[1][0],p[1][0])
        for i in [2,3]:
            for a,b in zip(result[i],p[i]):self.assertAlmostEqual(a,b)
        self.assertEqual(len(result),len(keys))

    def test_zero_temporal_strength_is_identity(self):
        p=[(.2,.3,.5)];self.assertEqual(smooth_states([('s','test','p','object','O',0)],p,[[1/3]*3]*3,0.),p)

    def test_transitions_use_only_adjacent_same_source_keys(self):
        keys=[('s','train','p','object','O',0),('s','train','p','object','O',1),('s','train','p','object','O',10)]
        t=learn_transitions(keys,[0,1,2],[1.,1.,1.]);self.assertGreater(t[0][1],t[0][2]);self.assertAlmostEqual(t[1][2],1/3)

    def test_invalid_model_settings_rejected(self):
        for cfg in [replace(StateModelConfig(),trees=0),replace(StateModelConfig(),prior_power=-1),replace(StateModelConfig(),temporal_strength=3)]:
            with self.assertRaises(ValueError):cfg.validate()


if __name__=='__main__':unittest.main()
