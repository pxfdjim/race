import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.b_assisted import (AssistanceConfig, BearingPrior, assist, bearing_candidates, unique_choice)
from src.dataio import Track, TrackObservation, InputConfig, read_b_tracks
from src.fusion import FusionObservation, LocalFrame
from src.local_grouping import Sequence


def track(name='b1',lon=140.01,lat=24.04,times=(0,1,2,3,4)):
    return Track(('s','test',name),[TrackObservation(int(t*1e9),int(t*1e9),(lon,lat,300.),'', '', '',i)
        for i,t in enumerate(times)])


def observation(number,time=1.,platform='p1',position=(140.,24.,0.),target=(140.01,24.04,300.)):
    frame=LocalFrame(*position[:2]);xyz=LocalFrame.ecef(*target);origin=LocalFrame.ecef(*position)
    delta=tuple(b-a for a,b in zip(origin,xyz))
    dx,dy=(sum(a*b for a,b in zip(delta,axis)) for axis in (frame.east,frame.north))
    doa=math.degrees(math.atan2(dx,dy))%360
    start=round(time*1e9)
    s=Sequence(('s','test',platform),str(number),start,start+25_200_000,64,'stable',400.,100.,1.,-30.,doa,.1,position)
    return FusionObservation(s,'mixed',frame.project(position),frame.direction(position,doa))


def baseline(obs,oid=''):
    return {o.key:{'object_id':oid} for o in obs}


class AssistanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(AssistanceConfig(),minimum_anchor_sequences=2,minimum_anchor_pulses=100,seed_requires_multiplatform=False)

    def test_no_extrapolation_or_smoothing_across_missing_B_span(self):
        prior=BearingPrior(track(times=(0,1,2,10,11)),self.cfg)
        self.assertIsNone(prior.position(-1))
        self.assertIsNone(prior.position(5_000_000_000))
        self.assertIsNone(prior.position(12_000_000_000))
        self.assertIsNotNone(prior.position(1_500_000_000))
        self.assertIsNotNone(prior.position(10_000_000_000))

    def test_moving_observer_position_and_geographic_north(self):
        priors=[BearingPrior(track(),self.cfg)]
        for i,position in enumerate(((140.,24.,0.),(140.04,24.05,5000.))):
            o=observation(i,position=position)
            cs=bearing_candidates(o,priors,self.cfg)
            self.assertLess(cs[('s','test','b1')]['residual'],1e-8)

    def test_unused_B_tracks_do_not_create_A_objects_or_force_group_count(self):
        obs=[observation(i,1.+i*.1) for i in range(5)]
        tracks={t.key:t for t in (track(),track('false',lon=139.9,lat=23.9))}
        links,anchors,_,stats=assist(obs,tracks,baseline(obs),self.cfg)
        self.assertEqual(len(anchors),1)
        self.assertEqual(len({v['object_id'] for v in links.values()}),1)
        self.assertEqual(stats['input_B_tracks'],2)
        self.assertFalse(stats['count_constraint_used'])

    def test_same_bearing_candidates_remain_ambiguous(self):
        obs=[observation(i,1.+i*.1) for i in range(5)]
        tracks={t.key:t for t in (track(),track('b2'))}
        links,anchors,_,_=assist(obs,tracks,baseline(obs),self.cfg)
        self.assertEqual(anchors,{})
        self.assertTrue(all(not v['object_id'] for v in links.values()))

    def test_production_default_requires_independent_platform_anchors(self):
        cfg=replace(self.cfg,seed_requires_multiplatform=True)
        obs=[observation(i,1.+i*.1) for i in range(5)]
        t=track()
        links,anchors,_,_=assist(obs,{t.key:t},baseline(obs),cfg)
        self.assertFalse(anchors)
        self.assertTrue(all(not v['object_id'] for v in links.values()))
        pair=observation('second',1.,platform='p2',position=(140.04,24.,0.))
        obs=[obs[0],pair];base=baseline(obs)
        for row in base.values():row['event_id']='E1'
        links,anchors,_,_=assist(obs,{t.key:t},base,cfg)
        self.assertEqual(len(anchors),1)
        self.assertTrue(all(v['object_id'] for v in links.values()))

    def test_baseline_survives_with_no_B_temporal_coverage(self):
        obs=[observation(i,20.+i) for i in range(3)]
        links,anchors,_,_=assist(obs,{track().key:track()},baseline(obs,'O00001'),self.cfg)
        self.assertFalse(anchors)
        self.assertEqual({v['object_id'] for v in links.values()},{'OA00001'})

    def test_four_platform_fragments_use_one_supported_B_anchor(self):
        obs=[observation(i,1.+i*.1,platform=f'p{i}') for i in range(4)]
        t=track();links,anchors,_,_=assist(obs,{t.key:t},baseline(obs),self.cfg)
        self.assertEqual(len(anchors),1)
        self.assertEqual(len({v['object_id'] for v in links.values()}),1)
        self.assertTrue(all(v['object_id'] for v in links.values()))

    def test_unavailable_class_and_truth_columns_cannot_change_inference(self):
        cfg=InputConfig.from_dict({'a_toa_unit':'us','origins':[{'scenario_id':'s','data_version':'test','datetime':'2026-01-01 00:00:00'}]})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'B.csv'
            header='scenario_id,data_version,track_id,datetime,target_lontitude,target_latitude,target_height'
            rows=[f's,test,b1,2026-01-01 00:00:0{i},140.01,24.04,300' for i in range(5)]
            p.write_text(header+',target_id,target_class,target_type\n'+'\n'.join(r+',SECRET,R99,large' for r in rows)+'\n')
            tracks=read_b_tracks(p,cfg)
            obs=[observation(i,1.+i*.1) for i in range(5)]
            result=assist(obs,tracks,baseline(obs),self.cfg)
            p.write_text(header+'\n'+'\n'.join(rows)+'\n')
            other=assist(obs,read_b_tracks(p,cfg),baseline(obs),self.cfg)
            self.assertEqual(result,other)

    def test_scenes_do_not_share_B_anchors(self):
        o=observation(1);o.sequence=replace(o.sequence,scope=('other','test','p1'))
        t=track();links,anchors,_,_=assist([o],{t.key:t},baseline([o]),self.cfg)
        self.assertFalse(anchors);self.assertFalse(links[o.key]['object_id'])

    def test_nearby_runner_up_outside_gate_still_counts_for_ambiguity(self):
        self.assertIsNone(unique_choice({'a':.99,'b':1.01},.2))

    def test_invalid_config(self):
        for cfg in (replace(self.cfg,bearing_base_gate_degrees=float('nan')),
                    replace(self.cfg,baseline_mapping_dominance=1.1),
                    replace(self.cfg,minimum_anchor_sequences=2.5)):
            with self.assertRaises(ValueError):cfg.validate()


if __name__=='__main__':unittest.main()
