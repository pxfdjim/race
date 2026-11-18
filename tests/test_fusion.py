import csv
import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.fusion import (C_M_PER_US,FusionConfig,LocalFrame,FusionObservation,build_events,
    geometry,tdoa_residual,fuse_observations,Event,associate_events)
from src.local_grouping import Sequence
from fuse_platforms import export_pulses
from evaluate_fusion import evaluate_sequences


def observation(platform,number,observer,target,emission_time=0.,rf=100.,pri=400.,local_group='L1'):
    dx,dy=target[0]-observer[0],target[1]-observer[1]
    distance=math.hypot(dx,dy)
    start=round(emission_time*1e9+distance/C_M_PER_US*1000)
    direction=dx/distance,dy/distance
    doa=math.degrees(math.atan2(dx,dy))%360
    s=Sequence(('s','train',platform),str(number),start,start+round(63*pri*1000),64,'stable',pri,
               rf,1.,-30.,doa,1.,(140.,24.,0.))
    return FusionObservation(s,local_group,observer,direction)


class FusionTests(unittest.TestCase):
    def test_wgs84_frame_projects_axes_and_preserves_geographic_direction(self):
        frame=LocalFrame(140.,24.)
        self.assertAlmostEqual(math.hypot(*frame.project((140.,24.,0.))),0.,places=8)
        east=frame.project((140.01,24.,0.));north=frame.project((140.,24.01,0.))
        self.assertGreater(east[0],1000);self.assertGreater(north[1],1100)
        self.assertAlmostEqual(frame.direction((140.,24.,0.),90)[0],1.)
        self.assertAlmostEqual(frame.direction((140.,24.,0.),0)[1],1.)

    def test_forward_geometry_and_true_differential_arrival(self):
        a=observation('p1',1,(0.,0.),(3000.,5000.))
        b=observation('p2',1,(10000.,0.),(3000.,5000.))
        g=geometry([a,b],FusionConfig())
        self.assertAlmostEqual(g['east_m'],3000.,places=6)
        self.assertAlmostEqual(g['north_m'],5000.,places=6)
        self.assertLess(tdoa_residual(a,b,g['ranges_m'],FusionConfig(),g['error_m']),.001)

    def test_parallel_or_backward_bearings_do_not_form_event(self):
        a=observation('p1',1,(0.,0.),(3000.,5000.))
        b=observation('p2',1,(10000.,0.),(3000.,5000.))
        parallel=replace(b,direction=a.direction)
        backward=replace(b,direction=tuple(-v for v in b.direction))
        self.assertIsNone(geometry([a,parallel],FusionConfig()))
        self.assertIsNone(geometry([a,backward],FusionConfig()))

    def test_intersection_with_incompatible_arrival_phase_is_rejected(self):
        a=observation('p1',1,(0.,0.),(3000.,5000.))
        b=observation('p2',1,(10000.,0.),(3000.,5000.))
        b.sequence=replace(b.sequence,start=b.sequence.start+100000,end=b.sequence.end+100000)
        self.assertEqual(build_events([a,b],FusionConfig())[0],[])

    def test_four_platform_event_and_unique_sequence_ownership(self):
        observers=[(0.,0.),(10000.,0.),(0.,10000.),(10000.,10000.)]
        obs=[observation(f'p{i}',1,p,(3500.,4500.)) for i,p in enumerate(observers)]
        events,_=build_events(obs,FusionConfig())
        self.assertEqual(len(events),1)
        self.assertEqual(len(events[0].members),4)
        self.assertEqual(len({o.key for e in events for o in e.members}),4)

    def test_pairwise_chain_does_not_become_a_three_platform_component(self):
        obs=[observation(p,1,pos,(3000.,5000.),rf=rf) for p,pos,rf in
             [('p1',(0.,0.),100.),('p2',(10000.,0.),102.),('p3',(0.,10000.),104.)]]
        events,_=build_events(obs,FusionConfig())
        self.assertEqual(len(events),1)
        self.assertEqual(len(events[0].members),2)

    def test_different_received_first_pulse_does_not_require_exact_alignment(self):
        a=observation('p1',1,(0.,0.),(3000.,5000.))
        b=observation('p2',1,(10000.,0.),(3000.,5000.))
        b.sequence=replace(b.sequence,start=b.sequence.start+800000,end=b.sequence.end+800000)
        self.assertEqual(len(build_events([a,b],FusionConfig())[0]),1)

    def test_many_local_fragments_become_one_object_with_parameter_changes(self):
        obs=[]
        for i in range(12):
            for platform,p in [('p1',(0.,0.)),('p2',(10000.,0.)),('p3',(0.,10000.))]:
                obs.append(observation(platform,i,p,(3000.+i*4,5000.),i*.1,
                    rf=100+(i%3)*40,pri=400+(i%2)*100,local_group=f'L{i//3}'))
        objects,events,links,_=fuse_observations(obs,FusionConfig())
        self.assertEqual(len(objects),1)
        self.assertEqual(len(events),12)
        self.assertTrue(all(l['object_id']=='O00001' for l in links.values()))

    def test_same_waveform_two_targets_do_not_chain_merge(self):
        obs=[]
        for i in range(6):
            for target in [(2500.,4000.),(7500.,6000.)]:
                for platform,p in [('p1',(0.,0.)),('p2',(10000.,0.)),('p3',(0.,10000.))]:
                    obs.append(observation(platform,f'{target[0]}-{i}',p,target,i*.1,local_group=str(target[0])))
        objects,events,links,_=fuse_observations(obs,FusionConfig())
        self.assertEqual(len(objects),2)
        for _,obj in objects:
            self.assertEqual(len({o.sequence.sequence_id.split('-')[0] for e in obj.events for o in e.members}),1)

    def test_one_mixed_local_group_can_contribute_to_distinct_objects(self):
        obs=[]
        for i,target in enumerate([(2500.,4000.),(7500.,6000.)]):
            for platform,p in [('p1',(0.,0.)),('p2',(10000.,0.)),('p3',(0.,10000.))]:
                obs.append(observation(platform,i,p,target,0.,local_group='same_mixed_group'))
        objects,_,links,_=fuse_observations(obs,FusionConfig())
        self.assertEqual(len(objects),2)
        self.assertEqual(len({links[o.key]['object_id'] for o in obs if o.platform=='p1'}),2)

    def test_single_platform_only_observation_can_use_nearby_fused_context(self):
        obs=[]
        for i in (0,2):
            for platform,p in [('p1',(0.,0.)),('p2',(10000.,0.))]:
                obs.append(observation(platform,i,p,(3000.,5000.),i*.1))
        single=observation('p1',1,(0.,0.),(3000.,5000.),.1)
        obs.append(single)
        _,_,links,_=fuse_observations(obs,FusionConfig())
        self.assertEqual(links[single.key]['basis'],'single_platform_context')
        self.assertTrue(links[single.key]['object_id'])

    def test_no_cross_platform_support_is_retained_unmatched(self):
        o=observation('p1',1,(0.,0.),(3000.,5000.))
        objects,events,links,_=fuse_observations([o],FusionConfig())
        self.assertEqual(objects,[]);self.assertEqual(events,[])
        self.assertEqual(links[o.key]['object_id'],'')

    def test_ambiguous_events_do_not_recursively_create_new_objects(self):
        cfg=replace(FusionConfig(),object_position_gate_m=1000.,object_ambiguity_margin=.3)
        events=[]
        for i,x in enumerate([2000.,4000.]+[3000.]*30):
            obs=tuple(observation(p,i,pos,(x,5000.),i*.1) for p,pos in [('p1',(0.,0.)),('p2',(10000.,0.))])
            g=geometry(obs,cfg)
            events.append(Event(obs,round(i*.1e9),g,0.,f'E{i}'))
        objects=associate_events(events,cfg)
        self.assertEqual(len(objects),2)

    def test_scene_and_version_are_never_combined(self):
        obs=[observation(p,1,pos,(3000.,5000.)) for p,pos in [('p1',(0.,0.)),('p2',(10000.,0.))]]
        other=[replace(o,sequence=replace(o.sequence,scope=('s2','train',o.platform))) for o in obs]
        objects,_,links,_=fuse_observations(obs+other,FusionConfig())
        self.assertEqual(len(objects),2);self.assertEqual(len(links),4)

    def test_pulse_export_uses_local_group_time_context_and_ignores_labels(self):
        o=observation('p1','s1',(0.,0.),(3000.,5000.))
        links={o.key:{'object_id':'O1','event_id':'E1','basis':'multiplatform_event','cost':0.}}
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            raw=[dict(scenario_id='s',data_version='train',platform_id='p1',pulse_id=str(i),toa=str(i*400),target_id='hidden') for i in range(3)]
            pred=[dict(scenario_id='s',data_version='train',platform_id='p1',pulse_id=str(i),
                       sequence_id='s1' if i==0 else '',local_cluster_id='L1' if i<2 else '') for i in range(3)]
            def write(path,rows):
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
            write(d/'a.csv',raw);write(d/'p.csv',pred)
            report,_=export_pulses(d/'a.csv',d/'p.csv',d/'first.csv',[o],links)
            self.assertEqual(report['input_output_pulses'],3)
            self.assertEqual(report['assigned_pulses'],2)
            for r in raw:r.pop('target_id')
            write(d/'a.csv',raw)
            export_pulses(d/'a.csv',d/'p.csv',d/'second.csv',[o],links)
            self.assertEqual((d/'first.csv').read_bytes(),(d/'second.csv').read_bytes())

    def test_invalid_configurations_rejected(self):
        for cfg in (replace(FusionConfig(),rf_pair_gate_mhz=0),replace(FusionConfig(),minimum_crossing_angle_degrees=90)):
            with self.assertRaises(ValueError):cfg.validate()


if __name__=='__main__':unittest.main()
