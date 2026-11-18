import csv
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from evaluate_local_groups import metrics
from group_sequences import export_pulses,read_sequences
from src.local_grouping import GroupConfig,Sequence,PulseAttacher,LocalGroup,associate,overlap_conflict,predict


def seq(i,time=0.,doa=10.,rf=100.,pw=1.,pri=100.,platform='p1',duration=None,model='stable'):
    duration=31*pri/1e6 if duration is None else duration
    return Sequence(('s','train',platform),str(i),round(time*1e9),round((time+duration)*1e9),
                    32,model,pri,rf,pw,-30.,doa,.8,(140.,24.,0.))


class LocalGroupingTests(unittest.TestCase):
    def test_nonoverlapping_mode_changes_stay_one_target(self):
        items=[seq(i,i*.1,rf=100+(i%3)*40,pw=1+(i%2)*.3,pri=100+(i%2)*50) for i in range(30)]
        groups,links=associate(items,GroupConfig())
        self.assertEqual(len(groups),1)
        self.assertEqual(len({r['local_cluster_id'] for r in links.values()}),1)

    def test_same_waveform_distinct_bearings_remain_two_instances(self):
        items=[seq(f'{k}-{i}',i*.1,doa=10+k*20) for i in range(30) for k in range(2)]
        groups,links=associate(items,GroupConfig())
        self.assertEqual(len(groups),2)
        for group in groups:
            self.assertEqual(len({s.doa for s in group.members}),1)

    def test_overlapping_incompatible_modes_cannot_merge_even_at_same_bearing(self):
        a,b=seq('a',0),seq('b',.001,pw=2.)
        self.assertTrue(overlap_conflict(a,b,GroupConfig()))
        self.assertEqual(len(associate([a,b],GroupConfig())[0]),2)

    def test_overlapping_stable_sources_with_distinct_phase_remain_separate(self):
        a,b=seq('a',0),seq('b',.00004)
        self.assertTrue(overlap_conflict(a,b,GroupConfig()))
        self.assertEqual(len(associate([a,b],GroupConfig())[0]),2)

    def test_wraparound_bearing_trend_and_observer_motion(self):
        items=[replace(seq(i,i*.1,doa=(359+i*.1)%360),position=(140+i*.00001,24,0)) for i in range(50)]
        self.assertEqual(len(associate(items,GroupConfig())[0]),1)

    def test_four_platforms_have_independent_scopes(self):
        items=[seq(f'{k}-{i}',i*.1,platform=str(k)) for i in range(8) for k in range(4)]
        groups,links=associate(items,GroupConfig())
        self.assertEqual(len(groups),4)
        self.assertEqual({g.name for g in groups},{'L00001'})
        self.assertEqual(len(links),32)

    def test_ambiguous_observation_is_deferred_then_left_unassigned(self):
        items=[seq('a',0,doa=9.5,pw=.5),seq('b',0,doa=10.5,pw=1.5),seq('c',.1,doa=10,pw=1.)]
        groups,links=associate(items,GroupConfig())
        self.assertEqual(len(groups),2)
        self.assertEqual(links[items[-1].key]['association_basis'],'ambiguous_unassigned_sequence')

    def test_long_gap_retains_two_tracklets_instead_of_forcing_merge(self):
        self.assertEqual(len(associate([seq(1,0),seq(2,61)],GroupConfig())[0]),2)

    def test_unknown_pri_attachment_and_ambiguity(self):
        items=[seq('a',0,doa=10,pw=.5),seq('b',0,doa=10.1,pw=1.5)]
        _,links=associate(items,GroupConfig())
        attacher=PulseAttacher(items,links,GroupConfig())
        self.assertTrue(attacher.attach(items[0].scope,1000000,100.,.5,-30.,10.)[0])
        self.assertFalse(attacher.attach(items[0].scope,2_000_000_000,100.,.5,-30.,10.)[0])
        close=[seq('a',0,doa=10),seq('b',.00004,doa=10.1)]
        _,links=associate(close,GroupConfig())
        attacher=PulseAttacher(close,links,GroupConfig())
        self.assertEqual(attacher.attach(close[0].scope,1000000,100.,1.,-30.,10.05)[2],'ambiguous')

    def test_merge_and_split_metrics_are_both_penalized(self):
        scope=('s','train','p1')
        merged=metrics(Counter({(scope,'g1','a'):100,(scope,'g1','b'):100}))
        split=metrics(Counter({(scope,'g1','a'):50,(scope,'g2','a'):50}))
        self.assertEqual(merged['bcubed_precision'],.5)
        self.assertEqual(merged['bcubed_recall'],1.)
        self.assertEqual(split['bcubed_precision'],1.)
        self.assertEqual(split['bcubed_recall'],.5)

    def test_invalid_configuration_rejected(self):
        for config in (replace(GroupConfig(),maximum_gap_seconds=0),replace(GroupConfig(),maximum_pri_ratio=1)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_rounded_360_degree_sequence_mean_normalizes_to_north(self):
        class Paths:
            def position(self,scope,time):
                return (140.,24.,0.)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sequences.csv'
            fields=['scenario_id','data_version','platform_id','sequence_id','start_toa_ns','end_toa_ns',
                    'pulse_count','pri_model','pri_center_us','rf_mean','pw_mean','pa_mean',
                    'doa_mean_degrees','doa_std_degrees']
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(fields)
                w.writerow(['s','train','p1','s1',0,3100000,32,'stable',100,100,1,-30,360,1])
            self.assertEqual(read_sequences(path,Paths())[0].doa,0.)

    def test_sparse_overlapping_bounding_intervals_do_not_prove_mode_conflict(self):
        self.assertFalse(overlap_conflict(seq(1,0,duration=.1),seq(2,.001,pw=2.,duration=.1),GroupConfig()))

    def test_cached_bearing_fit_matches_direct_fit(self):
        cfg=GroupConfig()
        g=LocalGroup(1,('s','train','p1'),0,0)
        for i in range(160):
            s=seq(i,400+i*.1,doa=(359+i*.07+(i%3)*.02)%360)
            g.add(s,cfg)
        expected,rms,_=predict(g.history,s.time+100000000,cfg)
        actual=(g.fit_angle+g.fit_rate*.1)%360
        self.assertAlmostEqual(expected,actual,places=6)
        self.assertAlmostEqual(rms,g.fit_rms,places=4)

    def test_export_retains_unassigned_and_ignores_truth_columns(self):
        items=[seq('s1',0)]
        _,links=associate(items,GroupConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            raw=[dict(scenario_id='s',data_version='train',platform_id='p1',pulse_id=str(i),
                toa=str(i*100),param1='100',param2='1',param3='-30',doa='10',target_id='secret') for i in range(3)]
            raw[-1]['param1']='9999'
            pred=[dict(scenario_id='s',data_version='train',platform_id='p1',pulse_id=str(i),
                sequence_id='s1' if i<2 else '',pri_model='stable' if i<2 else 'unknown',
                pri_center_us='100' if i<2 else '') for i in range(3)]
            def write(file,rows):
                with file.open('w',newline='') as f:
                    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
            write(path/'raw.csv',raw);write(path/'pri.csv',pred)
            report,_=export_pulses(path/'raw.csv',path/'pri.csv',path/'first.csv',items,links,GroupConfig())
            self.assertEqual(report['input_output_pulses'],3)
            self.assertEqual(report['assignment_status_counts']['unassigned_no_sequence_support'],1)
            for r in raw:
                r.pop('target_id')
            write(path/'raw.csv',raw)
            export_pulses(path/'raw.csv',path/'pri.csv',path/'second.csv',items,links,GroupConfig())
            self.assertEqual((path/'first.csv').read_bytes(),(path/'second.csv').read_bytes())


if __name__=='__main__':
    unittest.main()
