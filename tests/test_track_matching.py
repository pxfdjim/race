import unittest
from dataclasses import replace
from src.track_matching import MatchConfig,TrackPoint,BTrajectory,compare,match,motions


def points(offset=(0.,0.),speed=80.,start=0,count=41,step=.5,segment='T1'):
    return [TrackPoint(round((start+i*step)*1e9),(3000.+speed*(start+i*step)+offset[0],5000.+offset[1]),segment) for i in range(count)]


def trajectory(tid,ps,scene='s',version='test'):
    return BTrajectory((scene,version,tid),ps,3.)


class TrackMatchingTests(unittest.TestCase):
    def test_unique_matching_uses_trajectory_not_id(self):
        a={('s','test','O1'):points()}
        b=[trajectory('arbitrary',points(offset=(10.,-10.))),trajectory('1',points(offset=(2000.,0.)))]
        ds,pairs=match(a,{t.key:t for t in b},MatchConfig())
        self.assertEqual(ds[('s','test','O1')]['track_id'],'arbitrary')
        self.assertAlmostEqual(pairs[('s','test','O1','arbitrary')]['median_distance_m'],2**.5*10.)

    def test_equal_B_trajectories_remain_ambiguous(self):
        bs=[trajectory('1',points()),trajectory('999',points())]
        ds,_=match({('s','test','O1'):points()},{b.key:b for b in bs},MatchConfig())
        d=ds[('s','test','O1')]
        self.assertEqual(d['status'],'ambiguous');self.assertEqual(d['track_id'],'')

    def test_far_nearest_candidate_is_not_forced(self):
        b=trajectory('1',points(offset=(2000.,0.)))
        ds,_=match({('s','test','O1'):points()},{b.key:b},MatchConfig())
        self.assertEqual(ds[('s','test','O1')]['status'],'unmatched')
        self.assertIn('median_distance_gate',ds[('s','test','O1')]['reason'])

    def test_B_has_no_extrapolation_or_gap_bridge(self):
        ps=[TrackPoint(t*1_000_000_000,(float(t),0.),'ignored') for t in [0,2,10,12]]
        b=trajectory('1',ps)
        self.assertIsNone(b.sample(-1));self.assertIsNone(b.sample(13_000_000_000))
        self.assertIsNone(b.sample(6_000_000_000))
        self.assertEqual(b.sample(1_000_000_000).xy,(1.,0.))
        self.assertEqual(b.sample(10_000_000_000).xy,(10.,0.))

    def test_time_support_excludes_long_B_gaps(self):
        a=points(count=25)
        bs=[p for p in a if p.time<=2_000_000_000 or p.time>=10_000_000_000]
        p=compare(a,trajectory('1',bs),MatchConfig())
        self.assertAlmostEqual(p['supported_duration_seconds'],4.)
        self.assertFalse(p['temporal_eligible'])

    def test_spread_singletons_do_not_fabricate_duration(self):
        a=[replace(p,segment=str(i)) for i,p in enumerate(points(count=20))]
        p=compare(a,trajectory('1',a),MatchConfig())
        self.assertEqual(p['supported_duration_seconds'],0.)
        self.assertFalse(p['eligible'])

    def test_multiple_A_fragments_may_match_same_B(self):
        a={('s','test','O1'):points(count=21),('s','test','O2'):points(start=10,count=21)}
        b=trajectory('B',points())
        ds,_=match(a,{b.key:b},MatchConfig())
        self.assertEqual([d['track_id'] for d in ds.values()],['B','B'])

    def test_crossing_with_opposite_motion_does_not_match(self):
        a=points(speed=25.)
        b=trajectory('reverse',[replace(p,xy=(3500.-25.*(p.time/1e9),5000.)) for p in a])
        p=compare(a,b,MatchConfig())
        self.assertTrue(p['motion_used']);self.assertIn('heading_gate',p['rejection_reasons'])
        self.assertFalse(p['eligible'])

    def test_short_stationary_track_can_match_without_reliable_heading(self):
        a=points(speed=0.);b=trajectory('B',points(offset=(5.,0.),speed=0.))
        d,ps=match({('s','test','O'):a},{b.key:b},MatchConfig())
        self.assertEqual(d[('s','test','O')]['status'],'matched')
        self.assertFalse(ps[('s','test','O','B')]['motion_used'])

    def test_few_spurious_motion_vectors_do_not_veto_a_whole_track(self):
        a=points();b=trajectory('B',a)
        av={p.time:(80.,0.) for p in a[:5]}
        bv={p.time:(-80.,0.) for p in a[:5]}
        p=compare(a,b,MatchConfig(),av,bv)
        self.assertEqual(p['motion_comparison_count'],5)
        self.assertFalse(p['motion_used']);self.assertTrue(p['eligible'])

    def test_scene_and_version_never_cross(self):
        a={('s','test','O'):points()}
        b=trajectory('1',points(),version='train')
        d,p=match(a,{b.key:b},MatchConfig())
        self.assertEqual(len(p),0);self.assertEqual(d[('s','test','O')]['status'],'unmatched')

    def test_invalid_configuration_and_B_duplicates_rejected(self):
        for cfg in [replace(MatchConfig(),maximum_b_gap_seconds=0),replace(MatchConfig(),minimum_temporal_coverage=1.1),
                    replace(MatchConfig(),minimum_comparison_points=2.5)]:
            with self.assertRaises(ValueError):cfg.validate()
        with self.assertRaises(ValueError):trajectory('1',points()+points())

    def test_motion_estimation_does_not_cross_A_segment_boundaries(self):
        a=points(count=5,segment='a')+points(start=5,count=5,segment='b')
        self.assertEqual(motions(a,MatchConfig()),{})


if __name__=='__main__':unittest.main()
