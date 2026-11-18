import csv
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from src.localization import Bearing, HorizontalFrame, LocalizationConfig, localize, filter_track, bin_points
from src.localization_plot import chart
from localize_objects import read_bearings, run


def bearing(platform, xy, target=(3000., 5000.), seconds=1., noise=0., scope=('s','test')):
    dx, dy = target[0]-xy[0], target[1]-xy[1]
    angle = math.atan2(dy, dx)+math.radians(noise)
    t=round(seconds*1e9)
    return Bearing(scope+(platform,'S1'),t,t-10_000_000,t+10_000_000,xy,(math.cos(angle),math.sin(angle)),.15)


def point(t, x, y=5000., event='E1', variance=900.):
    return {'time_ns':round(t*1e9),'east_m':x,'north_m':y,'event_id':event,'covariance':(variance,0.,variance)}


class LocalizationTests(unittest.TestCase):
    def test_exact_two_platform_intersection(self):
        obs=[bearing('p1',(0.,0.)),bearing('p2',(10000.,0.))]
        p,reason=localize(obs,LocalizationConfig())
        self.assertEqual(reason,'')
        self.assertAlmostEqual(p['east_m'],3000.,places=7)
        self.assertAlmostEqual(p['north_m'],5000.,places=7)
        self.assertEqual(p['fit_status'],'two_bearing_wls')
        self.assertGreaterEqual(p['minor_error_proxy_m'],30.)

    def test_four_platform_joint_solution_uses_four_not_six_measurements(self):
        obs=[bearing(str(i),xy) for i,xy in enumerate([(0.,0.),(10000.,0.),(0.,10000.),(10000.,10000.)])]
        p,reason=localize(obs,LocalizationConfig())
        self.assertEqual(reason,'');self.assertEqual(len(p['selected']),4)
        self.assertAlmostEqual(p['east_m'],3000.,places=6)
        self.assertAlmostEqual(p['north_m'],5000.,places=6)
        self.assertEqual(p['fit_status'],'joint_multi_bearing_wls')

    def test_noisy_joint_fit_and_covariance(self):
        obs=[bearing(str(i),xy,noise=noise) for i,(xy,noise) in enumerate([
            ((0.,0.),.2),((10000.,0.),-.1),((0.,10000.),.15),((10000.,10000.),-.2)])]
        p,_=localize(obs,LocalizationConfig())
        self.assertLess(math.dist((p['east_m'],p['north_m']),(3000.,5000.)),40.)
        a,b,c=p['covariance'];self.assertGreater(a*c-b*b,0)
        self.assertLess(p['maximum_bearing_residual_degrees'],.4)

    def test_parallel_and_backward_not_localized(self):
        a,b=bearing('p1',(0.,0.)),bearing('p2',(10000.,0.))
        p,r=localize([a,replace(b,direction=a.direction)],LocalizationConfig())
        self.assertIsNone(p);self.assertEqual(r,'weak_crossing_angle')
        p,_=localize([a,replace(b,direction=tuple(-x for x in b.direction))],LocalizationConfig())
        self.assertIsNone(p)

    def test_single_platform_never_generates_position(self):
        self.assertEqual(localize([bearing('p1',(0.,0.))],LocalizationConfig())[1],'fewer_than_two_platforms')

    def test_duplicate_platform_and_cross_scene_are_errors(self):
        a,b=bearing('p1',(0.,0.)),bearing('p2',(10000.,0.))
        for invalid in [replace(b,key=('s','test','p1','S2')),replace(b,key=('other','test','p2','S2'))]:
            with self.assertRaises(ValueError):localize([a,invalid],LocalizationConfig())

    def test_moving_observers_at_asynchronous_times(self):
        # Different observer positions/times are retained, not replaced by a
        # common nominal platform location. The short window has small motion.
        target=lambda t:(3000.+80*t,5000.+25*t)
        obs=[bearing('p1',(0.,0.),target(1.),seconds=1.),
             bearing('p2',(10004.,2.),target(1.04),seconds=1.04),
             bearing('p3',(4.,10002.),target(1.06),seconds=1.06)]
        p,r=localize(obs,LocalizationConfig())
        self.assertEqual(r,'')
        self.assertLess(math.dist((p['east_m'],p['north_m']),target(p['time_ns']/1e9)),6.)
        self.assertAlmostEqual(p['time_span_seconds'],.06)
        late=replace(obs[-1],time=2_000_000_000)
        self.assertEqual(localize(obs[:2]+[late],LocalizationConfig())[1],'asynchronous_span_too_large')

    def test_one_outlier_on_four_platforms_has_unique_three_member_consensus(self):
        obs=[bearing(str(i),xy,noise=(20. if i==3 else 0.)) for i,xy in enumerate([
            (0.,0.),(10000.,0.),(0.,10000.),(10000.,10000.)])]
        p,r=localize(obs,LocalizationConfig())
        self.assertEqual(r,'');self.assertEqual(len(p['selected']),3)
        self.assertAlmostEqual(p['east_m'],3000.,places=5)

    def test_ambiguous_two_of_three_subsets_are_rejected(self):
        obs=[bearing(str(i),xy,noise=(20. if i==2 else 0.)) for i,xy in enumerate([(0.,0.),(10000.,0.),(0.,10000.)])]
        p,r=localize(obs,LocalizationConfig())
        self.assertIsNone(p);self.assertEqual(r,'ambiguous_inlier_subsets')

    def test_far_intersection_rejected(self):
        obs=[bearing('p1',(0.,0.),(500000.,500000.)),bearing('p2',(100000.,0.),(500000.,500000.))]
        self.assertIsNone(localize(obs,replace(LocalizationConfig(),minimum_crossing_angle_degrees=1.))[0])

    def test_frame_roundtrip_and_north_clockwise(self):
        frame=HorizontalFrame(142.,24.6)
        for lon,lat in [(142.,24.6),(142.5,24.4),(141.7,25.2)]:
            self.assertAlmostEqual(frame.inverse(*frame.project(lon,lat))[0],lon,places=10)
            self.assertAlmostEqual(frame.inverse(*frame.project(lon,lat))[1],lat,places=10)
        self.assertAlmostEqual(frame.direction(142.,24.6,0.)[1],1.)
        self.assertAlmostEqual(frame.direction(142.,24.6,90.)[0],1.)

    def test_heights_and_truth_are_not_read(self):
        columns=['scenario_id','data_version','platform_id','sequence_id','time_ns','start_ns','end_ns',
                 'observer_longitude','observer_latitude','doa_degrees','doa_standard_error_degrees','observer_height','target_id']
        row=['s','test','p1','S1','100','90','110','142','24.6','350','.2','INVALID HEIGHT','SECRET']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'b.csv'
            with p.open('w',newline='') as f:w=csv.writer(f);w.writerow(columns);w.writerow(row)
            a=read_bearings(p,{('s','test'):HorizontalFrame(142.,24.6)})
            with p.open('w',newline='') as f:w=csv.writer(f);w.writerow(columns[:-2]);w.writerow(row[:-2])
            self.assertEqual(a,read_bearings(p,{('s','test'):HorizontalFrame(142.,24.6)}))

    def test_track_bins_do_not_artificially_shrink_uncertainty_with_many_events(self):
        ps=[point(1.+i*.001,3000.,event=str(i)) for i in range(100)]
        result=bin_points(ps,LocalizationConfig())
        self.assertEqual(len(result),1);self.assertEqual(result[0]['covariance'][0],900.)

    def test_segment_gaps_are_not_filled(self):
        ps=[point(t,3000.+40*t,event=str(t)) for t in [0.,.5,1.,10.,10.5]]
        ts,rs=filter_track(ps,LocalizationConfig())
        self.assertEqual(len(ts),5);self.assertFalse(rs)
        self.assertEqual(len({p['segment_id'] for p in ts}),2)
        self.assertEqual([p['time_ns'] for p in ts],[p['time_ns'] for p in ps])

    def test_track_filter_rejects_position_spike_and_recovers(self):
        ps=[point(i*.5,3000.+i*20+(10000 if i==12 else 0),event=str(i)) for i in range(30)]
        ts,rs=filter_track(ps,LocalizationConfig())
        self.assertEqual(len(rs),1);self.assertEqual(rs[0]['time_ns'],6_000_000_000)
        self.assertLess(abs(ts[-1]['east_m']-(3000.+29*20)),1.)

    def test_invalid_config_rejected(self):
        for cfg in [replace(LocalizationConfig(),maximum_time_span_seconds=0),
                    replace(LocalizationConfig(),maximum_iterations=1.5),
                    replace(LocalizationConfig(),minimum_crossing_angle_degrees=90)]:
            with self.assertRaises(ValueError):cfg.validate()

    def test_svg_is_well_formed_and_object_names_escaped(self):
        ps=[point(1,3000.)|{'object_id':'<bad&name>'}]
        svg=chart(HorizontalFrame(142.,24.6),ps,[],{},['<bad&name>'],'Test','raw')
        self.assertEqual(ET.fromstring(svg).tag,'{http://www.w3.org/2000/svg}svg')
        self.assertIn('&lt;bad&amp;name&gt;',svg)


if __name__=='__main__':unittest.main()
