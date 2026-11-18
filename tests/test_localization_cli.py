import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from localize_objects import run, SEQ_KEYS
from prepare_inputs import digest
from src.localization import HorizontalFrame, LocalizationConfig
from verify_localization import verify


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(root):
    fusion=root/'output/fused_b_assisted_guarded/test'
    bearings=root/'output/grouped/test/bearing_observations.csv'
    positions=root/'output/group_inputs/test_platform_positions.csv'
    baseline=root/'output/fused/test/fusion_run.json'
    frame=HorizontalFrame(142.,24.6)
    bs,links,events=[],[],[]
    for i,t in enumerate([.5,1.,10.],1):
        oid='O1' if i<3 else ''
        for p,xy,doa in [('p1',(0.,0.),30.),('p2',(10000.,0.),330.)]:
            lon,lat=frame.inverse(*xy);ns=round(t*1e9)
            bs.append(dict(scenario_id='s',data_version='test',platform_id=p,sequence_id=f'S{i}',
                time_ns=ns,start_ns=ns-10_000_000,end_ns=ns+10_000_000,observer_longitude=lon,
                observer_latitude=lat,observer_height='deliberately invalid',doa_degrees=doa,
                doa_standard_error_degrees=.15,target_id='HIDDEN'))
            links.append(dict(zip(SEQ_KEYS,('s','test',p,f'S{i}')),object_id=oid,event_id=f'E{i}'))
        events.append(dict(scenario_id='s',data_version='test',event_id=f'E{i}',object_id=oid,time_ns=ns,
                           platform_count=2,platform_ids='p1|p2',sequence_ids=f'S{i}|S{i}',
                           coarse_east_m='DO NOT READ',coarse_north_m='DO NOT READ'))
    single=dict(bs[0],sequence_id='single',time_ns=2_000_000_000,start_ns=1_990_000_000,end_ns=2_010_000_000)
    bs.append(single);links.append(dict(zip(SEQ_KEYS,('s','test','p1','single')),object_id='O2',event_id=''))
    write(bearings,bs)
    write(positions,[dict(scenario_id='s',data_version='test',platform_id=p,toa_ns=0,longitude=142.,latitude=24.6,height='IGNORED') for p in ['p1','p2']])
    write(fusion/'cross_platform_events.csv',events)
    write(fusion/'sequence_object_links.csv',links)
    write(fusion/'fused_objects.csv',[
        dict(scenario_id='s',data_version='test',object_id='O1',sequence_count=4,event_count=2),
        dict(scenario_id='s',data_version='test',object_id='O2',sequence_count=1,event_count=0)])
    common=dict(bearing_source_sha256=digest(bearings),positions_source_sha256=digest(positions))
    baseline.parent.mkdir(parents=True,exist_ok=True)
    baseline.write_text(json.dumps(common|{'B_used':False,'frames':[dict(scenario_id='s',data_version='test',origin_longitude=142.,origin_latitude=24.6)]}))
    (fusion/'fusion_run.json').write_text(json.dumps(common|dict(B_used=True,event_count=3,
        baseline_run_sha256=digest(baseline),files={n:{'sha256':digest(fusion/n)} for n in
            ['cross_platform_events.csv','sequence_object_links.csv','fused_objects.csv']})))
    return fusion


class LocalizationCliTests(unittest.TestCase):
    def test_end_to_end_without_B_files_heights_or_coarse_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fusion=fixture(root);output=root/'out'
            with contextlib.redirect_stdout(io.StringIO()):report=run(root,fusion,output,'test',LocalizationConfig())
            self.assertEqual(report['localized_point_count'],2)
            self.assertEqual(report['track_point_count'],2)
            self.assertEqual(report['localized_object_count'],1)
            self.assertEqual(report['rejection_reasons'],{'unassigned_object_identity':1})
            self.assertFalse(report['height_used']);self.assertFalse(report['B_coordinates_used_in_localization'])
            self.assertTrue(verify(output)['passed'])
            self.assertTrue((output/'track_preview.html').exists())
            with (output/'localization_objects.csv').open() as f:summary=list(csv.DictReader(f))
            self.assertEqual(summary[1]['localization_status'],'no_usable_multiplatform_event')
            with self.assertRaises(FileExistsError):run(root,fusion,output,'test',LocalizationConfig())

    def test_input_changes_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fusion=fixture(root)
            with (fusion/'cross_platform_events.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed fusion artifact'):run(root,fusion,root/'out','test',LocalizationConfig())
            self.assertFalse((root/'out').exists())

    def test_verifier_detects_changed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fusion=fixture(root);output=root/'out'
            with contextlib.redirect_stdout(io.StringIO()):run(root,fusion,output,'test',LocalizationConfig())
            with (output/'localized_points.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed output'):verify(output)


if __name__=='__main__':unittest.main()
