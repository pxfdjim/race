import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from match_tracks import run
from prepare_inputs import digest
from src.localization import HorizontalFrame
from src.track_matching import MatchConfig
from verify_track_matching import verify


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(root):
    aroot=root/'output/localized_2d';directory=aroot/'test';frame=HorizontalFrame(142.,24.6)
    a=[dict(scenario_id='s',data_version='test',object_id='O1',segment_id='T1',time_ns=i*500_000_000,
            time_seconds=i*.5,east_m=3000.+i*40,north_m=5000.) for i in range(41)]
    write(directory/'track_points.csv',a)
    write(directory/'localization_objects.csv',[dict(scenario_id='s',data_version='test',object_id=k) for k in ['O1','O2']])
    meta=dict(dimension=2,height_used=False,split='test',upstream_association_B_assisted=True,
        frames=[dict(scenario_id='s',data_version='test',origin_longitude=142.,origin_latitude=24.6)],
        files={n:{'sha256':digest(directory/n)} for n in ['track_points.csv','localization_objects.csv']})
    (directory/'localization_run.json').write_text(json.dumps(meta))
    units={'a_toa_unit':'us','internal_time_unit':'ns','origins':[dict(scenario_id='s',data_version='test',datetime='2026-11-11 09:00:00')]}
    (root/'config').mkdir();(root/'config/input_units.json').write_text(json.dumps(units))
    rows=[]
    for tid,offset in [('1',0.),('2',2000.)]:
        for i in range(21):
            lon,lat=frame.inverse(3000.+80*i+offset,5000.)
            rows.append(dict(scenario_id='s',data_version='test',track_id=tid,datetime=f'2026-11-11 09:00:{i:02}',
                target_lontitude=lon,target_latitude=lat,target_height='INVALID HEIGHT',target_id='HIDDEN',
                target_class='HIDDEN',target_type='HIDDEN'))
    bfile=root/'B.csv';write(bfile,rows)
    prior=root/'output/fused_b_assisted_guarded/test/object_B_priors.csv'
    write(prior,[dict(scenario_id='s',data_version='test',object_id='O1',track_id='2')])
    return aroot,bfile,prior


class TrackMatchingCliTests(unittest.TestCase):
    def test_complete_outputs_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);aroot,bfile,prior=fixture(root)
            with patch('match_tracks.render'),contextlib.redirect_stdout(io.StringIO()):
                meta=run(root,aroot,root/'result','test',MatchConfig(),bfile)
            self.assertEqual(meta['matched_object_count'],1)
            self.assertEqual(meta['candidate_pair_count'],4)
            self.assertEqual(meta['input_A_track_points'],41)
            self.assertEqual(meta['auxiliary_prior_comparison']['different'],1)
            self.assertTrue(verify(root/'result')['passed'])
            with self.assertRaises(FileExistsError):run(root,aroot,root/'result','test',MatchConfig(),bfile)

    def test_removing_B_labels_heights_and_changing_prior_does_not_change_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);aroot,bfile,prior=fixture(root)
            with patch('match_tracks.render'),contextlib.redirect_stdout(io.StringIO()):
                run(root,aroot,root/'first','test',MatchConfig(),bfile)
            with bfile.open() as f:rows=list(csv.DictReader(f))
            for r in rows:
                for n in ['target_id','target_class','target_type','target_height']:del r[n]
            write(bfile,rows);prior.unlink()
            with patch('match_tracks.render'),contextlib.redirect_stdout(io.StringIO()):
                run(root,aroot,root/'second','test',MatchConfig(),bfile)
            for n in ['object_track_links.csv','pair_candidates.csv','matched_track_points.csv']:
                self.assertEqual((root/'first'/n).read_bytes(),(root/'second'/n).read_bytes())

    def test_modified_A_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);aroot,bfile,_=fixture(root)
            with (aroot/'test/track_points.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'A localization artifact changed'):
                run(root,aroot,root/'result','test',MatchConfig(),bfile)

    def test_modified_matching_result_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);aroot,bfile,_=fixture(root)
            with patch('match_tracks.render'),contextlib.redirect_stdout(io.StringIO()):
                run(root,aroot,root/'result','test',MatchConfig(),bfile)
            with (root/'result/object_track_links.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'Changed matching output'):verify(root/'result')


if __name__=='__main__':unittest.main()
