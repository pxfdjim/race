import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.dataio import (InputConfig, NS_PER_SECOND, iter_a_batches,
                       parse_timestamp, read_b_tracks, time_to_ns)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config/input_units.json').read_text())
HEADERS = ['scenario_id', 'data_version', 'pulse_id', 'datetime', 'toa', 'param1', 'param2', 'param3',
           'doa', 'sensor_lontitude', 'sensor_latitude', 'sensor_height', 'platform_id',
           'target_id', 'target_class', 'target_state']


def a_row(number=1, **values):
    row = dict(zip(HEADERS, ['P1A', 'test', str(number), '2026-11-11 09:00:00.000', '31.927',
        '28939.6', '.6', '40.5', '.8', '142.125865', '24.569929', '0', '104002', 'hidden', 'hidden', 'hidden']))
    row.update(values)
    return row


def write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def b_row(seconds, longitude=142., **values):
    row = dict(scenario_id='P1A', data_version='test', track_id='1',
               datetime=f'2026-11-11 09:00:{seconds:02d}', target_lontitude=str(longitude),
               target_latitude='24.', target_height='2400', target_type='medium', target_class='', target_id='hidden')
    row.update(values)
    return row


class DataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'input.csv'
        self.config = InputConfig.from_dict(CONFIG)

    def read_a(self, rows, **kwargs):
        write(self.path, rows)
        return [p for batch in iter_a_batches(self.path, self.config, **kwargs) for p in batch]

    def test_fractional_microseconds_are_exact_integer_nanoseconds(self):
        self.assertEqual(time_to_ns('31.927', 'us'), 31927)
        self.assertEqual(time_to_ns('36.14', 'us'), 36140)
        self.assertEqual(time_to_ns('500009943.529', 'us'), 500009943529)
        self.assertEqual(time_to_ns('1e-3', 'us'), 1)
        self.assertEqual(time_to_ns('0.000000001', 's'), 1)
        self.assertEqual(time_to_ns('31.927000', 'us'), 31927)

    def test_subnanosecond_precision_and_invalid_times_are_not_silently_truncated(self):
        for text in ('1.0001', '-1', 'NaN', 'Infinity', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                time_to_ns(text, 'us')

    def test_full_timestamps_preserve_ns_and_explicit_timezones(self):
        ticks, resolution, kind = parse_timestamp('2026-11-11 09:00:00.000000001')
        base = parse_timestamp('2026-11-11 09:00:00')[0]
        self.assertEqual(ticks - base, 1)
        self.assertEqual(resolution, 1)
        self.assertEqual(kind, 'naive_scene_local')
        local = parse_timestamp('2026-11-11T09:00:00+08:00')
        utc = parse_timestamp('2026-11-11T01:00:00Z')
        self.assertEqual(local, utc)
        self.assertEqual(parse_timestamp('2026/11/11 9:00:00')[0], base)

    def test_minute_only_B_time_is_rejected_instead_of_assigning_row_indices(self):
        write(self.path, [b_row(0, datetime='2026/11/11 9:00')])
        with self.assertRaisesRegex(ValueError, 'minute-only'):
            read_b_tracks(self.path, self.config)

    def test_streamed_four_platforms_keep_original_order_and_batches(self):
        rows = [a_row(i + 1, platform_id=('103001', '103002', '104001', '104002')[i % 4]) for i in range(9)]
        write(self.path, rows)
        batches = list(iter_a_batches(self.path, self.config, batch_size=4))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        records = [p for batch in batches for p in batch]
        self.assertEqual([p.pulse_id for p in records], [str(i) for i in range(1, 10)])
        self.assertEqual(len({p.platform for p in records}), 4)
        self.assertEqual(records[0].toa, 31927)
        self.assertEqual(records[0].toa_raw, '31.927')
        self.assertEqual(records[0].position, (142.125865, 24.569929, 0.))

    def test_A_truth_deletion_and_corruption_do_not_change_observations(self):
        original = self.read_a([a_row()])
        row = a_row(target_id='999999', target_class='wrong', target_state='not-a-number')
        self.assertEqual(original, self.read_a([row]))
        for field in ('target_id', 'target_class', 'target_state'):
            row.pop(field)
        self.assertEqual(original, self.read_a([row]))
        self.assertFalse(hasattr(original[0], 'target_id'))
        self.assertFalse(hasattr(original[0], 'target_class'))

    def test_coordinate_and_feature_aliases_match(self):
        original = self.read_a([a_row()])
        row = a_row()
        for old, new in [('sensor_lontitude', 'sensor_longitude'), ('param1', 'RF'), ('param2', 'PW'), ('param3', 'PA')]:
            row[new] = row.pop(old)
        self.assertEqual(original, self.read_a([row]))

    def test_conflicting_coordinate_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.read_a([a_row(sensor_longitude='140.')])

    def test_wrong_unit_or_scene_origin_is_detected_from_A_datetime(self):
        row = a_row(toa='1000000', datetime='2026-11-11 09:00:01.000')
        self.assertEqual(self.read_a([row])[0].toa, NS_PER_SECOND)
        bad = copy.deepcopy(CONFIG)
        bad['a_toa_unit'] = 'ns'
        write(self.path, [row])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            list(iter_a_batches(self.path, InputConfig.from_dict(bad)))

    def test_duplicate_pulse_ID_is_rejected_even_across_platforms(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.read_a([a_row(platform_id='103001'), a_row(platform_id='104001')])

    def test_sqlite_mode_allows_unordered_string_IDs_but_detects_duplicates(self):
        records = self.read_a([a_row(pulse_id='z'), a_row(pulse_id='a')], id_check='sqlite')
        self.assertEqual([p.pulse_id for p in records], ['z', 'a'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.read_a([a_row(pulse_id='z'), a_row(pulse_id='a'), a_row(pulse_id='z')], id_check='sqlite')

    def test_B_interpolation_uses_actual_times_and_keeps_start_offset(self):
        write(self.path, [b_row(2, 142.), b_row(4, 142.2), b_row(12, 142.4)])
        track = read_b_tracks(self.path, self.config)['P1A', 'test', '1']
        self.assertEqual(track.times_ns, (2 * NS_PER_SECOND, 4 * NS_PER_SECOND, 12 * NS_PER_SECOND))
        self.assertAlmostEqual(track.position(3 * NS_PER_SECOND, 3 * NS_PER_SECOND)[0], 142.1)
        self.assertIsNone(track.position(8 * NS_PER_SECOND, 3 * NS_PER_SECOND))
        self.assertIsNone(track.position(NS_PER_SECOND, 3 * NS_PER_SECOND))
        self.assertIsNone(track.position(13 * NS_PER_SECOND, 3 * NS_PER_SECOND))
        self.assertEqual(track.position(12 * NS_PER_SECOND, 3 * NS_PER_SECOND), (142.4, 24., 2400.))

    def test_B_is_sorted_by_actual_timestamp_and_truth_ID_is_ignored(self):
        write(self.path, [b_row(4), b_row(2)])
        before = read_b_tracks(self.path, self.config)
        self.assertTrue(before['P1A', 'test', '1'].reordered)
        write(self.path, [b_row(4, target_id='wrong'), b_row(2, target_id='other')])
        self.assertEqual(before, read_b_tracks(self.path, self.config))

    def test_duplicate_B_timestamp_is_rejected(self):
        write(self.path, [b_row(2), b_row(2, 143.)])
        with self.assertRaisesRegex(ValueError, 'Duplicate timestamp'):
            read_b_tracks(self.path, self.config)

    def test_B_coordinate_aliases_are_supported(self):
        row = b_row(2)
        write(self.path, [row])
        original = read_b_tracks(self.path, self.config)
        row['target_longitude'] = row.pop('target_lontitude')
        write(self.path, [row])
        self.assertEqual(original, read_b_tracks(self.path, self.config))

    def test_bad_measurement_or_missing_observer_pose_is_rejected(self):
        for key, value in [('param1', 'NaN'), ('doa', '361'), ('sensor_latitude', '99')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read_a([a_row(**{key: value})])
        row = a_row()
        row.pop('sensor_height')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.read_a([row])


if __name__ == '__main__':
    unittest.main()
