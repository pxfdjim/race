import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PreparationCLITests(unittest.TestCase):
    def test_full_input_count_is_preserved_when_only_A_sample_is_exported(self):
        self.run_case(False)

    def test_full_export_has_integer_ns_raw_TOA_and_no_A_truth_columns(self):
        self.run_case(True)

    def run_case(self, full):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'A.csv', root / 'B.csv', root / 'prepared'
            with a.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['scenario_id', 'data_version', 'pulse_id', 'datetime', 'toa', 'param1', 'param2',
                            'param3', 'doa', 'sensor_lontitude', 'sensor_latitude', 'sensor_height', 'platform_id', 'target_id'])
                for i in range(5):
                    w.writerow(['P1A', 'test', i + 1, '2026-11-11 09:00:00.000', '31.927', 28939.6,
                                .6, 40.5, .8, 142.1, 24.6, 0, ('103001', '103002', '104001', '104002')[i % 4], 'hidden'])
            with b.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['scenario_id', 'data_version', 'track_id', 'datetime', 'target_lontitude',
                            'target_latitude', 'target_height', 'target_type', 'target_class', 'target_id'])
                for second in (2, 4, 12):
                    w.writerow(['P1A', 'test', '1', f'2026-11-11 09:00:{second:02d}', 142.1, 24.6, 2400, 'medium', '', 'hidden'])
            original = hashlib.sha256(a.read_bytes()).hexdigest()
            command = [sys.executable, str(ROOT / 'prepare_inputs.py'), '--data-root', str(root), '--split', 'test',
                       '--a-file', str(a), '--b-file', str(b), '--output-dir', str(out), '--sample-rows', '2', '--batch-size', '2']
            if full:
                command.append('--write-normalized')
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / 'input_validation.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['datasets']['test']['A']['input_rows'], 5)
            self.assertEqual(report['datasets']['test']['A']['platform_count'], 4)
            self.assertEqual(report['datasets']['test']['A']['sample_rows'], 2)
            self.assertEqual(report['datasets']['test']['B']['tracks'][0]['gap_counts_seconds'], {'2.0': 1, '8.0': 1})
            self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(), original)
            with (out / 'test_A_sample.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['toa'], '31927')
            self.assertEqual(rows[0]['toa_raw'], '31.927')
            self.assertNotIn('target_id', rows[0])
            self.assertNotIn('target_class', rows[0])
            if full:
                with (out / 'test_A_normalized.csv').open() as f:
                    complete = list(csv.DictReader(f))
                self.assertEqual(len(complete), 5)
                self.assertEqual([r['pulse_id'] for r in complete], ['1', '2', '3', '4', '5'])
            else:
                self.assertFalse((out / 'test_A_normalized.csv').exists())


if __name__ == '__main__':
    unittest.main()
