import csv
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from estimate_pri import extract_file
from evaluate_pri import evaluate
from src.dataio import InputConfig, PulseObservation
from src.pri import PRIConfig, WindowedPRI


def pulse(toa, number, platform='p1', rf=100., pw=1., doa=10.):
    return PulseObservation('P1A', 'train', platform, str(number), toa, rf, pw, -30.,
                            doa, (140., 24., 0.), str(toa / 1000), '', toa, toa, 1)


def run(pulses, config=None):
    rows, sequences = [], []
    runner = WindowedPRI(config or PRIConfig(), sequences.append,
                         lambda r: rows.append((r.pulse, r.assignment)))
    for p in sorted(pulses, key=lambda p: (p.toa, p.pulse_id)):
        runner.feed(p)
    runner.finish()
    return rows, sequences, runner


class PRIExtractionTests(unittest.TestCase):
    def test_stable_physical_period_and_nanosecond_precision(self):
        rows, seqs, _ = run([pulse(31927 + i * 100013, i) for i in range(48)])
        self.assertEqual(len(rows), 48)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(seqs[0]['pri_center_ns'], 100013)
        self.assertEqual(seqs[0]['pri_model'], 'stable')

    def test_four_platforms_and_two_radio_sources_do_not_mix(self):
        pulses = [pulse(i * 100000 + platform * 10 + source, f'{platform}-{source}-{i}',
                        platform=str(platform), rf=100. + source * 100.)
                  for i in range(32) for platform in range(4) for source in range(2)]
        rows, seqs, _ = run(pulses)
        groups = {}
        for p, a in rows:
            self.assertIsNotNone(a)
            groups.setdefault(p.scope + (a['sequence_id'],), set()).add(p.rf)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(len(v) == 1 for v in groups.values()))
        self.assertEqual(len(seqs), 8)

    def test_bounded_jitter_is_inferred_without_class_labels(self):
        rng, time, pulses = random.Random(117), 0, []
        for i in range(128):
            pulses.append(pulse(time, i, doa=(359. + rng.uniform(-1., 1.)) % 360))
            time += 400000 + rng.randint(-20000, 20000)
        rows, seqs, _ = run(pulses)
        self.assertGreaterEqual(sum(a is not None for _, a in rows), 120)
        self.assertTrue(any(s['pri_model'] == 'jittered' for s in seqs))
        self.assertAlmostEqual(seqs[0]['pri_center_us'], 400., delta=5.)

    def test_dropped_pulses_keep_observed_interval_and_integer_multiple(self):
        times = [i * 200000 for i in range(50) if i not in (12, 24, 36)]
        rows, seqs, _ = run([pulse(t, i) for i, t in enumerate(times)])
        self.assertEqual(sum(s['pulse_count'] for s in seqs), len(times))
        self.assertEqual(seqs[0]['pri_center_ns'], 200000)
        doubled = [a for _, a in rows if a and a['interval_multiple'] == 2]
        self.assertEqual(len(doubled), 3)
        self.assertTrue(all(a['observed_interval_ns'] == 400000 for a in doubled))

    def test_interleaved_different_periods_use_graph_fallback(self):
        pulses = [pulse(i * 100000, f'a{i}') for i in range(50)]
        pulses += [pulse(37000 + i * 173000, f'b{i}') for i in range(29)]
        rows, seqs, _ = run(pulses)
        grouped = {}
        for p, a in rows:
            if a:
                grouped.setdefault(a['sequence_id'], set()).add(p.pulse_id[0])
        self.assertGreater(sum(a is not None for _, a in rows), 60)
        self.assertTrue(all(len(v) == 1 for v in grouped.values()))
        self.assertTrue(any(s['basis'] == 'multi_order_graph_validation' for s in seqs))

    def test_mode_changes_produce_different_short_sequences(self):
        pulses = [pulse(i * 100000, i) for i in range(32)]
        pulses += [pulse(4000000 + i * 150000, 32 + i) for i in range(32)]
        _, seqs, _ = run(pulses)
        self.assertEqual({s['pri_center_ns'] for s in seqs}, {100000., 150000.})
        self.assertEqual(sum(s['pulse_count'] for s in seqs), 64)

    def test_insufficient_evidence_is_unknown_and_never_dropped(self):
        rows, seqs, runner = run([pulse(i * 100000, i) for i in range(5)])
        self.assertEqual(len(rows), 5)
        self.assertEqual(seqs, [])
        self.assertTrue(all(a is None for _, a in rows))
        self.assertEqual(runner.input_count, runner.output_count)

    def test_lookahead_handles_boundary_without_double_claiming(self):
        pulses = [pulse(499600000 + i * 100000, i) for i in range(48)]
        pulses.append(pulse(800000000, 'isolated', rf=200.))
        rows, seqs, runner = run(pulses)
        self.assertEqual(len(rows), 49)
        self.assertEqual(sum(s['pulse_count'] for s in seqs), 48)
        self.assertEqual(len({p.pulse_id for p, _ in rows}), 49)
        self.assertGreaterEqual(runner.windows, 2)

    def test_maximum_period_is_supported(self):
        _, seqs, _ = run([pulse(i * 10000000, i) for i in range(16)])
        self.assertEqual(seqs[0]['pri_center_us'], 10000.)

    def test_unsorted_or_reappearing_scene_is_rejected(self):
        runner = WindowedPRI(PRIConfig(), lambda s: None, lambda p: None)
        runner.feed(pulse(1000000, 1))
        with self.assertRaisesRegex(ValueError, 'sorted'):
            runner.feed(pulse(0, 2))
        runner = WindowedPRI(PRIConfig(), lambda s: None, lambda p: None)
        runner.feed(pulse(0, 1))
        runner.feed(replace(pulse(0, 2), scenario='scene2'))
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            runner.feed(pulse(0, 3))

    def test_configuration_rejects_unsupported_ranges(self):
        for cfg in (replace(PRIConfig(), histogram_bin_us=0),
                    replace(PRIConfig(), lookahead_seconds=.001),
                    replace(PRIConfig(), pool_max_gap_us=1)):
            with self.assertRaises(ValueError):
                cfg.validate()


class PRIFileTests(unittest.TestCase):
    def test_batch_invariance_label_isolation_and_truth_evaluation(self):
        from test_dataio import a_row, write, CONFIG
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'A.csv'
            rows = [a_row(i + 1, toa=str(31.927 + i * 100),
                     datetime=f'2026-11-11 09:00:00.{int((31.927+i*100)//1000):03d}',
                     target_id='emitter', target_class='R01') for i in range(48)]
            write(source, rows)
            ic = InputConfig.from_dict(CONFIG)
            extract_file(source, directory / 'first', ic, PRIConfig(), batch_size=7, progress=False)
            for row in rows:
                for key in ('target_id', 'target_class', 'target_state'):
                    row.pop(key)
            write(source, rows)
            extract_file(source, directory / 'second', ic, PRIConfig(), batch_size=31, progress=False)
            for name in ('pulse_pri.csv', 'pri_sequences.csv'):
                self.assertEqual((directory / 'first' / name).read_bytes(),
                                 (directory / 'second' / name).read_bytes())
            for row in rows:
                row.update(target_id='emitter', target_class='R01')
            write(source, rows)
            report = evaluate(source, directory / 'first/pulse_pri.csv')
            self.assertEqual(report['weighted_sequence_purity'], 1.)
            self.assertEqual(report['same_source_edge_fraction'], 1.)
            self.assertEqual(report['supported_pulses'], 48)
            self.assertTrue((directory / 'first/pri_run.json').is_file())


if __name__ == '__main__':
    unittest.main()
