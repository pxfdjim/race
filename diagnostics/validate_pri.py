#!/usr/bin/env python3
"""Fixed training time-window development/validation split; labels are evaluation only."""
import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estimate_pri import extract_file
from evaluate_pri import evaluate
from prepare_inputs import digest
from src.dataio import InputConfig, time_to_ns
from src.pri import PRIConfig

WINDOWS = {'development': [(0, 1), (100, 101), (200, 201)],
           'validation': [(300, 301), (400, 401), (495, 496)]}


def cache(source, destination):
    destination.mkdir(parents=True)
    streams = []
    counts = {k: 0 for k in WINDOWS}
    try:
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            keep = [i for i, name in enumerate(header) if name not in ('target_id', 'target_class', 'target_state')]
            time_column = header.index('toa')
            writers = {}
            for split in WINDOWS:
                a = (destination / f'{split}_observations.csv').open('w', encoding='utf-8', newline='')
                b = (destination / f'{split}_truth.csv').open('w', encoding='utf-8', newline='')
                streams.extend((a, b))
                wa, wb = csv.writer(a), csv.writer(b)
                wa.writerow([header[i] for i in keep])
                wb.writerow(header)
                writers[split] = wa, wb
            bounds = [(a * 1_000_000_000, b * 1_000_000_000, split)
                      for split, windows in WINDOWS.items() for a, b in windows]
            for row in reader:
                time = time_to_ns(row[time_column], 'us')
                for start, end, split in bounds:
                    if start <= time < end:
                        wa, wb = writers[split]
                        wa.writerow([row[i] for i in keep])
                        wb.writerow(row)
                        counts[split] += 1
                        break
    finally:
        for stream in streams:
            stream.close()
    (destination / 'cache_manifest.json').write_text(json.dumps({
        'source': str(source.resolve()), 'source_sha256': digest(source),
        'windows_seconds': WINDOWS, 'counts': counts, 'selection_uses_labels': False,
        'limitation': 'Fixed time-window pilot, not a random or independent-scene benchmark.'},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/pri_validation')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if not (directory / 'cache/cache_manifest.json').exists():
        cache(args.source, directory / 'cache')
    else:
        manifest = json.loads((directory / 'cache/cache_manifest.json').read_text())
        if manifest['source_sha256'] != digest(args.source):
            raise ValueError('Cached source hash differs')
    if args.cache_only:
        return
    ic = InputConfig.from_dict(json.loads((ROOT / 'config/input_units.json').read_text()))
    base = PRIConfig(**json.loads((ROOT / 'config/pri_config.json').read_text()))
    results = []

    def ensure_run(observations, destination, config):
        if not destination.exists():
            extract_file(observations, destination, ic, config)
            return
        saved = json.loads((destination / 'pri_run.json').read_text())
        if saved['pri_config'] != asdict(config) or saved['source_sha256'] != digest(observations):
            raise ValueError(f'Existing pilot uses different inputs/parameters: {destination}; choose a new output directory')
        for name, detail in saved['files'].items():
            if digest(destination / name) != detail['sha256']:
                raise ValueError(f'Existing pilot output changed: {destination / name}')

    for gate in (3., 5., 8.):
        cfg = replace(base, rf_gate_mhz=gate)
        dest = directory / f'development_rf{int(gate)}'
        ensure_run(directory / 'cache/development_observations.csv', dest, cfg)
        metrics = evaluate(directory / 'cache/development_truth.csv', dest / 'pulse_pri.csv')
        results.append({'config': asdict(cfg), 'metrics': metrics})
        print(json.dumps({'gate': gate, 'coverage': metrics['supported_fraction'],
                         'purity': metrics['weighted_sequence_purity']}, ensure_ascii=False), flush=True)
    # Predeclared criterion: prioritize >=99.5% development purity, then coverage.
    eligible = [r for r in results if (r['metrics']['weighted_sequence_purity'] or 0) >= .995]
    winner = max(eligible or results, key=lambda r: (r['metrics']['supported_fraction'] if eligible else
                   r['metrics']['weighted_sequence_purity'], r['metrics']['supported_fraction'],
                   -r['config']['rf_gate_mhz']))
    cfg = PRIConfig(**winner['config'])
    dest = directory / 'validation'
    ensure_run(directory / 'cache/validation_observations.csv', dest, cfg)
    heldout = evaluate(directory / 'cache/validation_truth.csv', dest / 'pulse_pri.csv')
    report = {'windows_seconds': WINDOWS, 'selection_rule': 'Development purity >=0.995, then maximum coverage; '
              'if none qualifies, highest purity then coverage; ties prefer smaller RF gate.',
              'development_runs': results, 'selected_config': asdict(cfg),
              'heldout_validation': heldout, 'test_labels_used': False,
              'physical_PRI_accuracy_claimed': False}
    (directory / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'selected_pri_config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(heldout, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
