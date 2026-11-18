#!/usr/bin/env python3
"""Translate old PRI settings into physical units and record a train-only review grid."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def review(baseline_path, audit_path, output_path):
    baseline = json.loads(Path(baseline_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    units = baseline['assumed_toa_units_per_second']
    if units <= 0:
        raise ValueError('Invalid baseline time unit')
    to_us = 1e6 / units
    low, high = baseline['min_pri_raw'] * to_us, baseline['max_pri_raw'] * to_us
    evidence = []
    for row in audit['classes']:
        peaks = [p['interval_us'] for p in row['most_frequent_observed_intervals']]
        evidence.append({'target_class': row['target_class'], 'observed_interval_peaks_us': peaks,
                         'baseline_search_covers_listed_peaks': all(low <= p <= high for p in peaks),
                         'RF_PW_compatible_interval_fraction': row['RF_PW_compatible_intervals'] / row['short_same_source_observed_intervals'],
                         'top32_single_bin_fraction': row['top32_single_bin_fraction']})
    result = {
        'status': 'unit_review_complete; candidate gates are NOT validated estimator settings',
        'baseline_sha256': hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
        'training_audit_sha256': hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
        'baseline_physical_units': {'min_pri_us': low, 'max_pri_us': high,
            'histogram_bin_us': baseline['histogram_bin_raw'] * to_us,
            'absolute_tolerance_us': baseline['absolute_tolerance_raw'] * to_us,
            'effective_histogram_tolerance_floor_us': baseline['histogram_bin_raw'] * to_us / 2,
            'stable_relative_tolerance': baseline['relative_tolerance'],
            'RF_gate_MHz': baseline['rf_gate'], 'PW_gate_us': baseline['pw_gate']},
        'unit_consistent_search_ns': {'min_pri_raw': round(low * 1000), 'max_pri_raw': round(high * 1000),
            'histogram_bin_raw': round(baseline['histogram_bin_raw'] * to_us * 1000),
            'assumed_toa_units_per_second': 1_000_000_000},
        'candidate_grid_for_next_step': {'RF_gate_MHz': [1., 3., 5., 8.], 'PW_gate_us': [.2, .3, .4],
            'histogram_bin_us': [.05, .1, .2], 'stable_relative_tolerance': [.001, .002, .005]},
        'training_observed_interval_evidence': evidence,
        'decisions': [
            'Normalize fractional microseconds to integer ns. Do not change the physical period by 1000x.',
            'After this unit conversion the baseline search spans 1 us to 10000 us and covers all listed diagnostic peaks.',
            'RF/PW gates are explicit physical units; do not treat 500 as ns or reuse it as a narrow RF gate.',
            'The RF/PW error-width gate retained only a subset of known same-source observed edges; mode changes and observation fluctuations require empirical validation.',
            'Stable and jittered period validation belongs to step 2. No final RF/PW gate or jitter tolerance is selected here.',
            'Observed intervals and top32 concentration are diagnostics, not official PRI labels or inference accuracy.']}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--training-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/pri_parameter_review.json')
    args = parser.parse_args()
    result = review(args.baseline_config, args.training_audit, args.output)
    print(json.dumps({'output': str(args.output), 'baseline_physical_units': result['baseline_physical_units'],
                      'all_listed_training_peaks_in_range': all(r['baseline_search_covers_listed_peaks'] for r in result['training_observed_interval_evidence'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
