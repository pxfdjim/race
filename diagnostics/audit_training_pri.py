#!/usr/bin/env python3
"""TRAIN-only observed-interval diagnostic; target labels never enter test inference."""
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path


def quantile(histogram, q):
    threshold = q * (sum(histogram.values()) - 1)
    total = 0
    for index, count in sorted(histogram.items()):
        total += count
        if total > threshold:
            return index / 10.


def audit(source, output):
    counts, selected_counts, platforms, states = Counter(), Counter(), Counter(), Counter()
    histograms, targets, previous, edges = defaultdict(Counter), defaultdict(set), {}, Counter()
    rows, selected, begin = 0, 0, time.monotonic()
    with Path(source).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        fields = ('scenario_id', 'data_version', 'platform_id', 'target_id', 'target_class',
                  'target_state', 'toa', 'param1', 'param2')
        ix = {name: header.index(name) for name in fields}
        for row in reader:
            rows += 1
            scene, version = row[ix['scenario_id']], row[ix['data_version']]
            if version != 'train' or not row[ix['target_id']] or not row[ix['target_class']]:
                raise ValueError('Only fully labeled TRAIN observations are accepted')
            platform, target, label = row[ix['platform_id']], row[ix['target_id']], row[ix['target_class']]
            counts[label] += 1
            platforms[platform] += 1
            states[row[ix['target_state']]] += 1
            targets[label].add((scene, target))
            toa_us = float(row[ix['toa']])
            if not math.isfinite(toa_us) or toa_us < 0:
                raise ValueError('Invalid TOA')
            seconds = toa_us / 1e6
            if rows % 2_000_000 == 0:
                print(json.dumps({'scanned_rows': rows, 'sampled_rows': selected}), flush=True)
            # Five seconds every fifty seconds, plus the final five seconds.
            if seconds % 50. >= 5. and not 495 <= seconds < 501:
                continue
            selected += 1
            selected_counts[label] += 1
            toa_ns = round(toa_us * 1000.)
            rf, pw = float(row[ix['param1']]), float(row[ix['param2']])
            key = scene, version, platform, target
            old = previous.get(key)
            previous[key] = toa_ns, rf, pw
            if old is None:
                continue
            delta_ns = toa_ns - old[0]
            if not 0 < delta_ns <= 5_000_000:
                continue
            edges[label] += 1
            # D3 error widths for TWO observations. Compatibility does not prove a CPI.
            if abs(rf - old[1]) <= 1.00001 and abs(pw - old[2]) <= .20001:
                histograms[label][(delta_ns + 50) // 100] += 1
    summaries = []
    for label in sorted(counts):
        hist = histograms[label]
        n = sum(hist.values())
        summaries.append({'target_class': label, 'full_train_pulses': counts[label],
            'observed_target_instances': len(targets[label]), 'sampled_pulses': selected_counts[label],
            'short_same_source_observed_intervals': edges[label], 'RF_PW_compatible_intervals': n,
            'occupied_0_1us_bins': len(hist),
            'top32_single_bin_fraction': sum(v for _, v in hist.most_common(32)) / n if n else None,
            'observed_interval_p10_us': quantile(hist, .1),
            'observed_interval_median_us': quantile(hist, .5),
            'observed_interval_p90_us': quantile(hist, .9),
            'most_frequent_observed_intervals': [{'interval_us': k / 10., 'count': v} for k, v in hist.most_common(12)]})
        assert n <= edges[label] <= selected_counts[label]
    report = {'purpose': 'TRAIN-only diagnosis; not a PRI inference method or official physical PRI truth.',
        'source': str(source), 'full_train_rows': rows, 'sampled_rows': selected,
        'platform_pulse_counts': dict(platforms), 'state_counts': dict(states), 'classes': summaries,
        'sampling': 'First five seconds of each fifty-second period; extra tail [495,501) seconds.',
        'units': 'Input TOA us; differences computed in rounded integer ns; histogram bins 0.1 us.',
        'maximum_observed_interval_us': 5000.,
        'RF_PW_compatibility': 'abs(delta RF)<=1.00001 MHz; abs(delta PW)<=0.20001 us.',
        'limitations': ['TRAIN target_id isolates sources only for diagnosis; it is not a test-time feature.',
            'Intervals are actual adjacent observations within sampled source streams; misses and mode transitions may be included.',
            'Top32 concentration is not the coverage or accuracy of the old estimator.',
            'No true CPI boundaries or physical PRI labels are inferred.'],
        'elapsed_seconds': round(time.monotonic() - begin, 2)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'training_pri_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 新版训练数据的 PRI 前置诊断', '',
        'target_id 只用于诊断时隔离训练真实源，不是测试时输入，也不提供官方物理 PRI 真值。', '',
        f'扫描 {rows:,} 条训练脉冲，分时段选取 {selected:,} 条；各平台独立作同源时间差。', '',
        '| 型号 | 训练实例数 | 样本脉冲 | RF/PW相容短间隔 | 0.1μs桶数 | 前32单桶支持比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        fraction = f"{s['top32_single_bin_fraction']:.1%}" if s['top32_single_bin_fraction'] is not None else '—'
        lines.append(f"| {s['target_class']} | {s['observed_target_instances']} | {s['sampled_pulses']:,} | {s['RF_PW_compatible_intervals']:,} | {s['occupied_0_1us_bins']:,} | {fraction} |")
    lines += ['', '前32单桶支持比例仅描述间隔分布的集中程度，不是 PRI 提取准确率。统计已限制为短间隔且 RF/PW 相容，但仍可能包含漏收和模式切换。', '',
        '没有识别真实 CPI 边界，没有使用测试标签，没有改写现有模型、代码包或提交结果。完整峰值和口径见 training_pri_audit.json。', '']
    (output / 'PRI诊断.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-a', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    args = parser.parse_args()
    result = audit(args.train_a, args.output_dir)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
