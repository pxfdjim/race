"""Label-free local PRI extraction: stable chains, bounded jitter and graph fallback."""
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import median


def angle_delta(a, b):
    return (a - b + 180.) % 360. - 180.


def quantile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    i = int(x)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)


@dataclass(frozen=True)
class PRIConfig:
    window_seconds: float = .5
    lookahead_seconds: float = .08
    rf_gate_mhz: float = 8.
    pw_gate_us: float = .25
    doa_pair_gate_degrees: float = 7.
    pool_doa_gate_degrees: float = 4.5
    pool_warmup_doa_gate_degrees: float = 6.5
    pool_doa_span_degrees: float = 9.
    pool_max_gap_us: float = 30000.
    min_pri_us: float = 1.
    max_pri_us: float = 10000.
    histogram_bin_us: float = .1
    difference_order: int = 4
    min_candidate_votes: int = 6
    max_candidates: int = 12
    absolute_tolerance_us: float = .05
    stable_relative_tolerance: float = .001
    max_interval_multiple: int = 3
    min_sequence_pulses: int = 8
    min_direct_fraction: float = .6
    jitter_min_model_pulses: int = 16
    jitter_min_relative_spread: float = .012
    jitter_max_relative_half_width: float = .12
    jitter_min_adjacent_support: float = .85
    jitter_max_peak_fraction: float = .25

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('difference_order', 'min_candidate_votes', 'max_candidates', 'max_interval_multiple',
                    'min_sequence_pulses', 'jitter_min_model_pulses'):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f'{key} must be an integer')
        if self.min_pri_us >= self.max_pri_us or self.min_sequence_pulses < 3:
            raise ValueError('Invalid PRI range or minimum sequence size')
        if self.pool_max_gap_us < self.max_pri_us:
            raise ValueError('Radio neighborhood gap must support the maximum PRI')
        for key in ('min_direct_fraction', 'jitter_min_adjacent_support', 'jitter_max_peak_fraction'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'Invalid fraction {key}')
        if self.lookahead_seconds * 1e6 < (self.min_sequence_pulses - 1) * self.max_pri_us:
            raise ValueError('Lookahead must support at least one minimum-length sequence at maximum PRI')


@dataclass(slots=True)
class PendingPulse:
    index: int
    pulse: object
    assignment: dict | None = None


@dataclass
class RadioPool:
    items: list
    anchor: float
    rf_sum: float
    pw_sum: float
    angle_sum: float
    min_rf: float
    max_rf: float
    min_pw: float
    max_pw: float
    min_angle: float
    max_angle: float
    last_time: int

    @classmethod
    def create(cls, item):
        p = item.pulse
        return cls([item], p.doa, p.rf, p.pw, p.doa, p.rf, p.rf, p.pw, p.pw,
                   p.doa, p.doa, p.toa)

    def add(self, item):
        p = item.pulse
        angle = self.anchor + angle_delta(p.doa, self.anchor)
        self.items.append(item)
        self.rf_sum += p.rf
        self.pw_sum += p.pw
        self.angle_sum += angle
        self.min_rf, self.max_rf = min(self.min_rf, p.rf), max(self.max_rf, p.rf)
        self.min_pw, self.max_pw = min(self.min_pw, p.pw), max(self.max_pw, p.pw)
        self.min_angle, self.max_angle = min(self.min_angle, angle), max(self.max_angle, angle)
        self.last_time = p.toa


def make_pools(items, config):
    """Coarse neighborhoods only; they are neither CPI IDs nor target identities."""
    pools, index = [], defaultdict(list)
    width = 2 * config.rf_gate_mhz
    for item in sorted(items, key=lambda r: (r.pulse.toa, r.index)):
        if item.assignment is not None:
            continue
        p = item.pulse
        bucket = math.floor(p.rf / width)
        best = None
        for offset in (-1, 0, 1):
            for number in index.get(p.scope + (bucket + offset,), ()):
                pool = pools[number]
                if p.toa - pool.last_time > config.pool_max_gap_us * 1000:
                    continue
                n = len(pool.items)
                drf, dpw = abs(p.rf - pool.rf_sum / n), abs(p.pw - pool.pw_sum / n)
                angle = pool.anchor + angle_delta(p.doa, pool.anchor)
                da = abs(angle - pool.angle_sum / n)
                gate = config.pool_warmup_doa_gate_degrees if n < 8 else config.pool_doa_gate_degrees
                if drf > config.rf_gate_mhz or dpw > config.pw_gate_us or da > gate:
                    continue
                if (max(pool.max_rf, p.rf) - min(pool.min_rf, p.rf) > 2 * config.rf_gate_mhz or
                    max(pool.max_pw, p.pw) - min(pool.min_pw, p.pw) > 2 * config.pw_gate_us or
                    max(pool.max_angle, angle) - min(pool.min_angle, angle) > config.pool_doa_span_degrees):
                    continue
                cost = drf / config.rf_gate_mhz + dpw / config.pw_gate_us + da / gate
                if best is None or cost < best[0]:
                    best = cost, number
        if best is None:
            index[p.scope + (bucket,)].append(len(pools))
            pools.append(RadioPool.create(item))
        else:
            pools[best[1]].add(item)
    return pools


def compatible(a, b, config):
    return (abs(a.rf - b.rf) <= config.rf_gate_mhz and abs(a.pw - b.pw) <= config.pw_gate_us
            and abs(angle_delta(a.doa, b.doa)) <= config.doa_pair_gate_degrees)


def edge_cost(a, b, config):
    return (.15 * abs(a.rf - b.rf) / config.rf_gate_mhz +
            .15 * abs(a.pw - b.pw) / config.pw_gate_us +
            .2 * abs(angle_delta(a.doa, b.doa)) / config.doa_pair_gate_degrees +
            .02 * min(abs(a.pa - b.pa) / 10., 3.))


def tolerance(period, config):
    return max(config.absolute_tolerance_us * 1000, period * config.stable_relative_tolerance)


def stable_runs(items, config, emit):
    """Linear first pass for clean stable runs; failed starts do not claim pulses."""
    i, n = 0, len(items)
    while i + 1 < n:
        if items[i].assignment or items[i + 1].assignment:
            i += 1
            continue
        period = items[i + 1].pulse.toa - items[i].pulse.toa
        if not config.min_pri_us * 1000 <= period <= config.max_pri_us * 1000:
            i += 1
            continue
        path, multiples, j = [items[i]], [], i + 1
        while j < n and items[j].assignment is None:
            left, right = path[-1].pulse, items[j].pulse
            dt = right.toa - left.toa
            m = round(dt / period)
            if not 1 <= m <= config.max_interval_multiple or abs(dt - m * period) > m * tolerance(period, config) or not compatible(left, right, config):
                break
            path.append(items[j])
            multiples.append(m)
            j += 1
        if len(path) >= config.min_sequence_pulses and multiples.count(1) / len(multiples) >= config.min_direct_fraction:
            emit(path, multiples, 'stable', 'consecutive_interval_validation')
            i = j
        else:
            i += 1


def jitter_hypothesis(items, config):
    free = [r for r in items if r.assignment is None]
    if len(free) < config.jitter_min_model_pulses:
        return None
    values = [b.pulse.toa - a.pulse.toa for a, b in zip(free, free[1:])]
    values = [v for v in values if config.min_pri_us * 1000 <= v <= config.max_pri_us * 1000]
    if len(values) < config.jitter_min_model_pulses - 1:
        return None
    center = median(values)
    direct = [v for v in values if .5 * center <= v <= 1.5 * center]
    if len(direct) / len(values) < config.jitter_min_adjacent_support:
        return None
    center = median(direct)
    spread = (quantile(direct, .9) - quantile(direct, .1)) / center
    if not config.jitter_min_relative_spread <= spread <= 2 * config.jitter_max_relative_half_width:
        return None
    width = max(2 * tolerance(center, config), 1.)
    bins = Counter(round(v / width) for v in direct)
    if len(bins) < 6 or max(bins.values()) / len(direct) > config.jitter_max_peak_fraction:
        return None
    half = max(center - quantile(direct, .01), quantile(direct, .99) - center) * 1.15 + config.absolute_tolerance_us * 1000
    if half > config.jitter_max_relative_half_width * center:
        return None
    return center, half


def graph_paths(items, period, half_width, model, config, emit):
    n = len(items)
    times = [r.pulse.toa for r in items]
    successor, hop, length, direct = [-1] * n, [0] * n, [0] * n, [0] * n
    for i in range(n - 1, -1, -1):
        if items[i].assignment is not None:
            continue
        length[i], best = 1, 1.
        a = items[i].pulse
        for m in range(1, config.max_interval_multiple + 1):
            center = times[i] + m * period
            bound = m * half_width
            left = bisect.bisect_left(times, center - bound, i + 1)
            right = bisect.bisect_right(times, center + bound, left)
            for j in range(left, right):
                if items[j].assignment is not None or not compatible(a, items[j].pulse, config):
                    continue
                error = abs(times[j] - center) / bound
                quality = 1 + length[j] - .1 * (m - 1) - .1 * error - edge_cost(a, items[j].pulse, config)
                if quality > best:
                    best, successor[i], hop[i] = quality, j, m
                    length[i], direct[i] = 1 + length[j], direct[j] + (m == 1)
    roots = [i for i in range(n) if length[i] >= config.min_sequence_pulses]
    roots.sort(key=lambda i: (-length[i], -direct[i], times[i]))
    for i in roots:
        path, multiples = [], []
        while i >= 0 and items[i].assignment is None:
            path.append(items[i])
            j = successor[i]
            if j < 0 or items[j].assignment is not None:
                break
            multiples.append(hop[i])
            i = j
        if len(path) < config.min_sequence_pulses or multiples.count(1) / len(multiples) < config.min_direct_fraction:
            continue
        refined = median((b.pulse.toa - a.pulse.toa) / m for a, b, m in zip(path, path[1:], multiples))
        if abs(refined - period) > half_width:
            continue
        emit(path, multiples, model, 'bounded_jitter_validation' if model == 'jittered' else 'multi_order_graph_validation')


def candidate_periods(items, config):
    free = [r.pulse for r in items if r.assignment is None]
    votes, scores = Counter(), defaultdict(float)
    width = config.histogram_bin_us * 1000
    for j, b in enumerate(free):
        for lag in range(1, min(config.difference_order, j) + 1):
            a = free[j - lag]
            dt = b.toa - a.toa
            if dt > config.max_pri_us * 1000:
                break
            if dt < config.min_pri_us * 1000 or not compatible(a, b, config):
                continue
            period = round(dt / width) * width
            if period < config.min_pri_us * 1000:
                continue
            votes[period] += 1
            scores[period] += 1 / lag ** 2
    ranked = sorted((t for t in votes if votes[t] >= config.min_candidate_votes), key=lambda t: (-scores[t], t))
    return sorted(ranked[:config.max_candidates])


class PRIExtractor:
    def __init__(self, config, on_sequence):
        config.validate()
        self.config, self.on_sequence = config, on_sequence
        self.numbers, self.statistics = Counter(), Counter()

    def emit(self, path, multiples, model, basis):
        scope = path[0].pulse.scope
        if any(r.assignment is not None or r.pulse.scope != scope for r in path):
            raise RuntimeError('Sequence reclaims pulses or crosses a platform')
        intervals = [(b.pulse.toa - a.pulse.toa) / m for a, b, m in zip(path, path[1:], multiples)]
        center, low, high = median(intervals), quantile(intervals, .05), quantile(intervals, .95)
        direct = multiples.count(1) / len(multiples)
        confidence = min(.99, .6 + .3 * min(len(path) / 32, 1.) + .09 * direct)
        self.numbers[scope] += 1
        name = f'S{self.numbers[scope]:07d}'
        for i, item in enumerate(path):
            item.assignment = {'sequence_id': name, 'pri_model': model, 'pri_center_ns': center,
                'pri_center_us': center / 1000, 'pri_p05_us': low / 1000, 'pri_p95_us': high / 1000,
                'confidence': confidence, 'previous_pulse_id': path[i - 1].pulse.pulse_id if i else '',
                'observed_interval_ns': item.pulse.toa - path[i - 1].pulse.toa if i else '',
                'interval_multiple': multiples[i - 1] if i else '', 'status': 'sequence_supported'}
        first = path[0].pulse
        angles = [first.doa + angle_delta(r.pulse.doa, first.doa) for r in path]
        mean_angle = sum(angles) / len(angles)
        summary = dict(zip(('scenario_id', 'data_version', 'platform_id'), scope),
            sequence_id=name, pri_model=model, pulse_count=len(path), start_toa_ns=first.toa,
            end_toa_ns=path[-1].pulse.toa, pri_center_ns=center, pri_center_us=center / 1000,
            pri_p05_us=low / 1000, pri_p95_us=high / 1000, direct_interval_fraction=direct,
            confidence=confidence, rf_mean=sum(r.pulse.rf for r in path) / len(path),
            pw_mean=sum(r.pulse.pw for r in path) / len(path), pa_mean=sum(r.pulse.pa for r in path) / len(path),
            doa_mean_degrees=mean_angle % 360., doa_std_degrees=math.sqrt(sum((v - mean_angle) ** 2 for v in angles) / len(angles)),
            max_relative_interval_deviation=max(abs(v - center) for v in intervals) / center, basis=basis)
        self.statistics[model + '_sequences'] += 1
        self.statistics[model + '_pulses'] += len(path)
        self.on_sequence(summary)

    def process(self, pending):
        pools = make_pools(pending, self.config)
        self.statistics['radio_neighborhoods'] += len(pools)
        for pool in pools:
            items = pool.items
            if len(items) < self.config.min_sequence_pulses:
                continue
            stable_runs(items, self.config, self.emit)
            hypothesis = jitter_hypothesis(items, self.config)
            if hypothesis:
                graph_paths(items, *hypothesis, 'jittered', self.config, self.emit)
            if sum(r.assignment is None for r in items) < self.config.min_sequence_pulses:
                continue
            for period in candidate_periods(items, self.config):
                graph_paths(items, period, tolerance(period, self.config), 'stable', self.config, self.emit)
                if sum(r.assignment is None for r in items) < self.config.min_sequence_pulses:
                    break
            hypothesis = jitter_hypothesis(items, self.config)
            if hypothesis:
                graph_paths(items, *hypothesis, 'jittered', self.config, self.emit)


class WindowedPRI:
    """Fixed time windows independent of reader batches; assignments are immutable.

    A lookahead pulse can support a sequence in the preceding core window. It is
    retained until its own core is finalized, with its assignment unchanged.
    """
    def __init__(self, config, on_sequence, on_pulse):
        self.extractor = PRIExtractor(config, on_sequence)
        self.on_pulse = on_pulse
        self.window_ns = round(config.window_seconds * 1e9)
        self.lookahead_ns = round(config.lookahead_seconds * 1e9)
        if self.window_ns < 1 or self.lookahead_ns < 1:
            raise ValueError('Window durations must be representable as positive ns')
        self.pending, self.closed_scenes = [], set()
        self.scene, self.last_time, self.core_end = None, None, None
        self.input_count, self.output_count, self.max_pending, self.windows = 0, 0, 0, 0
        self.finished = False

    def flush(self, final=False):
        if not self.pending:
            return
        self.extractor.process(self.pending)
        self.windows += 1
        retained = []
        for item in self.pending:
            if final or item.pulse.toa < self.core_end:
                self.on_pulse(item)
                self.output_count += 1
            else:
                retained.append(item)
        self.pending = retained

    def feed(self, pulse):
        if self.finished:
            raise RuntimeError('Cannot append after finish')
        scene = pulse.scenario, pulse.version
        if scene != self.scene:
            self.flush(final=True)
            if self.scene is not None:
                self.closed_scenes.add(self.scene)
            if scene in self.closed_scenes:
                raise ValueError('Scene/version must be contiguous in the input')
            self.scene, self.last_time = scene, None
            self.core_end = (pulse.toa // self.window_ns + 1) * self.window_ns
        if self.last_time is not None and pulse.toa < self.last_time:
            raise ValueError('PRI input must be sorted by TOA within each scene/version')
        self.last_time = pulse.toa
        while pulse.toa >= self.core_end + self.lookahead_ns:
            self.flush()
            self.core_end += self.window_ns
            if not self.pending and pulse.toa >= self.core_end + self.lookahead_ns:
                self.core_end = (pulse.toa // self.window_ns + 1) * self.window_ns
        self.pending.append(PendingPulse(self.input_count, pulse))
        self.input_count += 1
        self.max_pending = max(self.max_pending, len(self.pending))

    def finish(self):
        if self.finished:
            return
        self.flush(final=True)
        if self.input_count != self.output_count:
            raise RuntimeError('PRI output coverage check failed')
        self.finished = True
