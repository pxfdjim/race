"""Ordered, observation-only context for short sequences in the zgy experiment."""

from collections import defaultdict
import math


NEIGHBORS = 3
MAX_GAP_NS = 250_000_000
TIME_BLOCKS_SECONDS = (
    ('development', 0, 300), ('check_300_320', 300, 320), ('selection', 320, 400),
    ('check_400_420', 400, 420), ('diagnostic_420_500', 420, 500),
)
SLOT_FIELDS = ('present', 'log_gap_ms', 'log_pulse_count_ratio', 'log_duration_ratio', 'pa_mean_difference_db')
CONTEXT_FEATURES = tuple(f'{side}_{i + 1}_{feature}' for side in ('previous', 'next')
                         for i in range(NEIGHBORS) for feature in SLOT_FIELDS) + (
                             'previous_count', 'next_count', 'log_observed_neighborhood_span_ms')


def time_block(window):
    for name, begin, end in TIME_BLOCKS_SECONDS:
        if begin * 1e9 <= window.first <= window.last < end * 1e9:
            return name
    return None


def ordered_context(windows, feature_names, confirmed_local_groups, *, partition_by_time):
    """Use adjacent observations from the same platform and source, with explicit masks."""
    amp = feature_names.index('observed_PA_dbm')
    groups = defaultdict(list)
    missing = (0.,) * len(CONTEXT_FEATURES)
    output = {}
    for key, w in windows.items():
        scene, version, platform, kind, source_bin, _ = key
        source, sep, _ = source_bin.rpartition('~')
        if not sep or not source:
            raise ValueError(f'Expected a short-sequence source key: {key}')
        block = time_block(w) if partition_by_time else 'test'
        if block is None or kind not in ('object', 'local') or (
                kind == 'local' and (scene, version, platform, source) not in confirmed_local_groups):
            output[key] = missing
        else:
            groups[scene, version, platform, kind, source, block].append(w)

    for group in groups.values():
        group.sort(key=lambda w: (w.first, w.last, w.key))
        for index, center in enumerate(group):
            slots = []
            visible, left, right = [], center.first, center.last
            for direction in (-1, 1):
                current = center
                used = 0
                for offset in range(1, NEIGHBORS + 1):
                    j = index + offset * direction
                    if not 0 <= j < len(group):
                        break
                    neighbor = group[j]
                    gap = (current.first - neighbor.last if direction < 0 else neighbor.first - current.last)
                    if not 0 <= gap <= MAX_GAP_NS:
                        break
                    slots.extend((1., math.log1p(gap / 1e6), math.log(neighbor.count / center.count),
                                  math.log((neighbor.last - neighbor.first + 1e6) /
                                           (center.last - center.first + 1e6)),
                                  neighbor.features[amp] - center.features[amp]))
                    left = min(left, neighbor.first)
                    right = max(right, neighbor.last)
                    current = neighbor
                    used += 1
                slots.extend((0.,) * ((NEIGHBORS - used) * len(SLOT_FIELDS)))
                visible.append(used)
            slots += [*visible, math.log1p((right - left) / 1e6) if any(visible) else 0.]
            output[center.key] = tuple(slots)
    if len(output) != len(windows):
        raise ValueError('A record was lost during the context join')
    return output
