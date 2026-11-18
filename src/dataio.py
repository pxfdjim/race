"""Streaming, label-free observations with explicit units and actual B timestamps."""
from __future__ import annotations

import bisect
import csv
import math
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

NS_PER_SECOND = 1_000_000_000
UNIT_NS = {'s': 1_000_000_000, 'ms': 1_000_000, 'us': 1000, 'ns': 1}
STAMP = re.compile(r'^(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})?$')


def time_to_ns(text, unit):
    """Exact decimal conversion. Never truncate fractional microseconds."""
    if unit not in UNIT_NS:
        raise ValueError(f'Unsupported time unit: {unit}')
    text = text.strip()
    whole, point, fraction = text.partition('.')
    if unit == 'us' and whole.isascii() and whole.isdigit() and (
            not point or fraction.isascii() and fraction.isdigit()):
        if fraction[3:].strip('0'):
            raise ValueError('TOA has sub-nanosecond precision; explicit rounding policy is required')
        return int(whole) * 1000 + int((fraction[:3] + '000')[:3])
    try:
        value = Decimal(text) * UNIT_NS[unit]
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError('Time must be nonnegative, finite and exactly representable in ns')
        return int(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f'Invalid {unit} time: {text!r}') from exc


def parse_timestamp(text):
    """Return integer clock ticks, display resolution, and clock-reference kind."""
    match = STAMP.fullmatch(text.strip())
    if not match:
        raise ValueError(f'Expected full date and seconds, got {text!r}; minute-only time is not a B sampling clock')
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    normalized = f'{year}-{month}-{day}T{int(hour):02d}:{minute}:{second}'
    if zone:
        normalized += '+00:00' if zone == 'Z' else zone
    stamp = datetime.fromisoformat(normalized)
    kind = 'aware_UTC' if stamp.tzinfo else 'naive_scene_local'
    if stamp.tzinfo:
        stamp = stamp.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    else:
        epoch = datetime(1970, 1, 1)
    delta = stamp - epoch
    fraction = fraction or ''
    ticks = (delta.days * 86400 + delta.seconds) * NS_PER_SECOND + int((fraction + '0' * 9)[:9])
    resolution = 10 ** (9 - len(fraction))
    return ticks, resolution, kind


@dataclass(frozen=True)
class InputConfig:
    a_toa_unit: str
    timestamp_tolerance_ns: int
    maximum_b_interpolation_gap_seconds: float
    origins: dict
    origin_source: str = ''

    @classmethod
    def from_dict(cls, value):
        if value['a_toa_unit'] not in UNIT_NS or value.get('internal_time_unit', 'ns') != 'ns':
            raise ValueError('Specify the source TOA unit explicitly; internal unit must be ns')
        origins = {}
        for row in value['origins']:
            key = row['scenario_id'], row['data_version']
            if key in origins:
                raise ValueError(f'Duplicate time origin: {key}')
            ticks, _, kind = parse_timestamp(row['datetime'])
            origins[key] = (ticks, kind)
        tolerance = value.get('timestamp_tolerance_ns', 1_000_000)
        gap = value.get('maximum_b_interpolation_gap_seconds', 3.)
        if not isinstance(tolerance, int) or tolerance < 0 or not math.isfinite(gap) or gap <= 0:
            raise ValueError('Invalid timestamp tolerance or B interpolation gap')
        return cls(value['a_toa_unit'], tolerance, gap, origins, value.get('origin_source', ''))

    def origin(self, scene, version):
        try:
            return self.origins[scene, version]
        except KeyError as exc:
            raise ValueError(f'No configured public/known time origin for {(scene, version)}') from exc


class Schema:
    def __init__(self, header, fields):
        if not header or len(header) != len(set(header)):
            raise ValueError('Missing or duplicate CSV header')
        self.width = len(header)
        self.indices = {}
        for canonical, aliases in fields.items():
            matches = [header.index(name) for name in aliases if name in header]
            if not matches:
                raise ValueError(f'Missing {canonical}; accepted names: {aliases}')
            self.indices[canonical] = matches

    def text(self, row, name):
        indices = self.indices[name]
        if len(indices) == 1:
            value = row[indices[0]].strip()
            if not value:
                raise ValueError(f'Empty/conflicting identifier {name}')
            return value
        values = [row[i].strip() for i in self.indices[name]]
        if not values[0] or any(v != values[0] for v in values[1:]):
            raise ValueError(f'Empty/conflicting identifier {name}')
        return values[0]

    def number(self, row, name):
        indices = self.indices[name]
        if len(indices) == 1:
            value = float(row[indices[0]])
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite/conflicting measurement {name}')
            return value
        values = [float(row[i]) for i in self.indices[name]]
        if not all(map(math.isfinite, values)) or any(v != values[0] for v in values[1:]):
            raise ValueError(f'Nonfinite/conflicting measurement {name}')
        return values[0]

    def check_width(self, row):
        if len(row) != self.width:
            raise ValueError('CSV row column count differs from header')


A_FIELDS = {name: (name,) for name in ('scenario_id', 'data_version', 'pulse_id', 'platform_id', 'datetime', 'toa')}
A_FIELDS.update({'rf': ('param1', 'rf_mhz', 'RF'), 'pw': ('param2', 'pw_us', 'PW'),
    'pa': ('param3', 'pa_dbm', 'PA'), 'doa': ('doa', 'doa_degrees'),
    'longitude': ('sensor_longitude', 'sensor_lontitude'),
    'latitude': ('sensor_latitude',), 'height': ('sensor_height',)})
B_FIELDS = {name: (name,) for name in ('scenario_id', 'data_version', 'track_id', 'datetime')}
B_FIELDS.update({'longitude': ('target_longitude', 'target_lontitude', 'longitude', 'lontitude'),
                 'latitude': ('target_latitude', 'latitude'), 'height': ('target_height', 'height')})


@dataclass(frozen=True, slots=True)
class PulseObservation:
    scenario: str
    version: str
    platform: str
    pulse_id: str
    toa: int                         # Integer ns since the configured scene origin.
    rf: float                        # MHz
    pw: float                        # us
    pa: float                        # dBm
    doa: float                       # North-clockwise geographic degrees.
    position: tuple
    toa_raw: str
    datetime_raw: str
    timestamp_ns: int                # TOA + scene origin, not corrected for clock bias.
    recorded_timestamp_ns: int       # Coarser timestamp as given in the CSV.
    timestamp_resolution_ns: int

    @property
    def scope(self):
        return self.scenario, self.version, self.platform

    @property
    def key(self):
        return self.scope + (self.pulse_id,)


class PulseIDCheck:
    """Bounded-memory fast path for ordered numeric IDs, disk-backed generic path."""
    def __init__(self, mode):
        self.mode, self.last, self.connection, self.temp = mode, {}, None, None
        if mode == 'sqlite':
            self.temp = tempfile.TemporaryDirectory(prefix='b04_id_check_')
            self.connection = sqlite3.connect(str(Path(self.temp.name) / 'ids.sqlite'))
            self.connection.execute('CREATE TABLE ids (scene TEXT, version TEXT, pulse TEXT, PRIMARY KEY(scene,version,pulse)) WITHOUT ROWID')
        elif mode != 'ordered':
            raise ValueError('ID checking must be ordered or sqlite')

    def add(self, scene, version, pulse):
        if self.connection:
            try:
                self.connection.execute('INSERT INTO ids VALUES (?,?,?)', (scene, version, pulse))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'Duplicate A pulse ID: {(scene, version, pulse)}') from exc
        else:
            if not pulse.isascii() or not pulse.isdigit() or str(int(pulse)) != pulse:
                raise ValueError('Ordered ID mode requires canonical numeric IDs; use sqlite mode for arbitrary IDs/order')
            number = int(pulse)
            key = scene, version
            if number <= self.last.get(key, -1):
                raise ValueError('Duplicate or non-increasing A pulse ID; use sqlite mode for unordered inputs')
            self.last[key] = number

    def close(self):
        if self.connection:
            self.connection.close()
        if self.temp:
            self.temp.cleanup()


def iter_a_batches(path, config, batch_size=20_000, id_check='ordered'):
    """Read only observation fields, never target_id/class/state; preserve file order."""
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    checker = PulseIDCheck(id_check)
    count, batch, last_stamp, cached_stamp = 0, [], None, None
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            schema = Schema(next(reader, None), A_FIELDS)
            for line, row in enumerate(reader, 2):
                try:
                    schema.check_width(row)
                    scene, version, platform = (sys.intern(schema.text(row, k)) for k in ('scenario_id', 'data_version', 'platform_id'))
                    pulse = schema.text(row, 'pulse_id')
                    checker.add(scene, version, pulse)
                    toa_raw, stamp_raw = schema.text(row, 'toa'), schema.text(row, 'datetime')
                    toa = time_to_ns(toa_raw, config.a_toa_unit)
                    if stamp_raw != last_stamp:
                        cached_stamp, last_stamp = parse_timestamp(stamp_raw), stamp_raw
                    recorded, resolution, kind = cached_stamp
                    origin, origin_kind = config.origin(scene, version)
                    absolute = origin + toa
                    if kind != origin_kind or abs(absolute - recorded) > max(resolution, config.timestamp_tolerance_ns):
                        raise ValueError('TOA and datetime disagree; check source unit, scene origin and clock reference')
                    rf, pw, pa, doa = (schema.number(row, k) for k in ('rf', 'pw', 'pa', 'doa'))
                    longitude, latitude, height = (schema.number(row, k) for k in ('longitude', 'latitude', 'height'))
                    if rf <= 0 or pw < 0 or not 0 <= doa <= 360 or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                        raise ValueError('Out-of-range RF, PW, DOA or position')
                    batch.append(PulseObservation(scene, version, platform, pulse, toa, rf, pw, pa, doa % 360.,
                        (longitude, latitude, height), toa_raw, stamp_raw, absolute, recorded, resolution))
                    count += 1
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            if count == 0:
                raise ValueError(f'Empty A observations: {path}')
    finally:
        checker.close()


@dataclass(frozen=True, slots=True)
class TrackObservation:
    timestamp_ns: int
    time_ns: int
    position: tuple
    datetime_raw: str
    target_type: str
    target_class: str                # Public B observation; not a truth identity.
    source_row: int


@dataclass
class Track:
    key: tuple
    observations: list
    reordered: bool = False
    times_ns: tuple = field(init=False)

    def __post_init__(self):
        self.times_ns = tuple(o.time_ns for o in self.observations)

    def position(self, time_ns, maximum_gap_ns):
        """Actual-time interpolation; no extrapolation or bridging long missing spans."""
        j = bisect.bisect_left(self.times_ns, time_ns)
        if j < len(self.times_ns) and self.times_ns[j] == time_ns:
            return self.observations[j].position
        if j == 0 or j == len(self.times_ns):
            return None
        left, right = self.observations[j - 1], self.observations[j]
        if right.time_ns - left.time_ns > maximum_gap_ns:
            return None
        fraction = (time_ns - left.time_ns) / (right.time_ns - left.time_ns)
        return tuple(a + fraction * (b - a) for a, b in zip(left.position, right.position))


def read_b_tracks(path, config):
    """Read public B columns only. No target_id, synthetic cadence, or smoothing."""
    grouped = defaultdict(list)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        schema = Schema(header, B_FIELDS)
        optional = {name: header.index(name) if name in header else None for name in ('target_type', 'target_class')}
        for line, row in enumerate(reader, 2):
            try:
                schema.check_width(row)
                scene, version, track_id = (schema.text(row, k) for k in ('scenario_id', 'data_version', 'track_id'))
                stamp_raw = schema.text(row, 'datetime')
                stamp, resolution, kind = parse_timestamp(stamp_raw)
                origin, origin_kind = config.origin(scene, version)
                if resolution > NS_PER_SECOND or kind != origin_kind:
                    raise ValueError('B requires actual second-or-finer timestamps in the configured clock reference')
                position = tuple(schema.number(row, k) for k in ('longitude', 'latitude', 'height'))
                if not -180 <= position[0] <= 180 or not -90 <= position[1] <= 90:
                    raise ValueError('Out-of-range B position')
                values = {k: row[i].strip() if i is not None else '' for k, i in optional.items()}
                grouped[scene, version, track_id].append(TrackObservation(stamp, stamp - origin, position,
                    stamp_raw, values['target_type'], values['target_class'], line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
    if not grouped:
        raise ValueError('Empty B observations')
    result = {}
    for key, observations in grouped.items():
        original = [o.time_ns for o in observations]
        observations.sort(key=lambda o: o.time_ns)
        if any(a.time_ns == b.time_ns for a, b in zip(observations, observations[1:])):
            raise ValueError(f'Duplicate timestamp within B track: {key}')
        result[key] = Track(key, observations, original != sorted(original))
    return result
