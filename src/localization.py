"""Horizontal A-bearing localization and segmented, measurement-only CV tracks.

Only two-dimensional observer positions and bearings enter the solver.  Heights,
B positions, labels, and upstream coarse intersection coordinates are not inputs.
The reported covariance is a model-based error proxy, not calibrated accuracy.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class LocalizationConfig:
    maximum_time_span_seconds: float = .08
    minimum_crossing_angle_degrees: float = 7.
    bearing_error_floor_degrees: float = .15
    residual_gate_degrees: float = 1.2
    residual_sigma_multiplier: float = 3.
    motion_bound_m_per_second: float = 180.
    maximum_range_m: float = 180000.
    maximum_condition_number: float = 1000.
    position_error_floor_m: float = 30.
    maximum_major_error_proxy_m: float = 5000.
    subset_ambiguity_distance_m: float = 500.
    maximum_iterations: int = 12
    track_bin_seconds: float = .5
    track_segment_gap_seconds: float = 5.
    track_acceleration_sigma_m_per_second2: float = 8.
    track_innovation_gate_squared: float = 25.

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if not isinstance(self.maximum_iterations, int) or self.maximum_iterations < 2:
            raise ValueError('maximum_iterations must be an integer >= 2')
        if self.minimum_crossing_angle_degrees >= 90 or self.maximum_condition_number <= 1:
            raise ValueError('Invalid geometric gate')
        if self.track_bin_seconds < 1e-9 or self.track_segment_gap_seconds < self.track_bin_seconds:
            raise ValueError('Invalid track time scale')


class HorizontalFrame:
    """WGS84 surface positions projected onto a common EN plane; no input height.

    Straight bearings in this plane approximate the local surface geometry. The
    inverse intersects the reference-normal line with the WGS84 ellipsoid.
    """
    A = 6378137.
    E2 = 6.6943799901413165e-3
    B2 = A*A*(1-E2)

    def __init__(self, longitude, latitude):
        if not math.isfinite(longitude) or not math.isfinite(latitude) or not -180 <= longitude <= 180 or not -89 <= latitude <= 89:
            raise ValueError('Invalid local frame origin')
        self.longitude, self.latitude = longitude, latitude
        lon, lat = math.radians(longitude), math.radians(latitude)
        self.east = (-math.sin(lon), math.cos(lon), 0.)
        self.north = (-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat))
        self.up = (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))
        self.origin = self.surface(longitude, latitude)

    @classmethod
    def surface(cls, longitude, latitude):
        lon, lat = math.radians(longitude), math.radians(latitude)
        n = cls.A/math.sqrt(1-cls.E2*math.sin(lat)**2)
        return n*math.cos(lat)*math.cos(lon), n*math.cos(lat)*math.sin(lon), n*(1-cls.E2)*math.sin(lat)

    def project(self, longitude, latitude):
        delta = tuple(p-q for p, q in zip(self.surface(longitude, latitude), self.origin))
        return tuple(sum(x*y for x, y in zip(delta, axis)) for axis in (self.east, self.north))

    def inverse(self, east, north):
        d = tuple(o+east*e+north*n for o, e, n in zip(self.origin, self.east, self.north))
        metric = (1/self.A**2, 1/self.A**2, 1/self.B2)
        a = sum(w*u*u for w, u in zip(metric, self.up))
        b = 2*sum(w*x*u for w, x, u in zip(metric, d, self.up))
        c = sum(w*x*x for w, x in zip(metric, d))-1
        disc = b*b-4*a*c
        if disc < 0:raise ValueError('Position outside surface projection')
        z = -2*c/(b+math.sqrt(disc))
        x, y, zz = (v+z*u for v, u in zip(d, self.up))
        return math.degrees(math.atan2(y, x)), math.degrees(math.atan2(zz/(1-self.E2), math.hypot(x, y)))

    def direction(self, longitude, latitude, doa):
        lon, lat, theta = map(math.radians, (longitude, latitude, doa))
        e = (-math.sin(lon), math.cos(lon), 0.)
        n = (-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat))
        ray = tuple(math.sin(theta)*x+math.cos(theta)*y for x, y in zip(e, n))
        dx, dy = (sum(x*y for x, y in zip(ray, axis)) for axis in (self.east, self.north))
        norm = math.hypot(dx, dy)
        return dx/norm, dy/norm


@dataclass(frozen=True, slots=True)
class Bearing:
    key: tuple                       # scene, version, platform, sequence
    time: int                        # integer ns from scene origin
    start: int
    end: int
    xy: tuple
    direction: tuple                 # unit vector in common EN
    error_degrees: float
    longitude: float = 0.
    latitude: float = 0.
    doa: float = 0.

    @property
    def platform(self):return self.key[2]


def inverse2(a, b, c):
    det = a*c-b*b
    if det <= 0 or not math.isfinite(det):return None
    return c/det, -b/det, a/det


def eigen2(a, b, c):
    radius = math.hypot(a-c, 2*b)
    return max(0., (a+c+radius)/2), max(0., (a+c-radius)/2)


def crossing(a, b):
    dot = abs(sum(x*y for x, y in zip(a.direction, b.direction)))
    return math.degrees(math.acos(min(1., dot)))


def metrics(o, point, reference, cfg):
    dx, dy = point[0]-o.xy[0], point[1]-o.xy[1]
    distance = math.hypot(dx, dy)
    forward = dx*o.direction[0]+dy*o.direction[1]
    residual = math.atan2(o.direction[0]*dy-o.direction[1]*dx, forward)
    # No invented synchronous measurement: observer position stays at o.time.
    # Target displacement across the short event and sequence averaging interval
    # is represented conservatively as extra angular model uncertainty.
    motion = cfg.motion_bound_m_per_second*(abs(o.time-reference)+(o.end-o.start)/2)/1e9
    sigma = math.hypot(math.radians(max(o.error_degrees, cfg.bearing_error_floor_degrees)),
                       math.atan2(motion, max(distance, 1.)))
    return distance, forward, residual, sigma


def line_start(obs, cfg):
    aa = ab = bb = ax = bx = 0.
    for o in obs:
        nx, ny = -o.direction[1], o.direction[0]
        w = 1/max(o.error_degrees, cfg.bearing_error_floor_degrees)**2
        rhs = nx*o.xy[0]+ny*o.xy[1]
        aa += w*nx*nx;ab += w*nx*ny;bb += w*ny*ny
        ax += w*nx*rhs;bx += w*ny*rhs
    inv = inverse2(aa, ab, bb)
    return None if inv is None else (inv[0]*ax+inv[1]*bx, inv[1]*ax+inv[2]*bx)


def normal(obs, point, reference, cfg):
    aa = ab = bb = gx = gy = cost = 0.
    for o in obs:
        distance, _, residual, sigma = metrics(o, point, reference, cfg)
        if distance < 1:return None
        dx, dy = point[0]-o.xy[0], point[1]-o.xy[1]
        jx, jy = -dy/distance**2, dx/distance**2
        w = 1/sigma**2
        aa += w*jx*jx;ab += w*jx*jy;bb += w*jy*jy
        gx += w*jx*residual;gy += w*jy*residual;cost += w*residual**2
    return aa, ab, bb, gx, gy, cost


def fit(obs, reference, cfg, initial=None):
    point = initial if initial is not None else line_start(obs, cfg)
    if point is None:return None
    for _ in range(cfg.maximum_iterations):
        n = normal(obs, point, reference, cfg)
        if n is None:return None
        a, b, c, gx, gy, cost = n
        inv = inverse2(a, b, c)
        if inv is None:return None
        sx, sy = inv[0]*gx+inv[1]*gy, inv[1]*gx+inv[2]*gy
        if math.hypot(sx, sy) < .001:break
        scale = 1.
        for _ in range(12):
            candidate = point[0]-sx*scale, point[1]-sy*scale
            nn = normal(obs, candidate, reference, cfg)
            if nn is not None and nn[-1] <= cost+1e-10:
                point = candidate;break
            scale *= .5
        else:break
    n = normal(obs, point, reference, cfg)
    if n is None:return None
    return point, n


def inliers(obs, point, reference, cfg):
    selected = []
    for o in obs:
        distance, forward, residual, sigma = metrics(o, point, reference, cfg)
        gate = max(cfg.residual_gate_degrees, cfg.residual_sigma_multiplier*math.degrees(sigma))
        if forward > 0 and distance <= cfg.maximum_range_m and abs(math.degrees(residual)) <= gate:
            selected.append(o)
    return tuple(selected)


def localize(obs, cfg):
    """One event, >=2 distinct platforms, joint angular WLS with subset consensus.

    Pair intersections initialize outlier hypotheses, never independent averaged
    position measurements. Equally supported, separated subsets are rejected.
    """
    obs = tuple(obs)
    if len(obs) < 2:return None, 'fewer_than_two_platforms'
    if len({o.key[:2] for o in obs}) != 1:raise ValueError('Mixed scenes/versions')
    if len({o.platform for o in obs}) != len(obs):raise ValueError('Duplicate platform in event')
    reference = round(median(o.time for o in obs))
    if max(o.time for o in obs)-min(o.time for o in obs) > cfg.maximum_time_span_seconds*1e9+1:
        return None, 'asynchronous_span_too_large'
    best_crossing = max(crossing(a, b) for a, b in itertools.combinations(obs, 2))
    if best_crossing < cfg.minimum_crossing_angle_degrees:return None, 'weak_crossing_angle'
    fitted = fit(obs, reference, cfg)
    selected = obs
    trimmed = False
    if fitted is None or len(inliers(obs, fitted[0], reference, cfg)) != len(obs):
        candidates = {}
        for pair in itertools.combinations(obs, 2):
            if crossing(*pair) < cfg.minimum_crossing_angle_degrees:continue
            first = fit(pair, reference, cfg)
            if first is None:continue
            subset = inliers(obs, first[0], reference, cfg)
            if len(subset) < 2:continue
            refined = fit(subset, reference, cfg, first[0])
            if refined is None or len(inliers(subset, refined[0], reference, cfg)) != len(subset):continue
            key = tuple(o.key for o in subset)
            candidates[key] = (subset, refined)
        if not candidates:return None, 'no_forward_inlier_consensus'
        ranked = sorted(candidates.values(), key=lambda c: (-len(c[0]), c[1][1][-1], tuple(o.key for o in c[0])))
        selected, fitted = ranked[0]
        for subset, other in ranked[1:]:
            if len(subset) != len(selected):break
            if math.dist(other[0], fitted[0]) > cfg.subset_ambiguity_distance_m:
                return None, 'ambiguous_inlier_subsets'
        trimmed = len(selected) != len(obs)
    point, n = fitted
    a, b, c, _, _, cost = n
    hi, lo = eigen2(a, b, c)
    condition = hi/lo if lo > 0 else float('inf')
    angle = max(crossing(x, y) for x, y in itertools.combinations(selected, 2))
    if angle < cfg.minimum_crossing_angle_degrees:return None, 'weak_inlier_crossing_angle'
    if condition > cfg.maximum_condition_number:return None, 'ill_conditioned_geometry'
    inv = inverse2(a, b, c)
    if inv is None:return None, 'singular_geometry'
    factor = max(1., cost/max(1, len(selected)-2))
    floor = cfg.position_error_floor_m**2
    cov = inv[0]*factor+floor, inv[1]*factor, inv[2]*factor+floor
    major2, minor2 = eigen2(*cov)
    if math.sqrt(major2) > cfg.maximum_major_error_proxy_m:return None, 'position_error_proxy_too_large'
    all_metrics = {o.key: metrics(o, point, reference, cfg) for o in obs}
    residuals = [math.degrees(all_metrics[o.key][2]) for o in selected]
    return {'time_ns': reference, 'east_m': point[0], 'north_m': point[1], 'covariance': cov,
            'major_error_proxy_m': math.sqrt(major2), 'minor_error_proxy_m': math.sqrt(minor2),
            'condition_number': condition, 'crossing_angle_degrees': angle,
            'rms_bearing_residual_degrees': math.sqrt(sum(x*x for x in residuals)/len(residuals)),
            'maximum_bearing_residual_degrees': max(abs(r) for r in residuals),
            'time_span_seconds': (max(o.time for o in obs)-min(o.time for o in obs))/1e9,
            'selected': selected, 'metrics': all_metrics,
            'fit_status': 'joint_wls_inlier_subset' if trimmed else ('two_bearing_wls' if len(obs)==2 else 'joint_multi_bearing_wls')}, ''


def bin_points(points, cfg):
    """Robust per-bin measurement summary; covariance is NOT divided by event N."""
    bins = {}
    period = round(cfg.track_bin_seconds*1e9)
    for p in points:bins.setdefault(p['time_ns']//period, []).append(p)
    rows = []
    for _, ps in sorted(bins.items()):
        x, y = median(p['east_m'] for p in ps), median(p['north_m'] for p in ps)
        sx = 1.4826*median(abs(p['east_m']-x) for p in ps)
        sy = 1.4826*median(abs(p['north_m']-y) for p in ps)
        xx = median(p['covariance'][0] for p in ps)+sx*sx
        yy = median(p['covariance'][2] for p in ps)+sy*sy
        xy = median(p['covariance'][1] for p in ps)
        xy = max(-.95*math.sqrt(xx*yy), min(.95*math.sqrt(xx*yy), xy))
        rows.append({'time_ns': round(median(p['time_ns'] for p in ps)), 'east_m': x, 'north_m': y,
                     'covariance': (xx, xy, yy), 'event_count': len(ps),
                     'event_ids': '|'.join(p['event_id'] for p in ps)})
    return rows


def transpose(a):return [list(x) for x in zip(*a)]


def multiply(a, b):
    bt = transpose(b)
    return [[sum(x*y for x, y in zip(row, col)) for col in bt] for row in a]


def filter_track(points, cfg):
    """CV Kalman filter, independent per object, gaps split segments, no gap fill.

    Only accepted measurement times produce track points. Rejected bin summaries
    remain auditable; a timeout since the last accepted point starts a new segment.
    """
    summaries = bin_points(points, cfg)
    accepted, rejected = [], []
    state = covariance = None
    last_time = last_accepted = None
    segment = 0
    for row in summaries:
        t, measurement = row['time_ns'], (row['east_m'], row['north_m'])
        xx, xy, yy = row['covariance']
        nis = 0.
        if state is None or t-last_accepted > cfg.track_segment_gap_seconds*1e9:
            segment += 1
            state = [*measurement, 0., 0.]
            v = cfg.motion_bound_m_per_second**2
            covariance = [[xx, xy, 0., 0.], [xy, yy, 0., 0.], [0., 0., v, 0.], [0., 0., 0., v]]
            status = 'segment_start'
        else:
            dt = (t-last_time)/1e9
            transition = [[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            state = [sum(a*b for a, b in zip(r, state)) for r in transition]
            covariance = multiply(multiply(transition, covariance), transpose(transition))
            q = cfg.track_acceleration_sigma_m_per_second2**2
            for i, j, v in ((0,0,dt**4/4),(1,1,dt**4/4),(0,2,dt**3/2),(2,0,dt**3/2),
                            (1,3,dt**3/2),(3,1,dt**3/2),(2,2,dt**2),(3,3,dt**2)):
                covariance[i][j] += q*v
            sxx, sxy, syy = covariance[0][0]+xx, covariance[0][1]+xy, covariance[1][1]+yy
            inv = inverse2(sxx, sxy, syy)
            if inv is None:raise ValueError('Invalid track innovation covariance')
            dx, dy = measurement[0]-state[0], measurement[1]-state[1]
            nis = inv[0]*dx*dx+2*inv[1]*dx*dy+inv[2]*dy*dy
            if nis > cfg.track_innovation_gate_squared:
                rejected.append(dict(row, segment_id=f'T{segment:04d}', innovation_squared=nis,
                                     reason='track_innovation_gate'))
                last_time = t
                continue
            gain = [[r[0]*inv[0]+r[1]*inv[1], r[0]*inv[1]+r[1]*inv[2]] for r in covariance]
            state = [v+k[0]*dx+k[1]*dy for v, k in zip(state, gain)]
            # Joseph update keeps positive covariance with repeated small steps.
            ikh = [[float(i==j)-(gain[i][j] if j<2 else 0.) for j in range(4)] for i in range(4)]
            pc = multiply(multiply(ikh, covariance), transpose(ikh))
            kr = multiply(multiply(gain, [[xx,xy],[xy,yy]]), transpose(gain))
            covariance = [[pc[i][j]+kr[i][j] for j in range(4)] for i in range(4)]
            status = 'measurement_update'
        last_time = last_accepted = t
        accepted.append(dict(row, east_m=state[0], north_m=state[1],
                             measurement_east_m=measurement[0], measurement_north_m=measurement[1],
                             velocity_east_m_per_second=state[2], velocity_north_m_per_second=state[3],
                             filter_covariance=(covariance[0][0], covariance[0][1], covariance[1][1]),
                             segment_id=f'T{segment:04d}', innovation_squared=nis, filter_status=status))
    return accepted, rejected
