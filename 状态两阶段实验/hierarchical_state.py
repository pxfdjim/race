"""Binary forests for observed search/tracking and conditional coarse/fine states."""

from dataclasses import asdict, dataclass
from itertools import accumulate
import math
import random


@dataclass(frozen=True, slots=True)
class ForestConfig:
    trees: int = 24
    maximum_depth: int = 12
    samples_per_tree: int = 18_000
    features_per_split: int = 6
    thresholds_per_feature: int = 12
    minimum_leaf_samples: int = 24
    prior_power: float = .5
    seed: int = 20261118

    def validate(self):
        for name in ('trees', 'maximum_depth', 'samples_per_tree', 'features_per_split',
                     'thresholds_per_feature', 'minimum_leaf_samples'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'Invalid binary forest configuration: {name}')
        if self.samples_per_tree < 2 or not 0 <= self.prior_power <= 1:
            raise ValueError('Invalid sampling size or prior correction')


def binary_gini(n, positive):
    return 2 * positive * (n - positive) / n if n else 0.


class BinaryForest:
    """Balanced bootstrap with leaf scores adjusted by the fitted positive prior."""

    def __init__(self, config, features):
        config.validate()
        self.config = config
        self.features = tuple(features)
        self.prior = None
        self.trees = []

    def fit(self, x, labels, weights):
        if not x or not len(x) == len(labels) == len(weights) or not self.features:
            raise ValueError('Binary feature, label or weight count mismatch')
        if any(len(v) != len(self.features) or not all(map(math.isfinite, v)) for v in x):
            raise ValueError('Invalid binary training vector')
        if any(y not in (0, 1) for y in labels) or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError('Invalid binary labels or weights')
        groups = [[i for i, label in enumerate(labels) if label == c] for c in (0, 1)]
        if not all(groups):
            raise ValueError('Both binary states need labeled development records')
        cumulative = [list(accumulate(weights[i] for i in group)) for group in groups]
        self.prior = cumulative[1][-1] / (cumulative[0][-1] + cumulative[1][-1])
        cfg = self.config
        rng = random.Random(cfg.seed)
        features = tuple(range(len(self.features)))

        def tree(indices, depth):
            n = len(indices)
            positive = sum(labels[i] for i in indices)
            leaf = {'p': (positive + .5) / (n + 1)}
            if depth >= cfg.maximum_depth or n < 2 * cfg.minimum_leaf_samples or positive in (0, n):
                return leaf
            parent = binary_gini(n, positive)
            cuts = {max(cfg.minimum_leaf_samples, min(n - cfg.minimum_leaf_samples,
                    round(n * i / (1 + cfg.thresholds_per_feature))))
                    for i in range(1, 1 + cfg.thresholds_per_feature)}
            best = None
            for feature in rng.sample(features, min(cfg.features_per_split, len(features))):
                ordered = sorted(indices, key=lambda i: x[i][feature])
                left_positive = 0
                for pos, i in enumerate(ordered, 1):
                    left_positive += labels[i]
                    if pos not in cuts:
                        continue
                    lo, hi = x[i][feature], x[ordered[pos]][feature]
                    if lo >= hi:
                        continue
                    gain = parent - binary_gini(pos, left_positive) - binary_gini(n - pos, positive - left_positive)
                    if gain > 1e-7 and (best is None or gain > best[0]):
                        best = gain, feature, (lo + hi) / 2
            if best is None:
                return leaf
            _, feature, cut = best
            left, right = [], []
            for i in indices:
                (left if x[i][feature] <= cut else right).append(i)
            if not left or not right:
                return leaf
            return {'f': feature, 't': cut, 'l': tree(left, depth + 1), 'r': tree(right, depth + 1)}

        self.trees = []
        for _ in range(cfg.trees):
            indices = []
            for c in (0, 1):
                indices.extend(rng.choices(groups[c], cum_weights=cumulative[c], k=cfg.samples_per_tree // 2))
            self.trees.append(tree(indices, 0))
        return self

    def score(self, values):
        if len(values) != len(self.features) or not self.trees or not all(map(math.isfinite, values)):
            raise ValueError('Invalid binary state input')
        total = 0.
        for root in self.trees:
            node = root
            while 'p' not in node:
                node = node['l'] if values[node['f']] <= node['t'] else node['r']
            total += node['p']
        p = total / len(self.trees)
        positive = p * self.prior ** self.config.prior_power
        negative = (1 - p) * (1 - self.prior) ** self.config.prior_power
        return positive / (positive + negative)

    def to_dict(self):
        return {'config': asdict(self.config), 'features': list(self.features),
                'positive_prior': self.prior, 'trees': self.trees}

    @classmethod
    def from_dict(cls, data):
        model = cls(ForestConfig(**data['config']), data['features'])
        model.trees, model.prior = data['trees'], data['positive_prior']
        if len(model.trees) != model.config.trees or not 0 < model.prior < 1:
            raise ValueError('Invalid stored binary state forest')
        return model


def joint_scores(follow, coarse):
    return 1 - follow, follow * coarse, follow * (1 - coarse)


def decide_state(follow, coarse, follow_threshold, coarse_threshold):
    if follow < follow_threshold:
        return 0
    return 1 if coarse >= coarse_threshold else 2
