"""Small standard-library state forest and gap-bounded temporal decoding."""
from __future__ import annotations
import math
import random
from dataclasses import asdict,dataclass

STATES=('0','1','2')


@dataclass(frozen=True)
class StateModelConfig:
    trees: int=24
    maximum_depth: int=8
    samples_per_tree: int=18000
    features_per_split: int=6
    thresholds_per_feature: int=12
    minimum_leaf_samples: int=24
    prior_power: float=0.
    temporal_strength: float=0.
    seed: int=20261117

    def validate(self):
        for k in ['trees','maximum_depth','samples_per_tree','features_per_split','thresholds_per_feature','minimum_leaf_samples']:
            if isinstance(getattr(self,k),bool) or not isinstance(getattr(self,k),int) or getattr(self,k)<1:raise ValueError(f'Invalid {k}')
        if not isinstance(self.seed,int) or not math.isfinite(self.prior_power) or not 0<=self.prior_power<=1 or not math.isfinite(self.temporal_strength) or not 0<=self.temporal_strength<=2:
            raise ValueError('Invalid state prior/temporal configuration')


def distribution(counts):
    n=sum(counts)+1.5
    return tuple((x+.5)/n for x in counts)


def gini(counts):
    n=sum(counts)
    return n-sum(x*x for x in counts)/n if n else 0.


class StateForest:
    def __init__(self,cfg,features):
        cfg.validate();self.config=cfg;self.features=list(features);self.trees=[];self.priors=(1/3,)*3
        self.transitions=[[1/3]*3 for _ in range(3)]

    def fit(self,x,y,weights):
        if not x or len(x)!=len(y) or len(x)!=len(weights) or any(len(v)!=len(self.features) for v in x):raise ValueError('Invalid state fit data')
        if any(not all(math.isfinite(z) for z in v) for v in x) or any(c not in [0,1,2] for c in y):raise ValueError('Invalid state features/labels')
        if any(not math.isfinite(w) or w<=0 for w in weights):raise ValueError('Invalid state fit weight')
        groups=[[i for i,c in enumerate(y) if c==s] for s in range(3)]
        if any(not g for g in groups):raise ValueError('Training must contain all three states')
        totals=[sum(weights[i] for i in g) for g in groups];total=sum(totals);self.priors=tuple(v/total for v in totals)
        rng=random.Random(self.config.seed);cfg=self.config
        def build(indices,depth):
            counts=[0,0,0]
            for i in indices:counts[y[i]]+=1
            if depth>=cfg.maximum_depth or len(indices)<2*cfg.minimum_leaf_samples or max(counts)==len(indices):
                return {'p':distribution(counts)}
            parent=gini(counts);best=None
            for feature in rng.sample(range(len(self.features)),min(cfg.features_per_split,len(self.features))):
                ordered=sorted(indices,key=lambda i:x[i][feature]);n=len(ordered)
                cuts={min(n-cfg.minimum_leaf_samples,max(cfg.minimum_leaf_samples,round(n*j/(cfg.thresholds_per_feature+1)))) for j in range(1,cfg.thresholds_per_feature+1)}
                left=[0,0,0];right=counts.copy()
                for position,i in enumerate(ordered,1):
                    left[y[i]]+=1;right[y[i]]-=1
                    if position not in cuts or position==n:continue
                    v,w=x[i][feature],x[ordered[position]][feature]
                    if v>=w:continue
                    gain=parent-gini(left)-gini(right)
                    if gain>1e-7 and (best is None or gain>best[0]):best=gain,feature,(v+w)/2
            if best is None:return {'p':distribution(counts)}
            _,feature,threshold=best
            left=[i for i in indices if x[i][feature]<=threshold];right=[i for i in indices if x[i][feature]>threshold]
            return {'f':feature,'t':threshold,'l':build(left,depth+1),'r':build(right,depth+1)}
        self.trees=[];per_state=max(1,cfg.samples_per_tree//3)
        for _ in range(cfg.trees):
            indices=[]
            for group in groups:indices.extend(rng.choices(group,weights=[weights[i] for i in group],k=per_state))
            self.trees.append(build(indices,0))
        return self

    def predict(self,features):
        if len(features)!=len(self.features) or not self.trees or not all(math.isfinite(x) for x in features):raise ValueError('Invalid state prediction features/model')
        probabilities=[0.,0.,0.]
        for tree in self.trees:
            node=tree
            while 'p' not in node:node=node['l'] if features[node['f']]<=node['t'] else node['r']
            for i,p in enumerate(node['p']):probabilities[i]+=p
        for i in range(3):probabilities[i]=probabilities[i]/len(self.trees)*self.priors[i]**self.config.prior_power
        total=sum(probabilities);return tuple(x/total for x in probabilities)

    def to_dict(self):
        return {'format':'B04_temporal_state_forest_v1','states':list(STATES),'features':self.features,
            'config':asdict(self.config),'priors':self.priors,'transitions':self.transitions,'trees':self.trees,
            'score_interpretation':'Uncalibrated state evidence weights'}

    @classmethod
    def from_dict(cls,data):
        if data['format']!='B04_temporal_state_forest_v1' or data['states']!=list(STATES):raise ValueError('Unknown state model')
        m=cls(StateModelConfig(**data['config']),data['features']);m.trees=data['trees'];m.priors=tuple(data['priors']);m.transitions=data['transitions']
        if not m.trees or len(m.priors)!=3 or any(x<=0 or not math.isfinite(x) for x in m.priors):raise ValueError('Invalid saved state model')
        return m


def learn_transitions(keys,labels,weights):
    counts=[[.5]*3 for _ in range(3)];lookup={k:(label,weight) for k,label,weight in zip(keys,labels,weights)}
    for k,(label,weight) in lookup.items():
        previous=lookup.get(k[:-1]+(k[-1]-1,))
        if previous is not None:counts[previous[0]][label]+=min(256.,weight,previous[1])
    return [[x/sum(row) for x in row] for row in counts]


def logsum(values):
    top=max(values);return top+math.log(sum(math.exp(x-top) for x in values))


def smooth_states(keys,probabilities,transitions,strength):
    """Forward/backward only through adjacent observed bins, without gap fill."""
    if len(keys)!=len(probabilities):raise ValueError('State smoothing count mismatch')
    if strength==0:return list(probabilities)
    order=sorted(range(len(keys)),key=lambda i:keys[i]);blocks=[]
    for i in order:
        if not blocks or keys[i][:-1]!=keys[blocks[-1][-1]][:-1] or keys[i][-1]!=keys[blocks[-1][-1]][-1]+1:blocks.append([])
        blocks[-1].append(i)
    out=[None]*len(keys);transition=[[strength*math.log(max(x,1e-12)) for x in row] for row in transitions]
    for block in blocks:
        emissions=[[math.log(max(p,1e-12)) for p in probabilities[i]] for i in block]
        forward=[emissions[0]]
        for e in emissions[1:]:
            row=[e[s]+logsum([forward[-1][j]+transition[j][s] for j in range(3)]) for s in range(3)]
            normal=max(row);forward.append([x-normal for x in row])
        backward=[[0.]*3 for _ in block]
        for j in range(len(block)-2,-1,-1):
            row=[logsum([transition[s][t]+emissions[j+1][t]+backward[j+1][t] for t in range(3)]) for s in range(3)]
            normal=max(row);backward[j]=[x-normal for x in row]
        for i,f,b in zip(block,forward,backward):
            logs=[x+y for x,y in zip(f,b)];normal=logsum(logs);out[i]=tuple(math.exp(x-normal) for x in logs)
    return out
