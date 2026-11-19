#!/usr/bin/env python3
"""Replay state 2-to-1 corrections on the original held-out pulse labels."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEQ = ['scenario_id', 'data_version', 'platform_id', 'sequence_id']
KEY = ['scenario_id', 'data_version', 'platform_id', 'source_kind', 'source_id', 'bin_index']
YCOL = ['state_0_pulses', 'state_1_pulses', 'state_2_pulses']
EXCLUDED = {'103004','103007','104004'}
TIME_FOLDS = {'check_300_320':(300,320), 'selection':(320,400),
              'check_400_420':(400,420), 'diagnostic_420_500':(420,500)}
MAX_ACCURACY_LOSS = .005


def load(root):
    source = root/'output/calibrated_sequence_state_features/train'
    meta = json.loads((source/'state_features_run.json').read_text())
    if meta['labels_used_in_features']:
        raise ValueError('State features must contain observations only')
    data = pd.read_csv(source/'window_features.csv', dtype={'platform_id':str})
    labels = pd.read_csv(source/'window_truth.csv', dtype={'platform_id':str})
    data = data.merge(labels,on=KEY,validate='one_to_one')
    if len(data) != len(labels) or not np.array_equal(data[YCOL].sum(axis=1),data.pulse_count):
        raise ValueError('Incomplete state truth')
    target = [json.loads(v) for v in data.pop('target_counts_json')]
    data['target'] = [next(iter(v)) if len(v)==1 else '' for v in target]
    data['mid_seconds'] = (data.first_time_ns + data.last_time_ns)/2e9
    baseline = pd.read_csv(root/'CPI时序状态实验/results/joint_class_state_validation.csv',
                           dtype={'platform_id':str})
    needed = SEQ+['fold','three_way_target_state','original_coarse_score','cpi_context_coarse_score']
    data = data.merge(baseline[needed],on=SEQ,how='left',validate='one_to_one')
    return data,meta['features']


def add_cadence(data, partition):
    """Measure short-window activity within an observation group and split."""
    group = data.source_id.str.rsplit('~',n=1).str[0]
    group = np.asarray(group, dtype=object)
    timing = data.mid_seconds.to_numpy()
    span = data.sequence_log_span_seconds.to_numpy()
    pri = data.observed_log_PRI_us.to_numpy()
    power = data.sequence_local_PA_rank.to_numpy()
    feature_names = ['cadence_%s_%s' % (d,n) for d in ('before','after') for n in ('025','100','500','2000')]
    for direction in ('before','after'):
        feature_names.extend('%s_delta_%s' % (direction,n) for n in ('span','pri','rank'))
        feature_names.append('%s_gap' % direction)
    out = np.full((len(data),len(feature_names)),np.nan,dtype=np.float32)
    keys = data[KEY[:4]].copy()
    keys['group'] = group
    keys['partition'] = partition
    for ids in keys.groupby(list(keys.columns),sort=False).indices.values():
        ix = ids[np.argsort(timing[ids],kind='stable')]
        t = timing[ix]
        valid = np.arange(len(ix))
        for j,w in enumerate((.025,.1,.5,2.)):
            out[ix,j] = np.log1p(valid-np.searchsorted(t,t-w,side='left'))
            out[ix,j+4] = np.log1p(np.searchsorted(t,t+w,side='right')-valid-1)
        for j,(v,w) in enumerate(zip((ix[1:],ix[:-1]), (ix[:-1],ix[1:]))):
            near = np.abs(timing[v]-timing[w]) <= .25
            v,w = v[near],w[near]
            for k,f in enumerate((span,pri,power)):
                out[v,8+4*j+k] = f[v]-f[w]
            out[v,11+4*j] = np.log1p(np.abs(timing[v]-timing[w])*1000)
    return pd.DataFrame(out,index=data.index,columns=feature_names)


def predictions_at(base, score, threshold):
    result = base.copy()
    result[(base == 2) & (score >= threshold)] = 1
    return result


def metrics(pred, truth):
    cm = np.zeros((3,3),dtype=np.int64)
    for i in range(3):
        cm[:,i] = truth[pred==i].sum(axis=0)
    diag = np.diag(cm)
    actual, reported = np.maximum(cm.sum(axis=1),1),np.maximum(cm.sum(axis=0),1)
    precision, recall = diag/reported,diag/actual
    f1 = 2*diag/(actual+reported)
    return {'pulses':int(cm.sum()),'accuracy':float(diag.sum()/cm.sum()),'macro_f1':float(f1.mean()),
            'coarse_precision':float(precision[1]),'coarse_recall':float(recall[1]),'coarse_f1':float(f1[1]),
            'coarse_tp':int(diag[1]),'coarse_predictions':int(cm[:,1].sum()),'confusion':cm.tolist()}


def select(base, score, truth):
    old = metrics(base,truth)
    options = score[(base==2) & (score>=0)]
    values = np.unique(np.r_[np.quantile(options,np.linspace(0,1,201)),1.01]) if options.size else [1.01]
    best = None
    for threshold in values:
        result = metrics(predictions_at(base,score,threshold),truth)
        if (result['accuracy'] >= old['accuracy']-MAX_ACCURACY_LOSS
                and result['coarse_precision'] >= old['coarse_precision']):
            record = (result['coarse_f1'],result['accuracy'],float(threshold),result)
            if best is None or record[:3]>best[:3]:
                best=record
    if best is None:
        raise ValueError('No eligible threshold, including unchanged predictions')
    return best[2],best[3]


def fit_booster(x, y, weights, n_leaves):
    model = lgb.LGBMClassifier(n_estimators=270,num_leaves=n_leaves,max_depth=6,learning_rate=.045,
                               min_child_samples=45,reg_lambda=2.,feature_fraction=.86,
                               deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=3,random_state=41)
    model.fit(x,y,sample_weight=weights/weights.mean())
    return model


def matched_fine_examples(training, labels):
    """Find difficult fine-track examples from the same target and observer."""
    result = np.zeros(len(training),dtype=bool)
    times = training.mid_seconds.to_numpy()
    rf = training.observed_RF_ghz.to_numpy()
    pw = training.observed_PW_us.to_numpy()
    for rows in training.groupby(['target','platform_id'],sort=False).indices.values():
        positive = rows[labels[rows]==1]
        if not len(positive):
            continue
        positive = positive[np.argsort(times[positive])]
        known = times[positive]
        for i in rows[labels[rows]==0]:
            lo,hi = np.searchsorted(known,[times[i]-8.,times[i]+8.])
            nearby = positive[lo:hi]
            result[i] = np.any((np.abs(rf[nearby]-rf[i])<=.12) & (np.abs(pw[nearby]-pw[i])<=.4))
    return result


def fit_classical(estimator, x, y, weights, scale=False):
    transforms = [SimpleImputer(strategy='median',add_indicator=True)]
    if scale:
        transforms.append(StandardScaler())
    model = make_pipeline(*transforms,estimator)
    weight_key = model.steps[-1][0]+'__sample_weight'
    model.fit(x,y,**{weight_key:weights/weights.mean()})
    return model


def nearby_coarse_anchors(data, base):
    """Trust only anchors from the frozen decision in a held-out data block."""
    one = np.full(len(data),-1.)
    both = one.copy()
    keys = data[KEY[:4]].copy()
    keys['group'] = data.source_id.str.rsplit('~',n=1).str[0]
    t,rf,pw,pri = (data[name].to_numpy() for name in ('mid_seconds','observed_RF_ghz','observed_PW_us','observed_log_PRI_us'))
    for ids in keys.groupby(list(keys.columns),sort=False).indices.values():
        anchors = ids[base[ids]==1]
        if not len(anchors):
            continue
        anchors = anchors[np.argsort(t[anchors])]
        times = t[anchors]
        for i in ids[base[ids]==2]:
            inner = anchors[np.searchsorted(times,t[i]-.25):np.searchsorted(times,t[i]+.25,side='right')]
            agree = (np.abs(rf[inner]-rf[i]) <= .12) & (np.abs(pw[inner]-pw[i]) <= .4) & (np.abs(pri[inner]-pri[i]) <= .33)
            dt = t[inner[agree]]-t[i]
            if not len(dt):
                continue
            one[i] = 1-np.min(np.abs(dt))/.25
            prev,nxt = dt[dt<0],dt[dt>0]
            if len(prev) and len(nxt):
                both[i] = 1-max(-prev.max(),nxt.min())/.25
    return one,both


def observed_events(root):
    path = root/'output/fused_b_assisted_guarded/train/cross_platform_events.csv'
    events = []
    cols = ['scenario_id','data_version','object_id','platform_ids','sequence_ids']
    for scene,version,oid,nodes,ids in pd.read_csv(path,usecols=cols,keep_default_na=False)[cols].itertuples(index=False,name=None):
        if oid:
            events.append([(scene,version,node,sid) for node,sid in zip(nodes.split('|'),ids.split('|'))])
    return events


def cross_platform_anchors(data, base, events):
    ix = {key:i for i,key in enumerate(data[SEQ].itertuples(index=False,name=None))}
    result = np.full(len(data),-1.)
    support = np.zeros(len(data),dtype=bool)
    for event in events:
        ids = [ix[k] for k in event if k in ix]
        if len(ids) < 2:
            continue
        support[ids] = True
        pred = base[ids]
        n_coarse = int((pred==1).sum())
        if n_coarse and not np.any(pred==0):
            for i in ids:
                if base[i]==2:
                    result[i] = n_coarse/(len(ids)-1)
    return result,support


def run(args):
    data, names = load(args.zgy)
    labels = data[YCOL].to_numpy(dtype=np.int64)
    state = labels.argmax(axis=1)
    fit = ((data.target != '') & ~data.target.isin(EXCLUDED) & (data.first_time_ns>=0)
           & (data.last_time_ns<300e9) & (labels.max(axis=1)>=.95*labels.sum(axis=1)) & (state>0))
    fold_names = data.fold.dropna().drop_duplicates().tolist()
    required = ['selection','diagnostic_420_500','instance_holdout']
    if any(f not in fold_names for f in required):
        raise ValueError('Expected saved validation folds, got %r' % fold_names)
    masks = {f:data.fold==f for f in fold_names}
    ref = {f: data.loc[m,'three_way_target_state'].to_numpy(dtype=int) for f,m in masks.items()}
    ref_metrics = {f:metrics(ref[f],labels[m]) for f,m in masks.items()}
    print('fitting pulses/records',int(labels[fit].sum()),int(fit.sum()), 'coarse records',int((fit & (state==1)).sum()),flush=True)

    partition = np.full(len(data),'other',dtype=object)
    start = data.first_time_ns.to_numpy(); end = data.last_time_ns.to_numpy()
    for name,(a,b) in dict(TIME_FOLDS,development=(0,300)).items():
        partition[(start>=a*1e9)&(end<b*1e9)] = name
    cadence = add_cadence(data,partition)
    relative = [n for n in names if not n.startswith('observed_')]
    x = pd.concat([data[names],cadence],axis=1)
    rowweights = np.minimum(data.loc[fit,'pulse_count'].to_numpy(dtype=float),256.)
    targets = data.loc[fit,'target'].to_numpy()
    y = (state[fit]==1).astype(int)
    target_totals = pd.Series(rowweights).groupby([targets,y]).transform('sum').to_numpy()
    adjusted = rowweights / np.sqrt(target_totals)

    scores = {name:{f:data.loc[m,col].to_numpy() for f,m in masks.items()} for name,col in (
                 ('original_two_stage_score','original_coarse_score'),('previous_CPI_score','cpi_context_coarse_score'))}
    models = [('boosted_sequence',names,rowweights,21),
              ('boosted_cadence',names+list(cadence.columns),rowweights,21),
              ('boosted_relative_cadence',relative+list(cadence.columns),adjusted,15),
              ('boosted_balanced_cadence',names+list(cadence.columns),adjusted,15)]
    for name,columns,weights,leaves in models:
        model = fit_booster(x.loc[fit,columns],y,weights,leaves)
        scores[name] = {f:model.predict_proba(x.loc[m,columns])[:,1] for f,m in masks.items()}
        print('trained',name,flush=True)

    matched = matched_fine_examples(data.loc[fit],y)
    hard_weights = adjusted * np.where((y==0)&~matched,.35,1.)
    classical_columns = relative+list(cadence.columns)
    for name,estimator,weights,scale in (
        ('logistic_relative',LogisticRegression(C=.3,max_iter=600),adjusted,True),
        ('logistic_matched_fine',LogisticRegression(C=.3,max_iter=600),hard_weights,True),
        ('extra_trees_matched_fine',ExtraTreesClassifier(n_estimators=240,max_depth=14,
                                                       min_samples_leaf=12,random_state=41,n_jobs=3),hard_weights,False),
        ('random_forest_matched_fine',RandomForestClassifier(n_estimators=240,max_depth=14,
                                                            min_samples_leaf=12,random_state=73,
                                                            n_jobs=3),hard_weights,False),
    ):
        model = fit_classical(estimator,x.loc[fit,classical_columns],y,weights,scale)
        scores[name] = {f:model.predict_proba(x.loc[m,classical_columns])[:,1] for f,m in masks.items()}
        print('trained',name,flush=True)

    events = observed_events(args.zgy)
    events_by_fold = {}
    for name in ('boosted_consensus','near_anchor_consensus','bracketed_coarse_anchor',
                 'cross_platform_anchor','cross_platform_anchor_and_consensus'):
        scores[name] = {}
    for f,m in masks.items():
        agree = np.min([scores[name][f] for name,_,_,_ in models],axis=0)
        one,both = nearby_coarse_anchors(data.loc[m],ref[f])
        peer,support = cross_platform_anchors(data.loc[m],ref[f],events)
        scores['boosted_consensus'][f] = agree
        scores['near_anchor_consensus'][f] = np.minimum(one,agree)
        scores['bracketed_coarse_anchor'][f] = both
        scores['cross_platform_anchor'][f] = peer
        scores['cross_platform_anchor_and_consensus'][f] = np.minimum(peer,agree)
        events_by_fold[f] = {'coarse_pulses_in_folds':int(labels[m,1].sum()),
                             'coarse_pulses_with_peer_event':int(labels[m,1][support].sum())}

    candidates = {}
    t = 'selection'
    for name,by_fold in scores.items():
        threshold,tuning = select(ref[t],by_fold[t],labels[masks[t]])
        metrics_by_fold = {f:metrics(predictions_at(ref[f],s,threshold),labels[masks[f]])
                           for f,s in by_fold.items()}
        holds = ('diagnostic_420_500','instance_holdout')
        passes = all(metrics_by_fold[f]['accuracy'] >= ref_metrics[f]['accuracy']-MAX_ACCURACY_LOSS
                     and metrics_by_fold[f]['coarse_precision'] >= ref_metrics[f]['coarse_precision']
                     and metrics_by_fold[f]['coarse_recall'] > ref_metrics[f]['coarse_recall'] for f in holds)
        candidates[name] = {'threshold':threshold,'metrics':metrics_by_fold,'passes_both_holdouts':bool(passes)}
        print('candidate',name,'threshold',threshold,'pass',passes,
              {f:[round(metrics_by_fold[f][k],4) for k in ('accuracy','coarse_precision','coarse_recall')]
               for f in holds},flush=True)
    best = max(candidates,key=lambda n:candidates[n]['metrics'][t]['coarse_f1'])
    coarse_by_target = {}
    for fold, mask in masks.items():
        cur = data.loc[mask,['target','state_1_pulses']].copy()
        cur['coarse_correct'] = cur.state_1_pulses.to_numpy()*(ref[fold]==1)
        count = cur.groupby('target').sum().sort_values('state_1_pulses',ascending=False)
        coarse_by_target[fold] = count[count.state_1_pulses>0].to_dict(orient='index')
    report = {'method':'Keep existing search and coarse predictions. Reconsider only fine tracking.',
              'fit':'Pure PRI sequences in 0–300 s, holding out three target instances.',
              'threshold_selection':'320–400 s only; no decrease in coarse precision and <=0.5 pp loss in accuracy.',
              'replay_scope':'Original scenario and frozen memberships. New cadence features stay within each time block. '
                             'Existing observable features retain the upstream offline context.',
              'baseline':ref_metrics,'candidates':candidates,'picked_on_selection':best,
              'picked_passes_holdouts':candidates[best]['passes_both_holdouts'],
              'baseline_coarse_by_target':coarse_by_target,
              'coarse_peer_event_coverage':events_by_fold,
              'matched_fine_fit_records':int(matched.sum()),
              'coarse_fit_records':int((fit & (state==1)).sum()), 'fine_fit_records':int((fit & (state==2)).sum())}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


NEURAL_FEATURES = {
    'waveform': ['observed_RF_ghz','observed_PW_us'],
    'pri': ['observed_log_PRI_us','sequence_relative_PRI_width','sequence_direct_interval_fraction'],
    'amplitude': ['observed_PA_dbm','PA_std_db','PA_range_db','sequence_local_PA_rank'],
    'activity': ['sequence_log_pulses','sequence_log_span_seconds','sequence_density',
                 'observed_span_fraction','within_bin_gap_cv'],
}


def neighboring_sequences(data):
    """Indices for ordered CPI candidates within one observer and one split."""
    times = data.mid_seconds.to_numpy()
    keys = data[KEY[:4]].copy()
    keys['group'] = data.source_id.str.rsplit('~',n=1).str[0]
    keys['split'] = data['neural_partition']
    order = np.full((len(data),7),-1,dtype=np.int32)
    order[:,3] = np.arange(len(data))
    for indices in keys.groupby(list(keys.columns),sort=False).indices.values():
        ids = indices[np.argsort(times[indices],kind='stable')]
        t = times[ids]
        for shift in (1,2,3):
            if len(ids)<=shift:
                continue
            valid = (t[shift:]-t[:-shift]) <= .35*shift
            left,right = ids[:-shift][valid],ids[shift:][valid]
            order[left,3+shift] = right
            order[right,3-shift] = left
    return order


def ordered_array(observations, times, order):
    missing = order<0
    normalized = observations[np.maximum(order,0)].copy()
    normalized[missing] = 0
    dt = times[np.maximum(order,0)]-times[order[:,3],None]
    dt = np.sign(dt)*np.log1p(np.abs(dt)*100)/5
    dt[missing] = 0
    return np.concatenate([normalized,dt[:,:,None],(~missing)[:,:,None]],axis=2).astype(np.float32)


def weighted_tracking_ap(scores, labels):
    """Average precision for coarse vs fine using exact pulse counts."""
    idx = np.argsort(-scores,kind='stable')
    positive = labels[idx,1].cumsum().astype(float)
    total = (labels[idx,1]+labels[idx,2]).cumsum().astype(float)
    if not positive[-1]:
        return None
    ends = np.r_[np.flatnonzero(np.diff(scores[idx])),len(idx)-1]
    p = positive[ends]
    return float(np.sum((p/np.maximum(total[ends],1))*np.diff(np.r_[0,p]))/p[-1])


def permute_groups(block, seed):
    sources = block[KEY[:4]].copy()
    sources['group'] = block.source_id.str.rsplit('~',n=1).str[0]
    rng = np.random.default_rng(seed)
    donor = np.arange(len(block))
    for ids in sources.groupby(list(sources.columns),sort=False).indices.values():
        donor[ids] = rng.permutation(ids)
    return donor


def neural_reference_diagnostics(data):
    fields = ['observed_PA_dbm','sequence_local_PA_rank','observed_log_PRI_us',
              'sequence_relative_PRI_width','sequence_log_span_seconds','sequence_log_pulses',
              'sequence_density','PA_range_db','log_previous_gap_seconds']
    reference, groups = {},{}
    for fold in ('selection','diagnostic_420_500','instance_holdout'):
        block = data.loc[data.fold==fold]
        counts = block[YCOL].to_numpy(dtype=np.int64)
        reference[fold] = {n:weighted_tracking_ap(block[c].to_numpy(),counts) for n,c in (
            ('original_two_stage','original_coarse_score'),('prior_CPI_context','cpi_context_coarse_score'))}
        state = counts.argmax(axis=1)
        pure = counts.max(axis=1)>=.95*counts.sum(axis=1)
        values = block[fields].to_numpy()
        t,rf,pw = (block[n].to_numpy() for n in ('mid_seconds','observed_RF_ghz','observed_PW_us'))
        by_target = {}
        for (target,_),ids in block.groupby(['target','platform_id']).indices.items():
            if not target:
                continue
            coarse = ids[pure[ids] & (state[ids]==1)]
            fine = ids[pure[ids] & (state[ids]==2)]
            fine = fine[np.argsort(t[fine],kind='stable')]
            known = t[fine]
            for i in coarse:
                lo,hi = np.searchsorted(known,[t[i]-8,t[i]+8])
                neighbors = fine[lo:hi]
                similar = neighbors[(np.abs(rf[neighbors]-rf[i])<=.12)&(np.abs(pw[neighbors]-pw[i])<=.4)]
                if len(similar):
                    j = similar[np.argmin(np.abs(t[similar]-t[i]))]
                    by_target.setdefault(target,[]).append(values[i]-values[j])
        groups[fold] = {'pure_coarse_sequences':int((pure & (state==1)).sum()),
                        'matched_pairs':sum(len(v) for v in by_target.values()),
                        'by_target':{target:{'pairs':len(diff),'median_coarse_minus_fine':dict(zip(fields,
                            map(float,np.median(diff,axis=0))))} for target,diff in by_target.items()}}
    return reference,groups


def neural_experiment(args):
    import torch
    from torch import nn

    class Classifier(nn.Module):
        def __init__(self, width, temporal):
            super().__init__()
            self.temporal = temporal
            if temporal:
                self.recur = nn.GRU(width,32,batch_first=True,bidirectional=True)
            self.head = nn.Sequential(nn.Linear(width+64 if temporal else width,64),nn.GELU(),
                                      nn.Dropout(.15),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

        def forward(self, rows):
            if self.temporal:
                temporal,_ = self.recur(rows)
                rows = torch.cat((rows[:,3],temporal[:,3]),dim=-1)
            return self.head(rows).squeeze(-1)

    torch.set_num_threads(4)
    data,_ = load(args.zgy)
    labels = data[YCOL].to_numpy(dtype=np.int64)
    truth = labels.argmax(axis=1)
    dev = ((data.target!='') & ~data.target.isin(EXCLUDED) & (data.last_time_ns<300e9))
    fit = dev & (truth>0) & (labels.max(axis=1)>=.95*labels.sum(axis=1))
    data['neural_partition'] = data.fold.fillna('unassigned')
    data.loc[dev,'neural_partition'] = 'development'
    names = [n for group in NEURAL_FEATURES.values() for n in group]
    obs = data[names].to_numpy(dtype=np.float32)
    center = obs[fit].mean(axis=0)
    scale = np.maximum(obs[fit].std(axis=0),1e-6)
    obs = np.clip((obs-center)/scale,-8,8)
    times = data.mid_seconds.to_numpy()
    order = neighboring_sequences(data)
    folds = ('selection','diagnostic_420_500','instance_holdout','check_300_320','check_400_420')
    idx = {f:np.flatnonzero(data.fold.to_numpy()==f) for f in folds}
    ref = {f:data.loc[ix,'three_way_target_state'].to_numpy(dtype=int) for f,ix in idx.items()}
    ref_metrics = {f:metrics(ref[f],labels[ix]) for f,ix in idx.items()}
    y = (truth[fit]==1).astype(np.float32)
    weights = np.minimum(data.loc[fit,'pulse_count'].to_numpy(dtype=np.float32),256.)
    target = data.loc[fit,'target'].to_numpy()
    totals = pd.Series(weights).groupby([target,y]).transform('sum').to_numpy()
    weights /= np.sqrt(totals)
    weights /= weights.mean()
    wy,ty = torch.from_numpy(weights),torch.from_numpy(y)
    fit_indices = np.flatnonzero(fit)
    output = {}
    for model_name, temporal, subset in [('single_sequence_mlp',False,names),('neighbor_gru',True,names),
                                        ('relative_neighbor_gru',True,[n for n in names if n not in
                                        ('observed_RF_ghz','observed_PA_dbm')])]:
        torch.manual_seed(291)
        rng = np.random.default_rng(291)
        columns = [names.index(n) for n in subset]
        values = obs[:,columns]
        width = len(subset)+2*int(temporal)
        model = Classifier(width,temporal)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0016,weight_decay=.005)

        def model_input(ix):
            return ordered_array(values,times,order[ix]) if temporal else values[ix].copy()

        x = torch.from_numpy(model_input(fit_indices))
        for _ in range(20):
            model.train()
            for batch in np.array_split(rng.permutation(len(x)),max(1,len(x)//512)):
                score = model(x[batch])
                loss = nn.functional.binary_cross_entropy_with_logits(score,ty[batch],weight=wy[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()

        def predict(a):
            with torch.no_grad():
                return np.concatenate([torch.sigmoid(model(torch.from_numpy(v))).numpy()
                                       for v in np.array_split(a,max(1,len(a)//2048))])

        test_x = {f:model_input(i) for f,i in idx.items()}
        score = {f:predict(a) for f,a in test_x.items()}
        threshold,_ = select(ref['selection'],score['selection'],labels[idx['selection']])
        reports = {f:metrics(predictions_at(ref[f],scores,threshold),labels[idx[f]]) for f,scores in score.items()}
        importance, ap = {},{}
        for f, indices in idx.items():
            ap[f] = weighted_tracking_ap(score[f],labels[indices])
            if f not in ('diagnostic_420_500','instance_holdout'):
                continue
            importance[f] = {}
            for group, members in NEURAL_FEATURES.items():
                chosen = [subset.index(n) for n in members if n in subset]
                drops = []
                for seed in range(3):
                    donor = permute_groups(data.loc[indices],seed)
                    modified = test_x[f].copy()
                    modified[...,chosen] = test_x[f][donor][...,chosen]
                    other = weighted_tracking_ap(predict(modified),labels[indices])
                    drops.append(ap[f]-other)
                importance[f][group] = float(np.mean(drops))
            if temporal:
                modified = test_x[f].copy()
                modified[:,:3] = 0
                modified[:,4:] = 0
                importance[f]['hide_neighbor_slots'] = ap[f]-weighted_tracking_ap(predict(modified),labels[indices])
        passed = all(reports[f]['accuracy']>=ref_metrics[f]['accuracy']-MAX_ACCURACY_LOSS
                     and reports[f]['coarse_precision']>=ref_metrics[f]['coarse_precision']
                     and reports[f]['coarse_recall']>ref_metrics[f]['coarse_recall']
                     for f in ('diagnostic_420_500','instance_holdout'))
        output[model_name] = {'inputs':subset,'threshold':threshold,'metrics':reports,'tracking_AP':ap,
                              'within_source_permutation_AP_loss':importance,'passes_both_holdouts':bool(passed)}
        print(model_name,'threshold',threshold,'pass',passed,'AP',ap,
              {f:[round(reports[f][k],4) for k in ('accuracy','coarse_precision','coarse_recall')]
               for f in ('diagnostic_420_500','instance_holdout')},flush=True)

    report = json.loads(args.report.read_text()) if args.report.exists() else {}
    baseline_ap,matched = neural_reference_diagnostics(data)
    report['deep_learning'] = {'baseline':ref_metrics,'models':output,'traditional_reference_AP':baseline_ap,
                               'same_target_paired_features':matched,
                               'training_sequences':int(fit.sum()),'fit_coarse_sequences':int(y.sum()),
                               'notes':'20 fixed epochs, same time/instance holdouts. Only fine-to-coarse changes. '
                                       'Source IDs select context only. Importance measures conditional AP change, not causality.'}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zgy',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--deep',action='store_true',help='Train only the PyTorch comparison and append its report')
    args = parser.parse_args()
    if args.deep:
        neural_experiment(args)
    else:
        run(args)


if __name__=='__main__':
    main()
