"""Independent endpoint-label/Ridge preparation oracle; no producer imports.

Real I/O is permitted only through an explicitly registered PLAN and completed
prepare flag. Synthetic callers supply their own arrays. The Ridge solver uses
augmented least squares, independently of the producer's normal equations.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import monthly_paired_target143_prepare_review as base

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'outputs/monthly_endpoint155'
LOCAL, TARGET, TRAIN_SOURCE, ARCH = base.LOCAL, base.TARGET, base.TRAIN_SOURCE, base.ARCH
old, prior, require = base.old, base.prior, base.require
NAMES = ('incumbent_local67', 'matched_mean_return_ridge', 'matched_loss_indicator_ridge')
EXTRA = ('mean_return_prediction', 'loss_prediction', 'mean_return_rank', 'negative_loss_rank', 'common_available')
LABEL_KEYS = ('dates', 'ends', 'symbols', 'return_target', 'loss_target', 'base_good', 'good')
INPUTS = ('proposal', 'meta', 'policy', 'original_orders', 'prices', 'protected_manifest', 'train_inputs', 'training_meta', 'design', 'calendar')
CACHE_FILES = base.CACHE_FILES
CHECKS = ('original27_arrays_byte_identical', 'training_identity_and_plain10_verified',
          'raw_prices_independently_verified', 'calendar_overlay_independently_verified',
          'endpoint_labels_independently_reconstructed', 'shared_maturity_rows_weights_scalers_verified',
          'annual_forecasts_independently_reconstructed', 'shared_query_mask_and_centered_ranks_verified',
          'all_three_score_matrices_and_ordered_choices_independently_verified', 'all_signal_prefixes_verified',
          'initial_partial_and_missing_exact_fallback_verified', 'one_episode_and_top2_lock_verified')
DESIGN = 'work/after154_endpoint_retest_design_v2.json'
ATOL = 1e-12
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
identity_oracle, day_vector, bool_vector = base.identity_oracle, base.day_vector, base.bool_vector
compare = prior.compare

def source_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    return sorted(set(base.source_files(root) + [root/'work'/n for n in (
        'monthly_endpoint155_prepare_review.py', 'test_monthly_endpoint155_prepare_review.py')]))

def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def month_ends(sessions):
    calendar = prior.canonical_axis(sessions)
    require(len(calendar) > 0, 'Nonempty session calendar')
    result = {}
    for day in calendar:
        result[day.to_period('M')] = day
    return result

def endpoint_schedule(sessions, dates):
    ends = month_ends(sessions)
    days = day_vector(dates, 'signals')
    out = []
    for d in days:
        date = pd.Timestamp(d); month = date.to_period('M')
        require(ends.get(month) == date, 'Training signal must be official month end')
        require(month + 1 in ends, 'Full following calendar month must be scheduled')
        out.append(ends[month + 1].to_datetime64())
    return np.asarray(out, dtype='datetime64[D]')

def query_monthends(sessions, dates):
    ends = month_ends(sessions); days = day_vector(dates, 'query')
    session_days = set(prior.canonical_axis(sessions))
    require(all(pd.Timestamp(d) in session_days for d in days), 'Every query must be an official session')
    values = []
    for i, d in enumerate(days):
        date = pd.Timestamp(d); valid = ends.get(date.to_period('M')) == date
        require(valid or (i == 0 and str(d) == '2016-08-17'), 'Only the initial partial query may be non-month-end')
        values.append(valid)
    return np.asarray(values, bool)

def labels_oracle(close, sessions, dates, symbols, common_good, eligible, finite_features, expected_ends=None):
    require(isinstance(close, pd.DataFrame), 'Price frame')
    prior.canonical_axis(close.index)
    require(close.columns.is_unique, 'Unique price securities')
    d = day_vector(dates, 'signals'); e = endpoint_schedule(sessions, d); n = len(d)
    s = np.asarray(symbols)
    require(s.shape == (n,) and all(isinstance(x, (str, np.str_)) and x for x in s), 'Explicit row identities')
    require(len(set(zip(d.astype(str), s))) == n, 'Unique security/month row')
    if expected_ends is not None:
        np.testing.assert_array_equal(day_vector(expected_ends, 'declared ends'), e)
    b = bool_vector(common_good,n,'common') & bool_vector(eligible,n,'eligible') & bool_vector(finite_features,n,'features')
    ret, loss = np.full(n,np.nan), np.full(n,np.nan)
    for i in np.flatnonzero(b):
        if s[i] not in close.columns or pd.Timestamp(d[i]) not in close.index or pd.Timestamp(e[i]) not in close.index:
            continue
        start, end = float(close.at[pd.Timestamp(d[i]),s[i]]), float(close.at[pd.Timestamp(e[i]),s[i]])
        if not (np.isfinite(start) and np.isfinite(end) and start>0 and end>0):
            continue
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            value = end/start-1.
        if np.isfinite(value):
            ret[i], loss[i] = value, float(end<start)
    good = b & np.isfinite(ret) & np.isfinite(loss)
    months = d.astype('datetime64[M]')
    for month in np.unique(months):
        rows = months == month
        if np.count_nonzero(good & rows) < 2:
            good[rows] = False
    return dict(dates=d, ends=e, symbols=s.copy(), return_target=ret, loss_target=loss, base_good=b, good=good)

def weights_oracle(dates, cutoff):
    d = day_vector(dates,'weight dates'); cutoff=np.datetime64(cutoff,'D')
    require(len(d)>0 and np.all(d<=cutoff), 'Mature nonempty weights')
    months = d.astype('datetime64[M]'); counts=Counter(months.astype(str))
    age = cutoff.astype('datetime64[M]').astype(int)-months.astype(int)
    raw = np.asarray([1./counts[str(m)] for m in months])*np.exp2(-age/12.)
    require(np.isfinite(raw).all() and raw.sum()>0, 'Positive finite weights')
    return raw*(len(raw)/raw.sum())

def fit_oracle(x,y,w):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    require(x.ndim==2 and x.shape[1]==10 and y.shape==w.shape==(len(x),), 'Ten-column weighted fit')
    require(len(x)>0 and np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all() and (w>0).all(), 'Finite positive fit inputs')
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    intercept=float(np.average(y,weights=w))
    design=np.vstack(((x-mean)/scale*np.sqrt(w[:,None]),np.sqrt(10.)*np.eye(10)))
    target=np.r_[(y-intercept)*np.sqrt(w),np.zeros(10)]
    coefficient=np.linalg.lstsq(design,target,rcond=None)[0]
    require(all(np.isfinite(v).all() for v in (mean,scale,coefficient,np.asarray(intercept))), 'Finite fit')
    return mean,scale,coefficient,intercept

def predict_oracle(fit,x):
    return np.sum(((x-fit[0])/fit[1])*fit[2],axis=1)+fit[3]

def model_oracle(labels,x,event_dates,event_x,event_good,event_monthend,fit_cache=None):
    require(set(labels)==set(LABEL_KEYS), 'Seven exact label arrays')
    d,e=day_vector(labels['dates'],'training'),day_vector(labels['ends'],'ends')
    qd=day_vector(event_dates,'query');n,q=len(d),len(qd)
    require(e.shape==(n,) and np.all(e>d) and np.all(e.astype('datetime64[M]')==d.astype('datetime64[M]')+1), 'Exact next-month target axes')
    x,qx=np.asarray(x,float),np.asarray(event_x,float)
    require(x.shape==(n,10) and qx.shape==(q,10), 'Original ten feature columns')
    good=bool_vector(labels['good'],n,'good') & np.isfinite(x).all(axis=1)
    targets=[np.asarray(labels[k],float) for k in ('return_target','loss_target')]
    require(all(t.shape==(n,) for t in targets), 'Target row axes')
    good &= np.isfinite(targets[0]) & np.isfinite(targets[1])
    require(np.isin(targets[1][np.isfinite(targets[1])], [0.,1.]).all(), 'Binary loss values')
    months=d.astype('datetime64[M]')
    for month in np.unique(months):
        if np.count_nonzero(good & (months==month))<2:good[months==month]=False
    valid=bool_vector(event_good,q,'query eligible') & bool_vector(event_monthend,q,'query monthend') & np.isfinite(qx).all(axis=1)
    years=qd.astype('datetime64[Y]').astype(int)+1970
    predictions=[np.full(q,np.nan),np.full(q,np.nan)];available=np.zeros(q,bool);annual=[];fits=0
    for year in np.unique(years):
        cutoff=np.datetime64('%04d-12-31'%(year-1),'D');start=np.datetime64('%04d-12-31'%(year-6),'D')
        train=good & (d>start) & (e<=cutoff);query=(years==year)&valid
        ids=np.flatnonzero(train);nmonths=len(np.unique(months[train]));status='fit' if nmonths>=18 and query.any() else 'cold_start' if nmonths<18 else 'no_queries'
        row=dict(year=int(year),cutoff=str(cutoff),lookback_exclusive_start=str(start),training_rows=len(ids),training_months=nmonths,
                 prediction_rows=int(query.sum()),first_training_signal=str(d[train].min()) if len(ids) else None,
                 latest_training_signal=str(d[train].max()) if len(ids) else None,latest_label_end=str(e[train].max()) if len(ids) else None,
                 training_indices_sha256=digest(ids),status=status)
        if status=='fit':
            require((np.abs(x[train])<=.5).all() and (np.abs(qx[query])<=.5).all(),'Original centered percentile features')
            weights=weights_oracle(d[train],cutoff)
            key=(int(year),digest(ids),digest(x[train]),digest(targets[0][train]),digest(targets[1][train]),digest(weights))
            models=fit_cache.get(key) if fit_cache is not None else None
            if models is None:
                models=[fit_oracle(x[train],t[train],weights) for t in targets]
                if fit_cache is not None:fit_cache[key]=models
            require(digest(models[0][0])==digest(models[1][0]) and digest(models[0][1])==digest(models[1][1]),'Shared training scaler')
            for target,fit in zip(predictions,models):target[query]=predict_oracle(fit,qx[query])
            available[query]=True;fits+=2
            row.update(weight_sum=float(weights.sum()),weights_sha256=digest(weights),scaler_mean_sha256=digest(models[0][0]),scaler_scale_sha256=digest(models[0][1]))
        annual.append(row)
    require(all(np.array_equal(np.isfinite(v),available) for v in predictions),'Both targets share finite query mask')
    return dict(mean_return_prediction=predictions[0],loss_prediction=predictions[1],common_available=available,
                audit=dict(annual=annual,new_pooled_model_fits=fits))

def rank_oracle(values):
    values=np.asarray(values,float);n=len(values)
    require(values.ndim==1 and np.isfinite(values).all(),'Finite rank population')
    if n<=1:return np.zeros(n)
    return np.asarray([(np.count_nonzero(values<x)+(np.count_nonzero(values==x)-1)/2)/(n-1)-.5 for x in values])

def ranked_arrays(forecast,shape):
    mean=np.asarray(forecast['mean_return_prediction']).reshape(shape).copy()
    loss=np.asarray(forecast['loss_prediction']).reshape(shape).copy()
    mask=np.asarray(forecast['common_available']).reshape(shape).copy()
    require(mask.dtype==bool and np.array_equal(np.isfinite(mean),mask) and np.array_equal(np.isfinite(loss),mask),'Common finite forecast mask')
    a,b=np.zeros(shape),np.zeros(shape)
    for i in range(shape[0]):
        a[i,mask[i]]=rank_oracle(mean[i,mask[i]])
        b[i,mask[i]]=rank_oracle(-loss[i,mask[i]])
    return dict(mean_return_prediction=mean,loss_prediction=loss,mean_return_rank=a,negative_loss_rank=b,common_available=mask)

def verify_arrays(original,arrays):
    require(len(original)==27 and set(arrays)==set(original)|set(EXTRA),'Original27 plus five forecasts')
    for k,v in original.items():
        require(arrays[k].shape==v.shape and arrays[k].dtype==v.dtype and arrays[k].tobytes()==v.tobytes(),'Original array bytes '+k)

def score_oracle(arrays,meta,n=None):
    helper=old.original_helper();x=helper.original_prefix(arrays,meta,n);n=x['n'];shape=x['core'].shape
    mask=np.asarray(arrays['common_available'])[:n]
    require(mask.shape==shape and mask.dtype==bool and not (mask & ~x['eligible']).any(),'Signal-time common mask subset')
    scores,choices={},{}
    for k,name in enumerate(NAMES):
        pre=x['local'].copy()
        if k:
            hand=np.asarray(arrays[EXTRA[k+1]])[:n]
            require(hand.shape==shape and hand.dtype==np.float64 and np.isfinite(hand).all() and (np.abs(hand)<=.5).all(), 'Centered bounded rank')
            require(not np.any(hand[~mask]), 'Missing correction is exact zero')
            use=mask & (hand!=0.)
            pre[use]+=.01*hand[use]
        choice,score,details=helper.episode(pre,x)
        require(all(set(row['top2'])<=set(row['selected']) if len(row['selected'])>=2 else True for row in details),'Frozen top-two lock')
        scores[name],choices[name]=score,choice
    return scores,choices

def ordered_orders(meta,scores,choices):
    return {name:[dict(date=d,signal_date=d,end=meta['ends'][i],symbols=[meta['symbols'][i][j] for j in row],scores=[float(scores[name][i,j]) for j in row])
                  for i,(d,row) in enumerate(zip(meta['dates'],choices[name]))] for name in NAMES}

def check_forecasts(actual,expected):
    error=0.
    for key in EXTRA:
        a,e=actual[key],expected[key]
        require(a.shape==e.shape and a.dtype==e.dtype,'Forecast shape/dtype '+key)
        if key in EXTRA[2:]:np.testing.assert_array_equal(a,e)
        else:
            np.testing.assert_allclose(a,e,rtol=0,atol=ATOL,equal_nan=True)
            if np.isfinite(e).any():error=max(error,float(np.max(np.abs(a[np.isfinite(e)]-e[np.isfinite(e)]))))
    # Independently derived ranks must agree exactly, even around near ties.
    return error

# I/O and proposal bindings are appended below once their public contract is fixed.

DESIGN_SHA = '91d8ed23f6aa53cde1ac62728bae2b08376f2f7ecc36076d6893d1bf37f2ba63'
PREVIOUS = 'outputs/monthly_financial_coverage154/COMPLETION.json'
PREVIOUS_SHA = 'cbc61cf6bd4501dedb3142be5c672a1b2f196ade55f42f9a3b14df66190fb749'
SOURCE_PLAN = 'outputs/monthly_paired_target143/PLAN.json'
SOURCE_PLAN_SHA = '2d73a3afb4745f5106f6cc870d0c25a775721181cd3c99ad78a7e1f56b58500b'

def selection_contract():
    return dict(primary=NAMES[2],controls=list(NAMES[:2]),first_months=88,
                same_close='strict_cagr_over_both_controls',next_open_5bp='noninferior_incumbent_strict_over_mean_return',
                cagr_epsilon=1e-12,sharpe_tolerance=.05,max_drawdown_tolerance=.03,
                early_cagr_tolerance=.025,recent_cagr_tolerance=.025,
                walk_forward=[[36,60],[60,76],[76,88]],later34_diagnostic_only=True)

def measurement_contract():
    return dict(design_sha256=DESIGN_SHA,predictor='unchanged_plain10',
        label='adjusted_close_next_calendar_month_end_over_signal_close_minus_one',
        loss='endpoint_close_strictly_below_signal_close',zero_is_loss=False,
        raw_simple_return=True,clip=False,center=False,spy_relative=False,
        shared_inherited_common_good=True,min_names_per_month=2,
        annual_prior_december31=True,exclusive_five_year_signal_bound=True,
        maturity='natural_month_endpoint_close_not_next_open',min_months=18,
        alpha=10.,half_life_months=12.,month_balanced=True,weighted_scaler=True,
        local_residual=False,class_balancing=False,probability_calibration=False,
        rank='average_rank0_over_n_minus1_minus_half',dose=.01,max_absolute_correction=.005,
        query_future_outcomes_used=False,first_partial='2016-08-17_exact_incumbent_fallback',
        final_partial='2026-09-24_replay_only_never_training_endpoint',original_arrays=27,
        extra_arrays=list(EXTRA),label_arrays=list(LABEL_KEYS),
        score_placement='unchanged_local67_pre_plus_dose_rank_then_one_episode')

def verify_proposal(reader,plan,pins):
    ref=plan['inputs']['proposal'];path=reader.path(ref['path'])
    require(reader.sha(path)==ref['sha256']==pins.get(str(path)),'Pinned proposal')
    p=reader.json(path)
    require(p.get('round')==155 and p.get('arms')==list(NAMES) and p.get('primary_challenger')==NAMES[2]
            and p.get('status')=='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION','Prospective three-arm proposal')
    compare(p.get('measurement'),measurement_contract(),'measurement')
    compare(p.get('selection_contract'),selection_contract(),'selection_contract')
    require(p.get('design')==dict(path=DESIGN,sha256=DESIGN_SHA) and
            p.get('previous_completion')==dict(path=PREVIOUS,sha256=PREVIOUS_SHA),'Design/predecessor refs')
    require(all(type(p.get(k)) is int and p[k]==0 for k in ('actual_new_labels','actual_new_model_fits','actual_new_orders','actual_new_returns')),'No preregistration actual work')
    for file,sha in ((DESIGN,DESIGN_SHA),(PREVIOUS,PREVIOUS_SHA),(SOURCE_PLAN,SOURCE_PLAN_SHA),(CALENDAR,CALENDAR_SHA)):
        require(reader.sha(file)==sha==pins.get(str(reader.path(file))),'Pinned inherited input '+file)
    return p

def validate_plan(reader,plan_path,plan_sha,stage='prepare'):
    require(re.fullmatch('[0-9a-f]{64}',plan_sha or '') is not None,'Explicit PLAN SHA')
    p=reader.path(STUDY);require(reader.path(plan_path)==p/'PLAN.json' and reader.sha(plan_path)==plan_sha,'Dedicated exact PLAN')
    plan=reader.json(plan_path)
    require(plan.get('round')==155 and plan.get('status')=='REGISTERED' and plan.get('arms')==list(NAMES)
            and plan.get('primary_challenger')==NAMES[2],'Registered endpoint protocol')
    require(reader.path(plan['output_directory'])==p,'Dedicated output directory')
    require(plan['runtime']==dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),'Pinned runtime')
    require(all(type(plan.get(k)) is int and plan[k]==0 for k in ('actual_label_calculations_before_registration','actual_feature_calculations_before_registration','actual_model_fits_before_registration','actual_returns_before_registration')),'Prospective scope')
    if stage=='prepare':
        require(not any((p/k).exists() for k in ('TRAIN_TRIALS.json','TRAIN_OPEN_TRIALS.json','FROZEN.json','RESULTS.json')),'Prepare review before outcomes')
    pins=reader.verify(plan['hashes']);require(set(plan['inputs'])==set(INPUTS),'Exact named inputs')
    for key,ref in plan['inputs'].items():
        require(reader.sha(ref['path'])==ref['sha256']==pins.get(str(reader.path(ref['path']))),'Bound named input '+key)
    verify_proposal(reader,plan,pins)
    sources={str(f):reader.sha(f) for f in source_files(reader.root)}
    require(reader.verify(plan['independent_prepare_source_hashes'])==sources and all(pins.get(k)==v for k,v in sources.items()),'Prepare source closure')
    original_plan=reader.json(LOCAL+'/PLAN.json')
    raw=reader.verify(plan['raw_source_hashes'])
    require(raw==reader.verify(original_plan['raw_source_hashes']) and len(raw)==64 and all(pins.get(k)==v for k,v in raw.items()),'Original raw64 identity')
    cached=reader.verify(reader.json(LOCAL+'/CACHE_HASHES.json'),reader.path(LOCAL))
    require(all(pins.get(k)==v for k,v in cached.items()),'Original cache closure')
    for key,file in (('meta','META.json'),('policy','INCUMBENT_POLICY.json'),('original_orders','local_linear_ORDERS.json')):
        require(reader.path(plan['inputs'][key]['path'])==reader.path(LOCAL+'/'+file),'Original input '+key)
    for key,path in (('prices',old.PRICES),('train_inputs',TARGET+'/TRAIN_INPUTS.npz'),('training_meta',TRAIN_SOURCE+'/META.json'),('design',DESIGN),('calendar',CALENDAR)):
        require(reader.path(plan['inputs'][key]['path'])==reader.path(path),'Original wiring '+key)
    require(plan['calendar']==original_plan['calendar'],'Calendar provenance')
    calendar_pins=reader.verify(plan['calendar']['source_hashes'])
    require(all(pins.get(k)==v for k,v in calendar_pins.items()),'Calendar source closure')
    require(plan['tolerances']['feature_atol']==ATOL and plan['tolerances']['feature_rtol']==0 and
            plan['tolerances']['scores_atol']==ATOL and plan['tolerances']['scores_rtol']==0,'Fixed tolerances')
    cache=reader.json(p/'CACHE.json');require(set(cache)==set(CACHE_FILES),'Exact cache files');reader.verify(cache,p)
    return p,plan,pins,sources,raw

def label_audit(labels):
    d=labels['dates']; records=[]
    for date in np.unique(d):
        rows=d==date;first=np.flatnonzero(rows)[0]
        endpoint=labels['base_good'][rows]&np.isfinite(labels['return_target'][rows])&np.isfinite(labels['loss_target'][rows])
        count=int(endpoint.sum())
        records.append(dict(signal_date=str(date),label_end=str(labels['ends'][first]),base_rows=int(labels['base_good'][rows].sum()),endpoint_rows=count,good_rows=int(labels['good'][rows].sum()),status='complete' if count>=2 else 'fewer_than_two_shared_rows'))
    return dict(targets=['return_target','loss_target'],rows=len(d),base_rows=int(labels['base_good'].sum()),good_rows=int(labels['good'].sum()),months=records,clipped=False,centered=False,
        loss_definition='1[adjusted_close[end] < adjusted_close[signal]]; exact zero is 0',maturity='next official calendar month-end close',
        inherited_common_good_retained=True,no_endpoint_shortening=True,calendar_completeness_certified_by_this_function=False)

def verify_model_audit(saved,forecast,labels,x):
    wanted=dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,
                new_pooled_model_fits=forecast['audit']['new_pooled_model_fits'],local_model_fits=0,
                shared_training_mask_sha256=digest(labels['good']&np.isfinite(x).all(axis=1)),
                training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False)
    require(set(saved)==set(wanted)|{'annual'},'Exact model audit fields')
    for k,v in wanted.items():compare(saved[k],v,'model/'+k)
    expected=forecast['audit']['annual'];require(len(saved['annual'])==len(expected),'All annual fits')
    for actual,row in zip(saved['annual'],expected):
        ids=np.flatnonzero(labels['good']&np.isfinite(x).all(axis=1)&(labels['dates']>np.datetime64(row['lookback_exclusive_start']))&(labels['ends']<=np.datetime64(row['cutoff'])))
        common={k:row[k] for k in ('year','cutoff','lookback_exclusive_start','training_rows','training_months','prediction_rows','latest_label_end','training_indices_sha256','status')}
        common.update(training_indices=ids.tolist(),training_dates_sha256=digest(labels['dates'][ids]),training_symbols_sha256=digest(labels['symbols'][ids]),model_fits=2 if row['status']=='fit' else 0)
        if row['status']=='fit':
            common.update({k:row[k] for k in ('weight_sum','weights_sha256','scaler_mean_sha256','scaler_scale_sha256')})
            common.update(target_sha256={k:digest(labels[k][ids]) for k in ('return_target','loss_target')},shared_rows_weights_scalers=True)
        compare(actual,common,'model/year/'+str(row['year']))

def rank_audit(arrays,meta):
    return dict(months=[dict(signal_index=i,date=d,shared_query_population=int(arrays['common_available'][i].sum()),
        available=bool(arrays['common_available'][i].any()),initial_partial_fallback=(d=='2016-08-17'),
        unchanged_return_correction=not bool(np.any(arrays['mean_return_rank'][i])),
        unchanged_loss_correction=not bool(np.any(arrays['negative_loss_rank'][i]))) for i,d in enumerate(meta['dates'])])

def verify_prefixes(arrays,meta,close,sessions,cached,symbols,expected,forecast,scores,choices):
    cache={};monthend=query_monthends(sessions,meta['dates']);qgood=arrays['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)
    for n in range(len(meta['dates'])+1):
        ss,cc=score_oracle(arrays,meta,n)
        for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Every score/order prefix')
        if not n:continue
        cutoff=pd.Timestamp(meta['dates'][n-1])
        lab=labels_oracle(close.loc[:cutoff],sessions,cached['dates'],symbols,cached['common_good'],cached['train_eligible'],np.isfinite(cached['train_plain']).all(axis=1),cached['ends'])
        with threadpool_limits(limits=1):f=model_oracle(lab,cached['train_plain'],cached['event_dates'][:n*49],cached['event_plain'][:n*49],qgood[:n*49],np.repeat(monthend[:n],49),fit_cache=cache)
        for key in ('mean_return_prediction','loss_prediction'):
            np.testing.assert_allclose(f[key],forecast[key][:n*49],rtol=0,atol=ATOL,equal_nan=True)
        np.testing.assert_array_equal(f['common_available'],forecast['common_available'][:n*49])
        ranks=ranked_arrays(f,(n,49))
        for key in EXTRA[2:]:np.testing.assert_array_equal(ranks[key],expected[key][:n])
    for kind in ('missing','zero'):
        test=dict(arrays,mean_return_rank=np.zeros_like(arrays['mean_return_rank']),negative_loss_rank=np.zeros_like(arrays['negative_loss_rank']))
        if kind=='missing':test['common_available']=np.zeros_like(arrays['common_available'])
        ss,cc=score_oracle(test,meta)
        for name in NAMES[1:]:require(cc[name]==choices[NAMES[0]] and ss[name].tobytes()==scores[NAMES[0]].tobytes(),'Exact '+kind+' fallback')

def audit(root,plan_path,plan_sha):
    r=old.Reader(root);p,plan,pins,sources,raw=validate_plan(r,plan_path,plan_sha)
    inputs={k:r.path(v['path']) for k,v in plan['inputs'].items()}
    meta=r.json(p/'META.json');require(meta==r.json(inputs['meta']),'Frozen metadata')
    require(len(meta['dates'])==len(meta['symbols'])==122 and meta['train_months']==88 and meta['early_months']==53 and len(meta['etfs'])==17 and all(len(v)==49 for v in meta['symbols']),'Original122x49 axis')
    a=r.npz(p/'DATA.npz');verify_arrays(r.npz(LOCAL+'/DATA.npz'),a)
    helper=old.original_helper();require(all(r.json(inputs['policy']).get(k)==v for k,v in helper.POLICY.items()),'Original episode policy')
    require(r.sha(old.LOADER)==old.LOADER_SHA,'Frozen price loader')
    spec=importlib.util.spec_from_file_location('endpoint155_independent_raw_loader',r.path(old.LOADER));loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    panel,info=loader.load_panel(meta['etfs']);used=dict(info['source_hashes']);used[str(r.path(old.LOADER))]=old.LOADER_SHA
    require({str(r.path(k)):v for k,v in used.items()}==raw,'Consumed raw64 closure');r.verify(used)
    price=r.npz(inputs['prices']);require(set(price)=={'dates','symbols','close','opening'},'Price cache schema')
    parity=old.price_parity(panel,price);compare(r.json(p/'PRICE_PARITY.json'),parity,'PRICE_PARITY')
    close=pd.DataFrame(price['close'],index=pd.DatetimeIndex(price['dates']),columns=price['symbols'])
    sessions=old.official_sessions(r,'2035-12-31');calendar=r.json(p/'CALENDAR.json')
    require(calendar==r.json(inputs['calendar']) and calendar['start']=='2008-01-01' and calendar['end']=='2035-12-31' and calendar['provenance']==plan['calendar'] and calendar['sessions']==[str(d.date()) for d in sessions],'Frozen full calendar and exception overlay')
    training=r.json(inputs['training_meta']);cached=r.npz(inputs['train_inputs'])
    ident=identity_oracle(meta,training,cached,r.npz(TRAIN_SOURCE+'/DATA.npz'),r.json(ARCH+'/META.json'))
    compare(r.json(p/'IDENTITY_AUDIT.json'),ident,'IDENTITY_AUDIT');np.testing.assert_array_equal(cached['eligible'],a['eligible'])
    symbols=np.asarray([s for row in training['training']['symbols'] for s in row])
    wanted=labels_oracle(close,sessions,cached['dates'],symbols,cached['common_good'],cached['train_eligible'],np.isfinite(cached['train_plain']).all(axis=1),cached['ends'])
    saved=r.npz(p/'LABELS.npz');require(set(saved)==set(LABEL_KEYS),'Seven label arrays')
    for key in LABEL_KEYS:
        require(saved[key].dtype==wanted[key].dtype and saved[key].shape==wanted[key].shape,'Label schema '+key)
        np.testing.assert_array_equal(saved[key],wanted[key])
    compare(r.json(p/'LABEL_AUDIT.json'),label_audit(wanted),'LABEL_AUDIT')
    qgood=a['eligible'].reshape(-1)&np.isfinite(cached['event_plain']).all(axis=1)
    with threadpool_limits(limits=1):forecast=model_oracle(wanted,cached['train_plain'],cached['event_dates'],cached['event_plain'],qgood,np.repeat(query_monthends(sessions,meta['dates']),49))
    verify_model_audit(r.json(p/'MODEL_AUDIT.json'),forecast,wanted,cached['train_plain'])
    expected=ranked_arrays(forecast,(122,49));error=check_forecasts(a,expected)
    independent=dict(a,**expected);scores,choices=score_oracle(independent,meta)
    compare(r.json(p/'RANK_AUDIT.json'),rank_audit(independent,meta),'RANK_AUDIT')
    orders=r.json(p/'PRETRAIN_ORDERS.json');actual_scores=r.npz(p/'SCORES.npz')
    compare(orders,ordered_orders(meta,scores,choices),'PRETRAIN_ORDERS');require(set(actual_scores)==set(NAMES),'Three score matrices')
    for name in NAMES:np.testing.assert_allclose(actual_scores[name],scores[name],rtol=0,atol=ATOL)
    require(actual_scores[NAMES[0]].tobytes()==scores[NAMES[0]].tobytes(),'Incumbent full score bytes')
    helper.assert_exact_tree(orders[NAMES[0]],r.json(inputs['original_orders']))
    for name in NAMES[1:]:helper.assert_exact_tree(orders[name][0],orders[NAMES[0]][0])
    verify_prefixes(independent,meta,close,sessions,cached,symbols,expected,forecast,scores,choices)
    required={str(p/k):r.sha(p/k) for k in (*CACHE_FILES,'PLAN.json','CACHE.json')};allowed=dict(pins,**required)
    for path,sha in list(r.seen.items()):require(r.sha(path)==sha==allowed.get(path),'Unbound consumed artifact '+path)
    return dict(schema_version=1,passed=True,stage='prepare',plan_sha256=plan_sha,cache_sha256=r.sha(p/'CACHE.json'),orders_sha256=r.sha(p/'PRETRAIN_ORDERS.json'),
        **{k:True for k in CHECKS},source_hashes=sources,artifact_hashes=dict(r.seen),producer_modules_imported=False,returns_or_nav_computed=False,
        orders_checked=366,candidate_cells_checked=5978,source_cells_checked=5978,stock_cells_checked=3904,training_rows_checked=len(symbols),months_checked=122,
        prefixes_checked=123,forecast_arrays_verified=5,new_model_fits=forecast['audit']['new_pooled_model_fits'],independent_forecast_ranks_and_orders_verified=True,
        independent_numerical_solver='augmented weighted least squares; no producer imports',max_forecast_error=error,price_parity=parity,
        limitations=['Historical repetition not excluded; not pristine out of sample.','Inherited full-path/common-good restrictions remain in both training targets.','Linear binary Ridge is not a calibrated probability.','Preparation checks no portfolio returns.'])

def synthetic_receipt(result,root=None):
    root=ROOT if root is None else Path(root).resolve()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    implementation=[root/'work'/('monthly_endpoint155_'+n+'.py') for n in ('model','runner','lifecycle')]
    inputs=[root/'work/monthly_endpoint155_proposal.json',root/DESIGN]
    return dict(round=155,stage='preregistration_prepare_review',passed=result.wasSuccessful(),synthetic_only=True,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        real_155_label_fit_or_return_calculations=0,proposal_sha256=sha(inputs[0]),
        source_hashes={str(p):sha(p) for p in source_files(root)},implementation_hashes={str(p):sha(p) for p in implementation},input_hashes={str(p):sha(p) for p in inputs},
        arms=list(NAMES),required_checks=list(CHECKS),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha',required=True);p.add_argument('--prepare-complete',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    require(args.prepare_complete,'Explicit completed preparation required')
    result=audit(args.root,args.plan or args.root/STUDY/'PLAN.json',args.plan_sha);out=args.output or args.root/STUDY/'INDEPENDENT_PREPARE.json'
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__=='__main__':main()
