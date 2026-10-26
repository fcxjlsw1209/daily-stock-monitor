"""Independent common-x stock/ETF mixture calibration and unchanged episode oracle.

No production234 computation is imported. New transforms use scalar integer
comparison counts. Frozen original independent pre-score and accounting helpers
retain their pinned contract; there is no real-data entry point or file output.
"""
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
SOURCE_HELPER = ROOT / 'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
ARMS = ('incumbent_local67', 'asset_mixture_equal', 'asset_mixture_pooled_control')
PATHS = (('same_close_zero','close',0.),('next_open_zero','open',0.),
         ('next_open_cost','open',.0005))
DELTA_KEYS = dict(asset_mixture_equal='candidate_delta', asset_mixture_pooled_control='control_delta')
VALUE_KEYS = ('support', 'weight', 'candidate', 'control', 'candidate_delta', 'control_delta')
COUNT_FIELDS = ('events','symbols_per_event','checked_cells','stock_cells','etf_cells',
    'both_group_rows','single_group_rows','constant_rows','active_forecast_cells',
    'group_cdf_evaluations','pooled_cdf_evaluations','new_model_fits','predictive_model_fits')
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','full_auxiliary_support_checked',
    'core_weights_checked','full_candidate_universe','asset_type_identity_checked',
    'common_x_group_cdfs_checked','equal_group_mass_checked','pooled_midcdf_control_checked',
    'exact_ties_checked','canonical_operation_order_checked','constant_rows_unchanged',
    'constant_subgroups_retained_checked','empty_group_copy_checked','first_partial_unchanged',
    'effective_support_weight_checked','no_eligibility_rank_filter','no_new_availability_filter',
    'no_future_or_outcome_inputs','audit_checked','descriptive_counts_checked','no_predictive_fits_checked')
DELTA_FLAGS = ('passed','independent_delta_rebuild','effective_support_weight_checked',
    'full_auxiliary_replacement_checked','first_partial_unchanged','constant_rows_unchanged',
    'core_weights_checked','no_eligibility_delta_filter','source_values_checked',
    'asset_type_identity_checked','empty_group_copy_checked','constant_subgroups_retained_checked')
SELECTION_FLAGS = ('passed','independent_original_pre_checked','original_support_checked',
    'original_budget_checked','full_auxiliary_delta_checked','additive_pre_correction_checked',
    'original_effects_unchanged','one_episode_per_arm','initial_partial_unchanged',
    'exact_scores_and_choices_checked')
MAX_DELTA = .5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('asset_mixture234_frozen_accounting', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | set(frozen_source_oracle().source_files(ROOT)) | {Path(__file__).resolve(), SOURCE_HELPER}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite cached-transform arithmetic')
    return value


def _numeric(raw, name, finite_required=True):
    require(all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))
                for x in raw.flat), name + ': real nonboolean numeric values required')
    try:
        result = raw.astype(np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ': binary64 conversion failed') from exc
    if finite_required:
        require(np.isfinite(result).all(), name + ': finite values required')
    return result


def _bool(raw, name):
    require(all(isinstance(x, (bool, np.bool_)) for x in raw.flat),
            name + ': exact boolean values required')
    return raw.astype(bool, copy=True)


def source_prefix(arrays, limit=None):
    require(isinstance(arrays, dict), 'Original arrays dictionary required')
    keys = ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow',
            'extension_available', 'forecast_available', 'eligible', 'rank_log_excess')
    require(all(key in arrays for key in keys), 'Missing original cached input')
    source = {key: np.asarray(arrays[key]) for key in keys}
    shape = source['baseline'].shape
    require(len(shape) == 2 and shape[1] >= 1, 'Nonempty candidate width required')
    full, width = shape
    require(limit is None or type(limit) is int and 0 <= limit <= full, 'Invalid prefix')
    n = full if limit is None else limit
    require(source['core'].shape == shape, 'Core shape differs')
    require(source['core_weight'].shape in ((full,), (full, 1), shape), 'Core weight shape differs')
    for key in ('state_calm10', 'state_weak_narrow'):
        require(source[key].shape == (full,), 'State shape differs: ' + key)
    for key in ('extension_available', 'forecast_available', 'eligible', 'rank_log_excess'):
        require(source[key].shape == shape, 'Matrix shape differs: ' + key)
    # Shapes may be inspected globally; values are consumed only after slicing.
    out = {}
    for key in ('baseline', 'core', 'rank_log_excess'):
        out[key] = _numeric(source[key][:n], key, key != 'rank_log_excess')
    for key in ('state_calm10', 'state_weak_narrow', 'extension_available',
                'forecast_available', 'eligible'):
        out[key] = _bool(source[key][:n], key)
    cw = _numeric(source['core_weight'][:n], 'core_weight')
    if cw.ndim == 1:
        cw = cw[:, None]
    out['core_weight'] = np.broadcast_to(cw, (n, width)).copy()
    for i in range(n):
        required = .50 if out['state_calm10'][i] else .52
        require(all(float(x) == required for x in out['core_weight'][i]), 'Core weight/state mismatch')
    return out, n, width


def metadata_prefix(meta, full, n, width):
    """Only explicit global ETF identities and consumed symbol rows are read."""
    require(isinstance(meta, dict), 'Metadata dictionary required')
    require(isinstance(meta.get('symbols'), list) and len(meta['symbols']) == full,
            'Full symbol-row axis required')
    etfs = meta.get('etfs')
    require(isinstance(etfs, list) and all(isinstance(x, str) and bool(x) for x in etfs)
            and len(set(etfs)) == len(etfs), 'Explicit unique ETF identities required')
    rows = []
    for row in meta['symbols'][:n]:
        require(isinstance(row, list) and len(row) == width and
                all(isinstance(x, str) and bool(x) for x in row) and len(set(row)) == width,
                'Consumed symbol row must contain unique nonempty strings')
        rows.append(row.copy())
    return rows, set(etfs)


def row_targets(support, stock_indices, etf_indices):
    """Every target evaluates both empirical CDFs at the SAME support value."""
    s=[float(x) for x in support]; width=len(s)
    require(width>0 and all(math.isfinite(x) for x in s),'Finite nonempty support row')
    require(sorted(stock_indices+etf_indices)==list(range(width)) and
        not set(stock_indices)&set(etf_indices),'Exact type partition required')
    ns,ne=len(stock_indices),len(etf_indices); constant=len(set(s))==1
    counts={k:[] for k in ('stock_less','stock_equal','etf_less','etf_equal','pooled_less','pooled_equal')}
    if constant:return [0.]*width,[0.]*width,counts
    candidate=[];control=[]
    for x in s:
        sl=sum(s[j]<x for j in stock_indices);se=sum(s[j]==x for j in stock_indices)
        el=sum(s[j]<x for j in etf_indices);ee=sum(s[j]==x for j in etf_indices)
        # Direct all-member counts are rebuilt independently of the two groups.
        pl=sum(y<x for y in s);pe=sum(y==x for y in s)
        for key,value in zip(counts,(sl,se,el,ee,pl,pe)):counts[key].append(value)
        pooled=(pl+.5*pe)/width-.5
        if ns and ne:
            fs=(sl+.5*se)/ns;fe=(el+.5*ee)/ne
            q=.5*fs+.5*fe-.5
        else:q=pooled
        candidate.append(finite(q));control.append(finite(pooled))
    return candidate,control,counts


def independent_features(arrays,meta,limit=None):
    source,n,width=source_prefix(arrays,limit)
    names,etfs=metadata_prefix(meta,np.asarray(arrays['baseline']).shape[0],n,width)
    values={key:np.zeros((n,width),dtype=np.float64) for key in VALUE_KEYS};events=[]
    totals={key:0 for key in COUNT_FIELDS}
    totals.update(events=n,symbols_per_event=width,checked_cells=n*width)
    for i in range(n):
        active_flags=[]
        for j in range(width):
            cw=float(source['core_weight'][i,j]);calm=bool(source['state_calm10'][i]);weak=bool(source['state_weak_narrow'][i])
            s=finite(finite(float(source['baseline'][i,j])-float(source['core'][i,j]))/(1.-cw))
            require(abs(s)<=.5+1e-12,'Original support exceeds validation bound')
            active=bool(calm and not weak and source['extension_available'][i,j] and
                source['forecast_available'][i,j] and source['eligible'][i,j] and
                math.isfinite(float(source['rank_log_excess'][i,j])))
            w=finite((1.-cw)-(.025*calm+.05*weak+.01*active))
            require(w>0.,'Effective original support weight must be positive')
            values['support'][i,j]=s;values['weight'][i,j]=w;active_flags.append(active)
        stocks=[j for j,s in enumerate(names[i]) if s not in etfs]
        funds=[j for j,s in enumerate(names[i]) if s in etfs]
        s=values['support'][i].tolist();q,p,counts=row_targets(s,stocks,funds)
        constant=len(set(s))==1;single=not stocks or not funds
        values['candidate'][i]=q;values['control'][i]=p
        if i and not constant:
            for j in range(width):
                for key in ('candidate','control'):
                    d=finite(float(values['weight'][i,j])*finite(float(values[key][i,j])-s[j]))
                    require(abs(d)<=MAX_DELTA,'Unbounded full auxiliary correction')
                    values[key+'_delta'][i,j]=d
        maximum=lambda key:float(max(map(abs,values[key][i]),default=0.))
        difference=float(max((abs(float(x)-float(y)) for x,y in zip(q,p)),default=0.))
        events.append(dict(query_index=i,symbols=names[i].copy(),stock_indices=stocks,etf_indices=funds,
            stock_count=len(stocks),etf_count=len(funds),distinct_levels=len(set(s)),constant=constant,
            single_group=bool(single),branch='constant_row' if constant else ('single_group_copy' if single else 'equal_mixture'),
            active_forecast_count=sum(active_flags),**counts,
            max_absolute_candidate_delta=maximum('candidate_delta'),
            max_absolute_control_delta=maximum('control_delta'),
            max_absolute_candidate_control_difference=difference))
        totals['stock_cells']+=len(stocks);totals['etf_cells']+=len(funds)
        totals['both_group_rows']+=int(not single);totals['single_group_rows']+=int(bool(single))
        totals['constant_rows']+=int(constant);totals['active_forecast_cells']+=sum(active_flags)
        if not constant:
            totals['pooled_cdf_evaluations']+=width
            if not single:totals['group_cdf_evaluations']+=2*width
    audit=dict(schema='asset_mixture234_features_v1',**totals,ranking_eligibility_filtered=False,
        new_availability_filter=False,portfolio_returns_calculated=False,event_history=events)
    return values,audit


def exact_array(actual,expected,label):
    a=np.asarray(actual);e=np.asarray(expected)
    require(a.dtype==e.dtype and a.shape==e.shape and a.tobytes()==e.tobytes(),label+': exact array bytes')


def values_check(values,expected):
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Six exact feature keys required')
    for key in VALUE_KEYS:exact_array(values[key],expected[key],'Independent '+key)


def verify_features(arrays,meta,values,audit,limit=None):
    expected,ea=independent_features(arrays,meta,limit);values_check(values,expected);record_check(audit,ea)
    counts={key:ea[key] for key in COUNT_FIELDS}
    return dict(**{key:True for key in FEATURE_FLAGS},**counts,counts=counts,
        computes_returns=False,producer_modules_imported=False,numerical_tolerance=0.,support_bound_tolerance=1e-12)


def verify_deltas(values,arrays,meta,deltas,limit=None):
    expected,_=independent_features(arrays,meta,limit);values_check(values,expected)
    require(isinstance(deltas,dict) and set(deltas)==set(DELTA_KEYS),'Two delta arms required')
    changed={};maxima={}
    for arm,key in DELTA_KEYS.items():
        exact_array(deltas[arm],expected[key],'Independent complete auxiliary delta '+arm)
        changed[arm]=int(np.count_nonzero(expected[key]));maxima[arm]=float(np.max(np.abs(expected[key]))) if expected[key].size else 0.
    return dict(**{key:True for key in DELTA_FLAGS},changed_cells=changed,checked_events=expected['support'].shape[0],
        max_absolute_delta=MAX_DELTA,observed_max_absolute_delta=max(maxima.values()),per_arm_max_absolute_delta=maxima,
        numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False)


def frozen_source_oracle():
    require(sha(SOURCE_HELPER)==SOURCE_SHA,'Frozen independent original source changed')
    spec=importlib.util.spec_from_file_location('mixture234_independent_original',SOURCE_HELPER)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def selection_oracle(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    # Features may be full-length; only this prefix is inspected, including source values.
    expected,_=independent_features(original,meta,limit=n)
    require(isinstance(values,dict) and set(values)==set(VALUE_KEYS),'Six exact feature keys')
    for key in VALUE_KEYS:exact_array(np.asarray(values[key])[:n],expected[key],'Consumed selection feature '+key)
    width=np.asarray(original['baseline']).shape[1]
    if n==0:return {arm:np.empty((0,width)) for arm in ARMS},{arm:[] for arm in ARMS}
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    require(x['local'].shape==(n,width),'Original source width')
    # Frozen NumPy association is part of the original episode contract.
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={}
    for arm in ARMS:
        pre=x['local'].copy()
        if arm in DELTA_KEYS:
            ds=expected[DELTA_KEYS[arm]]
            for i in range(n):
                for j in range(width):
                    if ds[i,j]!=0.:pre[i,j]+=float(ds[i,j])
        score=pre.copy();rows=[]
        for i,names in enumerate(x['names']):
            order=sorted(np.flatnonzero(x['eligible'][i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3:rows.append(order if len(order)==2 else []);continue
            js=[j for j in order[2:5] if float(pre[i,order[2]])-float(pre[i,j])<=.02]
            for j in js:
                target=min(.5,max(-.5,x['support'][i,j]+x['effects'][i,j]))
                score[i,j]+=budget[i]*(target-x['support'][i,j])
            third=min(js,key=lambda j:(-score[i,j],names[j]))
            rows.append(sorted(order[:2]+[third],key=lambda j:(-score[i,j],names[j])))
        scores[arm]=score;choices[arm]=rows
    return scores,choices


def verify_selection(original,meta,values,orders,*,scores=None,n=None):
    n=len(meta['dates']) if n is None else n
    expected,choices=selection_oracle(original,meta,values,n)
    require(set(orders)==set(ARMS),'Three order arms')
    wanted={arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(expected[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders,wanted,'Independent ordered selections')
    if scores is not None:
        require(set(scores)==set(ARMS),'Three score arms')
        for arm in ARMS:exact_array(scores[arm],expected[arm],'Full selection scores '+arm)
    return dict(**{k:True for k in SELECTION_FLAGS},ordered_choices_checked=3*n,
        full_scores_supplied=scores is not None,numerical_tolerance=0.,
        original_source_parity_tolerance=1e-12,maximum_absolute_delta=MAX_DELTA,
        producer_modules_imported=False,computes_returns=False)
