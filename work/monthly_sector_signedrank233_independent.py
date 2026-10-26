"""Independent exact-rational own-sector signed magnitude ranks and frozen accounts.

Feature calculations are self-contained and import no production strategy or
kernel. Only the pinned round189 mapping and independent accounts are reused.
No real-data command-line entry point, fitting, portfolio replay or writes.
"""
from datetime import date
from fractions import Fraction
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_transition189_independent.py'
FROZEN_HELPER_SHA = 'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
ARMS = ('incumbent_local67', 'sector_signed_magnitude', 'sector_sign_control')
RAW = dict(sector_signed_magnitude='candidate', sector_sign_control='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE, MAX_DELTA = .025, .01875

VALUE_KEYS=('candidate','control','available')
COUNT_FIELDS=('events','symbols_per_event','checked_cells','available_cells','unavailable_cells',
    'source_price_observations','relative_sign_observations','magnitude_rank_observations',
    'magnitude_pair_comparisons','signed_rank_estimates','sign_balance_estimates',
    'constant_magnitude_copies','predictive_model_fits')
FEATURE_FLAGS=('passed','independent_scalar_rebuild','own_sector_mapping_checked',
    'exact_rational_relative_ratios_checked','exact_rational_relative_signs_checked',
    'twelve_nonoverlapping_blocks_checked','four_neutral_pseudoblocks_checked',
    'full_253_price_support_checked','nonboolean_positive_real_sources_checked',
    'official_session_windows_checked','monthend_queries_checked','exact_common_availability',
    'zero_blocks_included_checked','doubled_average_magnitude_ranks_checked',
    'doubled_rank_sum_checked','fixed_denominators_checked','constant_magnitude_exact_copy_checked',
    'control_signed_balance_checked','no_cross_sectional_ranking','audit_checked',
    'descriptive_counts_checked','no_predictive_fits_checked','no_future_or_outcome_inputs')
MODEL_FLAGS=FEATURE_FLAGS
MODEL_COUNT_FIELDS=COUNT_FIELDS
DELTA_FLAGS=('passed','full_candidate_universe','common_available_only','common_availability',
    'initial_delta_zero','no_eligibility_or_model_filter','no_cross_sectional_ranking',
    'fixed_linear_mapping_checked','direct_additive_pre_score')
SELECTION_FLAGS=('passed','independent_original_pre_checked','original_support_checked',
    'original_budget_checked','fixed_raw_delta_checked','additive_pre_correction_checked',
    'original_effects_unchanged','one_episode_per_arm','initial_partial_unchanged',
    'exact_scores_and_choices_checked')
SOURCE_HELPER=ROOT/'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA='0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_helper():
    require(sha(FROZEN_HELPER) == FROZEN_HELPER_SHA, 'Frozen189 independent helper changed')
    spec = importlib.util.spec_from_file_location('signedrank233_frozen_mapping_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.MAPPING_SHA == MAPPING_SHA, 'Unexpected frozen mapping source')
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | set(frozen_source_oracle().source_files(ROOT)) | {Path(__file__).resolve(),SOURCE_HELPER}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['mapping_and_outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


def benchmark_for(day, symbol, sectors, etfs):
    """Expose the pinned independently stated original mapping for synthetics."""
    helper = frozen_helper()
    helper.mapping_constants()
    return helper.benchmark_for(day, symbol, sectors, etfs)


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': exact keys')
        for key, value in expected.items():
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    day = date.fromisoformat(value)
    require(day.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(day)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'sectors'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        idx = frame.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None
                and idx.equals(idx.normalize()), 'Invalid daily panel axis')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
                for s in frame.columns), 'Invalid panel identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF metadata')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(x.weekday() < 5 for x in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for raw, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(raw)
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        t = lookup[day]
        if queries or raw != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event name')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


def empty(reason):
    return dict(available=False, reason=reason, candidate=None, control=None,
        endpoint_indices=[], block_start_indices=[], block_end_indices=[], relative_signs=[],
        exact_relative_z=[], twice_magnitude_ranks=[], relative_positive_count=0,
        relative_negative_count=0, relative_tie_count=0, relative_signed_count=0,
        candidate_numerator=0, twice_rank_sum=0, constant_magnitude_copy=False,
        relative_sign_observations=0, magnitude_rank_observations=0, magnitude_pair_comparisons=0)


def relative_z(stock_end, stock_start, sector_end, sector_start):
    raw=(stock_end,stock_start,sector_end,sector_start)
    require(all(isinstance(v,numbers.Real) and not isinstance(v,(bool,np.bool_)) for v in raw),
        'Nonboolean real ratio endpoints required')
    values=[float(v) for v in raw]
    require(all(math.isfinite(v) and v>0 for v in values),'Positive finite ratio endpoints required')
    a,b,c,d=[Fraction.from_float(v) for v in values]
    own,sector=a/b,c/d
    return (own-sector)/(own+sector)


def measure_oracle(stock, benchmark):
    require(np.shape(stock)==np.shape(benchmark)==(253,),'Exactly253 prices per series required')
    vectors=[]
    for raw,name in ((stock,'stock'),(benchmark,'benchmark')):
        vector=[]
        for value in raw:
            if not isinstance(value,numbers.Real) or isinstance(value,(bool,np.bool_)):
                return empty('nonnumeric_'+name+'_price')
            try: v=float(value)
            except (ValueError,TypeError,OverflowError): return empty('invalid_'+name+'_price')
            if not math.isfinite(v) or v<=0.: return empty('invalid_'+name+'_price')
            vector.append(v)
        vectors.append(vector)
    own,sector=vectors;points=list(range(0,253,21));starts,ends=points[:-1],points[1:]
    z=[relative_z(own[b],own[a],sector[b],sector[a]) for a,b in zip(starts,ends)]
    signs=[int(v>0)-int(v<0) for v in z];magnitudes=[abs(v) for v in z]
    # Independent pair counting; integer doubled ranks preserve every exact tie.
    rank2=[2]*12
    for i in range(12):
        for j in range(i+1,12):
            if magnitudes[i]<magnitudes[j]: rank2[j]+=2
            elif magnitudes[i]>magnitudes[j]: rank2[i]+=2
            else: rank2[i]+=1;rank2[j]+=1
    require(sum(rank2)==156,'Doubled rank mass')
    numerator=sum(a*b for a,b in zip(signs,rank2));control=float(Fraction(sum(signs),16))
    constant=all(v==magnitudes[0] for v in magnitudes)
    candidate=control if constant else float(Fraction(numerator,208))
    out=empty('available')
    out.update(available=True,candidate=candidate,control=control,endpoint_indices=points,
        block_start_indices=starts,block_end_indices=ends,relative_signs=signs,
        exact_relative_z=[str(v) for v in z],twice_magnitude_ranks=rank2,
        relative_positive_count=signs.count(1),relative_negative_count=signs.count(-1),
        relative_tie_count=signs.count(0),relative_signed_count=sum(signs),
        candidate_numerator=numerator,twice_rank_sum=156,constant_magnitude_copy=constant,
        relative_sign_observations=12,magnitude_rank_observations=12,magnitude_pair_comparisons=66)
    return out


def independent_features(panel, meta, calendar, benchmark_map, limit=None):
    helper = frozen_helper()
    helper.mapping_constants()
    sessions, queries, width = schema(panel, meta, calendar)
    require(callable(benchmark_map), 'Original benchmark_map callable required')
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan), control=np.full((len(queries), width), np.nan),
                  available=np.zeros((len(queries), width), bool))
    history, reasons = [], {}
    for i, (day, names, t) in enumerate(queries):
        mapping = {s: helper.benchmark_for(day, s, panel['sectors'], meta['etfs']) for s in names}
        supplied = benchmark_map(str(day.date()), names, panel['sectors'].loc[:day].copy(deep=True), meta['etfs'])
        require(isinstance(supplied, dict) and supplied == mapping, 'Original own-sector mapping differs')
        days = sessions[t - 294:t - 41] if t >= 294 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else dict(formation=dict(price_start=dates[0], price_end=dates[-1], return_count=252),
            blocks=dict(first_start_date=dates[0], last_end_date=dates[-1], block_sessions=21, block_count=12))
        missing = [str(d.date()) for d in days if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            benchmark = mapping[symbol]
            reason = ('etf_neutral' if symbol in meta['etfs'] else 'missing_benchmark' if benchmark is None else
                'insufficient_history' if not days else 'missing_symbol' if symbol not in panel['adj'].columns else
                'missing_benchmark_symbol' if benchmark not in panel['adj'].columns else
                'missing_official_session' if missing else None)
            cell = empty(reason) if reason else measure_oracle(
                [panel['adj'].at[d, symbol] for d in days], [panel['adj'].at[d, benchmark] for d in days])
            cell.update(symbol=symbol, benchmark=benchmark, date=str(day.date()), windows=windows,
                endpoint_dates=[dates[k] for k in cell['endpoint_indices']],
                block_start_dates=[dates[k] for k in cell['block_start_indices']],
                block_end_dates=[dates[k] for k in cell['block_end_indices']])
            values['available'][i, j] = cell['available']
            for key in ('candidate', 'control'):
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=dates[0] if days else None, price_end=dates[-1] if days else None,
            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    copies=sum(cell['constant_magnitude_copy'] for event in history for cell in event['cells'])
    audit = dict(schema='monthly_sector_signedrank233_features_v1',events=len(queries),
        symbols_per_event=width,checked_cells=int(values['available'].size),
        price_count=253,skip_sessions=42,return_count=252,block_count=12,block_sessions=21,
        neutral_pseudo_blocks=4,candidate_denominator=208,control_denominator=16,
        expected_twice_rank_sum=156,available_cells=available,
        unavailable_cells=int(values['available'].size)-available,reason_counts=reasons,
        source_price_observations=506*available,relative_sign_observations=12*available,
        magnitude_rank_observations=12*available,magnitude_pair_comparisons=66*available,
        signed_rank_estimates=available,sign_balance_estimates=available,constant_magnitude_copies=copies,
        predictive_model_fits=0,common_availability=True,portfolio_returns_calculated=False,
        cross_sectional_ranking_performed=False,within_cell_ranking_performed=True,
        trading_eligibility_modified=False,event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values,dict) and set(values)=={'candidate','control','available'},'Value keys differ')
    a={k:np.asarray(v) for k,v in values.items()};mask=a['available']
    require(mask.ndim==2 and mask.dtype==bool,'Boolean availability matrix required')
    for key in ('candidate','control'):
        v=a[key]
        require(v.shape==mask.shape and v.dtype==np.dtype('float64'),'Binary64 raw shape/dtype differs')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v),mask),'Common raw availability differs')
        require(np.all(np.abs(v[mask])<=.75),'Raw outside[-.75,.75]')
    return a


def exact_array(actual,expected,label):
    a,e=np.asarray(actual),np.asarray(expected)
    require(a.shape==e.shape and a.dtype==e.dtype,label+': shape/dtype')
    if e.dtype.kind=='f':
        require(np.array_equal(np.isnan(a),np.isnan(e)) and not np.isinf(a).any(),label+': mask')
        good=np.isfinite(e)
        require(a[good].tobytes()==e[good].tobytes(),label+': exact binary64')
    else: require(np.array_equal(a,e),label+': values')


def verify_features(panel,meta,calendar,benchmark_map,values,audit,limit=None):
    actual=arrays(values)
    wanted,expected=independent_features(panel,meta,calendar,benchmark_map,limit)
    for key in wanted: exact_array(actual[key],wanted[key],'Independent feature '+key)
    record_check(audit,expected)
    counts={k:expected[k] for k in COUNT_FIELDS}
    return dict(**{k:True for k in FEATURE_FLAGS},**counts,counts=counts,
        mapping_source_sha256=MAPPING_SHA,mapping_and_outcome_adapter_sha256=FROZEN_HELPER_SHA,
        numerical_tolerance=0.,producer_modules_imported=False,computes_returns=False,
        portfolio_returns_calculated=False)


def independent_deltas(values,dose=DOSE):
    require(type(dose) in (float,int) and dose==DOSE,'Fixed dose .025 required')
    a=arrays(values);out={arm:np.zeros(a['available'].shape) for arm in RAW}
    for arm,key in RAW.items():
        for i,j in zip(*np.nonzero(a['available'])):
            if i: out[arm][i,j]=dose*float(a[key][i,j])
    return out


def verify_deltas(values,deltas,dose=DOSE):
    require(isinstance(deltas,dict) and set(deltas)==set(RAW),'Delta arms differ')
    expected=independent_deltas(values,dose);a=arrays(values)
    for arm,v in expected.items():
        exact_array(deltas[arm],v,'Exact fixed linear delta '+arm)
        require(np.all(np.abs(v)<=DOSE*.75),'Unbounded delta')
    return dict(**{k:True for k in DELTA_FLAGS},checked_events=a['available'].shape[0],
        dose=DOSE,max_absolute_delta=MAX_DELTA,numerical_tolerance=0.,
        changed_cells={arm:int(np.count_nonzero(v)) for arm,v in expected.items()},
        producer_modules_imported=False,computes_returns=False)


def frozen_source_oracle():
    require(sha(SOURCE_HELPER)==SOURCE_SHA,'Frozen independent original source changed')
    spec=importlib.util.spec_from_file_location('signedrank233_independent_original',SOURCE_HELPER)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def selection_oracle(original,meta,values,n=None):
    n=len(meta['dates']) if n is None else n
    require(type(n) is int and 0<=n<=len(meta['dates']),'Selection prefix')
    a=arrays(values);require(a['available'].shape[0]>=n,'Feature prefix short')
    if n==0:
        width=np.asarray(original['baseline']).shape[1]
        return {arm:np.empty((0,width)) for arm in ARMS},{arm:[] for arm in ARMS}
    x=frozen_source_oracle().old.original_helper().original_prefix(original,meta,n)
    require(a['available'].shape[1]==x['shape'][1],'Feature width')
    budget=np.minimum(.05,np.maximum(0.,.10-(.025*x['calm']+.05*x['weak'])))
    scores={};choices={}
    for arm in ARMS:
        pre=x['local'].copy()
        if arm in RAW:
            for i in range(n):
                if meta['dates'][i]=='2016-08-17': continue
                for j in range(pre.shape[1]):
                    if a['available'][i,j] and a[RAW[arm]][i,j]!=0.:
                        pre[i,j]+=DOSE*float(a[RAW[arm]][i,j])
        score=pre.copy();rows=[]
        for i,names in enumerate(x['names']):
            order=sorted(np.flatnonzero(x['eligible'][i]).tolist(),key=lambda j:(-pre[i,j],names[j]))
            if len(order)<3: rows.append(order if len(order)==2 else []);continue
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
        for arm in ARMS: exact_array(scores[arm],expected[arm],'Full selection scores '+arm)
    return dict(**{k:True for k in SELECTION_FLAGS},ordered_choices_checked=3*n,
        full_scores_supplied=scores is not None,numerical_tolerance=0.,
        original_source_parity_tolerance=1e-12,maximum_absolute_delta=MAX_DELTA,
        producer_modules_imported=False,computes_returns=False)
