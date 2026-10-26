"""Independent source, conditional-payoff and account verification for round189.

No production feature/strategy imports or real-data entry point. Mapping is
reconstructed from independently stated constants, verified against frozen AST.
The caller-provided original mapping is cross-checked on a query-date prefix.
"""
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction
import ast
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
MAPPING_SOURCE = ROOT / 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTOR = {'Basic Materials': 'XLB', 'Communication Services': 'XLC', 'Energy': 'XLE',
          'Financial Services': 'XLF', 'Industrials': 'XLI', 'Technology': 'XLK',
          'Consumer Defensive': 'XLP', 'Real Estate': 'XLRE', 'Utilities': 'XLU',
          'Healthcare': 'XLV', 'Consumer Cyclical': 'XLY'}
FOREIGN = {'ASML': 'XLK', 'ARM': 'XLK', 'TSM': 'XLK', 'BABA': 'XLY', 'NVO': 'XLV',
           'SAP': 'XLK', 'TM': 'XLY', 'SHEL': 'XLE', 'SKHY': 'XLK', 'BHP': 'XLB',
           'HSBC': 'XLF', 'SMFG': 'XLF', 'NVS': 'XLV', 'SNDK': 'XLK'}
ARMS = ('incumbent_local67', 'sector_trend_conditional_payoff', 'sector_unconditional_payoff')
RAW = dict(sector_trend_conditional_payoff='candidate', sector_unconditional_payoff='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE, SCALE = .02, .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING, MAPPING_SOURCE)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('sector189_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('sector189_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths']==9 and result['checked_slices']==45 and set(result['paths'])==set(ARMS), 'Incomplete independent account coverage')
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': mapping')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (got, wanted) in enumerate(zip(actual, expected)):
            record_check(got, wanted, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    day = date.fromisoformat(value)
    require(day.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(day)


def mapping_constants():
    require(sha(MAPPING_SOURCE) == MAPPING_SHA, 'Frozen benchmark mapping source changed')
    found = {}
    for node in ast.parse(MAPPING_SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('SECTOR', 'FOREIGN'):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    require(found == {'SECTOR': SECTOR, 'FOREIGN': FOREIGN}, 'Frozen mapping constants differ')


def benchmark_for(day, symbol, sectors, etfs):
    if symbol in etfs:
        return None
    if symbol in FOREIGN:
        return FOREIGN[symbol]
    if day not in sectors.index or symbol not in sectors.columns:
        return None
    return SECTOR.get(sectors.at[day, symbol])


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'sectors'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        require(not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing,
                'Invalid panel time axis')
        require(frame.index.tz is None and frame.index.equals(frame.index.normalize()), 'Daily naive dates required')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s for s in frame.columns),
                'Panel identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF metadata')
    sessions = [iso(value) for value in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], 0
    for raw, names in zip(meta['dates'], meta['symbols']):
        day = iso(raw)
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        require(isinstance(names, (list, tuple)) and names and len(names) == len(set(names)), 'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in names), 'Invalid event name')
        require(not queries or width == len(names), 'Changing event width')
        width = len(names); queries.append((day, list(names), lookup[day]))
    return sessions, queries, width



def empty(reason):
    return dict(available=False,reason=reason,candidate=None,control=None,state_start_indices=[],predictor_indices=[],
        label_end_indices=[],states=[],labels=[],query_start_index=None,query_end_index=None,query_state=None,
        matching_indices=[],matching_labels=[],matching_count=0,unconditional_mean=None,matching_sum=None,
        matching_mean=None,shrunken_payoff=None,all_states_match=False,
        conditional_mean_estimates=0,pooled_mean_estimates=0,training_label_observations=0,matched_label_observations=0)


def finite_sum(values):
    try:value=math.fsum(values)
    except (OverflowError,ValueError) as exc:raise FloatingPointError('Nonfinite conditional sum') from exc
    if not math.isfinite(value):raise FloatingPointError('Nonfinite conditional sum')
    return value


def relative_state(stock_end,stock_start,sector_end,sector_start):
    """Exact rationals of original binary floats; no rounded products or logs."""
    left=Fraction.from_float(float(stock_end))/Fraction.from_float(float(stock_start))
    right=Fraction.from_float(float(sector_end))/Fraction.from_float(float(sector_start))
    return int(left>right)-int(left<right)


def measure_oracle(stock,benchmark):
    require(np.shape(stock)==np.shape(benchmark)==(1471,),'Exactly 1471 prices for each series required')
    vectors=[]
    for raw,name in ((stock,'stock'),(benchmark,'benchmark')):
        try:v=[float(x) for x in raw]
        except (ValueError,TypeError,OverflowError):return empty('nonnumeric_'+name+'_price')
        if any(not math.isfinite(x) or x<=0 for x in v):return empty('invalid_'+name+'_price')
        vectors.append(v)
    stock,benchmark=vectors
    starts=list(range(0,1240,21));predictors=[x+210 for x in starts];ends=[x+21 for x in predictors]
    states=[relative_state(stock[a],stock[a-210],benchmark[a],benchmark[a-210]) for a in predictors]
    query=relative_state(stock[1470],stock[1260],benchmark[1470],benchmark[1260])
    try:
        with localcontext() as context:
            context.prec=70
            indices=sorted(set(predictors+ends))
            sl={i:Decimal.from_float(stock[i]).ln() for i in indices}
            bl={i:Decimal.from_float(benchmark[i]).ln() for i in indices}
            labels=[float((sl[b]-sl[a])-(bl[b]-bl[a])) for a,b in zip(predictors,ends)]
    except (ArithmeticError,ValueError,OverflowError) as exc:raise FloatingPointError('Nonfinite logarithmic label') from exc
    if any(not math.isfinite(x) for x in labels):raise FloatingPointError('Nonfinite excess label')
    matched=[i for i,x in enumerate(states) if x==query];n=len(matched)
    selected=[labels[i] for i in matched];mu=finite_sum(labels)/60.;total=finite_sum(selected)
    out=empty('insufficient_matches')
    out.update(state_start_indices=starts,predictor_indices=predictors,label_end_indices=ends,states=states,labels=labels,
        query_start_index=1260,query_end_index=1470,query_state=query,matching_indices=matched,matching_labels=selected,
        matching_count=n,unconditional_mean=mu,matching_sum=total,matching_mean=total/n if n else None,all_states_match=n==60)
    if n>=10:
        candidate=mu if n==60 else finite_sum([total,20.*mu])/(n+20)
        if not math.isfinite(candidate):raise FloatingPointError('Nonfinite shrunken payoff')
        out.update(available=True,reason='available',candidate=candidate,control=mu,shrunken_payoff=candidate,
            conditional_mean_estimates=1,pooled_mean_estimates=1,training_label_observations=60,matched_label_observations=n)
    return out


def independent_features(panel,meta,calendar,benchmark_map,limit=None):
    mapping_constants();sessions,queries,width=schema(panel,meta,calendar)
    require(callable(benchmark_map),'Original benchmark_map callable required')
    require(limit is None or type(limit) is int and 0<=limit<=len(queries),'Invalid prefix limit')
    queries=queries if limit is None else queries[:limit]
    values=dict(candidate=np.full((len(queries),width),np.nan),control=np.full((len(queries),width),np.nan),available=np.zeros((len(queries),width),bool))
    history=[];counts={};totals=dict(conditional_mean_estimates=0,pooled_mean_estimates=0,training_label_observations=0,matched_label_observations=0)
    for i,(day,names,t) in enumerate(queries):
        mapping={s:benchmark_for(day,s,panel['sectors'],meta['etfs']) for s in names}
        got=benchmark_map(str(day.date()),names,panel['sectors'].loc[:day].copy(deep=True),meta['etfs'])
        require(isinstance(got,dict) and got==mapping,'Original own-sector mapping disagrees with independent source')
        days=sessions[t-1512:t-41] if t>=1512 else [];dates=[str(x.date()) for x in days]
        windows=None if not days else dict(formation=dict(price_start=dates[0],price_end=dates[1470],return_count=1470),
            query_state=dict(price_start=dates[1260],price_end=dates[1470],return_count=210),
            training_labels=dict(price_start=dates[210],price_end=dates[1470],return_count=1260,block_sessions=21,block_count=60))
        missing=[str(d.date()) for d in days if d not in panel['adj'].index];cells=[]
        for j,symbol in enumerate(names):
            benchmark=mapping[symbol]
            reason=('etf_neutral' if symbol in meta['etfs'] else 'missing_benchmark' if benchmark is None else
                'insufficient_history' if not days else 'missing_symbol' if symbol not in panel['adj'].columns else
                'missing_benchmark_symbol' if benchmark not in panel['adj'].columns else 'missing_official_session' if missing else None)
            cell=empty(reason) if reason else measure_oracle([panel['adj'].at[d,symbol] for d in days],[panel['adj'].at[d,benchmark] for d in days])
            cell.update(symbol=symbol,benchmark=benchmark,date=str(day.date()),windows=windows,
                state_start_dates=[dates[k] for k in cell['state_start_indices']],predictor_dates=[dates[k] for k in cell['predictor_indices']],
                label_end_dates=[dates[k] for k in cell['label_end_indices']],
                matching_predictor_dates=[dates[cell['predictor_indices'][k]] for k in cell['matching_indices']],
                matching_label_end_dates=[dates[cell['label_end_indices'][k]] for k in cell['matching_indices']])
            values['available'][i,j]=cell['available']
            for key in ('candidate','control'):values[key][i,j]=cell[key]
            counts[cell['reason']]=counts.get(cell['reason'],0)+1
            for key in totals:totals[key]+=cell[key]
            cells.append(cell)
        history.append(dict(date=str(day.date()),symbols=names,session_position=t,price_start=dates[0] if dates else None,
            price_end=dates[-1] if dates else None,windows=windows,missing_official_sessions=missing,benchmarks=mapping,cells=cells))
    audit=dict(events=len(queries),symbols_per_event=width,checked_cells=int(values['available'].size),price_count=1471,
        skip_sessions=42,return_count=1470,state_return_count=210,label_sessions=21,training_label_count=60,prior_mass=20,minimum_matches=10,
        available_cells=int(values['available'].sum()),unavailable_cells=int((~values['available']).sum()),reason_counts=counts,
        **totals,predictive_model_fits=0,common_availability=True,portfolio_returns_calculated=False,ranking_performed=False,
        trading_eligibility_modified=False,event_history=history)
    return values,audit


def arrays(values):
    require(isinstance(values,dict) and set(values)=={'candidate','control','available'},'Value keys differ')
    a={k:np.asarray(v) for k,v in values.items()};mask=a['available']
    require(mask.ndim==2 and mask.dtype==bool,'Boolean availability matrix required')
    for key in ('candidate','control'):
        require(a[key].shape==mask.shape and np.issubdtype(a[key].dtype,np.floating),'Raw shape/dtype differs')
        require(not np.isinf(a[key]).any() and np.array_equal(np.isfinite(a[key]),mask),'Common raw availability differs')
    return a


def verify_features(panel,meta,calendar,benchmark_map,values,audit):
    actual=arrays(values);wanted,expected=independent_features(panel,meta,calendar,benchmark_map)
    require(np.array_equal(actual['available'],wanted['available']),'Independent common support differs')
    for key in ('candidate','control'):close(actual[key],wanted[key],'Independent conditional payoff '+key)
    record_check(audit,expected)
    for i,row in enumerate(expected['event_history']):
        for j,cell in enumerate(row['cells']):
            if cell['all_states_match']:
                require(actual['candidate'][i,j]==actual['control'][i,j],'All60 identity must be bit-for-bit')
    counts=('events','symbols_per_event','checked_cells','available_cells','unavailable_cells',
        'conditional_mean_estimates','pooled_mean_estimates','training_label_observations','matched_label_observations')
    return dict(passed=True,**{k:expected[k] for k in counts},independent_scalar_rebuild=True,own_sector_mapping_checked=True,
        exact_relative_state_signs_checked=True,matured_labels_checked=True,query_state_unlabelled_checked=True,
        minimum_10_matches_checked=True,all60_identity_checked=True,full_1471_price_support_checked=True,
        official_session_windows_checked=True,exact_common_availability=True,audit_checked=True,no_future_or_outcome_inputs=True,
        mapping_source_sha256=MAPPING_SHA,predictive_model_fits=0,computes_returns=False,portfolio_returns_calculated=False)


def independent_deltas(values,dose=DOSE):
    require(type(dose) in (int,float) and dose==DOSE,'Fixed dose .02 required')
    a=arrays(values);result={arm:np.zeros(a['available'].shape) for arm in RAW}
    for arm,key in RAW.items():
        for i,j in zip(*np.nonzero(a['available'])):
            scaled=float(a[key][i,j])/SCALE
            if not math.isfinite(scaled):raise FloatingPointError('Nonfinite tanh scaling')
            if i:result[arm][i,j]=dose*math.tanh(scaled)
    return result


def verify_deltas(values,deltas,dose=DOSE):
    require(isinstance(deltas,dict) and set(deltas)==set(RAW),'Delta arms differ')
    expected=independent_deltas(values,dose);a=arrays(values)
    for arm,target in expected.items():
        close(deltas[arm],target,'Fixed tanh delta '+arm,atol=2e-17)
        x=np.asarray(deltas[arm]);require(np.isfinite(x).all() and np.all(np.abs(x)<=DOSE),'Unbounded delta')
        neutral=~a['available'].copy()
        if len(neutral):neutral[0]=True
        require(np.array_equal(x[neutral],np.zeros(neutral.sum())),'Neutral delta must be exact zero')
    return dict(passed=True,checked_events=a['available'].shape[0],dose=DOSE,scale=SCALE,max_absolute_delta=DOSE,
        full_candidate_universe=True,common_available_only=True,common_availability=True,initial_delta_zero=True,
        no_eligibility_or_model_filter=True,no_cross_sectional_ranking=True,fixed_tanh_mapping_checked=True,
        direct_additive_pre_score=True,changed_cells={arm:int(np.count_nonzero(x)) for arm,x in expected.items()})
