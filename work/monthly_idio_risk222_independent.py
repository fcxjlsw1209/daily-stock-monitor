"""Independent cumulative-momentum/current63 projection-risk and half-core audit.

No producer calculation or real-data command is imported. Only the pinned,
frozen independent portfolio-accounting adapter is loaded for outcome audits.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'idiosyncratic_core', 'total_volatility_control')
RAW = {'idiosyncratic_core': 'candidate', 'total_volatility_control': 'control'}
BLEND = .5
COUNT_FIELDS = ('checked_cells', 'source_available_cells', 'available_cells', 'unavailable_cells',
    'formation_price_observations', 'risk_price_observations', 'valid_price_observations',
    'simple_returns', 'descriptive_beta_fits', 'normalized_momentum_scores', 'ranked_observations')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'skip42_cumulative_momentum_checked',
    'current63_risk_window_checked', 'full_source_windows_checked', 'stock_market_centering_checked',
    'exact_ols_projection_checked', 'residual_not_recentered_checked', 'denominator62_checked',
    'zero_risk_common_exclusion_checked', 'self_market_zero_residual_checked', 'common_rank_domain_checked',
    'original_pct_ranks_checked', 'official_session_windows_checked', 'monthend_queries_checked',
    'exact_common_availability', 'audit_checked', 'no_future_or_outcome_inputs',
    'descriptive_fit_counts_checked', 'no_predictive_fits_checked')
DELTA_FLAGS = ('passed', 'half_core_formula_checked', 'original_pct_ranks_preserved',
    'first_partial_unchanged', 'calm_rows_unchanged', 'missing_preserves_core',
    'core_weights_checked', 'no_second_rank', 'independent_delta_rebuild')
VALUE_KEYS = ('available', 'candidate', 'control', 'momentum', 'beta', 'idiosyncratic_risk',
    'total_risk', 'raw_candidate', 'raw_control')
SOURCE_KEYS = ('momentum', 'total_risk')
FIT_KEYS = ('beta', 'idiosyncratic_risk')
TARGET_KEYS = ('candidate', 'control', 'raw_candidate', 'raw_control')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('idio222_frozen_independent_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | {Path(__file__).resolve()}))


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
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Panel keys differ')
    frame = panel['adj']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Adjusted frame type')
    idx = frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
            idx.equals(idx.normalize()), 'Invalid adjusted daily axis')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
            for s in frame.columns), 'Invalid adjusted symbol labels')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    positions = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for value, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(value)
        require(day in positions and (not queries or day > queries[-1][0]), 'Event session/order')
        t = positions[day]
        if queries or value != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event symbol')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


MEASURE_KEYS = ('start_price', 'formation_end_price', 'stock_risk_start_price', 'stock_risk_end_price',
    'market_risk_start_price', 'market_risk_end_price', 'momentum', 'stock_mean', 'market_mean',
    'stock_centered_sum_squares', 'market_centered_sum_squares', 'cross_product_sum', 'beta',
    'residual_sum_squares', 'idiosyncratic_risk', 'total_risk', 'raw_candidate', 'raw_control')


def empty(reason):
    return dict(source_available=False, available=False, reason=reason,
                **{key: None for key in MEASURE_KEYS}, descriptive_beta_fit=0)


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from finite positive source prices')
    return value


def total(values):
    try:
        return finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite scalar sum') from exc


def source_prices(values, length, label):
    require(np.shape(values) == (length,), 'Wrong '+label+' source length')
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Real) for x in values):
        return None, 'nonnumeric_'+label+'_price'
    try:
        prices = [float(x) for x in values]
    except (TypeError, ValueError, OverflowError):
        return None, 'nonnumeric_'+label+'_price'
    if any(not math.isfinite(x) or x <= 0. for x in prices):
        return None, 'invalid_'+label+'_price'
    return prices, None


def square_sum(values):
    answer = total([finite(x*x) for x in values])
    if answer == 0. and any(x != 0. for x in values):
        raise FloatingPointError('Nonzero vector squares underflowed')
    return answer


def risk_from_sum(squares):
    variance = finite(squares / 62.)
    if squares > 0. and variance == 0.:
        raise FloatingPointError('Positive risk variance underflowed')
    answer = finite(math.sqrt(variance))
    if squares > 0. and answer == 0.:
        raise FloatingPointError('Positive risk underflowed')
    return answer


def measure_oracle(formation_prices, stock_risk_prices, market_risk_prices):
    """Scalar OLS projection: center63 returns, no recentering of projected residuals."""
    specifications = ((formation_prices,253,'formation'),
        (stock_risk_prices,64,'stock_risk'), (market_risk_prices,64,'market_risk'))
    for source, count, label in specifications:
        require(np.shape(source) == (count,), 'Wrong '+label+' source length')
    windows = []
    for source, count, label in specifications:
        p, reason = source_prices(source,count,label)
        if reason:
            return empty(reason)
        windows.append(p)
    formation, stock, market = windows
    try:
        m = finite(finite(formation[-1]/formation[0])-1.)
        stock_r = [finite(finite(b/a)-1.) for a,b in zip(stock,stock[1:])]
        market_r = [finite(finite(b/a)-1.) for a,b in zip(market,market[1:])]
        sm, mm = finite(total(stock_r)/63.), finite(total(market_r)/63.)
        y = [finite(value-sm) for value in stock_r]
        x = [finite(value-mm) for value in market_r]
        sy, sx = square_sum(y), square_sum(x)
        cross = total([finite(a*b) for a,b in zip(x,y)])
        total_risk = risk_from_sum(sy)
        risk_from_sum(sx)  # Validate nonzero market variance cannot underflow at the declared scale.
        out = empty('zero_market_variance' if sx == 0. else 'available')
        out.update(source_available=True,start_price=formation[0],formation_end_price=formation[-1],
            stock_risk_start_price=stock[0],stock_risk_end_price=stock[-1],
            market_risk_start_price=market[0],market_risk_end_price=market[-1],momentum=m,
            stock_mean=sm,market_mean=mm,stock_centered_sum_squares=sy,
            market_centered_sum_squares=sx,cross_product_sum=cross,total_risk=total_risk)
        if sx == 0.:
            return out
        beta = finite(cross/sx)
        residual = [finite(a-finite(beta*b)) for a,b in zip(y,x)]
        rss = square_sum(residual)
        idio = risk_from_sum(rss)
        out.update(beta=beta,residual_sum_squares=rss,idiosyncratic_risk=idio,descriptive_beta_fit=1)
        if total_risk == 0.:
            out['reason'] = 'zero_total_risk'
        elif idio == 0.:
            out['reason'] = 'zero_idiosyncratic_risk'
        else:
            out.update(available=True,raw_candidate=finite(m/idio),raw_control=finite(m/total_risk))
        return out
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise FloatingPointError('Invalid arithmetic from finite positive prices') from exc


def original_pct_ranks(raw, available):
    """Pair-counter average ascending percentile; never use producer or pandas.rank."""
    result = np.full(np.shape(raw), np.nan)
    for i in range(len(result)):
        members = [j for j, good in enumerate(available[i]) if good]
        n = len(members)
        for j in members:
            less = sum(raw[i, k] < raw[i, j] for k in members)
            equal = sum(raw[i, k] == raw[i, j] for k in members)
            result[i, j] = (less + (equal + 1) / 2.) / n - .5
    return result


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel,meta,calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    shape = (len(queries),width)
    values = {key: np.full(shape,np.nan) for key in VALUE_KEYS if key != 'available'}
    values['available'] = np.zeros(shape,bool)
    frame = panel['adj']
    history, reasons, source_counts, available_counts = [], {}, [], []
    beta_fits = 0
    for i,(day,names,t) in enumerate(queries):
        formation = sessions[t-294:t-41] if t >= 294 else []
        risk = sessions[t-63:t+1] if t >= 63 else []
        mf = [str(d.date()) for d in formation if d not in frame.index]
        mr = [str(d.date()) for d in risk if d not in frame.index]
        cells = []
        for j,symbol in enumerate(names):
            reason = ('insufficient_history' if not formation or not risk else
                'missing_symbol' if symbol not in frame.columns else
                'missing_market_symbol' if 'SPY' not in frame.columns else
                'missing_formation_session' if mf else 'missing_risk_session' if mr else None)
            cell = empty(reason) if reason else measure_oracle(
                [frame.at[d,symbol] for d in formation], [frame.at[d,symbol] for d in risk],
                [frame.at[d,'SPY'] for d in risk])
            cell.update(symbol=symbol,date=str(day.date()),candidate=None,control=None)
            values['available'][i,j] = cell['available']
            for key in (*SOURCE_KEYS,*FIT_KEYS,'raw_candidate','raw_control'):
                if cell[key] is not None:
                    values[key][i,j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'],0)+1
            cells.append(cell)
        source = [x['source_available'] for x in cells]
        available = [x['available'] for x in cells]
        fits = sum(x['descriptive_beta_fit'] for x in cells)
        beta_fits += fits
        source_counts.append(sum(source));available_counts.append(sum(available))
        start = str(formation[0].date()) if formation else None
        end = str(formation[-1].date()) if formation else None
        history.append(dict(date=str(day.date()),symbols=names,session_position=t,
            price_start=start,price_end=end,formation_price_start=start,formation_price_end=end,
            risk_price_start=str(risk[0].date()) if risk else None,
            risk_price_end=str(risk[-1].date()) if risk else None,
            first_risk_return_date=str(risk[1].date()) if risk else None,
            missing_formation_sessions=mf,missing_risk_sessions=mr,
            source_available=source,available=available,source_available_count=sum(source),
            available_count=sum(available),descriptive_beta_fits=fits,cells=cells))
    for raw,rank in (('raw_candidate','candidate'),('raw_control','control')):
        values[rank] = original_pct_ranks(values[raw],values['available'])
    for i,event in enumerate(history):
        for j,cell in enumerate(event['cells']):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i,j]),control=float(values['control'][i,j]))
    s,a = sum(source_counts),sum(available_counts)
    checked = int(values['available'].size)
    audit = dict(events=len(queries),symbols_per_event=width,checked_cells=checked,
        price_count=253,risk_price_count=64,risk_returns=63,skip_sessions=42,market_symbol='SPY',
        sample_ddof=1,residual_recentered=False,
        price_observation_counting='consumed observations: 253 formation + 64 stock risk + 64 market risk; overlaps repeated',
        source_available_cells=s,source_available_by_event=source_counts,
        available_cells=a,available_by_event=available_counts,unavailable_cells=checked-a,
        reason_counts=reasons,formation_price_observations=253*s,risk_price_observations=128*s,
        valid_price_observations=381*s,simple_returns=126*s,descriptive_beta_fits=beta_fits,
        descriptive_beta_fits_by_event=[event['descriptive_beta_fits'] for event in history],
        normalized_momentum_scores=2*a,ranked_observations=2*a,
        predictive_model_fits=0,new_model_fits=0,common_availability=True,ranking_performed=True,
        ranking_eligibility_filtered=False,trading_eligibility_modified=False,
        portfolio_returns_calculated=False,event_history=history)
    return values,audit


def feature_arrays(values):
    require(isinstance(values,dict) and set(values) == set(VALUE_KEYS),'Feature keys differ')
    a = {key:np.asarray(value) for key,value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool,'Boolean availability required')
    for key in set(VALUE_KEYS)-{'available'}:
        require(a[key].shape == mask.shape and np.issubdtype(a[key].dtype,np.floating), 'Float feature matrix: '+key)
        require(not np.isinf(a[key]).any(),'Infinite feature: '+key)
    source = np.isfinite(a['momentum'])
    require(np.array_equal(np.isfinite(a['total_risk']),source),'Total risk/source mask differs')
    fit = np.isfinite(a['beta'])
    require(np.array_equal(np.isfinite(a['idiosyncratic_risk']),fit),'Projection descriptor masks differ')
    require(not (fit & ~source).any(),'Projection without complete source')
    require(np.all(a['total_risk'][source]>=0.) and np.all(a['idiosyncratic_risk'][fit]>=0.),'Negative risk')
    require(np.array_equal(mask,fit & (a['total_risk']>0.) & (a['idiosyncratic_risk']>0.)), 'Common positive-risk mask differs')
    for key in TARGET_KEYS:
        require(np.array_equal(np.isfinite(a[key]),mask),'Common target mask differs: '+key)
    for i,j in zip(*np.nonzero(mask)):
        for raw,risk in (('raw_candidate','idiosyncratic_risk'),('raw_control','total_risk')):
            require(a[raw][i,j] == float(a['momentum'][i,j])/float(a[risk][i,j]),'Ratio differs: '+raw)
    for raw,rank in (('raw_candidate','candidate'),('raw_control','control')):
        require(np.array_equal(a[rank],original_pct_ranks(a[raw],mask),equal_nan=True),'Original percentile rank: '+rank)
    return a


def verify_features(panel,meta,calendar,values,audit,limit=None):
    actual = feature_arrays(values)
    wanted,expected = independent_features(panel,meta,calendar,limit)
    for key in wanted:
        require(np.array_equal(actual[key],wanted[key],equal_nan=True),'Independent feature differs: '+key)
        if key != 'available':
            used = np.isfinite(wanted[key])
            require(np.array_equal(np.signbit(actual[key][used]),np.signbit(wanted[key][used])), 'Independent sign differs: '+key)
    record_check(audit,expected)
    counters = ('events','symbols_per_event','source_available_by_event','available_by_event',
                'descriptive_beta_fits_by_event',*COUNT_FIELDS)
    return dict(**{key:True for key in FEATURE_FLAGS},**{key:expected[key] for key in counters},
        predictive_model_fits=0,new_model_fits=0,computes_returns=False,producer_modules_imported=False,
        numerical_comparison='exact scalar binary64 cumulative momentum, current63 projection risk, pair-counter ranks and full audit',
        numerical_tolerance=0.,outcome_adapter_sha256=FROZEN_HELPER_SHA)


review_features = verify_features


def original_prefix(original, n, width):
    require(isinstance(original, dict) and all(k in original for k in ('core','core_weight','state_calm10')),
            'Original core/state fields required')
    rawcore = np.asarray(original['core'])
    require(rawcore.ndim == 2 and rawcore.shape[1] == width and len(rawcore) >= n, 'Original core shape')
    full = len(rawcore)
    state = np.asarray(original['state_calm10'])
    require(state.shape == (full,), 'Original calm shape')
    require(all(isinstance(x, (bool, np.bool_)) for x in state[:n]), 'Original calm must be boolean')
    calm = np.asarray(state[:n], dtype=bool)
    rawweight = np.asarray(original['core_weight'])
    require(rawweight.shape in ((full,), (full,1), (full,width)), 'Original core weight shape')
    if rawweight.ndim == 1:
        rawweight = rawweight[:,None]
    require(all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))
                for x in rawcore[:n].flat) and all(isinstance(x, numbers.Real) and
                not isinstance(x, (bool, np.bool_)) for x in rawweight[:n].flat),
            'Original core/weights must be numeric nonboolean')
    try:
        weight = np.broadcast_to(np.asarray(rawweight[:n], dtype=float), (n,width)).copy()
        core = np.asarray(rawcore[:n], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Nonnumeric original core/weights') from exc
    require(np.isfinite(core).all() and np.isfinite(weight).all(), 'Nonfinite original core/weights')
    require(np.array_equal(weight, np.broadcast_to(np.where(calm,.5,.52)[:,None], (n,width))),
            'Core weights must exactly follow original calm')
    require(np.all(np.abs(core) <= .5*weight), 'Original weighted core outside rank bounds')
    return core, weight, full


def independent_deltas(values, original, targets, blend=BLEND):
    require(type(blend) in (float,int) and blend == BLEND, 'Fixed half-core blend required')
    a = feature_arrays(values)
    n, width = a['available'].shape
    core, weight, full = original_prefix(original,n,width)
    require(isinstance(targets,dict) and set(targets) == set(RAW), 'Target arms differ')
    result = {}
    for arm, raw in RAW.items():
        target = np.asarray(targets[arm])
        require(target.ndim == 2 and target.shape in ((n,width),(full,width)) and
                np.issubdtype(target.dtype,np.floating), 'Full/prefix floating targets required')
        require(not np.isinf(target).any(), 'Infinite target rejected')
        if len(target)>n:
            require(np.isnan(target[n:]).all(), 'Unfilled target suffix must remain NaN')
        wanted = a[raw].copy()
        calm = np.asarray(original['state_calm10'][:n], bool)
        wanted[calm] = np.nan
        if n:
            wanted[0] = np.nan
        require(np.array_equal(target[:n], wanted, equal_nan=True), 'Target prefix, calm or first partial differs')
        delta = np.zeros((n,width),float)
        for i in range(n):
            for j in range(width):
                if math.isfinite(float(wanted[i,j])):
                    delta[i,j] = blend*(float(weight[i,j])*float(wanted[i,j])-float(core[i,j]))
        require(np.isfinite(delta).all() and np.all(np.abs(delta) <= .26), 'Half-core delta exceeds bound')
        result[arm] = delta
    return result


def verify_deltas(values, arrays, targets, deltas, blend=BLEND):
    require(isinstance(deltas,dict) and set(deltas) == set(RAW), 'Delta arms differ')
    wanted = independent_deltas(values, arrays, targets, blend)
    for arm, expected in wanted.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == expected.shape and np.issubdtype(actual.dtype,np.floating) and
                np.isfinite(actual).all() and np.array_equal(actual,expected), 'Exact half-core delta differs: '+arm)
    n = np.shape(values['available'])[0]
    return dict(**{key: True for key in DELTA_FLAGS},checked_events=n,blend=BLEND,max_absolute_delta=.26,
        changed_cells={arm:int(np.count_nonzero(value)) for arm,value in wanted.items()},
        target_available_cells={arm:int(np.isfinite(np.asarray(targets[arm])[:n]).sum()) for arm in RAW})
