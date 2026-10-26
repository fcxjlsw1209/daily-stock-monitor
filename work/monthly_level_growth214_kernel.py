"""Skip-42 long level growth from two half-window geometric means.

Pure supplied-data features. Full295 positive prices share one availability;
center formation price is checked but not logged/used in the numerator.
The host alone suppresses initial/calm deltas and applies the half-core rule.
"""
from collections import Counter
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PRICE_COUNT, FORMATION_COUNT, HALF_COUNT = 295, 253, 126
RETURN_COUNT, SKIP, SEPARATION = 63, 42, 127
LIFT_SCALE = 252. / 127.
VALUE_KEYS = ('candidate', 'control', 'raw_candidate', 'raw_control',
              'numerator_candidate', 'numerator_control', 'volatility', 'log_lift', 'available')
COUNT_FIELDS = ('valid_price_observations', 'formation_log_observations',
    'simple_return_observations', 'segment_mean_evaluations', 'lift_evaluations',
    'numerator_evaluations', 'risk_evaluations', 'ranked_observations')


def dependency_files():
    return (Path(__file__).resolve(), Path(__file__).with_name('monthly_nested_trend199_kernel.py').resolve())


def _finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label + ' from valid finite source')
    return value


def _sum(values, label):
    try:
        return _finite(math.fsum(values), label)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite ' + label) from exc


def empty_measure(reason):
    return dict(source_available=False, available=False, reason=reason,
        start_price=None, formation_end_price=None, omitted_center_price=None,
        risk_start_price=None, end_price=None, early_logs=[], late_logs=[],
        early_log_mean=None, late_log_mean=None, mean_difference=None, log_lift=None,
        numerator_candidate=None, numerator_control=None, simple_returns=[],
        return_mean=None, volatility=None, raw_candidate=np.nan, raw_control=np.nan)


def measure(prices):
    """Two126-log means and fixed centroid-distance scaling; no approximate branches."""
    if np.shape(prices) != (PRICE_COUNT,):
        raise ValueError('Exactly295 adjusted prices required')
    raw = np.asarray(prices, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:
        p = [float(v) for v in raw]
    except (ValueError, TypeError, OverflowError):
        return empty_measure('nonnumeric_adjusted_price')
    if not all(math.isfinite(v) and v > 0. for v in p):
        return empty_measure('invalid_adjusted_price')
    try:
        early = [_finite(math.log(v), 'Early formation log') for v in p[:126]]
        late = [_finite(math.log(v), 'Late formation log') for v in p[127:253]]
        early_mean = _finite(_sum(early, 'Early log sum') / 126., 'Early log mean')
        late_mean = _finite(_sum(late, 'Late log sum') / 126., 'Late log mean')
        difference = _finite(late_mean - early_mean, 'Log mean difference')
        lift = _finite(LIFT_SCALE * difference, 'Scaled log level growth')
        numerator_candidate = _finite(math.expm1(lift), 'Level growth numerator')
        numerator_control = _finite(p[252] / p[0] - 1., 'Endpoint numerator')
        returns = [_finite(p[j] / p[j-1] - 1., 'Current simple return') for j in range(232, 295)]
        mean = _finite(_sum(returns, 'Return sum') / 63., 'Return mean')
        residual = [_finite(v - mean, 'Return residual') for v in returns]
        squares = [_finite(v*v, 'Return residual square') for v in residual]
        variance = _finite(_sum(squares, 'Return SSE') / 62., 'Sample return variance')
        vol = _finite(math.sqrt(variance), 'Sample volatility')
        result = empty_measure('zero_volatility' if vol == 0. else 'available')
        result.update(source_available=True, start_price=p[0], formation_end_price=p[252],
            omitted_center_price=p[126], risk_start_price=p[231], end_price=p[294],
            early_logs=early, late_logs=late, early_log_mean=early_mean,
            late_log_mean=late_mean, mean_difference=difference, log_lift=lift,
            numerator_candidate=numerator_candidate, numerator_control=numerator_control,
            simple_returns=returns, return_mean=mean, volatility=vol)
        if vol == 0.:
            return result
        q = _finite(numerator_candidate / vol, 'Risk-normalized level growth')
        c = _finite(numerator_control / vol, 'Risk-normalized endpoint growth')
        result.update(available=True, raw_candidate=q, raw_control=c)
        return result
    except OverflowError as exc:
        raise FloatingPointError('Nonfinite arithmetic from valid finite prices') from exc


def percentile_rank(raw, available):
    """Original average ascending percentile minus.5 on the common full-pool support."""
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.full(raw.shape, np.nan)
    ids = np.flatnonzero(available)
    if len(ids):
        result[ids] = pd.Series(raw[ids]).rank(method='average', pct=True).to_numpy() - .5
    return result


def build_features(panel, meta, calendar, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    adj = panel['adj']; _frame(adj)
    sessions = _calendar(calendar); queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('Invalid event prefix')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = {k: np.full(shape, np.nan) for k in VALUE_KEYS if k != 'available'}
    values['available'] = np.zeros(shape, bool)
    history, reasons, source_counts, counts = [], Counter(), [], []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date()); dates = sessions[t-294:t+1] if t >= 294 else None
        missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells = []
        for j, symbol in enumerate(symbols):
            if dates is None:
                result = empty_measure('insufficient_history')
            elif symbol not in adj.columns:
                result = empty_measure('missing_symbol')
            elif missing:
                result = empty_measure('missing_official_session')
            else:
                result = measure(adj.loc[dates, symbol].to_numpy(copy=True))
            values['available'][i, j] = result['available']
            if result['available']:
                for key in VALUE_KEYS[2:-1]:
                    values[key][i, j] = result[key]
            cell = dict(result, symbol=symbol, date=iso, candidate=None, control=None)
            if not result['available']:
                cell.update(raw_candidate=None, raw_control=None)
            cells.append(cell)
        for key in ('candidate', 'control'):
            values[key][i] = percentile_rank(values['raw_'+key][i], values['available'][i])
        for j, cell in enumerate(cells):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i, j]), control=float(values['control'][i, j]))
            reasons[cell['reason']] += 1
        source = sum(c['source_available'] for c in cells)
        available = int(values['available'][i].sum())
        source_counts.append(source); counts.append(available)
        def stamp(index):
            return str(dates[index].date()) if dates is not None else None
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=stamp(0), price_end=stamp(294), numerator_start=stamp(0), numerator_end=stamp(252),
            formation_price_start=stamp(0), formation_price_end=stamp(252),
            early_mean_start=stamp(0), early_mean_end=stamp(125), omitted_center_date=stamp(126),
            late_mean_start=stamp(127), late_mean_end=stamp(252),
            risk_price_start=stamp(231), risk_price_end=stamp(294), missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells], available=[c['available'] for c in cells],
            source_available_count=source, available_count=available, cells=cells))
    s, a = sum(source_counts), sum(counts)
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=values['available'].size,
        price_count=295, formation_price_count=253, used_log_price_count=252,
        half_price_count=126, omitted_center_index=126, centroid_distance=127, lift_scale=LIFT_SCALE,
        skip_sessions=42, current_return_count=63, ddof=1,
        source_available_cells=s, source_available_by_event=source_counts,
        available_cells=a, available_by_event=counts, unavailable_cells=values['available'].size-a,
        valid_price_observations=295*s, formation_log_observations=252*s,
        simple_return_observations=63*s, segment_mean_evaluations=2*s,
        lift_evaluations=s, numerator_evaluations=2*s, risk_evaluations=s, ranked_observations=2*a,
        descriptive_model_fits=0, new_model_fits=0, predictive_model_fits=0, new_predictive_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False,
        reason_counts=dict(reasons), event_history=history)
