"""Pure adjacent absolute-product versus RMS risk normalization.

Only long momentum skips42. Current63 simple returns include the query close.
Shared295-price support and joint positive denominators precede both original
percentile ranks. No source I/O, eligibility filter, model fit or trading.
"""
from collections import Counter
import math
import numbers
import numpy as np
import pandas as pd
from monthly_nested_trend199_kernel import _frame, _calendar, _queries

PRICE_COUNT, SKIP, RETURN_COUNT, ADJACENT_COUNT = 295, 42, 63, 62


def empty_measure(reason):
    return dict(source_available=False, available=False, reason=reason,
        start_price=None, formation_end_price=None, risk_start_price=None, end_price=None,
        numerator=None, simple_returns=[], absolute_adjacent_products=[], squared_returns=[],
        adjacent_sum=None, squared_sum=None, candidate_variance=None, control_variance=None,
        candidate_denominator=None, control_denominator=None,
        raw_candidate=np.nan, raw_control=np.nan, risk_estimates=0)


def measure(prices):
    if np.shape(prices) != (PRICE_COUNT,):
        raise ValueError('Exactly295 source prices required')
    raw = np.asarray(prices, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return empty_measure('nonnumeric_adjusted_price')
    try:
        p = [float(x) for x in raw]
    except (ValueError, TypeError, OverflowError):
        return empty_measure('nonnumeric_adjusted_price')
    if not all(math.isfinite(x) and x > 0 for x in p):
        return empty_measure('invalid_adjusted_price')
    try:
        numerator = p[252]/p[0]-1.
        returns = [p[i]/p[i-1]-1. for i in range(232, 295)]
        if not all(math.isfinite(x) for x in [numerator]+returns):
            raise FloatingPointError('Nonfinite ratio from finite source')
        products = [abs(returns[i])*abs(returns[i-1]) for i in range(1, 63)]
        squares = [r*r for r in returns]
        if not all(math.isfinite(x) for x in products+squares):
            raise FloatingPointError('Nonfinite risk product from finite returns')
        adjacent_sum, squared_sum = math.fsum(products), math.fsum(squares)
        if not all(math.isfinite(x) for x in (adjacent_sum, squared_sum)):
            raise FloatingPointError('Nonfinite risk sum')
        # Operation order is prospective: multiply by pi/2 before dividing62.
        candidate_variance = (math.pi/2.)*adjacent_sum/62.
        control_variance = squared_sum/63.
        if not all(math.isfinite(x) for x in (candidate_variance, control_variance)):
            raise FloatingPointError('Nonfinite risk variance')
        dpair, drms = math.sqrt(candidate_variance), math.sqrt(control_variance)
        if not all(math.isfinite(x) for x in (dpair, drms)):
            raise FloatingPointError('Nonfinite risk denominator')
        result = dict(source_available=True, available=False, reason='zero_risk',
            start_price=p[0], formation_end_price=p[252], risk_start_price=p[231], end_price=p[294],
            numerator=numerator, simple_returns=returns, absolute_adjacent_products=products,
            squared_returns=squares, adjacent_sum=adjacent_sum, squared_sum=squared_sum,
            candidate_variance=candidate_variance, control_variance=control_variance,
            candidate_denominator=dpair, control_denominator=drms,
            raw_candidate=np.nan, raw_control=np.nan, risk_estimates=2)
        if dpair == 0. or drms == 0.:
            return result
        q, c = numerator/dpair, numerator/drms
        if not all(math.isfinite(x) for x in (q, c)):
            raise FloatingPointError('Nonfinite normalized momentum')
        result.update(available=True, reason='available', raw_candidate=q, raw_control=c)
        return result
    except OverflowError as exc:
        raise FloatingPointError('Nonfinite arithmetic from finite source') from exc


def percentile_rank(raw, available):
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite values and boolean support required')
    result = np.full(raw.shape, np.nan)
    ids = np.flatnonzero(available)
    if len(ids):
        result[ids] = pd.Series(raw[ids]).rank(method='average', pct=True).to_numpy()-.5
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
    values = {k: np.full(shape, np.nan) for k in ('candidate', 'control', 'raw_candidate', 'raw_control')}
    values['available'] = np.zeros(shape, bool)
    history, reasons, source_counts, counts = [], Counter(), [], []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date()); dates = sessions[t-294:t+1] if t >= 294 else None
        missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist() if dates is not None else []
        cells = []
        for j, symbol in enumerate(symbols):
            if dates is None: result = empty_measure('insufficient_history')
            elif symbol not in adj.columns: result = empty_measure('missing_symbol')
            elif missing: result = empty_measure('missing_official_session')
            else: result = measure(adj.loc[dates, symbol].to_numpy(copy=True))
            values['available'][i, j] = result['available']
            for key in ('raw_candidate', 'raw_control'): values[key][i, j] = result[key]
            cell = dict(result, symbol=symbol, date=iso, candidate=None, control=None)
            if not result['available']: cell.update(raw_candidate=None, raw_control=None)
            cells.append(cell)
        for key in ('candidate', 'control'):
            values[key][i] = percentile_rank(values['raw_'+key][i], values['available'][i])
        for j, cell in enumerate(cells):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i, j]), control=float(values['control'][i, j]))
            reasons[cell['reason']] += 1
        source = sum(c['source_available'] for c in cells); count = int(values['available'][i].sum())
        source_counts.append(source); counts.append(count)
        def stamp(index): return str(dates[index].date()) if dates is not None else None
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=stamp(0), price_end=stamp(294), numerator_start=stamp(0), numerator_end=stamp(252),
            risk_price_start=stamp(231), risk_price_end=stamp(294), missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells], available=[c['available'] for c in cells],
            source_available_count=source, available_count=count, cells=cells))
    source_total, total = sum(source_counts), sum(counts)
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=295, skip_sessions=42, current_return_count=63, adjacent_product_count=62,
        source_available_cells=source_total, available_cells=total,
        unavailable_cells=int(values['available'].size)-total,
        source_available_by_event=source_counts, available_by_event=counts,
        valid_price_observations=295*source_total, simple_return_observations=63*source_total,
        adjacent_product_observations=62*source_total, squared_return_observations=63*source_total,
        risk_estimates=2*source_total, ratio_observations=2*total, ranked_score_observations=2*total,
        new_model_fits=0, predictive_model_fits=0, common_availability=True, ranking_performed=True,
        ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, reason_counts=dict(reasons), event_history=history)
