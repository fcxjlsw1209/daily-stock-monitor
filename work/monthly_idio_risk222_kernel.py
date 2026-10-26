"""Skip42 cumulative momentum with current63 market-projection risk.

Pure source transformation, with no I/O, eligibility filter, predictive model,
orders or returns. Projection RSS and total centered SS both divide by62.
The host alone suppresses initial/calm deltas and applies the half-core rule.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd
from monthly_tail_moment221_kernel import _frame, _calendar, _queries

PRICE_COUNT, RISK_PRICE_COUNT, RISK_RETURNS, SKIP = 253, 64, 63, 42
MARKET_SYMBOL = 'SPY'
FEATURE_KEYS = ('available', 'candidate', 'control', 'momentum', 'beta',
                'idiosyncratic_risk', 'total_risk', 'raw_candidate', 'raw_control')
COUNT_FIELDS = ('checked_cells', 'source_available_cells', 'available_cells',
    'unavailable_cells', 'formation_price_observations', 'risk_price_observations',
    'valid_price_observations', 'simple_returns', 'descriptive_beta_fits',
    'normalized_momentum_scores', 'ranked_observations')
FLOAT_FIELDS = ('momentum', 'beta', 'idiosyncratic_risk', 'total_risk',
                'raw_candidate', 'raw_control')
SCALAR_FIELDS = ('momentum', 'stock_mean', 'market_mean',
    'stock_centered_sum_squares', 'market_centered_sum_squares',
    'cross_product_sum', 'beta', 'residual_sum_squares',
    'idiosyncratic_risk', 'total_risk', 'raw_candidate', 'raw_control')
ENDPOINT_FIELDS = ('start_price', 'formation_end_price', 'stock_risk_start_price',
    'stock_risk_end_price', 'market_risk_start_price', 'market_risk_end_price')
PRICE_OBSERVATION_COUNTING = ('consumed observations: 253 formation + 64 stock risk + '
                              '64 market risk; overlaps repeated')


def empty_measure(reason):
    result = dict(source_available=False, available=False, reason=reason,
                  descriptive_beta_fit=0)
    result.update({key: None for key in ENDPOINT_FIELDS})
    result.update({key: np.nan for key in SCALAR_FIELDS})
    return result


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite calculation from finite source')
    return value


def _sum_squares(values):
    result = finite(math.fsum(finite(v*v) for v in values))
    if result == 0. and any(v != 0. for v in values):
        raise FloatingPointError('Nonzero vector squares underflowed')
    return result


def _risk(sum_squares):
    result = finite(math.sqrt(sum_squares/62.))
    if sum_squares > 0. and result == 0.:
        raise FloatingPointError('Positive variance underflowed')
    return result


def _prices(raw, size, label):
    if np.shape(raw) != (size,):
        raise ValueError('Exactly %s %s prices required' % (size, label))
    values = np.asarray(raw, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real)
           for x in values):
        return None, 'nonnumeric_'+label+'_price'
    try:
        values = [float(x) for x in values]
    except (TypeError, ValueError, OverflowError):
        return None, 'nonnumeric_'+label+'_price'
    if not all(math.isfinite(x) and x > 0. for x in values):
        return None, 'invalid_'+label+'_price'
    return values, None


def measure(formation_prices, stock_risk_prices, market_risk_prices):
    """Consume253/64/64 prices; only the momentum numerator skips42.

    Current63 simple returns are centered once. beta=fsum(x*y)/fsum(x*x),
    e=y-beta*x. The projection risk is sqrt(fsum(e*e)/62), without recentering
    residuals or an extra fitted-parameter degrees-of-freedom adjustment.
    Identical stock/market returns naturally yield beta1 and exact zero RSS.
    """
    inputs = ((formation_prices, PRICE_COUNT, 'formation'),
              (stock_risk_prices, RISK_PRICE_COUNT, 'stock_risk'),
              (market_risk_prices, RISK_PRICE_COUNT, 'market_risk'))
    # Shapes are input contracts even when another source contains bad values.
    for raw, size, label in inputs:
        if np.shape(raw) != (size,):
            raise ValueError('Exactly %s %s prices required' % (size, label))
    parsed = []
    for raw, size, label in inputs:
        values, reason = _prices(raw, size, label)
        if reason is not None:
            return empty_measure(reason)
        parsed.append(values)
    p, stock, market = parsed
    try:
        momentum = finite(p[-1]/p[0]-1.)
        stock_returns = [finite(stock[j+1]/stock[j]-1.) for j in range(RISK_RETURNS)]
        market_returns = [finite(market[j+1]/market[j]-1.) for j in range(RISK_RETURNS)]
        stock_mean = finite(math.fsum(stock_returns)/RISK_RETURNS)
        market_mean = finite(math.fsum(market_returns)/RISK_RETURNS)
        y = [finite(r-stock_mean) for r in stock_returns]
        x = [finite(r-market_mean) for r in market_returns]
        stock_ss = _sum_squares(y)
        market_ss = _sum_squares(x)
        cross = finite(math.fsum(finite(a*b) for a, b in zip(x, y)))
        total_risk = _risk(stock_ss)
        # A positive market SS that becomes zero at the matched sample scale
        # is a numerical failure, not evidence of a constant market.
        _risk(market_ss)
        result = empty_measure('zero_market_variance')
        result.update(source_available=True, start_price=p[0], formation_end_price=p[-1],
            stock_risk_start_price=stock[0], stock_risk_end_price=stock[-1],
            market_risk_start_price=market[0], market_risk_end_price=market[-1],
            momentum=momentum, stock_mean=stock_mean, market_mean=market_mean,
            stock_centered_sum_squares=stock_ss, market_centered_sum_squares=market_ss,
            cross_product_sum=cross, total_risk=total_risk)
        if market_ss == 0.:
            return result
        beta = finite(cross/market_ss)
        residuals = [finite(b-finite(beta*a)) for a, b in zip(x, y)]
        residual_ss = _sum_squares(residuals)
        idio_risk = _risk(residual_ss)
        result.update(beta=beta, residual_sum_squares=residual_ss,
                      idiosyncratic_risk=idio_risk, descriptive_beta_fit=1)
        if total_risk == 0.:
            result['reason'] = 'zero_total_risk'
            return result
        if idio_risk == 0.:
            result['reason'] = 'zero_idiosyncratic_risk'
            return result
        result.update(available=True, reason='available',
            raw_candidate=finite(momentum/idio_risk), raw_control=finite(momentum/total_risk))
        return result
    except (OverflowError, ZeroDivisionError) as exc:
        raise FloatingPointError('Arithmetic failure from finite source') from exc


def percentile_rank(raw, available):
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Finite shared rank domain required')
    result = np.full(raw.shape, np.nan)
    ids = np.flatnonzero(available)
    if len(ids):
        result[ids] = pd.Series(raw[ids]).rank(method='average', pct=True).to_numpy()-.5
    return result


def build_features(panel, meta, calendar, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Exactly adj panel required')
    adj = panel['adj']; _frame(adj)
    sessions = _calendar(calendar); queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('Integer event prefix required')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = {key: np.full(shape, np.nan) for key in FEATURE_KEYS if key != 'available'}
    values['available'] = np.zeros(shape, bool)
    history, reasons, source_counts, counts = [], Counter(), [], []
    beta_fits, beta_fit_counts = 0, []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        formation = sessions[t-294:t-42+1] if t >= 294 else None
        risk = sessions[t-63:t+1] if t >= 63 else None
        missing_formation = (formation.difference(adj.index).strftime('%Y-%m-%d').tolist()
                             if formation is not None else [])
        missing_risk = (risk.difference(adj.index).strftime('%Y-%m-%d').tolist()
                        if risk is not None else [])
        cells = []
        for j, symbol in enumerate(symbols):
            if formation is None or risk is None:
                row = empty_measure('insufficient_history')
            elif symbol not in adj.columns:
                row = empty_measure('missing_symbol')
            elif MARKET_SYMBOL not in adj.columns:
                row = empty_measure('missing_market_symbol')
            elif missing_formation:
                row = empty_measure('missing_formation_session')
            elif missing_risk:
                row = empty_measure('missing_risk_session')
            else:
                row = measure(adj.loc[formation, symbol].to_numpy(copy=True),
                              adj.loc[risk, symbol].to_numpy(copy=True),
                              adj.loc[risk, MARKET_SYMBOL].to_numpy(copy=True))
            values['available'][i, j] = row['available']
            for key in FLOAT_FIELDS:
                values[key][i, j] = row[key]
            cell = dict(row, symbol=symbol, date=iso, candidate=None, control=None)
            for key in SCALAR_FIELDS:
                if not math.isfinite(cell[key]):
                    cell[key] = None
            cells.append(cell)
        for key in ('candidate', 'control'):
            values[key][i] = percentile_rank(values['raw_'+key][i], values['available'][i])
        for j, cell in enumerate(cells):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i, j]),
                            control=float(values['control'][i, j]))
            reasons[cell['reason']] += 1
        source = sum(c['source_available'] for c in cells)
        available = int(values['available'][i].sum())
        event_fits = sum(c['descriptive_beta_fit'] for c in cells)
        source_counts.append(source); counts.append(available); beta_fits += event_fits
        beta_fit_counts.append(event_fits)
        fstart = str(formation[0].date()) if formation is not None else None
        fend = str(formation[-1].date()) if formation is not None else None
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=fstart, price_end=fend,
            formation_price_start=fstart, formation_price_end=fend,
            risk_price_start=str(risk[0].date()) if risk is not None else None,
            risk_price_end=str(risk[-1].date()) if risk is not None else None,
            first_risk_return_date=str(risk[1].date()) if risk is not None else None,
            missing_formation_sessions=missing_formation, missing_risk_sessions=missing_risk,
            source_available=[c['source_available'] for c in cells],
            available=[c['available'] for c in cells], source_available_count=source,
            available_count=available, descriptive_beta_fits=event_fits, cells=cells))
    source_total, total = sum(source_counts), sum(counts)
    audit = dict(events=len(queries), symbols_per_event=width,
        checked_cells=int(values['available'].size), price_count=PRICE_COUNT,
        risk_price_count=RISK_PRICE_COUNT, risk_returns=RISK_RETURNS, skip_sessions=SKIP,
        market_symbol=MARKET_SYMBOL, sample_ddof=1, residual_recentered=False,
        price_observation_counting=PRICE_OBSERVATION_COUNTING,
        source_available_cells=source_total, source_available_by_event=source_counts,
        available_cells=total, available_by_event=counts,
        unavailable_cells=int(values['available'].size)-total, reason_counts=dict(reasons),
        formation_price_observations=253*source_total, risk_price_observations=128*source_total,
        valid_price_observations=381*source_total, simple_returns=126*source_total,
        descriptive_beta_fits=beta_fits, descriptive_beta_fits_by_event=beta_fit_counts,
        normalized_momentum_scores=2*total,
        ranked_observations=2*total, predictive_model_fits=0, new_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False,
        event_history=history)
    return values, audit
