"""Annual own-security risk-forecast reliability from matured calendar months.

Only pure supplied-source arithmetic. No I/O, prediction of current outcomes,
eligibility filter, portfolio execution or source-model refitting is performed.
Two calibration means are estimated per supported unique (year, security).
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd
from monthly_tail_moment221_kernel import _frame, _calendar, _queries

HISTORY_MONTHS, MIN_MONTHS, FORECAST_RETURNS = 60, 18, 63
FEATURE_KEYS = ('available', 'candidate', 'control', 'raw_candidate', 'raw_control')
COUNT_FIELDS = ('checked_cells', 'available_cells', 'unavailable_cells',
    'year_security_groups', 'supported_groups', 'cold_start_groups',
    'historical_month_pairs', 'source_available_pairs', 'available_pairs',
    'valid_price_observations', 'simple_return_observations',
    'second_moment_estimates', 'error_evaluations', 'calibration_estimates',
    'ranked_observations')
PAIR_FLOAT_FIELDS = ('forecast_second_moment', 'prior_second_moment',
    'realized_second_moment', 'candidate_error', 'control_error')


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from finite source')
    return value


def _square(value):
    result = finite(value*value)
    if value != 0. and result == 0.:
        raise FloatingPointError('Nonzero squared value underflowed')
    return result


def empty_measure(reason):
    return dict(source_available=False, available=False, reason=reason,
                **{key: None for key in PAIR_FLOAT_FIELDS})


def _prices(raw, label):
    raw = np.asarray(raw, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return None, 'nonnumeric_'+label+'_price'
    try:
        result = [float(x) for x in raw]
    except (ValueError, TypeError, OverflowError):
        return None, 'nonnumeric_'+label+'_price'
    if not all(math.isfinite(x) and x > 0. for x in result):
        return None, 'invalid_'+label+'_price'
    return result, None


def second_moment(prices):
    returns = [finite(prices[i+1]/prices[i]-1.) for i in range(len(prices)-1)]
    total = finite(math.fsum(_square(r) for r in returns))
    result = finite(total/len(returns))
    if total > 0. and result == 0.:
        raise FloatingPointError('Positive second moment underflowed')
    return result


def error(realized, forecast):
    denominator = finite(realized+forecast)
    if denominator <= 0.:
        raise ValueError('Positive joint risk denominator required')
    difference = finite(realized-forecast)
    ratio = finite(difference/denominator)
    if difference != 0. and ratio == 0.:
        raise FloatingPointError('Nonzero normalized error underflowed')
    return _square(ratio)


def _negative_mean(errors):
    total = finite(math.fsum(errors))
    result = finite(total/len(errors))
    if total > 0. and result == 0.:
        raise FloatingPointError('Positive mean error underflowed')
    return -result


def measure(forecast_prices, prior_prices, realized_prices):
    inputs = ((forecast_prices, 'forecast'), (prior_prices, 'prior'),
              (realized_prices, 'realized'))
    for raw, label in inputs:
        shape = np.shape(raw)
        if len(shape) != 1 or (shape[0] != 64 if label == 'forecast' else shape[0] < 2):
            raise ValueError('Forecast requires64 prices; calendar-month windows require at least2')
    parsed = []
    for raw, label in inputs:
        values, reason = _prices(raw, label)
        if reason is not None:
            return empty_measure(reason)
        parsed.append(values)
    try:
        forecast, prior, realized = [second_moment(p) for p in parsed]
        result = empty_measure('zero_pair_denominator')
        result.update(source_available=True, forecast_second_moment=forecast,
                      prior_second_moment=prior, realized_second_moment=realized)
        d_forecast, d_prior = finite(realized+forecast), finite(realized+prior)
        if d_forecast == 0. or d_prior == 0.:
            return result
        result.update(available=True, reason='available',
                      candidate_error=error(realized, forecast), control_error=error(realized, prior))
        return result
    except (OverflowError, ZeroDivisionError) as exc:
        raise FloatingPointError('Risk-pair arithmetic failure from finite source') from exc


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


def _complete_month(period, buckets, sessions):
    ids = buckets.get(period)
    if not ids:
        raise ValueError('Missing official calendar month: '+str(period))
    lo, hi = ids[0], ids[-1]
    if (lo == 0 or hi+1 >= len(sessions)
            or sessions[lo-1].to_period('M') != period-1
            or sessions[hi+1].to_period('M') != period+1):
        raise ValueError('Unverified complete official calendar month: '+str(period))
    return lo, hi


def _year_windows(year, sessions, buckets):
    result = []
    for period in pd.period_range('%04d-01' % (year-5), '%04d-12' % (year-1), freq='M'):
        first, last = _complete_month(period, buckets, sessions)
        prev_first, prev_last = _complete_month(period-1, buckets, sessions)
        if first-1 != prev_last or prev_last < FORECAST_RETURNS:
            raise ValueError('Missing prior-month anchor or63-return forecast calendar warmup')
        forecast = sessions[prev_last-63:prev_last+1]
        prior = sessions[prev_first-1:prev_last+1]
        realized = sessions[prev_last:last+1]
        descriptor = dict(month=str(period), forecast_date=str(sessions[prev_last].date()),
            forecast_price_start=str(forecast[0].date()), forecast_price_end=str(forecast[-1].date()),
            prior_price_start=str(prior[0].date()), prior_price_end=str(prior[-1].date()),
            realized_price_start=str(realized[0].date()), realized_price_end=str(realized[-1].date()),
            prior_return_count=len(prior)-1, realized_return_count=len(realized)-1)
        result.append((descriptor, forecast, prior, realized))
    if len(result) != HISTORY_MONTHS:
        raise ValueError('Exactly60 historical calendar months required')
    return result


def _group(adj, year, symbol, windows):
    pairs, reasons = [], Counter()
    for descriptor, forecast, prior, realized in windows:
        if symbol not in adj.columns:
            measured = empty_measure('missing_symbol')
        elif len(forecast.difference(adj.index)):
            measured = empty_measure('missing_forecast_session')
        elif len(prior.difference(adj.index)):
            measured = empty_measure('missing_prior_session')
        elif len(realized.difference(adj.index)):
            measured = empty_measure('missing_realized_session')
        else:
            measured = measure(adj.loc[forecast, symbol].to_numpy(copy=True),
                               adj.loc[prior, symbol].to_numpy(copy=True),
                               adj.loc[realized, symbol].to_numpy(copy=True))
        pair = dict(descriptor, **measured)
        pairs.append(pair); reasons[pair['reason']] += 1
    valid = [pair for pair in pairs if pair['available']]
    supported = len(valid) >= MIN_MONTHS
    candidate = control = None
    if supported:
        try:
            candidate = _negative_mean([pair['candidate_error'] for pair in valid])
            control = _negative_mean([pair['control_error'] for pair in valid])
        except OverflowError as exc:
            raise FloatingPointError('Calibration error sum overflow') from exc
    return dict(group_id=str(year)+':'+symbol, year=year, symbol=symbol,
        cutoff='%04d-12-31' % (year-1), history_start_month='%04d-01' % (year-5),
        history_end_month='%04d-12' % (year-1),
        source_available_pairs=sum(p['source_available'] for p in pairs),
        available_pairs=len(valid), status='supported' if supported else 'cold_start',
        raw_candidate=candidate, raw_control=control, pair_reason_counts=dict(reasons), pairs=pairs)


def build_features(panel, meta, calendar, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Exactly adj panel required')
    adj = panel['adj']; _frame(adj)
    sessions = _calendar(calendar); queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('Integer event prefix required')
    queries = queries if limit is None else queries[:limit]
    buckets = {}
    for i, session in enumerate(sessions):
        buckets.setdefault(session.to_period('M'), []).append(i)
    keys = sorted({(day.year, symbol) for day, symbols, _ in queries for symbol in symbols})
    windows = {year: _year_windows(year, sessions, buckets) for year in sorted({y for y, _ in keys})}
    groups = [_group(adj, year, symbol, windows[year]) for year, symbol in keys]
    lookup = {(g['year'], g['symbol']): g for g in groups}
    shape = len(queries), width
    values = {key: np.full(shape, np.nan) for key in FEATURE_KEYS if key != 'available'}
    values['available'] = np.zeros(shape, bool)
    history, event_counts = [], []
    for i, (day, symbols, _) in enumerate(queries):
        cells, ids = [], []
        for j, symbol in enumerate(symbols):
            group = lookup[(day.year, symbol)]; available = group['status'] == 'supported'
            values['available'][i, j] = available
            for key in ('raw_candidate', 'raw_control'):
                if available:
                    values[key][i, j] = group[key]
            ids.append(group['group_id'])
            cells.append(dict(symbol=symbol, group_id=group['group_id'], available=available,
                raw_candidate=group['raw_candidate'], raw_control=group['raw_control'],
                candidate=None, control=None))
        for key in ('candidate', 'control'):
            values[key][i] = percentile_rank(values['raw_'+key][i], values['available'][i])
        for j, cell in enumerate(cells):
            if cell['available']:
                cell.update(candidate=float(values['candidate'][i, j]), control=float(values['control'][i, j]))
        count = int(values['available'][i].sum()); event_counts.append(count)
        history.append(dict(date=str(day.date()), symbols=symbols, group_ids=ids,
            available=[cell['available'] for cell in cells], available_count=count, cells=cells))
    source, usable, prices, returns = 0, 0, 0, 0
    pair_reasons = Counter()
    for group in groups:
        pair_reasons.update(group['pair_reason_counts'])
        for pair in group['pairs']:
            source += int(pair['source_available']); usable += int(pair['available'])
            if pair['source_available']:
                returns += 63+pair['prior_return_count']+pair['realized_return_count']
                prices += 64+(pair['prior_return_count']+1)+(pair['realized_return_count']+1)
    supported = sum(g['status'] == 'supported' for g in groups)
    total = sum(event_counts); estimates = 2*supported
    audit = dict(schema_version=1, method='annual_asof_risk_forecast_reliability',
        events=len(queries), symbols_per_event=width, history_months=HISTORY_MONTHS,
        min_months=MIN_MONTHS, forecast_returns=FORECAST_RETURNS,
        second_moment_centered=False, calendar_months_equal_weight=True,
        history_membership_filtered=False, pair_reason_counts=dict(pair_reasons),
        available_by_event=event_counts, checked_cells=int(values['available'].size),
        available_cells=total, unavailable_cells=int(values['available'].size)-total,
        year_security_groups=len(groups), supported_groups=supported,
        cold_start_groups=len(groups)-supported, historical_month_pairs=60*len(groups),
        source_available_pairs=source, available_pairs=usable,
        valid_price_observations=prices, simple_return_observations=returns,
        second_moment_estimates=3*source, error_evaluations=2*usable,
        calibration_estimates=estimates, ranked_observations=2*total,
        new_model_fits=estimates, predictive_model_fits=estimates,
        new_ridge_fits=0, new_tree_fits=0, common_availability=True,
        ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False,
        group_history=groups, event_history=history)
    return values, audit
