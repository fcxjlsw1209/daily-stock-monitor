"""Historical event payoff paired with its own pre-loss risk; pure calculations.

All comparisons use stored positive binary64 prices. No fit, price adjustment,
current pullback rule, eligibility filter, portfolio calculation, or I/O.
"""
from collections import Counter
import math
import numbers

import numpy as np
import pandas as pd

PRICE_COUNT = 253
ANCHOR_FIRST = 64
ANCHOR_LAST = 231
RISK_RETURNS = 63
PAYOFF_HORIZON = 21
COOLDOWN = 21
PRIOR_COUNT = 20
SKIP = 42
INITIAL_PARTIAL = '2016-08-17'


def source_prices(prices):
    """Validate all253 prices, including observations not used by any crossing."""
    if np.shape(prices) != (PRICE_COUNT,):
        raise ValueError('Expected exactly253 adjusted prices')
    raw = np.asarray(prices, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return None, 'nonnumeric_adjusted_price'
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, 'nonnumeric_adjusted_price'
    if not (np.isfinite(numeric) & (numeric > 0)).all():
        return None, 'invalid_adjusted_price'
    return numeric, 'available'


def empty_measure(reason):
    return dict(source_available=False, available=False, reason=reason,
        candidate=np.nan, control=np.nan, selected_anchors=0, valid_event_payoffs=0,
        anchor_indices=[], anchor_events=[], payoff_sum=None, inverse_risk_sum=None,
        standardized_payoff_sum=None, mean_inverse_risk=None, denominator=None, identity_branch=None)


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite '+label+' from finite positive source')
    return float(value)


def finite_sum(values, label):
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid '+label+' sum') from exc
    return finite(total, label)


def aggregate_payoffs(sigmas, payoffs):
    """Shared selected events; exact identity branches preserve equal Q/C bits."""
    if not sigmas or len(sigmas) != len(payoffs):
        raise ValueError('Aligned nonempty risk and payoff vectors required')
    risks = [finite(float(x), 'event sigma') for x in sigmas]
    labels = [finite(float(x), 'event payoff') for x in payoffs]
    if any(x < 0 for x in risks):
        raise ValueError('Sample risk cannot be negative')
    n = len(risks)
    total_y = finite_sum(labels, 'payoff')
    denominator = (n+PRIOR_COUNT)*math.sqrt(PAYOFF_HORIZON)
    inverse, normalized = [], []
    for risk, payoff in zip(risks, labels):
        inverse.append(finite(1./risk, 'inverse sigma') if risk else None)
        normalized.append(finite(payoff/risk, 'normalized payoff') if risk else None)
    result = dict(candidate=np.nan, control=np.nan, available=False,
        payoff_sum=total_y, inverse_risk_sum=None, standardized_payoff_sum=None,
        mean_inverse_risk=None, denominator=denominator, identity_branch=None)
    if any(risk == 0 for risk in risks):
        return result, inverse, normalized
    sum_inverse = finite_sum(inverse, 'inverse risk')
    sum_normalized = finite_sum(normalized, 'standardized payoff')
    mean_inverse = finite(sum_inverse/n, 'mean inverse risk')
    candidate = finite(sum_normalized/denominator, 'candidate score')
    branch = ('single_event' if n == 1 else 'equal_risks' if all(x == risks[0] for x in risks)
              else 'equal_payoffs' if all(x == labels[0] for x in labels) else None)
    if branch:
        control = candidate
    else:
        control = finite(finite(mean_inverse*total_y, 'control numerator')/denominator, 'control score')
    result.update(candidate=candidate, control=control, available=True,
        inverse_risk_sum=sum_inverse, standardized_payoff_sum=sum_normalized,
        mean_inverse_risk=mean_inverse, identity_branch=branch)
    return result, inverse, normalized


def measure(prices):
    source, reason = source_prices(prices)
    result = empty_measure(reason)
    if source is None:
        return result
    result['source_available'] = True
    selected = []
    for s in range(ANCHOR_FIRST, ANCHOR_LAST+1):
        if source[s] < source[s-1] and (not selected or s >= selected[-1]+COOLDOWN):
            selected.append(s)
    result.update(anchor_indices=selected, selected_anchors=len(selected))
    if not selected:
        result['reason'] = 'no_selected_anchors'
        return result
    events = []
    # Selection is complete before evaluating any event risk or realized label.
    for s in selected:
        old = source[s-RISK_RETURNS-1:s]
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            returns = old[1:]/old[:-1]-1.
            if not np.isfinite(returns).all():
                raise FloatingPointError('Nonfinite pre-event simple returns')
            sigma = finite(float(np.std(returns, ddof=1)), 'sample sigma')
        payoff = finite(math.log(float(source[s+PAYOFF_HORIZON]))-math.log(float(source[s])), 'log payoff')
        events.append(dict(anchor_index=s, reference_index=s-1, reference_price=float(source[s-1]),
            anchor_price=float(source[s]), risk_price_start_index=s-64, risk_price_end_index=s-1,
            risk_return_first_index=s-63, risk_return_last_index=s-1,
            pre_event_simple_returns=returns.tolist(), sigma=sigma, maturity_index=s+PAYOFF_HORIZON,
            maturity_price=float(source[s+PAYOFF_HORIZON]), payoff=payoff,
            inverse_sigma=None, standardized_payoff=None))
    aggregate, inverse, normalized = aggregate_payoffs([e['sigma'] for e in events], [e['payoff'] for e in events])
    for event, inv, norm in zip(events, inverse, normalized):
        event.update(inverse_sigma=inv, standardized_payoff=norm)
    result.update(aggregate, anchor_events=events,
        valid_event_payoffs=len(events) if aggregate['available'] else 0,
        reason='available' if aggregate['available'] else 'zero_pre_event_sigma')
    return result


def centered_rank(raw, available):
    raw, available = np.asarray(raw), np.asarray(available)
    if (raw.ndim != 1 or raw.shape != available.shape or available.dtype != bool
            or not np.isfinite(raw[available]).all()):
        raise ValueError('Aligned finite raw values and boolean availability required')
    result = np.zeros(raw.shape, dtype=np.float64)
    indices = np.flatnonzero(available)
    if len(indices) >= 2:
        ranked = pd.Series(raw[indices]).rank(method='average').to_numpy()
        result[indices] = (ranked-1)/(len(indices)-1)-.5
    return result


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label+' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label+' must be a naive midnight date')
    return day


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError('Adjusted prices require DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError('Price sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                         for s in frame.columns)):
        raise ValueError('Price columns must be unique nonempty security IDs')


def _calendar(calendar):
    if isinstance(calendar, (str, bytes)):
        raise ValueError('Calendar requires an explicit sequence')
    dates = []
    for value in calendar:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()) or day.weekday() >= 5:
            raise ValueError('Calendar requires ISO weekday sessions')
        if dates and day <= dates[-1]:
            raise ValueError('Calendar must be unique and increasing')
        dates.append(day)
    if not dates:
        raise ValueError('Calendar cannot be empty')
    return pd.DatetimeIndex(dates)


def _queries(meta, sessions):
    if not isinstance(meta, dict) or any(k not in meta for k in ('dates', 'symbols', 'etfs')):
        raise ValueError('Metadata requires dates, symbols and etfs')
    if isinstance(meta['dates'], (str, bytes)) or isinstance(meta['symbols'], (str, bytes)):
        raise ValueError('Metadata dates and symbols require sequences')
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF identities require an explicit sequence')
    etfs = list(meta['etfs'])
    if (any(not isinstance(s, str) or not s or s.strip() != s for s in etfs)
            or len(etfs) != len(set(etfs))):
        raise ValueError('ETF identities must be unique nonempty strings')
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        t = lookup[day]
        if not (not rows and str(day.date()) == INITIAL_PARTIAL):
            if t+1 >= len(sessions) or sessions[t+1].to_period('M') == day.to_period('M'):
                raise ValueError('Event must be a verified official calendar month-end')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)
                or len(symbols) != len(set(symbols))):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, t))
    return rows, width or 0


def build_features(panel, meta, calendar, limit=None):
    """Five aligned arrays; source-domain facts are separately retained in audit."""
    if not isinstance(panel, dict) or set(panel) != {'adj'}:
        raise ValueError('Panel requires exactly adj')
    adj = panel['adj']
    _frame(adj)
    sessions = _calendar(calendar)
    queries, width = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons, source_counts, available_counts = [], Counter(), [], []
    selected_counts, valid_counts = [], []
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        dates = sessions[t-(PRICE_COUNT-1)-SKIP:t-SKIP+1] if t >= PRICE_COUNT-1+SKIP else None
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
            values['candidate'][i, j] = result['candidate']
            values['control'][i, j] = result['control']
            cell = dict(result, symbol=symbol, date=iso, candidate_rank=0., control_rank=0.)
            if not result['available']:
                cell.update(candidate=None, control=None)
            cells.append(cell)
        for arm in ('candidate', 'control'):
            values[arm+'_rank'][i] = centered_rank(values[arm][i], values['available'][i])
        for j, cell in enumerate(cells):
            cell.update(candidate_rank=float(values['candidate_rank'][i, j]),
                        control_rank=float(values['control_rank'][i, j]))
            reasons[cell['reason']] += 1
        source_counts.append(sum(cell['source_available'] for cell in cells))
        available_counts.append(sum(cell['available'] for cell in cells))
        selected_counts.append(sum(cell['selected_anchors'] for cell in cells))
        valid_counts.append(sum(cell['valid_event_payoffs'] for cell in cells))
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            anchor_first_date=str(dates[ANCHOR_FIRST].date()) if dates is not None else None,
            anchor_last_date=str(dates[ANCHOR_LAST].date()) if dates is not None else None,
            missing_official_sessions=missing,
            source_available=[c['source_available'] for c in cells], available=[c['available'] for c in cells],
            source_available_count=source_counts[-1], available_count=available_counts[-1], cells=cells))
    source_count, available = sum(source_counts), int(values['available'].sum())
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, anchor_first=ANCHOR_FIRST, anchor_last=ANCHOR_LAST,
        risk_returns=RISK_RETURNS, payoff_horizon=PAYOFF_HORIZON, cooldown=COOLDOWN, prior_count=PRIOR_COUNT,
        source_available_cells=source_count, source_available_by_event=source_counts,
        available_cells=available, available_by_event=available_counts,
        unavailable_cells=int(values['available'].size)-available,
        reason_counts=dict(reasons), valid_price_observations=PRICE_COUNT*source_count,
        selected_anchors=sum(selected_counts), pre_event_return_observations=RISK_RETURNS*sum(selected_counts),
        matured_payoffs=sum(selected_counts), valid_event_payoffs=sum(valid_counts),
        selected_anchors_by_event=selected_counts, valid_event_payoffs_by_event=valid_counts,
        predictive_model_fits=0, new_model_fits=0, common_availability=True,
        ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False, event_history=history)
