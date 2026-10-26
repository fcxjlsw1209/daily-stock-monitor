"""Mature own-sector monthly payoffs conditional on long relative trend.

Pure source statistics only: no I/O, ranking, trading, or fitted prediction.
The query-date sector mapping is fixed throughout each 1471-price history.
"""
import math

import numpy as np
import pandas as pd

PRICE_COUNT = 1471
SKIP = 42
STATE_RETURNS = 210
LABEL_SESSIONS = 21
LABEL_COUNT = 60
PRIOR_MASS = 20
MINIMUM_MATCHES = 10


def _empty(reason):
    return dict(available=False, reason=reason, candidate=float('nan'), control=float('nan'),
        state_start_indices=[], predictor_indices=[], label_end_indices=[], states=[], labels=[],
        query_start_index=None, query_end_index=None, query_state=None,
        matching_indices=[], matching_labels=[], matching_count=0, unconditional_mean=None,
        matching_sum=None, matching_mean=None, shrunken_payoff=None, all_states_match=False,
        conditional_mean_estimates=0, pooled_mean_estimates=0,
        training_label_observations=0, matched_label_observations=0)


def _finite_sum(values):
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite sector-relative sum') from exc
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite sector-relative sum')
    return value


def _relative_sign(stock_end, stock_start, benchmark_end, benchmark_start):
    """Compare exact rational ratios, without floating products or division."""
    parts = []
    for value in (stock_end, stock_start, benchmark_end, benchmark_start):
        number = float(value)
        if not math.isfinite(number) or number <= 0.:
            raise ValueError('Relative-state endpoints must be finite positive prices')
        parts.append(number.as_integer_ratio())
    (sn, sd), (on, od), (bn, bd), (pn, pd_) = parts
    # S_end * B_start versus B_end * S_start, cross-multiplied as integers.
    left = sn * pn * bd * od
    right = bn * on * sd * pd_
    return 1 if left > right else -1 if left < right else 0


def _log_ratio(end, start):
    """Stable logarithm of a positive finite ratio, including extreme prices."""
    end, start = float(end), float(start)
    ratio = end / start
    if .5 <= ratio <= 2.:
        value = math.log1p((end-start)/start)
    elif math.isfinite(ratio) and ratio > 0.:
        value = math.log(ratio)
    else:
        em, ee = math.frexp(end)
        sm, se = math.frexp(start)
        value = math.log(em/sm) + (ee-se)*math.log(2.)
    if not math.isfinite(value):
        raise FloatingPointError('Positive finite prices produced a nonfinite log ratio')
    return value


def measure(stock, benchmark):
    """Measure 60 fully matured labels and their causal 210-session states."""
    if np.shape(stock) != (PRICE_COUNT,) or np.shape(benchmark) != (PRICE_COUNT,):
        raise ValueError('Expected exactly 1471 stock and benchmark prices')
    prices = []
    for name, source in (('stock', stock), ('benchmark', benchmark)):
        try:
            converted = np.asarray(source, dtype=float)
        except (ValueError, TypeError, OverflowError):
            return _empty('nonnumeric_' + name + '_price')
        if not (np.isfinite(converted) & (converted > 0.)).all():
            return _empty('invalid_' + name + '_price')
        prices.append(converted)
    stock, benchmark = prices
    starts = list(range(0, LABEL_COUNT*LABEL_SESSIONS, LABEL_SESSIONS))
    predictors = [s + STATE_RETURNS for s in starts]
    ends = [a + LABEL_SESSIONS for a in predictors]
    states, labels = [], []
    for s, a, b in zip(starts, predictors, ends):
        states.append(_relative_sign(stock[a], stock[s], benchmark[a], benchmark[s]))
        label = _log_ratio(stock[b], stock[a]) - _log_ratio(benchmark[b], benchmark[a])
        if not math.isfinite(label):
            raise FloatingPointError('Nonfinite relative monthly label')
        labels.append(label)
    query = _relative_sign(stock[1470], stock[1260], benchmark[1470], benchmark[1260])
    matches = [i for i, state in enumerate(states) if state == query]
    matching = [labels[i] for i in matches]
    count = len(matches)
    mean = _finite_sum(labels)/LABEL_COUNT
    total = _finite_sum(matching)
    result = _empty('insufficient_matches')
    result.update(state_start_indices=starts, predictor_indices=predictors, label_end_indices=ends,
        states=states, labels=labels, query_start_index=1260, query_end_index=1470, query_state=query,
        matching_indices=matches, matching_labels=matching, matching_count=count,
        unconditional_mean=mean, matching_sum=total, matching_mean=total/count if count else None,
        all_states_match=count == LABEL_COUNT)
    if count < MINIMUM_MATCHES:
        return result
    candidate = mean if count == LABEL_COUNT else _finite_sum([total, PRIOR_MASS*mean])/(count+PRIOR_MASS)
    if not math.isfinite(candidate):
        raise FloatingPointError('Finite labels produced a nonfinite shrunken payoff')
    result.update(available=True, reason='available', candidate=candidate, control=mean,
        shrunken_payoff=candidate, conditional_mean_estimates=1, pooled_mean_estimates=1,
        training_label_observations=LABEL_COUNT, matched_label_observations=count)
    return result


def _day(value, label):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(label + ' must be a naive midnight date')
    return day


def _frame(frame, label):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(label + ' requires DataFrame with DatetimeIndex')
    dates = frame.index
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz is not None or not dates.equals(dates.normalize())):
        raise ValueError(label + ' sessions must be unique increasing naive midnight dates')
    if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s for s in frame.columns)):
        raise ValueError(label + ' columns must be unique nonempty security IDs')


def _calendar(calendar):
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
    if len(meta['dates']) != len(meta['symbols']):
        raise ValueError('Metadata dates and symbols must align')
    if isinstance(meta['etfs'], (str, bytes)):
        raise ValueError('ETF identities require an explicit sequence')
    etfs = list(meta['etfs'])
    if len(etfs) != len(set(etfs)) or any(not isinstance(s, str) or not s or s.strip() != s for s in etfs):
        raise ValueError('ETF identities must be unique nonempty strings')
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, supplied in zip(meta['dates'], meta['symbols']):
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(supplied, (str, bytes)):
            raise ValueError('Event symbols require an explicit sequence')
        symbols = list(supplied)
        if (not symbols or len(symbols) != len(set(symbols))
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must be unique nonempty strings')
        if width is not None and len(symbols) != width:
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, lookup[day]))
    return rows, width or 0, etfs


def build_features(panel, meta, calendar, benchmark_map, limit=None):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'sectors'}:
        raise ValueError('Panel requires exactly adj and sectors')
    for key in panel:
        _frame(panel[key], key)
    if not callable(benchmark_map):
        raise TypeError('The frozen benchmark mapper must be callable')
    sessions = _calendar(calendar)
    queries, width, etfs = _queries(meta, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool))
    history, reasons = [], {}
    adj, sectors = panel['adj'], panel['sectors']
    for i, (day, symbols, t) in enumerate(queries):
        iso = str(day.date())
        mapping = benchmark_map(iso, symbols, sectors.loc[:day], etfs)
        if (not isinstance(mapping, dict) or set(mapping) != set(symbols)
                or any(b is not None and (not isinstance(b, str) or not b or b.strip() != b)
                       for b in mapping.values())):
            raise ValueError('Frozen mapping must contain exactly the event identities')
        if any(mapping[s] is not None for s in symbols if s in etfs):
            raise ValueError('Frozen mapping must keep candidate ETFs neutral')
        dates = sessions[t-1512:t-SKIP+1] if t >= 1512 else None
        missing, windows = [], None
        if dates is not None:
            missing = dates.difference(adj.index).strftime('%Y-%m-%d').tolist()
            date = lambda index: str(dates[index].date())
            windows = dict(formation=dict(price_start=date(0), price_end=date(1470), return_count=1470),
                query_state=dict(price_start=date(1260), price_end=date(1470), return_count=210),
                training_labels=dict(price_start=date(210), price_end=date(1470), return_count=1260,
                                     block_sessions=21, block_count=60))
        cells = []
        for j, symbol in enumerate(symbols):
            benchmark = mapping[symbol]
            if symbol in etfs:
                cell = _empty('etf_neutral')
            elif benchmark is None:
                cell = _empty('missing_benchmark')
            elif dates is None:
                cell = _empty('insufficient_history')
            elif symbol not in adj.columns:
                cell = _empty('missing_symbol')
            elif benchmark not in adj.columns:
                cell = _empty('missing_benchmark_symbol')
            elif missing:
                cell = _empty('missing_official_session')
            else:
                cell = measure(adj.loc[dates, symbol].to_numpy(copy=True),
                               adj.loc[dates, benchmark].to_numpy(copy=True))
            for key in values:
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0)+1
            cell = dict(cell, symbol=symbol, benchmark=benchmark, date=iso, windows=windows)
            cell['state_start_dates'] = [str(dates[a].date()) for a in cell['state_start_indices']]
            cell['predictor_dates'] = [str(dates[a].date()) for a in cell['predictor_indices']]
            cell['label_end_dates'] = [str(dates[a].date()) for a in cell['label_end_indices']]
            cell['matching_predictor_dates'] = [cell['predictor_dates'][a] for a in cell['matching_indices']]
            cell['matching_label_end_dates'] = [cell['label_end_dates'][a] for a in cell['matching_indices']]
            if not cell['available']:
                cell['candidate'] = cell['control'] = None
            cells.append(cell)
        history.append(dict(date=iso, symbols=symbols, session_position=t,
            price_start=str(dates[0].date()) if dates is not None else None,
            price_end=str(dates[-1].date()) if dates is not None else None,
            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    matched = sum(cell['matched_label_observations'] for event in history for cell in event['cells'])
    return values, dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=PRICE_COUNT, skip_sessions=SKIP, return_count=1470, state_return_count=STATE_RETURNS,
        label_sessions=LABEL_SESSIONS, training_label_count=LABEL_COUNT, prior_mass=PRIOR_MASS,
        minimum_matches=MINIMUM_MATCHES, available_cells=available,
        unavailable_cells=int(values['available'].size)-available, reason_counts=reasons,
        conditional_mean_estimates=available, pooled_mean_estimates=available,
        training_label_observations=LABEL_COUNT*available, matched_label_observations=matched,
        predictive_model_fits=0, common_availability=True, portfolio_returns_calculated=False,
        ranking_performed=False, trading_eligibility_modified=False, event_history=history)
