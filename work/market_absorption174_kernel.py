"""Pure market-adjusted historical shock absorption; no I/O or portfolios.

Sources are the caller's frozen price vintage. Stored close times volume is
activity, not observed investor flows or certified historical share turnover.
The label is a matured historical price change, not a forecast or trade fill.
"""
import math

import numpy as np
import pandas as pd


SKIP = 42
HISTORY = 1260
HORIZON = 21
COOLDOWN = 21
BASELINE = 20
PRIOR_MASS = 20
MIN_GROUP = 3
BETA_WINDOW = 252
ADJUSTED_COUNT = HISTORY + BETA_WINDOW + 1
LAST_ANCHOR = HISTORY - HORIZON
ACTIVITY_COUNT = LAST_ANCHOR + BASELINE
ADJUSTED_START_OFFSET = HISTORY + BETA_WINDOW + SKIP
ACTIVITY_START_OFFSET = HISTORY + SKIP + BASELINE - 1
ACTIVITY_END_OFFSET = SKIP + HORIZON


def _empty(reason):
    return dict(candidate=float('nan'), control=float('nan'), available=False,
                reason=reason, selected_anchors=[], high_anchors=[], low_anchors=[],
                invalid_activity_anchors=[], n_all=0, n_high=0, n_low=0,
                labels=[], raw_labels=[], classifications=[], market_labels=[], betas=[],
                beta_reasons=[], market_variances=[], residual_labels=[],
                extended_anchor_indices=[], beta_first_indices=[], beta_last_indices=[],
                invalid_beta_anchors=[], beta_attempts_count=0, beta_estimates_count=0)


def _activity_units(close, volume):
    """Exact dyadic products in one integer unit, with no float overflow.

    Every finite IEEE float is a rational with a power-of-two denominator.
    Common integer scaling preserves strict comparison to the arithmetic
    previous-20 mean, even at equality or a one-ULP difference.
    """
    products = []
    greatest_denominator_power = 0
    for c, v in zip(close, volume):
        cn, cd = float(c).as_integer_ratio()
        vn, vd = float(v).as_integer_ratio()
        power = cd.bit_length() + vd.bit_length() - 2
        products.append((cn * vn, power))
        greatest_denominator_power = max(greatest_denominator_power, power)
    return [numerator << (greatest_denominator_power - power)
            for numerator, power in products]


def _log_ratio(end, start):
    end, start = float(end), float(start)
    ratio = end / start
    if .5 <= ratio <= 2.:
        return math.log1p((end - start) / start)
    if math.isfinite(ratio) and ratio > 0.:
        return math.log(ratio)
    em, ee = math.frexp(end)
    sm, se = math.frexp(start)
    return math.log(em / sm) + (ee - se) * math.log(2.)


def _beta_from_returns(stock, market):
    """Centered OLS slope on exactly 252 *pre-anchor* log returns.

    Subtracting the first value before centering preserves exact degeneracy
    for constant return arrays. There is no variance floor, cap or shrinkage.
    """
    if len(stock) != BETA_WINDOW or len(market) != BETA_WINDOW:
        raise ValueError('Beta requires exactly 252 returns')
    if not all(math.isfinite(v) for v in (*stock, *market)):
        return None, 'nonfinite_beta_input', None
    x0, y0 = float(market[0]), float(stock[0])
    x, y = [float(v)-x0 for v in market], [float(v)-y0 for v in stock]
    xm, ym = math.fsum(x)/BETA_WINDOW, math.fsum(y)/BETA_WINDOW
    x, y = [v-xm for v in x], [v-ym for v in y]
    variance = math.fsum(v*v for v in x)
    if not math.isfinite(variance) or variance <= 0.:
        return None, 'zero_or_nonfinite_market_variance', variance if math.isfinite(variance) else None
    covariance = math.fsum(a*b for a, b in zip(x, y))
    beta = covariance/variance
    if not math.isfinite(beta):
        return None, 'nonfinite_beta', variance
    return beta, 'available', variance


def measure(adjusted, market, close, volume):
    """Measure 1513 stock/SPY prices and 1259 activity observations.

    Price index 0 is t-1554; activity index 0 is t-1321. Original173 anchor
    s uses extended price index a=s+252, activity index s+19 and label a+21.
    Its beta uses price[a-253:a], ending strictly before the anchor day.
    Sampling uses only negative price days and a common 21-session cooldown;
    activity and label values cannot move or remove a sampled anchor.
    """
    arrays = []
    for value, label in ((adjusted, 'adjusted_price'), (market, 'market_price'), (close, 'stored_close'),
                         (volume, 'stored_volume')):
        try:
            arrays.append(np.asarray(value, dtype=float))
        except (TypeError, ValueError, OverflowError):
            return _empty('nonnumeric_' + label)
    adjusted, market, close, volume = arrays
    if (adjusted.shape != (ADJUSTED_COUNT,) or market.shape != (ADJUSTED_COUNT,) or close.shape != (ACTIVITY_COUNT,)
            or volume.shape != (ACTIVITY_COUNT,)):
        raise ValueError('Expected exactly 1513 stock/SPY prices and 1259 activity rows')
    for value, label, positive in ((adjusted, 'adjusted_price', True),
                                   (market, 'market_price', True),
                                   (close, 'stored_close', True),
                                   (volume, 'stored_volume', False)):
        valid = np.isfinite(value) & ((value > 0.) if positive else (value >= 0.))
        if not valid.all():
            return _empty('invalid_' + label)
    result = _empty('insufficient_groups')
    next_allowed = 1
    for s in range(1, LAST_ANCHOR + 1):
        a = s + BETA_WINDOW
        if s >= next_allowed and adjusted[a] < adjusted[a-1]:
            result['selected_anchors'].append(s)
            next_allowed = s + COOLDOWN
    units = _activity_units(close, volume)
    sums = [0]
    for unit in units:
        sums.append(sums[-1] + unit)
    stock_returns = [_log_ratio(b, a) for a, b in zip(adjusted[:-1], adjusted[1:])]
    market_returns = [_log_ratio(b, a) for a, b in zip(market[:-1], market[1:])]
    high_labels, low_labels, high_residuals, low_residuals = [], [], [], []
    for s in result['selected_anchors']:
        a = s + BETA_WINDOW
        label = _log_ratio(adjusted[a + HORIZON], adjusted[a])
        market_label = _log_ratio(market[a + HORIZON], market[a])
        if not math.isfinite(label):
            raise ArithmeticError('Positive finite prices produced a nonfinite log label')
        result['labels'].append(label)
        result['raw_labels'].append(label)
        result['market_labels'].append(market_label)
        beta, beta_reason, variance = _beta_from_returns(
            stock_returns[a-BETA_WINDOW-1:a-1], market_returns[a-BETA_WINDOW-1:a-1])
        residual = label-beta*market_label if beta is not None else None
        if residual is not None and not math.isfinite(residual):
            residual, beta_reason = None, 'nonfinite_residual'
        result['betas'].append(beta)
        result['beta_reasons'].append(beta_reason)
        result['market_variances'].append(variance)
        result['residual_labels'].append(residual)
        if beta_reason != 'available':
            result['invalid_beta_anchors'].append(s)
        result['extended_anchor_indices'].append(a)
        result['beta_first_indices'].append(a-BETA_WINDOW-1)
        result['beta_last_indices'].append(a-1)
        j = s + BASELINE - 1
        prior = sums[j] - sums[j - BASELINE]
        if prior == 0:
            result['classifications'].append(None)
            result['invalid_activity_anchors'].append(s)
        elif BASELINE * units[j] > prior:
            result['classifications'].append('H')
            result['high_anchors'].append(s)
            high_labels.append(label)
            high_residuals.append(residual)
        else:
            result['classifications'].append('L')
            result['low_anchors'].append(s)
            low_labels.append(label)
            low_residuals.append(residual)
    result.update(n_all=len(result['selected_anchors']),
                  n_high=len(high_labels), n_low=len(low_labels),
                  beta_attempts_count=len(result['selected_anchors']),
                  beta_estimates_count=sum(b is not None for b in result['betas']))
    if result['invalid_activity_anchors']:
        result['reason'] = 'zero_prior_activity'
        return result
    if any(reason != 'available' for reason in result['beta_reasons']):
        result['reason'] = 'invalid_beta'
        return result
    if min(result['n_high'], result['n_low']) < MIN_GROUP:
        return result
    denominator_h = result['n_high'] + PRIOR_MASS
    denominator_l = result['n_low'] + PRIOR_MASS
    result.update(candidate=math.fsum(high_residuals)/denominator_h-math.fsum(low_residuals)/denominator_l,
                  control=math.fsum(high_labels)/denominator_h-math.fsum(low_labels)/denominator_l,
                  available=True, reason='available')
    return result


def _day(value, name):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a valid date') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError(name + ' must be a timezone-naive midnight date')
    return day


def _panel(panel):
    if not isinstance(panel, dict) or set(panel) != {'adj', 'close', 'volume'}:
        raise ValueError('Panel requires exactly adj, close, volume')
    for frame in panel.values():
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError('Panel values require DataFrames with DatetimeIndex')
        dates = frame.index
        if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
                or dates.tz is not None or not dates.equals(dates.normalize())):
            raise ValueError('Panel sessions must be unique, increasing, naive midnight dates')
        if (not frame.columns.is_unique or any(not isinstance(s, str) or not s or s.strip() != s
                                                for s in frame.columns)):
            raise ValueError('Security columns must be unique nonempty IDs')
        if not dates.equals(panel['adj'].index) or not frame.columns.equals(panel['adj'].columns):
            raise ValueError('Panel axes must be exactly aligned')


def _calendar(calendar):
    dates = []
    for value in calendar:
        day = _day(value, 'Calendar')
        if not isinstance(value, str) or value != str(day.date()) or day.weekday() >= 5:
            raise ValueError('Calendar requires ISO weekday session dates')
        if dates and day <= dates[-1]:
            raise ValueError('Calendar must be unique and increasing')
        dates.append(day)
    if not dates:
        raise ValueError('Calendar cannot be empty')
    return pd.DatetimeIndex(dates)


def _events(events, sessions):
    lookup = {day: i for i, day in enumerate(sessions)}
    rows, width = [], None
    for value, symbols in events:
        day = _day(value, 'Event')
        if day not in lookup or (rows and day <= rows[-1][0]):
            raise ValueError('Events must be unique increasing official sessions')
        if isinstance(symbols, (str, bytes)):
            raise TypeError('Event symbols must be an explicit sequence')
        symbols = list(symbols)
        if (not symbols or len(set(symbols)) != len(symbols)
                or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols)):
            raise ValueError('Event symbols must be unique nonempty IDs')
        if width is not None and width != len(symbols):
            raise ValueError('Event width must be constant')
        width = len(symbols)
        rows.append((day, symbols, lookup[day]))
    return rows, width or 0


def build_features(panel, events, calendar, limit=None):
    """Return values(candidate/control/available) and fully dated source audit.

    The supplied calendar is the host's pinned official calendar. A missing
    source session makes the whole measurement unavailable; reindexing here
    locates absence, never fills it. Numeric conversion follows time slicing.
    """
    _panel(panel)
    sessions = _calendar(calendar)
    queries, width = _events(events, sessions)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(queries)):
        raise ValueError('limit must be an integer event-prefix length')
    queries = queries if limit is None else queries[:limit]
    shape = len(queries), width
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool))
    history, reasons = [], {}
    for i, (day, symbols, t) in enumerate(queries):
        enough = t >= ADJUSTED_START_OFFSET
        adjusted_dates = sessions[t-ADJUSTED_START_OFFSET:t-SKIP+1] if enough else None
        activity_dates = sessions[t-ACTIVITY_START_OFFSET:t-ACTIVITY_END_OFFSET+1] if enough else None
        missing = []
        source, market_source = None, None
        if enough:
            needed = adjusted_dates.union(activity_dates)
            missing = needed.difference(panel['adj'].index).strftime('%Y-%m-%d').tolist()
            source = {key: frame.reindex(index=adjusted_dates if key == 'adj' else activity_dates,
                                         columns=symbols)
                      for key, frame in panel.items()}
            if 'SPY' in panel['adj'].columns:
                market_source = panel['adj'].loc[:, 'SPY'].reindex(adjusted_dates).to_numpy(copy=True)
        cells = []
        for j, symbol in enumerate(symbols):
            if not enough:
                result = _empty('insufficient_history')
            elif symbol not in panel['adj'].columns:
                result = _empty('missing_symbol')
            elif 'SPY' not in panel['adj'].columns:
                result = _empty('missing_market_symbol')
            elif missing:
                result = _empty('missing_official_session')
            else:
                result = measure(source['adj'].iloc[:, j].to_numpy(copy=True),
                                 market_source,
                                 source['close'].iloc[:, j].to_numpy(copy=True),
                                 source['volume'].iloc[:, j].to_numpy(copy=True))
            for key in values:
                values[key][i, j] = result[key]
            reasons[result['reason']] = reasons.get(result['reason'], 0) + 1
            cell = dict(result, symbol=symbol)
            for key in ('candidate', 'control'):
                cell[key] = float(result[key]) if math.isfinite(result[key]) else None
            for source_key, date_key in (('selected_anchors', 'selected_anchor_dates'),
                                         ('high_anchors', 'high_anchor_dates'),
                                         ('low_anchors', 'low_anchor_dates'),
                                         ('invalid_activity_anchors', 'invalid_activity_anchor_dates'),
                                         ('invalid_beta_anchors', 'invalid_beta_anchor_dates')):
                cell[date_key] = [str(adjusted_dates[s+BETA_WINDOW].date()) for s in result[source_key]]
            cell['label_end_dates'] = [str(adjusted_dates[s+BETA_WINDOW+HORIZON].date())
                                       for s in result['selected_anchors']]
            cell['beta_first_dates'] = [str(adjusted_dates[a].date()) for a in result['beta_first_indices']]
            cell['beta_last_dates'] = [str(adjusted_dates[a].date()) for a in result['beta_last_indices']]
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=list(symbols), session_position=t,
                            adjusted_start=str(adjusted_dates[0].date()) if enough else None,
                            adjusted_end=str(adjusted_dates[-1].date()) if enough else None,
                            activity_start=str(activity_dates[0].date()) if enough else None,
                            activity_end=str(activity_dates[-1].date()) if enough else None,
                            missing_official_sessions=missing, cells=cells))
    return values, dict(events=len(queries), symbols_per_event=width,
                        adjusted_count=ADJUSTED_COUNT, activity_count=ACTIVITY_COUNT,
                        history_sessions=HISTORY, beta_window_returns=BETA_WINDOW,
                        market_symbol='SPY', skip_sessions=SKIP, horizon_sessions=HORIZON,
                        cooldown_sessions=COOLDOWN, activity_baseline_sessions=BASELINE,
                        prior_mass=PRIOR_MASS, minimum_group_anchors=MIN_GROUP,
                        available_cells=int(values['available'].sum()),
                        unavailable_cells=int(values['available'].size-values['available'].sum()),
                        reason_counts=reasons, event_history=history, common_availability=True,
                        beta_attempts_count=sum(c['beta_attempts_count'] for r in history for c in r['cells']),
                        beta_estimates_count=sum(c['beta_estimates_count'] for r in history for c in r['cells']),
                        predictive_model_fits=0,
                        activity_threshold_arithmetic='exact_dyadic_products_and_integer_sum',
                        historical_matured_price_outcomes_used=True,
                        portfolio_returns_calculated=False, ranking_performed=False,
                        trading_eligibility_modified=False)
