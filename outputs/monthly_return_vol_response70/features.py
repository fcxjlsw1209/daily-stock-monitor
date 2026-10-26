"""Pure lagged own-return/squared-return response; no ranks, strategy or I/O.

Squared returns are formed before any centering. Lead/reverse raw correlations
retain their separate identifiability, while effective hands share one mask.
"""
import math

import numpy as np
import pandas as pd


PRICE_ROWS, RETURN_ROWS, PAIR_ROWS = 253, 252, 251
UNIVERSE_SIZE, ETF_COUNT = 49, 17
BOUND_TOLERANCE = 1e-12
LOG_TWO = math.log(2.)
FLOAT_KEYS = ('raw_lead', 'raw_reverse', 'effect_lead', 'effect_reverse')


def _finite(value, name):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite computed ' + name)
    return value


def _vector(values, count, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (count,):
        raise ValueError(name + ' requires exactly%d observations' % count)
    return [_finite(float(x), name) for x in values]


def _checked_correlation(value):
    value = _finite(value, 'correlation')
    if abs(value) > 1. + BOUND_TOLERANCE:
        raise FloatingPointError('Computed correlation outside [-1,1]')
    return min(1., max(-1., value))


def _pearson_center(values):
    """Apply the registered scale/center/scale sequence to a nonconstant leg."""
    scale = _finite(max(abs(x) for x in values), 'Pearson input scale')
    if scale == 0.:
        raise FloatingPointError('Unexpected zero Pearson input scale')
    normalized = [_finite(x / scale, 'Pearson normalized input') for x in values]
    mean = _finite(math.fsum(normalized) / PAIR_ROWS, 'Pearson input mean')
    centered = [_finite(x - mean, 'Pearson centered input') for x in normalized]
    centered_scale = _finite(max(abs(x) for x in centered), 'Pearson centered scale')
    if centered_scale == 0.:
        raise FloatingPointError('Unexpected zero Pearson centered scale')
    return [_finite(x / centered_scale, 'Pearson rescaled input') for x in centered]


def pearson(left, right):
    """Registered scalar Pearson of two251 vectors, or None if exactly constant.

    Undefined means exact input equality only. All arithmetic failures propagate
    and must never be relabeled as an unavailable market observation.
    """
    left = _vector(left, PAIR_ROWS, 'Pearson left input')
    right = _vector(right, PAIR_ROWS, 'Pearson right input')
    if (all(x == left[0] for x in left) or
            all(y == right[0] for y in right)):
        return None
    try:
        x, y = _pearson_center(left), _pearson_center(right)
        xx = _finite(math.fsum(_finite(v * v, 'Pearson left square') for v in x),
                     'Pearson left squared sum')
        yy = _finite(math.fsum(_finite(v * v, 'Pearson right square') for v in y),
                     'Pearson right squared sum')
        if xx <= 0. or yy <= 0.:
            raise FloatingPointError('Unexpected zero or negative Pearson variance')
        cross = _finite(math.fsum(_finite(a * b, 'Pearson cross product')
                                 for a, b in zip(x, y)), 'Pearson cross sum')
        denominator = _finite(math.sqrt(_finite(xx * yy, 'Pearson variance product')),
                              'Pearson denominator')
        if denominator <= 0.:
            raise FloatingPointError('Unexpected zero Pearson denominator')
        return _checked_correlation(cross / denominator)
    except OverflowError as exc:
        raise FloatingPointError('Overflow in computed Pearson correlation') from exc


def _detail():
    return dict(price_rows=PRICE_ROWS, return_rows=RETURN_ROWS, pair_rows=PAIR_ROWS,
                rho_lead=None, rho_reverse=None, effect_lead=None,
                effect_reverse=None, available=False, reason='insufficient_history')


def response_from_returns(returns):
    """Return JSON-safe response details for exactly252 finite log returns.

    This pure helper also exposes exact constant-leg synthetic fixtures without
    requiring an exponential round trip through synthetic prices.
    """
    returns = _vector(returns, RETURN_ROWS, 'log return')
    detail = _detail()
    detail['reason'] = 'lead_and_reverse_undefined'
    scale = max(abs(x) for x in returns)
    if scale == 0.:
        return detail
    try:
        normalized = [_finite(x / scale, 'normalized log return') for x in returns]
        if any(r != 0. and z == 0. for r, z in zip(returns, normalized)):
            raise FloatingPointError('Underflow in normalized nonzero log return')
        squared = [_finite(z * z, 'uncentered squared return') for z in normalized]
        if any(z != 0. and q == 0. for z, q in zip(normalized, squared)):
            raise FloatingPointError('Underflow in nonzero squared return')
        lead = pearson(normalized[:-1], squared[1:])
        reverse = pearson(normalized[1:], squared[:-1])
        detail.update(rho_lead=lead, rho_reverse=reverse)
        if lead is None:
            detail['reason'] = 'lead_and_reverse_undefined' if reverse is None else 'lead_undefined'
        elif reverse is None:
            detail['reason'] = 'reverse_undefined'
        else:
            detail.update(effect_lead=_finite(.5 * lead, 'lead effect'),
                          effect_reverse=_finite(.5 * reverse, 'reverse effect'),
                          available=True, reason='available')
    except OverflowError as exc:
        raise FloatingPointError('Overflow in computed return response') from exc
    return detail


def _log_returns(prices):
    """Registered frexp log ratio; both price-ratio overflow and log subtraction avoided."""
    parts = [math.frexp(float(p)) for p in prices]
    result = []
    try:
        for (previous_m, previous_e), (m, e) in zip(parts, parts[1:]):
            logarithm = _finite(math.log(m / previous_m), 'mantissa log ratio')
            exponent = _finite((e - previous_e) * LOG_TWO, 'exponent log difference')
            result.append(_finite(logarithm + exponent, 'log return'))
    except OverflowError as exc:
        raise FloatingPointError('Overflow in computed log return') from exc
    return result


def response_from_prices(prices):
    """Return JSON-safe details for one exact253-price window."""
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (PRICE_ROWS,):
        raise ValueError('Exactly253 adjusted closing prices required')
    if not (np.isfinite(prices) & (prices > 0.)).all():
        detail = _detail()
        detail['reason'] = 'invalid_adjusted_price'
        return detail
    return response_from_returns(_log_returns(prices))


def _ids(values, name, count=None):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(name + ' must be an explicit ID sequence')
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(name + ' must be an explicit ID sequence') from exc
    if count is not None and len(values) != count:
        raise ValueError(name + ' has wrong size')
    if (any(not isinstance(s, str) or not s or s.strip() != s for s in values)
            or len(set(values)) != len(values)):
        raise ValueError(name + ' has invalid or duplicate IDs')
    return values


def _sessions(index, name):
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
            or not index.equals(index.normalize())):
        raise ValueError(name + ' requires increasing unique midnight timezone-naive dates')
    return index


def build_features(adj, signal_dates, symbols_by_date, etfs, *, expected_sessions=None):
    """Compute only each signal's inclusive[t-252,t] adjusted-close window.

    The caller supplies and pins the official session calendar. When supplied,
    expected_sessions must equal the entire input date axis through the final
    signal. Future price values and unrequested columns are never converted.
    """
    if not isinstance(adj, pd.DataFrame):
        raise ValueError('Adjusted prices must be a DataFrame')
    sessions = _sessions(adj.index, 'adj.index')
    columns = _ids(adj.columns, 'adj.columns')
    dates = _sessions(pd.DatetimeIndex(signal_dates), 'signal_dates')
    if not len(dates):
        raise ValueError('At least one signal required')
    if expected_sessions is not None:
        expected = _sessions(pd.DatetimeIndex(expected_sessions), 'expected_sessions')
        if not sessions[sessions <= dates[-1]].equals(expected):
            raise ValueError('Incomplete official session calendar')
    universes = tuple(_ids(row, 'signal universe', UNIVERSE_SIZE) for row in symbols_by_date)
    if len(universes) != len(dates):
        raise ValueError('Universe and signal counts differ')
    etfs = frozenset(_ids(etfs, 'etfs', ETF_COUNT))
    positions = sessions.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError('Every signal must be an input session')
    for universe in universes:
        if not etfs.issubset(universe) or set(universe) - set(columns):
            raise ValueError('Universe ETFs or required price columns missing')
    shape = (len(dates), UNIVERSE_SIZE)
    arrays = {key: np.full(shape, np.nan, dtype=np.float64) for key in FLOAT_KEYS}
    arrays['response_available'] = np.zeros(shape, dtype=bool)
    audits = []
    for i, (day, t, universe) in enumerate(zip(dates, positions, universes)):
        rows = []
        for j, symbol in enumerate(universe):
            detail = _detail()
            detail.update(signal_date=day.date().isoformat(), query_symbol=symbol,
                          is_etf=symbol in etfs, price_start=None, max_price_date=None)
            if t >= RETURN_ROWS:
                prices = adj[symbol].iloc[t - RETURN_ROWS:t + 1].to_numpy(dtype=float, copy=True)
                detail.update(response_from_prices(prices),
                              price_start=sessions[t - RETURN_ROWS].date().isoformat(),
                              max_price_date=day.date().isoformat())
            for key, field in (('raw_lead', 'rho_lead'), ('raw_reverse', 'rho_reverse'),
                               ('effect_lead', 'effect_lead'), ('effect_reverse', 'effect_reverse')):
                if detail[field] is not None:
                    arrays[key][i, j] = detail[field]
            arrays['response_available'][i, j] = detail['available']
            rows.append(detail)
        audits.append(rows)
    return dict(arrays, dates=tuple(d.date().isoformat() for d in dates),
                symbols=universes, audit=audits)
