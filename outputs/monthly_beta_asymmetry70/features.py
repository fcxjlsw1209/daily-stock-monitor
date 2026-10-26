"""Fixed separate-intercept market-up minus market-down historical beta.

Pure feature preparation: no fitting of future targets, eligibility, state,
portfolio decisions, file I/O or model updates. Zero market returns are excluded.
"""
import numpy as np
import pandas as pd

PRICE_COUNT = 253
RETURN_COUNT = 252
FIRST_OFFSET = -294
LAST_OFFSET = -42
MIN_GROUP = 40
MARKET = 'SPY'


def _ids(values, name, count=None):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(name+' requires an explicit collection of IDs')
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError(name+' requires an explicit collection of IDs') from exc
    if any(not isinstance(s, (str, np.str_)) or not s or s.strip() != s for s in values):
        raise ValueError(name+' requires nonempty security IDs')
    if len(set(values)) != len(values) or (count is not None and len(values) != count):
        raise ValueError(name+' has duplicate IDs or wrong candidate count')
    return values


def _day(value):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Event date must be a valid daily session') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Event dates must be timezone-naive normalized daily sessions')
    return day


def _log_returns(prices):
    """Exactly252 log returns; frexp avoids price-quotient overflow/underflow.

    Algebraically log(P[d])-log(P[d-1]); common binary price-unit changes
    cancel exactly without subtracting two large log-price levels.
    """
    p = np.asarray(prices, float)
    if p.shape != (PRICE_COUNT,):
        raise ValueError('Expected exactly253 adjusted prices')
    if not (np.isfinite(p) & (p > 0.)).all():
        raise ValueError('Prices must be positive finite')
    mantissa, exponent = np.frexp(p)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        r = np.log(mantissa[1:]/mantissa[:-1])+np.diff(exponent)*np.log(2.)
    if not np.isfinite(r).all():
        raise FloatingPointError('Nonfinite computed log returns')
    return r


def _center(values):
    """Scale then separately center a group, retaining its original scale."""
    scale = float(np.abs(values).max())
    if scale == 0.:
        return np.zeros(len(values)), 1.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        z = values/scale
        z = z-z[0]
        z -= z.mean()
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite group centering')
    return z, scale


def _slope(market, security):
    """OLS slope WITH a separate group intercept, or NaN for constant market."""
    x, y = np.asarray(market, float), np.asarray(security, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError('Slope needs matching nontrivial one-dimensional groups')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError('Nonfinite slope input')
    xc, xs = _center(x)
    yc, ys = _center(y)
    # Rescale centered spreads so a valid tiny within-group variance is not
    # mistaken for zero by squared-sum underflow. No epsilon threshold.
    spread_x, spread_y = float(np.abs(xc).max()), float(np.abs(yc).max())
    if spread_x == 0.:
        return np.nan
    if spread_y == 0.:
        return 0.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        xx, yy = xc/spread_x, yc/spread_y
        coefficient = float(np.dot(xx, yy)/np.dot(xx, xx))
        # Multiply positive scales via mantissas/exponents. This avoids an
        # overflowing intermediate ratio when the final slope is representable.
        cm, ce = np.frexp(coefficient)
        ym, ye = np.frexp(ys); xm, xe = np.frexp(xs)
        dm, de = np.frexp(spread_y); bm, be = np.frexp(spread_x)
        slope = float(np.ldexp((cm*ym*dm)/(xm*bm), int(ce)+int(ye)+int(de)-int(xe)-int(be)))
    if not np.isfinite(slope):
        raise FloatingPointError('Nonfinite computed beta')
    return slope


def _beta_gap(security, market):
    """Return gap, up beta, down beta, up/down/zero counts and status."""
    y, x = np.asarray(security, float), np.asarray(market, float)
    if y.shape != (RETURN_COUNT,) or x.shape != y.shape:
        raise ValueError('Expected252 security and market returns')
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite computed historical returns')
    up, down = x > 0., x < 0.
    counts = (int(up.sum()), int(down.sum()), int((x == 0.).sum()))
    if min(counts[:2]) < MIN_GROUP:
        return np.nan, np.nan, np.nan, counts, 'insufficient_group'
    bu, bd = _slope(x[up], y[up]), _slope(x[down], y[down])
    if np.isnan(bu) or np.isnan(bd):
        return np.nan, bu, bd, counts, 'zero_market_variance'
    with np.errstate(over='raise', invalid='raise'):
        gap = bu-bd
    if not np.isfinite(gap):
        raise FloatingPointError('Nonfinite computed beta difference')
    return float(gap), bu, bd, counts, 'available'


def prepare(panel, events, etfs):
    """Return raw_beta_gap/rank_beta_gap/available [events,49] plus audit."""
    etfs = _ids(etfs, 'etfs', 17)
    etf_set = set(etfs)
    try:
        px = panel['adj']
    except (KeyError, TypeError) as exc:
        raise ValueError('panel must provide an adj DataFrame') from exc
    if not isinstance(px, pd.DataFrame) or not isinstance(px.index, pd.DatetimeIndex):
        raise ValueError('adj must be a DataFrame with a daily DatetimeIndex')
    if (px.index.hasnans or not px.index.is_unique or not px.index.is_monotonic_increasing or
            px.index.tz is not None or not px.index.equals(px.index.normalize())):
        raise ValueError('Price sessions must be valid, unique, increasing and normalized')
    _ids(px.columns, 'adj columns')
    normalized = []
    try:
        iterator = iter(events)
    except TypeError as exc:
        raise ValueError('events must be date/candidate pairs') from exc
    for item in iterator:
        if isinstance(item, (str, bytes)):
            raise ValueError('Each event must be a date/candidate pair')
        try:
            day, symbols = item
        except (TypeError, ValueError) as exc:
            raise ValueError('Each event must be a date/candidate pair') from exc
        day, symbols = _day(day), _ids(symbols, 'event candidates', 49)
        if not etf_set.issubset(symbols) or len(set(symbols)-etf_set) != 32:
            raise ValueError('Each event requires fixed17 ETFs plus32 distinct stocks')
        if day not in px.index or not set(symbols).issubset(px.columns):
            raise ValueError('Event session or candidate absent from price axes')
        if normalized and day <= normalized[-1][0]:
            raise ValueError('Event dates must be unique and strictly increasing')
        normalized.append((day, symbols))
    shape = (len(normalized), 49)
    arrays = dict(raw_beta_gap=np.full(shape, np.nan), rank_beta_gap=np.full(shape, np.nan),
                  available=np.zeros(shape, bool))
    reasons = ('available', 'insufficient_history', 'invalid_market_path', 'invalid_security_path',
               'insufficient_group', 'zero_market_variance')
    audit = dict(events=len(normalized), candidate_count=49, stock_count=32, etf_count=17,
        etfs=sorted(etfs), market=MARKET, price_count=PRICE_COUNT, return_count=RETURN_COUNT,
        first_price_offset=FIRST_OFFSET, last_price_offset=LAST_OFFSET, min_group=MIN_GROUP,
        formula='Cov(S,M|M>0)/Var(M|M>0)-Cov(S,M|M<0)/Var(M|M<0); separate group centering',
        returns='log(mantissa[d]/mantissa[d-1])+exponent_difference*log(2)',
        zero_market_returns='excluded from both groups', zero_market_variance='unavailable; no epsilon',
        valid_zero='constant security and market self if market groups valid',
        ranking='all49 finite candidates; average percentile minus0.5; missing stays NaN',
        eligibility_read=False, state_read=False, recent_42_prices_read=False,
        future_labels_used=False, beta_clipped=False, beta_shrinkage=False,
        **{reason+'_cells':0 for reason in reasons}, months=[])
    for i, (day, symbols) in enumerate(normalized):
        t = int(px.index.get_loc(day)); start, end = t+FIRST_OFFSET, t+LAST_OFFSET
        detail = dict(date=day.strftime('%Y-%m-%d'), first_price_date=None,
            first_return_date=None, last_price_date=None, up_count=None, down_count=None, zero_count=None,
            beta_up=[None]*49, beta_down=[None]*49, status=[None]*49)
        if start < 0:
            detail['status'] = ['insufficient_history']*49
        elif MARKET not in px.columns:
            detail['status'] = ['invalid_market_path']*49
        else:
            detail.update(first_price_date=px.index[start].strftime('%Y-%m-%d'),
                first_return_date=px.index[start+1].strftime('%Y-%m-%d'),
                last_price_date=px.index[end].strftime('%Y-%m-%d'))
            # Slice BEFORE float conversion: old and recent/future suffixes are
            # never inspected as numeric values, including market suffixes.
            market_prices = px[MARKET].iloc[start:end+1].to_numpy(dtype=float, na_value=np.nan)
            if not (np.isfinite(market_prices) & (market_prices > 0.)).all():
                detail['status'] = ['invalid_market_path']*49
            else:
                market = _log_returns(market_prices)
                detail.update(up_count=int((market > 0.).sum()), down_count=int((market < 0.).sum()),
                              zero_count=int((market == 0.).sum()))
                for j, symbol in enumerate(symbols):
                    prices = px[symbol].iloc[start:end+1].to_numpy(dtype=float, na_value=np.nan)
                    if not (np.isfinite(prices) & (prices > 0.)).all():
                        detail['status'][j] = 'invalid_security_path'
                        continue
                    raw, bu, bd, _, status = _beta_gap(_log_returns(prices), market)
                    arrays['raw_beta_gap'][i,j] = raw
                    arrays['available'][i,j] = status == 'available'
                    detail['status'][j] = status
                    detail['beta_up'][j] = float(bu) if np.isfinite(bu) else None
                    detail['beta_down'][j] = float(bd) if np.isfinite(bd) else None
        arrays['rank_beta_gap'][i] = (pd.Series(arrays['raw_beta_gap'][i]).rank(method='average', pct=True)-.5).to_numpy()
        for reason in reasons:
            audit[reason+'_cells'] += detail['status'].count(reason)
        audit['months'].append(detail)
    for key in ('raw_beta_gap', 'rank_beta_gap'):
        if not np.array_equal(np.isfinite(arrays[key]), arrays['available']):
            raise FloatingPointError(key+' finite mask differs from availability')
    if sum(audit[key+'_cells'] for key in reasons) != shape[0]*49:
        raise AssertionError('Incomplete feature coverage accounting')
    return arrays, audit
