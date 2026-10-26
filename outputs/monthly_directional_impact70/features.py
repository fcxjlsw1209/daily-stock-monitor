"""Skip42 directional response per stored dollar, with a price-only control.

Pure preparation: no files, labels, fitting, eligibility or portfolio actions.
Stored close/volume are heterogeneous proxies, not certified split-consistent
historical dollar volume. Reciprocal unit algebra does not certify vendors.
Adjusted returns include distributions; this is not observed order flow or
evidence of causal demand.
"""
import numpy as np
import pandas as pd

PRICE_COUNT = 253
RETURN_COUNT = 252
MIN_SIDE_COUNT = 40
FIRST_OFFSET = -294
RAW_FIRST_OFFSET = -293
LAST_OFFSET = -42
RAW_LAST_OFFSET = LAST_OFFSET
FLOAT_KEYS = ('raw_dollar', 'raw_price', 'rank_dollar', 'rank_price')
REASONS = ('available', 'insufficient_history', 'invalid_price_volume',
           'insufficient_sign_counts', 'undefined_denominator')


def _ids(values, name, expected_count=None):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(name+' must be an explicit collection of security IDs')
    try:
        ids = list(values)
    except TypeError as exc:
        raise ValueError(name+' must be an explicit collection of security IDs') from exc
    if any(not isinstance(s, (str, np.str_)) or not s or s.strip() != s for s in ids):
        raise ValueError(name+' requires nonempty security IDs')
    if len(set(ids)) != len(ids) or (expected_count is not None and len(ids) != expected_count):
        raise ValueError(name+' has duplicate IDs or the wrong number of candidates')
    return ids


def _day(value):
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Event date must be a valid daily session') from exc
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Event date must be a timezone-naive normalized daily session')
    return day


def _scaled_dollars(close, volume):
    """C*V with ONE power-of-two scale across all252 rows of a security.

    Multiplying mantissas avoids overflow/underflow of the original product.
    A scaling underflow or loss of a positive observation is an explicit error.
    """
    close, volume = np.asarray(close, float), np.asarray(volume, float)
    if close.shape != (RETURN_COUNT,) or volume.shape != close.shape:
        raise ValueError('Expected252 matching close and volume observations')
    if not (np.isfinite(close) & (close > 0.) & np.isfinite(volume) & (volume > 0.)).all():
        raise ValueError('Dollar-volume inputs must be positive finite')
    cm, ce = np.frexp(close)
    vm, ve = np.frexp(volume)
    dm, de = np.frexp(cm*vm)
    exponent = ce.astype(np.int64)+ve.astype(np.int64)+de.astype(np.int64)
    with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
        dollars = np.ldexp(dm, exponent-exponent.max())
    if not (np.isfinite(dollars) & (dollars > 0.)).all():
        raise FloatingPointError('Dollar-volume common scaling lost a positive finite observation')
    return dollars


def _asymmetry(up, down):
    if not np.isfinite(up) or not np.isfinite(down) or up < 0. or down < 0.:
        raise FloatingPointError('Invalid conditional directional mean')
    with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
        denominator = up+down
        if denominator == 0.:
            return np.nan
        result = float((up-down)/denominator)
    if not np.isfinite(result) or abs(result) > 1.+1e-12:
        raise FloatingPointError('Invalid directional asymmetry')
    return float(np.clip(result, -1., 1.))  # Mathematical-bound roundoff only.


def _directional_pair(returns, dollars):
    """Both ratios under common positive scales; never rescale sign groups."""
    returns, dollars = np.asarray(returns, float), np.asarray(dollars, float)
    if returns.shape != (RETURN_COUNT,) or dollars.shape != returns.shape:
        raise ValueError('Expected252 returns and252 scaled dollar observations')
    if not np.isfinite(returns).all() or not (np.isfinite(dollars) & (dollars > 0.)).all():
        raise FloatingPointError('Nonfinite returns or invalid scaled dollars')
    positive, negative = returns > 0., returns < 0.
    scale = np.max(np.abs(returns))
    if scale == 0. or not positive.any() or not negative.any():
        return np.nan, np.nan
    with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
        normalized = returns/scale
        inverse_dollars = np.min(dollars)/dollars
        weighted = normalized*inverse_dollars
        up, down = float(np.mean(weighted[positive])), float(np.mean(-weighted[negative]))
        up0, down0 = float(np.mean(normalized[positive])), float(np.mean(-normalized[negative]))
    if (not np.isfinite(normalized).all() or not np.isfinite(weighted).all() or
            not (np.isfinite(inverse_dollars) & (inverse_dollars > 0.)).all() or
            np.any((returns != 0.) & ((normalized == 0.) | (weighted == 0.)))):
        raise FloatingPointError('Directional common scaling lost a nonzero observation')
    dollar, price = _asymmetry(up, down), _asymmetry(up0, down0)
    if np.isnan(dollar) or np.isnan(price):
        return np.nan, np.nan
    return dollar, price


def _window_feature(adjusted, close, volume):
    """Compute isolated253 adjusted and252 raw close/volume rows per security."""
    adjusted, close, volume = (np.asarray(value, float) for value in (adjusted, close, volume))
    if (adjusted.ndim != 2 or adjusted.shape[0] != PRICE_COUNT or
            close.shape != (RETURN_COUNT, adjusted.shape[1]) or volume.shape != close.shape):
        raise ValueError('Feature requires253 adjusted and252 matching close/volume rows')
    width = adjusted.shape[1]
    complete = np.ones(width, bool)
    for data in (adjusted, close, volume):
        complete &= np.isfinite(data).all(axis=0) & (data > 0.).all(axis=0)
    dollar, price = np.full(width, np.nan), np.full(width, np.nan)
    n_positive, n_negative = np.zeros(width, np.int64), np.zeros(width, np.int64)
    status = np.full(width, 'invalid_price_volume', dtype=object)
    for j in np.flatnonzero(complete):
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            returns = adjusted[1:, j]/adjusted[:-1, j]-1.
        if not np.isfinite(returns).all():
            raise FloatingPointError('Nonfinite computed simple adjusted return')
        n_positive[j], n_negative[j] = np.count_nonzero(returns > 0.), np.count_nonzero(returns < 0.)
        # Validate common dollar scaling even if the complete path fails counts.
        dollars = _scaled_dollars(close[:, j], volume[:, j])
        if min(n_positive[j], n_negative[j]) < MIN_SIDE_COUNT:
            status[j] = 'insufficient_sign_counts'
            continue
        dollar[j], price[j] = _directional_pair(returns, dollars)
        status[j] = 'available' if np.isfinite(dollar[j]) else 'undefined_denominator'
    return dollar, price, n_positive, n_negative, status


def prepare(panel, events, etfs):
    """Return exactly7 [events,49] arrays and a JSON-safe formula/coverage audit."""
    etfs = _ids(etfs, 'etfs', 17)
    etf_set = set(etfs)
    try:
        frames = {key: panel[key] for key in ('adj', 'close', 'volume')}
    except (KeyError, TypeError) as exc:
        raise ValueError('panel must provide adj, close and volume DataFrames') from exc
    for key, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(key+' needs a DataFrame with a daily DatetimeIndex')
        if (frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing or
                frame.index.tz is not None or not frame.index.equals(frame.index.normalize())):
            raise ValueError('Price sessions must be valid, unique, increasing and normalized')
        _ids(frame.columns, key+' columns')
    px = frames['adj']
    for key in ('close', 'volume'):
        if not frames[key].index.equals(px.index) or not frames[key].columns.equals(px.columns):
            raise ValueError('Adjusted/raw price/volume axes must match exactly')
    normalized = []
    try:
        iterator = iter(events)
    except TypeError as exc:
        raise ValueError('events must be a sequence of date/candidate pairs') from exc
    for item in iterator:
        if isinstance(item, (str, bytes)):
            raise ValueError('Each event must be a date/candidate pair')
        try:
            day, symbols = item
        except (TypeError, ValueError) as exc:
            raise ValueError('Each event must be a date/candidate pair') from exc
        day, symbols = _day(day), _ids(symbols, 'event candidates', 49)
        if not etf_set.issubset(symbols) or len(set(symbols)-etf_set) != 32:
            raise ValueError('Each event requires the fixed17 ETFs plus32 distinct stocks')
        if day not in px.index or not set(symbols).issubset(px.columns):
            raise ValueError('Event session or candidate missing from price axes')
        if normalized and day <= normalized[-1][0]:
            raise ValueError('Event dates must be unique and strictly increasing')
        normalized.append((day, symbols))
    shape = (len(normalized), 49)
    arrays = {key: np.full(shape, np.nan) for key in FLOAT_KEYS}
    arrays.update(impact_available=np.zeros(shape, bool), n_positive=np.zeros(shape, np.int64),
                  n_negative=np.zeros(shape, np.int64))
    audit = dict(events=len(normalized), candidate_count=49, stock_count=32, etf_count=17,
        etfs=sorted(etfs), price_count=PRICE_COUNT, return_count=RETURN_COUNT,
        raw_count=RETURN_COUNT, min_side_count=MIN_SIDE_COUNT,
        first_offset=FIRST_OFFSET, last_offset=LAST_OFFSET,
        raw_first_offset=RAW_FIRST_OFFSET, raw_last_offset=RAW_LAST_OFFSET,
        returns='simple adjusted return A[1:]/A[:-1]-1; adjusted distributions included',
        formula='U=mean(r/D | r>0); L=mean(-r/D | r<0); (U-L)/(U+L); D=storedClose*storedVolume',
        price_control='U0=mean(r | r>0); L0=mean(-r | r<0); (U0-L0)/(U0+L0)',
        scaling='one frexp/ldexp power-of-two D scale, r/max(abs(r)), min(scaledD)/scaledD; common across signs',
        arithmetic_policy='nonfinite computed arithmetic or scaling loss raises; only <=1e-12 bound-roundoff clipping',
        shared_availability='complete positive finite paths, >=40 returns on each strict sign, both denominators defined',
        counts_policy='int64; zero before a complete valid path; retained after complete path even if either side<40',
        ranking='full49 finite candidates; average percentile minus0.5; NaN preserved, finite zero valid',
        interpretation='heterogeneous stored close/volume proxies, not certified split-consistent true historical dollar volume; reciprocal unit algebra does not certify vendors; not observed order flow or causal demand',
        eligibility_read=False, future_labels_used=False, recent_42_prices_read=False,
        target_last_day_raw_price_volume_read=True, unused_raw_first_row_read=False,
        months=[], **{reason+'_cells': 0 for reason in REASONS})
    for i, (day, symbols) in enumerate(normalized):
        t = int(px.index.get_loc(day))
        start, end = t+FIRST_OFFSET, t+LAST_OFFSET
        detail = dict(date=day.strftime('%Y-%m-%d'), first_price_date=None, last_price_date=None,
                      first_return_date=None, last_return_date=None,
                      **{reason: 0 for reason in REASONS})
        status = np.full(49, 'insufficient_history', dtype=object)
        if start >= 0:
            columns = px.columns.get_indexer(symbols)
            # Slice before conversion: exclude unused raw first row and all recent/future values.
            adjusted = px.iloc[start:end+1, columns].to_numpy(dtype=float, na_value=np.nan)
            close = frames['close'].iloc[start+1:end+1, columns].to_numpy(dtype=float, na_value=np.nan)
            volume = frames['volume'].iloc[start+1:end+1, columns].to_numpy(dtype=float, na_value=np.nan)
            dollar, price, n_positive, n_negative, status = _window_feature(adjusted, close, volume)
            arrays['raw_dollar'][i], arrays['raw_price'][i] = dollar, price
            arrays['n_positive'][i], arrays['n_negative'][i] = n_positive, n_negative
            arrays['impact_available'][i] = status == 'available'
            for raw_key, rank_key in (('raw_dollar', 'rank_dollar'), ('raw_price', 'rank_price')):
                arrays[rank_key][i] = (pd.Series(arrays[raw_key][i], index=symbols).rank(method='average', pct=True)-.5).to_numpy()
            detail.update(first_price_date=px.index[start].strftime('%Y-%m-%d'),
                          last_price_date=px.index[end].strftime('%Y-%m-%d'),
                          first_return_date=px.index[start+1].strftime('%Y-%m-%d'),
                          last_return_date=px.index[end].strftime('%Y-%m-%d'))
        for reason in REASONS:
            detail[reason] = int(np.count_nonzero(status == reason))
            audit[reason+'_cells'] += detail[reason]
        detail['status'] = dict(zip(symbols, status.tolist()))
        for key in ('n_positive', 'n_negative'):
            detail[key] = {symbol: int(value) for symbol, value in zip(symbols, arrays[key][i])}
        audit['months'].append(detail)
    for key in FLOAT_KEYS:
        if not np.array_equal(np.isfinite(arrays[key]), arrays['impact_available']):
            raise FloatingPointError(key+' availability differs from shared feature availability')
    if sum(audit[reason+'_cells'] for reason in REASONS) != shape[0]*49:
        raise AssertionError('Feature audit does not cover every candidate cell')
    return arrays, audit
