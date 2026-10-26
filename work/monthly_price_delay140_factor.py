"""Pure old-window market-price-delay features; no I/O, orders or returns replay."""
import math
import numpy as np
import pandas as pd

EXTRA = ('raw_nonfit', 'raw_delay', 'rank_nonfit', 'rank_delay', 'common_available')
LAG_SESSIONS = 42
WEEKLY_OBSERVATIONS = 52
MARKET_PRICE_ENDPOINTS = 57
STOCK_PRICE_ENDPOINTS = 53
RCOND = 1e-12
MAX_CONDITION = 1e8
R2_TOLERANCE = 1e-10
MIN_FULL_R2 = 1e-8


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _numeric(values, shape):
    raw = np.asarray(values)
    _require(raw.shape == shape, 'Exact numeric shape required')
    _require(all(isinstance(v, (int, float, np.integer, np.floating)) and
                 not isinstance(v, (bool, np.bool_)) for v in raw.flat),
             'Numeric nonboolean source required')
    return np.asarray(raw, dtype=np.float64)


def _sessions(values, label):
    idx = pd.DatetimeIndex(values)
    _require(idx.tz is None and not idx.hasnans and idx.is_unique and
             idx.is_monotonic_increasing and idx.equals(idx.normalize()),
             label + ' must be sorted unique naive session dates')
    return idx


def weekly_endpoints(sessions, signal):
    """The last 57 complete W-FRI weeks before the 42-session cutoff.

    Friday is the calendar label, not a required trading session. The endpoint
    is the last official session of that week; source-price availability never
    influences endpoint selection.
    """
    official = _sessions(sessions, 'official sessions')
    day = pd.Timestamp(signal)
    positions = official.get_indexer([day])
    _require(positions[0] >= 0, 'Signal must be an official session')
    if positions[0] < LAG_SESSIONS:
        return None, pd.DatetimeIndex([]), pd.DatetimeIndex([])
    cutoff = official[positions[0] - LAG_SESSIONS]
    historical = official[:positions[0] - LAG_SESSIONS + 1]
    labels = historical.to_period('W-FRI').end_time.normalize()
    full = labels <= cutoff
    table = pd.DataFrame({'nominal': labels[full], 'actual': historical[full]})
    table = table.groupby('nominal', sort=True)['actual'].last().tail(MARKET_PRICE_ENDPOINTS)
    return cutoff, pd.DatetimeIndex(table.index), pd.DatetimeIndex(table.values)


def _detail(reason):
    return dict(available=False, regression_available=False, reason=reason,
                restricted_r2=None, full_r2=None, restricted_condition=None,
                full_condition=None, restricted_rank=None, full_rank=None,
                response_variance=None, raw_nonfit=None, raw_delay=None)


def _standard_design(regressors):
    means = regressors.mean(axis=0)
    centered = regressors - means
    scales = np.sqrt(np.mean(centered * centered, axis=0))
    if not (np.isfinite(scales) & (scales > 0)).all():
        return None
    z = centered / scales
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite standardized market regressors')
    return np.column_stack((np.ones(len(z)), z))


def regression_delay(stock_prices, market_prices):
    """52 weekly responses on current market return and 4 market lags."""
    y_prices = _numeric(stock_prices, (STOCK_PRICE_ENDPOINTS,))
    x_prices = _numeric(market_prices, (MARKET_PRICE_ENDPOINTS,))
    _require((np.isfinite(y_prices) & (y_prices > 0)).all() and
             (np.isfinite(x_prices) & (x_prices > 0)).all(),
             'Complete positive finite endpoint prices required')
    out = _detail('zero_response_variance')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        y = y_prices[1:] / y_prices[:-1] - 1.
        market = x_prices[1:] / x_prices[:-1] - 1.
        x = np.column_stack([market[4-j:56-j] for j in range(5)])
        centered_y = y - y.mean()
        sst = float(np.dot(centered_y, centered_y))
        if not math.isfinite(sst):
            raise FloatingPointError('Nonfinite response variation')
        out['response_variance'] = sst / WEEKLY_OBSERVATIONS
        if sst <= 0:
            return out
        full = _standard_design(x)
        restricted = _standard_design(x[:, :1])
        if full is None or restricted is None:
            out['reason'] = 'zero_market_variance'
            return out
        fits = {}
        for label, design in (('restricted', restricted), ('full', full)):
            coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=RCOND)
            out[label+'_rank'] = int(rank)
            condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
            out[label+'_condition'] = condition
            if rank != design.shape[1]:
                out['reason'] = 'rank_deficient_' + label
                return out
            if condition is None or not math.isfinite(condition) or condition > MAX_CONDITION:
                out['reason'] = 'ill_conditioned_' + label
                return out
            error = y - design @ coef
            r2 = float(1. - np.dot(error, error)/sst)
            if not math.isfinite(r2) or r2 < -R2_TOLERANCE or r2 > 1.+R2_TOLERANCE:
                raise FloatingPointError('R-squared outside numerical tolerance')
            fits[label] = min(1., max(0., r2))
        if fits['restricted'] > fits['full'] + R2_TOLERANCE:
            raise FloatingPointError('Nested R-squared ordering violated')
        # Numerical ties preserve nesting without amplifying negative delay.
        fits['restricted'] = min(fits['restricted'], fits['full'])
        out.update(restricted_r2=fits['restricted'], full_r2=fits['full'])
        if fits['full'] < MIN_FULL_R2:
            out['reason'] = 'full_r2_below_numeric_floor'
            return out
        out.update(available=True, regression_available=True, reason='available',
                   raw_nonfit=1.-fits['restricted'],
                   raw_delay=1.-fits['restricted']/fits['full'])
    return out


def centered_midrank(values):
    """Average-rank scaling maps observed ranks to [-.5,.5]; ties are exact."""
    v = _numeric(values, (len(values),))
    result = np.full(len(v), np.nan)
    valid = np.isfinite(v)
    n = int(valid.sum())
    if n < 2:
        return result
    ranks = pd.Series(v[valid]).rank(method='average').to_numpy()
    result[valid] = (ranks-1.)/(n-1.)-.5
    return result


def build_features(panel, meta, sessions):
    prices = panel['adj']
    _require(isinstance(prices, pd.DataFrame), 'Explicit adjusted-price frame required')
    axis = _sessions(prices.index, 'prices')
    official = _sessions(sessions, 'official sessions')
    dates = _sessions(meta['dates'], 'signals')
    _require(len(official) > 0 and len(dates) > 0 and len(meta['symbols']) == len(dates),
             'Nonempty original axes')
    _require(prices.columns.is_unique, 'Unique source columns required')
    etfs = meta['etfs']
    _require(len(etfs) == 17 and len(set(etfs)) == 17, 'Original17 ETFs required')
    _require((official.get_indexer(dates) >= 0).all(), 'Signals must be official sessions')
    inspected = axis[(axis >= official[0]) & (axis <= dates[-1])]
    _require(inspected.isin(official).all(), 'Nonofficial historical source price dates')
    arrays = {key: np.full((len(dates), 49), np.nan) for key in EXTRA[:-1]}
    arrays['common_available'] = np.zeros((len(dates), 49), dtype=bool)
    cells = []
    for i, (day, names) in enumerate(zip(dates, meta['symbols'])):
        _require(len(names) == 49 and len(set(names)) == 49 and set(etfs) <= set(names),
                 'Original32 stocks plus17 ETFs required')
        cutoff, nominal, actual = weekly_endpoints(official, day)
        valid_market = False
        market = None
        if len(actual) == MARKET_PRICE_ENDPOINTS:
            market = _numeric(prices.reindex(index=actual, columns=['SPY']).to_numpy()[:, 0], (57,))
            valid_market = bool((np.isfinite(market) & (market > 0)).all())
        for j, symbol in enumerate(names):
            detail = _detail('etf_not_applicable')
            if symbol not in etfs:
                if len(actual) != MARKET_PRICE_ENDPOINTS:
                    detail = _detail('insufficient_history')
                elif not valid_market:
                    detail = _detail('invalid_market_path')
                else:
                    stock = _numeric(prices.reindex(index=actual[-53:], columns=[symbol]).to_numpy()[:, 0], (53,))
                    if not (np.isfinite(stock) & (stock > 0)).all():
                        detail = _detail('invalid_stock_path')
                    else:
                        detail = regression_delay(stock, market)
            cells.append(dict(signal_index=i, pool_index=j, signal_date=str(day.date()), symbol=symbol,
                benchmark=None if symbol in etfs else 'SPY',
                cutoff=None if cutoff is None else str(cutoff.date()),
                nominal_week_ends=[str(x.date()) for x in nominal],
                actual_week_ends=[str(x.date()) for x in actual],
                market_window_start=None if not len(actual) else str(actual[0].date()),
                stock_window_start=None if len(actual) < 53 else str(actual[-53].date()),
                window_end=None if not len(actual) else str(actual[-1].date()),
                **detail, rank_nonfit=None, rank_delay=None))
        row_cells = cells[i*49:(i+1)*49]
        valid = [j for j, cell in enumerate(row_cells) if cell['regression_available']]
        if len(valid) < 2:
            for j in valid:
                row_cells[j]['available'] = False
                row_cells[j]['reason'] = 'insufficient_cross_section'
            continue
        for j in valid:
            arrays['common_available'][i, j] = True
            for key in EXTRA[:2]:
                arrays[key][i, j] = row_cells[j][key]
        for raw, rank in zip(EXTRA[:2], EXTRA[2:4]):
            arrays[rank][i] = centered_midrank(arrays[raw][i])
            for j in valid:
                row_cells[j][rank] = float(arrays[rank][i, j])
    audit = dict(original_arrays_unchanged=True, original_months=len(dates), original_width=49,
        new_model_fits=0, monthly_label_models_fit=0, returns_or_nav_computed=False,
        new_model_fits_semantics='new predictive label/model fits only',
        attempted_stock_windows=sum(c['response_variance'] is not None for c in cells),
        descriptive_ols_calls=sum(int(c['restricted_rank'] is not None) + int(c['full_rank'] is not None) for c in cells),
        completed_regression_pairs=sum(c['full_rank'] is not None for c in cells),
        valid_regression_pairs=sum(c['regression_available'] for c in cells),
        feature_regressions_are_descriptive=True,
        original_auxiliary_states_and_core_coefficients_preserved=True, cells=cells)
    return dict(arrays=arrays, audit=audit)
