"""Pure paired annual Ridge forecasts of next-month path/terminal relative shortfalls.

Callers validate frozen source/cache/calendar provenance. This module does not
load market data, write artifacts, select holdings, or calculate portfolio returns.
Only the frozen numerical Ridge helper is imported. Exact rational labels use
complete stock/SPY paths on the supplied official calendar. Each path deficit
is accumulated as a rational number; only the final mean and endpoint are
converted to floats. Targets are neither clipped nor centered. Ridge forecasts
are ranking scores, not calibrated probabilities or causal risk estimates.
"""
from pathlib import Path
import hashlib
import math
from fractions import Fraction
import importlib.util

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

_NUMERIC_PATH = Path(__file__).resolve().parents[1] / 'outputs/monthly_architecture_restart/residual_model.py'
_SPEC = importlib.util.spec_from_file_location('relative_path209_frozen_numeric', _NUMERIC_PATH)
numeric = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(numeric)

TARGETS = ('candidate_target', 'control_target')
HANDS = ('candidate_rank', 'control_rank')
EXTRA = ('candidate_prediction', 'control_prediction', *HANDS, 'common_available')
LABELS = ('dates', 'ends', 'symbols', *TARGETS, 'base_good', 'good')
ALPHA = 10.0
HALF_LIFE_MONTHS = 12.0
MIN_MONTHS = 18
LOOKBACK_YEARS = 5
WIDTH = 10
INITIAL_PARTIAL_QUERY = np.datetime64('2016-08-17', 'D')


def dependency_files():
    return (Path(__file__).resolve(), _NUMERIC_PATH)


def source_files(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    return [root / 'work/monthly_relative_path209_model.py',
            root / 'work/test_monthly_relative_path209_model.py',
            root / 'outputs/monthly_architecture_restart/residual_model.py']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _days(value, name):
    value = np.asarray(value)
    require(value.ndim == 1, name + ' must be a date vector')
    if not value.size:
        return np.array([], dtype='datetime64[D]')
    require(value.dtype.kind not in 'biufc', name + ' cannot contain numeric dates')
    try:
        index = pd.DatetimeIndex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' contains invalid dates') from exc
    require(not index.hasnans and index.tz is None and index.equals(index.normalize()),
            name + ' must contain naive midnight dates without NaT')
    return index.to_numpy(dtype='datetime64[D]').copy()


def _bool(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind == 'b', name + ' must be an exact boolean array')
    return value.copy()


def _numbers(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'fiu', name + ' must be an exact real numeric array')
    return value.astype(float, copy=True)


def _symbols(value, n, name):
    value = np.asarray(value)
    require(value.shape == (n,), name + ' must align with dates')
    require(all(isinstance(x, str) and x and x.strip() == x for x in value), name + ' has invalid IDs')
    return value.astype(str, copy=True)


def _identities(dates, symbols, name):
    require(len(set(zip(dates.tolist(), symbols.tolist()))) == len(dates), name + ' repeats a security/date')
    require(not len(dates) or bool(np.all(dates[1:] >= dates[:-1])), name + ' dates must be ordered')


def _sessions(value):
    value = _days(value, 'calendar sessions')
    require(len(value) > 0 and bool(np.all(value[1:] > value[:-1])), 'Calendar must be unique, ordered and nonempty')
    months = {}
    for day in value:
        months[str(day)[:7]] = day
    return value, months


def _next_month(day):
    return str(day.astype('datetime64[M]') + 1)


def _anchors(sessions, dates, expected_ends=None):
    calendar, months = _sessions(sessions)
    members = set(calendar.tolist())
    ends = np.empty(len(dates), dtype='datetime64[D]')
    for day in np.unique(dates):
        require(day.tolist() in members and months.get(str(day)[:7]) == day,
                'Training signal must be the official calendar month-end')
        require(_next_month(day) in months, 'Calendar must cover the complete next target month')
        end = months[_next_month(day)]
        position = int(np.searchsorted(calendar, end))
        require(position + 1 < len(calendar) and
                calendar[position + 1].astype('datetime64[M]') != end.astype('datetime64[M]'),
                'Calendar must include the next official session after the complete target month')
        ends[dates == day] = end
    if expected_ends is not None:
        supplied = _days(expected_ends, 'expected label ends')
        require(np.array_equal(supplied, ends), 'Cached label ends differ from official natural-month ends')
    return ends


def _plain_bounds(value):
    require(not value.size or bool(np.all((value >= -0.5 - 1e-12) & (value <= 0.5 + 1e-12))),
            'Used plain10 features exceed their frozen centered-percentile bounds')


def relative_path(stock_path, spy_path):
    """Return mean daily relative deficit and terminal deficit, each rounded once.

    Paths include signal close at index0 and every target-session close after it.
    This helper has no date inference; the caller binds the exact official axis.
    Individual deficits and their sum stay rational until the final conversion.
    """
    stock, spy = np.asarray(stock_path, dtype=object), np.asarray(spy_path, dtype=object)
    require(stock.ndim == spy.ndim == 1 and len(stock) == len(spy) and len(stock) >= 2,
            'Two equal-length paths including signal and target closes required')
    for path in (stock, spy):
        require(all(isinstance(x, (int, float, np.integer, np.floating)) and
                    not isinstance(x, (bool, np.bool_)) for x in path),
                'Path prices must be real nonboolean numbers')
        require(all(math.isfinite(float(x)) and float(x) > 0 for x in path),
                'Path prices must be positive and finite')
    stock = tuple(Fraction.from_float(float(x)) for x in stock)
    spy = tuple(Fraction.from_float(float(x)) for x in spy)
    total = Fraction(0)
    terminal = Fraction(0)
    for price, market in zip(stock[1:], spy[1:]):
        left, right = price * spy[0], stock[0] * market
        terminal = (right - left) / right if left < right else Fraction(0)
        total += terminal
    candidate = float(total / (len(stock) - 1))
    control = float(terminal)
    require(math.isfinite(candidate) and math.isfinite(control) and
            0.0 <= candidate <= 1.0 and 0.0 <= control <= 1.0,
            'Path and terminal shortfalls require independent unit bounds')
    return candidate, control


def build_endpoint_targets(close, sessions, dates, symbols, common_good, eligible,
                           finite_features, *, expected_ends=None):
    """Compatibility API name; labels use COMPLETE next-calendar-month paths.

    Retain upstream complete-path/volatility/SPY requirements in common_good.
    Every supplied official target session is required for BOTH targets, without
    filling or compressing prices. Future query outcomes never filter queries.
    Complete singleton-month rows retain raw labels but are excluded from fits.
    """
    dates = _days(dates, 'training dates')
    n = len(dates)
    symbols = _symbols(symbols, n, 'training symbols')
    _identities(dates, symbols, 'Training')
    ends = _anchors(sessions, dates, expected_ends)
    calendar, _ = _sessions(sessions)
    base = (_bool(common_good, (n,), 'common_good') & _bool(eligible, (n,), 'train_eligible') &
            _bool(finite_features, (n,), 'features_finite'))
    require(isinstance(close, pd.DataFrame) and isinstance(close.index, pd.DatetimeIndex),
            'close must be a DataFrame with a DatetimeIndex')
    index = _days(close.index, 'close index')
    require(close.index.is_unique and close.index.is_monotonic_increasing and close.columns.is_unique,
            'Price axes must be ordered and unique')
    index = pd.DatetimeIndex(index)
    values = {key: np.full(n, np.nan) for key in TARGETS}
    complete = np.zeros(n, bool)
    path_checks = [None] * n
    windows = {}
    source_prices = target_sessions = 0
    for day in np.unique(dates):
        rows = np.flatnonzero(dates == day)
        end = ends[rows[0]]
        source_dates = calendar[(calendar >= day) & (calendar <= end)]
        require(len(source_dates) >= 2 and source_dates[0] == day and source_dates[-1] == end,
                'Complete signal-to-target official path required')
        target_dates = source_dates[1:]
        positions = index.get_indexer(source_dates)
        missing = [str(d) for d, p in zip(source_dates, positions) if p < 0]
        windows[str(day)] = (source_dates, target_dates)
        for row in rows:
            record = dict(row_index=int(row), signal_date=str(day), label_end=str(end),
                symbol=str(symbols[row]), base_good=bool(base[row]), path_complete=False,
                target_session_count=len(target_dates), status='inherited_base_unavailable',
                missing_official_sessions=[], invalid_stock_dates=[], invalid_spy_dates=[])
            path_checks[row] = record
            if not base[row]:
                continue
            if symbols[row] not in close.columns:
                record['status'] = 'missing_stock_symbol'
                continue
            if 'SPY' not in close.columns:
                record['status'] = 'missing_spy_symbol'
                continue
            if missing:
                record.update(status='missing_official_session', missing_official_sessions=missing[:])
                continue
            prices = []
            for symbol, field in ((symbols[row], 'invalid_stock_dates'), ('SPY', 'invalid_spy_dates')):
                column = close.columns.get_loc(symbol)
                path = []
                for date, position in zip(source_dates, positions):
                    value = close.iat[int(position), column]
                    valid = np.asarray(value).ndim == 0 and np.asarray(value).dtype.kind in 'fiu'
                    if valid:
                        value = float(value)
                        valid = math.isfinite(value) and value > 0
                    if not valid:
                        record[field].append(str(date))
                    path.append(value)
                prices.append(path)
            if record['invalid_stock_dates'] or record['invalid_spy_dates']:
                record['status'] = 'invalid_stock_price' if record['invalid_stock_dates'] else 'invalid_spy_price'
                continue
            values['candidate_target'][row], values['control_target'][row] = relative_path(*prices)
            complete[row] = True
            record.update(status='complete', path_complete=True)
            source_prices += 2 * len(source_dates)
            target_sessions += len(target_dates)
    good = base & complete
    records = []
    for day in np.unique(dates):
        rows = dates == day
        count = int(complete[rows].sum())
        if count < 2:
            good[rows] = False
        source_dates, target_dates = windows[str(day)]
        records.append(dict(signal_date=str(day), label_end=str(ends[np.flatnonzero(rows)[0]]),
            base_rows=int(base[rows].sum()), endpoint_rows=count, complete_path_rows=count,
            source_dates=[str(d) for d in source_dates], target_dates=[str(d) for d in target_dates],
            target_session_count=len(target_dates), good_rows=int(good[rows].sum()),
            status='complete' if count >= 2 else 'fewer_than_two_shared_rows'))
    return dict(dates=dates, ends=ends, symbols=symbols, **values, base_good=base, good=good,
        audit=dict(targets=list(TARGETS), rows=n, base_rows=int(base.sum()), good_rows=int(good.sum()),
            months=records, clipped=False, centered=False,
            relative_target_definitions=dict(candidate='mean_d max(0,1-R_d)', control='max(0,1-R_end)'),
            exact_binary_float_rational=True, rounded_targets_once=True,
            complete_positive_finite_paths=True, target_bounds_independent=True,
            maturity='next official calendar month-end close', inherited_common_good_retained=True,
            no_endpoint_shortening=True, calendar_completeness_certified_by_this_function=False,
            source_path_rows=int(complete.sum()), source_price_observations=source_prices,
            target_session_observations=target_sessions, missing_path_rows=int(base.sum()-complete.sum()),
            path_checks=path_checks))


def annual_forecasts(labels, train_plain, event_dates, event_symbols, event_plain,
                     event_good, *, event_monthend):
    """Fixed 18-month-minimum, five-year annual two-target pooled Ridge.

    event_monthend is supplied from the provenance-validated calendar by build.
    Neither target is reweighted. Constant-target samples are legitimate fits.
    """
    dates = _days(labels['dates'], 'training dates')
    ends = _days(labels['ends'], 'training ends')
    n = len(dates)
    symbols = _symbols(labels['symbols'], n, 'training symbols')
    _identities(dates, symbols, 'Training')
    require(ends.shape == (n,) and bool(np.all(ends > dates)) and
            bool(np.all(ends.astype('datetime64[M]') == dates.astype('datetime64[M]') + 1)),
            'Labels need aligned next-calendar-month ends')
    for day in np.unique(dates):
        require(len(np.unique(ends[dates == day])) == 1, 'A training month has multiple label ends')
    query_dates = _days(event_dates, 'event dates')
    q = len(query_dates)
    query_symbols = _symbols(event_symbols, q, 'event symbols')
    _identities(query_dates, query_symbols, 'Queries')
    X = _numbers(train_plain, (n, WIDTH), 'train_plain')
    query = _numbers(event_plain, (q, WIDTH), 'event_plain')
    targets = {key: _numbers(labels[key], (n,), key) for key in TARGETS}
    usable = _bool(labels['good'], (n,), 'label good') & np.isfinite(X).all(axis=1)
    usable &= np.isfinite(targets['candidate_target']) & np.isfinite(targets['control_target'])
    for key in TARGETS:
        target = targets[key][usable]
        require(bool(np.all((target >= 0.0) & (target <= 1.0))),
                'Both path and terminal targets must separately be bounded in [0,1]')
    for day in np.unique(dates):
        rows = dates == day
        if int(usable[rows].sum()) < 2:
            usable[rows] = False
    queries = (_bool(event_good, (q,), 'event_good') & _bool(event_monthend, (q,), 'event_monthend') &
               np.isfinite(query).all(axis=1))
    raw = {key: np.full(q, np.nan) for key in TARGETS}
    available = np.zeros(q, bool)
    years = query_dates.astype('datetime64[Y]').astype(int) + 1970
    annual = []
    fits = 0
    with threadpool_limits(limits=1):
        for year in np.unique(years):
            cutoff = np.datetime64('%04d-12-31' % (year - 1), 'D')
            lower = numeric._years_before(cutoff, LOOKBACK_YEARS)
            mask = usable & (dates > lower) & (ends <= cutoff)
            ids = np.flatnonzero(mask)
            predict = (years == year) & queries
            count = numeric._month_count(dates[mask])
            status = 'cold_start' if count < MIN_MONTHS else ('fit' if predict.any() else 'no_queries')
            record = dict(year=int(year), cutoff=str(cutoff), lookback_exclusive_start=str(lower),
                training_rows=len(ids), training_months=count, prediction_rows=int(predict.sum()),
                latest_label_end=str(ends[mask].max()) if len(ids) else None,
                training_indices=ids.tolist(), training_indices_sha256=_digest(ids),
                training_dates_sha256=_digest(dates[mask]), training_symbols_sha256=_digest(symbols[mask]),
                status=status,
                target_positive_counts={key: int(np.count_nonzero(targets[key][mask] > 0.0))
                                        for key in TARGETS},
                target_sums={key: float(targets[key][mask].sum()) for key in TARGETS},
                target_means={key: float(targets[key][mask].mean()) if len(ids) else None
                              for key in TARGETS},
                target_maxima={key: float(targets[key][mask].max()) if len(ids) else None
                               for key in TARGETS})
            if status == 'fit':
                _plain_bounds(X[mask]); _plain_bounds(query[predict])
                weights = numeric.month_balanced_weights(dates[mask], cutoff, HALF_LIFE_MONTHS)
                models = {key: numeric._fit_ridge(X[mask], targets[key][mask], weights, ALPHA) for key in TARGETS}
                require(np.array_equal(models[TARGETS[0]][0], models[TARGETS[1]][0]) and
                        np.array_equal(models[TARGETS[0]][1], models[TARGETS[1]][1]), 'Shared training scalers diverged')
                for key in TARGETS:
                    raw[key][predict] = numeric._predict(models[key], query[predict])
                require(all(np.isfinite(raw[key][predict]).all() for key in TARGETS), 'Nonfinite Ridge predictions')
                available[predict] = True
                fits += 2
                record.update(weight_sum=float(weights.sum()), weights_sha256=_digest(weights),
                    scaler_mean_sha256=_digest(models[TARGETS[0]][0]),
                    scaler_scale_sha256=_digest(models[TARGETS[0]][1]),
                    target_sha256={key: _digest(targets[key][mask]) for key in TARGETS},
                    shared_rows_weights_scalers=True, model_fits=2,
                    weighted_target_means={key: float(np.average(targets[key][mask], weights=weights))
                                                for key in TARGETS})
            else:
                record['model_fits'] = 0
            annual.append(record)
    return dict(candidate_prediction=raw['candidate_target'], control_prediction=raw['control_target'],
        common_available=available, audit=dict(alpha=ALPHA, half_life_months=HALF_LIFE_MONTHS,
            lookback_years=LOOKBACK_YEARS, minimum_mature_months=MIN_MONTHS,
            new_pooled_model_fits=fits, local_model_fits=0, shared_training_mask_sha256=_digest(usable),
            training_only_standardization=True, calibrated_probabilities=False,
            target_clipping=False, target_centering=False, annual=annual))


def rank_predictions(candidate, control, available, eligibility):
    """Common-population average centered ranks; never rank missing cells."""
    shape = np.asarray(available).shape
    require(len(shape) == 2, 'Rank inputs need a two-dimensional query grid')
    a = _numbers(candidate, shape, 'candidate_prediction')
    b = _numbers(control, shape, 'control_prediction')
    require(not np.isinf(a).any() and not np.isinf(b).any(), 'Infinite relative predictions')
    mask = (_bool(available, shape, 'common_available') & _bool(eligibility, shape, 'eligibility') &
            np.isfinite(a) & np.isfinite(b))
    ranked = {key: np.zeros(shape, float) for key in HANDS}
    counts = []
    for month in range(shape[0]):
        good = mask[month]
        n = int(good.sum()); counts.append(n)
        if n <= 1:
            continue
        for key, values in zip(HANDS, (-a[month, good], -b[month, good])):
            ranks = pd.Series(values).rank(method='average').to_numpy() - 1.0
            ranked[key][month, good] = ranks / (n - 1) - 0.5
    return dict(**ranked, common_available=mask,
                audit=dict(common_counts=counts, population='signal-time eligible with both finite forecasts',
                    mapping='average_rank0/(n-1)-0.5; both use negative prediction; n<=1 => 0',
                    missing_correction=0, max_absolute_rank=0.5, no_holdings_or_returns_computed=True))


def _meta(meta, name):
    require(isinstance(meta, dict) and all(key in meta for key in ('dates', 'ends', 'symbols')),
            name + ' requires dates/ends/symbols')
    dates = _days(meta['dates'], name + ' dates')
    ends = _days(meta['ends'], name + ' ends')
    require(ends.shape == dates.shape and bool(np.all(ends > dates)), name + ' invalid end dates')
    require(not len(dates) or bool(np.all(dates[1:] > dates[:-1])), name + ' dates must be strictly ordered')
    pools = meta['symbols']
    require(isinstance(pools, (list, tuple)) and len(pools) == len(dates), name + ' pools do not align')
    widths = []
    all_symbols = []
    for pool in pools:
        require(isinstance(pool, (list, tuple, np.ndarray)), name + ' pool must be a vector')
        symbols = _symbols(pool, len(pool), name + ' symbols')
        require(len(set(symbols.tolist())) == len(symbols), name + ' repeats a pool symbol')
        widths.append(len(pool)); all_symbols.extend(symbols.tolist())
    return dates, ends, np.array(widths, int), np.array(all_symbols, dtype=str)


def build(adj_close, sessions, trainmeta, meta, train_plain, event_plain,
          common_good, train_eligible, eligibility):
    """Unified pure interface; outputs five query matrices and seven labels.

    The sole admitted partial query is the original first 2016-08-17 event,
    which receives exact zero corrections. Query evaluation ends (including
    the final partial replay) never create or truncate a training target.
    """
    train_dates, train_ends, widths, symbols = _meta(trainmeta, 'training metadata')
    query_dates, query_ends, query_widths, query_symbols = _meta(meta, 'query metadata')
    eligible = np.asarray(eligibility)
    require(eligible.ndim == 2 and eligible.shape[0] == len(query_dates) and eligible.dtype.kind == 'b',
            'eligibility must be a boolean query grid')
    require(bool(np.all(query_widths == eligible.shape[1])), 'Query pools must match eligibility width')
    dates = np.repeat(train_dates, widths)
    ends = np.repeat(train_ends, widths)
    X = _numbers(train_plain, (len(dates), WIDTH), 'train_plain')
    calendar, calendar_ends = _sessions(sessions)
    members = set(calendar.tolist())
    monthend = np.ones(len(query_dates), bool)
    for i, day in enumerate(query_dates):
        require(day.tolist() in members, 'Query date is not an official session')
        is_monthend = calendar_ends[str(day)[:7]] == day
        if not is_monthend:
            require(i == 0 and day == INITIAL_PARTIAL_QUERY, 'Only the initial 2016-08-17 query may be non-month-end')
        monthend[i] = is_monthend
    labels = build_endpoint_targets(adj_close, calendar, dates, symbols, common_good, train_eligible,
                                   np.isfinite(X).all(axis=1), expected_ends=ends)
    forecast = annual_forecasts(labels, X, np.repeat(query_dates, query_widths), query_symbols,
        event_plain, eligible.reshape(-1), event_monthend=np.repeat(monthend, query_widths))
    shape = eligible.shape
    raw = {key: forecast[key].reshape(shape) for key in ('candidate_prediction', 'control_prediction')}
    ranked = rank_predictions(raw['candidate_prediction'], raw['control_prediction'],
                              forecast['common_available'].reshape(shape), eligible)
    arrays = dict(**raw, **{key: ranked[key] for key in (*HANDS, 'common_available')})
    return dict(arrays=arrays, labels={key: labels[key] for key in LABELS}, label_audit=labels['audit'],
        model_audit=forecast['audit'], rank_audit=dict(**ranked['audit'],
            non_monthend_fallback_indices=np.flatnonzero(~monthend).tolist(),
            query_evaluation_ends_not_used_for_labels=True))
