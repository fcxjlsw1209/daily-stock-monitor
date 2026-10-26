"""Pure paired annual Ridge forecasts of severe/any next-month SPY-relative loss.

Callers validate frozen source/cache/calendar provenance. This module does not
load market data, write artifacts, select holdings, or calculate portfolio returns.
Only the frozen numerical Ridge helper is imported. Exact rational labels use
four positive finite stored binary-float prices. Targets are neither clipped nor
centered; binary Ridge predictions are ranking scores, not calibrated odds.
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
_SPEC = importlib.util.spec_from_file_location('relative_tail190_frozen_numeric', _NUMERIC_PATH)
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
    return [root / 'work/monthly_relative_tail190_model.py',
            root / 'work/test_monthly_relative_tail190_model.py',
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
        ends[dates == day] = months[_next_month(day)]
    if expected_ends is not None:
        supplied = _days(expected_ends, 'expected label ends')
        require(np.array_equal(supplied, ends), 'Cached label ends differ from official natural-month ends')
    return ends


def _plain_bounds(value):
    require(not value.size or bool(np.all((value >= -0.5 - 1e-12) & (value <= 0.5 + 1e-12))),
            'Used plain10 features exceed their frozen centered-percentile bounds')


def relative_events(stock_start, stock_end, spy_start, spy_end):
    """Exact inequalities on binary floats, without a floating wealth ratio.

    Equality at 9/10 or 1 is non-event. Each source float is converted exactly;
    large/small finite prices cannot overflow these integer cross-products.
    """
    prices = (stock_start, stock_end, spy_start, spy_end)
    require(all(isinstance(x, (int, float, np.integer, np.floating)) and
                not isinstance(x, (bool, np.bool_)) for x in prices),
            'Four real nonboolean endpoint prices required')
    prices = tuple(float(x) for x in prices)
    require(all(math.isfinite(x) and x > 0 for x in prices),
            'Four positive finite endpoint prices required')
    ss, se, bs, be = map(Fraction.from_float, prices)
    left, right = se * bs, ss * be
    candidate, control = left * 10 < right * 9, left < right
    require(not candidate or control, 'Severe relative event must be a subset')
    return float(candidate), float(control)


def build_endpoint_targets(close, sessions, dates, symbols, common_good, eligible,
                           finite_features, *, expected_ends=None):
    """Exact next-calendar-month binary labels on a common inherited cohort.

    Retain upstream complete-path/volatility/SPY requirements in common_good.
    No query future is consulted and no absent endpoint is shortened or filled.
    A singleton keeps its raw binary labels but is excluded from training.
    """
    dates = _days(dates, 'training dates')
    n = len(dates)
    symbols = _symbols(symbols, n, 'training symbols')
    _identities(dates, symbols, 'Training')
    ends = _anchors(sessions, dates, expected_ends)
    base = (_bool(common_good, (n,), 'common_good') & _bool(eligible, (n,), 'train_eligible') &
            _bool(finite_features, (n,), 'features_finite'))
    require(isinstance(close, pd.DataFrame) and isinstance(close.index, pd.DatetimeIndex),
            'close must be a DataFrame with a DatetimeIndex')
    index = _days(close.index, 'close index')
    require(close.index.is_unique and close.index.is_monotonic_increasing and close.columns.is_unique,
            'Price axes must be ordered and unique')
    values = {key: np.full(n, np.nan) for key in TARGETS}
    endpoint_good = np.zeros(n, bool)
    for day in np.unique(dates):
        rows = np.flatnonzero((dates == day) & base)
        if not len(rows) or 'SPY' not in close.columns:
            continue
        positions = pd.DatetimeIndex(index).get_indexer([day, ends[rows[0]]])
        if (positions < 0).any():
            continue
        for row in rows:
            if symbols[row] not in close.columns:
                continue
            column, benchmark = close.columns.get_loc(symbols[row]), close.columns.get_loc('SPY')
            cells = [close.iat[int(p), c] for c in (column, benchmark) for p in positions]
            if any(np.asarray(x).ndim != 0 or np.asarray(x).dtype.kind not in 'fiu' for x in cells):
                continue
            cells = tuple(float(x) for x in cells)
            if not all(math.isfinite(x) and x > 0 for x in cells):
                continue
            values['candidate_target'][row], values['control_target'][row] = relative_events(*cells)
            endpoint_good[row] = True
    good = base & endpoint_good
    records = []
    for day in np.unique(dates):
        rows = dates == day
        count = int(good[rows].sum())
        if count < 2:
            good[rows] = False
        records.append(dict(signal_date=str(day), label_end=str(ends[np.flatnonzero(rows)[0]]),
                            base_rows=int(base[rows].sum()), endpoint_rows=count,
                            good_rows=int(good[rows].sum()),
                            status='complete' if count >= 2 else 'fewer_than_two_shared_rows'))
    return dict(dates=dates, ends=ends, symbols=symbols, **values, base_good=base, good=good,
                audit=dict(targets=list(TARGETS), rows=n, base_rows=int(base.sum()),
                    good_rows=int(good.sum()), months=records, clipped=False, centered=False,
                    relative_wealth_thresholds=dict(candidate=[9, 10], control=[1, 1]),
                    exact_binary_float_rational=True, candidate_is_subset=True,
                    four_positive_finite_endpoints=True,
                    maturity='next official calendar month-end close',
                    inherited_common_good_retained=True, no_endpoint_shortening=True,
                    calendar_completeness_certified_by_this_function=False))


def annual_forecasts(labels, train_plain, event_dates, event_symbols, event_plain,
                     event_good, *, event_monthend):
    """Fixed 18-month-minimum, five-year annual two-target pooled Ridge.

    event_monthend is supplied from the provenance-validated calendar by build.
    No classes are reweighted. Constant-class samples are legitimate fits.
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
    require(all(bool(np.isin(targets[key][usable], [0.0, 1.0]).all()) for key in TARGETS),
            'Both relative targets must be binary')
    require(bool(np.all(targets['candidate_target'][usable] <= targets['control_target'][usable])),
            'Severe relative target must be a subset of control')
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
                target_counts={key: int(targets[key][mask].sum()) for key in TARGETS},
                target_prevalence={key: float(targets[key][mask].mean()) if len(ids) else None
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
                    weighted_target_prevalence={key: float(np.average(targets[key][mask], weights=weights))
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
