"""Source-only early-query replay of the unchanged frozen local70 model.

No file I/O, feature construction, target construction, ranks, or orders occur
here. The caller supplies the independent 295-price signal-time warmup mask.
The original training labels/common cohort enter one annual model invocation;
query qualification never uses the query's future label or common-good value.
"""
from datetime import date
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = ROOT / 'outputs/monthly_local_linear70/model.py'
_spec = importlib.util.spec_from_file_location('early229_frozen_local', LOCAL_PATH)
local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local)
TARGETS = tuple(local.TARGETS)
RAW_KEYS = ('source_log', 'local_constant', 'local_linear')
UPPER_BOUND = '2016-08-17'
METHOD = 'early_original_query_history_extension'
QUERY_QUALIFICATION = ('finite cached plain10 AND (original train_eligible OR '
                       'caller-verified 295-price warmup)')
COUNT_FIELDS = (
    'query_rows', 'original_query_rows', 'expansion_requested_rows',
    'union_requested_rows', 'original_available_rows', 'added_available_rows',
    'available_rows', 'local_available_rows', 'source_target_fits',
    'pooled_log_refits', 'local_evaluations', 'local_solves',
    'actual_regression_solves',
)


def dependency_files():
    return tuple(sorted(set(local.dependency_files()) | {Path(__file__).resolve()}))


def _days(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(name + ' must be a one-dimensional date vector')
    if not raw.size:
        return np.empty(0, dtype='datetime64[D]')
    if raw.dtype.kind == 'M':
        if np.datetime_data(raw.dtype) != ('D', 1):
            raise ValueError(name + ' must use explicit calendar-day precision')
        result = raw.copy()
    elif raw.dtype.kind in 'UO':
        for value in raw:
            if not isinstance(value, (str, np.str_)):
                raise ValueError(name + ' must contain canonical ISO dates')
            try:
                parsed = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(name + ' must contain canonical ISO dates') from exc
            if parsed.isoformat() != value:
                raise ValueError(name + ' must contain canonical ISO dates')
        result = raw.astype('datetime64[D]')
    else:
        raise ValueError(name + ' must contain dates, not numerical offsets')
    if np.isnat(result).any():
        raise ValueError(name + ' contains NaT')
    return result


def _bool(values, length, name):
    result = np.asarray(values)
    if result.shape != (length,) or result.dtype.kind != 'b':
        raise ValueError(name + ' must be an exact boolean row vector')
    return result.copy()


def _numeric(values, shape, name):
    result = np.asarray(values)
    if result.shape != shape or result.dtype.kind not in 'iuf':
        raise ValueError(name + ' must be a real numeric array of shape ' + str(shape))
    # NaN/Inf are intentionally left to the unchanged finite-cohort rules.
    return result.astype(float, copy=True)


def _symbols(values, length, name):
    result = np.asarray(values)
    if result.shape != (length,) or any(
            not isinstance(x, (str, np.str_)) or not x or x.strip() != x
            for x in result):
        raise ValueError(name + ' must contain exact nonempty security IDs')
    return result.astype(str, copy=True)


def _identity(dates, symbols, name, ends=None):
    if len(dates) > 1 and np.any(dates[1:] < dates[:-1]):
        raise ValueError(name + ' dates must be nondecreasing')
    identities = list(zip(dates.astype(str), symbols))
    if len(set(identities)) != len(identities):
        raise ValueError(name + ' contains duplicate date/security identities')
    months = dates.astype('datetime64[M]')
    for month in np.unique(months):
        use = months == month
        if len(np.unique(dates[use])) != 1:
            raise ValueError(name + ' must have one signal date per calendar month')
        if ends is not None and len(np.unique(ends[use])) != 1:
            raise ValueError(name + ' must have one label end per calendar month')


def forecasts(dates, ends, symbols, X, targets, common_good, event_dates,
              event_symbols, event_X, event_good, *, expansion_allowed):
    """Replay only pre-2016-08-17 queries with the original annual estimator.

    ``event_good`` is original train_eligible, not common_good or future label
    availability. ``expansion_allowed`` is caller-verified signal-time warmup.
    Production hyperparameters are the fixed local70 defaults, not parameters
    of this wrapper. The query axis may be empty or a physical prefix.
    """
    dates, ends = _days(dates, 'dates'), _days(ends, 'ends')
    event_dates = _days(event_dates, 'event_dates')
    n, q = len(dates), len(event_dates)
    if (ends.shape != (n,) or (ends <= dates).any() or
            np.any(ends.astype('datetime64[M]') != dates.astype('datetime64[M]') + 1)):
        raise ValueError('Labels must end strictly after signals in the next calendar month')
    if np.any(event_dates >= np.datetime64(UPPER_BOUND, 'D')):
        raise ValueError('Early query dates must precede ' + UPPER_BOUND)
    symbols = _symbols(symbols, n, 'symbols')
    event_symbols = _symbols(event_symbols, q, 'event_symbols')
    _identity(dates, symbols, 'training', ends)
    _identity(event_dates, event_symbols, 'queries')
    X = _numeric(X, (n, 10), 'X')
    event_X = _numeric(event_X, (q, 10), 'event_X')
    common_good = _bool(common_good, n, 'common_good')
    event_good = _bool(event_good, q, 'event_good')
    expansion_allowed = _bool(expansion_allowed, q, 'expansion_allowed')
    if not isinstance(targets, dict) or set(targets) != set(TARGETS):
        raise ValueError('Exactly the original four target arrays are required')
    targets = {key: _numeric(targets[key], (n,), key) for key in TARGETS}
    finite = np.isfinite(event_X).all(axis=1)
    original = event_good & finite
    added = ~event_good & expansion_allowed & finite
    union = original | added
    result = local.annual_forecasts(dates, ends, symbols, X, targets, common_good,
                                   event_dates, event_symbols, event_X, union)
    replay = result['audit']
    source_fits = 4 * sum(row['status'] == 'fit' and row['prediction_rows'] > 0
                          for row in replay['source']['annual'])
    pooled_fits = sum(row['prediction_rows'] > 0 for row in replay['annual'])
    local_solves = sum(row['status'] == 'fit' for row in replay['local'])
    available, own = result['available'], result['local_available']
    if (np.any(available & ~union) or np.any(own & ~available) or
            any(not np.array_equal(np.isfinite(result['raw'][k]), available) for k in RAW_KEYS)):
        raise AssertionError('Frozen model changed the declared query support')
    counts = dict(query_rows=q, original_query_rows=int(original.sum()),
        expansion_requested_rows=int(added.sum()), union_requested_rows=int(union.sum()),
        original_available_rows=int((available & original).sum()),
        added_available_rows=int((available & added).sum()), available_rows=int(available.sum()),
        local_available_rows=int(own.sum()), source_target_fits=int(source_fits),
        pooled_log_refits=int(pooled_fits), local_evaluations=len(replay['local']),
        local_solves=int(local_solves),
        actual_regression_solves=int(source_fits + pooled_fits + local_solves))
    audit = dict(method=METHOD, query_rows=q, feature_count=10, raw_keys=list(RAW_KEYS),
        query_date_exclusive_upper_bound=UPPER_BOUND, original_query_mask=original.tolist(),
        expansion_query_mask=added.tolist(), union_query_mask=union.tolist(),
        expansion_allowed=expansion_allowed.tolist(), query_qualification=QUERY_QUALIFICATION,
        query_future_labels_used=False, training_targets_changed=False,
        training_common_good_changed=False, new_features_computed=False,
        annual_forecasts_calls=1, replay_audit=replay, counts=counts)
    return dict(raw=result['raw'], available=available, local_available=own, audit=audit)
