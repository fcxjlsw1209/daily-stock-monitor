"""Confirm cached personalized prediction direction using the previous formal month.

Pure caller-array transformation. No labels, fitting, prices, ranks or outcomes.
The pinned host validates formal signal dates against its trading calendar.
"""
from datetime import date
from pathlib import Path
import math
import numbers

import numpy as np

ARMS = ('incumbent_local67', 'persistence_confirmed', 'persistence_half')
INITIAL = '2016-08-17'
COUNT_FIELDS = ('query_rows', 'cached_forecast_rows', 'supported_rows',
                'unsupported_rows', 'current_zero_rows', 'confirmed_rows',
                'unconfirmed_rows', 'cross_year_supported_rows')


def dependency_files():
    return (Path(__file__).resolve(),)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value):
    _require(isinstance(value, str), 'Canonical ISO date string required')
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid ISO date') from exc
    _require(parsed.isoformat() == value, 'Noncanonical ISO date')
    return value


def _month(value):
    return int(value[:4])*12+int(value[5:7])-1


def _boolean(value, label):
    _require(isinstance(value, (bool, np.bool_)), 'Exact boolean required: '+label)
    return bool(value)


def _number(value, label):
    _require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
             'Real nonboolean value required: '+label)
    try:
        return float(value)
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Invalid binary64 '+label) from exc


def _finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite '+label)
    return value


def _inputs(query_meta, source, original_use):
    _require(isinstance(query_meta, dict) and isinstance(source, dict), 'Metadata/source dictionaries required')
    _require({'dates','ends','symbols'} <= set(query_meta), 'Explicit monthly identities required')
    _require({'raw_local_linear','raw_source_log','available'} <= set(source), 'Cached source fields missing')
    use = np.asarray(original_use)
    _require(use.ndim == 2 and use.shape[1] > 0, 'Original U must be a positive-width matrix')
    n, width = use.shape
    days, ends, names = (query_meta[k] for k in ('dates','ends','symbols'))
    _require(all(isinstance(v, list) and len(v) == n for v in (days, ends, names)), 'Metadata axes differ')
    formal = {}
    for i, (day, end, symbols) in enumerate(zip(days, ends, names)):
        day, end = _day(day), _day(end)
        partial = day == INITIAL
        _require(not partial or (i == 0 and end == '2016-08-31'), 'Initial partial must be first with its original end')
        _require(end > day and (_month(end) == _month(day)+1 or partial), 'Label interval must end in next month')
        _require(i == 0 or day > days[i-1], 'Signal dates must be unique and increasing')
        _require(isinstance(symbols, list) and len(symbols) == width
                 and all(isinstance(s, str) and s and s == s.strip() for s in symbols)
                 and len(set(symbols)) == width, 'Exact unique security IDs and fixed width required')
        if not partial:
            month = _month(day)
            _require(month not in formal, 'Duplicate formal signal month')
            formal[month] = i
    use = np.array([_boolean(v,'original U') for v in use.reshape(-1)], dtype=bool)
    q = n*width
    available = np.asarray(source['available'])
    _require(available.shape == (q,), 'Availability shape differs')
    available = np.array([_boolean(v,'source availability') for v in available], dtype=bool)
    numeric = {}
    for key in ('raw_local_linear','raw_source_log'):
        values = np.asarray(source[key])
        _require(values.shape == (q,), 'Cached prediction shape differs: '+key)
        values = np.array([_number(v,key) for v in values], dtype=float)
        _require(np.array_equal(np.isfinite(values), available) and not np.isinf(values).any(),
                 'Cached prediction finite mask differs: '+key)
        numeric[key] = values
    return days, names, formal, numeric['raw_local_linear'], numeric['raw_source_log'], available, use, width


def build_forecasts(query_meta, source, original_use):
    """Keep p only after consecutive-month sign confirmation; pair with half.

    Unsupported rows and zero current corrections copy p exactly. Prior-month
    matching is by calendar month and exact security ID, never column number.
    The common support only gates transformation; availability stays unchanged.
    """
    days, names, formal, raw, pooled, available, use, width = _inputs(query_meta, source, original_use)
    out = dict(confirmed=raw.copy(), half=raw.copy())
    counts = {k: 0 for k in COUNT_FIELDS}
    counts['query_rows'] = len(raw)
    counts['cached_forecast_rows'] = int(available.sum())
    records = []
    for j in range(len(raw)):
        if not available[j]:
            continue
        i, col = divmod(j, width)
        symbol, day = names[i][col], days[i]
        record = dict(row=j, date=day, symbol=symbol, prior_row=None,
                      status='unsupported', reason=None,
                      current_difference=None, previous_difference=None, cross_year=False)
        if not use[j]:
            record['reason'] = 'current_outside_universe'
        else:
            previous = formal.get(_month(day)-1)
            if previous is None:
                record['reason'] = 'missing_previous_formal_month'
            elif symbol not in names[previous]:
                record['reason'] = 'missing_previous_identity'
            else:
                prior = previous*width+names[previous].index(symbol)
                record['prior_row'] = prior
                if not available[prior]:
                    record['reason'] = 'previous_unavailable'
                elif not use[prior]:
                    record['reason'] = 'previous_outside_universe'
        if record['reason'] is not None:
            counts['unsupported_rows'] += 1
            records.append(record)
            continue
        prior = record['prior_row']
        record['cross_year'] = days[prior//width][:4] != day[:4]
        counts['supported_rows'] += 1
        counts['cross_year_supported_rows'] += int(record['cross_year'])
        difference = _finite(float(raw[j])-float(pooled[j]), 'current difference')
        record['current_difference'] = difference
        if difference == 0.:
            record['status'] = record['reason'] = 'current_zero'
            counts['current_zero_rows'] += 1
        else:
            previous_difference = _finite(float(raw[prior])-float(pooled[prior]), 'previous difference')
            record['previous_difference'] = previous_difference
            same = ((difference > 0. and previous_difference > 0.)
                    or (difference < 0. and previous_difference < 0.))
            status = 'confirmed' if same else 'unconfirmed'
            record['status'] = record['reason'] = status
            counts[status+'_rows'] += 1
            if not same:
                out['confirmed'][j] = pooled[j]
            out['half'][j] = _finite(float(pooled[j])+0.5*difference, 'half correction')
        records.append(record)
    return dict(raw=out, available=available.copy(), audit=dict(
        method='previous_formal_month_direction_confirmation', counts=counts, records=records,
        new_model_fits=0, predictive_model_fits=0, new_ridge_fits=0,
        new_tree_fits=0, new_parameters=0))
