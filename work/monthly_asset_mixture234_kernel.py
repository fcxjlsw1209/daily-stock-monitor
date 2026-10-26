"""Pure common-score stock/ETF mixture calibration of original auxiliary support.

Cached source validation and S/W arithmetic preserve round210's contract locally.
No source loading, model, order selection, eligibility filtering or return replay.
The host adds only nonzero deltas to original pre-score and keeps the original
support coordinate when it applies the original episode exactly once.
"""
import numbers

import numpy as np

REQUIRED = {'baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow',
            'extension_available', 'forecast_available', 'eligible', 'rank_log_excess'}
VALUE_KEYS = ('support', 'weight', 'candidate', 'control', 'candidate_delta', 'control_delta')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells', 'stock_cells', 'etf_cells',
    'both_group_rows', 'single_group_rows', 'constant_rows', 'active_forecast_cells',
    'group_cdf_evaluations', 'pooled_cdf_evaluations', 'new_model_fits', 'predictive_model_fits')
BOUND_TOL = 1e-12


def _prefix(value, shape, n, name, boolean=False, finite=True):
    source = np.asarray(value)
    if source.shape != shape:
        raise ValueError(name + ' shape differs from original event axes')
    # Slice before inspecting values or coercing numeric/object storage. Future
    # poison must not be inspected just because it shares a source array.
    raw = source[:n]
    if boolean:
        if any(not isinstance(x, (bool, np.bool_)) for x in raw.flat):
            raise ValueError(name + ' requires exact boolean values')
        return raw.astype(bool, copy=True)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw.flat):
        raise ValueError(name + ' requires real nonboolean numeric values')
    try:
        result = raw.astype(np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name + ' cannot be represented as float64') from exc
    if finite and not np.isfinite(result).all():
        raise ValueError(name + ' requires finite values')
    return result


def _sources(arrays, limit):
    if not isinstance(arrays, dict) or not REQUIRED <= set(arrays):
        raise ValueError('Original cached support, weights, states and masks required')
    shape = np.asarray(arrays['baseline']).shape
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError('Original baseline requires an event matrix of positive width')
    rows, width = shape
    if limit is not None and (type(limit) is not int or not 0 <= limit <= rows):
        raise ValueError('limit must be an integer event prefix')
    n = rows if limit is None else limit
    baseline = _prefix(arrays['baseline'], shape, n, 'baseline')
    core = _prefix(arrays['core'], shape, n, 'core')
    calm = _prefix(arrays['state_calm10'], (rows,), n, 'state_calm10', boolean=True)
    weak = _prefix(arrays['state_weak_narrow'], (rows,), n, 'state_weak_narrow', boolean=True)
    raw_weight = np.asarray(arrays['core_weight'])
    if raw_weight.shape == (rows,):
        raw_weight = raw_weight[:, None]
    if raw_weight.shape not in ((rows, 1), shape):
        raise ValueError('core_weight shape must be n, n-by-1 or n-by-width')
    cw = np.broadcast_to(_prefix(raw_weight, raw_weight.shape, n, 'core_weight'), (n, width)).copy()
    expected = np.broadcast_to(np.where(calm, .50, .52)[:, None], cw.shape)
    if not np.array_equal(cw, expected):
        raise ValueError('core_weight disagrees with original calm state')
    extension = _prefix(arrays['extension_available'], shape, n, 'extension_available', boolean=True)
    forecast = _prefix(arrays['forecast_available'], shape, n, 'forecast_available', boolean=True)
    eligible = _prefix(arrays['eligible'], shape, n, 'eligible', boolean=True)
    logrank = _prefix(arrays['rank_log_excess'], shape, n, 'rank_log_excess', finite=False)
    active = calm[:, None] & ~weak[:, None] & extension & forecast & eligible & np.isfinite(logrank)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        support = (baseline - core) / (1. - cw)
        weight = (1. - cw) - (.025 * calm[:, None] + .05 * weak[:, None] + .01 * active)
    if not np.isfinite(support).all() or (np.abs(support) > .5 + BOUND_TOL).any():
        raise ValueError('Original auxiliary support exceeds finite bounded domain')
    if not np.isfinite(weight).all() or (weight <= 0.).any():
        raise ValueError('Effective original support weight must be finite positive')
    return support, weight, active


def _groups(meta, rows, width, n):
    if not isinstance(meta, dict) or not {'symbols', 'etfs'} <= set(meta):
        raise ValueError('Explicit symbol rows and ETF identity list required')
    etfs = meta['etfs']
    if (not isinstance(etfs, list) or
            any(not isinstance(x, str) or not x for x in etfs) or
            len(set(etfs)) != len(etfs)):
        raise ValueError('etfs must be an explicit unique nonempty-string list')
    symbols = meta['symbols']
    if not isinstance(symbols, list) or len(symbols) != rows:
        raise ValueError('symbols must retain the original full event axis')
    selected = []
    for row in symbols[:n]:
        if (not isinstance(row, list) or len(row) != width or
                any(not isinstance(x, str) or not x for x in row) or
                len(set(row)) != width):
            raise ValueError('Each consumed symbols row must contain unique explicit identities')
        selected.append(row.copy())
    return selected, set(etfs)


def _maximum(values):
    return float(np.abs(values).max()) if len(values) else 0.


def build_features(arrays, meta, limit=None):
    """Return six finite float64 matrices over the unfiltered original universe.

    Fstock and Fetf are evaluated at the SAME support value for every security.
    Pooled control uses one direct integer-count formula; candidate only copies
    it when a group is absent. A full constant row preserves the old support via
    zero deltas. A constant subgroup in a varied row remains a valid step CDF.
    """
    support, weight, active = _sources(arrays, limit)
    n, width = support.shape
    symbols, etfs = _groups(meta, np.asarray(arrays['baseline']).shape[0], width, n)
    values = {key: np.zeros((n, width), dtype=np.float64) for key in VALUE_KEYS}
    values['support'], values['weight'] = support, weight
    history = []
    stocks = funds = both_rows = single_rows = constants = group_evals = pooled_evals = 0
    for i, (row, names) in enumerate(zip(support, symbols)):
        stock_ids = [j for j, name in enumerate(names) if name not in etfs]
        etf_ids = [j for j, name in enumerate(names) if name in etfs]
        ns, ne = len(stock_ids), len(etf_ids)
        stocks += ns; funds += ne
        single = ns == 0 or ne == 0
        single_rows += int(single); both_rows += int(not single)
        levels = int(len(np.unique(row)))
        constant = levels == 1
        constants += int(constant)
        counts = {key: [] for key in ('stock_less', 'stock_equal', 'etf_less', 'etf_equal',
                                     'pooled_less', 'pooled_equal')}
        branch = 'constant_row' if constant else 'single_group_copy' if single else 'equal_mixture'
        if not constant:
            stock = row[stock_ids]; etf = row[etf_ids]
            for j, x in enumerate(row):
                sl, se = int(np.count_nonzero(stock < x)), int(np.count_nonzero(stock == x))
                el, ee = int(np.count_nonzero(etf < x)), int(np.count_nonzero(etf == x))
                pl, pe = sl + el, se + ee
                for key, count in (('stock_less', sl), ('stock_equal', se), ('etf_less', el),
                                   ('etf_equal', ee), ('pooled_less', pl), ('pooled_equal', pe)):
                    counts[key].append(count)
                # Context fixes this literal count arithmetic; do not replace
                # pooled control with a differently associated weighted mixture.
                control = (pl + .5 * pe) / width - .5
                if single:
                    candidate = control
                else:
                    fs = (sl + .5 * se) / ns
                    fe = (el + .5 * ee) / ne
                    candidate = .5 * fs + .5 * fe - .5
                values['candidate'][i, j] = candidate
                values['control'][i, j] = control
            pooled_evals += width
            group_evals += 0 if single else 2 * width
            if i:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    values['candidate_delta'][i] = weight[i] * (values['candidate'][i] - row)
                    values['control_delta'][i] = weight[i] * (values['control'][i] - row)
        history.append(dict(query_index=i, symbols=names, stock_indices=stock_ids,
            etf_indices=etf_ids, stock_count=ns, etf_count=ne, distinct_levels=levels,
            constant=constant, single_group=single, branch=branch,
            active_forecast_count=int(active[i].sum()), **counts,
            max_absolute_candidate_delta=_maximum(values['candidate_delta'][i]),
            max_absolute_control_delta=_maximum(values['control_delta'][i]),
            max_absolute_candidate_control_difference=_maximum(values['candidate'][i]-values['control'][i])))
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite mixture auxiliary transform')
    for key in ('candidate', 'control'):
        if (np.abs(values[key]) > .5 + BOUND_TOL).any():
            raise FloatingPointError('CDF target exceeds bounded domain')
    return values, dict(schema='asset_mixture234_features_v1',events=n,symbols_per_event=width,
        checked_cells=n*width,stock_cells=stocks,etf_cells=funds,both_group_rows=both_rows,
        single_group_rows=single_rows,constant_rows=constants,active_forecast_cells=int(active.sum()),
        group_cdf_evaluations=group_evals,pooled_cdf_evaluations=pooled_evals,
        new_model_fits=0,predictive_model_fits=0,ranking_eligibility_filtered=False,
        new_availability_filter=False,portfolio_returns_calculated=False,event_history=history)


build = build_features
