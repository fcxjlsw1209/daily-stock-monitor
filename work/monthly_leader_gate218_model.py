"""Fixed six-leader choice set followed by the unchanged baseline score.

Pure prefix-only portfolio construction. No I/O, fitting, labels, forecasts,
state carry, episode logic, return calculation, or strategy-driver imports.
The host validates formal signal dates against its pinned trading calendar.
"""
from datetime import date
from pathlib import Path
import math

import numpy as np

ARMS = ('leader6_gate', 'baseline_score')
INITIAL = '2016-08-17'
COUNT_FIELDS = (
    'query_months', 'eligible_cells', 'shortlisted_cells', 'gated_months',
    'tie_expansion_months', 'tie_added_cells', 'changed_holding_months',
    'changed_ordered_months', 'cash_months', 'two_holding_months',
    'three_holding_months',
)


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
    return int(value[:4]) * 12 + int(value[5:7]) - 1


def _numeric(values, shape, n, name):
    value = np.asarray(values)
    _require(value.shape == shape, name + ' shape differs')
    _require(value.dtype.kind in 'fiu', name + ' must have real nonboolean numeric dtype')
    with np.errstate(over='ignore', invalid='ignore'):
        prefix = value[:n].astype(np.float64, copy=False)
    _require(np.isfinite(prefix).all(), name + ' prefix must be finite')
    return prefix


def _inputs(arrays, meta, n):
    _require(isinstance(arrays, dict) and isinstance(meta, dict),
             'Arrays and metadata must be dictionaries')
    _require({'baseline', 'core', 'core_weight', 'eligible'} <= set(arrays),
             'Original baseline/core/core_weight/eligible required')
    _require({'dates', 'ends', 'symbols'} <= set(meta),
             'Exact dates/ends/symbols required')
    baseline = np.asarray(arrays['baseline'])
    _require(baseline.ndim == 2 and baseline.shape[1] > 0,
             'Baseline must have nonempty candidate width')
    total, width = baseline.shape
    _require(isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))
             and 0 <= n <= total, 'Invalid monthly prefix length')
    n = int(n)
    baseline = _numeric(baseline, (total, width), n, 'baseline')
    core = _numeric(arrays['core'], (total, width), n, 'core')
    weight = _numeric(arrays['core_weight'], (total, 1), n, 'core_weight')
    _require((weight > 0).all(), 'core_weight prefix must be strictly positive')
    eligible = np.asarray(arrays['eligible'])
    _require(eligible.shape == (total, width) and eligible.dtype.kind == 'b',
             'Eligibility must exactly match original boolean matrix')
    eligible = eligible[:n]
    days, ends, symbols = (meta[k] for k in ('dates', 'ends', 'symbols'))
    _require(all(isinstance(v, list) and len(v) == total for v in (days, ends, symbols)),
             'Metadata row axes differ from arrays')
    formal_months = set()
    for i in range(n):
        day, end = _day(days[i]), _day(ends[i])
        partial = day == INITIAL
        _require(not partial or (i == 0 and end == '2016-08-31'),
                 'Initial partial must preserve original first interval')
        _require(end > day and (partial or _month(end) == _month(day) + 1),
                 'Formal holding interval must end in the next calendar month')
        _require(i == 0 or day > days[i-1], 'Signal dates must increase uniquely')
        if not partial:
            month = _month(day)
            _require(month not in formal_months, 'Duplicate formal signal month')
            formal_months.add(month)
        row_symbols = symbols[i]
        _require(isinstance(row_symbols, list) and len(row_symbols) == width,
                 'Security width differs from arrays')
        _require(all(isinstance(s, str) and s and s == s.strip() for s in row_symbols)
                 and len(set(row_symbols)) == width,
                 'Exact unique nonempty security IDs required')
    return baseline, core, weight, eligible, days[:n], ends[:n], symbols[:n]


def _chosen(indices, scores, symbols):
    ordered = sorted(indices, key=lambda j: (-float(scores[j]), symbols[j]))
    return ordered[:3] if len(ordered) >= 2 else []


def _order(day, end, ids, scores, symbols):
    return dict(date=day, end=end, signal_date=day,
                symbols=[symbols[j] for j in ids],
                scores=[float(scores[j]) for j in ids])


def build(arrays, meta, n):
    """Construct both fixed arms and their shortlist from the requested prefix.

    The sixth boundary is measured only within original E. Every exact tie is
    included; the final selection uses baseline scores and lexical symbol ties.
    Cached neutral core scores remain valid. Invalid consumed caches fail the
    whole call rather than silently changing eligibility.
    """
    baseline, core, weight, eligible, days, ends, symbols = _inputs(arrays, meta, n)
    n, width = baseline.shape
    shortlist = np.zeros((n, width), dtype=bool)
    orders = {arm: [] for arm in ARMS}
    counts = {key: 0 for key in COUNT_FIELDS}
    counts['query_months'] = n
    records = []
    for i in range(n):
        ids = [int(j) for j in np.flatnonzero(eligible[i])]
        ratios = {}
        for j in ids:
            ratio = float(core[i, j]) / float(weight[i, 0])
            _require(math.isfinite(ratio), 'Eligible normalized core overflow')
            ratios[j] = ratio
        threshold, tie_count = None, 0
        if len(ids) > 6:
            ordered_core = sorted(ids, key=lambda j: (-ratios[j], symbols[i][j]))
            threshold = ratios[ordered_core[5]]
            candidates = [j for j in ids if ratios[j] >= threshold]
            tie_count = sum(ratios[j] == threshold for j in ids)
        else:
            candidates = ids[:]
        shortlist[i, candidates] = True
        chosen = _chosen(candidates, baseline[i], symbols[i])
        control = _chosen(ids, baseline[i], symbols[i])
        orders['leader6_gate'].append(_order(days[i], ends[i], chosen, baseline[i], symbols[i]))
        orders['baseline_score'].append(_order(days[i], ends[i], control, baseline[i], symbols[i]))
        leader_names = [symbols[i][j] for j in chosen]
        control_names = [symbols[i][j] for j in control]
        changed_holdings = set(leader_names) != set(control_names)
        changed_order = leader_names != control_names
        added = max(len(candidates) - 6, 0)
        records.append(dict(
            row=i, date=days[i], eligible_count=len(ids), shortlist_count=len(candidates),
            sixth_core_threshold=threshold, sixth_tie_count=tie_count, tie_added_count=added,
            leader6_symbols=leader_names, baseline_symbols=control_names,
            changed_holdings=changed_holdings, changed_order=changed_order,
        ))
        counts['eligible_cells'] += len(ids)
        counts['shortlisted_cells'] += len(candidates)
        counts['gated_months'] += int(len(ids) > 6)
        counts['tie_expansion_months'] += int(added > 0)
        counts['tie_added_cells'] += added
        counts['changed_holding_months'] += int(changed_holdings)
        counts['changed_ordered_months'] += int(changed_order)
        counts['cash_months'] += int(len(ids) < 2)
        counts['two_holding_months'] += int(len(ids) == 2)
        counts['three_holding_months'] += int(len(ids) >= 3)
    audit = dict(method='eligible_core_top6_then_baseline_top3', counts=counts, records=records,
                 new_model_fits=0, predictive_model_fits=0, new_ridge_fits=0,
                 new_tree_fits=0, new_parameters=0)
    return dict(orders=orders, shortlist=shortlist, audit=audit)
