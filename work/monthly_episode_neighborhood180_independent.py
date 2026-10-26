"""Independent scalar selector audit for round180 and frozen outcome accounting.

This module does not import a production kernel, host, or strategy engine.  It
has no entry point and performs no source loading, model fitting, or writes.
The caller is responsible for binding raw-source and official-calendar hashes.
"""
from datetime import date
import hashlib
import importlib.util
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'score_neighborhood_episode', 'uniform_episode')
KEYS = ('candidate', 'control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
WIDTH = .02
BUDGET = .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('neighborhood180_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('neighborhood180_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': mapping')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (got, wanted) in enumerate(zip(actual, expected)):
            record_check(got, wanted, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def number(value):
    return float(value) if math.isfinite(value) else None


def inputs(pre, arrays, meta, limit):
    """Validate structure globally, convert only the requested chronological prefix."""
    pre = np.asarray(pre)
    require(pre.ndim == 2 and pre.shape[1] > 0, 'Nonempty-width pre matrix required')
    rows, width = pre.shape
    require(limit is None or (isinstance(limit, Integral) and not isinstance(limit, (bool, np.bool_)) and
                             0 <= limit <= rows), 'Invalid prefix limit')
    n = rows if limit is None else int(limit)
    require(isinstance(arrays, dict), 'Original arrays mapping required')
    out = {'pre': np.array(pre[:n], dtype=float, copy=True)}
    for key in ('baseline', 'core', 'effects', 'eligible'):
        value = np.asarray(arrays[key])
        require(value.shape == pre.shape, 'Original matrix shape: ' + key)
        if key == 'eligible':
            require(value.dtype == bool, 'Boolean eligibility required')
        out[key] = np.array(value[:n], dtype=bool if key == 'eligible' else float, copy=True)
    for key in ('state_calm10', 'state_weak_narrow'):
        value = np.asarray(arrays[key])
        require(value.shape == (rows,) and value.dtype == bool, 'Boolean state vector: ' + key)
        out[key] = value[:n].copy()
    weight = np.asarray(arrays['core_weight'])
    require(weight.shape in ((rows,), (rows, 1), pre.shape), 'Original core weight shape')
    prefix = np.array(weight[:n], dtype=float, copy=True)
    if prefix.ndim == 1:
        prefix = prefix[:, None]
    require(np.isfinite(prefix).all() and ((prefix > 0.) & (prefix < 1.)).all(), 'Core weights outside (0,1)')
    out['core_weight'] = np.broadcast_to(prefix, (n, width))
    require(not np.isinf(out['effects']).any(), 'Infinite episode effect')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols')), 'Event metadata required')
    require(len(meta['dates']) == rows and len(meta['symbols']) == rows, 'Event metadata lengths')
    dates, names = [], []
    for raw, symbols in zip(meta['dates'][:n], meta['symbols'][:n]):
        require(isinstance(raw, str), 'ISO event date required')
        day = date.fromisoformat(raw)
        require(day.isoformat() == raw and (not dates or raw > dates[-1]), 'Event date/order')
        require(isinstance(symbols, (list, tuple)) and len(symbols) == width and len(set(symbols)) == width and
                all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Event identities')
        dates.append(raw); names.append(list(symbols))
    mask = out['eligible']
    for key in ('pre', 'baseline', 'core'):
        require(np.isfinite(out[key][mask]).all(), 'Nonfinite eligible ' + key)
    return out, dates, names


def gaussian(distance):
    """Evaluate the bounded Gaussian even when the squared distance overflows."""
    if not math.isfinite(distance):
        return 0.
    ratio = distance / WIDTH
    square = ratio * ratio
    return math.exp(-.5 * square) if math.isfinite(square) else 0.


def independent_selection(pre, arrays, meta, limit=None):
    x, dates, names = inputs(pre, arrays, meta, limit)
    shape = x['pre'].shape
    scores = {key: x['pre'].copy() for key in KEYS}
    corrections = {key: np.zeros(shape, float) for key in KEYS}
    choices = {key: [] for key in KEYS}
    history = []
    eligible_count = finite_count = missing_count = active_count = low_count = 0
    for i, (day, symbols) in enumerate(zip(dates, names)):
        eligible = [j for j in range(shape[1]) if x['eligible'][i, j]]
        order = sorted(eligible, key=lambda j: (-x['pre'][i, j], symbols[j]))
        third = float(x['pre'][i, order[2]]) if len(order) >= 3 else None
        budget = min(BUDGET, max(0., .10 - .025 * bool(x['state_calm10'][i]) - .05 * bool(x['state_weak_narrow'][i])))
        active = i > 0 and len(order) >= 3
        suppression = 'initial_row' if i == 0 else ('fewer_than_three_eligible' if len(order) < 3 else None)
        active_count += int(active)
        low_count += int(len(order) < 3)
        eligible_count += len(eligible)
        cells = []
        for j, symbol in enumerate(symbols):
            good = bool(x['eligible'][i, j])
            p, effect = float(x['pre'][i, j]), float(x['effects'][i, j])
            b, c, weight = float(x['baseline'][i, j]), float(x['core'][i, j]), float(x['core_weight'][i, j])
            support = (b - c) / (1. - weight) if good else None
            if good and not math.isfinite(support):
                raise FloatingPointError('Nonfinite eligible original support')
            finite_count += int(good and math.isfinite(effect))
            missing_count += int(good and math.isnan(effect))
            unweighted = near = 0.
            if active and good:
                near = gaussian(p - third)
                if math.isfinite(effect):
                    shifted = support + effect
                    if not math.isfinite(shifted):
                        raise FloatingPointError('Nonfinite support-plus-effect before clipping')
                    clipped = max(-.5, min(.5, shifted))
                    unweighted = budget * (clipped - support)
                    if not math.isfinite(unweighted):
                        raise FloatingPointError('Nonfinite episode correction')
                    corrections['control'][i, j] = unweighted
                    corrections['candidate'][i, j] = near * unweighted
            for key in KEYS:
                if corrections[key][i, j] != 0.:
                    scores[key][i, j] += corrections[key][i, j]
                if good and not math.isfinite(scores[key][i, j]):
                    raise FloatingPointError('Nonfinite eligible final score')
            cells.append(dict(symbol=symbol, eligible=good, pre_score=number(p), support=number(support) if good else None,
                effect=number(effect), unweighted_correction=unweighted, neighborhood_weight=near,
                candidate_correction=float(corrections['candidate'][i, j]),
                control_correction=float(corrections['control'][i, j]),
                candidate_score=number(scores['candidate'][i, j]), control_score=number(scores['control'][i, j])))
        row_choices = {}
        for key in KEYS:
            chosen = sorted(eligible, key=lambda j: (-scores[key][i, j], symbols[j]))[:3] if len(eligible) >= 2 else []
            choices[key].append(chosen); row_choices[key] = chosen
        history.append(dict(date=day, symbols=symbols, eligible_indices=eligible, pre_order=order,
            third_score=third, budget=budget, active=active, suppression_reason=suppression,
            choices=row_choices, cells=cells))
    result = dict(scores=scores, corrections=corrections, choices=choices, eligible=x['eligible'].copy())
    audit = dict(events=shape[0], symbols_per_event=shape[1], checked_cells=int(x['eligible'].size),
        eligible_cells=eligible_count, finite_effect_eligible_cells=finite_count,
        missing_effect_eligible_cells=missing_count, active_rows=active_count,
        initial_rows_suppressed=int(shape[0] > 0), low_cardinality_rows=low_count,
        correction_cells={key: int(np.count_nonzero(corrections[key])) for key in KEYS},
        predictive_model_fits=0, portfolio_returns_calculated=False, eligibility_modified=False,
        episode_applications=1, gaussian_width=WIDTH, budget_cap=BUDGET, event_history=history)
    return result, audit


def verify_selection(pre, arrays, meta, result, audit, limit=None):
    wanted, expected = independent_selection(pre, arrays, meta, limit)
    require(isinstance(result, dict) and set(result) == set(wanted), 'Selector result keys differ')
    require(np.asarray(result['eligible']).dtype == bool and np.array_equal(result['eligible'], wanted['eligible']),
            'Original eligibility changed')
    for category in ('scores', 'corrections', 'choices'):
        require(isinstance(result[category], dict) and set(result[category]) == set(KEYS), 'Selector arm keys differ')
        for key in KEYS:
            if category == 'choices':
                record_check(result[category][key], wanted[category][key], 'choices/' + key)
            else:
                # Ineligible scores remain exactly what the caller supplied, including NaN.
                if category == 'scores':
                    actual = np.asarray(result[category][key], float)
                    require(actual.shape == wanted[category][key].shape, 'Score shape changed')
                    mask = wanted['eligible']
                    close(actual[mask], wanted[category][key][mask], category + '/' + key)
                    require(np.array_equal(actual[~mask], wanted[category][key][~mask], equal_nan=True),
                            'Ineligible score changed')
                else:
                    close(result[category][key], wanted[category][key], category + '/' + key)
                if category == 'corrections':
                    inactive = wanted[category][key] == 0.
                    require(np.array_equal(np.asarray(result[category][key])[inactive], np.zeros(inactive.sum())),
                            'Inactive correction must be exactly zero')
    require(audit.get('gaussian_width') == WIDTH and audit.get('budget_cap') == BUDGET, 'Fixed kernel constants differ')
    record_check(audit, expected)
    return dict(passed=True, events=expected['events'], symbols_per_event=expected['symbols_per_event'],
        checked_cells=expected['checked_cells'], eligible_cells=expected['eligible_cells'],
        correction_cells=expected['correction_cells'], independent_scalar_rebuild=True,
        original_support_checked=True, clipped_episode_budget_checked=True,
        gaussian_neighborhood_checked=True, global_top3_checked=True,
        first_partial_neutral_checked=True, exact_cardinality_checked=True,
        no_future_or_outcome_inputs=True, audit_checked=True, eligibility_unchanged=True,
        episode_applications=1, predictive_model_fits=0, computes_returns=False)


