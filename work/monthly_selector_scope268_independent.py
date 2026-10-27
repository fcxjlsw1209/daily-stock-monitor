"""Caller-input, unregistered independent selector-scope268 checker.

This module performs no file loading, model fitting, source admission or replay.
It independently reconstructs the fixed selector using scalar binary64 loops;
it does not import the producer. The frozen selector definition and NumPy
arithmetic semantics were inspected, so this is not a blind or library-diverse
review. A passed receipt validates supplied arrays, not their provenance.
"""
from datetime import date
import math
import numpy as np


SCHEMA = 'selector_scope268_independent_v1'
MODES = ('incumbent_local67', 'unlocked_top5_gap', 'unlocked_all')
INITIAL_PARTIAL = '2016-08-17'
FLAGS = ('original_rank_and_symbol_ties_exact', 'scope_and_gap_exact',
         'existing_support_budget_arithmetic_exact',
         'partial_and_small_universe_identity_exact',
         'scores_float64_bytes_exact', 'ordered_choices_exact')
WORK_FIELDS = ('selector_calls', 'event_rows', 'eligible_cells',
               'source_rank_calls', 'support_evaluations', 'delta_evaluations',
               'selection_rank_calls', 'partial_events', 'two_eligible_events',
               'cash_events', 'new_model_fits')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _matrix(value, shape, name, dtype=np.float64):
    need(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
         and value.shape == shape, name + ' exact matrix shape/dtype')


def _finite(value, name):
    result = float(value)
    need(math.isfinite(result), name + ' finite binary64')
    return result


def verify(pre, eligible, symbols, effects, baseline, core, core_weight,
           state_calm10, state_weak_narrow, dates, choices, scores, mode):
    """Verify one supplied mode, returning a descriptive execution receipt.

    Numeric matrices must already be canonical native float64 N-by-K, including
    core_weight; the host owns validation/broadcast of the original weight
    source. Eligibility is bool N-by-K and the two states are bool N-vectors.
    Only eligible P and effective adjustment cells are numerically consumed.
    Untouched output cells, including NaN payloads and signed zeros, are checked
    byte-for-byte. Dates/symbols are metadata and are checked over this prefix.
    This routine neither authenticates source files nor issues a capability.
    """
    need(type(mode) is str and mode in MODES, 'Declared selector mode')
    need(isinstance(pre, np.ndarray) and pre.dtype == np.dtype(np.float64)
         and pre.ndim == 2, 'P native float64 matrix')
    n, width = pre.shape
    need(width > 0, 'Positive source symbol width including empty prefixes')
    shape = (n, width)
    _matrix(eligible, shape, 'Eligibility', np.bool_)
    for key, value in (('Effects', effects), ('Baseline', baseline),
                       ('Core', core), ('Core weight', core_weight),
                       ('Scores', scores)):
        _matrix(value, shape, key)
    for key, value in (('Calm state', state_calm10),
                       ('Weak state', state_weak_narrow)):
        need(isinstance(value, np.ndarray) and value.dtype == np.dtype(np.bool_)
             and value.shape == (n,), key + ' bool event vector')
    need(isinstance(symbols, (list, tuple)) and len(symbols) == n,
         'Symbol event axis')
    need(isinstance(dates, (list, tuple)) and len(dates) == n, 'Date event axis')
    previous = None
    for i in range(n):
        names = symbols[i]
        need(isinstance(names, (list, tuple)) and len(names) == width
             and all(type(s) is str and s for s in names)
             and len(set(names)) == width, 'Unique complete row symbol identity')
        day = dates[i]
        need(type(day) is str, 'ISO date string')
        try:
            parsed = date.fromisoformat(day)
        except (TypeError, ValueError):
            raise ValueError('ISO date string') from None
        need(parsed.isoformat() == day and (previous is None or day > previous),
             'Strict chronological canonical dates')
        need(day != INITIAL_PARTIAL or i == 0, 'Named initial partial only first')
        previous = day
    need(type(choices) is list and len(choices) == n, 'Ordered choices event axis')
    normalized = []
    for selected in choices:
        need(type(selected) is list and all(isinstance(j, (int, np.integer))
             and not isinstance(j, (bool, np.bool_)) and 0 <= int(j) < width
             for j in selected), 'Ordered integer choices')
        normalized.append([int(j) for j in selected])

    work = dict.fromkeys(WORK_FIELDS, 0)
    work['selector_calls'] = 1
    expected = pre.copy()
    expected_choices = []
    for i in range(n):
        work['event_rows'] += 1
        names = symbols[i]
        members = [j for j in range(width) if bool(eligible[i, j])]
        work['eligible_cells'] += len(members)
        for j in members:
            _finite(pre[i, j], 'Eligible P')
        order = sorted(members, key=lambda j: (-float(pre[i, j]), names[j]))
        work['source_rank_calls'] += 1
        partial = dates[i] == INITIAL_PARTIAL
        work['partial_events'] += int(partial)
        if len(order) < 3:
            expected_choices.append(order if len(order) == 2 else [])
            work['two_eligible_events'] += int(len(order) == 2)
            work['cash_events'] += int(len(order) < 2)
            continue
        boundary = []
        p3 = float(pre[i, order[2]])
        for j in order[2:5]:
            gap = _finite(p3 - float(pre[i, j]), 'Original P3 gap')
            if gap <= .02:
                boundary.append(j)
        effective_mode = MODES[0] if partial else mode
        if effective_mode == MODES[0]:
            domain = boundary
        elif effective_mode == MODES[1]:
            domain = order[:2] + boundary
        else:
            domain = order
        old_budget = .025 * int(state_calm10[i]) + .05 * int(state_weak_narrow[i])
        budget = min(.05, max(0., .10 - old_budget))
        for j in domain:
            weight = _finite(core_weight[i, j], 'Consumed core weight')
            need(.5 <= weight < 1., 'Consumed core dominance and denominator')
            b = _finite(baseline[i, j], 'Consumed baseline')
            c = _finite(core[i, j], 'Consumed core')
            difference = _finite(b - c, 'Support numerator')
            support = _finite(difference / (1. - weight), 'Support')
            effect = _finite(effects[i, j], 'Consumed effect')
            shifted = _finite(support + effect, 'Support plus effect')
            target = min(.5, max(-.5, shifted))
            movement = _finite(target - support, 'Target minus support')
            delta = _finite(budget * movement, 'Budgeted delta')
            expected[i, j] = _finite(float(pre[i, j]) + delta, 'Final score')
            work['support_evaluations'] += 1
            work['delta_evaluations'] += 1
        if effective_mode == MODES[0]:
            winner = min(boundary, key=lambda j: (-float(expected[i, j]), names[j]))
            selected = order[:2] + [winner]
        else:
            selected = domain
        ranked = sorted(selected, key=lambda j: (-float(expected[i, j]), names[j]))
        expected_choices.append(ranked[:3])
        work['selection_rank_calls'] += 1
    need(normalized == expected_choices, 'Exact ordered choices')
    need(scores.tobytes(order='C') == expected.tobytes(order='C'),
         'Exact score binary64 bytes including NaN payload and signed zero')
    return dict(schema=SCHEMA, passed=True, mode=mode, events=n, width=width,
                flags=dict.fromkeys(FLAGS, True), work=work,
                numerical_tolerance=0., actual_coverage_claim=False,
                source_provenance_verified=False)
