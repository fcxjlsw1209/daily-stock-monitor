"""Pure selector-scope experiment. No source loading, fitting or returns.

The caller authenticates original Local67 P/E/effects/core and passes only the
consumed prefix. This module changes the adjustment/selection domain, never
the effects, core, support formula or existing episode budget. It does not
authenticate data merely because a source has the expected shape.
"""
from datetime import date

import numpy as np


ARMS = ('incumbent_local67', 'unlocked_top5_gap', 'unlocked_all')
INITIAL_PARTIAL = '2016-08-17'
POLICY = dict(budget=.05, ceiling=5, gap=.02)
AUDIT_KEYS = ('schema', 'mode', 'counts', 'events', 'original_inputs_preserved',
              'long_core_unchanged', 'original_eligibility_unchanged',
              'existing_effects_only', 'initial_partial_exact', 'computes_returns')
COUNT_FIELDS = ('events', 'symbols_per_event', 'eligible_cells',
                'initial_partial_events', 'regular_events', 'three_or_more_events',
                'two_name_events', 'cash_events', 'original_boundary_cells',
                'adjusted_cells', 'nonzero_delta_cells', 'changed_score_cells',
                'top2_exit_events', 'original_rankings', 'final_rankings',
                'boundary_winner_reductions', 'selector_calls',
                'new_model_fits', 'predictive_model_fits')
EVENT_KEYS = ('query_index', 'date', 'initial_partial', 'effective_mode',
              'eligible_indices', 'original_rank', 'original_boundary_indices',
              'adjustment_indices', 'selection_indices', 'selected_indices',
              'top2_exited_indices', 'episode_budget', 'adjustment_support',
              'adjustment_effects', 'adjustment_targets', 'adjustment_deltas',
              'changed_score_indices')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _inputs(pre, eligible, symbols, effects, baseline, core, core_weight,
            state_calm10, state_weak_narrow, dates, mode):
    need(type(mode) is str and mode in ARMS, 'Registered selector mode')
    need(isinstance(pre, np.ndarray) and pre.ndim == 2 and pre.dtype == np.dtype('float64'),
         'pre must be a float64 [events, symbols] array')
    n, width = pre.shape
    need(width > 0, 'Positive symbol width, including empty prefixes')
    for name, value in (('effects', effects), ('baseline', baseline),
                        ('core', core), ('core_weight', core_weight)):
        need(isinstance(value, np.ndarray) and value.shape == pre.shape
             and value.dtype == np.dtype('float64'), name + ' must be matching float64 matrix')
    need(isinstance(eligible, np.ndarray) and eligible.shape == pre.shape
         and eligible.dtype == np.dtype(bool), 'eligible must be matching bool matrix')
    for name, value in (('calm', state_calm10), ('weak', state_weak_narrow)):
        need(isinstance(value, np.ndarray) and value.shape == (n,)
             and value.dtype == np.dtype(bool), name + ' must be bool [events]')
    need(type(symbols) in (list, tuple) and len(symbols) == n, 'Symbols event axis')
    need(type(dates) in (list, tuple) and len(dates) == n, 'Dates event axis')
    last = None
    for i in range(n):
        ss = symbols[i]
        need(type(ss) in (list, tuple) and len(ss) == width, 'Symbols width')
        need(all(type(s) is str and bool(s) for s in ss) and len(set(ss)) == width,
             'Unique nonempty native symbol strings')
        day = dates[i]
        need(type(day) is str, 'Native ISO date string')
        try:
            parsed = date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError('Valid ISO date') from exc
        need(parsed.isoformat() == day and (last is None or day > last), 'Strictly increasing ISO dates')
        need(day != INITIAL_PARTIAL or i == 0, 'Initial partial may occur only at index zero')
        last = day
    return n, width


def select(pre, eligible, symbols, effects, baseline, core, core_weight,
           state_calm10, state_weak_narrow, dates, mode):
    """Select from already authenticated, physically sliced prefix matrices.

    All numeric matrices are float64 N x K (the host canonicalizes original
    core-weight broadcasting). Values outside the adjustment domain are not
    inspected, except P for every eligible name needed by the original sort.
    A zero effect is not globally an identity: clipping support and IEEE-754
    addition still follow the original selector. No extra epsilon is used.
    """
    n, width = _inputs(pre, eligible, symbols, effects, baseline, core,
                       core_weight, state_calm10, state_weak_narrow, dates, mode)
    scores = pre.copy()
    counts = {key: 0 for key in COUNT_FIELDS}
    counts.update(events=n, symbols_per_event=width, selector_calls=1)
    oldbudget = .025 * state_calm10 + .05 * state_weak_narrow
    budgets = np.minimum(.05, np.maximum(0., .10 - oldbudget))
    choices, events = [], []
    for i in range(n):
        ss = symbols[i]
        use = np.flatnonzero(eligible[i]).tolist()
        need(all(np.isfinite(pre[i, j]) for j in use), 'Finite eligible original P required')
        rank = sorted(use, key=lambda j: (-pre[i, j], ss[j]))
        partial = dates[i] == INITIAL_PARTIAL
        effective = ARMS[0] if partial else mode
        counts['eligible_cells'] += len(use)
        counts['initial_partial_events' if partial else 'regular_events'] += 1
        counts['original_rankings'] += 1
        boundary, adjust, domain, supports, es, targets, deltas = [], [], [], [], [], [], []
        if len(rank) < 3:
            selected = rank if len(rank) == 2 else []
            counts['two_name_events' if len(rank) == 2 else 'cash_events'] += 1
        else:
            counts['three_or_more_events'] += 1
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                for j in rank[2:5]:
                    gap = pre[i, rank[2]] - pre[i, j]
                    need(np.isfinite(gap), 'Finite original P gap required')
                    if gap <= .02:
                        boundary.append(j)
            if effective == ARMS[0]:
                adjust, domain = boundary, rank[:2] + boundary
            elif effective == ARMS[1]:
                adjust = rank[:2] + boundary
                domain = adjust
            else:
                adjust, domain = rank, rank
            for j in adjust:
                need(all(np.isfinite(x[i, j]) for x in (baseline, core, core_weight, effects)),
                     'Finite consumed support/core/weight/effect required')
                need(.5 <= core_weight[i, j] < 1., 'Consumed original core weight must be in [.5, 1)')
                # Keep the old NumPy operation order, including += for zero.
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    support = (baseline[i, j] - core[i, j]) / (1 - core_weight[i, j])
                    total = support + effects[i, j]
                    need(np.isfinite(support) and np.isfinite(total), 'Finite support and target sum required')
                    target = np.clip(total, -.5, .5)
                    delta = budgets[i] * (target - support)
                    scores[i, j] += delta
                need(np.isfinite(delta) and np.isfinite(scores[i, j]), 'Finite delta and adjusted score required')
                supports.append(float(support)); es.append(float(effects[i, j]))
                targets.append(float(target)); deltas.append(float(delta))
                counts['nonzero_delta_cells'] += int(delta != 0.)
            if effective == ARMS[0]:
                winner = min(boundary, key=lambda j: (-scores[i, j], ss[j]))
                selected = sorted(rank[:2] + [winner], key=lambda j: (-scores[i, j], ss[j]))
                counts['boundary_winner_reductions'] += 1
            else:
                selected = sorted(domain, key=lambda j: (-scores[i, j], ss[j]))[:3]
            counts['final_rankings'] += 1
        changed = [j for j in adjust if scores[i, j].tobytes() != pre[i, j].tobytes()]
        exited = [j for j in rank[:2] if len(rank) >= 3 and j not in selected]
        counts['original_boundary_cells'] += len(boundary)
        counts['adjusted_cells'] += len(adjust)
        counts['changed_score_cells'] += len(changed)
        counts['top2_exit_events'] += int(bool(exited))
        choices.append(selected)
        events.append(dict(query_index=i, date=dates[i], initial_partial=partial,
            effective_mode=effective, eligible_indices=use, original_rank=rank,
            original_boundary_indices=list(boundary), adjustment_indices=list(adjust),
            selection_indices=list(domain), selected_indices=list(selected),
            top2_exited_indices=exited, episode_budget=float(budgets[i]),
            adjustment_support=supports, adjustment_effects=es,
            adjustment_targets=targets, adjustment_deltas=deltas,
            changed_score_indices=changed))
    audit = dict(schema='monthly_selector_scope268_decisions_v1', mode=mode,
        counts=counts, events=events, original_inputs_preserved=True,
        long_core_unchanged=True, original_eligibility_unchanged=True,
        existing_effects_only=True, initial_partial_exact=True, computes_returns=False)
    return choices, scores, audit
