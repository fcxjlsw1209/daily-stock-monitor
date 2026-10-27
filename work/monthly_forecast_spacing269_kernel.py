"""Pure, fixed-budget spacing of already authenticated Local67 forecasts.

Only supplied prefix matrices are accepted. The host owns source authentication,
whole-row readiness and the original selector. This module does not fit, select
holdings, compute labels/returns, or read files. Endpoint matching retains the
range and ordering, not the mean/variance or absolute forecast calibration.
"""
from datetime import date

import numpy as np
import pandas as pd


ARMS = ('incumbent_local67', 'pooled_affine', 'mixed_affine')
INITIAL_PARTIAL = '2016-08-17'
POOLED_WEIGHT = .0375
LOCAL_WEIGHT = .0125
AUDIT_KEYS = ('schema', 'mode', 'counts', 'events', 'original_inputs_preserved',
              'nonconsumed_pre_preserved', 'initial_partial_exact', 'nonweak_exact',
              'not_ready_exact', 'consumed_source_ranks_checked', 'local_weight_fixed',
              'no_rerank', 'no_new_budget', 'source_files_authenticated', 'computes_returns')
COUNT_FIELDS = ('events', 'symbols_per_event', 'initial_partial_events', 'weak_events',
                'ready_events', 'use_cells', 'active_events', 'active_cells',
                'pooled_rank_checks', 'local_rank_checks', 'pooled_mapped_cells',
                'local_mapped_cells', 'pooled_identity_rows', 'local_identity_rows',
                'pooled_affine_rows', 'local_affine_rows', 'pooled_changed_rank_cells',
                'local_changed_rank_cells', 'pooled_nonzero_delta_cells',
                'local_nonzero_delta_cells', 'nonzero_delta_cells', 'changed_pre_cells',
                'transform_calls', 'new_model_fits', 'predictive_model_fits')
EVENT_KEYS = ('query_index', 'date', 'initial_partial', 'weak', 'ready', 'use_indices',
              'branch', 'pooled', 'local', 'delta', 'changed_pre_indices')
MAP_KEYS = ('raw', 'old_rank', 'new_rank', 'distinct_levels', 'branch', 'minimum',
            'maximum', 'rank_minimum', 'rank_maximum', 'delta')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _inputs(pre, pooled_raw, pooled_rank, local_raw, local_rank, use, ready, weak,
            dates, mode):
    need(type(mode) is str and mode in ARMS, 'Registered spacing mode')
    need(isinstance(pre, np.ndarray) and pre.ndim == 2
         and pre.dtype == np.dtype('float64'), 'pre must be a float64 matrix')
    n, width = pre.shape
    need(width > 0, 'Positive width required, including n0')
    for name, value in (('pooled_raw', pooled_raw), ('pooled_rank', pooled_rank),
                        ('local_raw', local_raw), ('local_rank', local_rank)):
        need(isinstance(value, np.ndarray) and value.shape == pre.shape
             and value.dtype == np.dtype('float64'), name + ' must be matching float64 matrix')
    need(isinstance(use, np.ndarray) and use.shape == pre.shape
         and use.dtype == np.dtype(bool), 'use must be matching bool matrix')
    for name, value in (('ready', ready), ('weak', weak)):
        need(isinstance(value, np.ndarray) and value.shape == (n,)
             and value.dtype == np.dtype(bool), name + ' must be bool event vector')
    need(type(dates) in (list, tuple) and len(dates) == n, 'Dates event axis')
    last = None
    for i, day in enumerate(dates):
        need(type(day) is str, 'Native ISO date required')
        try:
            parsed = date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError('Valid ISO date required') from exc
        need(parsed.isoformat() == day and (last is None or day > last),
             'Strictly increasing ISO dates required')
        need(day != INITIAL_PARTIAL or i == 0, 'Initial partial must be index0')
        last = day
    return n, width


def _changed(a, b):
    return np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))


def _map(raw, old_rank, weight):
    need(np.isfinite(raw).all() and np.isfinite(old_rank).all(),
         'Finite consumed raw forecasts and ranks required')
    # Same average percentile convention as frozen tweaks.py::ranks. Calling
    # rank on the selected U is equivalent to masking other rows to NaN first.
    expected = pd.Series(raw).rank(method='average', pct=True).to_numpy() - .5
    need(old_rank.tobytes() == expected.tobytes(), 'Original source ranks must match exactly')
    # First occurrence on equal extrema also fixes the signed-zero evidence.
    low, high = float(min(raw)), float(max(raw))
    rlow, rhigh = float(min(old_rank)), float(max(old_rank))
    levels = int(len(np.unique(raw)))
    new = old_rank.copy()
    branch = 'old_rank_copy'
    if levels > 2:
        branch = 'affine'
        # Literal binary64 order. No clipping, epsilon, variance floor or
        # subsequent ranking. Ordinary IEEE underflow is not a data fill.
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            span = np.float64(high) - np.float64(low)
            rank_span = np.float64(rhigh) - np.float64(rlow)
            middle = (raw != low) & (raw != high)
            new[middle] = ((raw[middle] - low) / span) * rank_span + rlow
        new[raw == low] = old_rank[raw == low]
        new[raw == high] = old_rank[raw == high]
        need(np.isfinite(new).all(), 'Finite affine values required')
        need((new >= rlow).all() and (new <= rhigh).all(), 'Affine values within original endpoints')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        delta = weight * (new - old_rank)
    need(np.isfinite(delta).all(), 'Finite weighted forecast delta required')
    evidence = dict(raw=raw.tolist(), old_rank=old_rank.tolist(), new_rank=new.tolist(),
        distinct_levels=levels, branch=branch, minimum=low, maximum=high,
        rank_minimum=rlow, rank_maximum=rhigh, delta=delta.tolist())
    return new, delta, evidence


def transform(pre, pooled_raw, pooled_rank, local_raw, local_rank, use, ready, weak,
              dates, mode):
    """Transform only nonpartial weak & ready cells in the original U.

    pooled_raw is the original raw_source_log, not a nearly equal earlier cache.
    Pooled-only mode never consumes local values. Incumbent, inactive rows and
    U-excluded cells do not inspect forecast/P values; their P bytes are copied.
    Active required values, including P, must be finite. The caller must slice
    every input to the desired prefix before calling; n0 consumes no body.
    """
    n, width = _inputs(pre, pooled_raw, pooled_rank, local_raw, local_rank, use,
                       ready, weak, dates, mode)
    out = pre.copy()
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts.update(events=n, symbols_per_event=width, transform_calls=1)
    events = []
    for i in range(n):
        ids = np.flatnonzero(use[i])
        partial = dates[i] == INITIAL_PARTIAL
        w, r = bool(weak[i]), bool(ready[i])
        counts['initial_partial_events'] += int(partial)
        counts['weak_events'] += int(w)
        counts['ready_events'] += int(r)
        counts['use_cells'] += len(ids)
        branch = ('initial_partial' if partial else 'incumbent' if mode == ARMS[0]
                  else 'nonweak' if not w else 'not_ready' if not r
                  else 'empty_use' if not len(ids) else 'active')
        maps = {'pooled': None, 'local': None}
        total, changed = [], []
        if branch == 'active':
            before = pre[i, ids]
            need(np.isfinite(before).all(), 'Finite consumed original pre-score required')
            counts['active_events'] += 1
            counts['active_cells'] += len(ids)
            contributions = []
            roles = [('pooled', pooled_raw, pooled_rank, POOLED_WEIGHT)]
            if mode == ARMS[2]:
                roles.append(('local', local_raw, local_rank, LOCAL_WEIGHT))
            for role, raw, old, weight in roles:
                new, delta, maps[role] = _map(raw[i, ids], old[i, ids], weight)
                counts[role + '_rank_checks'] += 1
                counts[role + '_mapped_cells'] += len(ids)
                counts[role + ('_identity_rows' if maps[role]['branch'] == 'old_rank_copy'
                               else '_affine_rows')] += 1
                counts[role + '_changed_rank_cells'] += len(_changed(new, old[i, ids]))
                counts[role + '_nonzero_delta_cells'] += int(np.count_nonzero(delta != 0.))
                contributions.append(delta)
            total = contributions[0]
            if len(contributions) == 2:
                with np.errstate(over='raise', invalid='raise', under='ignore'):
                    total = total + contributions[1]
            need(np.isfinite(total).all(), 'Finite total delta required')
            nonzero = total != 0.
            with np.errstate(over='raise', invalid='raise', under='ignore'):
                out[i, ids[nonzero]] += total[nonzero]
            need(np.isfinite(out[i, ids]).all(), 'Finite adjusted pre-score required')
            changed = ids[_changed(out[i, ids], before)].tolist()
            counts['nonzero_delta_cells'] += int(np.count_nonzero(nonzero))
            counts['changed_pre_cells'] += len(changed)
            total = total.tolist()
        events.append(dict(query_index=i, date=dates[i], initial_partial=partial,
            weak=w, ready=r, use_indices=ids.tolist(), branch=branch,
            pooled=maps['pooled'], local=maps['local'], delta=total,
            changed_pre_indices=changed))
    audit = dict(schema='monthly_forecast_spacing269_transform_v1', mode=mode,
        counts=counts, events=events, original_inputs_preserved=True,
        nonconsumed_pre_preserved=True, initial_partial_exact=True,
        nonweak_exact=True, not_ready_exact=True, consumed_source_ranks_checked=True,
        local_weight_fixed=True, no_rerank=True, no_new_budget=True,
        source_files_authenticated=False, computes_returns=False)
    return out, audit
