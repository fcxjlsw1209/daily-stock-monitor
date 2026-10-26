"""Exact rational support projection and independent Local67 decision audit.

Imports only a pinned independent original-source/accounting bridge. New
projection, ranks, budget, episode and decision audit are rebuilt here.
"""
from fractions import Fraction
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'work/monthly_core_veto241_independent.py'
PRIOR_SHA = '642958e877a0ed53347c2ca21aea0027e0148591af21985268976b91d9e22fb6'
PRIOR_TEST = ROOT / 'work/test_monthly_core_veto241_independent.py'
PRIOR_TEST_SHA = '4fc17fdcf385f30a87baf66513a1b2e0c19db2e0549dc779b6fe41f755e64087'
ARMS = ('incumbent_local67', 'core_conditioned_aux', 'aux_rank_control')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells', 'eligible_cells',
    'initial_partial_events', 'descriptive_ols_projections', 'constant_core_events',
    'constant_residual_events', 'nondegenerate_events', 'ranked_cells',
    'candidate_nonzero_delta_cells', 'control_nonzero_delta_cells',
    'candidate_changed_holdings', 'control_changed_holdings', 'original_episode_calls',
    'new_model_fits', 'predictive_model_fits')
EVENT_KEYS = ('query_index', 'date', 'initial_partial', 'branch', 'mean_core',
    'mean_support', 'cxx', 'cxs', 'beta', 'core_weight', 'episode_budget', 'cells',
    'incumbent_choices', 'candidate_choices', 'control_choices')
CELL_KEYS = ('symbol_index', 'symbol', 'eligible', 'active', 'core', 'core_rank',
    'support', 'weight', 'residual', 'residual_display', 'candidate_target',
    'control_target', 'candidate_delta', 'control_delta')
SELECTION_FLAGS = ('passed', 'independent_original_pre_checked',
    'original_model_rank_scope_preserved', 'original_eligibility_unchanged',
    'original_core_unchanged', 'original_effects_unchanged', 'original_support_checked',
    'effective_support_weight_checked', 'calm_nonweak_extension_budget_checked',
    'all_member_projection_domain_checked', 'exact_rational_projection_checked',
    'exact_zero_mean_residual_checked', 'exact_residual_orthogonality_checked',
    'core_conditioned_rank_checked', 'raw_rank_control_checked', 'exact_ties_checked',
    'degenerate_rows_unchanged', 'initial_partial_unchanged',
    'nonzero_delta_only_addition_checked', 'original_episode_once_per_arm_checked',
    'original_support_episode_anchor_checked', 'exact_scores_and_choices_checked',
    'full_audit_checked', 'descriptive_projection_counts_checked',
    'no_new_predictive_model_fits_checked')


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest() != PRIOR_SHA:
        raise ValueError('Frozen independent original-score/accounting bridge changed')
    spec = importlib.util.spec_from_file_location('core_conditioned243_prior', PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = prior()
require, exact_array, record_check = old.require, old.exact_array, old.record_check
PATHS, ACCOUNTING_SHA = old.PATHS, old.ACCOUNTING_SHA


def frozen_source_oracle():
    return old.frozen_source_oracle()


def accounting():
    return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest() == PRIOR_TEST_SHA,
            'Frozen synthetic original-source fixture changed')
    return tuple(sorted(set(old.dependency_files()) | {PRIOR, PRIOR_TEST, Path(__file__).resolve()}))


def pair(value):
    return None if value is None else [str(value.numerator), str(value.denominator)]


def rank_exact(values):
    """Symmetric average rank, comparisons on exact supplied fractions."""
    size = len(values)
    require(size > 1, 'Nondegenerate rank needs at least two members')
    output = []
    for value in values:
        smaller = sum(v < value for v in values)
        tied = sum(v == value for v in values)
        average = float(2 * smaller + tied + 1) / 2.0
        output.append((average - 1.0) / float(size - 1) - .5)
    return np.asarray(output, dtype=np.float64)


def project_exact(core_rank, support):
    """Project the exact values represented by finite input binary64 numbers."""
    require(len(core_rank) == len(support) and len(support) > 0, 'Equal nonempty projection axes')
    for value in list(core_rank) + list(support):
        require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
                and math.isfinite(float(value)), 'Finite real nonboolean projection inputs')
    xx = [Fraction.from_float(float(v)) for v in core_rank]
    ss = [Fraction.from_float(float(v)) for v in support]
    size = len(ss)
    mx, ms = sum(xx, Fraction()) / size, sum(ss, Fraction()) / size
    centered_x = [v - mx for v in xx]
    centered_s = [v - ms for v in ss]
    cxx = sum((v * v for v in centered_x), Fraction())
    cxs = sum((a * b for a, b in zip(centered_x, centered_s)), Fraction())
    beta = None
    residual = [None] * size
    displays = [None] * size
    target = np.zeros(size, dtype=np.float64)
    control = target.copy()
    branch = 'constant_core'
    if cxx:
        beta = cxs / cxx
        residual = [s - beta * x for s, x in zip(centered_s, centered_x)]
        require(sum(residual, Fraction()) == 0, 'Exact centered residual sum')
        require(sum((x * e for x, e in zip(centered_x, residual)), Fraction()) == 0,
                'Exact projection orthogonality')
        try:
            displays = [float(e) for e in residual]
        except (ValueError, OverflowError) as exc:
            raise FloatingPointError('Residual display overflow') from exc
        if not all(math.isfinite(v) for v in displays):
            raise FloatingPointError('Residual display must remain finite')
        branch = 'constant_residual' if all(e == residual[0] for e in residual) else 'ranked'
        if branch == 'ranked':
            target, control = rank_exact(residual), rank_exact(ss)
    return dict(branch=branch, mean_core=pair(mx), mean_support=pair(ms),
                cxx=pair(cxx), cxs=pair(cxs), beta=pair(beta),
                residual=[pair(e) for e in residual], residual_display=displays,
                candidate_target=target, control_target=control)


def _read(source, shape, n, label, *, boolean=False, finite=True):
    raw = np.asarray(source)
    require(raw.shape == shape, label + ' full shape')
    prefix = raw[:n]
    if boolean:
        require(raw.dtype == np.dtype(bool), label + ' original boolean dtype')
        require(all(isinstance(v, (bool, np.bool_)) for v in prefix.flat), label + ' exact booleans')
        return prefix.astype(bool, copy=True)
    require(all(isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
                for v in prefix.flat), label + ' real nonboolean values')
    try:
        output = prefix.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ' binary64 conversion') from exc
    require(not finite or np.isfinite(output).all(), label + ' finite values')
    return output


def sources(original, n):
    require(isinstance(original, dict), 'Original source dictionary')
    shape = np.asarray(original['baseline']).shape
    require(len(shape) == 2 and shape[1] > 0, 'Positive-width original event matrix')
    rows, width = shape
    require(type(n) is int and 0 <= n <= rows, 'Valid prefix')
    B = _read(original['baseline'], shape, n, 'baseline')
    C = _read(original['core'], shape, n, 'core')
    calm = _read(original['state_calm10'], (rows,), n, 'calm', boolean=True)
    weak = _read(original['state_weak_narrow'], (rows,), n, 'weak', boolean=True)
    raw_weight = np.asarray(original['core_weight'])
    if raw_weight.shape == (rows,):
        raw_weight = raw_weight[:, None]
    require(raw_weight.shape in ((rows, 1), shape), 'Original core-weight broadcast axes')
    cw = np.broadcast_to(_read(raw_weight, raw_weight.shape, n, 'core weight'), (n, width)).copy()
    exact_array(cw, np.broadcast_to(np.where(calm, .50, .52)[:, None], cw.shape).copy(),
                'Original common state-dependent core weight')
    E = _read(original['eligible'], shape, n, 'eligible', boolean=True)
    extension = _read(original['extension_available'], shape, n, 'extension', boolean=True)
    forecast = _read(original['forecast_available'], shape, n, 'forecast', boolean=True)
    logrank = _read(original['rank_log_excess'], shape, n, 'rank log', finite=False)
    A = (calm & ~weak)[:, None] & E & extension & forecast & np.isfinite(logrank)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        S = (B - C) / (1. - cw)
        X = C / cw
        W = (1. - cw) - (.025 * calm[:, None] + .05 * weak[:, None] + .01 * A)
    for arr, label in ((S, 'support'), (X, 'core rank')):
        require(np.isfinite(arr).all() and (np.abs(arr) <= .5 + 1e-12).all(), label + ' bounded domain')
    require(np.isfinite(W).all() and (W > 0.).all(), 'Finite positive effective support weight')
    return dict(shape=(n, width), core=C, cw=cw, support=S, X=X, W=W,
                calm=calm, weak=weak, eligible=E, active=A)


def _episode(pre, source, names, budgets):
    """Independent scalar original boundary selector, old support coordinates."""
    final = pre.copy()
    choices = []
    for i, ss in enumerate(names):
        eligible = [j for j in range(len(ss)) if source['eligible'][i, j]]
        ordered = sorted(eligible, key=lambda j: (-pre[i, j], ss[j]))
        selected = ordered.copy() if len(ordered) == 2 else []
        if len(ordered) >= 3:
            boundary = [j for j in ordered[2:5] if pre[i, ordered[2]] - pre[i, j] <= .02]
            for j in boundary:
                clipped = min(.5, max(-.5, source['support'][i, j] + source['effects'][i, j]))
                final[i, j] += budgets[i] * (clipped - source['support'][i, j])
            winner = min(boundary, key=lambda j: (-final[i, j], ss[j]))
            selected = sorted(ordered[:2] + [winner], key=lambda j: (-final[i, j], ss[j]))
        choices.append(selected)
    if not np.isfinite(final).all():
        raise FloatingPointError('Nonfinite post-episode score')
    return final, choices


def _selection(original, meta, n=None):
    n = len(meta['dates']) if n is None else n
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Valid consumed prefix')
    width, days, names = old.axes(original, meta, n)
    v = sources(original, n)
    require(v['shape'] == (n, width), 'Metadata/source width')
    counts = {key: 0 for key in COUNT_FIELDS}
    counts.update(events=n, symbols_per_event=width, checked_cells=n*width,
                  eligible_cells=int(v['eligible'].sum()), original_episode_calls=3)
    audit = dict(schema='monthly_core_conditioned243_decisions_v1', counts=counts, events=[])
    if not n:
        empty = {arm: np.empty((0, width), dtype=np.float64) for arm in ARMS}
        return empty, {arm: [] for arm in ARMS}, {arm: x.copy() for arm, x in empty.items()}, audit, dict(
            minimum_effective_support_weight=None, maximum_budget_sum_error=0., calm_nonweak_extension_count=0)
    base = frozen_source_oracle().old.original_helper().original_prefix(original, meta, n)
    exact_array(v['support'], base['support'], 'Original support parity')
    exact_array(v['core'], base['core'], 'Original weighted core parity')
    exact_array(v['eligible'], base['eligible'], 'Original eligibility parity')
    P = base['local']
    pre = {arm: P.copy() for arm in ARMS}
    budgets = np.minimum(.05, np.maximum(0., .10 - (.025*v['calm'] + .05*v['weak'])))
    maximum_error = 0.
    for i, ss in enumerate(names):
        q = project_exact(v['X'][i], v['support'][i])
        partial = str(days[i].date()) == '2016-08-17'
        active = q['branch'] == 'ranked' and not partial
        deltas = [np.zeros(width), np.zeros(width)]
        if active:
            with np.errstate(over='raise', invalid='raise'):
                deltas = [v['W'][i] * (q[key] - v['support'][i])
                          for key in ('candidate_target', 'control_target')]
            if not all(np.isfinite(d).all() for d in deltas):
                raise FloatingPointError('Nonfinite support replacement delta')
            for arm, delta in zip(ARMS[1:], deltas):
                mask = delta != 0.
                pre[arm][i, mask] += delta[mask]
        cells = []
        for j, symbol in enumerate(ss):
            coefficients = [float(v['cw'][i, j]), float(v['W'][i, j]),
                            .025*float(v['calm'][i]), .05*float(v['weak'][i]), .01*float(v['active'][i, j])]
            error = abs(math.fsum(coefficients) - 1.)
            require(error <= 1e-12, 'Original pre-episode support funding sum')
            maximum_error = max(maximum_error, error)
            cells.append(dict(symbol_index=j, symbol=symbol, eligible=bool(v['eligible'][i, j]),
                active=bool(v['active'][i, j]), core=float(v['core'][i, j]), core_rank=float(v['X'][i, j]),
                support=float(v['support'][i, j]), weight=float(v['W'][i, j]), residual=q['residual'][j],
                residual_display=q['residual_display'][j], candidate_target=float(q['candidate_target'][j]),
                control_target=float(q['control_target'][j]), candidate_delta=float(deltas[0][j]),
                control_delta=float(deltas[1][j])))
        audit['events'].append(dict(query_index=i, date=meta['dates'][i], initial_partial=partial,
            branch=q['branch'], mean_core=q['mean_core'], mean_support=q['mean_support'], cxx=q['cxx'],
            cxs=q['cxs'], beta=q['beta'], core_weight=float(v['cw'][i, 0]),
            episode_budget=float(budgets[i]), cells=cells,
            incumbent_choices=[], candidate_choices=[], control_choices=[]))
        counts['initial_partial_events'] += int(partial)
        counts['descriptive_ols_projections'] += int(q['branch'] != 'constant_core')
        counts['constant_core_events'] += int(q['branch'] == 'constant_core')
        counts['constant_residual_events'] += int(q['branch'] == 'constant_residual')
        counts['nondegenerate_events'] += int(q['branch'] == 'ranked')
        counts['ranked_cells'] += width * int(q['branch'] == 'ranked')
        counts['candidate_nonzero_delta_cells'] += int(np.count_nonzero(deltas[0]))
        counts['control_nonzero_delta_cells'] += int(np.count_nonzero(deltas[1]))
    if not all(np.isfinite(value).all() for value in pre.values()):
        raise FloatingPointError('Nonfinite modified pre-score')
    scores, choices = {}, {}
    for arm in ARMS:
        scores[arm], choices[arm] = _episode(pre[arm], base, names, budgets)
    for i, event in enumerate(audit['events']):
        for arm, field in zip(ARMS, ('incumbent_choices', 'candidate_choices', 'control_choices')):
            event[field] = choices[arm][i]
        counts['candidate_changed_holdings'] += int(set(choices[ARMS[1]][i]) != set(choices[ARMS[0]][i]))
        counts['control_changed_holdings'] += int(set(choices[ARMS[2]][i]) != set(choices[ARMS[0]][i]))
    return scores, choices, pre, audit, dict(minimum_effective_support_weight=float(v['W'].min()),
        maximum_budget_sum_error=maximum_error, calm_nonweak_extension_count=int(v['active'].sum()))


def selection_oracle(original, meta, n=None):
    scores, choices, pre, audit, _ = _selection(original, meta, n)
    return scores, choices, pre, audit


def verify_selection(original, meta, orders, *, scores, pre_scores, audit, n=None):
    n = len(meta['dates']) if n is None else n
    expected, choices, pre, wanted_audit, budget = _selection(original, meta, n)
    require(isinstance(orders, dict) and set(orders) == set(ARMS), 'Three order arms')
    wanted = {arm: [dict(date=meta['dates'][i], end=meta['ends'][i], signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids], scores=[float(expected[arm][i, j]) for j in ids])
        for i, ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders, wanted, 'Independent ordered choices')
    for got, reference, label in ((scores, expected, 'post-episode scores'), (pre_scores, pre, 'pre-scores')):
        require(isinstance(got, dict) and set(got) == set(ARMS), 'Three full '+label)
        for arm in ARMS:
            exact_array(got[arm], reference[arm], label+'/'+arm)
    record_check(audit, wanted_audit, 'Independent complete rational/decision audit')
    return dict(**{key: True for key in SELECTION_FLAGS}, **budget, counts=wanted_audit['counts'],
        checked_events=n, ordered_choices_checked=3*n, full_scores_supplied=True, full_pre_scores_supplied=True,
        full_audit_supplied=True, numerical_tolerance=0., original_source_parity_tolerance=1e-12,
        producer_modules_imported=False, computes_returns=False,
        holdings_counts={arm: {str(k): sum(len(row) == k for row in choices[arm]) for k in (0, 2, 3)} for arm in ARMS})


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arms')
    mapping = dict(zip(old.ARMS, ARMS))
    result = old.audit_outcomes(px, opening, *[{k: data[v] for k, v in mapping.items()}
        for data in (orders, exports, results)], meta)
    result['paths'] = {mapping[k]: v for k, v in result['paths'].items()}
    return result
