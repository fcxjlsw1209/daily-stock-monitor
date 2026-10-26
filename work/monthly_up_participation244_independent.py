"""Independent exact relative-win frequencies and funded Local67 integration.

Only pinned independent infrastructure is imported. New prices, frequencies,
pre-score corrections, scalar episode and complete audits are rebuilt here.
"""
from fractions import Fraction
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'work/monthly_block_sharpe242_independent.py'
PRIOR_SHA = 'd06100f847103e258ff56a4905e3b2b93401d008d6d81a58f78edfb43f07c180'
PRIOR_TEST = ROOT/'work/test_monthly_block_sharpe242_independent.py'
PRIOR_TEST_SHA = 'ad703b3d095fffdb1407f0a8b86759e551286e6b44e10b31f699a1dccf87c25e'
ARMS = ('incumbent_local67', 'up_participation', 'all_participation_control')
VALUE_KEYS = ('available', 'candidate', 'control')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells', 'source_available_cells',
    'benchmark_available_events', 'paired_source_cells', 'available_cells', 'unavailable_cells',
    'security_price_observations', 'benchmark_price_observations', 'relative_comparisons',
    'conditional_comparisons', 'conditional_estimates', 'unconditional_estimates',
    'new_model_fits', 'predictive_model_fits')
EVENT_KEYS = ('query_index', 'date', 'symbols', 'session_position', 'price_start', 'price_end',
    'source_dates', 'missing_official_sessions', 'benchmark_available', 'benchmark_reason',
    'benchmark_start_price', 'benchmark_end_price', 'up_count', 'down_count', 'flat_count',
    'source_available_indices', 'available_indices', 'cells')
CELL_KEYS = ('column_index', 'symbol', 'source_available', 'source_reason', 'start_price', 'end_price',
    'benchmark_available', 'benchmark_reason', 'benchmark_start_price', 'benchmark_end_price',
    'up_count', 'down_count', 'flat_count', 'available', 'reason', 'wins_up', 'ties_up', 'losses_up',
    'wins_all', 'ties_all', 'losses_all', 'candidate', 'control')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'exact_official_253_price_window_checked',
    'skip42_checked', 'full_positive_security_source_checked', 'full_positive_spy_source_checked',
    'security_source_independent_of_benchmark_checked', 'security_source_independent_of_up_count_checked',
    'exact_binary_rational_comparisons_checked', 'all_252_daily_outcomes_checked',
    'strict_positive_spy_days_checked', 'half_win_ties_checked', 'minimum_40_up_days_checked',
    'no_down_day_minimum_checked', 'flat_market_days_in_control_checked',
    'common_availability_checked', 'direct_probability_scale_checked', 'no_cross_sectional_ranking',
    'no_eligibility_filter', 'self_benchmark_neutral_checked', 'full_audit_checked',
    'descriptive_counts_checked', 'no_future_inputs', 'no_predictive_fits_checked')
DECISION_COUNT_FIELDS = ('events', 'symbols_per_event', 'active_events', 'available_active_cells',
    'candidate_nonzero_delta_cells', 'control_nonzero_delta_cells', 'original_episode_calls',
    'new_model_fits', 'predictive_model_fits')
DECISION_EVENT_KEYS = ('query_index', 'date', 'initial_partial', 'calm', 'weak', 'active',
    'available_indices', 'extension_active_indices', 'episode_budget', 'candidate_delta', 'control_delta',
    'original_remaining_support', 'candidate_remaining_support', 'control_remaining_support')
SELECTION_FLAGS = ('passed', 'independent_original_pre_checked', 'original_model_rank_scope_preserved',
    'original_eligibility_unchanged', 'original_core_unchanged', 'original_support_checked',
    'funded_hand_formula_checked', 'fixed_025_budget_checked', 'calm_nonweak_extension_budget_checked',
    'conservative_episode_reserve_checked', 'complete_remaining_support_checked',
    'valid_zero_probability_active', 'missing_preserves_original_score', 'all_states_active',
    'named_initial_partial_unchanged', 'original_effects_unchanged', 'single_original_episode',
    'modified_pre_top_two_preserved', 'exact_scores_and_choices_checked', 'no_second_rank',
    'full_audit_checked', 'no_predictive_fits_checked')


def prior():
    if hashlib.sha256(PRIOR.read_bytes()).hexdigest() != PRIOR_SHA:
        raise ValueError('Frozen independent source/accounting bridge changed')
    spec = importlib.util.spec_from_file_location('up244_prior_independent', PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = prior()
require, exact_array, record_check = old.require, old.exact_array, old.record_check
PATHS, ACCOUNTING_SHA = old.PATHS, old.ACCOUNTING_SHA


def frozen_source_oracle(): return old.frozen_source_oracle()
def accounting(): return old.accounting()


def dependency_files():
    require(hashlib.sha256(PRIOR_TEST.read_bytes()).hexdigest() == PRIOR_TEST_SHA,
            'Frozen synthetic original-source fixture changed')
    return tuple(sorted(set(old.dependency_files()) | {PRIOR, PRIOR_TEST, Path(__file__).resolve()}))


def source_path(values):
    require(np.shape(values) == (253,), 'Exactly253 stored source prices')
    if any(not isinstance(v, numbers.Real) or isinstance(v, (bool, np.bool_)) for v in values):
        return None, 'nonnumeric_adjusted_price'
    try:
        numbers64 = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError):
        return None, 'nonnumeric_adjusted_price'
    if any(not math.isfinite(v) or v <= 0. for v in numbers64):
        return None, 'invalid_adjusted_price'
    return numbers64, 'available'


def relative_outcome(start, end, market_start, market_end):
    """Fraction division independently compares exact binary64 gross returns."""
    own = Fraction.from_float(end) / Fraction.from_float(start)
    market = Fraction.from_float(market_end) / Fraction.from_float(market_start)
    return 1 if own > market else -1 if own < market else 0


def _measure(security, market):
    p, own_reason = security
    m, market_reason = market
    own_valid, market_valid = p is not None, m is not None
    up = [m[d] > m[d-1] for d in range(1, 253)] if market_valid else None
    up_count = sum(up) if market_valid else None
    down_count = sum(m[d] < m[d-1] for d in range(1, 253)) if market_valid else None
    flat_count = 252-up_count-down_count if market_valid else None
    counts = {key: None for key in ('wins_up', 'ties_up', 'losses_up', 'wins_all', 'ties_all', 'losses_all')}
    good = own_valid and market_valid and up_count >= 40
    reason = (own_reason if not own_valid else 'benchmark_'+market_reason if not market_valid
              else 'insufficient_up_days' if not good else 'available')
    candidate = control = None
    if own_valid and market_valid:
        counts = {key: 0 for key in counts}
        for d in range(1, 253):
            result = relative_outcome(p[d-1], p[d], m[d-1], m[d])
            key = 'wins' if result > 0 else 'losses' if result < 0 else 'ties'
            counts[key+'_all'] += 1
            if up[d-1]: counts[key+'_up'] += 1
        if good:
            candidate = (2*counts['wins_up']+counts['ties_up']-up_count) / (2*up_count)
            control = (2*counts['wins_all']+counts['ties_all']-252) / 504
            if up_count == 252: candidate = control
    return dict(source_available=own_valid, source_reason=own_reason,
        start_price=p[0] if own_valid else None, end_price=p[-1] if own_valid else None,
        benchmark_available=market_valid, benchmark_reason=market_reason,
        benchmark_start_price=m[0] if market_valid else None,
        benchmark_end_price=m[-1] if market_valid else None,
        up_count=up_count, down_count=down_count, flat_count=flat_count,
        available=bool(good), reason=reason, **counts, candidate=candidate, control=control)


def measure_oracle(prices, benchmark):
    # Validate both shapes first; source validity itself is independent.
    require(np.shape(prices) == np.shape(benchmark) == (253,), 'Two complete253 source paths')
    return _measure(source_path(prices), source_path(benchmark))


def independent_features(panel, meta, calendar, limit=None):
    frame, sessions, queries, width = old.axes(panel, meta, calendar, limit)
    shape = (len(queries), width)
    values = dict(available=np.zeros(shape, bool), candidate=np.full(shape, np.nan), control=np.full(shape, np.nan))
    counts = {key: 0 for key in COUNT_FIELDS}
    counts.update(events=len(queries), symbols_per_event=width, checked_cells=len(queries)*width)
    history, reasons = [], {}
    for i, (day, names, t) in enumerate(queries):
        dates = sessions[t-294:t-41] if t >= 294 else []
        missing = [str(d.date()) for d in dates if d not in frame.index]
        def path(symbol):
            reason = ('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                      else 'missing_official_session' if missing else None)
            return (None, reason) if reason else source_path([frame.at[d, symbol] for d in dates])
        market = path('SPY')
        market_values, market_reason = market
        mv = market_values is not None
        up_count = sum(market_values[d] > market_values[d-1] for d in range(1, 253)) if mv else None
        down_count = sum(market_values[d] < market_values[d-1] for d in range(1, 253)) if mv else None
        flat_count = 252-up_count-down_count if mv else None
        counts['benchmark_available_events'] += int(mv)
        cells = []
        for j, symbol in enumerate(names):
            measured = _measure(path(symbol), market)
            cell = dict(column_index=j, symbol=symbol, **measured)
            cells.append(cell)
            own, available = measured['source_available'], measured['available']
            paired = own and measured['benchmark_available']
            counts['source_available_cells'] += int(own)
            counts['paired_source_cells'] += int(paired)
            counts['available_cells'] += int(available)
            if paired: counts['conditional_comparisons'] += measured['up_count']
            values['available'][i, j] = available
            if available:
                values['candidate'][i, j] = measured['candidate']
                values['control'][i, j] = measured['control']
            reasons[measured['reason']] = reasons.get(measured['reason'], 0)+1
        history.append(dict(query_index=i, date=str(day.date()), symbols=names, session_position=t,
            price_start=str(dates[0].date()) if dates else None, price_end=str(dates[-1].date()) if dates else None,
            source_dates=[str(d.date()) for d in dates], missing_official_sessions=missing,
            benchmark_available=mv, benchmark_reason=market_reason,
            benchmark_start_price=market_values[0] if mv else None,
            benchmark_end_price=market_values[-1] if mv else None,
            up_count=up_count, down_count=down_count, flat_count=flat_count,
            source_available_indices=[j for j, c in enumerate(cells) if c['source_available']],
            available_indices=[j for j, c in enumerate(cells) if c['available']], cells=cells))
    counts.update(unavailable_cells=counts['checked_cells']-counts['available_cells'],
        security_price_observations=253*counts['source_available_cells'],
        benchmark_price_observations=253*counts['benchmark_available_events'],
        relative_comparisons=252*counts['paired_source_cells'],
        conditional_estimates=counts['available_cells'], unconditional_estimates=counts['available_cells'])
    return values, dict(schema='monthly_up_participation244_features_v1', events=len(queries),
        counts=counts, reason_counts=reasons, event_history=history)


def verify_features(panel, meta, calendar, values, audit, limit=None):
    wanted, report = independent_features(panel, meta, calendar, limit)
    require(isinstance(values, dict) and set(values) == set(VALUE_KEYS), 'Three feature arrays')
    for key in VALUE_KEYS: exact_array(values[key], wanted[key], 'Independent '+key)
    record_check(audit, report, 'Independent complete paired source and comparison audit')
    return dict(**{key: True for key in FEATURE_FLAGS}, **report['counts'], counts=report['counts'],
        numerical_tolerance=0., producer_modules_imported=False, computes_returns=False)


def selection_values(values, n, width):
    require(isinstance(values, dict) and set(values) == set(VALUE_KEYS), 'Three feature arrays')
    mask = np.asarray(values['available'])
    require(mask.ndim == 2 and mask.dtype == np.dtype(bool) and len(mask) >= n and
            (mask.shape[1] == width or n == 0 and mask.shape == (0, 0)), 'Feature mask axes')
    if not n: return dict(available=np.empty((0, width), bool), candidate=np.empty((0, width)), control=np.empty((0, width)))
    output = {'available': mask[:n].copy()}
    for key in VALUE_KEYS[1:]:
        raw = np.asarray(values[key])
        require(raw.shape == mask.shape and raw.dtype == np.dtype(np.float64), 'Binary64 probability array '+key)
        v = raw[:n]
        require(np.isfinite(v[output['available']]).all() and np.isnan(v[~output['available']]).all(),
                'Same probability support '+key)
        require((np.abs(v[output['available']]) <= .5).all(), 'Centered probability bound '+key)
        output[key] = v.copy()
    return output


def _selection(original, meta, values, n=None):
    n = len(meta['dates']) if n is None else n
    require(type(n) is int and 0 <= n <= len(meta['dates']), 'Selection prefix')
    width, days, names = old.old.axes(original, meta, n)
    v = selection_values(values, n, width)
    counts = {key: 0 for key in DECISION_COUNT_FIELDS}
    counts.update(events=n, symbols_per_event=width, original_episode_calls=3)
    decision = dict(schema='monthly_up_participation244_decisions_v1', counts=counts, events=[])
    if n == 0:
        empty = {arm: np.empty((0, width)) for arm in ARMS}
        return empty, {arm: [] for arm in ARMS}, {arm: a.copy() for arm, a in empty.items()}, decision, dict(
            minimum_remaining_support=None, maximum_budget_sum_error=0., calm_nonweak_extension_count=0,
            maximum_absolute_delta=0., changed_cells={arm: 0 for arm in ARMS})
    x = frozen_source_oracle().old.original_helper().original_prefix(original, meta, n)
    shape = (n, width)
    exact_array(x['cw'], np.broadcast_to(np.where(x['calm'], .5, .52)[:, None], shape), 'Original core weights')
    require((np.abs(x['core']) <= .5*x['cw']+1e-12).all(), 'Original weighted core bound')
    extension = np.asarray(original['extension_available'])[:n]
    forecast = np.asarray(original['forecast_available'])[:n]
    logrank = np.asarray(original['rank_log_excess'])[:n]
    require(extension.shape == forecast.shape == logrank.shape == shape and
            extension.dtype == forecast.dtype == np.dtype(bool), 'Original extension scope axes')
    A = (x['calm'] & ~x['weak'])[:, None] & extension & forecast & x['eligible'] & np.isfinite(logrank)
    budget = np.minimum(.05, np.maximum(0., .10-(.025*x['calm']+.05*x['weak'])))
    active = np.array([str(d.date()) != '2016-08-17' for d in days], dtype=bool)
    hand = active[:, None] & v['available']
    R0 = (1.-x['cw'])-(.025*x['calm'][:, None]+.05*x['weak'][:, None]+.01*A)-budget[:, None]
    Rnew = R0-.025*hand
    require(np.isfinite(R0).all() and np.isfinite(Rnew).all() and (Rnew >= .355-1e-12).all(),
            'Conservative full original and hand support budgets')
    maximum_error = 0.
    for i in range(n):
        for j in range(width):
            weights = [float(x['cw'][i, j]), .025*float(x['calm'][i]), .05*float(x['weak'][i]),
                       .01*float(A[i, j]), float(budget[i]), .025*float(hand[i, j]), float(Rnew[i, j])]
            error = abs(math.fsum(weights)-1.)
            require(error <= 1e-12, 'Funded support budget identity')
            maximum_error = max(maximum_error, error)
    scores, choices, pres, changed, deltas = {}, {}, {}, {}, {}
    maxdelta = 0.
    for arm, key in zip(ARMS, (None, 'candidate', 'control')):
        delta = np.zeros(shape)
        pre = x['local'].copy()
        if key is not None:
            for i in range(n):
                for j in range(width):
                    if hand[i, j]: delta[i, j] = .025*(float(v[key][i, j])-float(x['support'][i, j]))
        require(np.isfinite(delta).all() and (np.abs(delta) <= .025+1e-12).all(), 'Finite fixed funded hand delta')
        nonzero = delta != 0.
        pre[nonzero] += delta[nonzero]
        require(np.isfinite(pre).all(), 'Finite full modified pre')
        score = pre.copy()
        rows = []
        for i, ss in enumerate(names):
            order = sorted([j for j in range(width) if x['eligible'][i, j]], key=lambda j: (-pre[i, j], ss[j]))
            if len(order) < 3:
                rows.append(order if len(order) == 2 else [])
                continue
            group = [j for j in order[2:5] if pre[i, order[2]]-pre[i, j] <= .02]
            for j in group:
                target = min(.5, max(-.5, x['support'][i, j]+x['effects'][i, j]))
                score[i, j] += budget[i]*(target-x['support'][i, j])
            third = min(group, key=lambda j: (-score[i, j], ss[j]))
            rows.append(sorted(order[:2]+[third], key=lambda j: (-score[i, j], ss[j])))
        require(np.isfinite(score).all(), 'Finite post-episode scores')
        scores[arm], choices[arm], pres[arm], deltas[arm] = score, rows, pre, delta
        changed[arm] = int(nonzero.sum())
        maxdelta = max(maxdelta, float(np.max(np.abs(delta))))
    counts.update(active_events=int(active.sum()), available_active_cells=int(hand.sum()),
        candidate_nonzero_delta_cells=changed[ARMS[1]], control_nonzero_delta_cells=changed[ARMS[2]])
    for i, d in enumerate(days):
        decision['events'].append(dict(query_index=i, date=meta['dates'][i], initial_partial=str(d.date()) == '2016-08-17',
            calm=bool(x['calm'][i]), weak=bool(x['weak'][i]), active=bool(active[i]),
            available_indices=np.flatnonzero(v['available'][i]).tolist(), extension_active_indices=np.flatnonzero(A[i]).tolist(),
            episode_budget=float(budget[i]), candidate_delta=deltas[ARMS[1]][i].tolist(),
            control_delta=deltas[ARMS[2]][i].tolist(), original_remaining_support=R0[i].tolist(),
            candidate_remaining_support=Rnew[i].tolist(), control_remaining_support=Rnew[i].tolist()))
    return scores, choices, pres, decision, dict(minimum_remaining_support=float(Rnew.min()),
        maximum_budget_sum_error=maximum_error, calm_nonweak_extension_count=int(A.sum()),
        maximum_absolute_delta=maxdelta, changed_cells=changed)


def selection_oracle(original, meta, values, n=None): return _selection(original, meta, values, n)[:4]


def verify_selection(original, meta, values, orders, *, scores, pre_scores, audit, n=None):
    n = len(meta['dates']) if n is None else n
    expected, choices, pres, expected_audit, stats = _selection(original, meta, values, n)
    wanted = {arm: [dict(date=meta['dates'][i], end=meta['ends'][i], signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids], scores=[float(expected[arm][i, j]) for j in ids])
        for i, ids in enumerate(choices[arm])] for arm in ARMS}
    record_check(orders, wanted, 'Independent ordered selections')
    for got, reference, label in ((scores, expected, 'full scores'), (pre_scores, pres, 'full pre-scores')):
        require(isinstance(got, dict) and set(got) == set(ARMS), 'Three '+label)
        for arm in ARMS: exact_array(got[arm], reference[arm], label+'/'+arm)
    record_check(audit, expected_audit, 'Independent full decisions and budget audit')
    return dict(**{key: True for key in SELECTION_FLAGS}, **stats, counts=expected_audit['counts'],
        checked_events=n, ordered_choices_checked=3*n, full_scores_supplied=True, full_pre_scores_supplied=True,
        full_audit_supplied=True, numerical_tolerance=0., original_source_parity_tolerance=1e-12,
        producer_modules_imported=False, computes_returns=False)


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arms')
    mapping = dict(zip(old.ARMS, ARMS))
    result = old.audit_outcomes(px, opening, *[{k: data[v] for k, v in mapping.items()}
        for data in (orders, exports, results)], meta)
    result['paths'] = {mapping[k]: value for k, value in result['paths'].items()}
    return result
