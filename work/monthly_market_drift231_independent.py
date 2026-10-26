"""Independent current-market-conditioned historical-drift and selector oracle.

Only caller-supplied prices and original strategy arrays are consumed. The new
producer is never imported; frozen independent helpers rebuild the incumbent
and audit saved portfolio accounting. Import performs no real-data calculation.
"""
from datetime import date
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HELPER = ROOT/'work/monthly_endpoint155_prepare_review.py'
SOURCE_SHA = '0e594355beecc84a705517c8cd2774214e4300b5248e0c772a0306e73c38174f'
ACCOUNTING = ROOT/'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'market_conditioned_drift', 'unconditional_drift_control')
ARRAY_KEYS = ('candidate_raw', 'control_raw', 'candidate_effect', 'control_effect', 'ready')
COUNT_FIELDS = ('query_months', 'eligible_cells', 'supported_months', 'fallback_months',
    'initial_partial_months', 'domain_fallback_months', 'source_fallback_months',
    'state_support_fallback_months', 'checked_stock_cells', 'source_valid_stock_cells',
    'stock_price_observations', 'stock_log_returns', 'drift_estimates',
    'ranked_observations', 'market_state_evaluations', 'market_price_observations')
MODEL_COUNT_FIELDS = COUNT_FIELDS
MODEL_FLAGS = ('passed', 'independent_scalar_rebuild', 'original_calm_window_checked',
    'official_session_windows_checked', 'skip42_stock_window_checked',
    'current_market_state_checked', 'previous_day_market_states_checked',
    'strict_spy200_comparison_checked', 'mean_log_drift_checked',
    'same_window_unconditional_control_checked', 'fixed_63_shrinkage_checked',
    'minimum_42_matches_checked', 'complete_eligible_universe_checked',
    'whole_month_common_fallback_checked', 'symmetric_eligible_effect_ranks_checked',
    'constant_drift_zero_effect_checked', 'initial_partial_unchanged',
    'no_forward_labels', 'no_future_inputs', 'audit_checked', 'counts_checked',
    'no_predictive_fits_checked')
SELECTION_FLAGS = ('passed', 'independent_original_pre_checked', 'original_support_checked',
    'original_budget_checked', 'eligible_universe_reranked_checked',
    'replacement_not_additive_checked', 'one_episode_per_arm',
    'paired_fallback_checked', 'constant_effect_not_fallback_checked',
    'exact_scores_and_choices_checked')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_source_oracle():
    require(sha(SOURCE_HELPER) == SOURCE_SHA, 'Frozen independent source changed')
    spec = importlib.util.spec_from_file_location('market231_original_independent', SOURCE_HELPER)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def dependency_files():
    return tuple(sorted(set([Path(__file__).resolve(), ACCOUNTING,
        *frozen_source_oracle().source_files(ROOT)])))


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('market231_independent_accounts', ACCOUNTING)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            record_check(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def exact_array(actual, expected, label):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape and a.dtype == e.dtype, label + ': shape/dtype')
    if e.dtype.kind == 'f':
        require(not np.isinf(a).any() and np.array_equal(np.isnan(a), np.isnan(e)), label + ': finite mask')
        good = np.isfinite(e)
        require(a[good].tobytes() == e[good].tobytes(), label + ': exact binary64 values')
    else:
        require(np.array_equal(a, e), label + ': values')


def close(actual, expected, label, atol=0.):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape and np.array_equal(np.isfinite(a), np.isfinite(e))
        and np.array_equal(np.isnan(a), np.isnan(e)) and not np.isinf(a).any(), label+': array domain')
    good = np.isfinite(e)
    require(np.all(np.abs(a[good]-e[good]) <= atol), label+': numerical values')


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label)
    return value


def total(values, label):
    try:
        return finite(math.fsum(values), label)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Nonfinite ' + label) from exc


def iso(value):
    require(isinstance(value, str), 'Canonical ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Canonical ISO date string required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar, arrays, limit):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Adjusted-only panel required')
    frame = panel['adj']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Adjusted frame type')
    idx = frame.index
    require(not idx.hasnans and idx.tz is None and idx.is_unique and idx.is_monotonic_increasing
        and idx.equals(idx.normalize()), 'Adjusted date axis')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
        for s in frame.columns), 'Adjusted identity axis')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols')), 'Query metadata')
    require(isinstance(meta['dates'], list) and isinstance(meta['symbols'], list)
        and len(meta['dates']) == len(meta['symbols']), 'Query metadata axes')
    if 'ends' in meta:
        require(isinstance(meta['ends'], list) and len(meta['ends']) == len(meta['dates']), 'End metadata axis')
    sessions = [iso(s) for s in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions)
        and all(a < b for a, b in zip(sessions, sessions[1:])), 'Official calendar axis')
    positions = {d: i for i, d in enumerate(sessions)}
    require(isinstance(arrays, dict) and all(k in arrays for k in ('eligible', 'state_calm10')), 'Original array fields')
    eligible = np.asarray(arrays['eligible'])
    require(eligible.ndim == 2 and eligible.dtype == np.dtype(bool)
        and eligible.shape[0] == len(meta['dates']), 'Original eligible matrix')
    rows, width = eligible.shape
    calm = np.asarray(arrays['state_calm10'])
    require(calm.dtype == np.dtype(bool) and calm.shape in ((rows,), (rows, 1)), 'Original calm state vector')
    calm = calm.reshape(rows)
    require(limit is None or type(limit) is int and 0 <= limit <= rows, 'Physical prefix length')
    n = rows if limit is None else limit
    queries = []
    for i, (text, names) in enumerate(zip(meta['dates'][:n], meta['symbols'][:n])):
        d = iso(text)
        require(d in positions and (not queries or queries[-1][0] < d), 'Official increasing queries')
        t = positions[d]
        require(isinstance(names, (list, tuple)) and len(names) == width and width > 0
            and len(set(names)) == width and all(isinstance(s, str) and s and s.strip() == s for s in names), 'Query identities')
        if text == '2016-08-17':
            require(i == 0, 'Initial partial only at query start')
        else:
            require(t+1 < len(sessions) and sessions[t+1].to_period('M') != d.to_period('M'),
                    'Official month-end query required')
        queries.append((d, list(names), t))
    return frame, sessions, queries, eligible[:n], calm[:n], width


def prices_or_reason(frame, dates, symbol, role):
    if symbol not in frame.columns:
        return None, 'missing_market_symbol' if role == 'market' else 'missing_symbol'
    if any(d not in frame.index for d in dates):
        return None, 'missing_' + role + '_session'
    raw = [frame.at[d, symbol] for d in dates]
    if any(not isinstance(v, numbers.Real) or isinstance(v, (bool, np.bool_)) for v in raw):
        return None, 'nonnumeric_' + role + '_price'
    try:
        values = [float(v) for v in raw]
    except (ValueError, TypeError, OverflowError):
        return None, 'nonnumeric_' + role + '_price'
    if any(not math.isfinite(v) or v <= 0. for v in values):
        return None, 'invalid_' + role + '_price'
    return values, None


def market_state(prices):
    require(len(prices) == 200 and all(math.isfinite(float(p)) and float(p) > 0. for p in prices),
            'Exactly200 positive market prices')
    mean = finite(total([float(p) for p in prices], 'market close sum') / 200., 'market mean')
    return bool(prices[-1] > mean), mean


def drift_oracle(prices, matching):
    values = [float(v) for v in prices]
    L = len(values)-1
    mask = np.asarray(matching)
    require(L in (210, 252) and mask.shape == (L,) and mask.dtype == np.dtype(bool), 'Daily drift axes')
    require(all(math.isfinite(v) and v > 0. for v in values), 'Positive finite stock path')
    n = int(mask.sum())
    require(n >= 42, 'At least42 matched daily observations')
    returns = [finite(math.log(values[k+1])-math.log(values[k]), 'daily log difference') for k in range(L)]
    mu = finite(total(returns, 'unconditional log sum')/L, 'unconditional mean')
    cm = finite(total([v for v, use in zip(returns, mask) if use], 'matched log sum')/n, 'conditional mean')
    if n == L or cm == mu:
        candidate = mu
    else:
        difference = finite(cm-mu, 'conditional difference')
        shrinkage = n/(n+63.)
        correction = finite(shrinkage*difference, 'conditional correction')
        candidate = finite(mu+correction, 'conditional drift')
    return dict(unconditional_mean=mu, conditional_mean=cm, candidate_raw=candidate, control_raw=mu)


def effect_ranks(values):
    x = [float(v) for v in values]
    require(len(x) >= 2 and all(math.isfinite(v) for v in x), 'At least2 finite drift values')
    if all(v == x[0] for v in x):
        return [0.]*len(x)
    out = []
    for v in x:
        less = sum(z < v for z in x)
        equal = sum(z == v for z in x)
        average_rank = less+(equal+1)/2.
        out.append((average_rank-1.)/(len(x)-1.)-.5)
    return out


def independent_build(panel, meta, calendar, arrays, limit=None):
    frame, sessions, queries, eligible, calm, width = schema(panel, meta, calendar, arrays, limit)
    n = len(queries)
    result = {key: np.full((n, width), np.nan) for key in ARRAY_KEYS[:2]}
    result.update({key: np.zeros((n, width)) for key in ARRAY_KEYS[2:4]})
    result['ready'] = np.zeros(n, bool)
    events = []
    for i, (day, names, t) in enumerate(queries):
        E = np.flatnonzero(eligible[i]).tolist()
        L = 210 if calm[i] else 252
        c = {key: 0 for key in COUNT_FIELDS}
        c.update(query_months=1, eligible_cells=len(E))
        cells = [dict(column=j, symbol=names[j], source_checked=False, source_available=False,
            reason='not_checked', price_start=None, price_end=None, unconditional_mean=None,
            conditional_mean=None, candidate_raw=None, control_raw=None,
            candidate_effect=0., control_effect=0.) for j in E]
        e = dict(row=i, date=str(day.date()), symbols=names, eligible_indices=E, calm=bool(calm[i]),
            formation_returns=L, formation_start=None, formation_end=None,
            historical_state_first=None, historical_state_last=None, query_market_close=None,
            query_market_mean=None, query_state_up=None, historical_up_days=None,
            matching_state_days=None, ready=False, status=None, counts=c, cells=cells)
        if str(day.date()) == '2016-08-17':
            e['status'] = 'initial_partial'; c['initial_partial_months'] = 1
        elif len(E) < 2:
            e['status'] = 'fewer_than_two_eligible'; c['domain_fallback_months'] = 1
        elif t < L+42+199:
            e['status'] = 'insufficient_calendar'; c['source_fallback_months'] = 1
        else:
            u = t-42
            start = u-L
            stock_dates = sessions[start:u+1]
            historical_positions = list(range(start, u))
            e.update(formation_start=str(sessions[start].date()), formation_end=str(sessions[u].date()),
                historical_state_first=str(sessions[start].date()), historical_state_last=str(sessions[u-1].date()))
            required = sorted(set(k for p in [*historical_positions, t] for k in range(p-199, p+1)))
            market, reason = prices_or_reason(frame, [sessions[k] for k in required], 'SPY', 'market')
            if reason:
                e['status'] = reason; c['source_fallback_months'] = 1
            else:
                by_position = dict(zip(required, market))
                up, query_mean = market_state([by_position[k] for k in range(t-199, t+1)])
                prior_states = [market_state([by_position[k] for k in range(p-199, p+1)])[0]
                    for p in historical_positions]
                matched = np.asarray([s == up for s in prior_states], dtype=bool)
                e.update(query_market_close=by_position[t], query_market_mean=query_mean,
                    query_state_up=up, historical_up_days=sum(prior_states), matching_state_days=int(matched.sum()))
                c['market_state_evaluations'] = L+1
                c['market_price_observations'] = 200*(L+1)
                paths = {}
                for cell in cells:
                    j = cell['column']
                    p, reason = prices_or_reason(frame, stock_dates, names[j], 'stock')
                    cell.update(source_checked=True, source_available=reason is None,
                        reason='source_valid' if reason is None else reason)
                    c['checked_stock_cells'] += 1
                    if reason is None:
                        c['source_valid_stock_cells'] += 1
                        c['stock_price_observations'] += L+1
                        cell.update(price_start=p[0], price_end=p[-1]); paths[j] = p
                if len(paths) != len(E):
                    e['status'] = 'invalid_stock_source'; c['source_fallback_months'] = 1
                elif int(matched.sum()) < 42:
                    e['status'] = 'insufficient_matching_state'; c['state_support_fallback_months'] = 1
                else:
                    for cell in cells:
                        j = cell['column']
                        measured = drift_oracle(paths[j], matched)
                        cell.update(measured)
                        for key in ARRAY_KEYS[:2]: result[key][i, j] = measured[key]
                    for key in ('candidate', 'control'):
                        ranks = effect_ranks([cell[key+'_raw'] for cell in cells])
                        for cell, value in zip(cells, ranks):
                            cell[key+'_effect'] = value
                            result[key+'_effect'][i, cell['column']] = value
                    e.update(status='supported', ready=True); result['ready'][i] = True
                    c.update(supported_months=1, stock_log_returns=L*len(E),
                        drift_estimates=2*len(E), ranked_observations=2*len(E))
        c['fallback_months'] = int(not e['ready'])
        events.append(e)
    counts = {key: sum(e['counts'][key] for e in events) for key in COUNT_FIELDS}
    reasons = {status: sum(e['status'] == status for e in events) for status in sorted({e['status'] for e in events})}
    audit = dict(method='current_market_conditioned_skip42_daily_drift', arms=list(ARMS), events=n,
        symbols_per_event=width, skip_sessions=42, calm_returns=210, other_returns=252,
        market_symbol='SPY', market_mean_sessions=200, min_matching_days=42, shrinkage_days=63,
        market_price_observation_counting='200 per evaluated state; overlapping windows counted repeatedly',
        counts=counts, new_model_fits=0, predictive_model_fits=0, new_ridge_fits=0, new_tree_fits=0,
        reason_counts=reasons, event_history=events)
    return dict(arrays=result, audit=audit)


def check_build(panel, meta, calendar, arrays, built, limit=None):
    expected = independent_build(panel, meta, calendar, arrays, limit)
    require(isinstance(built, dict) and set(built) == {'arrays', 'audit'}, 'Build output fields')
    require(set(built['arrays']) == set(ARRAY_KEYS), 'Five output arrays')
    for key in ARRAY_KEYS:
        exact_array(built['arrays'][key], expected['arrays'][key], key)
    record_check(built['audit'], expected['audit'])
    return dict(**{key: True for key in MODEL_FLAGS}, counts=expected['audit']['counts'],
        query_rows=expected['audit']['events'], numerical_tolerance=0.,
        new_model_fits=0, predictive_model_fits=0, new_ridge_fits=0, new_tree_fits=0,
        producer_modules_imported=False, computes_returns=False)


def selection_oracle(original, meta, built):
    helper = frozen_source_oracle().old.original_helper()
    x = helper.original_prefix(original, meta, None)
    pre = x['local']; n, width = pre.shape
    a = built['arrays']
    require(set(a) == set(ARRAY_KEYS), 'Five selector arrays')
    ready = np.asarray(a['ready'])
    require(ready.shape == (n,) and ready.dtype == np.dtype(bool), 'Selector ready vector')
    effects = {arm: np.asarray(a[key+'_effect']) for arm, key in zip(ARMS[1:], ('candidate', 'control'))}
    budget = np.minimum(.05, np.maximum(0., .10-(.025*x['calm']+.05*x['weak'])))
    events = built['audit']['event_history']
    require(len(events) == n, 'Selector event axis')
    for i, e in enumerate(events):
        E = np.flatnonzero(x['eligible'][i]).tolist()
        require(e['row'] == i and e['date'] == meta['dates'][i] and e['eligible_indices'] == E
            and e['calm'] == bool(x['calm'][i]) and bool(ready[i]) == (e['status'] == 'supported'), 'Selector event provenance')
        if ready[i]:
            require(meta['dates'][i] != '2016-08-17' and len(E) >= 2, 'Unsupported selector readiness')
        support = np.asarray([ready[i] and j in E for j in range(width)], bool)
        for arm, key in zip(ARMS[1:], ('candidate', 'control')):
            risk = np.asarray(a[key+'_raw'])
            require(risk.shape == (n, width) and risk.dtype == np.float64 and not np.isinf(risk).any(), 'Raw selector shape')
            require(np.array_equal(np.isfinite(risk[i]), support), 'Raw common support')
            ef = effects[arm]
            require(ef.shape == (n, width) and ef.dtype == np.float64 and np.isfinite(ef).all()
                and np.all(np.abs(ef) <= .5) and not np.any(ef[i, ~support]), 'Selector effect support')
            if ready[i]: exact_array(ef[i, E], np.asarray(effect_ranks(risk[i, E])), 'Selector effect ranks')
    scores, choices = {}, {}
    for arm in ARMS:
        score = pre.copy(); selected = []
        for i, names in enumerate(x['names']):
            ordered = sorted(np.flatnonzero(x['eligible'][i]).tolist(), key=lambda j: (-pre[i, j], names[j]))
            if arm != ARMS[0] and ready[i]:
                for j in ordered:
                    target = min(.5, max(-.5, x['support'][i, j]+effects[arm][i, j]))
                    score[i, j] += budget[i]*(target-x['support'][i, j])
                ordered = sorted(ordered, key=lambda j: (-score[i, j], names[j]))
                selected.append(ordered[:3] if len(ordered) >= 2 else [])
            else:
                if len(ordered) < 3:
                    selected.append(ordered if len(ordered) == 2 else []); continue
                js = [j for j in ordered[2:5] if float(pre[i, ordered[2]])-float(pre[i, j]) <= .02]
                for j in js:
                    target = min(.5, max(-.5, x['support'][i, j]+x['effects'][i, j]))
                    score[i, j] += budget[i]*(target-x['support'][i, j])
                third = min(js, key=lambda j: (-score[i, j], names[j]))
                selected.append(sorted(ordered[:2]+[third], key=lambda j: (-score[i, j], names[j])))
        scores[arm] = score; choices[arm] = selected
    return scores, choices


def verify_selection(original, meta, built, scores, choices):
    require(set(scores) == set(choices) == set(ARMS), 'Three selection arms required')
    expected, ordered = selection_oracle(original, meta, built)
    for arm in ARMS:
        exact_array(scores[arm], expected[arm], arm+' full-eligible single-effect score')
        require(choices[arm] == ordered[arm], arm+' ordered choices differ')
    return dict(**{key: True for key in SELECTION_FLAGS}, ordered_choices_checked=3*len(meta['dates']),
        numerical_tolerance=0., original_source_parity_tolerance=1e-12,
        producer_modules_imported=False, computes_returns=False)


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = accounting()
    require(set(orders) == set(exports) == set(results) == set(ARMS), 'Outcome arm names differ')
    require(len(meta['dates']) == len(meta['ends']) == 122 and meta['ends'][:-1] == meta['dates'][1:], 'Outcome interval axis differs')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and meta['ends'][52] == meta['dates'][53] == '2020-12-31'
            and meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Outcome fixed slices differ')
    reports = {}
    for arm in ARMS:
        require(len(orders[arm]) == 122 and [o['date'] for o in orders[arm]] == meta['dates'] and
                [o['end'] for o in orders[arm]] == meta['ends'], 'Order axis differs')
        require(set(exports[arm]) == set(results[arm]) == {p[0] for p in PATHS}, 'Execution paths differ')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            data = exports[arm][name]
            require(set(data) == {'DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS'}, 'Incomplete execution exports')
            audited = helper.audit_path(px, opening, orders[arm], *[data[k] for k in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')], execution, cost)
            slices = helper.statistics_by_split(audited['days'], audited['values'], meta)
            wanted = results[arm][name]
            require(set(wanted['slices']) == set(slices), 'Fixed five slices differ')
            helper.check_metrics(slices['full122'], wanted['metrics'], arm + '/' + name)
            for split, value in slices.items():
                helper.check_metrics(value, wanted['slices'][split], arm + '/' + name + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                helper.near(audited[key], wanted['metrics'][key], key)
            annual = []
            for year in sorted({d[:4] for d in audited['days']}):
                ids = [i for i, day in enumerate(audited['days']) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                prior = audited['values'][ids[0] - 1] if ids[0] else 100000.
                annual.append(dict(year=int(year), return_=audited['values'][ids[-1]] / prior - 1., partial=int(year) in (2016, 2026)))
            observed = wanted['metrics']['annual_returns']
            require(isinstance(observed, list) and len(observed) == len(annual), 'Annual return count')
            for actual, expected in zip(observed, annual):
                require(set(actual) == set(expected) and actual['year'] == expected['year'] and
                        actual['partial'] == expected['partial'], 'Annual return identity')
                close(actual['return_'], expected['return_'], 'Annual return', atol=1e-12)
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
                independent_of_production_model_and_engine=True, terminal_liquidation=False,
                accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
