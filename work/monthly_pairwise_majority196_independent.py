"""Independent integer-cross-product monthly tournament and accounting audit.

No production feature imports. Comparisons are exact for canonical stored
binary64 prices, not a certification of vendor vintage or adjustment history.
Block order is irrelevant; shared opponents are not independent observations.
"""
from datetime import date
from fractions import Fraction
import hashlib
import importlib.util
import math
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'tournament_majority', 'tournament_balance_control')
RAW = {'tournament_majority': 'candidate', 'tournament_balance_control': 'control'}
DOSE = .01


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('majority196_frozen_independent_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_files():
    return tuple(sorted(set(frozen_helper().dependency_files()) | {Path(__file__).resolve()}))


def accounting():
    return frozen_helper().accounting()


def audit_outcomes(px, opening, orders, exports, results, meta):
    helper = frozen_helper()
    helper.ARMS = ARMS
    receipt = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(receipt['checked_paths'] == 9 and receipt['checked_slices'] == 45 and
            set(receipt['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    receipt['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return receipt


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


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj'}, 'Panel keys differ')
    frame = panel['adj']
    require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Adjusted frame type')
    idx = frame.index
    require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
            idx.equals(idx.normalize()), 'Invalid adjusted daily axis')
    require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
            for s in frame.columns), 'Invalid adjusted symbol labels')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    positions = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for value, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(value)
        require(day in positions and (not queries or day > queries[-1][0]), 'Event session/order')
        t = positions[day]
        if queries or value != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event symbol')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


def sign(value):
    return int(value > 0) - int(value < 0)


def cross_sign(end_i, start_i, end_j, start_j):
    """Each argument is an exact (integer numerator, denominator) pair."""
    nei, dei = end_i
    nsi, dsi = start_i
    nej, dej = end_j
    nsj, dsj = start_j
    left = nei * nsj * dej * dsi
    right = nej * nsi * dei * dsj
    return int(left > right) - int(left < right)


def compare_ratios(end_i, start_i, end_j, start_j):
    raw = (end_i, start_i, end_j, start_j)
    require(all(not isinstance(x, (bool, np.bool_)) and isinstance(x, numbers.Real) for x in raw),
            'Numeric binary64 endpoints required')
    values = [float(x) for x in raw]
    require(all(math.isfinite(x) and x > 0 for x in values), 'Positive finite endpoints required')
    return cross_sign(*[x.as_integer_ratio() for x in values])


def source_oracle(prices):
    require(np.shape(prices) == (253,), 'Exactly253 stored adjusted prices required')
    raw = list(prices)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return 'nonnumeric_adjusted_price', []
    try:
        values = [float(x) for x in raw]
    except (TypeError, ValueError, OverflowError):
        return 'nonnumeric_adjusted_price', []
    if any(not math.isfinite(x) or x <= 0 for x in values):
        return 'invalid_adjusted_price', []
    return 'available', [values[k] for k in range(0, 253, 21)]


def tournament_oracle(endpoints):
    """Independent ordered-pair rebuild on valid13-endpoint vectors."""
    require(all(len(v) == 13 and all(math.isfinite(x) and x > 0 for x in v) for v in endpoints),
            'Thirteen positive finite endpoints per comparison member required')
    n = len(endpoints)
    integer_ratios = [[float(x).as_integer_ratio() for x in member] for member in endpoints]
    margins = [[0] * n for _ in range(n)]
    observations = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            blocks = [cross_sign(integer_ratios[i][k + 1], integer_ratios[i][k],
                                 integer_ratios[j][k + 1], integer_ratios[j][k]) for k in range(12)]
            observations[(i, j)] = blocks
            margins[i][j] = sum(blocks)
    candidate, control = [], []
    for i in range(n):
        candidate.append(float(Fraction(sum(sign(m) for m in margins[i]), n - 1)) if n >= 2 else None)
        control.append(float(Fraction(sum(margins[i]), 12 * (n - 1))) if n >= 2 else None)
    return candidate, control, margins, observations


def centered_ranks(raw, available):
    out = np.zeros(np.shape(raw), float)
    for i in range(len(out)):
        members = [j for j, good in enumerate(available[i]) if good]
        n = len(members)
        if n < 2:
            continue
        for j in members:
            lower = sum(raw[i, k] < raw[i, j] for k in members)
            equal = sum(raw[i, k] == raw[i, j] for k in members)
            out[i, j] = (lower + (equal - 1) / 2) / (n - 1) - .5
    return out


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    if limit is not None:
        queries = queries[:limit]
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons, domain_counts = [], {}, []
    frame = panel['adj']
    for i, (day, names, t) in enumerate(queries):
        dates = sessions[t - 294:t - 41] if t >= 294 else []
        missing = [str(d.date()) for d in dates if d not in frame.index]
        sources, endpoints = [], []
        for symbol in names:
            reason = ('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                      else 'missing_official_session' if missing else None)
            if reason:
                endpoint = []
            else:
                reason, endpoint = source_oracle([frame.at[d, symbol] for d in dates])
            sources.append(reason)
            endpoints.append(endpoint)
        domain = [j for j, reason in enumerate(sources) if reason == 'available']
        n = len(domain)
        domain_counts.append(n)
        candidate, control, margins, comparisons = tournament_oracle([endpoints[j] for j in domain])
        pairs = [dict(left_index=domain[a], right_index=domain[b], block_signs=comparisons[(a, b)],
                      margin=margins[a][b], majority_sign=sign(margins[a][b]))
                 for a in range(n) for b in range(a + 1, n)]
        cells = []
        for j, symbol in enumerate(names):
            source_good = sources[j] == 'available'
            good = source_good and n >= 2
            reason = 'available' if good else 'insufficient_comparison_domain' if source_good else sources[j]
            q = candidate[domain.index(j)] if good else None
            f = control[domain.index(j)] if good else None
            cell = dict(symbol=symbol, date=str(day.date()), source_available=source_good, source_reason=sources[j],
                available=good, reason=reason, endpoint_prices=endpoints[j], opponent_count=n - 1 if good else 0,
                majority_sum=sum(sign(x) for x in margins[domain.index(j)]) if good else None,
                margin_sum=sum(margins[domain.index(j)]) if good else None,
                candidate=q, control=f, candidate_rank=0., control_rank=0.)
            values['available'][i, j] = good
            values['candidate'][i, j] = q
            values['control'][i, j] = f
            reasons[reason] = reasons.get(reason, 0) + 1
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=str(dates[0].date()) if dates else None,
            price_end=str(dates[-1].date()) if dates else None,
            endpoint_dates=[str(dates[k].date()) for k in range(0, 253, 21)] if dates else [],
            missing_official_sessions=missing, source_available=[r == 'available' for r in sources],
            source_indices=domain, domain_count=n, source_available_count=n, available_count=n if n >= 2 else 0,
            pair_margin_matrix=margins, pairs=pairs, cells=cells))
    for raw in ('candidate', 'control'):
        key = raw + '_rank'
        values[key] = centered_ranks(values[raw], values['available'])
        for i, event in enumerate(history):
            for j, cell in enumerate(event['cells']):
                cell[key] = float(values[key][i, j])
    available = int(values['available'].sum())
    source_available = sum(domain_counts)
    pair_count = sum(n * (n - 1) // 2 for n in domain_counts)
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=253, skip_sessions=42, block_count=12, block_sessions=21,
        source_available_cells=source_available, source_available_by_event=domain_counts,
        available_cells=available, unavailable_cells=int(values['available'].size) - available,
        reason_counts=reasons, valid_price_observations=253 * source_available,
        block_comparisons=12 * pair_count, pair_majority_decisions=pair_count,
        predictive_model_fits=0, new_model_fits=0, common_availability=True, ranking_performed=True,
        ranking_eligibility_filtered=False, trading_eligibility_modified=False,
        portfolio_returns_calculated=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) ==
            {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'}, 'Feature keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean availability required')
    require(all(int(row.sum()) != 1 for row in mask), 'Feature domain cannot have exactly one member')
    for raw in ('candidate', 'control'):
        v, r = a[raw], a[raw + '_rank']
        require(v.shape == r.shape == mask.shape and np.issubdtype(v.dtype, np.floating)
                and np.issubdtype(r.dtype, np.floating), 'Float feature/rank matrices required')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v), mask), 'Common raw availability differs')
        require(np.isfinite(r).all() and np.all(np.abs(r) <= .5), 'Finite bounded ranks required')
        require(np.all(np.abs(v[mask]) <= 1.), 'Raw outside[-1,1]')
        require(np.array_equal(r, centered_ranks(v, mask)), 'Independent centered ranks differ')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, limit)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape and
                np.array_equal(actual[key], wanted[key], equal_nan=True), 'Independent tournament differs: ' + key)
    record_check(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'source_available_cells', 'source_available_by_event',
                'available_cells', 'unavailable_cells', 'valid_price_observations', 'block_comparisons',
                'pair_majority_decisions')
    return dict(passed=True, **{key: expected[key] for key in counters}, independent_scalar_rebuild=True,
        exact_binary_ratio_comparisons_checked=True, opponent_specific_majorities_checked=True,
        matched_balance_checked=True, full253_price_support_checked=True, official_session_windows_checked=True,
        monthend_queries_checked=True, exact_common_availability=True, audit_checked=True,
        no_future_or_outcome_inputs=True, centered_high_tie_ranks_checked=True,
        predictive_model_fits=0, new_model_fits=0, computes_returns=False,
        outcome_adapter_sha256=FROZEN_HELPER_SHA)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (float, int) and dose == DOSE, 'Fixed dose .01 required')
    a = arrays(values)
    result = {arm: np.zeros(a['available'].shape, float) for arm in RAW}
    for arm, raw in RAW.items():
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                result[arm][i, j] = dose * float(a[raw + '_rank'][i, j])
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    wanted = independent_deltas(values, dose)
    for arm, expected in wanted.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == expected.shape and np.issubdtype(actual.dtype, np.floating)
                and np.isfinite(actual).all() and np.array_equal(actual, expected), 'Exact delta differs: ' + arm)
        require(np.all(np.abs(actual) <= DOSE * .5), 'Unbounded delta')
    return dict(passed=True, checked_events=np.shape(values['available'])[0], dose=DOSE, max_absolute_delta=.005,
        full_candidate_universe=True, initial_delta_zero=True, common_available_only=True,
        no_eligibility_or_model_filter=True, centered_cross_sectional_ranking=True, direct_additive_pre_score=True,
        fixed_linear_mapping_checked=True,
        changed_cells={arm: int(np.count_nonzero(value)) for arm, value in wanted.items()})
