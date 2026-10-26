"""Independent scalar event-risk and mature rebound-payoff verification.

Only frozen independent portfolio accounting is imported for outcome audits.
The factor oracle has no production imports or actual-data command-line mode.
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
ARMS = ('incumbent_local67', 'event_risk_payoff', 'inverse_risk_mass_control')
RAW = {'event_risk_payoff': 'candidate', 'inverse_risk_mass_control': 'control'}
DOSE = .01


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('riskpayoff198_frozen_independent_accounts', FROZEN_HELPER)
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


FLOAT_RTOL = 2e-12
FLOAT_ATOL = 1e-14


def close_record(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key in expected:
            close_record(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            close_record(a, b, label + '/' + str(i))
    elif isinstance(expected, float):
        # Scale estimates must agree relatively, including very small positive risk.
        atol = 0. if label.endswith('/sigma') else FLOAT_ATOL
        require(type(actual) is float and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=FLOAT_RTOL, abs_tol=atol), label + ': numeric')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': value/type')


def finite(value, label):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite ' + label)
    return value


def accurate_sum(values, label):
    try:
        return finite(math.fsum(values), label)
    except OverflowError as exc:
        raise FloatingPointError('Overflow ' + label) from exc


def empty(reason, source_available=False):
    return dict(source_available=source_available, available=False, reason=reason,
        candidate=None, control=None, selected_anchors=0, valid_event_payoffs=0,
        anchor_indices=[], anchor_events=[], payoff_sum=None, inverse_risk_sum=None,
        standardized_payoff_sum=None, mean_inverse_risk=None, denominator=None, identity_branch=None)


def measure_oracle(prices):
    require(np.shape(prices) == (253,), 'Exactly253 stored adjusted prices required')
    raw = list(prices)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
        return empty('nonnumeric_adjusted_price')
    try:
        p = [float(x) for x in raw]
    except (TypeError, ValueError, OverflowError):
        return empty('nonnumeric_adjusted_price')
    if any(not math.isfinite(x) or x <= 0 for x in p):
        return empty('invalid_adjusted_price')
    selected = []
    for s in range(64, 232):
        if p[s] < p[s - 1] and (not selected or s - selected[-1] >= 21):
            selected.append(s)
    if not selected:
        return empty('no_selected_anchors', True)
    require(len(selected) <= 8, 'Impossible cooldown count')
    events = []
    for s in selected:
        returns = [finite(p[j] / p[j - 1] - 1., 'pre-event simple return') for j in range(s - 63, s)]
        mean = accurate_sum(returns, 'pre-event return sum') / 63
        squared = [finite((x - mean) * (x - mean), 'pre-event squared deviation') for x in returns]
        variance = finite(accurate_sum(squared, 'pre-event squared sum') / 62, 'sample variance')
        sigma = finite(math.sqrt(variance), 'sample risk')
        payoff = finite(math.log(p[s + 21]) - math.log(p[s]), 'mature log payoff')
        inverse = finite(1. / sigma, 'inverse risk') if sigma > 0 else None
        standardized = finite(payoff / sigma, 'standardized payoff') if sigma > 0 else None
        events.append(dict(anchor_index=s, reference_index=s - 1, reference_price=p[s - 1], anchor_price=p[s],
            risk_price_start_index=s - 64, risk_price_end_index=s - 1,
            risk_return_first_index=s - 63, risk_return_last_index=s - 1,
            pre_event_simple_returns=returns, sigma=sigma, maturity_index=s + 21,
            maturity_price=p[s + 21], payoff=payoff, inverse_sigma=inverse, standardized_payoff=standardized))
    n = len(selected)
    result = empty('zero_pre_event_sigma', True)
    result.update(selected_anchors=n, anchor_indices=selected, anchor_events=events,
                  payoff_sum=accurate_sum([e['payoff'] for e in events], 'payoff sum'),
                  denominator=finite((n + 20) * math.sqrt(21.), 'denominator'))
    if any(e['sigma'] == 0 for e in events):
        return result
    inverse_sum = accurate_sum([e['inverse_sigma'] for e in events], 'inverse risk sum')
    standardized_sum = accurate_sum([e['standardized_payoff'] for e in events], 'standardized payoff sum')
    average_inverse = finite(inverse_sum / n, 'mean inverse risk')
    candidate = finite(standardized_sum / result['denominator'], 'candidate')
    risks, payoffs = [e['sigma'] for e in events], [e['payoff'] for e in events]
    identity = ('single_event' if n == 1 else 'equal_risks' if all(x == risks[0] for x in risks)
                else 'equal_payoffs' if all(x == payoffs[0] for x in payoffs) else None)
    if identity is not None:
        control = candidate
    else:
        numerator = finite(average_inverse * result['payoff_sum'], 'control numerator')
        control = finite(numerator / result['denominator'], 'control')
    result.update(available=True, reason='available', candidate=candidate, control=control,
        valid_event_payoffs=n, inverse_risk_sum=inverse_sum, standardized_payoff_sum=standardized_sum,
        mean_inverse_risk=average_inverse, identity_branch=identity)
    return result


def centered_ranks(raw, available):
    out = np.zeros(np.shape(raw), float)
    for i in range(len(out)):
        members = [j for j, good in enumerate(available[i]) if good]
        if len(members) < 2:
            continue
        for j in members:
            lower = sum(raw[i, k] < raw[i, j] for k in members)
            equal = sum(raw[i, k] == raw[i, j] for k in members)
            out[i, j] = (lower + (equal - 1) / 2) / (len(members) - 1) - .5
    return out


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons, source_counts, available_counts, selected_counts, payoff_counts = [], {}, [], [], [], []
    frame = panel['adj']
    for i, (day, names, t) in enumerate(queries):
        dates = sessions[t - 294:t - 41] if t >= 294 else []
        missing = [str(d.date()) for d in dates if d not in frame.index]
        cells = []
        for j, symbol in enumerate(names):
            reason = ('insufficient_history' if not dates else 'missing_symbol' if symbol not in frame.columns
                      else 'missing_official_session' if missing else None)
            cell = empty(reason) if reason else measure_oracle([frame.at[d, symbol] for d in dates])
            cell.update(symbol=symbol, date=str(day.date()), candidate_rank=0., control_rank=0.)
            values['available'][i, j] = cell['available']
            values['candidate'][i, j], values['control'][i, j] = cell['candidate'], cell['control']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        source_mask = [cell['source_available'] for cell in cells]
        available_mask = [cell['available'] for cell in cells]
        source_counts.append(sum(source_mask)); available_counts.append(sum(available_mask))
        selected_counts.append(sum(cell['selected_anchors'] for cell in cells))
        payoff_counts.append(sum(cell['valid_event_payoffs'] for cell in cells))
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=str(dates[0].date()) if dates else None, price_end=str(dates[-1].date()) if dates else None,
            anchor_first_date=str(dates[64].date()) if dates else None,
            anchor_last_date=str(dates[231].date()) if dates else None,
            missing_official_sessions=missing, source_available=source_mask, available=available_mask,
            source_available_count=sum(source_mask), available_count=sum(available_mask), cells=cells))
    for raw in ('candidate', 'control'):
        values[raw + '_rank'] = centered_ranks(values[raw], values['available'])
        for i, event in enumerate(history):
            for j, cell in enumerate(event['cells']):
                cell[raw + '_rank'] = float(values[raw + '_rank'][i, j])
    selected, source, available = sum(selected_counts), sum(source_counts), sum(available_counts)
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=253, skip_sessions=42, anchor_first=64, anchor_last=231,
        risk_returns=63, payoff_horizon=21, cooldown=21, prior_count=20,
        source_available_cells=source, source_available_by_event=source_counts,
        available_cells=available, available_by_event=available_counts,
        unavailable_cells=int(values['available'].size) - available, reason_counts=reasons,
        valid_price_observations=253 * source, selected_anchors=selected,
        selected_anchors_by_event=selected_counts, pre_event_return_observations=63 * selected,
        matured_payoffs=selected, valid_event_payoffs=sum(payoff_counts), valid_event_payoffs_by_event=payoff_counts,
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
    for raw in ('candidate', 'control'):
        v, r = a[raw], a[raw + '_rank']
        require(v.shape == r.shape == mask.shape and np.issubdtype(v.dtype, np.floating)
                and np.issubdtype(r.dtype, np.floating), 'Float feature/rank matrices required')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v), mask), 'Common raw availability differs')
        require(np.isfinite(r).all() and np.all(np.abs(r) <= .5), 'Finite bounded ranks required')
        require(np.array_equal(r, centered_ranks(v, mask)), 'Independent centered ranks differ')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, limit)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape, 'Feature shape differs')
        if key in ('candidate', 'control'):
            require(np.allclose(actual[key], wanted[key], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True),
                    'Independent event-risk raw differs: ' + key)
        else:
            require(np.array_equal(actual[key], wanted[key]), 'Independent availability/ranks differ: ' + key)
    close_record(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'source_available_cells', 'source_available_by_event',
                'available_cells', 'available_by_event', 'unavailable_cells', 'valid_price_observations',
                'selected_anchors', 'selected_anchors_by_event', 'pre_event_return_observations',
                'matured_payoffs', 'valid_event_payoffs', 'valid_event_payoffs_by_event')
    return dict(passed=True, **{key: expected[key] for key in counters}, independent_scalar_rebuild=True,
        negative_anchor_selection_checked=True, pre_event_risk_checked=True, matured_payoffs_checked=True,
        matched_inverse_risk_mass_checked=True, full253_price_support_checked=True,
        official_session_windows_checked=True, monthend_queries_checked=True, exact_common_availability=True,
        audit_checked=True, no_future_or_outcome_inputs=True, centered_high_tie_ranks_checked=True,
        predictive_model_fits=0, new_model_fits=0, computes_returns=False,
        numerical_comparison=dict(relative_tolerance=FLOAT_RTOL, absolute_tolerance=FLOAT_ATOL,
            sigma_absolute_tolerance=0., raw_ranks_exact=True, scalar_risk='math.fsum two-pass sample variance'),
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
