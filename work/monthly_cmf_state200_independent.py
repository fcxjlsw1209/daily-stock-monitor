"""Independent CMF-state oracle; synthetic/import APIs only, no data loader.

Canonical pandas rolling statistics retain the frozen source's float ties.
Every consumed sigma is also checked against a separately evaluated 63-return
sample standard deviation. That check alone has a numerical tolerance; ranks,
CDF comparisons, source regimes, and corrections are compared exactly.
"""
from datetime import date
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
ARMS = ('incumbent_local67', 'cmf_empirical_state', 'cmf_half_sign_control')
GATES = {'cmf_empirical_state': 'gate_candidate', 'cmf_half_sign_control': 'gate_control'}
PANEL_KEYS = {'adj', 'close', 'high', 'low', 'volume'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('cmf200_independent_accounts', FROZEN_HELPER)
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
    result = helper.audit_outcomes(px, opening, orders, exports, results, meta)
    require(result['checked_paths'] == 9 and result['checked_slices'] == 45 and
            set(result['paths']) == set(ARMS), 'Incomplete nine-path accounting')
    result['outcome_adapter_sha256'] = FROZEN_HELPER_SHA
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


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == PANEL_KEYS, 'Original five signal panels required')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Frame type')
        idx = frame.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
                idx.equals(idx.normalize()), 'Invalid daily axis')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
                for s in frame.columns), 'Invalid source symbol labels')
        require(idx.equals(panel['adj'].index), 'Original source axes differ')
    require('SPY' in panel['adj'], 'Original SPY series required')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = pd.DatetimeIndex([iso(x) for x in calendar])
    require(len(sessions) > 0 and all(d.weekday() < 5 for d in sessions) and
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


def gate_oracle(window):
    x = np.asarray(window, dtype=float)
    require(x.shape == (252,) and np.isfinite(x).all() and (x >= 0).all(),
            '252 finite nonnegative canonical sigmas required')
    current = float(x[-1])
    # pandas median is part of the original state, not an alternative threshold.
    median = float(pd.Series(x).median())
    less = sum(float(z) < current for z in x)
    equal = sum(float(z) == current for z in x)
    state = 1. if current > median else -1.
    gate = 0. if current == median else (2 * less + equal) / 252. - 1.
    require(-1. < gate < 1. and gate * state >= 0., 'Sign-preserving empirical gate')
    return dict(sigma=current, median=median, less=less, equal=equal, median_tie=current == median,
                state=state, gate_candidate=gate, gate_control=.5 * state)


def rank_oracle(raw):
    x = np.asarray(raw, dtype=float)
    require(x.ndim == 1 and not np.isinf(x).any(), 'Finite-or-missing CMF vector required')
    members = [float(z) for z in x if math.isfinite(float(z))]
    result = np.zeros(x.shape, dtype=float)
    for j, value in enumerate(x):
        if math.isfinite(float(value)):
            less = sum(z < value for z in members)
            equal = sum(z == value for z in members)
            result[j] = (less + (equal + 1) / 2.) / len(members) - .5
    return result


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    n = len(queries)
    values = {key: np.empty(n, float) for key in
              ('state', 'gate_candidate', 'gate_control', 'coefficient', 'sigma', 'median')}
    values.update(calm=np.empty(n, bool), cmf_rank=np.empty((n, width), float),
                  sigma_window=np.empty((n, 252), float))
    history, sigma_checked = [], set()
    if n:
        # Keep the original historical axis and physical prefix before conversion.
        prefix = {key: frame.loc[:queries[-1][0]].copy() for key, frame in panel.items()}
        spy = prefix['adj']['SPY'].astype(float)
        canonical = spy.pct_change(fill_method=None).rolling(63).std(ddof=1)
        close, high, low, volume = (prefix[key].astype(float) for key in ('close', 'high', 'low', 'volume'))
        spread = high - low
        location = ((2 * close - high - low) / spread.replace(0, np.nan)).where(spread != 0, 0.)
        numerator = (location * volume).rolling(63).sum()
        denominator = volume.rolling(63).sum()
        negative_cmf = (-numerator / denominator).replace([np.inf, -np.inf], np.nan)
        for i, (day, symbols, t) in enumerate(queries):
            require(t >= 314, 'Insufficient official SPY history')
            dates = sessions[t-314:t+1]
            require(prefix['adj'].loc[dates[0]:day].index.equals(dates), 'Exact315 official sessions required')
            price = spy.loc[dates].to_numpy()
            require(np.isfinite(price).all() and (price > 0).all(), 'Invalid SPY price; no fill')
            require(all(set(symbols).issubset(prefix[key].columns) for key in PANEL_KEYS), 'Query source symbol missing')
            window = canonical.loc[dates[-252:]].to_numpy(copy=True)
            state = gate_oracle(window)
            # Separate scalar window evaluation; never replaces the canonical tie values.
            for j, sigma_day in enumerate(dates[-252:]):
                if sigma_day in sigma_checked:
                    continue
                prices64 = price[j:j+64]
                with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                    returns63 = np.asarray([float(prices64[k]) / float(prices64[k-1]) - 1.
                                            for k in range(1, 64)], dtype=float)
                    scalar_sigma = float(np.std(returns63, ddof=1))
                require(math.isfinite(scalar_sigma) and math.isclose(float(window[j]), scalar_sigma,
                        rel_tol=1e-10, abs_tol=1e-12), 'Independent63-return sigma differs')
                sigma_checked.add(sigma_day)
            calm = bool(state['sigma'] * np.sqrt(252) < .10)
            coefficient = .075 if calm else .075 * (1 - (.10 if state['state'] > 0 else 0.) / .48)
            raw = negative_cmf.loc[day, symbols].to_numpy()
            values['cmf_rank'][i] = rank_oracle(raw)
            values['sigma_window'][i] = window
            values['calm'][i] = calm
            values['coefficient'][i] = coefficient
            for key in ('sigma', 'median', 'state', 'gate_candidate', 'gate_control'):
                values[key][i] = state[key]
            history.append(dict(date=str(day.date()), symbols=symbols,
                spy_window_start=str(dates[0].date()), spy_window_end=str(day.date()),
                sigma_window_start=str(dates[-252].date()), session_position=t,
                **{key: state[key] for key in ('less', 'equal', 'median_tie', 'state', 'gate_candidate', 'gate_control')},
                coefficient=coefficient, calm=calm, cmf_available_count=int(np.isfinite(raw).sum())))
    finite = sum(row['cmf_available_count'] for row in history)
    audit = dict(events=n, symbols_per_event=width, checked_cells=n*width,
        finite_cmf_cells=finite, neutral_cmf_cells=n*width-finite, spy_price_observations=315*n,
        volatility_rank_observations=252*n, new_model_fits=0, predictive_model_fits=0,
        portfolio_returns_calculated=False, event_history=history)
    return values, audit


def feature_arrays(values):
    keys = {'cmf_rank', 'state', 'gate_candidate', 'gate_control', 'coefficient', 'calm',
            'sigma', 'median', 'sigma_window'}
    require(isinstance(values, dict) and set(values) == keys, 'Feature keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    require(a['cmf_rank'].ndim == 2, 'CMF rank matrix required')
    n, width = a['cmf_rank'].shape
    for key in keys:
        v = a[key]
        shape = (n, width) if key == 'cmf_rank' else (n, 252) if key == 'sigma_window' else (n,)
        require(v.shape == shape, 'Feature shape: ' + key)
        if key == 'calm':
            require(v.dtype == bool, 'Boolean calm required')
        else:
            require(np.issubdtype(v.dtype, np.floating) and np.isfinite(v).all(), 'Finite float feature: ' + key)
    require((np.abs(a['cmf_rank']) <= .5).all(), 'Unbounded original percentile rank')
    for i in range(n):
        gate = gate_oracle(a['sigma_window'][i])
        for key in ('sigma', 'median', 'state', 'gate_candidate', 'gate_control'):
            require(a[key][i] == gate[key], 'Feature gate differs: ' + key)
        calm = bool(gate['sigma'] * np.sqrt(252) < .10)
        coefficient = .075 if calm else .075 * (1 - (.10 if gate['state'] > 0 else 0.) / .48)
        require(bool(a['calm'][i]) == calm and a['coefficient'][i] == coefficient, 'Original regime coefficient differs')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = feature_arrays(values)
    expected_values, expected_audit = independent_features(panel, meta, calendar, limit)
    for key in expected_values:
        require(np.array_equal(actual[key], expected_values[key]), 'Independent feature differs: ' + key)
    record_check(audit, expected_audit)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'finite_cmf_cells', 'neutral_cmf_cells',
                'spy_price_observations', 'volatility_rank_observations', 'new_model_fits', 'predictive_model_fits')
    receipt = {key: expected_audit[key] for key in counters}
    receipt.update(passed=True, independent_scalar_rebuild=True, canonical_sigma_checked=True,
        empirical_gate_checked=True, median_tie_neutral_checked=True, original_cmf_rank_checked=True,
        regime_coefficient_checked=True, strict315_official_spy_window_checked=True,
        monthend_queries_checked=True, audit_checked=True, no_future_or_outcome_inputs=True,
        computes_returns=False, scalar_sigma_tolerance={'atol': 1e-12, 'rtol': 1e-10},
        feature_comparison='exact; tolerance only for separate scalar sigma cross-check')
    return receipt


def independent_deltas(values, arrays):
    a = feature_arrays(values)
    n, width = a['cmf_rank'].shape
    require(isinstance(arrays, dict), 'Original arrays dictionary required')
    calm_full = np.asarray(arrays['state_calm10'])
    require(calm_full.ndim == 1 and len(calm_full) >= n, 'Original calm axis')
    full = len(calm_full)
    calm = calm_full[:n]
    weak_full = np.asarray(arrays['state_weak_narrow'])
    require(weak_full.shape == (full,), 'Original weak axis')
    weak = weak_full[:n]
    require(calm.dtype == bool and weak.dtype == bool, 'Original states must be boolean')
    require(np.array_equal(calm, a['calm']), 'Original calm/source mismatch')
    raw_cw = np.asarray(arrays['core_weight'])
    require(raw_cw.shape in ((full,), (full, 1), (full, width)), 'Original core-weight shape')
    cw = np.asarray(raw_cw[:n], dtype=float)
    if cw.ndim == 1:
        cw = cw[:, None]
    cw = np.broadcast_to(cw, (n, width))
    require(np.isfinite(cw).all() and np.array_equal(cw, np.broadcast_to(np.where(calm, .5, .52)[:, None], cw.shape)),
            'Original core weights differ')
    matrices = {}
    for key in ('extension_available', 'forecast_available', 'eligible', 'rank_log_excess'):
        raw = np.asarray(arrays[key])
        require(raw.shape == (full, width), 'Original matrix shape: ' + key)
        part = raw[:n]
        if key == 'rank_log_excess':
            matrices[key] = np.asarray(part, dtype=float)
        else:
            require(part.dtype == bool, 'Original boolean matrix: ' + key)
            matrices[key] = part
    corrections = {arm: np.zeros((n, width), float) for arm in GATES}
    scales = np.empty((n, width), float)
    active_cells = 0
    for i in range(n):
        for j in range(width):
            active = bool(calm[i] and not weak[i] and matrices['extension_available'][i, j] and
                          matrices['forecast_available'][i, j] and matrices['eligible'][i, j] and
                          math.isfinite(float(matrices['rank_log_excess'][i, j])))
            active_cells += int(active)
            budget = .025 * bool(calm[i]) + .05 * bool(weak[i]) + .01 * active
            scale = 1. - budget / (1. - float(cw[i, j]))
            require(math.isfinite(scale) and .85 <= scale <= 1., 'Original support scale out of range')
            scales[i, j] = scale
            if i == 0:
                continue
            for arm, gatekey in GATES.items():
                value = (float(a['coefficient'][i]) * float(a['cmf_rank'][i, j]) *
                         (float(a[gatekey][i]) - float(a['state'][i]))) * scale
                require(math.isfinite(value) and abs(value) <= .0375, 'CMF correction exceeds budget')
                corrections[arm][i, j] = value
    evidence = dict(effective_support_scale_min=float(scales.min()) if scales.size else None,
                    effective_support_scale_max=float(scales.max()) if scales.size else None,
                    active_cells=active_cells)
    return corrections, evidence


def verify_deltas(values, arrays, deltas):
    expected, evidence = independent_deltas(values, arrays)
    require(isinstance(deltas, dict) and set(deltas) == set(GATES), 'Delta arm keys differ')
    changed = {}
    for arm, wanted in expected.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == wanted.shape and np.issubdtype(actual.dtype, np.floating) and
                np.array_equal(actual, wanted), 'Independent CMF correction differs: ' + arm)
        changed[arm] = int(np.count_nonzero(wanted))
    return dict(passed=True, original_calm_and_core_weights_checked=True,
        cmf_state_delta_formula_checked=True, effective_support_scale_checked=True,
        original_state_preserved=True, first_partial_unchanged=True, missing_cmf_neutral=True,
        no_new_rank=True, bounded_correction_checked=True, independent_delta_rebuild=True,
        changed_cells=changed, max_absolute_delta=.0375, **evidence)
