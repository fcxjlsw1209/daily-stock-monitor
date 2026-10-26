"""Independent exact-rational block recurrence and frozen account verification.

Feature calculations are self-contained and import no production strategy or
kernel. Only the pinned round189 mapping and independent accounts are reused.
No real-data command-line entry point, fitting, portfolio replay or writes.
"""
from datetime import date
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_HELPER = ROOT / 'work/monthly_sector_transition189_independent.py'
FROZEN_HELPER_SHA = 'e83305d9f784ef9aa5ae0554cd1deb558d927b4182010af599b977efa69618dc'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
ARMS = ('incumbent_local67', 'sector_relative_recurrence', 'absolute_recurrence_control')
RAW = dict(sector_relative_recurrence='candidate', absolute_recurrence_control='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE, MAX_DELTA = .025, .01875


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_helper():
    require(sha(FROZEN_HELPER) == FROZEN_HELPER_SHA, 'Frozen189 independent helper changed')
    spec = importlib.util.spec_from_file_location('recurrence192_frozen_mapping_accounts', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.MAPPING_SHA == MAPPING_SHA, 'Unexpected frozen mapping source')
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
    result['mapping_and_outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return result


def benchmark_for(day, symbol, sectors, etfs):
    """Expose the pinned independently stated original mapping for synthetics."""
    helper = frozen_helper()
    helper.mapping_constants()
    return helper.benchmark_for(day, symbol, sectors, etfs)


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': exact keys')
        for key, value in expected.items():
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (a, b) in enumerate(zip(actual, expected)):
            record_check(a, b, label + '/' + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    day = date.fromisoformat(value)
    require(day.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(day)


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'sectors'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        idx = frame.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None
                and idx.equals(idx.normalize()), 'Invalid daily panel axis')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
                for s in frame.columns), 'Invalid panel identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF metadata')
    sessions = [iso(x) for x in calendar]
    require(sessions and all(x.weekday() < 5 for x in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {d: i for i, d in enumerate(sessions)}
    queries, width = [], 0
    for raw, symbols in zip(meta['dates'], meta['symbols']):
        day = iso(raw)
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        t = lookup[day]
        if queries or raw != '2016-08-17':
            require(t + 1 < len(sessions) and sessions[t + 1].to_period('M') != day.to_period('M'),
                    'Verified official month-end required')
        require(isinstance(symbols, (list, tuple)) and symbols and len(symbols) == len(set(symbols)),
                'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event name')
        require(not queries or width == len(symbols), 'Changing event width')
        width = len(symbols)
        queries.append((day, list(symbols), t))
    return sessions, queries, width


def empty(reason):
    return dict(available=False, reason=reason, candidate=None, control=None,
        endpoint_indices=[], block_start_indices=[], block_end_indices=[], relative_signs=[], absolute_signs=[],
        relative_positive_count=0, relative_negative_count=0, relative_tie_count=0,
        absolute_positive_count=0, absolute_negative_count=0, absolute_tie_count=0,
        relative_signed_count=0, absolute_signed_count=0,
        relative_sign_observations=0, absolute_sign_observations=0)


def relative_sign(stock_end, stock_start, sector_end, sector_start):
    values = [float(x) for x in (stock_end, stock_start, sector_end, sector_start)]
    require(all(math.isfinite(x) and x > 0 for x in values), 'Positive finite ratio endpoints required')
    a, b, c, d = [Fraction.from_float(x) for x in values]
    own, sector = a / b, c / d
    return int(own > sector) - int(own < sector)


def measure_oracle(stock, benchmark):
    require(np.shape(stock) == np.shape(benchmark) == (253,), 'Exactly253 prices per series required')
    vectors = []
    for raw, name in ((stock, 'stock'), (benchmark, 'benchmark')):
        try:
            vector = [float(x) for x in raw]
        except (TypeError, ValueError, OverflowError):
            return empty('nonnumeric_' + name + '_price')
        if any(not math.isfinite(x) or x <= 0 for x in vector):
            return empty('invalid_' + name + '_price')
        vectors.append(vector)
    own, sector = vectors
    points = list(range(0, 253, 21))
    starts, ends = points[:-1], points[1:]
    relative = [relative_sign(own[b], own[a], sector[b], sector[a]) for a, b in zip(starts, ends)]
    absolute = [int(own[b] > own[a]) - int(own[b] < own[a]) for a, b in zip(starts, ends)]
    out = empty('available')
    out.update(available=True, candidate=float(Fraction(sum(relative), 16)),
        control=float(Fraction(sum(absolute), 16)), endpoint_indices=points,
        block_start_indices=starts, block_end_indices=ends, relative_signs=relative, absolute_signs=absolute,
        relative_positive_count=relative.count(1), relative_negative_count=relative.count(-1),
        relative_tie_count=relative.count(0), absolute_positive_count=absolute.count(1),
        absolute_negative_count=absolute.count(-1), absolute_tie_count=absolute.count(0),
        relative_signed_count=sum(relative), absolute_signed_count=sum(absolute),
        relative_sign_observations=12, absolute_sign_observations=12)
    return out


def independent_features(panel, meta, calendar, benchmark_map, limit=None):
    helper = frozen_helper()
    helper.mapping_constants()
    sessions, queries, width = schema(panel, meta, calendar)
    require(callable(benchmark_map), 'Original benchmark_map callable required')
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan), control=np.full((len(queries), width), np.nan),
                  available=np.zeros((len(queries), width), bool))
    history, reasons = [], {}
    for i, (day, names, t) in enumerate(queries):
        mapping = {s: helper.benchmark_for(day, s, panel['sectors'], meta['etfs']) for s in names}
        supplied = benchmark_map(str(day.date()), names, panel['sectors'].loc[:day].copy(deep=True), meta['etfs'])
        require(isinstance(supplied, dict) and supplied == mapping, 'Original own-sector mapping differs')
        days = sessions[t - 294:t - 41] if t >= 294 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else dict(formation=dict(price_start=dates[0], price_end=dates[-1], return_count=252),
            blocks=dict(first_start_date=dates[0], last_end_date=dates[-1], block_sessions=21, block_count=12))
        missing = [str(d.date()) for d in days if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            benchmark = mapping[symbol]
            reason = ('etf_neutral' if symbol in meta['etfs'] else 'missing_benchmark' if benchmark is None else
                'insufficient_history' if not days else 'missing_symbol' if symbol not in panel['adj'].columns else
                'missing_benchmark_symbol' if benchmark not in panel['adj'].columns else
                'missing_official_session' if missing else None)
            cell = empty(reason) if reason else measure_oracle(
                [panel['adj'].at[d, symbol] for d in days], [panel['adj'].at[d, benchmark] for d in days])
            cell.update(symbol=symbol, benchmark=benchmark, date=str(day.date()), windows=windows,
                endpoint_dates=[dates[k] for k in cell['endpoint_indices']],
                block_start_dates=[dates[k] for k in cell['block_start_indices']],
                block_end_dates=[dates[k] for k in cell['block_end_indices']])
            values['available'][i, j] = cell['available']
            for key in ('candidate', 'control'):
                values[key][i, j] = cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=dates[0] if days else None, price_end=dates[-1] if days else None,
            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    available = int(values['available'].sum())
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=253, skip_sessions=42, return_count=252, block_count=12, block_sessions=21,
        neutral_pseudo_blocks=4, denominator=16, available_cells=available,
        unavailable_cells=int(values['available'].size) - available, reason_counts=reasons,
        relative_sign_observations=12 * available, absolute_sign_observations=12 * available,
        predictive_model_fits=0, common_availability=True, portfolio_returns_calculated=False,
        ranking_performed=False, trading_eligibility_modified=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean availability matrix required')
    for key in ('candidate', 'control'):
        v = a[key]
        require(v.shape == mask.shape and np.issubdtype(v.dtype, np.floating), 'Raw shape/dtype differs')
        require(not np.isinf(v).any() and np.array_equal(np.isfinite(v), mask), 'Common raw availability differs')
        require(np.all(np.abs(v[mask]) <= .75), 'Recurrence raw outside[-.75,.75]')
        require(np.all(v[mask] * 16 == np.rint(v[mask] * 16)), 'Raw must be signed integer count/16')
    return a


def verify_features(panel, meta, calendar, benchmark_map, values, audit):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, benchmark_map)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape and
                np.array_equal(actual[key], wanted[key], equal_nan=True), 'Independent recurrence differs: ' + key)
    record_check(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'available_cells', 'unavailable_cells',
                'relative_sign_observations', 'absolute_sign_observations')
    return dict(passed=True, **{key: expected[key] for key in counters}, independent_scalar_rebuild=True,
        own_sector_mapping_checked=True, exact_rational_relative_signs_checked=True, exact_absolute_signs_checked=True,
        twelve_nonoverlapping_blocks_checked=True, four_neutral_pseudoblocks_checked=True,
        full_253_price_support_checked=True, official_session_windows_checked=True, monthend_queries_checked=True,
        exact_common_availability=True, audit_checked=True, no_future_or_outcome_inputs=True,
        mapping_source_sha256=MAPPING_SHA, mapping_and_outcome_adapter_sha256=FROZEN_HELPER_SHA,
        predictive_model_fits=0, computes_returns=False, portfolio_returns_calculated=False)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (float, int) and dose == DOSE, 'Fixed dose .025 required')
    a = arrays(values)
    result = {arm: np.zeros(a['available'].shape) for arm in RAW}
    for arm, key in RAW.items():
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                result[arm][i, j] = dose * float(a[key][i, j])
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    wanted = independent_deltas(values, dose)
    a = arrays(values)
    for arm, expected in wanted.items():
        v = np.asarray(deltas[arm])
        require(v.shape == expected.shape and np.issubdtype(v.dtype, np.floating)
                and np.isfinite(v).all() and np.array_equal(v, expected), 'Exact fixed linear delta differs: ' + arm)
        require(np.all(np.abs(v) <= DOSE * .75), 'Unbounded delta')
    return dict(passed=True, checked_events=a['available'].shape[0], dose=DOSE, max_absolute_delta=MAX_DELTA,
        full_candidate_universe=True, common_available_only=True, common_availability=True, initial_delta_zero=True,
        no_eligibility_or_model_filter=True, no_cross_sectional_ranking=True, fixed_linear_mapping_checked=True,
        direct_additive_pre_score=True, changed_cells={arm: int(np.count_nonzero(v)) for arm, v in wanted.items()})
