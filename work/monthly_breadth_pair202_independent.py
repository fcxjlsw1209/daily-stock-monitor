"""Independent scalar sector-breadth/excess-return covariance audit.

No production feature imports or actual-data entry point. Frozen independent
mapping and portfolio-accounting helpers are loaded only after SHA checks.
All feature arithmetic and common-domain ranks are rebuilt here.
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
FROZEN_HELPER = ROOT / 'work/monthly_sector_recurrence192_independent.py'
FROZEN_HELPER_SHA = 'f78cc005827fd977ec431bdec4af0f4b7160694cb3d382df0115e8e00e0bedf8'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
ARMS = ('incumbent_local67', 'breadth_paired_relative', 'relative_mean_control')
RAW = {'breadth_paired_relative': 'candidate', 'relative_mean_control': 'control'}
SECTORS = ('XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')
DOSE = .01
COUNTERS = ('valid_price_observations', 'excess_block_observations', 'breadth_block_observations',
            'down_sign_comparisons', 'covariance_estimates', 'raw_score_observations',
            'ranked_score_observations')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'own_sector_mapping_checked',
    'excluded_own_benchmark_checked', 'twelve_nonoverlapping_blocks_checked',
    'log_excess_returns_checked', 'strict_down_breadth_checked', 'centered_covariance_checked',
    'matched_relative_mean_checked', 'exact_constant_branches_checked',
    'full253_price_support_checked', 'official_session_windows_checked', 'monthend_queries_checked',
    'exact_common_availability', 'audit_checked', 'no_future_or_outcome_inputs',
    'centered_high_tie_ranks_checked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent mapping/account helper changed')
    spec = importlib.util.spec_from_file_location('breadth202_frozen_independent', FROZEN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.MAPPING_SHA == MAPPING_SHA, 'Mapping source SHA differs')
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
    receipt['mapping_and_outcome_adapter_sha256'] = FROZEN_HELPER_SHA
    return receipt


def benchmark_for(day, symbol, sectors, etfs):
    helper = frozen_helper().frozen_helper()
    helper.mapping_constants()
    return helper.benchmark_for(day, symbol, sectors, etfs)


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
        require(type(actual) is type(expected) and actual == expected, label + ': exact value/type')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(parsed)


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
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF identities')
    sessions = [iso(value) for value in calendar]
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


def empty(reason, breadth_count):
    return dict(available=False, reason=reason, candidate=None, control=None,
        endpoint_indices=[], block_start_indices=[], block_end_indices=[],
        stock_log_gains=[], benchmark_log_gains=[], excess_log_returns=[], down_counts=[], breadth=[],
        mean_excess=None, mean_breadth=None, covariance=None, constant_branch=None,
        breadth_count=breadth_count, **{key: 0 for key in COUNTERS})


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite arithmetic from valid source')
    return value


def paired_summary(excess, breadth):
    require(len(excess) == len(breadth) == 12, 'Twelve paired blocks required')
    for value in list(excess) + list(breadth):
        finite(value)
    try:
        mean_excess = finite(math.fsum(excess) / 12)
        mean_breadth = finite(math.fsum(breadth) / 12)
    except (ValueError, OverflowError) as exc:
        raise FloatingPointError('Nonfinite summation from valid source') from exc
    branch = ('all_breadth_equal' if all(w == breadth[0] for w in breadth) else
              'all_excess_equal' if all(e == excess[0] for e in excess) else None)
    if branch is not None:
        covariance, candidate = 0., mean_excess
    else:
        terms = []
        for e, w in zip(excess, breadth):
            terms.append(finite(finite(w - mean_breadth) * finite(e - mean_excess)))
        try:
            covariance = finite(math.fsum(terms) / 12)
        except (ValueError, OverflowError) as exc:
            raise FloatingPointError('Nonfinite covariance summation') from exc
        candidate = finite(mean_excess + covariance)
    return dict(candidate=candidate, control=mean_excess, mean_excess=mean_excess,
                mean_breadth=mean_breadth, covariance=covariance, constant_branch=branch)


def measure_oracle(stock, benchmark, breadth):
    require(np.shape(stock) == np.shape(benchmark) == (253,), 'Exactly253 stock/benchmark prices required')
    shape = np.shape(breadth)
    require(len(shape) == 2 and shape[0] == 253 and shape[1] in (8, 9), 'Breadth253 by8or9 required')
    m = shape[1]
    vectors = []
    for raw, name in ((list(stock), 'stock'), (list(benchmark), 'benchmark'),
                      ([x for row in breadth for x in row], 'breadth')):
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) for x in raw):
            return empty('nonnumeric_' + name + '_price', m)
        try:
            vector = [float(x) for x in raw]
        except (TypeError, ValueError, OverflowError):
            return empty('nonnumeric_' + name + '_price', m)
        if any(not math.isfinite(x) or x <= 0 for x in vector):
            return empty('invalid_' + name + '_price', m)
        vectors.append(vector)
    own, sector, peers = vectors
    points = list(range(0, 253, 21))
    stock_gains, sector_gains, excess, down_counts, weights = [], [], [], [], []
    for a, b in zip(points[:-1], points[1:]):
        stock_gain = finite(math.log(own[b]) - math.log(own[a]))
        sector_gain = finite(math.log(sector[b]) - math.log(sector[a]))
        stock_gains.append(stock_gain); sector_gains.append(sector_gain)
        excess.append(finite(stock_gain - sector_gain))
        down = sum(peers[b * m + j] < peers[a * m + j] for j in range(m))
        down_counts.append(down); weights.append(down / m)
    result = empty('available', m)
    result.update(available=True, endpoint_indices=points, block_start_indices=points[:-1],
        block_end_indices=points[1:], stock_log_gains=stock_gains, benchmark_log_gains=sector_gains,
        excess_log_returns=excess, down_counts=down_counts, breadth=weights, **paired_summary(excess, weights),
        valid_price_observations=253 * (2 + m), excess_block_observations=12,
        breadth_block_observations=12, down_sign_comparisons=12 * m, covariance_estimates=1,
        raw_score_observations=2, ranked_score_observations=2)
    return result


def centered_ranks(raw, available):
    result = np.zeros(np.shape(raw), float)
    for i in range(len(result)):
        members = [j for j, flag in enumerate(available[i]) if flag]
        n = len(members)
        if n < 2:
            continue
        for j in members:
            lower = sum(raw[i, k] < raw[i, j] for k in members)
            equal = sum(raw[i, k] == raw[i, j] for k in members)
            result[i, j] = (lower + (equal - 1) / 2) / (n - 1) - .5
    return result


def independent_features(panel, meta, calendar, benchmark_map, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(callable(benchmark_map), 'Benchmark callback required')
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    if limit is not None:
        queries = queries[:limit]
    mapping_helper = frozen_helper().frozen_helper()
    mapping_helper.mapping_constants()
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons = [], {}
    totals = {key: 0 for key in COUNTERS}
    frame = panel['adj']
    for i, (day, names, t) in enumerate(queries):
        sector_prefix = panel['sectors'].loc[:day].copy(deep=True)
        mapping = {s: mapping_helper.benchmark_for(day, s, sector_prefix, meta['etfs']) for s in names}
        supplied = benchmark_map(str(day.date()), names, sector_prefix.copy(deep=True), meta['etfs'])
        require(isinstance(supplied, dict) and supplied == mapping, 'Original own-sector mapping differs')
        days = sessions[t - 294:t - 41] if t >= 294 else []
        dates = [str(d.date()) for d in days]
        missing = [str(d.date()) for d in days if d not in frame.index]
        cells = []
        for j, symbol in enumerate(names):
            benchmark = mapping[symbol]
            basket = [b for b in SECTORS if b != benchmark] if benchmark is not None and symbol not in meta['etfs'] else []
            if benchmark is not None and symbol not in meta['etfs']:
                require(len(set([symbol, benchmark] + basket)) == 2 + len(basket), 'Consumed series identities duplicate')
            reason = ('etf_neutral' if symbol in meta['etfs'] else 'missing_benchmark' if benchmark is None else
                'insufficient_history' if not days else 'missing_symbol' if symbol not in frame.columns else
                'missing_benchmark_symbol' if benchmark not in frame.columns else
                'missing_breadth_symbol' if any(b not in frame.columns for b in basket) else
                'missing_official_session' if missing else None)
            if reason is None:
                cell = measure_oracle([frame.at[d, symbol] for d in days], [frame.at[d, benchmark] for d in days],
                                      [[frame.at[d, b] for b in basket] for d in days])
            else:
                cell = empty(reason, len(basket))
            cell.update(symbol=symbol, benchmark=benchmark, date=str(day.date()), breadth_etfs=basket,
                endpoint_dates=[dates[k] for k in cell['endpoint_indices']],
                block_start_dates=[dates[k] for k in cell['block_start_indices']],
                block_end_dates=[dates[k] for k in cell['block_end_indices']], candidate_rank=0., control_rank=0.)
            for key in ('candidate', 'control', 'available'):
                values[key][i, j] = cell[key]
            for key in COUNTERS:
                totals[key] += cell[key]
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        valid = [c['available'] for c in cells]
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=dates[0] if dates else None, price_end=dates[-1] if dates else None,
            endpoint_dates=[dates[k] for k in range(0, 253, 21)] if dates else None,
            missing_official_sessions=missing, benchmarks=mapping, cells=cells, available=valid,
            available_count=sum(valid), eight_sector_cells=sum(c['available'] and c['breadth_count'] == 8 for c in cells),
            nine_sector_cells=sum(c['available'] and c['breadth_count'] == 9 for c in cells)))
    for raw in ('candidate', 'control'):
        key = raw + '_rank'
        values[key] = centered_ranks(values[raw], values['available'])
        for i, event in enumerate(history):
            for j, cell in enumerate(event['cells']):
                cell[key] = float(values[key][i, j])
    available = int(values['available'].sum())
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
        price_count=253, skip_sessions=42, block_count=12, block_sessions=21, breadth_etfs=list(SECTORS),
        available_cells=available, unavailable_cells=int(values['available'].size) - available,
        available_by_event=[row['available_count'] for row in history],
        eight_sector_cells_by_event=[row['eight_sector_cells'] for row in history],
        nine_sector_cells_by_event=[row['nine_sector_cells'] for row in history], **totals,
        reason_counts=reasons, event_history=history, new_model_fits=0, predictive_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) ==
            {'candidate', 'control', 'available', 'candidate_rank', 'control_rank'}, 'Feature keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean common availability required')
    for raw in ('candidate', 'control'):
        value, ranks = a[raw], a[raw + '_rank']
        require(value.shape == ranks.shape == mask.shape and np.issubdtype(value.dtype, np.floating)
                and np.issubdtype(ranks.dtype, np.floating), 'Float feature/rank matrices required')
        require(not np.isinf(value).any() and np.array_equal(np.isfinite(value), mask), 'Common finite raw domain differs')
        require(np.isfinite(ranks).all() and np.all(np.abs(ranks) <= .5), 'Finite bounded ranks required')
        require(np.array_equal(ranks, centered_ranks(value, mask)), 'Independent exact centered ranks differ')
    return a


def verify_features(panel, meta, calendar, benchmark_map, values, audit, limit=None):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, benchmark_map, limit)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape and np.array_equal(actual[key], wanted[key], equal_nan=True),
                'Independent breadth-paired feature differs: ' + key)
    record_check(audit, expected)
    counts = ('events', 'symbols_per_event', 'checked_cells', 'available_cells', 'unavailable_cells',
              'available_by_event', 'eight_sector_cells_by_event', 'nine_sector_cells_by_event') + COUNTERS
    return dict(**{key: True for key in FEATURE_FLAGS}, **{key: expected[key] for key in counts},
        mapping_source_sha256=MAPPING_SHA, mapping_and_outcome_adapter_sha256=FROZEN_HELPER_SHA,
        predictive_model_fits=0, new_model_fits=0, computes_returns=False)


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
