"""Independent source, rank and account verification for round183.

No production feature/strategy imports or real-data entry point. Mapping is
reconstructed from independently stated constants, verified against frozen AST.
The caller-provided original mapping is cross-checked on a query-date prefix.
"""
from datetime import date
import ast
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
MAPPING_SOURCE = ROOT / 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTOR = {'Basic Materials': 'XLB', 'Communication Services': 'XLC', 'Energy': 'XLE',
          'Financial Services': 'XLF', 'Industrials': 'XLI', 'Technology': 'XLK',
          'Consumer Defensive': 'XLP', 'Real Estate': 'XLRE', 'Utilities': 'XLU',
          'Healthcare': 'XLV', 'Consumer Cyclical': 'XLY'}
FOREIGN = {'ASML': 'XLK', 'ARM': 'XLK', 'TSM': 'XLK', 'BABA': 'XLY', 'NVO': 'XLV',
           'SAP': 'XLK', 'TM': 'XLY', 'SHEL': 'XLE', 'SKHY': 'XLK', 'BHP': 'XLB',
           'HSBC': 'XLF', 'SMFG': 'XLF', 'NVS': 'XLV', 'SNDK': 'XLK'}
ARMS = ('incumbent_local67', 'sector_two_state_floor', 'sector_relative_mean_control')
RAW = dict(sector_two_state_floor='candidate', sector_relative_mean_control='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .05


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING, MAPPING_SOURCE)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('sector183_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('sector183_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': mapping')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (got, wanted) in enumerate(zip(actual, expected)):
            record_check(got, wanted, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (float, int), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def iso(value):
    require(isinstance(value, str), 'ISO date string required')
    day = date.fromisoformat(value)
    require(day.isoformat() == value, 'Strict ISO date required')
    return pd.Timestamp(day)


def mapping_constants():
    require(sha(MAPPING_SOURCE) == MAPPING_SHA, 'Frozen benchmark mapping source changed')
    found = {}
    for node in ast.parse(MAPPING_SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('SECTOR', 'FOREIGN'):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    require(found == {'SECTOR': SECTOR, 'FOREIGN': FOREIGN}, 'Frozen mapping constants differ')


def benchmark_for(day, symbol, sectors, etfs):
    if symbol in etfs:
        return None
    if symbol in FOREIGN:
        return FOREIGN[symbol]
    if day not in sectors.index or symbol not in sectors.columns:
        return None
    return SECTOR.get(sectors.at[day, symbol])


def schema(panel, meta, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'sectors'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        require(not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing,
                'Invalid panel time axis')
        require(frame.index.tz is None and frame.index.equals(frame.index.normalize()), 'Daily naive dates required')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s for s in frame.columns),
                'Panel identities')
    require(isinstance(meta, dict) and all(k in meta for k in ('dates', 'symbols', 'etfs')), 'Metadata fields')
    require(len(meta['dates']) == len(meta['symbols']), 'Event metadata lengths')
    require(isinstance(meta['etfs'], (list, tuple, set, frozenset)) and
            len(meta['etfs']) == len(set(meta['etfs'])) and
            all(isinstance(s, str) and s and s.strip() == s for s in meta['etfs']), 'ETF metadata')
    sessions = [iso(value) for value in calendar]
    require(sessions and all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], 0
    for raw, names in zip(meta['dates'], meta['symbols']):
        day = iso(raw)
        require(day in lookup and (not queries or day > queries[-1][0]), 'Event session/order')
        require(isinstance(names, (list, tuple)) and names and len(names) == len(set(names)), 'Event identities')
        require(all(isinstance(s, str) and s and s.strip() == s for s in names), 'Invalid event name')
        require(not queries or width == len(names), 'Changing event width')
        width = len(names); queries.append((day, list(names), lookup[day]))
    return sessions, queries, width


def empty(reason):
    return dict(available=False, reason=reason, candidate=None, control=None,
                up_count=0, down_count=0, flat_count=0, mean_up=None, mean_down=None, pooled_mean=None,
                conditional_mean_estimates=0, pooled_mean_estimates=0,
                excess_log_returns=[], sector_states=[])


def finite_sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid excess return sum') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite excess return sum')
    return result


def measure_oracle(stock, benchmark):
    require(np.shape(stock) == np.shape(benchmark) == (211,), 'Exactly 211 prices for both series required')
    vectors = []
    for original, name in ((stock, 'stock'), (benchmark, 'benchmark')):
        try:
            numbers = [float(value) for value in original]
        except (ValueError, TypeError, OverflowError):
            return empty('nonnumeric_' + name + '_price')
        if any(not math.isfinite(x) or x <= 0. for x in numbers):
            return empty('invalid_' + name + '_price')
        vectors.append(numbers)
    stock, benchmark = vectors
    try:
        a, b = [math.log(x) for x in stock], [math.log(x) for x in benchmark]
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError('Invalid pointwise log') from exc
    if not all(math.isfinite(x) for x in a + b):
        raise FloatingPointError('Nonfinite pointwise log')
    result = empty('insufficient_groups')
    groups = {1: [], -1: []}
    for end in range(1, 211):
        state = 1 if benchmark[end] > benchmark[end-1] else -1 if benchmark[end] < benchmark[end-1] else 0
        excess = (a[end] - a[end-1]) - (b[end] - b[end-1])
        if not math.isfinite(excess):
            raise FloatingPointError('Nonfinite excess return')
        result['sector_states'].append(state)
        result['excess_log_returns'].append(excess)
        if state:
            groups[state].append(excess)
    up, down = groups[1], groups[-1]
    result.update(up_count=len(up), down_count=len(down), flat_count=210-len(up)-len(down))
    if min(len(up), len(down)) < 40:
        return result
    mean_up, mean_down = finite_sum(up) / len(up), finite_sum(down) / len(down)
    # Pool the same nonflat observations, retaining their actual counts.
    pooled = finite_sum([x for x, s in zip(result['excess_log_returns'], result['sector_states']) if s]) / (len(up)+len(down))
    if not all(math.isfinite(x) for x in (mean_up, mean_down, pooled)):
        raise FloatingPointError('Nonfinite conditional mean')
    result.update(available=True, reason='available', mean_up=mean_up, mean_down=mean_down,
                  pooled_mean=pooled, candidate=min(mean_up, mean_down), control=pooled,
                  conditional_mean_estimates=2, pooled_mean_estimates=1)
    return result


def independent_features(panel, meta, calendar, benchmark_map, limit=None):
    mapping_constants()
    sessions, queries, width = schema(panel, meta, calendar)
    require(callable(benchmark_map), 'Original benchmark_map callable required')
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan), control=np.full((len(queries), width), np.nan),
                  available=np.zeros((len(queries), width), bool))
    history, counts, conditional, pooled = [], {}, 0, 0
    for i, (day, names, t) in enumerate(queries):
        mapping = {s: benchmark_for(day, s, panel['sectors'], meta['etfs']) for s in names}
        got = benchmark_map(str(day.date()), names, panel['sectors'].loc[:day].copy(deep=True), meta['etfs'])
        require(isinstance(got, dict) and got == mapping, 'Original own-sector mapping disagrees with independent source')
        days = sessions[t-252:t-41] if t >= 252 else []
        dates = [str(d.date()) for d in days]
        windows = None if not days else dict(formation=dict(price_start=dates[0], price_end=dates[-1], return_count=210),
                                             first_return_date=dates[1], last_return_date=dates[-1])
        missing = [str(d.date()) for d in days if d not in panel['adj'].index]
        cells = []
        for j, symbol in enumerate(names):
            benchmark = mapping[symbol]
            if symbol in meta['etfs']:
                cell = empty('etf_neutral')
            elif benchmark is None:
                cell = empty('missing_benchmark')
            elif not days:
                cell = empty('insufficient_history')
            elif symbol not in panel['adj'].columns:
                cell = empty('missing_symbol')
            elif benchmark not in panel['adj'].columns:
                cell = empty('missing_benchmark_symbol')
            elif missing:
                cell = empty('missing_official_session')
            else:
                cell = measure_oracle([panel['adj'].at[d, symbol] for d in days],
                                      [panel['adj'].at[d, benchmark] for d in days])
            cell.update(symbol=symbol, benchmark=benchmark, date=str(day.date()), windows=windows)
            for key in values:
                values[key][i, j] = cell[key]
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            conditional += cell['conditional_mean_estimates']; pooled += cell['pooled_mean_estimates']
            cells.append(cell)
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
                            price_start=dates[0] if dates else None, price_end=dates[-1] if dates else None,
                            windows=windows, missing_official_sessions=missing, benchmarks=mapping, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=int(values['available'].size),
                 price_count=211, skip_sessions=42, return_count=210, minimum_group_count=40,
                 available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()),
                 reason_counts=counts, conditional_mean_estimates=conditional, pooled_mean_estimates=pooled,
                 predictive_model_fits=0, common_availability=True, portfolio_returns_calculated=False,
                 ranking_performed=False, trading_eligibility_modified=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    mask = a['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Boolean matrix availability required')
    for key in ('candidate', 'control'):
        require(a[key].shape == mask.shape and np.issubdtype(a[key].dtype, np.floating), 'Raw shape/dtype differs')
        require(not np.isinf(a[key]).any() and np.array_equal(np.isfinite(a[key]), mask), 'Common raw availability differs')
    return a


def verify_features(panel, meta, calendar, benchmark_map, values, audit):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, benchmark_map)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent common support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    actual_ranks, expected_ranks = independent_ranks(actual), independent_ranks(wanted)
    for arm in RAW:
        require(np.array_equal(actual_ranks[arm], expected_ranks[arm]), 'Independent raw rank differs: ' + arm)
    record_check(audit, expected)
    return dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
                checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
                unavailable_cells=int((~wanted['available']).sum()), independent_scalar_rebuild=True,
                own_sector_mapping_checked=True, exact_sector_price_sign_groups_checked=True,
                two_conditional_means_checked=True, count_weighted_control_checked=True,
                minimum_40_per_group_checked=True, official_session_windows_checked=True,
                full_211_price_support_checked=True, exact_common_availability=True, audit_checked=True,
                reconstructed_raw_ranks_exact=True,
                no_future_or_outcome_inputs=True, conditional_mean_estimates=expected['conditional_mean_estimates'],
                pooled_mean_estimates=expected['pooled_mean_estimates'], predictive_model_fits=0,
                mapping_source_sha256=MAPPING_SHA,
                computes_returns=False, portfolio_returns_calculated=False)


def independent_ranks(values):
    a = arrays(values)
    output = {arm: np.zeros(a['available'].shape) for arm in RAW}
    # Literal counting implements the average-rank convention, independently of pandas.
    for i, mask in enumerate(a['available']):
        ids = [j for j, valid in enumerate(mask) if valid]
        if len(ids) < 2:
            continue
        for arm, key in RAW.items():
            for j in ids:
                x = float(a[key][i, j])
                lower = sum(float(a[key][i, k]) < x for k in ids)
                equal = sum(float(a[key][i, k]) == x for k in ids)
                output[arm][i, j] = (lower + (equal + 1.) / 2.) / len(ids) - .5
    return output


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (float, int) and dose == DOSE, 'Fixed dose=.05 required')
    result = {arm: dose * rank for arm, rank in independent_ranks(values).items()}
    for delta in result.values():
        if len(delta):
            delta[0] = 0.
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose)
    a = arrays(values)
    for arm, target in expected.items():
        close(deltas[arm], target, 'Average-rank delta ' + arm, atol=2e-17)
        actual = np.asarray(deltas[arm])
        require(np.isfinite(actual).all() and np.all(np.abs(actual) <= .025), 'Unbounded delta')
        neutral = ~a['available'] | (np.sum(a['available'], axis=1) < 2)[:, None]
        if len(neutral):
            neutral[0] = True
        require(np.array_equal(actual[neutral], np.zeros(int(neutral.sum()))), 'Neutral cell delta not exactly zero')
    return dict(passed=True, checked_events=a['available'].shape[0], dose=DOSE,
                full_candidate_universe=True, common_available_only=True, common_availability=True,
                initial_delta_zero=True, no_eligibility_or_model_filter=True, direct_additive_pre_score=True,
                average_percentile_ranking_checked=True, no_additional_nonlinearity=True,
                direct_linear_mapping=True, max_absolute_delta=.025,
                changed_cells={arm: int(np.count_nonzero(value)) for arm, value in expected.items()})
