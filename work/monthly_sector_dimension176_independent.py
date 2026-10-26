"""Independent, source-only event oracle and pinned accounting for round176.

No producer feature helper or strategy engine is imported. The caller binds
source digests and the official calendar before supplying input objects. There
is no execution entry point, data loader, model fitting, or output mutation.
"""
from datetime import date
import ast
from itertools import combinations
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
ARMS = ('incumbent_local67', 'sector_dimension_timed', 'sector_dimension_mean')
RAW = dict(sector_dimension_timed='candidate', sector_dimension_mean='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .02
ETF9 = ('XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY')
MAPPING_SOURCE = ROOT / 'outputs/monthly_strict_relative/strict_study.py'
MAPPING_SHA = 'ddb11c0149e920de2952daba515ce28625a56df1f5c1039d4b9920c6d58d02c6'
SECTOR = {'Basic Materials': 'XLB', 'Communication Services': 'XLC', 'Energy': 'XLE',
          'Financial Services': 'XLF', 'Industrials': 'XLI', 'Technology': 'XLK',
          'Consumer Defensive': 'XLP', 'Real Estate': 'XLRE', 'Utilities': 'XLU',
          'Healthcare': 'XLV', 'Consumer Cyclical': 'XLY'}
FOREIGN = {'ASML': 'XLK', 'ARM': 'XLK', 'TSM': 'XLK', 'BABA': 'XLY', 'NVO': 'XLV',
           'SAP': 'XLK', 'TM': 'XLY', 'SHEL': 'XLE', 'SKHY': 'XLK', 'BHP': 'XLB',
           'HSBC': 'XLF', 'SMFG': 'XLF', 'NVS': 'XLV', 'SNDK': 'XLK'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING, MAPPING_SOURCE)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('dimension176_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_outcomes(px, opening, orders, exports, results, meta):
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('dimension176_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    result = module.audit_outcomes(px, opening, orders, exports, results, meta)
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': availability')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaNs')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    mask = np.isfinite(expected)
    require(np.all(np.abs(actual[mask] - expected[mask]) <= atol), label + ': numbers')


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
    """Verify frozen source constants without importing or executing its code."""
    require(sha(MAPPING_SOURCE) == MAPPING_SHA, 'Frozen benchmark mapping source changed')
    found = {}
    for node in ast.parse(MAPPING_SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('SECTOR', 'FOREIGN'):
                found[name] = ast.literal_eval(node.value)
    require(found == {'SECTOR': SECTOR, 'FOREIGN': FOREIGN}, 'Frozen mapping constants differ')
    return dict(source=str(MAPPING_SOURCE), sha256=MAPPING_SHA)


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
    require(sessions and all(d.weekday() < 5 for d in sessions)
            and all(a < b for a, b in zip(sessions, sessions[1:])), 'Calendar order/sessions')
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


def stable_log(first, last):
    relative = (last - first) / first
    if math.isfinite(relative) and relative > -1.:
        return math.log1p(relative)
    return math.log(last) - math.log(first)


def centered(vector):
    shift = sorted(vector)[len(vector) // 2]
    translated = [v - shift for v in vector]
    mean = math.fsum(translated) / len(translated)
    return [v - mean for v in translated]


def unit_interval(value, name):
    require(math.isfinite(value) and -1e-12 <= value <= 1. + 1e-12, name + ' outside roundoff tolerance')
    return min(1., max(0., value))


def correlation(left, right):
    dx, dy = centered(left), centered(right)
    normx, normy = math.sqrt(math.fsum(x * x for x in dx)), math.sqrt(math.fsum(y * y for y in dy))
    rho = math.fsum(x * y for x, y in zip(dx, dy)) / (normx * normy)
    require(math.isfinite(rho) and abs(rho) <= 1. + 1e-12, 'Correlation outside roundoff tolerance')
    return min(1., max(-1., rho))


def empty_market(reason):
    return dict(available=False, reason=reason, windows=[], dispersion_values=[],
                current_dispersion=None, historical_dispersion_mean=None,
                invalid_window_indices=[], correlation_window_estimates=0, pair_estimates=0)


def dimension_oracle(sector_prices):
    """61 trailing covariance windows, each 252 returns, nine fixed ETFs."""
    require(np.shape(sector_prices) == (1513, 9), 'Exactly 1513x9 sector prices required')
    try:
        columns = [[float(row[j]) for row in sector_prices] for j in range(9)]
    except (ValueError, TypeError, OverflowError):
        return empty_market('nonnumeric_market_price')
    if any(not math.isfinite(x) or x <= 0 for col in columns for x in col):
        return empty_market('invalid_market_price')
    returns = [[stable_log(a, b) for a, b in zip(col[:-1], col[1:])] for col in columns]
    result = empty_market('available')
    for k in range(61):
        end = 1512 - 21 * k
        values = [col[end - 252:end] for col in returns]
        deviations = [centered(col) for col in values]
        means = [math.fsum(col) / 252. for col in values]
        stds = [math.sqrt(math.fsum(x * x for x in col) / 251.) for col in deviations]
        if any(not math.isfinite(sd) or sd <= 1e-12 for sd in stds):
            correlations, dimension = [], None
            result['invalid_window_indices'].append(k)
        else:
            correlations = [correlation(values[i], values[j]) for i, j in combinations(range(9), 2)]
            dimension = unit_interval(1. - math.fsum(r * r for r in correlations) / 36., 'Dimension')
            result['correlation_window_estimates'] += 1
        result['windows'].append(dict(k=k, start_index=end - 252, end_index=end, means=means,
            standard_deviations=stds, pair_correlations=correlations, dispersion=dimension,
            available=dimension is not None, reason='available' if dimension is not None else 'market_volatility_floor'))
        result['dispersion_values'].append(dimension)
    result['pair_estimates'] = 36 * result['correlation_window_estimates']
    if result['invalid_window_indices']:
        result['reason'] = 'market_volatility_floor'
    else:
        result.update(available=True, current_dispersion=result['dispersion_values'][0],
            historical_dispersion_mean=math.fsum(result['dispersion_values'][1:]) / 60.)
    return result


def empty_quality(reason):
    return dict(available=False, reason=reason, relative_mean=None,
                relative_standard_deviation=None, relative_sharpe=None, quality=None)


def quality_oracle(stock, sector):
    require(np.shape(stock) == np.shape(sector) == (211,), 'Exactly 211 stock and sector prices required')
    vectors = []
    for original, label in ((stock, 'stock'), (sector, 'benchmark')):
        try:
            values = [float(x) for x in original]
        except (ValueError, TypeError, OverflowError):
            return empty_quality('nonnumeric_' + label + '_price')
        if any(not math.isfinite(x) or x <= 0 for x in values):
            return empty_quality('invalid_' + label + '_price')
        vectors.append(values)
    stock, sector = vectors
    relative = [stable_log(stock[i], stock[i + 1]) - stable_log(sector[i], sector[i + 1]) for i in range(210)]
    mean = math.fsum(relative) / 210.
    centered_relative = centered(relative)
    sd = math.sqrt(math.fsum(x * x for x in centered_relative) / 209.)
    result = empty_quality('relative_volatility_floor')
    result.update(relative_mean=mean, relative_standard_deviation=sd)
    if math.isfinite(sd) and sd > 1e-12:
        ratio = math.sqrt(252.) * mean / sd
        result.update(available=True, reason='available', relative_sharpe=ratio, quality=math.tanh(ratio))
    return result


def independent_features(panel, meta, calendar, limit=None):
    mapping_constants()
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan),
                  control=np.full((len(queries), width), np.nan), available=np.zeros((len(queries), width), bool))
    rows, counts = [], {}
    for i, (day, names, t) in enumerate(queries):
        enough = t >= 1554
        market_days = sessions[t - 1554:t - 41] if enough else []
        relative_days = sessions[t - 252:t - 41] if t >= 252 else []
        missing = [str(d.date()) for d in market_days if d not in panel['adj'].index]
        missing_symbols = [symbol for symbol in ETF9 if symbol not in panel['adj'].columns]
        if not enough:
            market = empty_market('insufficient_history')
        elif missing_symbols:
            market = empty_market('missing_market_symbol')
        elif missing:
            market = empty_market('missing_market_session')
        else:
            market = dimension_oracle([[panel['adj'].at[d, symbol] for symbol in ETF9] for d in market_days])
        for row in market['windows']:
            row['start_date'] = str(market_days[row['start_index']].date())
            row['end_date'] = str(market_days[row['end_index']].date())
        benchmarks = {symbol: benchmark_for(day, symbol, panel['sectors'], meta['etfs']) for symbol in names}
        cells = []
        for j, symbol in enumerate(names):
            benchmark = benchmarks[symbol]
            if symbol in meta['etfs']:
                q = empty_quality('etf_neutral')
            elif benchmark is None:
                q = empty_quality('missing_benchmark')
            elif not market['available']:
                q = empty_quality(market['reason'])
            elif symbol not in panel['adj'].columns:
                q = empty_quality('missing_symbol')
            elif benchmark not in panel['adj'].columns:
                q = empty_quality('missing_benchmark_symbol')
            elif any(d not in panel['adj'].index for d in relative_days):
                q = empty_quality('missing_relative_session')
            else:
                q = quality_oracle([panel['adj'].at[d, symbol] for d in relative_days],
                                   [panel['adj'].at[d, benchmark] for d in relative_days])
            cell = dict(q, symbol=symbol, benchmark=benchmark, candidate=None, control=None)
            if cell['available']:
                cell.update(candidate=market['current_dispersion'] * q['quality'],
                            control=market['historical_dispersion_mean'] * q['quality'])
                for key in ('candidate', 'control'):
                    values[key][i, j] = cell[key]
                values['available'][i, j] = True
            counts[cell['reason']] = counts.get(cell['reason'], 0) + 1
            cells.append(cell)
        rows.append(dict(date=str(day.date()), symbols=names, session_position=t,
            market_start=str(market_days[0].date()) if market_days else None,
            market_end=str(market_days[-1].date()) if market_days else None,
            relative_start=str(relative_days[0].date()) if relative_days else None,
            relative_end=str(relative_days[-1].date()) if relative_days else None,
            missing_market_sessions=missing, missing_market_symbols=missing_symbols,
            market=market, benchmarks=benchmarks, cells=cells))
    audit = dict(events=len(queries), symbols_per_event=width, sector_etfs=list(ETF9),
        pair_indices=[list(pair) for pair in combinations(range(9), 2)], skip_sessions=42,
        market_price_count=1513, market_return_sessions=252, market_window_count=61, market_window_spacing=21,
        historical_window_count=60, relative_price_count=211, relative_return_sessions=210,
        annualization_sessions=252, standard_deviation_floor=1e-12, roundoff_tolerance=1e-12,
        available_cells=int(values['available'].sum()), unavailable_cells=int((~values['available']).sum()),
        reason_counts=counts, event_history=rows,
        correlation_window_estimates=sum(r['market']['correlation_window_estimates'] for r in rows),
        pair_estimates=sum(r['market']['pair_estimates'] for r in rows),
        relative_quality_estimates=int(values['available'].sum()), predictive_model_fits=0,
        common_availability=True, portfolio_returns_calculated=False,
        ranking_performed=False, trading_eligibility_modified=False)
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


def verify_features(panel, meta, calendar, values, audit):
    actual = arrays(values)
    wanted, expected_audit = independent_features(panel, meta, calendar)
    require(np.array_equal(actual['available'], wanted['available']), 'Independent state support differs')
    for key in ('candidate', 'control'):
        close(actual[key], wanted[key], 'Independent ' + key)
    record_check(audit, expected_audit)
    return dict(passed=True, events=len(meta['dates']), symbols_per_event=wanted['available'].shape[1],
        checked_cells=int(wanted['available'].size), available_cells=int(wanted['available'].sum()),
        unavailable_cells=int((~wanted['available']).sum()), independent_scalar_oracle=True,
        independent_scalar_rebuild=True, exact_common_availability=True, sector_dimension_windows_checked=True,
        relative_stock_sector_checked=True, frozen_mapping_checked=True, etf_neutrality_checked=True,
        no_future_or_outcome_inputs=True, audit_checked=True, no_cross_sectional_ranking=True,
        correlation_window_estimates=expected_audit['correlation_window_estimates'],
        pair_estimates=expected_audit['pair_estimates'], relative_quality_estimates=expected_audit['relative_quality_estimates'],
        predictive_model_fits=0, computes_returns=False, portfolio_returns_calculated=False,
        mapping_source_sha256=MAPPING_SHA)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (int, float) and dose == DOSE, 'Fixed dose=.02 required')
    a = arrays(values)
    result = {}
    for arm, key in RAW.items():
        raw = a[key]
        require(np.all(np.abs(raw[a['available']]) <= 1.), 'Raw value outside [-1,1]')
        delta = np.zeros(a['available'].shape)
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                delta[i, j] = dose * float(raw[i, j])
        result[arm] = delta
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose)
    for arm, target in expected.items():
        close(deltas[arm], target, 'Direct delta ' + arm, atol=2e-17)
        actual = np.asarray(deltas[arm])
        require(np.isfinite(actual).all() and np.all(np.abs(actual) <= dose), 'Unbounded delta')
        require(not len(actual) or np.array_equal(actual[0], np.zeros(actual.shape[1])), 'Initial delta not exactly zero')
        require(np.array_equal(actual[~np.asarray(values['available'])], np.zeros((~np.asarray(values['available'])).sum())),
                'Missing source delta not exactly zero')
    return dict(passed=True, checked_events=arrays(values)['available'].shape[0], dose=DOSE,
        full_candidate_universe=True, common_available_only=True, initial_delta_zero=True,
        no_cross_sectional_ranking=True, no_eligibility_or_model_filter=True, direct_linear_mapping=True,
        changed_cells={arm: int(np.count_nonzero(v)) for arm, v in expected.items()})
