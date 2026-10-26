"""Independent scalar volume-stability sign-reversal oracle for round208.

The only reusable implementation is frozen independent portfolio accounting.
Stored volume is constant-unit invariant, not certified split-adjusted turnover.
Canonical arithmetic and ranks are exact; a separately evaluated pairwise
squared-distance identity checks each CV2 within absolute1e-12. No tolerance
changes raw scores, ties, support or portfolio deltas.
Positive dispersion values and ranks remain unchanged from round207. Only the
fixed additive dose is negative. This is an outcome-informed counterhypothesis,
not new volume information, certified share turnover or a pressure-direction proof.
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
ARMS = ('incumbent_local67', 'down_volume_stability', 'all_volume_stability_control')
RAW = {'down_volume_stability': 'candidate', 'all_volume_stability_control': 'control'}
DOSE = -.025
COUNTERS = ('valid_price_observations', 'valid_volume_observations', 'down_comparisons',
            'dispersion_estimates', 'raw_score_observations', 'ranked_score_observations')
FEATURE_FLAGS = ('passed', 'independent_scalar_rebuild', 'strict_down_price_comparisons_checked',
    'current63_alignment_checked', 'population_cv_squared_checked', 'matched_all_volume_dispersion_checked',
    'common_minimum_down_support_checked', 'exact_constant_branches_checked', 'full64_price_support_checked',
    'full63_volume_support_checked', 'official_session_windows_checked', 'monthend_queries_checked',
    'exact_common_availability', 'audit_checked', 'no_future_or_outcome_inputs', 'centered_high_tie_ranks_checked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def frozen_helper():
    require(hashlib.sha256(FROZEN_HELPER.read_bytes()).hexdigest() == FROZEN_HELPER_SHA,
            'Frozen192 independent accounting helper changed')
    spec = importlib.util.spec_from_file_location('volumestability208_frozen_independent_accounts', FROZEN_HELPER)
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
    require(isinstance(panel, dict) and set(panel) == {'adj', 'volume'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Frame type')
        idx = frame.index
        require(not idx.hasnans and idx.is_unique and idx.is_monotonic_increasing and idx.tz is None and
                idx.equals(idx.normalize()), 'Invalid daily axis')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
                for s in frame.columns), 'Invalid symbol labels')
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


def empty(reason):
    return dict(source_available=False, available=False, reason=reason, candidate=None, control=None,
        prices=[], volumes=[], down_flags=[], down_indices=[], down_count=0, down_volumes=[],
        candidate_scale=None, control_scale=None, candidate_scaled=[], control_scaled=[],
        candidate_mean=None, control_mean=None, candidate_squared_deviation_sum=None,
        control_squared_deviation_sum=None, candidate_constant=False, control_constant=False,
        dispersion_estimates=0)


def positive_values(values, name):
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in values):
        return None, 'nonnumeric_' + name
    try:
        result = [float(v) for v in values]
    except (ValueError, TypeError, OverflowError):
        return None, 'nonnumeric_' + name
    if any(not math.isfinite(v) or v <= 0 for v in result):
        return None, 'invalid_' + name
    return result, None


def cv_squared(values):
    """Independent ordered-scalar canonical statistic, plus pair-distance check."""
    require(len(values) >= 2 and all(math.isfinite(v) and v > 0 for v in values),
            'At least two positive finite volume values required')
    scale = max(values)
    scaled = [v / scale for v in values]
    count = len(values)
    mean = math.fsum(scaled) / count
    constant = all(v == values[0] for v in values)
    centered = 0. if constant else math.fsum((v - mean) ** 2 for v in scaled)
    result = 0. if constant else centered / (count * mean * mean)
    if not all(math.isfinite(v) for v in (scale, mean, centered, result)):
        raise FloatingPointError('Nonfinite volume dispersion arithmetic')
    # Independent identity: sum_{i<j}(z_i-z_j)^2 / (sum z)^2 = population CV2.
    distances = math.fsum((scaled[i] - scaled[j]) ** 2
                         for i in range(count) for j in range(i))
    total = math.fsum(scaled)
    alternate = distances / (total * total)
    require(math.isfinite(alternate) and abs(alternate - result) <= 1e-12,
            'Pairwise dispersion identity differs')
    return result, dict(scale=scale, scaled=scaled, mean=mean,
                        squared_deviation_sum=centered, constant=constant)


def measure_oracle(prices, volume):
    require(np.shape(prices) == (64,) and np.shape(volume) == (63,),
            'Exactly64 prices and63 stored volume observations required')
    prices, reason = positive_values(list(prices), 'adjusted_price')
    if reason:
        return empty(reason)
    volume, reason = positive_values(list(volume), 'volume')
    if reason:
        return empty(reason)
    down = [prices[k + 1] < prices[k] for k in range(63)]
    indices = [k for k in range(63) if down[k]]
    selected = [volume[k] for k in indices]
    result = empty('insufficient_down_days')
    result.update(source_available=True, prices=prices, volumes=volume,
                  down_flags=down, down_indices=indices, down_count=len(indices), down_volumes=selected)
    if len(indices) < 2:
        return result
    candidate, detail = cv_squared(selected)
    control, all_detail = cv_squared(volume)
    result.update(available=True, reason='available', candidate=candidate, control=control,
                  dispersion_estimates=2)
    for prefix, stats in (('candidate', detail), ('control', all_detail)):
        result.update({prefix + '_' + k: v for k, v in stats.items()})
    return result


def centered_ranks(raw, available):
    out = np.zeros(np.shape(raw), float)
    for i in range(len(out)):
        columns = [j for j, good in enumerate(available[i]) if good]
        n = len(columns)
        if n <= 1:
            continue
        for j in columns:
            lesser = sum(raw[i, k] < raw[i, j] for k in columns)
            equal = sum(raw[i, k] == raw[i, j] for k in columns)
            rank0 = lesser + (equal - 1) / 2
            out[i, j] = rank0 / (n - 1) - .5
    return out


def independent_features(panel, meta, calendar, limit=None):
    sessions, queries, width = schema(panel, meta, calendar)
    require(limit is None or type(limit) is int and 0 <= limit <= len(queries), 'Invalid prefix limit')
    if limit is not None:
        queries = queries[:limit]
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
        available=np.zeros(shape, bool), candidate_rank=np.zeros(shape), control_rank=np.zeros(shape))
    history, reasons = [], {}
    prices, volumes = panel['adj'], panel['volume']
    source_counts, available_counts = [], []
    for i, (day, names, t) in enumerate(queries):
        pdays = sessions[t - 63:t + 1] if t >= 63 else []
        vdays = pdays[1:]
        pmissing = [str(d.date()) for d in pdays if d not in prices.index]
        vmissing = [str(d.date()) for d in vdays if d not in volumes.index]
        cells = []
        for j, symbol in enumerate(names):
            reason = ('insufficient_history' if not pdays else
                'missing_price_symbol' if symbol not in prices.columns else
                'missing_volume_symbol' if symbol not in volumes.columns else
                'missing_official_price_session' if pmissing else
                'missing_official_volume_session' if vmissing else None)
            cell = empty(reason) if reason else measure_oracle(
                [prices.at[d, symbol] for d in pdays], [volumes.at[d, symbol] for d in vdays])
            cell.update(symbol=symbol, date=str(day.date()), candidate_rank=0., control_rank=0.)
            values['available'][i, j] = cell['available']
            values['candidate'][i, j] = cell['candidate']
            values['control'][i, j] = cell['control']
            reasons[cell['reason']] = reasons.get(cell['reason'], 0) + 1
            cells.append(cell)
        source = [c['source_available'] for c in cells]
        available = [c['available'] for c in cells]
        source_counts.append(sum(source)); available_counts.append(sum(available))
        history.append(dict(date=str(day.date()), symbols=names, session_position=t,
            price_start=str(pdays[0].date()) if pdays else None,
            price_end=str(pdays[-1].date()) if pdays else None,
            volume_start=str(vdays[0].date()) if vdays else None,
            volume_end=str(vdays[-1].date()) if vdays else None,
            price_dates=[str(d.date()) for d in pdays], volume_dates=[str(d.date()) for d in vdays],
            missing_official_price_sessions=pmissing, missing_official_volume_sessions=vmissing,
            source_available=source, available=available,
            source_available_count=sum(source), available_count=sum(available), cells=cells))
    for raw in ('candidate', 'control'):
        key = raw + '_rank'
        values[key] = centered_ranks(values[raw], values['available'])
        for i, event in enumerate(history):
            for j, cell in enumerate(event['cells']):
                cell[key] = float(values[key][i, j])
    count = int(values['available'].size); source = sum(source_counts); available = sum(available_counts)
    audit = dict(events=len(queries), symbols_per_event=width, checked_cells=count,
        price_count=64, volume_count=63, minimum_down_days=2,
        source_available_cells=source, source_available_by_event=source_counts,
        available_cells=available, available_by_event=available_counts, unavailable_cells=count-available,
        valid_price_observations=64*source, valid_volume_observations=63*source, down_comparisons=63*source,
        dispersion_estimates=2*available, raw_score_observations=2*available, ranked_score_observations=2*available,
        reason_counts=reasons, predictive_model_fits=0, new_model_fits=0,
        common_availability=True, ranking_performed=True, ranking_eligibility_filtered=False,
        trading_eligibility_modified=False, portfolio_returns_calculated=False, event_history=history)
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
        require(np.all(v[mask] >= 0.), 'Negative volume dispersion')
        require(np.array_equal(r, centered_ranks(v, mask)), 'Independent centered ranks differ')
    return a


def verify_features(panel, meta, calendar, values, audit, limit=None):
    actual = arrays(values)
    wanted, expected = independent_features(panel, meta, calendar, limit)
    for key in wanted:
        require(actual[key].shape == wanted[key].shape and
                np.array_equal(actual[key], wanted[key], equal_nan=True), 'Independent feature differs: ' + key)
    record_check(audit, expected)
    counters = ('events', 'symbols_per_event', 'checked_cells', 'source_available_cells',
                'source_available_by_event', 'available_cells', 'available_by_event', 'unavailable_cells') + COUNTERS
    return dict(**{key: True for key in FEATURE_FLAGS}, **{key: expected[key] for key in counters},
        predictive_model_fits=0, new_model_fits=0, computes_returns=False,
        canonical_feature_comparison='exact', rank_comparison='exact', pairwise_identity_atol=1e-12,
        outcome_adapter_sha256=FROZEN_HELPER_SHA)


def independent_deltas(values, dose=DOSE):
    require(type(dose) in (float, int) and dose == DOSE, 'Fixed negative dose -.025 required')
    a = arrays(values)
    result = {arm: np.zeros(a['available'].shape, float) for arm in RAW}
    for arm, raw in RAW.items():
        for i, j in zip(*np.nonzero(a['available'])):
            if i:
                result[arm][i, j] = dose * float(a[raw + '_rank'][i, j])
    return result


def verify_deltas(values, deltas, dose=DOSE):
    require(isinstance(deltas, dict) and set(deltas) == set(RAW), 'Delta arms differ')
    expected = independent_deltas(values, dose)
    for arm, wanted in expected.items():
        actual = np.asarray(deltas[arm])
        require(actual.shape == wanted.shape and np.issubdtype(actual.dtype, np.floating)
                and np.isfinite(actual).all() and np.array_equal(actual, wanted), 'Exact delta differs: ' + arm)
        require(np.all(np.abs(actual) <= abs(DOSE) * .5), 'Unbounded delta')
    return dict(passed=True, checked_events=np.shape(values['available'])[0], dose=DOSE, max_absolute_delta=.0125,
        full_candidate_universe=True, initial_delta_zero=True, common_available_only=True,
        no_eligibility_or_model_filter=True, centered_cross_sectional_ranking=True, direct_additive_pre_score=True,
        fixed_linear_mapping_checked=True,
        changed_cells={arm: int(np.count_nonzero(value)) for arm, value in expected.items()})
