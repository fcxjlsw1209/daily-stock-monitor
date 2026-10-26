"""Independent source-only oracle and execution audit for price distribution171.

No producer kernel, strategy or execution engine is imported. Decimal scalar
arithmetic deliberately differs from production log weights / pairwise scaling.
The caller must verify registered input digests and official calendar provenance.
There is no file writing, real-data entrypoint, fitting, or automatic execution.
"""
import hashlib
import importlib.util
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution')
RAW = dict(price_activity_distribution='candidate', price_only_distribution='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .02


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), ACCOUNTING)


def accounting():
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Independent accounting source changed')
    spec = importlib.util.spec_from_file_location('distribution171_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, expected, label, atol=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, label + ': shape')
    require(np.array_equal(np.isfinite(actual), np.isfinite(expected)), label + ': finite mask')
    require(np.array_equal(np.isnan(actual), np.isnan(expected)), label + ': NaN mask')
    require(not np.isinf(actual).any() and not np.isinf(expected).any(), label + ': infinity')
    use = np.isfinite(expected)
    require(np.all(np.abs(actual[use] - expected[use]) <= atol), label + ': values')


def record_check(actual, expected, label='audit'):
    """Compare every oracle field; descriptive producer-only fields may remain."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (value, wanted) in enumerate(zip(actual, expected)):
            record_check(value, wanted, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float), label + ': scalar')
        close(actual, expected, label)
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def _day(value):
    day = pd.Timestamp(value)
    require(not pd.isna(day) and day.tzinfo is None and day == day.normalize(), 'Non-daily or zoned date')
    return day


def _schema(panel, events, calendar):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'close', 'volume'}, 'Panel keys differ')
    for frame in panel.values():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Panel frame type')
        require(not frame.index.hasnans and frame.index.is_unique and frame.index.is_monotonic_increasing,
                'Panel session axis')
        require(frame.index.tz is None and frame.index.equals(frame.index.normalize()), 'Panel date units')
        require(frame.columns.is_unique and all(isinstance(s, str) and s for s in frame.columns), 'Panel identities')
        require(frame.index.equals(panel['adj'].index) and frame.columns.equals(panel['adj'].columns), 'Panel axes differ')
    sessions = []
    for value in calendar:
        day = _day(value)
        require(isinstance(value, str) and value == str(day.date()) and day.weekday() < 5, 'Calendar session format')
        require(not sessions or sessions[-1] < day, 'Calendar order or duplicate')
        sessions.append(day)
    require(bool(sessions), 'Empty calendar')
    lookup = {day: i for i, day in enumerate(sessions)}
    queries, width = [], None
    for value, symbols in events:
        day = _day(value)
        require(day in lookup and (not queries or queries[-1][0] < day), 'Event session or order')
        require(not isinstance(symbols, (str, bytes, dict)), 'Event symbols require sequence')
        names = list(symbols)
        require(names and len(set(names)) == len(names) and all(isinstance(s, str) and s for s in names), 'Event identities')
        require(width is None or width == len(names), 'Event widths differ')
        width = len(names)
        queries.append((day, names, lookup[day]))
    return sessions, queries, width or 0


def measure_oracle(adjusted, stored_close, stored_volume):
    """211/210/210 exact values; Decimal weights never multiply float C*V."""
    arrays = []
    for values, label in ((adjusted, 'adjusted_price'), (stored_close, 'stored_close'), (stored_volume, 'stored_volume')):
        try:
            array = np.asarray(values, float)
        except (TypeError, ValueError, OverflowError):
            return None, None, 'nonnumeric_' + label, 0
        arrays.append(array)
    a, c, v = arrays
    require(a.shape == (211,) and c.shape == v.shape == (210,), 'Exact 211/210/210 samples required')
    for array, label, allow_zero in ((a, 'adjusted_price', False), (c, 'stored_close', False), (v, 'stored_volume', True)):
        if any(not math.isfinite(float(x)) or (x < 0 if allow_zero else x <= 0) for x in array):
            return None, None, 'invalid_' + label, 0
    positive = sum(float(x) > 0 for x in v)
    if positive == 0:
        return None, None, 'zero_total_activity', 0
    with localcontext() as ctx:
        ctx.prec = 80
        ref = Decimal.from_float(float(a[-1]))
        g = [(ref - Decimal.from_float(float(p))) / (ref + Decimal.from_float(float(p))) for p in a[:-1]]
        activity = [Decimal.from_float(float(x)) * Decimal.from_float(float(y)) for x, y in zip(c, v)]
        control = sum(g, Decimal(0)) / Decimal(210)
        weighted = sum((w * value for w, value in zip(activity, g)), Decimal(0)) / sum(activity, Decimal(0))
        candidate = weighted - control
        # These mathematical identities must produce exact zero, not rank noise.
        if all(value == g[0] for value in g) or all(value == activity[0] for value in activity):
            candidate = Decimal(0)
        return float(candidate), float(control), 'available', positive


def independent_features(panel, events, calendar, limit=None):
    sessions, queries, width = _schema(panel, events, calendar)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Bad prefix limit')
    queries = queries if limit is None else queries[:limit]
    values = dict(candidate=np.full((len(queries), width), np.nan), control=np.full((len(queries), width), np.nan),
                  available=np.zeros((len(queries), width), bool))
    rows, counts = [], {}
    for i, (day, symbols, position) in enumerate(queries):
        enough = position >= 252
        dates = sessions[position - 252:position - 42 + 1] if enough else []
        missing = [str(d.date()) for d in dates if d not in panel['adj'].index]
        reasons, positives = [], []
        for j, symbol in enumerate(symbols):
            if not enough:
                result = (None, None, 'insufficient_history', 0)
            elif symbol not in panel['adj'].columns:
                result = (None, None, 'missing_symbol', 0)
            elif missing:
                result = (None, None, 'missing_official_session', 0)
            else:
                result = measure_oracle([panel['adj'].at[d, symbol] for d in dates],
                    [panel['close'].at[d, symbol] for d in dates[:-1]],
                    [panel['volume'].at[d, symbol] for d in dates[:-1]])
            candidate, control, reason, positive = result
            if reason == 'available':
                values['candidate'][i, j], values['control'][i, j] = candidate, control
                values['available'][i, j] = True
            reasons.append(reason); positives.append(positive)
            counts[reason] = counts.get(reason, 0) + 1
        rows.append(dict(date=str(day.date()), symbols=symbols, session_position=position,
            window_start=str(dates[0].date()) if dates else None,
            history_end=str(dates[-2].date()) if dates else None,
            reference_date=str(dates[-1].date()) if dates else None,
            required_adjusted_count=211, required_activity_count=210,
            missing_official_sessions=missing, available=values['available'][i].tolist(),
            candidate=[float(v) if math.isfinite(v) else None for v in values['candidate'][i]],
            control=[float(v) if math.isfinite(v) else None for v in values['control'][i]],
            reasons=reasons, positive_activity_days=positives))
    audit = dict(events=len(queries), symbols_per_event=width, lookback_sessions=252, skip_sessions=42,
        history_sessions=210, adjusted_count=211, common_availability=True,
        available_cells=int(values['available'].sum()), unavailable_cells=int(values['available'].size-values['available'].sum()),
        reason_counts=counts, ranking_performed=False, trading_eligibility_modified=False,
        outcomes_used=False, event_history=rows)
    return values, audit


def _values(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Value keys differ')
    a = {key: np.asarray(value) for key, value in values.items()}
    require(a['available'].ndim == 2 and a['available'].dtype == np.dtype(bool), 'Boolean availability matrix required')
    for name in ('candidate', 'control'):
        require(a[name].shape == a['available'].shape, 'Factor shapes differ')
        require(not np.isinf(a[name]).any() and np.array_equal(np.isfinite(a[name]), a['available']), 'Common factor availability differs')
    return a


def independent_rank_deltas(values):
    a = _values(values)
    ranks = {arm: np.zeros(a['available'].shape) for arm in RAW}
    for row, mask in enumerate(a['available']):
        ids = [i for i, yes in enumerate(mask) if yes]
        if len(ids) < 2:
            continue
        for arm, raw in RAW.items():
            for i in ids:
                lower = sum(a[raw][row, j] < a[raw][row, i] for j in ids)
                equal = sum(a[raw][row, j] == a[raw][row, i] for j in ids)
                ranks[arm][row, i] = (lower + (equal + 1.) / 2.) / len(ids) - .5
    deltas = {arm: DOSE * rank for arm, rank in ranks.items()}
    for delta in deltas.values():
        if len(delta):
            delta[0] = 0.
    return ranks, deltas


def verify_features(panel, events, calendar, values, audit=None):
    actual = _values(values)
    expected, expected_audit = independent_features(panel, events, calendar)
    require(np.array_equal(actual['available'], expected['available']), 'Source availability differs')
    for name in ('candidate', 'control'):
        close(actual[name], expected[name], 'Feature ' + name)
    # Small numeric differences are tolerated only if they cannot alter ranks.
    expected_ranks, _ = independent_rank_deltas(expected)
    actual_ranks, _ = independent_rank_deltas(actual)
    for arm in RAW:
        close(actual_ranks[arm], expected_ranks[arm], 'Independent raw rank ' + arm, atol=0.)
    if audit is not None:
        record_check(audit, expected_audit)
    return dict(passed=True, checked_events=len(expected['available']), checked_cells=int(expected['available'].size),
        events=len(expected['available']), symbols_per_event=int(expected['available'].shape[1]),
        available_cells=int(expected['available'].sum()),
        unavailable_cells=int(expected['available'].size-expected['available'].sum()), computes_returns=False,
        decimal_scalar_oracle=True,
        exact_common_availability=True, independent_raw_ranks_equal=True,
        no_future_or_outcome_inputs=True, audit_checked=audit is not None)


def verify_rank_deltas(values, ranks, deltas):
    require(set(ranks) == set(deltas) == set(RAW), 'Rank arm keys differ')
    wanted_ranks, wanted_deltas = independent_rank_deltas(values)
    for arm in RAW:
        close(ranks[arm], wanted_ranks[arm], arm + ' full-pool midranks', atol=0.)
        close(deltas[arm], wanted_deltas[arm], arm + ' 2-percent delta', atol=0.)
    return dict(passed=True, checked_events=int(np.asarray(values['available']).shape[0]),
        full_original_universe=True, full_candidate_universe=True, common_available_only=True, no_eligibility_or_model_filter=True,
        dose=DOSE, initial_delta_zero=True, fewer_than_two_zero=True,
        changed_cells={arm: int(np.count_nonzero(delta)) for arm, delta in wanted_deltas.items()})


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
            record_check(wanted['metrics']['annual_returns'], annual, 'Annual returns')
            require(wanted['accounting']['exported_fill_ledger_matches'] is True and
                    wanted['accounting']['terminal_liquidation'] is False, 'Account declaration differs')
            for key in ('days', 'values', 'daily_cash_nav'):
                audited.pop(key)
            reports[arm][name] = dict(audited, passed=True, slices=slices)
    return dict(passed=True, checked_paths=9, checked_slices=45, independent_accounts=True,
        independent_of_production_model_and_engine=True, terminal_liquidation=False,
        accounting_source_sha256=ACCOUNTING_SHA, paths=reports)
