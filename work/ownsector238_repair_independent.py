"""Independent source repair and VOX import checker; no financial arithmetic.

Only the frozen source-independent endpoint classifier is reused. The new repair
producer is never imported. Official endpoint positions are inherited from the
hash-bound original audit; this API checks their internal identity/clock binding,
not a fresh reconstruction from a calendar which it does not receive.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta
import math
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

import ownsector238_source_independent as frozen

COUNT_FIELDS = frozen.COUNT_FIELDS
RULES = {
    'AZN': ('XLV', '2023-03-10', '2023-03-11', None),
    'BUD': ('XLP', '2016-03-14', '2016-03-15', None),
    'CHL': ('VOX', '2016-04-27', '2016-04-28', None),
    'GOOGL': ('XLK', '2016-02-11', '2016-02-12', '2018-09-24'),
    'PDD': ('XLY', '2020-04-24', '2020-04-25', None),
    'TSLA': ('XLY', '2020-02-13', '2020-02-14', None),
}
FLAGS = ('passed', 'independent_repair_rebuild', 'fixed_dated_rules_checked',
    'original_required_identity_preserved', 'sealed_endpoint_clock_binding_checked',
    'named_partial_bypass_checked', 'etf_spy_precedence_checked',
    'all_original_xlc_replaced', 'nonmissing_other_mappings_preserved',
    'no_missing_price_fallback', 'missing_and_invalid_evidence_separated',
    'regular_market_endpoints_checked', 'exact_report_and_counts_checked',
    'no_future_price_consumption', 'no_strategy_math_computed')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dependency_files():
    return tuple(dict.fromkeys((Path(__file__).resolve(),
        Path(frozen.__file__).resolve()) + frozen.dependency_files()))


def day(value):
    require(isinstance(value, str), 'ISO day string required')
    parsed = date.fromisoformat(value)
    require(parsed.isoformat() == value, 'Canonical ISO day required')
    return parsed


def policy_rules(policy):
    require(isinstance(policy, dict) and policy.get('schema') ==
        'ownsector238_repair_source_policy_v1', 'Repair policy schema')
    rows = policy.get('missing_mapping_rules')
    require(isinstance(rows, list) and len(rows) == 6, 'Exactly six dated mappings')
    found = {}
    for row in rows:
        require(isinstance(row, dict), 'Mapping record required')
        symbol = row.get('symbol')
        require(symbol in RULES and symbol not in found, 'Duplicate or unknown mapping')
        actual = (row.get('benchmark'), row.get('public_date'),
                  row.get('valid_from'), row.get('valid_until_exclusive'))
        require(actual == RULES[symbol], 'Fixed mapping or publication clock changed')
        require(day(actual[2]) == day(actual[1]) + timedelta(days=1),
                'Following-calendar-date admission only')
        if actual[3] is not None:
            require(day(actual[3]) > day(actual[2]), 'Mapping expiry')
        require(isinstance(row.get('source_file'), str) and bool(row['source_file']),
                'Mapping source file required')
        refs = row.get('source_ids')
        require(isinstance(refs, list) and len(refs) > 0 and
                all(isinstance(v, str) and v for v in refs), 'Mapping evidence required')
        found[symbol] = row
    comm = policy.get('communication_benchmark')
    require(isinstance(comm, dict) and comm.get('old') == 'XLC' and
            comm.get('new') == 'VOX' and comm.get('price_field') == 'adjClose' and
            comm.get('fill_or_splice') is False, 'Fixed VOX proxy contract')
    return found


def resolve(record, policy):
    rules = policy_rules(policy)
    require(type(record.get('is_etf')) is bool, 'Explicit ETF identity required')
    query = day(record['date'])
    old = record['benchmark']
    require(old is None or isinstance(old, str) and old and old.strip() == old,
            'Original benchmark identity')
    if record['is_etf']:
        require(old == 'SPY', 'Original ETF dispatch must be SPY')
        return 'SPY', 'unchanged', None
    if old == 'XLC':
        return 'VOX', 'full_period_communication_proxy', None
    if old is not None:
        return old, 'unchanged', None
    row = rules.get(record['symbol'])
    if row is not None and query >= day(row['valid_from']) and (
            row['valid_until_exclusive'] is None or query < day(row['valid_until_exclusive'])):
        return row['benchmark'], 'dated_business_proxy', deepcopy(row['source_ids'])
    return None, 'unresolved', None


def _indices(value, width, label):
    require(isinstance(value, list) and all(type(j) is int and 0 <= j < width for j in value)
            and value == sorted(set(value)), label)


def _original_event(original, event, index, width, previous):
    require(event['query_index'] == index and type(event['query_index']) is int,
            'Original event order')
    query = day(event['date'])
    require(query.weekday() < 5 and (previous is None or query > previous),
            'Original query day order')
    partial = event['date'] == '2016-08-17'
    require(type(event['initial_partial']) is bool and event['initial_partial'] == partial and
            type(event['active']) is bool and event['active'] == (not partial),
            'Original partial/active flags')
    require(not partial or index == 0, 'Named partial must be first')
    for key in ('original_eligible_indices', 'required_indices'):
        _indices(event[key], width, 'Original ' + key)
    require(event['required_indices'] == ([] if partial else event['original_eligible_indices']),
            'Original required eligibility binding')
    require(event['required_count'] == len(event['required_indices']), 'Original required count')
    required = [r for r in original['records'] if r['query_index'] == index]
    require([r['column_index'] for r in required] == event['required_indices'],
            'Original required record order')
    if partial:
        require(event['source_dates'] == event['source_positions'] == event['market_endpoints'] == []
                and required == [], 'Partial must not have source endpoints')
    else:
        positions = event['source_positions']; dates = event['source_dates']
        require(isinstance(positions, list) and len(positions) == 2 and
                all(type(v) is int and v >= 0 for v in positions) and
                positions[1] - positions[0] == 210, 'Original endpoint session positions')
        require(isinstance(dates, list) and len(dates) == 2 and
                day(dates[0]) < day(dates[1]) < query, 'Original endpoint day clock')
        require(len(event['market_endpoints']) == 2, 'Original SPY endpoints')
        for point, role, raw in zip(event['market_endpoints'], ('market_start', 'market_end'), dates):
            require(point['role'] == role and point['symbol'] == 'SPY' and point['date'] == raw,
                    'Original SPY endpoint identity')
        symbols = []
        for row in required:
            require(row['date'] == event['date'] and isinstance(row['symbol'], str) and
                    row['symbol'] and row['symbol'].strip() == row['symbol'], 'Original cell identity')
            symbols.append(row['symbol'])
            require(type(row['is_etf']) is bool, 'Original ETF identity')
            require(len(row['endpoints']) == 4, 'Four original endpoint records required')
            expected = [('security_start', row['symbol'], dates[0]),
                        ('security_end', row['symbol'], dates[1]),
                        ('benchmark_start', row['benchmark'], dates[0]),
                        ('benchmark_end', row['benchmark'], dates[1])]
            for point, (role, symbol, raw) in zip(row['endpoints'], expected):
                require((point['role'], point['symbol'], point['date']) == (role, symbol, raw),
                        'Original cell endpoint identity/clock')
        require(len(set(symbols)) == len(symbols), 'Original query symbols must be unique')
    return required, query


def repair_sources(original_report, adj, vox_frame, policy, limit=None):
    policy_rules(policy)
    frozen.frame_schema(adj, 'original adjusted prices')
    frozen.frame_schema(vox_frame, 'dedicated VOX')
    require(list(vox_frame.columns) == ['VOX'], 'Dedicated single-column VOX frame')
    require(isinstance(original_report, dict) and original_report.get('schema') ==
            'ownsector238_source_inspection_v1', 'Original source report schema')
    require(original_report.get('source_only') is True and original_report.get('model_fits') == 0,
            'Original source-only scope')
    for flag in ('momentum_computed', 'eligibility_changed', 'rankings_computed',
                 'orders_computed', 'returns_computed', 'strict_pit'):
        require(original_report.get(flag) is False, 'Original source scope ' + flag)
    total = len(original_report['events'])
    require(limit is None or type(limit) is int and 0 <= limit <= total, 'Invalid prefix')
    n = total if limit is None else limit
    width = original_report['counts']['symbols_per_event']
    require(type(width) is int and width > 0, 'Original source width')
    counts = {key: 0 for key in COUNT_FIELDS}
    counts.update(events=n, symbols_per_event=width, checked_cells=n * width)
    events, records, gaps = [], [], []
    repairs = dict.fromkeys(('unchanged', 'full_period_communication_proxy',
                             'dated_business_proxy', 'unresolved'), 0)
    previous = None
    for i in range(n):
        original_event = original_report['events'][i]
        required, previous = _original_event(original_report, original_event, i, width, previous)
        event = deepcopy(original_event)
        event.update(admitted_count=0, gap_count=0, market_endpoints=[])
        counts['original_eligible_cells'] += len(event['original_eligible_indices'])
        counts['initial_partial_events'] += int(event['initial_partial'])
        counts['regular_events'] += int(event['active'])
        if event['active']:
            for label, raw in zip(('market_start', 'market_end'), event['source_dates']):
                point = frozen.endpoint(adj, label, 'SPY', pd.Timestamp(day(raw)))
                event['market_endpoints'].append(point)
                counts[point['status'] + '_endpoints'] += 1
                counts['market_endpoint_checks'] += 1
            reasons = list(dict.fromkeys(p['reason'] for p in event['market_endpoints'] if p['reason']))
            if reasons:
                gaps.append(dict(scope='market', query_index=i, column_index=None, date=event['date'],
                                 symbol='SPY', benchmark='SPY', reasons=reasons))
                counts['market_gap_events'] += 1
        for original in required:
            benchmark, operation, evidence = resolve(original, policy)
            row = deepcopy(original)
            row.update(original_benchmark=original['benchmark'], benchmark=benchmark,
                       mapping_repair=operation, mapping_evidence=evidence,
                       mapping_status='available' if benchmark is not None else 'missing',
                       original_mapping_source=original['mapping_source'],
                       mapping_source=original['mapping_source'] if operation == 'unchanged' else operation)
            source_days = [pd.Timestamp(day(v)) for v in event['source_dates']]
            points = []
            for label, symbol, when, frame in (
                    ('security_start', row['symbol'], source_days[0], adj),
                    ('security_end', row['symbol'], source_days[1], adj),
                    ('benchmark_start', benchmark, source_days[0], vox_frame if benchmark == 'VOX' else adj),
                    ('benchmark_end', benchmark, source_days[1], vox_frame if benchmark == 'VOX' else adj)):
                point = frozen.endpoint(frame, label, symbol, when)
                points.append(point)
                counts[point['status'] + '_endpoints'] += 1
            reasons = [] if benchmark is not None else ['missing_benchmark_mapping']
            reasons += [p['reason'] for p in points if p['reason']]
            reasons = list(dict.fromkeys(reasons))
            row.update(endpoints=points, admitted=not reasons, gap_reasons=reasons)
            records.append(row)
            repairs[operation] += 1
            counts['required_cells'] += 1
            counts['security_benchmark_endpoint_checks'] += 4
            counts['etf_required_cells' if row['is_etf'] else 'stock_required_cells'] += 1
            counts['mapping_available_cells' if benchmark is not None else 'mapping_missing_cells'] += 1
            if reasons:
                event['gap_count'] += 1; counts['gap_cells'] += 1
                gaps.append(dict(scope='cell', query_index=i, column_index=row['column_index'],
                                 date=row['date'], symbol=row['symbol'], benchmark=benchmark, reasons=reasons))
            else:
                event['admitted_count'] += 1; counts['admitted_cells'] += 1
        events.append(event)
    return dict(schema='ownsector238_repaired_source_report_v1',
        status='NOT_ADMITTED' if gaps else 'ADMITTED', admitted=not gaps, counts=counts,
        events=events, records=records, gaps=gaps, repair_counts=repairs,
        source_only=True, momentum_computed=False, eligibility_changed=False,
        rankings_computed=False, orders_computed=False, returns_computed=False,
        model_fits=0, strict_pit=False)


def check_repaired(original_report, adj, vox_frame, policy, actual, limit=None):
    expected = repair_sources(original_report, adj, vox_frame, policy, limit)
    frozen.record_check(actual, expected, 'independent repaired source')
    counts = expected['counts']
    return dict(**{key: True for key in FLAGS}, cell_count=counts['required_cells'],
        endpoint_count=counts['security_benchmark_endpoint_checks'] + counts['market_endpoint_checks'],
        counts=counts.copy(), repair_counts=expected['repair_counts'].copy(),
        status=expected['status'], admitted=expected['admitted'], producer_modules_imported=False,
        official_clock_reconstructed_from_calendar=False,
        official_clock_basis='sealed original source report; internal binding independently checked',
        no_price_values_exported=True, momentum_computed=False, eligibility_changed=False,
        rankings_computed=False, orders_computed=False, returns_computed=False,
        model_fits=0, numerical_tolerance=0.)


def check_vox(rows, calendar, produced_frame):
    require(isinstance(calendar, (list, tuple)), 'Official session list required')
    sessions = [day(v) for v in calendar]
    require(all(d.weekday() < 5 for d in sessions) and
            all(a < b for a, b in zip(sessions, sessions[1:])), 'Canonical ordered official sessions')
    require(isinstance(rows, list) and bool(rows), 'Nonempty raw VOX rows')
    dates, values = [], []
    session_set = set(sessions)
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('date'), str), 'Raw VOX row/date')
        try:
            stamp = datetime.fromisoformat(row['date'].replace('Z', '+00:00'))
        except (ValueError, TypeError) as exc:
            raise ValueError('Raw VOX ISO timestamp') from exc
        require(stamp.tzinfo is not None and stamp.utcoffset() == timedelta(0) and
                (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0),
                'Daily UTC midnight required')
        d = stamp.date()
        require(date(2015, 1, 1) <= d <= date(2026, 9, 24) and d in session_set,
                'Raw VOX requested bounds / official session')
        require(not dates or d > dates[-1], 'Raw VOX unique increasing dates')
        value = row.get('adjClose')
        require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
                'Raw VOX adjusted close real nonbool')
        try:
            converted = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Raw VOX adjusted close conversion') from exc
        require(math.isfinite(converted) and converted > 0., 'Raw VOX adjusted close finite positive')
        dates.append(d); values.append(converted)
    expected = pd.DataFrame({'VOX': np.array(values, dtype=np.float64)},
                            index=pd.DatetimeIndex(dates))
    frozen.frame_schema(produced_frame, 'imported VOX')
    require(list(produced_frame.columns) == ['VOX'] and produced_frame.index.equals(expected.index)
            and produced_frame.shape == expected.shape, 'Imported VOX identity/date/row parity')
    actual = produced_frame['VOX'].to_numpy()
    desired = expected['VOX'].to_numpy()
    require(actual.dtype == desired.dtype and actual.tobytes() == desired.tobytes(),
            'Imported VOX dtype and byte parity')
    return dict(passed=True, independent_raw_vox_rebuild=True, row_count=len(rows),
        first_date=str(dates[0]), last_date=str(dates[-1]), official_sessions_checked=True,
        requested_bounds_checked=True, utc_midnight_checked=True,
        unique_dates_checked=True, adjusted_close_exact=True, missing_rows_filled=False,
        producer_modules_imported=False, momentum_computed=False, returns_computed=False,
        numerical_tolerance=0.)
