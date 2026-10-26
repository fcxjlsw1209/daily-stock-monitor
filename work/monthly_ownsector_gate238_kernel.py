"""Pure own-sector eligibility gate using admitted repaired source identities.

No file access, fitting, ranking, portfolio selection or returns. A stock has one
assigned benchmark; only explicit ETFs use SPY. Source admission is hash-bound
by the host, while this kernel validates consumed identities and endpoint clocks.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np
import pandas as pd

INITIAL_PARTIAL = '2016-08-17'
START_OFFSET, END_OFFSET, RETURN_COUNT = 252, 42, 210
VALUE_KEYS = ('original_eligible', 'candidate_eligible', 'required', 'momentum',
              'benchmark_momentum', 'threshold')
COUNT_FIELDS = ('events', 'symbols_per_event', 'checked_cells', 'original_eligible_cells',
    'weak_events', 'active_events', 'required_cells', 'stock_required_cells',
    'etf_required_cells', 'source_price_observations', 'benchmark_price_observations',
    'spy_price_observations', 'rejected_cells', 'candidate_eligible_cells',
    'new_model_fits', 'predictive_model_fits')
ORIGINAL_STOCK_BENCHMARKS = frozenset(('XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'))
DATED_RULES = {
    'AZN': ('XLV', '2023-03-11', None, ['azn_20230310_issuer']),
    'BUD': ('XLP', '2016-03-15', None, ['bud_20160314_issuer_sec', 'bud_2015_20f_index']),
    'CHL': ('VOX', '2016-04-28', None, ['chl_2015_20f', 'chl_20160427_issuer_release']),
    'GOOGL': ('XLK', '2016-02-12', '2018-09-24', ['alphabet_google_2015_index', 'alphabet_google_2015_body']),
    'PDD': ('XLY', '2020-04-25', None, ['pdd_2019_20f', 'pdd_2019_20f_index']),
    'TSLA': ('XLY', '2020-02-14', None, ['tesla_2019_10k_index', 'tesla_2019_10k_business']),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _day(value):
    require(isinstance(value, str), 'Canonical ISO day required')
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid ISO day') from exc
    require(not pd.isna(result) and result.tzinfo is None and result == result.normalize()
            and value == str(result.date()), 'Canonical naive ISO day required')
    return result


def _frames(panel):
    require(isinstance(panel, dict) and set(panel) == {'adj', 'vox'}, 'Exactly adj and vox frames required')
    for label, frame in panel.items():
        require(isinstance(frame, pd.DataFrame) and isinstance(frame.index, pd.DatetimeIndex), 'Price DataFrame required')
        ix = frame.index
        require(not ix.hasnans and ix.is_unique and ix.is_monotonic_increasing and ix.tz is None
                and ix.equals(ix.normalize()), 'Price date axis invalid')
        require(frame.columns.is_unique and all(isinstance(s, str) and s and s.strip() == s
                for s in frame.columns), 'Price identity axis invalid')
    require(list(panel['vox'].columns) == ['VOX'], 'Dedicated VOX column required')


def _inputs(panel, meta, calendar, arrays, limit):
    _frames(panel)
    require(isinstance(arrays, dict) and {'eligible', 'state_weak_narrow'} <= set(arrays), 'Original E and weak state required')
    e, w = np.asarray(arrays['eligible']), np.asarray(arrays['state_weak_narrow'])
    require(e.ndim == 2 and e.shape[1] > 0 and w.shape == (e.shape[0],), 'Original array axes invalid')
    total, width = e.shape
    require(limit is None or type(limit) is int and 0 <= limit <= total, 'Invalid prefix limit')
    n = total if limit is None else limit
    require(all(isinstance(v, (bool, np.bool_)) for v in e[:n].flat)
            and all(isinstance(v, (bool, np.bool_)) for v in w[:n]), 'Consumed E/state must be boolean')
    require(isinstance(meta, dict) and all(isinstance(meta.get(k), list) and len(meta[k]) == total
            for k in ('dates', 'symbols')), 'Metadata event axes invalid')
    etfs = meta.get('etfs')
    require(isinstance(etfs, (list, tuple)) and all(isinstance(s, str) and s and s.strip() == s for s in etfs)
            and len(set(etfs)) == len(etfs), 'Explicit unique ETF identities required')
    require(isinstance(calendar, (list, tuple)), 'Official calendar sequence required')
    sessions = [_day(d) for d in calendar]
    require(all(d.weekday() < 5 for d in sessions) and all(a < b for a, b in zip(sessions, sessions[1:])),
            'Official sessions must be unique increasing weekdays')
    lookup = {d: i for i, d in enumerate(sessions)}
    queries = []
    for i in range(n):
        day = _day(meta['dates'][i]); names = meta['symbols'][i]
        require(day in lookup and (not queries or day > queries[-1][0]), 'Queries must increase on official sessions')
        require(isinstance(names, list) and len(names) == width and
                all(isinstance(s, str) and s and s.strip() == s for s in names) and len(set(names)) == width,
                'Query identities must be unique full-width IDs')
        partial = meta['dates'][i] == INITIAL_PARTIAL
        require(not partial or i == 0, 'Named partial must be first')
        pos = lookup[day]
        if not partial:
            require(pos >= START_OFFSET and pos + 1 < len(sessions)
                    and sessions[pos + 1].to_period('M') != day.to_period('M'), 'Official history/month-end witness missing')
        queries.append((day, list(names), pos, partial))
    return np.array(e[:n], dtype=bool, copy=True), np.array(w[:n], dtype=bool, copy=True), queries, sessions, set(etfs), width


def _index_list(actual, expected, label):
    require(isinstance(actual, list) and all(type(j) is int for j in actual) and actual == expected, label)


def _point(point, role, symbol, day):
    require(isinstance(point, dict) and point.get('role') == role and point.get('symbol') == symbol
            and point.get('date') == day and point.get('status') == 'valid' and point.get('reason') is None,
            'Admitted endpoint identity/status mismatch')


def _mapping(record, is_etf, day):
    require(type(record.get('is_etf')) is bool and record['is_etf'] == is_etf, 'Source ETF identity mismatch')
    benchmark = record.get('benchmark'); original = record.get('original_benchmark')
    source = record.get('mapping_source'); origin = record.get('original_mapping_source')
    repair, evidence = record.get('mapping_repair'), record.get('mapping_evidence')
    if is_etf:
        require(original == benchmark == 'SPY' and repair == 'unchanged' and evidence is None
                and source == origin == 'original_META.etfs_SPY', 'ETF must retain SPY provenance')
    else:
        require(origin in ('query_date_sector_panel', 'frozen_FOREIGN_override'), 'Original stock mapping origin invalid')
        if original == 'XLC':
            require(benchmark == 'VOX' and repair == source == 'full_period_communication_proxy'
                    and evidence is None, 'All original XLC stock mappings require VOX')
        elif original is not None:
            require(isinstance(original, str) and original in ORIGINAL_STOCK_BENCHMARKS
                    and benchmark == original and repair == 'unchanged' and source == origin and evidence is None,
                    'Original nonmissing stock mapping changed')
        else:
            rule = DATED_RULES.get(record['symbol'])
            require(rule is not None and day >= rule[1] and (rule[2] is None or day < rule[2]),
                    'No admitted dated business rule for required stock')
            require(benchmark == rule[0] and repair == source == 'dated_business_proxy' and evidence == rule[3],
                    'Dated business mapping provenance mismatch')
    require(record.get('mapping_status') == 'available' and record.get('admitted') is True
            and record.get('gap_reasons') == [], 'Required source record not admitted')
    return benchmark


def _source_records(report, eligible, queries, sessions, etfs, width):
    n = len(queries)
    require(isinstance(report, dict) and report.get('schema') == 'ownsector238_repaired_source_report_v1'
            and report.get('status') == 'ADMITTED' and report.get('admitted') is True
            and report.get('gaps') == [] and report.get('source_only') is True, 'Admitted repaired report required')
    require(all(report.get(k) is False for k in ('momentum_computed', 'eligibility_changed', 'rankings_computed',
            'orders_computed', 'returns_computed', 'strict_pit')) and type(report.get('model_fits')) is int
            and report['model_fits'] == 0, 'Source report scope invalid')
    require(isinstance(report.get('counts'), dict) and type(report['counts'].get('symbols_per_event')) is int
            and report['counts']['symbols_per_event'] == width, 'Source report width differs')
    events, records = report.get('events'), report.get('records')
    require(isinstance(events, list) and len(events) >= n and isinstance(records, list), 'Source report event/record axes invalid')
    rows = [[] for _ in range(n)]; order = []
    for record in records:
        require(isinstance(record, dict) and type(record.get('query_index')) is int and record['query_index'] >= 0,
                'Source record canonical query index required')
        i = record['query_index']
        if i >= n:
            continue  # Full-report future content is not consumed by this prefix.
        require(type(record.get('column_index')) is int, 'Source record canonical column index required')
        rows[i].append(record); order.append((i, record['column_index']))
    expected_order = []
    for i, (day, names, pos, partial) in enumerate(queries):
        event = events[i]; d = str(day.date()); ids = np.flatnonzero(eligible[i]).tolist()
        required = [] if partial else ids
        positions = [] if partial else [pos - START_OFFSET, pos - END_OFFSET]
        dates = [str(sessions[k].date()) for k in positions]
        require(isinstance(event, dict) and type(event.get('query_index')) is int and event['query_index'] == i
                and event.get('date') == d and event.get('initial_partial') is partial
                and event.get('active') is (not partial), 'Source event identity/partial mismatch')
        _index_list(event.get('original_eligible_indices'), ids, 'Source original eligibility differs')
        _index_list(event.get('required_indices'), required, 'Source required domain differs')
        _index_list(event.get('source_positions'), positions, 'Source official endpoint positions differ')
        require(event.get('source_dates') == dates, 'Source official endpoint dates differ')
        for field, count in (('required_count', len(required)), ('admitted_count', len(required)), ('gap_count', 0)):
            require(type(event.get(field)) is int and event[field] == count, 'Source event support count differs')
        market = event.get('market_endpoints')
        require(isinstance(market, list) and len(market) == (0 if partial else 2), 'Source market evidence shape differs')
        for point, role, dd in zip(market, ('market_start', 'market_end'), dates):
            _point(point, role, 'SPY', dd)  # Metadata only; never read unused SPY prices.
        _index_list([r['column_index'] for r in rows[i]], required, 'Required records missing/extra/duplicate/out of order')
        expected_order.extend((i, j) for j in required)
        for record, j in zip(rows[i], required):
            require(record.get('date') == d and record.get('symbol') == names[j], 'Required source security/date differs')
            benchmark = _mapping(record, names[j] in etfs, d)
            points = record.get('endpoints')
            require(isinstance(points, list) and len(points) == 4, 'Four admitted cell endpoints required')
            for point, role, symbol, dd in zip(points,
                    ('security_start', 'security_end', 'benchmark_start', 'benchmark_end'),
                    (names[j], names[j], benchmark, benchmark), (dates[0], dates[1], dates[0], dates[1])):
                _point(point, role, symbol, dd)
    require(order == expected_order, 'Source records must retain query/column order')
    return rows


def _momentum(frame, symbol, start, end):
    require(symbol in frame.columns and start in frame.index and end in frame.index, 'Required price endpoint missing')
    values = []
    for day in (start, end):
        value = frame.at[day, symbol]
        require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Real), 'Required price real nonboolean')
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Required price conversion invalid') from exc
        require(math.isfinite(number) and number > 0., 'Required price finite positive')
        values.append(number)
    try:
        result = values[1] / values[0] - 1.
    except (ZeroDivisionError, OverflowError) as exc:
        raise FloatingPointError('Derived momentum nonfinite') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Derived momentum nonfinite')
    return result


def build_features(panel, meta, calendar, arrays, source_report, limit=None):
    eligible, weak, queries, sessions, etfs, width = _inputs(panel, meta, calendar, arrays, limit)
    rows = _source_records(source_report, eligible, queries, sessions, etfs, width)
    n = len(queries)
    values = dict(original_eligible=eligible, candidate_eligible=eligible.copy(),
        required=np.zeros((n, width), bool), momentum=np.full((n, width), np.nan),
        benchmark_momentum=np.full((n, width), np.nan), threshold=np.full((n, width), np.nan))
    history, cells = [], []
    stocks = funds = active_count = 0
    for i, (day, names, pos, partial) in enumerate(queries):
        d = str(day.date()); ids = np.flatnonzero(eligible[i]).tolist()
        required = [] if partial else ids.copy()
        stock_ids = [j for j in required if names[j] not in etfs]
        etf_ids = [j for j in required if names[j] in etfs]
        rejected = []; positions = [] if partial else [pos - START_OFFSET, pos - END_OFFSET]
        dates = [str(sessions[k].date()) for k in positions]
        start, end = (None, None) if partial else (sessions[positions[0]], sessions[positions[1]])
        for record, j in zip(rows[i], required):
            benchmark = record['benchmark']; label = 'vox' if benchmark == 'VOX' else 'adj'
            ms = _momentum(panel['adj'], names[j], start, end)
            require(ms > 0., 'Original eligible/source positive momentum mismatch')
            mb = _momentum(panel[label], benchmark, start, end)
            threshold = max(0., mb); accepted = ms > threshold
            values['required'][i, j] = True
            values['momentum'][i, j] = ms
            values['benchmark_momentum'][i, j] = mb
            values['threshold'][i, j] = threshold
            values['candidate_eligible'][i, j] = accepted
            if not accepted:
                rejected.append(j)
            cells.append(dict(query_index=i, column_index=j, date=d, symbol=names[j],
                is_etf=names[j] in etfs, benchmark=benchmark, benchmark_source=label,
                original_benchmark=record['original_benchmark'], mapping_repair=record['mapping_repair'],
                mapping_source=record['mapping_source'], mapping_evidence=deepcopy(record['mapping_evidence']),
                source_start=dates[0], source_end=dates[1], momentum=ms,
                benchmark_momentum=mb, threshold=threshold, accepted=accepted))
        active_count += int(not partial); stocks += len(stock_ids); funds += len(etf_ids)
        history.append(dict(query_index=i, date=d, symbols=names, session_position=pos,
            weak_narrow=bool(weak[i]), initial_partial=partial, active=not partial,
            branch='initial_partial' if partial else 'ownsector_relative_gate',
            source_start=dates[0] if dates else None, source_end=dates[1] if dates else None,
            source_positions=positions, original_eligible_indices=ids, required_indices=required,
            stock_required_indices=stock_ids, etf_required_indices=etf_ids,
            rejected_indices=rejected, candidate_eligible_indices=np.flatnonzero(values['candidate_eligible'][i]).tolist()))
    r = stocks + funds; original = int(eligible.sum()); candidate = int(values['candidate_eligible'].sum())
    audit = dict(schema='monthly_ownsector_gate238_features_v1', events=n, symbols_per_event=width,
        checked_cells=n*width, original_eligible_cells=original, weak_events=int(weak.sum()),
        active_events=active_count, required_cells=r, stock_required_cells=stocks,
        etf_required_cells=funds, source_price_observations=2*r, benchmark_price_observations=2*r,
        spy_price_observations=2*funds, rejected_cells=original-candidate,
        candidate_eligible_cells=candidate, new_model_fits=0, predictive_model_fits=0,
        start_offset=START_OFFSET, end_offset=END_OFFSET, return_count=RETURN_COUNT,
        rankings_recomputed=False, source_missing_filled=False, portfolio_returns_calculated=False,
        event_history=history, cell_history=cells)
    return values, audit


build = build_features
