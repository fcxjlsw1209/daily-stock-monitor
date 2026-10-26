"""Independent rational source-feature oracle and frozen accounting reuse.

Pure APIs only: no source loading, real-data entrypoint, fitting or output writes.
The caller must verify registered input digests before supplying source objects.
"""
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_HELPER = ROOT / 'work/price_distribution171_independent.py'
OUTCOME_SHA = '94e2a984affad28ae1ba62de245519ffd9f4763c1a6ab3a5cd95dea02e0a1841'
ACCOUNTING = ROOT / 'work/ptr169_results_independent.py'
ACCOUNTING_SHA = '6264571ad6ddacbc408449180c74a32b37659757c9bdfc5ce0e2672467607d59'
ARMS = ('incumbent_local67', 'annual_margin_confirmation', 'annual_income_confirmation')
RAW = dict(annual_margin_confirmation='candidate', annual_income_confirmation='control')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
DOSE = .01
SYMBOLS = frozenset(('AAPL', 'ABBV', 'MSFT'))
NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_files():
    return (Path(__file__).resolve(), OUTCOME_HELPER, ACCOUNTING)


def accounting():
    """Expose the pinned scalar-accounting API for frozen training checks."""
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('margin172_isolated_accounting', ACCOUNTING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome_helper():
    require(sha(OUTCOME_HELPER) == OUTCOME_SHA, 'Frozen independent outcome helper changed')
    require(sha(ACCOUNTING) == ACCOUNTING_SHA, 'Frozen independent accounting changed')
    spec = importlib.util.spec_from_file_location('margin172_isolated_outcome', OUTCOME_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A fresh isolated module object; no frozen source or shared module changed.
    require(module.ARMS == ('incumbent_local67', 'price_activity_distribution', 'price_only_distribution'),
            'Unexpected original outcome arms')
    module.ARMS = ARMS
    return module


def day(value):
    require(isinstance(value, str), 'Strict ISO date string required')
    result = date.fromisoformat(value)
    require(result.isoformat() == value, 'Strict YYYY-MM-DD required')
    return result


def amount(value):
    require(isinstance(value, str) and NUMBER.fullmatch(value), 'Normalized finite decimal string required')
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('Normalized finite decimal string required') from exc
    require(parsed.is_finite(), 'Nonfinite amount')
    return Fraction(parsed)


def record_check(actual, expected, label='audit'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            record_check(actual[key], value, label + '/' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for i, (value, target) in enumerate(zip(actual, expected)):
            record_check(value, target, label + '/' + str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and actual == expected, label + ': scalar')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': differs')


def metadata(records, events):
    require(isinstance(records, (list, tuple)) and isinstance(events, (list, tuple)), 'Sequences required')
    catalog, seen_ids, seen_clocks = [], set(), set()
    for row in records:
        require(isinstance(row, Mapping), 'Source record must be a mapping')
        symbol, rid, year = row.get('symbol'), row.get('report_id'), row.get('fiscal_year')
        require(symbol in SYMBOLS, 'Source record outside supported issuers')
        require(isinstance(rid, str) and rid and rid.strip() == rid, 'Nonempty report identity required')
        require(type(year) is int and 2 <= year <= 9999, 'Fiscal year integer required')
        known, release = day(row.get('known_session')), day(row.get('release_date'))
        require(release <= known, 'Knowledge time precedes release')
        require((symbol, rid) not in seen_ids and (symbol, known) not in seen_clocks, 'Ambiguous source identity or knowledge clock')
        seen_ids.add((symbol, rid)); seen_clocks.add((symbol, known))
        catalog.append((symbol, known, row))
    queries, width = [], 0
    for query in events:
        require(isinstance(query, (list, tuple)) and len(query) == 2, 'Event pair required')
        when, symbols = day(query[0]), query[1]
        require(not queries or when > queries[-1][0], 'Events must be unique and increasing')
        require(isinstance(symbols, (list, tuple)) and symbols, 'Nonempty event symbol sequence required')
        require(all(isinstance(s, str) and s and s.strip() == s for s in symbols), 'Invalid event symbol')
        require(len(set(symbols)) == len(symbols), 'Duplicate event symbol')
        require(not queries or width == len(symbols), 'Event width changed')
        width = len(symbols)
        queries.append((when, list(symbols)))
    return catalog, queries, width


def measure_selected(row, cell):
    usable, holds = row.get('usable'), row.get('hold_reasons')
    require(type(usable) is bool, 'Selected source usable flag must be bool')
    require(isinstance(holds, list) and all(isinstance(x, str) and x for x in holds), 'Selected hold reasons must be strings')
    cell['hold_reasons'] = list(holds)
    if not usable:
        return None, ['latest_report_unusable']
    parsed, failures = {}, []
    for role in ('current', 'comparative'):
        source = row.get(role)
        if not isinstance(source, Mapping):
            failures.append('missing_' + role + '_period')
            continue
        duration = source.get('days')
        cell['input_period'][role].update(days_inspected=True, days=duration if type(duration) is int else None)
        if duration is None:
            failures.append('missing_' + role + '_days')
        elif type(duration) is not int or not 330 <= duration <= 400:
            failures.append('invalid_' + role + '_days')
        else:
            parsed[role + '_days'] = duration
        for field in ('revenue', 'operating_income'):
            try:
                number = amount(source.get(field))
                require(field != 'revenue' or number > 0, 'Revenue must be positive')
                parsed[role + '_' + field] = number
            except ValueError:
                failures.append('invalid_' + role + '_' + field)
    if failures:
        return None, failures
    # Explicit rational divisions differ from production cross-product method.
    margins = [parsed[role + '_operating_income'] / parsed[role + '_revenue'] for role in ('current', 'comparative')]
    daily = [parsed[role + '_operating_income'] / parsed[role + '_days'] for role in ('current', 'comparative')]
    sign = lambda value: float((value > 0) - (value < 0))
    return (sign(margins[0] - margins[1]), sign(daily[0] - daily[1])), []


def independent_features(records, events, limit=None):
    catalog, queries, width = metadata(records, events)
    require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)), 'Invalid prefix limit')
    queries = queries if limit is None else queries[:limit]
    shape = (len(queries), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan), available=np.zeros(shape, bool))
    history, reason_counts, failure_counts, hold_counts = [], {}, {}, {}
    for i, (when, symbols) in enumerate(queries):
        cells = []
        for j, symbol in enumerate(symbols):
            cell = dict(symbol=symbol, selected_report_id=None, known_session=None, release_date=None,
                        fiscal_year=None, age_days=None, input_period=None, hold_reasons=[],
                        available=False, candidate=None, control=None, reason=None, failure_reasons=[])
            possible = [(known, row) for name, known, row in catalog if name == symbol and known <= when]
            measured = None
            if symbol not in SYMBOLS:
                failures = ['missing_symbol']
            elif not possible:
                failures = ['no_known_report']
            else:
                known, row = max(possible, key=lambda pair: pair[0])
                year = row['fiscal_year']
                cell.update(selected_report_id=row['report_id'], known_session=known.isoformat(),
                    release_date=row['release_date'], fiscal_year=year, age_days=(when - known).days,
                    input_period={role: dict(fiscal_year=year - offset, days=None, days_inspected=False)
                                  for role, offset in (('current', 0), ('comparative', 1))})
                if cell['age_days'] > 400:
                    failures = ['latest_report_stale']
                else:
                    measured, failures = measure_selected(row, cell)
            if measured is not None:
                values['candidate'][i, j], values['control'][i, j] = measured
                values['available'][i, j] = True
                cell.update(available=True, candidate=measured[0], control=measured[1])
            cell.update(reason=failures[0] if failures else 'available', failure_reasons=failures)
            reason_counts[cell['reason']] = reason_counts.get(cell['reason'], 0) + 1
            for label in failures:
                failure_counts[label] = failure_counts.get(label, 0) + 1
            for label in set(cell['hold_reasons']):
                hold_counts[label] = hold_counts.get(label, 0) + 1
            cells.append(cell)
        history.append(dict(date=when.isoformat(), symbols=symbols, cells=cells, reasons=[c['reason'] for c in cells]))
    audit = dict(events=len(queries), symbols_per_event=width, source_scope=sorted(SYMBOLS),
        max_age_days=400, age_origin='known_session', age_limit_inclusive=True, annual_duration_days=[330, 400],
        common_availability=True, available_cells=int(values['available'].sum()),
        unavailable_cells=int(values['available'].size - values['available'].sum()),
        reason_counts=reason_counts, failure_reason_counts=failure_counts, inherited_hold_reason_counts=hold_counts,
        ranking_performed=False, initial_allocation_neutralized=False, trading_eligibility_modified=False,
        outcomes_used=False, event_history=history)
    return values, audit


def arrays(values):
    require(isinstance(values, dict) and set(values) == {'candidate', 'control', 'available'}, 'Feature keys differ')
    result = {key: np.asarray(value) for key, value in values.items()}
    mask = result['available']
    require(mask.ndim == 2 and mask.dtype == bool, 'Two-dimensional boolean availability required')
    for key in ('candidate', 'control'):
        a = result[key]
        require(a.shape == mask.shape and np.issubdtype(a.dtype, np.number), 'Raw sign shape/type differs')
        require(not np.isinf(a).any() and np.array_equal(np.isfinite(a), mask), 'Common sign availability differs')
        require(np.all(np.isin(a[mask], [-1., 0., 1.])), 'Raw factor must be an exact sign')
    return result


def verify_features(records, events, values, audit=None):
    actual = arrays(values)
    expected, wanted_audit = independent_features(records, events)
    require(np.array_equal(actual['available'], expected['available']), 'Independent availability differs')
    for name in ('candidate', 'control'):
        require(np.array_equal(actual[name], expected[name], equal_nan=True), 'Independent rational sign differs: ' + name)
    if audit is not None:
        record_check(audit, wanted_audit)
    return dict(passed=True, events=len(events), symbols_per_event=expected['available'].shape[1],
        checked_events=len(events), checked_cells=int(expected['available'].size),
        available_cells=int(expected['available'].sum()), unavailable_cells=int((~expected['available']).sum()),
        rational_scalar_oracle=True, exact_common_availability=True, independent_raw_signs_equal=True,
        no_future_or_outcome_inputs=True, no_fallback=True, no_percentile_rank=True,
        computes_returns=False, audit_checked=audit is not None)


def independent_rank_deltas(values):
    a = arrays(values)
    raw = {arm: np.where(a['available'], a[key], 0.) for arm, key in RAW.items()}
    deltas = {arm: DOSE * scores for arm, scores in raw.items()}
    for delta in deltas.values():
        if len(delta):
            delta[0] = 0.
    return raw, deltas


def verify_rank_deltas(values, rawscores, deltas):
    require(set(rawscores) == set(deltas) == set(RAW), 'Score arm keys differ')
    wanted_raw, wanted_delta = independent_rank_deltas(values)
    for arm in RAW:
        require(np.array_equal(rawscores[arm], wanted_raw[arm]), 'Direct sign score differs: ' + arm)
        require(np.array_equal(deltas[arm], wanted_delta[arm]), 'One-percent sign delta differs: ' + arm)
    return dict(passed=True, checked_events=arrays(values)['available'].shape[0], full_candidate_universe=True,
        full_original_universe=True, initial_delta_zero=True, common_available_only=True,
        no_eligibility_or_model_filter=True, no_cross_sectional_ranking=True, dose=DOSE,
        changed_cells={arm: int(np.count_nonzero(delta)) for arm, delta in wanted_delta.items()})


def verify_source_records(records, html, aapl, review):
    """Verify normalized 35-record input against registered source-review objects.

    No financial ratio is calculated. Dates and original strings are compared.
    External file digests remain the caller's precondition, not inferred here.
    """
    html = html['reports'] if isinstance(html, dict) else html
    aapl = aapl['records'] if isinstance(aapl, dict) else aapl
    scope = {(symbol, year) for symbol, years in [('MSFT', range(2015, 2027)),
             ('ABBV', range(2013, 2026)), ('AAPL', range(2016, 2026))] for year in years}
    require(review.get('passed') is True and review.get('source_reports') == 35
            and review.get('checked_annual_cells') == 140, 'Completed 35-source independent review required')
    groups = []
    for rows in (records, html + aapl, review['reports']):
        mapped = {}
        for row in rows:
            key = (row.get('issuer', row.get('symbol')), row['fiscal_year'])
            require(key not in mapped, 'Duplicate normalized/source/review year')
            mapped[key] = row
        require(set(mapped) == scope, 'Fixed 35-source scope differs')
        groups.append(mapped)
    normalized, sources, checked = groups
    unavailable = []
    for (symbol, year), row in normalized.items():
        source, witness = sources[symbol, year], checked[symbol, year]
        require(witness.get('passed') is True and witness.get('checked_annual_cells') == 4
                and witness.get('next_session_clock_rederived') is True, 'Unreviewed source record')
        apple = symbol == 'AAPL'
        fields = source['fields'] if apple else source['cells']
        rid = 'AAPL_FY' + str(year) if apple else source['report_id']
        source_sha = source['source']['sha256'] if apple else source['source']['raw']['sha256']
        require(source_sha == witness['source_sha256'], 'Source-review raw digest differs')
        require(row['report_id'] == rid and row['release_date'] == witness['release_date']
                and row['known_session'] == witness['known_session'], 'Source identity or clock changed')
        if 'source_sha256' in row:
            require(row['source_sha256'] == source_sha, 'Normalized source digest changed')
        require(row['hold_reasons'] == source.get('holds', []), 'Inherited source holds changed')
        day(row['known_session']); day(row['release_date'])
        durations = []
        for role in ('current', 'comparative'):
            revenue, income = fields['revenue'][role], fields['operating_income'][role]
            period = revenue['period'] if apple else source['periods'][role]
            start, end = period.get('start'), period.get('end')
            duration = None if start is None or end is None else (day(end) - day(start)).days + 1
            require(period.get('days_inclusive') == duration, 'Source annual duration does not match dates')
            require(duration is None or 330 <= duration <= 400, 'Source period not annual')
            durations.append(duration)
            require(row[role] == dict(revenue=revenue['amount_decimal'], operating_income=income['amount_decimal'], days=duration),
                    'Normalized four cells or annual duration changed')
            amount(row[role]['revenue']); amount(row[role]['operating_income'])
            if apple:
                require(income['period'] == period and income['known_at_session'] == revenue['known_at_session'] == row['known_session'],
                        'Apple source pairing or knowledge clock differs')
                require(source['publication']['release_date'] == row['release_date'], 'Apple publication changed')
            else:
                require(revenue['period_start'] == income['period_start'] == start
                        and revenue['period_end'] == income['period_end'] == end, 'HTML annual periods differ')
                require(revenue['known_by'] == income['known_by'] == witness['known_by'], 'HTML source knowledge clocks differ')
                require(revenue['release_date'] == income['release_date'] == row['release_date'], 'HTML release dates differ')
        usable = all(n is not None for n in durations)
        require(type(row['usable']) is bool and row['usable'] == usable, 'Source usability differs')
        if not usable:
            unavailable.append([symbol, year])
    require(unavailable == [['AAPL', 2016]], 'Unexpected source exclusion scope')
    return dict(passed=True, checked_records=35, source_reports=35, checked_annual_cells=140,
        exact_source_cells=True, exact_source_clocks=True, exact_source_durations=True,
        source_hold_preservation=True, unavailable_records=unavailable,
        financial_ratios_computed=False, compares_pinned_objects_only=True)


def audit_outcomes(px, opening, orders, exports, results, meta):
    """Reuse the frozen independent nine-path/five-slice accounting checks."""
    result = outcome_helper().audit_outcomes(px, opening, orders, exports, results, meta)
    result['outcome_helper_sha256'] = OUTCOME_SHA
    return result
