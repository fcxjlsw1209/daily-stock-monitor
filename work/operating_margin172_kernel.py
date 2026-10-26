"""Pure same-statement operating-margin and daily-income sign features.

No source loading, market data, ranking, orders, returns or file I/O. The caller
owns source qualification and strategy registration. Only the latest report
known by each event is examined; unavailable reports never fall back to older
reports. Initial-allocation score neutralization belongs to the scoring host.
"""
from bisect import bisect_right
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

import numpy as np


SYMBOLS = frozenset({'MSFT', 'ABBV', 'AAPL'})
MAX_AGE_DAYS = 400
MIN_PERIOD_DAYS = 330
MAX_PERIOD_DAYS = 400


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _day(value, label):
    _require(isinstance(value, str), label + ' must be an ISO date string')
    try:
        result = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(label + ' must be an ISO date string') from exc
    _require(result.isoformat() == value, label + ' must be strict YYYY-MM-DD')
    return result


def _events(events):
    _require(isinstance(events, (list, tuple)), 'Events must be an ordered sequence')
    result, width = [], None
    for event in events:
        _require(isinstance(event, (list, tuple)) and len(event) == 2,
                 'Each event needs a date and member sequence')
        day = _day(event[0], 'Event')
        _require(not result or day > result[-1][0], 'Events must be unique and increasing')
        names = event[1]
        _require(isinstance(names, (list, tuple)), 'Event members must be a sequence')
        _require(bool(names) and all(isinstance(s, str) and s and s.strip() == s for s in names),
                 'Event members must be nonempty security strings')
        _require(len(set(names)) == len(names), 'Event members must be unique')
        _require(width is None or len(names) == width, 'Event membership width must be constant')
        width = len(names)
        result.append((day, list(names)))
    return result, width or 0


def _index(records):
    """Validate selection metadata only, without reading future fact values."""
    _require(isinstance(records, (list, tuple)), 'Records must be a sequence')
    grouped = {symbol: [] for symbol in SYMBOLS}
    identities, clocks = set(), set()
    for record in records:
        _require(isinstance(record, Mapping), 'Each record must be a mapping')
        symbol, report_id = record.get('symbol'), record.get('report_id')
        _require(isinstance(symbol, str) and symbol in SYMBOLS, 'Record symbol outside fixed source scope')
        _require(isinstance(report_id, str) and report_id and report_id.strip() == report_id,
                 'A nonempty report_id is required')
        year = record.get('fiscal_year')
        _require(type(year) is int and 2 <= year <= 9999, 'A valid integer fiscal_year is required')
        known = _day(record.get('known_session'), 'known_session')
        release = _day(record.get('release_date'), 'release_date')
        _require(release <= known, 'Known session precedes release date')
        identity, clock = (symbol, report_id), (symbol, known)
        _require(identity not in identities, 'Duplicate report identity')
        _require(clock not in clocks, 'Ambiguous reports at the same known session')
        identities.add(identity)
        clocks.add(clock)
        grouped[symbol].append((known, record))
    result = {}
    for symbol, items in grouped.items():
        items.sort(key=lambda item: item[0])
        result[symbol] = ([item[0] for item in items], [item[1] for item in items])
    return result


def _amount(value):
    """Exact rational representation; Decimal context cannot round the input."""
    if not isinstance(value, str) or re.fullmatch(
            r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?', value) is None:
        raise ValueError('Normalized decimal string required')
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('Finite decimal amount required') from exc
    if not number.is_finite():
        raise ValueError('Finite decimal amount required')
    return Fraction(number)


def _sign(value):
    return float((value > 0) - (value < 0))


def _measure(record, cell):
    """Read only an already selected, fresh report. Both arms share support."""
    usable = record.get('usable')
    _require(type(usable) is bool, 'Selected record usable must be boolean')
    holds = record.get('hold_reasons')
    _require(isinstance(holds, list) and all(isinstance(v, str) and v for v in holds),
             'Selected record hold_reasons must be a string list')
    cell['hold_reasons'] = list(holds)
    if not usable:
        return None, ['latest_report_unusable']
    quantities, failures = {}, []
    for side in ('current', 'comparative'):
        period = record.get(side)
        if not isinstance(period, Mapping):
            failures.append('missing_' + side + '_period')
            continue
        days = period.get('days')
        cell['input_period'][side]['days_inspected'] = True
        cell['input_period'][side]['days'] = days if type(days) is int else None
        if days is None:
            failures.append('missing_' + side + '_days')
        elif type(days) is not int or not MIN_PERIOD_DAYS <= days <= MAX_PERIOD_DAYS:
            failures.append('invalid_' + side + '_days')
        else:
            quantities[side + '_days'] = days
        for measure in ('revenue', 'operating_income'):
            try:
                value = _amount(period.get(measure))
                if measure == 'revenue' and value <= 0:
                    raise ValueError('Strictly positive revenue required')
            except ValueError:
                failures.append('invalid_' + side + '_' + measure)
            else:
                quantities[side + '_' + measure] = value
    if failures:
        return None, failures
    current_oi, prior_oi = (quantities[side + '_operating_income']
                            for side in ('current', 'comparative'))
    # All denominators are positive. Exact cross products avoid rounded ratio
    # subtraction even when the difference is much smaller than either ratio.
    margin = (current_oi * quantities['comparative_revenue'] -
              prior_oi * quantities['current_revenue'])
    income = (current_oi * quantities['comparative_days'] -
              prior_oi * quantities['current_days'])
    return (_sign(margin), _sign(income)), []


def build_features(records, events, limit=None):
    """Return candidate/control/available arrays and exhaustive cell audit.

    ``limit`` is a prefix length, including zero. Record identities and clocks
    are checked globally; fact values and qualification flags are read only
    after selecting a report known by an included event. Age is counted from
    known_session, with day 400 included. Inherited hold reasons do not override
    a true usable flag. Durations must be annual integers in [330, 400].
    """
    queries, width = _events(events)
    _require(limit is None or (type(limit) is int and 0 <= limit <= len(queries)),
             'limit must be an integer event-prefix length')
    index = _index(records)
    selected = queries if limit is None else queries[:limit]
    shape = (len(selected), width)
    values = dict(candidate=np.full(shape, np.nan), control=np.full(shape, np.nan),
                  available=np.zeros(shape, dtype=bool))
    history, reason_counts, failure_counts, inherited_counts = [], {}, {}, {}
    for row, (day, symbols) in enumerate(selected):
        cells = []
        for column, symbol in enumerate(symbols):
            cell = dict(symbol=symbol, selected_report_id=None, known_session=None,
                        release_date=None, fiscal_year=None, age_days=None, input_period=None,
                        hold_reasons=[], available=False, candidate=None, control=None,
                        reason=None, failure_reasons=[])
            result, failures = None, []
            if symbol not in SYMBOLS:
                failures = ['missing_symbol']
            else:
                clocks, reports = index[symbol]
                position = bisect_right(clocks, day) - 1
                if position < 0:
                    failures = ['no_known_report']
                else:
                    record, known = reports[position], clocks[position]
                    year = record['fiscal_year']
                    cell.update(selected_report_id=record['report_id'], known_session=known.isoformat(),
                                release_date=record['release_date'], fiscal_year=year,
                                age_days=(day - known).days,
                                input_period={side: dict(fiscal_year=year - offset, days=None,
                                                         days_inspected=False)
                                              for side, offset in (('current', 0), ('comparative', 1))})
                    if cell['age_days'] > MAX_AGE_DAYS:
                        failures = ['latest_report_stale']
                    else:
                        result, failures = _measure(record, cell)
            if result is not None:
                values['candidate'][row, column], values['control'][row, column] = result
                values['available'][row, column] = True
                cell.update(available=True, candidate=result[0], control=result[1])
            cell['reason'] = failures[0] if failures else 'available'
            cell['failure_reasons'] = failures
            reason_counts[cell['reason']] = reason_counts.get(cell['reason'], 0) + 1
            for reason in failures:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            for reason in set(cell['hold_reasons']):
                inherited_counts[reason] = inherited_counts.get(reason, 0) + 1
            cells.append(cell)
        history.append(dict(date=day.isoformat(), symbols=list(symbols), cells=cells,
                            reasons=[cell['reason'] for cell in cells]))
    count = int(values['available'].sum())
    audit = dict(events=len(selected), symbols_per_event=width,
                 source_scope=sorted(SYMBOLS), max_age_days=MAX_AGE_DAYS,
                 age_origin='known_session', age_limit_inclusive=True,
                 annual_duration_days=[MIN_PERIOD_DAYS, MAX_PERIOD_DAYS],
                 selection='latest known_session; no fallback after selection',
                 candidate_formula='sign(current_operating_income/current_revenue - comparative_operating_income/comparative_revenue)',
                 control_formula='sign(current_operating_income/current_days - comparative_operating_income/comparative_days)',
                 arithmetic='exact rational cross products from decimal strings',
                 common_availability=True, available_cells=count,
                 unavailable_cells=int(values['available'].size) - count,
                 reason_counts=reason_counts, failure_reason_counts=failure_counts,
                 inherited_hold_reason_counts=inherited_counts,
                 ranking_performed=False, initial_allocation_neutralized=False,
                 trading_eligibility_modified=False, outcomes_used=False,
                 event_history=history)
    return values, audit
