"""Pure missing-cell overlay and first88 admission; no I/O or source admission.

The caller authenticates the complete44 catalog, accepted exact-three package,
and frozen154 baseline. Only a selected, live increment's financial body is read.
Stock eligibility affects the later .01 correction, never source availability.
"""
from bisect import bisect_right
from datetime import timedelta

import numpy as np
import monthly_cash_quality146_factor as legacy
from monthly_same_report145_runner import guard
import roa_field_scope270_v2 as source

ARMS = ('incumbent_local67', 'final154_expanded_earnings_control', 'exact3_roa_increment_candidate')
EVENT_IDS = ('AAPL:FY2016:2016-10-25', 'AMD:FY2015:2016-01-19', 'MSFT:FY2015:2015-07-21')
COUNT_FIELDS = ('events','symbols_per_event','baseline_available_cells','catalog_headers_checked',
    'latest_lookups','selected_increment_cells','event_bindings','required_cell_checks',
    'ratio_evaluations','bounded_evaluations','filled_cells','zero_fills','new_model_fits')
AUDIT_KEYS = ('schema','counts','cells','baseline_bytes_preserved','availability_independent_of_E',
    'source_files_authenticated','computes_returns')
CELL_KEYS = ('signal_index','pool_index','date','symbol','event_id','reason','filled',
    'ratio_fraction','bounded_fraction','hand_value','required_cell_sha256')
require, day, fraction = legacy._require, legacy._date, legacy._fraction


def _equal(a, b, message):
    require(source.canonical_sha256(a) == source.canonical_sha256(b), message)


def _axes(meta, hand, mask):
    require(type(hand) is np.ndarray and hand.dtype == np.float64 and hand.ndim == 2
        and hand.shape[1] > 0, 'Physical float64 hand matrix required')
    require(type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.shape == hand.shape,
        'Physical bool mask required')
    require(not np.isinf(hand).any() and np.array_equal(np.isfinite(hand), mask), 'Frozen finite-mask equivalence')
    dates, symbols, etfs = meta['dates'], meta['symbols'], meta['etfs']
    n, width = hand.shape
    require(type(dates) is list and type(symbols) is list and len(dates) == len(symbols) == n,
        'Already consumed-prefix metadata required')
    require(type(etfs) is list and all(legacy._text(s) for s in etfs) and len(set(etfs)) == len(etfs), 'ETF identities')
    parsed = [day(x) for x in dates]
    require(parsed == sorted(set(parsed)), 'Strict increasing query dates')
    for names in symbols:
        require(type(names) is list and len(names) == width and all(legacy._text(s) for s in names)
            and len(set(names)) == width, 'Unique symbol axes')
    return dates, symbols, set(etfs)


def _live(event, signal, sessions):
    """Validate selected metadata before any financial-body access."""
    released, available = day(event['release_date']), day(event['available_from'])
    current = event['periods']['current']
    period, error = legacy._period(current)
    require(error is None, 'Complete annual current period required')
    start, end = period['start'], period['end']
    j = bisect_right(sessions, released.isoformat())
    require(0 < j < len(sessions) and sessions[j] == available.isoformat()
        and end <= released < available <= signal, 'Release and next-session clock')
    require(day(event['periods']['comparison']['period_end']) == start-timedelta(days=1), 'Opening asset endpoint')
    ident = event['identity']
    require(legacy._text(ident['issuer_id']) and ident['security_id'] == event['symbol']
        and ident['identity_valid_from'] == available.isoformat()
        and ident['identity_valid_through'] == (end+timedelta(days=550)).isoformat(), 'Selected report identity interval')
    return (signal-end).days <= 550


def _measure(event, accepted, counts):
    """Reuse146 exact decimal parsing and bounded operation; no flow comparison."""
    require(accepted['increment_admitted'] is True, 'Explicit separately accepted increment required')
    _equal(accepted['event_id'], event['event_id'], 'Accepted event identity')
    esha = source.canonical_sha256(event); counts['event_bindings'] += 1
    _equal(accepted['event_sha256'], esha, 'Complete selected event binding')
    decision = accepted['saved_scope_decision']
    require(decision['schema'] == source.SCHEMA and decision['definition'] == source.DEFINITION
        and decision['admitted'] is True and decision['reasons'] == [] and decision['source_only'] is True
        and decision['strict_pit'] is False and decision['ratios_computed'] is False and decision['chain_admitted'] is False,
        'Saved three-field admission scope')
    _equal(decision['event_sha256'], esha, 'Decision event binding')
    require(set(accepted['fields']) == set(source.CELL_NAMES), 'Exactly three accepted fields')
    source._source(event['source'])
    values, hashes, currencies = {}, {}, []
    start, end = day(event['periods']['current']['period_start']), day(event['periods']['current']['period_end'])
    for side, name in source.REQUIRED:
        key = side+'.'+name; cell = event['fields'][side][name]
        _equal(cell, accepted['fields'][key], 'Accepted cell body')
        hashes[key] = source.canonical_sha256(cell)
        _equal(cell['source'], event['source'], 'Same-report source')
        require(cell['available_from'] == event['available_from'] and cell['currency_literal'] == '$'
            and cell['currency_iso'] in (None, 'USD'), 'Same-report availability and currency')
        currencies.append(cell['currency_iso'])
        expected = end if side == 'current' else start-timedelta(days=1)
        require(day(cell['period']['period_end']) == expected and day(cell['locator']['period_end']) == expected,
            'Selected cell period/locator endpoint')
        if cell.get('as_of') is not None: require(day(cell['as_of']) == expected, 'Explicit as_of mismatch')
        if name == 'net_income':
            _equal(cell['period'], event['periods']['current'], 'NI current annual duration')
            require(cell['locator']['ni_scope'] == 'indirect_method_consolidated_start', 'Consolidated NI scope')
        value, scale = fraction(cell['value_decimal']), fraction(cell['scale'])
        require(scale > 0 and (name != 'total_assets' or value > 0), 'Positive scale and asset endpoints')
        values[key] = value*scale; counts['required_cell_checks'] += 1
    require(len(set(currencies)) == 1, 'Same-report ISO declaration')
    _equal(hashes, decision['required_cell_sha256'], 'Required cell digest binding')
    average = (values['current.total_assets']+values['comparison.total_assets'])/2
    ratio = values['current.net_income']/average; counts['ratio_evaluations'] += 1
    bounded = ratio/(1+abs(ratio))
    hand = legacy.bounded(ratio); counts['bounded_evaluations'] += 1
    require(np.isfinite(hand), 'Finite bounded hand')
    return hand, [ratio.numerator,ratio.denominator], [bounded.numerator,bounded.denominator], hashes


def overlay_prefix(meta, baseline_hand, baseline_mask, full_catalog, accepted_records, sessions):
    dates, symbols, etfs = _axes(meta, baseline_hand, baseline_mask)
    hand, mask = baseline_hand.copy(), baseline_mask.copy()
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts.update(events=len(dates), symbols_per_event=hand.shape[1], baseline_available_cells=int(mask.sum()))
    cells = []
    audit = dict(schema='monthly_roa_increment272_overlay_v1', counts=counts, cells=cells,
        baseline_bytes_preserved=True, availability_independent_of_E=True,
        source_files_authenticated=False, computes_returns=False)
    if not dates: return hand, mask, audit
    require(type(full_catalog) is list and len(full_catalog) == 44, 'Complete44 source catalog required')
    require(type(accepted_records) is list and len(accepted_records) == 3
        and [r['event_id'] for r in accepted_records] == list(EVENT_IDS), 'Exact predeclared three records')
    accepted = dict(zip(EVENT_IDS, accepted_records))
    require(type(sessions) in (list,tuple) and len(sessions) > 1, 'Pinned session sequence required')
    parsed = [day(x) for x in sessions]
    require(parsed == sorted(set(parsed)), 'Strict increasing sessions')
    require(set(dates) <= set(sessions), 'Query dates must belong to pinned sessions')
    headers, ids = {}, set()
    for event in full_catalog:
        key, symbol = event['event_id'], event['symbol']
        require(legacy._text(key) and legacy._text(symbol) and key not in ids, 'Unique catalog event IDs')
        ids.add(key); available = day(event['available_from'])
        headers.setdefault(symbol, []).append((available, key, event))
        counts['catalog_headers_checked'] += 1
    require(set(EVENT_IDS) <= ids, 'Three accepted records must remain in complete catalog')
    for i, (signal, names) in enumerate(zip(dates, symbols)):
        now = day(signal)
        for j, symbol in enumerate(names):
            row = dict(signal_index=i,pool_index=j,date=signal,symbol=symbol,event_id=None,
                reason='baseline_preserved' if mask[i,j] else 'no_known_event',filled=False,
                ratio_fraction=None,bounded_fraction=None,hand_value=None,required_cell_sha256=None)
            cells.append(row)
            if symbol in etfs:
                require(not mask[i,j], 'No financial factor for ETF'); row['reason']='etf_not_applicable'; continue
            counts['latest_lookups'] += 1
            known = [x for x in headers.get(symbol,()) if x[0] <= now]
            if not known: continue
            latest = max(x[0] for x in known); chosen = [x for x in known if x[0] == latest]
            if len(chosen) != 1: row['reason']='ambiguous_latest'; continue
            _, key, event = chosen[0]; row['event_id']=key
            if key not in accepted:
                if not mask[i,j]: row['reason']='latest_not_increment'
                continue
            counts['selected_increment_cells'] += 1
            if not _live(event,now,sessions): row['reason']='stale_latest'; continue
            require(not mask[i,j], 'Deduplication conflict: live new interval is already available in154')
            value, ratio, bounded, hashes = _measure(event,accepted[key],counts)
            hand[i,j]=value;mask[i,j]=True
            row.update(reason='filled',filled=True,ratio_fraction=ratio,bounded_fraction=bounded,
                hand_value=value,required_cell_sha256=hashes)
            counts['filled_cells'] += 1;counts['zero_fills'] += int(value == 0.)
    untouched = ~(mask & ~baseline_mask)
    require(hand.view(np.uint64)[untouched].tobytes() == baseline_hand.view(np.uint64)[untouched].tobytes()
        and mask[untouched].tobytes() == baseline_mask[untouched].tobytes(), 'Untouched baseline bytes')
    return hand, mask, audit


def select_first88(rows, *, execution='close', cost_per_side=0.0, training_months=88):
    require(execution == 'close' and type(cost_per_side) is float and cost_per_side == 0.0
        and type(training_months) is int and training_months == 88, 'First88 same-close ZERO only')
    require(type(rows) is list and len(rows) == 3 and [r['name'] for r in rows] == list(ARMS), 'Three fixed arms in order')
    checks = {ARMS[i]:guard(rows[2], rows[i], strict=True) for i in (0,1)}
    passed = all(checks.values())
    return dict(schema='monthly_roa_increment272_first88_v1',primary=ARMS[2] if passed else ARMS[0],
        admitted=passed,guards_against=checks,selection_execution='close',cost_per_side=0.0,
        training_months=88,early_months=53,recent_months=35,guard_evaluations=2,
        metric_scalar_reads=20,later_used=False,next_open_used=False,cost_diagnostics_used=False,
        control_promotable=False)
