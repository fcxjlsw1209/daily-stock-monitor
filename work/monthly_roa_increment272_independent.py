"""Independent pure audit of the272 source overlay and one binary64 P correction.

No imports of the producer, source assessor, legacy financial code or runner.
The caller authenticates files and supplies physically consumed-prefix arrays.
"""
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import re
from urllib.parse import urlsplit

import numpy as np

EVENT_IDS = ('AAPL:FY2016:2016-10-25', 'AMD:FY2015:2016-01-19', 'MSFT:FY2015:2015-07-21')
FIELDS = ('current.net_income', 'current.total_assets', 'comparison.total_assets')
COUNT_FIELDS = ('events', 'symbols_per_event', 'baseline_available_cells', 'catalog_headers_checked',
                'latest_lookups', 'selected_increment_cells', 'event_bindings', 'required_cell_checks',
                'ratio_evaluations', 'bounded_evaluations', 'filled_cells', 'zero_fills', 'new_model_fits')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def text(value):
    return type(value) is str and bool(value) and value.strip() == value


def iso(value):
    need(type(value) is str, 'Canonical date string')
    parsed = date.fromisoformat(value)
    need(parsed.isoformat() == value, 'Canonical date spelling')
    return parsed


def native(value):
    if type(value) is dict:
        need(all(type(k) is str for k in value), 'Native JSON keys')
        for v in value.values(): native(v)
    elif type(value) is list:
        for v in value: native(v)
    else:
        need(type(value) in (str, int, float, bool, type(None)), 'Native JSON values')
        need(type(value) is not float or math.isfinite(value), 'Finite JSON numbers')


def canonical(value):
    native(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(a, b, why):
    need(canonical(a) == canonical(b), why)


def decimal_ratio(value):
    need(text(value) and re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value),
         'Explicit finite decimal string')
    number = Decimal(value)
    need(number.is_finite(), 'Finite decimal')
    numerator, denominator = number.as_integer_ratio()
    return Fraction(numerator, denominator)


def reference(value):
    need(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact reference')
    need(type(value['path']) is str and bool(value['path'].strip()), 'Reference path')
    need(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'SHA256')


def rich_source(value):
    keys = {'path', 'sha256', 'receipt', 'url'}
    need(type(value) is dict and keys <= set(value) <= keys | {'receipt_entry_key'}, 'Rich source schema')
    reference({k: value[k] for k in ('path', 'sha256')}); reference(value['receipt'])
    if 'receipt_entry_key' in value:
        need(value['receipt_entry_key'] is None or type(value['receipt_entry_key']) is str
             and bool(value['receipt_entry_key'].strip()), 'Receipt entry key')
    u = value['url']
    need(type(u) is str and bool(u) and not any(c.isspace() or ord(c) < 32 for c in u), 'Source URL')
    parsed = urlsplit(u)
    need(parsed.scheme == 'https' and bool(parsed.hostname) and parsed.username is None
         and parsed.password is None and '#' not in u, 'HTTPS source identity')
    parsed.port


def matrix(value, shape=None, dtype=np.float64):
    need(type(value) is np.ndarray and value.dtype == np.dtype(dtype) and value.ndim == 2,
         'Physical native matrix')
    need(value.shape[1] > 0 and (shape is None or value.shape == shape), 'Matrix axes')


def axes(meta, hand, mask):
    matrix(hand); matrix(mask, hand.shape, np.bool_)
    need(not np.isinf(hand).any() and np.array_equal(np.isfinite(hand), mask), 'Baseline finite-mask equivalence')
    n, width = hand.shape
    dates, names, etfs = meta['dates'], meta['symbols'], meta['etfs']
    need(type(dates) is list and type(names) is list and len(dates) == len(names) == n, 'Physical prefix axes')
    days = [iso(d) for d in dates]
    need(all(a < b for a, b in zip(days, days[1:])), 'Increasing dates')
    need(type(etfs) is list and all(text(s) for s in etfs) and len(etfs) == len(set(etfs)), 'ETF names')
    for row in names:
        need(type(row) is list and len(row) == width and all(text(s) for s in row)
             and len(set(row)) == width, 'Unique symbol row')
    return days, set(etfs)


def current_window(event, now, session_days):
    release, available = iso(event['release_date']), iso(event['available_from'])
    period = event['periods']['current']
    need(type(period) is dict, 'Current period object')
    opening, closing = iso(period['period_start']), iso(period['period_end'])
    duration = (closing - opening).days + 1
    need(330 <= duration <= 400 and type(period.get('days_inclusive')) is int
         and period['days_inclusive'] == duration, 'Current annual duration')
    after = next((s for s in session_days if s > release), None)
    need(session_days[0] <= release and after == available and closing <= release < available <= now,
         'Release-to-next-session clock')
    need(iso(event['periods']['comparison']['period_end']) == opening - timedelta(days=1), 'Opening instant')
    identity = event['identity']
    need(text(identity['issuer_id']) and identity['security_id'] == event['symbol']
         and identity['identity_valid_from'] == available.isoformat()
         and identity['identity_valid_through'] == (closing + timedelta(days=550)).isoformat(), 'Identity interval')
    return (now - closing).days <= 550, opening, closing


def independently_measure(event, record, opening, closing):
    need(record['increment_admitted'] is True, 'Separate increment admission')
    same(record['event_id'], event['event_id'], 'Accepted identity')
    event_hash = digest(event)
    same(record['event_sha256'], event_hash, 'Full event binding')
    d = record['saved_scope_decision']
    need(d['schema'] == 'roa_field_scope270_assessment_v2'
         and d['definition'] == 'current_annual_net_income_average_assets_field_scope_v1'
         and d['admitted'] is True and d['reasons'] == [] and d['source_only'] is True
         and d['strict_pit'] is False and d['ratios_computed'] is False and d['chain_admitted'] is False,
         'Accepted single-report scope')
    same(d['event_sha256'], event_hash, 'Decision event binding')
    need(set(record['fields']) == set(FIELDS), 'Three financial cells only')
    rich_source(event['source'])
    amounts, hashes, currencies = [], {}, []
    for field_path in FIELDS:
        side, name = field_path.split('.')
        c = event['fields'][side][name]
        same(c, record['fields'][field_path], 'Complete accepted cell')
        same(c['source'], event['source'], 'Same-report source')
        need(c['available_from'] == event['available_from'] and c['currency_literal'] == '$'
             and c['currency_iso'] in (None, 'USD'), 'Availability and literal currency')
        endpoint = closing if side == 'current' else opening - timedelta(days=1)
        need(iso(c['period']['period_end']) == endpoint and iso(c['locator']['period_end']) == endpoint,
             'Cell period and physical endpoint')
        if c.get('as_of') is not None: need(iso(c['as_of']) == endpoint, 'Explicit instant')
        if name == 'net_income':
            same(c['period'], event['periods']['current'], 'Annual NI duration')
            need(c['locator']['ni_scope'] == 'indirect_method_consolidated_start', 'Consolidated NI')
        amount, scale = decimal_ratio(c['value_decimal']), decimal_ratio(c['scale'])
        need(scale > 0 and (name == 'net_income' or amount > 0), 'Positive scale/assets')
        amounts.append(amount * scale); currencies.append(c['currency_iso']); hashes[field_path] = digest(c)
    need(currencies[0] == currencies[1] == currencies[2], 'Consistent ISO declaration')
    same(hashes, d['required_cell_sha256'], 'Decision field hashes')
    # Algebraically independent from average-assets then bounded(): one rational
    # numerator2NI and denominator(Aclose+Aopen+abs(2NI)), rounded once.
    numerator = 2 * amounts[0]
    asset_sum = amounts[1] + amounts[2]
    raw = numerator / asset_sum
    bounded = numerator / (asset_sum + abs(numerator))
    value = float(bounded)
    need(math.isfinite(value), 'Finite converted h')
    return value, [raw.numerator, raw.denominator], [bounded.numerator, bounded.denominator], hashes


def verify_overlay(meta, baseline_hand, baseline_mask, full_catalog, accepted_records,
                   sessions, actual_hand, actual_mask, audit):
    days, etfs = axes(meta, baseline_hand, baseline_mask)
    shape = baseline_hand.shape; n, width = shape
    matrix(actual_hand, shape); matrix(actual_mask, shape, np.bool_)
    result, available = baseline_hand.copy(), baseline_mask.copy()
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts.update(events=n, symbols_per_event=width, baseline_available_cells=int(baseline_mask.sum()))
    rows = []
    if n:
        need(type(full_catalog) is list and len(full_catalog) == 44, 'Complete44 catalog')
        need(type(accepted_records) is list and len(accepted_records) == 3
             and [x['event_id'] for x in accepted_records] == list(EVENT_IDS), 'Predeclared three records')
        need(type(sessions) in (list, tuple) and len(sessions) > 1, 'Session sequence')
        calendar = [iso(x) for x in sessions]
        need(all(a < b for a, b in zip(calendar, calendar[1:])) and set(days) <= set(calendar), 'Session order/membership')
        headers = []
        for e in full_catalog:
            need(text(e['event_id']) and text(e['symbol']), 'Catalog header text')
            headers.append((iso(e['available_from']), e['symbol'], e['event_id'], e))
        need(len({h[2] for h in headers}) == 44 and set(EVENT_IDS) <= {h[2] for h in headers}, 'Unique44 catalog IDs')
        counts['catalog_headers_checked'] = 44
        accepted = {x['event_id']: x for x in accepted_records}
        for i, now in enumerate(days):
            for j, symbol in enumerate(meta['symbols'][i]):
                row = dict(signal_index=i, pool_index=j, date=meta['dates'][i], symbol=symbol, event_id=None,
                    reason='baseline_preserved' if baseline_mask[i,j] else 'no_known_event', filled=False,
                    ratio_fraction=None, bounded_fraction=None, hand_value=None, required_cell_sha256=None)
                rows.append(row)
                if symbol in etfs:
                    need(not baseline_mask[i,j], 'ETF unavailable'); row['reason'] = 'etf_not_applicable'; continue
                counts['latest_lookups'] += 1
                eligible_headers = sorted((h for h in headers if h[1] == symbol and h[0] <= now),
                                          key=lambda h: h[0], reverse=True)
                if not eligible_headers: continue
                selected = eligible_headers[0]
                if len(eligible_headers) > 1 and eligible_headers[1][0] == selected[0]:
                    row['reason'] = 'ambiguous_latest'; continue
                key, event = selected[2], selected[3]; row['event_id'] = key
                if key not in accepted:
                    if not baseline_mask[i,j]: row['reason'] = 'latest_not_increment'
                    continue
                counts['selected_increment_cells'] += 1
                live, opening, closing = current_window(event, now, calendar)
                if not live: row['reason'] = 'stale_latest'; continue
                need(not baseline_mask[i,j], 'Deduplication conflict with baseline154')
                value, raw, bounded, hashes = independently_measure(event, accepted[key], opening, closing)
                result[i,j] = value; available[i,j] = True
                row.update(reason='filled', filled=True, ratio_fraction=raw, bounded_fraction=bounded,
                           hand_value=value, required_cell_sha256=hashes)
                for k in ('event_bindings', 'ratio_evaluations', 'bounded_evaluations', 'filled_cells'): counts[k] += 1
                counts['required_cell_checks'] += 3; counts['zero_fills'] += int(value == 0.)
    expected_audit = dict(schema='monthly_roa_increment272_overlay_v1', counts=counts, cells=rows,
        baseline_bytes_preserved=True, availability_independent_of_E=True, source_files_authenticated=False, computes_returns=False)
    same(audit, expected_audit, 'Exact independent overlay audit')
    need(actual_hand.tobytes(order='C') == result.tobytes(order='C')
         and actual_mask.tobytes(order='C') == available.tobytes(order='C'), 'Exact overlay bytes')
    return dict(schema='monthly_roa_increment272_overlay_independent_v1', passed=True, events=n, width=width,
        counts=counts, overlay_verifications=1, audit_matched=True, baseline_bytes_preserved=True,
        availability_independent_of_E=True, source_files_authenticated=False, computes_returns=False)


def verify_pre_score(original_pre, hand, eligible, actual_pre):
    matrix(original_pre); shape = original_pre.shape; n, width = shape
    matrix(hand, shape); matrix(eligible, shape, np.bool_); matrix(actual_pre, shape)
    need(np.isfinite(original_pre).all() and not np.isinf(hand).any(), 'Finite original P and finite-or-missing hand')
    finite = np.isfinite(hand)
    need(not (np.abs(hand[finite]) > .5).any(), 'Original adapter .5 hard bound, including E-false cells')
    expected = original_pre.copy(); active_count = 0
    for i in range(n):
        for j in range(width):
            if eligible[i,j] and finite[i,j] and hand[i,j] != 0.:
                delta = np.float64(.01) * hand[i,j]
                expected[i,j] = original_pre[i,j] + delta
                active_count += 1
    need(np.isfinite(expected).all() and actual_pre.tobytes(order='C') == expected.tobytes(order='C'),
         'One literal .01 multiply then P addition')
    return dict(schema='monthly_roa_increment272_pre_score_independent_v1', passed=True, events=n, width=width,
        P_modified_once=True, eligible_only=True, adapter_bound_checked=True, source_files_authenticated=False,
        computes_returns=False, work=dict(pre_score_verifications=1, cells_compared=n*width,
        eligible_delta_cells=active_count, hand_bound_cells=int(finite.sum()), new_model_fits=0))
