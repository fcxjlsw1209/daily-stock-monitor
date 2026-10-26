"""Offline VOX source integrity only. No collector import, network or strategy code.

Pure verify_source accepts explicit bytes and digest pins; the CLI additionally
requires frozen historical pins. Differences across vintages are diagnostics,
not repairs or permission to use either series in a strategy.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OLD = 'outputs/monthly_ownsector238_vox_sources'
NEW = 'outputs/monthly_residual246_vox2014_sources'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
COLLECTOR = 'work/collect_residual246_vox2014.py'
HELPER = 'work/collect_ownsector238_vox.py'
START, OLD_START, END = '2014-06-18', '2015-01-01', '2026-09-24'
SCHEMA = 'residual246_vox2014_offline_source_verification_v1'
NEW_SCHEMA = 'residual246_vox2014_acquisition_v1'
OLD_SCHEMA = 'ownsector238_vox_acquisition_v1'
FACTOR_RTOL = 1e-8
FROZEN_COLLECTOR_SHA = '52556425440e7dfbd1209f94a8b4bfc728175cb1a4ccb413ce8d4fa701547980'
PRICE_FIELDS = ('open', 'high', 'low', 'close', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose')
FIELDS = PRICE_FIELDS + ('volume', 'adjVolume', 'divCash', 'splitFactor')
FROZEN_PINS = {
    CALENDAR: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    HELPER: '3d885637e5cf848827550080f1e8906a3a1664aaa1f0f390df63b4881a08bf10',
    OLD + '/metadata.raw.json': '087a0d420a47c2f1ade00e224b20e78e9b3df077d9ef07f5e2eace12450059c1',
    OLD + '/metadata.receipt.json': '24c5e8cdb601318bb602cf1f0b7f63f055d30ff088910bb9af3f5c794ac45a88',
    OLD + '/metadata.attempt.json': '85e496e326c4d8f96d3f44ce5e3f1092ef3b1ba12cfa7b0db9b86653ffe5a942',
    OLD + '/prices.raw.json': '8911a120d54c3180feb50e4f8efee538e0fc9c26d58e2aa140802c0e55078680',
    OLD + '/prices.receipt.json': '15ea7af71f585f805749d636714f6ec776ee99a18ae96383243d3c4816bfae15',
    OLD + '/prices.attempt.json': 'fdb0d7914455bfb9def90671f3f14d26c2b488dcd79755c98d7f96e4dfdbf376',
}
FIXED_INPUTS = set(FROZEN_PINS) | {COLLECTOR} | {NEW + '/' + x for x in
    ('SCOPE.json', 'prices.raw.json', 'prices.receipt.json', 'prices.attempt.json')}
FLAGS = (
    'all_input_digests_checked', 'single_get_receipt_chain_checked',
    'cached_security_metadata_checked', 'requested_windows_checked',
    'full_official_session_coverage_checked', 'ordered_unique_utc_dates_checked',
    'finite_positive_raw_adjusted_ohlc_checked', 'ohlc_extrema_checked',
    'within_row_adjustment_factor_checked', 'corporate_action_fields_checked',
    'complete_overlap_compared', 'source_return_consistency_diagnostics_checked',
)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    require(type(raw) is bytes, 'input_not_bytes')
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate_json_key')
            out[k] = v
        return out
    def bad_constant(_):
        raise ValueError('nonstandard_json_constant')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, RecursionError):
        raise ValueError('invalid_json') from None


def iso_day(value):
    require(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is not None, 'invalid_iso_day')
    require(date.fromisoformat(value).isoformat() == value, 'invalid_iso_day')
    return value


def utc_clock(value):
    require(type(value) is str, 'invalid_utc_clock')
    try:
        d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('invalid_utc_clock') from None
    require(d.tzinfo is not None and d.utcoffset().total_seconds() == 0, 'non_utc_clock')
    return d


def finite(value, code, positive=False, nonnegative=False):
    require(type(value) in (int, float), code)
    try:
        x = float(value)
    except (ValueError, OverflowError):
        raise ValueError(code) from None
    require(math.isfinite(x), code)
    require(not positive or x > 0, code)
    require(not nonnegative or x >= 0, code)
    return x


def derived(x):
    require(math.isfinite(x), 'nonfinite_derived_diagnostic')
    return x


def descriptor(phase, start):
    endpoint = 'https://api.tiingo.com/tiingo/daily/VOX' + ('/prices' if phase == 'prices' else '')
    params = {} if phase == 'metadata' else {'startDate': start, 'endDate': END, 'format': 'json', 'resampleFreq': 'daily'}
    return dict(phase=phase, symbol='VOX', endpoint=endpoint, parameters=params,
                request_url=endpoint + ('?' + urlencode(params) if params else ''),
                method='GET', redirects_permitted=False, retries_permitted=False, maximum_gets=1)


def equal(actual, expected, code):
    # JSON type equality matters for flags/counts (True must not equal 1).
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), code)


def metadata_screen(obj):
    require(type(obj) is dict and obj.get('ticker') == 'VOX', 'security_identity_mismatch')
    name = obj.get('name')
    require(type(name) is str and 'vanguard' in name.lower() and
            ('communication' in name.lower() or 'telecommunication' in name.lower()), 'security_name_mismatch')
    require(obj.get('exchangeCode') == 'NYSE', 'cached_exchange_mismatch')
    require(iso_day(obj.get('startDate')) <= START and iso_day(obj.get('endDate')) >= END, 'metadata_window_insufficient')
    return dict(ticker='VOX', name=name, exchangeCode=obj['exchangeCode'],
                startDate=obj['startDate'], endDate=obj['endDate'], identity_history_screen_passed=True,
                exchange_listing_certified=False, history_rows_verified=False, source_admitted=False)


def acquisition_screen(rows):
    require(type(rows) is list and rows and all(type(r) is dict and 'adjClose' in r for r in rows), 'price_rows_schema')
    return dict(rows_acquired=len(rows), adjClose_key_present_in_each_row=True,
                numeric_values_verified=False, dates_unique_verified=False, date_window_verified=False,
                session_coverage_verified=False, adjustment_vintage_verified=False, source_admitted=False)


def acquisition(doc, blobs, pins, directory, phase, binding, schema, screen):
    attempt_path = directory + '/' + phase + '.attempt.json'
    raw_path = directory + '/' + phase + '.raw.json'
    receipt = doc[directory + '/' + phase + '.receipt.json']
    attempt = doc[attempt_path]
    require(type(receipt) is dict and type(attempt) is dict, 'receipt_schema')
    equal(receipt.get('binding'), binding, 'receipt_binding')
    equal(attempt.get('binding'), binding, 'attempt_binding')
    require(receipt.get('schema') == schema and attempt.get('schema') == schema, 'receipt_schema')
    require(attempt.get('attempt_budget_consumed') is True and type(attempt.get('http_gets_maximum')) is int and
            attempt['http_gets_maximum'] == 1, 'single_attempt_required')
    require(receipt.get('attempt_sha256') == pins[attempt_path] and receipt.get('raw_sha256') == pins[raw_path], 'receipt_digest_binding')
    require(type(receipt.get('raw_bytes')) is int and receipt['raw_bytes'] == len(blobs[raw_path]), 'receipt_byte_length')
    for k, v in (('request_count', 1), ('http_status', 200), ('redirects_followed', 0)):
        require(type(receipt.get(k)) is int and receipt[k] == v, 'receipt_' + k)
    require(receipt.get('outcome') == 'ACQUIRED_NOT_ADMITTED', 'receipt_outcome')
    for k in ('source_admitted', 'credential_persisted', 'features_computed', 'orders_computed', 'returns_computed'):
        require(receipt.get(k) is False, 'receipt_scope_' + k)
    t0, t1 = utc_clock(receipt.get('request_started_utc')), utc_clock(receipt.get('request_finished_utc'))
    require(t0 == utc_clock(attempt.get('attempt_started_utc')) and t1 >= t0, 'receipt_time_chain')
    headers = receipt.get('safe_response_headers')
    require(type(headers) is dict, 'receipt_headers')
    allowed = {'Date','Content-Type','Content-Length','Content-Encoding','Retry-After','ETag','Last-Modified',
               'X-RateLimit-Limit','X-RateLimit-Remaining','X-RateLimit-Reset'}
    require(set(headers) <= allowed and all(type(x) is str for x in headers.values()), 'unsafe_receipt_headers')
    require('application/json' in headers.get('Content-Type', '').lower(), 'response_content_type')
    if 'Content-Length' in headers:
        require(headers['Content-Length'].isdigit() and int(headers['Content-Length']) == len(blobs[raw_path]), 'content_length_mismatch')
    equal(receipt.get('screen'), screen, 'acquisition_screen_mismatch')
    return receipt


def calendar_sessions(obj):
    require(type(obj) is dict and type(obj.get('sessions')) is list and obj['sessions'], 'calendar_schema')
    days = [iso_day(d) for d in obj['sessions']]
    require(all(date.fromisoformat(d).weekday() < 5 for d in days), 'calendar_weekend')
    require(all(a < b for a, b in zip(days, days[1:])), 'calendar_order')
    require(days[0] <= START and days[-1] >= END and START in days and END in days, 'calendar_scope')
    return days


def inspect_rows(rows, sessions, start):
    require(type(rows) is list and bool(rows), 'empty_price_rows')
    expected = [d for d in sessions if start <= d <= END]
    require(expected, 'empty_requested_calendar')
    observed, by_day = [], {}
    factor_errors, dividends, splits, zero_volumes = [], [], [], []
    extras = set()
    for raw in rows:
        require(type(raw) is dict and set(('date',) + FIELDS) <= set(raw), 'daily_row_schema')
        extras.update(set(raw) - set(('date',) + FIELDS))
        for field in ('ticker', 'symbol'):
            if field in raw:
                require(raw[field] == 'VOX', 'row_security_mismatch')
        stamp = utc_clock(raw['date'])
        require((stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0), 'price_date_not_midnight')
        d = stamp.date().isoformat()
        require(start <= d <= END, 'price_outside_request')
        require(not observed or d > observed[-1], 'price_dates_not_unique_ordered')
        observed.append(d)
        row = {k: finite(raw[k], 'invalid_' + k, positive=k in PRICE_FIELDS or k == 'splitFactor',
                         nonnegative=k in ('volume', 'adjVolume', 'divCash')) for k in FIELDS}
        require(row['volume'].is_integer(), 'raw_volume_not_integer')
        for fields in (('open','high','low','close'), ('adjOpen','adjHigh','adjLow','adjClose')):
            op, hi, lo, cl = [row[x] for x in fields]
            require(lo <= min(op, cl) <= max(op, cl) <= hi, 'ohlc_extrema_invalid')
        ratios = [derived(row['adj' + k.title()] / row[k]) for k in ('open','high','low','close')]
        require(all(x > 0 for x in ratios), 'adjustment_factor_underflow')
        error = derived(max(ratios) / min(ratios) - 1.0)
        require(error <= FACTOR_RTOL, 'within_row_adjustment_factor_inconsistent')
        factor_errors.append(error)
        row['adjustment_factor'] = ratios[-1]
        by_day[d] = row
        if row['divCash'] != 0: dividends.append(d)
        if row['splitFactor'] != 1: splits.append(d)
        if row['volume'] == 0: zero_volumes.append(d)
    # Equality, not intersection: no common day may disappear silently.
    require(observed == expected, 'official_session_coverage_mismatch')
    return by_day, dict(rows=len(rows), first_date=observed[0], last_date=observed[-1],
        requested_start=start, requested_end=END, official_sessions=len(expected), missing_sessions=[],
        unexpected_sessions=[], dividend_dates=dividends, split_dates=splits, zero_raw_volume_dates=zero_volumes,
        maximum_within_row_factor_relative_spread=max(factor_errors), adjustment_factor_relative_tolerance=FACTOR_RTOL,
        extra_row_fields=sorted(extras))


def differences(pairs):
    changed, absolutes, relatives = [], [], []
    for day, old, new in pairs:
        delta = derived(new - old)
        if old != new: changed.append(day)
        absolutes.append(abs(delta))
        if old != 0: relatives.append(abs(derived(delta / old)))
    return dict(observations=len(pairs), changed_count=len(changed), changed_dates=changed,
                maximum_absolute_difference=max(absolutes, default=0.0),
                mean_absolute_difference=derived(math.fsum(absolutes) / len(absolutes)) if absolutes else 0.0,
                maximum_relative_difference_nonzero_old=max(relatives, default=0.0),
                relative_difference_denominator='abs(old); zero-old observations excluded only from relative statistic',
                zero_old_count=sum(old == 0 for _, old, _ in pairs))


def overlap_diagnostics(old, new, sessions):
    dates = [d for d in sessions if OLD_START <= d <= END]
    require(list(old) == dates and all(d in new for d in dates), 'overlap_dates_incomplete')
    field_stats = {k: differences([(d, old[d][k], new[d][k]) for d in dates])
                   for k in FIELDS + ('adjustment_factor',)}
    return_stats = {}
    for key in ('close', 'adjClose'):
        pairs = []
        for prev, day in zip(dates, dates[1:]):
            ratios = [derived(rows[day][key] / rows[prev][key]) for rows in (old, new)]
            require(all(x > 0 for x in ratios), 'source_return_ratio_underflow')
            pairs.append((day, derived(ratios[0] - 1.0), derived(ratios[1] - 1.0)))
        return_stats[key] = differences(pairs)
    factor_ratios = [derived(new[d]['adjustment_factor'] / old[d]['adjustment_factor']) for d in dates]
    require(all(x > 0 for x in factor_ratios), 'factor_ratio_underflow')
    return dict(first_date=dates[0], last_date=dates[-1], dates=len(dates), compared_dates=dates,
                fields=field_stats, source_daily_return_consistency=return_stats,
                adjustment_factor_new_over_old_min=min(factor_ratios),
                adjustment_factor_new_over_old_max=max(factor_ratios),
                adjustment_factor_ratio_exactly_constant=len(set(factor_ratios)) == 1,
                difference_review_required=any(x['changed_count'] for x in field_stats.values()),
                comparison_tolerance=0.0, discrepancies_are_diagnostics_only=True,
                returns_are_security_source_consistency_not_strategy_returns=True)


def manifest_paths(manifest):
    require(type(manifest) is dict and manifest.get('schema') == 'residual246_vox2014_verification_inputs_v1', 'manifest_schema')
    require(set(manifest) == {'schema','authorization_path','source_preflight_path','input_sha256'}, 'manifest_fields')
    auth, preflight = manifest['authorization_path'], manifest['source_preflight_path']
    for p in (auth, preflight):
        require(type(p) is str and p.startswith('work/') and p.endswith('.json') and '..' not in Path(p).parts, 'manifest_evidence_path')
    pins = manifest['input_sha256']
    require(type(pins) is dict and auth != preflight and set(pins) == FIXED_INPUTS | {auth, preflight}, 'manifest_input_set')
    require(all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in pins.values()), 'manifest_sha256')
    return pins, auth, preflight


def verify_source(blobs, manifest):
    """Pure independent check: no file access; supplied blobs must match manifest pins."""
    pins, auth_path, preflight_path = manifest_paths(manifest)
    require(type(blobs) is dict and set(blobs) == set(pins), 'blob_input_set')
    require(all(digest(blobs[k]) == pins[k] for k in pins), 'input_digest_mismatch')
    doc = {k: parse(v) for k, v in blobs.items() if k.endswith('.json')}
    pf = doc[preflight_path]
    require(type(pf) is dict and pf.get('schema') == 'residual246_vox2014_source_preflight_v1', 'source_preflight_schema')
    require(pf.get('source_only') is True and all(pf.get(k) is False for k in
            ('source_admitted','strategy_registered','strict_pit')), 'source_preflight_scope')
    metadata = doc[OLD + '/metadata.raw.json']
    ms = metadata_screen(metadata)
    metadata_pins = {name: pins[OLD + '/' + name] for name in
                     ('metadata.raw.json','metadata.receipt.json','metadata.attempt.json')}
    mb = dict(descriptor('metadata', OLD_START), collector_sha256=pins[HELPER], operator_metadata_review_sha256=None)
    mr = acquisition(doc, blobs, pins, OLD, 'metadata', mb, OLD_SCHEMA, ms)
    ob = dict(descriptor('prices', OLD_START), collector_sha256=pins[HELPER],
              operator_metadata_review_sha256=pins[OLD + '/metadata.raw.json'])
    old_receipt = acquisition(doc, blobs, pins, OLD, 'prices', ob, OLD_SCHEMA, acquisition_screen(doc[OLD + '/prices.raw.json']))
    require(utc_clock(mr['request_finished_utc']) <= utc_clock(old_receipt['request_started_utc']), 'metadata_not_prior_to_old_prices')
    auth = doc[auth_path]
    expected_auth = dict(schema='residual246_vox2014_source_authorization_v1', provider_host='api.tiingo.com',
                         symbol='VOX', requested_start=START, requested_end=END, metadata_file_sha256=metadata_pins)
    require(type(auth) is dict, 'authorization_schema')
    for k, v in expected_auth.items(): equal(auth.get(k), v, 'authorization_' + k)
    for k in ('metadata_exchange_reviewed','known_provider_stop_scope_reviewed','local_research_no_concurrent_collection','local_research_exclusive_execution_confirmed'):
        require(auth.get(k) is True, 'authorization_' + k)
    require(type(auth.get('known_stop_files')) is list and auth['known_stop_files'], 'authorization_stop_inventory')
    known = auth['known_stop_files']
    require(all(type(x) is dict and set(x) == {'path','sha256'} and type(x['path']) is str for x in known), 'authorization_stop_schema')
    names = [x['path'] for x in known]
    require(len(set(names)) == len(names) and OLD + '/STOPPED.json' in names, 'authorization_stop_inventory')
    preflight_ref = dict(path=preflight_path, sha256=pins[preflight_path])
    equal(auth.get('source_preflight'), preflight_ref, 'authorization_preflight_binding')
    require(auth.get('collector_sha256') == pins[COLLECTOR] and
            auth.get('frozen_helper_sha256') == pins[HELPER], 'authorization_code_binding')
    require(auth.get('external_account_activity_unknown') is True, 'authorization_external_scope')
    reviewed = utc_clock(auth.get('reviewed_at_utc'))
    require(auth.get('review_date_utc') == reviewed.date().isoformat() and
            auth.get('review_date_america_los_angeles') == reviewed.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat(), 'authorization_date_labels')
    review_ref = dict(path=auth_path, sha256=pins[auth_path], known_stop_paths=sorted(names),
                      source_preflight=preflight_ref, review_is_operator_attestation=True,
                      external_account_activity_unknown=True, shared_global_stop_mechanism=False)
    nb = dict(descriptor('prices', START), collector_sha256=pins[COLLECTOR], frozen_helper_sha256=pins[HELPER],
              source_review=review_ref, metadata_file_sha256=metadata_pins)
    nr = acquisition(doc, blobs, pins, NEW, 'prices', nb, NEW_SCHEMA, acquisition_screen(doc[NEW + '/prices.raw.json']))
    started = utc_clock(nr['request_started_utc'])
    require(utc_clock(mr['request_finished_utc']) <= started and
            utc_clock(old_receipt['request_finished_utc']) <= started, 'historical_cache_not_prior_to_new_prices')
    require(reviewed <= started, 'authorization_after_acquisition')
    require(reviewed.date() == started.date() and reviewed.astimezone(ZoneInfo('America/Los_Angeles')).date() == started.astimezone(ZoneInfo('America/Los_Angeles')).date(), 'authorization_not_acquisition_day')
    cached = dict(directory=OLD, file_sha256=metadata_pins, screen=ms, metadata_http_requests_this_invocation=0,
                  metadata_cached_verification=True, price_rows_or_coverage_certified=False, source_admitted=False)
    equal(nr.get('metadata_cached_verification'), cached, 'cached_metadata_chain')
    for k in ('strict_pit_certified','old_price_cache_read','price_splicing_performed'):
        require(nr.get(k) is False, 'new_receipt_scope_' + k)
    require(type(nr.get('metadata_http_requests_this_invocation')) is int and nr['metadata_http_requests_this_invocation'] == 0, 'new_metadata_request')
    equal(doc[NEW + '/SCOPE.json'], dict(schema=NEW_SCHEMA,binding=nb,provider_host='api.tiingo.com',source_admitted=False,
        metadata_request_permitted=False,price_splicing_permitted=False,shared_stop_ledger_modified=False,
        other_scripts_automatically_blocked=False), 'acquisition_scope_mismatch')
    sessions = calendar_sessions(doc[CALENDAR])
    new_rows, new_quality = inspect_rows(doc[NEW + '/prices.raw.json'], sessions, START)
    old_rows, old_quality = inspect_rows(doc[OLD + '/prices.raw.json'], sessions, OLD_START)
    overlap = overlap_diagnostics(old_rows, new_rows, sessions)
    return dict(schema=SCHEMA, passed=True, flags={k: True for k in FLAGS}, input_sha256=dict(pins),
        identity=ms, new_source=new_quality, old_source=old_quality, overlap=overlap,
        newly_covered_sessions=[d for d in new_rows if d < OLD_START],
        source_only=True, source_admitted=False, strategy_admitted=False, strategy_trials=0,
        strict_pit=False, historical_known_at_certified=False, currency_certified=False,
        currency=metadata.get('currency'), currency_evidence='provider metadata field only; no independent currency certificate',
        complete_corporate_action_adjustment_certified=False, collector_modules_imported=False,
        network_requests=0, strategy_features_computed=False, orders_computed=False, strategy_returns_computed=False,
        source_return_diagnostics_computed=True, sources_spliced=False, missing_rows_filled=False,
        original_source_files_modified=False, frozen_246_247_inputs_changed=False,
        provenance_scope='Hash-bound recorded GET/attempt/receipt consistency; not independent network packet evidence.',
        limitations=['Provider current-vintage values do not establish historical availability.',
                     'OHLC adjustment-factor agreement checks units, not currency or complete corporate-action correctness.',
                     'Differences from the old cache do not authorize stitching, repair, or strategy admission.',
                     'VOX has changing index composition and is not identical historical XLC exposure.'])


def safe_path(root, relative):
    require(type(relative) is str and not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'invalid_relative_path')
    p = root / relative
    require(not any(q.is_symlink() for q in (p, *p.parents)), 'symlink_forbidden')
    return p


def run_offline(manifest_path, manifest_sha256, output, *, root=ROOT):
    root = Path(root).resolve()
    mp = safe_path(root, manifest_path)
    raw = mp.read_bytes()
    require(digest(raw) == manifest_sha256, 'manifest_digest_mismatch')
    manifest = parse(raw)
    pins, _, _ = manifest_paths(manifest)
    require(all(pins.get(k) == v for k, v in FROZEN_PINS.items()), 'frozen_historical_pin_mismatch')
    require(pins[COLLECTOR] == FROZEN_COLLECTOR_SHA, 'frozen_collector_pin_mismatch')
    require(output in ('work/residual246_vox2014_SOURCE_AUDIT.json',
                       'outputs/monthly_residual246_vox2014_source_audit/REPORT.json'), 'output_path_not_dedicated')
    target = safe_path(root, output)
    require(not target.exists(), 'output_already_exists')
    blobs = {k: safe_path(root, k).read_bytes() for k in pins}
    result = verify_source(blobs, manifest)
    require(all(digest(safe_path(root, k).read_bytes()) == v for k, v in pins.items()), 'inputs_changed_during_audit')
    result.update(verification_manifest_sha256=manifest_sha256, verifier_sha256=digest(Path(__file__).read_bytes()),
                  historical_pin_policy_checked=True, checked_at_utc=datetime.now(timezone.utc).isoformat())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return dict(passed=True, report=str(target), report_sha256=digest(target.read_bytes()),
                rows=result['new_source']['rows'], overlap_dates=result['overlap']['dates'], source_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', default='work/residual246_vox2014_SOURCE_AUDIT.json')
    a = parser.parse_args()
    print(json.dumps(run_offline(a.manifest, a.manifest_sha256, a.output), sort_keys=True))


if __name__ == '__main__':
    main()
