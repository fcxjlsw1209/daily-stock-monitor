"""Source-only PTR169 quote audit. No valuation, pool, features, or returns."""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_ptr_pool_source169'
PLAN_SHA = '2a25ff6a4b7a3f7732ed9814a886fe53c244ae0902b8c65bfa94e2af1f83681f'
RECEIPTS = {
    '601857.SS.json': '59cd1f6307f3e6d4fceb8ef8bb36c99ccc59e5f680d0055b8948a17239855840',
    '0857.HK.json': 'c0823706db17a78bebde432bef5daeff31a17e80bdb3506d0d64383d82c98963',
    'CNY_X.json': '59a8e3629a16c5e6ce59a018898a79d78d8b54e379130befe4fb8c50492261d9',
    'HKD_X.json': 'd6d91e9e308a1b273cbe3eac1d4ce8f8dc40f7e8dcc7d7e2e7d0b4112abb70fc',
    'petrochina_2015_annual.pdf': 'e27bcc74451f920309f165572a30049585ffce5bc89e7ad263502d11bd80f7e5',
}
EXPECTED = {
    '601857.SS.json': ('601857.SS', 'CNY', 'Asia/Shanghai', 'SHH', 'EQUITY'),
    '0857.HK.json': ('0857.HK', 'HKD', 'Asia/Hong_Kong', 'HKG', 'EQUITY'),
    'CNY_X.json': ('CNY=X', 'CNY', 'Europe/London', 'CCY', 'CURRENCY'),
    'HKD_X.json': ('HKD=X', 'HKD', 'Europe/London', 'CCY', 'CURRENCY'),
}
FIELDS = ('open', 'high', 'low', 'close', 'volume')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def iso_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def valid_price(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def normalize_chart(body, expected):
    """Use the IANA exchange timezone, never current gmtoffset or UTC dates."""
    chart = body['chart']
    assert chart.get('error') is None and len(chart['result']) == 1
    result = chart['result'][0]
    meta = result['meta']
    for field, value in zip(('symbol', 'currency', 'exchangeTimezoneName',
                             'exchangeName', 'instrumentType'), expected):
        assert meta[field] == value, (field, meta.get(field), value)
    assert meta['dataGranularity'] == '1d'
    if expected[-1] == 'EQUITY':
        assert meta['longName'] == 'PetroChina Company Limited'
    else:
        assert meta['longName'] == 'USD/' + expected[1]
        assert meta['shortName'] == 'USD/' + expected[1]
    tz = ZoneInfo(meta['exchangeTimezoneName'])
    stamps = result['timestamp']
    assert stamps and stamps == sorted(set(stamps)), 'Duplicate/unordered timestamps'
    assert len(result['indicators']['quote']) == 1
    quotes = result['indicators']['quote'][0]
    assert all(len(quotes[k]) == len(stamps) for k in FIELDS)
    adjusted = result['indicators'].get('adjclose', [])
    assert len(adjusted) <= 1
    adj = adjusted[0]['adjclose'] if adjusted else [None] * len(stamps)
    assert len(adj) == len(stamps)
    rows = {}
    for i, stamp in enumerate(stamps):
        assert isinstance(stamp, int)
        local = datetime.fromtimestamp(stamp, tz)
        day = local.date().isoformat()
        assert day not in rows, 'Multiple bars on an exchange-local date'
        row = {k: quotes[k][i] for k in FIELDS}
        for k in FIELDS:
            v = row[k]
            if v is not None:
                assert isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                assert v >= 0 if k == 'volume' else v > 0, (day, k, v)
        complete_ohlc = all(row[k] is not None for k in ('open', 'high', 'low', 'close'))
        row['ohlc_range_consistent'] = (row['low'] <= min(row['open'], row['close'])
                                       <= max(row['open'], row['close']) <= row['high']) if complete_ohlc else None
        if adj[i] is not None:
            assert valid_price(adj[i])
        row.update(timestamp=stamp, timestamp_utc=iso_utc(stamp),
                   timestamp_exchange_local=local.isoformat(),
                   exchange_local_date=day, utc_date=iso_utc(stamp)[:10],
                   adjusted_close=adj[i])
        rows[day] = row
    return meta, rows, result.get('events') or {}


def target_row(signal, asof, rows):
    """Missing foreign bars remain missing; neighboring bars are not substitutes."""
    out = dict(signal_date=signal, asof=asof, lookup='exact_exchange_local_date',
               filled=False, price_available=False, close=None)
    row = rows.get(asof)
    if row is None:
        out.update(status='missing_bar', missing_reason='Not determined: exchange closure, suspension, or provider omission require independent evidence.',
                   exchange_holiday_confirmed=False)
        return out
    out.update(row)
    out['price_available'] = valid_price(row['close'])
    out['status'] = 'exact_bar' if out['price_available'] else 'null_close'
    out['missing_fields'] = [k for k in FIELDS if row[k] is None]
    out['ohlc_quality_passed'] = row['ohlc_range_consistent'] is True
    out['historical_close_publication_time_utc'] = None
    out['historical_close_available_by_us_cutoff_verified'] = False
    return out


def audit_chart(body, expected, targets):
    meta, rows, events = normalize_chart(body, expected)
    records = [target_row(x['date'], x['asof'], rows) for x in targets]
    missing = [x['asof'] for x in records if not x['price_available']]
    days = sorted(rows)
    all_weekdays = []
    day, last = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    while day <= last:
        if day.weekday() < 5 and day.isoformat() not in rows:
            all_weekdays.append(day.isoformat())
        day += timedelta(days=1)
    event_records = {}
    for kind, values in events.items():
        assert isinstance(values, dict)
        event_records[kind] = []
        for key, event in sorted(values.items(), key=lambda x: int(x[0])):
            stamp = event['date']
            assert int(key) == stamp
            local = datetime.fromtimestamp(stamp, ZoneInfo(expected[2]))
            event_records[kind].append(dict(provider_event=event, timestamp_utc=iso_utc(stamp), exchange_local_date=local.date().isoformat()))
    utc_lookup = {row['utc_date']: row for row in rows.values()}
    wrong_utc = []
    for target in targets:
        match = utc_lookup.get(target['asof'])
        if match is None or match['exchange_local_date'] != target['asof']:
            wrong_utc.append(dict(asof=target['asof'], naive_utc_matched_exchange_date=match['exchange_local_date'] if match else None))
    local_times = sorted({x['timestamp_exchange_local'][11:] for x in rows.values()})
    is_fx = expected[-1] == 'CURRENCY'
    return dict(
        metadata={k: meta[k] for k in ('symbol', 'currency', 'exchangeName', 'fullExchangeName', 'instrumentType', 'longName', 'shortName', 'exchangeTimezoneName', 'dataGranularity')},
        current_snapshot_fields_excluded=['gmtoffset', 'timezone', 'regularMarketTime', 'regularMarketPrice', 'currentTradingPeriod'],
        rows=len(rows), first_exchange_date=days[0], last_exchange_date=days[-1],
        all_quote_fields=list(FIELDS), raw_price_field='chart.result[0].indicators.quote[0].close',
        adjusted_price_field_not_used_for_valuation='chart.result[0].indicators.adjclose[0].adjclose',
        raw_and_adjusted_differing_rows=sum(x['close'] is not None and x['adjusted_close'] is not None and x['close'] != x['adjusted_close'] for x in rows.values()),
        local_timestamp_clock_values=local_times,
        utc_date_differs_from_exchange_date_rows=sum(x['utc_date'] != x['exchange_local_date'] for x in rows.values()),
        naive_utc_target_date_lookup_errors=wrong_utc,
        missing_weekdays_not_holiday_classified=all_weekdays,
        null_close_dates=[d for d, x in rows.items() if x['close'] is None],
        all_null_quote_dates=[d for d, x in rows.items() if all(x[k] is None for k in FIELDS)],
        inconsistent_ohlc_rows=[dict(date=d, **{k:x[k] for k in FIELDS}) for d,x in rows.items() if x['ohlc_range_consistent'] is False],
        target_ohlc_quality_failures=[x['asof'] for x in records if x.get('ohlc_range_consistent') is False],
        source_event_counts={k:len(v) for k,v in event_records.items()}, events=event_records,
        splits_in_response=len(events.get('splits', {})),
        absence_of_reported_splits_is_not_full_adjustment_certification=True,
        provider_declared_unit=(expected[1] + ' per USD, supported by USD/' + expected[1] + ' longName and quote currency') if is_fx else expected[1] + ' per locally listed share; A/H class identification requires separate issuer evidence',
        fx_direction_provider_metadata_supported=is_fx,
        fx_direction_independently_verified=False if is_fx else None,
        historical_bar_finalization_or_publication_time_available=False,
        bar_timestamp_is_not_close_publication_time=True,
        historical_us_cutoff_availability='unverified: historical bar end/publication timestamps are absent; currentTradingPeriod belongs to the retrieval-date snapshot',
        identity_unit_status='provider metadata matches request; issuer class/ratio and fully unadjusted historical-unit certification remain separate',
        target_count=len(records), available_target_close_count=len(records)-len(missing),
        missing_target_close_dates=missing, target_records=records,
        bars_on_or_after_request_exclusive_end=[d for d,x in rows.items() if x['timestamp'] >= 1609459200],
    )


def run():
    output = OUT / 'QUOTE_AUDIT.json'
    assert not output.exists(), 'Refuse to overwrite existing audit'
    planfile = OUT / 'SOURCE_PLAN.json'
    assert sha(planfile) == PLAN_SHA
    plan = read(planfile)
    pins = {str(planfile.relative_to(ROOT)): PLAN_SHA}
    for name, expected in plan['existing_input_pins'].items():
        assert sha(ROOT / name) == expected, name
        pins[name] = expected
    collector = ROOT / 'work/ptr169_collect.py'
    assert sha(collector) == plan['collector_sha256']
    pins[str(collector.relative_to(ROOT))] = plan['collector_sha256']
    requests = {x['name']: x for x in plan['requests']}
    assert set(requests) == set(RECEIPTS)
    receipts = {}
    for name, receipt_sha in RECEIPTS.items():
        receiptfile, bodyfile = OUT / (name + '.receipt.json'), OUT / name
        assert sha(receiptfile) == receipt_sha, name
        receipt = read(receiptfile)
        assert receipt['source_plan_sha256'] == PLAN_SHA
        assert receipt['status'] == 200 and receipt['url'] == requests[name]['url']
        assert receipt['source_kind'] == requests[name]['kind']
        assert receipt['bytes'] == bodyfile.stat().st_size and sha(bodyfile) == receipt['body_sha256']
        assert datetime.fromisoformat(receipt['requested_utc']) <= datetime.fromisoformat(receipt['finished_utc'])
        pins[str(receiptfile.relative_to(ROOT))] = receipt_sha
        pins[str(bodyfile.relative_to(ROOT))] = receipt['body_sha256']
        receipts[name] = receipt
    assert (OUT / 'petrochina_2015_annual.pdf').read_bytes().startswith(b'%PDF-')
    targets = read(ROOT / 'outputs/monthly_2016_extension/EARLY_RANKS.json')
    assert len(targets) == 53
    assert [x['date'] for x in targets] == sorted({x['date'] for x in targets})
    assert all(x['asof'] < x['date'] for x in targets)
    assert targets[0]['date'] == '2016-08-17' and targets[-1]['date'] == '2020-11-30'
    audits = {name: audit_chart(read(OUT / name), expected, targets) for name, expected in EXPECTED.items()}
    missing = {k: v['missing_target_close_dates'] for k,v in audits.items()}
    result = dict(
        schema='ptr169_quote_source_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Receipt integrity, quote identity/unit metadata and exact-date coverage only; no cap/pool/model/return computation.',
        source_plan_sha256=PLAN_SHA, audit_code_sha256=sha(Path(__file__)),
        runtime=dict(python=sys.version, timezone_conversion='zoneinfo IANA exchangeTimezoneName for every historical timestamp'),
        all_receipts_and_original_body_hashes_verified=True, input_sha256=pins, receipts=receipts,
        target_scope=dict(count=53, first_signal=targets[0]['date'], last_signal=targets[-1]['date'],
                          first_asof=targets[0]['asof'], last_asof=targets[-1]['asof'],
                          source_plan_text=plan['period'], source_plan_text_end_does_not_match_archived_targets=True,
                          resolution='Use exactly the 53 frozen EARLY_RANKS signal/asof pairs; no December2020 target added.'),
        instruments=audits, summary=dict(available_closes={k:v['available_target_close_count'] for k,v in audits.items()}, missing_dates=missing),
        holidays=dict(authoritative_foreign_exchange_calendar_supplied=False,
                      rule='Missing bars/null values are explicit gaps. Do not forward fill or classify them as holidays without an independent exchange calendar.'),
        existing_ptrcy_used=False,
        remaining_limits=[
            'Original Yahoo JSON is a 2026 retrieval of historical prices, not a contemporaneous data vintage.',
            'Neither bar timestamps nor currentTradingPeriod establish when historical daily closes became available relative to a US cutoff.',
            'FX direction is explicitly labelled by provider metadata, but independent source verification and historical finalization timing remain open.',
            'Share-class counts, their publication dates, and ADS conversion evidence require separate annual-report verification.',
            'Raw quote.close is distinct from adjclose; an empty 2016-2020 split event list does not certify all historical split adjustments or future corporate actions.',
            'Any OHLC range inconsistencies are retained as source-quality exceptions; positive close coverage does not certify those bars.',
            'The annual PDF was hash and file-signature verified here, not interpreted.',
        ],
        cap_calculation_ready=False, pools_computed=False, features_computed=False,
        models_computed=False, strategy_returns_computed=False, observations_modified=False,
    )
    for name, expected in pins.items():
        assert sha(ROOT / name) == expected, 'Input changed during audit: ' + name
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), sha256=sha(output), **result['summary']), ensure_ascii=False))


class SyntheticTests(unittest.TestCase):
    def body(self, stamp, close=7.0):
        return {'chart': {'error': None, 'result': [{
            'meta': dict(symbol='CNY=X', currency='CNY', exchangeTimezoneName='Europe/London', exchangeName='CCY', instrumentType='CURRENCY', dataGranularity='1d', longName='USD/CNY', shortName='USD/CNY'),
            'timestamp': [stamp], 'indicators': {'quote': [{k: [close if k != 'volume' else 0] for k in FIELDS}]}}]}}

    def test_london_summer_midnight(self):
        stamp = int(datetime(2018, 6, 27, 23, tzinfo=timezone.utc).timestamp())
        _, rows, _ = normalize_chart(self.body(stamp), EXPECTED['CNY_X.json'])
        self.assertEqual(list(rows), ['2018-06-28'])
        self.assertEqual(rows['2018-06-28']['utc_date'], '2018-06-27')

    def test_no_forward_fill(self):
        _, rows, _ = normalize_chart(self.body(1451606400), EXPECTED['CNY_X.json'])
        record = target_row('2016-01-05', '2016-01-04', rows)
        self.assertEqual(record['status'], 'missing_bar')
        self.assertIsNone(record['close'])
        self.assertFalse(record['filled'])

    def test_null_is_not_missing_bar(self):
        _, rows, _ = normalize_chart(self.body(1451606400, None), EXPECTED['CNY_X.json'])
        self.assertEqual(target_row('2016-01-04', '2016-01-01', rows)['status'], 'null_close')

    def test_wrong_currency_rejected(self):
        body = self.body(1451606400)
        body['chart']['result'][0]['meta']['currency'] = 'USD'
        with self.assertRaises(AssertionError):
            normalize_chart(body, EXPECTED['CNY_X.json'])

    def test_duplicate_timestamp_rejected(self):
        body = self.body(1451606400)
        body['chart']['result'][0]['timestamp'] *= 2
        with self.assertRaises(AssertionError):
            normalize_chart(body, EXPECTED['CNY_X.json'])

    def test_array_length_rejected(self):
        body = self.body(1451606400)
        body['chart']['result'][0]['indicators']['quote'][0]['close'] = []
        with self.assertRaises(AssertionError):
            normalize_chart(body, EXPECTED['CNY_X.json'])

    def test_invalid_price_rejected(self):
        with self.assertRaises(AssertionError):
            normalize_chart(self.body(1451606400, float('inf')), EXPECTED['CNY_X.json'])

    def test_ohlc_exception_preserved(self):
        body = self.body(1451606400)
        body['chart']['result'][0]['indicators']['quote'][0]['close'] = [8.0]
        _, rows, _ = normalize_chart(body, EXPECTED['CNY_X.json'])
        self.assertFalse(rows['2016-01-01']['ohlc_range_consistent'])
        self.assertEqual(rows['2016-01-01']['close'], 8.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    else:
        run()
