"""Independent offline audit of the single missing PTR A-share quotation.

No producer imports; probe by default, exclusive-create the review with --write.
This does not fill prices or calculate market caps, pool membership or returns.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_ptr_pool_source169'
OUT = BASE / 'missing_quote'
PINS = {}
FIELDS = ('open', 'close', 'high', 'low')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path, expected=None):
    path = Path(path)
    actual = sha(path)
    assert expected is None or actual == expected, str(path)
    PINS[str(path.relative_to(ROOT))] = actual
    return path


def read(path, expected=None):
    return json.loads(track(path, expected).read_text())


def valid_ohlc(row):
    assert all(math.isfinite(row[k]) and row[k] > 0 for k in FIELDS)
    assert row['low'] <= min(row['open'], row['close'])
    assert row['high'] >= max(row['open'], row['close'])


def tencent_rows(body):
    assert body['code'] == 0 and body['msg'] == ''
    assert set(body['data']) == {'sh601857'}
    data = body['data']['sh601857']
    assert 'day' in data and not {'qfqday', 'hfqday'}.intersection(data)
    rows = {}
    for bar in data['day']:
        assert len(bar) == 6
        date = datetime.strptime(bar[0], '%Y-%m-%d').date().isoformat()
        assert date == bar[0] and date not in rows
        assert '2019-04-22' <= date <= '2019-04-30'
        row = dict(zip(FIELDS, map(float, bar[1:5])))
        valid_ohlc(row)
        rows[date] = row
    assert list(rows) == sorted(rows)
    assert '2019-04-29' in rows
    return rows


def negative_controls(body, original):
    changed = copy.deepcopy(body)
    changed['data']['sh601857']['qt'] = {'untrusted_current_price': 999999}
    assert tencent_rows(changed) == original
    cases = []
    adjusted = copy.deepcopy(body)
    adjusted['data']['sh601857']['qfqday'] = adjusted['data']['sh601857'].pop('day')
    cases.append(adjusted)
    missing = copy.deepcopy(body)
    missing['data']['sh601857']['day'] = [x for x in missing['data']['sh601857']['day'] if x[0] != '2019-04-29']
    cases.append(missing)
    wrong = copy.deepcopy(body)
    wrong['data']['sh600000'] = wrong['data'].pop('sh601857')
    cases.append(wrong)
    for case in cases:
        try:
            tencent_rows(case)
        except AssertionError:
            continue
        raise AssertionError('Expected parser rejection')


def main(write=False):
    result = read(OUT/'RESOLUTION.json', '1da1dfd4778b4ea355f523b33e9ed22508fff1518afc2c1f6401505a9aae4e54')
    for name, expected in result['input_sha256'].items():
        track(ROOT/name, expected)
    track(ROOT/'work/ptr169_quote_resolve.py', result['code_sha256'])
    plan = read(OUT/'PLAN.json')
    extraction = read(OUT/'EXTRACTION.json')
    for name, expected in extraction['source_and_output_sha256'].items():
        track(OUT/name, expected)
    track(ROOT/'work/ptr169_missing_quote_sources.py', plan['code_sha256'])
    track(ROOT/'work/ptr169_missing_quote_audit.py', extraction['code_sha256'])
    track(BASE/'QUOTE_AUDIT.json', plan['quote_audit_sha256'])
    tplan = read(OUT/'TENCENT_PLAN.json')
    track(ROOT/'work/ptr169_tencent_quote.py', tplan['code_sha256'])
    assert tplan['prior_extraction_sha256'] == sha(OUT/'EXTRACTION.json')
    for name, prior, url in (
            ('tencent_201904', 'TENCENT_PLAN.json', tplan['url']),
            ('cjsc_20190430.pdf', 'PLAN.json', plan['sources']['cjsc_20190430.pdf'])):
        receipt = read(OUT/(name+'.receipt.json'))
        body = OUT/(name+'.json') if name == 'tencent_201904' else OUT/name
        assert receipt['status'] == 200 and receipt['body_sha256'] == sha(body)
        assert receipt['bytes'] == body.stat().st_size
        assert receipt['url'] == receipt['final_url'] == url
        assert receipt['plan_sha256'] == sha(OUT/prior)
    assert tplan['url'].endswith('param=sh601857,day,2019-04-22,2019-04-30,20,')
    raw = read(OUT/'tencent_201904.json')
    rows = tencent_rows(raw)
    assert len(rows) == 7
    negative_controls(raw, rows)
    assert rows['2019-04-29'] == result['ohlc'] == dict(open=7.41, close=7.4, high=7.46, low=7.37)

    ypath = BASE/'601857.SS.json'
    yreceipt = read(BASE/'601857.SS.json.receipt.json')
    assert yreceipt['status'] == 200 and yreceipt['body_sha256'] == sha(ypath)
    assert yreceipt['bytes'] == ypath.stat().st_size
    track(BASE/'SOURCE_PLAN.json', yreceipt['source_plan_sha256'])
    chart = read(ypath)['chart']
    assert chart['error'] is None and len(chart['result']) == 1
    y = chart['result'][0]
    assert y['meta']['symbol'] == '601857.SS' and y['meta']['currency'] == 'CNY'
    assert y['meta']['exchangeTimezoneName'] == 'Asia/Shanghai'
    quote, = y['indicators']['quote']
    assert all(len(quote[k]) == len(y['timestamp']) for k in FIELDS)
    all_dates, valid, outside_scope_quality_issues = {}, {}, []
    for i, stamp in enumerate(y['timestamp']):
        date = datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).date().isoformat()
        assert date not in all_dates
        row = {k: quote[k][i] for k in FIELDS}
        all_dates[date] = row
        if all(row[k] is not None for k in FIELDS):
            try:
                valid_ohlc(row)
            except AssertionError:
                assert date not in rows, ('Scoped Yahoo bar invalid', date, row)
                outside_scope_quality_issues.append(dict(date=date, ohlc=row))
                continue
            valid[date] = row
    overlap, missing = [], []
    for date, row in rows.items():
        if date not in valid:
            missing.append(dict(date=date, raw_timestamp_present=date in all_dates,
                                raw_ohlc=all_dates.get(date)))
            continue
        differences = {k: abs(row[k]-valid[date][k]) for k in FIELDS}
        assert max(differences.values()) < 0.00001
        overlap.append(dict(date=date, tencent=row, yahoo=valid[date], absolute_differences=differences))
    assert [r['date'] for r in overlap] == ['2019-04-22','2019-04-23','2019-04-24','2019-04-25','2019-04-26']
    assert [r['date'] for r in missing] == result['yahoo_missing_in_probe'] == ['2019-04-29','2019-04-30']
    assert [{k:r[k] for k in ('date','absolute_differences')} for r in overlap] == result['five_nearby_OHLC_rows_match_yahoo']

    runtime = read(ROOT/'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json', extraction['runtime_sha256'])
    for name, expected in runtime['pdf_runtime_hashes'].items():
        assert sha(name) == expected
    sys.path.append(runtime['pdf_runtime']['bundle_site_packages'])
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument((OUT/'cjsc_20190430.pdf').read_bytes())
    fresh = {}
    for number in (1,7,8):
        text = pdf[number-1].get_textpage().get_text_range()
        path = track(OUT/('cjsc_page%02d.txt' % number))
        track(OUT/('cjsc_page%02d.png' % number))
        assert ' '.join(text.split()) == ' '.join(path.read_text().split())
        fresh[number] = re.sub(r'\s+', '', text)
    pdf.close()
    assert '2019/04/30' in fresh[1] and '昨日（4月29日）' in fresh[1]
    assert '港股（港元）' in fresh[7] and 'A股（人民币）' in fresh[7]
    assert '表6：A+H股相对表现' in fresh[7]
    assert '中国石油股份0857.HK601857.SH5.067.4041.321.70' in fresh[8]
    assert '资料来源：长江证券、Wind' in fresh[7]
    report = dict(
        schema='ptr169_missing_quote_independent_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        verdict='CORROBORATED_FOR_CONDITIONAL_HISTORICAL_REPLAY_NOT_PIT_VINTAGE',
        code_sha256=sha(__file__), input_sha256=PINS, pdf_runtime_sha256=runtime['pdf_runtime_hashes'],
        method='Fresh original PDF extraction, actual visual inspection of pages 1/7/8, independent raw daily-bar parser and named-field Yahoo comparison; no producer imports.',
        quote=dict(instrument='601857.SH', share_class='A', currency='CNY', date='2019-04-29',
                   ohlc=rows['2019-04-29'], raw_tencent_row=raw['data']['sh601857']['day'][5]),
        nearby_comparison=dict(tolerance=0.00001, all_five_pass=True, rows=overlap, missing_yahoo=missing,
                               initial_six_overlap_expectation_corrected_explicitly=True, april30_not_admitted=True),
        outside_scope_yahoo_quality_issues=outside_scope_quality_issues,
        broker_evidence=dict(report_date='2019-04-30', report_page=1, discussed_prior_session='2019-04-29',
                             header_page=7, row_page=8, table='6 A+H relative performance',
                             A_CNY=7.40, H_HKD=5.06, source='Changjiang Securities / Wind',
                             row_date_explicit=False, row_date_inferred_from_report_context=True,
                             independent_visual_review_complete=True),
        handling=dict(raw_day_only=True, adjusted_series_rejected=True, current_qt_ignored=True,
                      volume_not_admitted=True, missing_date_cannot_be_supplied_by_current_qt=True),
        limitations=[
            'Both API bodies were retrieved in 2026, not preserved 2019 API vintages; agreement supports transcription and units, not historical source availability.',
            'The broker table supplies a matching A-share close, not independent open/high/low evidence for April29.',
            'The table row has no individual date; April29 follows the Apr30 report date and prior-session context. Exact historical publication timestamp is not certified.',
            'The Apr30 morning discussion is consistent with publication after the Apr29 US close; it must not be used as evidence available before that close.',
            'Nearby named-field OHLC agreement supports the Tencent array order and nominal price basis. This is not a full corporate-action or vendor revision audit.',
            'An initial probe applying range validation to every historical Yahoo bar stopped on the pre-existing January7 2019 range inconsistency. It is retained below as outside this Apr22-Apr30 verification scope, not corrected or admitted.',
            'Sina redirected to a current page and remains rejected; neither current quote metadata nor present share counts are evidence for this historical bar.',
            'Downstream use is limited to an explicitly registered 2026-data-version conditional historical replay. No existing price cache is patched by this audit.'
        ],
        checks=dict(source_receipts_and_hashes=True, original_pdf_reextracted_pages=3, visual_pages=3,
                    negative_parser_controls=3, current_qt_invariance=True),
        no_network=True, existing_files_modified=False, price_cache_filled=False,
        market_caps_pools_or_returns_computed=False)
    for name, expected in PINS.items():
        assert sha(ROOT/name) == expected
    if write:
        with (OUT/'INDEPENDENT_REVIEW.json').open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(verdict=report['verdict'], overlap=len(overlap), missing_yahoo=missing,
                          input_files_verified=len(PINS), report_written=write)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)
