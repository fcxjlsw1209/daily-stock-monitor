"""Independent source-only VVIX coverage audit; no producer imports or factors."""
import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/after159_vvix_source/'
DEST = 'work/after159_vvix_admission_review.json'
SELF = 'work/after159_vvix_admission_independent.py'
VALIDATION_SHA = '5fdadf917a444e6193198108c6ab51fd2734e30296fa719f1ca9064c0368e7aa'
SUPPLEMENT_SHA = '76a8b45255887205417854445b01f73f639288fce221b48f4cbead745749dbfe'
REVIEW_SHA = '9ad9f26e2a379e126151fbdfd143486ef7cccb061db1feebcc3a4277d2e0e17d'
SCHEMAS = {'VVIX': ['DATE', 'VVIX'], 'VIX': ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def number_status(raw):
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return 'not_numeric'
    if not number.is_finite():
        return 'nonfinite'
    return None if number > 0 else 'nonpositive'


def source_rows(name, raw):
    """Require exact raw schema and chronology; preserve literals and line numbers."""
    reader = csv.reader(io.StringIO(raw.decode('utf-8'), newline=''), strict=True)
    require(next(reader) == SCHEMAS[name], 'Unexpected CSV header')
    result = []
    for cells in reader:
        require(len(cells) == len(SCHEMAS[name]), 'CSV width')
        require(re.fullmatch(r'[0-9]{2}/[0-9]{2}/[0-9]{4}', cells[0]), 'CSV date shape')
        stamp = datetime.strptime(cells[0], '%m/%d/%Y').date().isoformat()
        require(not result or result[-1]['date'] < stamp, 'CSV duplicate or out-of-order date')
        result.append({'line': reader.line_num, 'date': stamp, 'cells': cells,
                       'value_status': {key: number_status(value.strip())
                                        for key, value in zip(SCHEMAS[name][1:], cells[1:])}})
    require(result, 'Empty source CSV')
    return result


def previous_window(calendar, signal):
    require(calendar == sorted(set(calendar)), 'Unique ordered calendar')
    require(signal in calendar, 'Signal outside calendar')
    past = [day for day in calendar if day < signal]
    require(len(past) >= 252, 'Insufficient official history')
    return past[-252:]


def window_status(rows, needed, field):
    lookup = {row['date']: row for row in rows}
    missing = [day for day in needed if day not in lookup]
    invalid = [day for day in needed if day in lookup and lookup[day]['value_status'][field] is not None]
    return missing, invalid


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.urls.append(dict(attrs).get('href'))


class Reader:
    def __init__(self):
        self.hashes = {}
    def path(self, name):
        p = Path(name)
        p = (p if p.is_absolute() else ROOT / p).resolve()
        require(not any(word in str(p) for word in ('OneDrive', 'CloudStorage', 'stock-observer-runtime')), 'Forbidden path')
        return p
    def read(self, name):
        p = self.path(name); raw = p.read_bytes(); h = digest(raw)
        require(str(p) not in self.hashes or self.hashes[str(p)] == h, 'Input changed during audit')
        self.hashes[str(p)] = h
        return raw
    def bind(self, name, expected):
        require(digest(self.read(name)) == expected, 'Hash mismatch: '+str(name))
    def json(self, name):
        def unique(pairs):
            value = {}
            for key, entry in pairs:
                require(key not in value, 'Duplicate JSON key'); value[key] = entry
            return value
        return json.loads(self.read(name), object_pairs_hook=unique)
    def refresh(self):
        for name in list(self.hashes):
            self.read(name)


class Synthetic(unittest.TestCase):
    def calendar(self):
        first = date(2010, 1, 1)
        return [(first+timedelta(days=i)).isoformat() for i in range(400)]
    def test_strictly_prior_252(self):
        cal = self.calendar(); got = previous_window(cal, cal[300])
        self.assertEqual(got, cal[48:300]); self.assertNotIn(cal[300], got)
    def test_calendar_closure_not_calendar_day(self):
        cal = self.calendar(); target = cal[300]; closure = cal.pop(100)
        got = previous_window(cal, target)
        self.assertEqual(len(got), 252); self.assertNotIn(closure, got); self.assertEqual(got[0], cal[47])
    def test_short_window_fails(self):
        cal = self.calendar()
        with self.assertRaises(ValueError): previous_window(cal, cal[251])
    def test_missing_does_not_fill(self):
        rows = source_rows('VVIX', b'DATE,VVIX\n01/01/2010,10\n01/03/2010,20\n')
        self.assertEqual(window_status(rows, ['2010-01-01', '2010-01-02'], 'VVIX'), (['2010-01-02'], []))
    def test_future_poison_not_consumed(self):
        rows = source_rows('VVIX', b'DATE,VVIX\n01/01/2010,10\n01/03/2010,NaN\n')
        self.assertEqual(window_status(rows, ['2010-01-01'], 'VVIX'), ([], []))
        self.assertEqual(window_status(rows, ['2010-01-03'], 'VVIX'), ([], ['2010-01-03']))
    def test_duplicate_and_invalid_date_rejected(self):
        for raw in (b'DATE,VVIX\n01/01/2010,10\n01/01/2010,20\n', b'DATE,VVIX\n02/30/2010,10\n'):
            with self.assertRaises(ValueError): source_rows('VVIX', raw)
    def test_invalid_numerics(self):
        self.assertEqual([number_status(s) for s in ('NaN', '-2', '0', 'bad', '2')],
                         ['nonfinite', 'nonpositive', 'nonpositive', 'not_numeric', None])


def synthetic():
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Synthetic))
    require(result.wasSuccessful(), 'Independent synthetic audit tests failed')
    return {'passed': True, 'tests_run': result.testsRun, 'source_only': True}


def review():
    own_tests = synthetic()
    rd = Reader(); require(not rd.path(DEST).exists(), 'Write-once review already exists')
    for file, expected in ((OUT+'VALIDATION.json', VALIDATION_SHA), (OUT+'SUPPLEMENT.json', SUPPLEMENT_SHA),
                           ('work/after159_vvix_source_review.json', REVIEW_SHA)):
        rd.bind(file, expected)
    validation, supplement, prior = [rd.json(p) for p in
        (OUT+'VALIDATION.json', OUT+'SUPPLEMENT.json', 'work/after159_vvix_source_review.json')]
    for pinset in (validation['input_hashes'], prior['input_hashes']):
        for file, expected in pinset.items(): rd.bind(file, expected)
    for file in (SELF, 'work/after159_vvix_install_supplement.py', 'run_local.sh', 'local_run.py', 'local.sandbox.sb', sys.executable):
        rd.read(file)
    cal = rd.json('outputs/monthly_paired_target143/CALENDAR.json')
    meta = rd.json('outputs/monthly_local_linear70/META.json')
    for file, expected in cal['provenance']['source_hashes'].items():
        rd.bind(file, expected)
        require(validation['input_hashes'].get(str(rd.path(file))) == expected, 'Calendar dependency missing from validation')
    require(len(cal['provenance']['source_hashes']) == 9, 'Calendar closure')
    plan = rd.json('work/after159_vvix_source_plan.json')
    collection = rd.json(OUT+'COLLECTION.json')
    require(collection['plan_sha256'] == validation['plan_sha256'] == prior['plan_sha256'], 'Plan linkage')
    require(len(collection['sources']) == 5 and collection['complete'] is True, 'Five complete original sources')
    for key, item in collection['sources'].items():
        rd.bind(item['file'], item['sha256'])
        require(len(rd.read(item['file'])) == item['bytes'] and item['status'] == 200 and
                item['requested_url'] == item['final_url'] == plan['sources'][key], 'Original collection identity')
    page = Anchors(); page.feed(rd.read(OUT+'historical_index_page.html').decode())
    require(all(plan['sources'][name] in page.urls for name in SCHEMAS), 'Official links missing')
    require(supplement['schema'] == 'after159_vvix_methodology_supplement_v1' and supplement['status'] == 200 and
            supplement['file'] == OUT+'filter_change_20240708.pdf' and
            supplement['headers']['content-type'].split(';')[0] == 'application/pdf', 'Supplement identity')
    rd.bind(supplement['file'], supplement['sha256'])
    require(supplement['sha256'] == 'a52428a70355692ab5e37fb88f8a5dd9e7f8689dbb445c9a5d69b9d047a9049b' and
            len(rd.read(supplement['file'])) == supplement['bytes'] and rd.read(supplement['file']).startswith(b'%PDF-'), 'Supplement raw PDF')
    log_file = '/private/tmp/after159_vvix_validate_tests.log'
    log = rd.read(log_file).decode()
    require('\nOK\n' in log and re.findall(r'Ran (\d+) tests in', log) == ['21'], 'Producer test log')
    markers = [line.split('=', 1)[1] for line in log.splitlines() if line.startswith('SYNTHETIC_TEST_RECEIPT=')]
    require(len(markers) == 1 and json.loads(markers[0]) == validation['synthetic_tests'], 'Test receipt linkage')
    test_result = validation['synthetic_tests']
    require(all(test_result[k] is True for k in ('passed', 'synthetic_only', 'hashes_unchanged')), 'Producer synthetic failure')
    for file, expected in test_result['implementation_hashes'].items(): rd.bind(file, expected)
    require(set(test_result['implementation_hashes']) == {'work/after159_vvix_validate.py', 'work/test_after159_vvix_validate.py'}, 'Producer code/test closure')
    dates, calendar = meta['dates'], cal['sessions']
    require(len(dates) == len(meta['symbols']) == len(meta['ends']) == 122 and all(len(x) == 49 for x in meta['symbols']) and
            dates == sorted(set(dates)) and dates[0] == '2016-08-17' and dates[-1] == '2026-08-31', 'Original axis')
    require(all(date.fromisoformat(d).isoformat() == d for d in calendar), 'Real calendar dates')
    raw = {name: source_rows(name, rd.read(OUT+name+'_History.csv')) for name in SCHEMAS}
    all_needed = set(); windows = []
    require(len(validation['windows']) == 122, 'All window records required')
    initial = validation['windows'][0]
    require(initial == {'first_required_date': None, 'initial_partial': True, 'last_required_date': None,
        'passed': True, 'required_sessions': 0, 'signal_date': dates[0], 'signal_index': 0,
        'sources': {}, 'status': 'neutral_initial_partial'}, 'Initial neutral record differs')
    for index, signal in enumerate(dates[1:], start=1):
        needed = previous_window(calendar, signal); all_needed.update(needed)
        expected = {'available_calendar_sessions': 252, 'first_required_date': needed[0], 'initial_partial': False,
            'last_required_date': needed[-1], 'passed': True, 'required_sessions': 252,
            'required_sessions_sha256': digest(json.dumps(needed, separators=(',', ':')).encode()),
            'signal_date': signal, 'signal_index': index, 'sources': {}}
        for name in SCHEMAS:
            missing, bad = window_status(raw[name], needed, 'VVIX' if name == 'VVIX' else 'CLOSE')
            require(not missing and not bad, 'Missing/invalid original required source rows')
            expected['sources'][name] = {'invalid_required_rows': [], 'missing_dates': [], 'passed': True,
                                         'valid_required_sessions': 252}
        require(expected == validation['windows'][index], 'Window mismatch at '+signal)
        windows.append({'signal_index': index, 'signal_date': signal, 'count': 252,
                        'first_required_date': needed[0], 'last_required_date': needed[-1],
                        'required_sessions_sha256': expected['required_sessions_sha256']})
    summary = {'signals_checked': 122, 'noninitial_signals': 121, 'first_required_date': min(all_needed),
               'last_required_date': max(all_needed), 'distinct_required_sessions': len(all_needed)}
    require(all(validation[key] == value for key, value in summary.items()), 'Coverage summary mismatch')
    source_summary = {}; calset = set(calendar)
    for name, rows in raw.items():
        got = validation['sources'][name]; field = 'VVIX' if name == 'VVIX' else 'CLOSE'
        noncalendar = [r for r in rows if calendar[0] <= r['date'] <= calendar[-1] and r['date'] not in calset]
        outside = [r['date'] for r in rows if not calendar[0] <= r['date'] <= calendar[-1]]
        future = [r for r in rows if r['date'] > max(all_needed)]
        invalid = [r for r in rows if r['value_status'][field] is not None]
        expected = {'header': SCHEMAS[name], 'row_count': len(rows), 'first_date': rows[0]['date'], 'last_date': rows[-1]['date'],
            'schema_and_chronology_passed': True, 'schema_errors': [], 'order_errors': [], 'duplicate_dates': [],
            'noncalendar_rows': noncalendar, 'outside_calendar_extent_dates': outside,
            'future_beyond_last_required': {'row_count': len(future), 'first_date': min((r['date'] for r in future), default=None),
                'last_date': max((r['date'] for r in future), default=None), 'invalid_rows': [r for r in future if r in invalid]},
            'invalid_consumed_rows': [r for r in invalid if r['date'] in all_needed],
            'invalid_unconsumed_rows': [r for r in invalid if r['date'] not in all_needed],
            'unused_OHLC_issues': [r for r in rows if name == 'VIX' and any(r['value_status'][k] is not None for k in ('OPEN', 'HIGH', 'LOW'))],
            'covered_noninitial_signals': 121}
        require(expected == got, 'Source diagnostics mismatch: '+name)
        require(not all_needed.intersection(r['date'] for r in noncalendar), 'Noncalendar date consumed')
        source_summary[name] = {key: expected[key] for key in ('row_count', 'first_date', 'last_date', 'covered_noninitial_signals')}
        source_summary[name].update(noncalendar_count=len(noncalendar), noncalendar_dates=[r['date'] for r in noncalendar],
            outside_calendar_count=len(outside), future_rows=len(future), noncalendar_dates_consumed=0)
    require(validation['passed'] is validation['coverage_passed'] is validation['schema_and_chronology_passed'] is True, 'Validation did not pass')
    require(validation['source_admission_changed'] is False and all(validation[k] == 0 for k in ('new_features', 'new_ranks', 'new_returns', 'new_orders')), 'Source-only boundary')
    require(not any(name in sys.modules for name in ('after159_vvix_validate', 'after159_vvix_collect')), 'Producer import')
    rd.refresh()
    report = {'schema': 'after159_vvix_independent_admission_review_v1', 'reviewed_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'conditional_research_sources_ready': True, 'strict_pit_vintage': False, 'strategy_admitted': False,
        'validation_sha256': VALIDATION_SHA, 'source_review_sha256': REVIEW_SHA, 'supplement_sha256': SUPPLEMENT_SHA,
        'producer_modules_imported': False, 'input_hashes': dict(sorted(rd.hashes.items())), 'differences': [],
        'checks': {'validation_all_input_hashes_verified': True, 'source_review_all_input_hashes_verified': True,
            'original_five_sources_unchanged': True, 'current_producer_code_tests_and_21_test_log_bound': True,
            'calendar_nine_dependency_closure': True, 'all121_exact_prior252_windows_recomputed': True,
            'initial_partial_neutral': True, 'two_series_all_required_dates_unique_positive_finite': True,
            'all_raw_row_counts_and_diagnostics_recomputed': True, 'noncalendar_rows_retained_not_consumed': True,
            'supplement_and_original_pdf_same_expected_bytes': True, 'zero_feature_rank_return_order_calculations': True},
        'producer_synthetic_tests': dict(test_result, independently_rerun=False, evidence='Pinned successful log read and code/test hashes rechecked'),
        'independent_synthetic_tests': own_tests, 'summary': summary, 'sources': source_summary, 'windows': windows,
        'known_source_limitations': [
            'This is exact source-window coverage under a declared previous-XNYS-session rule, not a historical as-published-vintage certification.',
            '2012-03-14 issuer launch announcement conflicts with current methodology table launch 2014-02-01; both precede the earliest consumed date.',
            'Current VVIX dissemination extends to 16:15 ET; no same-date index close may enter a 16:00 equity decision. Exact historical daily publication latency remains unverified.',
            'Cboe notice C2024052400 dated 2024-05-24 implements VVIX RTH filtering on 2024-07-08: 0.5 volatility points, 30 seconds. Original prior source review remains unchanged; this review now binds the supplemental PDF.',
            'The supplement records a later receipt-writing time and HTTP server Date, not an invented original client retrieval timestamp.',
            'Current CSVs do not establish absence of historical revisions or complete methodology-change history.'],
        'activity': {'new_features': 0, 'new_ranks': 0, 'new_returns': 0, 'new_orders': 0, 'new_strategy_paths': 0,
                     'existing_inputs_modified': False, 'observer_access': False}}
    with rd.path(DEST).open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({'passed': True, 'path': str(rd.path(DEST)), 'sha256': digest(rd.path(DEST).read_bytes()),
                      'input_hashes': len(rd.hashes), 'summary': summary, 'source_summary': source_summary}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=['test', 'review'])
    if parser.parse_args().phase == 'test':
        print(json.dumps(synthetic()))
    else:
        review()
