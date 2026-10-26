"""Pure synthetic source histories/HTML/receipts; never consumes collected files."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import after159_vvix_validate as v


def sessions(count=500):
    values = []; day = date(2014, 1, 1)
    while len(values) < count:
        if day.weekday() < 5:
            values.append(day.isoformat())
        day += timedelta(days=1)
    return values


def source(name, days, replacements=None):
    replacements = replacements or {}
    rows = [','.join(v.HEADERS[name])]
    for day in days:
        numbers = replacements.get(day, ['15'] if name == 'VVIX' else ['14', '16', '13', '15'])
        rows.append(','.join([date.fromisoformat(day).strftime('%m/%d/%Y')]+numbers))
    return ('\n'.join(rows)+'\n').encode()


def fixture(replacements=None, omitted=None):
    cal = sessions(); signals = [cal[270], cal[300], cal[330]]
    data = {name: v.parse_csv(name, source(name, [d for d in cal if d not in (omitted or {}).get(name, [])],
                                        (replacements or {}).get(name))) for name in v.HEADERS}
    return cal, signals, data


def covered(cal, signals, data):
    return v.coverage(data, cal, signals, initial=signals[0])


def collection_fixture():
    urls = {k: 'https://example.test/'+f for k, f in v.FILENAMES.items()}
    plan = dict(schema='after159_vvix_source_collection_v1', output_directory=v.OUT,
                fixed_scope=['VVIX', 'VIX'], sources=urls)
    bodies = {v.OUT+'/'+f: b'%PDF-1.7\n' if k == 'methodology' else (k+' raw').encode()
              for k, f in v.FILENAMES.items()}
    records = {k: dict(requested_url=urls[k], final_url=urls[k], status=200,
        file=v.OUT+'/'+f, bytes=len(bodies[v.OUT+'/'+f]), sha256=v.sha_bytes(bodies[v.OUT+'/'+f]),
        headers={'Content-Type': v.MIMES[k]+'; charset=UTF-8'}, retrieved_utc='2026-10-21T17:00:00+00:00')
        for k, f in v.FILENAMES.items()}
    return plan, dict(complete=True, plan_sha256='a'*64, sources=records), bodies


class CSVTests(unittest.TestCase):
    def test_exact_observed_headers_only(self):
        for name, raw in [('VVIX', b'DATE,CLOSE\n01/01/2015,2\n'),
                          ('VIX', b'DATE,CLOSE,OPEN,HIGH,LOW\n01/01/2015,1,2,3,4\n')]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'header'):
                v.parse_csv(name, raw)
    def test_invalid_real_date_and_column_count_are_preserved(self):
        data = v.parse_csv('VVIX', b'DATE,VVIX\n02/30/2015,5\n03/02/2015,5,6\n03/03/2015,5\n')
        self.assertEqual([r['reason'] for r in data['schema_errors']], ['invalid_date', 'column_count'])
        self.assertEqual(data['schema_errors'][0]['cells'], ['02/30/2015', '5'])
        self.assertFalse(data['schema_and_chronology_passed'])
    def test_duplicates_and_reverse_dates_rejected_without_overwrite(self):
        data = v.parse_csv('VVIX', b'DATE,VVIX\n01/02/2015,5\n01/01/2015,6\n01/02/2015,7\n')
        self.assertEqual(len(data['rows']), 3)
        self.assertEqual(data['duplicate_dates'], ['2015-01-02'])
        self.assertEqual(len(data['order_errors']), 1)
        self.assertFalse(data['schema_and_chronology_passed'])
    def test_all_bad_numeric_kinds_are_distinguished(self):
        self.assertEqual(v.numeric_status('0'), 'nonpositive')
        self.assertEqual(v.numeric_status('-1'), 'nonpositive')
        self.assertEqual(v.numeric_status('NaN'), 'nonfinite')
        self.assertEqual(v.numeric_status('Infinity'), 'nonfinite')
        self.assertEqual(v.numeric_status(''), 'not_numeric')
        self.assertEqual(v.numeric_status('unknown'), 'not_numeric')
        self.assertIsNone(v.numeric_status(' 0.00001 '))


class CoverageTests(unittest.TestCase):
    def test_exact_previous252_endpoint_and_full_no_mutation(self):
        cal, signals, data = fixture(); before = deepcopy(data)
        out = covered(cal, signals, data)
        self.assertTrue(out['coverage_passed'])
        for i in (1, 2):
            t = cal.index(signals[i]); row = out['windows'][i]
            self.assertEqual(row['first_required_date'], cal[t-252])
            self.assertEqual(row['last_required_date'], cal[t-1])
            self.assertEqual(row['sources']['VVIX']['valid_required_sessions'], 252)
        self.assertEqual(data, before)
    def test_current_signal_observation_is_not_used_in_its_window(self):
        cal, signals, data = fixture()
        signals = signals[:2]
        data['VVIX'] = v.parse_csv('VVIX', source('VVIX', cal, {signals[-1]: ['NaN']}))
        out = covered(cal, signals, data)
        self.assertTrue(out['coverage_passed'])
        self.assertEqual(out['sources']['VVIX']['invalid_consumed_rows'], [])
    def test_missing_previous_session_does_not_use_latest_present(self):
        cal, signals, data = fixture(); missing = cal[299]
        data['VIX'] = v.parse_csv('VIX', source('VIX', [d for d in cal if d != missing]))
        out = covered(cal, signals, data)
        self.assertFalse(out['coverage_passed'])
        self.assertEqual(out['windows'][1]['last_required_date'], missing)
        self.assertEqual(out['windows'][1]['sources']['VIX']['missing_dates'], [missing])
    def test_invalid_required_close_fails_without_affecting_other_series(self):
        cal, signals, data = fixture(); bad = cal[299]
        data['VVIX'] = v.parse_csv('VVIX', source('VVIX', cal, {bad: ['0']}))
        out = covered(cal, signals, data)
        self.assertFalse(out['windows'][1]['sources']['VVIX']['passed'])
        self.assertTrue(out['windows'][1]['sources']['VIX']['passed'])
        self.assertEqual(out['windows'][1]['sources']['VVIX']['invalid_required_rows'][0]['reason'], 'nonpositive')
    def test_unused_ohlc_bad_values_are_diagnostic_not_consumed(self):
        cal, signals, data = fixture()
        data['VIX'] = v.parse_csv('VIX', source('VIX', cal, {cal[299]: ['NaN', '0', '-1', '15']}))
        out = covered(cal, signals, data)
        self.assertTrue(out['coverage_passed'])
        self.assertEqual(len(out['sources']['VIX']['unused_OHLC_issues']), 1)
    def test_future_poison_changes_diagnostics_not_historical_windows(self):
        cal, signals, data = fixture(); baseline = covered(cal, signals, data)
        data['VVIX'] = v.parse_csv('VVIX', source('VVIX', cal, {cal[400]: ['not observed']}))
        altered = covered(cal, signals, data)
        self.assertEqual(altered['windows'], baseline['windows'])
        self.assertTrue(altered['coverage_passed'])
        self.assertEqual(altered['sources']['VVIX']['future_beyond_last_required']['invalid_rows'][0]['date'], cal[400])
    def test_noncalendar_row_is_retained_but_does_not_fill_a_gap(self):
        cal, signals, data = fixture(); saturday = '2015-02-21'
        self.assertNotIn(saturday, cal)
        absent = cal[299]
        days = sorted(set(cal+[saturday])-{absent})
        data['VVIX'] = v.parse_csv('VVIX', source('VVIX', days))
        out = covered(cal, signals, data)
        self.assertEqual(out['sources']['VVIX']['noncalendar_rows'][0]['date'], saturday)
        self.assertEqual(out['windows'][1]['sources']['VVIX']['missing_dates'], [absent])
        self.assertFalse(out['coverage_passed'])
    def test_calendar_closure_never_counts_as_an_official_session(self):
        cal, signals, data = fixture(); closure = cal[200]
        cal.remove(closure)
        out = covered(cal, signals, data)
        self.assertTrue(out['coverage_passed'])
        self.assertEqual(out['windows'][1]['first_required_date'], cal[cal.index(signals[1])-252])
        self.assertEqual(out['sources']['VIX']['noncalendar_rows'][0]['date'], closure)
    def test_initial_alone_requires_no_history(self):
        cal, signals, data = fixture(); cal = cal[270:]
        out = covered(cal, signals[:1], data)
        self.assertTrue(out['coverage_passed'])
        self.assertEqual(out['distinct_required_sessions'], 0)
        self.assertEqual(out['windows'][0]['required_sessions'], 0)
        self.assertIsNone(out['windows'][0]['last_required_date'])
    def test_short_warmup_fails_not_a_shorter_window(self):
        cal, signals, data = fixture(); cal = cal[100:]
        out = covered(cal, signals, data)
        self.assertFalse(out['windows'][1]['passed'])
        self.assertEqual(out['windows'][1]['required_sessions'], 252)
        self.assertEqual(out['windows'][1]['available_calendar_sessions'], 200)
    def test_future_duplicate_does_not_change_coverage_but_fails_global_chronology(self):
        cal, signals, data = fixture()
        future = cal[400]
        raw = source('VVIX', cal)
        line = source('VVIX', [future]).decode().splitlines()[1]+'\n'
        data['VVIX'] = v.parse_csv('VVIX', raw+line.encode())
        out = covered(cal, signals, data)
        self.assertTrue(out['coverage_passed'])
        self.assertFalse(out['schema_and_chronology_passed'])
        self.assertEqual(out['sources']['VVIX']['duplicate_dates'], [future])
    def test_non_session_signal_or_repeated_calendar_fails(self):
        cal, signals, data = fixture()
        with self.assertRaises(ValueError):v.coverage(data, cal, [signals[0], '2015-02-21'], initial=signals[0])
        with self.assertRaises(ValueError):covered(cal+[cal[-1]], signals, data)


class ProvenanceTests(unittest.TestCase):
    def test_collection_exact_five_sources_and_no_mutation(self):
        plan, receipt, bodies = collection_fixture(); saved = deepcopy(receipt)
        self.assertEqual(set(v.collection_records(plan, receipt, 'a'*64, bodies.__getitem__)), set(v.FILENAMES))
        self.assertEqual(receipt, saved)
    def test_bad_status_url_mime_hash_or_scope_fails(self):
        for fault in ('status', 'url', 'mime', 'hash', 'bytes', 'scope', 'plan', 'complete', 'stop', 'pdf'):
            plan, receipt, bodies = collection_fixture()
            r = receipt['sources']['VVIX']
            if fault == 'status':r['status'] = 429
            elif fault == 'url':r['final_url'] += '?other=1'
            elif fault == 'mime':r['headers']['Content-Type'] = 'text/html'
            elif fault == 'hash':r['sha256'] = '0'*64
            elif fault == 'bytes':r['bytes'] += 1
            elif fault == 'scope':del receipt['sources']['VIX']
            elif fault == 'plan':receipt['plan_sha256'] = 'b'*64
            elif fault == 'complete':receipt['complete'] = False
            elif fault == 'stop':receipt['stop_for_429'] = True
            else:
                r = receipt['sources']['methodology']; bodies[r['file']] = b'not pdf'
                r.update(bytes=7, sha256=v.sha_bytes(b'not pdf'))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                v.collection_records(plan, receipt, 'a'*64, bodies.__getitem__)
    def test_anchor_href_required_url_text_does_not_count(self):
        urls = dict(VVIX='https://x/VVIX.csv', VIX='https://x/VIX.csv')
        good = '\n'.join('<a href="'+url+'">history</a>' for url in urls.values()).encode()
        self.assertEqual(set(v.official_links(good, urls)), set(urls))
        with self.assertRaises(ValueError):v.official_links(str(urls).encode(), urls)
        with self.assertRaises(ValueError):v.official_links(good.replace(b'VVIX.csv', b'VVIX-other.csv'), urls)
    def test_launch_uses_publication_not_retrieval_or_modified(self):
        obj = {'@type': 'NewsArticle', 'mainEntityOfPage': {'@id': 'https://x/launch'},
            'headline': 'CBOE Introduces VIX of VIX', 'datePublished': '2012-03-14T08:16:00-04:00',
            'dateModified': '2026-10-21T10:00:00-04:00', 'description': 'CBOE began publishing values for VVIX'}
        raw = lambda: ('<script type="application/ld+json">'+json.dumps(obj)+'</script>').encode()
        record = v.launch_evidence(raw(), 'https://x/launch', '2016-08-17')
        self.assertEqual(record['datePublished'], obj['datePublished'])
        self.assertEqual(record['dateModified'], obj['dateModified'])
        obj['datePublished'] = '2026-01-01T00:00:00Z'
        with self.assertRaises(ValueError):v.launch_evidence(raw(), 'https://x/launch', '2016-08-17')
    def test_reader_detects_changed_source_and_forbidden_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source'; path.write_text('original')
            reader = v.Reader(tmp); reader.bytes(path); path.write_text('changed')
            with self.assertRaises(ValueError):reader.refresh()
        for path in ('/tmp/OneDrive/test', '/tmp/stock-observer-runtime/test'):
            with self.assertRaises(ValueError):v.Reader('/tmp').path(path)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    paths = (v.SELF, v.TEST)
    before = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    result = unittest.main(exit=False).result
    after = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    passed = result.wasSuccessful() and before == after
    if passed:
        print('SYNTHETIC_TEST_RECEIPT='+json.dumps(dict(passed=True, synthetic_only=True,
            hashes_unchanged=True, tests_run=result.testsRun, implementation_hashes=after), sort_keys=True))
    sys.exit(0 if passed else 1)
