"""Synthetic-only checks. All collection uses injected in-memory responses."""
from copy import deepcopy
from datetime import datetime, timezone
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import monthly_profitability141_sources as source

CIK = '0000123456'
ACC = '0000123456-23-000001'
NOW = datetime(2026, 10, 20, 0, 30, tzinfo=timezone.utc)


def facts():
    period = dict(accn=ACC, filed='2023-02-15', form='10-K', start='2022-01-01', end='2022-12-31', val=4)
    asset = dict(period, val=10)
    del asset['start']
    return dict(cik=int(CIK), entityName='Synthetic Corporation', facts={'us-gaap': {
        'GrossProfit': {'units': {'USD': [period]}},
        'Assets': {'units': {'USD': [asset]}}}})


def encode(obj):
    return json.dumps(obj).encode()


class Response(io.BytesIO):
    def __init__(self, raw, code=200, headers=None, url=None):
        super().__init__(raw)
        self.code = code
        self.headers = Message()
        for key, val in dict({'Content-Type': 'application/json'}, **(headers or {})).items():
            self.headers[key] = val
        self.url = url or source.endpoint(CIK)

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def open(self, req, timeout):
        self.calls.append((req, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MetadataTests(unittest.TestCase):
    def inspect(self, obj=None, cutoff=None):
        return source.inspect_facts(encode(obj or facts()), CIK, cutoff)

    def test_pairs_without_values_or_factor(self):
        result = self.inspect()
        self.assertEqual(result['pair_status_counts'], {'structurally_paired_not_admitted': 1})
        pair = result['annual_gp_assets_pairs'][0]
        self.assertEqual(pair['accession'], ACC)
        self.assertFalse(result['factor_admitted'])
        self.assertFalse(result['factor_computed'])
        self.assertFalse(result['returns_computed'])
        self.assertNotIn('"val"', json.dumps(result))
        self.assertNotIn('ratio', pair)
        self.assertIn('independent audit', result['temporal_semantics'])

    def test_asof_does_not_backfill_comparative_year(self):
        self.assertEqual(self.inspect(cutoff='2023-02-14')['annual_gp_assets_pairs'], [])
        self.assertEqual(len(self.inspect(cutoff='2023-02-15')['annual_gp_assets_pairs']), 1)
        self.assertEqual(self.inspect(cutoff='2023-02-14')['tag_summaries']['GrossProfit']['units']['USD']['after_filed_cutoff'], 1)

    def test_negative_and_zero_gross_profit_are_allowed(self):
        for value in (0, -3):
            obj = facts()
            obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['val'] = value
            self.assertEqual(self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'], [])

    def test_nonpositive_and_boolean_amounts_are_flagged(self):
        for tag, value, reason in [('Assets', 0, 'nonpositive_assets'), ('Assets', -1, 'nonpositive_assets'),
                                   ('Assets', True, 'invalid_assets_amount'), ('GrossProfit', False, 'invalid_gross_profit_amount')]:
            obj = facts()
            obj['facts']['us-gaap'][tag]['units']['USD'][0]['val'] = value
            self.assertIn(reason, self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'])

    def test_cannot_join_different_vintages_currencies_or_duration_assets(self):
        for change in ({'accn': '0000123456-24-000001'}, {'filed': '2023-02-16'},
                       {'end': '2021-12-31'}, {'form': '10-Q'}, {'start': '2022-01-01'}):
            obj = facts()
            obj['facts']['us-gaap']['Assets']['units']['USD'][0].update(change)
            self.assertIn('missing_same_filing_instant_assets', self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'])
        obj = facts()
        obj['facts']['us-gaap']['Assets']['units']['EUR'] = obj['facts']['us-gaap']['Assets']['units'].pop('USD')
        self.assertIn('missing_same_filing_instant_assets', self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'])

    def test_duplicates_conflicts_and_ambiguous_durations(self):
        for tag, reason in [('GrossProfit', 'conflicting_gross_profit_values'), ('Assets', 'conflicting_assets_values')]:
            obj = facts()
            rows = obj['facts']['us-gaap'][tag]['units']['USD']
            rows.append(deepcopy(rows[0]))
            self.assertEqual(self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'], [])
            rows[1]['val'] += 1
            self.assertIn(reason, self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'])
        obj = facts()
        rows = obj['facts']['us-gaap']['GrossProfit']['units']['USD']
        rows.append(dict(rows[0], start='2022-01-02'))
        self.assertIn('ambiguous_annual_duration', self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'])

    def test_quarters_amendments_and_bad_dates_not_annual_substitutes(self):
        for change in ({'start': '2022-10-01'}, {'form': '10-Q'}, {'form': '10-K/A'},
                       {'start': 'bad'}, {'filed': '2022-12-01'}):
            obj = facts()
            obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0].update(change)
            self.assertEqual(self.inspect(obj)['annual_gp_assets_pairs'], [])
        obj = facts()
        obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]['filed'] = 'bad'
        self.assertEqual(self.inspect(obj)['tag_summaries']['GrossProfit']['units']['USD']['invalid_filed_rows'], 1)

    def test_strict_document_and_identity(self):
        for raw in (b'[]', b'<html>denied</html>', b'{"cik":123456,"facts":{},"x":NaN}',
                    b'{"cik":123456,"facts":{},"x":Infinity}', b'{"cik":123456,"cik":123456,"facts":{}}',
                    b'{"cik":true,"facts":{}}'):
            with self.assertRaises((ValueError, TypeError)):
                source.validate_facts(raw, CIK)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            source.validate_facts(encode(facts()), '0000123457')
        with self.assertRaises(ValueError):
            self.inspect(cutoff='20230215')

    def test_optional_fiscal_metadata_conflicts_and_instant_coverage(self):
        obj = facts()
        gp = obj['facts']['us-gaap']['GrossProfit']['units']['USD'][0]
        asset = obj['facts']['us-gaap']['Assets']['units']['USD'][0]
        gp.update(fp='FY', fy=2023)
        asset.update(fp='FY', fy=2023)
        self.assertEqual(self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons'], [])
        self.assertEqual(self.inspect(obj)['tag_summaries']['Assets']['units']['USD']['instant_ends'], ['2022-12-31'])
        gp['fp'] = 'Q4'
        asset['fy'] = 2022
        reasons = self.inspect(obj)['annual_gp_assets_pairs'][0]['reasons']
        self.assertIn('nonannual_fiscal_period_metadata', reasons)
        self.assertIn('conflicting_filing_fiscal_years', reasons)

    def test_original_filing_vintages_stay_separate(self):
        obj = facts()
        for tag in ('GrossProfit', 'Assets'):
            rows = obj['facts']['us-gaap'][tag]['units']['USD']
            rows.append(dict(rows[0], accn='0000123456-24-000001', filed='2024-02-15'))
        result = self.inspect(obj)
        self.assertEqual(len(result['annual_gp_assets_pairs']), 2)
        self.assertEqual(len(self.inspect(obj, '2023-12-31')['annual_gp_assets_pairs']), 1)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.plan = self.root / 'DATA_PLAN.json'
        self.plan.write_bytes(encode(dict(schema_version=1, purpose=source.PURPOSE,
            allowed_ciks=[CIK, '0000123457'], output_dir='outputs/monthly_profitability141_sources',
            source_stop_timezone='America/Los_Angeles')))
        self.digest = source.sha(self.plan.read_bytes())
        self.output = self.root / 'outputs/monthly_profitability141_sources'

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self, opener, **kwargs):
        return source.collect(self.plan, self.digest, kwargs.pop('cik', CIK), 'synthetic-agent',
                              root=self.root, opener=opener, now=kwargs.pop('now', NOW), **kwargs)

    def test_exact_raw_receipt_and_success_cache(self):
        raw = b' \n' + encode(facts()) + b'\n'
        opener = Opener(Response(raw, headers={'Content-Type': 'application/json', 'Set-Cookie': 'do-not-save'}))
        result = self.collect(opener)
        self.assertEqual(result['status'], 'saved')
        self.assertEqual((self.output / 'raw' / ('CIK' + CIK + '.json')).read_bytes(), raw)
        receipt = json.loads((self.output / 'receipts' / ('CIK' + CIK + '.json')).read_bytes())
        self.assertEqual(receipt['sha256'], source.sha(raw))
        self.assertNotIn('set-cookie', receipt['headers'])
        self.assertNotIn('synthetic-agent', json.dumps(receipt))
        self.assertEqual(opener.calls[0][1], 15)
        self.assertEqual(opener.calls[0][0].full_url, source.endpoint(CIK))
        self.assertEqual(self.collect(opener)['status'], 'reused')
        self.assertEqual(len(opener.calls), 1)

    def test_corrupt_cache_and_unfrozen_plan_never_fetch(self):
        opener = Opener(Response(encode(facts())))
        self.digest = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.collect(opener)
        self.assertEqual(opener.calls, [])
        self.digest = source.sha(self.plan.read_bytes())
        self.collect(opener)
        (self.output / 'raw' / ('CIK' + CIK + '.json')).write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.collect(opener)
        self.assertEqual(len(opener.calls), 1)

    def test_plan_user_agent_and_cache_plan_binding(self):
        opener = Opener(Response(encode(facts())))
        plan = json.loads(self.plan.read_bytes())
        plan['user_agent'] = 'required-test-agent'
        self.plan.write_bytes(encode(plan))
        self.digest = source.sha(self.plan.read_bytes())
        with self.assertRaisesRegex(ValueError, 'User-Agent'):
            self.collect(opener)
        self.assertEqual(opener.calls, [])
        plan['user_agent'] = 'synthetic-agent'
        self.plan.write_bytes(encode(plan))
        self.digest = source.sha(self.plan.read_bytes())
        self.collect(opener)
        plan['additional_note'] = 'new plan cannot silently inherit prior receipt'
        self.plan.write_bytes(encode(plan))
        self.digest = source.sha(self.plan.read_bytes())
        with self.assertRaisesRegex(ValueError, 'receipt schema'):
            self.collect(opener)
        self.assertEqual(len(opener.calls), 1)

    def test_invalid_html_and_mismatched_identity_not_success(self):
        for body in (b'<html>bad</html>', encode(dict(cik=987654, facts={}))):
            result = self.collect(Opener(Response(body)))
            self.assertEqual(result['status'], 'failed')
            self.assertFalse((self.output / 'raw').exists())
        self.assertEqual(len(list((self.output / 'failures').glob('*.body'))), 2)

    def test_wrong_content_type_and_invalid_success_cache_receipt(self):
        result = self.collect(Opener(Response(encode(facts()), headers={'Content-Type': 'text/html'})))
        self.assertEqual(result['error'], 'response_content_type_not_json')
        opener = Opener(Response(encode(facts())))
        self.collect(opener)
        receipt_path = self.output / 'receipts' / ('CIK' + CIK + '.json')
        receipt = json.loads(receipt_path.read_bytes())
        receipt['retrieved_at_utc'] = '2026-10-20T00:00:00'
        receipt_path.write_bytes(encode(receipt))
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.collect(opener)
        self.assertEqual(len(opener.calls), 1)

    def test_forbidden_stops_local_day_across_utc_midnight(self):
        before = datetime(2026, 10, 19, 23, 59, tzinfo=timezone.utc)
        opener = Opener(Response(b'forbidden', 403))
        self.assertEqual(self.collect(opener, now=before)['status'], 'failed')
        self.assertTrue((self.output / 'STOPPED_2026-10-19.json').is_file())
        for instant in (NOW, datetime(2026, 10, 20, 6, 59, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, 'stopped'):
                self.collect(opener, now=instant, cik='0000123457')
        self.assertEqual(len(opener.calls), 1)
        source.check_stops(self.output, datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc), 'America/Los_Angeles')

    def test_retry_after_delta_and_http_date_are_obeyed(self):
        for header, expected in [('172800', '2026-10-22T00:30:00+00:00'),
                                 ('Thu, 22 Oct 2026 00:30:00 GMT', '2026-10-22T00:30:00+00:00')]:
            with self.subTest(header=header):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    stop = dict(source_stop_timezone='America/Los_Angeles', local_date='2026-10-19',
                                retry_not_before_utc=source.retry_instant(header, NOW).isoformat())
                    self.assertEqual(stop['retry_not_before_utc'], expected)
                    (output / 'STOPPED_2026-10-19.json').write_bytes(encode(stop))
                    with self.assertRaisesRegex(ValueError, 'stopped'):
                        source.check_stops(output, datetime(2026, 10, 21, tzinfo=timezone.utc), 'America/Los_Angeles')
        opener = Opener(Response(b'limited', 429, {'Retry-After': '172800'}))
        self.collect(opener)
        stop = json.loads((self.output / 'STOPPED_2026-10-19.json').read_bytes())
        self.assertEqual(stop['retry_after'], '172800')
        self.assertEqual(stop['retry_not_before_utc'], '2026-10-22T00:30:00+00:00')

    def test_redirect_not_followed_and_oversize_cannot_be_success(self):
        self.assertIsNone(source.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://evil.example/'))
        result = self.collect(Opener(Response(encode(facts()), url='https://evil.example/')))
        self.assertEqual(result['status'], 'failed')
        with patch.object(source, 'MAX_BYTES', 32):
            result = self.collect(Opener(Response(b'x' * 33)))
        self.assertEqual(result['error'], 'response_exceeds_size_limit')
        bodies = list((self.output / 'failures').glob('*.body'))
        self.assertIn(32, [p.stat().st_size for p in bodies])

    def test_http_error_body_is_saved_and_timeout_does_not_leak(self):
        headers = Message()
        headers['Retry-After'] = '60'
        error = urllib.error.HTTPError(source.endpoint(CIK), 429, 'limited', headers, io.BytesIO(b'original-error'))
        self.assertEqual(self.collect(Opener(error))['status'], 'failed')
        self.assertEqual(next((self.output / 'failures').glob('*.body')).read_bytes(), b'original-error')

    def test_exclusive_install_preserves_existing_bytes(self):
        path = self.root / 'original'
        source.exclusive_bytes(path, b'a')
        with self.assertRaises(FileExistsError):
            source.exclusive_bytes(path, b'b')
        self.assertEqual(path.read_bytes(), b'a')


if __name__ == '__main__':
    unittest.main()
