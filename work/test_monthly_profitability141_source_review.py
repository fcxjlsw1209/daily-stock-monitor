"""Independent synthetic-only source review; no network or market inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import monthly_profitability141_sources as source

CIK = '0000320193'
OTHER_CIK = '0000789019'
ACC = '0000320193-20-000096'


def row(value, *, instant=False, **changes):
    out = dict(end='2019-09-28', val=value, accn=ACC, filed='2020-01-31',
               form='10-K', fy=2019, fp='FY')
    if not instant:
        out['start'] = '2018-09-30'
    out.update(changes)
    return out


def payload(gross=None, assets=None, *, asset_unit='USD'):
    if gross is None:
        gross = [row(10)]
    if assets is None:
        assets = [row(100, instant=True)]
    return dict(cik=320193, entityName='Synthetic Review Issuer', facts={'us-gaap': {
        'GrossProfit': dict(units={'USD': gross}),
        'Assets': dict(units={asset_unit: assets})}})


def raw(obj):
    return json.dumps(obj, indent=3).encode() + b'  \n'


def inspect(obj, cutoff=None):
    return source.inspect_facts(raw(obj), CIK, cutoff)


class Response:
    def __init__(self, body, code=200, headers=None, url=None):
        self.body = body
        self.code = code
        self.headers = headers or {'Content-Type': 'application/json'}
        self.url = url or source.endpoint(CIK)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def geturl(self):
        return self.url
    def read(self, size):
        return self.body[:size]


class Opener:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        return self.response


class AnnualPairReview(unittest.TestCase):
    def test_pair_is_metadata_only_and_not_admission(self):
        report = inspect(payload())
        pair = report['annual_gp_assets_pairs'][0]
        self.assertEqual(pair['status'], 'structurally_paired_not_admitted')
        self.assertFalse(report['factor_computed'])
        self.assertFalse(report['factor_admitted'])
        self.assertFalse(report['returns_computed'])
        serialized = json.dumps(report)
        self.assertNotIn('"val"', serialized)
        self.assertNotIn('profitability_ratio', serialized)
        self.assertIn('no factor admission', report['temporal_semantics'].lower())

    def test_negative_and_zero_gross_profit_remain_structural(self):
        for value in [-10, 0]:
            with self.subTest(value=value):
                pair = inspect(payload(gross=[row(value)]))['annual_gp_assets_pairs'][0]
                self.assertEqual(pair['status'], 'structurally_paired_not_admitted')

    def test_assets_need_same_accession_filed_yearend_and_currency(self):
        cases = [dict(accn='0000320193-21-000001'), dict(filed='2020-02-01'),
                 dict(end='2019-09-29'), dict(form='10-K/A')]
        for changes in cases:
            with self.subTest(changes=changes):
                pair = inspect(payload(assets=[row(100, instant=True, **changes)]))['annual_gp_assets_pairs'][0]
                self.assertIn('missing_same_filing_instant_assets', pair['reasons'])
        pair = inspect(payload(asset_unit='EUR'))['annual_gp_assets_pairs'][0]
        self.assertIn('missing_same_filing_instant_assets', pair['reasons'])

    def test_assets_duration_is_not_silently_instant(self):
        pair = inspect(payload(assets=[row(100)]))['annual_gp_assets_pairs'][0]
        self.assertIn('missing_same_filing_instant_assets', pair['reasons'])

    def test_exact_value_duplicate_ok_conflict_blocks(self):
        report = inspect(payload(gross=[row(10), row(10, frame='CY2019')]))
        self.assertEqual(report['annual_gp_assets_pairs'][0]['status'], 'structurally_paired_not_admitted')
        for field in ['gross', 'assets']:
            values = [row(10, instant=(field == 'assets')), row(11, instant=(field == 'assets'))]
            report = inspect(payload(**{field: values}))
            reason = 'conflicting_gross_profit_values' if field == 'gross' else 'conflicting_assets_values'
            self.assertIn(reason, report['annual_gp_assets_pairs'][0]['reasons'])

    def test_distinct_annual_start_dates_do_not_collapse(self):
        pair = inspect(payload(gross=[row(10), row(10, start='2018-10-01')]))['annual_gp_assets_pairs'][0]
        self.assertIn('ambiguous_annual_duration', pair['reasons'])

    def test_quarter_and_amendment_not_admitted_as_annual_original(self):
        for changes in [dict(start='2019-07-01'), dict(form='10-Q'), dict(form='10-K/A')]:
            with self.subTest(changes=changes):
                self.assertEqual(inspect(payload(gross=[row(10, **changes)]))['annual_gp_assets_pairs'], [])

    def test_future_filing_cannot_backfill_old_fiscal_year(self):
        future = dict(accn='0000320193-21-000001', filed='2021-01-31', fy=2020)
        obj = payload(gross=[row(10, **future)], assets=[row(100, instant=True, **future)])
        report = inspect(obj, '2020-12-31')
        self.assertEqual(report['annual_gp_assets_pairs'], [])
        self.assertEqual(report['tag_summaries']['GrossProfit']['units']['USD']['after_filed_cutoff'], 1)

    def test_original_and_later_comparatives_remain_distinct_vintages(self):
        later = dict(accn='0000320193-21-000001', filed='2021-01-31', fy=2020)
        obj = payload(gross=[row(10), row(15, **later)],
                      assets=[row(100, instant=True), row(120, instant=True, **later)])
        before = inspect(obj, '2020-12-31')['annual_gp_assets_pairs']
        after = inspect(obj, '2021-12-31')['annual_gp_assets_pairs']
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 2)
        self.assertEqual(before[0]['filed'], '2020-01-31')
        self.assertEqual({p['filed'] for p in after}, {'2020-01-31', '2021-01-31'})

    def test_nonpositive_assets_and_boolean_amount_block(self):
        for value in [0, -1, True, '100']:
            with self.subTest(value=value):
                pair = inspect(payload(assets=[row(value, instant=True)]))['annual_gp_assets_pairs'][0]
                self.assertEqual(pair['status'], 'ambiguous_or_unusable')

    def test_wrong_cik_duplicate_keys_and_nonfinite_fail(self):
        obj = payload()
        obj['cik'] = 789019
        with self.assertRaises(ValueError):
            inspect(obj)
        for data in [b'{"cik":320193,"cik":320193,"facts":{}}',
                     b'{"cik":320193,"facts":{},"bad":NaN}']:
            with self.assertRaises(ValueError):
                source.validate_facts(data, CIK)

    def test_invalid_top_level_schema_is_rejected(self):
        for change in [dict(entityName=''), dict(facts={'us-gaap': []})]:
            obj = payload()
            obj.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                source.validate_facts(raw(obj), CIK)

    def test_fiscal_metadata_conflict_blocks(self):
        obj = payload(gross=[row(10), row(10, fy=2020)])
        pair = inspect(obj)['annual_gp_assets_pairs'][0]
        self.assertEqual(pair['status'], 'ambiguous_or_unusable')
        obj = payload(assets=[row(100, instant=True, fp='Q4')])
        pair = inspect(obj)['annual_gp_assets_pairs'][0]
        self.assertEqual(pair['status'], 'ambiguous_or_unusable')


class CollectionReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='profitability141-review-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / 'DATA_PLAN.json'
        data = dict(schema_version=1, purpose=source.PURPOSE,
                    allowed_ciks=[CIK, OTHER_CIK],
                    output_dir='outputs/monthly_profitability141_sources',
                    source_stop_timezone='America/Los_Angeles')
        self.plan.write_bytes(raw(data))
        self.digest = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.now = datetime(2026, 10, 19, 20, 0, tzinfo=timezone.utc)
        self.output = self.root / data['output_dir']
    def collect(self, opener, cik=CIK, now=None):
        return source.collect(self.plan, self.digest, cik, 'synthetic-review@example.invalid',
                              root=self.root, opener=opener, now=now or self.now)

    def test_success_preserves_exact_raw_bytes_and_safe_headers(self):
        body = raw(payload())
        opener = Opener(Response(body, headers={'Content-Type': 'application/json',
                                                'Set-Cookie': 'do-not-store'}))
        result = self.collect(opener)
        self.assertEqual(result['status'], 'saved')
        self.assertEqual(Path(result['raw_path']).read_bytes(), body)
        receipt = json.loads((self.output / 'receipts' / ('CIK'+CIK+'.json')).read_bytes())
        self.assertEqual(receipt['sha256'], hashlib.sha256(body).hexdigest())
        self.assertEqual(receipt['plan_sha256'], self.digest)
        self.assertNotIn('set-cookie', receipt['headers'])
        self.assertEqual(len(opener.calls), 1)
        replay = Opener(Response(b'not used'))
        self.assertEqual(self.collect(replay)['status'], 'reused')
        self.assertEqual(replay.calls, [])

    def test_wrong_entity_is_failure_and_not_success_cache(self):
        body = payload()
        body['cik'] = 789019
        result = self.collect(Opener(Response(raw(body))))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.output / 'raw' / ('CIK'+CIK+'.json')).exists())
        receipt = json.loads(Path(result['receipt_path']).read_bytes())
        self.assertEqual(receipt['sha256'], hashlib.sha256(raw(body)).hexdigest())

    def test_403_and_429_stop_different_cik_without_second_request(self):
        for code in [403, 429]:
            with self.subTest(code=code):
                if list(self.output.glob('STOPPED_*.json')):
                    for path in self.output.glob('STOPPED_*.json'):
                        path.unlink()
                result = self.collect(Opener(Response(b'blocked', code=code,
                    headers={'Content-Type': 'text/plain', 'Retry-After': '172800'})))
                self.assertEqual(result['status'], 'failed')
                blocked = Opener(Response(raw(payload()), url=source.endpoint(OTHER_CIK)))
                with self.assertRaises(ValueError):
                    self.collect(blocked, cik=OTHER_CIK)
                self.assertEqual(blocked.calls, [])
                tomorrow = self.now.replace(day=20)
                with self.assertRaises(ValueError):
                    self.collect(blocked, cik=OTHER_CIK, now=tomorrow)
                self.assertEqual(blocked.calls, [])

    def test_plan_digest_and_unplanned_cik_fail_before_request(self):
        opener = Opener(Response(raw(payload())))
        self.plan.write_bytes(self.plan.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            self.collect(opener)
        self.assertEqual(opener.calls, [])
        self.digest = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            self.collect(opener, cik='0000000001')
        self.assertEqual(opener.calls, [])

    def test_200_html_content_type_not_success(self):
        result = self.collect(Opener(Response(raw(payload()), headers={'Content-Type': 'text/html'})))
        self.assertEqual(result['status'], 'failed')

    def test_frozen_user_agent_mismatch_blocks_before_request(self):
        plan = json.loads(self.plan.read_bytes())
        plan['user_agent'] = 'ExactFrozenAgent/1.0'
        self.plan.write_bytes(raw(plan))
        self.digest = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        opener = Opener(Response(raw(payload())))
        with self.assertRaises(ValueError):
            self.collect(opener)
        self.assertEqual(opener.calls, [])

    def test_cached_receipt_from_other_plan_cannot_be_reused(self):
        self.collect(Opener(Response(raw(payload()))))
        receipt_path = self.output / 'receipts' / ('CIK'+CIK+'.json')
        receipt = json.loads(receipt_path.read_bytes())
        receipt['plan_sha256'] = '0' * 64
        receipt_path.write_bytes(raw(receipt))
        opener = Opener(Response(raw(payload())))
        with self.assertRaises(ValueError):
            self.collect(opener)
        self.assertEqual(opener.calls, [])

    def test_redirected_response_not_success(self):
        result = self.collect(Opener(Response(raw(payload()), url='https://example.invalid/x')))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.output / 'raw' / ('CIK'+CIK+'.json')).exists())


class FrozenScopeReview(unittest.TestCase):
    def test_plan_preserves_all_original_direct_stock_names_and_unknowns(self):
        root = Path(__file__).resolve().parent.parent
        path = root / 'outputs/monthly_profitability141_sources/DATA_PLAN.json'
        body = path.read_bytes()
        self.assertEqual(hashlib.sha256(body).hexdigest(),
                         '214fb91e749d6d6dc54ff4b479e6230a1b4cecac48bcca69a15b017e4110d112')
        plan = json.loads(body)
        meta_path = root / 'outputs/monthly_architecture_restart/META.json'
        meta_raw = meta_path.read_bytes()
        self.assertEqual(hashlib.sha256(meta_raw).hexdigest(),
                         plan['input_hashes']['outputs/monthly_architecture_restart/META.json'])
        meta = json.loads(meta_raw)['monthly']
        original = set().union(*(set(row) for row in meta['symbols'])) - set(meta['etfs'])
        self.assertEqual(len(original), 73)
        self.assertEqual(set(plan['candidate_universe']), original)
        evidence = plan['candidate_cik_locating_evidence']
        self.assertEqual({item['symbol'] for item in evidence}, original)
        self.assertEqual(len(evidence), 73)
        located = set().union(*(set(item['candidate_ciks']) for item in evidence))
        self.assertEqual(set(plan['allowed_ciks']), located)
        self.assertEqual(len(located), 43)
        self.assertEqual(sum(not item['candidate_ciks'] for item in evidence), 30)
        self.assertTrue(all(item['identity_status'] ==
                            'locating_evidence_only_not_historical_identity_admission'
                            for item in evidence))
        self.assertEqual(plan['first_probe_cik'], CIK)
        self.assertEqual(plan['first_probe_requests'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
