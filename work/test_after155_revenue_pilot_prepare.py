"""Fabricated driver tests. No actual source grid or financial outcome is read."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import after155_revenue_pilot_prepare as driver


class Close:
    def __init__(self, day):
        self.day = day
        self.value = date.fromisoformat(day).toordinal() * 86400 * 10**9

    def isoformat(self):
        return self.day+'T21:00:00+00:00'

    def tz_convert(self, zone):
        return self

    def date(self):
        return date.fromisoformat(self.day)


def currency():
    return dict(policy_sha256='p'*64, currency='reported_dollar', currency_basis=driver.DOLLAR_BASIS,
                currency_iso=None, currency_iso_certified=False, actual_source_preconditions_verified=True,
                within_issuer_assumption_admitted=True, scope_fiscal_years=list(range(2013, 2026)))


def catalog():
    originals, checks, closes = [], [], {}
    for year in range(2013, 2026):
        day, release = f'{year+1}-02-03', f'{year+1}-01-31'
        close = Close(day)
        closes[day] = close
        source = dict(raw=dict(path=f'raw/{year}', sha256='a'*64, bytes=4),
                      receipt=dict(path=f'raw/{year}.meta', sha256='b'*64, bytes=4),
                      metadata=dict(symbol='ABBV', source_id=f'ABBV_{year}'))
        clock = dict(release_date=release, precision='date_only', retrieved_at='2026-10-20T00:00:00+00:00',
                     available_session=None if year == 2013 else day)
        cells, reviewed = {}, {}
        for role, fiscal in [('current', year), ('comparative', year-1)]:
            cells[role] = dict(amount_decimal=str(fiscal), scale=1000000, fact_year=fiscal,
                period_start=f'{fiscal}-01-01', period_end=f'{fiscal}-12-31', cell_html_sha256='c'*64,
                currency_iso=None, currency_literal='$', source_raw_sha256='a'*64, locator={},
                period_derivation='annual source header', measure_label='Net revenues',
                currency_status='HOLD_BARE_DOLLAR_NOT_ISO_CERTIFICATION', measure='consolidated_total_net_revenue')
            reviewed[role] = dict(exact_amount_verified=True, period_verified=True,
                own_annual_column_dollar_evidence_verified=True, cell_sha256='c'*64,
                scale=1000000, fiscal_year=fiscal)
        originals.append(dict(symbol='ABBV', fiscal_year=year, source=source,
                              publication_clock=clock, revenue=cells, holds=['CURRENCY_ISO_UNRESOLVED']))
        checks.append(dict(fiscal_year=year, source_path=f'raw/{year}', source_sha256='a'*64,
                           receipt_verified=True, reported_gaap_policy_verified=True,
                           release_date=release, independent_available_session=day,
                           independent_known_by_close=close.isoformat(), cells=reviewed))
    return {'reports': originals}, {'reports': checks}, {'currency_policy_review': currency()}, sorted(closes), closes


def normalized():
    extraction, source, accounting, days, closes = catalog()
    return driver.normalize_abbv(extraction, source, accounting, 's'*64, 'r'*64, 'p'*64, days, closes)


def edge_fixture():
    reports = normalized()
    edges, checked = [], []
    for earlier, later in zip(reports, reports[1:]):
        y = earlier['fiscal_year']
        evidence = [dict(report_id=r['report_id'], source_path=r['source']['raw']['path'],
                         source_sha256=r['source_sha256'], known_ns=r['known_ns']) for r in [earlier, later]]
        edge = dict(edge_id=f'e{y}', earlier_fiscal_year=y, later_fiscal_year=y+1,
            earlier_report_id=earlier['report_id'], later_report_id=later['report_id'],
            known_ns=later['known_ns'], evidence_known_ns=[earlier['known_ns'], later['known_ns']],
            currency_policy_sha256='p'*64, evidence=evidence, status='checked_reported_total_edge')
        edges.append(edge)
        check = {k: edge[k] for k in ['edge_id', 'earlier_fiscal_year', 'later_fiscal_year',
            'earlier_report_id', 'later_report_id', 'known_ns', 'evidence_known_ns', 'status']}
        check.update(admitted=True, source_review_sha256='p'*64)
        checked.append(check)
    return {'edges': edges}, {'edges': checked}, reports


class ReviewAndCatalogTests(unittest.TestCase):
    def test_local_closure_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input'
            p.write_bytes(b'before')
            inputs = driver.Inputs(folder)
            inputs.read('input')
            p.write_bytes(b'after')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                inputs.unchanged()

    def test_nonlocal_input_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Nonlocal'):
                driver.Inputs(folder).read('../escape')

    def test_missing_review_digest_fails(self):
        class FakeInputs:
            def read(self, *args):
                return b''
        with self.assertRaisesRegex(ValueError, 'review SHA missing'):
            driver.review_inputs(FakeInputs(), {})

    def test_early_clock_integrates_independent_value_and_keeps_original_hold(self):
        reports = normalized()
        self.assertEqual(reports[0]['known_by'], '2014-02-03T21:00:00+00:00')
        self.assertIsNone(reports[0]['original_publication_clock']['available_session'])
        self.assertEqual(reports[0]['original_holds_preserved'], ['CURRENCY_ISO_UNRESOLVED'])
        self.assertEqual(reports[0]['current'], '2013')
        self.assertEqual(reports[0]['comparative'], '2012')
        self.assertIsNone(reports[0]['currency_iso'])
        self.assertIs(reports[0]['currency_iso_certified'], False)

    def test_synthetic_currency_review_cannot_admit_actual_sources(self):
        args = list(catalog())
        args[2]['currency_policy_review']['actual_source_preconditions_verified'] = False
        with self.assertRaisesRegex(ValueError, 'not reviewed/admitted'):
            driver.normalize_abbv(*args[:3], 's'*64, 'r'*64, 'p'*64, *args[3:])

    def test_currency_certification_escalation_fails(self):
        review = {'currency_policy_review': currency()}
        review['currency_policy_review']['currency_iso_certified'] = True
        with self.assertRaisesRegex(ValueError, 'certified ISO'):
            driver.currency_review(review, 'p'*64)

    def test_missing_catalog_year_fails(self):
        args = list(catalog())
        args[0]['reports'].pop(0)
        with self.assertRaisesRegex(ValueError, 'Incomplete bounded'):
            driver.normalize_abbv(*args[:3], 's'*64, 'r'*64, 'p'*64, *args[3:])

    def test_wrong_independent_cell_hash_fails(self):
        args = list(catalog())
        args[1]['reports'][0]['cells']['current']['cell_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'cell review binding'):
            driver.normalize_abbv(*args[:3], 's'*64, 'r'*64, 'p'*64, *args[3:])

    def test_wrong_independent_clock_fails(self):
        args = list(catalog())
        args[1]['reports'][0]['independent_known_by_close'] = '2014-02-03T20:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'close mismatch'):
            driver.normalize_abbv(*args[:3], 's'*64, 'r'*64, 'p'*64, *args[3:])

    def test_edge_holds_override_producer_positive(self):
        proposal, review, reports = edge_fixture()
        review['edges'][4].update(admitted=False, status='pending_identified_reclassification_note')
        result = driver.normalize_edges(proposal, review, reports, 'i'*64, 'p'*64, 'p'*64)
        self.assertEqual(result[4]['status'], 'pending_identified_reclassification_note')
        self.assertEqual(result[4]['source_review_sha256'], 'i'*64)
        self.assertEqual(proposal['edges'][4]['status'], 'checked_reported_total_edge')

    def test_missing_edge_disposition_fails(self):
        proposal, review, reports = edge_fixture()
        del review['edges'][0]['admitted']
        with self.assertRaisesRegex(ValueError, 'explicit per-edge disposition'):
            driver.normalize_edges(proposal, review, reports, 'i'*64, 'p'*64, 'p'*64)

    def test_rejected_positive_status_fails(self):
        proposal, review, reports = edge_fixture()
        review['edges'][0]['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'Rejected edge has positive'):
            driver.normalize_edges(proposal, review, reports, 'i'*64, 'p'*64, 'p'*64)

    def test_missing_edge_review_fails(self):
        proposal, review, reports = edge_fixture()
        review['edges'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete ABBV accounting-edge'):
            driver.normalize_edges(proposal, review, reports, 'i'*64, 'p'*64, 'p'*64)

    def test_future_note_fails(self):
        proposal, review, reports = edge_fixture()
        proposal['edges'][0]['evidence'][0]['known_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'note source/clock'):
            driver.normalize_edges(proposal, review, reports, 'i'*64, 'p'*64, 'p'*64)


class GridTests(unittest.TestCase):
    def setUp(self):
        dates = [(date(2020, 1, 1)+timedelta(days=i)).isoformat() for i in range(122)]
        names = ['MSFT', 'AAPL', 'ABBV', 'ETF']+[f'X{i}' for i in range(45)]
        self.meta = dict(dates=dates, symbols=[list(names) for _ in dates], etfs=['ETF'])
        self.closes = {d: Close(d) for d in dates}
        self.prior = [dict(signal_index=i, candidate_index=j, symbol=s, decision_date=d,
            decision_cutoff_ns=self.closes[d].value, status='prior_hold',
            factor_computed=False, factor_admitted=False, strategy_admitted=False,
            policy_id='original_msft_contract', reasons=['original'])
            for i, d in enumerate(dates) for j, s in enumerate(names)]
        self.shas = dict(msft='m'*64, msft_rows='f'*64, abbv_source='s'*64,
                         abbv_accounting='a'*64, chain='c'*64)

    def evaluate(self, issuer, reports, edges, cutoff, day):
        self.assertEqual(issuer, 'ABBV')
        self.assertEqual(cutoff, self.closes[day].value)
        return dict(status='missing_required_source', reasons=['synthetic'], report_ids=[], edge_ids=[],
                    latest_report_id=None, factor_computed=False)

    def run_grid(self):
        return driver.grid(self.meta, self.prior, [], [], self.closes, 'policy', self.shas, self.evaluate)

    def test_axis_msft_contract_apple_hold_and_etf_preserved(self):
        before = deepcopy(self.prior)
        result = self.run_grid()
        self.assertEqual(len(result), 122*49)
        self.assertEqual(self.prior, before)
        for original, row in zip(self.prior, result):
            self.assertEqual((row['signal_index'], row['candidate_index'], row['symbol']),
                             (original['signal_index'], original['candidate_index'], original['symbol']))
            if row['symbol'] == 'MSFT':
                self.assertTrue(all(row[k] == v for k, v in original.items()))
            if row['symbol'] == 'AAPL':
                self.assertEqual(row['status'], 'unresolved_source_or_comparability')
                self.assertEqual(row['report_ids'], [])
            if row['symbol'] == 'ETF':
                self.assertEqual(row['status'], 'not_applicable_etf')

    def test_wrong_previous_grid_clock_fails(self):
        self.prior[1]['decision_cutoff_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'axis/clock mismatch'):
            self.run_grid()

    def test_dropped_candidate_fails(self):
        self.meta['symbols'][2].pop()
        with self.assertRaisesRegex(ValueError, 'candidate/calendar axis'):
            self.run_grid()

    def test_duplicate_previous_slot_fails(self):
        self.prior.append(deepcopy(self.prior[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate previous grid'):
            self.run_grid()

    def test_usable_without_provenance_fails(self):
        row = dict(status=driver.USABLE, report_ids=[], edge_ids=[], decision_cutoff_ns=1)
        with self.assertRaisesRegex(ValueError, 'four originals/three edges'):
            driver.attach_provenance(row, [], [], {})


if __name__ == '__main__':
    unittest.main()
