"""Independent synthetic review harness. Never reads original financial rows."""
import argparse
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import unittest

class AppleIndependentTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('after155_aapl_revenue_source')
        self.texts=['Apple Inc.\nCONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)\n(In millions, except per share)\nThree Months Ended Twelve Months Ended\nSeptember 28, September 30, September 28, September 30,\n2024 2023 2024 2023\nTotal net sales $ 1.00 $ 2.00 $ 30.50 $ 40.25\n','Apple Inc.\nCONDENSED CONSOLIDATED BALANCE SHEETS (Unaudited)\nSeptember 28, September 30,\n2024 2023\n']
        self.spec={'fiscal_year':2024,'fiscal_year_end':'2024-09-28','release_date':'2024-10-31'}
    def test_annual_selection_and_noncertified_currency(self):
        row=self.m.parse_revenue(self.texts,self.spec)
        self.assertEqual(row['revenue']['current']['amount_decimal'],'30.50')
        self.assertEqual(row['revenue']['comparative']['amount_decimal'],'40.25')
        self.assertIsNone(row['revenue']['current']['iso_currency'])
        self.assertFalse(row['revenue']['current']['ISO_currency_verified'])
    def test_no_mutation_and_unavailable_prior_start(self):
        before=deepcopy((self.texts,self.spec));row=self.m.parse_revenue(self.texts,self.spec)
        self.assertEqual((self.texts,self.spec),before)
        self.assertIsNone(row['revenue']['comparative']['period']['start'])
        self.m.attach_prior_duration(row,None)
        self.assertIsNone(row['revenue']['comparative']['period']['start'])
    def test_competing_total_row_rejected(self):
        self.texts[0]+='Net sales $ 1 $ 2 $ 3 $ 4\n'
        with self.assertRaises(ValueError):self.m.parse_revenue(self.texts,self.spec)
    def test_nonconsecutive_period_rejected(self):
        self.texts[0]=self.texts[0].replace('2024 2023 2024 2023','2024 2022 2024 2022').replace('September 30,','September 24,')
        self.texts[1]=self.texts[1].replace('2024 2023','2024 2022').replace('September 30,','September 24,')
        with self.assertRaises(ValueError):self.m.parse_revenue(self.texts,self.spec)
    def test_current_release_date_cannot_equal_period_end(self):
        self.spec['release_date']='2024-09-28'
        with self.assertRaises(ValueError):self.m.parse_revenue(self.texts,self.spec)
    def test_notes_are_leads_at_current_availability(self):
        rows=self.m.note_contexts('An accounting change retrospectively recast revenue.',{'path':'fabricated'},'2024-11-01')
        self.assertTrue(rows)
        self.assertTrue(all(x['known_at_session']=='2024-11-01' and x['requires_review'] and not x['confirmed_accounting_change'] for x in rows))
    def test_wrong_parent_link_rejected(self):
        raw=b'<h1>Apple reports fourth quarter results</h1><span class="category-eyebrow__date">October 31, 2024</span><script type="application/ld+json">{"datePublished":"2024-10-31"}</script><a href="/other.pdf">PDF</a>'
        with self.assertRaises(ValueError):self.m.publication(raw,dict(self.spec,source_url='https://www.apple.com/fake.pdf'),'https://www.apple.com/release')

class ChainIndependentTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('after155_multi_issuer_chain')
        bounds=['2019-09-28','2020-09-26','2021-09-25','2022-09-24','2023-09-30']
        amounts=['100','120','125','130']
        from datetime import date,timedelta
        self.r=[];self.e=[]
        for i,y in enumerate(range(2020,2024)):
            r=dict(issuer='AAPL',fiscal_year=y,report_id='original_'+str(y),known_ns=(i+1)*10,
                source_checked=True,currency_basis_checked=True,source_sha256='a'*64,source_review_sha256='b'*64,
                period_start=(date.fromisoformat(bounds[i])+timedelta(days=1)).isoformat(),period_end=bounds[i+1],
                current=amounts[i],scale='1000000',measure='consolidated_gaap_total_revenue',
                currency='reported_dollar',currency_basis='declared_within_issuer_reported_dollar_v1',
                currency_iso=None,currency_iso_certified=False,currency_policy_sha256='d'*64)
            if i:
                r.update(comparative=amounts[i-1],comparative_scale='1000000',
                    comparative_period_start=self.r[-1]['period_start'],comparative_period_end=self.r[-1]['period_end'],
                    comparative_currency='reported_dollar',comparative_currency_basis=r['currency_basis'],
                    comparative_currency_iso=None,comparative_currency_iso_certified=False,
                    comparative_measure=r['measure'],comparative_source_report_id=r['report_id'])
                self.e.append(dict(issuer='AAPL',earlier_fiscal_year=y-1,later_fiscal_year=y,
                    earlier_report_id=self.r[-1]['report_id'],later_report_id=r['report_id'],edge_id='edge_'+str(y),
                    known_ns=r['known_ns'],evidence_known_ns=[self.r[-1]['known_ns'],r['known_ns']],
                    source_review_sha256='c'*64,status='checked_reported_total_edge'))
            self.r.append(r)
    def evaluate(self,cutoff=45):return self.m.evaluate('AAPL',self.r,self.e,cutoff,'2024-01-31')
    def assertHeld(self):self.assertNotEqual(self.evaluate()['status'],'checked_reported_total_chain_with_declared_source_scope')
    def test_independent_literal_dollar_fixture_and_no_mutation(self):
        before=deepcopy((self.r,self.e));out=self.evaluate()
        self.assertEqual(out['status'],'checked_reported_total_chain_with_declared_source_scope')
        self.assertFalse(out['factor_computed']);self.assertFalse(out['economic_homogeneity_certified'])
        self.assertEqual((self.r,self.e),before)
    def test_known_at_exact_cutoff_is_included(self):self.assertEqual(self.evaluate(40)['status'],'checked_reported_total_chain_with_declared_source_scope')
    def test_before_latest_availability_missing_original(self):self.assertEqual(self.evaluate(39)['status'],'missing_required_source')
    def test_later_source_failure_does_not_retroactively_block(self):
        extra=deepcopy(self.r[-1]);extra.update(fiscal_year=2024,report_id='future',known_ns=50,source_checked=False)
        self.r.append(extra)
        self.assertEqual(self.evaluate()['status'],'checked_reported_total_chain_with_declared_source_scope')
        self.assertNotEqual(self.evaluate(50)['status'],'checked_reported_total_chain_with_declared_source_scope')
    def test_future_edge_does_not_retroactively_conflict(self):
        extra=deepcopy(self.e[-1]);extra.update(known_ns=50,edge_id='future_edge');self.e.append(extra)
        self.assertEqual(self.evaluate()['status'],'checked_reported_total_chain_with_declared_source_scope')
    def test_known_unchecked_original_is_held(self):self.r[0]['source_checked']=False;self.assertHeld()
    def test_currency_policy_must_match(self):self.r[1]['currency_policy_sha256']='e'*64;self.assertHeld()
    def test_original_member_scales_must_match_policy(self):
        self.r[1]['scale']='1000';self.r[2]['comparative_scale']='1000';self.assertHeld()
    def test_currency_boolean_not_integer(self):self.r[0]['currency_iso_certified']=0;self.assertHeld()
    def test_false_iso_assertion_is_held(self):self.r[0]['currency_iso']='USD';self.assertHeld()
    def test_comparative_false_certification_is_held(self):self.r[1]['comparative_currency_iso_certified']=True;self.assertHeld()
    def test_mixed_declared_and_evidenced_basis_is_held(self):
        self.r[0].update(currency='USD',currency_basis='iso_evidenced',currency_iso='USD',currency_iso_certified=True);self.assertHeld()
    def test_missing_source_review_is_held(self):self.r[0].pop('source_review_sha256');self.assertHeld()
    def test_revenue_definition_change_is_held(self):self.r[1]['measure']='segment_revenue';self.assertHeld()
    def test_comparative_rebasing_is_held(self):
        self.r[1]['comparative']='101';self.assertEqual(self.evaluate()['status'],'blocked_known_basis_break')
    def test_equal_scaled_amount_does_not_override_policy(self):
        self.r[1]['comparative']='100000';self.r[1]['comparative_scale']='1000';self.assertHeld()
    def test_duplicate_edge_reference_is_held(self):self.e[1]['edge_id']=self.e[0]['edge_id'];self.assertHeld()
    def test_nondict_record_is_clean_hold(self):self.r.append(None);self.assertHeld()
    def test_unreviewed_note_is_held(self):self.e[0]['status']='unreviewed_note';self.assertHeld()
    def test_note_after_cutoff_is_held(self):self.e[0].update(known_ns=50,evidence_known_ns=[50]);self.assertHeld()
    def test_boundary_mismatch_is_held(self):self.r[1]['period_start']='2020-09-26';self.assertHeld()
    def test_msft_cannot_use_new_dollar_assumption(self):
        from datetime import date
        for r in self.r:
            y=r['fiscal_year'];r.update(issuer='MSFT',period_start=f'{y-1}-07-01',period_end=f'{y}-06-30')
        for e in self.e:e['issuer']='MSFT'
        out=self.m.evaluate('MSFT',self.r,self.e,45,'2024-01-31')
        self.assertNotEqual(out['status'],'checked_reported_total_chain_with_declared_source_scope')


def main():
    import hashlib
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['aapl','chain'],default='chain');args=p.parse_args()
    source=Path(__file__).parent/'after155_multi_issuer_chain.py';tests=Path(__file__).parent/'test_after155_multi_issuer_chain.py'
    expected={'source':'f106f40624afd6fc233263301df06cf85ed8961bca8b822007c9161f65b80bba','tests':'2d1c82442d594c0021581dc58ab4966c1626024a29ebd112a83e89e04cd099d9'}
    observed={'source':hashlib.sha256(source.read_bytes()).hexdigest(),'tests':hashlib.sha256(tests.read_bytes()).hexdigest()}
    if args.target=='chain' and observed!=expected:raise ValueError('Reviewed source/test hashes changed')
    suite=unittest.TestSuite()
    if args.target=='aapl':
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module('after155_aapl_revenue_tests')))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AppleIndependentTests))
    else:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module('test_after155_multi_issuer_chain')))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ChainIndependentTests))
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    print(output.getvalue())
    if args.target=='chain' and observed!={'source':hashlib.sha256(source.read_bytes()).hexdigest(),'tests':hashlib.sha256(tests.read_bytes()).hexdigest()}:raise ValueError('Reviewed source/test hashes changed during tests')
    print(json.dumps({'target':args.target,'tests':result.testsRun,'passed':result.wasSuccessful(),'real_source_amount_reads':0,'new_source_chains':0,'new_factors':0,'new_returns':0,'source_hashes':observed if args.target=='chain' else None}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
