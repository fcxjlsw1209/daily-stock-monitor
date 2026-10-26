"""Synthetic adversarial checks; no actual source or outcome is used."""
import copy
import unittest
from after155_revenue_coverage_gate import coverage_gate, USABLE


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.meta = {'dates': [f'{2000+i//12:04d}-{i%12+1:02d}-28' for i in range(122)],
                     'symbols': [['MSFT', 'AAPL', 'ABBV']+[f'X{j}' for j in range(46)] for _ in range(122)],
                     'etfs': ['X0']}
        self.plan = {'fixed_issuer_scope': ['MSFT','AAPL','ABBV'], 'gate_before_strategy_registration': {
            'minimum_issuers':2,'minimum_usable_candidate_rows_per_qualifying_issuer_in_first88':12,
            'minimum_distinct_latest_original_annual_endpoints_per_qualifying_issuer_in_first88':2,
            'minimum_first88_cutoffs_with_at_least_two_usable_issuers':12}}
        self.rows = [dict(signal_index=i,candidate_index=j,decision_date=self.meta['dates'][i],
                     symbol=s,status='missing_required_source',latest_report_id=None)
                     for i, ss in enumerate(self.meta['symbols']) for j,s in enumerate(ss)]

    def fill(self, issuer, indices):
        for r in self.rows:
            if r['symbol']==issuer and r['signal_index'] in indices:
                r.update(status=USABLE,latest_report_id=f"{issuer}:{r['signal_index']//6}")

    def gate(self): return coverage_gate(self.meta,self.rows,self.plan)

    def test_exact_threshold(self):
        self.fill('MSFT',range(12)); self.fill('AAPL',range(12))
        self.assertTrue(self.gate()['source_coverage_floor_passed'])

    def test_short_overlap(self):
        self.fill('MSFT',range(12)); self.fill('AAPL',range(1,13))
        self.assertFalse(self.gate()['source_coverage_floor_passed'])

    def test_one_endpoint(self):
        self.fill('MSFT',range(12)); self.fill('AAPL',range(12))
        for r in self.rows:
            if r['symbol']=='AAPL': r['latest_report_id']='only_one'
        self.assertFalse(self.gate()['source_coverage_floor_passed'])

    def test_later_holdout_never_qualifies(self):
        self.fill('MSFT',range(88,122)); self.fill('AAPL',range(88,122))
        self.assertFalse(self.gate()['source_coverage_floor_passed'])
        self.assertEqual(len(self.gate()['segments']['later34']['cutoffs_with_minimum_issuers']),34)

    def test_missing_slot(self):
        self.rows.pop()
        with self.assertRaises(ValueError): self.gate()

    def test_duplicate_slot(self):
        self.rows[-1]=copy.deepcopy(self.rows[0])
        with self.assertRaises(ValueError): self.gate()

    def test_changed_candidate(self):
        self.rows[0]['symbol']='WRONG'
        with self.assertRaises(ValueError): self.gate()

    def test_outside_pilot(self):
        self.rows[3].update(status=USABLE,latest_report_id='ETF')
        with self.assertRaises(ValueError): self.gate()

    def test_endpoint_required(self):
        self.rows[0]['status']=USABLE
        with self.assertRaises(ValueError): self.gate()

    def test_no_hidden_mutation(self):
        before=copy.deepcopy((self.meta,self.rows,self.plan))
        result=self.gate()
        self.assertEqual((self.meta,self.rows,self.plan),before)
        self.assertEqual(result['rows_verified'],5978)
        self.assertFalse(result['source_reviews_verified_by_this_function'])


if __name__=='__main__': unittest.main()
