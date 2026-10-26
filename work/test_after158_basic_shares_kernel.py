"""Synthetic checks only; importing/running never loads actual financial data."""
from copy import deepcopy
from fractions import Fraction
import unittest
import after158_basic_shares_kernel as k


def event(symbol='A', **kwargs):
    e=dict(event_id=symbol+'-2020', symbol=symbol, known_at='2021-01-05T21:00:00+00:00',
           usable_at='2021-01-07T21:00:00+00:00', period_end='2020-12-31',
           prior_period_end='2019-12-31', current_shares='90', prior_shares='100',
           conditional_source_ready=True, same_report_annual_pair=True,
           same_share_basis=True, unresolved_reasons=[])
    e.update(kwargs)
    return e


class Tests(unittest.TestCase):
    t='2021-03-31T20:00:00+00:00'

    def test_negative_growth(self):
        self.assertEqual(k.measure(event(), self.t), (Fraction(1,10), None))

    def test_common_scale_invariant(self):
        self.assertEqual(k.measure(event(current_shares='90000',prior_shares='100000'),self.t),(Fraction(1,10),None))

    def test_shrinkage_and_growth_rank(self):
        m,r=k.transform([.1, -.2, None, 0.])
        self.assertEqual(r,[1.,-1.,0.,0.])
        self.assertGreater(m[0],0);self.assertLess(m[1],0)

    def test_ties(self):
        self.assertEqual(k.transform([1.,1.,2.])[1],[-.5,-.5,1.])
        self.assertEqual(k.transform([0.,0.])[1],[0.,0.])
        self.assertEqual(k.transform([.1,.1]),([0.,0.],[0.,0.]))

    def test_missing_not_zero(self):
        self.assertEqual(k.transform([None, .1]),([0.,0.],[0.,0.]))
        self.assertEqual(k.transform([0., .1])[1],[-1.,1.])

    def test_zero_invalid_and_no_imputation(self):
        for value in ['0','-1','NaN','Infinity',None,True]:
            self.assertEqual(k.measure(event(prior_shares=value),self.t)[1],'invalid_share_denominator')

    def test_unverified_split_is_not_inferred(self):
        self.assertEqual(k.measure(event(current_shares='400',same_share_basis=False),self.t)[1],
                         'same_share_basis_unverified')

    def test_known_bad_latest_blocks_prior(self):
        old=event(period_end='2019-12-31',prior_period_end='2018-12-31',event_id='old')
        new=event(conditional_source_ready=False)
        out=k.evaluate_month([old,new],['A'],[True],self.t)
        self.assertEqual(out['audit'][0]['event_id'],'A-2020')
        self.assertEqual(out['negative_growth'],[None])

    def test_known_unusable_latest_blocks_prior(self):
        latest=event(usable_at='2021-04-01T20:00:00+00:00')
        self.assertEqual(k.evaluate_month([latest],['A'],[True],self.t)['audit'][0]['reason'],
                         'latest_not_yet_usable')

    def test_future_values_unread(self):
        future=dict(symbol='A',known_at='2022-01-01T21:00:00+00:00')
        self.assertEqual(k.evaluate_month([event(),future],['A'],[True],self.t),
                         k.evaluate_month([event()],['A'],[True],self.t))

    def test_initial_partial_no_lookup(self):
        self.assertEqual(k.evaluate_month([{}],['A'],[True],self.t,True)['negative_growth'],[None])

    def test_exact_cutoff_usable(self):
        self.assertEqual(k.measure(event(),event()['usable_at']),(Fraction(1,10),None))

    def test_corroboration_cannot_precede_knowledge(self):
        with self.assertRaises(ValueError):
            k.measure(event(usable_at='2021-01-04T21:00:00+00:00'),self.t)

    def test_stale_annual_report(self):
        self.assertEqual(k.measure(event(),'2023-03-31T20:00:00+00:00')[1],
                         'latest_annual_period_stale')

    def test_duplicate_period_is_error(self):
        with self.assertRaises(ValueError):k.latest_known([event(),event()], 'A',self.t)

    def test_unknown_and_unresolved_sources(self):
        self.assertEqual(k.evaluate_month([event()],['B'],[True],self.t)['audit'][0]['reason'],'no_known_original')
        self.assertEqual(k.measure(event(unresolved_reasons=['x']),self.t)[1],'unresolved_source_reasons')

    def test_no_input_mutation(self):
        source=[event(),event('B',current_shares='110')]; before=deepcopy(source)
        k.evaluate_month(source,['A','B'],[True,True],self.t)
        self.assertEqual(source,before)

    def test_unavailable_candidates_have_no_score_premium(self):
        m,r=k.transform([.1,.2,.3,None])
        self.assertAlmostEqual(sum(m),0.)
        self.assertAlmostEqual(sum(r),0.)
        self.assertEqual((m[-1],r[-1]),(0.,0.))
        self.assertTrue(all(abs(x)<=1 for x in m+r))

    def test_original_eligibility_cannot_be_widened(self):
        out=k.evaluate_month([event(),event('B',current_shares='110')],
                              ['A','B'],[False,True],self.t)
        self.assertEqual(out['audit'][0]['reason'],'original_ineligible')
        self.assertEqual(out['rank'],[0.,0.])

    def test_fixed_economic_scale(self):
        m,r=k.transform([.05,-.05])
        self.assertEqual(m,[.25,-.25])
        self.assertEqual(r,[1.,-1.])

    def test_exact_fractional_ties_and_near_ties(self):
        left=Fraction(1,3);right=left+Fraction(1,10**30)
        self.assertEqual(float(left),float(right))
        self.assertEqual(k.transform([left,right])[1],[-1.,1.])
        first,_=k.measure(event(current_shares='200',prior_shares='300'),self.t)
        second,_=k.measure(event(current_shares='20',prior_shares='30'),self.t)
        self.assertEqual(k.transform([first,second]),([0.,0.],[0.,0.]))


if __name__=='__main__':unittest.main(verbosity=2)
