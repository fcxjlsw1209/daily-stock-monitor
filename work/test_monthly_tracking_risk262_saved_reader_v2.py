"""Synthetic tests for saved diagnostic-fee representation exception only."""
import copy
import math
import unittest
import monthly_tracking_risk262_saved_reader_v2 as reader

class DisplayFeeReviewTests(unittest.TestCase):
    def sample(self):
        return [dict(signal_date='2000-01-31',execution_date='2000-02-01',symbols=['A','B','C'],scores=[.3,.2,.1],fee=50.)]
    def check(self,current,previous):return reader.diagnostic_holdings_parity(current,previous)
    def test_exact(self):
        x=self.sample();self.assertTrue(self.check(x,copy.deepcopy(x))['exact'])
    def test_one_ulp(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['fee']=math.nextafter(50.,math.inf)
        r=self.check(x,y);self.assertEqual(len(r['differences']),1);self.assertEqual(r['differences'][0]['ulp_difference'],1.)
    def test_two_ulp(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['fee']=math.nextafter(math.nextafter(50.,math.inf),math.inf)
        self.assertEqual(self.check(x,y)['differences'][0]['ulp_difference'],2.)
    def test_large_difference_fails(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['fee']+=1e-8
        with self.assertRaises(ValueError):self.check(x,y)
    def test_zero_exact(self):
        x=self.sample();x[0]['fee']=0.;self.assertTrue(self.check(x,copy.deepcopy(x))['exact'])
    def test_zero_nonzero_fails(self):
        x=self.sample();y=copy.deepcopy(x);x[0]['fee']=0.;y[0]['fee']=math.ulp(0.)
        with self.assertRaises(ValueError):self.check(x,y)
    def test_invalid_fees_fail(self):
        for bad in [True,-1.,float('inf'),float('nan'),'50']:
            x=self.sample();y=copy.deepcopy(x);y[0]['fee']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.check(x,y)
    def test_changed_symbols_fail(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['symbols']=['A','C','B']
        with self.assertRaises(ValueError):self.check(x,y)
    def test_changed_score_fails(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['scores'][0]=math.nextafter(.3,math.inf)
        with self.assertRaises(ValueError):self.check(x,y)
    def test_changed_date_fails(self):
        x=self.sample();y=copy.deepcopy(x);y[0]['execution_date']='2000-02-02'
        with self.assertRaises(ValueError):self.check(x,y)
    def test_missing_fields_or_rows_fail(self):
        x=self.sample();y=copy.deepcopy(x);del y[0]['fee']
        with self.assertRaises(ValueError):self.check(x,y)
        with self.assertRaises(ValueError):self.check(x,[])
    def test_more_than_six_terms_fails(self):
        x=self.sample();x[0]['symbols']=list('ABCDEFG');y=copy.deepcopy(x);y[0]['fee']=math.nextafter(50.,math.inf)
        with self.assertRaises(ValueError):self.check(x,y)

if __name__=='__main__':unittest.main(verbosity=2)
