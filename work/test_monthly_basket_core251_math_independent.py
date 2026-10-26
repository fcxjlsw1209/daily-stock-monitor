"""Invented prices and rational blocks only; bilateral imports occur only in final test."""
import copy
from fractions import Fraction as F
import math
import sys
import unittest
import numpy as np
import pandas as pd
import monthly_basket_core251_math_independent as subject


def fixture():
    return [float(64+i/8+((i*7)%13)/16) for i in range(253)]


def blocks():
    return [[F(1,16)+F((b+i)%3-1,64) for b in range(12)] for i in range(3)]


def constant_prices(gross=F(17,16)):
    out=[1.]*253
    for b in range(12):
        for j in range(21*b,21*(b+1)):out[j]=float(gross**b)
    out[-1]=float(gross**12);return out

class MathTests(unittest.TestCase):
    def test_clock_telescoping_and_interior_support(self):
        p=fixture();m=subject.measure(p);rs=list(map(subject.rational,m['block_returns']));product=F(1)
        for r in rs:product*=1+r
        self.assertEqual(product,1+subject.rational(m['momentum']));self.assertEqual(len(rs),12)
        self.assertEqual(subject.rational(m['momentum']),F.from_float(p[252])/F.from_float(p[0])-1)
        p[1]=None
        with self.assertRaises(ValueError):subject.measure(p)

    def test_window_shape(self):
        for p in (fixture()[:-1],fixture()+[1.],[[1.]*253]):
            with self.assertRaises(ValueError):subject.measure(p)

    def test_all_invalid_source_types(self):
        for bad in (None,pd.NA,np.nan,True,np.bool_(False),'1',float('inf'),-1.,0.,-0.,10**1000,F(1,10**1000)):
            p=fixture();p[99]=bad
            with self.assertRaises(ValueError):subject.measure(p)

    def test_all_source_is_scanned(self):
        p=fixture();p[0]=None;p[-1]=float('inf')
        with self.assertRaisesRegex(ValueError,'252'):subject.measure(p)

    def test_extreme_exact_gains_do_not_overflow(self):
        p=[1.]*253;p[0]=float.fromhex('0x0.0000000000001p-1022');p[-1]=float.fromhex('0x1.fffffffffffffp+1023')
        out=subject.measure(p);m=subject.rational(out['momentum'])
        self.assertGreater(m,F(10)**600)

    def test_no_source_input_mutation(self):
        p=fixture();before=copy.deepcopy(p);subject.measure(p);self.assertEqual(p,before)

    def test_exact_representable_common_scale_invariance(self):
        p=fixture();self.assertEqual(subject.measure(p),subject.measure([x*8 for x in p]))

    def test_constant_gross_means_positive_zero_risk(self):
        m=subject.measure(constant_prices());b=subject.basket([m['block_returns']]*3)
        self.assertEqual(subject.rational(b['mean']),F(1,16));self.assertEqual(b['reason'],'zero_both_variances')
        self.assertFalse(b['supported']);self.assertGreater(subject.rational(m['momentum']),0)

    def test_second_moment_covariance_identity(self):
        r=blocks();b=subject.basket(r);means=[sum(x)/12 for x in r]
        cov=[[sum((r[i][k]-means[i])*(r[j][k]-means[j]) for k in range(12))/11 for j in range(3)] for i in range(3)]
        self.assertEqual(subject.rational(b['joint_variance']),sum(map(sum,cov))/9)
        self.assertEqual(subject.rational(b['diagonal_variance']),sum(cov[i][i] for i in range(3))/9)

    def test_exact_cancellation_shared_domain(self):
        r=[[F(1,16)+(F(1,64) if b%2 else -F(1,64)) for b in range(12)],
           [F(1,16)+(-F(1,64) if b%2 else F(1,64)) for b in range(12)],[F(1,16)]*12]
        out=subject.basket(r);self.assertEqual(out['reason'],'zero_joint_variance');self.assertFalse(out['supported'])
        self.assertGreater(subject.rational(out['diagonal_variance']),0)

    def test_identical_series_covariance_upper_bound(self):
        r=blocks()[0];out=subject.basket([r]*3)
        self.assertEqual(subject.rational(out['joint_variance']),3*subject.rational(out['diagonal_variance']))

    def test_positive_annual_gate_rejects_negative_or_zero(self):
        for row in ([F(0)]*12,[F(-1,100)]*12):
            with self.assertRaises(ValueError):subject.basket([row]*3)

    def test_negative_block_allowed_with_positive_annual_product(self):
        row=[F(1,2),F(-1,4)]*6;out=subject.basket([row]*3)
        self.assertGreater(subject.rational(out['mean']),0);self.assertTrue(out['supported'])

    def test_invalid_block_shape_or_gross(self):
        for r in (blocks()[:2],[x[:11] for x in blocks()],[[F(-1)]+x[1:] for x in blocks()]):
            with self.assertRaises(ValueError):subject.basket(r)

    def test_common_block_and_member_permutation(self):
        r=blocks();b=subject.basket(r);rev=subject.basket([x[::-1] for x in r[::-1]])
        for key in ('mean','joint_variance','diagonal_variance','supported','reason'):self.assertEqual(b[key],rev[key])
        self.assertEqual(b['portfolio_blocks'][::-1],rev['portfolio_blocks'])

    def test_arbitrarily_tiny_exact_positive_risk(self):
        epsilon=F(1,10**400);row=[F(1,16)+epsilon*(b%2) for b in range(12)]
        out=subject.basket([row]*3);self.assertTrue(out['supported']);self.assertGreater(subject.rational(out['joint_variance']),0)
        self.assertEqual(float(subject.rational(out['joint_variance'])),0.)

    def test_ratio_compare_ties_and_near_ties(self):
        self.assertEqual(subject.compare_ratios(F(1),F(4),F(2),F(16)),0)
        self.assertEqual(subject.compare_ratios(F(1)+F(1,10**500),F(4),F(2),F(16)),1)
        self.assertEqual(subject.compare_ratios(F(1),F(4),F(1),F(4)-F(1,10**500)),-1)

    def test_ratio_compare_requires_strict_positive(self):
        for args in ((0,1,1,1),(1,0,1,1),(-1,1,1,1)):
            with self.assertRaises(ValueError):subject.compare_ratios(*map(F,args))

    def test_canonical_pairs_reject_float_boolean_and_unreduced(self):
        for value in (1.,True,dict(numerator='2',denominator='4'),dict(numerator='-0',denominator='1'),dict(numerator='1',denominator='0')):
            with self.assertRaises(ValueError):subject.rational(value)

    def test_verify_full_audit_tamper(self):
        p=fixture();m=subject.measure(p);subject.verify_measure(p,m);m['momentum']['numerator']='999'
        with self.assertRaises(ValueError):subject.verify_measure(p,m)
        b=subject.basket(blocks());b['supported']=1
        with self.assertRaises(ValueError):subject.verify_basket(blocks(),b)

class SoleProducerCross(unittest.TestCase):
    def test_pure_math_fixtures_and_extremes(self):
        import monthly_basket_core251_math as producer
        import test_monthly_basket_core251_math as fixtures
        for p in (fixtures.fixture(),fixture(),constant_prices(),[1.]*253):subject.exact(producer.measure(p),subject.measure(p))
        for r in (fixtures.block_fixture(),blocks(),[[F(1,16)]*12]*3):subject.exact(producer.basket(r),subject.basket(r))
        for a in ((F(1),F(4),F(2),F(16)),(F(1)+F(1,10**500),F(4),F(2),F(16))):self.assertEqual(producer.compare_ratios(*a),subject.compare_ratios(*a))

if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only');result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MathTests));sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
