"""Synthetic exact-math tests; no market inputs."""
from fractions import Fraction as F
import math
import unittest
import numpy as np
import monthly_basket_core251_math as m


def fixture():return [100.+j*.031+2*math.sin(j/17) for j in range(253)]
def block_fixture():return [[F(1,50)+F((b+i)%3,1000) for b in range(12)] for i in range(3)]

class MathTests(unittest.TestCase):
    def test_measure_schema_and_boundaries(self):
        p=fixture();z=m.measure(p);self.assertEqual(tuple(z),m.MEASURE_KEYS);self.assertEqual(z['price_count'],253)
        self.assertEqual(m._rational(z['momentum']),F.from_float(p[252])/F.from_float(p[0])-1)
        for b,r in enumerate(z['block_returns']):self.assertEqual(m._rational(r),F.from_float(p[21*(b+1)])/F.from_float(p[21*b])-1)
    def test_interior_does_not_change_endpoint_math(self):
        p=fixture();a=m.measure(p);p[1]*=3;self.assertEqual(a,m.measure(p))
    def test_interior_invalid_still_hard(self):
        p=fixture();p[1]=0
        with self.assertRaises(ValueError):m.measure(p)
    def test_shape(self):
        for p in ([],fixture()[:-1],fixture()+[1],np.ones((253,1)),set(fixture()),'x'):
            with self.subTest(type=type(p)),self.assertRaises(ValueError):m.measure(p)
    def test_missing_invalid(self):
        for bad in (None,float('nan'),float('inf'),-1,0,False,np.bool_(True),'1',1j):
            p=fixture();p[120]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):m.measure(p)
    def test_full_scan(self):
        p=fixture();p[0]=None;p[-1]=0
        with self.assertRaisesRegex(ValueError,'Price 252'):m.measure(p)
    def test_conversion_overflow_underflow(self):
        for bad in (10**1000,F(1,10**1000)):
            p=fixture();p[0]=bad
            with self.assertRaises(ValueError):m.measure(p)
    def test_binary64_exact(self):
        p=[.1]*253;p[-1]=.2;z=m.measure(p);self.assertEqual(z['momentum'],m._pair(F(1)))
    def test_exact_scale(self):
        p=fixture();self.assertEqual(m.measure(p),m.measure([x*2 for x in p]))
    def test_constant_price_valid_zero(self):self.assertEqual(m.measure([1]*253)['momentum'],m._pair(0))
    def test_negative_momentum_measure_valid(self):
        p=[2.]*253;p[-1]=1.;self.assertEqual(m.measure(p)['momentum'],m._pair(F(-1,2)))
    def test_basket_centered_statistics(self):
        rr=block_fixture();z=m.basket(rr);self.assertEqual(tuple(z),m.BASKET_KEYS);q=[sum(row[b] for row in rr)/3 for b in range(12)];N=sum(q)/12
        self.assertEqual(m._rational(z['mean']),N);self.assertEqual(m._rational(z['joint_variance']),sum((v-N)**2 for v in q)/11)
        self.assertEqual(m._rational(z['diagonal_variance']),sum(sum((v-sum(row)/12)**2 for v in row) for row in rr)/99)
    def test_common_variance_zero(self):
        z=m.basket([[F(1,100)]*12]*3);self.assertFalse(z['supported']);self.assertEqual(z['reason'],'zero_both_variances')
    def test_negative_covariance_zero_joint(self):
        x=[F(1,100) if b%2 else F(3,100) for b in range(12)];y=[F(4,100)-v for v in x];z=m.basket([x,y,[F(2,100)]*12]);self.assertEqual(z['reason'],'zero_joint_variance');self.assertGreater(m._rational(z['diagonal_variance']),0)
    def test_basket_permutation(self):
        r=block_fixture();self.assertEqual(m.basket(r),m.basket(r[::-1]))
    def test_constituent_domain(self):
        for r in ([[F(0)]*12]*3,[[F(-1)]*12]*3,[[F(-1,10)]*12]*3):
            with self.assertRaises(ValueError):m.basket(r)
    def test_basket_shape(self):
        for r in ([],block_fixture()[:2],[[F(1)]*11]*3):
            with self.assertRaises(ValueError):m.basket(r)
    def test_compare_exact(self):
        self.assertEqual(m.compare_ratios(F(1),F(1),F(2),F(4)),0);self.assertEqual(m.compare_ratios(F(1),F(1),F(1),F(2)),1)
        self.assertEqual(m.compare_ratios(F(1),F(2),F(1),F(1)),-1)
    def test_compare_ulp_unbounded(self):
        q=F(10**500);self.assertEqual(m.compare_ratios(q+1,F(1),q,F(1)),1)
    def test_compare_invalid(self):
        for a in (F(0),F(-1),1.,True):
            with self.assertRaises(ValueError):m.compare_ratios(a,F(1),F(1),F(1))
    def test_canonical_pairs(self):
        for x in ({'numerator':'01','denominator':'1'},{'numerator':'-0','denominator':'1'},{'numerator':'2','denominator':'2'},{'numerator':1,'denominator':'1'},{'numerator':'1','denominator':'0'}):
            with self.assertRaises(ValueError):m._rational(x)
    def test_inputs_unchanged(self):
        p=fixture();r=block_fixture();pp=p.copy();rr=[x.copy() for x in r];m.measure(p);m.basket(r);self.assertEqual((p,r),(pp,rr))

if __name__=='__main__':unittest.main(verbosity=2)
