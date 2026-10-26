"""Synthetic-only source contracts, exact covariance arithmetic and49-way ranks."""
from copy import deepcopy
from fractions import Fraction as F
import json
import math
import sys
import unittest
import numpy as np
import pandas as pd
import monthly_volume_breadth249_math as m


def fixture():
    return ([1.0 + (q % 17) / 32.0 for q in range(253)],
            [1.0 + (q % 7) for q in range(315)])


def signed_fixture(signs):
    volume = [1.0] * 63 + [2.0 if q % 3 == 0 else 1.0 for q in range(252)]
    price = [1.0]
    for q in range(252):
        factor = (2.0 if signs[q//21] > 0 else .5 if signs[q//21] < 0 else 1.0) if q % 3 == 0 else 1.0
        price.append(price[-1] * factor)
    return price, volume


def fraction(pair):
    return F(int(pair['numerator']), int(pair['denominator']))


class SourceTests(unittest.TestCase):
    def test_complete_schema_and_json(self):
        z=m.measure(*fixture())
        self.assertEqual(tuple(z),m.MEASURE_KEYS)
        self.assertTrue(z['available']);self.assertEqual(z['reason'],'available')
        self.assertEqual((z['price_count'],z['volume_count'],z['return_count'],z['block_count']),(253,315,252,12))
        self.assertEqual(len(z['high_flags']),252);self.assertEqual(len(z['blocks']),12)
        for b in z['blocks']:self.assertEqual(tuple(b),m.BLOCK_KEYS)
        json.dumps(z,allow_nan=False)

    def test_exact_shapes(self):
        p,v=fixture()
        for a,b in ((p[:-1],v),(p+[1.],v),(p,v[:-1]),(p,v+[1.]),(np.ones((253,1)),v),(p,np.ones((1,315)))):
            with self.assertRaises(ValueError):m.measure(a,b)

    def test_no_unordered_text_mapping_or_generator_inputs(self):
        p,v=fixture()
        for bad in ('1'*253,bytes(253),{i:1. for i in range(253)},set(range(253)),iter(p)):
            with self.assertRaises(ValueError):m.measure(bad,v)

    def test_none_real_nan_and_pandas_na_are_missing(self):
        p,v=fixture();p[0]=None;p[252]=np.float64('nan');v[0]=pd.NA;v[314]=float('nan')
        z=m.measure(p,v)
        self.assertFalse(z['available']);self.assertEqual(z['reason'],'missing_source')
        self.assertEqual(z['missing_price_indices'],[0,252]);self.assertEqual(z['missing_volume_indices'],[0,314])
        self.assertEqual(z['high_flags'],[]);self.assertEqual(z['blocks'],[])
        for k in ('raw_breadth','raw_mean','positive_block_count','zero_block_count','negative_block_count'):self.assertIsNone(z[k])
        json.dumps(z,allow_nan=False)

    def test_missing_does_not_hide_later_price_invalid(self):
        p,v=fixture();p[0]=None;p[252]=float('inf')
        with self.assertRaisesRegex(ValueError,r'prices\[252\]: nonfinite'):m.measure(p,v)

    def test_missing_price_does_not_hide_volume_invalid(self):
        p,v=fixture();p[0]=None;v[314]=False
        with self.assertRaisesRegex(ValueError,r'volumes\[314\]'):m.measure(p,v)

    def test_missing_volume_does_not_hide_price_invalid(self):
        p,v=fixture();p[252]=-1.;v[0]=None
        with self.assertRaisesRegex(ValueError,r'prices\[252\]'):m.measure(p,v)

    def test_entire_both_windows_scanned_for_invalid(self):
        p,v=fixture();p[0]=None;p[2]=-1;v[3]='1';v[314]=float('inf')
        with self.assertRaises(ValueError) as exc:m.measure(p,v)
        self.assertIn('prices[2]',str(exc.exception));self.assertIn('volumes[3]',str(exc.exception));self.assertIn('volumes[314]',str(exc.exception))

    def test_bool_nonreal_nat_nonpositive_and_inf_hard(self):
        for bad in (True,np.bool_(False),'1',complex(1,0),pd.NaT,np.datetime64('NaT'),0.,-0.,-1.,float('inf'),-float('inf')):
            p,v=fixture();p[17]=bad
            with self.subTest(bad=repr(bad)),self.assertRaises(ValueError):m.measure(p,v)
            p,v=fixture();v[91]=bad
            with self.subTest(volume=repr(bad)),self.assertRaises(ValueError):m.measure(p,v)

    def test_unrepresentable_integer_and_fraction_underflow_hard(self):
        for bad in (10**1000,F(1,10**1000)):
            p,v=fixture();p[1]=bad
            with self.assertRaises(ValueError):m.measure(p,v)

    def test_numpy_real_inputs_equal_lists(self):
        p,v=fixture()
        self.assertEqual(m.measure(np.array(p),np.array(v)),m.measure(p,v))

    def test_inputs_not_mutated(self):
        p,v=fixture();before=deepcopy((p,v));m.measure(p,v)
        self.assertEqual((p,v),before)


class ExactMathTests(unittest.TestCase):
    def test_positive_blocks_known_fraction(self):
        z=m.measure(*signed_fixture([1]*12))
        self.assertEqual(fraction(z['raw_breadth']),F(1,2));self.assertEqual(fraction(z['raw_mean']),F(2,9))
        self.assertEqual(z['positive_block_count'],12)
        for b in z['blocks']:
            self.assertEqual((b['high_count'],fraction(b['return_sum']),fraction(b['high_return_sum'])),(7,F(7),F(7)))
            self.assertEqual(fraction(b['covariance']),F(2,9))

    def test_negative_blocks_known_fraction(self):
        z=m.measure(*signed_fixture([-1]*12))
        self.assertEqual(fraction(z['raw_breadth']),F(-1,2));self.assertEqual(fraction(z['raw_mean']),F(-1,9))
        self.assertEqual(z['negative_block_count'],12)

    def test_breadth_zero_does_not_imply_mean_zero(self):
        z=m.measure(*signed_fixture([1]*6+[-1]*6))
        self.assertEqual(fraction(z['raw_breadth']),0);self.assertEqual(fraction(z['raw_mean']),F(1,18))
        self.assertEqual((z['positive_block_count'],z['zero_block_count'],z['negative_block_count']),(6,0,6))

    def test_zero_blocks_half_credit(self):
        z=m.measure(*signed_fixture([1]*4+[0]*4+[-1]*4))
        self.assertEqual(fraction(z['raw_breadth']),0);self.assertEqual(fraction(z['raw_mean']),F(1,27))
        self.assertEqual((z['positive_block_count'],z['zero_block_count'],z['negative_block_count']),(4,4,4))

    def test_constant_volume_and_constant_return_supported(self):
        p,v=fixture()
        for a,b in ((p,[1.]*315),([1.]*253,v),([2.**q for q in range(253)],v)):
            z=m.measure(a,b);self.assertTrue(z['available']);self.assertEqual(z['zero_block_count'],12)
            self.assertEqual(z['raw_breadth'],{'numerator':'0','denominator':'1'});self.assertEqual(z['raw_mean'],z['raw_breadth'])

    def test_centered_fraction_reference(self):
        p,v=fixture();z=m.measure(p,v)
        r=[F.from_float(p[q+1])/F.from_float(p[q])-1 for q in range(252)]
        for k,b in enumerate(z['blocks']):
            lo=21*k;h=z['high_flags'][lo:lo+21];x=r[lo:lo+21];mh=F(sum(h),21);mr=sum(x,F(0))/21
            c=sum(((F(a)-mh)*(q-mr) for a,q in zip(h,x)),F(0))/21
            self.assertEqual(fraction(b['covariance']),c)

    def test_first_last_window_and_strict_median_ties(self):
        p=[1.]*253;v=[1.]*315;v[63]=2.;v[314]=2.
        z=m.measure(p,v);self.assertEqual([i for i,x in enumerate(z['high_flags']) if x],[0,251])
        self.assertEqual(z['blocks'][0]['first_return_index'],0);self.assertEqual(z['blocks'][-1]['last_return_index'],251)
        # Prior63 has32 low and31 high: including current would change the median.
        v=[1.]*32+[3.]*31+[2.]*252
        self.assertEqual(m.measure(p,v)['high_flags'][0],1)
        v[63]=1.;self.assertEqual(m.measure(p,v)['high_flags'][0],0)

    def test_initial_volume_and_final_volume_are_consumed(self):
        p,v=fixture();v[0]=None;self.assertFalse(m.measure(p,v)['available'])
        p,v=fixture();v[-1]=None;self.assertFalse(m.measure(p,v)['available'])

    def test_binary64_cast_precedes_fraction(self):
        p,v=fixture();p[0]=2**53+1
        q=list(p);q[0]=float(p[0]);self.assertEqual(m.measure(p,v),m.measure(q,v))

    def test_sub_ulp_return_not_rounded_before_rational(self):
        p,v=signed_fixture([0]*12);p[1]=math.nextafter(1.,math.inf)
        z=m.measure(p,v)
        expected=F.from_float(p[1])-1
        self.assertEqual(fraction(z['blocks'][0]['high_return_sum']),expected)
        self.assertNotEqual(fraction(z['blocks'][0]['covariance']),0)

    def test_extreme_finite_sources_raw_not_cast_to_float(self):
        p,v=fixture();p=[sys.float_info.max if q%2 else math.nextafter(0.,1.) for q in range(253)]
        z=m.measure(p,v);self.assertTrue(z['available']);json.dumps(z,allow_nan=False)
        self.assertTrue(any(abs(fraction(b['covariance']))>F.from_float(sys.float_info.max) for b in z['blocks']))
        # These exact raw values remain legal inputs to the bounded rank conversion.
        r=m.rank49([z['raw_mean']]*49);self.assertEqual(r['targets'],[float(F(1,98))]*49)

    def test_power_of_two_unit_scaling_preserves_measure(self):
        p,v=fixture();z=m.measure(p,v)
        self.assertEqual(z,m.measure([x*8 for x in p],[x*4 for x in v]))


class RankTests(unittest.TestCase):
    def test_all_tie_is_one_over98(self):
        z=m.rank49([F(0)]*49);self.assertEqual(tuple(z),m.RANK_KEYS)
        self.assertEqual(z['less'],[0]*49);self.assertEqual(z['equal'],[49]*49)
        self.assertEqual(z['targets'],[float(F(1,98))]*49)

    def test_unique_ranks_and_full49_domain(self):
        z=m.rank49([F(q) for q in range(49)])
        self.assertEqual(z['less'],list(range(49)));self.assertEqual(z['equal'],[1]*49)
        self.assertEqual(z['targets'],[float(F(2*q-47,98)) for q in range(49)])
        self.assertEqual(z['targets'][-1],.5)

    def test_exact_tie_group_and_zero_target(self):
        values=[F(-1)]*24+[F(0)]*2+[F(1)]*23;z=m.rank49(values)
        self.assertEqual(z['less'][24:26],[24,24]);self.assertEqual(z['equal'][24:26],[2,2])
        # A two-member tie at positions24 and25 yields exactly0.
        values=[F(-1)]*23+[F(0)]*2+[F(1)]*24
        self.assertEqual(m.rank49(values)['targets'][23:25],[0.,0.])

    def test_close_distinct_rationals_never_float_tie(self):
        x=F(1);y=x+F(1,10**400);values=[x]*24+[y]*25;z=m.rank49(values)
        self.assertEqual(float(x),float(y));self.assertEqual(z['equal'],[24]*24+[25]*25)
        self.assertLess(z['targets'][0],z['targets'][-1])

    def test_pairs_fraction_equivalence_and_no_mutation(self):
        values=[{'numerator':str(q),'denominator':'1'} for q in range(49)];before=deepcopy(values)
        self.assertEqual(m.rank49(values),m.rank49([F(q) for q in range(49)]));self.assertEqual(values,before)

    def test_canonical_pairs_enforced(self):
        bads=[{'numerator':'01','denominator':'1'},{'numerator':'+1','denominator':'1'},
              {'numerator':'-0','denominator':'1'},{'numerator':'2','denominator':'2'},
              {'numerator':'0','denominator':'2'},{'numerator':'1','denominator':'-1'},
              {'numerator':'1','denominator':'0'},{'numerator':1,'denominator':'1'},
              {'numerator':'1','denominator':'1','extra':0}]
        for bad in bads:
            values=[F(0)]*49;values[3]=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):m.rank49(values)

    def test_rank_rejects_missing_float_integer_and_wrong_domain(self):
        for bad in (None,float('nan'),0.,1,True):
            values=[F(0)]*49;values[-1]=bad
            with self.subTest(bad=repr(bad)),self.assertRaises(ValueError):m.rank49(values)
        for n in (0,48,50):
            with self.assertRaises(ValueError):m.rank49([F(0)]*n)

    def test_rank_permutation_equivariance(self):
        values=[F(q%25,24)-F(1,2) for q in range(49)];z=m.rank49(values);r=m.rank49(values[::-1])
        for k in ('less','equal','targets'):self.assertEqual(r[k],z[k][::-1])


if __name__=='__main__':unittest.main(verbosity=2)
