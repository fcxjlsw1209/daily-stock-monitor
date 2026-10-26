"""Synthetic-only exact reference gain and strict source-shape tests."""
from copy import deepcopy
from fractions import Fraction as F
import json, math, sys, unittest
import numpy as np
import pandas as pd
import monthly_reference_gain250_math as m

def fixture():
    return ([1.+(q%17)/32 for q in range(253)],[1.+q%7 for q in range(252)])
def fraction(pair):return F(int(pair['numerator']),int(pair['denominator']))
def gains(z):return fraction(z['raw_volume_gain']),fraction(z['raw_equal_gain'])

class SourceTests(unittest.TestCase):
    def test_schema_and_json(self):
        z=m.measure(*fixture());self.assertEqual(tuple(z),m.MEASURE_KEYS)
        self.assertTrue(z['available']);self.assertEqual((z['price_count'],z['volume_count'],z['historical_count']),(253,252,252))
        json.dumps(z,allow_nan=False)
    def test_exact_shapes_and_no_reference_volume(self):
        p,v=fixture()
        for a,b in ((p[:-1],v),(p+[1.],v),(p,v[:-1]),(p,v+[1.]),(np.ones((253,1)),v),(p,np.ones((1,252)))):
            with self.assertRaises(ValueError):m.measure(a,b)
    def test_sequence_schema(self):
        p,v=fixture()
        for x in ('1'*253,bytes(253),{i:1. for i in range(253)},set(range(253)),iter(p)):
            with self.assertRaises(ValueError):m.measure(x,v)
    def test_all_missing_locations_preserved(self):
        p,v=fixture();p[0]=None;p[252]=np.float64('nan');v[0]=pd.NA;v[251]=float('nan')
        z=m.measure(p,v);self.assertFalse(z['available']);self.assertEqual(z['missing_price_indices'],[0,252]);self.assertEqual(z['missing_volume_indices'],[0,251])
        for k in m.MEASURE_KEYS[8:]:self.assertIsNone(z[k])
        json.dumps(z,allow_nan=False)
    def test_missing_never_hides_any_invalid(self):
        p,v=fixture();p[0]=None;p[252]=float('inf');v[1]='1';v[251]=-1.
        with self.assertRaises(ValueError) as c:m.measure(p,v)
        for label in ('prices[252]','volumes[1]','volumes[251]'):self.assertIn(label,str(c.exception))
    def test_nonreal_boolean_nonpositive_and_inf(self):
        for bad in (True,np.bool_(False),'1',complex(1,0),pd.NaT,np.datetime64('NaT'),0.,-0.,-1.,float('inf'),-float('inf')):
            p,v=fixture();p[17]=bad
            with self.subTest(price=repr(bad)),self.assertRaises(ValueError):m.measure(p,v)
            p,v=fixture();v[91]=bad
            with self.subTest(volume=repr(bad)),self.assertRaises(ValueError):m.measure(p,v)
    def test_conversion_overflow_and_nonzero_underflow(self):
        for bad in (10**1000,F(1,10**1000)):
            for which in ('p','v'):
                p,v=fixture();(p if which=='p' else v)[1]=bad
                with self.subTest(which=which),self.assertRaises(ValueError):m.measure(p,v)
    def test_array_equal_and_inputs_unchanged(self):
        p,v=fixture();before=deepcopy((p,v));z=m.measure(p,v)
        self.assertEqual(z,m.measure(np.asarray(p),np.asarray(v)));self.assertEqual((p,v),before)
    def test_first_last_history_and_reference_consumed(self):
        p,v=fixture()
        for q in (0,251,252):
            a=list(p);a[q]=None;self.assertFalse(m.measure(a,v)['available'])
        for q in (0,251):
            b=list(v);b[q]=None;self.assertFalse(m.measure(p,b)['available'])
    def test_binary64_cast_before_fraction(self):
        p,v=fixture();p[0]=2**53+1;q=list(p);q[0]=float(q[0]);self.assertEqual(m.measure(p,v),m.measure(q,v))

class ExactMathTests(unittest.TestCase):
    def test_hand_computable_weighted_equal(self):
        p=[1.]*126+[3.]*126+[4.];v=[1.]*126+[3.]*126;z=m.measure(p,v)
        self.assertEqual(fraction(z['weighted_reference']),F(5,2));self.assertEqual(fraction(z['equal_reference']),F(2))
        self.assertEqual(gains(z),(F(3,8),F(1,2)))
    def test_constant_volume(self):
        p,v=fixture();z=m.measure(p,[3.]*252);self.assertEqual(*gains(z))
    def test_constant_history_distinct_reference(self):
        p,v=fixture();z=m.measure([3.]*252+[6.],v);self.assertEqual(gains(z),(F(1,2),F(1,2)))
    def test_all_constant_prices_raw_zero_valid(self):
        p,v=fixture();z=m.measure([3.]*253,v);self.assertTrue(z['available']);self.assertEqual(gains(z),(F(0),F(0)))
    def test_covariance_difference_identity(self):
        p,v=fixture();z=m.measure(p,v);ps=[F.from_float(x) for x in p[:-1]];vs=[F.from_float(x) for x in v]
        mp=sum(ps,F(0))/252;mv=sum(vs,F(0))/252;cov=sum(((a-mp)*(b-mv) for a,b in zip(ps,vs)),F(0))/252
        a,b=gains(z);self.assertEqual(a-b,-cov/(mv*F.from_float(p[-1])))
    def test_exact_scaling_only_gain_invariance(self):
        p,v=fixture();z=m.measure(p,v);w=m.measure([x*8 for x in p],[x*4 for x in v]);self.assertEqual(gains(z),gains(w))
        self.assertEqual(fraction(w['weighted_reference']),8*fraction(z['weighted_reference']))
    def test_paired_permutation_and_reference_fixed(self):
        p,v=fixture();z=m.measure(p,v);self.assertEqual(z,m.measure(p[:-1][::-1]+p[-1:],v[::-1]))
    def test_reference_monotonicity(self):
        p,v=fixture();z=m.measure(p,v);p[-1]*=2;w=m.measure(p,v)
        self.assertTrue(all(b>a for a,b in zip(gains(z),gains(w))))
    def test_reference_excluded_from_historical_average(self):
        p=[1.]*252+[3.];v=[2.]*252;z=m.measure(p,v)
        self.assertEqual(fraction(z['historical_price_sum']),252);self.assertEqual(fraction(z['weighted_price_sum']),504)
        self.assertEqual(gains(z),(F(2,3),F(2,3)))
    def test_negative_gain_unbounded_without_raw_float(self):
        p=[sys.float_info.max]*252+[math.nextafter(0.,1.)];v=[sys.float_info.max]*252;z=m.measure(p,v)
        self.assertTrue(all(x<-F.from_float(sys.float_info.max) for x in gains(z)))
        json.dumps(z,allow_nan=False);self.assertEqual(m.rank49([z['raw_volume_gain']]*49)['targets'],[float(F(1,98))]*49)
    def test_exact_small_gain_and_strict_bound(self):
        p=[1.]*252+[math.nextafter(1.,math.inf)];v=[1.]*252;z=m.measure(p,v)
        expected=(F.from_float(p[-1])-1)/F.from_float(p[-1]);self.assertEqual(gains(z),(expected,expected))
        self.assertTrue(all(x<1 for x in gains(z)))
    def test_rank_equivalence_to_reference_ratio(self):
        p,v=fixture();zs=[]
        for q in range(49):
            pp=list(p);pp[-1]=1.+q/16;zs.append(m.measure(pp,v))
        for raw,reference in [('raw_volume_gain','weighted_reference'),('raw_equal_gain','equal_reference')]:
            h=[fraction(z['reference_price'])/fraction(z[reference])-1 for z in zs]
            self.assertEqual(m.rank49([z[raw] for z in zs]),m.rank49(h))

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
