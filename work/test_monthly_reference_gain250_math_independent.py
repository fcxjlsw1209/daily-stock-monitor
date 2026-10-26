"""Synthetic exact reference-gain math only; no source loaders or I/O."""
import ast
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import monthly_reference_gain250_math_independent as subject

def fixture():
    return [float(20+(i*13)%37)/8 for i in range(253)], [float(10+(i*17)%31) for i in range(252)]

def value(pair): return Fraction(int(pair['numerator']),int(pair['denominator']))

class MeasureTests(unittest.TestCase):
    def test_fixed_schema_clock_and_complete_audit(self):
        p,v=fixture();out=subject.measure_oracle(p,v)
        self.assertEqual(tuple(out),subject.MEASURE_KEYS)
        self.assertEqual([out[k] for k in ('price_count','volume_count','historical_count')],[253,252,252])
        self.assertTrue(subject.check_measure(p,v,out)['passed'])

    def test_reference_is_excluded_from_both_historical_references(self):
        p=[2.]*252+[4.];v=[3.]*252;a=subject.measure_oracle(p,v)
        self.assertEqual(value(a['weighted_reference']),2);self.assertEqual(value(a['equal_reference']),2)
        self.assertEqual(value(a['raw_volume_gain']),Fraction(1,2))
        p[-1]=8.;b=subject.measure_oracle(p,v)
        self.assertEqual(a['weighted_reference'],b['weighted_reference'])
        self.assertEqual(value(b['raw_volume_gain']),Fraction(3,4))
        with self.assertRaises(ValueError):subject.measure_oracle(p,v+[0.])

    def test_constant_volume_exact_arm_equality(self):
        p,v=fixture();out=subject.measure_oracle(p,[7.]*252)
        self.assertEqual(out['weighted_reference'],out['equal_reference'])
        self.assertEqual(out['raw_volume_gain'],out['raw_equal_gain'])

    def test_constant_historical_price_and_all_constant_zero_are_available(self):
        _,v=fixture()
        for reference,wanted in ((4.,Fraction(1,2)),(2.,Fraction(0)),(1.,Fraction(-1))):
            out=subject.measure_oracle([2.]*252+[reference],v)
            self.assertTrue(out['available']);self.assertEqual(value(out['raw_volume_gain']),wanted)
            self.assertEqual(out['raw_volume_gain'],out['raw_equal_gain'])

    def test_exact_covariance_contrast_identity(self):
        p,v=fixture();out=subject.measure_oracle(p,v)
        pp=list(map(Fraction.from_float,p[:-1]));vv=list(map(Fraction.from_float,v))
        mp=sum(pp)/252;mv=sum(vv)/252
        cov=sum((x-mp)*(w-mv) for x,w in zip(pp,vv))/252
        self.assertEqual(value(out['raw_volume_gain'])-value(out['raw_equal_gain']),-cov/(mv*Fraction.from_float(p[-1])))

    def test_weight_alignment_first_last_and_paired_permutation(self):
        p=[1.]*252+[4.];p[0]=2.;p[251]=3.;v=[1.]*252;v[0]=4.;v[-1]=7.
        out=subject.measure_oracle(p,v)
        self.assertEqual(value(out['weighted_price_sum']),250+8+21)
        perm=list(range(251,-1,-1));other=subject.measure_oracle([p[i] for i in perm]+[p[-1]],[v[i] for i in perm])
        self.assertEqual(out,other)
        wrong=subject.measure_oracle(p,v[::-1]);self.assertNotEqual(out['raw_volume_gain'],wrong['raw_volume_gain'])

    def test_exact_common_scaling_and_input_preservation(self):
        p,v=fixture();before=deepcopy((p,v));out=subject.measure_oracle(p,v)
        scaled=subject.measure_oracle([x*8 for x in p],[x*4 for x in v])
        for k in ('raw_volume_gain','raw_equal_gain'):self.assertEqual(out[k],scaled[k])
        self.assertEqual((p,v),before)

    def test_reference_monotonicity_and_conventional_ratio_rank_equivalence(self):
        p,v=fixture();out=subject.measure_oracle(p,v);p[-1]*=2;other=subject.measure_oracle(p,v)
        for k in ('raw_volume_gain','raw_equal_gain'):self.assertGreater(value(other[k]),value(out[k]))
        gs=[];hs=[]
        for ref in range(1,50):
            x=subject.measure_oracle([2.]*252+[float(ref)],[1.]*252)
            gs.append(value(x['raw_volume_gain']));hs.append(Fraction(ref,2)-1)
        self.assertEqual(subject.rank49_oracle(gs),subject.rank49_oracle(hs))

    def test_missing_source_full_scan_and_no_partial_sums(self):
        p,v=fixture();p[0]=None;p[-1]=np.nan;v[0]=pd.NA;v[-1]=float('nan')
        out=subject.measure_oracle(p,v)
        self.assertFalse(out['available']);self.assertEqual(out['missing_price_indices'],[0,252]);self.assertEqual(out['missing_volume_indices'],[0,251])
        for k in subject.MEASURE_KEYS[8:]:self.assertIsNone(out[k])
        seen=[]
        class Probe(float):
            def __float__(self):seen.append('read');return float.__float__(self)
        p,v=fixture();p[0]=None;p[9]=0.;p[-1]=Probe(1.);v[-1]=Probe(1.)
        with self.assertRaises(ValueError):subject.measure_oracle(p,v)
        self.assertEqual(seen,['read','read'])

    def test_hard_invalid_zero_underflow_and_types(self):
        bad=[True,np.bool_(False),'1',pd.NaT,complex(1),float('inf'),-float('inf'),0.,-0.,-1.,10**1000,Fraction(1,10**1000)]
        for x in bad:
            for which in (0,1):
                with self.subTest(x=repr(x)[:30],which=which):
                    args=list(fixture());args[which][100]=x
                    with self.assertRaises(ValueError):subject.measure_oracle(*args)

    def test_exact_shapes_and_binary64_admission(self):
        p,v=fixture()
        for a,b in ((p[:-1],v),(p+[1.],v),(p,v[:-1]),(p,v+[1.]),([[x] for x in p],v)):
            with self.assertRaises(ValueError):subject.measure_oracle(a,b)
        p[0]=Fraction(1,3);v[0]=Fraction(2,3)
        self.assertEqual(subject.measure_oracle(p,v),subject.measure_oracle([float(x) for x in p],[float(x) for x in v]))

    def test_extreme_raw_gain_never_cast_to_float(self):
        tiny=math.ldexp(1.,-1074);huge=sys.float_info.max
        out=subject.measure_oracle([huge]*252+[tiny],[huge]*252)
        self.assertLess(value(out['raw_volume_gain']),-Fraction.from_float(huge));self.assertTrue(out['available'])
        self.assertEqual(out['raw_volume_gain'],out['raw_equal_gain'])
        subject.rank49_oracle([out['raw_volume_gain']]*49)

    def test_exact_raw_differences_survive_rounded_raw_gain_collapse(self):
        values=[]
        for j in range(49):
            p=[1.]*252+[sys.float_info.max];p[j]=math.nextafter(1.,2.)
            v=[1.]*252;v[j]=float(j+1)
            values.append(value(subject.measure_oracle(p,v)['raw_volume_gain']))
        self.assertEqual(len(set(map(float,values))),1);self.assertEqual(len(set(values)),49)
        self.assertEqual(len(set(subject.rank49_oracle(values)['targets'])),49)

    def test_measure_checker_rejects_every_algebra_and_type_tamper(self):
        p,v=fixture();expected=subject.measure_oracle(p,v)
        for key in subject.MEASURE_KEYS[8:]:
            wrong=deepcopy(expected);wrong[key]={'numerator':'0','denominator':'1'}
            with self.assertRaises(ValueError):subject.check_measure(p,v,wrong)
        for key,val in (('available',1),('price_count',252),('volume_count',253)):
            wrong=deepcopy(expected);wrong[key]=val
            with self.assertRaises(ValueError):subject.check_measure(p,v,wrong)

class RankTests(unittest.TestCase):
    def test_ascending_reverse_and_range(self):
        values = [Fraction(i) for i in range(49)]
        result = subject.rank49_oracle(values)
        self.assertEqual(tuple(result),subject.RANK_KEYS)
        self.assertEqual(result['less'],list(range(49))); self.assertEqual(result['equal'],[1]*49)
        self.assertEqual(result['targets'],[float(Fraction(i+1,49)-Fraction(1,2)) for i in range(49)])
        self.assertEqual(subject.rank49_oracle(values[::-1])['targets'],result['targets'][::-1])
        self.assertEqual(result['targets'][-1],.5)

    def test_all_tied_and_canonical_pair_input(self):
        a = subject.rank49_oracle([Fraction(0)]*49)
        b = subject.rank49_oracle([subject.pair(Fraction(-7,13))]*49)
        self.assertEqual(a,b); self.assertEqual(a['targets'],[float(Fraction(1,98))]*49)

    def test_exact_ties_and_zero_target(self):
        values = [Fraction(i) for i in range(49)]; values[24] = values[23]
        result = subject.rank49_oracle(values)
        self.assertEqual(result['less'][23:25],[23,23]); self.assertEqual(result['equal'][23:25],[2,2])
        self.assertEqual(result['targets'][23:25],[0.,0.])

    def test_no_float_or_implicit_integer_raw_ranking(self):
        for bad in (0.,1,True,np.float64(1.),None):
            values = [Fraction(0)]*49; values[7] = bad
            with self.assertRaises(ValueError): subject.rank49_oracle(values)
        for width in (0,48,50):
            with self.assertRaises(ValueError): subject.rank49_oracle([Fraction(0)]*width)

    def test_canonical_rational_encoding_is_strict(self):
        bad = [
            {'numerator':'2','denominator':'2'},{'numerator':'-0','denominator':'1'},
            {'numerator':'+1','denominator':'1'},{'numerator':'01','denominator':'1'},
            {'numerator':'1','denominator':'01'},{'numerator':'1','denominator':'0'},
            {'numerator':'1','denominator':'-2'},{'numerator':1,'denominator':'1'},
            {'numerator':'1','denominator':'1','extra':0},{'numerator':'1'},
            {'numerator':' 1','denominator':'1'}]
        for pair in bad:
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError): subject.rank49_oracle([pair]*49)

    def test_tiny_exact_differences_survive_raw_float_collapse(self):
        values = [Fraction(2**1000)+Fraction(i,2**1000) for i in range(49)]
        self.assertEqual(float(values[0]),float(values[-1]))
        self.assertEqual(subject.rank49_oracle(values)['less'],list(range(49)))

    def test_permutation_equivariance(self):
        values = [Fraction((i*17)%25,24) for i in range(49)]
        permutation = list(range(0,49,2))+list(range(1,49,2))
        first = subject.rank49_oracle(values); other = subject.rank49_oracle([values[i] for i in permutation])
        for key in ('less','equal','targets'):
            self.assertEqual(other[key],[first[key][i] for i in permutation])

    def test_rank_checker_rejects_float_bytes_and_discrete_tamper(self):
        values = [Fraction(i) for i in range(49)]; values[24] = values[23]
        good = subject.rank49_oracle(values); receipt=subject.check_rank49(values,good)
        self.assertTrue(all(receipt[k] is True for k in subject.RANK_FLAGS))
        for mutate in (lambda x:x['targets'].__setitem__(23,-0.),
                       lambda x:x['targets'].__setitem__(0,float('nan')),
                       lambda x:x['equal'].__setitem__(0,True),lambda x:x.update(width=48)):
            bad=deepcopy(good);mutate(bad)
            with self.assertRaises(ValueError):subject.check_rank49(values,bad)

    def test_oracle_has_no_producer_host_or_io_calls(self):
        tree=ast.parse(Path(subject.__file__).read_text())
        forbidden={'open','read_bytes','read_text','write_text','write_bytes','load','save','loadtxt'}
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                self.assertFalse(any('reference_gain250_math' in n or 'host' in n for n in names))
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
                self.assertNotIn(name,forbidden)


class SoleProducerCrossTests(unittest.TestCase):
    def test_math_and_rank_full_evidence(self):
        import monthly_reference_gain250_math as producer
        import test_monthly_reference_gain250_math as fixtures
        for p,v in (fixture(),fixtures.fixture(),([2.]*253,[7.]*252),([sys.float_info.max]*252+[math.ldexp(1.,-1074)],[1.]*252)):
            subject.check_measure(p,v,producer.measure(p,v))
        p,v=fixture();p[0]=None;v[-1]=pd.NA;subject.check_measure(p,v,producer.measure(p,v))
        for invalid in (True,0.,-1.,float('inf'),Fraction(1,10**1000)):
            p,v=fixture();p[0]=None;v[-1]=invalid
            with self.assertRaises(ValueError):producer.measure(p,v)
            with self.assertRaises(ValueError):subject.measure_oracle(p,v)
        for raw in ([Fraction(i) for i in range(49)],[Fraction(0)]*49,[Fraction(2**1000)+Fraction(i,2**1000) for i in range(49)]):
            subject.check_rank49(raw,producer.rank49(raw))

if __name__=='__main__':
    if '--self-only' in sys.argv:
        sys.argv.remove('--self-only');suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (MeasureTests,RankTests))
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not result.wasSuccessful())
    unittest.main(verbosity=2)
