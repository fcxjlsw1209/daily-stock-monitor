"""Invented-data tests for the independent249 math; no real source is loaded."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

import monthly_volume_breadth249_math_independent as subject

PRODUCER_SHA = 'c632e74a76fbc1975fff3e56d96b4ff1fccd314828e3099bd9495d7424a73ee9'
FIXTURE_SHA = '5b0b2291775b9bde7947926b3cbfecad1fcba90107943f47f162a31b5f6271ad'


def fixture():
    volumes = [float(20+(17*i)%31) for i in range(315)]
    prices = [1.]
    for q in range(252):
        prices.append(prices[-1]*(1.+((7*q)%13-6)/512.))
    return prices, volumes


def brute_flags(volumes):
    return [int(volumes[q+63] > sorted(volumes[q:q+63])[31]) for q in range(252)]


def signed_fixture(signs):
    if len(signs) != 12:
        raise ValueError('Twelve synthetic signs')
    volumes = [100.]*63+[50. if q%2 == 0 else 200. for q in range(252)]
    flags = brute_flags(volumes)
    prices = [1.]
    for q, high in enumerate(flags):
        sign = signs[q//21]
        factor = (2. if sign > 0 else .5) if high and sign else 1.
        prices.append(prices[-1]*factor)
    return prices, volumes


def value(pair):
    return Fraction(int(pair['numerator']), int(pair['denominator']))


class MeasureTests(unittest.TestCase):
    def test_fixed_schema_and_counts(self):
        result = subject.measure_oracle(*fixture())
        self.assertEqual(tuple(result), subject.MEASURE_KEYS)
        self.assertTrue(result['available'])
        self.assertEqual([result[k] for k in ('price_count','volume_count','return_count','block_count')], [253,315,252,12])
        self.assertTrue(all(tuple(b) == subject.BLOCK_KEYS for b in result['blocks']))
        self.assertEqual(len(result['high_flags']), 252)

    def test_strictly_previous63_and_current_exclusion(self):
        prices = [1.]*253
        volumes = [1.]*32+[10.]*31+[10.]+[1.]*251
        self.assertEqual(subject.measure_oracle(prices, volumes)['high_flags'][0], 1)
        changed = volumes.copy(); changed[63] = 1.
        self.assertEqual(subject.measure_oracle(prices, changed)['high_flags'][0], 0)
        self.assertEqual(volumes[0:63], changed[0:63])

    def test_all252_order_statistics_and_last_alignment(self):
        p, v = fixture(); result = subject.measure_oracle(p, v)
        self.assertEqual(result['high_flags'], brute_flags(v))
        self.assertEqual(result['high_flags'][-1], int(v[314] > sorted(v[251:314])[31]))
        changed = v.copy(); changed[314] = 1e6
        other = subject.measure_oracle(p, changed)
        self.assertEqual(other['high_flags'][:-1], result['high_flags'][:-1])
        self.assertEqual(other['high_flags'][-1], 1)

    def test_disjoint21_blocks_and_full_partition(self):
        result = subject.measure_oracle(*fixture())
        for k, block in enumerate(result['blocks']):
            self.assertEqual((block['first_return_index'],block['last_return_index']), (21*k,21*k+20))
            self.assertEqual(block['high_count'], sum(result['high_flags'][21*k:21*k+21]))
        self.assertEqual(sum(b['high_count'] for b in result['blocks']), sum(result['high_flags']))

    def test_positive_negative_and_zero_blocks(self):
        signs = [1,-1,0]*4
        result = subject.measure_oracle(*signed_fixture(signs))
        self.assertEqual([b['sign'] for b in result['blocks']], signs)
        self.assertEqual((result['positive_block_count'],result['negative_block_count'],result['zero_block_count']), (4,4,4))
        self.assertEqual(value(result['raw_breadth']), 0)

    def test_constant_volume_valid_zero_covariance(self):
        p, _ = fixture()
        result = subject.measure_oracle(p, [100.]*315)
        self.assertTrue(result['available'])
        self.assertEqual(result['high_flags'], [0]*252)
        self.assertEqual(result['zero_block_count'], 12)
        self.assertEqual(value(result['raw_breadth']), 0)
        self.assertEqual(value(result['raw_mean']), 0)

    def test_constant_returns_valid_zero_covariance(self):
        _, v = fixture()
        for p in ([1.]*253, [math.ldexp(1., i) for i in range(253)]):
            result = subject.measure_oracle(p, v)
            self.assertEqual(result['zero_block_count'], 12)
            self.assertEqual(value(result['raw_mean']), 0)

    def test_covariance_matches_exact_unscaled_sum_identity(self):
        p, v = fixture(); result = subject.measure_oracle(p, v)
        r = [Fraction.from_float(p[q+1])/Fraction.from_float(p[q])-1 for q in range(252)]
        for k, block in enumerate(result['blocks']):
            rr = r[k*21:(k+1)*21]; h = result['high_flags'][k*21:(k+1)*21]
            sumr = sum(rr, Fraction()); sumhr = sum((x for x, flag in zip(rr,h) if flag), Fraction())
            self.assertEqual(value(block['return_sum']), sumr)
            self.assertEqual(value(block['high_return_sum']), sumhr)
            self.assertEqual(value(block['covariance']), (21*sumhr-sum(h)*sumr)/441)
        self.assertEqual(value(result['raw_mean']), sum((value(b['covariance']) for b in result['blocks']), Fraction())/12)

    def test_breadth_ignores_return_amplitude_mean_does_not(self):
        p, v = signed_fixture([1]*12)
        flags = brute_flags(v); p4 = [1.]
        for h in flags: p4.append(p4[-1]*(4. if h else 1.))
        a, b = subject.measure_oracle(p,v), subject.measure_oracle(p4,v)
        self.assertEqual(a['raw_breadth'], b['raw_breadth'])
        self.assertEqual(value(a['raw_breadth']), Fraction(1,2))
        self.assertEqual(value(b['raw_mean']), 3*value(a['raw_mean']))

    def test_zero_blocks_receive_half_credit(self):
        result = subject.measure_oracle(*signed_fixture([1]*3+[-1]*2+[0]*7))
        self.assertEqual(value(result['raw_breadth']), Fraction(1,24))
        self.assertEqual(value(result['raw_breadth']), Fraction(3,12)+Fraction(7,24)-Fraction(1,2))

    def test_missing_source_has_no_partial_statistics(self):
        p, v = fixture(); p[0]=None; p[252]=np.nan; v[0]=pd.NA; v[314]=float('nan')
        result = subject.measure_oracle(p,v)
        self.assertFalse(result['available']); self.assertEqual(result['reason'],'missing_source')
        self.assertEqual(result['missing_price_indices'], [0,252])
        self.assertEqual(result['missing_volume_indices'], [0,314])
        self.assertEqual(result['blocks'],[]); self.assertEqual(result['high_flags'],[])
        for key in ('raw_breadth','raw_mean','positive_block_count','zero_block_count','negative_block_count'):
            self.assertIsNone(result[key])
        self.assertEqual(result['volume_count'],315)

    def test_missing_does_not_hide_invalid_and_both_windows_are_scanned(self):
        seen = []
        class Probe(float):
            def __new__(cls, number, label):
                obj = float.__new__(cls,number); obj.label = label; return obj
            def __float__(self):
                seen.append(self.label); return float.__float__(self)
        p, v = fixture(); p[0]=None; p[100]=0.; p[-1]=Probe(1.,'price_tail')
        v[0]=pd.NA; v[100]=-1.; v[-1]=Probe(1.,'volume_tail')
        with self.assertRaises(ValueError): subject.measure_oracle(p,v)
        self.assertEqual(seen, ['price_tail','volume_tail'])
        p, v = fixture(); p[0]=None; v[-1]=True
        with self.assertRaises(ValueError): subject.measure_oracle(p,v)
        p, v = fixture(); p[-1]=False; v[0]=None
        with self.assertRaises(ValueError): subject.measure_oracle(p,v)

    def test_all_invalid_types_and_values_hard_fail(self):
        bad = [True,np.bool_(False),'1','nan',pd.NaT,complex(1),float('inf'),-float('inf'),0.,-1.,10**1000]
        for invalid in bad:
            for which in (0,1):
                with self.subTest(invalid=repr(invalid)[:30],which=which):
                    args = list(fixture()); args[which][100] = invalid
                    with self.assertRaises(ValueError): subject.measure_oracle(*args)

    def test_lengths_and_shapes_are_exact(self):
        p, v = fixture()
        for a, b in ((p[:-1],v),(p+[1.],v),(p,v[:-1]),(p,v+[1.]),([[x] for x in p],v),(p,[[x] for x in v])):
            with self.assertRaises(ValueError): subject.measure_oracle(a,b)

    def test_binary64_source_conversion_precedes_exact_ratios(self):
        p, v = fixture(); p[0] = Fraction(1,3); p[1] = Fraction(2,3)
        expected = subject.measure_oracle([float(x) for x in p],v)
        self.assertEqual(subject.measure_oracle(p,v), expected)

    def test_extreme_finite_prices_keep_exact_raw_without_float_overflow(self):
        tiny = math.ldexp(1.,-1074); huge = sys.float_info.max
        p = [tiny if q%2 == 0 else huge for q in range(253)]
        _, v = fixture(); result = subject.measure_oracle(p,v)
        self.assertTrue(result['available'])
        covariances = [value(b['covariance']) for b in result['blocks']]
        self.assertTrue(any(abs(x) > Fraction.from_float(huge) for x in covariances))
        subject.check_measure(p,v,result)

    def test_exact_power_two_scaling_invariance_and_no_mutation(self):
        p, v = fixture(); before = deepcopy((p,v))
        result = subject.measure_oracle(p,v)
        scaled = subject.measure_oracle([x*2 for x in p],[x*4 for x in v])
        self.assertEqual(result,scaled); self.assertEqual((p,v),before)

    def test_measure_checker_checks_full_audit_and_strict_types(self):
        p, v = signed_fixture([1]*12); expected = subject.measure_oracle(p,v)
        receipt = subject.check_measure(p,v,expected)
        self.assertTrue(all(receipt[k] is True for k in subject.MEASURE_FLAGS))
        for mutate in (lambda x:x.update(price_count=252),lambda x:x.update(available=1),
                       lambda x:x['blocks'][0].update(sign=True),
                       lambda x:x['blocks'][0]['covariance'].update(numerator='0'),
                       lambda x:x.update(extra=True),lambda x:x['high_flags'].__setitem__(0,True)):
            wrong = deepcopy(expected); mutate(wrong)
            with self.assertRaises(ValueError): subject.check_measure(p,v,wrong)


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
                self.assertFalse(any('breadth249_math' in n or 'host' in n for n in names))
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
                self.assertNotIn(name,forbidden)


class SoleProducerCrossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent
        for name,digest in (('monthly_volume_breadth249_math.py',PRODUCER_SHA),
                            ('test_monthly_volume_breadth249_math.py',FIXTURE_SHA)):
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen synthetic cross dependency changed: '+name)
        cls.producer=importlib.import_module('monthly_volume_breadth249_math')
        cls.fixture_module=importlib.import_module('test_monthly_volume_breadth249_math')

    def test_both_fixture_families_full_measure_and_invalid_missing_cross(self):
        cases=[fixture(),self.fixture_module.fixture()]
        for signs in ([1]*12,[-1]*12,[0]*12,[1,-1,0]*4):
            cases.extend((signed_fixture(signs),self.fixture_module.signed_fixture(signs)))
        for args in cases:
            actual=self.producer.measure(*args)
            self.assertTrue(subject.check_measure(*args,actual)['passed'])
        p,v=fixture();p[3]=None;v[314]=pd.NA
        subject.check_measure(p,v,self.producer.measure(p,v))
        for bad in (True,'1',pd.NaT,0.,-1.,float('inf'),10**1000):
            p,v=fixture();p[0]=None;v[-1]=bad
            with self.assertRaises(ValueError):self.producer.measure(p,v)
            with self.assertRaises(ValueError):subject.measure_oracle(p,v)

    def test_all_rank_audit_and_exact_raw_extremes_cross(self):
        cases=[
            [Fraction(i) for i in range(49)],
            [Fraction((i*13)%25,24) for i in range(49)],
            [Fraction(0)]*49,
            [Fraction(2**1000)+Fraction(i,2**1000) for i in range(49)]]
        for raw in cases:
            for encoded in (raw,[subject.pair(x) for x in raw]):
                self.assertTrue(subject.check_rank49(encoded,self.producer.rank49(encoded))['passed'])
        p=[math.ldexp(1.,-1074) if q%2==0 else sys.float_info.max for q in range(253)]
        _,v=fixture()
        subject.check_measure(p,v,self.producer.measure(p,v))


if __name__=='__main__':
    unittest.main(verbosity=2)
