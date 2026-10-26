"""Synthetic scoring integration using the unchanged frozen episode adapter."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import copy
import importlib.util
import unittest
import numpy as np
import monthly_roa_stability220_scoring as s

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roa220_frozen_adapter', ROOT/'outputs/monthly_core_timing70_research/core_adapter.py')
adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
POLICY = dict(budget=.05, ceiling=5, gap=.02)


def fixture():
    shape = (4, 5)
    core = np.tile([.20, .18, .16, .15, .12], (4, 1))
    a = dict(core=core, baseline=core+.03, core_weight=np.full((4, 1), .52),
        eligible=np.ones(shape, bool), effects=np.zeros(shape),
        state_calm10=np.zeros(4, bool), state_weak_narrow=np.zeros(4, bool))
    a['eligible'][2, 0] = False
    v = dict(available=np.tile([True, True, True, True, False], (4, 1)),
        candidate=np.tile([.9, .1, 0., -.9, np.nan], (4, 1)),
        control=np.tile([.99, .2, 0., -.1, np.nan], (4, 1)))
    m = dict(dates=['s0','s1','s2','s3'], symbols=[list('ABCDE') for _ in range(4)], etfs=['E'])
    return a, m, v


class Study:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.rp = SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs(): return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(spec, a, n):
        assert spec == dict(id=4)
        return a['baseline'].copy() + np.array([.011,.012,.013,.014,.015])
    @staticmethod
    def boundary(pre, eligible, symbols, effects, original, policy):
        assert policy == POLICY
        choices=[]
        for scores, good, ss in zip(pre, eligible, symbols):
            ids=sorted(np.flatnonzero(good), key=lambda j:(-scores[j],ss[j]))
            choices.append(ids[:2]+[ids[-1]] if len(ids)>=3 else ids if len(ids)==2 else [])
        return choices, pre.copy()
    def choices(self, spec, a, m, n, *, policy):
        return self.ep.select_boundary(self.modified_pre_score(spec,a,n)[:n],a['eligible'][:n],
            m['symbols'][:n],a['effects'][:n],a,policy)
    @staticmethod
    def orders(choices, score, meta):
        return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],
                     score=score[i].tolist()) for i,ids in enumerate(choices)]


class ScoreTests(unittest.TestCase):
    def test_only_fixed_eligible_nonzero_delta(self):
        a,m,v=fixture(); t,d=s.scale_and_delta(v,a,4)
        expected=.01*np.nan_to_num(v['candidate']);expected[~a['eligible']]=0.
        np.testing.assert_array_equal(d[s.NAMES[1]],expected)
        self.assertEqual(d[s.NAMES[1]][2,0],0.)
        self.assertTrue(np.signbit(t[s.NAMES[1]][0,3]))
        self.assertEqual(d[s.NAMES[1]][0,2].hex(),'0x0.0p+0')

    def test_h_over_half_allowed_without_rank_adapter_error(self):
        a,m,v=fixture(); study=Study(); before={k:x.tobytes() for k,x in a.items()}
        actual,_,d=s.decisions(study,a,m,v,4,policy=POLICY,adapter=adapter)
        self.assertEqual(study.ep.select_boundary.call_count,3)
        for arm in s.NAMES[1:]:
            pre=study.modified_pre_score(dict(id=4),a,4)
            changed=d[arm]!=0.;pre[changed]+=d[arm][changed]
            ch,score=Study.boundary(pre,a['eligible'],m['symbols'],a['effects'],a,POLICY)
            self.assertEqual(actual[arm],Study.orders(ch,score,m))
        self.assertEqual(before,{k:x.tobytes() for k,x in a.items()})
        # Deliberately non-global fake episode keeps E rather than third-ranked C.
        self.assertEqual(actual[s.NAMES[1]][0]['symbols'],list('ABE'))
        for call in study.ep.select_boundary.call_args_list[1:]:
            orig=call.args[4]
            np.testing.assert_array_equal(orig['core'],a['core'])
            np.testing.assert_array_equal(orig['baseline'],a['baseline'])
            np.testing.assert_array_equal(orig['core_weight'],np.broadcast_to(a['core_weight'],a['core'].shape))

    def test_missing_and_zero_exact_incumbent_fallback(self):
        for value in [np.nan,0.]:
            a,m,v=fixture();v['candidate'][:]=value;v['control'][:]=value
            v['candidate'][:,-1]=np.nan;v['control'][:,-1]=np.nan
            v['available']=np.isfinite(v['candidate'])
            orders,_,d=s.decisions(Study(),a,m,v,4,policy=POLICY,adapter=adapter)
            for arm in s.NAMES[1:]:
                self.assertEqual(orders[arm],orders[s.NAMES[0]])
                self.assertEqual(np.count_nonzero(d[arm]),0)

    def test_all_prefixes_exact_no_input_mutation(self):
        a,m,v=fixture(); before=copy.deepcopy(v)
        whole,tt,dd=s.decisions(Study(),a,m,v,4,policy=POLICY,adapter=adapter)
        for n in range(5):
            orders,t,d=s.decisions(Study(),a,m,v,n,policy=POLICY,adapter=adapter)
            self.assertEqual(orders,{k:val[:n] for k,val in whole.items()})
            for arm in s.NAMES[1:]:
                np.testing.assert_array_equal(t[arm],tt[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
        for k in v:np.testing.assert_array_equal(v[k],before[k])

    def test_future_invalid_values_not_read(self):
        a,m,v=fixture();ref=s.scale_and_delta(v,a,2)
        v['candidate'][2:]=np.inf;v['control'][2:]=-np.inf
        actual=s.scale_and_delta(v,a,2)
        for original, result in zip(ref,actual):
            for key in original:np.testing.assert_array_equal(original[key],result[key])
        with self.assertRaises(ValueError):s.scale_and_delta(v,a,3)

    def test_invalid_available_mask_fails(self):
        a,m,v=fixture();v['available']=v['available'].astype(int)
        with self.assertRaises(ValueError):s.scale_and_delta(v,a,4)
        a,m,v=fixture();v['candidate'][0,0]=np.nan
        with self.assertRaises(ValueError):s.scale_and_delta(v,a,4)

    def test_out_of_range_and_penalty_reversal_fail(self):
        for value in [1.01,float('inf')]:
            a,m,v=fixture();v['control'][0,0]=value
            with self.assertRaises(ValueError):s.scale_and_delta(v,a,4)
        a,m,v=fixture();v['candidate'][0,0]=1.
        with self.assertRaises(ValueError):s.scale_and_delta(v,a,4)

    def test_etf_never_has_financial_target(self):
        a,m,v=fixture();v['candidate'][0,4]=0.;v['control'][0,4]=0.;v['available'][0,4]=True
        with self.assertRaises(ValueError):s.decisions(Study(),a,m,v,4,policy=POLICY,adapter=adapter)

    def test_bad_prefix_and_schema_fail(self):
        a,m,v=fixture()
        for n in [-1,5,True,1.]:
            with self.assertRaises(ValueError):s.scale_and_delta(v,a,n)
        v['extra']=np.zeros((4,5))
        with self.assertRaises(ValueError):s.scale_and_delta(v,a,4)


if __name__=='__main__':unittest.main(verbosity=2)
