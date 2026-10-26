"""Driver invariants and integration against immutable monthly inputs."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('monthly_session_driver_test',P/'study.py')
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class DriverTests(unittest.TestCase):
    def test_fixed_grid_and_small_budgets(self):
        rows=s.grid()
        self.assertEqual(len(rows),54)
        self.assertEqual([r['id'] for r in rows],list(range(54)))
        self.assertTrue(all(r['budget']<=.075 for r in rows))

    def test_zero_missing_and_tiny_prediction(self):
        b=np.array([[.21,.20,.19,.18]])
        a=dict(base_original65=b,core_original65=b/2,cw_original65=np.array([[.5]]),
               plain_pooled=np.zeros_like(b),valid_plain_pooled=np.ones_like(b,dtype=bool))
        row=dict(base='original65',family='type',key='plain_pooled',budget=.075,direction=1)
        np.testing.assert_array_equal(s.score(row,a,1),b)
        a['plain_pooled'][:]=1e6;a['valid_plain_pooled'][:]=False
        np.testing.assert_array_equal(s.score(row,a,1),b)
        a['plain_pooled'][:]=1e-12;a['valid_plain_pooled'][:]=True
        self.assertLess(np.max(np.abs(s.score(row,a,1)-b)),1e-10)

    def test_correction_is_convex_support_mix_with_preserved_core(self):
        b=np.array([[.2,.1,-.2]])
        core=np.array([[.1,.05,-.1]])
        a=dict(base_original65=b,core_original65=core,cw_original65=np.array([[.5]]),
               session_log_spread=np.array([[.5,-.5,.3]]),valid_session_log_spread=np.ones_like(b,dtype=bool))
        row=dict(base='original65',family='session',key='session_log_spread',budget=.075,direction=1)
        support=(b-core)/.5
        expected=core+(.5-.075)*support+.075*np.clip(support+a['session_log_spread'],-.5,.5)
        np.testing.assert_allclose(s.score(row,a,1),expected,rtol=0,atol=1e-16)

    def test_early_segment_failure_not_overridden_by_large_total_return(self):
        b={k:dict(cagr=.3,sharpe=1.2,max_drawdown=-.3) for k in ['training','early','recent']}
        z={k:dict(v) for k,v in b.items()}
        z['training']['cagr']=3.;z['early']['cagr']=.29
        self.assertFalse(s.admitted(z,b))
        self.assertTrue(s.admitted(b,b))

    def test_prepared_control_and_eligibility(self):
        a=dict(np.load(P/'DATA.npz'))
        old=np.load(s.arch.P/'DATA.npz')
        np.testing.assert_array_equal(a['eligible'],old['eligible'])
        self.assertEqual(a['train_plain'].shape,(len(a['dates']),10))
        self.assertEqual(a['train_state'].shape,(len(a['dates']),28))
        for lo,hi in zip(a['offsets'][:-1],a['offsets'][1:]):
            self.assertLessEqual(a['boundary_top3'][lo:hi].sum(),3)
        for row in s.grid():
            if row['family']=='session':
                sc=s.score(row,a,122)
                missing=~a['valid_'+row['key']]
                np.testing.assert_array_equal(sc[missing],a['base_'+row['base']][missing])

    def test_all_slots_free_and_insufficient_candidates_cash(self):
        m={'symbols':[['A','B','C','D','E','F']]}
        b=np.array([[.01,.009,.008,.007,.006,.005]])
        a=dict(base_original65=b,core_original65=np.zeros_like(b),cw_original65=np.array([[.5]]),
               session_log_spread=np.array([[-.5,-.5,-.5,.5,.5,.5]]),valid_session_log_spread=np.ones_like(b,dtype=bool),
               eligible=np.ones_like(b,dtype=bool))
        row=dict(base='original65',family='session',key='session_log_spread',budget=.075,direction=1)
        self.assertEqual(s.choose(row,a,m,1)[0],[[3,4,5]])
        a['eligible'][:]=False;a['eligible'][0,0]=True
        self.assertEqual(s.choose(row,a,m,1)[0],[[]])


if __name__=='__main__':unittest.main()
