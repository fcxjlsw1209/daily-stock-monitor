"""Execution, budget, missing-data and training-only driver invariants."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('trend_driver_test',P/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def example():
    base=np.array([[.30,.20,.10,.099,.098,.097]])
    a=dict(base_original65=base,core_original65=.5*base,cw_original65=np.array([[.5]]),
        eligible=np.ones_like(base,dtype=bool),full=np.array([0.,0.,0.,.05,0.,0.]),
        folds=np.tile([0.,0.,0.,.05,0.,0.],(5,1)),available=np.ones(6,dtype=bool),
        fold_available=np.ones((5,6),dtype=bool),volume_confirmation=np.zeros_like(base),
        valid_volume_confirmation=np.ones_like(base,dtype=bool))
    m=dict(symbols=[list('ABCDEF')])
    row=dict(base='original65',family='ml',key='block_stability',budget=.025,k=1.)
    return a,m,row


class DriverTests(unittest.TestCase):
    def test_grid_is_fixed_and_small(self):
        g=s.grid();self.assertEqual(len(g),22)
        self.assertEqual([x['id'] for x in g],list(range(22)))
        self.assertTrue(all(x['budget']<=.075 for x in g))

    def test_hand_zero_and_missing_are_exact_nochange(self):
        a,m,row=example();row.update(family='hand',key='volume_confirmation',budget=.075)
        np.testing.assert_array_equal(s.score(row,a,1),a['base_original65'])
        a['volume_confirmation'][:]=1.;a['valid_volume_confirmation'][:]=False
        np.testing.assert_array_equal(s.score(row,a,1),a['base_original65'])

    def test_hand_is_convex_auxiliary_mix(self):
        a,m,row=example();row.update(family='hand',key='volume_confirmation',budget=.075)
        a['volume_confirmation'][:]=[-.5,.5,0.,.2,-.3,.1]
        core=a['core_original65'];support=(a['base_original65']-core)/.5
        want=core+(.5-.075)*support+.075*np.clip(support+a['volume_confirmation'],-.5,.5)
        np.testing.assert_allclose(s.score(row,a,1),want,atol=1e-16,rtol=0)

    def test_stable_pair_may_replace_third_but_not_first_two(self):
        a,m,row=example();ch,sc=s.choose(row,a,m,1)
        self.assertEqual(ch,[[0,1,3]])
        np.testing.assert_array_equal(sc[0,:3],a['base_original65'][0,:3])
        self.assertLessEqual(sc[0,3]-a['base_original65'][0,3],.5*row['budget'])

    def test_no_model_or_exact_zero_means_exact_original(self):
        a,m,row=example();a['available'][:]=False
        ch,sc=s.choose(row,a,m,1);self.assertEqual(ch,[[0,1,2]])
        np.testing.assert_array_equal(sc,a['base_original65'])
        a['available'][:]=True;a['full'][:]=0;a['folds'][:]=0
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])

    def test_disagreement_stops_swap_but_k0_is_control(self):
        a,m,row=example();a['folds'][:,3]=[-.15,.25,-.15,.25,.05]
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])
        row['k']=0.;self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,3]])

    def test_near_boundary_only_and_score_gap_must_be_paid(self):
        a,m,row=example();a['base_original65'][0,3]=.075
        a['base_original65'][0,4:]=[.074,.073]
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])
        a['base_original65'][0,3]=.085
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])
        a,m,row=example();a['full'][3]=0;a['full'][5]=100;a['folds'][:,3]=0;a['folds'][:,5]=100
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])

    def test_eligibility_not_forecast_validity_and_cash_rule(self):
        a,m,row=example();a['eligible'][0,3]=False
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1,2]])
        a['eligible'][:]=False;a['eligible'][0,:2]=True
        self.assertEqual(s.choose(row,a,m,1)[0],[[0,1]])
        a['eligible'][0,1]=False;self.assertEqual(s.choose(row,a,m,1)[0],[[]])

    def test_future_rows_cannot_change_first_decision(self):
        a,m,row=example();expected=s.choose(row,a,m,1)
        for k in ['base_original65','core_original65','cw_original65','eligible']:
            a[k]=np.concatenate([a[k],a[k]],axis=0)
        a['full']=np.r_[a['full'],np.full(6,1000.)]
        a['folds']=np.column_stack([a['folds'],np.full((5,6),1000.)])
        a['available']=np.r_[a['available'],np.ones(6,dtype=bool)]
        a['fold_available']=np.column_stack([a['fold_available'],np.ones((5,6),dtype=bool)])
        m['symbols'].append(list('ABCDEF'))
        actual=s.choose(row,a,m,1)
        self.assertEqual(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])

    def test_prepared_original_orders_and_missing_neutrality(self):
        if not (P/'DATA.npz').exists():self.skipTest('prepare not run')
        a=dict(np.load(P/'DATA.npz'));m=s.read(P/'META.json')['monthly']
        ch=s.rp.choose(a['base_original65'],a['eligible'],m['symbols'],3)
        self.assertEqual(s.rp.make_orders(ch,a['base_original65'],m),s.read(s.ep.P/'original65_ORDERS.json'))
        old=np.load(s.prev.P/'DATA.npz')
        np.testing.assert_array_equal(a['eligible'],old['eligible'])
        for row in s.grid():
            if row['family']=='hand':
                sc=s.score(row,a,122);miss=~a['valid_'+row['key']]
                np.testing.assert_array_equal(sc[miss],a['base_'+row['base']][miss])


if __name__=='__main__':unittest.main()
