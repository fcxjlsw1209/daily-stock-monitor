"""Synthetic only: no historical forecasts, scores, or portfolio returns."""
import copy
import json
import math
import unittest

import numpy as np

import monthly_episode_neighborhood180_kernel as k


def fixture(rows=3, width=6):
    pre = np.tile(np.linspace(.40, .30, width), (rows, 1))
    support = np.tile(np.linspace(-.3, .3, width), (rows, 1))
    arrays = dict(baseline=.2+.48*support, core=np.full((rows,width),.2),
                  core_weight=np.full((rows,1),.52), eligible=np.ones((rows,width),bool),
                  effects=np.tile(np.linspace(.18,-.12,width),(rows,1)),
                  state_calm10=np.zeros(rows,bool),state_weak_narrow=np.zeros(rows,bool))
    meta = dict(dates=['2020-%02d-28'%(i+1) for i in range(rows)],
                symbols=[[chr(65+j) for j in range(width)] for _ in range(rows)])
    return pre, arrays, meta


class SelectionTests(unittest.TestCase):
    def test_independent_agreement_on_both_fixtures_and_edge_cases(self):
        import monthly_episode_neighborhood180_independent as independent
        from test_monthly_episode_neighborhood180_independent import fixture as other_fixture
        cases=[fixture(),other_fixture()]
        pre,a,m=fixture();pre[1]=.2;a['effects'][1]=0.;m['symbols'][1]=list('FEDCBA')
        cases.append((pre,a,m))
        pre,a,m=fixture();a['eligible'][1,2:]=False;a['effects'][2,1]=np.nan
        a['state_calm10'][:]=True;a['state_weak_narrow'][:]=True;cases.append((pre,a,m))
        pre,a,m=fixture();pre[1]=[1e308,5e307,0.,-5e307,-1e308,-1.7e308]
        cases.append((pre,a,m))
        for pre,a,m in cases:
            for n in range(len(pre)+1):
                result,audit=k.apply_episode(pre,a,m,n)
                receipt=independent.verify_selection(pre,a,m,result,audit,n)
                self.assertTrue(receipt['passed'])

    def test_exact_formula_scores_choices_and_json(self):
        pre,a,m=fixture();r,au=k.apply_episode(pre,a,m)
        for i in range(1,3):
            q=sorted(pre[i],reverse=True)[2]
            for j in range(6):
                support=(a['baseline'][i,j]-a['core'][i,j])/.48
                d=.05*(min(.5,max(-.5,support+a['effects'][i,j]))-support)
                weight=math.exp(-.5*((pre[i,j]-q)/.02)**2)
                self.assertEqual(r['corrections']['control'][i,j],d)
                self.assertEqual(r['corrections']['candidate'][i,j],weight*d)
            for arm in k.ARMS:
                expected=sorted(range(6),key=lambda j:(-r['scores'][arm][i,j],m['symbols'][i][j]))[:3]
                self.assertEqual(r['choices'][arm][i],expected)
        self.assertEqual(au['active_rows'],2)
        self.assertEqual(au['eligible_cells'],18)
        self.assertEqual(au['finite_effect_eligible_cells'],18)
        json.dumps(au,allow_nan=False)

    def test_first_row_never_corrected(self):
        pre,a,m=fixture();a['effects'][0]=1e300;r,au=k.apply_episode(pre,a,m)
        for arm in k.ARMS:
            np.testing.assert_array_equal(r['scores'][arm][0],pre[0])
            np.testing.assert_array_equal(r['corrections'][arm][0],np.zeros(6))
        self.assertEqual(au['event_history'][0]['suppression_reason'],'initial_row')

    def test_all_cardinalities_and_ineligible_preserved(self):
        for number in range(7):
            pre,a,m=fixture();a['eligible'][1]=False;a['eligible'][1,:number]=True
            pre[1,number:]=np.nan;a['baseline'][1,number:]=np.nan;a['core'][1,number:]=np.inf
            r,au=k.apply_episode(pre,a,m)
            for arm in k.ARMS:
                self.assertEqual(len(r['choices'][arm][1]),min(3,number) if number>=2 else 0)
                np.testing.assert_array_equal(r['scores'][arm][1,number:],pre[1,number:])
                self.assertFalse(r['corrections'][arm][1,number:].any())
                if number<3:self.assertFalse(r['corrections'][arm][1].any())
            json.dumps(au,allow_nan=False)

    def test_ticker_ties_are_only_tie_break(self):
        pre,a,m=fixture();pre[:]=.2;a['effects'][:]=0.;m['symbols'][1]=list('FEDCBA')
        r,au=k.apply_episode(pre,a,m)
        for arm in k.ARMS:self.assertEqual(r['choices'][arm][1],[5,4,3])
        self.assertEqual(au['event_history'][1]['third_score'],.2)
        self.assertTrue(all(c['neighborhood_weight']==1. for c in au['event_history'][1]['cells']))

    def test_global_selection_can_change_top_two_and_admit_below_fifth(self):
        pre,a,m=fixture();pre[1]=[.31,.30,.29,.28,.27,.26]
        a['baseline'][1]=.2;a['effects'][1]=[-1.,-1.,-1.,0.,0.,1.]
        r,_=k.apply_episode(pre,a,m)
        self.assertIn(5,r['choices']['control'][1])
        self.assertNotEqual(r['choices']['control'][1],[0,1,2])

    def test_rank_two_three_and_fifth_sixth_boundaries_continuous(self):
        pre,a,m=fixture();pre[1]=[.5,.4,.4,.38,.37,.37]
        for j in (1,2,4,5):
            outputs=[]
            for epsilon in (-1e-10,1e-10):
                p=pre.copy();p[1,j]+=epsilon
                outputs.append(k.apply_episode(p,a,m)[0]['corrections']['candidate'][1])
            np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-9,rtol=0.)
        r,_=k.apply_episode(pre,a,m)
        self.assertNotEqual(r['corrections']['candidate'][1,0],0.)
        self.assertNotEqual(r['corrections']['candidate'][1,5],0.)

    def test_missing_effect_neutral_but_infinite_is_error(self):
        pre,a,m=fixture();a['effects'][1,2]=np.nan;r,au=k.apply_episode(pre,a,m)
        for arm in k.ARMS:self.assertEqual(r['corrections'][arm][1,2],0.)
        self.assertEqual(au['event_history'][1]['cells'][2]['neighborhood_weight'],1.)
        self.assertEqual(au['missing_effect_eligible_cells'],1)
        for value in (np.inf,-np.inf):
            a['effects'][1,2]=value
            with self.assertRaises(ValueError):k.apply_episode(pre,a,m)

    def test_budget_states_and_target_clipping(self):
        for calm,weak,budget in ((False,False,.05),(True,False,.05),(False,True,.05),(True,True,.025)):
            pre,a,m=fixture();a['state_calm10'][:]=calm;a['state_weak_narrow'][:]=weak
            a['effects'][1,2]=1e300;r,au=k.apply_episode(pre,a,m)
            self.assertAlmostEqual(au['event_history'][1]['budget'],budget,places=16)
            support=(a['baseline'][1,2]-.2)/.48
            self.assertAlmostEqual(r['corrections']['control'][1,2],budget*(.5-support),places=16)

    def test_weight_broadcast_forms_same_and_not_state_guessed(self):
        pre,a,m=fixture();expected=k.apply_episode(pre,a,m)
        for value in (np.full(3,.52),np.full((3,1),.52),np.full((3,6),.52)):
            a['core_weight']=value;r,au=k.apply_episode(pre,a,m)
            for arm in k.ARMS:np.testing.assert_array_equal(r['scores'][arm],expected[0]['scores'][arm])
            self.assertEqual(au,expected[1])
        a['core_weight']=np.full((3,6),.4)
        k.apply_episode(pre,a,m)
        for value in (0.,1.,-1.,np.nan,np.inf):
            a['core_weight']=np.full((3,6),.52);a['core_weight'][1,5]=value
            with self.assertRaises(ValueError):k.apply_episode(pre,a,m)

    def test_invalid_eligible_and_arithmetic_fail_closed(self):
        for field in ('baseline','core','pre'):
            pre,a,m=fixture();(pre if field=='pre' else a[field])[1,2]=np.nan
            with self.assertRaises(ValueError):k.apply_episode(pre,a,m)
        pre,a,m=fixture();a['baseline'][1,2]=1e308;a['core'][1,2]=-1e308
        with self.assertRaises(FloatingPointError):k.apply_episode(pre,a,m)

    def test_extreme_finite_score_distance_and_effect_saturate(self):
        pre,a,m=fixture();pre[1]=[1e308,5e307,0.,-5e307,-1e308,-1.7e308]
        a['effects'][1]=1e308;r,au=k.apply_episode(pre,a,m)
        self.assertEqual(au['event_history'][1]['cells'][0]['neighborhood_weight'],0.)
        self.assertEqual(au['event_history'][1]['cells'][2]['neighborhood_weight'],1.)
        self.assertTrue(np.isfinite(r['scores']['candidate'][1]).all())

    def test_nonfinite_result_is_error(self):
        pre,a,m=fixture();a['core_weight'][:]=.1
        pre[1]=np.finfo(float).max;a['baseline'][1]=-1e308;a['core'][1]=0.;a['effects'][1]=1e308
        with self.assertRaises(FloatingPointError):k.apply_episode(pre,a,m)

    def test_finite_inputs_cannot_hide_shifted_support_overflow_with_clip(self):
        pre,a,m=fixture();a['core'][1,2]=0.;a['baseline'][1,2]=.48e308
        a['effects'][1,2]=1e308
        with self.assertRaises(FloatingPointError):k.apply_episode(pre,a,m)

    def test_input_bytes_unchanged(self):
        pre,a,m=fixture();a['effects'][1,1]=np.nan
        before={name:(v.dtype,v.shape,v.tobytes()) for name,v in dict(a,pre=pre).items()}
        metadata=copy.deepcopy(m);k.apply_episode(pre,a,m)
        for name,v in dict(a,pre=pre).items():self.assertEqual(before[name],(v.dtype,v.shape,v.tobytes()))
        self.assertEqual(m,metadata)

    def test_prefixes_and_future_poison(self):
        pre,a,m=fixture();r,au=k.apply_episode(pre,a,m)
        for n in range(4):
            short,sa=k.apply_episode(pre,a,m,n)
            for category in ('scores','corrections'):
                for arm in k.ARMS:np.testing.assert_array_equal(short[category][arm],r[category][arm][:n])
            for arm in k.ARMS:self.assertEqual(short['choices'][arm],r['choices'][arm][:n])
            self.assertEqual(sa['event_history'],au['event_history'][:n])
        class Poison:
            def __float__(self):raise AssertionError('Future value inspected')
        pre=pre.astype(object);pre[2]=Poison()
        for name in ('baseline','core','effects','core_weight'):
            a[name]=a[name].astype(object);a[name][2]=Poison()
        m['dates'][2]='bad';m['symbols'][2]=None
        short,_=k.apply_episode(pre,a,m,2)
        np.testing.assert_array_equal(short['scores']['candidate'],r['scores']['candidate'][:2])

    def test_schema_and_prefix_limits(self):
        pre,a,m=fixture()
        for limit in (-1,4,True,1.5,'1'):
            with self.assertRaises(ValueError):k.apply_episode(pre,a,m,limit)
        for name in ('eligible','state_calm10','state_weak_narrow'):
            bad=copy.deepcopy(a);bad[name]=bad[name].astype(int)
            with self.assertRaises(ValueError):k.apply_episode(pre,bad,m)
        for value in (np.ones((3,2)),np.ones((1,6))):
            bad=copy.deepcopy(a);bad['core_weight']=value
            with self.assertRaises(ValueError):k.apply_episode(pre,bad,m)
        bad=copy.deepcopy(m);bad['symbols'][1][1]='A'
        with self.assertRaises(ValueError):k.apply_episode(pre,a,bad)
        bad=copy.deepcopy(m);bad['dates'][1]=bad['dates'][0]
        with self.assertRaises(ValueError):k.apply_episode(pre,a,bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
