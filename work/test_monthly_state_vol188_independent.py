"""Synthetic-only independent checks; no real research input or output is read."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_state_vol188_independent as audit


def fixture():
    idx=pd.bdate_range('2009-01-05',periods=1120)
    t=np.arange(len(idx)-1,dtype=float)
    sr=np.where((t.astype(int)//140)%2==0,.0025,.017)*np.cos(t*.73)+.0002
    ar=(.007+.004*np.sin(t/90.))*np.sin(t*1.13)+.0003
    br=.009*np.cos(t*.47)+.0007
    frame=pd.DataFrame({s:np.r_[100.,100.*np.cumprod(1+r)] for s,r in [('A',ar),('FUND',br),('SPY',sr)]},index=idx)
    dates=[str(idx[p].date()) for p in (950,1000,1050)]
    return {'adj':frame},{'dates':dates,'symbols':[['A','FUND'] for _ in dates],'etfs':['FUND']},[str(d.date()) for d in idx]


def single(meta,n=1):
    return dict(dates=meta['dates'][:n],symbols=meta['symbols'][:n],etfs=meta['etfs'])


class ScalarTests(unittest.TestCase):
    def test_hand_computed_sample_sigma_and_window(self):
        prices=[100.,110.,99.,118.8]
        returns=[.1,-.1,.2]
        expected=math.sqrt(sum((r-sum(returns)/3)**2 for r in returns)/2)
        self.assertAlmostEqual(audit.sample_sigma(prices),expected,places=14)
        p,m,c=fixture();prices=p['adj'].A.to_numpy()
        self.assertNotEqual(audit.sample_sigma(prices[824:951]),audit.sample_sigma(prices[887:951]))

    def test_exact_midpoint_and_distinct_control(self):
        x=[1.]*126+[2.]*126+[3.]*126+[4.]*378
        s=[True]*378+[False]*378
        r=audit.cdf_oracle(2.,True,x,s)
        self.assertEqual((r['candidate'],r['control']),(0.,-.25))
        self.assertEqual((r['reference_equal'],r['matched_equal']),(126,126))
        for cur,wanted in [(0.,-.5),(5.,.5)]:
            r=audit.cdf_oracle(cur,True,x,s)
            self.assertEqual(r['candidate'],wanted)

    def test_joint_and_matched_minima_inclusive(self):
        x=[1.]*252+[None]*504;s=[True]*63+[False]*189+[None]*504
        self.assertTrue(audit.cdf_oracle(1.,True,x,s)['available'])
        s[62]=False
        self.assertEqual(audit.cdf_oracle(1.,True,x,s)['reason'],'insufficient_matched_history')
        x[251]=None
        r=audit.cdf_oracle(1.,True,x,s)
        self.assertEqual(r['reason'],'insufficient_joint_history')
        self.assertIsNone(r['control'])

    def test_missing_market_is_not_noncalm(self):
        r=audit.cdf_oracle(1.,None,[1.]*756,[False]*756)
        self.assertEqual(r['reference_count'],756)
        self.assertEqual(r['matched_count'],0)
        self.assertEqual(r['reason'],'current_market_state_unavailable')
        self.assertIsNone(r['matched_below'])
        r=audit.cdf_oracle(1.,False,[1.]*756,[None]*756)
        self.assertEqual(r['reference_count'],0)
        self.assertEqual(r['matched_count'],0)

    def test_missing_current_keeps_joint_counts_zero_sigma_valid(self):
        r=audit.cdf_oracle(None,True,[0.]*756,[True]*756)
        self.assertEqual((r['reference_count'],r['matched_count']),(756,756))
        self.assertIsNone(r['reference_below'])
        z=audit.cdf_oracle(0.,True,[0.]*756,[True]*756)
        self.assertEqual((z['candidate'],z['control']),(0.,0.))
        self.assertTrue(z['available'])

    def test_bad_reference_shape_and_types(self):
        with self.assertRaises(ValueError):audit.cdf_oracle(1.,True,[1.]*755,[True]*755)
        with self.assertRaises(ValueError):audit.cdf_oracle(1.,1,[1.]*756,[True]*756)
        with self.assertRaises(ValueError):audit.cdf_oracle(1.,False,[-1.]*756,[False]*756)
        with self.assertRaises(FloatingPointError):audit.sample_sigma([1.,1e200,1.])


class SourceTests(unittest.TestCase):
    def test_complete_audit_counts_and_cached_scalar_windows(self):
        p,m,c=fixture();checks={}
        v,a=audit.independent_features(p,m,c,checks=checks)
        receipt=audit.verify_features(p,m,c,v,a)
        self.assertEqual(a['reference_observations'],6*756)
        self.assertTrue(v['available'].all())
        self.assertGreater(checks['scalar_volatility_checks'],0)
        self.assertLess(checks['unique_volatility_windows_checked'],3*3*757)
        for flag in ['excludes_current_reference','market_state_validity_checked','midpoint_cdf_checked','official_session_windows_checked']:
            self.assertTrue(receipt[flag])
        for r in a['event_history']:
            t=r['session_position'];self.assertEqual(r['reference_start'],c[t-756]);self.assertEqual(r['reference_end'],c[t-1])
            self.assertEqual(r['own_price_start'],c[t-126]);self.assertEqual(r['market_price_start'],c[t-63])

    def test_constant_zero_returns_all_exact_ties(self):
        p,m,c=fixture();p['adj'].loc[:,:]=100.
        v,a=audit.independent_features(p,m,c)
        self.assertTrue(v['available'].all())
        np.testing.assert_array_equal(v['candidate'],np.zeros((3,2)))
        np.testing.assert_array_equal(v['control'],np.zeros((3,2)))
        self.assertTrue(all(r['current_calm'] is True for r in a['event_history']))
        self.assertTrue(all(z['reference_equal']==756 for r in a['event_history'] for z in r['cells']))

    def test_history_not_shortened_and_source_start_official(self):
        p,m,c=fixture();m['dates']=[c[755],c[756]];m['symbols']=m['symbols'][:2]
        v,a=audit.independent_features(p,m,c)
        self.assertFalse(v['available'][0].any())
        self.assertEqual(a['event_history'][0]['reference_slot_count'],0)
        self.assertEqual(a['event_history'][0]['cells'][0]['reference_count'],0)
        self.assertIsNotNone(a['event_history'][0]['cells'][0]['current_sigma'])
        self.assertEqual(a['event_history'][1]['cells'][0]['reference_count'],630)
        self.assertEqual(a['event_history'][0]['source_start'],c[0])

    def test_missing_market_column_and_invalid_current_not_false(self):
        p,m,c=fixture();m=single(m)
        for bad in [None,0.,-1.,np.inf,np.nan]:
            q=copy.deepcopy(p)
            if bad is None:q['adj']=q['adj'].drop(columns='SPY')
            else:q['adj'].loc[m['dates'][0],'SPY']=bad
            v,a=audit.independent_features(q,m,c)
            self.assertFalse(v['available'].any())
            self.assertIsNone(a['event_history'][0]['current_calm'])
            self.assertEqual(a['reason_counts'],{'current_market_state_unavailable':2})

    def test_invalid_own_price_and_unknown_member_common_neutral(self):
        p,m,c=fixture();m=single(m)
        for bad in [0.,-1.,np.inf,np.nan]:
            q=copy.deepcopy(p);q['adj'].loc[m['dates'][0],'A']=bad
            v,a=audit.independent_features(q,m,c)
            self.assertFalse(v['available'][0,0]);self.assertTrue(v['available'][0,1])
            self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'current_volatility_unavailable')
        q={'adj':p['adj'].drop(columns='A')}
        v,a=audit.independent_features(q,m,c)
        self.assertEqual(a['event_history'][0]['cells'][0]['reason'],'missing_symbol')

    def test_missing_official_session_reindexed_no_compression(self):
        p,m,c=fixture();m=single(m);q=copy.deepcopy(p)
        q['adj'].loc[c[500],:]=np.nan
        removed={'adj':p['adj'].drop(pd.Timestamp(c[500]))}
        v,a=audit.independent_features(q,m,c);w,b=audit.independent_features(removed,m,c)
        for key in v:np.testing.assert_array_equal(v[key],w[key])
        self.assertEqual(a['event_history'][0]['cells'],b['event_history'][0]['cells'])
        self.assertEqual(b['event_history'][0]['missing_official_sessions'],[c[500]])
        self.assertEqual(a['event_history'][0]['cells'][0]['reference_count'],629)

    def test_prefix_future_and_unused_poison(self):
        p,m,c=fixture();v,a=audit.independent_features(p,m,c)
        for n in range(4):
            q=copy.deepcopy(p);q['adj']=q['adj'].astype(object)
            after=0 if n==0 else [950,1000,1050][n-1]+1
            q['adj'].iloc[after:,:]='future poison';q['adj']['UNUSED']='ignored'
            w,b=audit.independent_features(q,m,c,limit=n)
            for key in v:np.testing.assert_array_equal(w[key],v[key][:n])
            self.assertEqual(b['event_history'],a['event_history'][:n])
        pre=pd.DataFrame('ignored',index=[pd.Timestamp('2009-01-02')],columns=q['adj'].columns)
        r={'adj':pd.concat([pre,p['adj']])}
        w,b=audit.independent_features(r,m,c)
        for key in v:np.testing.assert_array_equal(w[key],v[key])

    def test_current_observation_never_enters_reference(self):
        p,m,c=fixture();m=single(m);v,a=audit.independent_features(p,m,c)
        q=copy.deepcopy(p);q['adj'].loc[m['dates'][0],'A']*=4
        w,b=audit.independent_features(q,m,c)
        z=a['event_history'][0]['cells'][0];zz=b['event_history'][0]['cells'][0]
        self.assertEqual(z['reference_count'],zz['reference_count'])
        self.assertEqual(zz['reference_count'],756)
        self.assertEqual(zz['reference_below'],756)
        self.assertEqual(zz['reference_equal'],0)
        self.assertEqual(w['control'][0,0],.5)

    def test_overflow_is_error_not_missing_and_input_unchanged(self):
        p,m,c=fixture();m=single(m);before=copy.deepcopy(p)
        audit.independent_features(p,m,c)
        pd.testing.assert_frame_equal(p['adj'],before['adj'])
        for small,big in [(1.,1e200),(5e-324,1e308)]:
            q=copy.deepcopy(p);q['adj'].iloc[200,0]=small;q['adj'].iloc[201,0]=big
            with self.assertRaises(FloatingPointError):audit.independent_features(q,m,c)

    def test_schema_and_past_nonnumeric_rejected(self):
        p,m,c=fixture()
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,limit=True)
        with self.assertRaises(ValueError):audit.independent_features(p,m,c[::-1])
        bad=copy.deepcopy(m);bad.pop('etfs')
        with self.assertRaises(ValueError):audit.independent_features(p,bad,c)
        bad=copy.deepcopy(m);bad['symbols'][0]=['A','A']
        with self.assertRaises(ValueError):audit.independent_features(p,bad,c)
        q=copy.deepcopy(p);q['adj']=q['adj'].astype(object);q['adj'].iloc[500,0]='past poison'
        with self.assertRaises(ValueError):audit.independent_features(q,m,c)

    def test_independent_rejects_audit_count_state_tie_and_raw_mutation(self):
        p,m,c=fixture();m=single(m);v,a=audit.independent_features(p,m,c)
        for key in ['reference_count','reference_equal','matched_count']:
            bad=copy.deepcopy(a);bad['event_history'][0]['cells'][0][key]+=1
            with self.assertRaises(ValueError):audit.verify_features(p,m,c,v,bad)
        bad=copy.deepcopy(a);bad['event_history'][0]['current_calm']=not bad['event_history'][0]['current_calm']
        with self.assertRaises(ValueError):audit.verify_features(p,m,c,v,bad)
        bad=copy.deepcopy(v);bad['control'][0,0]=np.nextafter(bad['control'][0,0],np.inf)
        with self.assertRaises(ValueError):audit.verify_features(p,m,c,bad,a)


class DeltaTests(unittest.TestCase):
    def values(self):
        return dict(candidate=np.array([[.5,-.5],[.5,np.nan],[0.,-.25]]),control=np.array([[0.,.5],[-.5,np.nan],[.5,.25]]),available=np.array([[1,1],[1,0],[1,1]],bool))

    def test_direct_bounded_mapping_initial_and_missing_exact_zero(self):
        v=self.values();d=audit.independent_deltas(v);r=audit.verify_deltas(v,d)
        np.testing.assert_array_equal(d[audit.ARMS[1]],[[0.,0.],[.005,0.],[0.,-.0025]])
        self.assertEqual(r['max_absolute_delta'],.005)
        self.assertTrue(r['no_cross_sectional_ranking'])
        self.assertEqual(r['changed_cells'][audit.ARMS[1]],2)

    def test_reject_wrong_dose_reordered_arms_firstrow_missing_and_bounds(self):
        v=self.values();d=audit.independent_deltas(v)
        with self.assertRaises(ValueError):audit.independent_deltas(v,dose=.02)
        for i,j in [(0,0),(1,1),(2,0)]:
            bad=copy.deepcopy(d);bad[audit.ARMS[1]][i,j]=.006
            with self.assertRaises(ValueError):audit.verify_deltas(v,bad)
        bad=copy.deepcopy(v);bad['candidate'][1,1]=0.
        with self.assertRaises(ValueError):audit.independent_deltas(bad)


class OutcomeTests(unittest.TestCase):
    def test_three_arms_nine_paths_45_slices_and_tamper(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        receipt=audit.audit_outcomes(*args)
        self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(9,45))
        self.assertEqual(set(receipt['paths']),set(audit.ARMS))
        bad=copy.deepcopy(args)
        path=next(iter(bad[4][audit.ARMS[1]]))
        bad[4][audit.ARMS[1]][path]['metrics']['cagr']+=.01
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


if __name__=='__main__':unittest.main(verbosity=2)
