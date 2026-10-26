"""Synthetic-only independent191 tests; never load actual training/price arrays."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_relative_severity191_independent as r


def fixture():
    sessions=pd.bdate_range('2012-01-01','2021-03-31')
    _,last=r.calendar_map(sessions)
    months=pd.period_range('2013-01','2019-12',freq='M')
    sig=[last[np.datetime64(str(m),'M')] for m in months]
    end=[last[np.datetime64(str(m+1),'M')] for m in months]
    tm=dict(dates=[str(v) for v in sig],ends=[str(v) for v in end],symbols=[['A','B','C'] for _ in sig])
    meta=dict(dates=['2016-08-17','2016-08-31','2017-01-31','2018-01-31'],ends=['2016-08-31','2016-09-30','2017-02-28','2018-02-28'],symbols=[['A','B','C'] for _ in range(4)])
    t=np.arange(len(sessions),dtype=float)
    adj=pd.DataFrame({s:100*np.exp(.0001*t+.16*np.sin(t/(15+3*j))) for j,s in enumerate(['A','B','C','SPY'])},index=sessions)
    rng=np.random.default_rng(9191)
    return [adj,sessions,tm,meta,rng.uniform(-.5,.5,(len(sig)*3,10)),rng.uniform(-.5,.5,(12,10)),np.ones(len(sig)*3,bool),np.ones(len(sig)*3,bool),np.ones((4,3),bool)]


def prefix(f,n,truncate=False):
    f=copy.deepcopy(f)
    for key in ('dates','ends','symbols'):f[3][key]=f[3][key][:n]
    f[5]=f[5][:n*3];f[8]=f[8][:n]
    if truncate and n:f[0]=f[0].loc[:f[3]['dates'][-1]]
    return f


def small_labels():
    sessions=pd.bdate_range('2020-01-01','2020-03-31')
    adj=pd.DataFrame(100.,index=sessions,columns=['A','B','C','SPY'])
    args=[adj,sessions,['2020-01-31']*3,['A','B','C'],np.ones(3,bool),np.ones(3,bool),np.ones(3,bool)]
    return args


class Labels(unittest.TestCase):
    def test_severity_rounds_only_after_exact_subtraction(self):
        epsilon=2.**-52
        prices=(1.,1.+epsilon,1.-epsilon,1.)
        relative=r.Fraction.from_float(prices[1])*r.Fraction.from_float(prices[2])
        self.assertEqual(float(relative),1.)
        self.assertEqual(r.exact_targets(*prices),(2.**-104,1.))
        self.assertNotEqual(r.exact_targets(*prices)[0],max(0.,1.-float(relative)))
    def test_exact_thresholds_and_wealth_not_percentage_points(self):
        self.assertEqual(r.exact_targets(10,9,1,1),(.1,1.))
        self.assertGreater(r.exact_targets(10,np.nextafter(9.,0),1,1)[0],.1)
        self.assertEqual(r.exact_targets(10,10,1,1),(0.,0.))
        self.assertGreater(r.exact_targets(10,np.nextafter(10.,0),1,1)[0],0.)
        self.assertEqual(r.exact_targets(100,110,100,130),(float(r.Fraction(2,13)),1.))
        self.assertEqual(r.exact_targets(100,80,100,70),(0.,0.))
    def test_float_ratio_overflow_and_underflow_remain_exact(self):
        self.assertEqual(r.exact_targets(1e-300,1e300,1e-300,1e300),(0.,0.))
        self.assertEqual(r.exact_targets(1e300,1e-300,1e-300,1e300),(1.,1.))
        self.assertEqual(r.exact_targets(1e-300,1e300,1e300,1e-300),(0.,0.))
    def test_continuous_severity_order_and_near_total_rounding(self):
        results=[r.exact_targets(100.,end,100.,100.) for end in (99.,90.,50.,1.)]
        self.assertEqual([x[0] for x in results],[.01,.1,.5,.99])
        self.assertEqual([x[1] for x in results],[1.]*4)
        self.assertEqual(r.exact_targets(1e300,1e-300,1.,1.),(1.,1.))
    def test_four_endpoints_and_no_filling(self):
        f=small_labels();f[0].loc['2020-02-28']=[90,80,110,100]
        lab,_=r.labels_oracle(*f);np.testing.assert_array_equal(lab['candidate_target'],[.1,.2,0]);np.testing.assert_array_equal(lab['control_target'],[1,1,0])
        for value in (np.nan,np.inf,0.,-1.,'bad'):
            z=copy.deepcopy(f);z[0]=z[0].astype(object);z[0].loc['2020-02-28','SPY']=value
            a,_=r.labels_oracle(*z);self.assertFalse(a['good'].any());self.assertTrue(np.isnan(a['candidate_target']).all())
        f[0]=f[0].drop('2020-02-28');self.assertFalse(r.labels_oracle(*f)[0]['good'].any())
    def test_internal_path_unread_and_power2_scale_invariance(self):
        f=small_labels();f[0].loc['2020-02-28']=[70,90,110,100];a=r.labels_oracle(*f)[0]
        f[0]=f[0].astype(object);f[0].loc['2020-02-12']='unread';b=r.labels_oracle(*f)[0]
        for k in r.TARGETS:np.testing.assert_array_equal(a[k],b[k])
        f=small_labels();f[0].loc['2020-02-28']=[70,90,110,100];f[0]*=[2,4,.5,8]
        for k in r.TARGETS:np.testing.assert_array_equal(a[k],r.labels_oracle(*f)[0][k])
    def test_singleton_retains_raw_but_not_training(self):
        f=small_labels();f[4][1:]=False;lab,a=r.labels_oracle(*f)
        self.assertFalse(lab['good'].any());self.assertEqual(lab['candidate_target'][0],0);self.assertEqual(a['months'][0]['status'],'fewer_than_two_shared_rows')
    def test_calendar_exact_end_not_final_partial(self):
        cal=pd.bdate_range('2026-08-01','2026-10-31');adj=pd.DataFrame(100.,index=cal[cal<='2026-09-24'],columns=['A','B','SPY'])
        a,_=r.labels_oracle(adj,cal,['2026-08-31']*2,['A','B'],[True]*2,[True]*2,[True]*2)
        self.assertTrue((a['ends']==np.datetime64('2026-09-30')).all());self.assertFalse(a['good'].any())
        with self.assertRaises(ValueError):r.endpoint_schedule(cal,['2026-08-28'])
        with self.assertRaises(ValueError):r.endpoint_schedule(cal,['2026-08-31'],['2026-09-24'])
    def test_masks_and_identities_rejected(self):
        f=small_labels();f[4]=np.ones(3,int)
        with self.assertRaises(ValueError):r.labels_oracle(*f)
        f=small_labels();f[3][1]='A'
        with self.assertRaises(ValueError):r.labels_oracle(*f)


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=fixture();cls.b=r.independent_build(*cls.f)
    def test_same_source_full_build_and_counts(self):
        a=r.verify_build(*self.f,self.b);self.assertTrue(a['passed']);self.assertEqual(a['new_pooled_model_fits'],6)
        self.assertEqual(a['independent_model_fits'],6);self.assertFalse(a['computes_portfolio_returns'])
        self.assertTrue(all(z['cutoff']==str(z['year']-1)+'-12-31' for z in self.b['model_audit']['annual']))
    def test_weighted_augmented_solution_and_intercept(self):
        x=np.zeros((4,10));x[:,0]=[-.5,-.25,.25,.5];y=np.array([0.,0.,1.,1.]);w=np.ones(4)
        m,s,b,i=r.fit_oracle(x,y,w);z=(x[:,0]-m[0])/s[0]
        self.assertAlmostEqual(b[0],sum(z*(y-.5))/(sum(z*z)+10),14);self.assertEqual(i,.5)
        c=r.fit_oracle(x,np.ones(4),w);np.testing.assert_array_equal(c[2],np.zeros(10));self.assertEqual(c[3],1)
    def test_month_balance_decay_and_row_normalization(self):
        w=r.weights_oracle(['2018-01-31','2018-01-31','2019-01-31'],'2019-12-31')
        self.assertAlmostEqual(w.sum(),3);self.assertAlmostEqual(2*w[:2].sum(),w[2])
    def test_exact_18_month_boundary(self):
        f=copy.deepcopy(self.f);f[6][:]=False
        td=np.repeat(np.array(f[2]['dates'],dtype='datetime64[D]'),3);te=np.repeat(np.array(f[2]['ends'],dtype='datetime64[D]'),3)
        valid=(td>=np.datetime64('2014-06-01'))&(te<=np.datetime64('2015-12-31'));f[6][valid]=True
        a=r.independent_build(*f);self.assertEqual(a['model_audit']['annual'][0]['training_months'],18);self.assertTrue(a['arrays']['common_available'][1].all())
        f[6][td<np.datetime64('2014-07-01')]=False;a=r.independent_build(*f);self.assertEqual(a['model_audit']['annual'][0]['training_months'],17);self.assertFalse(a['arrays']['common_available'][1].any())
    def test_prefix_price_truncation_no_future_query_labels(self):
        for n in range(5):
            args=prefix(self.f,n,True);a=r.independent_build(*args)
            for key in r.EXTRA:
                np.testing.assert_array_equal(a['arrays'][key],self.b['arrays'][key][:n])
    def test_future_training_poison_does_not_change_earlier_fit(self):
        f=copy.deepcopy(self.f);late=np.repeat(np.array(f[2]['ends'],dtype='datetime64[D]'),3)>np.datetime64('2015-12-31');f[4][late]=.5
        f[0].loc['2016-01-01':]*=2
        a=r.independent_build(*prefix(f,2));
        for k in r.EXTRA:np.testing.assert_array_equal(a['arrays'][k],self.b['arrays'][k][:2])
    def test_training_only_scaler_and_statistics_audit_tampering(self):
        f=copy.deepcopy(self.f);f[5]*=-1;a=r.independent_build(*f)
        for x,y in zip(a['model_audit']['annual'],self.b['model_audit']['annual']):self.assertEqual(x.get('scaler_mean_sha256'),y.get('scaler_mean_sha256'))
        bad=copy.deepcopy(self.b);bad['model_audit']['annual'][0]['target_positive_counts']['candidate_target']+=1
        with self.assertRaises(ValueError):r.verify_build(*self.f,bad)
    def test_continuous_annual_statistics_and_bounds(self):
        row=self.b['model_audit']['annual'][0];ids=row['training_indices']
        y=self.b['labels']['candidate_target'][ids]
        self.assertEqual(row['target_positive_counts']['candidate_target'],int(np.count_nonzero(y>0)))
        self.assertEqual(row['target_sums']['candidate_target'],float(y.sum()))
        self.assertEqual(row['target_means']['candidate_target'],float(y.mean()))
        self.assertEqual(row['target_maxima']['candidate_target'],float(y.max()))
        self.assertNotEqual(row['target_sums']['candidate_target'],row['target_positive_counts']['candidate_target'])
        for value in (-.1,1.1):
            lab=copy.deepcopy(self.b['labels']);lab['candidate_target'][0]=value;lab['control_target'][0]=1.
            with self.assertRaises(ValueError):r.annual_oracle(lab,self.f[4],['2016-08-31'],self.f[5][:1],[True],[True])
        bad=copy.deepcopy(self.b);bad['model_audit']['annual'][0]['weighted_target_means']['candidate_target']+=.01
        with self.assertRaises(ValueError):r.verify_build(*self.f,bad)
        bad=copy.deepcopy(self.b);bad['label_audit']['rounded_severity_once']=False
        with self.assertRaises(ValueError):r.verify_build(*self.f,bad)
    def test_constant_continuous_target_and_empty_statistics(self):
        labels=copy.deepcopy(self.b['labels']);labels['candidate_target'][:]=.25;labels['control_target'][:]=1.
        prediction,audit=r.annual_oracle(labels,self.f[4],['2016-08-31'],self.f[5][:1],[True],[True])
        np.testing.assert_array_equal(prediction['candidate_prediction'],[.25])
        self.assertEqual(audit['annual'][0]['weighted_target_means']['candidate_target'],.25)
        labels['good'][:]=False
        _,audit=r.annual_oracle(labels,self.f[4],['2016-08-31'],self.f[5][:1],[True],[True])
        self.assertEqual(audit['new_pooled_model_fits'],0)
        row=audit['annual'][0]
        self.assertEqual(row['target_positive_counts'],{k:0 for k in r.TARGETS})
        self.assertEqual(row['target_sums'],{k:0. for k in r.TARGETS})
        self.assertEqual(row['target_means'],{k:None for k in r.TARGETS})
        self.assertEqual(row['target_maxima'],{k:None for k in r.TARGETS})
    def test_control_exactly_reuses190_on_synthetic_inputs(self):
        import monthly_relative_tail190_independent as previous
        old=previous.independent_build(*self.f)
        for key in ('dates','ends','symbols','control_target','base_good','good'):
            a,b=self.b['labels'][key],old['labels'][key]
            self.assertEqual((a.dtype,a.shape,a.tobytes()),(b.dtype,b.shape,b.tobytes()))
        for key in ('control_prediction','control_rank','common_available'):
            a,b=self.b['arrays'][key],old['arrays'][key]
            self.assertEqual((a.dtype,a.shape,a.tobytes()),(b.dtype,b.shape,b.tobytes()))
        bad=copy.deepcopy(self.b);bad['model_audit']['new_pooled_model_fits']=0
        with self.assertRaises(ValueError):r.verify_build(*self.f,bad)
    def test_label_and_rank_tampering_rejected(self):
        bad=copy.deepcopy(self.b);bad['labels']['candidate_target'][0]=1-bad['labels']['candidate_target'][0]
        with self.assertRaises(AssertionError):r.verify_build(*self.f,bad)
        bad=copy.deepcopy(self.b);bad['arrays']['candidate_rank'][1,0]+=1e-14
        with self.assertRaises(AssertionError):r.verify_build(*self.f,bad)
    def test_zero_one_ties_negative_ranks_and_deltas(self):
        np.testing.assert_array_equal(r.rank_oracle([2.,2.,2.]),[0,0,0]);np.testing.assert_array_equal(r.rank_oracle([1]),[0])
        np.testing.assert_array_equal(r.rank_oracle(-np.array([.1,.8,.8])),[.5,-.25,-.25])
        d=r.independent_deltas(self.b['arrays']);a=r.verify_deltas(self.b['arrays'],d)
        self.assertTrue(a['initial_delta_zero']);self.assertLessEqual(max(np.max(abs(v)) for v in d.values()),.005)
        d['candidate'][1,0]+=1e-14
        with self.assertRaises(AssertionError):r.verify_deltas(self.b['arrays'],d)
    def test_constant_labels_zero_rank_and_query_missing(self):
        f=copy.deepcopy(self.f);f[0][:]=100.;a=r.independent_build(*f)
        for k in ('candidate_rank','control_rank'):np.testing.assert_array_equal(a['arrays'][k],np.zeros((4,3)))
        f[5][4,0]=np.nan;a=r.independent_build(*f);self.assertFalse(a['arrays']['common_available'][1,1]);self.assertTrue(a['arrays']['common_available'][1,0])
    def test_unchanged_inputs_and_no_production_import(self):
        before=copy.deepcopy(self.f);r.independent_build(*self.f)
        pd.testing.assert_frame_equal(before[0],self.f[0])
        for i in (4,5,6,7,8):np.testing.assert_array_equal(before[i],self.f[i])
        tree=ast.parse(Path(r.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):self.assertFalse(any('relative_severity191_model' in x.name for x in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn('relative_severity191_model',node.module or '')
    def test_accounting_helper_sha_guard(self):
        self.assertTrue(callable(r.accounting().audit_path))
        with patch.object(r,'ACCOUNTING_SHA','0'*64):
            with self.assertRaises(ValueError):r.accounting()
    def test_original_episode_ordered_choices_and_guard(self):
        from test_monthly_price_delay140_prepare_review import score_fixture
        a,m=score_fixture();m.update(dates=['2016-08-17','2016-08-31'],ends=['2016-08-31','2016-09-30'])
        helper=r.frozen_source_oracle().old.original_helper();x=helper.original_prefix(a,m,None)
        d=dict(candidate=np.zeros((2,5)),control=np.zeros((2,5)));d['candidate'][1,2]=.005
        scores={};choices={}
        for arm,key in zip(r.ARMS,(None,'candidate','control')):
            pre=x['local'].copy()
            if key:mask=d[key]!=0;pre[mask]+=d[key][mask]
            choices[arm],scores[arm],_=helper.episode(pre,x)
        self.assertTrue(r.verify_selection(a,m,d,scores,choices)['one_original_episode_checked'])
        choices[r.ARMS[1]][1]=list(reversed(choices[r.ARMS[1]][1]))
        with self.assertRaises(ValueError):r.verify_selection(a,m,d,scores,choices)
    def test_cross_production_both_synthetic_fixtures(self):
        import monthly_relative_severity191_model as p
        from test_monthly_relative_severity191_model import build_fixture
        for args in (fixture(),build_fixture()):
            got=p.build(*args);self.assertTrue(r.verify_build(*args,got)['passed'])
            for n in (0,1,2):
                part=prefix(args,n,True);self.assertTrue(r.verify_build(*part,p.build(*part))['passed'])


if __name__=='__main__':unittest.main(verbosity=2)
