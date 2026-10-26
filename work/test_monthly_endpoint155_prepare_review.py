"""Synthetic-only tests of the independent endpoint audit; no real arrays."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_endpoint155_prepare_review as r
from test_monthly_price_delay140_prepare_review import score_fixture


def fixture():
    sessions=pd.bdate_range('2013-01-01','2025-03-31')
    months=pd.period_range('2013-01','2023-12',freq='M')
    signals=np.array([sessions[sessions.to_period('M')==m][-1] for m in months],dtype='datetime64[D]')
    days=np.repeat(signals,3);symbols=np.tile(np.array(['A','B','C']),len(signals));good=np.ones(len(days),bool)
    t=np.arange(len(sessions))[:,None];j=np.arange(3)[None,:]
    prices=100*np.exp(.0001*t+.02*np.sin(t*(.02+j*.01)))
    close=pd.DataFrame(prices,index=sessions,columns=['A','B','C'])
    x=.4*np.sin(np.arange(len(days))[:,None]*.17+np.arange(10)[None,:]*.3)
    return close,sessions,days,symbols,good,x

def labels(f=None):
    close,sessions,d,s,g,x=fixture() if f is None else f
    return r.labels_oracle(close,sessions,d,s,g,g,np.isfinite(x).all(axis=1))

def model_fixture():
    f=fixture();lab=labels(f)
    events=np.repeat(np.array(['2020-01-31','2020-06-30','2021-03-31'],dtype='datetime64[D]'),3)
    x=.3*np.cos(np.arange(9)[:,None]*.13+np.arange(10)[None,:]*.2)
    return lab,f[-1],events,x,np.ones(9,bool),np.ones(9,bool)

def score_data():
    a,m=score_fixture();shape=a['baseline'].shape
    m=dict(m,dates=['2020-01-31','2020-02-28'],ends=['2020-02-28','2020-03-31'])
    available=a['eligible'].copy();p=np.tile(np.arange(shape[1]),(shape[0],1)).astype(float)
    p[~available]=np.nan
    a.update(r.ranked_arrays(dict(mean_return_prediction=p,loss_prediction=-p,common_available=available),shape))
    return a,m


class EndpointTests(unittest.TestCase):
    def setUp(self):self.f=fixture()
    def test_positive_negative_exact_zero(self):
        f=list(self.f);f[0]=f[0].copy();d=f[2][0];end=r.endpoint_schedule(f[1],[d])[0]
        f[0].loc[pd.Timestamp(d)]=[100,100,100];f[0].loc[pd.Timestamp(end)]=[110,90,100]
        a=labels(f);np.testing.assert_allclose(a['return_target'][:3],[.1,-.1,0]);np.testing.assert_array_equal(a['loss_target'][:3],[0,1,0])
    def test_no_centering_or_clipping(self):
        f=list(self.f);f[0]=f[0].copy();f[0].iloc[:]=100;f[0].loc['2013-02-28']=[500,300,200]
        a=labels(f);np.testing.assert_array_equal(a['return_target'][:3],[4,2,1]);np.testing.assert_array_equal(a['loss_target'][:3],[0,0,0])
    def test_no_benchmark_needed(self):self.assertTrue(labels(self.f)['good'].all())
    def test_no_path_extrema(self):
        f=list(self.f);f[0]=f[0].copy();f[0].loc['2013-02-12','A']=.001
        np.testing.assert_array_equal(labels(f)['return_target'],labels(self.f)['return_target'])
    def test_missing_end_never_shortens(self):
        f=list(self.f);f[0]=f[0].drop('2013-02-28');a=labels(f)
        self.assertEqual(str(a['ends'][0]),'2013-02-28');self.assertTrue(np.isnan(a['return_target'][:3]).all());self.assertFalse(a['good'][:3].any())
    def test_final_september_not_24(self):
        sessions=pd.bdate_range('2026-08-01','2026-10-31');c=pd.DataFrame(100.,index=sessions[sessions<='2026-09-24'],columns=['A','B'])
        a=r.labels_oracle(c,sessions,['2026-08-31']*2,['A','B'],[True]*2,[True]*2,[True]*2)
        self.assertEqual(str(a['ends'][0]),'2026-09-30');self.assertFalse(a['good'].any())
    def test_december_january(self):self.assertEqual(str(r.endpoint_schedule(self.f[1],['2019-12-31'])[0]),'2020-01-31')
    def test_leap_year(self):self.assertEqual(str(r.endpoint_schedule(self.f[1],['2020-01-31'])[0]),'2020-02-28')
    def test_declared_endpoint_mismatch(self):
        with self.assertRaises(AssertionError):r.labels_oracle(*self.f[:4],self.f[4],self.f[4],self.f[4],expected_ends=self.f[2])
    def test_missing_symbol(self):
        f=list(self.f);f[0]=f[0].drop(columns='A');a=labels(f);self.assertFalse(a['good'][::3].any())
    def test_zero_negative_nan_infinite_endpoints(self):
        for value in (0.,-1.,np.nan,np.inf):
            f=list(self.f);f[0]=f[0].copy();f[0].loc['2013-02-28','A']=value
            self.assertFalse(labels(f)['good'][0])
    def test_raw_overflow_no_label(self):
        f=list(self.f);f[0]=f[0].copy();f[0].loc['2013-01-31','A']=1e-300;f[0].loc['2013-02-28','A']=1e300
        self.assertFalse(labels(f)['good'][0])
    def test_singleton_month(self):
        f=list(self.f);f[4]=f[4].copy();f[4][1:3]=False;a=labels(f);self.assertFalse(a['good'][:3].any())
    def test_inherited_common_mask_preserved(self):
        f=list(self.f);f[4]=f[4].copy();f[4][0]=False;a=labels(f);self.assertFalse(a['base_good'][0]);self.assertTrue(np.isnan(a['return_target'][0]))
    def test_finite_predictors_required(self):
        f=list(self.f);f[-1]=f[-1].copy();f[-1][0,0]=np.nan;self.assertFalse(labels(f)['good'][0])
    def test_scale_invariance(self):
        f=list(self.f);f[0]=f[0]*[.5,2,4];a,b=labels(f),labels(self.f)
        np.testing.assert_allclose(a['return_target'],b['return_target'],atol=1e-14,rtol=0);np.testing.assert_array_equal(a['loss_target'],b['loss_target'])
    def test_non_monthend_rejected(self):
        with self.assertRaises(ValueError):r.endpoint_schedule(self.f[1],['2020-01-30'])
    def test_duplicate_row_identity(self):
        f=list(self.f);f[3]=f[3].copy();f[3][1]=f[3][0]
        with self.assertRaises(ValueError):labels(f)
    def test_nonboolean_mask_rejected(self):
        f=list(self.f);f[4]=f[4].astype(int)
        with self.assertRaises(ValueError):labels(f)
    def test_calendar_order_rejected(self):
        with self.assertRaises(ValueError):r.endpoint_schedule(self.f[1][::-1],self.f[2])
    def test_partial_query_must_be_in_calendar(self):
        sessions=self.f[1][self.f[1]!=pd.Timestamp('2016-08-17')]
        with self.assertRaises(ValueError):r.query_monthends(sessions,['2016-08-17','2016-08-31'])
    def test_partial_first_fallback(self):
        np.testing.assert_array_equal(r.query_monthends(self.f[1],['2016-08-17','2016-08-31']),[False,True])
    def test_other_partial_rejected(self):
        with self.assertRaises(ValueError):r.query_monthends(self.f[1],['2016-08-31','2016-09-15'])
    def test_unknown_initial_partial_rejected(self):
        with self.assertRaises(ValueError):r.query_monthends(self.f[1],['2016-08-18'])
    def test_inputs_unchanged(self):
        before=self.f[0].copy();mask=self.f[4].copy();labels(self.f);pd.testing.assert_frame_equal(before,self.f[0]);np.testing.assert_array_equal(mask,self.f[4])


class FitTests(unittest.TestCase):
    def setUp(self):self.f=model_fixture()
    def run_model(self,f=None,**kw):return r.model_oracle(*(self.f if f is None else f),**kw)
    def test_annual_previous_december(self):
        a=self.run_model()['audit']['annual'][0];self.assertEqual(a['cutoff'],'2019-12-31');self.assertEqual(a['latest_label_end'],'2019-12-31')
    def test_five_year_exclusive(self):
        a=self.run_model()['audit']['annual'][0];self.assertEqual(a['lookback_exclusive_start'],'2014-12-31');self.assertEqual(a['first_training_signal'],'2015-01-30')
    def test_identical_targets_identical_predictions(self):
        f=copy.deepcopy(self.f);f[0]['return_target']=f[0]['loss_target'].copy();v=self.run_model(f);np.testing.assert_array_equal(v['mean_return_prediction'],v['loss_prediction'])
    def test_shared_target_validity(self):
        f=copy.deepcopy(self.f);f[0]['return_target'][:250]=np.nan;a=self.run_model(f);b=self.run_model();self.assertLess(a['audit']['annual'][0]['training_rows'],b['audit']['annual'][0]['training_rows'])
    def test_weight_month_totals_and_half_life(self):
        d=np.array(['2018-01-31','2018-01-31','2019-01-31'] ,dtype='datetime64[D]');w=r.weights_oracle(d,np.datetime64('2019-12-31'));self.assertAlmostEqual(w.sum(),3);self.assertAlmostEqual(w[:2].sum()*2,w[2])
    def test_train_only_scaler(self):
        f=copy.deepcopy(self.f);f[3][:6]*=-1;a=self.run_model();b=self.run_model(f)
        for left,right in zip(a['audit']['annual'],b['audit']['annual']):self.assertEqual(left['scaler_mean_sha256'],right['scaler_mean_sha256'])
    def test_future_labels_ignored_by_old_fit(self):
        f=copy.deepcopy(self.f);late=f[0]['ends']>np.datetime64('2019-12-31');f[0]['return_target'][late]=1000.;f[0]['loss_target'][late]=1.;f[1][late]=.5
        a,b=self.run_model(),self.run_model(f);np.testing.assert_array_equal(a['mean_return_prediction'][:6],b['mean_return_prediction'][:6]);np.testing.assert_array_equal(a['loss_prediction'][:6],b['loss_prediction'][:6])
    def test_query_future_target_unneeded(self):
        f=copy.deepcopy(self.f);late=f[0]['dates']>=np.datetime64('2020-01-01');f[0]['good'][late]=False
        self.assertTrue(self.run_model(f)['common_available'][:6].all())
    def test_exactly_18_mature_months(self):
        f=copy.deepcopy(self.f);f[0]['good'][:]=False;pick=(f[0]['dates']>=np.datetime64('2018-06-01'))&(f[0]['ends']<=np.datetime64('2019-12-31'));f[0]['good'][pick]=True
        self.assertEqual(len(np.unique(f[0]['dates'][pick].astype('datetime64[M]'))),18);self.assertTrue(self.run_model(f)['common_available'][:6].all())
    def test_17_mature_months_cold(self):
        f=copy.deepcopy(self.f);f[0]['good'][:]=False;pick=(f[0]['dates']>=np.datetime64('2018-07-01'))&(f[0]['ends']<=np.datetime64('2019-12-31'));f[0]['good'][pick]=True
        self.assertFalse(self.run_model(f)['common_available'][:6].any())
    def test_constant_binary_zero_intercept(self):
        f=copy.deepcopy(self.f);f[0]['loss_target'][:]=0.;a=self.run_model(f);np.testing.assert_array_equal(a['loss_prediction'],np.zeros(9))
    def test_constant_binary_one_intercept(self):
        f=copy.deepcopy(self.f);f[0]['loss_target'][:]=1.;a=self.run_model(f);np.testing.assert_allclose(a['loss_prediction'],np.ones(9),atol=1e-15)
    def test_unpenalized_intercept(self):
        x=np.zeros((6,10));fit=r.fit_oracle(x,np.full(6,7.),np.ones(6));self.assertEqual(fit[3],7.);np.testing.assert_array_equal(fit[2],np.zeros(10))
    def test_direct_one_feature_ridge(self):
        x=np.zeros((4,10));x[:,0]=[-.5,-.25,.25,.5];y=np.array([0,0,1,1]);w=np.ones(4);fit=r.fit_oracle(x,y,w)
        z=x[:,0]/np.sqrt(np.mean(x[:,0]**2));expected=np.sum(z*(y-.5))/(np.sum(z*z)+10)
        self.assertAlmostEqual(fit[2][0],expected,14)
    def test_query_nonfinite_only_itself_missing(self):
        f=copy.deepcopy(self.f);f[3][0,0]=np.nan;a=self.run_model(f);self.assertFalse(a['common_available'][0]);self.assertTrue(a['common_available'][1:].all())
    def test_query_maturity_flag_required(self):
        f=copy.deepcopy(self.f);f[-1][0]=False;a=self.run_model(f);self.assertFalse(a['common_available'][0])
    def test_nonbinary_label_rejected(self):
        f=copy.deepcopy(self.f);f[0]['loss_target'][0]=.5
        with self.assertRaises(ValueError):self.run_model(f)
    def test_memoization_does_not_change_predictions(self):
        cache={};a=self.run_model(fit_cache=cache);b=self.run_model(fit_cache=cache);self.assertTrue(cache);np.testing.assert_array_equal(a['loss_prediction'],b['loss_prediction'])
    def test_future_queries_do_not_affect_prefix(self):
        f=copy.deepcopy(self.f);f[3][6:]=np.inf;np.testing.assert_array_equal(self.run_model(f)['loss_prediction'][:6],self.run_model()['loss_prediction'][:6])
    def test_ten_columns_only(self):
        f=list(copy.deepcopy(self.f));f[1]=f[1][:,:9]
        with self.assertRaises(ValueError):self.run_model(f)


class RankScoreTests(unittest.TestCase):
    def test_average_rank_endpoints(self):np.testing.assert_array_equal(r.rank_oracle([2,1,3]),[0,-.5,.5])
    def test_ties(self):np.testing.assert_array_equal(r.rank_oracle([1,1,3]),[-.25,-.25,.5])
    def test_all_ties_zero(self):np.testing.assert_array_equal(r.rank_oracle([1,1,1]),[0,0,0])
    def test_zero_and_one_population(self):self.assertEqual(len(r.rank_oracle([])),0);np.testing.assert_array_equal(r.rank_oracle([3]),[0])
    def test_negative_loss_prefers_lower(self):
        a=r.ranked_arrays(dict(mean_return_prediction=[1.,2.],loss_prediction=[.1,.9],common_available=[True,True]),(1,2));np.testing.assert_array_equal(a['negative_loss_rank'],[[.5,-.5]])
    def test_missing_rank_zero(self):
        a=r.ranked_arrays(dict(mean_return_prediction=[np.nan,2.],loss_prediction=[np.nan,.1],common_available=[False,True]),(1,2));np.testing.assert_array_equal(a['mean_return_rank'],[[0,0]])
    def test_asymmetric_missing_rejected(self):
        with self.assertRaises(ValueError):r.ranked_arrays(dict(mean_return_prediction=[1.],loss_prediction=[np.nan],common_available=[True]),(1,1))
    def test_single_episode_per_arm_and_additive_dose(self):
        a,m=score_data();h=r.old.original_helper();original=h.episode;seen=[]
        def capture(pre,x):seen.append(pre.copy());return original(pre,x)
        with patch.object(r.old,'original_helper',return_value=h),patch.object(h,'episode',side_effect=capture):r.score_oracle(a,m)
        self.assertEqual(len(seen),3)
        for pre,key in zip(seen[1:],r.EXTRA[2:4]):np.testing.assert_allclose(pre-seen[0],.01*a[key],atol=1e-16,rtol=0)
    def test_zero_corrections_exact_baseline(self):
        a,m=score_data();a['mean_return_rank'][:]=0;a['negative_loss_rank'][:]=0;s,c=r.score_oracle(a,m)
        for n in r.NAMES[1:]:self.assertEqual(c[n],c[r.NAMES[0]]);self.assertEqual(s[n].tobytes(),s[r.NAMES[0]].tobytes())
    def test_missing_exact_baseline(self):
        a,m=score_data();a['common_available'][:]=False;a['mean_return_rank'][:]=0;a['negative_loss_rank'][:]=0;s,c=r.score_oracle(a,m)
        for n in r.NAMES[1:]:self.assertEqual(c[n],c[r.NAMES[0]]);self.assertEqual(s[n].tobytes(),s[r.NAMES[0]].tobytes())
    def test_score_prefixes(self):
        a,m=score_data();s,c=r.score_oracle(a,m)
        for n in range(3):
            ss,cc=r.score_oracle(a,m,n)
            for arm in r.NAMES:self.assertEqual(cc[arm],c[arm][:n]);self.assertEqual(ss[arm].tobytes(),s[arm][:n].tobytes())
    def test_out_of_range_rejected(self):
        a,m=score_data();a['mean_return_rank'][0,0]=.51
        with self.assertRaises(ValueError):r.score_oracle(a,m)
    def test_nonzero_missing_rejected(self):
        a,m=score_data();a['common_available'][0,0]=False;a['mean_return_rank'][0,0]=.2
        with self.assertRaises(ValueError):r.score_oracle(a,m)
    def test_forecast_near_tie_rank_mismatch_rejected(self):
        e=r.ranked_arrays(dict(mean_return_prediction=[0.,1e-14],loss_prediction=[0.,1.],common_available=[True,True]),(1,2));a=copy.deepcopy(e);a['mean_return_rank']*=-1
        with self.assertRaises(AssertionError):r.check_forecasts(a,e)
    def test_source_has_no_producer_import(self):
        tree=ast.parse(Path(r.__file__).read_text());names=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):names.extend(x.name for x in n.names)
            elif isinstance(n,ast.ImportFrom):names.append(n.module or '')
        self.assertFalse(any('endpoint155_model' in x or 'endpoint155_runner' in x for x in names))



class AuditSchemaTests(unittest.TestCase):
    def test_label_audit_contains_every_month(self):
        a=labels();audit=r.label_audit(a);self.assertEqual(len(audit['months']),132);self.assertEqual(audit['rows'],396)
        self.assertEqual(audit['good_rows'],int(a['good'].sum()));self.assertFalse(audit['clipped']);self.assertFalse(audit['centered'])
    def test_shared_model_audit_reconstructed(self):
        f=model_fixture();forecast=r.model_oracle(*f);lab,x=f[:2];annual=[]
        for row in forecast['audit']['annual']:
            ids=np.flatnonzero(lab['good']&(lab['dates']>np.datetime64(row['lookback_exclusive_start']))&(lab['ends']<=np.datetime64(row['cutoff'])))
            item={k:row[k] for k in ('year','cutoff','lookback_exclusive_start','training_rows','training_months','prediction_rows','latest_label_end','training_indices_sha256','status')}
            item.update(training_indices=ids.tolist(),training_dates_sha256=r.digest(lab['dates'][ids]),training_symbols_sha256=r.digest(lab['symbols'][ids]),model_fits=2)
            item.update({k:row[k] for k in ('weight_sum','weights_sha256','scaler_mean_sha256','scaler_scale_sha256')})
            item.update(target_sha256={k:r.digest(lab[k][ids]) for k in ('return_target','loss_target')},shared_rows_weights_scalers=True);annual.append(item)
        saved=dict(alpha=10.,half_life_months=12.,lookback_years=5,minimum_mature_months=18,new_pooled_model_fits=4,local_model_fits=0,shared_training_mask_sha256=r.digest(lab['good']),training_only_standardization=True,calibrated_probabilities=False,target_clipping=False,target_centering=False,annual=annual)
        r.verify_model_audit(saved,forecast,lab,x)
        for key in ('training_indices_sha256','weights_sha256','scaler_mean_sha256','scaler_scale_sha256'):
            bad=copy.deepcopy(saved);bad['annual'][0][key]='0'*64
            with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):r.verify_model_audit(bad,forecast,lab,x)
    def test_query_mask_does_not_reuse_old_forecast_availability(self):
        a,m=score_data();a['forecast_available'][:]=False
        scores,choices=r.score_oracle(a,m);self.assertEqual(set(scores),set(r.NAMES))
    def test_123_independent_score_prefixes(self):
        a,m=score_data()
        for key,value in list(a.items()):
            if isinstance(value,np.ndarray) and value.shape and value.shape[0]==2:a[key]=np.tile(value,(61,)+tuple(1 for _ in value.shape[1:]))
        m=dict(m,dates=['2020-01-31']*122,ends=['2020-02-28']*122,symbols=m['symbols']*61)
        full,choices=r.score_oracle(a,m)
        for n in range(123):
            short,cc=r.score_oracle(a,m,n)
            for name in r.NAMES:self.assertEqual(cc[name],choices[name][:n]);self.assertEqual(short[name].tobytes(),full[name][:n].tobytes())

class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.p=self.root/r.STUDY
        def write(rel,obj):
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p
        self.write=write;self.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.design=write(r.DESIGN,{});previous=write(r.PREVIOUS,{});sourceplan=write(r.SOURCE_PLAN,{});calendar=write(r.CALENDAR,{})
        for name,path in [('DESIGN_SHA',self.design),('PREVIOUS_SHA',previous),('SOURCE_PLAN_SHA',sourceplan),('CALENDAR_SHA',calendar)]:
            p=patch.object(r,name,self.sha(path));p.start();self.addCleanup(p.stop)
        reviewer=write('reviewer.py',{})
        patcher=patch.object(r,'source_files',return_value=[reviewer]);patcher.start();self.addCleanup(patcher.stop)
        proposal=dict(round=155,status='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION',arms=list(r.NAMES),primary_challenger=r.NAMES[2],measurement=r.measurement_contract(),selection_contract=r.selection_contract(),design=dict(path=r.DESIGN,sha256=r.DESIGN_SHA),previous_completion=dict(path=r.PREVIOUS,sha256=r.PREVIOUS_SHA),actual_new_labels=0,actual_new_model_fits=0,actual_new_orders=0,actual_new_returns=0)
        pp=write('proposal.json',proposal)
        original=write(r.LOCAL+'/DATA.npz',{})
        write(r.LOCAL+'/CACHE_HASHES.json',{'DATA.npz':self.sha(original)})
        rawfiles=[write('raw/'+str(i)+'.json',{}) for i in range(64)];raw={str(p):self.sha(p) for p in rawfiles}
        calendar_source=write('calendar_source.py',{});prov={'source_hashes':{str(calendar_source):self.sha(calendar_source)}}
        write(r.LOCAL+'/PLAN.json',dict(raw_source_hashes=raw,calendar=prov))
        paths={'proposal':pp,'design':self.design,'calendar':calendar,'meta':write(r.LOCAL+'/META.json',{}),'policy':write(r.LOCAL+'/INCUMBENT_POLICY.json',{}),'original_orders':write(r.LOCAL+'/local_linear_ORDERS.json',[]),'prices':write(r.old.PRICES,{}),'protected_manifest':write('protect.json',{}),'train_inputs':write(r.TARGET+'/TRAIN_INPUTS.npz',{}),'training_meta':write(r.TRAIN_SOURCE+'/META.json',{})}
        hashes={str(p):self.sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.plan=dict(round=155,status='REGISTERED',arms=list(r.NAMES),primary_challenger=r.NAMES[2],output_directory=str(self.p),runtime=dict(python=r.sys.version,numpy=np.__version__,pandas=pd.__version__),hashes=hashes,raw_source_hashes=raw,calendar=prov,independent_prepare_source_hashes={str(reviewer):self.sha(reviewer)},inputs={k:dict(path=str(p),sha256=self.sha(p)) for k,p in paths.items()},tolerances=dict(feature_atol=1e-12,feature_rtol=0,scores_atol=1e-12,scores_rtol=0),actual_label_calculations_before_registration=0,actual_feature_calculations_before_registration=0,actual_model_fits_before_registration=0,actual_returns_before_registration=0)
        self.plan_path=write(r.STUDY+'/PLAN.json',self.plan)
        cache={}
        for name in r.CACHE_FILES:cache[name]=self.sha(write(r.STUDY+'/'+name,{}))
        write(r.STUDY+'/CACHE.json',cache)
    def check(self):return r.validate_plan(r.old.Reader(self.root),self.plan_path,self.sha(self.plan_path))
    def test_actual_metadata_binding_path(self):self.assertEqual(self.check()[0],self.p)
    def test_wrong_hash_rejected(self):
        with self.assertRaises(ValueError):r.validate_plan(r.old.Reader(self.root),self.plan_path,'0'*64)
    def test_actual_phase_cannot_begin_after_returns(self):
        self.write(r.STUDY+'/RESULTS.json',{})
        with self.assertRaises(ValueError):self.check()
    def test_registered_runtime_guard(self):
        self.plan['runtime']['python']='wrong';self.write(r.STUDY+'/PLAN.json',self.plan)
        with self.assertRaises(ValueError):self.check()
    def test_proposal_semantic_change_rejected(self):
        self.plan['inputs']['design']['sha256']='0'*64;self.write(r.STUDY+'/PLAN.json',self.plan)
        with self.assertRaises(ValueError):self.check()
    def test_missing_original_raw_member(self):
        self.plan['raw_source_hashes'].pop(next(iter(self.plan['raw_source_hashes'])));self.write(r.STUDY+'/PLAN.json',self.plan)
        with self.assertRaises(ValueError):self.check()
    def test_wrong_source_closure_rejected(self):
        self.plan['independent_prepare_source_hashes']={str(self.design):self.sha(self.design)};self.write(r.STUDY+'/PLAN.json',self.plan)
        with self.assertRaises(ValueError):self.check()
    def test_cache_must_include_all_labels_and_audits(self):
        cache=json.loads((self.p/'CACHE.json').read_text());del cache['LABELS.npz'];self.write(r.STUDY+'/CACHE.json',cache)
        with self.assertRaises(ValueError):self.check()
    def test_novelty_model_not_reclassified_as_new(self):
        self.assertEqual(r.measurement_contract()['dose'],.01);self.assertFalse(r.measurement_contract()['probability_calibration'])

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--receipt');args=cli.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if args.receipt:
        with Path(args.receipt).open('x') as f:json.dump(r.synthetic_receipt(result),f,indent=2,sort_keys=True);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
