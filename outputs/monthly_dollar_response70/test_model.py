"""Synthetic fixed-base projection, annual cohort and fallback tests."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dollar126_model_test',P/'model.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def fixture():
    rng=np.random.default_rng(456)
    months=pd.period_range('2010-01',periods=72,freq='M')
    d=np.repeat(months.to_timestamp(how='end').normalize().to_numpy(),3)
    ends=np.repeat((months+1).to_timestamp(how='end').normalize().to_numpy(),3)
    x=rng.uniform(-.45,.45,(len(d),10));c=np.repeat(np.sin(np.arange(72)),3)
    c=c+x[:,2]*.2; y=.4*x[:,0]+.3*c+.01*rng.normal(size=len(d))
    q=rng.uniform(-.45,.45,(6,10));ed=np.repeat(np.array(['2016-02-29','2017-02-28'],dtype='datetime64[D]'),3)
    tc=dict(feature_raw_dollar=c,feature_partial_dollar=-c,response_available=np.ones(len(d),bool))
    ec=dict(feature_raw_dollar=np.array([.4]*3+[-.2]*3),feature_partial_dollar=np.array([-.4]*3+[.2]*3),response_available=np.ones(6,bool))
    return dict(dates=d,ends=ends,X=x,targets={name:y.copy()for name in f.TARGETS},common_good=np.ones(len(d),bool),
        event_dates=ed,event_X=q,event_good=np.ones(6,bool),train_features=tc,event_features=ec)

class AddonTests(unittest.TestCase):
    def setUp(self):
        data=fixture();self.x=data['X'];self.y=data['targets']['log_excess'];self.c=data['train_features']['feature_raw_dollar']
        self.w=np.linspace(.5,1.5,len(self.x));self.q=data['event_X'];self.qc=data['event_features']['feature_raw_dollar']
        self.base=f.numeric._fit_ridge(self.x,self.y,self.w,10.);self.pred=f.numeric._predict(self.base,self.q)
    def runfit(self,c=None,q=None,qc=None,base=None):
        q=self.q if q is None else q;fit=self.base if base is None else base
        return f.fit_addon(self.x,self.y,self.w,fit,self.c if c is None else c,q,self.qc if qc is None else qc,f.numeric._predict(fit,q))
    def test_constant_feature_zero_exact_source(self):
        for c in (0.,1.,-.25):
            result,a=self.runfit(np.full(len(self.x),c),qc=np.full(len(self.q),c))
            self.assertTrue(a['zero_addon']);self.assertEqual(result.tobytes(),self.pred.tobytes());self.assertEqual(a['beta'],0.)
    def test_nonzero_projection_orthogonal(self):
        result,a=self.runfit();self.assertFalse(a['zero_addon'])
        self.assertTrue(np.all(np.abs(a['normalized_orthogonality'])<=np.asarray(a['orthogonality_bounds'])))
        self.assertTrue(np.isfinite(result).all());self.assertGreater(a['beta'],0.)
    def test_scalar_formula_and_prediction_reconstruction(self):
        result,a=self.runfit();b=np.column_stack([np.ones(len(self.x)),(self.x-self.base[0])/self.base[1]])
        e=self.c-b.dot(a['gamma']);z=e/a['residual_rms']
        expected=np.sum(self.w*z*(self.y-f.numeric._predict(self.base,self.x)))/(np.sum(self.w*z*z)+10.)
        self.assertAlmostEqual(a['beta'],expected,13)
        qb=np.column_stack([np.ones(len(self.q)),(self.q-self.base[0])/self.base[1]])
        want=self.pred+a['beta']*(self.qc-qb.dot(a['gamma']))/a['residual_rms']
        np.testing.assert_allclose(result,want,atol=1e-13,rtol=0.)
    def test_base_fit_never_mutated(self):
        before=copy.deepcopy(self.base);self.runfit()
        for x,y in zip(before,self.base):np.testing.assert_array_equal(x,y)
    def test_queries_do_not_fit_projection_or_scale(self):
        _,a=self.runfit();q=self.q.copy();q[:]=1e6
        _,b=self.runfit(q=q,qc=np.full(6,-1.))
        self.assertEqual(a,b)
    def test_query_prefix_exact(self):
        full,a=self.runfit()
        for n in range(7):
            result,b=self.runfit(q=self.q[:n],qc=self.qc[:n]);self.assertEqual(a,b)
            self.assertEqual(result.tobytes(),full[:n].tobytes())
    def test_rank_deficient_base_still_projects(self):
        self.x[:,1]=self.x[:,0];self.x[:,4]=0.;base=f.numeric._fit_ridge(self.x,self.y,self.w,10.)
        _,a=self.runfit(base=base);self.assertLess(a['lstsq_rank'],11);self.assertFalse(a['zero_addon'])
    def test_tiny_feature_is_exact_zero(self):
        result,a=self.runfit(c=self.c*1e-14,qc=self.qc*1e-14)
        self.assertTrue(a['zero_addon']);self.assertEqual(result.tobytes(),self.pred.tobytes())
    def test_large_finite_raw_feature_is_not_a_bounded_context(self):
        result,a=self.runfit(c=self.c*100.,qc=self.qc*100.)
        self.assertTrue(np.isfinite(result).all());self.assertFalse(a['zero_addon'])
    def test_feature_in_original_span_zero_exact(self):
        feature=4.+2.*self.x[:,3]-.7*self.x[:,9]
        query=4.+2.*self.q[:,3]-.7*self.q[:,9]
        result,a=self.runfit(c=feature,qc=query)
        self.assertTrue(a['zero_addon']);self.assertEqual(result.tobytes(),self.pred.tobytes())
    def test_bad_feature_and_bad_weights_fail(self):
        for value in (np.nan,np.inf,-np.inf):
            c=self.c.copy();c[0]=value
            with self.assertRaises((ValueError,FloatingPointError)):self.runfit(c=c)
        self.w[0]=0.
        with self.assertRaises(ValueError):self.runfit()
    def test_orthogonality_guard_fails_bad_projection(self):
        with patch.object(f.np.linalg,'lstsq',return_value=(np.zeros(11),np.array([]),11,np.ones(11))):
            with self.assertRaises(FloatingPointError):self.runfit()

class AnnualTests(unittest.TestCase):
    def setUp(self):self.args=fixture()
    def runfit(self):return f.annual_residual_forecasts(**self.args)
    def test_exact_original_base_parity(self):
        got=self.runfit();args={k:v for k,v in self.args.items()if k not in ('train_features','event_features')}
        args['train_good']=args.pop('common_good');expected=f.original.annual_forecasts(**args)
        self.assertEqual(got['base_raw'].tobytes(),expected['log_excess'].tobytes());np.testing.assert_array_equal(got['base_available'],expected['available'])
    def test_prior_cutoff_and_exclusive_lower_bound(self):
        got=self.runfit()
        for row in got['audit']['annual']:
            ids=row['training_ids'];self.assertTrue(np.all(self.args['dates'][ids]>np.datetime64(row['lookback_exclusive_start'])))
            self.assertTrue(np.all(self.args['ends'][ids]<=np.datetime64(row['cutoff'])))
    def test_one_missing_training_feature_whole_year_fallback(self):
        baseline=self.runfit();idx=baseline['audit']['annual'][0]['training_ids'][0]
        self.args['train_features']['response_available'][idx]=False
        result=self.runfit();self.assertFalse(result['available'][:3].any())
        for name in f.NAMES:self.assertEqual(result['raw'][name][:3].tobytes(),result['base_raw'][:3].tobytes())
        self.assertEqual(result['audit']['annual'][0]['training_ids'],baseline['audit']['annual'][0]['training_ids'])
    def test_missing_event_feature_preserves_other_predictions(self):
        baseline=self.runfit();self.args['event_features']['response_available'][0]=False
        got=self.runfit();self.assertFalse(got['available'][0]);self.assertTrue(got['available'][1:].all())
        for name in f.NAMES:
            self.assertEqual(got['raw'][name][0],got['base_raw'][0]);np.testing.assert_array_equal(got['raw'][name][1:],baseline['raw'][name][1:])
    def test_constant_both_features_collapse(self):
        for features in (self.args['train_features'],self.args['event_features']):
            for key in f.FEATURE_KEYS:features[key][:]=.3
        got=self.runfit();self.assertTrue(got['available'].all())
        for name in f.NAMES:self.assertEqual(got['raw'][name].tobytes(),got['base_raw'].tobytes())
        self.assertEqual(got['audit']['counts']['scalar_ridge_fits'],0)
    def test_future_poison_does_not_affect_first_year(self):
        baseline=self.runfit();self.args['event_dates']=self.args['event_dates'][:3];self.args['event_X']=self.args['event_X'][:3];self.args['event_good']=self.args['event_good'][:3]
        self.args['event_features']={k:v[:3]for k,v in self.args['event_features'].items()}
        future=self.args['ends']>np.datetime64('2015-12-31')
        for key in f.FEATURE_KEYS:self.args['train_features'][key][future]=np.inf
        got=self.runfit()
        for name in f.NAMES:self.assertEqual(got['raw'][name].tobytes(),baseline['raw'][name][:3].tobytes())
    def test_all_four_labels_and_singletons_preserved(self):
        self.args['targets']['loss_utility'][45:47]=np.nan
        got=self.runfit();self.assertFalse(any(i in got['audit']['annual'][0]['training_ids']for i in (45,46,47)))
    def test_cold_start_no_fit(self):
        self.args['common_good'][:]=False;got=self.runfit();self.assertFalse(got['base_available'].any())
        self.assertEqual(got['audit']['counts']['base_ridge_fits'],0)
    def test_inputs_unchanged(self):
        before=copy.deepcopy(self.args);self.runfit()
        for k,v in before.items():
            if isinstance(v,dict):
                for q in v:np.testing.assert_array_equal(v[q],self.args[k][q])
            else:np.testing.assert_array_equal(v,self.args[k])

if __name__=='__main__':unittest.main()
