"""Synthetic257 tests only. No market/cache reads or registered phases."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_correlation_distance257_kernel as k

FIXTURE_PATH=k.ROOT/'work/test_monthly_local_distance256_kernel.py'
FIXTURE_SHA='f4b9feef9c998ea22e28cbc684fc78ba9e3ed48e4473306ce07f7cccb826f621'
if hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()!=FIXTURE_SHA:raise ValueError('Frozen256 synthetic fixture changed')
spec=importlib.util.spec_from_file_location('correlation257_frozen_fixture',FIXTURE_PATH)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
fixture,physical,refresh_original=frozen.fixture,frozen.physical,frozen.refresh_original

def fit(args,n=None,**kw):return k.fit_models(args[1],args[2],args[3],n,**kw)
def decide(args,bundle,n=None,**kw):return k.decisions(args[0],args[3],args[2]['monthly'],bundle,n,**kw)
def exact(a,b):np.testing.assert_array_equal(a,b);assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def geometry_fixture():
    rng=np.random.default_rng(257);X=rng.uniform(-.45,.45,(40,10));w=np.linspace(.1,2.,40)
    own=X[:24].copy();q=rng.uniform(-.45,.45,10)
    dates=pd.date_range('2018-01-31',periods=24,freq='ME').values.astype('datetime64[D]')
    return X,w,dates,np.datetime64('2020-12-31'),own,q


def orthogonal(scales=None):
    scales=np.ones(10)*.2 if scales is None else np.asarray(scales)
    return np.r_[np.diag(scales),-np.diag(scales)],np.ones(20)


class GeometryTests(unittest.TestCase):
    def test_canonical_full_moment_formula(self):
        X,w,*_=geometry_fixture();m=k.correlation_metric(X,w)
        self.assertEqual(tuple(m),k.MOMENT_KEYS)
        mean=np.average(X,axis=0,weights=w);c=X-mean;sigma=np.sqrt(np.average(c*c,axis=0,weights=w));s=np.where(sigma>1e-12,sigma,1.)
        C=((c*w[:,None]).T@c)/w.sum();R=np.eye(10)
        for j in range(10):
            for h in range(j+1,10):R[j,h]=R[h,j]=C[j,h]/(s[j]*s[h])
        A=(R+np.eye(10))/2;L=np.linalg.cholesky(A);B=np.linalg.solve(L,np.diag(1/s));T=float(np.sum(B*B))
        for key,value in [('source_mean',mean),('raw_scale',sigma),('source_scale',s),('covariance',C),('correlation',R),('shrinkage',A),('cholesky',L),('transform',B),('metric',B.T@B/T)]:np.testing.assert_array_equal(m[key],value)
        self.assertEqual(m['trace'],T);self.assertFalse(m['identity_correlation'])
    def test_identity_equal_scales_reuses_exact_original(self):
        X,w=orthogonal();m=k.correlation_metric(X,w);_,_,d,c,own,q=geometry_fixture()
        out,g=k.correlation_weights(d,c,own,q,m);self.assertTrue(g['identity_correlation'])
        exact(out,k.model.local_weights(d,c,own,q));self.assertIsNotNone(g['diagonal_geometry'])
    def test_identity_unequal_scales_reuses_frozen256(self):
        X,w=orthogonal(np.linspace(.05,.4,10));m=k.correlation_metric(X,w);_,_,d,c,own,q=geometry_fixture()
        out,g=k.correlation_weights(d,c,own,q,m);ref,a=k.previous.diagonal_weights(d,c,own,q,np.array(m['source_scale']))
        exact(out,ref);self.assertEqual(g['diagonal_geometry'],a);self.assertTrue(g['identity_correlation'])
    def test_equal_scales_does_not_ignore_correlation(self):
        X=np.tile(np.linspace(-.4,.4,40)[:,None],(1,10));m=k.correlation_metric(X,np.ones(40));_,_,d,c,own,q=geometry_fixture()
        self.assertTrue(np.all(np.array(m['source_scale'])==m['source_scale'][0]));self.assertFalse(m['identity_correlation'])
        out,g=k.correlation_weights(d,c,own,q,m);self.assertIsNone(g['diagonal_geometry']);self.assertNotEqual(out.tobytes(),k.model.local_weights(d,c,own,q).tobytes())
    def test_original_floor_coordinates_retained_with_identity(self):
        X,w,*_=geometry_fixture();X[:,0]=.125;X[:,1]=np.linspace(-1e-13,1e-13,len(X));m=k.correlation_metric(X,w)
        self.assertTrue(m['floored'][0] and m['floored'][1]);self.assertEqual(m['source_scale'][:2],[1.,1.])
        R=np.array(m['correlation']);np.testing.assert_array_equal(R[:2,:2],np.eye(2));np.testing.assert_array_equal(R[:2,2:],0.);self.assertEqual(R.shape,(10,10))
    def test_all_constant_identity_metric(self):
        X=np.zeros((24,10));m=k.correlation_metric(X,np.ones(24));self.assertTrue(m['identity_correlation']);self.assertEqual(m['floored'],[True]*10)
        _,_,d,c,own,q=geometry_fixture();out,_=k.correlation_weights(d,c,own,q,m);exact(out,k.model.local_weights(d,c,own,q))
    def test_perfect_positive_and_negative_duplicates_spd(self):
        X,w,*_=geometry_fixture();X[:,1]=X[:,0];X[:,2]=-X[:,0];m=k.correlation_metric(X,w)
        eig=np.linalg.eigvalsh(m['shrinkage']);self.assertGreater(eig.min(),.49);self.assertLess(eig.max(),5.51)
        np.testing.assert_allclose(np.array(m['cholesky'])@np.array(m['cholesky']).T,m['shrinkage'],atol=1e-12,rtol=1e-12)
    def test_exact_formula_mass_and_positive_support(self):
        X,w,d,c,own,q=geometry_fixture();m=k.correlation_metric(X,w);out,g=k.correlation_weights(d,c,own,q,m)
        B=np.array(m['transform']);dist=np.sum(((own-q)@B.T)**2,axis=1)/m['trace'];ages=c.astype('datetime64[M]').astype(int)-d.astype('datetime64[M]').astype(int)
        u=np.exp2(-ages/12.)*np.exp(-dist/(2*.5**2));old=k.model.local_weights(d,c,own,q);expected=u*(old.sum()/u.sum())
        exact(out,expected);np.testing.assert_array_equal(g['distance'],dist);self.assertTrue((out>0).all());self.assertLessEqual(abs(out.sum()-old.sum()),k.MASS_RTOL*old.sum());self.assertEqual(tuple(g),k.GEOMETRY_KEYS)
    def test_zero_displacement_exact_weights(self):
        X,w,d,c,own,q=geometry_fixture();m=k.correlation_metric(X,w);own[:]=q
        out,g=k.correlation_weights(d,c,own,q,m);exact(out,k.model.local_weights(d,c,own,q));self.assertEqual(g['distance'],[0.]*24)
    def test_tiny_nonzero_correlation_is_not_identity(self):
        X,w=orthogonal();X[:,1]+=X[:,0]*1e-10;m=k.correlation_metric(X,w)
        self.assertFalse(m['identity_correlation']);self.assertNotEqual(m['correlation'][0][1],0.)
    def test_extreme_legal_nonfloor_scales_finite(self):
        X,w,*_=geometry_fixture();X[:,0]*=1e-11;X[:,1]=0.;m=k.correlation_metric(X,w)
        self.assertFalse(m['floored'][0]);self.assertTrue(m['floored'][1]);self.assertTrue(np.isfinite(m['metric']).all())
        _,_,d,c,own,q=geometry_fixture();out,_=k.correlation_weights(d,c,own,q,m);self.assertTrue(np.isfinite(out).all() and (out>0).all())
    def test_scale_parity_invalid_input_and_cholesky_failure(self):
        X,w,*_=geometry_fixture()
        with self.assertRaises(AssertionError):k.correlation_metric(X,w,source_scale=np.ones(10))
        for kind in ('nan','negative','zero','shape'):
            Y=X.copy();v=w.copy()
            if kind=='nan':Y[0,0]=np.nan
            elif kind=='negative':v[0]=-1.
            elif kind=='zero':v[:]=0.
            else:Y=Y[:,:9]
            with self.subTest(kind=kind),self.assertRaises(ValueError):k.correlation_metric(Y,v)
        with patch.object(k.np.linalg,'cholesky',side_effect=np.linalg.LinAlgError('blocked')):
            with self.assertRaises(np.linalg.LinAlgError):k.correlation_metric(X,w)
    def test_matmul_nonfinite_outputs_are_not_suppressed(self):
        class Product:
            def __init__(self,value):self.value=value
            def __matmul__(self,other):return np.full((2,2),self.value)
        for bad in (np.nan,np.inf,-np.inf):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'Nonfinite matrix product'):k._matmul(Product(bad),np.eye(2))
    def test_matmul_only_status_isolated_no_error_or_output_repair(self):
        class Spurious:
            def __matmul__(self,other):
                np.divide(np.float64(1.),np.float64(0.))
                return np.array([[1.,2.],[3.,4.]])
        with np.errstate(all='raise'):exact(k._matmul(Spurious(),np.eye(2)),np.array([[1.,2.],[3.,4.]]))
        class Failed:
            def __matmul__(self,other):raise np.linalg.LinAlgError('real failure')
        with self.assertRaises(np.linalg.LinAlgError):k._matmul(Failed(),np.eye(2))

    def test_geometry_does_not_mutate_inputs(self):
        X,w,d,c,own,q=geometry_fixture();before=[v.copy() for v in (X,w,d,own,q)];m=k.correlation_metric(X,w);saved=deepcopy(m);k.correlation_weights(d,c,own,q,m)
        for a,b in zip((X,w,d,own,q),before):exact(a,b)
        self.assertEqual(m,saved)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.old=k.previous.fit_models(cls.args[1],cls.args[2],cls.args[3])
    def test_control_and_diagonal_full_exact256(self):
        for current,old in [('control','control'),('diagonal','candidate')]:
            self.assertEqual(self.bundle[current]['audit'],self.old[old]['audit'])
            for key in k.RAW_KEYS:exact(self.bundle[current]['raw'][key],self.old[old]['raw'][key])
            for key in ('available','local_available'):exact(self.bundle[current][key],self.old[old][key])
    def test_complete_schemas_counts_and_no_pooled_candidate_solves(self):
        b=self.bundle;c=b['audit']['counts'];self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(c),k.MODEL_COUNT_FIELDS)
        self.assertEqual(c['correlation_matrix_builds'],2);self.assertEqual(c['control_target_fits'],8);self.assertEqual(c['control_log_refits'],2)
        for role in ('diag','candidate'):self.assertEqual(c[role+'_pooled_fits'],0);self.assertEqual(c[role+'_log_refits'],0)
        self.assertEqual(c['actual_regression_solves'],sum(c[x] for x in ('control_target_fits','control_log_refits','control_local_solves','diag_local_solves','candidate_local_solves')))
        for row in b['candidate']['audit']['annual']:self.assertEqual(tuple(row),k.CANDIDATE_ANNUAL_KEYS);self.assertEqual(tuple(row['moments']),k.MOMENT_KEYS)
        for row in b['candidate']['audit']['local']:self.assertEqual(tuple(row),k.CANDIDATE_LOCAL_KEYS)
        json.dumps(b['audit'],allow_nan=False);json.dumps(b['candidate']['audit'],allow_nan=False)
    def test_pooled_constants_labels_support_exact(self):
        b=self.bundle
        for role in ('diagonal','candidate'):
            for key in ('source_log','local_constant'):exact(b[role]['raw'][key],b['control']['raw'][key])
            for key in ('available','local_available'):exact(b[role][key],b['control'][key])
            for row,old in zip(b[role]['audit']['local'],b['control']['audit']['local']):self.assertEqual(row['constant_correction'],old['constant_correction'])
        self.assertEqual(tuple(b['label_preservation']),k.LABEL_KEYS);exact(b['label_values'],self.old['label_values'])
        self.assertEqual(b['label_preservation']['target_sha256'],self.old['label_preservation']['target_sha256'])
    def test_control_gate_before_both_candidates(self):
        with patch.object(k.previous,'candidate_forecasts',side_effect=AssertionError('early diag')):
            with self.assertRaisesRegex(ValueError,'model gate failed'):fit(self.args,control_validator=lambda _:dict(passed=False))
    def test_diagnostic_gate_before_moments(self):
        with patch.object(k,'correlation_metric',side_effect=AssertionError('early correlation')):
            with self.assertRaisesRegex(ValueError,'diagonal model gate failed'):fit(self.args,diagnostic_validator=lambda _:dict(passed=False))
    def test_callback_order_and_receipts(self):
        seen=[]
        def control(_):seen.append('control');return dict(passed=True,kind='original')
        def diag(d):self.assertEqual(seen,['control']);seen.append('diagonal');return dict(passed=True,kind='saved256')
        b=fit(self.args,control_validator=control,diagnostic_validator=diag)
        self.assertEqual(seen,['control','diagonal']);self.assertEqual(b['control_gate'],dict(passed=True,kind='original'));self.assertEqual(b['diagnostic_gate'],dict(passed=True,kind='saved256'))
    def test_input_bytes_and_metadata_preserved(self):
        s,t,m,a=self.args;before=k.bridge._snapshot(a);train={x:(v.shape,v.dtype.str,v.tobytes()) for x,v in t.items()};meta=deepcopy(m);fit(self.args)
        self.assertEqual(before,k.bridge._snapshot(a));self.assertEqual(meta,m);self.assertEqual(train,{x:(v.shape,v.dtype.str,v.tobytes()) for x,v in t.items()})
    def test_annual_moments_and_geometry_reconstruction(self):
        args=k.model_inputs(self.args[1],self.args[2],self.args[3]);dates=args[0];X=args[3];Q=args[8]
        for row in self.bundle['candidate']['audit']['annual']:
            ids=row['training_row_ids'];weights=k.model.numeric.month_balanced_weights(dates[ids],np.datetime64(row['cutoff']),12.)
            self.assertEqual(row['moments'],k.correlation_metric(X[ids],weights,source_scale=row['source_scale']))
            for local in [r for r in self.bundle['candidate']['audit']['local'] if r['year']==row['year'] and r['geometry'] is not None]:
                oi=local['own_row_ids'];_,expected=k.correlation_weights(dates[oi],row['cutoff'],X[oi],Q[local['event_row']],row['moments']);self.assertEqual(local['geometry'],expected)
    def test_all_model_logical_and_physical_prefixes(self):
        for n in range(5):
            a=fit(self.args,n);b=fit(physical(self.args,n))
            for role in ('control','diagonal','candidate'):
                self.assertEqual(a[role]['audit'],b[role]['audit'])
                for key in k.RAW_KEYS:exact(a[role]['raw'][key],b[role]['raw'][key]);exact(a[role]['raw'][key],self.bundle[role]['raw'][key][:6*n])
    def test_future_object_poison_before_cast(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);future=args[1]['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+x for x in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][future]='never cast'
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]='never cast'
        b=fit(args,2)
        for role in ('control','diagonal','candidate'):
            for key in k.RAW_KEYS:exact(b[role]['raw'][key],self.bundle[role]['raw'][key][:12])
    def test_n0_no_numeric_moment_or_fit(self):
        args=(self.args[0],)+deepcopy(self.args[1:])
        for key in ('train_plain','event_plain')+tuple('target_'+x for x in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][:]='never cast'
        with patch.object(k,'correlation_metric',side_effect=AssertionError('no moments')):b=fit(args,0)
        self.assertEqual(b['audit']['counts']['actual_regression_solves'],0);self.assertEqual(b['audit']['counts']['correlation_matrix_builds'],0)
    def test_no_own_history_original_fallback(self):
        args=(self.args[0],)+deepcopy(self.args[1:])
        for names in args[2]['monthly']['symbols']:names[-1]='NO_HISTORY'
        refresh_original(args);b=fit(args)
        for j in range(5,24,6):self.assertFalse(b['candidate']['local_available'][j]);self.assertEqual(b['candidate']['raw']['local_linear'][j],b['control']['raw']['source_log'][j])
    def test_saved_control_corruption_stops_before_covariance(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);args[3]['raw_local_linear'][1,0]+=.01
        with patch.object(k,'correlation_metric',side_effect=AssertionError('too early')):
            with self.assertRaises(AssertionError):fit(args)


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.out=decide(cls.args,cls.bundle)
    def test_original_and_diag_full_exact_frozen256(self):
        s,t,m,a=self.args;b=k.previous.fit_models(t,m,a);ref=k.previous.decisions(s,a,m['monthly'],b)
        for arm in k.ARMS[:2]:
            self.assertEqual(self.out[0][arm],ref[0][arm])
            for i in (1,2):exact(self.out[i][arm],ref[i][arm])
    def test_weak_formula_full_U_and_no_own_filter(self):
        for arm in k.ARMS[1:]:
            r,d,au=k.rank_and_delta(self.args[3],self.args[2]['monthly'],self.bundle,arm=arm);exact(r['new_pool'],r['old_pool']);self.assertEqual(au['ranking_cells'],24)
            for i,ids in enumerate(au['active_indices']):np.testing.assert_array_equal(d[i,ids],.0125*(r['new_local'][i,ids]-r['old_local'][i,ids]))
        b=deepcopy(self.bundle);b['candidate']['local_available'][:]=False;self.assertEqual(k.rank_and_delta(self.args[3],self.args[2]['monthly'],b)[2]['ranking_cells'],24)
    def test_three_episodes_and_both_gates_order(self):
        seen=[];study=self.args[0]
        with patch.object(study.ep,'select_boundary',wraps=study.ep.select_boundary) as ep:
            decide(self.args,self.bundle,incumbent_validator=lambda *x:(seen.append(ep.call_count) or dict(passed=True)),diagnostic_validator=lambda *x:(seen.append(ep.call_count) or dict(passed=True)))
        self.assertEqual(seen,[1,2]);self.assertEqual(ep.call_count,3)
    def test_inc_gate_prevents_all_new_ranks(self):
        with patch.object(k,'rank_and_delta',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'incumbent gate failed'):decide(self.args,self.bundle,incumbent_validator=lambda *x:dict(passed=False))
    def test_diag_gate_prevents_corr_ranking_and_episode(self):
        study=self.args[0]
        with patch.object(k,'rank_and_delta',wraps=k.rank_and_delta) as ranks,patch.object(study.ep,'select_boundary',wraps=study.ep.select_boundary) as ep:
            with self.assertRaisesRegex(ValueError,'diagonal gate failed'):decide(self.args,self.bundle,diagnostic_validator=lambda *x:dict(passed=False))
        self.assertEqual(ranks.call_count,1);self.assertEqual(ep.call_count,2)
    def test_expected_endpoints_mismatch(self):
        for name in ('expected_incumbent_orders','expected_diagnostic_orders'):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Frozen'):decide(self.args,self.bundle,**{name:[]})
    def test_partial_nonweak_exact_and_zero_delta(self):
        for arm in k.ARMS[1:]:
            for i in (0,2):
                self.assertEqual(self.out[0][arm][i],self.out[0][k.ARMS[0]][i])
                for key in (1,2):exact(self.out[key][arm][i],self.out[key][k.ARMS[0]][i])
        b=deepcopy(self.bundle);b['candidate']=deepcopy(b['control']);r,d,a=k.rank_and_delta(self.args[3],self.args[2]['monthly'],b);exact(d,np.zeros_like(d))
    def test_audit_schemas_and_no_fits(self):
        au=self.out[3];self.assertEqual(tuple(au['counts']),k.COUNT_FIELDS);self.assertTrue(all(tuple(row)==k.EVENT_KEYS for row in au['events']));self.assertEqual(set(au['rank_delta']),set(k.ARMS[1:]));json.dumps(au,allow_nan=False)
        for key in ('new_model_fits','predictive_model_fits'):self.assertEqual(au['counts'][key],0)
    def test_all_cached_physical_and_fullbundlephysical_prefixes(self):
        for n in range(5):
            small=physical(self.args,n);outputs=(decide(self.args,self.bundle,n),decide(small,fit(small)),decide(small,self.bundle))
            self.assertEqual(outputs[0][3],outputs[1][3]);self.assertEqual(outputs[0][3],outputs[2][3])
            for out in outputs:
                for arm in k.ARMS:
                    self.assertEqual(out[0][arm],self.out[0][arm][:n])
                    for i in (1,2):exact(out[i][arm],self.out[i][arm][:n])
    def test_future_forecast_values_not_consumed(self):
        for n in (0,2):
            b=deepcopy(self.bundle)
            for role in ('control','diagonal','candidate'):
                for key in k.RAW_KEYS:b[role]['raw'][key][n*6:]=np.inf
                b[role]['available'][n*6:]=False;b[role]['local_available'][n*6:]=True
            out=decide(physical(self.args,n),b)
            for arm in k.ARMS:
                self.assertEqual(out[0][arm],self.out[0][arm][:n])
                for i in (1,2):exact(out[i][arm],self.out[i][arm][:n])
    def test_no_model_or_moment_rebuild_in_decisions(self):
        with patch.object(k,'correlation_metric',side_effect=AssertionError('no moments')),patch.object(k,'candidate_forecasts',side_effect=AssertionError('no models')):decide(self.args,self.bundle)
    def test_consumed_invalid_forecast_hard_failure(self):
        b=deepcopy(self.bundle);b['candidate']['raw']['local_linear'][6]=np.inf
        with self.assertRaisesRegex(ValueError,'Raw support'):decide(self.args,b)


if __name__=='__main__':unittest.main(verbosity=2)
