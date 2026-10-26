"""Synthetic caller-owned cache/model tests; no market or saved NPZ access."""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_local_distance256_kernel as k


def fixture():
    spec=importlib.util.spec_from_file_location('distance256_synthetic_study',k.ROOT/'outputs/monthly_local_linear70/study.py')
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    names=list('ABCDEF');w=len(names)
    days=pd.date_range('2010-12-31','2017-12-31',freq='ME')
    dates=np.repeat(days.values.astype('datetime64[D]'),w);ends=np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'),w)
    symbols=np.tile(names,len(days));rng=np.random.default_rng(255)
    X=rng.uniform(-.48,.48,(len(dates),10))
    raw=.13*X[:,0]-.09*X[:,1]+.06*X[:,3]**2+np.tile([.01,.03,-.02,-.01,-.04,.15],len(days))
    y=(raw.reshape(-1,w)-raw.reshape(-1,w).mean(axis=1,keepdims=True)).reshape(-1)
    good=np.ones(len(dates),bool);good[7]=False
    targets={name:(y*(i+1)).copy() for i,name in enumerate(k.TARGETS)}
    for v in targets.values():v[~good]=np.nan
    q=['2016-08-17','2016-08-31','2017-01-31','2017-02-28'];qe=['2016-08-31','2016-09-30','2017-02-28','2017-03-31'];n=len(q)
    m=dict(dates=q,ends=qe,symbols=[names.copy() for _ in q],etfs=['F'])
    tm=dict(dates=days.strftime('%Y-%m-%d').tolist(),ends=(days+pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d').tolist(),symbols=[names.copy() for _ in days])
    eventX=rng.uniform(-.48,.48,(n*w,10));event_dates=np.repeat(np.asarray(q,dtype='datetime64[D]'),w)
    train=dict(dates=dates,ends=ends,train_plain=X,event_plain=eventX,event_dates=event_dates,common_good=good,offsets=np.arange(len(days)+1)*w,**{'target_'+name:value for name,value in targets.items()})
    shape=(n,w);z=np.zeros(shape);b=np.ones(shape,bool);calm=np.array([False,False,True,False]);weak=np.array([True,True,False,True]);cw=np.where(calm,.5,.52)[:,None]
    core=cw*np.tile(np.linspace(-.2,.2,w),(n,1))
    a=dict(baseline=core+.07,core=core,core_weight=cw,eligible=b.copy(),effects=np.tile([0.,0.,0.,.5,-.5,0.],(n,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.31,.305,.30,.295,.29,.285],(n,1)),old_ml4=z.copy(),extension_available=b.copy())
    original=k.model.annual_forecasts(dates,ends,symbols,X,targets,good,event_dates,np.tile(names,n),eventX,np.ones(n*w,bool))
    a['forecast_available']=original['available'].reshape(shape).copy();a['model_available']=a['forecast_available'].copy();a['local_available']=original['local_available'].reshape(shape).copy()
    for key in k.RAW_KEYS:a['raw_'+key]=original['raw'][key].reshape(shape).copy()
    a['raw_log_excess']=a['raw_source_log'].copy();use=a['forecast_available']&a['eligible']
    a['rank_log_excess']=k._rank(a['raw_source_log'],use,z)
    ranks,ready=s.ranked_forecasts({key:a['raw_'+key] for key in ('local_constant','local_linear')},a['model_available'],a,n)
    for key,v in ranks.items():a['rank_'+key]=v
    a['paired_month_ready']=ready
    return s,train,dict(monthly=m,training=tm),a


def fit(args,n=None,**kw):return k.fit_models(args[1],args[2],args[3],n,**kw)

def decide(args,bundle,n=None,**kw):return k.decisions(args[0],args[3],args[2]['monthly'],bundle,n,**kw)


def physical(args,n):
    s,t,meta,a=args;out=deepcopy(meta)
    for key in ('dates','ends','symbols'):out['monthly'][key]=out['monthly'][key][:n]
    small={key:value[:n].copy() for key,value in a.items()}
    train=deepcopy(t);train['event_dates']=train['event_dates'][:n*6];train['event_plain']=train['event_plain'][:n*6]
    # Physical source rows are a complete prefix of exact dated monthly cohorts.
    cutoff=np.datetime64('%04d-12-31'%(int(out['monthly']['dates'][-1][:4])-1)) if n else None
    keep=t['ends']<=cutoff if n else np.zeros(len(t['ends']),bool);months=int(keep.sum())//6
    for key in ('dates','ends','symbols'):out['training'][key]=out['training'][key][:months]
    for key in ('dates','ends','train_plain','common_good')+tuple('target_'+x for x in k.TARGETS):train[key]=train[key][keep]
    train['offsets']=np.arange(months+1)*6
    return s,train,out,small


def refresh_original(args):
    study,train,meta,a=args;n,w=a['baseline'].shape
    out=k.model.annual_forecasts(*k.model_inputs(train,meta,a))
    a['forecast_available']=out['available'].reshape(n,w).copy();a['model_available']=a['forecast_available'].copy();a['local_available']=out['local_available'].reshape(n,w).copy()
    for key in k.RAW_KEYS:a['raw_'+key]=out['raw'][key].reshape(n,w).copy()
    a['raw_log_excess']=a['raw_source_log'].copy();a['rank_log_excess']=k._rank(a['raw_source_log'],a['forecast_available']&a['eligible'],np.zeros((n,w)))
    ranks,ready=study.ranked_forecasts({key:a['raw_'+key] for key in ('local_constant','local_linear')},a['model_available'],a,n)
    for key,v in ranks.items():a['rank_'+key]=v
    a['paired_month_ready']=ready
    return args


class GeometryTests(unittest.TestCase):
    def case(self):
        rng=np.random.default_rng(256);X=rng.uniform(-.5,.5,(24,10));q=rng.uniform(-.5,.5,10)
        dates=pd.date_range('2018-01-31',periods=24,freq='ME').values.astype('datetime64[D]')
        return dates,np.datetime64('2020-12-31'),X,q,np.linspace(.1,.5,10)
    def test_equal_scale_exact_original_branch(self):
        d,c,X,q,s=self.case();s[:]=.25;w,a=k.diagonal_weights(d,c,X,q,s)
        self.assertTrue(a['equal_scales']);self.assertEqual(tuple(a),k.GEOMETRY_KEYS);self.assertEqual(w.tobytes(),k.model.local_weights(d,c,X,q).tobytes());self.assertEqual(a['mass_multiplier'],1.);self.assertEqual(a['distance'],a['old_distance']);self.assertEqual(a['unscaled_weights'],a['original_weights'])
    def test_near_equal_not_equal_branch(self):
        d,c,X,q,s=self.case();s[:]=.25;s[-1]=np.nextafter(.25,np.inf);w,a=k.diagonal_weights(d,c,X,q,s);self.assertFalse(a['equal_scales']);inv=1./(s*s);np.testing.assert_array_equal(a['coefficients'],inv/inv.sum())
    def test_exact_as_written_formula_mass_and_no_repair(self):
        d,c,X,q,s=self.case();w,a=k.diagonal_weights(d,c,X,q,s);inv=1./(s*s);alpha=inv/inv.sum();dist=np.sum((X-q)**2*alpha,axis=1);age=c.astype('datetime64[M]').astype(int)-d.astype('datetime64[M]').astype(int)
        u=np.exp2(-age/12.)*np.exp(-dist/(2*.5**2));orig=k.model.local_weights(d,c,X,q);expected=u*(orig.sum()/u.sum())
        self.assertEqual(w.tobytes(),expected.tobytes());np.testing.assert_array_equal(a['coefficients'],alpha);np.testing.assert_array_equal(a['distance'],dist);self.assertLessEqual(abs(w.sum()-orig.sum()),k.MASS_RTOL*orig.sum());self.assertTrue((w>0).all())
    def test_zero_distance_exact_original(self):
        d,c,X,q,s=self.case();X[:]=q;w,a=k.diagonal_weights(d,c,X,q,s);self.assertEqual(w.tobytes(),k.model.local_weights(d,c,X,q).tobytes());self.assertEqual(a['distance'],[0.]*24)
    def test_extreme_original_scales_finite_support(self):
        d,c,X,q,s=self.case();s[0]=np.nextafter(1e-12,np.inf);s[-1]=1.;w,a=k.diagonal_weights(d,c,X,q,s);self.assertTrue(np.isfinite(w).all() and (w>0).all());self.assertTrue(np.isfinite(a['coefficients']).all() and (np.asarray(a['coefficients'])>0).all())
    def test_invalid_scale_and_nonfinite_inputs_abort(self):
        for bad in (0.,-1.,np.inf,np.nan):
            d,c,X,q,s=self.case();s[0]=bad
            with self.assertRaises(ValueError):k.diagonal_weights(d,c,X,q,s)
        d,c,X,q,s=self.case();X[0,0]=np.nan
        with self.assertRaises(ValueError):k.diagonal_weights(d,c,X,q,s)
    def test_arithmetic_overflow_is_not_fallback(self):
        d,c,X,q,s=self.case();s[0]=1e-300
        with self.assertRaises((FloatingPointError,ValueError)):k.diagonal_weights(d,c,X,q,s)
    def test_constant_design_intercept_and_zero_residual(self):
        d,c,X,q,s=self.case();X[:]=q;w,_=k.diagonal_weights(d,c,X,q,s);r=np.linspace(-.2,.2,len(X))
        linear,a=k._linear_only(X,q,s,r,w);expected=np.einsum('n,n->',w,r,optimize=False)/(w.sum()+12.)
        self.assertEqual(linear,k.model.local_corrections(X,q,s,r,w)[1]);self.assertLessEqual(abs(linear-expected),16*np.finfo(float).eps);self.assertEqual(a['status'],'fit');linear,a=k._linear_only(X,q,s,np.zeros(len(X)),w);self.assertEqual(linear,0.);self.assertEqual(a['status'],'zero_residual')
    def test_unequal_scales_reorder_analogues(self):
        d=np.array(['2020-01-31']*2);X=np.zeros((2,10));X[0,0]=.2;X[1,1]=.3;s=np.array([.1]+[.5]*9)
        w,a=k.diagonal_weights(d,'2020-12-31',X,np.zeros(10),s);self.assertLess(a['old_distance'][0],a['old_distance'][1]);self.assertGreater(a['distance'][0],a['distance'][1]);self.assertLess(w[0],w[1])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.full=fit(cls.args)
    def test_control_exact_raw_audit_original_pure_interface(self):
        ref=k.model.annual_forecasts(*k.model_inputs(self.args[1],self.args[2],self.args[3]));self.assertEqual(ref['audit'],self.full['control']['audit']);k.control_parity(self.args[3],self.full['control'])
        for key in k.RAW_KEYS:self.assertEqual(ref['raw'][key].tobytes(),self.full['control']['raw'][key].tobytes())
    def test_schemas_and_actual_fit_counts(self):
        b=self.full;c=b['audit']['counts'];self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(c),k.MODEL_COUNT_FIELDS);self.assertEqual(c['control_target_fits'],8);self.assertEqual(c['control_log_refits'],2);self.assertEqual(c['candidate_pooled_fits'],0);self.assertEqual(c['candidate_log_refits'],0)
        self.assertEqual(c['actual_regression_solves'],c['control_target_fits']+c['control_log_refits']+c['control_local_solves']+c['candidate_local_solves']);self.assertGreater(c['candidate_local_solves'],0)
        self.assertEqual(tuple(b['candidate']['audit']),k.CANDIDATE_AUDIT_KEYS)
        for x in b['candidate']['audit']['annual']:self.assertEqual(tuple(x),k.CANDIDATE_ANNUAL_KEYS)
        for x in b['candidate']['audit']['local']:self.assertEqual(tuple(x),k.CANDIDATE_LOCAL_KEYS)
    def test_original_labels_alltargets_exact_and_invalid_nan_preserved(self):
        args=k.model_inputs(self.args[1],self.args[2],self.args[3]);lp=self.full['label_preservation'];self.assertEqual(tuple(lp),k.LABEL_KEYS);self.assertFalse(lp['recentered']);self.assertTrue(lp['labels_unchanged'])
        self.assertEqual(self.full['label_values'].tobytes(),args[4]['log_excess'].tobytes())
        for key in k.TARGETS:self.assertEqual(lp['target_sha256'][key],k._target_digest(args[4][key]))
    def test_pooled_and_constant_exactcopies_candidate_onlylinear(self):
        for key in ('source_log','local_constant'):self.assertEqual(self.full['candidate']['raw'][key].tobytes(),self.full['control']['raw'][key].tobytes())
        self.assertTrue(np.any(self.full['candidate']['raw']['local_linear']!=self.full['control']['raw']['local_linear']))
        for x,y in zip(self.full['candidate']['audit']['local'],self.full['control']['audit']['local']):self.assertEqual(x['constant_correction'],y['constant_correction'])
        with patch.object(k.model,'_fit_pooled',wraps=k.model._fit_pooled) as pool,patch.object(k.model,'local_corrections',wraps=k.model.local_corrections) as lc:
            b=fit(self.args)
        self.assertEqual(pool.call_count,2);self.assertEqual(lc.call_count,b['audit']['counts']['candidate_kernel_evaluations'])
    def test_control_gate_precedes_candidate_weights(self):
        with patch.object(k,'diagonal_weights',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,control_validator=lambda x:{'passed':False})
    def test_corrupt_original_cache_before_candidate(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);args[3]['raw_local_linear'][1,0]+=.01
        with patch.object(k,'candidate_forecasts',side_effect=AssertionError('too early')):
            with self.assertRaises(AssertionError):fit(args)
    def test_input_immutability(self):
        _,t,m,a=self.args;before=k.bridge._snapshot(a);ts={x:(v.dtype.str,v.shape,v.tobytes()) for x,v in t.items()};ms=deepcopy(m);fit(self.args)
        self.assertEqual(before,k.bridge._snapshot(a));self.assertEqual(ts,{x:(v.dtype.str,v.shape,v.tobytes()) for x,v in t.items()});self.assertEqual(m,ms)
    def test_query_cutoff_future_object_poison_before_cast(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);future=args[1]['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+x for x in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][future]='future must not cast'
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]='future query must not cast'
        b=fit(args,2)
        for key in k.RAW_KEYS:np.testing.assert_array_equal(b['candidate']['raw'][key],self.full['candidate']['raw'][key][:12])
    def test_n0_zero_fits_no_weight_or_numeric_consumption(self):
        args=(self.args[0],)+deepcopy(self.args[1:])
        for key in ('train_plain','event_plain')+tuple('target_'+x for x in k.TARGETS):args[1][key]=args[1][key].astype(object);args[1][key][:]='never cast'
        with patch.object(k,'diagonal_weights',side_effect=AssertionError('no weight')):b=fit(args,0)
        self.assertEqual(b['audit']['counts']['actual_regression_solves'],0);self.assertEqual(len(b['label_values']),0)
    def test_equal_design_scales_reproduce_original_model_exact(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);args[1]['train_plain'][:]=args[1]['train_plain'][:,0,None];args[1]['event_plain'][:]=args[1]['event_plain'][:,0,None];refresh_original(args);b=fit(args)
        self.assertEqual(b['audit']['counts']['candidate_kernel_evaluations'],b['audit']['counts']['candidate_equal_scale_evaluations'])
        for key in k.RAW_KEYS:self.assertEqual(b['candidate']['raw'][key].tobytes(),b['control']['raw'][key].tobytes())
    def test_no_own_history_preserves_pooled_and_support(self):
        args=(self.args[0],)+deepcopy(self.args[1:])
        for names in args[2]['monthly']['symbols']:names[-1]='NO_HISTORY'
        refresh_original(args);b=fit(args)
        for j in range(5,24,6):self.assertFalse(b['candidate']['local_available'][j]);self.assertEqual(b['candidate']['raw']['local_linear'][j],b['control']['raw']['source_log'][j])
        np.testing.assert_array_equal(b['candidate']['available'],b['control']['available'])
    def test_all_logical_and_physical_model_prefixes(self):
        for n in range(5):
            x=fit(self.args,n);y=fit(physical(self.args,n))
            for arm in ('control','candidate'):
                for key in k.RAW_KEYS:np.testing.assert_array_equal(x[arm]['raw'][key],y[arm]['raw'][key]);np.testing.assert_array_equal(x[arm]['raw'][key],self.full[arm]['raw'][key][:n*6])
    def test_geometry_evidence_uses_same_rows_and_original_scales(self):
        args=k.model_inputs(self.args[1],self.args[2],self.args[3]);scales={a['year']:a['source_scale'] for a in self.full['candidate']['audit']['annual']}
        for detail in self.full['candidate']['audit']['local']:
            g=detail['geometry']
            if g is not None:
                ids=detail['own_row_ids'];self.assertEqual(len(ids),detail['own_rows']);self.assertEqual(g['source_scale'],scales[detail['year']]);self.assertEqual(tuple(g),k.GEOMETRY_KEYS)
                w,expected=k.diagonal_weights(args[0][ids],detail['cutoff'],args[3][ids],args[8][detail['event_row']],np.asarray(g['source_scale']));self.assertEqual(expected,g)
    def test_bad_metadata_and_good_target_fail(self):
        args=(self.args[0],)+deepcopy(self.args[1:]);args[1]['offsets'][1]+=1
        with self.assertRaisesRegex(ValueError,'offsets'):fit(args)
        args=(self.args[0],)+deepcopy(self.args[1:]);args[1]['target_log_excess'][0]=np.nan
        with patch.object(k,'_control_forecasts',side_effect=AssertionError('no fit')):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):fit(args)


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.result=decide(cls.args,cls.bundle)
    def test_exact_original_control_scores_orders(self):
        s,t,meta,a=self.args;m=meta['monthly'];c,post=s.choices(s.configs()[4],a,m,4,policy=dict(s.POLICY_FIELDS))
        o,sc,pre,au=self.result;self.assertEqual(o[k.ARMS[0]],s.rp.make_orders(c,post,m));np.testing.assert_array_equal(sc[k.ARMS[0]],post);np.testing.assert_array_equal(pre[k.ARMS[0]],s.modified_pre_score(s.configs()[4],a,4))
    def test_full_weak_replacement_formula(self):
        ranks,d,audit=k.rank_and_delta(self.args[3],self.args[2]['monthly'],self.bundle)
        for i,active in enumerate(audit['active_indices']):
            j=np.asarray(active,int);op,ol,npool,nlocal=(ranks[x][i,j] for x in k.RANK_KEYS)
            expected=.0125*(nlocal-ol);np.testing.assert_array_equal(d[i,j],expected)
        np.testing.assert_array_equal(ranks['new_pool'],ranks['old_pool'])
    def test_nonweak_calm_and_partial_exact(self):
        o,sc,p,a=self.result
        for i in (0,2):
            self.assertEqual(p[k.ARMS[0]][i].tobytes(),p[k.ARMS[1]][i].tobytes());self.assertEqual(sc[k.ARMS[0]][i].tobytes(),sc[k.ARMS[1]][i].tobytes());self.assertEqual(o[k.ARMS[0]][i],o[k.ARMS[1]][i])
    def test_equal_rank_pairs_exactzero(self):
        b=deepcopy(self.bundle);b['candidate']=deepcopy(b['control']);r,d,a=k.rank_and_delta(self.args[3],self.args[2]['monthly'],b);np.testing.assert_array_equal(d,np.zeros_like(d));o,sc,p,a=decide(self.args,b);np.testing.assert_array_equal(sc[k.ARMS[0]],sc[k.ARMS[1]])
    def test_rank_all_original_u_not_by_type(self):
        r,d,au=k.rank_and_delta(self.args[3],self.args[2]['monthly'],self.bundle);raw=self.bundle['candidate']['raw']['source_log'].reshape(4,6)
        for i in range(4):np.testing.assert_array_equal(r['new_pool'][i],pd.Series(raw[i]).rank(pct=True).to_numpy()-.5)
    def test_rank_does_not_require_own_history(self):
        b=deepcopy(self.bundle);b['candidate']['local_available'][:]=False;r,d,au=k.rank_and_delta(self.args[3],self.args[2]['monthly'],b);self.assertEqual(au['ranking_cells'],24)
    def test_two_single_episodes_and_incgate_before_candidate(self):
        s=self.args[0];calls=[]
        with patch.object(s.ep,'select_boundary',wraps=s.ep.select_boundary) as ep:
            decide(self.args,self.bundle,incumbent_validator=lambda *x:(calls.append(ep.call_count) or {'passed':True}))
        self.assertEqual(calls,[1]);self.assertEqual(ep.call_count,2)
    def test_rejected_incgate_precedes_new_ranks(self):
        with patch.object(k,'rank_and_delta',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'gate failed'):decide(self.args,self.bundle,incumbent_validator=lambda *x:{'passed':False})
    def test_frozen_orders_mismatch(self):
        with self.assertRaisesRegex(ValueError,'Frozen incumbent'):decide(self.args,self.bundle,expected_incumbent_orders=[])
    def test_full_audit_schema_no_returns(self):
        au=self.result[3];self.assertEqual(tuple(au['counts']),k.COUNT_FIELDS);self.assertTrue(all(tuple(e)==k.EVENT_KEYS for e in au['events']));json.dumps(au,allow_nan=False);self.assertFalse(au['computes_returns']);self.assertEqual(au['counts']['new_model_fits'],0)
    def test_all_cached_physical_prefixes(self):
        for n in range(5):
            out=decide(self.args,self.bundle,n);small=physical(self.args,n);b=fit(small);other=decide(small,b)
            self.assertEqual(out[3],other[3])
            for arm in k.ARMS:
                self.assertEqual(out[0][arm],self.result[0][arm][:n]);self.assertEqual(out[0][arm],other[0][arm])
                for j in (1,2):np.testing.assert_array_equal(out[j][arm],self.result[j][arm][:n]);np.testing.assert_array_equal(out[j][arm],other[j][arm])
    def test_physical_cache_with_full_bundle_host_seam(self):
        for n in range(5):
            a=physical(self.args,n);out=decide(a,self.bundle)
            for arm in k.ARMS:
                self.assertEqual(out[0][arm],self.result[0][arm][:n])
                for j in (1,2):np.testing.assert_array_equal(out[j][arm],self.result[j][arm][:n])
    def test_future_forecast_values_not_consumed(self):
        for n in (0,2):
            b=deepcopy(self.bundle)
            for name in ('control','candidate'):
                for key in k.RAW_KEYS:b[name]['raw'][key][n*6:]=np.inf
                b[name]['available'][n*6:]=False;b[name]['local_available'][n*6:]=True
            a=physical(self.args,n);out=decide(a,b)
            for arm in k.ARMS:
                self.assertEqual(out[0][arm],self.result[0][arm][:n])
                for j in (1,2):np.testing.assert_array_equal(out[j][arm],self.result[j][arm][:n])
    def test_consumed_forecast_inf_rejected(self):
        b=deepcopy(self.bundle);b['candidate']['raw']['source_log'][6]=np.inf
        with self.assertRaisesRegex(ValueError,'Raw support'):decide(self.args,b)
    def test_decisions_do_not_refit_or_recenter(self):
        with patch.object(k,'candidate_forecasts',side_effect=AssertionError('no model')),patch.object(k,'_control_forecasts',side_effect=AssertionError('no model')):decide(self.args,self.bundle)


if __name__=='__main__':unittest.main(verbosity=2)
