"""Synthetic caller-owned cache/model tests; no market or saved NPZ access."""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_asset_target255_kernel as k


def fixture():
    spec=importlib.util.spec_from_file_location('asset255_synthetic_study',k.ROOT/'outputs/monthly_local_linear70/study.py')
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


class LabelTests(unittest.TestCase):
    def basic(self):
        return np.array(['2020-01-31']*4),np.array(list('ABCD')),np.array([2.,4.,7.,np.nan]),np.array([True,True,True,False]),['C']
    def test_group_means_singleton_no_reclip(self):
        a=self.basic();y,au=k.recenter_labels(*a);np.testing.assert_array_equal(y,[-1.,1.,0.,np.nan]);self.assertEqual(tuple(au['counts']),k.LABEL_COUNT_FIELDS);self.assertEqual(tuple(au['groups'][0]),k.LABEL_GROUP_KEYS);self.assertEqual(au['counts']['singleton_groups'],1);self.assertGreater(a[2][2],1.);self.assertEqual(y[2].hex(),'0x0.0p+0')
    def test_invalid_payload_inf_signedzero_unchanged(self):
        a=list(self.basic());a[2]=np.array([2.,4.,np.inf,-0.]);a[3]=np.array([True,True,False,False]);y,_=k.recenter_labels(*a);self.assertEqual(y[~a[3]].tobytes(),a[2][~a[3]].tobytes())
    def test_original_order_numpy_mean(self):
        a=list(self.basic());a[2]=np.array([.1,.2,.3,4.]);a[3][:]=True;a[4]=[];y,au=k.recenter_labels(*a);np.testing.assert_array_equal(y,a[2]-np.mean(a[2]));self.assertEqual(au['counts']['empty_groups'],1)
    def test_all_ties_valid_zero(self):
        a=list(self.basic());a[2]=np.ones(4);a[3][:]=True;y,_=k.recenter_labels(*a);np.testing.assert_array_equal(y,np.zeros(4))
    def test_empty_n0(self):
        y,au=k.recenter_labels(np.array([],dtype='datetime64[D]'),np.array([],dtype=str),np.array([],float),np.array([],bool),[]);self.assertEqual(len(y),0);self.assertEqual(au['groups'],[])
    def test_nonfinite_valid_hardfail(self):
        for v in (np.inf,np.nan):
            a=list(self.basic());a[2][0]=v
            with self.assertRaises(ValueError):k.recenter_labels(*a)
    def test_boolean_mask_must_be_bool(self):
        a=list(self.basic());a[3]=a[3].astype(int)
        with self.assertRaises(ValueError):k.recenter_labels(*a)
    def test_duplicate_security_month_rejected(self):
        a=list(self.basic());a[1][1]='A'
        with self.assertRaises(ValueError):k.recenter_labels(*a)
    def test_malformed_dates_rejected(self):
        a=list(self.basic());a[0]=['2020-01-31','2020-1-31','2020-01-31','2020-01-31']
        with self.assertRaises(ValueError):k.recenter_labels(*a)
    def test_overflow_not_silent(self):
        a=list(self.basic());a[2][:]=1e308;a[3][:]=True
        with self.assertRaises((ValueError,FloatingPointError)):k.recenter_labels(*a)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.full=fit(cls.args)
    def test_schema_control_exact_and_real_fit_accounting(self):
        b=self.full;self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(b['audit']['counts']),k.MODEL_COUNT_FIELDS)
        k.control_parity(self.args[3],b['control']);c=b['audit']['counts'];self.assertEqual(c['control_target_fits'],8);self.assertEqual(c['control_log_refits'],2)
        self.assertEqual(c['actual_regression_solves'],sum(c[p+'_'+s] for p in ('control','candidate') for s in ('target_fits','log_refits','local_solves')));self.assertGreater(c['candidate_local_solves'],0)
    def test_label_used_by_both_pooled_and_local(self):
        args=list(k.model_inputs(self.args[1],self.args[2],self.args[3]));args[4]=deepcopy(args[4]);args[4]['log_excess']=self.full['label_values'].copy()
        expected=k.model.annual_forecasts(*args)
        for key in k.RAW_KEYS:np.testing.assert_array_equal(expected['raw'][key],self.full['candidate']['raw'][key])
        self.assertTrue(np.any(self.full['candidate']['raw']['source_log']!=self.full['control']['raw']['source_log']))
    def test_control_gate_before_new_labels_or_fit(self):
        calls=[]
        def reject(control):calls.append(control);return {'passed':False}
        with patch.object(k,'recenter_labels',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,control_validator=reject)
        self.assertEqual(len(calls),1)
    def test_corrupt_cache_aborts_before_candidate(self):
        a=(self.args[0],)+deepcopy(self.args[1:]);a[3]['raw_local_linear'][1,0]+=.01
        with patch.object(k,'recenter_labels',side_effect=AssertionError('too early')):
            with self.assertRaises(AssertionError):fit(a)
    def test_input_immutability(self):
        s,t,m,a=self.args;before=k.bridge._snapshot(a);train={x:(v.dtype.str,v.shape,v.tobytes()) for x,v in t.items()};meta=deepcopy(m);fit(self.args)
        self.assertEqual(before,k.bridge._snapshot(a));self.assertEqual(train,{x:(v.dtype.str,v.shape,v.tobytes()) for x,v in t.items()});self.assertEqual(m,meta)
    def test_query_prefix_cutoff_and_future_poison(self):
        a=(self.args[0],)+deepcopy(self.args[1:]);cut=np.datetime64('2015-12-31');future=a[1]['ends']>cut
        for key in ('train_plain',)+tuple('target_'+x for x in k.TARGETS):
            a[1][key]=a[1][key].astype(object);a[1][key][future]='future must not cast'
        b=fit(a,2)
        for key in k.RAW_KEYS:np.testing.assert_array_equal(b['candidate']['raw'][key],self.full['candidate']['raw'][key][:12])
        self.assertTrue((a[1]['ends'][b['training_row_ids']]<=cut).all())
    def test_actual_control_fit_is_original_pure_interface(self):
        inputs=k.model_inputs(self.args[1],self.args[2],self.args[3]);ref=k.model.annual_forecasts(*inputs)
        self.assertEqual(ref['audit'],self.full['control']['audit'])
        for key in k.RAW_KEYS:np.testing.assert_array_equal(ref['raw'][key],self.full['control']['raw'][key])
    def test_n0_zero_fits_and_empty_inputs(self):
        b=fit(self.args,0);self.assertEqual(b['audit']['counts']['actual_regression_solves'],0);self.assertEqual(len(b['label_values']),0)
    def test_identity_offsets_rejected(self):
        a=(self.args[0],)+deepcopy(self.args[1:]);a[1]['offsets'][1]+=1
        with self.assertRaisesRegex(ValueError,'offsets'):fit(a)
    def test_nonfinite_valid_label_rejected_before_fit(self):
        a=(self.args[0],)+deepcopy(self.args[1:]);a[1]['target_log_excess'][0]=np.nan
        with patch.object(k.model,'annual_forecasts',side_effect=AssertionError('no fit')):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):fit(a)
    def test_original_good_not_new_label_mask(self):
        self.assertEqual(self.full['label_audit']['counts']['valid_rows'],int(k.model_inputs(self.args[1],self.args[2],self.args[3])[5].sum()))
        np.testing.assert_array_equal(self.full['candidate']['available'],self.full['control']['available']);np.testing.assert_array_equal(self.full['candidate']['local_available'],self.full['control']['local_available'])
    def test_all_logical_physical_model_prefixes(self):
        for n in range(5):
            x=fit(self.args,n);y=fit(physical(self.args,n))
            for arm in ('control','candidate'):
                for key in k.RAW_KEYS:
                    np.testing.assert_array_equal(x[arm]['raw'][key],y[arm]['raw'][key]);np.testing.assert_array_equal(x[arm]['raw'][key],self.full[arm]['raw'][key][:n*6])


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
            expected=.05*(npool-op)+.0125*((nlocal-npool)-(ol-op));np.testing.assert_array_equal(d[i,j],expected)
        self.assertGreater(np.count_nonzero(d),0)
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
        with patch.object(k,'recenter_labels',side_effect=AssertionError('no labels')),patch.object(k.model,'annual_forecasts',side_effect=AssertionError('no model')):decide(self.args,self.bundle)


if __name__=='__main__':unittest.main(verbosity=2)
