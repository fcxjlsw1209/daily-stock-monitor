"""Synthetic-only teacher, official-clock, annual-router, and selector checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_leader_utility260_kernel as k
import test_monthly_joint_identity259_kernel as old

class Poison:
    def __float__(self):raise AssertionError('Future/unselected numeric accessed')
    def __deepcopy__(self,memo):raise AssertionError('Future metadata accessed')
    def __eq__(self,other):raise AssertionError('Future metadata compared')


def fixture():
    study,train,meta,a=old.fixture()
    days=pd.date_range('2010-12-31','2017-12-31',freq='BME');ends=days+pd.offsets.BMonthEnd(1);w=a['baseline'].shape[1]
    train['dates']=np.repeat(days.values.astype('datetime64[D]'),w)
    train['ends']=np.repeat(ends.values.astype('datetime64[D]'),w)
    train['train_eligible']=np.ones(len(train['dates']),bool)
    meta['training']['dates']=days.strftime('%Y-%m-%d').tolist();meta['training']['ends']=ends.strftime('%Y-%m-%d').tolist()
    study,train,meta,a=old.refresh_original((study,train,meta,a))
    cal=pd.bdate_range('2008-01-02','2018-02-28');t=np.arange(len(cal),dtype=float)
    adj=pd.DataFrame({'SPY':np.exp(.0002*t+.04*np.sin(t/47.)), 'XLY':np.exp(.0003*t+.03*np.cos(t/31.)), 'XLP':np.exp(.0001*t+.02*np.sin(t/69.))},index=cal)
    return study,train,meta,a,adj,cal.strftime('%Y-%m-%d').tolist()


def source(args,n=None):
    inp=k.router_inputs(args[1],args[2],args[3],n);target=k.build_expert_targets(inp)
    tx,qx,audit=k.build_market_inputs(args[4],args[5],target['dates'],inp['event_dates'])
    return inp,target,tx,qx,audit


def route(args,n=None):
    inp,target,tx,qx,src=source(args,n)
    return inp,k.fit_router(target,tx,qx,inp['event_dates']),target,src


def decide(args,bundle,n=None,**kwargs):return k.decisions(args[0],args[3],args[2]['monthly'],bundle,args[1]['event_plain'],n,**kwargs)


def physical(args,n):
    study,train,meta,a,adj,cal=args;w=a['baseline'].shape[1]
    mm=deepcopy(meta)
    for key in ('dates','ends','symbols'):mm['monthly'][key]=mm['monthly'][key][:n]
    dates=mm['monthly']['dates'];ordinary=[d for d in dates if d!=k.INITIAL_PARTIAL]
    cutoff=np.datetime64('%04d-12-31'%(int(ordinary[-1][:4])-1)) if ordinary else None
    keep=train['ends']<=cutoff if cutoff is not None else np.zeros(len(train['ends']),bool)
    months=int(keep.sum())//w;tt={}
    for key,value in train.items():
        if key in ('event_plain','event_dates'):tt[key]=value[:n*w].copy()
        elif key=='offsets':tt[key]=np.arange(months+1)*w
        else:tt[key]=value[keep].copy()
    for key in ('dates','ends','symbols'):mm['training'][key]=mm['training'][key][:months]
    end=dates[-1] if n else cal[0]
    return study,tt,mm,{key:v[:n].copy() for key,v in a.items()},adj.loc[:end].copy(),cal


def tiny(plain=None,eligible=None,good=None,labels=None,names=None):
    names=list('ABCD') if names is None else names;w=len(names)
    plain=np.tile(np.linspace(.4,-.4,w)[:,None],(1,10)) if plain is None else np.asarray(plain,float)
    inp=dict(schema=k._schema('inputs'),n=1,width=w,training_row_ids=np.arange(w),dates=np.full(w,np.datetime64('2014-01-31')),ends=np.full(w,np.datetime64('2014-02-28')),symbols=np.asarray(names),offsets=np.array([0,w]),plain=plain,eligible=np.ones(w,bool) if eligible is None else np.asarray(eligible,bool),common_good=np.ones(w,bool) if good is None else np.asarray(good,bool),labels=np.arange(w)/10. if labels is None else np.asarray(labels),event_dates=np.array(['2016-01-29'],dtype='datetime64[D]'),event_symbols=[names],event_plain=np.zeros((w,10)))
    return inp


def model_fixture(months=48):
    dates=pd.date_range('2011-12-30',periods=months,freq='BME').values.astype('datetime64[D]')
    ends=(pd.DatetimeIndex(dates)+pd.offsets.BMonthEnd(1)).values.astype('datetime64[D]')
    z=np.arange(months,dtype=float);X=np.column_stack((np.sin(z/5.),np.cos(z/7.),np.sin(z/11.)))
    y=.07+.11*X[:,0]-.09*X[:,1]+.015*np.cos(z/3.)
    targets=dict(schema=k._schema('targets'),dates=dates,ends=ends,valid=np.ones(months,bool),values=y,records=[],counts={})
    return targets,X,np.array([[.2,-.1,.3],[.6,.1,-.4]]),np.array(['2016-01-29','2016-02-29'],dtype='datetime64[D]')


class TeacherTests(unittest.TestCase):
    def test_selected_invalid_not_replaced(self):
        x=tiny(good=[False,True,True,True]);r=k.build_expert_targets(x)
        self.assertFalse(r['valid'][0]);self.assertEqual(r['records'][0]['raw_choices'],['A','B','C']);self.assertEqual(r['counts']['selected_label_observations'],3)
    def test_unselected_label_poison_ignored(self):
        x=tiny(labels=np.array([.1,.2,.3,Poison()],object));self.assertEqual(k.build_expert_targets(x)['values'][0],0.)
    def test_cash_does_not_touch_any_labels(self):
        x=tiny(eligible=[True,False,False,False],labels=np.array([Poison()]*4,object));r=k.build_expert_targets(x)
        self.assertEqual(r['records'][0]['reason'],'cash');self.assertEqual(r['counts']['selected_label_observations'],0)
    def test_two_names_identical_after_validity(self):
        x=tiny(eligible=[True,False,True,False]);x['plain'][[0,2],1]=[-.4,.4]
        r=k.build_expert_targets(x);self.assertEqual(r['records'][0]['k'],2);self.assertEqual(r['values'][0],0.)
        x['labels'][2]=np.nan;self.assertFalse(k.build_expert_targets(x)['valid'][0])
    def test_symmetric_difference_and_symbol_ties(self):
        x=tiny(names=['D','A','C','B']);x['plain'][:,0]=[.3,.3,.1,-.1];x['plain'][:,1]=[-.1,.1,.3,.3];x['labels']=np.array([.8,.1,.4,-.2])
        r=k.build_expert_targets(x);row=r['records'][0]
        self.assertEqual(row['raw_choices'],['A','D','C']);self.assertEqual(row['risk_choices'],['B','C','A'])
        self.assertEqual(row['raw_only'],['D']);self.assertEqual(row['risk_only'],['B']);self.assertEqual(r['values'][0],(np.float64(.8)-np.float64(-.2))/3.)
    def test_teacher_does_not_filter_common_good_before_selection(self):
        x=tiny();x['common_good'][0]=False;x['labels'][0]=np.inf
        self.assertEqual(k.build_expert_targets(x)['records'][0]['raw_choices'],['A','B','C'])
    def test_finite_all10_domain_not_just_two_columns(self):
        x=tiny();x['plain'][0,9]=np.nan;r=k.build_expert_targets(x)
        self.assertEqual(r['records'][0]['domain_indices'],[1,2,3])
    def test_selected_type_is_hard_error(self):
        for bad in (True,'0.1'):
            x=tiny(labels=np.array([bad,.1,.2,.3],object))
            with self.assertRaises(ValueError):k.build_expert_targets(x)
    def test_targets_do_not_modify_inputs(self):
        x=tiny();before={key:v.tobytes() for key,v in x.items() if isinstance(v,np.ndarray)};k.build_expert_targets(x)
        self.assertEqual(before,{key:v.tobytes() for key,v in x.items() if isinstance(v,np.ndarray)})


class ModelTests(unittest.TestCase):
    def test_canonical_and_augmented_svd(self):
        target,X,Q,events=model_fixture();r=k.fit_router(target,X,Q,events);a=r['audit']['annual'][0]
        W=np.array(a['weights']);Z=(X-np.array(a['mean_x']))/np.array(a['scale_x']);y=target['values']-a['mean_d']
        A=np.vstack((np.sqrt(W)[:,None]*Z,np.sqrt(10.)*np.eye(3)));b=np.r_[np.sqrt(W)*y,np.zeros(3)]
        beta=np.linalg.lstsq(A,b,rcond=None)[0]
        np.testing.assert_allclose(beta,a['beta'],rtol=2e-10,atol=2e-10)
        self.assertAlmostEqual(W.sum(),len(X),places=12);self.assertTrue(r['active'].all());self.assertEqual(r['audit']['counts']['ridge_solves'],1)
        self.assertEqual(r['audit']['counts']['original_model_fits'],0)
    def test_minimum_months_and_constant_target_copy(self):
        target,X,Q,events=model_fixture(17);r=k.fit_router(target,X,Q,events)
        self.assertFalse(r['active'].any());self.assertEqual(r['audit']['annual'][0]['status'],'insufficient_months')
        target,X,Q,events=model_fixture();target['values'][:]=.125;r=k.fit_router(target,X,Q,events)
        self.assertFalse(r['active'].any());self.assertEqual(r['audit']['counts']['ridge_solves'],0)
        self.assertEqual(r['audit']['annual'][0]['status'],'constant_target')
    def test_constant_state_floor_is_one(self):
        target,X,Q,events=model_fixture();X[:,2]=0.;Q[:,2]=0.;r=k.fit_router(target,X,Q,events)
        self.assertEqual(r['audit']['annual'][0]['scale_x'][2],1.);self.assertEqual(r['audit']['annual'][0]['beta'][2],0.)
    def test_zero_state_conditional_equals_intercept(self):
        target,X,Q,events=model_fixture();X[:]=0.;Q[:]=0.;r=k.fit_router(target,X,Q,events)
        self.assertEqual(r['state_p'].tobytes(),r['intercept_p'].tobytes())
    def test_fit_gate_precedes_any_new_moments(self):
        def fail():return {'passed':False}
        with patch.object(np.linalg,'solve',side_effect=AssertionError('solve before gate')):
            with self.assertRaises(ValueError):k.fit_router(*model_fixture(),control_validator=fail)
    def test_solve_failure_no_fallback(self):
        with patch.object(np.linalg,'solve',side_effect=np.linalg.LinAlgError('fixture')):
            with self.assertRaises(np.linalg.LinAlgError):k.fit_router(*model_fixture())
    def test_nonfinite_solution_rejected(self):
        for bad in (np.nan,np.inf):
            with patch.object(np.linalg,'solve',return_value=np.full(3,bad)):
                with self.assertRaises(ValueError):k.fit_router(*model_fixture())
    def test_exact_maturity_and_five_year_lower_boundary(self):
        target,X,Q,events=model_fixture();target['dates'][0]=np.datetime64('2010-12-31');target['ends'][0]=np.datetime64('2011-01-31')
        target['ends'][-1]=np.datetime64('2016-01-29');r=k.fit_router(target,X,Q,events)
        ids=r['audit']['annual'][0]['training_indices'];self.assertNotIn(0,ids);self.assertNotIn(len(X)-1,ids)
    def test_future_numeric_poison_not_consumed(self):
        target,X,Q,events=model_fixture();target['dates']=np.r_[target['dates'],np.datetime64('2016-01-29')];target['ends']=np.r_[target['ends'],np.datetime64('2016-02-29')];target['valid']=np.r_[target['valid'],True];target['values']=np.r_[target['values'].astype(object),Poison()];X=np.vstack((X.astype(object),[Poison()]*3))
        r=k.fit_router(target,X,Q,events);self.assertTrue(r['active'].all())
    def test_partial_query_never_reads_states(self):
        target,X,_,_=model_fixture();events=np.array([k.INITIAL_PARTIAL],dtype='datetime64[D]');Q=np.array([[Poison()]*3],object)
        r=k.fit_router(target,X,Q,events);self.assertFalse(r['active'][0]);self.assertEqual(r['audit']['counts']['ridge_solves'],0)
    def test_label_scale_invariance_prediction_direction(self):
        target,X,Q,events=model_fixture();Q[1]=Q[0];Q[1,0]+=.4;r=k.fit_router(target,X,Q,events);target['values']*=2.;s=k.fit_router(target,X,Q,events)
        np.testing.assert_array_equal(r['state_p'],s['state_p']);self.assertGreater(r['state_prediction'][1],r['state_prediction'][0])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.inp,cls.bundle,cls.targets,cls.source=route(cls.args)
    def test_exact_source_identity_and_cutoff(self):
        inp=self.inp;self.assertEqual(set(inp),set(k.INPUT_KEYS));self.assertTrue(np.all(inp['ends']<=np.datetime64('2016-12-31')))
        self.assertEqual(inp['labels'].tobytes(),self.args[1]['target_simple_excess'][:len(inp['labels'])].tobytes())
    def test_n0_and_partial_only_no_training_body(self):
        args=list(self.args);t=deepcopy(args[1]);t['train_plain']=np.full(t['train_plain'].shape,Poison(),object);t['target_simple_excess']=np.full(t['target_simple_excess'].shape,Poison(),object);args[1]=t
        for n in (0,1):
            inp=k.router_inputs(t,args[2],args[3],n);self.assertEqual(len(inp['labels']),0)
            tar=k.build_expert_targets(inp);tx,qx,_=k.build_market_inputs(pd.DataFrame(index=pd.DatetimeIndex([])),args[5],tar['dates'],inp['event_dates']);r=k.fit_router(tar,tx,qx,inp['event_dates']);self.assertEqual(r['audit']['counts']['ridge_solves'],0)
    def test_input_future_object_poison_before_cast(self):
        t=deepcopy(self.args[1]);cut=t['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain','target_simple_excess'):
            t[key]=t[key].astype(object);t[key][cut]=Poison()
        result=k.router_inputs(t,self.args[2],self.args[3],2);self.assertTrue(np.isfinite(result['plain']).all())
    def test_future_training_identity_body_poison(self):
        meta=deepcopy(self.args[2]);ends=np.asarray(meta['training']['ends'],dtype='datetime64[D]');count=int(np.count_nonzero(ends<=np.datetime64('2015-12-31')))
        meta['training']['symbols'][count:]=[Poison()]*(len(ends)-count)
        inp=k.router_inputs(self.args[1],meta,self.args[3],2);self.assertEqual(len(inp['offsets']),count+1)
        empty=k.router_inputs(self.args[1],meta,self.args[3],0);self.assertEqual(empty['symbols'].dtype.kind,'U')
    def test_input_metadata_identity_tamper(self):
        t=deepcopy(self.args[1]);t['offsets'][1]+=1
        with self.assertRaises(ValueError):k.router_inputs(t,self.args[2],self.args[3])
    def test_market_formula_and_counts(self):
        d='2016-08-31';tx,qx,audit=k.build_market_inputs(self.args[4],self.args[5],[],[d]);t=self.args[5].index(d);px=self.args[4]['SPY'].to_numpy()
        self.assertEqual(qx[0,1],px[t-42]/px[t-252]-1.);self.assertEqual(audit['counts']['price_observations'],507)
        self.assertEqual(audit['queries'][0]['spy_start'],self.args[5][t-252])
    def test_market_missing_official_session_hardfail(self):
        d='2016-08-31';t=self.args[5].index(d);adj=self.args[4].drop(pd.Timestamp(self.args[5][t-100]))
        with self.assertRaises(ValueError):k.build_market_inputs(adj,self.args[5],[],[d])
    def test_market_bad_prices_and_no_fill(self):
        d='2016-08-31'
        for bad in (0.,-1.,np.inf,np.nan,True,'1'):
            adj=self.args[4].astype(object);adj.loc[d,'XLY']=bad
            with self.assertRaises(ValueError):k.build_market_inputs(adj,self.args[5],[],[d])
    def test_market_future_poison_irrelevant(self):
        d='2016-08-31';adj=self.args[4].astype(object);adj.loc[adj.index>pd.Timestamp(d),:]=Poison()
        actual=k.build_market_inputs(adj,self.args[5],[],[d]);expected=k.build_market_inputs(self.args[4],self.args[5],[],[d]);np.testing.assert_array_equal(actual[1],expected[1]);self.assertEqual(actual[2],expected[2])
    def test_market_skips42_momentum_but_vol_uses_current(self):
        d='2016-08-31';adj=self.args[4].copy();adj.loc[d,'SPY']*=1.1
        before=k.build_market_inputs(self.args[4],self.args[5],[],[d])[1];after=k.build_market_inputs(adj,self.args[5],[],[d])[1]
        self.assertEqual(before[0,1],after[0,1]);self.assertNotEqual(before[0,0],after[0,0])
    def test_incumbent_gate_and_partial_exact(self):
        o,s,p=k.original_decisions(self.args[0],self.args[3],self.args[2]['monthly']);calls=[]
        def gate(oo,ss,pp):
            self.assertEqual(oo,o);self.assertEqual(ss.tobytes(),s.tobytes());self.assertEqual(pp.tobytes(),p.tobytes());calls.append(1);return dict(passed=True,scope='test')
        out=decide(self.args,self.bundle,incumbent_validator=gate,expected_incumbent_orders=o)
        self.assertEqual(calls,[1]);self.assertEqual(out[0][k.ARMS[0]],o)
        for arm in k.ARMS:
            self.assertEqual(out[1][arm][0].tobytes(),s[0].tobytes());self.assertEqual(out[2][arm][0].tobytes(),p[0].tobytes())
    def test_failed_incumbent_gate_prevents_candidate(self):
        b=deepcopy(self.bundle);b['active']=Poison()
        with self.assertRaises(ValueError):decide(self.args,b,incumbent_validator=lambda *x:{'passed':False})
    def test_weighted_core_units_both_states_and_equal_ranks(self):
        args=list(self.args);args[1]=deepcopy(args[1]);w=args[3]['baseline'].shape[1];r=args[1]['event_plain'].reshape(4,w,10);r[:,:,0]=.3;r[:,:,1]=.3
        out=decide(args,self.bundle)
        for i in (1,2,3):
            if self.bundle['active'][i]:
                expected=.10*(.3-args[3]['core'][i]/args[3]['core_weight'][i])
                np.testing.assert_array_equal(out[3]['events'][i]['state_delta'],expected)
                self.assertTrue(np.count_nonzero(expected)>0)
        self.assertEqual(float(args[3]['core_weight'][1,0]),.52);self.assertEqual(float(args[3]['core_weight'][2,0]),.5)
    def test_zero_delta_preserves_original_pre_bytes(self):
        args=list(self.args);args[1]=deepcopy(args[1]);w=args[3]['baseline'].shape[1];r=args[1]['event_plain'].reshape(4,w,10);c=args[3]['core']/args[3]['core_weight'];r[:,:,0]=c;r[:,:,1]=c
        out=decide(args,self.bundle)
        for arm in k.ARMS[1:]:self.assertEqual(out[2][arm].tobytes(),out[2][k.ARMS[0]].tobytes())
    def test_original_E_not_forecast_population(self):
        args=list(self.args);args[3]=deepcopy(args[3]);args[3]['forecast_available'][1]=False
        out=decide(args,self.bundle);self.assertEqual(len(out[3]['events'][1]['eligible_indices']),6)
        self.assertTrue(np.count_nonzero(out[3]['events'][1]['state_delta'])>0)
    def test_invalid_eligible_rank_hardfail(self):
        args=list(self.args);args[1]=deepcopy(args[1]);args[1]['event_plain'][6,0]=np.nan
        with self.assertRaises(ValueError):decide(args,self.bundle)
    def test_one_ulp_rank_outside_original_range_rejected(self):
        args=list(self.args);args[1]=deepcopy(args[1]);args[1]['event_plain'][6,0]=np.nextafter(.5,np.inf)
        with self.assertRaises(ValueError):decide(args,self.bundle)
    def test_cold_routes_exact_copy(self):
        b=deepcopy(self.bundle);b['active'][:]=False;b['intercept_prediction'][:]=0.;b['state_prediction'][:]=0.;b['intercept_p'][:]=.5;b['state_p'][:]=.5
        out=decide(self.args,b)
        for arm in k.ARMS[1:]:self.assertEqual(out[0][arm],out[0][k.ARMS[0]]);self.assertEqual(out[1][arm].tobytes(),out[1][k.ARMS[0]].tobytes())
    def test_all_logical_physical_and_fullbundle_prefixes(self):
        full=decide(self.args,self.bundle)
        for n in range(5):
            part=physical(self.args,n);inp,b,_,_=route(part);x=decide(part,b);y=decide(self.args,self.bundle,n);z=decide(part,self.bundle)
            for v in (x,y,z):
                for arm in k.ARMS:
                    self.assertEqual(v[0][arm],full[0][arm][:n]);self.assertEqual(v[1][arm].tobytes(),full[1][arm][:n].tobytes());self.assertEqual(v[2][arm].tobytes(),full[2][arm][:n].tobytes())
                self.assertEqual(v[3]['events'],full[3]['events'][:n])
    def test_future_bundle_and_metadata_body_poison(self):
        b=deepcopy(self.bundle);args=list(self.args);args[2]=deepcopy(args[2]);args[1]=deepcopy(args[1]);n=2
        for key in ('intercept_prediction','state_prediction','intercept_p','state_p'):
            b[key]=b[key].astype(object);b[key][n:]=Poison()
        for key in ('dates','ends','symbols'):args[2]['monthly'][key][n:]=[Poison()]*(4-n)
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]=Poison()
        x=decide(args,b,n);y=decide(self.args,self.bundle,n);self.assertEqual(x[0],y[0]);self.assertEqual(x[3],y[3])
    def test_all_original_arrays_unchanged_and_calls(self):
        before=k.bridge._snapshot(self.args[3]);real=self.args[0].ep.select_boundary;calls=[]
        def counted(*a,**kw):calls.append(1);return real(*a,**kw)
        with patch.object(self.args[0].ep,'select_boundary',side_effect=counted):out=decide(self.args,self.bundle)
        self.assertEqual(len(calls),3);self.assertEqual(k.bridge._snapshot(self.args[3]),before)
        self.assertEqual(set(out[3]['counts']),set(k.COUNT_FIELDS));self.assertEqual(set(out[3]['events'][0]),set(k.EVENT_KEYS))


if __name__=='__main__':unittest.main(verbosity=2)
