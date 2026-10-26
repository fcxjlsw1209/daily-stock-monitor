"""Synthetic source/control/dummy/selection tests; never load actual input caches."""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_joint_identity259_kernel as k

def fixture():
    spec=importlib.util.spec_from_file_location('identity259_synthetic_study',k.ROOT/'outputs/monthly_local_linear70/study.py')
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    names=list('ABCDEF');width=6
    days=pd.date_range('2010-12-31','2017-12-31',freq='ME')
    dates=np.repeat(days.values.astype('datetime64[D]'),width)
    ends=np.repeat((days+pd.offsets.MonthEnd(1)).values.astype('datetime64[D]'),width)
    rng=np.random.default_rng(259)
    X=rng.uniform(-.48,.48,(len(dates),10))
    raw=.13*X[:,0]-.09*X[:,1]+.06*X[:,3]**2+np.tile([.01,.03,-.02,-.01,-.04,.15],len(days))
    y=(raw.reshape(-1,width)-raw.reshape(-1,width).mean(axis=1,keepdims=True)).reshape(-1)
    good=np.ones(len(dates),bool);good[7]=False
    targets={name:(y*(i+1)).copy() for i,name in enumerate(k.TARGETS)}
    for values in targets.values():values[~good]=np.nan
    query=['2016-08-17','2016-08-31','2017-01-31','2017-02-28']
    query_end=['2016-08-31','2016-09-30','2017-02-28','2017-03-31']
    m=dict(dates=query,ends=query_end,symbols=[names.copy() for _ in query],etfs=['F'])
    tm=dict(dates=days.strftime('%Y-%m-%d').tolist(),ends=(days+pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d').tolist(),symbols=[names.copy() for _ in days])
    eventX=rng.uniform(-.48,.48,(24,10))
    train=dict(dates=dates,ends=ends,train_plain=X,event_plain=eventX,
        event_dates=np.repeat(np.asarray(query,dtype='datetime64[D]'),width),
        common_good=good,offsets=np.arange(len(days)+1)*width,
        **{'target_'+name:value for name,value in targets.items()})
    shape=(4,width);zero=np.zeros(shape);yes=np.ones(shape,bool)
    calm=np.array([False,False,True,False]);weak=np.array([True,True,False,True])
    cw=np.where(calm,.5,.52)[:,None];core=cw*np.tile(np.linspace(-.2,.2,width),(4,1))
    arrays=dict(baseline=core+.07,core=core,core_weight=cw,eligible=yes.copy(),
        effects=np.tile([0.,0.,0.,.5,-.5,0.],(4,1)),state_calm10=calm,state_weak_narrow=weak,
        pre_score=np.tile([.31,.305,.30,.295,.29,.285],(4,1)),old_ml4=zero.copy(),extension_available=yes.copy())
    return refresh_original((study,train,dict(monthly=m,training=tm),arrays))

def refresh_original(args, **kwargs):
    study,train,meta,a=args;n,width=a['baseline'].shape
    out=k.model.annual_forecasts(*k.model_inputs(train,meta,a),**kwargs)
    a['forecast_available']=out['available'].reshape(n,width).copy()
    a['model_available']=a['forecast_available'].copy()
    a['local_available']=out['local_available'].reshape(n,width).copy()
    for key in k.RAW_KEYS:a['raw_'+key]=out['raw'][key].reshape(n,width).copy()
    a['raw_log_excess']=a['raw_source_log'].copy()
    refresh_ranks(args)
    return args

def refresh_ranks(args):
    study,train,meta,a=args;n,width=a['baseline'].shape
    a['rank_log_excess']=k._rank(a['raw_source_log'],a['forecast_available']&a['eligible'],np.zeros((n,width)))
    ranks,ready=study.ranked_forecasts({key:a['raw_'+key] for key in ('local_constant','local_linear')},a['model_available'],a,n)
    for key,value in ranks.items():a['rank_'+key]=value
    a['paired_month_ready']=ready

def fit(args,n=None,**kwargs):return k.fit_models(args[1],args[2],args[3],n,**kwargs)
def decide(args,bundle,n=None,**kwargs):return k.decisions(args[0],args[3],args[2]['monthly'],bundle,n,**kwargs)
def clone(args):return (args[0],)+deepcopy(args[1:])

def physical(args,n):
    study,train,meta,arrays=args;out=deepcopy(meta);width=arrays['baseline'].shape[1]
    for key in ('dates','ends','symbols'):out['monthly'][key]=out['monthly'][key][:n]
    small={key:value[:n].copy() for key,value in arrays.items()}
    target=deepcopy(train);target['event_dates']=target['event_dates'][:n*width];target['event_plain']=target['event_plain'][:n*width]
    cutoff=np.datetime64('%04d-12-31'%(int(out['monthly']['dates'][-1][:4])-1)) if n else None
    keep=train['ends']<=cutoff if n else np.zeros(len(train['ends']),bool)
    months=int(keep.sum())//width
    for key in ('dates','ends','symbols'):out['training'][key]=out['training'][key][:months]
    for key in ('dates','ends','train_plain','common_good')+tuple('target_'+x for x in k.TARGETS):target[key]=target[key][keep]
    target['offsets']=np.arange(months+1)*width
    return study,target,out,small

class AlgebraTests(unittest.TestCase):
    def test_block_orthogonal_case_joint_matches_sequential(self):
        Z=np.zeros((8,10));Z[:,0]=np.tile([.25,-.25],4)
        H=np.column_stack((np.repeat([.5,-.5],4),np.repeat([-.5,.5],4)))
        w=np.ones(8);y=.125*Z[:,0]+.25*H[:,0]
        self.assertTrue(np.all(np.einsum('ni,n,nj->ij',Z,w,H,optimize=False)==0.))
        beta,_,_=k._solve(Z,w,y,np.full(10,10.))
        delta,_,_=k._solve(H,w,y-np.einsum('ij,j->i',Z,beta,optimize=False),np.full(2,12.))
        theta,_,_=k._solve(np.column_stack((Z,H)),w,y,np.r_[np.full(10,10.),np.full(2,12.)])
        np.testing.assert_allclose(theta,np.r_[beta,delta],rtol=0,atol=2e-16)
    def test_nonzero_crossblock_creates_feedback(self):
        Z=np.zeros((8,10));Z[:,0]=np.repeat([.25,-.25],4)
        H=np.column_stack((np.repeat([.5,-.5],4),np.repeat([-.5,.5],4)))
        w=np.ones(8);y=.25*H[:,0]
        beta,_,_=k._solve(Z,w,y,np.full(10,10.))
        theta,_,_=k._solve(np.column_stack((Z,H)),w,y,np.r_[np.full(10,10.),np.full(2,12.)])
        self.assertNotEqual(theta[0],beta[0])
    def test_collinear_dummies_positive_penalty_unique(self):
        H=np.column_stack((np.repeat([.5,-.5],4),np.repeat([-.5,.5],4)))
        theta,gram,rhs=k._solve(H,np.ones(8),np.arange(8)/10.,np.full(2,12.))
        self.assertTrue(np.linalg.eigvalsh(gram).min()>0.)
        np.testing.assert_allclose(gram@theta,rhs,rtol=0,atol=1e-15)
    def test_constant_response_is_valid_zero_solve(self):
        H=np.array([[.5,-.5],[-.5,.5]])
        theta,_,_=k._solve(H,np.ones(2),np.zeros(2),np.full(2,12.))
        self.assertEqual(theta.tobytes(),np.zeros(2).tobytes())
    def test_solve_failure_is_not_fallback(self):
        with patch.object(np.linalg,'solve',side_effect=np.linalg.LinAlgError('singular')):
            with self.assertRaises(np.linalg.LinAlgError):k._solve(np.ones((3,1)),np.ones(3),np.ones(3),np.array([12.]))
    def test_nonfinite_new_coefficient_rejected(self):
        for value in (np.nan,np.inf):
            with patch.object(np.linalg,'solve',return_value=np.array([value])):
                with self.assertRaises(ValueError):k._solve(np.ones((3,1)),np.ones(3),np.ones(3),np.array([12.]))
    def test_full_training_dummy_build_not_query_dependent(self):
        dates=np.array(['2014-01-31','2014-01-31','2014-02-28','2014-02-28','2014-03-31'])
        symbols=np.array(['A','B','A','C','A']);X=np.zeros((5,10));w=np.array([1.,2.,3.,4.,5.])
        d=k.identity_design(dates.astype('datetime64[D]'),symbols,X,w,(np.zeros(10),np.ones(10),np.zeros(10),0.),2)
        self.assertEqual(d['qualified'],['A']);self.assertEqual(d['own_month_counts'],{'A':3,'B':1,'C':1})
        self.assertEqual(d['H'].shape,(5,1));np.testing.assert_array_equal(d['H'][:,0],[1,0,1,0,1])
        self.assertEqual(d['dummy_mean'][0],9./15.)
    def test_exact_constant_dummy_design_copy_branch(self):
        dates=pd.date_range('2012-01-31',periods=18,freq='ME').values.astype('datetime64[D]')
        symbols=np.full(18,'A');X=np.zeros((18,10));ends=dates+np.timedelta64(28,'D')
        pooled=(np.zeros(10),np.ones(10),np.zeros(10),.125)
        args=(dates,ends,symbols,X,{key:np.full(18,.125) for key in k.TARGETS},np.ones(18,bool),np.array(['2014-01-31'],dtype='datetime64[D]'),np.array(['A']),np.zeros((1,10)),np.ones(1,bool))
        raw={key:np.array([.125]) for key in k.RAW_KEYS};control=dict(raw=raw,available=np.ones(1,bool),local_available=np.ones(1,bool),audit={})
        context=dict(year=2014,cutoff=np.datetime64('2013-12-31'),ids=np.arange(18),pooled=pooled,predict=np.array([0]))
        # Uniform exact weights make Hc exactlyzero: the branch is mathematical,
        # not a tolerance shortcut. This isolates the defensive pure helper.
        with patch.object(k.model.numeric,'month_balanced_weights',return_value=np.ones(18)),patch.object(k,'_solve',side_effect=AssertionError('no new solve')):
            seq,joint=k.new_forecasts(args,control,[context])
        for out in (seq,joint):
            self.assertEqual(out['audit']['annual'][0]['status'],'zero_centered_design')
            self.assertEqual(out['raw']['local_linear'].tobytes(),raw['source_log'].tobytes())

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.full=fit(cls.args)
    def test_control_exact_source_raw_and_audit(self):
        reference=k.model.annual_forecasts(*k.model_inputs(self.args[1],self.args[2],self.args[3]))
        self.assertEqual(reference['audit'],self.full['control']['audit'])
        for key in k.RAW_KEYS:self.assertEqual(reference['raw'][key].tobytes(),self.full['control']['raw'][key].tobytes())
    def test_exact_schemas_and_counts(self):
        b=self.full;c=b['audit']['counts']
        self.assertEqual(tuple(b),k.BUNDLE_KEYS);self.assertEqual(tuple(c),k.MODEL_COUNT_FIELDS)
        self.assertEqual(c['control_target_fits'],8);self.assertEqual(c['control_log_refits'],2)
        self.assertEqual(c['annual_design_builds'],2)
        self.assertEqual(c['diagnostic_dummy_solves'],2);self.assertEqual(c['candidate_joint_solves'],2)
        self.assertEqual(c['diagnostic_pooled_fits'],0);self.assertEqual(c['candidate_pooled_fits'],0)
        self.assertEqual(c['qualified_query_rows']+c['cold_query_rows'],c['control_local_evaluations'])
        self.assertEqual(c['actual_new_model_solves'],4)
        self.assertEqual(c['actual_regression_solves'],sum(c[x] for x in ('control_target_fits','control_log_refits','control_local_solves','actual_new_model_solves')))
        for role in ('diagnostic','candidate'):
            self.assertEqual(tuple(b[role]['audit']),k.FORECAST_AUDIT_KEYS)
            for row in b[role]['audit']['annual']:self.assertEqual(tuple(row),k.ANNUAL_KEYS)
        json.dumps(b['audit'],allow_nan=False)
    def test_native_security_identity_audit_metadata(self):
        for role in ('diagnostic','candidate'):
            for row in self.full[role]['audit']['annual']:
                self.assertTrue(all(type(symbol) is str for symbol in row['qualified_symbols']))
                self.assertTrue(all(type(symbol) is str for symbol in row['own_month_counts']))
                self.assertTrue(all(type(count) is int for count in row['own_month_counts'].values()))
                self.assertEqual(json.loads(json.dumps(row)),row)
    def test_actual_solve_instrumentation_and_no_geometry(self):
        with patch.object(k,'_solve',wraps=k._solve) as solve,patch.object(k.base,'diagonal_weights',side_effect=AssertionError('geometry forbidden')):
            b=fit(self.args)
        self.assertEqual(solve.call_count,b['audit']['counts']['actual_new_model_solves'])
    def test_source_constant_support_and_labels_byte_preserved(self):
        b=self.full;inp=k.model_inputs(self.args[1],self.args[2],self.args[3])
        self.assertEqual(tuple(b['label_preservation']),k.LABEL_KEYS)
        self.assertFalse(b['label_preservation']['recentered'])
        self.assertEqual(b['label_values'].tobytes(),inp[4]['log_excess'].tobytes())
        for key in k.TARGETS:self.assertEqual(b['label_preservation']['target_sha256'][key],k._digest(inp[4][key]))
        for role in ('diagnostic','candidate'):
            for key in ('source_log','local_constant'):self.assertEqual(b[role]['raw'][key].tobytes(),b['control']['raw'][key].tobytes())
            for key in ('available','local_available'):self.assertEqual(b[role][key].tobytes(),b['control'][key].tobytes())
    def test_augmented_svd_matches_both_penalized_solutions(self):
        args=k.model_inputs(self.args[1],self.args[2],self.args[3])
        for role in ('diagnostic','candidate'):
            for row in self.full[role]['audit']['annual']:
                ids=np.asarray(row['training_row_ids']);w=np.asarray(row['weights'])
                Z=(args[3][ids]-np.asarray(row['source_mean']))/np.asarray(row['source_scale'])
                H=np.asarray(args[2][ids,None]==np.asarray(row['qualified_symbols'])[None,:],float)-np.asarray(row['dummy_mean'])
                y=args[4]['log_excess'][ids]-row['source_intercept']
                if role=='diagnostic':
                    D=H;y=y-np.einsum('ij,j->i',Z,np.asarray(row['source_beta']),optimize=False);got=np.asarray(row['dummy_delta'])
                else:D=np.column_stack((Z,H));got=np.r_[row['global_beta'],row['dummy_delta']]
                penalty=np.asarray(row['penalty']);A=np.vstack((np.sqrt(w)[:,None]*D,np.diag(np.sqrt(penalty))));rhs=np.r_[np.sqrt(w)*y,np.zeros(len(penalty))]
                expected=np.linalg.lstsq(A,rhs,rcond=None)[0]
                np.testing.assert_allclose(got,expected,atol=2e-10,rtol=2e-10)
                self.assertEqual(k._digest(H),row['centered_dummy_sha256']);self.assertEqual(k._digest(Z),row['standardized_design_sha256'])
    def test_joint_beta_differs_with_nonzero_crossblock(self):
        rows=self.full['candidate']['audit']['annual']
        self.assertTrue(any(np.any(np.asarray(r['feature_dummy_cross'])!=0.) for r in rows))
        self.assertTrue(any(np.any(np.asarray(r['global_beta'])!=np.asarray(r['source_beta'])) for r in rows))
    def test_qualified_never_queried_security_still_column(self):
        args=clone(self.args)
        for names in args[2]['monthly']['symbols']:names[-1]='NEW'
        refresh_original(args);b=fit(args)
        for row in b['candidate']['audit']['annual']:
            self.assertIn('F',row['qualified_symbols']);self.assertNotIn('NEW',row['qualified_symbols'])
        for role in ('diagnostic','candidate'):
            for j in range(5,24,6):self.assertEqual(b[role]['raw']['local_linear'][j].tobytes(),b['control']['raw']['source_log'][j].tobytes())
    def test_seventeen_month_security_training_rows_retained_query_poolcopy(self):
        args=clone(self.args);t=args[1];is_f=np.tile(np.array([False]*5+[True]),len(args[2]['training']['dates']))
        keep=(t['dates']>=np.datetime64('2014-07-31'))&(t['dates']<=np.datetime64('2015-11-30'))
        t['common_good'][is_f&~keep]=False
        refresh_original(args);b=fit(args)
        for row in b['candidate']['audit']['annual']:
            self.assertEqual(row['own_month_counts']['F'],17);self.assertNotIn('F',row['qualified_symbols'])
            inp=k.model_inputs(args[1],args[2],args[3])
            self.assertEqual(sum(inp[2][row['training_row_ids']]=='F'),17)
        for role in ('diagnostic','candidate'):
            for j in range(5,24,6):self.assertEqual(b[role]['raw']['local_linear'][j].tobytes(),b['control']['raw']['source_log'][j].tobytes())
    def test_no_qualified_columns_no_new_solves(self):
        args=clone(self.args);refresh_original(args,min_own_months=1000)
        with patch.object(k,'_solve',side_effect=AssertionError('no new solve')):b=fit(args,min_own_months=1000)
        self.assertEqual(b['audit']['counts']['no_qualified_columns_years'],2)
        for role in ('diagnostic','candidate'):self.assertEqual(b[role]['raw']['local_linear'].tobytes(),b['control']['raw']['source_log'].tobytes())
    def test_no_qualified_queries_keeps_all_columns_no_new_solves(self):
        args=clone(self.args)
        for i,names in enumerate(args[2]['monthly']['symbols']):args[2]['monthly']['symbols'][i]=['NEW'+s for s in names]
        refresh_original(args)
        with patch.object(k,'_solve',side_effect=AssertionError('no new solve')):b=fit(args)
        self.assertEqual(b['audit']['counts']['no_qualified_query_years'],2)
        self.assertEqual(b['audit']['counts']['qualified_security_columns'],12)
        for role in ('diagnostic','candidate'):self.assertEqual(b[role]['raw']['local_linear'].tobytes(),b['control']['raw']['source_log'].tobytes())
    def test_control_gate_before_design_or_new_fit(self):
        with patch.object(k,'identity_design',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'gate failed'):fit(self.args,control_validator=lambda c:{'passed':False})
    def test_gate_receipt_preserved_and_payload_isolated(self):
        receipt={'passed':True,'identity':'independent fake'}
        def gate(control):
            control['raw']['source_log'][:]=1e5
            return receipt
        b=fit(self.args,control_validator=gate)
        self.assertEqual(b['control_gate'],receipt)
        self.assertEqual(b['control']['raw']['source_log'].tobytes(),self.full['control']['raw']['source_log'].tobytes())
    def test_corrupt_original_cache_fails_before_new_fit(self):
        args=clone(self.args);args[3]['raw_local_linear'][1,0]+=.01
        with patch.object(k,'new_forecasts',side_effect=AssertionError('too early')):
            with self.assertRaises(AssertionError):fit(args)
    def test_input_immutability(self):
        _,train,meta,a=self.args
        snap=k.bridge._snapshot(a);tr={x:(v.dtype.str,v.shape,v.tobytes()) for x,v in train.items()};mt=deepcopy(meta)
        fit(self.args)
        self.assertEqual(snap,k.bridge._snapshot(a));self.assertEqual(tr,{x:(v.dtype.str,v.shape,v.tobytes()) for x,v in train.items()});self.assertEqual(mt,meta)
    def test_future_numeric_object_poison_not_decoded(self):
        args=clone(self.args);future=args[1]['ends']>np.datetime64('2015-12-31')
        for key in ('train_plain',)+tuple('target_'+x for x in k.TARGETS):
            args[1][key]=args[1][key].astype(object);args[1][key][future]='unread future'
        args[1]['event_plain']=args[1]['event_plain'].astype(object);args[1]['event_plain'][12:]='unread future'
        b=fit(args,2)
        for role in ('control','diagnostic','candidate'):
            for key in k.RAW_KEYS:np.testing.assert_array_equal(b[role]['raw'][key],self.full[role]['raw'][key][:12])
    def test_n0_no_numeric_or_new_design(self):
        args=clone(self.args)
        for key in ('train_plain','event_plain')+tuple('target_'+x for x in k.TARGETS):
            args[1][key]=args[1][key].astype(object);args[1][key][:]='unread'
        with patch.object(k,'identity_design',side_effect=AssertionError('no design')):b=fit(args,0)
        self.assertEqual(b['audit']['counts']['actual_regression_solves'],0);self.assertEqual(b['audit']['counts']['annual_design_builds'],0)
    def test_full_logical_physical_model_prefixes(self):
        for n in range(5):
            left=fit(self.args,n);right=fit(physical(self.args,n))
            self.assertEqual(left['audit'],right['audit'])
            for role in ('control','diagnostic','candidate'):
                self.assertEqual(left[role]['audit'],right[role]['audit'])
                for key in k.RAW_KEYS:
                    self.assertEqual(left[role]['raw'][key].tobytes(),right[role]['raw'][key].tobytes())
                    self.assertEqual(left[role]['raw'][key].tobytes(),self.full[role]['raw'][key][:n*6].tobytes())
    def test_malformed_metadata_nonfinite_target_and_invalid_prefix(self):
        args=clone(self.args);args[1]['offsets'][1]+=1
        with self.assertRaisesRegex(ValueError,'offsets'):fit(args)
        args=clone(self.args);args[1]['target_log_excess'][0]=np.nan
        with patch.object(k,'_control_forecasts',side_effect=AssertionError('no fit')):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):fit(args)
        for n in (-1,5,True):
            with self.assertRaises(ValueError):fit(self.args,n)

class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.args=fixture();cls.bundle=fit(cls.args);cls.result=decide(cls.args,cls.bundle)
    def test_original_scores_pre_orders_exact(self):
        study,train,meta,a=self.args;m=meta['monthly']
        choices,post=study.choices(study.configs()[4],a,m,4,policy=dict(study.POLICY_FIELDS))
        self.assertEqual(self.result[0][k.ARMS[0]],study.rp.make_orders(choices,post,m))
        np.testing.assert_array_equal(self.result[1][k.ARMS[0]],post)
        np.testing.assert_array_equal(self.result[2][k.ARMS[0]],study.modified_pre_score(study.configs()[4],a,4))
    def test_full_weak_and_calm_slot_formula(self):
        b=deepcopy(self.bundle)
        for role in ('diagnostic','candidate'):b[role]['raw']['local_linear'][:]=np.tile(np.arange(6)[::-1],4)
        a=self.args[3];m=self.args[2]['monthly']
        for arm in k.ARMS[1:]:
            r,d,au=k.rank_and_delta(a,m,b,arm=arm)
            coeff=.25*(.05*a['state_weak_narrow'][:,None]+.01*(a['state_calm10']&~a['state_weak_narrow'])[:,None]*a['extension_available'])
            for i,ids in enumerate(au['active_indices']):
                js=np.asarray(ids,int);np.testing.assert_array_equal(d[i,js],coeff[i,js]*(r['new_local'][i,js]-r['old_local'][i,js]))
            self.assertTrue(np.any(d[2]!=0.));self.assertEqual(au['coefficient'][2],[.0025]*6)
            self.assertFalse(np.any(d[0]))
    def test_partial_exact_and_zero_delta_preserves_bytes(self):
        for arm in k.ARMS[1:]:
            for pos in (1,2):self.assertEqual(self.result[pos][arm][0].tobytes(),self.result[pos][k.ARMS[0]][0].tobytes())
            self.assertEqual(self.result[0][arm][0],self.result[0][k.ARMS[0]][0])
        b=deepcopy(self.bundle)
        for role in ('diagnostic','candidate'):b[role]=deepcopy(b['control'])
        out=decide(self.args,b)
        for arm in k.ARMS[1:]:
            for pos in (1,2):self.assertEqual(out[pos][arm].tobytes(),out[pos][k.ARMS[0]].tobytes())
    def test_rank_full_original_u_not_local_mask(self):
        b=deepcopy(self.bundle)
        for role in ('diagnostic','candidate'):b[role]['local_available'][:]=False
        for arm in k.ARMS[1:]:
            r,d,au=k.rank_and_delta(self.args[3],self.args[2]['monthly'],b,arm=arm)
            self.assertEqual(au['ranking_cells'],24)
            raw=b[k.ROLES[arm]]['raw']['local_linear'].reshape(4,6)
            for i in range(4):np.testing.assert_array_equal(r['new_local'][i],pd.Series(raw[i]).rank(pct=True).to_numpy()-.5)
    def test_single_episode_each_and_inc_gate_before_new_rank(self):
        study=self.args[0];seen=[]
        with patch.object(study.ep,'select_boundary',wraps=study.ep.select_boundary) as ep:
            decide(self.args,self.bundle,incumbent_validator=lambda *x:(seen.append(ep.call_count) or {'passed':True}))
        self.assertEqual(seen,[1]);self.assertEqual(ep.call_count,3)
        with patch.object(k,'rank_and_delta',side_effect=AssertionError('too early')):
            with self.assertRaisesRegex(ValueError,'gate failed'):decide(self.args,self.bundle,incumbent_validator=lambda *x:{'passed':False})
    def test_expected_incumbent_guard(self):
        with self.assertRaisesRegex(ValueError,'Frozen incumbent'):decide(self.args,self.bundle,expected_incumbent_orders=[])
    def test_exact_decision_schema_counts_no_new_fits(self):
        audit=self.result[3];self.assertEqual(tuple(audit['counts']),k.COUNT_FIELDS)
        self.assertTrue(all(tuple(row)==k.EVENT_KEYS for row in audit['events']))
        for key in ('incumbent_episode_calls','diagnostic_episode_calls','candidate_episode_calls'):self.assertEqual(audit['counts'][key],1)
        self.assertEqual(audit['counts']['new_model_fits'],0);self.assertEqual(audit['counts']['predictive_model_fits'],0)
        json.dumps(audit,allow_nan=False)
    def test_all_cached_logical_physical_prefixes_and_fullbundle_seam(self):
        for n in range(5):
            left=decide(self.args,self.bundle,n);small=physical(self.args,n);right=decide(small,fit(small));seam=decide(small,self.bundle)
            self.assertEqual(left[3],right[3]);self.assertEqual(left[3],seam[3])
            for arm in k.ARMS:
                self.assertEqual(left[0][arm],self.result[0][arm][:n])
                self.assertEqual(left[0][arm],right[0][arm]);self.assertEqual(left[0][arm],seam[0][arm])
                for pos in (1,2):np.testing.assert_array_equal(left[pos][arm],right[pos][arm]);np.testing.assert_array_equal(left[pos][arm],seam[pos][arm])
    def test_future_bundle_numeric_not_consumed(self):
        for n in (0,2):
            b=deepcopy(self.bundle)
            for role in ('control','diagnostic','candidate'):
                for key in k.RAW_KEYS:b[role]['raw'][key][n*6:]=np.inf
                b[role]['available'][n*6:]=False;b[role]['local_available'][n*6:]=True
            out=decide(physical(self.args,n),b)
            for arm in k.ARMS:
                for pos in (1,2):np.testing.assert_array_equal(out[pos][arm],self.result[pos][arm][:n])
    def test_future_metadata_body_never_copied_or_compared(self):
        class Poison:
            def __deepcopy__(self,memo):raise AssertionError('future deepcopy')
            def __eq__(self,other):raise AssertionError('future equality')
            def __str__(self):raise AssertionError('future text')
        for n in (0,2):
            args=clone(self.args)
            for key in ('dates','ends','symbols'):args[2]['monthly'][key][n:]=[Poison()]*(4-n)
            out=decide(args,self.bundle,n)
            for arm in k.ARMS:self.assertEqual(out[0][arm],self.result[0][arm][:n])
    def test_invalid_active_forecast_is_hard_error(self):
        for role in ('diagnostic','candidate'):
            b=deepcopy(self.bundle);b[role]['raw']['local_linear'][6]=np.inf
            with self.assertRaisesRegex(ValueError,'Raw support'):decide(self.args,b)
    def test_source_copy_tamper_is_hard_error(self):
        b=deepcopy(self.bundle);b['candidate']['raw']['source_log'][6]=np.nextafter(b['candidate']['raw']['source_log'][6],np.inf)
        with self.assertRaisesRegex(ValueError,'Preserved raw'):decide(self.args,b)
    def test_decisions_no_fit_and_no_mutation(self):
        b=deepcopy(self.bundle);before=k.bridge._snapshot(self.args[3]);m=deepcopy(self.args[2])
        with patch.object(k,'new_forecasts',side_effect=AssertionError('no fit')),patch.object(k,'_control_forecasts',side_effect=AssertionError('no fit')):decide(self.args,b)
        self.assertEqual(before,k.bridge._snapshot(self.args[3]));self.assertEqual(m,self.args[2])
        for role in ('control','diagnostic','candidate'):
            for key in k.RAW_KEYS:self.assertEqual(b[role]['raw'][key].tobytes(),self.bundle[role]['raw'][key].tobytes())
    def test_two_eligible_half_and_single_eligible_cash(self):
        args=clone(self.args);args[3]['eligible'][1,2:]=False;args[3]['eligible'][2,1:]=False;refresh_ranks(args)
        out=decide(args,self.bundle)
        for arm in k.ARMS:
            self.assertEqual(len(out[0][arm][1]['symbols']),2);self.assertEqual(out[0][arm][2]['symbols'],[])

if __name__=='__main__':unittest.main(verbosity=2)
