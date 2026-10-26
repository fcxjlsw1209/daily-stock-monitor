"""260 independent synthetic sources, teacher, canonical fit and integration."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_leader_utility260_independent as audit

def state_fixture():
    cal=pd.bdate_range('2010-01-01','2020-12-31');t=np.arange(len(cal),dtype=float)
    frame=pd.DataFrame({s:100.*np.exp(.0002*t+.01*np.sin(t/(11+j*3))) for j,s in enumerate(('SPY','XLY','XLP'))},index=cal)
    return frame,cal,{v:i for i,v in enumerate(cal)}

def ridge_fixture(n=30):
    t=np.arange(n,dtype=float)
    X=np.column_stack((np.sin(t*.4),np.cos(t*.7),t/n));d=.03+.2*X[:,0]-.1*X[:,1]+.02*np.sin(t)
    dates=pd.date_range('2016-01-31',periods=n,freq='ME').to_numpy().astype('datetime64[D]')
    return X,d,dates,'2019-12-31',X[-3:].copy()

class PureFitTests(unittest.TestCase):
    def test_independent_svd_canonical_and_actual_calls(self):
        args=ridge_fixture()
        with patch.object(audit.numeric,'least_squares',wraps=audit.numeric.least_squares) as svd,patch.object(audit.np.linalg,'solve',wraps=audit.np.linalg.solve) as normal:
            fit=audit.ridge_oracle(*args)
        self.assertEqual(svd.call_count,fit['svd_solves']);self.assertEqual(normal.call_count,fit['canonical_solves'])
        self.assertAlmostEqual(fit['weight_sum'],len(args[1]),places=13)
        self.assertGreater(np.linalg.eigvalsh(fit['gram']).min(),0.)

    def test_17_months_and_constant_target_no_solve(self):
        for args in (ridge_fixture(17),list(ridge_fixture())):
            if isinstance(args,list):args[1][:]=.05
            with patch.object(audit.np.linalg,'solve',side_effect=AssertionError('cold solve')):
                out=audit.ridge_oracle(*args)
            self.assertIsNone(out['beta']);self.assertEqual(out['canonical_solves']+out['svd_solves'],0)

    def test_18_months_and_collinear_or_constant_state_columns(self):
        args=list(ridge_fixture(18));args[0][:,2]=args[0][:,0];args[4]=args[0][-3:].copy()
        self.assertEqual(audit.ridge_oracle(*args)['canonical_solves'],1)
        args[0][:,2]=3.;args[4]=args[0][-3:].copy();out=audit.ridge_oracle(*args)
        self.assertEqual(out['state_scale'][2],1.)

    def test_target_sign_and_units(self):
        args=list(ridge_fixture());a=audit.ridge_oracle(*args);args[1]=-args[1];b=audit.ridge_oracle(*args)
        np.testing.assert_allclose(a['prediction'],-b['prediction'],atol=2e-15,rtol=0)
        self.assertEqual(a['target_sd'],b['target_sd'])

    def test_duplicate_month_and_invalid_numeric_rejected(self):
        args=list(ridge_fixture());args[2][1]=args[2][0]
        with self.assertRaises(ValueError):audit.ridge_oracle(*args)
        for x in (np.nan,np.inf,True,'bad'):
            args=list(ridge_fixture());args[0]=args[0].astype(object);args[0][0,0]=x
            with self.subTest(value=x),self.assertRaises(ValueError):audit.ridge_oracle(*args)

class MarketMathTests(unittest.TestCase):
    def test_offsets_relative_gross_and_sample_vol(self):
        adj,cal,pos=state_fixture();t=600;out=audit._market_state(adj,cal,pos,cal[t]);spy=adj.SPY.to_numpy()
        want=[np.std(spy[t-62:t+1]/spy[t-63:t]-1,ddof=1)*np.sqrt(252),spy[t-42]/spy[t-252]-1,
          (adj.XLY.iloc[t]/adj.XLY.iloc[t-126])/(adj.XLP.iloc[t]/adj.XLP.iloc[t-126])-1]
        np.testing.assert_array_equal(out,want)

    def test_missing_interior_row_is_not_compressed_or_filled(self):
        adj,cal,pos=state_fixture();adj=adj.drop(cal[590])
        with self.assertRaisesRegex(ValueError,'Missing official'):audit._market_state(adj,cal,pos,cal[600])

    def test_every_required_price_positive_finite_and_real(self):
        for x in (0.,-1.,np.nan,np.inf,True,'bad'):
            adj,cal,pos=state_fixture();adj=adj.astype(object);adj.loc[cal[400],'SPY']=x
            with self.subTest(value=x),self.assertRaises(ValueError):audit._market_state(adj,cal,pos,cal[600])

def teacher_fixture(months=36):
    width=6;dates=pd.date_range('2013-01-31',periods=months,freq='ME');names=['ALPHA','BETA','GAMMA','DELTA','EPSILON','ZETA']
    rows=months*width;plain=np.zeros((rows,10));plain[:,0]=np.tile([.5,.4,.3,.2,.1,0.],months);plain[:,1]=np.tile([0.,.1,.2,.3,.4,.5],months)
    y=np.sin(np.arange(rows)*.3)*.1
    return dict(schema='monthly_leader_utility260_inputs_v1',n=3,width=width,training_row_ids=np.arange(rows,dtype=np.int64),
      dates=np.repeat(dates.to_numpy().astype('datetime64[D]'),width),ends=np.repeat((dates+pd.offsets.MonthEnd()).to_numpy().astype('datetime64[D]'),width),
      symbols=np.tile(names,months),offsets=np.arange(months+1,dtype=np.int64)*width,plain=plain,eligible=np.ones(rows,bool),common_good=np.ones(rows,bool),labels=y,
      event_dates=np.asarray(['2016-08-17','2016-08-31','2017-01-31'],dtype='datetime64[D]'),event_symbols=[names.copy() for _ in range(3)],event_plain=np.tile(plain[:width],(3,1)))

def raw_fixture():
    import test_monthly_asset_target255_independent as old
    train,meta,arrays=old.fixture();train['train_eligible']=np.ones(len(train['dates']),bool)
    return old.study(),train,meta,arrays

class TeacherTests(unittest.TestCase):
    def test_bad_selected_label_no_replacement_all_union_checked(self):
        i=teacher_fixture(1);i['labels'][0]=np.nan
        out=audit.targets_oracle(i);r=out['records'][0]
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'selected_label_unavailable');self.assertEqual(out['counts']['selected_label_observations'],6)
        self.assertEqual(r['raw_choices'],['ALPHA','BETA','GAMMA']);self.assertTrue(np.isnan(out['values'][0]))
    def test_same_set_requires_labels_two_names_cash_and_zero(self):
        for n in (0,1,2,3):
            i=teacher_fixture(1);i['eligible'][n:]=False;i['plain'][:,1]=i['plain'][:,0]
            out=audit.targets_oracle(i);self.assertEqual(out['values'][0],0.)
            if n<2:
                i['labels']=np.full(6,'not read',object);self.assertEqual(audit.targets_oracle(i)['values'][0],0.)
            else:
                i['common_good'][0]=False;self.assertFalse(audit.targets_oracle(i)['valid'][0])
    def test_unselected_label_ignored_finite10_before_selection(self):
        i=teacher_fixture(1);i['plain'][:,1]=i['plain'][:,0];i['labels']=i['labels'].astype(object);i['labels'][5]='not selected'
        self.assertTrue(audit.targets_oracle(i)['valid'][0]);i['plain'][0,9]=np.nan
        self.assertEqual(audit.targets_oracle(i)['records'][0]['raw_choices'],['BETA','GAMMA','DELTA'])
    def test_lexical_symmetric_difference_and_clip_not_reconstructed(self):
        i=teacher_fixture(1);i['labels']=np.array([.3,-.2,.1,.4,-.1,.2]);out=audit.targets_oracle(i)
        want=(np.sum(i['labels'][[0,1,2]])-np.sum(i['labels'][[3,4,5]]))/3
        self.assertEqual(out['values'][0].hex(),float(want).hex());i['labels']+=.125
        self.assertAlmostEqual(audit.targets_oracle(i)['values'][0],out['values'][0])
    def test_tied_ranks_lexical_not_row_order_and_bad_selected_type(self):
        i=teacher_fixture(1);i['plain'][:,:2]=0.
        self.assertEqual(audit.targets_oracle(i)['records'][0]['raw_choices'],['ALPHA','BETA','DELTA'])
        for value in (True,'bad'):
            j=copy.deepcopy(i);j['labels']=j['labels'].astype(object);j['labels'][0]=value
            with self.assertRaises(ValueError):audit.targets_oracle(j)
    def test_complete_target_audit_mutations_rejected(self):
        i=teacher_fixture();b=audit.targets_oracle(i);self.assertTrue(audit.verify_targets(i,b)['passed'])
        for fault in ('choice','count','value'):
            t=copy.deepcopy(b)
            if fault=='choice':t['records'][0]['raw_choices'].reverse()
            if fault=='count':t['counts']['selected_label_observations']-=1
            if fault=='value':t['values'][0]=np.nextafter(t['values'][0],np.inf)
            with self.assertRaises(ValueError):audit.verify_targets(i,t)

class SourceAndInputTests(unittest.TestCase):
    def test_input_physical_cutoff_and_unused_future_poison(self):
        _,t,m,a=raw_fixture();wanted=audit.router_inputs(t,m,a,2);future=t['ends']>np.datetime64('2015-12-31')
        t['train_plain']=t['train_plain'].astype(object);t['train_plain'][future]='future'
        t['target_simple_excess']=t['target_simple_excess'].astype(object);t['target_simple_excess'][future]='future'
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]='future'
        class Poison:
            def __iter__(self):raise AssertionError('future identity body')
            def __deepcopy__(self,memo):raise AssertionError('future copy')
        for k,date in enumerate(m['training']['ends']):
            if date>'2015-12-31':m['training']['symbols'][k]=Poison()
        m['monthly']['symbols'][2:]=[Poison(),Poison()]
        actual=audit.router_inputs(t,m,a,2)
        for key in ('dates','ends','plain','eligible','common_good'):audit.exact_array(actual[key],wanted[key],key)
        np.testing.assert_array_equal(actual['event_plain'].astype(float),wanted['event_plain'])
        np.testing.assert_array_equal(actual['labels'].astype(float),wanted['labels'])
        self.assertEqual(actual['event_dates'].shape,(2,));self.assertEqual(len(actual['event_symbols']),2)
    def test_n0_partial_only_no_train_bodies(self):
        _,t,m,a=raw_fixture();t['train_plain']=np.full(t['train_plain'].shape,'never',object);t['target_simple_excess']=np.full(len(t['dates']),'never',object)
        for n in (0,1):
            inp=audit.router_inputs(t,m,a,n);self.assertEqual(len(inp['dates']),0);self.assertEqual(audit.targets_oracle(inp)['counts']['months'],0)
    def test_input_identity_commonmask_and_original_rows_checked(self):
        _,t,m,a=raw_fixture();inp=audit.router_inputs(t,m,a)
        self.assertTrue(audit.verify_inputs(t,m,a,inp)['passed'])
        for key in ('training_row_ids','eligible','common_good','labels'):
            b=copy.deepcopy(inp);b[key][0]=not b[key][0] if b[key].dtype==bool else b[key][0]+1
            with self.assertRaises(ValueError):audit.verify_inputs(t,m,a,b)
    def test_market_full_audit_partial_and_required_days(self):
        adj,cal,_=state_fixture();td=['2014-05-30'];qd=['2016-08-17','2016-08-31']
        x,q,au=audit.market_oracle(adj,cal,td,qd);proof=audit.verify_market_inputs(adj,cal,td,qd,x,q,au)
        self.assertEqual(proof['counts']['price_observations'],1014);self.assertTrue((q[0]==0).all())
        adj=adj.drop(pd.Timestamp('2016-08-17'));x,q,au=audit.market_oracle(adj,cal,[],['2016-08-17']);self.assertEqual(au['counts']['state_vectors'],0)
        with self.assertRaises(ValueError):audit.market_oracle(adj,cal,[],['2016-08-31'])
    def test_strict_dates_duplicates_nonofficial_and_source_tamper(self):
        for dates in (['2016-8-1'],['2016-08-01T12:00:00'],[np.datetime64('2016-08-01T00:00:01')]):
            with self.assertRaises(ValueError):audit._days(dates,'strict')
        adj,cal,_=state_fixture()
        with self.assertRaises(ValueError):audit.market_oracle(pd.concat([adj.iloc[:1],adj]),cal,[],[])
        x,q,au=audit.market_oracle(adj,cal,[],['2016-08-31']);q[0,1]=np.nextafter(q[0,1],np.inf)
        with self.assertRaises(ValueError):audit.verify_market_inputs(adj,cal,[],['2016-08-31'],x,q,au)

class RouterAndSelectionTests(unittest.TestCase):
    def fixture(self):
        i=teacher_fixture(48);t=audit.targets_oracle(i);X=np.c_[np.sin(np.arange(48)*.3),np.cos(np.arange(48)*.2),np.arange(48)/48]
        dates=np.asarray(['2016-08-17','2016-08-31','2017-01-31'],dtype='datetime64[D]');Q=np.array([[0.,0.,0.],[.2,.1,.4],[.3,.1,.5]])
        b,_=audit.router_oracle(t,X,Q,dates);return t,X,Q,dates,b
    def test_complete_annual_canonical_svd_proofs_counts_and_tamper(self):
        t,X,Q,d,b=self.fixture();r=audit.verify_models(t,X,Q,d,b)
        self.assertEqual(r['conditional_ridge_solves'],2);self.assertEqual(r['independent_svd_solves'],2);self.assertEqual(r['old_model_solves'],0)
        for fault in ('weight','p','beta','cutoff','count'):
            c=copy.deepcopy(b)
            if fault=='weight':c['audit']['annual'][0]['weights'][0]+=1e-12
            if fault=='p':c['state_p'][1]=np.nextafter(c['state_p'][1],np.inf)
            if fault=='beta':c['audit']['annual'][0]['beta'][0]+=1e-12
            if fault=='cutoff':c['audit']['annual'][0]['cutoff']='2016-01-01'
            if fault=='count':c['audit']['counts']['ridge_solves']-=1
            with self.assertRaises(ValueError):audit.verify_models(t,X,Q,d,c)
    def test_annual_exact_boundaries_cold_target_sd_and_unavailable_months(self):
        t,X,Q,d,b=self.fixture();r=b['audit']['annual'][0];ids=r['training_indices']
        self.assertTrue(all(t['ends'][j]<=np.datetime64(r['cutoff']) and t['dates'][j]>np.datetime64(r['lower_bound']) for j in ids))
        t['valid'][:]=False;t['valid'][:17]=True;t['values'][~t['valid']]=np.inf
        c,work=audit.router_oracle(t,X,Q,d);self.assertFalse(c['active'].any());self.assertEqual(work['independent_svd_solves'],0)
        t['valid'][:]=True;t['values'][:]=0.;c,_=audit.router_oracle(t,X,Q,d);self.assertFalse(c['active'].any());self.assertEqual(c['audit']['annual'][0]['status'],'constant_target')
    def test_n0_and_partial_models_no_solve(self):
        t,X,Q,d,b=self.fixture()
        for n in (0,1):
            with patch.object(np.linalg,'solve',side_effect=AssertionError('fit')):
                c,work=audit.router_oracle(t,X,Q[:n],d[:n]);self.assertEqual(c['audit']['counts']['years'],0);self.assertEqual(work['independent_canonical_solves'],0)
    def test_failed_original_gate_before_any_new_moments_or_solve(self):
        t,X,Q,d,b=self.fixture();b['control_gate']['passed']=False
        with patch.object(audit,'ridge_oracle',side_effect=AssertionError('premature router')):
            with self.assertRaises(ValueError):audit.verify_models(t,X,Q,d,b)
    def test_selection_full_core_slot_equal_rank_not_cold_and_original_pres(self):
        _,tr,m,a=raw_fixture();dates=np.asarray(m['monthly']['dates'],dtype='datetime64[D]');n=len(dates)
        b=dict(schema='monthly_leader_utility260_router_v1',n=n,event_dates=dates,active=np.array([False,True,True,False]),
          intercept_prediction=np.array([0.,.1,.1,0.]),state_prediction=np.array([0.,.2,.2,0.]),intercept_p=np.array([.5,.55,.55,.5]),state_p=np.array([.5,.65,.65,.5]))
        q=tr['event_plain'].copy();q[6:12,1]=q[6:12,0]
        s,ch,p,au=audit.selection_oracle(a,m['monthly'],b,q);base=audit.original_base(a,m['monthly'],n)
        want=.1*(q[6:12,0]-base['core'][1]/base['cw'][1]);audit.exact_array(p[audit.ARMS[1]][1],base['P'][1]+want,'Equal blend still replaces incumbentcore')
        for arm in audit.ARMS:audit.exact_array(p[arm][[0,3]],base['P'][[0,3]],'Partial/cold exact')
        o=audit.orders_from_choices(s,ch,m['monthly']);proof=audit.verify_selection(a,m['monthly'],b,q,o,scores=s,pre_scores=p,audit=au)
        self.assertEqual(proof['original_episode_calls'],3);self.assertEqual(proof['ordered_choices_checked'],3*n)
    def test_selection_prefix_future_forecasts_poison_and_n0(self):
        _,tr,m,a=raw_fixture();dates=np.asarray(m['monthly']['dates'],dtype='datetime64[D]');n=len(dates)
        b=dict(schema='monthly_leader_utility260_router_v1',n=n,event_dates=dates,active=np.zeros(n,bool),intercept_prediction=np.zeros(n),state_prediction=np.zeros(n),intercept_p=np.full(n,.5),state_p=np.full(n,.5))
        for key in ('intercept_prediction','state_prediction','intercept_p','state_p'):b[key]=b[key].astype(object);b[key][2:]='future'
        q=tr['event_plain'].astype(object);q[12:]='future'
        for count in (0,1,2):
            s,ch,p,au=audit.selection_oracle(a,m['monthly'],b,q,count);self.assertEqual(au['counts']['events'],count)
            for arm in audit.ARMS:audit.exact_array(p[arm],audit.original_base(a,m['monthly'],count)['P'],'Cold source preserved')

class BilateralCross(unittest.TestCase):
    def test_source_model_selection_host_gates_and_prefixes(self):
        # Producer is imported only here, as the subject under test, never by the oracle.
        import monthly_leader_utility260_kernel as kernel
        import monthly_leader_utility260_host as host
        import test_monthly_leader_utility260_kernel as producer_fixture
        args=producer_fixture.fixture();study,train,meta,arrays,adj,calendar=args;n=len(meta['monthly']['dates'])
        def expected(a,m,k):
            base=audit.original_base(a,m,k);F,C=audit.accounts.episode(base['P'],base)
            rows=[dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],symbols=[m['symbols'][i][j] for j in ids],scores=[float(F[i,j]) for j in ids]) for i,ids in enumerate(C)]
            return rows,F,base['P']
        original=expected(arrays,meta['monthly'],n)
        full,artifacts,result,proof=host.prepare_one(*args,n,*original,kernel=kernel,ind=audit)
        self.assertTrue(proof['models']['passed']);self.assertEqual(proof['models']['conditional_ridge_solves'],full['audit']['counts']['ridge_solves'])
        self.assertEqual(proof['models']['conditional_ridge_solves'],proof['models']['independent_svd_solves'])
        for count in sorted({0,1,max(1,n-1),n}):
            part=producer_fixture.physical(args,count);ss,tt,mm,aa,pp,cc=part;old=expected(aa,mm['monthly'],count)
            b,parts,out,pr=host.prepare_one(*part,count,*old,kernel=kernel,ind=audit)
            host.bundle_parity(full,b,count)
            cached,cp=host.decision_pass(ss,aa,mm['monthly'],full,artifacts['inputs']['event_plain'],count,*old,kernel=kernel,ind=audit)
            audit.record_check(cached[0],out[0],'Physical vs cached ordered choices')
            for arm in audit.ARMS:
                audit.exact_array(cached[1][arm],out[1][arm],'Physical/cached F');audit.exact_array(cached[2][arm],out[2][arm],'Physical/cached P')
            self.assertEqual(cp['selection']['original_episode_calls'],3)
            self.assertEqual(pr['early_incumbent']['original_episode_calls'],1)
        count=min(2,n);part=producer_fixture.physical(args,count);ss,tt,mm,aa,pp,cc=part;old=expected(aa,mm['monthly'],count)
        poisoned=copy.deepcopy(full)
        for key in ('intercept_prediction','state_prediction','intercept_p','state_p'):
            poisoned[key]=poisoned[key].astype(object);poisoned[key][count:]='unconsumed future'
        q=artifacts['inputs']['event_plain'].astype(object);q[count*aa['baseline'].shape[1]:]='unconsumed future'
        host.decision_pass(ss,aa,mm['monthly'],poisoned,q,count,*old,kernel=kernel,ind=audit)
        bad=copy.deepcopy(original[0])
        if n:bad[0]['symbols']=[]
        with patch.object(kernel,'build_expert_targets',side_effect=AssertionError('Teacher before original gate')):
            with self.assertRaises(ValueError):host.prepare_one(*args,n,bad,original[1],original[2],kernel=kernel,ind=audit)

if __name__=='__main__':unittest.main(verbosity=2)
