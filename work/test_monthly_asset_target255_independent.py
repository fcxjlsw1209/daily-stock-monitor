"""255 synthetic-only labels/models/selection and sole bilateral integration."""
import copy
from functools import lru_cache
import hashlib
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_asset_target255_independent as audit

@lru_cache(None)
def study():
    path=audit.ROOT/'outputs/monthly_local_linear70/study.py'
    audit.require(hashlib.sha256(path.read_bytes()).hexdigest()=='e8cfe0c27a6805cbee4b59d24ce9771ece7caa01c29f1a30aa2d5acf9837f3fb','Frozen study fixture')
    spec=importlib.util.spec_from_file_location('asset255_self_synthetic_study',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def original_model(inp,targets):
    return study().model.annual_forecasts(inp['dates'],inp['ends'],inp['symbols'],inp['X'],targets,inp['good'],
      inp['query_dates'],inp['query_symbols'],inp['query_X'],inp['query_good'])

@lru_cache(None)
def _fixture():
    names=['ALPHA','BETA','GAMMA','DELTA','ETF_ONE','ETF_TWO'];w=len(names)
    months=pd.date_range('2011-01-31','2019-12-31',freq='ME');size=len(months)*w
    rng=np.random.default_rng(8255);X=rng.uniform(-.49,.49,(size,10));dates=np.repeat(months.to_numpy().astype('datetime64[D]'),w)
    ends=np.repeat((months+pd.offsets.MonthEnd()).to_numpy().astype('datetime64[D]'),w)
    y=.31*X[:,1]-.19*X[:,7]+.08*X[:,8]**2+np.tile([.15,-.11,.03,.07,-.09,-.05],len(months))
    y=(y.reshape(-1,w)-np.mean(y.reshape(-1,w),axis=1,keepdims=True)).ravel()
    good=np.ones(size,bool);good[13]=False;y[13]=np.nan
    q=['2016-08-17','2016-08-31','2017-01-31','2018-03-30'];qe=['2016-08-31','2016-09-30','2017-02-28','2018-04-30'];n=len(q)
    meta=dict(monthly=dict(dates=q,ends=qe,symbols=[names.copy() for _ in q],etfs=names[-2:]),
      training=dict(dates=months.strftime('%Y-%m-%d').tolist(),ends=(months+pd.offsets.MonthEnd()).strftime('%Y-%m-%d').tolist(),symbols=[names.copy() for _ in months]))
    train=dict(dates=dates,ends=ends,train_plain=X,event_plain=rng.uniform(-.49,.49,(n*w,10)),event_dates=np.repeat(np.asarray(q,dtype='datetime64[D]'),w),
      common_good=good,offsets=np.arange(len(months)+1,dtype=np.int64)*w,**{'target_'+k:y*(1+.2*i) for i,k in enumerate(audit.TARGETS)})
    shape=(n,w);calm=np.array([False,True,False,False]);weak=np.array([True,False,True,True]);cw=np.where(calm,.5,.52)[:,None]
    core=cw*np.tile(np.linspace(-.2,.2,w),(n,1));z=np.zeros(shape);b=np.ones(shape,bool)
    arrays=dict(baseline=core+.04,core=core,core_weight=cw,eligible=b.copy(),effects=np.tile([0.,0.,.15,-.1,.5,-.5],(n,1)),
      state_calm10=calm,state_weak_narrow=weak,pre_score=np.tile([.31,.305,.3,.295,.29,.28],(n,1)),old_ml4=z.copy(),extension_available=b.copy())
    inp=audit.model_inputs(train,meta,arrays);model=original_model(inp,inp['targets'])
    arrays['model_available']=model['available'].reshape(shape).copy();arrays['forecast_available']=arrays['model_available'].copy()
    arrays['local_available']=model['local_available'].reshape(shape).copy()
    for k in audit.RAW_KEYS:arrays['raw_'+k]=model['raw'][k].reshape(shape).copy()
    arrays['raw_log_excess']=arrays['raw_source_log'].copy();arrays['rank_log_excess']=np.vstack([audit._rank(v) for v in arrays['raw_source_log']])
    ranks,ready=study().ranked_forecasts({k:arrays['raw_'+k] for k in ('local_constant','local_linear')},arrays['model_available'],arrays,n)
    arrays.update({'rank_'+k:v for k,v in ranks.items()});arrays['paired_month_ready']=ready
    return train,meta,arrays

def fixture():return copy.deepcopy(_fixture())

def bundle_from_frozen(train,meta,arrays,n=None):
    inp=audit.model_inputs(train,meta,arrays,n);control=original_model(inp,inp['targets'])
    new,la=audit.recenter_labels(inp['dates'],inp['symbols'],inp['targets']['log_excess'],inp['good'],inp['etfs'])
    candidate=original_model(inp,dict(inp['targets'],log_excess=new));n=inp['n']
    parity=dict(passed=True,events=n,original_raw_exact=True,original_availability_exact=True,original_ranks_exact=True,candidate_models_computed=False)
    au=dict(schema='monthly_asset_target255_model_audit_v1',counts=audit.model_counts(inp,la,control,candidate),original_cache_parity=parity,
      original_row_order_preserved=True,original_common_good_preserved=True,all_other_targets_preserved=True,new_labels_feed_pooled_and_local=True,
      inherited_model_audit_target_text='Frozen helper wording unchanged cached centered log_excess describes its argument; candidate argument IS recentered. Wrapper label_audit is authoritative.',
      residuals_are_mature_in_sample=True,residuals_are_oof=False,computes_returns=False)
    return dict(schema='monthly_asset_target255_models_v1',n=n,width=inp['width'],training_row_ids=inp['training_row_ids'],control=control,candidate=candidate,
      label_values=new,label_audit=la,control_gate=parity,audit=au)

@lru_cache(None)
def _bundle():return bundle_from_frozen(*_fixture())
def full_bundle():return copy.deepcopy(_bundle())
def run(a,m,b,n=None):
    s,c,p,au=audit.selection_oracle(a,m,b,n);return audit.orders_from_choices(s,c,m),s,p,au
def verify(a,m,b,out,n=None):
    o,s,p,au=out;return audit.verify_selection(a,m,b,o,scores=s,pre_scores=p,audit=au,n=n)

class LabelTests(unittest.TestCase):
    def basic(self):return np.array(['2020-01-31']*4),np.array(list('ABCD')),np.array([1.,3.,5.,np.nan]),np.array([True,True,True,False]),['C']
    def test_exact_center_singleton_empty_and_invalid(self):
        args=self.basic();y,au=audit.recenter_labels(*args);np.testing.assert_array_equal(y,[-1.,1.,0.,np.nan]);self.assertEqual(y[2].hex(),'0x0.0p+0')
        self.assertEqual(au['counts']['singleton_groups'],1);self.assertEqual(au['groups'][0]['row_ids'],[0,1]);self.assertEqual(au['groups'][1]['row_ids'],[2])
        out=audit.recenter_labels(*args[:4],[])[1];self.assertEqual(out['counts']['empty_groups'],1)
    def test_mean_is_saved_numpy_row_order_not_fsum_or_reclip(self):
        a=list(self.basic());a[2]=np.array([1e16,1.,-1e16,np.nan]);a[4]=[]
        y,au=audit.recenter_labels(*a);self.assertEqual(au['groups'][0]['mean'],0.);self.assertEqual(y[1],1.)
    def test_invalid_payload_bytes_are_untouched(self):
        a=list(self.basic());a[2]=np.array([1.,3.,-0.,np.inf]);a[3][2:]=False
        y,_=audit.recenter_labels(*a);self.assertEqual(y[2:].tobytes(),a[2][2:].tobytes())
    def test_nonfinite_good_duplicate_dates_bad_mask_rejected(self):
        for fault in ('nan','inf','duplicate','mask','date','etfs'):
            a=list(self.basic())
            if fault in ('nan','inf'):a[2][0]=float(fault)
            if fault=='duplicate':a[1][1]='A'
            if fault=='mask':a[3]=a[3].astype(int)
            if fault=='date':a[0][0]='bad'
            if fault=='etfs':a[4]=['C','C']
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.recenter_labels(*a)
    def test_all_equal_valid_is_zero_and_empty_is_empty(self):
        a=list(self.basic());a[2]=np.ones(4);a[3][:]=True;y,_=audit.recenter_labels(*a);np.testing.assert_array_equal(y,np.zeros(4))
        y,au=audit.recenter_labels(np.array([],dtype='datetime64[D]'),[],np.array([],float),np.array([],bool),[]);self.assertEqual(len(y),0);self.assertEqual(au['groups'],[])

class ModelTests(unittest.TestCase):
    def test_independent_svd_against_frozen_model_full_audit_counts(self):
        t,m,a=fixture();b=full_bundle();r=audit.verify_models(t,m,a,b)
        self.assertTrue(all(r[k] for k in audit.MODEL_FLAGS));self.assertGreater(r['counts']['actual_regression_solves'],0)
        self.assertEqual(r['counts'],b['audit']['counts']);self.assertEqual(r['rank_tolerance'],0.)
        own=r['independent_solve_counts'];self.assertEqual(own['total_regression_solves'],own['control']['total_regression_solves']+own['candidate']['total_regression_solves'])
        years=own['control']['pooled_log_solves']+own['candidate']['pooled_log_solves']
        self.assertEqual(r['counts']['actual_regression_solves']-own['total_regression_solves'],4*years)
        self.assertEqual(r['producer_equivalent_counts'],r['counts'])
    def test_control_gate_no_candidate_work_and_nextafter_cache_rejected(self):
        t,m,a=fixture();b=full_bundle()
        with patch.object(audit,'recenter_labels',side_effect=AssertionError('premature label')):r=audit.verify_control(t,m,a,b['control'])
        self.assertFalse(r['candidate_models_computed']);self.assertFalse(r['candidate_labels_computed']);self.assertEqual(r['episode_calls'],0)
        self.assertEqual(r['producer_equivalent_fit_counts']['actual_regression_solves']-r['independent_solve_counts']['total_regression_solves'],4*r['independent_solve_counts']['pooled_log_solves'])
        a['raw_source_log'][0,0]=np.nextafter(a['raw_source_log'][0,0],np.inf)
        with self.assertRaises(ValueError):audit.verify_control(t,m,a,b['control'])
    def test_annual_cutoff_five_year_and_original_all_four_target_support(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);z=audit.annual_oracle(inp,inp['targets'])
        for row in z['audit']['annual']:
            self.assertGreater(row['first_training_signal'],row['lookback_exclusive_start']);self.assertLessEqual(row['latest_label_end'],row['cutoff'])
        targets=copy.deepcopy(inp['targets']);targets['loss_utility'][:12]=np.nan
        other=audit.annual_oracle(inp,targets);self.assertLess(other['audit']['annual'][0]['training_rows'],z['audit']['annual'][0]['training_rows'])
    def test_new_group_labels_feed_both_pooled_and_local(self):
        b=full_bundle();mask=b['candidate']['available']
        for key in ('source_log','local_linear'):self.assertFalse(np.array_equal(b['candidate']['raw'][key][mask],b['control']['raw'][key][mask]))
        self.assertTrue(b['audit']['new_labels_feed_pooled_and_local']);self.assertEqual(b['label_audit']['counts']['invalid_rows'],1)
    def test_pooled_ready_no_own_history_is_evaluation_not_solve(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a);inp['query_symbols'][:]='NEW_ID'
        old=audit.annual_oracle(inp,inp['targets']);y,la=audit.recenter_labels(inp['dates'],inp['symbols'],inp['targets']['log_excess'],inp['good'],inp['etfs'])
        new=audit.annual_oracle(inp,dict(inp['targets'],log_excess=y));c=audit.model_counts(inp,la,old,new)
        self.assertEqual(c['control_local_evaluations'],24);self.assertEqual(c['candidate_local_evaluations'],24)
        self.assertEqual(c['control_local_solves'],0);self.assertFalse(old['local_available'].any())
        audit.exact_array(old['raw']['local_linear'],old['raw']['source_log'],'No own exact source')
    def test_future_numeric_poison_is_not_consumed(self):
        t,m,a=fixture();inp=audit.model_inputs(t,m,a,2);expected=audit.annual_oracle(inp,inp['targets']);future=t['ends']>np.datetime64('2015-12-31')
        t['train_plain']=t['train_plain'].astype(object);t['train_plain'][future]='future poison'
        for k in audit.TARGETS:t['target_'+k]=t['target_'+k].astype(object);t['target_'+k][future]='future poison'
        t['event_plain']=t['event_plain'].astype(object);t['event_plain'][12:]='future query poison'
        inp=audit.model_inputs(t,m,a,2);actual=audit.annual_oracle(inp,inp['targets']);audit.check_forecast(actual,expected,'Future ignored')
    def test_n0_has_zero_fit_and_empty_label_cohorts(self):
        t,m,a=fixture();b=bundle_from_frozen(t,m,a,0);r=audit.verify_models(t,m,a,b,0)
        self.assertTrue(all(v==0 for v in r['counts'].values()));self.assertEqual(b['candidate']['audit']['annual'],[])
    def test_raw_labels_audits_counts_and_row_identity_corruption_fail(self):
        t,m,a=fixture()
        for fault in ('raw','label','group','rows','count','cutoff','gate','availability'):
            b=full_bundle()
            if fault=='raw':b['candidate']['raw']['local_linear'][0]+=.1
            if fault=='label':b['label_values'][0]+=.1
            if fault=='group':b['label_audit']['groups'][0]['row_ids'].reverse()
            if fault=='rows':b['training_row_ids'][0]=1
            if fault=='count':b['audit']['counts']['actual_regression_solves']-=1
            if fault=='cutoff':b['candidate']['audit']['annual'][0]['cutoff']='2099-12-31'
            if fault=='gate':b['control_gate']['passed']=False
            if fault=='availability':b['candidate']['available'][0]=False
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.verify_models(t,m,a,b)
    def test_invalid_shapes_original_bool_and_duplicate_identity_fail(self):
        for fault in ('shape','mask','duplicate','query','offset'):
            t,m,a=fixture()
            if fault=='shape':t['train_plain']=t['train_plain'][:,:9]
            if fault=='mask':t['common_good']=t['common_good'].astype(int)
            if fault=='duplicate':m['training']['symbols'][0][1]=m['training']['symbols'][0][0]
            if fault=='query':t['event_dates'][0]=np.datetime64('2020-01-31')
            if fault=='offset':t['offsets'][1]=0
            with self.subTest(fault=fault),self.assertRaises(ValueError):audit.model_inputs(t,m,a)

class SelectionTests(unittest.TestCase):
    def test_complete_two_arm_scores_orders_budget_and_audit(self):
        _,m,a=fixture();b=full_bundle();out=run(a,m['monthly'],b);r=verify(a,m['monthly'],b,out)
        self.assertTrue(all(r[k] for k in audit.SELECTION_FLAGS));self.assertEqual(r['ordered_choices_checked'],8)
        self.assertEqual(tuple(out[3]['counts']),audit.COUNT_FIELDS);self.assertEqual(tuple(out[3]['events'][0]),audit.EVENT_KEYS)
        self.assertEqual(out[3]['counts']['new_model_fits'],0);self.assertEqual(out[3]['counts']['incumbent_episode_calls'],1)
    def test_weak_only_partial_and_calm_extension_exact(self):
        _,m,a=fixture();out=run(a,m['monthly'],full_bundle())
        for i in (0,1):
            self.assertEqual(out[0][audit.ARMS[0]][i],out[0][audit.ARMS[1]][i]);self.assertEqual(out[1][audit.ARMS[0]][i].tobytes(),out[1][audit.ARMS[1]][i].tobytes())
            self.assertEqual(out[2][audit.ARMS[0]][i].tobytes(),out[2][audit.ARMS[1]][i].tobytes())
        self.assertTrue(any(out[3]['events'][i]['changed_delta_indices'] for i in (2,3)))
    def test_exact_hand_delta_including_zero_new_rank(self):
        _,m,a=fixture();b=full_bundle();r,d,U,_,active=audit.rank_oracle(a,m['monthly'],b)
        for i in (2,3):
            wanted=.05*(r['new_pool'][i]-r['old_pool'][i])+.0125*((r['new_local'][i]-r['new_pool'][i])-(r['old_local'][i]-r['old_pool'][i]))
            audit.exact_array(d[i],wanted,'Hand delta');self.assertEqual(active[i],list(range(6)))
    def test_no_change_rank_pair_exact_preserve_prebytes(self):
        _,m,a=fixture();b=full_bundle();b['candidate']=copy.deepcopy(b['control']);out=run(a,m['monthly'],b)
        self.assertEqual(out[0][audit.ARMS[0]],out[0][audit.ARMS[1]])
        self.assertEqual(out[2][audit.ARMS[0]].tobytes(),out[2][audit.ARMS[1]].tobytes())
    def test_no_own_history_filter_and_whole_domain_missing_hardfail(self):
        _,m,a=fixture();b=full_bundle();b['candidate']['local_available'][:]=False
        _,_,U,_,_=audit.rank_oracle(a,m['monthly'],b);self.assertEqual(int(U.sum()),24)
        b['candidate']['raw']['local_linear'][12]=np.nan
        with self.assertRaises(ValueError):audit.rank_oracle(a,m['monthly'],b)
    def test_n0_and_all_logical_physical_prefixes(self):
        _,m,a=fixture();b=full_bundle();full=run(a,m['monthly'],b)
        for n in range(5):
            out=run(a,m['monthly'],b,n);verify(a,m['monthly'],b,out,n)
            mm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m['monthly'].items()};aa={k:v[:n].copy() for k,v in a.items()}
            small=run(aa,mm,b,n)
            for arm in audit.ARMS:
                self.assertEqual(out[0][arm],full[0][arm][:n]);audit.exact_array(out[1][arm],full[1][arm][:n],'Prefix scores');audit.exact_array(small[1][arm],out[1][arm],'Physical')
    def test_future_forecast_poison_is_not_consumed(self):
        _,m,a=fixture();b=full_bundle();expected=run(a,m['monthly'],b,2)
        for v in b['candidate']['raw'].values():v[12:]=np.inf
        actual=run(a,m['monthly'],b,2);self.assertEqual(actual[0],expected[0])
    def test_fullscore_pre_order_and_count_tamper_rejected(self):
        _,m,a=fixture();b=full_bundle()
        for fault in ('score','pre','order','count','audit'):
            o,s,p,au=copy.deepcopy(run(a,m['monthly'],b))
            if fault=='score':s[audit.ARMS[1]][2,5]+=1e-9
            if fault=='pre':p[audit.ARMS[1]][2,5]+=1e-9
            if fault=='order':o[audit.ARMS[1]][2]['symbols'].reverse()
            if fault=='count':au['counts']['changed_delta_cells']+=1
            if fault=='audit':au['events'][2]['delta'][0]+=1e-9
            with self.subTest(fault=fault),self.assertRaises(ValueError):verify(a,m['monthly'],b,(o,s,p,au))
    def test_two_names_cash(self):
        _,m,a=fixture();b=full_bundle();a['eligible'][2]=False;a['eligible'][2,:2]=True;a['eligible'][3]=False;a['eligible'][3,0]=True
        for i in (2,3):
            use=a['eligible'][i];a['rank_log_excess'][i,use]=audit._rank(a['raw_source_log'][i,use]);a['rank_local_linear'][i]=a['rank_log_excess'][i]
            a['rank_local_linear'][i,use]=audit._rank(a['raw_local_linear'][i,use])
        out=run(a,m['monthly'],b);self.assertEqual(len(out[0][audit.ARMS[1]][2]['symbols']),2);self.assertEqual(out[0][audit.ARMS[1]][3]['symbols'],[])

class BilateralCross(unittest.TestCase):
    def test_both_fixtures_full_models_all_prefixes_exact_decisions_and_host_gates(self):
        # This is the sole new-producer import, confined to synthetic integration.
        import monthly_asset_target255_kernel as kernel
        import test_monthly_asset_target255_kernel as producer_tests
        import monthly_asset_target255_host as host
        for filename,digest in (('monthly_asset_target255_kernel.py','340d38228f4dc113a642871ffedce88868c521e506d3c17a718f01f3a8f6718c'),
          ('test_monthly_asset_target255_kernel.py','5b291ca456e98e712a2b3d3be553efd262de41dde45146151586f9e164086b87')):
            self.assertEqual(hashlib.sha256((audit.ROOT/'work'/filename).read_bytes()).hexdigest(),digest)
        t,m,a=fixture()
        for args in ((study(),t,m,a),producer_tests.fixture()):
            s,t,m,a=args;n=len(m['monthly']['dates']);before=copy.deepcopy((t,m,a))
            callback,control_receipts=host.model_callbacks(audit,t,m,a,n)
            b=kernel.fit_models(t,m,a,n,control_validator=callback)
            proof=audit.verify_models(t,m,a,b,n);host.receipt(proof,n,audit.MODEL_FLAGS,'models')
            self.assertEqual(control_receipts['control'],b['control_gate'])
            r,d,ra=audit.rank_and_delta(a,m['monthly'],b,n);pr,pd,pa=kernel.rank_and_delta(a,m['monthly'],b,n)
            for k in r:audit.exact_array(pr[k],r[k],'Bilateral exact rank/'+k)
            audit.exact_array(pd,d,'Bilateral exact delta');audit.record_check(pa,ra,'Bilateral complete rank audit')
            expected=run(a,m['monthly'],b,n)
            callback,inc_receipts=host.incumbent_callback(audit,a,m['monthly'],n)
            out=kernel.decisions(s,a,m['monthly'],b,n,incumbent_validator=callback,expected_incumbent_orders=expected[0][audit.ARMS[0]])
            receipt=verify(a,m['monthly'],b,out,n);host.receipt(receipt,n,audit.SELECTION_FLAGS,'selection')
            self.assertEqual(out[3]['incumbent_gate'],inc_receipts['incumbent'])
            for count in range(n+1):
                # Full saved bundle: prefix selection consumes no future forecasts.
                inc,receipts=host.incumbent_callback(audit,a,m['monthly'],count)
                logical=kernel.decisions(s,a,m['monthly'],b,count,incumbent_validator=inc,expected_incumbent_orders=out[0][audit.ARMS[0]][:count])
                verify(a,m['monthly'],b,logical,count);host.exact_prefix(out,logical,count)
                ss,tt,mm,aa=producer_tests.physical(args,count)
                control,_=host.model_callbacks(audit,tt,mm,aa,count)
                small=kernel.fit_models(tt,mm,aa,count,control_validator=control)
                modelproof=audit.verify_models(tt,mm,aa,small,count);host.receipt(modelproof,count,audit.MODEL_FLAGS,'models');host.bundle_parity(b,small,count)
                inc,_=host.incumbent_callback(audit,aa,mm['monthly'],count)
                physical=kernel.decisions(ss,aa,mm['monthly'],small,count,incumbent_validator=inc,expected_incumbent_orders=out[0][audit.ARMS[0]][:count])
                verify(aa,mm['monthly'],small,physical,count);host.exact_prefix(out,physical,count)
                # Legal physical original arrays with a full saved forecast bundle.
                same=kernel.decisions(ss,aa,mm['monthly'],b,count)
                verify(aa,mm['monthly'],b,same,count);host.exact_prefix(out,same,count)
            poison=copy.deepcopy(b)
            for v in poison['candidate']['raw'].values():v[12:]=np.inf
            got=kernel.decisions(s,a,m['monthly'],poison,2);verify(a,m['monthly'],poison,got,2);host.exact_prefix(out,got,2)
            for k,v in t.items():audit.exact_array(v,before[0][k],'No training mutation/'+k)
            self.assertEqual(m,before[1])
            for k,v in a.items():audit.exact_array(v,before[2][k],'No cache mutation/'+k)

if __name__=='__main__':unittest.main()
