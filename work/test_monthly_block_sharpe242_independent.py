"""Invented paths only: sampling frequency, original scoring and accounting."""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_block_sharpe242_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest()!=audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen synthetic fixture changed')
    spec=importlib.util.spec_from_file_location('block_sharpe242_prior_fixture',audit.PRIOR_TEST)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old_fixture=helper()


def fixture():
    ix=pd.bdate_range('2014-01-01','2017-01-04');t=np.arange(len(ix),dtype=float)
    panel={'adj':pd.DataFrame({'A':100.+.03*t+3.*np.sin(t*.13),
        'B':200.+.02*t+5.*np.cos(t*.23),'FUND':80.+.02*t+4.*np.sin(t*.37)},index=ix)}
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND'])
    return panel,meta,ix.strftime('%Y-%m-%d').tolist()


def physical(args,n):
    p,m,c=copy.deepcopy(args);day=m['dates'][n-1] if n else None
    m={k:(v[:n] if k in ('dates','symbols','ends') else v) for k,v in m.items()}
    p={'adj':p['adj'].loc[:day].copy() if n else p['adj'].iloc[:0].copy()}
    c=c[:c.index(day)+2] if n else []
    return p,m,c


def selection_fixture():
    a,m=old_fixture.fixture();shape=a['eligible'].shape
    available=np.ones(shape,bool)
    block=np.tile(np.array([-4.,-3.,-2.,-1.,-1.,1.,2.,3.,4.]),(shape[0],1))
    daily=block[:,::-1].copy()
    values=dict(available=available,block_sharpe=block,daily_sharpe=daily,
        candidate=audit.percentile(block,available),control=audit.percentile(daily,available))
    return a,m,values


def selection_physical(args,n):
    a,m,v=copy.deepcopy(args);a,m=old_fixture.physical((a,m),n)
    return a,m,{k:x[:n].copy() for k,x in v.items()}


def run_selection(args,n=None):
    s,c,p,r=audit.selection_oracle(*args,n=n)
    return orders(s,c,args[1]),s,p,r


def orders(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


class FeatureTests(unittest.TestCase):
    def test_exact_twelve_block_endpoint_statistics_and_ddof(self):
        p=100.+np.arange(253.)+7.*np.sin(np.arange(253.)*.11)
        result=audit.measure_oracle(p)
        for key,returns in [('daily',[float(p[i+1])/float(p[i])-1. for i in range(252)]),
                            ('block',[float(p[i+21])/float(p[i])-1. for i in range(0,252,21)])]:
            mean=math.fsum(returns)/len(returns)
            square=math.fsum((x-mean)*(x-mean) for x in returns)
            self.assertEqual(result[key+'_return_sum'],math.fsum(returns))
            self.assertEqual(result[key+'_mean'],mean)
            self.assertEqual(result[key+'_centered_square_sum'],square)
            self.assertEqual(result[key+'_sample_vol'],math.sqrt(square/(len(returns)-1)))
            self.assertNotEqual(result[key+'_sample_vol'],math.sqrt(square/len(returns)))
            self.assertEqual(result[key+'_sharpe'],mean/result[key+'_sample_vol'])

    def test_equal_block_endpoints_ignore_internal_noise_but_daily_changes(self):
        end=100.+np.arange(13.)+np.sin(np.arange(13.))
        p=np.interp(np.arange(253),np.arange(0,253,21),end);q=p.copy()
        q[np.arange(253)%21!=0]*=1.1
        left,right=audit.measure_oracle(p),audit.measure_oracle(q)
        self.assertTrue(left['available'] and right['available'])
        for key in ('block_return_sum','block_mean','block_centered_square_sum','block_sample_vol','block_sharpe'):
            self.assertEqual(left[key],right[key])
        self.assertNotEqual(left['daily_sharpe'],right['daily_sharpe'])

    def test_both_zero_and_block_only_zero_are_jointly_unavailable(self):
        for p in (np.ones(253),np.exp2(np.arange(253.))):
            result=audit.measure_oracle(p)
            self.assertTrue(result['source_available']);self.assertFalse(result['available'])
            self.assertEqual(result['reason'],'zero_both_sample_volatility')
            self.assertIsNone(result['block_sharpe']);self.assertIsNone(result['daily_sharpe'])
        p=np.where(np.arange(253)%21==1,2.,1.)
        result=audit.measure_oracle(p)
        self.assertGreater(result['daily_sample_vol'],0.);self.assertEqual(result['block_sample_vol'],0.)
        self.assertEqual(result['reason'],'zero_block_sample_volatility')
        self.assertIsNone(result['daily_sharpe'])

    def test_negative_mean_valid_and_scalar_source_errors(self):
        p=400.-np.arange(253.)+np.sin(np.arange(253.))
        result=audit.measure_oracle(p)
        self.assertTrue(result['available']);self.assertLess(result['daily_sharpe'],0.);self.assertLess(result['block_sharpe'],0.)
        for value,reason in [(True,'nonnumeric_adjusted_price'),('100','nonnumeric_adjusted_price'),
            (None,'nonnumeric_adjusted_price'),(0.,'invalid_adjusted_price'),(-1.,'invalid_adjusted_price'),
            (np.nan,'invalid_adjusted_price'),(np.inf,'invalid_adjusted_price')]:
            q=p.astype(object);q[100]=value
            self.assertEqual(audit.measure_oracle(q)['reason'],reason)
        q=np.ones(253);q[0]=np.nextafter(0.,1.);q[1]=1e308
        with self.assertRaises(FloatingPointError):audit.measure_oracle(q)

    def test_nonzero_underflow_hardfails_without_epsilon_or_floor(self):
        tiny=np.nextafter(0.,1.)
        for returns in ([tiny,0.],[1e-200,-1e-200],[2.**-537,-2.**-537]+[0.]*250):
            with self.subTest(returns=returns[:2]),self.assertRaises(FloatingPointError):audit.sample_statistics(returns)
        with self.assertRaises(FloatingPointError):audit.normalized_mean(tiny,2.)
        with self.assertRaises(FloatingPointError):audit.simple_return(tiny,2.)
        p=np.ones(253);p[0]=1e308;p[1]=tiny
        with self.assertRaises(FloatingPointError):audit.measure_oracle(p)
        self.assertEqual(audit.normalized_mean(0.,1.),0.)

    def test_all_cells_audit_counts_full_source_and_original_pct_ties(self):
        args=fixture();v,r=audit.independent_features(*args)
        result=audit.verify_features(*args,v,r)
        self.assertTrue(all(result[k] for k in audit.FEATURE_FLAGS));self.assertEqual(result['checked_cells'],9)
        self.assertEqual(result['valid_price_observations'],253*9);self.assertEqual(result['block_returns'],12*9)
        ev=r['event_history'][1]
        self.assertEqual(len(ev['source_dates']),253);self.assertEqual(ev['block_endpoint_dates'],ev['source_dates'][::21])
        self.assertEqual(ev['price_start'],ev['source_dates'][0]);self.assertEqual(ev['price_end'],ev['source_dates'][-1])
        raw=np.array([[1.,1.,1.],[0.,1.,1.]])
        got=audit.percentile(raw,np.ones((2,3),bool))
        np.testing.assert_array_equal(got[0],np.full(3,2/3-.5))
        np.testing.assert_array_equal(got[1],np.array([1/3-.5,2.5/3-.5,2.5/3-.5]))
        self.assertEqual(audit.percentile(np.array([[2.]]),np.ones((1,1),bool))[0,0],.5)

    def test_missing_interior_price_and_official_session_not_filled(self):
        p,m,c=fixture();t=c.index(m['dates'][1]);when=c[t-180]
        p['adj'].loc[when,'A']=np.nan
        v,r=audit.independent_features(p,m,c)
        self.assertFalse(v['available'][1,0]);self.assertTrue(np.isnan(v['candidate'][1,0]))
        p,m,c=fixture();p['adj']=p['adj'].drop(pd.Timestamp(when))
        v,r=audit.independent_features(p,m,c)
        self.assertFalse(v['available'][1].any());self.assertIn(when,r['event_history'][1]['missing_official_sessions'])
        self.assertTrue(all(x['reason']=='missing_official_session' for x in r['event_history'][1]['cells']))

    def test_official_clocks_exclude_recent42_and_no_future_numeric_reads(self):
        args=fixture();v,r=audit.independent_features(*args,limit=1)
        changed=copy.deepcopy(args);t=changed[2].index(changed[1]['dates'][0])
        changed[0]['adj'].loc[changed[2][t-41]:]=np.nan
        changed[1]['dates'][1:] = ['garbage',None];changed[1]['symbols'][1:]=[None,True]
        other,rr=audit.independent_features(*changed,limit=1)
        for key in audit.VALUE_KEYS:audit.exact_array(v[key],other[key],key)
        audit.record_check(r,rr)

    def test_physical_and_logical_prefix_exact_and_input_unchanged(self):
        args=fixture();before=copy.deepcopy(args);v,r=audit.independent_features(*args)
        for n in range(4):
            small=physical(args,n);vv,rr=audit.independent_features(*small)
            logical,lr=audit.independent_features(*args,limit=n)
            if n:
                for key in audit.VALUE_KEYS:audit.exact_array(vv[key],v[key][:n],key);audit.exact_array(vv[key],logical[key],key)
                audit.record_check(rr,lr)
            else:self.assertEqual(vv['available'].shape,(0,0));self.assertEqual(logical['available'].shape,(0,3))
            self.assertEqual(rr['event_history'],r['event_history'][:n])
            audit.verify_features(*small,vv,rr)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[1:],before[1:])

    def test_metadata_structure_rejected_and_insufficient_history_is_unavailable(self):
        for which in ('duplicate','weekend','midmonth','etfs','badprefix'):
            p,m,c=fixture()
            if which=='duplicate':m['symbols'][1][0]='B'
            elif which=='weekend':c[10]='2014-01-18'
            elif which=='midmonth':m['dates'][1]='2016-08-30'
            elif which=='etfs':m['etfs']=['FUND','FUND']
            with self.subTest(which=which),self.assertRaises((ValueError,TypeError)):
                audit.independent_features(p,m,c,limit=True if which=='badprefix' else None)
        p,m,c=fixture();m['dates']=['2014-01-31'];m['symbols']=m['symbols'][:1]
        v,r=audit.independent_features(p,m,c)
        self.assertFalse(v['available'].any());self.assertEqual(r['event_history'][0]['source_dates'],[])

    def test_tampered_raw_rank_mask_counts_and_audit_rejected(self):
        args=fixture();v,r=audit.independent_features(*args)
        for key in audit.VALUE_KEYS:
            bad=copy.deepcopy(v)
            if key=='available':bad[key][1,0]=False
            else:bad[key][1,0]+=1e-6
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_features(*args,bad,r)
        bad=copy.deepcopy(r);bad['event_history'][1]['block_endpoint_dates'][1]=bad['event_history'][1]['source_dates'][20]
        with self.assertRaises(ValueError):audit.verify_features(*args,v,bad)


class SelectionTests(unittest.TestCase):
    def test_half_core_original_support_and_one_frozen_episode_each_arm(self):
        a,m,v=selection_fixture();s,c,p,r=audit.selection_oracle(a,m,v)
        study=old_fixture.real_study();adapter=old_fixture.fixture_helper.helper.load(
            'outputs/monthly_core_timing70_research/core_adapter.py','block242_test_core_adapter')
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        calls=[]
        for arm,key in zip(audit.ARMS,(None,'candidate','control')):
            target=np.full(a['eligible'].shape,np.nan)
            if key is not None:target[1]=v[key][1]
            def boundary(pre,eligible,names,effects,original,rule):
                calls.append(arm);audit.exact_array(pre,p[arm],'Modified pre')
                audit.exact_array(original['baseline'],x['baseline'],'Original support baseline')
                audit.exact_array(original['core'],x['core'],'Original support core')
                return study.ep.select_boundary(pre,eligible,names,effects,original,rule)
            cc,ss=adapter.score_and_choose('incumbent66',.5,target,a,m,3,pre_score=x['local'],effects=a['effects'],
                episode_policy=dict(budget=.05,ceiling=5,gap=.02),boundary_selector=boundary)
            self.assertEqual(cc,c[arm]);audit.exact_array(ss,s[arm],arm)
        self.assertEqual(calls,list(audit.ARMS))
        oo=orders(s,c,m);out=audit.verify_selection(a,m,v,oo,scores=s,pre_scores=p,audit=r)
        self.assertTrue(all(out[k] for k in audit.SELECTION_FLAGS));self.assertEqual(r['counts']['original_episode_calls'],3)

    def test_named_partial_calm_unchanged_first_regular_active(self):
        args=selection_fixture();a,m,v=args;s,c,p,r=audit.selection_oracle(*args)
        self.assertEqual([x['active'] for x in r['events']],[False,True,False])
        for arm in audit.ARMS:
            for i in (0,2):audit.exact_array(p[arm][i],p[audit.ARMS[0]][i],arm);audit.exact_array(s[arm][i],s[audit.ARMS[0]][i],arm)
        aa={k:x[1:2].copy() for k,x in a.items()};mm={k:(x[1:2] if k in ('dates','ends','symbols') else x) for k,x in m.items()}
        vv={k:x[1:2].copy() for k,x in v.items()};ss,cc,pp,rr=audit.selection_oracle(aa,mm,vv)
        self.assertTrue(rr['events'][0]['active']);self.assertGreater(rr['counts']['candidate_nonzero_delta_cells'],0)

    def test_zero_rank_active_missing_preserves_pre_and_ineligible_not_filtered(self):
        a,m,v=selection_fixture();self.assertEqual(v['candidate'][1,3],0.)
        a['eligible'][1,3]=False
        s,c,p,r=audit.selection_oracle(a,m,v)
        self.assertEqual(r['events'][1]['candidate_delta'][3],-.5*a['core'][1,3])
        self.assertNotEqual(p[audit.ARMS[1]][1,3],p[audit.ARMS[0]][1,3])
        v['available'][1,3]=False
        for k in audit.VALUE_KEYS[1:]:v[k][1,3]=np.nan
        for k,raw in [('candidate','block_sharpe'),('control','daily_sharpe')]:v[k]=audit.percentile(v[raw],v['available'])
        _,_,pp,rr=audit.selection_oracle(a,m,v)
        self.assertEqual(pp[audit.ARMS[1]][1,3],pp[audit.ARMS[0]][1,3])

    def test_entirely_missing_and_two_cash_original_choices(self):
        for count in (0,1,2,3):
            a,m,v=selection_fixture();a['eligible'][:,count:]=False;v['available'][:]=False
            for k in audit.VALUE_KEYS[1:]:v[k][:]=np.nan
            s,c,p,r=audit.selection_oracle(a,m,v)
            for arm in audit.ARMS:
                self.assertEqual(c[arm],c[audit.ARMS[0]]);audit.exact_array(s[arm],s[audit.ARMS[0]],arm)
                self.assertEqual(len(c[arm][1]),0 if count<2 else count)

    def test_budget_and_unchanged_original_model_inputs(self):
        a,m,v=selection_fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['rank_log_excess'][:]=.2;a['model_available'][:]=True;a['raw_local_linear'][:]=np.arange(9)
        before=copy.deepcopy(a);o,s,p,r=run_selection((a,m,v));out=audit.verify_selection(a,m,v,o,scores=s,pre_scores=p,audit=r)
        self.assertEqual(out['calm_nonweak_extension_count'],9);self.assertGreaterEqual(out['minimum_remaining_support'],.38-1e-12)
        self.assertLessEqual(out['maximum_absolute_delta'],.26)
        for k in a:audit.exact_array(a[k],before[k],k)

    def test_every_physical_logical_prefix_and_future_poison(self):
        args=selection_fixture();o,s,p,r=run_selection(args)
        for n in range(4):
            small=selection_physical(args,n);oo,ss,pp,rr=run_selection(small)
            audit.verify_selection(*small,oo,scores=ss,pre_scores=pp,audit=rr)
            logical=audit.selection_oracle(*args,n=n);audit.record_check(rr,logical[3])
            self.assertEqual(rr['events'],r['events'][:n])
            for arm in audit.ARMS:audit.exact_array(ss[arm],s[arm][:n],arm);self.assertEqual(oo[arm],o[arm][:n])
        a,m,v=copy.deepcopy(args);a['pre_score'][2]=np.nan;a['core'][2]=np.inf;m['dates'][2]='bad'
        for k in audit.VALUE_KEYS[1:]:v[k][2]=np.inf
        poisoned=audit.selection_oracle(a,m,v,n=2)
        for arm in audit.ARMS:audit.exact_array(poisoned[0][arm],s[arm][:2],arm)

    def test_selection_tamper_and_dependency_closure(self):
        args=selection_fixture();a,m,v=args;o,s,p,r=run_selection(args)
        for field in ('scores','pre_scores'):
            kwargs=dict(scores=copy.deepcopy(s),pre_scores=copy.deepcopy(p),audit=r)
            kwargs[field][audit.ARMS[1]][1,8]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(*args,o,**kwargs)
        bad=copy.deepcopy(r);bad['counts']['original_episode_calls']=2
        with self.assertRaises(ValueError):audit.verify_selection(*args,o,scores=s,pre_scores=p,audit=bad)
        self.assertIn(audit.PRIOR,audit.dependency_files());self.assertIn(audit.PRIOR_TEST,audit.dependency_files())
        tree=ast.parse(Path(audit.__file__).read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('monthly_block_sharpe242_kernel',imports);self.assertNotIn('monthly_block_sharpe242_host',imports)


class AccountingTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_fee_tamper(self):
        import test_monthly_upside_retention181_independent as fixture_module
        with patch.object(fixture_module,'audit',audit):args=fixture_module.account_fixture()
        out=audit.audit_outcomes(*args);self.assertEqual((out['checked_paths'],out['checked_slices']),(9,45))
        self.assertEqual(set(out['paths']),set(audit.ARMS))
        wrong=copy.deepcopy(args);wrong[4][audit.ARMS[1]]['same_close_zero']['metrics']['fees_paid']+=1.
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unrepresentable_real_price_has_same_missing_reason_and_full_audit(self):
        import monthly_block_sharpe242_kernel as producer
        huge=[1.]*253;huge[10]=10**1000
        self.assertEqual(audit.measure_oracle(huge)['reason'],'nonnumeric_adjusted_price')
        self.assertEqual(producer.measure(huge)['reason'],'nonnumeric_adjusted_price')
        panel,meta,calendar=fixture();t=calendar.index(meta['dates'][0])
        panel['adj']=panel['adj'].astype(object);panel['adj'].at[pd.Timestamp(calendar[t-100]),'A']=10**1000
        values,report=producer.build_features(panel,meta,calendar)
        audit.verify_features(panel,meta,calendar,values,report)

    def test_unique_both_fixtures_all_feature_prefixes_exact_values_and_audit(self):
        import monthly_block_sharpe242_kernel as producer
        from test_monthly_block_sharpe242_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            v,r=producer.build_features(*args);audit.verify_features(*args,v,r)
            for n in range(len(args[1]['dates'])+1):
                small=physical(args,n);vv,rr=producer.build_features(*small);audit.verify_features(*small,vv,rr)
                logical,lr=producer.build_features(*args,limit=n);audit.verify_features(*args,logical,lr,limit=n)
                self.assertEqual(rr['event_history'],r['event_history'][:n])
                if n:
                    for k in audit.VALUE_KEYS:audit.exact_array(vv[k],v[k][:n],k)
                    audit.record_check(rr,lr)

    def test_unique_both_decision_fixtures_all_prefixes_complete_audit(self):
        import monthly_block_sharpe242_kernel as producer
        from test_monthly_block_sharpe242_kernel import selection_fixture as other_fixture
        study,a,m,v=other_fixture()
        for study,a,m,v in ((study,a,m,v),(old_fixture.real_study(),*selection_fixture())):
            o,s,p,r=producer.decisions(study,a,m,v,len(m['dates']))
            audit.verify_selection(a,m,v,o,scores=s,pre_scores=p,audit=r)
            for n in range(len(m['dates'])+1):
                aa,mm,vv=selection_physical((a,m,v),n)
                oo,ss,pp,rr=producer.decisions(study,aa,mm,vv,n)
                audit.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=pp,audit=rr)
                self.assertEqual(rr['events'],r['events'][:n])
                for arm in audit.ARMS:
                    audit.exact_array(ss[arm],s[arm][:n],arm);audit.exact_array(pp[arm],p[arm][:n],arm)
                    self.assertEqual(oo[arm],o[arm][:n])


if __name__=='__main__':unittest.main(verbosity=2)
