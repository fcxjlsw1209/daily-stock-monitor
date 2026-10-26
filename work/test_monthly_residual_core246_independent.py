"""Synthetic independent lagged FWL, source admission, full scoring and ledgers."""
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

import monthly_residual_core246_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest()!=audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen fixture changed')
    spec=importlib.util.spec_from_file_location('residual246_prior_test',audit.PRIOR_TEST)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


old_fixture=helper()


def price(returns):
    p=[100.]
    for r in returns:p.append(p[-1]*(1.+float(r)))
    return p


def returns_fixture(length=504):
    k=np.arange(length,dtype=float)
    m=.001+np.sin(k*.17)*.012
    s=.0007+np.cos(k*.31)*.009+.2*m
    y=.0008+.7*m+.35*s+np.sin(k*.53)*.002
    return y.tolist(),m.tolist(),s.tolist()


def fixture():
    ix=pd.bdate_range('2013-01-01','2017-01-04');y,m,s=returns_fixture(len(ix)-1)
    panel={'adj':pd.DataFrame(dict(A=price(y),B=price(np.array(y)*.5+np.sin(np.arange(len(y))*.47)*.003),
        SPY=price(m),XLK=price(s),FUND=np.ones(len(ix))),index=ix),
        'vox':pd.DataFrame({'VOX':price(np.array(s)*.8+np.cos(np.arange(len(s))*.11)*.002)},index=ix)}
    meta=dict(dates=['2016-08-17','2016-08-31','2016-09-30'],symbols=[['A','B','FUND'] for _ in range(3)],etfs=['FUND'])
    mapping=[dict(query_index=i,column_index=j,date=d,symbol=symbol,issuer_key='issuer:'+symbol,
        original_sector='XLK',sector='XLK',mapping_origin='query_date_sector_panel',mapping_repair='unchanged',
        mapping_evidence=None) for i,d in enumerate(meta['dates']) for j,symbol in enumerate(['A','B'])]
    return panel,meta,ix.strftime('%Y-%m-%d').tolist(),dict(mapping_records=mapping)


def physical(args,n):
    p,m,c,r=copy.deepcopy(args);day=m['dates'][n-1] if n else None
    m={k:(v[:n] if k in ('dates','symbols','ends') else v) for k,v in m.items()}
    p={k:(frame.loc[:day].copy() if n else frame.iloc[:0].copy()) for k,frame in p.items()}
    c=c[:c.index(day)+2] if n else [];r['mapping_records']=[x for x in r['mapping_records'] if x['query_index']<n]
    return p,m,c,r


def selection_fixture():
    a,m=old_fixture.old_fixture.fixture();m['etfs']=[m['symbols'][0][-1]]
    shape=a['eligible'].shape;available=np.ones(shape,bool);available[:,-1]=False
    raw=np.tile(np.arange(shape[1],dtype=float),(shape[0],1));candidate=audit.percentile(raw,available)
    control=audit.percentile(-raw,available)
    return a,m,dict(available=available,candidate=candidate,control=control)


def selection_physical(args,n):
    a,m,v=copy.deepcopy(args);a,m=old_fixture.old_fixture.physical((a,m),n)
    return a,m,{k:x[:n].copy() for k,x in v.items()}


def orders(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


class MathematicsTests(unittest.TestCase):
    def test_scalar_fwl_recovers_slopes_and_does_not_subtract_alpha(self):
        _,m,s=returns_fixture(252);y=[.003+.6*x+.4*z for x,z in zip(m,s)]
        fit=audit.fit_fwl(y,m,s)
        self.assertTrue(fit['fit_available']);self.assertAlmostEqual(fit['beta_market_joint'],.6,13)
        self.assertAlmostEqual(fit['beta_sector_joint'],.4,13)
        residual=[(a-fit['beta_market_joint']*b)-fit['beta_sector_joint']*c for a,b,c in zip(y,m,s)]
        self.assertAlmostEqual(math.fsum(residual)/252.,.003,14)
        self.assertGreater(abs(math.fsum(residual)),.5)

    def test_505_window_first_last_fit_and_application_exclusion(self):
        y,m,s=returns_fixture();p=[price(x) for x in (y,m,s)];out=audit.measure_oracle(*p)
        actual=[[float(z[k])/float(z[k-1])-1. for k in range(1,505)] for z in p]
        for h in (0,251):
            fit=audit.fit_fwl(*[x[h:h+252] for x in actual])
            for key in ('beta_market_only','beta_market_joint','beta_sector_joint'):
                self.assertEqual(out['fit_arrays'][key][h],fit[key])
        changed=copy.deepcopy(p);changed[0][-1]*=1.01;later=audit.measure_oracle(*changed)
        for key in ('beta_market_only','beta_market_joint','beta_sector_joint'):
            self.assertEqual(out['fit_arrays'][key],later['fit_arrays'][key])
        self.assertNotEqual(out['fit_arrays']['adjusted_joint'][-1],later['fit_arrays']['adjusted_joint'][-1])

    def test_same_sample_stock_sd_and_signed_adjusted_drift(self):
        y,m,s=returns_fixture();p=[price(x) for x in (y,m,s)];out=audit.measure_oracle(*p)
        f=[float(p[0][k])/float(p[0][k-1])-1. for k in range(253,505)]
        mean=math.fsum(f)/252;ss=math.fsum((x-mean)*(x-mean) for x in f);sd=math.sqrt(ss/251.)
        self.assertEqual(out['stock_sample_vol'],sd)
        self.assertEqual(out['raw_candidate'],out['joint_adjusted_mean']/sd)
        self.assertEqual(out['raw_control'],out['market_adjusted_mean']/sd)
        p=[price(np.array(y)-.004),price(m),price(s)];out=audit.measure_oracle(*p)
        self.assertTrue(out['available']);self.assertLess(out['joint_adjusted_mean'],0.)

    def test_zero_stock_sd_retains_all_fits_and_zero_summaries(self):
        _,m,s=returns_fixture();out=audit.measure_oracle([100.]*505,price(m),price(s))
        self.assertFalse(out['available']);self.assertEqual(out['reason'],'zero_stock_sample_volatility')
        self.assertTrue(all(out['fit_arrays']['fit_available']));self.assertEqual(out['joint_adjusted_mean'],0.)
        self.assertIsNone(out['raw_candidate']);self.assertIsNone(out['raw_control'])

    def test_zero_factor_and_collinearity_common_exclusion(self):
        y,m,s=returns_fixture()
        for market,sector,reason in (([100.]*505,price(s),'zero_market_variance'),
            (price(m),[100.]*505,'zero_sector_variance'),(price(m),price(m),'collinear_factors')):
            out=audit.measure_oracle(price(y),market,sector)
            self.assertFalse(out['available']);self.assertEqual(out['reason'],reason)
            self.assertEqual(len(out['fit_arrays']['fit_reason']),252)
            self.assertTrue(all(x==reason for x in out['fit_arrays']['fit_reason']))
            self.assertTrue(all(x is None for x in out['fit_arrays']['beta_market_only']))

    def test_condition_threshold_sides_and_scale_invariant_gate(self):
        k=np.arange(252);m=np.where(k%2,1.,-1.);orth=np.where((k//2)%2,1.,-1.)
        y=.1*m+.2*orth
        for epsilon,want in ((5e-5,False),(2e-4,True)):
            s=m+epsilon*orth;fit=audit.fit_fwl(y.tolist(),m.tolist(),s.tolist())
            self.assertEqual(fit['fit_available'],want)
            scaled=audit.fit_fwl(y.tolist(),(m*3).tolist(),(s*7).tolist())
            self.assertEqual(scaled['fit_available'],want)
        with self.assertRaises(ValueError):
            audit.check_normal_equations(y.tolist(),m.tolist(),orth.tolist(),252.,252.,10.,10.,10.)

    def test_small_variance_cannot_be_tolerantly_changed_to_negative(self):
        y,m,s=returns_fixture();p=[price(np.array(x)*1e-8) for x in (y,m,s)]
        out=audit.measure_oracle(*p);fit=out['fit_arrays'];bad=copy.deepcopy(fit)
        self.assertLess(fit['market_ss'][0],audit.RAW_ATOL)
        bad['market_ss'][0]=-fit['market_ss'][0]
        with self.assertRaises(ValueError):audit.verify_fit(bad,fit,*p,'negative variance')

    def test_invalid_sources_and_derived_underflow_fail_without_floor(self):
        y,m,s=returns_fixture();base=price(y)
        for bad,reason in ((True,'nonnumeric_adjusted_price'),('100','nonnumeric_adjusted_price'),
            (10**1000,'nonnumeric_adjusted_price'),(float('inf'),'invalid_adjusted_price'),
            (0.,'invalid_adjusted_price'),(float('nan'),'invalid_adjusted_price')):
            p=base.copy();p[100]=bad
            self.assertEqual(audit.measure_oracle(p,price(m),price(s))['stock_source_reason'],reason)
        with self.assertRaises(FloatingPointError):audit.square(1e-200)
        with self.assertRaises(FloatingPointError):audit.simple_return(np.nextafter(0.,1.),2.)
        with self.assertRaises(FloatingPointError):audit.mean([np.nextafter(0.,1.),0.])


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.values,cls.report=audit.independent_features(*cls.args)

    def test_full_inventory_clocks_counts_etfs_and_finite_ranks(self):
        v,r=self.values,self.report;out=audit.verify_features(*self.args,v,r)
        self.assertTrue(all(out[k] for k in audit.FEATURE_FLAGS));self.assertTrue(out['exact_discrete'])
        self.assertEqual(r['counts']['stock_cells'],6);self.assertEqual(r['counts']['rolling_windows_checked'],1512)
        self.assertEqual(r['counts']['stock_price_observations'],6*505)
        self.assertFalse(v['available'][:,-1].any());self.assertTrue(np.isnan(v['candidate'][:,-1]).all())
        ev=r['event_history'][1];ds=ev['source_dates'];self.assertEqual(len(ds),505)
        self.assertEqual(ev['fit_dates'][0],dict(training_price_start=ds[0],training_return_start=ds[1],
            training_return_end=ds[252],application_date=ds[253]))
        self.assertEqual(ev['fit_dates'][-1]['training_return_end'],ds[503])
        self.assertEqual(ev['fit_dates'][-1]['application_date'],ds[504])

    def test_unresolved_mapping_does_not_hide_stock_source(self):
        args=physical(self.args,1);args[3]['mapping_records'][0]['sector']=None
        v,r=audit.independent_features(*args);cell=r['event_history'][0]['cells'][0]
        self.assertTrue(cell['stock_source_available']);self.assertTrue(cell['market_source_available'])
        self.assertEqual(cell['reason'],'unresolved_mapping');self.assertFalse(v['available'][0,0])
        dates=r['event_history'][0]['source_dates'];args[0]['adj'].at[pd.Timestamp(dates[100]),'A']=np.nan
        _,r=audit.independent_features(*args);self.assertFalse(r['event_history'][0]['cells'][0]['stock_source_available'])
        self.assertEqual(r['event_history'][0]['cells'][0]['reason'],'stock_invalid_adjusted_price')

    def test_dedicated_vox_and_missing_full_window_no_market_fallback(self):
        args=physical(self.args,1);args[3]['mapping_records'][0]['sector']='VOX'
        args[0]['vox']=args[0]['vox'].iloc[600:]
        v,r=audit.independent_features(*args);cell=r['event_history'][0]['cells'][0]
        self.assertEqual(cell['reason'],'sector_missing_official_session');self.assertFalse(v['available'][0,0])
        self.assertTrue(cell['stock_source_available']);self.assertTrue(v['available'][0,1])
        self.assertTrue(r['event_history'][0]['missing_vox_sessions'])

    def test_mapping_duplicate_issuer_wrong_identity_missing_records_fail(self):
        for kind in ('missing','duplicate','issuer','identity','bool_index'):
            args=physical(self.args,1);rr=args[3]['mapping_records']
            if kind=='missing':rr.pop()
            elif kind=='duplicate':rr.append(copy.deepcopy(rr[0]))
            elif kind=='issuer':rr[1]['issuer_key']=rr[0]['issuer_key']
            elif kind=='identity':rr[0]['symbol']='wrong'
            else:rr[0]['column_index']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_future_numeric_metadata_and_mapping_poison_ignored(self):
        args=copy.deepcopy(self.args);t=args[2].index(args[1]['dates'][0])
        for frame in args[0].values():frame.loc[args[2][t-41]:]=np.nan
        args[1]['dates'][1:]=['bad',None];args[1]['symbols'][1:]=[None,True]
        args[3]['mapping_records']=[r if r['query_index']==0 else {'query_index':r['query_index'],'poison':object()}
            for r in args[3]['mapping_records']]
        v,r=audit.independent_features(*args,limit=1)
        for key in audit.VALUE_KEYS:audit.exact_array(v[key],self.values[key][:1],key)
        self.assertEqual(r['event_history'],self.report['event_history'][:1])

    def test_physical_logical_prefix_no_mutation(self):
        before=copy.deepcopy(self.args)
        for n in range(4):
            small=physical(self.args,n);v,r=audit.independent_features(*small)
            logical,lr=audit.independent_features(*self.args,limit=n)
            self.assertEqual(r['event_history'],self.report['event_history'][:n])
            if n:
                for key in audit.VALUE_KEYS:audit.exact_array(v[key],self.values[key][:n],key)
                audit.record_check(r,lr)
            else:self.assertEqual(v['available'].shape,(0,0));self.assertEqual(logical['available'].shape,(0,3))
        for k in before[0]:pd.testing.assert_frame_equal(before[0][k],self.args[0][k])
        self.assertEqual(before[1:],self.args[1:])

    def test_fit_raw_gate_rank_count_and_clock_tampering_detected(self):
        args=physical(self.args,1);v,r=audit.independent_features(*args)
        for kind in ('slope','mask','rank','count','clock','raw','gate'):
            bad=copy.deepcopy(r);vv=copy.deepcopy(v);cell=bad['event_history'][0]['cells'][0]
            if kind=='slope':cell['fit_arrays']['beta_market_joint'][0]+=.001
            elif kind=='mask':cell['fit_arrays']['fit_available'][0]=False
            elif kind=='rank':vv['candidate'][0,0]+=.01
            elif kind=='count':bad['counts']['descriptive_joint_estimates']-=1
            elif kind=='clock':bad['event_history'][0]['fit_dates'][0]['training_return_end']='2016-01-01'
            elif kind=='raw':cell['raw_candidate']+=.001
            else:cell['fit_arrays']['conditioning'][0]=1e-8
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.verify_features(*args,vv,bad)

    def test_fixed_tolerance_does_not_create_rank_ties(self):
        raw=np.array([[1.,1.+1e-12,1.]])
        np.testing.assert_array_equal(audit.percentile(raw,np.ones((1,3),bool)),np.array([[0.,.5,0.]]))
        audit.close_stat(1.+1e-11,1.,'within bound')
        with self.assertRaises(ValueError):audit.close_stat(1.+1e-7,1.,'outside bound')


class SelectionTests(unittest.TestCase):
    def test_full_original_adapter_scores_orders_and_one_episode_all_states(self):
        a,m,v=selection_fixture();scores,choices,pres,r=audit.selection_oracle(a,m,v)
        study=old_fixture.old_fixture.real_study();adapter=old_fixture.old_fixture.fixture_helper.helper.load(
            'outputs/monthly_core_timing70_research/core_adapter.py','residual246_original_adapter')
        x=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3);calls=[]
        for arm,key in zip(audit.ARMS,(None,'candidate','control')):
            target=np.full(a['eligible'].shape,np.nan)
            if key is not None:target[1:]=v[key][1:]
            def boundary(pre,E,names,effects,original,policy):
                calls.append(arm);audit.exact_array(pre,pres[arm],'Full modified pre')
                audit.exact_array(original['core'],x['core'],'Original core stays support anchor')
                return study.ep.select_boundary(pre,E,names,effects,original,policy)
            cc,ss=adapter.score_and_choose('incumbent66',.5,target,a,m,3,pre_score=x['local'],effects=a['effects'],
                episode_policy=dict(budget=.05,ceiling=5,gap=.02),boundary_selector=boundary)
            self.assertEqual(cc,choices[arm]);audit.exact_array(ss,scores[arm],arm)
        self.assertEqual(calls,list(audit.ARMS));self.assertEqual([e['active'] for e in r['events']],[False,True,True])
        out=audit.verify_selection(a,m,v,orders(scores,choices,m),scores=scores,pre_scores=pres,audit=r)
        self.assertTrue(all(out[k] for k in audit.SELECTION_FLAGS))

    def test_partial_etfs_and_missing_noop_finite_zero_is_active(self):
        a,m,v=selection_fixture();v['candidate'][1,3]=0.;a['eligible'][1,3]=False
        scores,choices,pres,r=audit.selection_oracle(a,m,v)
        self.assertEqual(r['events'][1]['candidate_delta'][3],-.5*a['core'][1,3])
        for arm in audit.ARMS:
            audit.exact_array(pres[arm][0],pres[audit.ARMS[0]][0],arm)
            self.assertEqual(pres[arm][1,-1],pres[audit.ARMS[0]][1,-1])
        v['available'][1,3]=False;v['candidate'][1,3]=v['control'][1,3]=np.nan
        _,_,p,_=audit.selection_oracle(a,m,v);self.assertEqual(p[audit.ARMS[1]][1,3],p[audit.ARMS[0]][1,3])
        v['available'][1,-1]=True;v['candidate'][1,-1]=v['control'][1,-1]=0.
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,v)

    def test_budget_extra_extension_original_inputs_unchanged(self):
        a,m,v=selection_fixture();a['extension_available'][:]=True;a['forecast_available'][:]=True
        a['rank_log_excess'][:]=.2;a['model_available'][:]=True;a['raw_local_linear'][:]=np.arange(9)
        before=copy.deepcopy(a);s,c,p,r=audit.selection_oracle(a,m,v)
        out=audit.verify_selection(a,m,v,orders(s,c,m),scores=s,pre_scores=p,audit=r)
        self.assertEqual(out['calm_nonweak_extension_count'],9);self.assertGreaterEqual(out['minimum_remaining_support'],.38-1e-12)
        for event in r['events']:
            self.assertEqual(event['original_remaining_support'],event['candidate_remaining_support'])
        for key in a:audit.exact_array(a[key],before[key],key)

    def test_selection_all_prefixes_future_poison_and_tamper(self):
        args=selection_fixture();s,c,p,r=audit.selection_oracle(*args)
        for n in range(4):
            small=selection_physical(args,n);ss,cc,pp,rr=audit.selection_oracle(*small)
            audit.verify_selection(*small,orders(ss,cc,small[1]),scores=ss,pre_scores=pp,audit=rr)
            logical=audit.selection_oracle(*args,n=n);audit.record_check(rr,logical[3])
            for arm in audit.ARMS:audit.exact_array(ss[arm],s[arm][:n],arm)
        a,m,v=copy.deepcopy(args);a['pre_score'][2]=np.nan;a['core'][2]=np.inf;m['dates'][2]='bad'
        v['candidate'][2]=np.inf
        ss,_,_,_=audit.selection_oracle(a,m,v,n=2)
        for arm in audit.ARMS:audit.exact_array(ss[arm],s[arm][:2],arm)
        bad=copy.deepcopy(p);bad[audit.ARMS[1]][1,8]+=.001
        with self.assertRaises(ValueError):audit.verify_selection(*args,orders(s,c,args[1]),scores=s,pre_scores=bad,audit=r)

    def test_dependency_closure_and_no_new_producer_import(self):
        files=audit.dependency_files();self.assertIn(audit.PRIOR,files);self.assertIn(audit.PRIOR_TEST,files)
        tree=ast.parse(Path(audit.__file__).read_text())
        imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        self.assertNotIn('monthly_residual_core246_kernel',imports);self.assertNotIn('monthly_residual_core246_host',imports)


class AccountingTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_fees_are_independent(self):
        import test_monthly_upside_retention181_independent as fixture_module
        with patch.object(fixture_module,'audit',audit):args=fixture_module.account_fixture()
        out=audit.audit_outcomes(*args);self.assertEqual((out['checked_paths'],out['checked_slices']),(9,45))
        wrong=copy.deepcopy(args);wrong[4][audit.ARMS[1]]['same_close_zero']['metrics']['fees_paid']+=1.
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unique_both_feature_fixtures_complete_fit_arrays_and_all_prefixes(self):
        import monthly_residual_core246_kernel as producer
        from test_monthly_residual_core246_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            v,r=producer.build_features(*args)
            audit.verify_features(*args,v,r)
            for n in range(len(args[1]['dates'])+1):
                small=physical(args,n);vv,rr=producer.build_features(*small)
                audit.verify_features(*small,vv,rr)
                logical,lr=producer.build_features(*args,limit=n)
                if n:
                    for key in audit.VALUE_KEYS:audit.exact_array(vv[key],v[key][:n],key)
                    audit.record_check(rr,lr)
                self.assertEqual(rr['event_history'],r['event_history'][:n])
                if n==0:
                    audit.verify_features(*args,logical,lr,limit=0)

    def test_unique_both_decision_fixtures_full_pres_scores_orders_all_prefixes(self):
        import monthly_residual_core246_kernel as producer
        from test_monthly_residual_core246_kernel import selection_fixture as other_fixture
        for study,a,m,v in (other_fixture(),(old_fixture.old_fixture.real_study(),*selection_fixture())):
            o,s,p,r=producer.decisions(study,a,m,v,len(m['dates']))
            audit.verify_selection(a,m,v,o,scores=s,pre_scores=p,audit=r)
            for n in range(len(m['dates'])+1):
                aa,mm,vv=selection_physical((a,m,v),n)
                oo,ss,pp,rr=producer.decisions(study,aa,mm,vv,n)
                audit.verify_selection(aa,mm,vv,oo,scores=ss,pre_scores=pp,audit=rr)
                lo,ls,lp,lr=producer.decisions(study,a,m,v,n)
                audit.record_check(lo,oo);audit.record_check(lr,rr)
                for arm in audit.ARMS:
                    audit.exact_array(ss[arm],s[arm][:n],arm);audit.exact_array(pp[arm],p[arm][:n],arm)
                    audit.exact_array(ls[arm],ss[arm],arm);audit.exact_array(lp[arm],pp[arm],arm)
                    self.assertEqual(oo[arm],o[arm][:n])


if __name__=='__main__':unittest.main(verbosity=2)
