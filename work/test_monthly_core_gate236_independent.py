"""Only invented source paths and portfolios; no real-data loads."""
import copy
import importlib.util
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_core_gate236_independent as audit


def fixture():
    calendar=pd.bdate_range('2015-01-01','2017-02-01');names=['A','B','C','D']
    frame=pd.DataFrame(100.,index=calendar,columns=names)
    dates=['2016-08-17','2016-08-31','2016-09-30']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],ends=['2016-08-31','2016-09-30','2016-10-31'])
    a=dict(eligible=np.array([[True]*4,[False,True,True,False],[True]*4]),
           state_calm10=np.array([False,False,True]),ignored=np.full((3,4),'poison',object))
    args=({'adj':frame},meta,[str(d.date()) for d in calendar],a)
    f,s294,s252,u,t=active_endpoints(args)
    f.loc[s294,['A','B','C']]=[80.,160.,80.];f.loc[s252,['A','B','C']]=[150.,80.,80.]
    f.loc[u,['A','B','C']]=120.;f.at[calendar[calendar.get_loc(s294)+10],'D']=math.nan
    return args


def active_endpoints(args,row=1):
    p,m,c,a=args;t=c.index(m['dates'][row])
    return (p['adj'],*[pd.Timestamp(c[k]) for k in (t-294,t-252,t-42,t)])


def physical_prefix(args,n):
    p,m,c,a=copy.deepcopy(args);mm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
    aa={k:v[:n].copy() for k,v in a.items()}
    if n:
        cutoff=pd.Timestamp(m['dates'][n-1]);witness=c.index(m['dates'][n-1])+1
        pp={'adj':p['adj'].loc[:cutoff].copy()};cc=c[:witness+1]
    else:pp={'adj':p['adj'].iloc[:0].copy()};cc=[]
    return pp,mm,cc,aa


def selection_fixture():
    p,m,c,_=fixture();names=['A','B','C','D','E'];n,w=3,5;shape=(n,w)
    frame=p['adj'].reindex(columns=names).fillna(100.);m['symbols']=[names.copy() for _ in m['dates']]
    pre=np.tile([.24,.095,.18,.10,.09],(n,1));z=np.zeros(shape);f=np.zeros(shape,bool)
    calm=np.array([False,False,True]);old=np.ones(shape,bool);old[1,0]=False
    a=dict(baseline=pre.copy(),core=np.zeros(shape),core_weight=np.where(calm,.5,.52)[:,None],
        eligible=old,effects=np.tile([0.,-.4,0.,.4,.5],(n,1)),
        state_calm10=calm,state_weak_narrow=np.array([True,True,False]),pre_score=pre.copy(),
        old_ml4=z.copy(),rank_log_excess=z.copy(),extension_available=f.copy(),forecast_available=f.copy(),
        local_available=f.copy(),model_available=f.copy(),raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    args=({'adj':frame},m,c,a);f,s294,s252,u,t=active_endpoints(args)
    f.loc[s294,names]=[80.,160.,80.,80.,80.];f.loc[s252,names]=[150.,80.,80.,80.,80.];f.loc[u,names]=120.
    return args


def order_rows(scores,choices,m):
    return {arm:[dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],
        symbols=[m['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def original_production():
    path=audit.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('gate236_test_original',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FeatureTests(unittest.TestCase):
    def test_hand_known_add_remove_and_counts(self):
        args=fixture();v,r=audit.independent_features(*args)
        self.assertEqual(v['candidate_eligible'].tolist(),[[True]*4,[True,False,True,False],[True]*4])
        self.assertEqual(v['momentum210'][1,:3].tolist(),[120./150.-1.,.5,.5])
        self.assertEqual(v['momentum252'][1,:3].tolist(),[.5,-.25,.5]);self.assertEqual(v['warm'].sum(),3)
        self.assertEqual({k:r[k] for k in audit.COUNT_FIELDS},dict(events=3,symbols_per_event=4,checked_cells=12,
            original_eligible_cells=10,calm_events=1,active_events=1,required_cells=4,warm_cells=3,
            warm_observations=1180,source_price_observations=12,added_cells=1,rejected_cells=1,
            candidate_eligible_cells=10,new_model_fits=0,predictive_model_fits=0))
        proof=audit.verify_features(*args,v,r)
        self.assertTrue(all(proof[k] is True for k in audit.FEATURE_FLAGS));self.assertEqual(proof['numerical_tolerance'],0.)
        self.assertEqual(r['event_history'][1]['added_indices'],[0]);self.assertEqual(r['event_history'][1]['rejected_indices'],[1])

    def test_initial_and_calm_bypass_no_values_or_columns_consumed(self):
        p,m,c,a=physical_prefix(fixture(),1);p['adj']=pd.DataFrame(index=pd.DatetimeIndex([]))
        v,r=audit.independent_features(p,m,c,a);self.assertEqual(r['active_events'],0)
        p,m,c,a=fixture();a['state_calm10'][:]=True;p['adj']=p['adj'].astype(object);p['adj'].loc[:,:]='poison'
        v,r=audit.independent_features(p,m,c,a);audit.exact_array(v['candidate_eligible'],a['eligible'],'Calm copy')
        self.assertEqual(r['warm_observations'],0);self.assertTrue(np.isnan(v['momentum210']).all())

    def test_first_regular_query_does_not_bypass(self):
        p,m,c,a=fixture();m={k:v[1:2] for k,v in m.items()};a={k:v[1:2] for k,v in a.items()}
        v,r=audit.independent_features(p,m,c,a);self.assertEqual(r['active_events'],1);self.assertEqual(r['added_cells'],1)

    def test_nan_warmfalse_skips_numeric_endpoints_but_checks_old_E(self):
        args=fixture();args[0]['adj']=args[0]['adj'].astype(object);f,s294,s252,u,t=active_endpoints(args)
        f.at[t,'D']='poison';f.at[s294,'D']=math.inf
        v,_=audit.independent_features(*args);self.assertFalse(v['warm'][1,3]);self.assertTrue(math.isnan(v['momentum252'][1,3]))
        args[3]['eligible'][1,3]=True
        with self.assertRaisesRegex(ValueError,'Original E'):audit.independent_features(*args)

    def test_nonpositive_finite_current_is_not_warm(self):
        for bad in (0.,-1.):
            args=fixture();f,s294,s252,u,t=active_endpoints(args);f.at[t,'A']=bad
            v,_=audit.independent_features(*args);self.assertFalse(v['warm'][1,0]);self.assertFalse(v['candidate_eligible'][1,0])
            self.assertTrue(math.isnan(v['momentum210'][1,0]))

    def test_nonfinite_or_nonreal_current_hardfails_when_all_notna(self):
        for bad in (math.inf,-math.inf,True,False,'100'):
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);f,s294,s252,u,t=active_endpoints(args);f.at[t,'A']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_current_nan_is_warmfalse_not_numeric_failure(self):
        args=fixture();f,s294,s252,u,t=active_endpoints(args);f.at[t,'A']=math.nan
        v,_=audit.independent_features(*args);self.assertFalse(v['warm'][1,0])

    def test_interior_only_notna_not_stronger_price_filter(self):
        for value in (math.inf,-math.inf,-1.,0.,True,'poison'):
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);f,s294,s252,u,t=active_endpoints(args)
            day=f.index[f.index.get_loc(s294)+10];f.at[day,'A']=value
            v,_=audit.independent_features(*args);self.assertTrue(v['warm'][1,0]);self.assertTrue(v['candidate_eligible'][1,0])

    def test_invalid_warm_endpoint_fails_and_nan_endpoint_is_not_warm(self):
        for offset in (0,1,2):
            for bad in (0.,-1.,math.inf,-math.inf,True,'100'):
                args=fixture();args[0]['adj']=args[0]['adj'].astype(object);f,*ds=active_endpoints(args);f.at[ds[offset],'A']=bad
                with self.subTest(offset=offset,bad=bad),self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();f,s294,s252,u,t=active_endpoints(args);f.at[u,'A']=math.nan
        v,_=audit.independent_features(*args);self.assertFalse(v['warm'][1,0])

    def test_full295_axis_must_be_exact_no_missing_or_inserted_date(self):
        args=fixture();f,s294,s252,u,t=active_endpoints(args);interior=f.index[f.index.get_loc(s294)+10]
        args[0]['adj']=f.drop(index=interior)
        with self.assertRaisesRegex(ValueError,'axis'):audit.independent_features(*args)
        args=fixture();f,s294,s252,u,t=active_endpoints(args);weekend=s294
        while weekend.weekday()<5:weekend+=pd.Timedelta(days=1)
        f.loc[weekend]=100.;args[0]['adj']=f.sort_index()
        with self.assertRaisesRegex(ValueError,'axis'):audit.independent_features(*args)

    def test_missing_old_ineligible_column_is_hardfailure(self):
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='D')
        with self.assertRaisesRegex(ValueError,'column'):audit.independent_features(*args)

    def test_old_parity_all_active_names_and_strict_zero_sign(self):
        for column in (0,1):
            args=fixture();args[3]['eligible'][1,column]=not args[3]['eligible'][1,column]
            with self.assertRaisesRegex(ValueError,'Original E'):audit.independent_features(*args)
        args=fixture();f,s294,s252,u,t=active_endpoints(args);f.at[u,'A']=f.at[s294,'A']
        v,_=audit.independent_features(*args);self.assertEqual(v['momentum252'][1,0],0.);self.assertFalse(v['candidate_eligible'][1,0])
        f.at[u,'A']=np.nextafter(f.at[s294,'A'],math.inf)
        v,_=audit.independent_features(*args);self.assertTrue(v['candidate_eligible'][1,0])

    def test_canonical_binary64_overflow_fails(self):
        with self.assertRaises(FloatingPointError):audit.momentum(np.nextafter(0.,1.),np.finfo(float).max)
        self.assertEqual(audit.momentum(4.,5.),.25)
        args=fixture();f,s294,s252,u,t=active_endpoints(args);f.at[s294,'A']=np.nextafter(0.,1.)
        with self.assertRaises(FloatingPointError):audit.independent_features(*args)

    def test_latest42_change_only_warm_not_momentum_and_future_unread(self):
        args=fixture();full=audit.independent_features(*args);f,s294,s252,u,t=active_endpoints(args)
        f.loc[(f.index>u)&(f.index<=t),:]=777.;f.loc[f.index>t,:]=math.nan
        changed=audit.independent_features(*args)
        for k in audit.VALUE_KEYS:audit.exact_array(full[0][k],changed[0][k],k)
        audit.record_check(full[1],changed[1])

    def test_all_logical_physical_prefixes_and_future_poison(self):
        args=fixture();full=audit.independent_features(*args)
        for n in range(4):
            logical=audit.independent_features(*args,limit=n);physical=audit.independent_features(*physical_prefix(args,n))
            audit.record_check(logical[1],physical[1])
            for k in audit.VALUE_KEYS:
                audit.exact_array(logical[0][k],physical[0][k],k);audit.exact_array(full[0][k][:n],physical[0][k],k)
        args[1]['dates'][2]='future poison';args[1]['symbols'][2]=['bad']
        for key in ('eligible','state_calm10'):
            args[3][key]=args[3][key].astype(object);args[3][key][2]='future poison'
        audit.independent_features(*args,limit=2)

    def test_schema_calendar_and_bool_validation(self):
        for mutate in (
            lambda x:x[3].update(state_calm10=np.ones((3,1),bool)),
            lambda x:x[3].update(eligible=np.ones((3,4),int)),
            lambda x:x[1]['symbols'].__setitem__(1,['A']*4),
            lambda x:x[1]['dates'].__setitem__(1,'2016-08-30'),
            lambda x:x[2].__setitem__(5,x[2][4]),
        ):
            args=fixture();mutate(args)
            with self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture()
        with self.assertRaises(ValueError):audit.independent_features(*args,limit=True)
        with self.assertRaises(ValueError):audit.independent_features(args[0],args[1],args[2][:args[2].index(args[1]['dates'][-1])+1],args[3])

    def test_no_input_mutation_and_all_values_audit_tamper_rejected(self):
        args=fixture();before=copy.deepcopy(args);v,r=audit.independent_features(*args)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[1:3],before[1:3])
        for k in args[3]:audit.exact_array(args[3][k],before[3][k],k)
        for key in audit.VALUE_KEYS:
            wrong=copy.deepcopy(v);wrong[key].flat[0]=not wrong[key].flat[0] if wrong[key].dtype==bool else 0.
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_features(*args,wrong,r)
        wrong=copy.deepcopy(r);wrong['added_cells']+=1
        with self.assertRaises(ValueError):audit.verify_features(*args,v,wrong)


class SelectionTests(unittest.TestCase):
    def test_new_eligibility_before_single_frozen_selector_matches_exactly(self):
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a);expected,choices,pres=audit.selection_oracle(a,m,v)
        study=original_production();policy=dict(id=7,family='level',model_id=2,budget=.05,ceiling=5,gap=.02)
        orders={};scores={};calls={}
        for arm in audit.ARMS:
            calls[arm]=0;gate=v['original_eligible'] if arm==audit.ARMS[0] else v['candidate_eligible']
            def selector(pre,originalE,names,effects,source,rule):
                calls[arm]+=1;audit.exact_array(originalE,a['eligible'],'Original E');audit.exact_array(pre,pres[arm],'Original pre')
                return study.ep.select_boundary(pre,gate.copy(),names,effects,source,rule)
            chosen,score=study.choices(study.configs()[4],a,m,3,policy=policy,boundary_selector=selector)
            self.assertEqual(chosen,choices[arm]);audit.exact_array(score,expected[arm],arm)
            orders[arm]=study.rp.make_orders(chosen,score,m);scores[arm]=score
        self.assertEqual(list(calls.values()),[1,1]);self.assertIn(0,choices[audit.ARMS[1]][1]);self.assertNotIn(1,choices[audit.ARMS[1]][1])
        proof=audit.verify_selection(a,m,v,orders,scores=scores,pre_scores=pres)
        self.assertTrue(all(proof[k] is True for k in audit.SELECTION_FLAGS));self.assertTrue(proof['full_pre_scores_supplied'])
        self.assertFalse(np.array_equal(expected[audit.ARMS[0]][1],expected[audit.ARMS[1]][1]))

    def test_identity_gate_and_all_order_prefixes(self):
        args=selection_fixture();f,s294,s252,u,t=active_endpoints(args);f.loc[s294,args[1]['symbols'][1]]=f.loc[s252,args[1]['symbols'][1]].to_numpy()
        p,m,c,a=args;v,_=audit.independent_features(*args);sc,ch,pr=audit.selection_oracle(a,m,v)
        audit.exact_array(sc[audit.ARMS[0]],sc[audit.ARMS[1]],'Identity gate');self.assertEqual(ch[audit.ARMS[0]],ch[audit.ARMS[1]])
        for n in range(4):
            s2,c2,p2=audit.selection_oracle(a,m,v,n);oo=order_rows(s2,c2,{k:val[:n] for k,val in m.items()})
            audit.verify_selection(a,m,v,oo,scores=s2,pre_scores=p2,n=n)
            for arm in audit.ARMS:audit.exact_array(s2[arm],sc[arm][:n],arm)

    def test_two_halves_and_below_two_cash(self):
        for surviving in (0,1,2):
            args=selection_fixture();f,s294,s252,u,t=active_endpoints(args);args[3]['eligible'][1]=True;f.loc[s252,:]=80.
            for j,name in enumerate(args[1]['symbols'][1]):f.at[s294,name]=80. if j<surviving else 160.
            p,m,c,a=args;v,_=audit.independent_features(*args);sc,ch,pr=audit.selection_oracle(a,m,v)
            self.assertEqual(len(ch[audit.ARMS[1]][1]),2 if surviving==2 else 0)
            audit.verify_selection(a,m,v,order_rows(sc,ch,m),scores=sc,pre_scores=pr)

    def test_added_name_preserves_old_model_rank_domain_and_fallback(self):
        p,m,c,a=selection_fixture();a['forecast_available'][:]=True;a['model_available'][:]=True
        a['raw_local_constant'][:]=[1.,2.,3.,4.,5.];a['raw_local_linear'][:]=[5.,1.,4.,2.,3.];a['raw_source_log'][:]=0.
        v,_=audit.independent_features(p,m,c,a);sc,ch,pr=audit.selection_oracle(a,m,v)
        original=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        audit.exact_array(pr[audit.ARMS[1]],original['local'],'Original E model domain')
        mutated=copy.deepcopy(a);mutated['eligible']=v['candidate_eligible'].copy()
        wrong=audit.frozen_source_oracle().old.original_helper().original_prefix(mutated,m,3)
        self.assertFalse(np.array_equal(wrong['local'][1],pr[audit.ARMS[1]][1]))
        self.assertEqual(pr[audit.ARMS[1]][1,0],a['baseline'][1,0])

    def test_selection_evidence_semantics_and_full_pre_score_order_tamper(self):
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a);sc,ch,pr=audit.selection_oracle(a,m,v);o=order_rows(sc,ch,m)
        for target in ('scores','pre_scores'):
            kw=dict(scores=copy.deepcopy(sc),pre_scores=copy.deepcopy(pr));kw[target][audit.ARMS[1]][1,0]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,v,o,**kw)
        for key in audit.VALUE_KEYS:
            bad=copy.deepcopy(v)
            if bad[key].dtype==bool:bad[key][1,0]=not bad[key][1,0]
            else:bad[key][0,0]=0.
            with self.subTest(key=key),self.assertRaises(ValueError):audit.selection_oracle(a,m,bad)
        wrong=copy.deepcopy(o);wrong[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,wrong,scores=sc,pre_scores=pr)

    def test_no_new_producer_import_and_pinned_primitives(self):
        self.assertNotIn('import monthly_core_gate236_kernel',inspect.getsource(audit))
        with patch.object(audit,'ACCOUNTING_SHA','bad'),self.assertRaises(ValueError):audit.accounting()
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a)
        with patch.object(audit,'SOURCE_SHA','bad'),self.assertRaises(ValueError):audit.selection_oracle(a,m,v)


class OutcomeTests(unittest.TestCase):
    def test_two_arms_six_paths_30_slices_and_bad_metric(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        r=audit.audit_outcomes(*args);self.assertEqual((r['checked_paths'],r['checked_slices']),(6,30))
        bad=copy.deepcopy(args);bad[4][audit.ARMS[1]]['same_close_zero']['metrics']['cagr']+=.01
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)

    def test_cost_path_fee_and_missing_path_rejected(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        bad=copy.deepcopy(args);del bad[3][audit.ARMS[1]]['next_open_cost']
        with self.assertRaises(ValueError):audit.audit_outcomes(*bad)
        bad=copy.deepcopy(args);bad[4][audit.ARMS[1]]['next_open_cost']['metrics']['fees_paid']+=1.
        with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


class BilateralTests(unittest.TestCase):
    def test_both_fixtures_all_prefixes_full_values_and_audit(self):
        import monthly_core_gate236_kernel as producer
        from test_monthly_core_gate236_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            p,m,c,a=args;full=producer.build_features(*args);audit.verify_features(*args,*full)
            for n in range(len(a['eligible'])+1):
                logical=producer.build_features(*args,limit=n);audit.verify_features(*args,*logical,limit=n)
                small=physical_prefix(args,n);out=producer.build_features(*small);audit.verify_features(*small,*out)
                audit.record_check(logical[1],out[1])
                for key in audit.VALUE_KEYS:audit.exact_array(full[0][key][:n],out[0][key],key)


if __name__=='__main__':unittest.main(verbosity=2)
