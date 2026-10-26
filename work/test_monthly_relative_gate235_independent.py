"""Invented source paths and portfolios only; no real data entry points."""
import copy
import importlib.util
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_relative_gate235_independent as audit


def fixture():
    calendar=pd.bdate_range('2015-01-01','2017-02-01')
    names=['A','B','C','D']
    frame=pd.DataFrame(100.,index=calendar,columns=names+['SPY'])
    dates=['2016-08-17','2016-08-31','2016-09-30']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],
        ends=['2016-08-31','2016-09-30','2016-10-31'])
    a=dict(eligible=np.array([[True]*4,[True,True,True,False],[True]*4]),
           state_weak_narrow=np.array([True,True,False]),ignored=np.full((3,4),'poison',object))
    t=calendar.get_loc(pd.Timestamp(dates[1]));s,u=calendar[t-252],calendar[t-42]
    frame.loc[u,['A','B','C','SPY']]=[150.,125.,112.5,125.]
    frame.loc[s,'D']=math.nan
    return {'adj':frame},meta,[str(d.date()) for d in calendar],a


def physical_prefix(args,n):
    panel,meta,calendar,arrays=copy.deepcopy(args)
    mm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in meta.items()}
    aa={k:v[:n].copy() for k,v in arrays.items()}
    if n:
        cutoff=pd.Timestamp(meta['dates'][n-1]);days=[pd.Timestamp(x) for x in calendar]
        witness=days.index(cutoff)+1
        pp={'adj':panel['adj'].loc[:cutoff].copy()};cc=calendar[:witness+1]
    else:
        pp={'adj':panel['adj'].iloc[:0].copy()};cc=[]
    return pp,mm,cc,aa


def active_endpoints(args,row=1,symbol='SPY'):
    panel,m,c,a=args;t=c.index(m['dates'][row]);return panel['adj'],pd.Timestamp(c[t-252]),pd.Timestamp(c[t-42])


def selection_fixture():
    panel,meta,calendar,a=fixture();names=['A','B','C','D','E'];n,w=3,5;shape=(n,w)
    frame=panel['adj'].reindex(columns=names+['SPY']).fillna(100.)
    meta['symbols']=[names.copy() for _ in meta['dates']]
    pre=np.tile([.24,.18,.10,.095,.09],(n,1));z=np.zeros(shape);f=np.zeros(shape,bool)
    a=dict(baseline=pre.copy(),core=np.zeros(shape),core_weight=np.full((n,1),.5),
        eligible=np.ones(shape,bool),effects=np.tile([0.,0.,-.4,.4,.5],(n,1)),
        state_calm10=np.ones(n,bool),state_weak_narrow=np.array([True,True,False]),pre_score=pre.copy(),
        old_ml4=z.copy(),rank_log_excess=z.copy(),extension_available=f.copy(),forecast_available=f.copy(),
        local_available=f.copy(),model_available=f.copy(),raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    args=({'adj':frame},meta,calendar,a);frame,s,u=active_endpoints(args)
    frame.loc[s,names+['SPY']]=100.;frame.loc[u,names+['SPY']]=[112.5,150.,150.,150.,150.,125.]
    return args


def order_rows(scores,choices,m):
    return {arm:[dict(date=m['dates'][i],end=m['ends'][i],signal_date=m['dates'][i],
        symbols=[m['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def original_production():
    path=audit.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('gate235_test_original',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class GateTests(unittest.TestCase):
    def test_hand_known_strict_gate_and_exact_counts(self):
        p,m,c,a=fixture();v,r=audit.independent_features(p,m,c,a)
        self.assertEqual(v['candidate_eligible'].tolist(),[[True]*4,[True,False,False,False],[True]*4])
        self.assertEqual(v['required'].sum(),3);self.assertEqual(v['momentum'][1,:3].tolist(),[.5,.25,.125])
        self.assertEqual(v['spy_momentum'][1],.25);self.assertEqual(v['threshold'][1],.25)
        self.assertEqual({k:r[k] for k in audit.COUNT_FIELDS},dict(events=3,symbols_per_event=4,checked_cells=12,
            original_eligible_cells=11,weak_events=2,active_events=1,required_cells=3,source_price_observations=6,
            spy_price_observations=2,rejected_cells=2,candidate_eligible_cells=9,new_model_fits=0,predictive_model_fits=0))
        receipt=audit.verify_features(p,m,c,a,v,r)
        self.assertTrue(all(receipt[k] is True for k in audit.FEATURE_FLAGS));self.assertEqual(receipt['counts'],{k:r[k] for k in audit.COUNT_FIELDS})
        self.assertEqual(receipt['numerical_tolerance'],0.);self.assertFalse(receipt['computes_returns'])

    def test_nonpositive_spy_is_redundant_with_original_positive_E(self):
        for price in (50.,100.):
            args=fixture();frame,s,u=active_endpoints(args);frame.at[u,'SPY']=price
            v,_=audit.independent_features(*args)
            audit.exact_array(v['candidate_eligible'],args[3]['eligible'],'Redundant gate')
            self.assertEqual(v['threshold'][1],0.)

    def test_initial_partial_bypasses_but_regular_first_row_does_not(self):
        args=fixture();p,m,c,a=physical_prefix(args,1);p['adj']=p['adj'].iloc[:0]
        v,r=audit.independent_features(p,m,c,a);self.assertEqual(r['active_events'],0)
        self.assertTrue(np.isnan(v['momentum']).all())
        p,m,c,a=fixture();m={k:v[1:2] for k,v in m.items()};a={k:v[1:2] for k,v in a.items()}
        v,r=audit.independent_features(p,m,c,a);self.assertEqual(r['active_events'],1);self.assertEqual(r['rejected_cells'],2)

    def test_inactive_and_ineligible_sources_not_read(self):
        p,m,c,a=fixture();p['adj']=p['adj'].astype(object)
        frame,s,u=active_endpoints((p,m,c,a));frame.at[s,'D']='not numeric';frame.at[u,'D']=False
        audit.independent_features(p,m,c,a)
        a['state_weak_narrow'][:]=False;p['adj'].loc[:,:]='poison'
        v,_=audit.independent_features(p,m,c,a);audit.exact_array(v['candidate_eligible'],a['eligible'],'Inactive')

    def test_empty_E_still_requires_weak_market_evidence(self):
        p,m,c,a=fixture();a['eligible'][1]=False
        v,r=audit.independent_features(p,m,c,a);self.assertEqual(r['required_cells'],0);self.assertEqual(r['spy_price_observations'],2)
        p['adj']=p['adj'].drop(columns='SPY')
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,a)

    def test_missing_required_endpoint_and_nonreal_bad_prices_hard_fail(self):
        for bad in (0.,-1.,math.nan,math.inf,-math.inf,True,'100',None):
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);f,s,u=active_endpoints(args);f.at[u,'A']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();f,s,u=active_endpoints(args);args[0]['adj']=f.drop(index=s)
        with self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();args[0]['adj']=args[0]['adj'].drop(columns='A')
        with self.assertRaises(ValueError):audit.independent_features(*args)

    def test_original_E_reconstructed_nonpositive_momentum_is_integrity_failure(self):
        for value in (100.,75.):
            args=fixture();f,s,u=active_endpoints(args);f.at[u,'A']=value
            with self.assertRaisesRegex(ValueError,'Original E'):audit.independent_features(*args)

    def test_derived_overflow_and_canonical_float_equality(self):
        with self.assertRaises(FloatingPointError):audit.momentum(np.nextafter(0.,1.),np.finfo(float).max)
        self.assertEqual(audit.momentum(4.,5.),.25)
        args=fixture();f,s,u=active_endpoints(args);f.at[u,'A']=125.
        v,_=audit.independent_features(*args);self.assertFalse(v['candidate_eligible'][1,0])
        f.at[u,'A']=np.nextafter(125.,math.inf)
        v,_=audit.independent_features(*args);self.assertTrue(v['candidate_eligible'][1,0])

    def test_only_two_endpoints_not_new_full_path_admission(self):
        args=fixture();f,s,u=active_endpoints(args);interior=f.index[f.index.get_loc(s)+10]
        f.loc[interior,['A','SPY']]=math.nan
        v,_=audit.independent_features(*args);self.assertTrue(v['candidate_eligible'][1,0])

    def test_no_final42_or_future_numeric_dependency(self):
        args=fixture();base=audit.independent_features(*args)
        f,s,u=active_endpoints(args);f.loc[f.index>u,:]=math.nan
        changed=audit.independent_features(*args)
        for key in audit.VALUE_KEYS:audit.exact_array(base[0][key],changed[0][key],key)
        audit.record_check(base[1],changed[1])

    def test_all_physical_and_logical_prefixes(self):
        args=fixture();full=audit.independent_features(*args)
        for n in range(4):
            logical=audit.independent_features(*args,limit=n);physical=audit.independent_features(*physical_prefix(args,n))
            audit.record_check(logical[1],physical[1])
            for k in audit.VALUE_KEYS:
                audit.exact_array(logical[0][k],physical[0][k],k);audit.exact_array(logical[0][k],full[0][k][:n],k)
        args[1]['dates'][2]='future poison';args[1]['symbols'][2]=['bad']
        args[3]['eligible']=args[3]['eligible'].astype(object);args[3]['eligible'][2]='future poison'
        args[3]['state_weak_narrow']=args[3]['state_weak_narrow'].astype(object);args[3]['state_weak_narrow'][2]=1
        audit.independent_features(*args,limit=2)

    def test_schema_monthend_calendar_and_state_errors(self):
        for mutate in (
            lambda x:x[3].update(state_weak_narrow=np.ones((3,1),bool)),
            lambda x:x[3].update(eligible=np.ones((3,4),int)),
            lambda x:x[1]['symbols'].__setitem__(1,['A']*4),
            lambda x:x[1]['dates'].__setitem__(1,'2016-08-30'),
            lambda x:x[2].__setitem__(5,x[2][4]),
        ):
            args=fixture();mutate(args)
            with self.assertRaises(ValueError):audit.independent_features(*args)
        p,m,c,a=fixture()
        with self.assertRaises(ValueError):audit.independent_features(p,m,c,a,True)
        last=c.index(m['dates'][-1])
        with self.assertRaises(ValueError):audit.independent_features(p,m,c[:last+1],a)

    def test_no_inputs_changed_and_poisoned_receipts_rejected(self):
        args=fixture();before=copy.deepcopy(args);v,r=audit.independent_features(*args)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj']);self.assertEqual(args[1:3],before[1:3])
        for k in args[3]:audit.exact_array(args[3][k],before[3][k],k)
        for key in audit.VALUE_KEYS:
            wrong=copy.deepcopy(v)
            if wrong[key].dtype==bool:wrong[key].flat[0]=not wrong[key].flat[0]
            else:wrong[key].flat[0]=0.
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_features(*args,wrong,r)
        bad=copy.deepcopy(r);bad['rejected_cells']+=1
        with self.assertRaises(ValueError):audit.verify_features(*args,v,bad)


class SelectionTests(unittest.TestCase):
    def test_original_pre_and_new_boundary_match_frozen_selector_exactly(self):
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a)
        expected,choices,pres=audit.selection_oracle(a,m,v);study=original_production()
        policy=dict(id=7,family='level',model_id=2,budget=.05,ceiling=5,gap=.02);scores={};orders={};calls={}
        for arm in audit.ARMS:
            calls[arm]=0;gate=v['original_eligible'] if arm==audit.ARMS[0] else v['candidate_eligible']
            def selector(pre,originalE,names,effects,source,rule):
                calls[arm]+=1;audit.exact_array(originalE,a['eligible'],'Original selector E')
                audit.exact_array(pre,pres[arm],'Original pre')
                return study.ep.select_boundary(pre,gate.copy(),names,effects,source,rule)
            chosen,score=study.choices(study.configs()[4],a,m,3,policy=policy,boundary_selector=selector)
            self.assertEqual(chosen,choices[arm]);audit.exact_array(score,expected[arm],arm)
            orders[arm]=study.rp.make_orders(chosen,score,m);scores[arm]=score
        self.assertEqual(list(calls.values()),[1,1])
        receipt=audit.verify_selection(a,m,v,orders,scores=scores,pre_scores=pres)
        self.assertTrue(all(receipt[k] is True for k in audit.SELECTION_FLAGS));self.assertTrue(receipt['full_pre_scores_supplied'])
        self.assertFalse(np.array_equal(expected[audit.ARMS[0]][1],expected[audit.ARMS[1]][1]))
        self.assertEqual(choices[audit.ARMS[1]][1],[1,3,2])
        self.assertNotEqual([j for j in choices[audit.ARMS[0]][1] if v['candidate_eligible'][1,j]],choices[audit.ARMS[1]][1])

    def test_unchanged_gate_full_exact_and_order_prefixes(self):
        args=selection_fixture();f,s,u=active_endpoints(args);f.at[u,'SPY']=100.
        p,m,c,a=args;v,_=audit.independent_features(*args);scores,ch,pre=audit.selection_oracle(a,m,v)
        audit.exact_array(scores[audit.ARMS[0]],scores[audit.ARMS[1]],'Identity gate');self.assertEqual(ch[audit.ARMS[0]],ch[audit.ARMS[1]])
        for n in range(4):
            s2,c2,p2=audit.selection_oracle(a,m,v,n);oo=order_rows(s2,c2,{k:v[:n] for k,v in m.items()})
            audit.verify_selection(a,m,v,oo,scores=s2,pre_scores=p2,n=n)
            for arm in audit.ARMS:audit.exact_array(s2[arm],scores[arm][:n],arm)

    def test_two_halves_and_below_two_cash(self):
        for surviving in (0,1,2):
            args=selection_fixture();f,s,u=active_endpoints(args)
            for j,name in enumerate(args[1]['symbols'][1]):f.at[u,name]=150. if j<surviving else 112.5
            p,m,c,a=args;v,_=audit.independent_features(*args);scores,ch,pres=audit.selection_oracle(a,m,v)
            self.assertEqual(len(ch[audit.ARMS[1]][1]),2 if surviving==2 else 0)
            oo=order_rows(scores,ch,m);r=audit.verify_selection(a,m,v,oo,scores=scores,pre_scores=pres)
            self.assertEqual(r['holdings_counts'][audit.ARMS[1]][str(2 if surviving==2 else 0)],1)

    def test_original_model_rank_scope_survives_removed_candidate(self):
        p,m,c,a=selection_fixture();a['forecast_available'][:]=True;a['model_available'][:]=True
        a['raw_local_constant'][:]=[1.,2.,3.,4.,5.];a['raw_local_linear'][:]=[5.,1.,4.,2.,3.]
        a['raw_source_log'][:]=0.;v,_=audit.independent_features(p,m,c,a)
        scores,ch,pres=audit.selection_oracle(a,m,v)
        original=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        audit.exact_array(pres[audit.ARMS[1]],original['local'],'Original full ranking domain')
        mutated=copy.deepcopy(a);mutated['eligible']=v['candidate_eligible'].copy()
        wrong=audit.frozen_source_oracle().old.original_helper().original_prefix(mutated,m,3)
        self.assertFalse(np.array_equal(wrong['local'][1],pres[audit.ARMS[1]][1]))

    def test_score_pre_order_and_mask_tamper_rejected(self):
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a);s,ch,pr=audit.selection_oracle(a,m,v);o=order_rows(s,ch,m)
        for target in ('scores','pre_scores'):
            kwargs=dict(scores=copy.deepcopy(s),pre_scores=copy.deepcopy(pr));kwargs[target][audit.ARMS[1]][1,0]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,v,o,**kwargs)
        wrong=copy.deepcopy(o);wrong[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,wrong,scores=s,pre_scores=pr)
        wrong=copy.deepcopy(v);wrong['candidate_eligible'][1,0]=True
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,wrong)

    def test_no_producer_import_or_changed_pinned_primitive(self):
        self.assertNotIn('import monthly_relative_gate235_kernel',inspect.getsource(audit))
        with patch.object(audit,'ACCOUNTING_SHA','bad'),self.assertRaises(ValueError):audit.accounting()
        p,m,c,a=selection_fixture();v,_=audit.independent_features(p,m,c,a)
        with patch.object(audit,'SOURCE_SHA','bad'),self.assertRaises(ValueError):audit.selection_oracle(a,m,v)


class OutcomeTests(unittest.TestCase):
    def test_two_arms_six_paths_30_slices_and_bad_metric(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        receipt=audit.audit_outcomes(*args);self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(6,30))
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
        import monthly_relative_gate235_kernel as producer
        from test_monthly_relative_gate235_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            p,m,c,a=args;full=producer.build_features(*args);audit.verify_features(*args,*full)
            for n in range(len(a['eligible'])+1):
                logical=producer.build_features(*args,limit=n);audit.verify_features(*args,*logical,limit=n)
                small=physical_prefix(args,n);out=producer.build_features(*small);audit.verify_features(*small,*out)
                audit.record_check(logical[1],out[1])
                for key in audit.VALUE_KEYS:audit.exact_array(full[0][key][:n],out[0][key],key)


if __name__=='__main__':unittest.main(verbosity=2)
