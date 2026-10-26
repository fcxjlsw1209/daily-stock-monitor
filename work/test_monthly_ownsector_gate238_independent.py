"""Synthetic endpoints, source certificates and accounts only."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_ownsector_gate238_independent as audit


def source_report(meta, calendar, arrays, assignments=None):
    events=[];records=[];width=arrays['eligible'].shape[1]
    for i,(raw,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        partial=raw=='2016-08-17';t=calendar.index(raw)
        ids=np.flatnonzero(arrays['eligible'][i]).tolist();required=[] if partial else ids
        positions=[] if partial else [t-252,t-42]
        dates=[calendar[k] for k in positions]
        market=[dict(role=r,symbol='SPY',date=d,status='valid',reason=None)
                for r,d in zip(('market_start','market_end'),dates)]
        events.append(dict(query_index=i,date=raw,initial_partial=partial,active=not partial,
            original_eligible_indices=ids,required_indices=required,source_positions=positions,
            source_dates=dates,market_endpoints=market,required_count=len(required),
            admitted_count=len(required),gap_count=0))
        for j in required:
            name=names[j];etf=name in meta['etfs']
            benchmark='SPY' if etf else (assignments or {}).get(name,'XLK')
            old='XLC' if benchmark=='VOX' else benchmark
            origin='original_META.etfs_SPY' if etf else 'query_date_sector_panel'
            operation='full_period_communication_proxy' if old=='XLC' else 'unchanged'
            points=[dict(role=r,symbol=s,date=d,status='valid',reason=None) for r,s,d in (
                ('security_start',name,dates[0]),('security_end',name,dates[1]),
                ('benchmark_start',benchmark,dates[0]),('benchmark_end',benchmark,dates[1]))]
            records.append(dict(query_index=i,column_index=j,date=raw,symbol=name,is_etf=etf,
                benchmark=benchmark,original_benchmark=old,mapping_source=origin if operation=='unchanged' else operation,
                original_mapping_source=origin,mapping_repair=operation,mapping_evidence=None,
                mapping_status='available',endpoints=points,admitted=True,gap_reasons=[]))
    return dict(schema='ownsector238_repaired_source_report_v1',status='ADMITTED',admitted=True,
        counts=dict(symbols_per_event=width),events=events,records=records,gaps=[],source_only=True,
        momentum_computed=False,eligibility_changed=False,rankings_computed=False,
        orders_computed=False,returns_computed=False,model_fits=0,strict_pit=False)


def fixture():
    axis=pd.bdate_range('2015-01-01','2017-02-01');calendar=[str(d.date()) for d in axis]
    names=['A','B','C','FUND','UNUSED'];dates=['2016-08-17','2016-08-31','2016-09-30']
    meta=dict(dates=dates,symbols=[names.copy() for _ in dates],etfs=['FUND'],
        ends=['2016-08-31','2016-09-30','2016-10-31'])
    arrays=dict(eligible=np.array([[True]*5,[True]*4+[False],[True]*4+[False]],bool),
                state_weak_narrow=np.array([True,True,False]))
    panel=dict(adj=pd.DataFrame(100.,index=axis,columns=names+['SPY','XLK','XLP']),
               vox=pd.DataFrame({'VOX':np.full(len(axis),100.)},index=axis))
    for d in dates[1:]:
        t=calendar.index(d);u=calendar[t-42]
        panel['adj'].loc[u,names+['SPY','XLK','XLP']]=[125.,150.,125.,150.,100.,137.5,112.5,175.]
        panel['vox'].loc[u,'VOX']=125.
    report=source_report(meta,calendar,arrays,{'A':'XLK','B':'XLP','C':'VOX'})
    return panel,meta,calendar,arrays,report


def physical_prefix(args,n):
    panel,meta,calendar,arrays,report=copy.deepcopy(args)
    mm={k:(v[:n] if k in ('dates','ends','symbols') else v) for k,v in meta.items()}
    aa={k:v[:n].copy() for k,v in arrays.items()}
    report['events']=report['events'][:n]
    report['records']=[r for r in report['records'] if r['query_index']<n]
    if n:
        cutoff=meta['dates'][n-1];pp={k:v.loc[:cutoff].copy() for k,v in panel.items()}
        cc=calendar[:calendar.index(cutoff)+2]
    else:pp={k:v.iloc[:0].copy() for k,v in panel.items()};cc=[]
    return pp,mm,cc,aa,report


def endpoints(args,row=1):
    calendar=args[2];t=calendar.index(args[1]['dates'][row])
    return pd.Timestamp(calendar[t-252]),pd.Timestamp(calendar[t-42])


def selection_fixture():
    panel,meta,calendar,_,_=fixture();names=['A','B','C','D','E'];n,w=3,5;shape=(n,w)
    meta['symbols']=[names.copy() for _ in meta['dates']];meta['etfs']=[]
    panel['adj']=pd.DataFrame(100.,index=panel['adj'].index,columns=names+['XLK'])
    pre=np.tile([.24,.18,.10,.095,.09],(n,1));z=np.zeros(shape);f=np.zeros(shape,bool)
    arrays=dict(baseline=pre.copy(),core=np.zeros(shape),core_weight=np.full((n,1),.5),
        eligible=np.ones(shape,bool),effects=np.tile([0.,0.,-.4,.4,.5],(n,1)),
        state_calm10=np.ones(n,bool),state_weak_narrow=np.array([True,True,False]),pre_score=pre.copy(),
        old_ml4=z.copy(),rank_log_excess=z.copy(),extension_available=f.copy(),forecast_available=f.copy(),
        local_available=f.copy(),model_available=f.copy(),raw_local_constant=z.copy(),raw_local_linear=z.copy(),raw_source_log=z.copy())
    for row in (1,2):
        s,u=endpoints((panel,meta,calendar,arrays),row)
        panel['adj'].loc[u,names+['XLK']]=[112.5,150.,150.,150.,150.,125.]
    return panel,meta,calendar,arrays,source_report(meta,calendar,arrays)


def order_rows(scores,choices,meta):
    return {arm:[dict(date=meta['dates'][i],end=meta['ends'][i],signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids],scores=[float(scores[arm][i,j]) for j in ids])
        for i,ids in enumerate(choices[arm])] for arm in audit.ARMS}


def original_production():
    path=audit.ROOT/'outputs/monthly_local_linear70/study.py'
    spec=importlib.util.spec_from_file_location('ownsector238_test_original',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FeatureTests(unittest.TestCase):
    def test_hand_known_ownsector_not_dual_spy_gate_and_counts(self):
        args=fixture();v,r=audit.independent_features(*args)
        self.assertEqual(v['candidate_eligible'].tolist(),[[True]*5,[True,False,False,True,False],[True,False,False,True,False]])
        self.assertEqual(v['momentum'][1,:4].tolist(),[.25,.5,.25,.5])
        self.assertEqual(v['benchmark_momentum'][1,:4].tolist(),[.125,.75,.25,.375])
        self.assertTrue(v['candidate_eligible'][1,0])  # A lags SPY but beats its own sector.
        self.assertFalse(v['candidate_eligible'][1,1])  # B beats SPY but lags its own sector.
        self.assertEqual({k:r[k] for k in audit.COUNT_FIELDS},dict(events=3,symbols_per_event=5,checked_cells=15,
            original_eligible_cells=13,weak_events=2,active_events=2,required_cells=8,stock_required_cells=6,
            etf_required_cells=2,source_price_observations=16,benchmark_price_observations=16,
            spy_price_observations=4,rejected_cells=4,candidate_eligible_cells=9,new_model_fits=0,predictive_model_fits=0))
        receipt=audit.verify_features(*args,v,r)
        self.assertTrue(all(receipt[k] is True for k in audit.FEATURE_FLAGS))
        self.assertEqual(len(r['cell_history']),8)

    def test_etf_dispatch_synthetic237_parity(self):
        import monthly_allstate_gate237_independent as prior
        p,m,c,a,report=fixture();v,_=audit.independent_features(p,m,c,a,report)
        before,_=prior.independent_features({'adj':p['adj']},m,c,a)
        for row in (1,2):
            self.assertEqual(v['candidate_eligible'][row,3],before['candidate_eligible'][row,3])
            self.assertEqual(v['benchmark_momentum'][row,3],before['spy_momentum'][row])

    def test_state_flags_are_descriptive_only(self):
        args=fixture();base=audit.independent_features(*args)
        for state in (False,True):
            other=copy.deepcopy(args);other[3]['state_weak_narrow'][:]=state
            other[3]['state_calm10']=np.full(3,not state,bool)
            v,r=audit.independent_features(*other)
            for k in audit.VALUE_KEYS:audit.exact_array(v[k],base[0][k],k)
            self.assertEqual(r['active_events'],2);self.assertEqual(r['weak_events'],3*int(state))

    def test_partial_empty_and_stock_only_do_not_read_spy(self):
        args=physical_prefix(fixture(),1);args[0]['adj']=args[0]['adj'].iloc[:0];args[0]['vox']=args[0]['vox'].iloc[:0]
        v,r=audit.independent_features(*args);self.assertEqual(r['required_cells'],0)
        p,m,c,a,_=fixture();a['eligible'][1:]=False;p['adj']=p['adj'].drop(columns='SPY')
        report=source_report(m,c,a)
        v,r=audit.independent_features(p,m,c,a,report);self.assertEqual(r['benchmark_price_observations'],0)
        args=selection_fixture();v,r=audit.independent_features(*args)
        self.assertNotIn('SPY',args[0]['adj'].columns);self.assertEqual(r['spy_price_observations'],0)

    def test_regular_first_row_still_active(self):
        p,m,c,a,_=fixture();m={k:(v[1:2] if k in ('dates','ends','symbols') else v) for k,v in m.items()}
        a={k:v[1:2] for k,v in a.items()};report=source_report(m,c,a,{'A':'XLK','B':'XLP','C':'VOX'})
        _,r=audit.independent_features(p,m,c,a,report);self.assertEqual(r['active_events'],1)

    def test_nonpositive_benchmark_preserves_positive_E(self):
        for value in (75.,100.):
            args=fixture()
            for row in (1,2):
                s,u=endpoints(args,row);args[0]['adj'].loc[u,['SPY','XLK','XLP']]=value;args[0]['vox'].loc[u,'VOX']=value
            v,_=audit.independent_features(*args);audit.exact_array(v['candidate_eligible'],args[3]['eligible'],'Redundant')

    def test_strict_tie_adjacent_float_and_overflow(self):
        args=fixture();s,u=endpoints(args);args[0]['adj'].at[u,'A']=112.5
        self.assertFalse(audit.independent_features(*args)[0]['candidate_eligible'][1,0])
        args[0]['adj'].at[u,'A']=np.nextafter(112.5,math.inf)
        self.assertTrue(audit.independent_features(*args)[0]['candidate_eligible'][1,0])
        with self.assertRaises(FloatingPointError):audit.momentum(np.nextafter(0.,1.),np.finfo(float).max)

    def test_required_endpoint_invalid_and_original_positive_parity(self):
        for name,value in [('A',100.),('A',0.),('A',True),('XLK',np.nan),('XLK',math.inf),('XLK','112.5')]:
            args=fixture();args[0]['adj']=args[0]['adj'].astype(object);s,u=endpoints(args)
            args[0]['adj'].at[u,name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):audit.independent_features(*args)

    def test_dedicated_vox_no_fallback(self):
        args=fixture();s,u=endpoints(args);args[0]['adj']['VOX']=150.
        args[0]['vox']=args[0]['vox'].drop(index=u)
        with self.assertRaises(ValueError):audit.independent_features(*args)
        args=fixture();args[0]['vox'].columns=['XLC']
        with self.assertRaises(ValueError):audit.independent_features(*args)

    def test_only_required_endpoints_no_recent_or_ineligible_source(self):
        args=physical_prefix(fixture(),2);base=audit.independent_features(*args);s,u=endpoints(args)
        for frame in args[0].values():frame.loc[(frame.index>s)&(frame.index!=u),:]=np.nan
        args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj']['UNUSED']='poison'
        after=audit.independent_features(*args)
        for k in audit.VALUE_KEYS:audit.exact_array(base[0][k],after[0][k],k)
        audit.record_check(base[1],after[1])

    def test_report_missing_extra_duplicate_identity_clock_or_state_fails(self):
        for mutate in (
            lambda r:r.update(admitted=False),lambda r:r.update(status='NOT_ADMITTED'),
            lambda r:r['records'].pop(),lambda r:r['records'].append(copy.deepcopy(r['records'][0])),
            lambda r:r['records'][0].update(symbol='WRONG'),lambda r:r['records'][0].update(is_etf=True),
            lambda r:r['records'][0].update(column_index=99),
            lambda r:r['records'][0]['endpoints'][0].update(date='2015-01-02'),
            lambda r:r['events'][1].update(source_positions=[0,210]),
            lambda r:r['events'][1].update(required_indices=[0]),
            lambda r:r['events'][1]['market_endpoints'][0].update(status='missing'),
            lambda r:r['records'][0].update(query_index=True),
        ):
            args=fixture();mutate(args[4])
            with self.assertRaises(ValueError):audit.independent_features(*args)

    def test_mapping_provenance_tamper_and_nonmissing_preservation(self):
        for mutate in (
            lambda r:r.update(benchmark='SPY'),lambda r:r.update(mapping_source='now'),
            lambda r:r.update(mapping_evidence=['new']),lambda r:r.update(original_benchmark='VOX'),
            lambda r:r.update(original_mapping_source='unverified'),
        ):
            args=fixture();mutate(args[4]['records'][0])
            with self.assertRaises(ValueError):audit.independent_features(*args)

    def test_fixed_six_dated_business_provenance(self):
        for name,(bench,start,end,evidence) in audit.BUSINESS_RULES.items():
            row=dict(is_etf=False,original_benchmark=None,original_mapping_source='query_date_sector_panel',
                benchmark=bench,mapping_repair='dated_business_proxy',mapping_source='dated_business_proxy',mapping_evidence=evidence)
            self.assertEqual(audit._assignment(row,False,start,name),bench)
            with self.assertRaises(ValueError):audit._assignment(row,False,'2000-01-01',name)
            if end:
                with self.assertRaises(ValueError):audit._assignment(row,False,end,name)
            row['mapping_evidence']=['wrong']
            with self.assertRaises(ValueError):audit._assignment(row,False,start,name)

    def test_all_logical_physical_prefixes_future_report_poison(self):
        args=fixture();full=audit.independent_features(*args)
        for n in range(4):
            logical=audit.independent_features(*args,limit=n);physical=audit.independent_features(*physical_prefix(args,n))
            audit.record_check(logical[1],physical[1])
            for k in audit.VALUE_KEYS:audit.exact_array(logical[0][k],physical[0][k],k)
        args[1]['dates'][2]='future poison';args[1]['symbols'][2]=['bad']
        args[3]['eligible']=args[3]['eligible'].astype(object);args[3]['eligible'][2]='future poison'
        args[4]['events'][2]={'future':'ignored'}
        for row in args[4]['records']:
            if row['query_index']==2:row.clear();row.update(query_index=2,unknown_future='poison')
        logical=audit.independent_features(*args,limit=2)
        for k in audit.VALUE_KEYS:audit.exact_array(logical[0][k],full[0][k][:2],k)

    def test_metadata_calendar_bad_shapes_and_zero_prefix(self):
        for mutate in (
            lambda x:x[1].update(etfs=['FUND','FUND']),
            lambda x:x[3].update(eligible=np.ones((3,5),int)),
            lambda x:x[3].update(state_weak_narrow=np.ones((3,1),bool)),
            lambda x:x[1]['dates'].__setitem__(1,'2016-08-30'),
            lambda x:x[2].__setitem__(5,x[2][4]),
        ):
            args=fixture();mutate(args)
            with self.assertRaises(ValueError):audit.independent_features(*args)
        args=physical_prefix(fixture(),0);v,r=audit.independent_features(*args)
        self.assertEqual(v['threshold'].shape,(0,5));self.assertEqual(r['cell_history'],[])

    def test_input_immutability_and_feature_receipt_tamper(self):
        args=fixture();before=copy.deepcopy(args);v,r=audit.independent_features(*args)
        for k in args[0]:pd.testing.assert_frame_equal(args[0][k],before[0][k])
        self.assertEqual(args[1],before[1]);self.assertEqual(args[4],before[4])
        for k in args[3]:audit.exact_array(args[3][k],before[3][k],k)
        for key in audit.VALUE_KEYS:
            wrong=copy.deepcopy(v);wrong[key].flat[0]=False if wrong[key].dtype==bool else 0.
            if key=='required':wrong[key].flat[0]=True
            with self.subTest(key=key),self.assertRaises(ValueError):audit.verify_features(*args,wrong,r)
        wrong=copy.deepcopy(r);wrong['benchmark_price_observations']+=2
        with self.assertRaises(ValueError):audit.verify_features(*args,v,wrong)


class SelectionTests(unittest.TestCase):
    def test_original_pre_and_single_episode_exact(self):
        args=selection_fixture();p,m,c,a,_=args;v,_=audit.independent_features(*args)
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
        self.assertEqual(list(calls.values()),[1,1]);self.assertEqual(choices[audit.ARMS[1]][1],[1,3,2])
        receipt=audit.verify_selection(a,m,v,orders,scores=scores,pre_scores=pres)
        self.assertTrue(all(receipt[k] is True for k in audit.SELECTION_FLAGS))
        self.assertTrue(receipt['full_pre_scores_supplied'])

    def test_order_prefixes_unchanged_gate_and_cash_cases(self):
        for surviving in (0,1,2,5):
            args=selection_fixture();s,u=endpoints(args)
            for j,name in enumerate(args[1]['symbols'][1]):args[0]['adj'].at[u,name]=150. if j<surviving else 112.5
            _,m,_,a,_=args;v,_=audit.independent_features(*args);scores,ch,pres=audit.selection_oracle(a,m,v)
            self.assertEqual(len(ch[audit.ARMS[1]][1]),3 if surviving==5 else 2 if surviving==2 else 0)
            for n in range(4):
                ss,cc,pp=audit.selection_oracle(a,m,v,n);oo=order_rows(ss,cc,m)
                audit.verify_selection(a,m,v,oo,scores=ss,pre_scores=pp,n=n)
                for arm in audit.ARMS:audit.exact_array(ss[arm],scores[arm][:n],arm)

    def test_original_model_ranking_domain_not_shrunk(self):
        args=selection_fixture();_,m,_,a,_=args
        a['forecast_available'][:]=True;a['model_available'][:]=True
        a['raw_local_constant'][:]=[1.,2.,3.,4.,5.];a['raw_local_linear'][:]=[5.,1.,4.,2.,3.]
        v,_=audit.independent_features(*args);_,_,pres=audit.selection_oracle(a,m,v)
        original=audit.frozen_source_oracle().old.original_helper().original_prefix(a,m,3)
        audit.exact_array(pres[audit.ARMS[1]],original['local'],'Original ranking domain')
        bad=copy.deepcopy(a);bad['eligible']=v['candidate_eligible'].copy()
        mutated=audit.frozen_source_oracle().old.original_helper().original_prefix(bad,m,3)
        self.assertFalse(np.array_equal(mutated['local'][1],pres[audit.ARMS[1]][1]))

    def test_evidence_masks_pre_scores_scores_and_orders_tamper(self):
        args=selection_fixture();_,m,_,a,_=args;v,_=audit.independent_features(*args)
        scores,ch,pres=audit.selection_oracle(a,m,v);orders=order_rows(scores,ch,m)
        for target in ('scores','pre_scores'):
            supplied=dict(scores=copy.deepcopy(scores),pre_scores=copy.deepcopy(pres))
            supplied[target][audit.ARMS[1]][1,0]+=.001
            with self.assertRaises(ValueError):audit.verify_selection(a,m,v,orders,**supplied)
        bad=copy.deepcopy(v);bad['threshold'][1,0]=0.
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,bad)
        bad=copy.deepcopy(v);bad['benchmark_momentum']=np.zeros(3)
        with self.assertRaises(ValueError):audit.selection_oracle(a,m,bad)
        bad=copy.deepcopy(orders);bad[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError):audit.verify_selection(a,m,v,bad,scores=scores,pre_scores=pres)

    def test_no_new_producer_imports_and_pinned_dependencies(self):
        tree=ast.parse(Path(audit.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('monthly_ownsector_gate238_kernel',imports)
        self.assertNotIn('monthly_ownsector_gate238_host',imports)
        with patch.object(audit,'ACCOUNTING_SHA','bad'),self.assertRaises(ValueError):audit.accounting()


class OutcomeTests(unittest.TestCase):
    def test_six_paths_thirty_slices_and_fee_metric_tamper(self):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):args=helper.account_fixture()
        receipt=audit.audit_outcomes(*args)
        self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(6,30))
        for path,key in [('same_close_zero','cagr'),('next_open_cost','fees_paid')]:
            bad=copy.deepcopy(args);bad[4][audit.ARMS[1]][path]['metrics'][key]+=1.
            with self.assertRaises((ValueError,AssertionError)):audit.audit_outcomes(*bad)


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixtures_all_prefixes_and_full_audit(self):
        import monthly_ownsector_gate238_kernel as producer
        from test_monthly_ownsector_gate238_kernel import fixture as other_fixture
        for args in (fixture(),other_fixture()):
            full=producer.build_features(*args);audit.verify_features(*args,*full)
            for n in range(len(args[1]['dates'])+1):
                logical=producer.build_features(*args,limit=n);audit.verify_features(*args,*logical,limit=n)
                small=physical_prefix(args,n);out=producer.build_features(*small);audit.verify_features(*small,*out)
                audit.record_check(logical[1],out[1])
                for k in audit.VALUE_KEYS:audit.exact_array(full[0][k][:n],out[0][k],k)


if __name__=='__main__':unittest.main(verbosity=2)
