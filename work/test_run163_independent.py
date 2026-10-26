"""Synthetic-only independent163 tests; no producer imports or actual numerical data."""
from copy import deepcopy
import ast
from datetime import date,timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import run163_independent_prepare as p
import run163_independent_outcome as o
import run163_independent as f


def feature_fixture(count=3,width=3):
    index=pd.bdate_range('2014-01-01','2026-10-02')
    calendar=[str(d.date()) for d in index]
    monthends=[calendar[i] for i in range(len(calendar)-1) if calendar[i][:7]!=calendar[i+1][:7]]
    dates=(['2016-08-17']+[d for d in monthends if '2016-08-31'<=d<='2026-08-31'])[:count]
    names=['S%d'%i for i in range(width)]
    meta=dict(dates=dates,symbols=[names[:] for _ in dates])
    x=np.arange(len(index)-1)
    paths={}
    for j,name in enumerate(names):
        returns=.002*np.sin(x*(j+1)*.017)+.001*np.cos(x*.003*(j+2))
        paths[name]=np.r_[100.,100*np.cumprod(1+returns)]
    return meta,calendar,dict(dates=[],symbols=[]),pd.DataFrame(paths,index=index)


def block_prices(signs):
    """Integer-valued synthetic endpoints; arbitrary valid interiors are immaterial."""
    assert len(signs)==10
    endpoints=[100.]
    for sign in signs:endpoints.append(endpoints[-1]+sign)
    prices=np.full(211,100.)
    prices[::21]=endpoints
    return prices


class FeatureTests(unittest.TestCase):
    def test_terminal_lengths_and_zero_sign(self):
        for sign in (-1,1):
            for length in range(1,11):
                signs=[-sign]*(10-length)+[sign]*length
                self.assertEqual(p.run_oracle(block_prices(signs)),sign*length/10)
                if length<10:
                    signs[-length-1]=0
                    self.assertEqual(p.run_oracle(block_prices(signs)),sign*length/10)
        self.assertEqual(p.run_oracle(block_prices([1]*9+[0])),0.)
        self.assertEqual(p.run_oracle(block_prices([0]*10)),0.)
    def test_terminal_only_not_longest_or_total_count(self):
        self.assertEqual(p.run_oracle(block_prices([1]*6+[-1,-1,1,1])),.2)
        self.assertEqual(p.run_oracle(block_prices([-1]*6+[1,1,-1,-1])),-.2)
        rng=random.Random(163)
        for _ in range(400):
            signs=[rng.choice((-1,0,1)) for _ in range(10)]
            last=signs[-1]
            first=9
            while first>0 and signs[first-1]==last:first-=1
            expected=last*(10-first)/10 if last else 0.
            self.assertEqual(p.run_oracle(block_prices(signs)),expected)
    def test_endpoints_and_21_session_block_boundaries(self):
        prices=block_prices([1]*7+[-1,1,1])
        self.assertEqual(p.run_oracle(prices),.2)
        prices[np.arange(211)%21!=0]=1e200
        self.assertEqual(p.run_oracle(prices),.2)
        prices[189]=prices[210]
        self.assertEqual(p.run_oracle(prices),0.)
    def test_exact_price_count_and_each_of211_values_required(self):
        prices=block_prices([1]*10)
        for bad in (prices[:-1],np.r_[prices,101.],[]):self.assertIsNone(p.run_oracle(bad))
        for i in range(211):
            for value in (0.,-1.,np.nan,np.inf,-np.inf):
                changed=prices.copy();changed[i]=value
                self.assertIsNone(p.run_oracle(changed),(i,value))
    def test_endpoint_signs_avoid_floating_division_overflow_and_underflow(self):
        prices=np.full(211,np.finfo(float).tiny)
        prices[::21]=[np.finfo(float).max if j%2 else np.nextafter(0.,1.) for j in range(11)]
        self.assertEqual(p.run_oracle(prices),-.1)
        # Adjacent representable prices still count; no return epsilon is allowed.
        endpoints=[1.]
        for _ in range(10):endpoints.append(np.nextafter(endpoints[-1],np.inf))
        prices[::21]=endpoints
        self.assertEqual(p.run_oracle(prices),1.)
    def test_midranks_ties_singleton_and_missing(self):
        self.assertEqual(p.midranks([2.,1.,1.,None]),[.5,0.,0.,None])
        self.assertEqual(p.midranks([None,7.,None]),[None,.5,None])
        self.assertEqual(p.midranks([None]),[None])
    def test_initial_records_measured_values_but_forces_positive_zero(self):
        got=p.features_oracle(*feature_fixture());row=got['audit']['event_history'][0]
        self.assertTrue(all(x is not None for x in row['raw_run']))
        self.assertEqual(row['common_support'],[False]*3)
        for value in got['arrays'].values():
            self.assertTrue(np.all(value[0]==0));self.assertFalse(np.signbit(value[0]).any())
        self.assertEqual(got['audit']['common_support_cells'],6)
    def test_exact_window_and_skipped_recent_prices(self):
        args=list(feature_fixture());got=p.features_oracle(*args);day=args[0]['dates'][1];t=args[1].index(day)
        row=got['audit']['event_history'][1]
        self.assertEqual((row['window_start'],row['window_end']),(args[1][t-252],args[1][t-42]))
        changed=args[3].copy();changed.loc[changed.index>pd.Timestamp(args[1][t-42])]=np.nan;args[3]=changed
        prefix=p.features_oracle(*args,n=2)
        for key in p.EXTRA:self.assertEqual(prefix['arrays'][key].tobytes(),got['arrays'][key][:2].tobytes())
        p.compare(prefix['audit']['event_history'],got['audit']['event_history'][:2])
    def test_all123_feature_prefixes(self):
        args=feature_fixture(122,1);full=p.features_oracle(*args)
        for n in range(123):
            prefix=p.features_oracle(*args,n=n)
            for key in p.EXTRA:self.assertEqual(prefix['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
            p.compare(prefix['audit']['event_history'],full['audit']['event_history'][:n])
    def test_future_memberships_and_unused_training_have_no_effect(self):
        args=list(feature_fixture());base=p.features_oracle(*args,n=2)
        args[0]=deepcopy(args[0]);args[0]['symbols'][2]=['X','Y','Z'];args[2]={'poison':object()}
        p.compare(p.features_oracle(*args,n=2)['audit'],base['audit'])
    def test_nonpositive_nonfinite_missing_symbol_or_session_neutral(self):
        args=feature_fixture();t=args[1].index(args[0]['dates'][1]);day=args[1][t-100]
        for bad in (0.,-1.,np.nan,np.inf,-np.inf):
            px=args[3].copy();px.at[pd.Timestamp(day),'S0']=bad
            got=p.features_oracle(*args[:3],px);self.assertIsNone(got['audit']['event_history'][1]['raw_run'][0])
            for value in got['arrays'].values():self.assertEqual(value[1,0],0.)
        for px in (args[3].drop(columns='S0'),args[3].drop(index=pd.Timestamp(day))):
            got=p.features_oracle(*args[:3],px);self.assertEqual(got['arrays'][p.EXTRA[0]][1,0],0.)
    def test_no_spy_dependency_and_flat_series_valid_zero_run(self):
        args=feature_fixture();base=p.features_oracle(*args)
        self.assertNotIn('SPY',args[3].columns)
        for replacement in (100.,0.,np.nan,np.inf):
            px=args[3].copy();px['SPY']=replacement
            p.compare(p.features_oracle(*args[:3],px)['audit'],base['audit'])
        px=args[3].copy();px[:]=100.
        got=p.features_oracle(*args[:3],px);row=got['audit']['event_history'][1]
        self.assertEqual(row['raw_run'],[0.,0.,0.])
        self.assertEqual(row['common_support'],[True]*3)
        self.assertEqual(row['ranks'],[2/3-.5]*3)
        np.testing.assert_allclose(got['arrays'][p.EXTRA[1]][1],[.02/6]*3,rtol=0,atol=1e-18)
    def test_stock_etf_same_treatment_and_current_pool_ties(self):
        args=list(feature_fixture());args[0]['etfs']=['S1'];px=args[3].copy();px['S1']=px['S0']
        row=p.features_oracle(*args[:3],px)['audit']['event_history'][1]
        self.assertEqual(row['raw_run'][0],row['raw_run'][1]);self.assertEqual(row['ranks'][0],row['ranks'][1])
    def test_incomplete_calendar_history_and_extra_offcalendar_prices(self):
        args=list(feature_fixture());args[1]=[x for x in args[1] if x>='2016-08-01']
        got=p.features_oracle(*args)
        self.assertIsNone(got['audit']['event_history'][0]['window_start']);self.assertIsNone(got['audit']['event_history'][0]['window_end'])
        self.assertFalse(got['arrays'][p.EXTRA[0]].any())
        args=list(feature_fixture());extra=pd.DataFrame(7.,index=[pd.Timestamp('2016-08-20')],columns=args[3].columns)
        got=p.features_oracle(*args);args[3]=pd.concat([args[3],extra]).sort_index()
        p.compare(p.features_oracle(*args)['audit'],got['audit'])
    def test_scale_column_permutation_and_exact_duplicate_tie(self):
        args=feature_fixture();base=p.features_oracle(*args)
        px=(args[3]*2)[list(reversed(args[3].columns))]
        p.compare(p.features_oracle(*args[:3],px)['audit'],base['audit'])
        px=args[3].copy();px['S1']=px['S0'];row=p.features_oracle(*args[:3],px)['audit']['event_history'][1]
        self.assertEqual(row['ranks'][0],row['ranks'][1])
    def test_duplicate_chronology_and_membership_schema(self):
        args=feature_fixture()
        for px in (args[3].iloc[::-1],pd.concat([args[3],args[3].tail(1)]),args[3].rename(columns={'S1':'S0'})):
            with self.assertRaises(ValueError):p.features_oracle(*args[:3],px)
        for dates in (list(reversed(args[0]['dates'])),[args[0]['dates'][0]]*3):
            meta=deepcopy(args[0]);meta['dates']=dates
            with self.assertRaises(ValueError):p.features_oracle(meta,*args[1:])
        meta=deepcopy(args[0]);meta['symbols'][1][1]='S0'
        with self.assertRaises(ValueError):p.features_oracle(meta,*args[1:])
        with self.assertRaises(ValueError):p.features_oracle(*args,n=True)
    def test_official_calendar_nonmonthend_and_bad_axis_rejected(self):
        args=feature_fixture()
        calendar=sorted(args[1]+['2016-08-20'])
        with self.assertRaises(ValueError):p.features_oracle(args[0],calendar,*args[2:])
        meta=deepcopy(args[0]);meta['dates'][1]='2016-08-30'
        with self.assertRaisesRegex(ValueError,'month-end'):p.features_oracle(meta,*args[1:])
        with self.assertRaises(ValueError):p.features_oracle(*args[:3],args[3].rename(columns={'S0':7}))
    def test_current_membership_ranking_and_original_eligibility_ignored(self):
        args=list(feature_fixture());base=p.features_oracle(*args)
        args[0]=deepcopy(args[0]);args[0]['eligible']=[[False]*3 for _ in args[0]['dates']]
        p.compare(p.features_oracle(*args)['audit'],base['audit'])
        args[0]['symbols']=[['S0','ABSENT1','ABSENT2'] for _ in args[0]['dates']]
        got=p.features_oracle(*args);row=got['audit']['event_history'][1]
        self.assertEqual(row['ranks'],[.5,None,None]);self.assertEqual(got['arrays'][p.EXTRA[1]][1,0],.01)
    def test_neutral_zero_rank_must_be_exact_even_inside_tolerance(self):
        got=p.features_oracle(*feature_fixture());row=got['audit']['event_history'][1]
        row['ranks'][0]=0.;got['arrays'][p.EXTRA[0]][1,0]=0.;got['arrays'][p.EXTRA[1]][1,0]=0.
        other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][1,0]=-1e-20;other[p.EXTRA[1]][1,0]=1e-20
        with self.assertRaisesRegex(ValueError,'Exact zero-rank'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_saved_states_raw_audit_and_delta_tampering(self):
        got=p.features_oracle(*feature_fixture());states=p.state_values(got['audit'])
        p.verify_feature_output(got,got['arrays'],got['audit'],states)
        states['common_support'][1][0]=False
        with self.assertRaises(ValueError):p.verify_feature_output(got,got['arrays'],got['audit'],states)
        states=p.state_values(got['audit']);bad=deepcopy(got['audit']);bad['event_history'][1]['raw_run'][0]+=.001
        with self.assertRaises(ValueError):p.verify_feature_output(got,got['arrays'],bad,states)
        for target,value in (((0,0),1e-20),((1,0),.0100000000001)):
            other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][target]=value
            with self.assertRaises((ValueError,AssertionError)):p.verify_feature_output(got,other,got['audit'],states)
    def test_complete_support_counts_and_corruptions(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args)
        support=o.support_oracle(got['audit'],args[0],got['arrays'])
        self.assertEqual(support['candidate_cells'],5978);self.assertIsNone(support['coverage_threshold'])
        self.assertEqual(support['common_support_cells'],121*49)
        self.assertEqual(support['coverage']['early53']['common_support_cells'],52*49)
        for field in ('common_support','ranks'):
            bad=deepcopy(got['audit']);bad['event_history'][1][field][0]=None
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        for value in (.05,1.1,-1.1,True):
            bad=deepcopy(got['audit']);bad['event_history'][1]['raw_run'][0]=value
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        bad={k:v.copy() for k,v in got['arrays'].items()};bad[p.EXTRA[0]][0,0]=.001
        with self.assertRaises(ValueError):o.support_oracle(got['audit'],args[0],bad)


def score_fixture():
    shape=(2,49);baseline=np.tile(np.linspace(.20,.10,49),(2,1))
    arrays=dict(baseline=baseline,core=.5*baseline,core_weight=np.full(2,.5),eligible=np.ones(shape,bool),
        state_weak_narrow=np.zeros(2,bool),state_calm10=np.ones(2,bool),pre_score=baseline.copy(),
        effects=np.zeros(shape),forecast_available=np.zeros(shape,bool),rank_log_excess=np.zeros(shape),
        old_ml4=np.zeros(shape),extension_available=np.zeros(shape,bool),model_available=np.zeros(shape,bool),
        raw_local_constant=np.zeros(shape),raw_local_linear=np.zeros(shape),raw_source_log=np.zeros(shape))
    names=['S'+str(i) for i in range(49)]
    meta=dict(symbols=[names[:],names[:]],etfs=names[-17:],dates=['2016-08-17','2016-08-31'],ends=['2016-08-31','2016-09-30'])
    hands={key:np.zeros(shape) for key in p.HANDS}
    hands[p.HANDS[0]][1]=np.linspace(-.01,.01,49)
    hands[p.HANDS[1]][1]=np.linspace(.01,-.01,49)
    return arrays,meta,hands


class ScoreTests(unittest.TestCase):
    def test_all123_score_prefixes(self):
        source,meta,hands=score_fixture();meta['dates']=feature_fixture(122,1)[0]['dates']
        meta['ends']=meta['dates'][1:]+['2026-09-30'];meta['symbols']=[meta['symbols'][0][:] for _ in range(122)]
        arrays={k:np.concatenate([v]*61,axis=0) for k,v in source.items()}
        hands={k:np.concatenate([v]*61,axis=0) for k,v in hands.items()}
        full=p.score_oracle(arrays,meta,hands)
        for n in range(123):
            part=p.score_oracle(arrays,meta,hands,n)
            for name in p.NAMES:
                self.assertEqual(part['scores'][name].tobytes(),full['scores'][name][:n].tobytes())
                self.assertEqual(part['choices'][name],full['choices'][name][:n])
    def test_delta_once_original_episode_and_top2(self):
        arrays,meta,hands=score_fixture();result=p.score_oracle(arrays,meta,hands)
        for name,key in zip(p.NAMES[1:],p.HANDS):
            np.testing.assert_array_equal(result['deltas'][name],hands[key])
            modified=arrays['pre_score'][1]+hands[key][1]
            top=sorted(range(49),key=lambda j:(-modified[j],meta['symbols'][1][j]))[:2]
            self.assertEqual(result['episodes'][name][1]['top2'],[meta['symbols'][1][j] for j in top])
            self.assertTrue(set(top).issubset(result['choices'][name][1]))
    def test_exact_ordered_holdings_fail_on_swapped_or_changed_score(self):
        arrays,meta,hands=score_fixture();orders=p.order_oracle(meta,p.score_oracle(arrays,meta,hands))
        p.verify_orders(orders,orders)
        swapped=deepcopy(orders);swapped[p.NAMES[1]][1]['symbols'].reverse()
        with self.assertRaisesRegex(ValueError,'Exact ordered holdings'):p.verify_orders(swapped,orders)
        changed=deepcopy(orders);changed[p.NAMES[2]][1]['scores'][0]+=1e-8
        with self.assertRaises(AssertionError):p.verify_orders(changed,orders)
    def test_initial_zero_and_input_unchanged(self):
        arrays,meta,hands=score_fixture();before={k:v.tobytes() for k,v in arrays.items()};result=p.score_oracle(arrays,meta,hands)
        for name in p.NAMES[1:]:self.assertEqual(result['scores'][name][0].tobytes(),result['scores'][p.NAMES[0]][0].tobytes())
        self.assertEqual(before,{k:v.tobytes() for k,v in arrays.items()})
        for hand in hands.values():hand[:]=0
        result=p.score_oracle(arrays,meta,hands)
        for name in p.NAMES[1:]:self.assertEqual(result['scores'][name].tobytes(),result['scores'][p.NAMES[0]].tobytes())
    def test_ineligible_and_etf_keep_delta_but_selector_enforces(self):
        arrays,meta,hands=score_fixture();arrays['eligible'][1,0]=False;result=p.score_oracle(arrays,meta,hands)
        for name,key in zip(p.NAMES[1:],p.HANDS):
            self.assertNotIn(0,result['choices'][name][1]);self.assertEqual(result['deltas'][name][1,0],hands[key][1,0])
            self.assertNotEqual(result['deltas'][name][1,-1],0)
    def test_invalid_delta_and_initial_nonzero_fail(self):
        arrays,meta,hands=score_fixture();hands[p.HANDS[0]][1,0]=.01001
        with self.assertRaises(ValueError):p.score_oracle(arrays,meta,hands)
        hands[p.HANDS[0]][1,0]=.01;hands[p.HANDS[0]][0,0]=.01
        with self.assertRaises(ValueError):p.score_oracle(arrays,meta,hands)
    def test_score_future_original_array_poison(self):
        arrays,meta,hands=score_fixture();first=p.score_oracle(arrays,meta,hands,1)
        for value in arrays.values():
            if value.dtype.kind=='f':value[1:]=np.nan
        last=p.score_oracle(arrays,meta,hands,1)
        for name in p.NAMES:self.assertEqual(first['scores'][name].tobytes(),last['scores'][name].tobytes())
    def test_original27_arrays_and_only2_extra(self):
        original={str(i):np.zeros((122,49)) for i in range(26)};original['baseline']=np.zeros((122,49))
        prepared={k:v.copy() for k,v in original.items()};prepared.update({k:np.zeros((122,49)) for k in p.HANDS})
        p.verify_original_arrays(original,prepared);prepared['baseline'][1,1]=1
        with self.assertRaises(ValueError):p.verify_original_arrays(original,prepared)


class SelectionTests(unittest.TestCase):
    def rows(self,gains=(0.,.006,.006)):
        return [dict(name=n,training=dict(cagr=.4+g,sharpe=1.5,max_drawdown=-.2),early=dict(cagr=.4),recent=dict(cagr=.4)) for n,g in zip(p.NAMES,gains)]
    def test_changed_set_first88_and_both_eras(self):
        orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in p.NAMES}
        orders[p.NAMES[1]][0]['symbols'].reverse();orders[p.NAMES[1]][100]['symbols']=['D','E','F']
        self.assertFalse(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
        for i in (1,2,53):orders[p.NAMES[1]][i]['symbols']=['A','B','D']
        self.assertEqual(o.changed_support_oracle(orders,p.NAMES[1])['months'],[1,2,53]);self.assertTrue(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
    def test_fixed_tie_and_source_guard(self):
        rows=self.rows();changed={n:dict(passed=True) for n in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=False),changed)['primary'],p.NAMES[0])
    def test_minimum_gain_and_open_cost_guard(self):
        changed={n:dict(passed=True) for n in p.NAMES[1:]};rows=self.rows((0,.0049,.004))
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[0])
        rows=self.rows();opened=self.rows((0,-.0001,-.0001))
        self.assertEqual(o.select_first88_oracle(rows,opened,dict(passed=True),changed)['primary'],p.NAMES[0])
    def test_exact_epsilon_threshold_order(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        edge=baseline['training']['cagr']+.005-1e-12
        candidate['training']['cagr']=edge
        self.assertTrue(o.metric_guard(candidate,baseline,.005))
        candidate['training']['cagr']=float(np.nextafter(edge,-np.inf))
        self.assertFalse(o.metric_guard(candidate,baseline,.005))
        rows=self.rows((0.,.006,.006));rows[2]['training']['cagr']=rows[1]['training']['cagr']+1e-12
        changed={n:dict(passed=True) for n in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
        rows[2]['training']['cagr']=float(np.nextafter(rows[2]['training']['cagr'],np.inf))
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[2])
    def test_registered_decimal_epsilon_edge_fixtures(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        baseline['training']['cagr']=.2;candidate['training']['cagr']=.204999999999
        self.assertTrue(o.metric_guard(candidate,baseline,.005))
        self.assertLess(candidate['training']['cagr']-baseline['training']['cagr'],.005-1e-12)
        rows=self.rows();rows[0]['training']['cagr']=.19
        rows[1]['training']['cagr']=.2;rows[2]['training']['cagr']=.20000000000100002
        changed={name:dict(passed=True) for name in p.NAMES[1:]}
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])
    def test_later_values_irrelevant_and_nonfinite_guard(self):
        rows=self.rows((0,.006,.008));changed={n:dict(passed=True) for n in p.NAMES[1:]}
        for row in rows:row['later34']=dict(cagr=-10000.)
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[2])
        rows[2]['training']['sharpe']=float('nan')
        self.assertEqual(o.select_first88_oracle(rows,rows,dict(passed=True),changed)['primary'],p.NAMES[1])


class PhaseAndAccountingTests(unittest.TestCase):
    def test_invalid_plan_before_any_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/f.OUT;directory.mkdir(parents=True);path=directory/'PLAN.json';path.write_text(json.dumps(dict(round=159,status='REGISTERED',arms=list(p.NAMES))))
            reader=f.Reader(tmp)
            with self.assertRaisesRegex(ValueError,'Fixed registered'):f.bind(reader,hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
            self.assertEqual(set(reader.seen),{str(path.resolve())})
    def test_explicit_plan_sha_required(self):
        with self.assertRaisesRegex(ValueError,'Explicit PLAN'):f.bind(f.Reader('/private/tmp'),None,'prepare')
    def test_prepare_blocks_prior_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/f.OUT;directory.mkdir(parents=True)
            plan=dict(round=163,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory.resolve()),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),measurement=f.measurement_contract(),selection_contract=f.selection_contract(),actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan));(directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):f.bind(f.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_exact_queue_and_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/f.OUT;directory.mkdir(parents=True);control=root/f.STATE;control.parent.mkdir(parents=True)
            before=dict(completed=[dict(trials=6)]*161+[dict(trials=35)],registered_queue=[])
            raw=json.dumps(before);(directory/'STATE_BEFORE.json').write_text(raw)
            plan=dict(registered_utc='2026-10-21T01:00:00+00:00',registration_precondition=dict(path=str(control),sha256=hashlib.sha256(raw.encode()).hexdigest()))
            entry=dict(round=163,directory='monthly_run163',plan=dict(path=str(directory/'PLAN.json'),sha256='a'*64),registered_utc=plan['registered_utc'])
            current=dict(completed=before['completed'],registered_queue=[entry]);control.write_text(json.dumps(current));f.registration(root,plan,'a'*64)
            current['best_training_selected_candidate']={'changed':True};control.write_text(json.dumps(current))
            with self.assertRaisesRegex(ValueError,'incumbent metadata'):f.registration(root,plan,'a'*64)
            current.pop('best_training_selected_candidate')
            current['registered_queue']=[];control.write_text(json.dumps(current))
            with self.assertRaisesRegex(ValueError,'registered queue'):f.registration(root,plan,'a'*64)
    def test_duplicate_json_and_hash_change_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):f.Reader(tmp).json(path)
            reader=f.Reader(tmp);reader.sha(path);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):reader.sha(path)
    def test_forbidden_paths_and_import_guard(self):
        with self.assertRaises(ValueError):f.Reader('/private/tmp').path('/Users/test/OneDrive/x')
        with patch.dict(sys.modules,{'run163_kernel':object()}):
            with self.assertRaisesRegex(ValueError,'imported'):p.independent_imports()
    def test_closure_and_exact_cache_schema(self):
        names={path.name for path in f.source_files()}
        self.assertTrue({'run163_independent.py','run163_independent_prepare.py','run163_independent_outcome.py','test_run163_independent.py','monthly_ridge_support70_result_review.py','monthly_paired_target143_outcome_review.py','monthly_latent_state134_prepare_review.py','inputs.py'}<=names)
        self.assertEqual(len(f.CACHE_FILES),8);self.assertIn('STATE_VALUES.json',f.CACHE_FILES)
    def test_source_only_static_host_interfaces(self):
        # Parse source only. Never import or execute the new host/runner/kernel.
        root=Path(__file__).resolve().parent
        host=ast.parse((root/'run163_host.py').read_text())
        assignments={target.id:node.value for node in host.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(set(ast.literal_eval(assignments['CACHE'])),set(f.CACHE_FILES))
        self.assertEqual(ast.literal_eval(assignments['PROTECTED_SHA']),'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')
        runner=ast.parse((root/'run163_runner.py').read_text())
        declarations=[node for node in runner.body if isinstance(node,ast.FunctionDef) and node.name in ('measurement_contract','selection_contract')]
        # Validate before evaluating declaration-only AST; never execute imports,
        # decorators, defaults, or any producer body other than a literal return.
        for declaration in declarations:
            self.assertEqual(len(declaration.body),1)
            self.assertIsInstance(declaration.body[0],ast.Return)
            self.assertFalse(declaration.decorator_list or declaration.args.args or declaration.args.defaults)
            allowed=(ast.Return,ast.Call,ast.Name,ast.Load,ast.Constant,ast.List,ast.Tuple,ast.keyword,ast.UnaryOp,ast.USub)
            self.assertTrue(all(isinstance(node,allowed) for node in ast.walk(declaration.body[0])))
            calls=[node for node in ast.walk(declaration) if isinstance(node,ast.Call)]
            self.assertTrue(all(isinstance(call.func,ast.Name) and call.func.id in ('dict','list') for call in calls))
        scope={'NAMES':p.NAMES,'EXTRA':p.EXTRA,'MAX_ABS_DELTA':.01}
        exec(compile(ast.Module(body=declarations,type_ignores=[]),'static_literal_contracts','exec'),scope)
        self.assertEqual(scope['measurement_contract'](),f.measurement_contract())
        self.assertEqual(scope['selection_contract'](),f.selection_contract())
        p.independent_imports()
    def test_known_cash_ew2_fixed_units_and_open_cost_accounting(self):
        days=pd.bdate_range('2020-01-01',periods=6)
        close=pd.DataFrame({'A':[100.,120.,140.,150.,160.,170.],
                            'B':[100.,90.,80.,70.,60.,50.]},index=days)
        opening=pd.DataFrame({'A':[100.,110.,130.,145.,150.,160.],
                              'B':[100.,100.,90.,75.,70.,60.]},index=days)
        orders=[dict(date=str(days[0].date()),signal_date=str(days[0].date()),symbols=['A','B'],scores=[.2,.1]),
                dict(date=str(days[2].date()),signal_date=str(days[2].date()),symbols=[],scores=[])]
        helper=o.accounting_helper()
        _,nav,trades,_=helper.replay(close,opening,orders,days,'close',0.)
        np.testing.assert_allclose(nav,[100000.,105000.,110000.,110000.,110000.,110000.],rtol=0,atol=1e-9)
        self.assertEqual([t['date'] for t in trades],[str(days[0].date())]*2+[str(days[2].date())]*2)
        for cost in (0.,.0005):
            _,nav,trades,_=helper.replay(close,opening,orders,days,'open',cost)
            invested=100000./(1+cost);units_a=invested/220;units_b=invested/200
            final=(145*units_a+75*units_b)*(1-cost)
            wanted=[100000.,120*units_a+90*units_b,140*units_a+80*units_b,final,final,final]
            np.testing.assert_allclose(nav,wanted,rtol=0,atol=1e-9)
            self.assertEqual([t['date'] for t in trades],[str(days[1].date())]*2+[str(days[3].date())]*2)
            self.assertAlmostEqual(sum(t['cost'] for t in trades),100000.-invested+(145*units_a+75*units_b)*cost,places=9)
    def test_nine_paths_synthetic_fixed_units_and_tamper(self):
        dates=pd.bdate_range('2020-01-01',periods=245);names=['A','B','C']
        close=pd.DataFrame(100.,index=dates,columns=names);opening=close.copy()
        signals=[str(x.date()) for x in dates[::2][:-1]];ends=[str(x.date()) for x in dates[::2][1:]]
        meta=dict(dates=signals,ends=ends,symbols=[names[:] for _ in signals])
        orders={name:[dict(date=d,signal_date=d,end=e,symbols=names[:],scores=[.3,.2,.1]) for d,e in zip(signals,ends)] for name in p.NAMES}
        result=o.nine_path_oracle(close,opening,orders,meta)
        self.assertEqual((result['paths'],result['orders'],len(result['details'])),(9,366,9))
        artifacts={'RESULTS.json':dict(arms=result['results']),'TRAIN_TRIALS.json':result['training'],'TRAIN_OPEN_TRIALS.json':result['opened']}
        for name,values in result['details'].items():
            for key in ('daily','trades','holdings'):artifacts[name+'_'+key.upper()+'.json']=deepcopy(values[key])
        self.assertEqual(o.compare_nine_paths(result,artifacts)['paths'],9)
        for name in p.NAMES:
            self.assertEqual(result['results'][name]['same_close']['metrics']['total_return'],0.)
            self.assertLess(result['results'][name]['next_open_cost']['metrics']['total_return'],0.)
        artifacts[p.NAMES[0]+'_same_close_DAILY.json'][1]['equity']+=1
        with self.assertRaises(AssertionError):o.compare_nine_paths(result,artifacts)


if __name__=='__main__':
    p.independent_imports();unittest.main(verbosity=2)

