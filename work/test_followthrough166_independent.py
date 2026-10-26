"""Synthetic-only independent166 tests; no producer imports or actual numerical data."""
from copy import deepcopy
import ast
from datetime import date,timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import followthrough166_independent_prepare as p
import followthrough166_independent_outcome as o
import followthrough166_independent as f


def feature_fixture(count=3,width=3):
    index=pd.bdate_range('2014-01-01','2026-10-02')
    calendar=[str(d.date()) for d in index]
    monthends=[calendar[i] for i in range(len(calendar)-1) if calendar[i][:7]!=calendar[i+1][:7]]
    dates=(['2016-08-17']+[d for d in monthends if '2016-08-31'<=d<='2026-08-31'])[:count]
    names=['S%d'%i for i in range(width)]
    meta=dict(dates=dates,symbols=[names[:] for _ in dates])
    data={key:{} for key in ('adj','volume')}
    for j,name in enumerate(names):
        rng=np.random.default_rng(166+j)
        returns=rng.normal(.0003,.02,len(index)-1)
        data['adj'][name]=100*np.exp(np.r_[0.,np.cumsum(returns)])
        data['volume'][name]=1e6*np.exp(rng.normal(0,.6,len(index)))
    return meta,calendar,{}, {key:pd.DataFrame(values,index=index) for key,values in data.items()}


def paired_fixture(direction=None):
    rng=np.random.default_rng(166);logs=rng.normal(0.,.6,231)
    activity=np.array([logs[j+21]-sum(logs[j:j+21])/21 for j in range(210)])
    if direction is None:returns=rng.normal(.001,.02,211)
    else:
        returns=[.01]
        for a in activity:returns.append(.4*returns[-1]+direction*.3*returns[-1]*a+.02*a+.002)
        returns=np.array(returns)
    return np.exp(np.r_[0.,np.cumsum(returns)]),np.exp(logs),returns,activity


class FeatureTests(unittest.TestCase):
    def test_partial_corr_against_lstsq_with_intercept_and_two_controls(self):
        adj,volume,r,a=paired_fixture();got=p.paired_oracle(adj,volume)
        design=np.column_stack([np.ones(210),r[:-1],a]);q=(r[:-1]-r[:-1].mean())*(a-a.mean());y=r[1:]
        residual_q=q-design@np.linalg.lstsq(design,q,rcond=None)[0]
        residual_y=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        self.assertAlmostEqual(got[0],np.corrcoef(residual_q,residual_y)[0,1],places=12)
        self.assertAlmostEqual(got[1],np.corrcoef(r[:-1],y)[0,1],places=12)
    def test_known_positive_negative_interaction_after_removing_controls(self):
        for sign in (-1,1):
            adj,volume,_,_=paired_fixture(sign)
            self.assertAlmostEqual(p.paired_oracle(adj,volume)[0],sign,places=11)
    def test_21_prior_volume_values_exclude_current_volume(self):
        adj,volume,r,a=paired_fixture();actual=p.paired_oracle(adj,volume)
        logs=np.log(volume)
        wrong=[logs[j+21]-sum(logs[j+1:j+22])/21 for j in range(210)]
        self.assertGreater(abs(actual[0]-p.partial_oracle(r[:-1],wrong,r[1:])[0]),1e-6)
        changed=volume.copy();changed[0]*=2
        self.assertGreater(abs(p.paired_oracle(adj,changed)[0]-actual[0]),1e-7)
    def test_exact_counts_and_every_consumed_source_required(self):
        args=paired_fixture()[:2]
        for key in range(2):
            for bad_source in (args[key][:-1],np.r_[args[key],1.],np.ones((len(args[key]),1))):
                bad=list(args);bad[key]=bad_source;self.assertEqual(p.paired_oracle(*bad),(None,None))
            for index in range(len(args[key])):
                for value in (0.,-1.,np.nan,np.inf,-np.inf):
                    bad=[a.copy() for a in args];bad[key][index]=value
                    self.assertEqual(p.paired_oracle(*bad),(None,None),(key,index,value))
    def test_positive_source_units_do_not_change_correlations(self):
        adj,volume,_,_=paired_fixture();got=p.paired_oracle(adj,volume)
        for am,vm in ((1e100,1e150),(1e-100,1e-150)):
            np.testing.assert_allclose(p.paired_oracle(adj*am,volume*vm),got,atol=1e-10,rtol=0)
    def test_relative_centering_and_all_degenerate_components(self):
        r=np.sin(np.arange(210)*.19);a=np.cos(np.arange(210)*.31);y=np.sin(np.arange(210)*.43)
        self.assertIsNone(p.center_unit(np.ones(210)));self.assertIsNone(p.center_unit(np.zeros(210)))
        signs=(-1.)**np.arange(210)
        self.assertIsNone(p.center_unit(1.+signs*1e-13));self.assertIsNotNone(p.center_unit(1.+signs*1e-11))
        for args in ((np.ones(210),a,y),(r,np.ones(210),y),(r,a,np.ones(210)),(r,r,y),(r,-r,y),(r,a,r+.2*a)):
            self.assertEqual(p.partial_oracle(*args),(None,None))
        r=np.tile([-3.,-2.,-1.,1.,2.,3.],35)
        self.assertEqual(p.partial_oracle(r,1/r,y),(None,None)) # Constant interaction.
        r=np.tile([-1.,0.,1.],70)
        self.assertEqual(p.partial_oracle(r,r*r,y),(None,None)) # Interaction aliases return control.
    def test_dependent_control_relative_cutoff(self):
        r=np.sin(np.arange(210)*.19);a=np.cos(np.arange(210)*.31);y=np.sin(np.arange(210)*.43)
        self.assertEqual(p.partial_oracle(r,r+1e-14*a,y),(None,None))
        self.assertIsNotNone(p.partial_oracle(r,r+1e-8*a,y)[0])
    def test_constant_volume_or_price_invalidates_both(self):
        adj,volume,_,_=paired_fixture()
        self.assertEqual(p.paired_oracle(adj,np.ones(231)),(None,None))
        self.assertEqual(p.paired_oracle(np.ones(212),volume),(None,None))
    def test_roundoff_only_clip(self):
        e=np.finfo(float).eps
        for sign in (-1,1):
            self.assertEqual(p.correlation_clip(sign*(1+64*e)),float(sign))
            self.assertIsNone(p.correlation_clip(sign*(1+65*e)))
        self.assertIsNone(p.correlation_clip(np.nan))
    def test_midranks_ties_singleton_and_missing(self):
        self.assertEqual(p.midranks([2.,1.,1.,None]),[.5,0.,0.,None])
        self.assertEqual(p.midranks([None,7.,None]),[None,.5,None]);self.assertEqual(p.midranks([None]),[None])
    def test_initial_records_measurement_but_forces_positive_zero(self):
        got=p.features_oracle(*feature_fixture());row=got['audit']['event_history'][0]
        self.assertTrue(all(x is not None for x in row['raw_followthrough']+row['raw_control']))
        self.assertEqual(row['common_support'],[False]*3)
        for value in got['arrays'].values():
            self.assertTrue(np.all(value[0]==0));self.assertFalse(np.signbit(value[0]).any())
        self.assertEqual(got['audit']['common_support_cells'],6)
    def test_exact_windows_and_poison_before_and_after_each_source(self):
        args=feature_fixture();got=p.features_oracle(*args);t=args[1].index(args[0]['dates'][0])
        row=got['audit']['event_history'][0]
        self.assertEqual((row['window_start'],row['window_end'],row['volume_window_start'],row['volume_window_end'],row['pair_start'],row['pair_end']),
            (args[1][t-253],args[1][t-42],args[1][t-273],args[1][t-43],args[1][t-252],args[1][t-43]))
        for key,lo,hi in (('adj',t-253,t-42),('volume',t-273,t-43)):
            panel={k:v.copy() for k,v in args[3].items()};frame=panel[key]
            frame.loc[(frame.index<pd.Timestamp(args[1][lo]))|(frame.index>pd.Timestamp(args[1][hi]))]=np.nan
            prefix=p.features_oracle(*args[:3],panel,n=1)
            p.compare(prefix['audit']['event_history'],got['audit']['event_history'][:1])
    def test_all123_prefixes_with_exact_union_window_poison(self):
        args=feature_fixture(122,1);full=p.features_oracle(*args)
        for n in range(123):
            prefix=p.features_oracle(*args,n=n)
            poisoned=p.poison_outside_windows(args[3],args[0],args[1],n)
            alternate=p.features_oracle(*args[:3],poisoned,n=n)
            for key in p.EXTRA:
                self.assertEqual(prefix['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
                self.assertEqual(alternate['arrays'][key].tobytes(),prefix['arrays'][key].tobytes())
            p.compare(alternate['audit']['event_history'],full['audit']['event_history'][:n])
    def test_poison_helper_uses_source_specific_exact_bounds(self):
        args=feature_fixture();t=args[1].index(args[0]['dates'][0]);panel=p.poison_outside_windows(args[3],args[0],args[1],1)
        for key,lo,hi in (('adj',t-253,t-42),('volume',t-273,t-43)):
            for i in (lo,hi):self.assertTrue(panel[key].loc[args[1][i]].notna().all())
            for i in (lo-1,hi+1):self.assertTrue(panel[key].loc[args[1][i]].isna().all())
        self.assertTrue(all(frame.isna().all().all() for frame in p.poison_outside_windows(args[3],args[0],args[1],0).values()))
    def test_future_membership_and_unused_metadata_no_effect(self):
        args=list(feature_fixture());base=p.features_oracle(*args,n=2)
        args[0]=deepcopy(args[0]);args[0]['symbols'][2]=['X','Y','Z'];args[2]={'poison':object()}
        p.compare(p.features_oracle(*args,n=2)['audit'],base['audit'])
    def test_missing_source_neutralizes_both_features(self):
        args=feature_fixture();t=args[1].index(args[0]['dates'][1]);day=args[1][t-100]
        for key in ('adj','volume'):
            for bad in (0.,-1.,np.nan,np.inf,-np.inf):
                px={k:v.copy() for k,v in args[3].items()};px[key].at[pd.Timestamp(day),'S0']=bad
                row=p.features_oracle(*args[:3],px)['audit']['event_history'][1]
                self.assertTrue(all(row[k][0] is None for k in ('raw_followthrough','raw_control','ranks_followthrough','ranks_control')))
        for px in ({k:v.drop(columns='S0') for k,v in args[3].items()}, {k:v.drop(index=pd.Timestamp(day)) for k,v in args[3].items()}):
            got=p.features_oracle(*args[:3],px);self.assertEqual(got['arrays'][p.EXTRA[0]][1,0],0.)
    def test_etfs_same_treatment_and_duplicate_series_tie(self):
        args=list(feature_fixture());args[0]['etfs']=['S1'];px={k:v.copy() for k,v in args[3].items()}
        for frame in px.values():frame['S1']=frame['S0']
        row=p.features_oracle(*args[:3],px)['audit']['event_history'][1]
        for key in ('raw_followthrough','raw_control','ranks_followthrough','ranks_control'):self.assertEqual(row[key][0],row[key][1])
        self.assertTrue(row['common_support'][1])
    def test_incomplete_calendar_and_offcalendar_sources(self):
        args=list(feature_fixture());args[1]=[x for x in args[1] if x>='2016-08-01']
        got=p.features_oracle(*args);self.assertIsNone(got['audit']['event_history'][0]['volume_window_start']);self.assertFalse(got['arrays'][p.EXTRA[0]].any())
        args=list(feature_fixture());got=p.features_oracle(*args)
        args[3]={k:pd.concat([v,pd.DataFrame(7.,index=[pd.Timestamp('2016-08-20')],columns=v.columns)]).sort_index() for k,v in args[3].items()}
        p.compare(p.features_oracle(*args)['audit'],got['audit'])
    def test_original_eligibility_ignored_single_shared_member_full_dose(self):
        args=list(feature_fixture());base=p.features_oracle(*args)
        args[0]=deepcopy(args[0]);args[0]['eligible']=[[False]*3 for _ in args[0]['dates']]
        p.compare(p.features_oracle(*args)['audit'],base['audit'])
        args[0]['symbols']=[['S0','ABSENT1','ABSENT2'] for _ in args[0]['dates']]
        got=p.features_oracle(*args);row=got['audit']['event_history'][1]
        for key in ('ranks_followthrough','ranks_control'):self.assertEqual(row[key],[.5,None,None])
        for key in p.EXTRA:self.assertEqual(got['arrays'][key][1,0],.02)
    def test_daily_axes_current_pool_monthend_and_extra_rawclose_rejections(self):
        args=feature_fixture()
        for changed in (args[3]['adj'].iloc[::-1],pd.concat([args[3]['adj'],args[3]['adj'].tail(1)]),args[3]['adj'].rename(columns={'S1':'S0'}),args[3]['adj'].rename(columns={'S0':7})):
            px=dict(args[3]);px['adj']=changed
            with self.assertRaises(ValueError):p.features_oracle(*args[:3],px)
        px=dict(args[3],close=args[3]['adj'])
        with self.assertRaises(ValueError):p.features_oracle(*args[:3],px)
        for dates in (list(reversed(args[0]['dates'])),[args[0]['dates'][0]]*3):
            meta=deepcopy(args[0]);meta['dates']=dates
            with self.assertRaises(ValueError):p.features_oracle(meta,*args[1:])
        meta=deepcopy(args[0]);meta['symbols'][1][1]='S0'
        with self.assertRaises(ValueError):p.features_oracle(meta,*args[1:])
        with self.assertRaises(ValueError):p.features_oracle(*args,n=True)
        meta=deepcopy(args[0]);meta['dates'][1]='2016-08-30'
        with self.assertRaisesRegex(ValueError,'month-end'):p.features_oracle(meta,*args[1:])
        with self.assertRaises(ValueError):p.features_oracle(args[0],sorted(args[1]+['2016-08-20']),*args[2:])
    def test_delta_tamper_initial_bound_and_exact_zero_rank(self):
        got=p.features_oracle(*feature_fixture());states=p.state_values(got['audit']);p.verify_feature_output(got,got['arrays'],got['audit'],states)
        self.assertEqual(set(states),{'raw_followthrough','raw_control','ranks_followthrough','ranks_control','common_support'})
        states['common_support'][1][0]=False
        with self.assertRaises(ValueError):p.verify_feature_output(got,got['arrays'],got['audit'],states)
        for target,value in (((0,0),1e-20),((1,0),.0200000000001)):
            other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][target]=value
            with self.assertRaises((ValueError,AssertionError)):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
        got['audit']['event_history'][1]['ranks_followthrough'][0]=0.;got['arrays'][p.EXTRA[0]][1,0]=0.
        other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][1,0]=1e-20
        with self.assertRaisesRegex(ValueError,'Exact zero-rank'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_support_rejects_bad_endpoints_and_numerical_grid(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args)
        for field,value in (('pair_end',None),('volume_window_end',got['audit']['event_history'][1]['window_end']),('volume_window_start',got['audit']['event_history'][1]['window_start']),('pair_start',got['audit']['event_history'][1]['window_start']),('window_start','2014-1-1')):
            bad=deepcopy(got['audit']);bad['event_history'][1][field]=value
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        bad=deepcopy(got['audit']);bad['window_pairs']=210.
        with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
    def test_complete_support_counts_and_corruptions(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args);support=o.support_oracle(got['audit'],args[0],got['arrays'])
        self.assertEqual(support['candidate_cells'],5978);self.assertIsNone(support['coverage_threshold']);self.assertEqual(support['common_support_cells'],121*49)
        self.assertEqual(support['coverage']['early53']['common_support_cells'],52*49)
        for field in ('common_support','ranks_followthrough','ranks_control','raw_followthrough'):
            bad=deepcopy(got['audit']);bad['event_history'][1][field][0]=None
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        for value in (1.1,-1.1,True):
            bad=deepcopy(got['audit']);bad['event_history'][1]['raw_control'][0]=value
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
        arrays,meta,hands=score_fixture();hands[p.HANDS[0]][1,0]=.02001
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
    def select(self,rows=None,opened=None,support=True,changed=True):
        rows=self.rows() if rows is None else rows
        return o.select_first88_oracle(rows,rows if opened is None else opened,dict(passed=support),{n:dict(passed=changed) for n in p.NAMES[1:]})
    def test_changed_set_first88_and_both_eras(self):
        orders={name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in p.NAMES}
        orders[p.NAMES[1]][0]['symbols'].reverse();orders[p.NAMES[1]][100]['symbols']=['D','E','F']
        self.assertFalse(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
        for i in (1,2,53):orders[p.NAMES[1]][i]['symbols']=['A','B','D']
        self.assertEqual(o.changed_support_oracle(orders,p.NAMES[1])['months'],[1,2,53]);self.assertTrue(o.changed_support_oracle(orders,p.NAMES[1])['passed'])
    def test_followthrough_only_and_source_changed_guards(self):
        self.assertEqual(self.select()['primary'],p.NAMES[1])
        self.assertEqual(self.select(support=False)['primary'],p.NAMES[0])
        self.assertEqual(self.select(changed=False)['primary'],p.NAMES[0])
        got=self.select(self.rows((0,.006,.05)))
        self.assertEqual(got['primary'],p.NAMES[0]);self.assertEqual(got['guards'][p.NAMES[2]],{'promotable':False})
    def test_incumbent_gain_open_cost_and_control_noninferiority(self):
        for rows,opened in ((self.rows((0,.0049,.003)),None),
                            (self.rows(),self.rows((0,-.0001,-.001))),
                            (self.rows((0,.006,.007)),None),
                            (self.rows(),self.rows((0,.006,.007)))):
            self.assertEqual(self.select(rows,opened)['primary'],p.NAMES[0])
    def test_exact_epsilon_threshold_and_control_boundary(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        edge=baseline['training']['cagr']+.005-1e-12
        candidate['training']['cagr']=edge;self.assertTrue(o.metric_guard(candidate,baseline,.005))
        candidate['training']['cagr']=float(np.nextafter(edge,-np.inf));self.assertFalse(o.metric_guard(candidate,baseline,.005))
        rows[2]['training']['cagr']=rows[1]['training']['cagr']+1e-12
        self.assertEqual(self.select(rows)['primary'],p.NAMES[1])
        rows[2]['training']['cagr']=float(np.nextafter(rows[2]['training']['cagr'],np.inf))
        self.assertEqual(self.select(rows)['primary'],p.NAMES[0])
    def test_registered_decimal_epsilon_edge(self):
        rows=self.rows();baseline=rows[0];candidate=deepcopy(rows[1])
        baseline['training']['cagr']=.2;candidate['training']['cagr']=.204999999999
        self.assertTrue(o.metric_guard(candidate,baseline,.005))
    def test_all_metric_guards_apply_against_incumbent_and_control_both_paths(self):
        for source in (0,2):
            for path in ('close','open'):
                for group,key,advantage in (('training','sharpe',.051),('training','max_drawdown',.031),('early','cagr',.026),('recent','cagr',.026)):
                    rows=self.rows();opened=deepcopy(rows);active=rows if path=='close' else opened
                    active[source][group][key]=active[1][group][key]+advantage
                    self.assertEqual(self.select(rows,opened)['primary'],p.NAMES[0],(source,path,group,key))
    def test_later_diagnostics_ignored_and_nonfinite_guards_fail(self):
        rows=self.rows()
        for row in rows:row['later34']=dict(cagr=-10000.)
        self.assertEqual(self.select(rows)['primary'],p.NAMES[1])
        for index in (0,1,2):
            bad=deepcopy(rows);bad[index]['training']['sharpe']=float('nan')
            self.assertEqual(self.select(bad)['primary'],p.NAMES[0])


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
            plan=dict(round=166,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory.resolve()),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),measurement=f.measurement_contract(),selection_contract=f.selection_contract(),actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan));(directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):f.bind(f.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_exact_queue_and_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/f.OUT;directory.mkdir(parents=True);control=root/f.STATE;control.parent.mkdir(parents=True)
            before=dict(completed=[dict(trials=6)]*164+[dict(trials=26)],registered_queue=[])
            raw=json.dumps(before);(directory/'STATE_BEFORE.json').write_text(raw)
            plan=dict(registered_utc='2026-10-21T01:00:00+00:00',registration_precondition=dict(path=str(control),sha256=hashlib.sha256(raw.encode()).hexdigest()))
            entry=dict(round=166,directory='monthly_followthrough166',plan=dict(path=str(directory/'PLAN.json'),sha256='a'*64),registered_utc=plan['registered_utc'])
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
        with patch.dict(sys.modules,{'followthrough166_kernel':object()}):
            with self.assertRaisesRegex(ValueError,'imported'):p.independent_imports()
    def test_closure_and_exact_cache_schema(self):
        names={path.name for path in f.source_files()}
        self.assertTrue({'followthrough166_independent.py','followthrough166_independent_prepare.py','followthrough166_independent_outcome.py','test_followthrough166_independent.py','monthly_ridge_support70_result_review.py','monthly_paired_target143_outcome_review.py','monthly_latent_state134_prepare_review.py','inputs.py'}<=names)
        self.assertEqual(len(f.CACHE_FILES),8);self.assertIn('STATE_VALUES.json',f.CACHE_FILES)
    def test_source_only_static_host_interfaces(self):
        # Parse source only. Never import or execute the new host/runner/kernel.
        root=Path(__file__).resolve().parent
        host=ast.parse((root/'followthrough166_host.py').read_text())
        assignments={target.id:node.value for node in host.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(set(ast.literal_eval(assignments['CACHE'])),set(f.CACHE_FILES))
        self.assertEqual(ast.literal_eval(assignments['PROTECTED_SHA']),'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')
        runner=ast.parse((root/'followthrough166_runner.py').read_text())
        declarations=[node for node in runner.body if isinstance(node,ast.FunctionDef) and node.name in ('measurement_contract','selection_contract')]
        # Validate before evaluating declaration-only AST; never execute imports,
        # decorators, defaults, or any producer body other than a literal return.
        for declaration in declarations:
            self.assertEqual(len(declaration.body),1)
            self.assertIsInstance(declaration.body[0],ast.Return)
            self.assertFalse(declaration.decorator_list or declaration.args.args or declaration.args.defaults)
            allowed=(ast.Return,ast.Call,ast.Name,ast.Load,ast.Constant,ast.List,ast.Tuple,ast.keyword,ast.UnaryOp,ast.USub,ast.Subscript)
            self.assertTrue(all(isinstance(node,allowed) for node in ast.walk(declaration.body[0])))
            calls=[node for node in ast.walk(declaration) if isinstance(node,ast.Call)]
            self.assertTrue(all(isinstance(call.func,ast.Name) and call.func.id in ('dict','list') for call in calls))
        scope={'NAMES':p.NAMES,'EXTRA':p.EXTRA,'MAX_ABS_DELTA':.02}
        exec(compile(ast.Module(body=declarations,type_ignores=[]),'static_literal_contracts','exec'),scope)
        self.assertEqual(scope['measurement_contract'](),f.measurement_contract())
        self.assertEqual(scope['selection_contract'](),f.selection_contract())
        p.independent_imports()
    def test_frozen_loader_closure_parity_and_volume_panel_exposure(self):
        import monthly_latent_state134_prepare_review as old
        root=Path('/private/tmp/followthrough166_synthetic_loader')
        loader_path=root/old.LOADER
        raw={str(root/('source_%d'%i)):old.LOADER_SHA for i in range(63)}
        raw[str(loader_path)]=old.LOADER_SHA
        index=pd.bdate_range('2020-01-01',periods=4);columns=['B','A']
        cached=dict(dates=np.array([str(x.date()) for x in index[:3]]),symbols=np.array(['A']),
                    close=np.array([[100.],[110.],[90.]]),opening=np.array([[99.],[109.],[89.]]))
        source={key:pd.DataFrame(value,index=index,columns=columns)
                for key,value in (('adj',200.),('close',50.),('volume',7000.))}
        info=dict(source_hashes={k:v for k,v in raw.items() if k!=str(loader_path)})
        loader=SimpleNamespace(load_panel=lambda etfs:(source,info))
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
        class SyntheticReader:
            def path(self,value):return root/value
            def verify(self,pins):return dict(pins)
            def npz(self,value):return cached
            def sha(self,value):return old.LOADER_SHA
            def json(self,value):return dict(synthetic_parity=True)
        with patch.object(f.importlib.util,'spec_from_file_location',return_value=spec), \
             patch.object(f.importlib.util,'module_from_spec',return_value=loader), \
             patch.object(old,'price_parity',return_value=dict(synthetic_parity=True)) as parity:
            close,opening,panel=f.prices(SyntheticReader(),dict(raw_source_hashes=raw),raw,dict(prices='prices.npz'),dict(etfs=[]),True)
            parity.assert_called_once()
            self.assertTrue(panel['adj'].equals(close));self.assertEqual(close.iloc[0,0],100.)
            self.assertEqual(opening.iloc[0,0],99.);self.assertEqual(set(panel),{'adj','volume'})
            self.assertEqual(panel['volume'].iloc[0,0],7000.)
            for frame in panel.values():self.assertTrue(frame.index.equals(close.index) and frame.columns.equals(close.columns))
            info['source_hashes'].pop(next(iter(info['source_hashes'])))
            with self.assertRaisesRegex(ValueError,'Actually consumed'):
                f.prices(SyntheticReader(),dict(raw_source_hashes=raw),raw,dict(prices='prices.npz'),dict(etfs=[]),True)
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

    def test_nine_distinct_nonflat_paths_reject_cross_arm_and_execution_swaps(self):
        days=pd.bdate_range('2020-01-01',periods=245);names=['A','B','C','D'];t=np.arange(len(days))
        close=pd.DataFrame({name:100*np.exp((j+1)*.0004*t+.035*np.sin(t/(5+j)+j)) for j,name in enumerate(names)},index=days)
        opening=close*pd.DataFrame({name:1+.007*np.cos(t/(3+j)+j) for j,name in enumerate(names)},index=days)
        signals=[str(x.date()) for x in days[::2][:-1]];ends=[str(x.date()) for x in days[::2][1:]]
        meta=dict(dates=signals,ends=ends,symbols=[names[:] for _ in signals])
        holdings=(lambda i:[] if i%7==0 else ['A','B'],
                  lambda i:['A','D'] if i%5==0 else ['B','C','D'],
                  lambda i:[] if i%11==0 else ['A','C'])
        orders={name:[dict(date=d,signal_date=d,end=e,symbols=holdings[j](i),
                          scores=[.3-.1*k for k in range(len(holdings[j](i)))])
                      for i,(d,e) in enumerate(zip(signals,ends))] for j,name in enumerate(p.NAMES)}
        result=o.nine_path_oracle(close,opening,orders,meta)
        paths={name:tuple(row['equity'] for row in rows['daily']) for name,rows in result['details'].items()}
        self.assertEqual(len(set(paths.values())),9)
        self.assertTrue(all(len(set(nav))>200 for nav in paths.values()))
        self.assertEqual(len({nav[-1] for nav in paths.values()}),9)
        artifacts={'RESULTS.json':dict(arms=result['results']),'TRAIN_TRIALS.json':result['training'],'TRAIN_OPEN_TRIALS.json':result['opened']}
        for name,values in result['details'].items():
            for key in ('daily','trades','holdings'):artifacts[name+'_'+key.upper()+'.json']=deepcopy(values[key])
        self.assertEqual(o.compare_nine_paths(result,artifacts)['paths'],9)
        for left,right in ((p.NAMES[0]+'_same_close',p.NAMES[1]+'_same_close'),
                           (p.NAMES[1]+'_next_open_zero',p.NAMES[1]+'_next_open_cost'),
                           (p.NAMES[2]+'_same_close',p.NAMES[2]+'_next_open_zero')):
            for kind in ('DAILY','TRADES','HOLDINGS'):
                swapped=deepcopy(artifacts)
                swapped[left+'_'+kind+'.json']=deepcopy(artifacts[right+'_'+kind+'.json'])
                with self.assertRaises((AssertionError,ValueError),msg=(left,right,kind)):
                    o.compare_nine_paths(result,swapped)


if __name__=='__main__':
    p.independent_imports();unittest.main(verbosity=2)

