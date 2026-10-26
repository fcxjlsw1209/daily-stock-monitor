"""Synthetic-only independent165 tests; no producer imports or actual numerical data."""
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
import recovery165_independent_prepare as p
import recovery165_independent_outcome as o
import recovery165_independent as f


def feature_fixture(count=3,width=3):
    index=pd.bdate_range('2014-01-01','2026-10-02')
    calendar=[str(d.date()) for d in index]
    monthends=[calendar[i] for i in range(len(calendar)-1) if calendar[i][:7]!=calendar[i+1][:7]]
    dates=(['2016-08-17']+[d for d in monthends if '2016-08-31'<=d<='2026-08-31'])[:count]
    names=['S%d'%i for i in range(width)];etfs=names[-17:] if width==49 else []
    meta=dict(dates=dates,symbols=[names[:] for _ in dates],etfs=etfs)
    data={};days=np.arange(len(index))
    for j,name in enumerate(names):
        period=6+j%3;phase=days%period
        data[name]=np.where(phase==1,80.-j*.1,np.where(phase==2,90.-j*.1,100.))
    data['XLK']=np.where(days%6==1,90.,np.where(days%6==2,92.,np.where(days%6==3,95.,100.)))
    data['XLF']=np.where(days%7==1,85.,100.)
    mapping=dict(etfs=etfs[:],by_date={day:{name:None if name in etfs else 'XLK' for name in names} for day in dates})
    return meta,calendar,mapping,pd.DataFrame(data,index=index)


def race_fixture(count=24,stock_delay=1,sector_delay=3,stock_drop=20.,sector_drop=10.):
    stock=np.full(211,100.);sector=stock.copy()
    for start in [1+8*j for j in range(count)]:
        stock[start:start+stock_delay]=100.-stock_drop
        sector[start:start+sector_delay]=100.-sector_drop
    return stock,sector


class FeatureTests(unittest.TestCase):
    def test_known_recovery_and_impact_signs(self):
        stock,sector=race_fixture()
        recovery,control,count=p.paired_oracle(stock,sector)
        self.assertEqual(count,24);self.assertEqual(recovery,2/21)
        self.assertAlmostEqual(control,-1/3,places=14)
        reverse=p.paired_oracle(sector,stock)
        self.assertEqual(reverse,(-recovery,-control,count))
    def test_pre_event_level_and_first_passage_equality(self):
        prices=np.full(211,80.);prices[0]=100.;prices[2]=90.;prices[3]=100.;prices[4]=120.
        self.assertEqual(p.recovery_time(prices,1),2)
        prices[3]=99.;self.assertEqual(p.recovery_time(prices,1),3)
    def test_exact_21_session_horizon_and_censor22(self):
        prices=np.full(211,80.);prices[0]=100.;prices[22]=100.
        self.assertEqual(p.recovery_time(prices,1),21)
        prices[22]=99.;prices[23]=101.
        self.assertEqual(p.recovery_time(prices,1),22)
    def test_last_start189_followup_consumes210_and190_is_excluded(self):
        stock,sector=race_fixture(20)
        stock[189]=80.;sector[189:210]=90.
        got=p.paired_oracle(stock,sector)
        self.assertEqual(got[2],21)
        self.assertEqual(got[0],float(Fraction(20*2+20,21*21)))
        # The 190th return cannot add another event, and source cell210 is consumed.
        stock[190]=70.;sector[190]=80.
        self.assertEqual(p.paired_oracle(stock,sector)[2],21)
        stock[210]=np.nan;self.assertEqual(p.paired_oracle(stock,sector),(None,None,None))
    def test_minimum_exactly20_joint_down_starts(self):
        self.assertEqual(p.paired_oracle(*race_fixture(19)),(None,None,19))
        self.assertEqual(p.paired_oracle(*race_fixture(20))[2],20)
        self.assertIsNotNone(p.paired_oracle(*race_fixture(20))[0])
    def test_joint_strict_down_only_no_zero_or_single_series_event(self):
        stock,sector=race_fixture(20);sector[1]=100.
        self.assertEqual(p.paired_oracle(stock,sector),(None,None,19))
        sector[1]=101.;self.assertEqual(p.paired_oracle(stock,sector),(None,None,19))
        self.assertEqual(p.paired_oracle(np.ones(211),np.ones(211)),(None,None,0))
    def test_overlapping_and_double_censored_events_are_kept(self):
        falling=np.linspace(500.,100.,211)
        got=p.paired_oracle(falling,falling)
        self.assertEqual(got,(0.,0.,189))
    def test_maximal_race_and_reverse(self):
        stock=np.where(np.arange(211)%2,80.,100.)
        sector=100.*(.99**((np.arange(211)+1)//2))
        self.assertEqual(p.paired_oracle(stock,sector)[0],1.)
        self.assertEqual(p.paired_oracle(sector,stock)[0],-1.)
        self.assertEqual(p.paired_oracle(stock,sector)[2],95)
    def test_exact_rational_mean_preserves_equal_discrete_scores(self):
        for count in (20,21,22,23,24):
            self.assertEqual(p.paired_oracle(*race_fixture(count))[0],2/21)
    def test_all211_paired_values_required_and_exact_length(self):
        args=race_fixture()
        for source in (0,1):
            for n in (210,212):
                bad=list(args);bad[source]=np.full(n,100.)
                self.assertEqual(p.paired_oracle(*bad),(None,None,None))
            for index in range(211):
                for value in (0.,-1.,np.nan,np.inf,-np.inf):
                    bad=[a.copy() for a in args];bad[source][index]=value
                    self.assertEqual(p.paired_oracle(*bad),(None,None,None),(source,index,value))
    def test_positive_rescaling_and_exact_price_comparisons(self):
        stock,sector=race_fixture()
        got=p.paired_oracle(stock,sector)
        for multiplier in (1e-200,1e200):
            scaled=p.paired_oracle(stock*multiplier,sector*multiplier)
            np.testing.assert_allclose(scaled,got,atol=1e-14,rtol=0)
        close=np.ones(211);close[1::2]=np.nextafter(1.,0.)
        self.assertEqual(p.paired_oracle(close,close),(0.,0.,95))
    def test_midranks_ties_singleton_and_missing(self):
        self.assertEqual(p.midranks([2.,1.,1.,None]),[.5,0.,0.,None])
        self.assertEqual(p.midranks([None,7.,None]),[None,.5,None])
        self.assertEqual(p.midranks([None]),[None])
    def test_initial_audit_measured_but_deltas_positive_zero(self):
        args=feature_fixture();got=p.features_oracle(*args);row=got['audit']['event_history'][0]
        self.assertIsNotNone(row['raw_recovery'][0]);self.assertEqual(row['source_status'][0],'available')
        self.assertEqual(row['common_support'],[False]*3)
        for value in got['arrays'].values():
            self.assertTrue(np.all(value[0]==0));self.assertFalse(np.signbit(value[0]).any())
    def test_exact_endpoints_and_postwindow_stock_sector_poison(self):
        args=feature_fixture();got=p.features_oracle(*args);t=args[1].index(args[0]['dates'][1]);row=got['audit']['event_history'][1]
        self.assertEqual((row['window_start'],row['window_end']),(args[1][t-252],args[1][t-42]))
        for names in (['S0','S1','S2'],['XLK','XLF']):
            prices=args[3].copy();prices.loc[prices.index>pd.Timestamp(args[1][t-42]),names]=np.nan
            prefix=p.features_oracle(*args[:3],prices,n=2)
            for key in p.EXTRA:self.assertEqual(prefix['arrays'][key].tobytes(),got['arrays'][key][:2].tobytes())
            p.compare(prefix['audit']['event_history'],got['audit']['event_history'][:2])
    def test_all123_feature_prefixes(self):
        args=feature_fixture(122,1);full=p.features_oracle(*args)
        for n in range(123):
            prefix=p.features_oracle(*args,n=n)
            for key in p.EXTRA:self.assertEqual(prefix['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
            p.compare(prefix['audit']['event_history'],full['audit']['event_history'][:n])
    def test_future_mapping_and_membership_do_not_change_prefix(self):
        args=list(feature_fixture());base=p.features_oracle(*args,n=2)
        args[0]=deepcopy(args[0]);args[2]=deepcopy(args[2]);day=args[0]['dates'][2]
        args[0]['symbols'][2]=['FUTURE0','FUTURE1','FUTURE2']
        args[2]['by_date'][day]=dict.fromkeys(args[0]['symbols'][2],None)
        p.compare(p.features_oracle(*args,n=2)['audit'],base['audit'])
    def test_missing_and_invalid_stock_or_sector_neutralize_both(self):
        args=feature_fixture();t=args[1].index(args[0]['dates'][1]);day=args[1][t-100]
        for name,reason in (('S0','invalid_stock_path'),('XLK','invalid_sector_path')):
            for value in (0.,-1.,np.nan,np.inf,-np.inf):
                prices=args[3].copy();prices.at[pd.Timestamp(day),name]=value
                row=p.features_oracle(*args[:3],prices)['audit']['event_history'][1]
                self.assertIsNone(row['raw_recovery'][0]);self.assertIsNone(row['raw_control'][0]);self.assertIsNone(row['joint_down_counts'][0])
                self.assertEqual(row['source_status'][0],reason)
            row=p.features_oracle(*args[:3],args[3].drop(columns=name))['audit']['event_history'][1]
            self.assertEqual(row['source_status'][0],reason)
    def test_etfs_unknown_mapping_and_insufficient_history_are_neutral(self):
        args=list(feature_fixture());args[2]=deepcopy(args[2]);args[2]['etfs']=['S1']
        for mapping in args[2]['by_date'].values():mapping['S1']=None;mapping['S2']=None
        row=p.features_oracle(*args)['audit']['event_history'][1]
        self.assertEqual(row['source_status'][1:],['etf_not_applicable','missing_sector_mapping'])
        self.assertEqual(row['ranks_recovery'],[.5,None,None]);self.assertEqual(row['ranks_control'],[.5,None,None])
        got=p.features_oracle(*args);self.assertEqual(got['arrays'][p.EXTRA[0]][1,0],.02)
        args[1]=[x for x in args[1] if x>='2016-08-01']
        row=p.features_oracle(*args)['audit']['event_history'][1]
        self.assertEqual(row['source_status'][0],'insufficient_history')
    def test_own_sector_only_no_market_or_othersector_fallback(self):
        args=feature_fixture();base=p.features_oracle(*args);prices=args[3].copy()
        prices['SPY']=np.nan;prices['XLF']=np.nan
        p.compare(p.features_oracle(*args[:3],prices)['audit'],base['audit'])
        prices=prices.drop(columns='XLK');row=p.features_oracle(*args[:3],prices)['audit']['event_history'][1]
        self.assertEqual(row['source_status'],['invalid_sector_path']*3)
    def test_eligibility_ignored_and_separate_rankings_on_shared_support(self):
        args=list(feature_fixture());base=p.features_oracle(*args)
        args[0]=deepcopy(args[0]);args[0]['eligible']=[[False]*3 for _ in args[0]['dates']]
        p.compare(p.features_oracle(*args)['audit'],base['audit'])
        prices=args[3].copy();prices['S1']=prices['S0']
        row=p.features_oracle(*args[:3],prices)['audit']['event_history'][1]
        self.assertEqual(row['ranks_recovery'][0],row['ranks_recovery'][1])
        self.assertEqual(row['ranks_control'][0],row['ranks_control'][1])
    def test_daily_axes_monthend_mapping_rejections(self):
        args=feature_fixture()
        for changed in (args[3].iloc[::-1],pd.concat([args[3],args[3].tail(1)]),args[3].rename(columns={'S1':'S0'}),args[3].rename(columns={'S0':7})):
            with self.assertRaises(ValueError):p.features_oracle(*args[:3],changed)
        with self.assertRaises(ValueError):p.features_oracle(*args,n=True)
        meta=deepcopy(args[0]);meta['dates'][1]='2016-08-30';mapping=deepcopy(args[2]);mapping['by_date']['2016-08-30']=mapping['by_date'].pop('2016-08-31')
        with self.assertRaisesRegex(ValueError,'month-end'):p.features_oracle(meta,args[1],mapping,args[3])
        for value in ('SPY','UNKNOWN',7):
            mapping=deepcopy(args[2]);mapping['by_date'][args[0]['dates'][1]]['S0']=value
            with self.assertRaises(ValueError):p.features_oracle(*args[:2],mapping,args[3])
        mapping=deepcopy(args[2]);mapping['etfs']=['S0']
        with self.assertRaisesRegex(ValueError,'neutral mapping'):p.features_oracle(*args[:2],mapping,args[3])
    def test_delta_tamper_initial_bound_and_exact_zero_rank(self):
        got=p.features_oracle(*feature_fixture());states=p.state_values(got['audit'])
        p.verify_feature_output(got,got['arrays'],got['audit'],states)
        self.assertEqual(set(states),{'raw_recovery','raw_control','ranks_recovery','ranks_control','joint_down_counts','sector_etfs','common_support'})
        states['joint_down_counts'][1][0]=0
        with self.assertRaises(ValueError):p.verify_feature_output(got,got['arrays'],got['audit'],states)
        for target,value in (((0,0),1e-20),((1,0),.0200000000001)):
            other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][target]=value
            with self.assertRaises((ValueError,AssertionError)):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
        got['audit']['event_history'][1]['ranks_recovery'][0]=0.;got['arrays'][p.EXTRA[0]][1,0]=0.
        other={k:v.copy() for k,v in got['arrays'].items()};other[p.EXTRA[0]][1,0]=1e-20
        with self.assertRaisesRegex(ValueError,'Exact zero-rank'):p.verify_feature_output(got,other,got['audit'],p.state_values(got['audit']))
    def test_complete_support_counts_and_source_corruptions(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args)
        support=o.support_oracle(got['audit'],args[0],got['arrays'])
        self.assertEqual(support['candidate_cells'],5978);self.assertIsNone(support['coverage_threshold'])
        # S0 shares every sixth-session sector decline and is always supported.
        self.assertTrue(all(row['common_support'][0] for row in got['audit']['event_history'][1:]))
        self.assertTrue(all(not any(row['common_support'][-17:]) for row in got['audit']['event_history']))
        self.assertEqual(support['common_support_cells'],sum(sum(row['common_support']) for row in got['audit']['event_history']))
        for field,value in (('common_support',None),('ranks_recovery',None),('ranks_control',None),('raw_recovery',None),('source_status','UNKNOWN'),('joint_down_counts',19),('sector_etfs','SPY')):
            bad=deepcopy(got['audit']);bad['event_history'][1][field][0]=value
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        for value in (1.1,-1.1,True):
            bad=deepcopy(got['audit']);bad['event_history'][1]['raw_control'][0]=value
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
    def test_support_rejects_bad_endpoints_and_grid(self):
        args=feature_fixture(122,49);got=p.features_oracle(*args)
        for field,value in (('window_start','2014-1-1'),('window_end',None),('window_start',got['audit']['event_history'][1]['window_end'])):
            bad=deepcopy(got['audit']);bad['event_history'][1][field]=value
            with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])
        bad=deepcopy(got['audit']);bad['start_count']=189.
        with self.assertRaises(ValueError):o.support_oracle(bad,args[0],got['arrays'])


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
    def test_recovery_only_and_source_changed_guards(self):
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
            plan=dict(round=165,status='REGISTERED',arms=list(p.NAMES),output_directory=str(directory.resolve()),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),measurement=f.measurement_contract(),selection_contract=f.selection_contract(),actual_features_before_registration=0,actual_orders_before_registration=0,actual_returns_before_registration=0,no_observer_writes=True)
            path=directory/'PLAN.json';path.write_text(json.dumps(plan));(directory/'TRAIN_TRIALS.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'precede all outcomes'):f.bind(f.Reader(tmp),hashlib.sha256(path.read_bytes()).hexdigest(),'prepare')
    def test_exact_queue_and_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory=root/f.OUT;directory.mkdir(parents=True);control=root/f.STATE;control.parent.mkdir(parents=True)
            before=dict(completed=[dict(trials=6)]*163+[dict(trials=29)],registered_queue=[])
            raw=json.dumps(before);(directory/'STATE_BEFORE.json').write_text(raw)
            plan=dict(registered_utc='2026-10-21T01:00:00+00:00',registration_precondition=dict(path=str(control),sha256=hashlib.sha256(raw.encode()).hexdigest()))
            entry=dict(round=165,directory='monthly_recovery165',plan=dict(path=str(directory/'PLAN.json'),sha256='a'*64),registered_utc=plan['registered_utc'])
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
        with patch.dict(sys.modules,{'recovery165_kernel':object()}):
            with self.assertRaisesRegex(ValueError,'imported'):p.independent_imports()
    def test_closure_and_exact_cache_schema(self):
        names={path.name for path in f.source_files()}
        self.assertTrue({'recovery165_independent.py','recovery165_independent_prepare.py','recovery165_independent_outcome.py','test_recovery165_independent.py','monthly_ridge_support70_result_review.py','monthly_paired_target143_outcome_review.py','monthly_latent_state134_prepare_review.py','inputs.py','strict_study.py','sectors.csv.gz'}<=names)
        self.assertEqual(len(f.CACHE_FILES),8);self.assertIn('STATE_VALUES.json',f.CACHE_FILES)
    def test_source_only_static_host_interfaces(self):
        # Parse source only. Never import or execute the new host/runner/kernel.
        root=Path(__file__).resolve().parent
        host=ast.parse((root/'recovery165_host.py').read_text())
        assignments={target.id:node.value for node in host.body if isinstance(node,ast.Assign)
                     for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(set(ast.literal_eval(assignments['CACHE'])),set(f.CACHE_FILES))
        self.assertEqual(ast.literal_eval(assignments['PROTECTED_SHA']),'89e82ac5cbec3b998f1d8aab93f1f8fd7c5e19522321b2dd1da2f88dfff922cb')
        runner=ast.parse((root/'recovery165_runner.py').read_text())
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
        scope={'NAMES':p.NAMES,'EXTRA':p.EXTRA,'MAX_ABS_DELTA':.02,'MAPPING_SHA':f.MAPPER_SHA}
        exec(compile(ast.Module(body=declarations,type_ignores=[]),'static_literal_contracts','exec'),scope)
        self.assertEqual(scope['measurement_contract'](),f.measurement_contract())
        self.assertEqual(scope['selection_contract'](),f.selection_contract())
        p.independent_imports()
    def test_source_review_cannot_claim_certified_historical_sectors(self):
        review=dict(passed=True,conditional_research_sources_ready=True,strict_pit_vintage=False,
                    strict_pit_membership=False,strategy_admitted=False,proposal_sha256='synthetic')
        reader=SimpleNamespace(json=lambda path:review,sha=lambda path:'synthetic')
        for value in (None,True,'false'):
            review['strict_pit_sector_classification']=value
            with self.assertRaisesRegex(ValueError,'Conditional metadata review'):
                f.source_inputs(reader,{}, {},dict(source_review='review',proposal='proposal'))
        review.pop('strict_pit_sector_classification')
        with self.assertRaisesRegex(ValueError,'Conditional metadata review'):
            f.source_inputs(reader,{}, {},dict(source_review='review',proposal='proposal'))
    def test_frozen_loader_closure_and_price_parity(self):
        import monthly_latent_state134_prepare_review as old
        root=Path('/private/tmp/recovery165_synthetic_loader');loader_path=root/old.LOADER
        raw={str(root/('source_%d'%i)):old.LOADER_SHA for i in range(63)};raw[str(loader_path)]=old.LOADER_SHA
        index=pd.bdate_range('2020-01-01',periods=3)
        cached=dict(dates=np.array([str(x.date()) for x in index]),symbols=np.array(['A']),
                    close=np.array([[100.],[110.],[90.]]),opening=np.array([[99.],[109.],[89.]]))
        source=dict(adj=pd.DataFrame(200.,index=index,columns=['A']))
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
            close,opening=f.prices(SyntheticReader(),dict(raw_source_hashes=raw),raw,dict(prices='prices.npz'),dict(etfs=[]),True)
            parity.assert_called_once();self.assertEqual(close.iloc[0,0],100.);self.assertEqual(opening.iloc[0,0],99.)
            info['source_hashes'].pop(next(iter(info['source_hashes'])))
            with self.assertRaisesRegex(ValueError,'Actually consumed'):
                f.prices(SyntheticReader(),dict(raw_source_hashes=raw),raw,dict(prices='prices.npz'),dict(etfs=[]),True)
    def test_independent_mapping_from_frozen_ast_and_exact_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapper=root/f.MAPPER;mapper.parent.mkdir(parents=True)
            mapper.write_bytes((Path(__file__).resolve().parents[1]/f.MAPPER).read_bytes())
            sectors=root/f.SECTORS;sectors.parent.mkdir(parents=True)
            pd.DataFrame({'IBM':['Technology'],'X':['Unknown'],'ASML':['Energy']},index=pd.to_datetime(['2020-01-31'])).to_csv(sectors,compression='gzip')
            reader=f.Reader(root);pins={str(mapper.resolve()):reader.sha(mapper),str(sectors.resolve()):reader.sha(sectors)}
            names=['IBM','X','ASML','ETF'];meta=dict(dates=['2020-01-31','2020-02-28'],symbols=[names[:],names[:]],etfs=['ETF'])
            actual=f.sector_mapping(reader,pins,dict(sector_mapper=str(mapper),sectors=str(sectors)),meta)
            self.assertEqual(actual['by_date']['2020-01-31'],dict(IBM='XLK',X=None,ASML='XLK',ETF=None))
            self.assertEqual(actual['by_date']['2020-02-28'],dict(IBM=None,X=None,ASML='XLK',ETF=None))
            # ETF classification takes precedence over the inherited foreign assumption.
            meta['etfs'].append('ASML')
            self.assertIsNone(f.sector_mapping(reader,pins,dict(sector_mapper=str(mapper),sectors=str(sectors)),meta)['by_date']['2020-01-31']['ASML'])
            mapper.write_text(mapper.read_text()+'\n# changed')
            with self.assertRaises(ValueError):f.sector_mapping(reader,pins,dict(sector_mapper=str(mapper),sectors=str(sectors)),meta)
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

