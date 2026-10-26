"""Synthetic-only202 rank-slot wiring, selection and source gate tests."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_breadth202_host', Path(__file__).with_name('monthly_breadth_pair202_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def add_ranks(values):
    for key in ('candidate','control'):
        result=np.zeros_like(values[key],dtype=float)
        for i,valid in enumerate(values['available']):
            ids=np.flatnonzero(valid)
            if len(ids)>1:
                for j in ids:
                    x=values[key][i,ids]
                    result[i,j]=(np.count_nonzero(x<values[key][i,j])+(np.count_nonzero(x==values[key][i,j])-1)/2)/(len(ids)-1)-.5
        values[key+'_rank']=result
    return values

def fixture():
    shape=(6,5)
    values=dict(candidate=np.tile([0.,.2,.4,.6,.8],(6,1)),control=np.tile([.99,.98,.97,.96,.95],(6,1)),available=np.ones(shape,bool))
    values['available'][2,:4]=False;values['available'][3,:]=False
    for key in ('candidate','control'):values[key][~values['available']]=np.nan
    add_ranks(values)
    a=dict(baseline=np.tile([.2,.199,.198,.202,.196],(6,1)),eligible=np.ones(shape,bool),effects=np.zeros(shape),core=np.zeros(shape),core_weight=np.full((6,1),.52),state_calm10=np.zeros(6,bool),state_weak_narrow=np.zeros(6,bool))
    a['eligible'][:,-1]=False
    return a,dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE') for _ in range(6)]),values


def counts_fixture(eight=2,nine=1):
    available=eight+nine;m=8*eight+9*nine
    return dict(available_by_event=[available]+[0]*121,eight_sector_cells_by_event=[eight]+[0]*121,
        nine_sector_cells_by_event=[nine]+[0]*121,available_cells=available,
        valid_price_observations=253*(2*available+m),excess_block_observations=12*available,
        breadth_block_observations=12*available,down_sign_comparisons=12*m,
        covariance_estimates=available,raw_score_observations=2*available,ranked_score_observations=2*available)


class SyntheticStudy:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp = SimpleNamespace(make_orders=self.orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def modified_pre_score(specification, a, n):
        assert specification == dict(id=4)
        return a['baseline'].copy() + .11  # Includes all already-frozen slots.

    @staticmethod
    def boundary(pre, a, m, n):
        choices = []
        for i in range(n):
            ids = sorted(np.flatnonzero(a['eligible'][i]), key=lambda j: (-pre[i, j], j))
            choices.append([m['symbols'][i][j] for j in ids[:3]] if len(ids) >= 2 else [])
        return choices, pre[:n].copy()

    def choices(self, specification, a, m, n, *, policy):
        return self.boundary(self.modified_pre_score(specification, a, n), a, m, n)

    @staticmethod
    def orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=symbols, scores=scores[i].tolist()) for i, symbols in enumerate(choices)]

    def choose(self, base, weight, extra, a, m, n, *, pre_score, effects, episode_policy, boundary_selector):
        assert base == 'incumbent66' and weight == 0.
        assert extra.shape == a['baseline'].shape and np.isnan(extra).all()
        assert effects is a['effects'] and boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score, a, m, n)


class ScoreTests(unittest.TestCase):
    def test_higher_rank_exact_bound_ineligible_receives_original_slot(self):
        a,m,v=fixture();r,d=h.scale_and_delta(v,6)
        np.testing.assert_array_equal(r[h.NAMES[1]][1],[-.5,-.25,0.,.25,.5])
        np.testing.assert_array_equal(d[h.NAMES[1]][1],[-.005,-.0025,0.,.0025,.005])
        self.assertFalse(a['eligible'][1,-1]);self.assertEqual(d[h.NAMES[1]][1,-1],.005)
        self.assertTrue(np.all(d[h.NAMES[1]][0]==0))
    def test_missing_singleton_and_ties_neutral(self):
        _,_,v=fixture();v['candidate'][1]=.1;add_ranks(v);r,d=h.scale_and_delta(v,6)
        for i in (1,2,3):np.testing.assert_array_equal(d[h.NAMES[1]][i],np.zeros(5))
    def test_average_ties(self):
        _,_,v=fixture();v['candidate'][1]=[0.,.4,.4,.4,.8];add_ranks(v)
        r,d=h.scale_and_delta(v,6);np.testing.assert_array_equal(r[h.NAMES[1]][1],[-.5,0.,0.,0.,.5])
    def test_corrupt_rank_bounds_missing_and_raw_rejected(self):
        for kind in ('rankwrong','rankinf','rankbound','missingrank','rawnan','rawinf','extra','mask','shape'):
            _,_,v=fixture()
            if kind=='rankwrong':v['candidate_rank'][1,0]=0.
            if kind=='rankinf':v['candidate_rank'][1,0]=np.inf
            if kind=='rankbound':v['candidate_rank'][1,0]=.5000001
            if kind=='missingrank':v['candidate_rank'][3,0]=.1
            if kind=='rawnan':v['candidate'][1,0]=np.nan
            if kind=='rawinf':v['control'][1,0]=np.inf
            if kind=='extra':v['extra']=np.ones((6,5))
            if kind=='mask':v['available']=v['available'].astype(int)
            if kind=='shape':v['control_rank']=v['control_rank'][:,:-1]
            with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):h.scale_and_delta(v,6)
    def test_future_feature_values_not_inspected(self):
        _,_,v=fixture();rr,dd=h.scale_and_delta(v,6)
        for n in range(7):
            vv=copy.deepcopy(v)
            for key in ('candidate','control','candidate_rank','control_rank'):
                vv[key]=vv[key].astype(object);vv[key][n:]='future poison'
            r,d=h.scale_and_delta(vv,n)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(r[arm],rr[arm][:n]);np.testing.assert_array_equal(d[arm],dd[arm][:n])
    def test_invalid_prefix(self):
        _,_,v=fixture()
        for n in (-1,7,True,1.5,np.int64(2)):
            with self.assertRaises(ValueError):h.scale_and_delta(v,n)
    def test_full_score_exactly_one_episode_no_input_mutation(self):
        a,m,v=fixture();aa,vv=copy.deepcopy(a),copy.deepcopy(v);study=SyntheticStudy();policy={'synthetic':True}
        original=study.modified_pre_score(dict(id=4),a,6)
        with patch.object(h,'read',return_value=policy):orders,r,d=h.decisions(study,a,m,v,6)
        self.assertEqual(study.adapter.score_and_choose.call_count,2);self.assertEqual(study.ep.select_boundary.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            expected=original.copy();mask=d[arm]!=0;expected[mask]+=d[arm][mask]
            np.testing.assert_array_equal(call.kwargs['pre_score'],expected);self.assertIs(call.kwargs['episode_policy'],policy)
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
        for k in a:np.testing.assert_array_equal(a[k],aa[k])
        for k in v:np.testing.assert_array_equal(v[k],vv[k])
    def test_all_decision_prefixes_and_zero_missing_fallback(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):
            oo,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                small,_,_=h.decisions(SyntheticStudy(),a,m,v,n)
                for arm in h.NAMES:self.assertEqual(small[arm],oo[arm][:n])
            for key in ('candidate','control'):v[key][:]=np.nan
            v['available'][:]=False;add_ranks(v)
            neutral,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
            self.assertTrue(all(neutral[arm]==neutral[h.NAMES[0]] for arm in h.NAMES))


def selection_fixture():
    metrics = {n: {p: {e: dict(cagr=.5, sharpe=1.5, max_drawdown=-.3) for e in ('training', 'early', 'recent')}
                  for p in ('same_close_zero', 'next_open_cost')} for n in h.NAMES}
    orders = {n: [dict(symbols=['A', 'B', 'C']) for _ in range(122)] for n in h.NAMES}
    for path in metrics[h.NAMES[1]]: metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
    return metrics, orders


class SelectionAndBindingTests(unittest.TestCase):
    def test_later34_and_extra_metrics_never_choose_arm(self):
        metrics, orders = selection_fixture(); expected = h.select_first88(metrics, orders)
        self.assertEqual(expected['primary'], h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            for i in range(88, 122): orders[name][i]['symbols'] = [name]
        self.assertEqual(h.select_first88(metrics, orders), expected)

    def test_relative_mean_control_cannot_promote_and_must_be_beaten(self):
        for path in ('same_close_zero', 'next_open_cost'):
            metrics, orders = selection_fixture(); metrics[h.NAMES[2]][path]['training']['cagr'] = 2.
            result = h.select_first88(metrics, orders)
            self.assertEqual(result['primary'], h.NAMES[0]); self.assertFalse(result['admitted'])

    def test_exact_gain_era_risk_and_changed_set_guards(self):
        for path, era, key, value in [('same_close_zero', 'training', 'cagr', .504),
                ('next_open_cost', 'training', 'cagr', .49), ('same_close_zero', 'early', 'cagr', .474),
                ('same_close_zero', 'recent', 'cagr', .474), ('same_close_zero', 'training', 'sharpe', 1.449),
                ('same_close_zero', 'training', 'max_drawdown', -.331)]:
            metrics, orders = selection_fixture(); metrics[h.NAMES[1]][path][era][key] = value
            self.assertFalse(h.select_first88(metrics, orders)['admitted'])
        metrics, orders = selection_fixture()
        for i in (1, 2, 53): orders[h.NAMES[1]][i]['symbols'] = ['C', 'B', 'A']
        for i in (90, 100, 110): orders[h.NAMES[1]][i]['symbols'] = ['A', 'B', 'D']
        self.assertEqual(h.select_first88(metrics, orders)['changed_months'], [])

    def test_candidate_must_change_holdings_vs_control_in_both_eras(self):
        metrics, orders = selection_fixture()
        orders[h.NAMES[2]] = copy.deepcopy(orders[h.NAMES[1]])
        decision = h.select_first88(metrics, orders)
        self.assertFalse(decision['admitted'])
        self.assertEqual(decision['changed_months_vs_control'], [])
        for i in (1,2,3): orders[h.NAMES[2]][i]['symbols'] = ['A','E','F']
        self.assertFalse(h.select_first88(metrics, orders)['guards']['changed_support_vs_control'])
        orders[h.NAMES[2]][53]['symbols'] = ['A','E','F']
        self.assertTrue(h.select_first88(metrics, orders)['admitted'])


    def test_inherited_pin_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fake-source'; path.write_text('synthetic')
            pins = {}; h.bind(pins, path); before = dict(pins)
            h.bind(pins, path, h.sha(path)); self.assertEqual(pins, before)
            path.write_text('changed')
            with self.assertRaises(ValueError): h.bind(pins, path)
            self.assertEqual(pins, before)

    def test_rejected_plan_cannot_load_source_or_compute_features(self):
        with patch.object(h, 'gate', side_effect=ValueError('unregistered')), patch.object(h, 'local') as loader:
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                h.prepare('bad-plan-sha')
        loader.assert_not_called()




class IntegrationTests(unittest.TestCase):
    def test_signed_raw_above_control_and_calm_keep_rank_dose(self):
        a,m,v=fixture();a['state_calm10'][1]=True;a['core_weight'][1]=.5
        v['candidate'][1]=[-8.,-2.,0.,4.,12.];v['control'][1]=[-12.,-4.,0.,2.,8.];add_ranks(v)
        r,d=h.scale_and_delta(v,6)
        np.testing.assert_array_equal(d[h.NAMES[1]][1],[-.005,-.0025,0.,.0025,.005])
        with patch.object(h,'read',return_value={'synthetic':True}):
            o,_,_=h.decisions(SyntheticStudy(),a,m,v,6)
        self.assertNotEqual(o[h.NAMES[1]][1]['scores'],o[h.NAMES[0]][1]['scores'])

    def test_nonzero_delta_applies_to_zero_score_and_zero_delta_keeps_negative_zero(self):
        a,m,v=fixture();study=SyntheticStudy()
        pre=np.zeros(a['baseline'].shape);pre[:,2]=-0.
        study.modified_pre_score=Mock(side_effect=lambda *args:pre.copy())
        with patch.object(h,'read',return_value={'synthetic':True}):
            _,_,d=h.decisions(study,a,m,v,6)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            actual=call.kwargs['pre_score']
            self.assertEqual(actual[1,4],d[arm][1,4]);self.assertNotEqual(actual[1,4],0.)
            self.assertTrue(np.signbit(actual[1,2]))

    def test_real_frozen_path_adapter_keeps_original_support_and_returned_choices(self):
        a,m,v=fixture();saved=copy.deepcopy(a);study=SyntheticStudy()
        path=h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
        study.adapter=h.module('synthetic_202_real_adapter',path)
        calls=[]
        def boundary(pre,eligible,symbols,effects,original,policy):
            calls.append((pre.copy(),eligible.copy(),effects.copy(),copy.deepcopy(original)))
            choices=[]
            for row,good in zip(pre,eligible):
                ids=list(np.flatnonzero(good));choices.append([ids[-1],ids[0]] if len(ids)>=2 else [])
            return [[ss[j] for j in ids] for ss,ids in zip(symbols,choices)],pre.copy()
        study.ep.select_boundary=boundary
        policy=dict(budget=.05,ceiling=5,gap=.02)
        with patch.object(h,'read',return_value=policy):orders,r,d=h.decisions(study,a,m,v,6)
        self.assertEqual(len(calls),2)
        base=study.modified_pre_score(dict(id=4),a,6)
        for arm,(pre,good,effects,original) in zip(h.NAMES[1:],calls):
            expected=base.copy();mask=d[arm]!=0;expected[mask]+=d[arm][mask]
            np.testing.assert_array_equal(pre,expected)
            for key in ('baseline','core','state_calm10','state_weak_narrow'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal((original['baseline']-original['core'])/(1-original['core_weight']),
                                         (a['baseline']-a['core'])/(1-a['core_weight']))
            np.testing.assert_array_equal(good,a['eligible']);np.testing.assert_array_equal(effects,a['effects'])
            self.assertEqual(orders[arm][1]['symbols'],['D','A'])
        for key in a:np.testing.assert_array_equal(a[key],saved[key])

    def test_actual_producer_oracle_host_interfaces_all_prefixes(self):
        import monthly_breadth_pair202_kernel as kernel
        import monthly_breadth_pair202_independent as independent
        from test_monthly_breadth_pair202_kernel import fixture as source_fixture,mapper
        panel,meta,calendar=source_fixture();values,audit=kernel.build_features(panel,meta,calendar,mapper)
        self.assertTrue(independent.verify_features(panel,meta,calendar,mapper,values,audit)['passed'])
        for n in range(len(meta['dates'])+1):
            small={key:val[:n] for key,val in values.items()};r,d=h.scale_and_delta(values,n)
            check=independent.verify_deltas(small,d,dose=h.DOSE);h.require_delta_audit(check,d)
            pp=h.feature_prefix_panel(panel,meta,n)
            got,aa=kernel.build_features(pp,meta,calendar,mapper,limit=n)
            for key in got:np.testing.assert_array_equal(got[key],values[key][:n])
            self.assertEqual(aa['event_history'],audit['event_history'][:n])

    def test_source_admission_cell_support_and_identity_rechecked(self):
        import monthly_breadth_pair202_kernel as kernel
        from test_monthly_breadth_pair202_kernel import fixture as source_fixture,mapper
        p,m,c=source_fixture();v,a=kernel.build_features(p,m,c,mapper)
        rows=[]
        for event in a['event_history']:
            rows.append(dict(query=event['date'],price_start=event['price_start'],price_end=event['price_end'],
                benchmarks=event['benchmarks'],endpoint_dates=event['endpoint_dates'],
                cells=[{key:cell[key] for key in ('symbol','benchmark','breadth_etfs','available')} for cell in event['cells']]))
        coverage=dict(common_available_cells=a['available_cells'],accepted_source_price_observations=a['valid_price_observations'],
            available_cells_by_basket_size={'8':sum(a['eight_sector_cells_by_event']),'9':sum(a['nine_sector_cells_by_event'])},by_query=rows)
        with patch.object(h,'read',return_value={'source_window_coverage':coverage}):self.assertTrue(h.require_source_support(a,v))
        for kind in ('mask','basket','date','mapping','count'):
            bad=copy.deepcopy(coverage)
            if kind=='mask':bad['by_query'][0]['cells'][0]['available']=False
            if kind=='basket':bad['by_query'][0]['cells'][0]['breadth_etfs']=[]
            if kind=='date':bad['by_query'][0]['price_end']='2099-01-01'
            if kind=='mapping':bad['by_query'][0]['benchmarks']['A']='XLF'
            if kind=='count':bad['accepted_source_price_observations']+=1
            with patch.object(h,'read',return_value={'source_window_coverage':bad}),self.assertRaises(ValueError):h.require_source_support(a,v)


class SourceAndLifecycleTests(unittest.TestCase):
    def test_metadata_source_sector_binding_and_feature_boundary_flags(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.price_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        sector=h.sector_source_binding(raw)
        self.assertEqual(set(sector['sources']),set(h.SECTOR_ETFS))
        self.assertEqual(sector['mapping_source']['sha256'],h.MAPPING_SHA)
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64):
            with self.assertRaises(ValueError):h.price_source_binding(raw)
        missing=dict(raw);missing.pop(str(h.SECTORS_PATH))
        with self.assertRaises(ValueError):h.price_source_binding(missing)
        with self.assertRaises(ValueError):h.sector_source_binding(missing)
        receipt=h.read(h.SOURCE_REVIEW)
        for key in ('actual_signal_returns_computed','actual_breadth_computed','actual_covariance_computed','actual_ranks_computed','actual_deltas_computed'):
            changed=copy.deepcopy(receipt);changed[key]=True
            with patch.object(h,'read',return_value=changed),self.assertRaises(ValueError):h.price_source_binding(raw)

    def test_predecessor_metadata_and_central_counter_contract(self):
        state=h.read(h.CENTRAL);binding=h.predecessor_binding(state)
        self.assertEqual(binding['directory'],'monthly_median_risk201')
        rows=[dict(trials=1,directory='old') for _ in range(199)];rows[0]['trials']=916;rows[-1]['directory']=h.PRIOR.name
        synthetic=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(synthetic);synthetic['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(synthetic)
        text=Path(h.__file__).read_text();self.assertIn("len(state['completed']) == 200",text);self.assertIn("== 1117",text)

    def test_independent_code_review_gate_rejects_missing_or_stale(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        with patch.object(h,'read',return_value={'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with self.assertRaises(ValueError):h.code_review_binding()

    def test_source_prefix_zero_physically_empty_in_both_frames(self):
        idx=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'])
        p={'adj':pd.DataFrame({'A':[1.,2.,'future poison']},index=idx),
           'sectors':pd.DataFrame({'A':['Technology','Technology',object()]},index=idx)}
        m={'dates':['2020-01-03','2020-01-06']}
        self.assertTrue(all(len(x)==0 for x in h.feature_prefix_panel(p,m,0).values()))
        small=h.feature_prefix_panel(p,m,1);self.assertTrue(all(len(x)==2 for x in small.values()))
        self.assertEqual(small['adj'].iloc[-1,0],2.);self.assertEqual(small['sectors'].iloc[-1,0],'Technology')
        with self.assertRaises(ValueError):h.feature_prefix_panel({'adj':p['adj']},m,1)

    def test_fixed_contract_and_exact_basket_consumption_arithmetic(self):
        c=h.contract();self.assertEqual(c['dose'],.01);self.assertEqual(c['max_absolute_delta'],.005)
        self.assertEqual(c['source_window']['price_count'],253);self.assertEqual(c['source_window']['latest_offset'],-42)
        self.assertIn(h.ROOT/'work/monthly_nested_trend199_kernel.py',h.code_files())
        self.assertEqual(len(h.PREPARED_FILES),7);self.assertEqual(c['execution_paths'],9)
        for eight,nine in ((0,0),(1,0),(0,1),(40,9)):
            counts=counts_fixture(eight,nine);self.assertEqual(h.estimate_counts(counts)['valid_price_observations'],253*(10*eight+11*nine))
            for key in h.ESTIMATE_FIELDS:
                bad=copy.deepcopy(counts);bad[key]+=1
                with self.assertRaises(ValueError):h.estimate_counts(bad)

    def test_counts_require_common_support_integer_width_and_prefix_length(self):
        for change in ('support','total','bool','width','length'):
            counts=counts_fixture()
            if change=='support':counts['eight_sector_cells_by_event'][0]=3
            if change=='total':counts['available_cells']=4
            if change=='bool':counts['available_by_event'][0]=True
            if change=='width':counts['available_by_event'][0]=50
            if change=='length':counts['available_by_event'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):h.estimate_counts(counts)

    def test_prepared_receipt_requires_all_invariants_and_counts(self):
        mask=np.zeros((122,49),bool);mask[0,:3]=True
        counts=dict(counts_fixture(),unavailable_cells=5975)
        feature=dict({k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
            computes_returns=False,predictive_model_fits=0,new_model_fits=0,mapping_source_sha256=h.MAPPING_SHA,**counts)
        deltas={arm:np.zeros((122,49)) for arm in h.NAMES[1:]}
        keys=('passed','full_candidate_universe','initial_delta_zero','common_available_only','no_eligibility_or_model_filter','centered_cross_sectional_ranking','direct_additive_pre_score','fixed_linear_mapping_checked')
        delta=dict({k:True for k in keys},dose=.01,max_absolute_delta=.005,changed_cells={k:0 for k in deltas})
        base=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**{k:counts[k] for k in h.ESTIMATE_FIELDS})
        audit=dict(base,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            feature_prefix_sources_truncated=True,source_support_exact=True,original_orders_exact=True,original_arrays_unchanged=True,
            original_core_metadata_and_support_preserved=True,single_original_episode=True)
        receipt=dict(base,new_returns_computed=False,artifacts_sha256={k:'synthetic' for k in h.PREPARED_FILES})
        h.verify_prepared_evidence(receipt,{'available':mask},deltas,audit)
        for key in ('source_support_exact','original_core_metadata_and_support_preserved','single_original_episode','original_arrays_unchanged','feature_prefix_sources_truncated'):
            bad=copy.deepcopy(audit);bad[key]=False
            with self.assertRaises(ValueError):h.verify_prepared_evidence(receipt,{'available':mask},deltas,bad)
        for key in ('new_predictive_model_fits',)+h.ESTIMATE_FIELDS:
            bad=copy.deepcopy(receipt);bad[key]=None
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,{'available':mask},deltas,audit)
        for key in h.FEATURE_FLAGS:
            bad=dict(feature);bad.pop(key)
            with self.assertRaises(ValueError):h.require_feature_audit(bad,{'available':mask})
        for key in keys:
            bad=dict(delta);bad.pop(key)
            with self.assertRaises(ValueError):h.require_delta_audit(bad,deltas)
    def test_outcomes_require_all_three_arms_nine_paths_fortyfive_slices(self):
        slices={'full122':{},'first88':{},'early53':{},'recent35':{},'later34':{}}
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,
            accounting_source_sha256=h.sha(h.ROOT/'work/ptr169_results_independent.py'),
            paths={arm:{p[0]:dict(passed=True,monthly_executions=122,slices=copy.deepcopy(slices)) for p in h.ev.PATHS} for arm in h.NAMES})
        h.require_outcome_audit(check)
        for mutation in ('arm','path','slice','terminal'):
            bad=copy.deepcopy(check)
            if mutation=='arm':bad['paths'].pop(h.NAMES[2])
            if mutation=='path':bad['paths'][h.NAMES[1]].pop(h.ev.PATHS[2][0])
            if mutation=='slice':bad['paths'][h.NAMES[1]][h.ev.PATHS[0][0]]['slices'].pop('later34')
            if mutation=='terminal':bad['terminal_liquidation']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):h.require_outcome_audit(bad)
    def test_write_once_and_protection_before_publishing_and_central_cas(self):
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        parts={x.name:ast.get_source_segment(code,x) for x in tree.body if isinstance(x,ast.FunctionDef)}
        register=parts['register'];self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))
        self.assertLess(register.index('len(protected) == 21444'),register.index('OUT.mkdir()'))
        self.assertIn('Code changed since tests',register)
        self.assertEqual(parts['commit_state'].count('CENTRAL.read_bytes() == before'),2)
        self.assertIn("not (OUT / 'prepared').exists()",parts['prepare'])
        self.assertIn("not (OUT / 'selected').exists()",parts['select'])
        self.assertIn('Selection escaped first88 cutoff',parts['select'])


if __name__=='__main__':unittest.main()

