"""Synthetic-only195 rank-slot wiring, selection and source gate tests."""
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

spec = importlib.util.spec_from_file_location('tested_volumecluster195_host', Path(__file__).with_name('monthly_volume_cluster195_host.py'))
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
    values=dict(candidate=np.tile([-.2,-.1,0.,.1,.2],(6,1)),control=np.tile([.9,.7,.5,.3,.1],(6,1)),available=np.ones(shape,bool))
    values['available'][2,:4]=False;values['available'][3,:]=False
    for key in ('candidate','control'):values[key][~values['available']]=np.nan
    add_ranks(values)
    a=dict(baseline=np.tile([.2,.199,.198,.197,.196],(6,1)),eligible=np.ones(shape,bool),effects=np.zeros(shape),core=np.zeros(shape),core_weight=np.full((6,1),.52),state_calm10=np.zeros(6,bool),state_weak_narrow=np.zeros(6,bool))
    a['eligible'][:,-1]=False
    return a,dict(dates=['synthetic-%d'%i for i in range(6)],symbols=[list('ABCDE') for _ in range(6)]),values


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
        _,_,v=fixture();v['candidate'][1]=[-.2,0.,0.,0.,.2];add_ranks(v)
        r,d=h.scale_and_delta(v,6);np.testing.assert_array_equal(r[h.NAMES[1]][1],[-.5,0.,0.,0.,.5])
    def test_corrupt_rank_bounds_missing_and_raw_rejected(self):
        for kind in ('rankwrong','rankinf','rankbound','missingrank','rawnan','rawinf','rawbound','mask','shape'):
            _,_,v=fixture()
            if kind=='rankwrong':v['candidate_rank'][1,0]=0.
            if kind=='rankinf':v['candidate_rank'][1,0]=np.inf
            if kind=='rankbound':v['candidate_rank'][1,0]=.5000001
            if kind=='missingrank':v['candidate_rank'][3,0]=.1
            if kind=='rawnan':v['candidate'][1,0]=np.nan
            if kind=='rawinf':v['control'][1,0]=np.inf
            if kind=='rawbound':v['candidate'][1,0]=.251
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

    def test_absolute_control_cannot_promote_and_must_be_beaten(self):
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



class SourceAndLifecycleTests(unittest.TestCase):
    def test_metadata_source_binding_pins(self):
        raw=h.read(h.LOCAL/'PLAN.json')['raw_source_hashes'];binding=h.volume_source_binding(raw)
        self.assertEqual(binding['sha256'],h.SOURCE_REVIEW_SHA)
        with patch.object(h,'SOURCE_REVIEW_SHA','0'*64):
            with self.assertRaises(ValueError):h.volume_source_binding(raw)
        missing=dict(raw);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):h.volume_source_binding(missing)
    def test_independent_review_gate_rejects_stale_or_failed(self):
        with patch.object(h,'read',return_value={'passed':False}):
            with self.assertRaises(ValueError):h.code_review_binding()
        with patch.object(h,'read',return_value={'passed':True,'unresolved_blockers':[],'code_sha256':{}}):
            with self.assertRaises(ValueError):h.code_review_binding()
    def test_original_volume_and_prefix_zero(self):
        volume=pd.DataFrame({'A':[1.,2.,3.]},index=pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06']))
        source={'volume':volume};m={'etfs':[str(i) for i in range(17)],'dates':['2020-01-03','2020-01-06']}
        p=h.volume_inputs(source,m);self.assertIs(p['volume'],volume)
        self.assertEqual(len(h.feature_prefix_panel(p,m,0)['volume']),0)
        self.assertEqual(len(h.feature_prefix_panel(p,m,1)['volume']),2)
        with self.assertRaises(ValueError):h.volume_inputs({},m)
    def test_counts_source_period_and_contract_fixed(self):
        c=h.contract();self.assertEqual(c['dose'],.01);self.assertEqual(c['max_absolute_delta'],.005)
        self.assertEqual(c['source_window']['volume_count'],273);self.assertEqual(c['source_window']['latest_offset'],42)
        self.assertIn('UNKNOWN',c['context']['novelty_scope']);self.assertIn(h.CONTEXT,h.code_files())
        ok=dict(available_cells=3,event_observations=630,adjacent_pair_observations=627,rolling_medians_computed=630)
        self.assertEqual(h.estimate_counts(ok)['rolling_medians_computed'],630)
        for key in h.ESTIMATE_FIELDS:
            bad=dict(ok);bad[key]+=1
            with self.assertRaises(ValueError):h.estimate_counts(bad)
    def test_completed_predecessor_state(self):
        rows=[dict(trials=1,directory='old') for _ in range(192)];rows[0]['trials']=902;rows[-1]['directory']=h.PRIOR.name
        state=dict(completed=rows,best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state);state['completed'][-1]['directory']='wrong'
        with self.assertRaises(ValueError):h.state_check(state)
    def test_gate_before_real_loader_and_no_missing_review(self):
        with patch.object(h,'gate',side_effect=ValueError('unregistered')),patch.object(h,'local') as loader:
            with self.assertRaises(ValueError):h.prepare('bad')
        loader.assert_not_called()
        code=Path(h.__file__).read_text();tree=ast.parse(code)
        register=ast.get_source_segment(code,next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='register'))
        self.assertLess(register.index('code_review_binding()'),register.index('OUT.mkdir()'))

if __name__=='__main__':unittest.main()
