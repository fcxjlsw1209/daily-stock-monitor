"""Synthetic-only component routing, original episode, and governance tests for182."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('tested_trend182_host',Path(__file__).with_name('monthly_trend_gate182_host.py'))
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape=(6,5)
    a=dict(baseline=np.tile([.2,.199,.198,.197,.196],(6,1)),core=np.zeros(shape),
           core_weight=np.full((6,1),.52),eligible=np.ones(shape,bool),effects=np.zeros(shape),
           state_calm10=np.zeros(6,bool),state_weak_narrow=np.zeros(6,bool),
           local_increment=np.tile([.01,-.005,0.,.005,0.],(6,1)))
    a['eligible'][:,-1]=False
    values=dict(consistent=np.tile([True,False,True,False,False],(6,1)),available=np.ones(shape,bool))
    values['available'][2,0]=False;values['consistent'][2,0]=False
    values['available'][3]=False;values['consistent'][3]=False
    m=dict(dates=['2020-0%d-28'%(i+1) for i in range(6)],symbols=[list('ABCDE') for _ in range(6)])
    return a,m,values


class SyntheticStudy:
    def __init__(self):
        self.scope=SimpleNamespace(modified_pre_score=Mock(side_effect=self.upstream))
        self.score_delta=Mock(side_effect=self.increment)
        self.source_spec=Mock(side_effect=lambda spec: dict(upstream=spec['id']))
        self.modified_pre_score=Mock(side_effect=self.pre)
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp=SimpleNamespace(make_orders=self.orders)

    @staticmethod
    def configs(): return [dict(id=i) for i in range(5)]

    @staticmethod
    def upstream(spec,a,n):
        assert spec==dict(upstream=4)
        out=np.zeros(a['baseline'].shape);out[:n]=a['baseline'][:n]+.11
        return out

    @staticmethod
    def increment(spec,a,n):
        assert spec==dict(id=4)
        return np.asarray(a['local_increment'][:n],float).copy()

    def pre(self,spec,a,n):
        out=self.upstream(dict(upstream=spec['id']),a,n)
        d=self.increment(spec,a,n);v=out[:n].copy();use=d!=0;v[use]+=d[use];out[:n]=v
        return out

    @staticmethod
    def boundary(pre,a,m,n):
        chosen=[]
        for i in range(n):
            ids=sorted(np.flatnonzero(a['eligible'][i]),key=lambda j:(-pre[i,j],m['symbols'][i][j]))
            chosen.append([int(j) for j in ids[:3]] if len(ids)>=2 else [])
        return chosen,pre[:n].copy()

    def choices(self,spec,a,m,n,*,policy): return self.boundary(self.pre(spec,a,n),a,m,n)

    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=[meta['symbols'][i][j] for j in ids],
                     scores=[float(scores[i,j]) for j in ids]) for i,ids in enumerate(choices)]

    def choose(self,base,weight,extra,a,m,n,*,pre_score,effects,episode_policy,boundary_selector):
        assert base=='incumbent66' and weight==0. and extra.shape==a['baseline'].shape and np.isnan(extra).all()
        assert effects is a['effects'] and boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score,a,m,n)


class ComponentAndRoutingTests(unittest.TestCase):
    def test_components_are_explicit_original_calls_not_subtraction(self):
        a,_,_=fixture();study=SyntheticStudy()
        result=h.original_components(study,a,6)
        study.modified_pre_score.assert_called_once_with(dict(id=4),a,6)
        study.source_spec.assert_called_once_with(dict(id=4))
        study.scope.modified_pre_score.assert_called_once_with(dict(upstream=4),a,6)
        study.score_delta.assert_called_once_with(dict(id=4),a,6)
        np.testing.assert_array_equal(result['local_increment'],a['local_increment'])
        self.assertFalse(np.array_equal(result['upstream_pre_score'],a['baseline']))

    def test_original_identity_ineligible_and_limit_fail_closed(self):
        for failure in ('identity','ineligible','limit','infinity','nan'):
            a,_,_=fixture();study=SyntheticStudy()
            if failure=='identity': study.modified_pre_score.side_effect=lambda spec,a,n: study.pre(spec,a,n)+.001
            if failure=='ineligible': a['local_increment'][1,-1]=.001
            if failure=='limit': a['local_increment'][1,0]=.0126
            if failure=='infinity': a['local_increment'][1,0]=np.inf
            if failure=='nan': a['local_increment'][1,0]=np.nan
            with self.subTest(failure=failure),self.assertRaises((ValueError,AssertionError)):h.original_components(study,a,6)

    def test_component_prefix_does_not_read_future_increment(self):
        a,_,_=fixture();expected=h.original_components(SyntheticStudy(),a,2)
        a['local_increment']=a['local_increment'].astype(object);a['local_increment'][2:]='future'
        actual=h.original_components(SyntheticStudy(),a,2)
        for k in actual:np.testing.assert_array_equal(actual[k],expected[k])

    def test_feature_boolean_contract_and_prefix(self):
        _,_,v=fixture()
        self.assertEqual(h.feature_prefix(v,0)['available'].shape,(0,5))
        for failure in ('key','shape','type','unavailable'):
            bad=copy.deepcopy(v)
            if failure=='key':bad['raw']=np.zeros((6,5))
            if failure=='shape':bad['consistent']=bad['consistent'][:,:4]
            if failure=='type':bad['available']=bad['available'].astype(int)
            if failure=='unavailable':bad['consistent'][3,0]=True
            with self.subTest(failure=failure),self.assertRaises(ValueError):h.feature_prefix(bad,6)
        for n in (True,False,np.int64(1),1.5,-1,7):
            with self.subTest(n=n),self.assertRaises(ValueError):h.feature_prefix(v,n)

    def test_routing_exact_copies_and_matched_budget(self):
        a,m,v=fixture();study=SyntheticStudy()
        with patch.object(h,'read',return_value={'synthetic':True}):o,r,audit=h.decisions(study,a,m,v,6)
        p,b,d=(r[k] for k in ('pre_local67','upstream_pre_score','local_increment'))
        for arm in h.NAMES[1:]:
            np.testing.assert_array_equal(r[arm][0],p[0])
            np.testing.assert_array_equal(r[arm][3],p[3])
            np.testing.assert_array_equal(r[arm][d==0],p[d==0])
        self.assertEqual(r['trend_gated_local'][1,0],b[1,0])
        self.assertEqual(r['trend_gated_local'][1,1],p[1,1])
        self.assertEqual(r['uniform_removed_fraction'][1],.5)
        np.testing.assert_allclose(r['candidate_removed_abs_increment'],r['control_removed_abs_increment'],atol=1e-12,rtol=0)
        self.assertEqual(r['candidate_removed_abs_increment'][1],.01)
        self.assertEqual(o[h.NAMES[1]][0],o[h.NAMES[0]][0])

    def test_routed_pre_has_one_original_episode_and_no_input_mutation(self):
        a,m,v=fixture();before=copy.deepcopy((a,v));study=SyntheticStudy()
        with patch.object(h,'read',return_value={'synthetic':True}):orders,r,audit=h.decisions(study,a,m,v,6)
        self.assertEqual(study.adapter.score_and_choose.call_count,2)
        self.assertEqual(study.ep.select_boundary.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            np.testing.assert_array_equal(call.kwargs['pre_score'],r[arm])
            self.assertIs(call.args[3],a)
        for key in a:np.testing.assert_array_equal(a[key],before[0][key])
        for key in v:np.testing.assert_array_equal(v[key],before[1][key])

    def test_all_order_and_routing_prefixes(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):
            full,r,audit=h.decisions(SyntheticStudy(),a,m,v,6)
            for n in range(7):
                small,s,_=h.decisions(SyntheticStudy(),a,m,v,n)
                self.assertEqual(small,{k:rows[:n] for k,rows in full.items()})
                for k in h.ROUTING_KEYS:np.testing.assert_array_equal(s[k],r[k][:n])

    def test_independent_routing_receipt_and_tamper(self):
        import monthly_trend_gate182_independent as independent
        a,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):_,r,audit=h.decisions(SyntheticStudy(),a,m,v,6)
        def verify(scores):return independent.verify_routing(r['pre_local67'],r['upstream_pre_score'],r['local_increment'],
            v['consistent'],v['available'],scores,audit)
        scores={arm:r[arm] for arm in h.NAMES[1:]}
        h.require_routing_audit(verify(scores),r)
        wrong=copy.deepcopy(scores);wrong[h.NAMES[1]][1,0]+=.001
        with self.assertRaises(ValueError):verify(wrong)


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

    def test_uniform_shrink_control_cannot_promote_and_must_be_beaten(self):
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

    def test_completed_predecessor_and_counters_required(self):
        state = dict(completed=[dict(trials=6, directory='old')] * 174 + [dict(trials=x, directory='old') for x in (2, 1, 1, 1)] + [dict(trials=3, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)



class AuditReceiptTests(unittest.TestCase):
    def test_feature_receipt_fields_and_coverage_required(self):
        values=dict(available=np.ones((122,49),bool),consistent=np.zeros((122,49),bool))
        check=dict(passed=True,events=122,symbols_per_event=49,checked_cells=5978,available_cells=5978,
            unavailable_cells=0,consistent_cells=0,block_return_estimates=17934,
            independent_scalar_rebuild=True,exact_common_availability=True,no_future_or_outcome_inputs=True,
            audit_checked=True,full_211_price_support_checked=True,three_70_return_blocks_checked=True,
            strict_positive_consistency_checked=True,predictive_model_fits=0,computes_returns=False)
        h.require_feature_audit(check,values)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):h.require_feature_audit(bad,values)
        for key,value in (('block_return_estimates',17931),('consistent_cells',1),('checked_cells',5978.),('predictive_model_fits',False)):
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_feature_audit(dict(check,**{key:value}),values)

    def test_routing_receipt_missing_flags_and_episode_rejected(self):
        a,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):_,r,_=h.decisions(SyntheticStudy(),a,m,v,6)
        p=r['pre_local67']
        check=dict(passed=True,independent_scalar_rebuild=True,original_increment_identity_checked=True,exact_unchanged_cells_checked=True,
            matched_absolute_removal_checked=True,no_cross_sectional_ranking=True,initial_scores_unchanged=True,
            no_new_eligibility_filter=True,original_pre_reconstruction_exact=True,unchanged_cell_copy_exact=True,
            matched_removed_budgets=True,predictive_model_fits=0,episode_applications=0,
            events=6,symbols_per_event=5,checked_cells=30,portfolio_returns_calculated=False,
            uniform_removed_fraction=r['uniform_removed_fraction'].tolist(),
            changed_cells={arm:int(np.count_nonzero(r[arm]!=p)) for arm in h.NAMES[1:]})
        h.require_routing_audit(check,r)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(missing=key),self.assertRaises((ValueError,AssertionError)):h.require_routing_audit(bad,r)
        with self.assertRaises(ValueError):h.require_routing_audit(dict(check,episode_applications=1),r)

    def test_prepared_manifest_invariants_and_budget_receipt(self):
        c=dict(artifacts_sha256={n:'fake' for n in h.PREPARED_FILES},new_returns_computed=False,
            new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,
            model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,block_return_estimates=36)
        detail={k:[0.,.01] for k in h.ROUTING_VECTORS}
        r={k:np.asarray(v) for k,v in detail.items()}
        audit=dict(feature=dict(available_cells=12,block_return_estimates=36),routing={},routing_detail=detail,
            prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),original_orders_exact=True,
            original_arrays_unchanged=True,original_array_count=27,new_model_fits=0,predictive_model_fits=0,
            new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,block_return_estimates=36,
            original_components_direct=True,original_ineligible_increment_zero=True,original_episode_once_per_arm=True)
        with patch.object(h,'require_feature_audit'),patch.object(h,'require_routing_audit'):
            h.verify_prepared_evidence(c,{},r,audit)
            for name in h.PREPARED_FILES:
                bad=copy.deepcopy(c);bad['artifacts_sha256'].pop(name)
                with self.subTest(file=name),self.assertRaises(ValueError):h.verify_prepared_evidence(bad,{},r,audit)
            for key in ('original_components_direct','original_ineligible_increment_zero','original_episode_once_per_arm','original_arrays_unchanged'):
                with self.subTest(key=key),self.assertRaises(ValueError):h.verify_prepared_evidence(c,{},r,dict(audit,**{key:False}))
            bad=copy.deepcopy(audit);bad['routing_detail']['uniform_removed_fraction'][1]=.02
            with self.assertRaises(AssertionError):h.verify_prepared_evidence(c,{},r,bad)

    def test_frozen_training_matches_independent_slices(self):
        training,_=selection_fixture()
        audit=dict(paths={a:{p:dict(slices={full:copy.deepcopy(s[short]) for short,full in
            [('training','first88'),('early','early53'),('recent','recent35')]}) for p,s in paths.items()} for a,paths in training.items()})
        h.verify_training_against_audit(training,audit)
        training[h.NAMES[1]]['same_close_zero']['training']['cagr']+=.01
        with self.assertRaises(ValueError):h.verify_training_against_audit(training,audit)

    def test_nine_paths_five_slices_receipt(self):
        one=dict(passed=True,monthly_executions=122,slices={s:{} for s in ('full122','first88','early53','recent35','later34')})
        check=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
            checked_paths=9,checked_slices=45,terminal_liquidation=False,accounting_source_sha256='fake',
            paths={a:{p[0]:copy.deepcopy(one) for p in h.ev.PATHS} for a in h.NAMES})
        with patch.object(h,'sha',return_value='fake'):
            h.require_outcome_audit(check)
            check['paths'][h.NAMES[1]]['same_close_zero']['slices'].pop('later34')
            with self.assertRaises(ValueError):h.require_outcome_audit(check)


class SourceBindingTests(unittest.TestCase):
    def test_prospective_context_and_declared_local_evidence_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = root / 'source.txt'; evidence.write_text('source')
            context = root / 'context.json'
            context.write_text(json.dumps(dict(local_evidence_sha256={'source.txt': h.sha(evidence)})))
            with patch.multiple(h, ROOT=root, CONTEXT=context, CONTEXT_SHA=h.sha(context)):
                pins = h.context_binding()
                self.assertEqual(set(pins), {str(context), str(evidence)})
                evidence.write_text('changed')
                with self.assertRaises(ValueError): h.context_binding()

    def test_original_source_adjusted_panel_only(self):
        adj = pd.DataFrame({'A': [1., 2.], 'SPY': [2., 3.]})
        source = dict(adj=adj, close=pd.DataFrame({'A': [100., 200.]}))
        result = h.feature_inputs(source)
        self.assertEqual(set(result), {'adj'})
        self.assertIs(result['adj'], adj)
        for invalid in ({}, {'adj': np.ones((2, 2))}):
            with self.assertRaises(ValueError): h.feature_inputs(invalid)


    def test_contract_limits_route_to_existing_increment(self):
        c=h.contract()
        self.assertEqual(c['source_window']['price_count'],211)
        self.assertEqual(c['source_window']['block_sessions'],70)
        self.assertIn('Never infer D by subtraction',c['component_binding'])
        self.assertIn('no infinities',c['numerical_policy'])
        self.assertIn('exactly one original episode',c['scoring'])
        self.assertIn('theoretical L1',c['budget'])
        self.assertIn('only one historical holding month',c['limitation'])
        self.assertIn(h.CONTEXT,h.code_files())
        self.assertTrue(c['selection']['require_noninferior_to_uniform_shrink'])

    def test_block_estimate_count_exact_not_fit_count(self):
        self.assertEqual(h.estimate_counts(dict(available_cells=2,block_return_estimates=6)),{'block_return_estimates':6})
        for count in (True,6.,-3,3,7):
            with self.subTest(count=count),self.assertRaises(ValueError):h.estimate_counts(dict(available_cells=2,block_return_estimates=count))


def predecessor_fixture(root):
    prior = root / 'outputs/monthly_upside_retention181'
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return dict(path=str(path.relative_to(root)), sha256=h.sha(path))
    old_amendment = put(root / 'outputs/old/AMENDMENT.json', dict(old=True))
    old_repair = put(root / 'outputs/old/REPAIR_COMPLETION.json', dict(old=True))
    old_code = put(root / 'work/old_repair.py', 'historical code')
    inherited = {str(root / record['path']): record['sha256'] for record in (old_amendment, old_repair, old_code)}
    before = put(prior / 'STATE_BEFORE.json', dict(active_execution_amendment=old_amendment,
        latest_execution_repair_receipt=old_repair))
    plan = put(prior / 'PLAN.json', dict(input_sha256=inherited,
        predecessor_binding=dict(input_sha256=inherited), prior_central_sha256=before['sha256']))
    result = put(prior / 'results/COMPLETION.json', dict(status='COMPLETE', plan_sha256=plan['sha256']))
    closed = put(prior / 'COMPLETION.json', dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',
        plan_sha256=plan['sha256'], result_completion_sha256=result['sha256'],
        protected_files=21444, observers_modified=False))
    expected = {name: h.sha(prior / name) for name in h.PRIOR_EXPECTED}
    state = dict(completed=[dict(completion=closed)], active_execution_amendment=old_amendment,
        latest_execution_repair_receipt=old_repair)
    return prior, expected, state


class PredecessorTests(unittest.TestCase):
    def test_closed_predecessor_without_new_repair_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); prior, expected, state = predecessor_fixture(root)
            with patch.multiple(h, ROOT=root, PRIOR=prior, PRIOR_EXPECTED=expected):
                binding = h.predecessor_binding(state)
                self.assertFalse((prior / 'AMENDMENT_01.json').exists())
                self.assertFalse((prior / 'REPAIR_COMPLETION_01.json').exists())
                self.assertFalse(binding['predecessor_requires_new_repair'])
                self.assertEqual(binding['inherited_evidence_count'], 3)
                self.assertIn(str(root / 'work/old_repair.py'), binding['input_sha256'])
                for field in ('active_execution_amendment', 'latest_execution_repair_receipt'):
                    wrong = copy.deepcopy(state); wrong.pop(field)
                    with self.assertRaises(ValueError): h.predecessor_binding(wrong)
                    wrong = copy.deepcopy(state); wrong[field]['sha256'] = 'wrong'
                    with self.assertRaises(ValueError): h.predecessor_binding(wrong)

    def test_inherited_historical_code_and_evidence_are_digest_bound(self):
        for relative in ('work/old_repair.py', 'outputs/old/AMENDMENT.json',
                         'outputs/old/REPAIR_COMPLETION.json', 'outputs/monthly_upside_retention181/STATE_BEFORE.json'):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); prior, expected, state = predecessor_fixture(root)
                (root / relative).write_text('changed')
                with patch.multiple(h, ROOT=root, PRIOR=prior, PRIOR_EXPECTED=expected):
                    with self.assertRaises(ValueError): h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
        plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                    prior_central_sha256='sha', predecessor_binding={'registered':True})
        state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
        def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
        with patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
             patch.object(h, 'verify'), patch.object(h, 'state_check'), patch.object(h, 'context_binding', return_value={}), \
             patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
            self.assertEqual(h.gate('sha'), plan)
            predecessor.assert_called_once_with(state)
            predecessor.return_value = {'registered':False}
            with self.assertRaises(ValueError): h.gate('sha')


if __name__=='__main__':
    unittest.main()
