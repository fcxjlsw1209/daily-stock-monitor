"""Synthetic-only host tests: half-core wiring, conditional identification and immutable governance."""
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

spec=importlib.util.spec_from_file_location('tested_eligible187_host',Path(__file__).with_name('monthly_eligible_core187_host.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def fixture():
    n,width=8,5; shape=(n,width)
    calm=np.array([False,True,True,True,False,False,False,False])
    cw=np.where(calm,.5,.52)[:,None]
    u=np.tile([-.3,-.1,.1,.3,.5],(n,1))
    original=dict(baseline=cw*u+.1*(1-cw),core=cw*u,
        core_weight=cw,state_calm10=calm,state_weak_narrow=np.zeros(n,bool),
        eligible=np.ones(shape,bool),effects=np.zeros(shape))
    original['eligible'][:,1]=False
    source={key:original[key].copy() for key in ('core','core_weight')}
    available=original['eligible'].copy();available[[0,4]]=False
    values=dict(candidate=np.tile([-.25,np.nan,0.,.25,.5],(n,1)),
        control=np.tile([-.25,np.nan,.125,.3125,.5],(n,1)),available=available)
    for key in ('candidate','control'):values[key][~available]=np.nan
    meta=dict(dates=['synthetic-%d'%i for i in range(n)],symbols=[list('ABCDE') for _ in range(n)])
    return original,source,meta,values


class SyntheticStudy:
    def __init__(self):
        self.ep=SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter=SimpleNamespace(score_and_choose=Mock(side_effect=self.choose))
        self.rp=SimpleNamespace(make_orders=self.orders)
    @staticmethod
    def configs():return [dict(id=i) for i in range(5)]
    @staticmethod
    def modified_pre_score(spec,a,n):
        assert spec==dict(id=4)
        return a['baseline'].copy()+.11
    @staticmethod
    def boundary(pre,a,m,n):
        choices=[]
        for i in range(n):
            ids=sorted(np.flatnonzero(a['eligible'][i]),key=lambda j:(-pre[i,j],j))
            choices.append([m['symbols'][i][j] for j in ids[:3]] if len(ids)>=2 else [])
        return choices,pre[:n].copy()
    def choices(self,spec,a,m,n,*,policy):return self.boundary(self.modified_pre_score(spec,a,n),a,m,n)
    @staticmethod
    def orders(choices,scores,meta):
        return [dict(date=meta['dates'][i],symbols=ss,scores=scores[i].tolist()) for i,ss in enumerate(choices)]
    def choose(self,base,weight,extra,a,m,n,*,pre_score,effects,episode_policy,boundary_selector):
        assert base=='incumbent66' and weight==0. and effects is a['effects']
        assert np.isnan(extra).all() and extra.shape==a['baseline'].shape
        return boundary_selector(pre_score,a,m,n)


def original_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class CoreMappingAndWiringTests(unittest.TestCase):
    def test_exact_half_core_operation_order_and_ineligible_neutral(self):
        import monthly_eligible_core187_kernel as kernel
        a,s,_,v=fixture();dd,audit=kernel.build_deltas(v,a,s)
        for key in ('candidate','control'):
            expected=np.zeros(a['core'].shape);mask=v['available']
            cw=np.broadcast_to(a['core_weight'],expected.shape)
            expected[mask]=.5*(cw[mask]*v[key][mask]-a['core'][mask])
            np.testing.assert_array_equal(dd[key],expected)
            self.assertTrue((dd[key][~a['eligible']]==0).all())
            self.assertTrue((dd[key][0]==0).all())
        self.assertTrue(audit['original_support_unchanged'])
        self.assertFalse(audit['episode_applied'])

    def test_changing_target_never_recomputes_source_support_or_models(self):
        a,s,m,v=fixture();study=SyntheticStudy();before=copy.deepcopy(a)
        with patch.object(h,'read',return_value={}):h.decisions(study,a,m,v,s,8)
        for call in study.adapter.score_and_choose.call_args_list:
            self.assertIs(call.args[3],a)
            np.testing.assert_array_equal((call.args[3]['baseline']-call.args[3]['core'])/(1-a['core_weight']),
                                          (before['baseline']-before['core'])/(1-before['core_weight']))
        for key in a:np.testing.assert_array_equal(a[key],before[key])

    def test_single_original_episode_and_original_arrays_unchanged(self):
        a,s,m,v=fixture();before=copy.deepcopy(a);study=SyntheticStudy()
        with patch.object(h,'read',return_value={'synthetic':True}):
            orders,audit,deltas=h.decisions(study,a,m,v,s,8)
        self.assertEqual(study.adapter.score_and_choose.call_count,2)
        self.assertEqual(study.ep.select_boundary.call_count,2)
        for arm,call in zip(h.NAMES[1:],study.adapter.score_and_choose.call_args_list):
            expected=study.modified_pre_score(dict(id=4),a,8);mask=deltas[arm]!=0.;expected[mask]+=deltas[arm][mask]
            np.testing.assert_array_equal(call.kwargs['pre_score'],expected)
            self.assertIs(call.args[3],a)
            self.assertEqual(orders[arm][0],orders[h.NAMES[0]][0])
            self.assertTrue(np.all(deltas[arm][4]==0))
        for key in a:np.testing.assert_array_equal(a[key],before[key])

    def test_every_prefix_zero_and_one_included(self):
        a,s,m,v=fixture()
        with patch.object(h,'read',return_value={'synthetic':True}):
            orders,audit,deltas=h.decisions(SyntheticStudy(),a,m,v,s,8)
            for n in range(9):
                oo,aa,dd=h.decisions(SyntheticStudy(),a,m,v,s,n)
                for arm in h.NAMES:self.assertEqual(oo[arm],orders[arm][:n])
                for arm in h.RAW:np.testing.assert_array_equal(dd[arm],deltas[arm][:n])
                for key in ('core_weights',):self.assertEqual(aa[key],audit[key][:n])

    def test_weight_broadcasts_and_future_poison_are_prefix_safe(self):
        import monthly_eligible_core187_kernel as kernel
        a,s,_,v=fixture();expected,_=kernel.build_deltas(v,a,s,limit=3)
        for shape in ('vector','full'):
            aa=copy.deepcopy(a);ss=copy.deepcopy(s)
            for obj,key in ((aa,'core_weight'),(ss,'core_weight')):
                obj[key]=obj[key][:,0] if shape=='vector' else np.broadcast_to(obj[key],a['core'].shape).copy()
            got,_=kernel.build_deltas(v,aa,ss,limit=3)
            for key in got:np.testing.assert_array_equal(got[key],expected[key])
        for key in ('candidate','control'):v[key]=v[key].astype(object);v[key][3:]='future poison'
        got,_=kernel.build_deltas(v,a,s,limit=3)
        for key in got:np.testing.assert_array_equal(got[key],expected[key])

    def test_real_boundary_uses_original_support(self):
        a,s,m,v=fixture();study=SyntheticStudy()
        ns=original_functions(h.ROOT/'outputs/monthly_daily_episodes/episodes.py',['select_boundary'],{'np':np})
        study.adapter=h.module('eligible187_real_adapter',h.ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
        study.ep=SimpleNamespace(select_boundary=Mock(side_effect=ns['select_boundary']))
        policy=dict(budget=.05,ceiling=5,gap=.02)
        def choices(spec,a,m,n,*,policy):
            return study.adapter.score_and_choose('incumbent66',0.,np.full(a['baseline'].shape,np.nan),a,m,n,
                pre_score=study.modified_pre_score(spec,a,n),effects=a['effects'],episode_policy=policy,boundary_selector=ns['select_boundary'])
        study.choices=choices
        with patch.object(h,'read',return_value=policy):orders,_,dd=h.decisions(study,a,m,v,s,8)
        self.assertEqual(study.ep.select_boundary.call_count,2)
        for call in study.ep.select_boundary.call_args_list:
            original=call.args[4]
            for key in ('baseline','core'):np.testing.assert_array_equal(original[key],a[key])
            np.testing.assert_array_equal(original['core_weight'],np.broadcast_to(a['core_weight'],a['core'].shape))
            np.testing.assert_array_equal(call.args[1],a['eligible'])


class SourceAndContractTests(unittest.TestCase):
    def test_context_and_local_evidence_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();evidence=root/'source';evidence.write_text('source')
            context=root/'context.json';context.write_text(json.dumps(dict(local_evidence_sha256={'source':h.sha(evidence)})))
            numerics=root/'numerics.json';numerics.write_text(json.dumps(dict(context_sha256=h.sha(context))))
            with patch.multiple(h,ROOT=root,CONTEXT=context,CONTEXT_SHA=h.sha(context),NUMERICS=numerics,NUMERICS_SHA=h.sha(numerics)):
                pins=h.context_binding();self.assertEqual(set(pins),{str(evidence),str(context),str(numerics)})
                evidence.write_text('changed')
                with self.assertRaises(ValueError):h.context_binding()

    def test_numerical_supplement_hash_and_context_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();context=root/'context.json'
            context.write_text(json.dumps(dict(local_evidence_sha256={})))
            numerics=root/'numerics.json'
            numerics.write_text(json.dumps(dict(context_sha256=h.sha(context))))
            with patch.multiple(h,ROOT=root,CONTEXT=context,CONTEXT_SHA=h.sha(context),NUMERICS=numerics,NUMERICS_SHA=h.sha(numerics)):
                h.context_binding()
                numerics.write_text(json.dumps(dict(context_sha256='wrong-context')))
                with self.assertRaises(ValueError):h.context_binding()
                with patch.object(h,'NUMERICS_SHA',h.sha(numerics)):
                    with self.assertRaises(ValueError):h.context_binding()

    def test_original_quality_arrays_exact_and_shapes_not_squeezed(self):
        a,s,_,_=fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'quality.npz'
            np.savez(path,**{key:a[key] for key in ('baseline','core','core_weight','state_calm10')})
            with patch.multiple(h,QUALITY=path,QUALITY_SHA=h.sha(path)):
                got,audit=h.quality_inputs(a)
                self.assertTrue(audit['original_quality_arrays_exact']);self.assertEqual(set(got),set(s))
                for key in s:np.testing.assert_array_equal(got[key],s[key])
                bad=copy.deepcopy(a);bad['core_weight']=bad['core_weight'][:,0]
                with self.assertRaises(AssertionError):h.quality_inputs(bad)
                bad=copy.deepcopy(a);bad['baseline'][1,1]+=.001
                with self.assertRaises(AssertionError):h.quality_inputs(bad)
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):h.quality_inputs(a)

    def test_fixed_contract_and_seven_artifact_set(self):
        c=h.contract();self.assertEqual(h.HALF,.5)
        self.assertEqual(c['control_nonpromotable'],h.NAMES[2]);self.assertEqual(len(h.PREPARED_FILES),7)
        self.assertEqual(c['prospective_context_sha256'],h.CONTEXT_SHA)
        for text in ('average_rank','endpoints exactly','two distinct'):self.assertIn(text,c['measure'])
        for text in ('row0','all49','undefined'):self.assertIn(text,c['availability'])
        for text in ('halfCORE','only nonzero','exactly one original episode'):self.assertIn(text,c['scoring'])
        self.assertEqual(c['selection']['changed_support_against'],[h.NAMES[0],h.NAMES[2]])
        self.assertIn(h.CONTEXT,h.code_files())
        self.assertIn(h.NUMERICS,h.code_files())
        self.assertEqual(c['prospective_numerics_sha256'],h.NUMERICS_SHA)
        self.assertIn('Inf becomes NaN before eligibility',c['numerical_policy'])



def audit_values():
    available=np.zeros((122,49),bool);available[1:,:48]=True
    return dict(available=available,candidate=np.where(available,.1,np.nan),control=np.where(available,.2,np.nan))


def feature_receipt():
    count=int(audit_values()['available'].sum())
    return dict(**{k:True for k in h.FEATURE_FLAGS},events=122,symbols_per_event=49,checked_cells=5978,
        available_cells=count,unavailable_cells=5978-count,eligible_cells=122*48,raw_core_finite_cells=5978,
        ready_rows=121,fallback_rows=1,equivalent_ready_rows=0,candidate_control_different_rows=121,
        core_rank_reconstructions=122,max_candidate_control_difference=.1,predictive_model_fits=0,computes_returns=False)


def delta_receipt():
    return dict(**{k:True for k in h.DELTA_FLAGS},replacement_fraction=.5,predictive_model_fits=0,episode_applications=0,
        changed_cells={'candidate':0,'control':0})


class AuditTests(unittest.TestCase):
    def test_feature_and_delta_evidence_fail_closed(self):
        values=audit_values();check=feature_receipt();h.require_feature_audit(check,values)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_feature_audit(bad,values)
        for key,value in (('events',True),('available_cells',5977),('predictive_model_fits',False),('ready_rows',123),('candidate_control_different_rows',123)):
            with self.assertRaises(ValueError):h.require_feature_audit(dict(check,**{key:value}),values)
        dd={k:np.zeros((122,49)) for k in h.RAW};check=delta_receipt();h.require_delta_audit(check,dd)
        for key in check:
            bad=dict(check);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.require_delta_audit(bad,dd)
        dd[h.NAMES[1]][1,0]=.001
        with self.assertRaises(ValueError):h.require_delta_audit(check,dd)

    def test_real_independent_delta_full_receipt_and_tamper(self):
        import monthly_eligible_core187_kernel as kernel
        import monthly_eligible_core187_independent as independent
        a,s,_,v=fixture();dd,audit=kernel.build_deltas(v,a,s)
        check=independent.verify_deltas(v,a,s,dd,audit)
        h.require_delta_audit(check,{arm:dd[key] for arm,key in h.RAW.items()})
        bad=copy.deepcopy(dd);bad['candidate'][1,0]+=.001
        with self.assertRaises(ValueError):independent.verify_deltas(v,a,s,bad,audit)

    def test_prepared_manifest_and_all_invariants(self):
        counts=h.estimate_counts(feature_receipt())
        common=dict(new_model_fits=0,predictive_model_fits=0,new_predictive_model_fits=0,model_fit_accounting=h.MODEL_FIT_ACCOUNTING,original_array_count=27,**counts)
        c=dict(common,artifacts_sha256={n:'fake' for n in h.PREPARED_FILES},new_returns_computed=False)
        audit=dict(common,feature=feature_receipt(),delta=delta_receipt(),prefixes=123,feature_prefixes=list(h.FEATURE_PREFIXES),
            original_orders_exact=True,original_arrays_unchanged=True,original_quality_arrays_exact=True,quality_arrays_unchanged=True)
        values=audit_values();dd={arm:np.zeros((122,49)) for arm in h.RAW}
        h.verify_prepared_evidence(c,values,dd,audit)
        for key in h.PREPARED_FILES:
            bad=copy.deepcopy(c);bad['artifacts_sha256'].pop(key)
            with self.assertRaises(ValueError):h.verify_prepared_evidence(bad,values,dd,audit)
        for key in ('prefixes','original_orders_exact','original_arrays_unchanged','original_quality_arrays_exact','quality_arrays_unchanged','new_predictive_model_fits','ready_rows'):
            bad=copy.deepcopy(audit);bad.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):h.verify_prepared_evidence(c,values,dd,bad)

    def test_frozen_training_matches_independent_slices(self):
        training, _ = selection_fixture()
        audit = dict(paths={a: {p: dict(slices={full: copy.deepcopy(s[short])
                    for short, full in [('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')]})
                    for p, s in paths.items()} for a, paths in training.items()})
        h.verify_training_against_audit(training, audit)
        for arm in h.NAMES:
            for path in training[arm]:
                for segment in training[arm][path]:
                    bad = copy.deepcopy(training); bad[arm][path][segment]['cagr'] += .01
                    with self.assertRaises(ValueError): h.verify_training_against_audit(bad, audit)

    def test_nine_paths_five_slices_exact_receipt_and_source(self):
        one = dict(passed=True, monthly_executions=122, slices={s: {} for s in ('full122', 'first88', 'early53', 'recent35', 'later34')})
        receipt = dict(passed=True, independent_accounts=True, independent_of_production_model_and_engine=True,
                       checked_paths=9, checked_slices=45, terminal_liquidation=False, accounting_source_sha256='fake',
                       paths={a: {p[0]: copy.deepcopy(one) for p in h.ev.PATHS} for a in h.NAMES})
        with patch.object(h, 'sha', return_value='fake'):
            h.require_outcome_audit(receipt)
            for change in ('path', 'months', 'slices', 'pass', 'sha'):
                bad = copy.deepcopy(receipt); path = bad['paths'][h.NAMES[0]]['same_close_zero']
                if change == 'path': bad['paths'][h.NAMES[0]].pop('next_open_zero')
                if change == 'months': path['monthly_executions'] = 121
                if change == 'slices': path['slices'].pop('later34')
                if change == 'pass': path['passed'] = False
                if change == 'sha': bad['accounting_source_sha256'] = 'wrong'
                with self.subTest(change=change), self.assertRaises(ValueError): h.require_outcome_audit(bad)


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

    def test_affine_span_control_cannot_promote_and_must_be_beaten(self):
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

    def test_control_identification_requires_three_changes_and_both_eras(self):
        for indices in ([], [1,53], [1,2,3], [53,54,55]):
            metrics,orders=selection_fixture()
            orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
            for i in indices:orders[h.NAMES[2]][i]['symbols']=['A','B','Z']
            result=h.select_first88(metrics,orders)
            self.assertTrue(result['guards']['changed_support'])
            self.assertFalse(result['guards']['changed_support_vs_control'])
            self.assertEqual(result['changed_months_vs_control'],indices)
            self.assertFalse(result['admitted'])
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        for i in (1,2,53):orders[h.NAMES[2]][i]['symbols']=['A','B','Z']
        self.assertTrue(h.select_first88(metrics,orders)['admitted'])

    def test_control_later34_cannot_supply_identification(self):
        metrics,orders=selection_fixture();orders[h.NAMES[2]]=copy.deepcopy(orders[h.NAMES[1]])
        for i in (90,100,110):orders[h.NAMES[2]][i]['symbols']=['A','B','Z']
        result=h.select_first88(metrics,orders)
        self.assertFalse(result['admitted']);self.assertEqual(result['changed_months_vs_control'],[])

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
        state = dict(completed=[dict(trials=6, directory='old')] * 176 + [dict(trials=x, directory='old') for x in (2, 2, 1, 1, 1, 1, 1)] + [dict(trials=4, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


def predecessor_fixture(root):
    prior = root / 'outputs/monthly_aux_reallocation186'
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
                         'outputs/old/REPAIR_COMPLETION.json', 'outputs/monthly_aux_reallocation186/STATE_BEFORE.json'):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve(); prior, expected, state = predecessor_fixture(root)
                (root / relative).write_text('changed')
                with patch.multiple(h, ROOT=root, PRIOR=prior, PRIOR_EXPECTED=expected):
                    with self.assertRaises(ValueError): h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
        plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                    prior_central_sha256='sha', predecessor_binding={'registered':True},raw_source_hashes={'raw':'sha'},quality_source_binding={'quality':True})
        state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
        def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
        with patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
             patch.object(h, 'verify'), patch.object(h, 'state_check'), patch.object(h, 'context_binding', return_value={}), \
             patch.object(h, 'quality_source_binding',return_value={'quality':True}), \
             patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
            self.assertEqual(h.gate('sha'), plan)
            predecessor.assert_called_once_with(state)
            predecessor.return_value = {'registered':False}
            with self.assertRaises(ValueError): h.gate('sha')



if __name__=='__main__':
    if sys.argv[1:]==['--combined-synthetic']:
        names=('host','kernel','independent')
        modules=[h.module('dry_eligible187_'+name,Path(__file__).with_name('test_monthly_eligible_core187_'+name+'.py')) for name in names]
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()

