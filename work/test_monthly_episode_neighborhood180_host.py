"""Synthetic-only governance and single-operator wiring tests for round180."""
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

spec = importlib.util.spec_from_file_location('tested_episode180_host', Path(__file__).with_name('monthly_episode_neighborhood180_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    arrays = dict(baseline=np.tile([.20, .19, .18, .17, .16], (6, 1)),
        core=np.tile([.15, .14, .13, .12, .11], (6, 1)), core_weight=np.full((6, 1), .52),
        eligible=np.ones(shape, bool), effects=np.tile([.01, -.02, .03, .04, .05], (6, 1)),
        state_calm10=np.zeros(6, bool), state_weak_narrow=np.zeros(6, bool))
    arrays['eligible'][2, 2:] = False
    arrays['eligible'][3, 1:] = False
    arrays['eligible'][4] = False
    meta = dict(dates=['2020-0%d-28' % (i+1) for i in range(6)], symbols=[list('ABCDE') for _ in range(6)])
    pre = arrays['baseline'] + .11
    result = dict(eligible=arrays['eligible'].copy(), scores={}, corrections={}, choices={})
    for key in ('candidate', 'control'):
        result['scores'][key] = pre.copy()
        result['corrections'][key] = np.zeros(shape)
        result['choices'][key] = [[0, 1, 2], [0, 1, 2], [0, 1], [], [], [0, 1, 2]]
    return arrays, meta, pre, result


class SyntheticStudy:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=AssertionError('new arms must not call old episode')))
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=AssertionError('new arms must not call old adapter')))
        self.rp = SimpleNamespace(make_orders=self.orders)
        self.choices = Mock(side_effect=self.original)

    @staticmethod
    def configs(): return [dict(id=i) for i in range(5)]

    @staticmethod
    def modified_pre_score(specification, arrays, n):
        assert specification == dict(id=4)
        return arrays['baseline'].copy() + .11

    @staticmethod
    def orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in ids],
                     scores=[float(scores[i, j]) for j in ids]) for i, ids in enumerate(choices)]

    def original(self, specification, arrays, meta, n, *, policy):
        assert policy == {'synthetic': True}
        score = self.modified_pre_score(specification, arrays, n)[:n]
        # Deliberately different old first-row choice proves no new-arm identity assumption.
        choices = [[0, 1, 3], [0, 1, 2], [0, 1], [], [], [0, 1, 2]][:n]
        return choices, score


class SingleOperatorTests(unittest.TestCase):
    def test_new_arms_use_returned_scores_and_indices_without_old_episode(self):
        arrays, meta, pre, result = fixture(); study = SyntheticStudy()
        result['scores']['candidate'][1, 3] = .32
        result['corrections']['candidate'][1, 3] = .04
        result['choices']['candidate'][1] = [3, 0, 1]
        before = copy.deepcopy((arrays, result))
        with patch.object(h, 'read', return_value={'synthetic': True}):
            orders = h.decisions(study, arrays, meta, result, 6)
        study.choices.assert_called_once()
        study.adapter.score_and_choose.assert_not_called(); study.ep.select_boundary.assert_not_called()
        self.assertEqual(orders[h.NAMES[1]][1]['symbols'], ['D', 'A', 'B'])
        self.assertEqual(orders[h.NAMES[1]][1]['scores'][0], .32)
        self.assertNotEqual(orders[h.NAMES[1]][0], orders[h.NAMES[0]][0])
        for key in arrays: np.testing.assert_array_equal(arrays[key], before[0][key])
        for key, value in h.result_arrays(result).items(): np.testing.assert_array_equal(value, h.result_arrays(before[1])[key])

    def test_all_prefix_orders_are_causal_and_empty_prefix_works(self):
        arrays, meta, pre, result = fixture()
        with patch.object(h, 'read', return_value={'synthetic': True}):
            full = h.decisions(SyntheticStudy(), arrays, meta, result, 6)
            for n in range(7):
                orders = h.decisions(SyntheticStudy(), arrays, meta, result, n)
                self.assertEqual(orders, {key: value[:n] for key, value in full.items()})

    def test_prefix_does_not_convert_future_poison(self):
        arrays, meta, pre, result = fixture()
        for field in ('scores', 'corrections'):
            for key in result[field]:
                result[field][key] = result[field][key].astype(object)
                result[field][key][2:] = 'future'
        small = h.prefix_result(result, 2)
        self.assertEqual(small['eligible'].shape, (2, 5))
        for key in small['scores']:
            np.testing.assert_array_equal(small['scores'][key], pre[:2])


    def test_invalid_prefixes_fail(self):
        *_, result = fixture()
        for n in (True, False, np.int64(1), 1.1, -1, 7):
            with self.subTest(n=n), self.assertRaises(ValueError): h.prefix_result(result, n)

    def test_result_validation_rejects_wrong_shapes_masks_choices_and_numerics(self):
        for failure in ('extra', 'shape', 'dtype', 'ineligible', 'duplicate', 'boolean', 'cardinality', 'inf', 'first', 'outside'):
            *_, result = fixture()
            if failure == 'extra': result['extra'] = None
            if failure == 'shape': result['scores']['candidate'] = np.zeros((2, 2))
            if failure == 'dtype': result['eligible'] = result['eligible'].astype(int)
            if failure == 'ineligible': result['choices']['candidate'][2] = [0, 4]
            if failure == 'duplicate': result['choices']['candidate'][1] = [0, 0, 1]
            if failure == 'boolean': result['choices']['candidate'][1] = [False, 1, 2]
            if failure == 'cardinality': result['choices']['candidate'][3] = [0]
            if failure == 'inf': result['scores']['candidate'][1, 0] = np.inf
            if failure == 'first': result['corrections']['candidate'][0, 0] = .01
            if failure == 'outside': result['corrections']['candidate'][2, 4] = .01
            with self.subTest(failure=failure), self.assertRaises(ValueError): h.result_arrays(result)

    def test_frozen_support_column_vector_and_matrix_weights(self):
        arrays, meta, pre, result = fixture()
        for weights in (arrays['core_weight'], arrays['core_weight'][:, 0], np.broadcast_to(arrays['core_weight'], pre.shape).copy()):
            arrays['core_weight'] = weights
            check = h.verify_frozen_support(pre, arrays, result, 6)
            self.assertTrue(check['eligible_support_bound_checked'])

    def test_frozen_support_does_not_clip_or_accept_excess_budget(self):
        for failure in ('support', 'budget', 'eligibility', 'initial', 'ineligible_score', 'weight'):
            arrays, meta, pre, result = fixture()
            if failure == 'support': arrays['core'][1, 0] = -1.
            if failure == 'budget': result['corrections']['candidate'][1, 0] = .051
            if failure == 'eligibility': arrays['eligible'][1, 4] = False
            if failure == 'initial': result['scores']['candidate'][0, 0] += .01
            if failure == 'ineligible_score': result['scores']['candidate'][2, 4] += .01
            if failure == 'weight': arrays['core_weight'][1, 0] = 1.
            with self.subTest(failure=failure), self.assertRaises((ValueError, AssertionError)): h.verify_frozen_support(pre, arrays, result, 6)

    def test_frozen_support_prefix_checks_only_selected_rows(self):
        arrays, meta, pre, result = fixture()
        small = h.prefix_result(result, 2)
        for name in ('baseline', 'core', 'core_weight'):
            arrays[name] = arrays[name].astype(object); arrays[name][2:] = 'future'
        pre = pre.astype(object); pre[2:] = 'future'
        h.verify_frozen_support(pre, arrays, small, 2)

    def test_frozen_support_tolerance_is_not_data_clipping(self):
        arrays, meta, pre, result = fixture()
        arrays['core'][1, 0] = arrays['baseline'][1, 0] - (.5 + 5e-13) * .48
        before = arrays['core'].copy()
        h.verify_frozen_support(pre, arrays, result, 6)
        np.testing.assert_array_equal(arrays['core'], before)

    def test_production_and_independent_selector_integrate_without_raw_prices(self):
        import monthly_episode_neighborhood180_kernel as kernel
        import monthly_episode_neighborhood180_independent as independent
        arrays, meta, pre, _ = fixture()
        result, audit = kernel.apply_episode(pre, arrays, meta)
        check = independent.verify_selection(pre, arrays, meta, result, audit)
        h.require_selection_audit(check, result)
        h.verify_frozen_support(pre, arrays, result, 6)
        with patch.object(h, 'read', return_value={'synthetic': True}):
            full = h.decisions(SyntheticStudy(), arrays, meta, result, 6)
            for n in range(7):
                small, small_audit = kernel.apply_episode(pre, arrays, meta, limit=n)
                h.require_selection_audit(independent.verify_selection(pre, arrays, meta, small, small_audit, limit=n), small)
                self.assertEqual(h.decisions(SyntheticStudy(), arrays, meta, small, n), {key: rows[:n] for key, rows in full.items()})


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

    def test_uniform_episode_control_cannot_promote_and_must_be_beaten(self):
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
        state = dict(completed=[dict(trials=6, directory='old')] * 173 + [dict(trials=x, directory='old') for x in (2, 1, 2)] + [dict(trials=3, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)



class AuditReceiptTests(unittest.TestCase):
    @staticmethod
    def selector_receipt(result):
        return dict(passed=True, independent_scalar_rebuild=True, original_support_checked=True,
            clipped_episode_budget_checked=True, gaussian_neighborhood_checked=True, global_top3_checked=True,
            first_partial_neutral_checked=True, exact_cardinality_checked=True, no_future_or_outcome_inputs=True,
            audit_checked=True, eligibility_unchanged=True, episode_applications=1, predictive_model_fits=0,
            computes_returns=False, events=6, symbols_per_event=5, checked_cells=30,
            eligible_cells=int(result['eligible'].sum()), correction_cells={'candidate':0, 'control':0})

    def test_selector_receipt_flags_and_exact_coverage_required(self):
        *_, result = fixture(); check = self.selector_receipt(result)
        h.require_selection_audit(check, result)
        for key in check:
            bad = dict(check); bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError): h.require_selection_audit(bad, result)
        for key, value in (('events', 5), ('checked_cells', 30.), ('eligible_cells', True), ('episode_applications', 2),
                           ('predictive_model_fits', False), ('computes_returns', True), ('correction_cells', {'candidate':1,'control':0})):
            with self.subTest(key=key), self.assertRaises(ValueError): h.require_selection_audit(dict(check, **{key:value}), result)

    def test_prepared_invariants_and_support_receipt_required(self):
        c = dict(artifacts_sha256={n:'fake' for n in h.PREPARED_FILES}, new_returns_computed=False,
                 new_model_fits=0, predictive_model_fits=0, new_predictive_model_fits=0,
                 new_raw_features=0, new_descriptive_estimates=0, original_array_count=27,
                 model_fit_accounting=h.MODEL_FIT_ACCOUNTING)
        audit = dict(c, selection={}, prefixes=123, selector_prefixes=list(h.SELECTOR_PREFIXES),
                     original_orders_exact=True, original_pre_score_exact=True, original_arrays_unchanged=True,
                     new_arms_direct_make_orders=True, original_episode_not_reapplied=True,
                     frozen_support=dict(eligible_support_bound_checked=True, correction_budget_checked=True,
                         initial_pre_score_unchanged=True, original_eligibility_exact=True,
                         support_abs_limit=.5, correction_abs_limit=.05, tolerance=1e-12))
        result = dict(eligible=np.ones((122,49),bool))
        with patch.object(h, 'require_selection_audit'):
            h.verify_prepared_evidence(c, result, audit)
            for name in h.PREPARED_FILES:
                bad=copy.deepcopy(c); bad['artifacts_sha256'].pop(name)
                with self.subTest(file=name), self.assertRaises(ValueError): h.verify_prepared_evidence(bad, result, audit)
            for key in ('original_orders_exact','original_pre_score_exact','original_arrays_unchanged',
                        'new_arms_direct_make_orders','original_episode_not_reapplied'):
                with self.subTest(key=key), self.assertRaises(ValueError): h.verify_prepared_evidence(c,result,dict(audit,**{key:False}))
            for key in audit['frozen_support']:
                bad=copy.deepcopy(audit);bad['frozen_support'].pop(key)
                with self.subTest(support=key), self.assertRaises(ValueError): h.verify_prepared_evidence(c,result,bad)
            for key in ('new_raw_features','new_descriptive_estimates','new_model_fits','predictive_model_fits','new_predictive_model_fits'):
                with self.subTest(key=key), self.assertRaises(ValueError): h.verify_prepared_evidence(dict(c,**{key:1}),result,audit)

    def test_nine_paths_five_slices_exact_receipt_and_source(self):
        one = dict(passed=True, monthly_executions=122, slices={s: {} for s in ('full122','first88','early53','recent35','later34')})
        check = dict(passed=True, independent_accounts=True, independent_of_production_model_and_engine=True,
            checked_paths=9, checked_slices=45, terminal_liquidation=False, accounting_source_sha256='fake',
            paths={a:{p[0]:copy.deepcopy(one) for p in h.ev.PATHS} for a in h.NAMES})
        with patch.object(h,'sha',return_value='fake'):
            h.require_outcome_audit(check)
            for fault in ('path','months','slices','passed','sha'):
                bad=copy.deepcopy(check); p=bad['paths'][h.NAMES[0]]['same_close_zero']
                if fault=='path': bad['paths'][h.NAMES[0]].pop('next_open_zero')
                if fault=='months': p['monthly_executions']=121
                if fault=='slices': p['slices'].pop('later34')
                if fault=='passed': p['passed']=False
                if fault=='sha': bad['accounting_source_sha256']='wrong'
                with self.subTest(fault=fault), self.assertRaises(ValueError): h.require_outcome_audit(bad)


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


    def test_contract_has_one_episode_no_new_data_and_honest_novelty(self):
        contract=h.contract()
        self.assertEqual(contract['control_nonpromotable'],h.NAMES[2])
        self.assertIn('0.02',contract['measure'])
        self.assertIn('do not call the old adapter/episode',contract['scoring'])
        self.assertIn('NaN effects give zero',contract['numerical_policy'])
        self.assertIn('infinite effects raise',contract['numerical_policy'])
        self.assertIn('episode_retention70',contract['novelty'])
        self.assertEqual(contract['new_raw_features'],0)
        self.assertEqual(contract['new_descriptive_estimates'],0)
        self.assertTrue(contract['selection']['require_noninferior_to_uniform_episode'])
        self.assertIn(h.CONTEXT,h.code_files())

    def test_prepare_and_new_arm_wiring_have_no_raw_feature_or_second_episode_call(self):
        tree=ast.parse(Path(h.__file__).read_text())
        decision=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decisions')
        attributes=[n.attr for n in ast.walk(decision) if isinstance(n,ast.Attribute)]
        self.assertNotIn('score_and_choose',attributes)
        self.assertNotIn('select_boundary',attributes)
        self.assertEqual(attributes.count('choices'),1)
        prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
        attributes=[n.attr for n in ast.walk(prepare) if isinstance(n,ast.Attribute)]
        self.assertNotIn('build_features',attributes)
        self.assertIn('apply_episode',attributes)
        self.assertIn('verify_panel_parity',attributes)
        self.assertIn('modified_pre_score',attributes)


def predecessor_fixture(root):
    prior = root / 'outputs/monthly_downside_mix179'
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
                         'outputs/old/REPAIR_COMPLETION.json', 'outputs/monthly_downside_mix179/STATE_BEFORE.json'):
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


if __name__ == '__main__':
    unittest.main()
