"""Synthetic-only tests of188 direct transforms, frozen-score wiring and governance."""
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

spec = importlib.util.spec_from_file_location('tested_statevol188_host', Path(__file__).with_name('monthly_state_vol188_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    values = dict(candidate=np.tile([-.1, -.2, -.3, -.4, -.5], (6, 1)),
                  control=np.tile([-.5, -.4, -.3, -.2, -.1], (6, 1)), available=np.ones(shape, bool))
    values['available'][2, :4] = False
    values['available'][3, :] = False
    for key in ('candidate', 'control'):
        values[key][~values['available']] = np.nan
    a = dict(baseline=np.tile([.2, .199, .198, .197, .196], (6, 1)),
             eligible=np.ones(shape, bool), effects=np.zeros(shape),
             core=np.zeros(shape), core_weight=np.full((6, 1), .52),
             state_calm10=np.zeros(6, bool), state_weak_narrow=np.zeros(6, bool))
    a['eligible'][:, -1] = False  # Still receives its raw delta even though not a buy candidate.
    m = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDE') for _ in range(6)])
    return a, m, values


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


class DirectMappingAndWiringTests(unittest.TestCase):
    def test_direct_mapping_includes_ineligible_cells_without_second_tanh(self):
        a, m, values = fixture()
        values['candidate'][1] = [-.10, -.05, 0., -.025, -.05]
        transforms, deltas = h.scale_and_delta(values, 6)
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], np.array([-.10, -.05, 0., -.025, -.05]), atol=1e-15)
        self.assertFalse(a['eligible'][1, -1])
        self.assertAlmostEqual(deltas[h.NAMES[1]][1, -1], .01*(-.05))
        self.assertTrue(np.all(deltas[h.NAMES[1]][0] == 0.))
        self.assertTrue(all(np.max(np.abs(x)) <= .01 for x in deltas.values()))

    def test_missing_cells_neutral_but_singleton_not_discarded(self):
        _, _, values = fixture()
        transforms, deltas = h.scale_and_delta(values, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(transforms[name][2, :4], np.zeros(4))
            np.testing.assert_array_equal(deltas[name][3], np.zeros(5))
            self.assertLess(deltas[name][2, 4], 0.)

    def test_no_cross_sectional_scaling_or_rank(self):
        _, _, values = fixture(); values['candidate'][1] = [-.001, -.002, -.002, -.002, -.003]
        transforms, delta = h.scale_and_delta(values, 2)
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], values['candidate'][1], atol=1e-15)
        old = delta[h.NAMES[1]][1, 1]
        values['candidate'][1, 0] = -.5
        _, changed = h.scale_and_delta(values, 2)
        self.assertEqual(old, changed[h.NAMES[1]][1, 1])
        self.assertLess(old, .001)

    def test_common_mask_and_invalid_numerics_enforced(self):
        for kind in ('mask', 'infinity', 'shape', 'dtype', 'bound', 'positive_bound'):
            _, _, values = fixture()
            if kind == 'mask': values['candidate'][1, 0] = np.nan
            if kind == 'infinity': values['control'][1, 0] = np.inf
            if kind == 'shape': values['control'] = values['control'][:, :-1]
            if kind == 'bound': values['candidate'][1, 0] = -.500001
            if kind == 'positive_bound': values['candidate'][1, 0] = .50001
            if kind == 'dtype': values['available'] = values['available'].astype(int)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                h.scale_and_delta(values, 6)

    def test_prefix_does_not_consume_future_features(self):
        _, _, values = fixture(); fullr, fulld = h.scale_and_delta(values, 6)
        for n in range(7):
            poisoned = copy.deepcopy(values)
            for key in ('candidate', 'control'): poisoned[key][n:] = np.inf
            ranks, deltas = h.scale_and_delta(poisoned, n)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(ranks[arm], fullr[arm][:n])
                np.testing.assert_array_equal(deltas[arm], fulld[arm][:n])

    def test_prefix_does_not_convert_future_object_poison(self):
        _, _, values = fixture(); expected, _ = h.scale_and_delta(values, 2)
        for key in ('candidate', 'control'):
            values[key] = values[key].astype(object)
            values[key][2:] = 'future-not-numeric'
        actual, _ = h.scale_and_delta(values, 2)
        for arm in actual: np.testing.assert_array_equal(actual[arm], expected[arm])

    def test_additive_small_slot_and_exactly_one_episode(self):
        a, m, values = fixture(); olda, oldv = copy.deepcopy(a), copy.deepcopy(values)
        study = SyntheticStudy(); policy = dict(synthetic=True)
        original = study.modified_pre_score(dict(id=4), a, 6)
        with patch.object(h, 'read', return_value=policy) as reader:
            orders, ranks, deltas = h.decisions(study, a, m, values, 6)
        reader.assert_called_once_with(h.LOCAL / 'INCUMBENT_POLICY.json')
        self.assertEqual(study.adapter.score_and_choose.call_count, 2)
        self.assertEqual(study.ep.select_boundary.call_count, 2)
        for arm, call in zip(h.NAMES[1:], study.adapter.score_and_choose.call_args_list):
            expected = original.copy(); use = deltas[arm] != 0.
            expected[use] += deltas[arm][use]
            np.testing.assert_array_equal(call.kwargs['pre_score'], expected)
            self.assertIs(call.kwargs['episode_policy'], policy)
            self.assertEqual(orders[arm][0], orders[h.NAMES[0]][0])
        for key in a: np.testing.assert_array_equal(a[key], olda[key])
        for key in values: np.testing.assert_array_equal(values[key], oldv[key])

    def test_every_decision_prefix_and_no_models_called(self):
        a, m, values = fixture()
        with patch.object(h, 'read', return_value={'synthetic': True}):
            allorders, _, _ = h.decisions(SyntheticStudy(), a, m, values, 6)
            for n in range(7):
                orders, _, _ = h.decisions(SyntheticStudy(), a, m, values, n)
                for name in h.NAMES: self.assertEqual(orders[name], allorders[name][:n])

    def test_prefix_validation_rejects_bool_fraction_and_out_of_range(self):
        _, _, values = fixture()
        for n in (True, False, 1.5, np.int64(1), -1, 7):
            with self.subTest(n=n), self.assertRaises(ValueError):
                h.scale_and_delta(values, n)
        transforms, deltas = h.scale_and_delta(values, 0)
        for result in (transforms, deltas):
            self.assertTrue(all(array.shape == (0, 5) for array in result.values()))


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

    def test_current_volatility_control_cannot_promote_and_must_be_beaten(self):
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

    def test_positive_zero_and_negative_cdf_scores_are_permitted(self):
        _, _, values=fixture();values['candidate'][1]=[-.5,-.25,0.,.25,.5]
        _, delta=h.scale_and_delta(values,2)
        np.testing.assert_array_equal(delta[h.NAMES[1]][1],np.array([-.005,-.0025,0.,.0025,.005]))

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
        state = dict(completed=[dict(trials=1, directory='old')] * 184 + [dict(trials=888, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


class SourceBindingTests(unittest.TestCase):
    def test_original_source_adjusted_panel_only(self):
        adj = pd.DataFrame({'A': [1., 2.], 'SPY': [2., 3.]})
        source = dict(adj=adj, close=pd.DataFrame({'A': [100., 200.]}))
        result = h.feature_inputs(source)
        self.assertEqual(set(result), {'adj'})
        self.assertIs(result['adj'], adj)
        for invalid in ({}, {'adj': np.ones((2, 2))}):
            with self.assertRaises(ValueError): h.feature_inputs(invalid)

    def test_contract_fixed_conditioning_and_small_score(self):
        contract = h.contract()
        self.assertEqual(contract['own_volatility_sessions'],126)
        self.assertEqual(contract['market_volatility_sessions'],63)
        self.assertEqual(contract['history_sessions'],756)
        self.assertEqual(contract['joint_minimum'],252)
        self.assertEqual(contract['same_state_minimum'],63)
        self.assertTrue(contract['current_excluded'])
        self.assertEqual(contract['control_nonpromotable'], h.NAMES[2])
        self.assertEqual(contract['max_absolute_delta'], .005)
        self.assertIn('Delta=0.01*raw',contract['context']['score'])
        self.assertIn('no claim',contract['context']['novelty'])
        self.assertIn('not pristineOOS',contract['limitation'])


    def test_real_adapter_and_original_boundary_with_column_weights(self):
        # Compile the exact original boundary function without importing any driver or prices.
        path = h.ROOT / 'outputs/monthly_daily_episodes/episodes.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'select_boundary')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        adapter = h.module('statevol188_test_original_adapter', h.ROOT / 'outputs/monthly_path70_research/incumbent_adapter.py')
        a, m, values = fixture()
        a['state_calm10'][2] = True
        a['core_weight'][2, 0] = .50
        a['effects'][1, 3] = .2
        before = {key: value.tobytes() for key, value in a.items()}
        study = SyntheticStudy()
        study.adapter = adapter
        study.ep = SimpleNamespace(select_boundary=Mock(side_effect=namespace['select_boundary']))
        policy = dict(budget=.05, ceiling=5, gap=.02)
        def choices(specification, arrays, meta, n, *, policy):
            return adapter.score_and_choose('incumbent66', 0., np.full(arrays['baseline'].shape, np.nan),
                arrays, meta, n, pre_score=study.modified_pre_score(specification, arrays, n),
                effects=arrays['effects'], episode_policy=policy, boundary_selector=namespace['select_boundary'])
        study.choices = choices
        with patch.object(h, 'read', return_value=policy):
            orders, _, deltas = h.decisions(study, a, m, values, 6)
        self.assertEqual(study.ep.select_boundary.call_count, 2)
        for arm, call in zip(h.NAMES[1:], study.ep.select_boundary.call_args_list):
            score, eligible, symbols, effects, original, passed_policy = call.args
            np.testing.assert_array_equal(score, study.modified_pre_score(dict(id=4), a, 6) + deltas[arm])
            np.testing.assert_array_equal(original['core'], a['core'])
            np.testing.assert_array_equal(original['baseline'], a['baseline'])
            np.testing.assert_array_equal(original['core_weight'], np.broadcast_to(a['core_weight'], (6, 5)))
            np.testing.assert_array_equal(eligible, a['eligible'])
            self.assertEqual(orders[arm][0], orders[h.NAMES[0]][0])
        self.assertEqual(before, {key: value.tobytes() for key, value in a.items()})


class AuditReceiptTests(unittest.TestCase):
    def test_prepared_manifest_and_invariants_are_required(self):
        c = dict(artifacts_sha256={n: 'fake' for n in h.PREPARED_FILES}, new_returns_computed=False,
                 new_model_fits=0, predictive_model_fits=0, model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
                 original_array_count=27, new_predictive_model_fits=0, reference_observations=3024, matched_observations=756)
        audit = dict(feature=dict(available_cells=12, reference_observations=3024, matched_observations=756), delta={}, prefixes=123,
                     feature_prefixes=list(h.FEATURE_PREFIXES), original_orders_exact=True,
                     original_arrays_unchanged=True, original_array_count=27, new_model_fits=0, predictive_model_fits=0,
                     model_fit_accounting=h.MODEL_FIT_ACCOUNTING, new_predictive_model_fits=0, reference_observations=3024, matched_observations=756)
        with patch.object(h, 'require_feature_audit'), patch.object(h, 'require_delta_audit'):
            h.verify_prepared_evidence(c, {}, {}, audit)
            for name in h.PREPARED_FILES:
                bad = copy.deepcopy(c); bad['artifacts_sha256'].pop(name)
                with self.assertRaises(ValueError): h.verify_prepared_evidence(bad, {}, {}, audit)
            for key, value in [('prefixes', 122), ('feature_prefixes', [0, 1]), ('original_orders_exact', False),
                               ('original_arrays_unchanged', False), ('original_array_count', 26), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('new_predictive_model_fits', 1), ('reference_observations', 3023)]:
                with self.assertRaises(ValueError):
                    h.verify_prepared_evidence(c, {}, {}, dict(audit, **{key: value}))
            for key, value in [('new_returns_computed', True), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('original_array_count', 26), ('new_predictive_model_fits', 1),
                               ('reference_observations', 3023)]:
                with self.assertRaises(ValueError):
                    h.verify_prepared_evidence(dict(c, **{key: value}), {}, {}, audit)

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

    def test_feature_and_delta_receipts_fail_closed(self):
        values = dict(available=np.ones((122, 49), bool))
        check = dict(passed=True, events=122, symbols_per_event=49, checked_cells=5978,
                     available_cells=5978, unavailable_cells=0, computes_returns=False,
                     independent_scalar_rebuild=True, exact_common_availability=True,
                     no_future_or_outcome_inputs=True, audit_checked=True,
                     excludes_current_reference=True, market_state_validity_checked=True, midpoint_cdf_checked=True,
                     official_session_windows_checked=True, predictive_model_fits=0, reference_observations=5978*252, matched_observations=5978*63)
        h.require_feature_audit(check, values)
        for key, wrong in [('passed', 1), ('events', 121), ('symbols_per_event', 48),
                           ('checked_cells', 5977), ('available_cells', 1), ('unavailable_cells', 1),
                           ('computes_returns', True), ('independent_scalar_rebuild', False),
                           ('exact_common_availability', False), ('no_future_or_outcome_inputs', False),
                           ('audit_checked', False), ('excludes_current_reference', False),
                           ('official_session_windows_checked', False), ('market_state_validity_checked', False), ('midpoint_cdf_checked', False),
                           ('reference_observations', 5978*252-1), ('reference_observations', 5978*756+1), ('matched_observations', 5978*63-1),
                           ('predictive_model_fits', 1), ('predictive_model_fits', False),
                           ('unavailable_cells', False), ('checked_cells', 5978.)]:
            bad = dict(check, **{key: wrong})
            with self.subTest(key=key), self.assertRaises(ValueError): h.require_feature_audit(bad, values)
        for key in check:
            bad = dict(check); bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError): h.require_feature_audit(bad, values)
        deltas = {n: np.zeros((122, 49)) for n in h.NAMES[1:]}
        check = dict(passed=True, full_candidate_universe=True, initial_delta_zero=True,
                     common_available_only=True, no_eligibility_or_model_filter=True,
                     no_cross_sectional_ranking=True, dose=.01, direct_linear_mapping=True, changed_cells={n: 0 for n in h.NAMES[1:]})
        h.require_delta_audit(check, deltas)
        for key in ('passed', 'full_candidate_universe', 'initial_delta_zero',
                    'common_available_only', 'no_eligibility_or_model_filter', 'no_cross_sectional_ranking', 'direct_linear_mapping'):
            with self.assertRaises(ValueError): h.require_delta_audit(dict(check, **{key: False}), deltas)
        with self.assertRaises(ValueError): h.require_delta_audit(dict(check, dose=.02), deltas)
        deltas[h.NAMES[1]][1, 0] = .01
        with self.assertRaises(ValueError): h.require_delta_audit(check, deltas)

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


def predecessor_fixture(root):
    prior = root/'outputs/monthly_eligible_core187'
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path.resolve()),sha256=h.sha(path))
    plan=put(prior/'PLAN.json',{'input_sha256':{}})
    result=put(prior/'results/COMPLETION.json',dict(status='COMPLETE',plan_sha256=plan['sha256'],paths=9,slices=45))
    closed=put(prior/'COMPLETION.json',dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256=plan['sha256'],
        result_completion_sha256=result['sha256'],protected_files=21444,observers_modified=False))
    context=root/'work/context.json'
    evidence=put(root/'work/evidence.json',{'synthetic':True})
    put(context,dict(evidence_sha256={'work/evidence.json':evidence['sha256']}))
    return prior,{n:h.sha(prior/n) for n in h.PRIOR_EXPECTED},dict(completed=[dict(completion=closed)]),context


class PredecessorBindingTests(unittest.TestCase):
    def test_predecessor_completion_and_result_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
            with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context):
                binding=h.predecessor_binding(state)
                self.assertIn(str(prior/'results/COMPLETION.json'),binding['input_sha256'])
                self.assertIn(str(root/'work/evidence.json'),binding['input_sha256'])
                for key,value in [('path',str(root/'wrong')),('sha256','wrong')]:
                    wrong=copy.deepcopy(state);wrong['completed'][0]['completion'][key]=value
                    with self.assertRaises(ValueError):h.predecessor_binding(wrong)

    def test_predecessor_and_context_evidence_cannot_change(self):
        for relative in ('outputs/monthly_eligible_core187/PLAN.json','outputs/monthly_eligible_core187/results/COMPLETION.json','work/evidence.json'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();prior,expected,state,context=predecessor_fixture(root)
                (root/relative).write_text('changed')
                with patch.multiple(h,ROOT=root,PRIOR=prior,PRIOR_EXPECTED=expected,CONTEXT=context):
                    with self.assertRaises(ValueError):h.predecessor_binding(state)

    def test_all_phase_gates_recheck_registered_predecessor_binding(self):
        plan = dict(contract=h.contract(), runtime=h.ev.runtime(), input_sha256={'x':'y'},
                    prior_central_sha256='sha', predecessor_binding={'registered':True})
        state = dict(registered_queue=[dict(directory=h.OUT.name, plan_sha256='sha')])
        def reader(path): return plan if path == h.OUT / 'PLAN.json' else state
        with patch.object(h, 'contract', return_value=plan['contract']), patch.object(h, 'sha', return_value='sha'), patch.object(h, 'read', side_effect=reader), \
             patch.object(h, 'verify'), patch.object(h, 'state_check'), \
             patch.object(h, 'predecessor_binding', return_value={'registered':True}) as predecessor:
            self.assertEqual(h.gate('sha'), plan)
            predecessor.assert_called_once_with(state)
            predecessor.return_value = {'registered':False}
            with self.assertRaises(ValueError): h.gate('sha')

    def test_original_column_weight_shape_does_not_enter_new_delta_api(self):
        import monthly_state_vol188_independent as independent
        original, _, values = fixture()
        self.assertEqual(original['core_weight'].shape, (6, 1))
        _, deltas = h.scale_and_delta(values, 6)
        check = independent.verify_deltas(values, deltas, dose=h.DOSE)
        h.require_delta_audit(check, deltas)
        wrong = copy.deepcopy(deltas); wrong[h.NAMES[1]][1, 0] += .001
        with self.assertRaises(ValueError): independent.verify_deltas(values, wrong, dose=h.DOSE)

    def test_reference_counts_include_inadequate_cells_without_claiming_fits(self):
        self.assertEqual(h.estimate_counts(dict(available_cells=1,reference_observations=300,matched_observations=70)),
                         dict(reference_observations=300,matched_observations=70))
        for change in [dict(reference_observations=251),dict(matched_observations=62),dict(matched_observations=301),
                       dict(reference_observations=True),dict(matched_observations=63.),dict(available_cells=-1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                h.estimate_counts(dict(dict(available_cells=1,reference_observations=300,matched_observations=70),**change))


if __name__ == '__main__':
    unittest.main()
