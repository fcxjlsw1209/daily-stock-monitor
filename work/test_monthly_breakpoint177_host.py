"""Synthetic-only177 core replacement, source bindings and governance tests."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('tested_breakpoint177_host', Path(__file__).with_name('monthly_breakpoint177_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    baseline = np.tile([1., 2., 3., 4., 5.], (6, 1))
    values = dict(baseline=baseline, candidate=baseline.copy(),
                  control=baseline.copy(), available=np.ones(shape, bool))
    values['candidate'][1] = [5., 4., 3., 2., 1.]
    values['available'][2, 0] = False
    for key in ('candidate', 'control'): values[key][2, 0] = np.nan
    values['baseline'][4, 0] = np.nan
    values['available'][4, 0] = False
    for key in ('candidate', 'control'): values[key][4, 0] = np.nan
    values['candidate'][4, 1:] = [5., 4., 3., 2.]
    calm = np.array([False, False, False, True, False, False])
    cw = np.where(calm, .5, .52)[:, None]
    ranks = pd.DataFrame(baseline).rank(axis=1, pct=True, method='average').fillna(.5).to_numpy()-.5
    core = cw*ranks
    a = dict(core=core, core_weight=cw, baseline=core+(1-cw)*.1,
             eligible=np.ones(shape, bool), effects=np.zeros(shape),
             state_calm10=calm, state_weak_narrow=np.zeros(shape[0], bool))
    a['eligible'][:, -1] = False
    m = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDE') for _ in range(6)])
    return a, m, values


def policy():
    return dict(budget=.05, ceiling=5, gap=.02)


class SyntheticStudy:
    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.rp = SimpleNamespace(make_orders=self.orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def modified_pre_score(specification, a, n):
        assert specification == dict(id=4)
        return a['baseline'].copy() + .011

    @staticmethod
    def boundary(score, eligible, symbols, effects, original, frozen_policy):
        choices = []
        for row, good, names in zip(score, eligible, symbols):
            ids = sorted(np.flatnonzero(good), key=lambda j: (-row[j], names[j]))
            choices.append(ids[:3] if len(ids) >= 2 else [])
        return choices, score.copy()

    def choices(self, specification, a, m, n, *, policy):
        return self.boundary(self.modified_pre_score(specification, a, n)[:n], a['eligible'][:n],
            m['symbols'][:n], a['effects'][:n], a, policy)

    @staticmethod
    def orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in selected],
                     scores=scores[i].tolist()) for i, selected in enumerate(choices)]


class CoreReplacementTests(unittest.TestCase):
    def test_half_core_delta_and_complete_universe_rank(self):
        a, _, values = fixture()
        ranks, deltas = h.core_ranks_and_deltas(values, a, 6)
        rank = pd.Series(values['candidate'][1]).rank(pct=True).to_numpy()-.5
        np.testing.assert_array_equal(ranks[h.NAMES[1]][1], rank)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][1], .5*(.52*rank-a['core'][1]))
        self.assertFalse(a['eligible'][1, -1]); self.assertNotEqual(deltas[h.NAMES[1]][1, -1], 0.)
        self.assertTrue(all(np.max(np.abs(delta)) <= .26 for delta in deltas.values()))

    def test_initial_calm_and_incomplete_baseline_member_neutral_entire_row(self):
        a, _, values = fixture()
        values['candidate'][3] = [5., 4., 3., 2., 1.]
        ranks, deltas = h.core_ranks_and_deltas(values, a, 6)
        for arm in h.NAMES[1:]:
            for row in (0, 2, 3):
                self.assertTrue(np.isnan(ranks[arm][row]).all())
                np.testing.assert_array_equal(deltas[arm][row], np.zeros(5))
        self.assertTrue(np.any(deltas[h.NAMES[1]][4] != 0.))
        self.assertTrue(np.isnan(ranks[h.NAMES[1]][4, 0]))
        self.assertEqual(deltas[h.NAMES[1]][4, 0], 0.)

    def test_unbroken_inputs_reproduce_core_exactly(self):
        a, _, values = fixture()
        for key in ('candidate', 'control'):
            values[key] = values['baseline'].copy()
            values[key][~values['available']] = np.nan
        _, deltas = h.core_ranks_and_deltas(values, a, 6)
        for value in deltas.values(): np.testing.assert_array_equal(value, np.zeros((6, 5)))

    def test_invalid_domain_mask_weights_and_core_fail_closed(self):
        for change in ('domain', 'mask', 'inf', 'shape', 'bool', 'core', 'weights', 'state'):
            a, _, values = fixture()
            if change == 'domain':
                values['available'][4, 0] = True
                values['candidate'][4, 0] = values['control'][4, 0] = 1.
            if change == 'mask': values['candidate'][1, 0] = np.nan
            if change == 'inf': values['baseline'][1, 0] = np.inf
            if change == 'shape': values['control'] = values['control'][:, :-1]
            if change == 'bool': values['available'] = values['available'].astype(int)
            if change == 'core': a['core'][1, 0] += 1e-16
            if change == 'weights': a['core_weight'][1] = .5
            if change == 'state': a['state_calm10'] = a['state_calm10'].astype(int)
            with self.subTest(change=change), self.assertRaises(ValueError): h.core_ranks_and_deltas(values, a, 6)

    def test_exact_tied_ranks_and_single_baseline_member(self):
        a, _, values = fixture()
        values['baseline'][1] = [2., 2., 4., np.nan, np.nan]
        values['available'][1] = [True, True, True, False, False]
        for key in ('candidate', 'control'): values[key][1] = values['baseline'][1]
        a['core'][1] = .52*(pd.Series(values['baseline'][1]).rank(pct=True).fillna(.5).to_numpy()-.5)
        ranks, deltas = h.core_ranks_and_deltas(values, a, 2)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][1], np.zeros(5))
        self.assertEqual(ranks[h.NAMES[1]][1, 0], 0.)
        for key in ('baseline', 'candidate', 'control'):
            values[key][1] = [2., np.nan, np.nan, np.nan, np.nan]
        values['available'][1] = [True, False, False, False, False]
        a['core'][1] = [.26, 0., 0., 0., 0.]
        ranks, deltas = h.core_ranks_and_deltas(values, a, 2)
        self.assertEqual(ranks[h.NAMES[1]][1, 0], .5)
        np.testing.assert_array_equal(deltas[h.NAMES[1]][1], np.zeros(5))

    def test_prefix_does_not_convert_future_object_values(self):
        a, _, values = fixture(); fullr, fulld = h.core_ranks_and_deltas(values, a, 6)
        for n in range(7):
            poisoned = copy.deepcopy(values); pa = copy.deepcopy(a)
            for key in ('baseline', 'candidate', 'control'):
                poisoned[key] = poisoned[key].astype(object); poisoned[key][n:] = 'future-invalid'
            for key in ('core', 'core_weight'):
                pa[key] = pa[key].astype(object); pa[key][n:] = 'future-invalid'
            ranks, deltas = h.core_ranks_and_deltas(poisoned, pa, n)
            for arm in h.NAMES[1:]:
                np.testing.assert_array_equal(ranks[arm], fullr[arm][:n])
                np.testing.assert_array_equal(deltas[arm], fulld[arm][:n])

    def test_prefix_validation_and_core_weight_shapes(self):
        a, _, values = fixture()
        for n in (True, False, 1.5, np.int64(1), -1, 7):
            with self.subTest(n=n), self.assertRaises(ValueError): h.core_ranks_and_deltas(values, a, n)
        expected = h.core_ranks_and_deltas(values, a, 6)
        for weights in (a['core_weight'][:, 0], np.broadcast_to(a['core_weight'], (6, 5))):
            result = h.core_ranks_and_deltas(values, dict(a, core_weight=weights), 6)
            for index in (0, 1):
                for arm in h.NAMES[1:]: np.testing.assert_array_equal(result[index][arm], expected[index][arm])

    def test_real_adapter_updates_metadata_but_single_episode_gets_original_support(self):
        a, m, values = fixture(); olda = copy.deepcopy(a); study = SyntheticStudy()
        adapter = h.core_adapter()
        original = study.modified_pre_score(dict(id=4), a, 6)
        with patch.object(h, 'read', return_value=policy()), patch.object(h, 'core_adapter', return_value=adapter), \
                patch.object(adapter.path_adapter, 'score_and_choose', wraps=adapter.path_adapter.score_and_choose) as entry:
            orders, ranks, deltas = h.decisions(study, a, m, values, 6)
        self.assertEqual(entry.call_count, 2); self.assertEqual(study.ep.select_boundary.call_count, 2)
        for arm, call, episode in zip(h.NAMES[1:], entry.call_args_list, study.ep.select_boundary.call_args_list):
            updated = call.args[3]
            expected_base = a['baseline'].copy(); expected_core = a['core'].copy(); expected_pre = original.copy()
            changed = deltas[arm] != 0.
            expected_base[changed] += deltas[arm][changed]
            expected_core[changed] += deltas[arm][changed]
            expected_pre[changed] += deltas[arm][changed]
            np.testing.assert_array_equal(updated['baseline'], expected_base)
            np.testing.assert_array_equal(updated['core'], expected_core)
            np.testing.assert_array_equal(call.kwargs['pre_score'], expected_pre)
            old = episode.args[4]
            np.testing.assert_array_equal(old['baseline'], a['baseline'])
            np.testing.assert_array_equal(old['core'], a['core'])
            np.testing.assert_array_equal((old['baseline']-old['core'])/(1-old['core_weight']),
                                          (a['baseline']-a['core'])/(1-a['core_weight']))
            self.assertEqual(orders[arm][0], orders[h.NAMES[0]][0])
        for key in a: np.testing.assert_array_equal(a[key], olda[key])

    def test_independent_delta_rebuild_agrees_on_readiness_and_original_domain(self):
        import monthly_breakpoint177_independent as independent
        a, _, values = fixture()
        a['core_weight'] = a['core_weight'][:, 0]
        _, deltas = h.core_ranks_and_deltas(values, a, 6)
        check = independent.verify_deltas(values, deltas, a, dose=h.DOSE)
        h.require_delta_audit(check, deltas)
        broken = copy.deepcopy(deltas)
        broken[h.NAMES[1]][2, 1] = .01
        with self.assertRaises(ValueError): independent.verify_deltas(values, broken, a, dose=h.DOSE)

    def test_every_decision_prefix_and_no_input_mutation(self):
        a, m, values = fixture(); before = copy.deepcopy(values)
        with patch.object(h, 'read', return_value=policy()):
            allorders, _, _ = h.decisions(SyntheticStudy(), a, m, values, 6)
            for n in range(7):
                orders, _, _ = h.decisions(SyntheticStudy(), a, m, values, n)
                for name in h.NAMES: self.assertEqual(orders[name], allorders[name][:n])
        for key in values: np.testing.assert_array_equal(values[key], before[key])


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

    def test_midpoint_control_cannot_promote_and_must_be_beaten(self):
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
        state = dict(completed=[dict(trials=6, directory='old')] * 172 + [dict(trials=2, directory='old'), dict(trials=3, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


class SourceBindingTests(unittest.TestCase):
    def test_frozen_adapter_source_pins_are_required(self):
        with patch.object(h, 'sha', return_value='wrong'), patch.object(h, 'module') as loader:
            with self.assertRaises(ValueError): h.core_adapter()
        loader.assert_not_called()

    def test_contract_fixes_noncalm_half_core_and_bounded_novelty(self):
        contract = h.contract()
        self.assertEqual(contract['source_window']['formation_prices'], 253)
        self.assertEqual(contract['source_window']['volatility_prices'], 64)
        self.assertEqual(contract['source_window']['midpoint'], 126)
        self.assertEqual(contract['source_window']['formation_log_std_floor'], 1e-12)
        self.assertEqual(contract['source_window']['split_sse_relative_tolerance'], 1e-12)
        self.assertIn('minimum_SSE+1e-12*SSE0', contract['formation'])
        self.assertIn('numerically constant', contract['criterion'])
        self.assertIn('statistic=None', contract['criterion'])
        self.assertIn('canonical inactive placeholder', contract['criterion'])
        self.assertEqual(contract['control_nonpromotable'], h.NAMES[2])
        self.assertIn('0.5*', contract['scoring'])
        self.assertIn('Calm', contract['scoring'])
        self.assertIn('bounded distinction', contract['novelty'])
        self.assertIn('original baseline/core/weights', contract['metadata'])


class AuditReceiptTests(unittest.TestCase):
    def test_prepared_manifest_and_invariants_are_required(self):
        c = dict(artifacts_sha256={n: 'fake' for n in h.PREPARED_FILES}, new_returns_computed=False,
                 new_model_fits=0, predictive_model_fits=0, model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
                 original_array_count=27, new_predictive_model_fits=0, constant_mean_estimates=12, split_mean_estimates=1524, candidate_breaks=4, control_breaks=2)
        audit = dict(feature=dict(available_cells=12, constant_mean_estimates=12, split_mean_estimates=1524, candidate_breaks=4, control_breaks=2), delta={}, prefixes=123,
                     feature_prefixes=list(h.FEATURE_PREFIXES), original_orders_exact=True,
                     original_arrays_unchanged=True, original_array_count=27, original_core_metadata_and_support_preserved=True,
                     single_original_episode=True, new_model_fits=0, predictive_model_fits=0,
                     model_fit_accounting=h.MODEL_FIT_ACCOUNTING, new_predictive_model_fits=0, constant_mean_estimates=12, split_mean_estimates=1524, candidate_breaks=4, control_breaks=2)
        with patch.object(h, 'require_feature_audit'), patch.object(h, 'require_delta_audit'):
            h.verify_prepared_evidence(c, {}, {}, audit)
            for name in h.PREPARED_FILES:
                bad = copy.deepcopy(c); bad['artifacts_sha256'].pop(name)
                with self.assertRaises(ValueError): h.verify_prepared_evidence(bad, {}, {}, audit)
            for key, value in [('prefixes', 122), ('feature_prefixes', [0, 1]), ('original_orders_exact', False),
                               ('original_arrays_unchanged', False), ('original_core_metadata_and_support_preserved', False), ('single_original_episode', False), ('original_array_count', 26), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('new_predictive_model_fits', 1), ('constant_mean_estimates', 11), ('split_mean_estimates', 1523), ('candidate_breaks', 5), ('control_breaks', 3)]:
                with self.assertRaises(ValueError):
                    h.verify_prepared_evidence(c, {}, {}, dict(audit, **{key: value}))
            for key, value in [('new_returns_computed', True), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('original_array_count', 26), ('new_predictive_model_fits', 1),
                               ('constant_mean_estimates', 11), ('split_mean_estimates', 1523), ('candidate_breaks', 5), ('control_breaks', 3)]:
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
                     breakpoint_search_checked=True, fixed_midpoint_control_checked=True,
                     formation_and_volatility_dates_checked=True, endpoint_fallback_checked=True,
                     numerical_constant_fallback_checked=True, raw_rank_parity_checked=True, predictive_model_fits=0,
                     constant_mean_estimates=5978, split_mean_estimates=759206, candidate_breaks=2000, control_breaks=1000)
        h.require_feature_audit(check, values)
        for key, wrong in [('passed', 1), ('events', 121), ('symbols_per_event', 48),
                           ('checked_cells', 5977), ('available_cells', 1), ('unavailable_cells', 1),
                           ('computes_returns', True), ('independent_scalar_rebuild', False),
                           ('exact_common_availability', False), ('no_future_or_outcome_inputs', False),
                           ('audit_checked', False), ('breakpoint_search_checked', False),
                           ('fixed_midpoint_control_checked', False), ('formation_and_volatility_dates_checked', False),
                           ('endpoint_fallback_checked', False), ('numerical_constant_fallback_checked', False),
                           ('raw_rank_parity_checked', False), ('constant_mean_estimates', 5979),
                           ('split_mean_estimates', 1), ('candidate_breaks', 5979), ('control_breaks', -1),
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
                     noncalm_core_parity_checked=True, baseline_rank_domain_checked=True, whole_row_readiness_checked=True, calm_rows_neutral=True, dose=.5, changed_cells={n: 0 for n in h.NAMES[1:]})
        h.require_delta_audit(check, deltas)
        for key in ('passed', 'full_candidate_universe', 'initial_delta_zero',
                    'common_available_only', 'no_eligibility_or_model_filter', 'noncalm_core_parity_checked', 'baseline_rank_domain_checked', 'whole_row_readiness_checked', 'calm_rows_neutral'):
            with self.assertRaises(ValueError): h.require_delta_audit(dict(check, **{key: False}), deltas)
        with self.assertRaises(ValueError): h.require_delta_audit(dict(check, dose=.01), deltas)
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


if __name__ == '__main__':
    unittest.main()
