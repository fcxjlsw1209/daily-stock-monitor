"""Synthetic-only tests of174 ranking, frozen-score wiring and governance."""
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

spec = importlib.util.spec_from_file_location('tested_market174_host', Path(__file__).with_name('market_absorption174_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    values = dict(candidate=np.tile([1., 2., 3., 4., 5.], (6, 1)),
                  control=np.tile([5., 4., 3., 2., 1.], (6, 1)), available=np.ones(shape, bool))
    values['available'][2, :4] = False
    values['available'][3, :] = False
    for key in ('candidate', 'control'):
        values[key][~values['available']] = np.nan
    a = dict(baseline=np.tile([.2, .199, .198, .197, .196], (6, 1)),
             eligible=np.ones(shape, bool), effects=np.zeros((6, 5, 2)))
    a['eligible'][:, -1] = False  # Still ranks even though not a buy candidate.
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


class RankAndWiringTests(unittest.TestCase):
    def test_direct_mapping_includes_ineligible_cells_and_preserves_shrink(self):
        a, m, values = fixture()
        values['candidate'][1] = [-.10, -.05, 0., .025, .05]
        transforms, deltas = h.scale_and_delta(values, 6)
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], np.tanh(np.array([-2., -1., 0., .5, 1.])), atol=1e-15)
        self.assertFalse(a['eligible'][1, -1])
        self.assertAlmostEqual(deltas[h.NAMES[1]][1, -1], .01*np.tanh(1.))
        self.assertTrue(np.all(deltas[h.NAMES[1]][0] == 0.))
        self.assertTrue(all(np.max(np.abs(x)) <= .01 for x in deltas.values()))

    def test_missing_cells_neutral_but_singleton_not_discarded(self):
        _, _, values = fixture()
        transforms, deltas = h.scale_and_delta(values, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(transforms[name][2, :4], np.zeros(4))
            np.testing.assert_array_equal(deltas[name][3], np.zeros(5))
            self.assertGreater(deltas[name][2, 4], 0.)

    def test_no_cross_sectional_scaling_or_rank(self):
        _, _, values = fixture(); values['candidate'][1] = [.001, .002, .002, .002, .003]
        transforms, delta = h.scale_and_delta(values, 2)
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], np.tanh(values['candidate'][1]/.05), atol=1e-15)
        old = delta[h.NAMES[1]][1, 1]
        values['candidate'][1, 0] = 1000.
        _, changed = h.scale_and_delta(values, 2)
        self.assertEqual(old, changed[h.NAMES[1]][1, 1])
        self.assertLess(old, .001)

    def test_common_mask_and_invalid_numerics_enforced(self):
        for kind in ('mask', 'infinity', 'shape', 'dtype'):
            _, _, values = fixture()
            if kind == 'mask': values['candidate'][1, 0] = np.nan
            if kind == 'infinity': values['control'][1, 0] = np.inf
            if kind == 'shape': values['control'] = values['control'][:, :-1]
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

    def test_raw_contrast_control_cannot_promote_and_must_be_beaten(self):
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

    def test_completed_predecessor_and_counters_required(self):
        state = dict(completed=[dict(trials=6, directory='old')] * 170 + [dict(trials=8, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


class MarketSourceTests(unittest.TestCase):
    def test_spy_requires_matching_original_manifest_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'SPY.csv'; source.write_text('synthetic source only')
            manifest = Path(directory) / 'ETF_SOURCES.json'
            row = dict(symbol='SPY', file=str(source), sha256=h.sha(source))
            manifest.write_text(json.dumps([row]))
            hashes = {str(p): h.sha(p) for p in (source, manifest)}
            with patch.object(h, 'SPY_SOURCE', source), patch.object(h, 'SPY_MANIFEST', manifest):
                result = h.market_source(hashes)
                self.assertEqual(result['source_sha256'], h.sha(source))
                for missing in hashes:
                    with self.assertRaises(ValueError):
                        h.market_source({k: v for k, v in hashes.items() if k != missing})
                source.write_text('changed source')
                with self.assertRaises(ValueError): h.market_source(hashes)

    def test_duplicate_or_redirected_spy_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'SPY.csv'; source.write_text('synthetic source only')
            manifest = Path(directory) / 'ETF_SOURCES.json'
            row = dict(symbol='SPY', file=str(source), sha256=h.sha(source))
            with patch.object(h, 'SPY_SOURCE', source), patch.object(h, 'SPY_MANIFEST', manifest):
                for rows in ([], [row, row], [dict(row, file=str(source.with_name('OTHER.csv')))]):
                    manifest.write_text(json.dumps(rows))
                    hashes = {str(p): h.sha(p) for p in (source, manifest)}
                    with self.assertRaises(ValueError): h.market_source(hashes)


class AuditReceiptTests(unittest.TestCase):
    def test_prepared_manifest_and_invariants_are_required(self):
        c = dict(artifacts_sha256={n: 'fake' for n in h.PREPARED_FILES}, new_returns_computed=False,
                 new_model_fits=0, predictive_model_fits=0, model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
                 fixed_ols_beta_attempts=12, fixed_ols_beta_estimates=11)
        audit = dict(feature=dict(beta_attempts_count=12, beta_estimates_count=11), delta={}, prefixes=123,
                     feature_prefixes=list(h.FEATURE_PREFIXES), original_orders_exact=True,
                     original_arrays_unchanged=True, new_model_fits=0, predictive_model_fits=0,
                     model_fit_accounting=h.MODEL_FIT_ACCOUNTING)
        with patch.object(h, 'require_feature_audit'), patch.object(h, 'require_delta_audit'):
            h.verify_prepared_evidence(c, {}, {}, audit)
            for name in h.PREPARED_FILES:
                bad = copy.deepcopy(c); bad['artifacts_sha256'].pop(name)
                with self.assertRaises(ValueError): h.verify_prepared_evidence(bad, {}, {}, audit)
            for key, value in [('prefixes', 122), ('feature_prefixes', [0, 1]), ('original_orders_exact', False),
                               ('original_arrays_unchanged', False), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero')]:
                with self.assertRaises(ValueError):
                    h.verify_prepared_evidence(c, {}, {}, dict(audit, **{key: value}))
            for key, value in [('new_returns_computed', True), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('fixed_ols_beta_attempts', 11), ('fixed_ols_beta_estimates', 12)]:
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

    def test_feature_and_rank_receipts_fail_closed(self):
        values = dict(available=np.ones((122, 49), bool))
        check = dict(passed=True, events=122, symbols_per_event=49, available_cells=5978,
                     unavailable_cells=0, computes_returns=False, independent_scalar_rebuild=True,
                     exact_common_availability=True, common_event_cooldown_checked=True,
                     no_future_or_outcome_inputs=True, audit_checked=True,
                     pre_anchor_beta_checked=True, stock_and_market_label_dates_checked=True,
                     shared_beta_support_checked=True, beta_attempts_count=12, beta_estimates_count=11,
                     predictive_model_fits=0)
        h.require_feature_audit(check, values)
        for key, wrong in [('passed', 1), ('events', 121), ('symbols_per_event', 48),
                           ('available_cells', 1), ('unavailable_cells', 1), ('computes_returns', True),
                           ('independent_scalar_rebuild', False), ('exact_common_availability', False),
                           ('common_event_cooldown_checked', False), ('no_future_or_outcome_inputs', False),
                           ('audit_checked', False), ('pre_anchor_beta_checked', False),
                           ('stock_and_market_label_dates_checked', False), ('shared_beta_support_checked', False),
                           ('beta_attempts_count', -1), ('beta_attempts_count', True),
                           ('beta_estimates_count', 13), ('beta_estimates_count', 1.5),
                           ('predictive_model_fits', 1), ('predictive_model_fits', False)]:
            bad = dict(check, **{key: wrong})
            with self.subTest(key=key), self.assertRaises(ValueError): h.require_feature_audit(bad, values)
        for key in ('pre_anchor_beta_checked', 'stock_and_market_label_dates_checked', 'shared_beta_support_checked',
                    'beta_attempts_count', 'beta_estimates_count', 'predictive_model_fits'):
            bad = dict(check); bad.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError): h.require_feature_audit(bad, values)
        deltas = {n: np.zeros((122, 49)) for n in h.NAMES[1:]}
        check = dict(passed=True, full_candidate_universe=True, initial_delta_zero=True,
                     common_available_only=True, no_eligibility_or_model_filter=True,
                     no_cross_sectional_ranking=True, dose=.01, scale=.05, changed_cells={n: 0 for n in h.NAMES[1:]})
        h.require_delta_audit(check, deltas)
        for key in ('passed', 'full_candidate_universe', 'initial_delta_zero',
                    'common_available_only', 'no_eligibility_or_model_filter', 'no_cross_sectional_ranking'):
            with self.assertRaises(ValueError): h.require_delta_audit(dict(check, **{key: False}), deltas)
        with self.assertRaises(ValueError): h.require_delta_audit(dict(check, dose=.03), deltas)
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
