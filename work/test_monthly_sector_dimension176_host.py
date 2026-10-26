"""Synthetic-only tests of176 direct transforms, frozen-score wiring and governance."""
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

spec = importlib.util.spec_from_file_location('tested_dimension176_host', Path(__file__).with_name('monthly_sector_dimension176_host.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def fixture():
    shape = (6, 5)
    values = dict(candidate=np.tile([.1, .2, .3, .4, .5], (6, 1)),
                  control=np.tile([.5, .4, .3, .2, .1], (6, 1)), available=np.ones(shape, bool))
    values['available'][2, :4] = False
    values['available'][3, :] = False
    for key in ('candidate', 'control'):
        values[key][~values['available']] = np.nan
    a = dict(baseline=np.tile([.2, .199, .198, .197, .196], (6, 1)),
             eligible=np.ones(shape, bool), effects=np.zeros((6, 5, 2)))
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
        values['candidate'][1] = [-.10, -.05, 0., .025, .05]
        transforms, deltas = h.scale_and_delta(values, 6)
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], np.array([-.10, -.05, 0., .025, .05]), atol=1e-15)
        self.assertFalse(a['eligible'][1, -1])
        self.assertAlmostEqual(deltas[h.NAMES[1]][1, -1], .02*.05)
        self.assertTrue(np.all(deltas[h.NAMES[1]][0] == 0.))
        self.assertTrue(all(np.max(np.abs(x)) <= .02 for x in deltas.values()))

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
        np.testing.assert_allclose(transforms[h.NAMES[1]][1], values['candidate'][1], atol=1e-15)
        old = delta[h.NAMES[1]][1, 1]
        values['candidate'][1, 0] = 1.
        _, changed = h.scale_and_delta(values, 2)
        self.assertEqual(old, changed[h.NAMES[1]][1, 1])
        self.assertLess(old, .001)

    def test_common_mask_and_invalid_numerics_enforced(self):
        for kind in ('mask', 'infinity', 'shape', 'dtype', 'bound'):
            _, _, values = fixture()
            if kind == 'mask': values['candidate'][1, 0] = np.nan
            if kind == 'infinity': values['control'][1, 0] = np.inf
            if kind == 'shape': values['control'] = values['control'][:, :-1]
            if kind == 'bound': values['candidate'][1, 0] = 1.00001
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

    def test_mean_dimension_control_cannot_promote_and_must_be_beaten(self):
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
        state = dict(completed=[dict(trials=6, directory='old')] * 172 + [dict(trials=2, directory=h.PRIOR.name)],
                     best_training_selected_candidate=dict(directory='monthly_local_linear70'))
        h.state_check(state)
        state['completed'][-1] = dict(trials=8, directory='wrong')
        with self.assertRaises(ValueError): h.state_check(state)


class SourceBindingTests(unittest.TestCase):
    def test_sector_manifest_sources_are_exact_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            mapper = root / 'mapping.py'; mapper.write_text('synthetic mapping')
            sectors = root / 'sectors.csv'; sectors.write_text('synthetic sectors')
            manifests = [root / ('manifest%d.json' % n) for n in range(3)]
            entries = []
            raw = {str(sectors): h.sha(sectors)}
            for symbol in h.SECTOR_ETFS + ('XLC', 'XLRE'):
                path = root / (symbol + '.csv'); path.write_text('synthetic ' + symbol)
                digest = h.sha(path); raw[str(path)] = digest
                entries.append(dict(symbol=symbol, file=str(path), sha256=digest))
            for i, manifest in enumerate(manifests):
                manifest.write_text(json.dumps(entries if i == 0 else []))
                raw[str(manifest)] = h.sha(manifest)
            with patch.multiple(h, ROOT=root, MAPPING_PATH=mapper, MAPPING_SHA=h.sha(mapper),
                                SECTORS_PATH=sectors, ETF_MANIFESTS=tuple(manifests)):
                binding = h.sector_source_binding(raw)
                self.assertEqual(set(binding['sources']), set(h.SECTOR_ETFS) | {'XLC', 'XLRE'})
                self.assertEqual(binding['relative_only_etfs'], ['XLC', 'XLRE'])
                self.assertNotIn('XLC', binding['dimension_etfs'])
                missing = dict(raw); missing.pop(str(sectors))
                with self.assertRaises(ValueError): h.sector_source_binding(missing)
                manifests[1].write_text(json.dumps([entries[0]])); raw[str(manifests[1])] = h.sha(manifests[1])
                with self.assertRaises(ValueError): h.sector_source_binding(raw)

    def test_original_source_panel_and_mapping_are_used(self):
        callback = Mock()
        study = SimpleNamespace(ex=SimpleNamespace(t=SimpleNamespace(strict=SimpleNamespace(
            __file__=str(h.MAPPING_PATH), benchmark_map=callback))))
        adj = pd.DataFrame(columns=list(h.SECTOR_ETFS) + ['XLC', 'XLRE', 'A'])
        sectors = pd.DataFrame(); source = dict(adj=adj, sectors=sectors)
        meta = dict(etfs=['E%d' % n for n in range(17)])
        with patch.object(h, 'sha', return_value=h.MAPPING_SHA):
            panel, mapper = h.sector_inputs(study, source, meta)
            self.assertIs(panel['adj'], adj); self.assertIs(panel['sectors'], sectors)
            self.assertIs(mapper, callback); callback.assert_not_called()
            with self.assertRaises(ValueError): h.sector_inputs(study, dict(adj=adj), meta)
            with self.assertRaises(ValueError): h.sector_inputs(study, dict(adj=adj[['A']], sectors=sectors), meta)
            study.ex.t.strict.__file__ = str(h.ROOT / 'work/replacement.py')
            with self.assertRaises(ValueError): h.sector_inputs(study, source, meta)

    def test_contract_is_fixed_linear_and_states_bounded_novelty(self):
        contract = h.contract()
        self.assertEqual(contract['source_window']['dimension_prices'], 1513)
        self.assertEqual(contract['source_window']['correlation_windows'], 61)
        self.assertEqual(contract['source_window']['relative_prices'], 211)
        self.assertEqual(contract['control_nonpromotable'], h.NAMES[2])
        self.assertIn('0.02*raw directly', contract['scoring'])
        self.assertIn('not proof', contract['novelty'])
        self.assertIn('XLC/XLRE are permitted', contract['mapping'])


class AuditReceiptTests(unittest.TestCase):
    def test_prepared_manifest_and_invariants_are_required(self):
        c = dict(artifacts_sha256={n: 'fake' for n in h.PREPARED_FILES}, new_returns_computed=False,
                 new_model_fits=0, predictive_model_fits=0, model_fit_accounting=h.MODEL_FIT_ACCOUNTING,
                 original_array_count=27, new_predictive_model_fits=0, correlation_window_estimates=61, pair_estimates=2196, relative_quality_estimates=12)
        audit = dict(feature=dict(available_cells=12, correlation_window_estimates=61, pair_estimates=2196, relative_quality_estimates=12), delta={}, prefixes=123,
                     feature_prefixes=list(h.FEATURE_PREFIXES), original_orders_exact=True,
                     original_arrays_unchanged=True, original_array_count=27, new_model_fits=0, predictive_model_fits=0,
                     model_fit_accounting=h.MODEL_FIT_ACCOUNTING, new_predictive_model_fits=0, correlation_window_estimates=61, pair_estimates=2196, relative_quality_estimates=12)
        with patch.object(h, 'require_feature_audit'), patch.object(h, 'require_delta_audit'):
            h.verify_prepared_evidence(c, {}, {}, audit)
            for name in h.PREPARED_FILES:
                bad = copy.deepcopy(c); bad['artifacts_sha256'].pop(name)
                with self.assertRaises(ValueError): h.verify_prepared_evidence(bad, {}, {}, audit)
            for key, value in [('prefixes', 122), ('feature_prefixes', [0, 1]), ('original_orders_exact', False),
                               ('original_arrays_unchanged', False), ('original_array_count', 26), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('new_predictive_model_fits', 1), ('correlation_window_estimates', 60), ('pair_estimates', 2195), ('relative_quality_estimates', 11)]:
                with self.assertRaises(ValueError):
                    h.verify_prepared_evidence(c, {}, {}, dict(audit, **{key: value}))
            for key, value in [('new_returns_computed', True), ('new_model_fits', 1),
                               ('predictive_model_fits', 1), ('model_fit_accounting', 'unqualified zero'),
                               ('original_array_count', 26), ('new_predictive_model_fits', 1),
                               ('correlation_window_estimates', 60), ('pair_estimates', 2195), ('relative_quality_estimates', 11)]:
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
                     sector_dimension_windows_checked=True, relative_stock_sector_checked=True,
                     frozen_mapping_checked=True, etf_neutrality_checked=True, predictive_model_fits=0,
                     mapping_source_sha256=h.MAPPING_SHA, correlation_window_estimates=7442,
                     pair_estimates=267912, relative_quality_estimates=5978)
        h.require_feature_audit(check, values)
        for key, wrong in [('passed', 1), ('events', 121), ('symbols_per_event', 48),
                           ('checked_cells', 5977), ('available_cells', 1), ('unavailable_cells', 1),
                           ('computes_returns', True), ('independent_scalar_rebuild', False),
                           ('exact_common_availability', False), ('no_future_or_outcome_inputs', False),
                           ('audit_checked', False), ('sector_dimension_windows_checked', False),
                           ('relative_stock_sector_checked', False), ('frozen_mapping_checked', False),
                           ('etf_neutrality_checked', False), ('mapping_source_sha256', 'wrong'),
                           ('correlation_window_estimates', 7443), ('pair_estimates', 1),
                           ('relative_quality_estimates', 5977),
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
                     no_cross_sectional_ranking=True, dose=.02, direct_linear_mapping=True, changed_cells={n: 0 for n in h.NAMES[1:]})
        h.require_delta_audit(check, deltas)
        for key in ('passed', 'full_candidate_universe', 'initial_delta_zero',
                    'common_available_only', 'no_eligibility_or_model_filter', 'no_cross_sectional_ranking', 'direct_linear_mapping'):
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
