"""Synthetic host wiring checks; no market arrays or model fitting."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tested_local_year170_host', P/'local_year170_host.py')
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def rank(raw, use):
    result = np.zeros(raw.shape, float)
    for i in range(len(raw)):
        values = pd.Series(raw[i, use[i]])
        result[i, use[i]] = values.rank(method='average', pct=True).to_numpy()-.5
    return result


def fixture():
    shape = (6, 6)
    source = np.tile(np.array([0., 2., 1., 5., 3., 4.]), (shape[0], 1))
    old = np.tile(np.array([1., 0., 5., 3., 2., 4.]), (shape[0], 1))
    use = np.ones(shape, bool); use[:, -1] = False
    a = dict(baseline=np.tile([.25, .02, -.1, .13, -.05, .07], (6, 1)),
        eligible=use, forecast_available=np.ones(shape, bool), model_available=np.ones(shape, bool),
        raw_source_log=source, raw_local_linear=old, raw_local_constant=old.copy(),
        rank_log_excess=rank(source, use),
        state_weak_narrow=np.array([True, True, False, False, False, True]),
        state_calm10=np.array([False, True, True, False, True, False]),
        extension_available=np.ones(shape, bool), effects=np.zeros((6, 6, 2)))
    a['extension_available'][2, 1] = False
    a['extension_available'][4] = False
    f = dict(raw=dict(local_year_stability=np.tile(np.arange(6., 0., -1.), (6, 1)).reshape(-1),
                      local_half=((source+old)/2.).reshape(-1)), available=np.ones(shape, bool).reshape(-1))
    meta = dict(dates=['synthetic-%d' % i for i in range(6)], symbols=[list('ABCDEF') for _ in range(6)])
    return a, meta, f


class SyntheticStudy:
    """A deterministic ranker/selector double with observable adapter calls."""
    ARMS = ('local_constant', 'local_linear')

    def __init__(self):
        self.ep = SimpleNamespace(select_boundary=Mock(side_effect=self.boundary))
        self.adapter = SimpleNamespace(score_and_choose=Mock(side_effect=self.choose_adapter))
        self.rp = SimpleNamespace(make_orders=self.make_orders)

    @staticmethod
    def configs():
        return [dict(id=i) for i in range(5)]

    @staticmethod
    def original_rank_inputs(a, n):
        return a['rank_log_excess'][:n].copy(), a['forecast_available'][:n] & a['eligible'][:n]

    def ranked_forecasts(self, values, available, a, n):
        use = self.original_rank_inputs(a, n)[1]
        result = {key:a['rank_log_excess'][:n].copy() for key in self.ARMS}
        ready = np.array([bool(use[i].any() and available[i, use[i]].all() and
            all(np.isfinite(values[key][i, use[i]]).all() for key in self.ARMS)) for i in range(n)], bool)
        for key in self.ARMS:
            if np.isinf(values[key][:n]).any():
                raise FloatingPointError('Infinite synthetic forecast')
            for i in np.flatnonzero(ready):
                result[key][i, use[i]] = pd.Series(values[key][i, use[i]]).rank(
                    method='average', pct=True).to_numpy()-.5
        return result, ready

    def modified_pre_score(self, spec, a, n):
        assert spec == dict(id=4)
        old = rank(a['raw_local_linear'], a['eligible'] & a['forecast_available'])
        coef = .25*(.05*a['state_weak_narrow'][:, None]+.01*
            (a['state_calm10'] & ~a['state_weak_narrow'])[:, None]*a['extension_available'])
        # Existing global-model contribution is deliberately large and distinct.
        return a['baseline']+.07*a['rank_log_excess']+coef*(old-a['rank_log_excess'])

    @staticmethod
    def boundary(pre, a, meta, n):
        result = []
        for i in range(n):
            ids = np.flatnonzero(a['eligible'][i])
            ids = sorted(ids, key=lambda j:(-pre[i, j], j))
            chosen = ids[:3] if len(ids) >= 2 else []
            result.append([meta['symbols'][i][j] for j in chosen])
        return result, pre[:n].copy()

    def choices(self, spec, a, meta, n, *, policy):
        return self.boundary(self.modified_pre_score(spec, a, n), a, meta, n)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=ss, scores=scores[i].tolist())
                for i, ss in enumerate(choices)]

    def choose_adapter(self, base, weight, residual, a, meta, n, *, pre_score,
                       effects, episode_policy, boundary_selector):
        assert base == 'incumbent66' and weight == 0.
        assert residual.shape == a['baseline'].shape and np.isnan(residual).all()
        assert effects is a['effects']
        assert boundary_selector is self.ep.select_boundary
        return boundary_selector(pre_score, a, meta, n)


class RankSlotTests(unittest.TestCase):
    def test_exact_regime_coefficients_and_partial_first_event(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        ranks, deltas = h.rank_and_delta(s, a, forecast, 6)
        use = a['eligible'] & a['forecast_available']; old = rank(a['raw_local_linear'], use)
        coefficient = np.zeros(a['baseline'].shape)
        coefficient[1, :] = .0125
        coefficient[2, :] = .0025; coefficient[2, 1] = 0.
        coefficient[5, :] = .0125
        for name in h.NAMES[1:]:
            want_rank = rank(forecast['raw'][h.RAW[name]].reshape(a['baseline'].shape), use)
            np.testing.assert_array_equal(ranks[name], want_rank)
            want_delta = coefficient*(want_rank-old); want_delta[~use] = 0.
            np.testing.assert_array_equal(deltas[name], want_delta)
            np.testing.assert_array_equal(deltas[name][0], np.zeros(6))
            np.testing.assert_array_equal(deltas[name][3:5], np.zeros((2, 6)))
        self.assertTrue(np.any(deltas[h.NAMES[1]][1] != 0.))
        self.assertTrue(np.any(deltas[h.NAMES[1]][2] != 0.))

    def test_ineligible_and_unavailable_rows_do_not_gain_delta(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        forecast['available'].reshape(6, 6)[1, 0] = False
        forecast['raw']['local_year_stability'].reshape(6, 6)[5, 1] = np.nan
        _, deltas = h.rank_and_delta(s, a, forecast, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(deltas[name][1], np.zeros(6))
            np.testing.assert_array_equal(deltas[name][:, -1], np.zeros(6))
        np.testing.assert_array_equal(deltas[h.NAMES[1]][5], np.zeros(6))

    def test_identical_old_raw_preserves_every_pre_score_and_order(self):
        a, meta, forecast = fixture(); s = SyntheticStudy()
        for key in forecast['raw']:
            forecast['raw'][key] = a['raw_local_linear'].reshape(-1).copy()
        before = s.modified_pre_score(s.configs()[4], a, 6)
        policy = dict(synthetic=True)
        with patch.object(h, 'read', return_value=policy):
            orders, ranks, deltas = h.decisions(s, a, meta, forecast, 6)
        for name in h.NAMES[1:]:
            np.testing.assert_array_equal(deltas[name], np.zeros((6, 6)))
            self.assertEqual(orders[name], orders[h.NAMES[0]])
        for call in s.adapter.score_and_choose.call_args_list:
            self.assertEqual(call.kwargs['pre_score'].tobytes(), before.tobytes())

    def test_replaces_local_quarter_preserves_global_and_runs_episode_once(self):
        a, meta, forecast = fixture(); s = SyntheticStudy(); policy = dict(synthetic=True)
        original_pre = s.modified_pre_score(s.configs()[4], a, 6)
        with patch.object(h, 'read', return_value=policy) as reader:
            orders, ranks, deltas = h.decisions(s, a, meta, forecast, 6)
        reader.assert_called_once_with(h.LOCAL/'INCUMBENT_POLICY.json')
        self.assertEqual(s.adapter.score_and_choose.call_count, 2)
        self.assertEqual(s.ep.select_boundary.call_count, 2)
        for name, call in zip(h.NAMES[1:], s.adapter.score_and_choose.call_args_list):
            expected = original_pre.copy(); changed = deltas[name] != 0.
            expected[changed] += deltas[name][changed]
            self.assertEqual(call.kwargs['pre_score'].tobytes(), expected.tobytes())
            self.assertIs(call.kwargs['effects'], a['effects'])
            self.assertIs(call.kwargs['episode_policy'], policy)
            # A whole-model replacement would move the large .07 global slot;
            # this test instead verifies only the existing local-quarter delta.
            np.testing.assert_array_equal(call.kwargs['pre_score'][0], original_pre[0])
        self.assertEqual(set(orders), set(h.NAMES))

    def test_every_small_prefix_and_poisoned_future_and_no_input_mutation(self):
        a, meta, forecast = fixture(); before_a = copy.deepcopy(a); before_f = copy.deepcopy(forecast)
        with patch.object(h, 'read', return_value={'synthetic':True}):
            all_orders, all_ranks, all_deltas = h.decisions(SyntheticStudy(), a, meta, forecast, 6)
            for n in range(7):
                orders, ranks, deltas = h.decisions(SyntheticStudy(), a, meta, forecast, n)
                for name in h.NAMES:
                    self.assertEqual(orders[name], all_orders[name][:n])
                for name in h.NAMES[1:]:
                    np.testing.assert_array_equal(ranks[name], all_ranks[name][:n])
                    np.testing.assert_array_equal(deltas[name], all_deltas[name][:n])
            for n in (1, 3, 5):
                poison = copy.deepcopy(forecast)
                for values in poison['raw'].values():
                    values.reshape(6, 6)[n:] = np.inf
                orders, ranks, deltas = h.decisions(SyntheticStudy(), a, meta, poison, n)
                for name in h.NAMES:
                    self.assertEqual(orders[name], all_orders[name][:n])
        for key, value in a.items():
            np.testing.assert_array_equal(value, before_a[key])
        for key, value in forecast['raw'].items():
            np.testing.assert_array_equal(value, before_f['raw'][key])
        np.testing.assert_array_equal(forecast['available'], before_f['available'])


def selection_fixture():
    metric = {era:dict(cagr=.5, sharpe=1.5, max_drawdown=-.3) for era in ('training','early','recent')}
    metrics = {name:{path:copy.deepcopy(metric) for path in ('same_close_zero','next_open_cost')}
               for name in h.NAMES}
    orders = {name:[dict(symbols=['A','B','C']) for _ in range(122)] for name in h.NAMES}
    for path in metrics[h.NAMES[1]]:
        metrics[h.NAMES[1]][path]['training']['cagr'] += .01
    for i in (1, 2, 53):
        orders[h.NAMES[1]][i]['symbols'] = ['A','B','D']
    return metrics, orders


class BindingAndSelectionTests(unittest.TestCase):
    def test_bind_adds_or_accepts_exact_digest_and_rejects_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic-source'; path.write_bytes(b'synthetic only\n')
            pins = {}; h.bind(pins, path)
            self.assertEqual(pins, {str(path):h.sha(path)})
            h.bind(pins, path)
            old = dict(pins); path.write_bytes(b'changed synthetic\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting inherited'):
                h.bind(pins, path)
            self.assertEqual(pins, old)

    def test_selection_ignores_later_orders_and_extra_report_only_metrics(self):
        metrics, orders = selection_fixture()
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[1])
        for name in h.NAMES:
            metrics[name]['later34'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            metrics[name]['full122'] = dict(cagr=1000. if name != h.NAMES[1] else -1.)
            for i in range(88, 122):
                orders[name][i]['symbols'] = [name+str(i)]
        self.assertEqual(h.select_first88(metrics, orders), result)

    def test_half_cannot_be_promoted_and_candidate_must_clear_both_controls(self):
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['same_close_zero']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['close_vs_half'])
        metrics, orders = selection_fixture()
        metrics[h.NAMES[2]]['next_open_cost']['training']['cagr'] = 2.
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['open_vs_half'])

    def test_first88_changed_set_support_cannot_come_from_later34(self):
        metrics, orders = selection_fixture()
        for i in (1, 2, 53):
            orders[h.NAMES[1]][i]['symbols'] = ['C','B','A']
        for i in (90, 100, 110):
            orders[h.NAMES[1]][i]['symbols'] = ['A','B','D']
        result = h.select_first88(metrics, orders)
        self.assertEqual(result['changed_months'], [])
        self.assertEqual(result['primary'], h.NAMES[0])
        self.assertFalse(result['guards']['changed_support'])


class AuditReceiptTests(unittest.TestCase):
    @staticmethod
    def model_receipt():
        forecast = dict(available=np.array([True, True, False, True]), audit=dict(
            annual=[dict(synthetic=i) for i in range(11)],
            local=[dict(folds=[{}, {}]), dict(folds=[{}]), dict(folds=[])],
            counts=dict(deletion_evaluations=3)))
        check = dict(passed=True, independent_augmented_least_squares=True,
            original_source_and_local_parity=True, availability_unchanged=True,
            not_held_out_validation=True, computes_returns=False,
            annual_cohorts=11, local_queries=3, deletion_folds=3)
        return check, forecast

    @staticmethod
    def slot_receipt():
        deltas = {h.NAMES[1]:np.array([[0., .1], [0., -.2]]),
                  h.NAMES[2]:np.array([[0., 0.], [.3, 0.]])}
        check = dict(passed=True, full_original_universe=True, includes_eligible_etfs=True,
            no_local_history_filter=True, replacement_not_additive_local_slot=True,
            ready_rows={name:2 for name in h.RAW},
            changed_cells={h.NAMES[1]:2, h.NAMES[2]:1})
        return check, deltas

    @staticmethod
    def outcome_receipt():
        one = dict(passed=True, monthly_executions=122,
            slices={name:dict(synthetic=True) for name in
                    ('full122','first88','early53','recent35','later34')})
        return dict(passed=True, independent_accounts=True,
            independent_of_production_model_and_engine=True, checked_paths=9, checked_slices=45,
            terminal_liquidation=False, accounting_source_sha256='synthetic-code-sha',
            paths={arm:{path[0]:copy.deepcopy(one) for path in h.ev.PATHS} for arm in h.NAMES})

    def test_valid_model_receipt_and_each_required_boolean(self):
        check, forecast = self.model_receipt()
        h.require_model_audit(check, forecast)
        for key in ('passed','independent_augmented_least_squares','original_source_and_local_parity',
                    'availability_unchanged','not_held_out_validation'):
            for value in (None, False, 1):
                bad = copy.deepcopy(check)
                if value is None:
                    bad.pop(key)
                else:
                    bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    h.require_model_audit(bad, forecast)
        for value in (None, True, 0):
            bad = copy.deepcopy(check); bad['computes_returns'] = value
            with self.subTest(computes_returns=value), self.assertRaises(ValueError):
                h.require_model_audit(bad, forecast)

    def test_model_coverage_requires_matching_receipt_arrays_and_fold_counts(self):
        check, forecast = self.model_receipt()
        for key in ('annual_cohorts','local_queries','deletion_folds'):
            bad = copy.deepcopy(check); bad[key] -= 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.require_model_audit(bad, forecast)
        changes = ('annual','local','available','count','folds')
        for change in changes:
            bad = copy.deepcopy(forecast)
            if change == 'annual':
                bad['audit']['annual'].pop()
            elif change == 'local':
                bad['audit']['local'].pop()
            elif change == 'available':
                bad['available'][0] = False
            elif change == 'count':
                bad['audit']['counts']['deletion_evaluations'] = 4
            else:
                bad['audit']['local'][0]['folds'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                h.require_model_audit(check, bad)

    def test_valid_slot_receipt_flags_arm_keys_readiness_and_actual_changed_counts(self):
        check, deltas = self.slot_receipt()
        h.require_slot_audit(check, deltas)
        for key in ('passed','full_original_universe','includes_eligible_etfs',
                    'no_local_history_filter','replacement_not_additive_local_slot'):
            bad = copy.deepcopy(check); bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.require_slot_audit(bad, deltas)
        for change in ('ready_missing','ready_extra','ready_mismatch','count_wrong','count_missing','count_extra'):
            bad = copy.deepcopy(check)
            if change == 'ready_missing':
                bad['ready_rows'].pop(h.NAMES[1])
            elif change == 'ready_extra':
                bad['ready_rows']['unknown_arm'] = 2
            elif change == 'ready_mismatch':
                bad['ready_rows'][h.NAMES[1]] = 1
            elif change == 'count_wrong':
                bad['changed_cells'][h.NAMES[1]] = 1
            elif change == 'count_missing':
                bad['changed_cells'].pop(h.NAMES[1])
            else:
                bad['changed_cells']['unknown_arm'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                h.require_slot_audit(bad, deltas)
        bad_deltas = copy.deepcopy(deltas); bad_deltas[h.NAMES[1]][0, 1] = 0.
        with self.assertRaises(ValueError):
            h.require_slot_audit(check, bad_deltas)

    def test_valid_outcome_receipt_binds_exact_accounting_source(self):
        check = self.outcome_receipt()
        with patch.object(h, 'sha', return_value='synthetic-code-sha') as hashing:
            h.require_outcome_audit(check)
        hashing.assert_called_once_with(h.ROOT/'work/ptr169_results_independent.py')
        with patch.object(h, 'sha', return_value='different-code-sha'):
            with self.assertRaisesRegex(ValueError, 'accounting source'):
                h.require_outcome_audit(check)

    def test_outcome_rejects_missing_flags_wrong_counts_terminal_and_nested_failures(self):
        source = self.outcome_receipt()
        for key in ('passed','independent_accounts','independent_of_production_model_and_engine'):
            for value in (None, False, 1):
                bad = copy.deepcopy(source)
                if value is None:
                    bad.pop(key)
                else:
                    bad[key] = value
                with patch.object(h, 'sha', return_value='synthetic-code-sha'):
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        h.require_outcome_audit(bad)
        for key, value in (('checked_paths',8), ('checked_slices',44), ('terminal_liquidation',True)):
            bad = copy.deepcopy(source); bad[key] = value
            with patch.object(h, 'sha', return_value='synthetic-code-sha'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    h.require_outcome_audit(bad)
        for change in ('arm_missing','arm_extra','path_missing','path_extra',
                       'month_count','path_failed','path_missing_passed','slice_missing','slice_extra'):
            bad = copy.deepcopy(source); arm = bad['paths'][h.NAMES[0]]; path = arm[h.ev.PATHS[0][0]]
            if change == 'arm_missing':
                bad['paths'].pop(h.NAMES[0])
            elif change == 'arm_extra':
                bad['paths']['unknown_arm'] = copy.deepcopy(arm)
            elif change == 'path_missing':
                arm.pop(h.ev.PATHS[0][0])
            elif change == 'path_extra':
                arm['unknown_path'] = copy.deepcopy(path)
            elif change == 'month_count':
                path['monthly_executions'] = 121
            elif change == 'path_failed':
                path['passed'] = False
            elif change == 'path_missing_passed':
                path.pop('passed')
            elif change == 'slice_missing':
                path['slices'].pop('later34')
            else:
                path['slices']['unknown_slice'] = {}
            with patch.object(h, 'sha', return_value='synthetic-code-sha'):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    h.require_outcome_audit(bad)


if __name__ == '__main__':
    unittest.main()
