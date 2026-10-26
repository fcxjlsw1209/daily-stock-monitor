"""Synthetic only: frozen selector integration, support and fixed accounting."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import numpy as np
import pandas as pd
import vvix160_runner as r

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.base.module('vvix160_synthetic_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
EPISODE = r.base.module('vvix160_synthetic_episode', ROOT/'outputs/monthly_daily_episodes/episodes.py')
POLICY = dict(budget=.05, ceiling=5, gap=.02)
ORIGINAL_KEYS = ('baseline', 'core', 'core_weight', 'state_calm10', 'state_weak_narrow', 'eligible',
                 'effects', 'old_ml4', 'pre_score', 'forecast_available', 'rank_simple_excess',
                 'rank_log_excess', 'rank_vol_scaled', 'rank_loss_utility', 'raw_simple_excess',
                 'raw_log_excess', 'raw_vol_scaled', 'raw_loss_utility', 'extension_available',
                 'raw_source_log', 'model_available', 'local_available', 'raw_local_constant',
                 'raw_local_linear', 'rank_local_constant', 'rank_local_linear', 'paired_month_ready')


def fixture(n=4):
    names = ['S'+str(i) for i in range(32)] + ['ETF'+str(i) for i in range(17)]
    dates = [(date(2017, 1, 31)+timedelta(days=31*i)).isoformat() for i in range(n)]
    meta = dict(dates=dates, ends=dates, symbols=[names[:] for _ in dates], etfs=names[-17:], train_months=88)
    shape = (n, 49)
    a = {key: np.zeros(shape) for key in ORIGINAL_KEYS}
    a.update(baseline=np.tile(np.linspace(.2, .1, 49), (n, 1)),
             core_weight=np.full(n, .5), eligible=np.ones(shape, bool),
             state_calm10=np.ones(n, bool), state_weak_narrow=np.zeros(n, bool))
    a['core'] = .5*a['baseline']
    a['delta_vix'] = np.tile(np.linspace(-.01, .01, 49), (n, 1))
    a['delta_vvix'] = -a['delta_vix'].copy()
    return a, meta


class SyntheticBackend:
    adapter = ADAPTER

    def __init__(self):
        self.calls = []

    def pre(self, arrays, n):
        pre = np.zeros_like(arrays['baseline'])
        pre[:n] = arrays['baseline'][:n]
        return pre

    def selector(self, scores, good, symbols, effects, original, policy):
        self.calls.append(deepcopy(dict(scores=scores, good=good, symbols=symbols,
                                       effects=effects, original=original, policy=policy)))
        return EPISODE.select_boundary(scores, good, symbols, effects, original, policy)

    def baseline(self, a, m, n, policy):
        return self.adapter.score_and_choose('incumbent66', 0., np.zeros_like(a['baseline']), a, m, n,
                                             pre_score=self.pre(a, n), effects=a['effects'],
                                             episode_policy=policy, boundary_selector=self.selector)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(date=meta['dates'][i], symbols=[meta['symbols'][i][j] for j in choice],
                     scores=[float(scores[i, j]) for j in choice]) for i, choice in enumerate(choices)]


class Scoring(unittest.TestCase):
    def test_actual_delta_applied_once_then_one_frozen_episode(self):
        a, m = fixture(); b = SyntheticBackend()
        choices, scores, _ = r.choices_all(a, m, 4, b, POLICY)
        self.assertEqual(len(b.calls), 3)
        for index, key in enumerate(r.HANDS, 1):
            call = b.calls[index]
            np.testing.assert_array_equal(call['scores'], b.pre(a, 4)+a[key])
            np.testing.assert_array_equal(call['original']['baseline'], a['baseline'])
            np.testing.assert_array_equal(call['effects'], a['effects'])
            expected = EPISODE.select_boundary(call['scores'], a['eligible'], m['symbols'],
                                                a['effects'], call['original'], POLICY)
            self.assertEqual(choices[r.NAMES[index]], expected[0])
            np.testing.assert_array_equal(scores[r.NAMES[index]], expected[1])

    def test_zero_deltas_exact_bytes_and_single_baseline_call(self):
        a, m = fixture(); b = SyntheticBackend()
        for key in r.HANDS: a[key][:] = 0
        choices, scores, _ = r.choices_all(a, m, 4, b, POLICY)
        self.assertEqual(len(b.calls), 1)
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name], choices[r.NAMES[0]])
            self.assertEqual(scores[name].tobytes(), scores[r.NAMES[0]].tobytes())

    def test_initial_and_zero_rows_exact_in_otherwise_active_arm(self):
        a, m = fixture(); m['dates'][0] = r.INITIAL_PARTIAL
        for key in r.HANDS: a[key][:2] = 0
        choices, scores, _ = r.choices_all(a, m, 4, SyntheticBackend(), POLICY)
        for name in r.NAMES[1:]:
            self.assertEqual(choices[name][:2], choices[r.NAMES[0]][:2])
            self.assertEqual(scores[name][:2].tobytes(), scores[r.NAMES[0]][:2].tobytes())

    def test_nonzero_initial_rejected(self):
        a, m = fixture(); m['dates'][0] = r.INITIAL_PARTIAL
        with self.assertRaisesRegex(ValueError, 'Initial partial'):
            r.choices_all(a, m, 4, SyntheticBackend(), POLICY)

    def test_full49_including_etfs_and_ineligible_are_not_reranked(self):
        a, m = fixture(); a['eligible'][:, 0] = False; b = SyntheticBackend()
        a['delta_vix'][:] = .003  # Deliberately uncentered, including ETFs.
        choices, _, _ = r.choices_all(a, m, 4, b, POLICY)
        np.testing.assert_array_equal(b.calls[1]['scores'], b.pre(a, 4)+.003)
        self.assertTrue(all(0 not in row for row in choices[r.NAMES[1]]))

    def test_invalid_delta_values_and_bounds(self):
        for bad in (np.nan, np.inf, -.010001, .010001, True, 'bad'):
            a, m = fixture(); a['delta_vix'] = a['delta_vix'].astype(object); a['delta_vix'][0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.choices_all(a, m, 4, SyntheticBackend(), POLICY)

    def test_wrong_shape_axes_boolean_mask_or_policy(self):
        for fault in ('shape', 'axes', 'mask', 'duplicates', 'policy'):
            a, m = fixture(); policy = POLICY
            if fault == 'shape': a['delta_vix'] = a['delta_vix'][:, :-1]
            if fault == 'axes': m['dates'] = m['dates'][:-1]
            if fault == 'mask': a['eligible'] = a['eligible'].astype(int)
            if fault == 'duplicates': m['symbols'][0][0] = m['symbols'][0][1]
            if fault == 'policy': policy = dict(POLICY, budget=.06)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                r.choices_all(a, m, 4, SyntheticBackend(), policy)

    def test_all123_prefixes_and_original27_are_unchanged(self):
        a, m = fixture(122); before = {k: v.tobytes() for k, v in a.items()}
        self.assertEqual(len(set(a)-set(r.EXTRA)), 27)
        with patch.object(r, 'choices_all', wraps=r.choices_all) as choose:
            orders, scores, audit = r.panels(a, m, SyntheticBackend(), POLICY)
        self.assertEqual([x.args[2] for x in choose.call_args_list], [122]+list(range(123)))
        self.assertEqual(sum(map(len, orders.values())), 366)
        self.assertEqual(len(audit), 122)
        self.assertEqual(before, {k: v.tobytes() for k, v in a.items()})

    def test_poison_future_deltas_and_metadata_ignored(self):
        a, m = fixture(); expected = r.choices_all(a, m, 1, SyntheticBackend(), POLICY)
        for key in r.HANDS:
            a[key] = a[key].astype(object); a[key][1:] = 'future'
        m['symbols'][1:] = [None]*3
        actual = r.choices_all(a, m, 1, SyntheticBackend(), POLICY)
        self.assertEqual(expected[0], actual[0])
        for name in r.NAMES: self.assertEqual(expected[1][name].tobytes(), actual[1][name].tobytes())

    def test_empty_prefix_ignores_all_future_values(self):
        a, m = fixture()
        for key in r.HANDS: a[key] = np.full(a[key].shape, 'future', dtype=object)
        choices, scores, audit = r.choices_all(a, m, 0, SyntheticBackend(), POLICY)
        self.assertEqual(audit, [])
        self.assertTrue(all(choices[name] == [] and scores[name].shape == (0, 49) for name in r.NAMES))

    def test_original_two_half_and_cash_choice_counts(self):
        for count in (1, 2):
            a, m = fixture(); a['eligible'][:] = False; a['eligible'][:, :count] = True
            choices, _, _ = r.choices_all(a, m, 4, SyntheticBackend(), POLICY)
            self.assertTrue(all(len(row) == (0 if count == 1 else 2) for name in r.NAMES for row in choices[name]))

    def test_backend_inherits_original_config4_and_price_parity(self):
        b = object.__new__(r.Backend)
        specs = [{'id': i} for i in range(5)]
        b.local = NS(configs=Mock(return_value=specs), modified_pre_score=Mock(return_value='pre'),
                     choices=Mock(return_value='choices'))
        b.selector = Mock()
        self.assertEqual(b.pre('arrays', 3), 'pre')
        b.local.modified_pre_score.assert_called_once_with(specs[4], 'arrays', 3)
        self.assertEqual(b.baseline('arrays', 'meta', 3, POLICY), 'choices')
        self.assertEqual(b.local.choices.call_args.args[:4], (specs[4], 'arrays', 'meta', 3))
        self.assertIs(r.Backend.price_parity, r.previous.Backend.price_parity)


def source_fixture():
    _, meta = fixture(122); meta['dates'][0] = r.INITIAL_PARTIAL
    monthly = []
    for i, day in enumerate(meta['dates']):
        count = 0 if i == 0 else 252
        monthly.append(dict(signal_index=i, date=day, initial_partial=i == 0,
                            window_start=None if i == 0 else '2010-01-01',
                            window_end=None if i == 0 else '2011-01-01', window_size=count,
                            VIX=dict(passed=True, observations=count, m=0.),
                            VVIX=dict(passed=True, observations=count, m=0.)))
    return monthly, meta


class SourceSupport(unittest.TestCase):
    def test_real_kernel_interface_with_synthetic_sources(self):
        import vvix160_kernel as kernel
        start = date(2014, 1, 1)
        sessions = [(start+timedelta(days=i)).isoformat() for i in range(3200)
                    if (start+timedelta(days=i)).weekday() < 5]
        initial = sessions.index(r.INITIAL_PARTIAL)
        dates = [r.INITIAL_PARTIAL]+[sessions[initial+1+10*i] for i in range(121)]
        values = {'VIX': {d: float(i+1) for i, d in enumerate(sessions)},
                  'VVIX': {d: float(len(sessions)-i) for i, d in enumerate(sessions)}}
        states = kernel.build_states(dates, sessions, values)
        a, meta = fixture(122); meta.update(dates=dates, ends=dates)
        support = r.source_support(states['audit']['monthly'], meta)
        self.assertTrue(support['passed'])
        vol = np.full((122, 49), .5)  # No centered-rank assumption.
        a.update(kernel.build_deltas(states, vol))
        backend = SyntheticBackend()
        r.choices_all(a, meta, 122, backend, POLICY)
        np.testing.assert_array_equal(backend.calls[1]['scores'][1:], backend.pre(a, 122)[1:]-.01)
        np.testing.assert_array_equal(backend.calls[2]['scores'][1:], backend.pre(a, 122)[1:]+.01)

    def test_all121_windows_in_fixed_eras(self):
        rows, meta = source_fixture(); result = r.source_support(rows, meta)
        self.assertTrue(result['passed'])
        self.assertEqual([result[k] for k in ('full_window_months', 'first88', 'early53', 'recent35', 'later34')],
                         [121, 87, 52, 35, 34])

    def test_later_source_missing_also_blocks_pre_outcome_support(self):
        rows, meta = source_fixture(); rows[-1]['VVIX']['passed'] = False
        result = r.source_support(rows, meta)
        self.assertFalse(result['passed']); self.assertEqual(result['missing_months'], [121])

    def test_wrong_source_count_not_admitted(self):
        rows, meta = source_fixture(); rows[2]['VIX']['observations'] = 251
        self.assertFalse(r.source_support(rows, meta)['passed'])

    def test_malformed_source_axis_or_initial_is_rejected(self):
        for fault in ('short', 'date', 'index', 'bool', 'window', 'initial'):
            rows, meta = source_fixture()
            if fault == 'short': rows.pop()
            if fault == 'date': rows[1]['date'] = 'wrong'
            if fault == 'index': rows[1]['signal_index'] = True
            if fault == 'bool': rows[1]['VIX']['passed'] = 1
            if fault == 'window': rows[1]['window_size'] = 251
            if fault == 'initial': rows[0]['VVIX']['m'] = .5
            with self.subTest(fault=fault), self.assertRaises(ValueError): r.source_support(rows, meta)


class Selection(unittest.TestCase):
    def rows(self):
        return [dict(name=n, training=dict(cagr=.2, sharpe=1., max_drawdown=-.2),
                     early=dict(cagr=.2), recent=dict(cagr=.2)) for n in r.NAMES]

    def select(self, rows, opened=None, support=None, changed=None):
        return r.select_first88(rows, self.rows() if opened is None else opened,
                                {'passed': True} if support is None else support,
                                {n: {'passed': True} for n in r.NAMES[1:]} if changed is None else changed)

    def test_close_gain_and_open_cost_noninferiority(self):
        rows = self.rows(); rows[1]['training']['cagr'] = .2049
        self.assertFalse(self.select(rows)['admitted'])
        rows[1]['training']['cagr'] = .205
        self.assertEqual(self.select(rows)['primary'], r.NAMES[1])
        opened = self.rows(); opened[1]['training']['cagr'] = .199
        self.assertFalse(self.select(rows, opened)['admitted'])

    def test_each_close_and_open_guard(self):
        for phase in ('close', 'open'):
            for field, value in (('sharpe', .949), ('max_drawdown', -.231), ('early', .174), ('recent', .174)):
                rows, opened = self.rows(), self.rows(); rows[1]['training']['cagr'] = .21
                target = rows if phase == 'close' else opened
                if field in ('early', 'recent'): target[1][field]['cagr'] = value
                else: target[1]['training'][field] = value
                with self.subTest(phase=phase, field=field): self.assertFalse(self.select(rows, opened)['admitted'])

    def test_fixed_ties_and_highest_first88_close_only(self):
        rows = self.rows(); rows[1]['training']['cagr'] = rows[2]['training']['cagr'] = .21
        self.assertEqual(self.select(rows)['primary'], r.NAMES[1])
        rows[2]['training']['cagr'] += 5e-13
        self.assertEqual(self.select(rows)['primary'], r.NAMES[1])
        rows[2]['training']['cagr'] = .211
        self.assertEqual(self.select(rows)['primary'], r.NAMES[2])

    def test_nonfinite_metric_and_missing_support_do_not_pass(self):
        rows = self.rows(); rows[1]['training']['cagr'] = np.inf
        self.assertFalse(self.select(rows)['admitted'])
        rows[1]['training']['cagr'] = .4
        self.assertFalse(self.select(rows, support={'passed': False})['admitted'])
        self.assertFalse(self.select(rows, changed={n: {'passed': False} for n in r.NAMES[1:]})['admitted'])

    def test_later34_metrics_do_not_change_selection(self):
        rows = self.rows(); rows[1]['training']['cagr'] = .21
        rows[2]['later34'] = {'cagr': 999.}; rows[1]['later34'] = {'cagr': -1.}
        result = self.select(rows)
        self.assertEqual(result['primary'], r.NAMES[1]); self.assertEqual(result['selection_count'], 1)
        self.assertFalse(result['later34_used_for_selection'])

    def test_wrong_arm_order_rejected(self):
        with self.assertRaises(ValueError): self.select(list(reversed(self.rows())))

    def test_changes_require_three_sets_and_both_training_eras(self):
        orders = {n: [dict(symbols=['S0', 'S1', 'S2']) for _ in range(122)] for n in r.NAMES}
        name = r.NAMES[1]; orders[name][0]['symbols'].reverse()
        orders[name][100]['symbols'] = ['S0', 'S1', 'S3']
        self.assertEqual(r.changed_support(orders, name)['months'], [])
        for i in (1, 2, 3): orders[name][i]['symbols'] = ['S0', 'S1', 'S3']
        self.assertFalse(r.changed_support(orders, name)['passed'])
        orders[name][53]['symbols'] = ['S0', 'S1', 'S3']
        result = r.changed_support(orders, name)
        self.assertTrue(result['passed']); self.assertEqual(result['months'], [1, 2, 3, 53])

    def test_duplicate_holdings_rejected(self):
        orders = {n: [dict(symbols=['S0', 'S1', 'S1']) for _ in range(88)] for n in r.NAMES}
        with self.assertRaises(ValueError): r.changed_support(orders, r.NAMES[1])


class AccountingWiring(unittest.TestCase):
    def backend(self):
        _, m = fixture(122)
        orders = {name: [dict(date=d, symbols=['S0', 'S1']) for d in m['dates']] for name in r.NAMES}
        index = pd.date_range(m['dates'][0], m['ends'][-1])
        eq = pd.Series(np.linspace(100., 120., len(index)), index=index)
        metrics = dict(training={'cagr': .1}, early={'cagr': .1}, recent={'cagr': .1})
        rp = NS(paths=Mock(return_value=(None, None, None, None)), fast_returns=Mock(return_value=np.zeros(10)),
                stats_rr=Mock(return_value={'cagr': .1}))
        replay = Mock(side_effect=lambda *a, **kw: ({'annual_returns': [{'year': 2016}, {'year': 2026}]}, eq.copy(), [], []))
        b = object.__new__(r.Backend)
        b.local = NS(rp=rp, old=NS(session=NS(measure=Mock(return_value=metrics))),
                     ex=NS(e=NS(engine=NS(replay=replay))),
                     arch=NS(old=NS(prior=NS(old=NS(previous=NS(cuts=Mock(return_value={})))))))
        return b, m, orders, rp, replay

    def test_training_only_first88_no_fold_reselection(self):
        b, m, orders, rp, replay = self.backend(); rows, opened = b.training(orders, m, None, None)
        self.assertEqual(rp.paths.call_count, 1); self.assertEqual(rp.paths.call_args.args[2], 88)
        self.assertEqual(replay.call_count, 3)
        self.assertEqual([x['name'] for x in rows], list(r.NAMES))
        self.assertEqual([x['name'] for x in opened], list(r.NAMES))
        for call in replay.call_args_list:
            self.assertEqual(len(call.args[2]), 88); self.assertEqual(call.kwargs['cost'], .0005)
            self.assertEqual(call.kwargs['execution'], 'open')
            self.assertEqual(call.kwargs['end'], pd.Timestamp(m['ends'][87]))

    def test_diagnostics_exact_nine_paths_and27_artifacts(self):
        b, m, orders, _, replay = self.backend(); written = []
        result = b.diagnostics(orders, m, None, None, lambda name, value: written.append(name))
        self.assertEqual(replay.call_count, 9); self.assertEqual(len(set(written)), 27)
        self.assertEqual(set(result), set(r.NAMES))
        for name in r.NAMES: self.assertEqual(set(result[name]), {'same_close', 'next_open_zero', 'next_open_cost'})
        self.assertTrue(all(len(call.args[2]) == 122 for call in replay.call_args_list))

    def test_contracts_match_fixed_interface(self):
        self.assertEqual(r.EXTRA, ('delta_vix', 'delta_vvix'))
        self.assertEqual(r.measurement_contract()['max_abs_delta'], .01)
        self.assertEqual(r.selection_contract()['execution_paths'], 9)
        self.assertEqual(r.selection_contract()['required_complete_source_months'], 121)
        self.assertNotIn('minimum_changed_source_events', r.selection_contract())
        self.assertIn('STATE_VALUES.json', r.CACHE_FILES)

    def test_shared_import_implementation_closure_is_pinned(self):
        # Hash implementation files only; source_files never loads market arrays.
        pins = r.source_files(ROOT)
        for name in ('vvix160_kernel.py', 'test_vvix160_kernel.py', 'vvix160_runner.py',
                     'test_vvix160_runner.py', 'basic_shares159_host.py',
                     'basic_shares159_runner_draft.py', 'after158_basic_shares_kernel.py',
                     'monthly_annual_lag158_runner.py', 'monthly_annual_lag158_model.py',
                     'monthly_formation142_runner.py', 'monthly_formation142_features.py',
                     'monthly_formation142_selector.py'):
            self.assertEqual(pins[str(ROOT/'work'/name)], r.base.sha(ROOT/'work'/name))


if __name__ == '__main__':
    unittest.main()
