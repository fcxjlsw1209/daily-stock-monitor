"""Synthetic only: frozen selector integration, support and fixed accounting."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import numpy as np
import pandas as pd
import run163_runner as r

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = r.base.module('run163_synthetic_adapter', ROOT/'outputs/monthly_path70_research/incumbent_adapter.py')
EPISODE = r.base.module('run163_synthetic_episode', ROOT/'outputs/monthly_daily_episodes/episodes.py')
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
    a['delta_negative'] = np.tile(np.linspace(-.01, .01, 49), (n, 1))
    a['delta_positive'] = -a['delta_negative'].copy()
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
        a['delta_negative'][:] = .003  # Deliberately uncentered, including ETFs.
        choices, _, _ = r.choices_all(a, m, 4, b, POLICY)
        np.testing.assert_array_equal(b.calls[1]['scores'], b.pre(a, 4)+.003)
        self.assertTrue(all(0 not in row for row in choices[r.NAMES[1]]))

    def test_invalid_delta_values_and_bounds(self):
        for bad in (np.nan, np.inf, -.010001, .010001, True, 'bad'):
            a, m = fixture(); a['delta_negative'] = a['delta_negative'].astype(object); a['delta_negative'][0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.choices_all(a, m, 4, SyntheticBackend(), POLICY)

    def test_wrong_shape_axes_boolean_mask_or_policy(self):
        for fault in ('shape', 'axes', 'mask', 'duplicates', 'policy'):
            a, m = fixture(); policy = POLICY
            if fault == 'shape': a['delta_negative'] = a['delta_negative'][:, :-1]
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


def source_fixture(supported=True):
    _, meta = fixture(122)
    dates = [r.INITIAL_PARTIAL]+[str(x.date()) for x in pd.date_range('2016-08-31', periods=121, freq='BME')]
    meta.update(dates=dates, ends=dates)
    rows = []
    deltas = {k: np.zeros((122, 49)) for k in r.EXTRA}
    for i, day in enumerate(dates):
        active = bool(i and supported)
        raw = [((j % 21)-10)/10. for j in range(49)] if supported else [None]*49
        ranks = (pd.Series(raw).rank(method='average', ascending=True, pct=True)-.5).tolist() if supported else [None]*49
        rows.append(dict(date=day, symbols=meta['symbols'][i],
                         window_start=str((pd.Timestamp(day)-pd.Timedelta(days=360)).date()),
                         window_end=str((pd.Timestamp(day)-pd.Timedelta(days=60)).date()),
                         common_support=[active]*49, raw_run=raw, ranks=ranks,
                         initial_forced_neutral=i == 0))
        if active:
            deltas[r.EXTRA[0]][i] = -.02*np.asarray(ranks)
            deltas[r.EXTRA[1]][i] = .02*np.asarray(ranks)
    count = 121*49 if supported else 0
    audit = dict(events=122, symbols_per_event=49, window_returns=210, skip_sessions=42,
                 trading_eligibility_modified=False, outcomes_used=False, event_history=rows,
                 common_support_cells=count, neutral_cells=122*49-count)
    return audit, meta, deltas


class SourceSupport(unittest.TestCase):
    def test_real_kernel_interface_with_synthetic_sources(self):
        import run163_kernel as kernel
        calendar = pd.bdate_range('2013-01-01', '2027-01-31')
        dates = [r.INITIAL_PARTIAL]+[str(x.date()) for x in pd.date_range('2016-08-31', periods=121, freq='BME')]
        a, meta = fixture(122); meta.update(dates=dates, ends=dates)
        names = meta['symbols'][0]
        t = np.arange(len(calendar), dtype=float)
        prices = pd.DataFrame({name: 100.+j+.003*t+5.*np.sin(t*.017+j)
                               for j, name in enumerate(names)}, index=calendar)
        # SPY is intentionally absent: no external market dependency.
        deltas, audit = kernel.build_features(prices, {}, list(zip(dates, meta['symbols'])),
                                              [str(x.date()) for x in calendar])
        support = r.source_support(audit, meta, deltas)
        self.assertTrue(support['passed']); self.assertEqual(support['candidate_cells'], 5978)
        self.assertEqual(support['common_support_cells'], 121*49)
        self.assertTrue(all(np.all(deltas[k][0] == 0) for k in r.EXTRA))
        a.update(deltas); r.choices_all(a, meta, 122, SyntheticBackend(), POLICY)

    def test_coverage_reports_all_cells_and_fixed_eras(self):
        audit, meta, deltas = source_fixture(); result = r.source_support(audit, meta, deltas)
        self.assertTrue(result['passed']); self.assertIsNone(result['coverage_threshold'])
        self.assertEqual([result['coverage'][k]['candidate_cells'] for k in ('full122', 'first88', 'early53', 'recent35', 'later34')],
                         [5978, 4312, 2597, 1715, 1666])
        self.assertEqual([result['coverage'][k]['common_support_cells'] for k in ('full122', 'first88', 'early53', 'recent35', 'later34')],
                         [5929, 4263, 2548, 1715, 1666])

    def test_all_missing_is_neutral_and_valid_without_coverage_threshold(self):
        audit, meta, deltas = source_fixture(False); result = r.source_support(audit, meta, deltas)
        self.assertTrue(result['passed']); self.assertEqual(result['common_support_cells'], 0)
        self.assertTrue(all(x['coverage'] == 0 for x in result['coverage'].values()))

    def test_missing_neutral_and_initial_are_enforced(self):
        for bad in (.001, np.nan):
            audit, meta, deltas = source_fixture(False); deltas[r.EXTRA[0]][10, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): r.source_support(audit, meta, deltas)
        audit, meta, deltas = source_fixture(); deltas[r.EXTRA[1]][0, 0] = .001
        with self.assertRaises(ValueError): r.source_support(audit, meta, deltas)

    def test_initial_measured_values_are_preserved_but_neutral(self):
        audit, meta, deltas = source_fixture()
        self.assertTrue(all(x is not None for x in audit['event_history'][0]['raw_run']))
        self.assertTrue(r.source_support(audit, meta, deltas)['passed'])
        audit['event_history'][0]['common_support'][0] = True
        with self.assertRaises(ValueError): r.source_support(audit, meta, deltas)

    def test_percentile_ties_singleton_and_zero_rank_keep_measured_support(self):
        for raw, ranks in (([.2], [.5]), ([.1, .2], [0., .5]), ([.1, .1, .2], [0., 0., .5]), ([0.], [.5])):
            audit, meta, deltas = source_fixture(False)
            row = audit['event_history'][1]; n = len(raw)
            row['raw_run'][:n] = raw; row['ranks'][:n] = ranks; row['common_support'][:n] = [True]*n
            audit['common_support_cells'] = n; audit['neutral_cells'] -= n
            deltas[r.EXTRA[0]][1, :n] = -.02*np.asarray(ranks)
            deltas[r.EXTRA[1]][1, :n] = .02*np.asarray(ranks)
            with self.subTest(raw=raw):
                result = r.source_support(audit, meta, deltas)
                self.assertEqual(result['common_support_cells'], n)

    def test_percentile_is_not_minmax_centered_or_reranked_by_eligibility(self):
        audit, meta, deltas = source_fixture()
        row = audit['event_history'][1]
        row['ranks'] = np.linspace(-.5, .5, 49).tolist()
        with self.assertRaisesRegex(ValueError, 'percentiles'): r.source_support(audit, meta, deltas)

    def test_wrong_sign_double_dose_and_wrong_dtype_fail(self):
        for fault in ('sign', 'dose', 'dtype'):
            audit, meta, deltas = source_fixture()
            if fault == 'sign': deltas[r.EXTRA[0]] = deltas[r.EXTRA[1]].copy()
            if fault == 'dose': deltas[r.EXTRA[1]] *= .5
            if fault == 'dtype': deltas[r.EXTRA[1]] = deltas[r.EXTRA[1]].astype(np.float32)
            with self.subTest(fault=fault), self.assertRaises(ValueError): r.source_support(audit, meta, deltas)

    def test_missing_window_start_or_end_cannot_supply_measurements(self):
        for field in ('window_start', 'window_end'):
            audit, meta, deltas = source_fixture(); audit['event_history'][1][field] = None
            with self.subTest(field=field), self.assertRaises(ValueError): r.source_support(audit, meta, deltas)
        audit, meta, deltas = source_fixture(False)
        audit['event_history'][1]['window_start'] = None
        self.assertTrue(r.source_support(audit, meta, deltas)['passed'])
        audit['event_history'][1]['window_end'] = None
        self.assertTrue(r.source_support(audit, meta, deltas)['passed'])

    def test_malformed_source_axes_masks_or_aggregate_are_rejected(self):
        for fault in ('short', 'date', 'initial', 'returns', 'skip', 'future', 'window_order', 'mask',
                      'support', 'raw_nan', 'raw_bool', 'raw_outside', 'raw_nonlattice', 'rank_missing', 'rank_wrong', 'aggregate', 'outcomes', 'eligibility'):
            audit, meta, deltas = source_fixture(); row = audit['event_history'][1]
            if fault == 'short': audit['event_history'].pop()
            if fault == 'date': row['date'] = 'wrong'
            if fault == 'initial': row['initial_forced_neutral'] = True
            if fault == 'returns': audit['window_returns'] = 209
            if fault == 'skip': audit['skip_sessions'] = 41
            if fault == 'future': row['window_end'] = '2027-01-01'
            if fault == 'window_order': row['window_start'] = row['window_end']
            if fault == 'mask': row['common_support'][0] = 1
            if fault == 'support': row['common_support'][0] = False
            if fault == 'raw_nan': row['raw_run'][0] = float('nan')
            if fault == 'raw_bool': row['raw_run'][0] = True
            if fault == 'raw_outside': row['raw_run'][0] = 1.1
            if fault == 'raw_nonlattice': row['raw_run'][0] = .15
            if fault == 'rank_missing': row['ranks'][0] = None
            if fault == 'rank_wrong': row['ranks'][0] = .1
            if fault == 'aggregate': audit['common_support_cells'] -= 1
            if fault == 'outcomes': audit['outcomes_used'] = True
            if fault == 'eligibility': audit['trading_eligibility_modified'] = True
            with self.subTest(fault=fault), self.assertRaises(ValueError): r.source_support(audit, meta, deltas)


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
        self.assertEqual(r.EXTRA, ('delta_negative', 'delta_positive'))
        self.assertEqual(r.measurement_contract()['max_abs_delta'], .01)
        self.assertEqual(r.measurement_contract()['blocks'], 10)
        self.assertEqual(r.measurement_contract()['block_sessions'], 21)
        self.assertEqual(r.measurement_contract()['nuisance_estimation'], 'none')
        self.assertNotIn('market_proxy', r.measurement_contract())
        self.assertEqual(r.NAMES, ('incumbent_local67', 'run_negative', 'run_positive'))
        self.assertEqual(r.selection_contract()['execution_paths'], 9)
        self.assertIsNone(r.selection_contract()['minimum_common_support_cells'])
        self.assertNotIn('minimum_changed_source_events', r.selection_contract())
        self.assertIn('STATE_VALUES.json', r.CACHE_FILES)

    def test_shared_import_implementation_closure_is_pinned(self):
        # Hash implementation files only; source_files never loads market arrays.
        pins = r.source_files(ROOT)
        for name in ('run163_kernel.py', 'test_run163_kernel.py', 'run163_runner.py',
                     'test_run163_runner.py', 'basic_shares159_host.py',
                     'basic_shares159_runner_draft.py', 'after158_basic_shares_kernel.py',
                     'monthly_annual_lag158_runner.py', 'monthly_annual_lag158_model.py',
                     'monthly_formation142_runner.py', 'monthly_formation142_features.py',
                     'monthly_formation142_selector.py'):
            self.assertEqual(pins[str(ROOT/'work'/name)], r.base.sha(ROOT/'work'/name))


if __name__ == '__main__':
    unittest.main()
