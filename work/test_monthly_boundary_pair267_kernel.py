"""Synthetic267 sources only; no real prices, labels, arrays or replay loaded."""
import ast
import calendar
from copy import deepcopy
from datetime import date
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import monthly_boundary_pair267_kernel as k


class Poison:
    def __len__(self):
        raise AssertionError('unused numerical body accessed')
    def __getitem__(self, _):
        raise AssertionError('unused numerical body accessed')
    def __float__(self):
        raise AssertionError('unused numerical scalar accessed')
    def __array__(self, *args, **kwargs):
        raise AssertionError('unused numerical array accessed')


def month(year, number, n=6):
    signal = date(year, number, calendar.monthrange(year, number)[1]).isoformat()
    ey, em = year + (number == 12), number % 12 + 1
    end = date(ey, em, calendar.monthrange(ey, em)[1]).isoformat()
    x = np.zeros((n, 10))
    x[:, 0] = np.linspace(.45, -.45, n)
    x[:, 1] = np.linspace(-.2, .2, n)
    return dict(signal=signal, end=end, symbols=[chr(65 + i) for i in range(n)],
                pre_scores=np.array([1. - .1 * i for i in range(n)]), eligible=np.ones(n, bool),
                features=x, signal_prices=np.ones(n), end_prices=np.array([1.3 - .05 * i for i in range(n)]))


def history(n=18):
    return [month(2017 + i // 12, i % 12 + 1) for i in range(n)]


def event():
    return dict(signal='2019-01-31', symbols=['A', 'B', 'C', 'D', 'E', 'F'],
                pre_scores=np.array([.9, .8, .7, .69, .685, .5]), eligible=np.ones(6, bool),
                features=np.zeros((6, 10)), effects=np.array([-.0, 0., 0., -.0, .4, 0.]))


def ready(beta=1., year=2019):
    b = np.zeros(10); b[0] = beta
    return dict(schema=k._schema('model'), arm=k.ARMS[1], query_year=year, status='ready', common_months=18,
                beta=b, alpha=10., intercept=0., rescaled=False, reverse_augmented=False)


class DatesAndCohort(unittest.TestCase):
    def test_topk_is_original_score_then_symbol(self):
        p = np.array([.2, .4, .4, .5, .3, .1])
        chosen, n = k.original_topk(p, [True] * 6, ['Z', 'B', 'A', 'D', 'E', 'F'])
        self.assertEqual(chosen, [3, 2, 1, 4, 0]); self.assertEqual(n, 6)
    def test_original_eligibility_is_not_target_eligibility(self):
        m = month(2017, 1)
        m['eligible'][0] = False
        m['pre_scores'][0] = np.nan
        result = k.build_cohort([m], 2019)
        self.assertEqual(result['months'][0]['selected_indices'], [1, 2, 3, 4, 5])
    def test_nonfinite_eligible_P_hard_fails(self):
        m = month(2017, 1); m['pre_scores'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Eligible original P'):
            k.build_cohort([m], 2019)
    def test_top_five_selected_before_bad_label_no_fill(self):
        m = month(2017, 1); m['end_prices'][0] = np.nan
        c = k.build_cohort([m], 2019)
        self.assertEqual(c['common_months'], 0)
        self.assertEqual(c['months'][0]['selected_indices'], [0, 1, 2, 3, 4])
        for arm in k.ARMS: self.assertEqual(len(c['arms'][arm]['y']), 0)
    def test_invalid_rank_five_excludes_candidate_rank34_too(self):
        m = month(2017, 1); m['features'][4, 8] = np.inf
        c = k.build_cohort([m], 2019)
        self.assertEqual(c['common_months'], 0)
        self.assertEqual(c['months'][0]['invalid_selected_indices'], [4])
    def test_unused_sixth_name_bodies_not_read(self):
        m = month(2017, 1)
        for key in ('features', 'signal_prices', 'end_prices'):
            m[key] = list(m[key]); m[key][5] = Poison()
        self.assertEqual(k.build_cohort([m], 2019)['common_months'], 1)
    def test_no_training_gap_filter(self):
        m = month(2017, 1)
        self.assertGreater(m['pre_scores'][2] - m['pre_scores'][3], .02)
        c = k.build_cohort([m], 2019)
        self.assertEqual(c['common_months'], 1); self.assertFalse(c['training_gap_filter'])
    def test_four_names_six_control_pairs(self):
        c = k.build_cohort([month(2017, 1, 4)], 2019)
        self.assertEqual(len(c['arms'][k.ARMS[0]]['y']), 6)
        self.assertEqual(len(c['arms'][k.ARMS[1]]['y']), 1)
    def test_less_than_four_no_features_or_prices(self):
        m = month(2017, 1, 3)
        for key in ('features', 'signal_prices', 'end_prices'): m[key] = Poison()
        self.assertEqual(k.build_cohort([m], 2019)['common_months'], 0)
    def test_dates_before_future_old_and_partial_bodies(self):
        rows = [dict(signal='2013-12-31', end='2014-01-31'),
                dict(signal='2016-08-17', end='2016-08-31'),
                dict(signal='2018-12-31', end='2019-01-31'),
                dict(signal='2025-01-31', end='2025-02-28')]
        for row in rows:
            for key in k.BODY_KEYS: row[key] = Poison()
        c = k.build_cohort(rows, 2019)
        self.assertEqual([x['status'] for x in c['months']], ['outside_window', 'initial_partial', 'not_mature', 'not_mature'])
    def test_cutoff_inclusive_lower_exclusive(self):
        rows = [month(2013, 12), month(2014, 1), month(2018, 11), month(2018, 12)]
        c = k.build_cohort(rows, 2019)
        self.assertEqual(c['common_source_indices'], [1, 2])
        self.assertEqual(c['decay_ages'], [59, 1])
    def test_duplicate_month_and_wrong_next_month_rejected(self):
        a = month(2017, 1); b = deepcopy(a); b['signal'] = '2017-01-30'
        with self.assertRaises(ValueError): k.build_cohort([b, a], 2019)
        a['end'] = '2017-03-01'
        with self.assertRaisesRegex(ValueError, 'Next natural-month'): k.build_cohort([a], 2019)
    def test_unsorted_dates_and_bad_year_rejected(self):
        with self.assertRaises(ValueError): k.build_cohort(list(reversed(history(2))), 2019)
        for year in (True, 2019., 0):
            with self.assertRaises(ValueError): k.build_cohort([], year)
    def test_identity_and_E_types_rejected(self):
        m = month(2017, 1); m['symbols'][1] = m['symbols'][0]
        with self.assertRaises(ValueError): k.build_cohort([m], 2019)
        m = month(2017, 1); m['eligible'] = [1] * 6
        with self.assertRaises(ValueError): k.build_cohort([m], 2019)
    def test_positive_price_and_rank_range_validity(self):
        for field, value in [('signal_prices', 0.), ('end_prices', -1.)]:
            m = month(2017, 1); m[field][2] = value
            self.assertEqual(k.build_cohort([m], 2019)['common_months'], 0)
        m = month(2017, 1); m['features'][2, 0] = math.nextafter(.5, math.inf)
        self.assertEqual(k.build_cohort([m], 2019)['common_months'], 0)
    def test_label_exact_operation_order_not_gross_difference(self):
        m = month(2017, 1); m['end_prices'][2:4] = [.1, .2]
        y = k.build_cohort([m], 2019)['arms'][k.ARMS[1]]['y'][0]
        self.assertEqual(y.hex(), ((.1 - 1.) - (.2 - 1.)).hex())
        self.assertNotEqual(y, .1 - .2)
    def test_raw_label_not_clipped_excess_or_basket_third(self):
        m = month(2017, 1); m['end_prices'][2:4] = [4., 1.]
        c = k.build_cohort([m], 2019)
        self.assertEqual(c['arms'][k.ARMS[1]]['y'][0], 3.)
        np.testing.assert_array_equal(c['arms'][k.ARMS[1]]['X'][0], m['features'][2] - m['features'][3])
    def test_gross_overflow_and_underflow_fail(self):
        for start, stop in [(1e-308, 1e308), (1e308, 1e-308)]:
            m = month(2017, 1); m['signal_prices'][2] = start; m['end_prices'][2] = stop
            with self.assertRaisesRegex(ValueError, 'adjusted gross'): k.build_cohort([m], 2019)
    def test_source_arrays_not_modified(self):
        rows = history(2); before = deepcopy(rows); k.build_cohort(rows, 2019)
        for a, b in zip(rows, before):
            for key in ('features', 'eligible', 'pre_scores', 'signal_prices', 'end_prices'):
                self.assertEqual(a[key].tobytes(), b[key].tobytes())


class WeightsAndFit(unittest.TestCase):
    def test_one_month_control_total_one_not_ten(self):
        c = k.build_cohort([month(2017, 1)], 2019)
        for arm in k.ARMS:
            self.assertEqual(c['arms'][arm]['weight_mass'], 1.)
        np.testing.assert_array_equal(c['arms'][k.ARMS[0]]['weights'], [.1] * 10)
    def test_canonical_decay_and_month_unit_sharing(self):
        c = k.build_cohort(history(), 2019)
        expected = [(18 * float(np.exp2(-a / 12.))) / math.fsum(c['decays']) for a in c['decay_ages']]
        self.assertEqual([x.hex() for x in expected], c['month_weights_hex'])
        for m, wm in enumerate(expected):
            for arm, count in [(k.ARMS[0], 10), (k.ARMS[1], 1)]:
                ids = [i for i, row in enumerate(c['arms'][arm]['pairs']) if row['common_month_index'] == m]
                self.assertEqual(len(ids), count)
                self.assertTrue(all(c['arms'][arm]['weights'][i] == wm / count for i in ids))
        for arm in k.ARMS:
            self.assertEqual(c['arms'][arm]['weight_mass_hex'], math.fsum(c['arms'][arm]['weights']).hex())
            self.assertAlmostEqual(c['arms'][arm]['weight_mass'], 18., places=13)
    def test_eighteen_common_months_required_for_both(self):
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('cold fit')):
            cold = k.fit_models(history(17), 2019)
        self.assertEqual(cold['actual_solve_calls'], 0)
        rows = history(); rows[0]['end_prices'][4] = np.nan
        self.assertEqual(k.fit_models(rows, 2019)['actual_solve_calls'], 0)
        fitted = k.fit_models(history(), 2019)
        self.assertEqual(fitted['actual_solve_calls'], 2)
        self.assertTrue(all(m['status'] == 'ready' for m in fitted['models'].values()))
    def test_no_intercept_closed_form(self):
        rows = history()
        for m in rows:
            m['features'][:] = 0.; m['features'][2, 0] = .2; m['features'][3, 0] = -.2
            m['end_prices'][2] = 1.2; m['end_prices'][3] = 1.
        out = k.fit_models(rows, 2019)
        model = out['models'][k.ARMS[1]]
        mass = out['cohort']['arms'][k.ARMS[1]]['weight_mass']
        want = mass * .4 * (1.2 - 1.) / (10. + mass * .4 * .4)
        self.assertAlmostEqual(model['beta'][0], want, places=15)
        self.assertTrue(np.all(model['beta'][1:] == 0.))
        self.assertEqual(model['intercept'], 0.)
        self.assertFalse(model['rescaled']); self.assertFalse(model['reverse_augmented'])
    def test_identical_features_nonzero_labels_fit_zero_prediction(self):
        rows = history()
        for m in rows: m['features'][:] = .1
        out = k.fit_models(rows, 2019)
        for model in out['models'].values(): np.testing.assert_array_equal(model['beta'], np.zeros(10))
    def test_pair_loss_set_is_only_model_difference(self):
        out = k.fit_models(history(), 2019)
        for key in ('query_year', 'common_months', 'alpha', 'intercept', 'rescaled', 'reverse_augmented', 'status'):
            self.assertEqual(out['models'][k.ARMS[0]][key], out['models'][k.ARMS[1]][key])
        self.assertEqual([r['original_ranks'] for r in out['cohort']['arms'][k.ARMS[0]]['pairs'][:10]],
                         [[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]])
        self.assertEqual(out['cohort']['arms'][k.ARMS[1]]['pairs'][0]['original_ranks'], [3,4])
    def test_fit_evidence_reconstructs_exact_arrays(self):
        out = k.fit_models(history(), 2019)
        for model in out['models'].values():
            for key in ('beta', 'gram', 'rhs'):
                p = model['float_evidence'][key]
                a = np.array([float.fromhex(v) for v in p['values']], dtype=np.dtype(p['dtype'])).reshape(p['shape'])
                self.assertEqual(a.tobytes(), model[key].tobytes())
    def test_no_solver_fallback(self):
        with patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('synthetic solver failure')):
            with self.assertRaises(np.linalg.LinAlgError): k.fit_models(history(), 2019)
    def test_zero_months_empty_shape_and_no_fits(self):
        out = k.fit_models([], 2019)
        self.assertEqual(out['actual_solve_calls'], 0)
        for arm in k.ARMS:
            self.assertEqual(out['cohort']['arms'][arm]['X'].shape, (0,10))
            self.assertEqual(out['cohort']['arms'][arm]['weight_mass'], 0.)
    def test_future_body_poison_does_not_change_fit_bits(self):
        a = k.fit_models(history(), 2019)
        rows = history() + [dict(signal='2020-01-31', end='2020-02-29', **{key:Poison() for key in k.BODY_KEYS})]
        b = k.fit_models(rows, 2019)
        for arm in k.ARMS:
            for key in ('beta', 'gram', 'rhs'):
                self.assertEqual(a['models'][arm][key].tobytes(), b['models'][arm][key].tobytes())


class Deployment(unittest.TestCase):
    def test_relative_prediction_sign_and_only_pair_effects(self):
        e = event(); e['features'][2,0] = .2; e['features'][3,0] = -.2
        a, audit = k.adjust_effects(e, ready())
        self.assertGreater(a[2], 0.); self.assertLess(a[3], 0.)
        self.assertEqual(audit['pair_indices'], [2,3]); self.assertEqual(audit['changed_indices'], [2,3])
        self.assertEqual(a[[0,1,4,5]].tobytes(), e['effects'][[0,1,4,5]].tobytes())
        b, _ = k.adjust_effects(e, ready(-1.))
        self.assertLess(b[2], 0.); self.assertGreater(b[3], 0.)
    def test_zero_preserves_signed_zero_exactly(self):
        e = event(); result, a = k.adjust_effects(e, ready(0.))
        self.assertEqual(result.tobytes(), e['effects'].tobytes())
        self.assertEqual(a['status'], 'zero_prediction')
    def test_clipping_and_original_effect_addition(self):
        e = event(); e['features'][2,0] = .5; e['features'][3,0] = -.5
        e['effects'][2:4] = [.4,-.4]
        result, a = k.adjust_effects(e, ready())
        self.assertEqual(result[2], .5); self.assertEqual(result[3], -.5)
        self.assertEqual(a['q'], .5 * math.tanh(1. / .05))
    def test_partial_and_cold_never_read_numeric_query_body(self):
        for partial in (False, True):
            e = event(); model = ready(year=2016 if partial else 2019)
            if partial: e['signal'] = '2016-08-17'
            else: model['status'] = 'cold'; model['common_months'] = 17
            for key in ('symbols','pre_scores','eligible','features'): e[key] = Poison()
            result, _ = k.adjust_effects(e, model)
            self.assertEqual(result.tobytes(), e['effects'].tobytes())
    def test_gap_outside_does_not_read_features(self):
        e = event(); e['pre_scores'][3] = .67; e['pre_scores'][4] = .66; e['features'] = Poison()
        result, a = k.adjust_effects(e, ready())
        self.assertEqual(a['status'], 'outside_deployment_gap')
        self.assertEqual(result.tobytes(), e['effects'].tobytes())
    def test_gap_equal_threshold_included(self):
        e = event(); e['pre_scores'][2:] = [.02, 0., -.01, -.02]
        e['features'][2,0] = .2
        result, a = k.adjust_effects(e, ready())
        self.assertEqual(a['status'], 'adjusted'); self.assertGreater(result[2], 0.)
    def test_invalid_required_pair_features_fail(self):
        e = event(); e['features'][3,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid required boundary'):
            k.adjust_effects(e, ready())
    def test_wrong_query_year_and_nonfinite_model_rejected(self):
        with self.assertRaises(ValueError): k.adjust_effects(event(), ready(year=2020))
        model = ready(); model['beta'][0] = np.inf
        with self.assertRaises(ValueError): k.adjust_effects(event(), model)
    def test_unbacked_ready_status_rejected(self):
        model = ready(); model['common_months'] = 17
        with self.assertRaisesRegex(ValueError, 'eighteen-month'):
            k.adjust_effects(event(), model)
    def test_no_pair_for_fewer_than_four(self):
        e = event(); e['eligible'][3:] = False; e['features'] = Poison()
        a, proof = k.adjust_effects(e, ready())
        self.assertEqual(proof['status'], 'fewer_than_four_eligible')
        self.assertEqual(a.tobytes(), e['effects'].tobytes())
    def test_selector_called_once_fixed_policy_and_copies(self):
        calls = []
        p = np.array([[.9,.8,.7,.69,.68]])
        eligible = np.ones_like(p, bool); effects = np.zeros_like(p); source = {'sentinel': np.array([1.])}
        def selector(a,b,c,d,s,policy):
            calls.append(policy); a[0,0] = 5.; b[0,0] = False; d[:] = 1.; s['sentinel'][:] = 3.; c[0][0] = 'changed'
            return [[0,1,2]], a
        symbols = [['A','B','C','D','E']]
        _, _, audit = k.select_once(selector, p, eligible, symbols, effects, source)
        self.assertEqual(calls, [dict(budget=.05,ceiling=5,gap=.02)])
        self.assertEqual(audit['original_selector_calls'], 1)
        self.assertEqual(p[0,0], .9); self.assertTrue(eligible[0,0]); self.assertTrue((effects==0.).all())
        self.assertEqual(source['sentinel'][0], 1.); self.assertEqual(symbols[0][0], 'A')
    def test_actual_original_selector_source_allows_P5_win(self):
        # Read code only, extract one pure function; never import its loaders.
        path = Path(__file__).resolve().parents[1] / 'outputs/monthly_daily_episodes/episodes.py'
        source_tree = ast.parse(path.read_text())
        function = next(x for x in source_tree.body if isinstance(x,ast.FunctionDef) and x.name=='select_boundary')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'), namespace)
        e = event(); e['features'][2,0] = -.1; e['features'][3,0] = .1
        e['effects'][4] = .5
        effects, _ = k.adjust_effects(e, ready(.005))
        p = e['pre_scores'][None,:]; E = e['eligible'][None,:]
        src = dict(baseline=np.zeros_like(p),core=np.zeros_like(p),core_weight=np.full_like(p,.52),
                   eligible=E,state_calm10=np.zeros(1,bool),state_weak_narrow=np.zeros(1,bool))
        choices, _, proof = k.select_once(namespace['select_boundary'],p,E,[e['symbols']],effects[None,:],src)
        self.assertEqual(set(choices[0]), {0,1,4})
        self.assertFalse(proof['forced_pair_winner'])
    def test_selector_identity_preserves_ineligible_nan_payload_and_zero(self):
        p = np.array([[.9,.8,.7,-.0,0.]])
        p.view(np.uint64)[0,4] = 0x7ff8000000000042
        use = np.array([[True,True,True,True,False]])
        calls=[]
        def selector(a,b,c,d,s,policy): calls.append(1); return [[0,1,2]],a
        _, final, _=k.select_once(selector,p,use,[['A','B','C','D','E']],np.zeros_like(p),{})
        self.assertEqual(calls,[1]);self.assertEqual(final.tobytes(),p.tobytes())
    def test_empty_prefix_still_one_callback(self):
        calls=[]
        def selector(a,b,c,d,s,policy): calls.append(1); return [], a
        choices, final, audit=k.select_once(selector,np.zeros((0,5)),np.zeros((0,5),bool),[],np.zeros((0,5)),{})
        self.assertEqual(calls,[1]); self.assertEqual(choices,[]); self.assertEqual(final.shape,(0,5))
        self.assertEqual(audit['original_selector_calls'],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
