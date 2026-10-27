"""Caller-only synthetic verifier tests; never import the 267 producer.

The supplied-model fixtures use explicit normal equations, while the verifier
uses augmented SVD. The frozen selector's single pure function is extracted
from source for policy parity; no source loader or actual dataset is executed.
"""
import ast
import calendar
from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import monthly_boundary_pair267_independent as ind


class Poison:
    def __float__(self):
        raise AssertionError('Future numeric body consumed')

    def __len__(self):
        raise AssertionError('Future numeric body sized')

    def __iter__(self):
        raise AssertionError('Future numeric body iterated')

    def __getitem__(self, key):
        raise AssertionError('Future numeric body indexed')

    def __array__(self, *args, **kwargs):
        raise AssertionError('Future numeric body converted')


def pack(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), values=[float(v).hex() for v in a.flat])


def month(year=2018, number=1, width=6, eligible_count=5, ordinal=0):
    ny, nm = (year + 1, 1) if number == 12 else (year, number + 1)
    signal = '%04d-%02d-%02d' % (year, number, calendar.monthrange(year, number)[1])
    end = '%04d-%02d-%02d' % (ny, nm, calendar.monthrange(ny, nm)[1])
    features = np.array([[((j * 7 + k * 3 + ordinal * (k + 1)) % 19 - 9) / 24.
                          for k in range(10)] for j in range(width)], dtype=np.float64)
    return dict(signal=signal, end=end, symbols=['S%02d' % j for j in range(width)],
                pre_scores=np.linspace(.4, -.2, width),
                eligible=np.array([j < eligible_count for j in range(width)], dtype=bool),
                features=features, signal_prices=np.full(width, 100.),
                end_prices=np.array([100. + (j - 2) * .7 + ((ordinal + j) % 5) * .1 for j in range(width)]))


def months(count=18):
    return [month(2018 + i // 12, i % 12 + 1, ordinal=i) for i in range(count)]


def bundle(rows=None, year=2020):
    rows = months() if rows is None else rows
    cohort, _ = ind.reference_cohort(rows, year)
    models = {}
    for arm in ind.ARMS:
        a = cohort['arms'][arm]
        if cohort['status'] == 'ready':
            gram = np.einsum('ni,n,nj->ij', a['X'], a['weights'], a['X'], optimize=False) + 10. * np.eye(10)
            rhs = np.einsum('ni,n,n->i', a['X'], a['weights'], a['y'], optimize=False)
            beta = np.linalg.solve(gram, rhs)
            fits = 1
        else:
            beta, gram, rhs, fits = np.zeros(10), None, None, 0
        models[arm] = dict(schema='monthly_boundary_pair267_model_v1', arm=arm, query_year=year,
                           status=cohort['status'], common_months=cohort['common_months'],
                           alpha=10., intercept=0., rescaled=False, reverse_augmented=False,
                           beta=beta, gram=gram, rhs=rhs,
                           float_evidence={k: None if a is None else pack(a)
                                           for k, a in [('beta', beta), ('gram', gram), ('rhs', rhs)]},
                           rows=len(a['y']), actual_solve_calls=fits)
    return dict(schema='monthly_boundary_pair267_models_v1', query_year=year, cohort=cohort, models=models,
                actual_solve_calls=sum(m['actual_solve_calls'] for m in models.values()),
                pair_loss_set_only_difference=True, actual_coverage_claim=False)


def event():
    x = np.zeros((6, 10)); x[2, 0] = .2; x[3, 0] = -.1
    return dict(signal='2020-01-31', symbols=['A', 'B', 'C', 'D', 'E', 'F'],
                pre_scores=np.array([.4, .3, .1, .09, .085, -.2]), eligible=np.ones(6, dtype=bool),
                features=x, effects=np.array([0., -0., .02, -.03, .49, 0.]))


def header_model(status='ready', year=2020, beta=None):
    return dict(schema='monthly_boundary_pair267_model_v1', arm=ind.ARMS[1], query_year=year,
                status=status, common_months=18 if status == 'ready' else 17,
                alpha=10., intercept=0., rescaled=False, reverse_augmented=False,
                beta=np.zeros(10) if beta is None else beta)


def audit(ev, model, status=None):
    """Expected synthetic audit, fixed fixture identities; not a selector."""
    result = ev['effects'].copy()
    status = status or ('initial_partial' if ev['signal'] == ind.INITIAL_PARTIAL else
                        'cold' if model['status'] == 'cold' else 'active')
    out = dict(signal=ev['signal'], arm=model['arm'], status=status,
               pair_indices=[], predicted_return_gap=None, q=0., changed_indices=[],
               original_selector_calls=0, original_pre_scores_unchanged=True,
               new_global_score_ranks_computed=False, winner_selected=False)
    if status in ('initial_partial', 'cold', 'fewer_than_four_eligible'):
        return result, out
    out.update(pair_indices=[2, 3], original_gap=float(ev['pre_scores'][2]) - float(ev['pre_scores'][3]))
    if status == 'outside_deployment_gap':
        return result, out
    value = float(np.dot(ev['features'][2] - ev['features'][3], model['beta']))
    q = .5 * math.tanh(value / .05)
    out.update(predicted_return_gap=value, predicted_return_gap_hex=value.hex(), q=q, q_hex=q.hex(),
               status='zero_prediction' if q == 0. else 'adjusted')
    if q != 0.:
        result[2] = min(.5, max(-.5, float(result[2]) + q))
        result[3] = min(.5, max(-.5, float(result[3]) - q))
        out['changed_indices'] = [j for j in (2, 3) if result[j].tobytes() != ev['effects'][j].tobytes()]
    return result, out


def original_selector():
    path = Path(__file__).resolve().parents[1] / 'outputs/monthly_daily_episodes/episodes.py'
    parsed = ast.parse(path.read_text())
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'select_boundary']
    assert len(nodes) == 1
    scope = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['select_boundary']


def selection_inputs(n=1, width=6):
    p = np.tile(np.array([.4, .3, .1, .09, .085, -.2])[:width], (n, 1))
    e = np.ones((n, width), dtype=bool)
    names = [[chr(65 + j) for j in range(width)] for _ in range(n)]
    effects = np.zeros((n, width)); effects[:, 4:5] = .49
    source = dict(baseline=np.zeros((n, width)), core=np.zeros((n, width)), core_weight=np.full((n, 1), .52),
                  state_calm10=np.zeros(n), state_weak_narrow=np.zeros(n))
    return p, e, names, effects, source


class CohortTests(unittest.TestCase):
    def test_single_month_known_rows_signs_and_no_gap_filter(self):
        row = month()
        row['pre_scores'] = np.array([.5, .4, .3, -.1, -.2, -.3])
        c, work = ind.reference_cohort([row], 2020)
        self.assertEqual(c['common_months'], 1)
        self.assertEqual(c['month_weights'], [1.])
        control, candidate = [c['arms'][a] for a in ind.ARMS]
        self.assertEqual(control['X'].shape, (10, 10)); self.assertEqual(candidate['X'].shape, (1, 10))
        self.assertEqual(candidate['pairs'][0]['indices'], [2, 3])
        self.assertEqual(candidate['pairs'][0]['original_ranks'], [3, 4])
        expected = (float(row['end_prices'][2] / 100.) - 1.) - (float(row['end_prices'][3] / 100.) - 1.)
        self.assertEqual(candidate['y'][0].item().hex(), expected.hex())
        np.testing.assert_array_equal(candidate['X'][0], row['features'][2] - row['features'][3])
        self.assertEqual(work['endpoint_return_evaluations'], 5)
        self.assertEqual(work['control_pair_rows'], 10); self.assertEqual(work['candidate_pair_rows'], 1)

    def test_zero_and_seventeen_are_cold_eighteen_ready(self):
        for n, status in [(0, 'cold'), (17, 'cold'), (18, 'ready')]:
            c, _ = ind.reference_cohort(months(n), 2020)
            self.assertEqual(c['status'], status)
            self.assertEqual(c['common_months'], n)

    def test_ignored_old_future_and_partial_bodies_are_not_read(self):
        records = [dict(signal='2014-12-31', end='2015-01-30'),
                   dict(signal=ind.INITIAL_PARTIAL, end='2016-08-31'),
                   month(), dict(signal='2019-12-31', end='2020-01-31')]
        for i in (0, 1, 3):
            records[i].update({k: Poison() for k in ('symbols', 'pre_scores', 'eligible', 'features', 'signal_prices', 'end_prices')})
        c, work = ind.reference_cohort(records, 2020)
        self.assertEqual([r['status'] for r in c['months']], ['outside_window', 'initial_partial', 'common_valid_month', 'not_mature'])
        self.assertEqual(work['mature_month_bodies'], 1)

    def test_cutoff_inclusive_lower_exclusive_and_january(self):
        rows = [month(2014, 12), month(2015, 1), month(2019, 11), month(2019, 12)]
        c, _ = ind.reference_cohort(rows, 2020)
        self.assertEqual(c['common_source_indices'], [1, 2])
        self.assertEqual(c['decay_ages'], [59, 1])

    def test_all_headers_validated_before_any_numeric_body(self):
        early = dict(signal='2018-01-31', end='2018-02-28', symbols=Poison())
        with self.assertRaisesRegex(ValueError, 'Next natural-month'):
            ind.reference_cohort([early, dict(signal='2018-02-28', end='2018-04-30')], 2020)

    def test_strict_date_native_year_and_order(self):
        for bad in (True, np.int64(2020), 6, 10000):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ind.reference_cohort([], bad)
        for records in ([month(), month()], [month(2018, 2), month()],
                        [dict(signal='2018-1-31', end='2018-02-28')],
                        [dict(signal='2018-02-30', end='2018-03-31')]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                ind.reference_cohort(records, 2020)

    def test_topk_first_bad_fifth_drops_whole_month_no_rank6_fill(self):
        row = month(eligible_count=6)
        row['features'][4, 0] = np.nan
        c, work = ind.reference_cohort([row], 2020)
        self.assertEqual(c['common_months'], 0)
        self.assertEqual(c['months'][0]['invalid_selected_indices'], [4])
        self.assertEqual(work['endpoint_return_evaluations'], 0)
        for arm in ind.ARMS:
            self.assertEqual(c['arms'][arm]['X'].shape, (0, 10))

    def test_unused_sixth_feature_and_prices_poison_safe(self):
        row = month(eligible_count=6)
        for key in ('features', 'signal_prices', 'end_prices'):
            row[key] = list(row[key]); row[key][5] = Poison()
        c, _ = ind.reference_cohort([row], 2020)
        self.assertEqual(c['common_months'], 1)

    def test_fewer_than_four_never_reads_labels_or_features(self):
        row = month(eligible_count=3)
        row.update(features=Poison(), signal_prices=Poison(), end_prices=Poison())
        c, _ = ind.reference_cohort([row], 2020)
        self.assertEqual(c['months'][0]['status'], 'fewer_than_four_eligible')

    def test_original_symbol_ties_and_original_E(self):
        row = month(eligible_count=6)
        row['pre_scores'] = np.zeros(6)
        row['symbols'] = ['F', 'E', 'D', 'C', 'B', 'A']
        row['eligible'][5] = False
        c, _ = ind.reference_cohort([row], 2020)
        self.assertEqual(c['months'][0]['selected_indices'], [4, 3, 2, 1, 0])
        self.assertEqual(c['arms'][ind.ARMS[1]]['pairs'][0]['indices'], [2, 1])

    def test_invalid_selected_values_common_exclusion(self):
        for key, bad in [('signal_prices', 0.), ('end_prices', -1.), ('signal_prices', True),
                         ('end_prices', np.inf), ('end_prices', complex(1, 0))]:
            row = month(); row[key] = list(row[key]); row[key][2] = bad
            with self.subTest(key=key, bad=bad):
                self.assertEqual(ind.reference_cohort([row], 2020)[0]['common_months'], 0)
        for bad in ([0.] * 9, [True] * 10, [np.nan] * 10, [0.5000001] * 10, np.zeros((1, 10)), np.array(0.)):
            row = month(); row['features'] = list(row['features']); row['features'][2] = bad
            with self.subTest(bad=bad):
                self.assertEqual(ind.reference_cohort([row], 2020)[0]['common_months'], 0)

    def test_bad_original_E_P_and_identity_are_errors(self):
        cases = [('eligible', [True, True, 1, True, True, False]),
                 ('pre_scores', [1., .8, np.nan, .4, .2, 0.]),
                 ('pre_scores', [1., .8, True, .4, .2, 0.]),
                 ('symbols', ['A'] * 6)]
        for key, bad in cases:
            row = month(); row[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                ind.reference_cohort([row], 2020)

    def test_underflow_and_overflow_gross_fail_not_drop(self):
        for start, stop in [(np.finfo(float).max, math.ulp(0.)), (math.ulp(0.), np.finfo(float).max)]:
            row = month(); row['signal_prices'][2] = start; row['end_prices'][2] = stop
            with self.assertRaisesRegex(ValueError, 'gross'):
                ind.reference_cohort([row], 2020)

    def test_two_return_subtractions_not_cancelled(self):
        row = month(); row['signal_prices'][:] = 1.
        row['end_prices'][2] = .1; row['end_prices'][3] = .2
        c, _ = ind.reference_cohort([row], 2020)
        got = c['arms'][ind.ARMS[1]]['y'][0].item()
        self.assertEqual(got.hex(), ((.1 - 1.) - (.2 - 1.)).hex())
        self.assertNotEqual(got.hex(), (.1 - .2).hex())

    def test_mixed_top4_top5_common_month_mass_and_canonical_weights(self):
        rows = [month(), month(2018, 2, eligible_count=4)]
        c, work = ind.reference_cohort(rows, 2020)
        ds = [float(np.exp2(-23 / 12.)), float(np.exp2(-22 / 12.))]
        wm = [2 * d / math.fsum(ds) for d in ds]
        self.assertEqual(c['month_weights_hex'], [v.hex() for v in wm])
        control = c['arms'][ind.ARMS[0]]; candidate = c['arms'][ind.ARMS[1]]
        self.assertEqual(control['weights'].tolist(), [wm[0] / 10] * 10 + [wm[1] / 6] * 6)
        self.assertEqual(candidate['weights'].tolist(), wm)
        self.assertEqual(control['weight_mass_hex'], math.fsum(control['weights']).hex())
        self.assertEqual(work['control_pair_rows'], 16)


class ModelTests(unittest.TestCase):
    def test_ready_models_two_independent_svd_solves(self):
        rows = months(); supplied = bundle(rows)
        receipt = ind.verify_models(rows, 2020, supplied)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['work']['independent_svd_solves'], 2)
        self.assertEqual(receipt['work']['independent_equation_checks'], 2)
        self.assertEqual(receipt['work']['endpoint_return_evaluations'], 90)
        self.assertTrue(all(receipt[k] for k in ind.MODEL_FLAGS))
        self.assertEqual(set(receipt['svd_beta']), set(ind.ARMS))

    def test_cold_does_not_call_svd_or_equations(self):
        rows = months(17); supplied = bundle(rows)
        with mock.patch.object(ind.np.linalg, 'svd', side_effect=AssertionError('No cold solve')):
            receipt = ind.verify_models(rows, 2020, supplied)
        self.assertEqual(receipt['work']['independent_svd_solves'], 0)
        self.assertEqual(receipt['svd_beta'], dict.fromkeys(ind.ARMS))

    def test_known_scalar_ridge_and_collinear_solution(self):
        X = np.zeros((4, 10)); X[:, 0] = [1., 2., 3., 4.]
        y = np.array([2., 4., 6., 8.]); weights = np.ones(4)
        answer = ind._svd(X, y, weights)
        self.assertAlmostEqual(answer[0], 60. / 40., places=14)
        np.testing.assert_allclose(answer[1:], 0., atol=1e-15)
        X[:, 1] = X[:, 0]
        answer = ind._svd(X, y, weights)
        np.testing.assert_allclose(answer[:2], [60. / 70.] * 2, atol=1e-14, rtol=0.)

    def test_cohort_bits_weights_and_pair_identity_tamper_fail(self):
        for what in ('X', 'y', 'weights', 'pair', 'decay'):
            rows = months(); supplied = bundle(rows); c = supplied['cohort']; a = c['arms'][ind.ARMS[0]]
            if what in ('X', 'y', 'weights'):
                a[what].flat[0] = np.nextafter(a[what].flat[0], np.inf)
            elif what == 'pair':
                a['pairs'][0]['indices'].reverse()
            else:
                c['month_weights'][0] = np.nextafter(c['month_weights'][0], np.inf).item()
            with self.subTest(what=what), self.assertRaisesRegex(ValueError, 'Exact'):
                ind.verify_models(rows, 2020, supplied)

    def test_gram_and_rhs_one_ulp_exact_not_tolerance(self):
        for key in ('gram', 'rhs'):
            rows = months(); supplied = bundle(rows); model = supplied['models'][ind.ARMS[0]]
            model[key].flat[0] = np.nextafter(model[key].flat[0], np.inf)
            model['float_evidence'][key] = pack(model[key])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Exact array bytes'):
                ind.verify_models(rows, 2020, supplied)

    def test_beta_continuous_tolerance_not_unbounded(self):
        rows = months(); supplied = bundle(rows); m = supplied['models'][ind.ARMS[0]]
        m['beta'][0] += 1e-11; m['float_evidence']['beta'] = pack(m['beta'])
        self.assertTrue(ind.verify_models(rows, 2020, supplied)['passed'])
        m['beta'][0] += 1e-5; m['float_evidence']['beta'] = pack(m['beta'])
        with self.assertRaisesRegex(ValueError, 'SVD tolerance'):
            ind.verify_models(rows, 2020, supplied)

    def test_parameters_roles_counts_dtype_and_saved_evidence(self):
        for key, value in [('alpha', 1.), ('intercept', 1.), ('rescaled', True), ('reverse_augmented', True),
                           ('arm', ind.ARMS[1]), ('actual_solve_calls', 0), ('common_months', True)]:
            rows = months(); supplied = bundle(rows); supplied['models'][ind.ARMS[0]][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ind.verify_models(rows, 2020, supplied)
        rows = months(); supplied = bundle(rows)
        supplied['models'][ind.ARMS[0]]['beta'] = supplied['models'][ind.ARMS[0]]['beta'].astype(np.float32)
        with self.assertRaises(ValueError):
            ind.verify_models(rows, 2020, supplied)
        supplied = bundle(rows); supplied['actual_solve_calls'] = 1
        with self.assertRaises(ValueError):
            ind.verify_models(rows, 2020, supplied)
        supplied = bundle(rows); supplied['models'][ind.ARMS[0]]['float_evidence']['beta']['values'][0] = '0x0.0p+0'
        with self.assertRaises(ValueError):
            ind.verify_models(rows, 2020, supplied)

    def test_cold_negative_zero_beta_rejected(self):
        rows = months(1); supplied = bundle(rows)
        supplied['models'][ind.ARMS[0]]['beta'][0] = -0.
        with self.assertRaisesRegex(ValueError, 'cold beta'):
            ind.verify_models(rows, 2020, supplied)

    def test_solver_failure_has_no_fallback(self):
        rows = months(); supplied = bundle(rows)
        with mock.patch.object(ind.np.linalg, 'svd', side_effect=np.linalg.LinAlgError('Synthetic SVD failure')):
            with self.assertRaises(np.linalg.LinAlgError):
                ind.verify_models(rows, 2020, supplied)

    def test_readonly_inputs_and_model_binding_mutation(self):
        rows = months(); supplied = bundle(rows)
        for model in supplied['models'].values():
            for key in ('beta', 'gram', 'rhs'):
                model[key].flags.writeable = False
        receipt = ind.verify_models(rows, 2020, supplied)
        model = deepcopy(supplied['models'][ind.ARMS[0]])
        before = receipt['model_sha256'][ind.ARMS[0]]
        self.assertEqual(before, ind.model_fingerprint(model))
        model['beta'][0] = np.nextafter(model['beta'][0], np.inf)
        self.assertNotEqual(before, ind.model_fingerprint(model))


class AdjustmentTests(unittest.TestCase):
    def test_positive_negative_sign_and_untouched_effects(self):
        for coefficient in (.05, -.05):
            ev = event(); beta = np.zeros(10); beta[0] = coefficient; model = header_model(beta=beta)
            effects, expected = audit(ev, model)
            receipt = ind.verify_adjustment(ev, model, effects, expected)
            self.assertTrue(receipt['passed']); self.assertFalse(receipt['independent_svd_prediction_checked'])
            self.assertEqual(receipt['work']['independent_query_predictions'], 1)
            self.assertEqual(effects[[0, 1, 4, 5]].tobytes(), ev['effects'][[0, 1, 4, 5]].tobytes())
            self.assertEqual(math.copysign(1., effects[2] - ev['effects'][2]), math.copysign(1., coefficient))

    def test_zero_prediction_preserves_every_bit(self):
        ev = event(); ev['effects'][2] = -0.; model = header_model()
        effects, expected = audit(ev, model)
        ind.verify_adjustment(ev, model, effects, expected)
        self.assertEqual(effects.tobytes(), ev['effects'].tobytes())
        effects[2] = 0.
        with self.assertRaisesRegex(ValueError, 'effect bytes'):
            ind.verify_adjustment(ev, model, effects, expected)

    def test_cold_and_partial_do_not_consume_features_or_beta(self):
        for partial in (False, True):
            ev = event(); model = header_model('ready' if partial else 'cold', year=2016 if partial else 2020)
            if partial:
                ev['signal'] = ind.INITIAL_PARTIAL
            ev.update(features=Poison(), symbols=Poison(), eligible=Poison(), pre_scores=Poison())
            model['beta'] = Poison()
            effects, expected = audit(ev, model)
            receipt = ind.verify_adjustment(ev, model, effects, expected)
            self.assertEqual(receipt['work']['independent_query_predictions'], 0)
            self.assertIsNone(receipt['model_sha256'])

    def test_outside_gap_does_not_read_query_or_model_body(self):
        ev = event(); ev['pre_scores'][3] = .07; ev['pre_scores'][4] = .06
        ev['features'] = Poison(); model = header_model(beta=Poison())
        effects, expected = audit(ev, model, 'outside_deployment_gap')
        receipt = ind.verify_adjustment(ev, model, effects, expected)
        self.assertIsNone(receipt['model_sha256'])
        self.assertEqual(receipt['work']['independent_query_predictions'], 0)

    def test_insufficient_original_E_no_query_body(self):
        ev = event(); ev['eligible'][3:] = False; ev['features'] = Poison(); model = header_model(beta=Poison())
        effects, expected = audit(ev, model, 'fewer_than_four_eligible')
        self.assertTrue(ind.verify_adjustment(ev, model, effects, expected)['passed'])

    def test_exact_gap_threshold_included(self):
        ev = event(); ev['pre_scores'][2:5] = [.02, 0., -.01]
        model = header_model(); effects, expected = audit(ev, model)
        self.assertEqual(ind.verify_adjustment(ev, model, effects, expected)['status'], 'zero_prediction')

    def test_active_bad_required_feature_is_hard_error(self):
        ev = event(); model = header_model(); effects, expected = audit(ev, model)
        ev['features'][2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Required pair feature'):
            ind.verify_adjustment(ev, model, effects, expected)

    def test_same_year_native_availability_and_model_parameters(self):
        for key, value in [('query_year', 2019), ('common_months', True), ('alpha', 5.), ('rescaled', True)]:
            ev = event(); model = header_model(); effects, expected = audit(ev, model); model[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ind.verify_adjustment(ev, model, effects, expected)

    def test_verified_svd_query_no_refit_and_hash_bound(self):
        rows = months(); supplied = bundle(rows); proof = ind.verify_models(rows, 2020, supplied)
        ev = event(); model = supplied['models'][ind.ARMS[1]]; effects, expected = audit(ev, model)
        with mock.patch.object(ind.np.linalg, 'svd', side_effect=AssertionError('No deployment refit')):
            receipt = ind.verify_adjustment(ev, model, effects, expected, model_verification=proof)
        self.assertTrue(receipt['independent_svd_prediction_checked'])
        self.assertEqual(receipt['work']['independent_query_predictions'], 2)
        self.assertEqual(receipt['work']['independent_svd_solves'], 0)
        model['beta'][0] = np.nextafter(model['beta'][0], np.inf)
        effects, expected = audit(ev, model)
        with self.assertRaisesRegex(ValueError, 'model binding'):
            ind.verify_adjustment(ev, model, effects, expected, model_verification=proof)

    def test_svd_prediction_and_receipt_evidence_checked(self):
        rows = months(); supplied = bundle(rows); proof = ind.verify_models(rows, 2020, supplied)
        ev = event(); model = supplied['models'][ind.ARMS[1]]; effects, expected = audit(ev, model)
        for what in ('prediction', 'flags', 'atol', 'encoding'):
            wrong = deepcopy(proof)
            if what == 'prediction':
                wrong['svd_beta'][model['arm']]['values'][0] = (1.).hex()
            elif what == 'flags':
                wrong[ind.MODEL_FLAGS[0]] = False
            elif what == 'atol':
                wrong['continuous_atol'] = 1.
            else:
                wrong['svd_beta'][model['arm']]['values'][0] = 'nan'
            with self.subTest(what=what), self.assertRaises(ValueError):
                ind.verify_adjustment(ev, model, effects, expected, model_verification=wrong)

    def test_q_and_effects_one_ulp_never_use_continuous_tolerance(self):
        ev = event(); beta = np.zeros(10); beta[0] = .05; model = header_model(beta=beta)
        effects, expected = audit(ev, model)
        bad = effects.copy(); bad[2] = np.nextafter(bad[2], np.inf)
        with self.assertRaises(ValueError):
            ind.verify_adjustment(ev, model, bad, expected)
        expected['q'] = np.nextafter(expected['q'], np.inf).item()
        with self.assertRaises(ValueError):
            ind.verify_adjustment(ev, model, effects, expected)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = staticmethod(original_selector())

    def check(self, inputs):
        p, e, names, effects, source = inputs
        choices, final = self.selector(p.copy(), e.copy(), deepcopy(names), effects.copy(), deepcopy(source),
                                       dict(budget=.05, ceiling=5, gap=.02))
        receipt = ind.verify_selection(p, e, names, effects, source, choices, final)
        return choices, final, receipt

    def test_original_rank_five_can_win_top_two_preserved(self):
        choices, _, receipt = self.check(selection_inputs())
        self.assertEqual(choices, [[0, 1, 4]])
        self.assertEqual(receipt['work']['selector_candidate_cells'], 3)
        self.assertEqual(receipt['original_episode_calls'], 1)

    def test_empty_prefix_one_selector_call_zero_event_work(self):
        choices, final, receipt = self.check(selection_inputs(0))
        self.assertEqual(choices, []); self.assertEqual(final.shape, (0, 6))
        self.assertEqual(receipt['work']['independent_selector_calls'], 1)
        self.assertEqual(receipt['work']['selector_events'], 0)

    def test_two_eligible_halves_and_one_or_zero_cash(self):
        for count in (0, 1, 2):
            args = selection_inputs(); args[1][0, count:] = False
            choices, _, _ = self.check(args)
            self.assertEqual(choices, [[0, 1]] if count == 2 else [[]])

    def test_symbol_ties_and_state_budget(self):
        args = selection_inputs(); args[0][:] = 0.
        args[2][0] = ['F', 'E', 'D', 'C', 'B', 'A']
        args[3][:] = .5; args[4]['state_calm10'][:] = 1.; args[4]['state_weak_narrow'][:] = 1.
        choices, final, _ = self.check(args)
        self.assertEqual(choices[0], [3, 5, 4])
        self.assertEqual(final[0, 3].item().hex(), ((.10 - (.025 + .05)) * .5).hex())

    def test_preserve_ineligible_nan_payload_and_signed_zero(self):
        args = selection_inputs(); args[1][0, 5] = False
        args[0][0, 5] = np.array([0x7ff8000000000042], dtype=np.uint64).view(np.float64)[0]
        args[0][0, 0] = -0.; args[0][0, 1:5] = [-.01, -.02, -.03, -.035]
        _, final, _ = self.check(args)
        self.assertEqual(final[0, 5].tobytes(), args[0][0, 5].tobytes())
        self.assertEqual(final[0, 0].tobytes(), args[0][0, 0].tobytes())

    def test_final_score_choice_order_and_bool_index_tamper_fail(self):
        args = selection_inputs(); choices, final, _ = self.check(args)
        for what in ('ulp', 'order', 'bool', 'dtype'):
            c, f = deepcopy(choices), final.copy()
            if what == 'ulp':
                f[0, 4] = np.nextafter(f[0, 4], np.inf)
            elif what == 'order':
                c[0].reverse()
            elif what == 'bool':
                c[0][0] = False
            else:
                f = f.astype(np.float32)
            with self.subTest(what=what), self.assertRaises(ValueError):
                ind.verify_selection(*args, c, f)

    def test_full_source_future_numeric_rows_are_not_consumed(self):
        args = list(selection_inputs())
        class PrefixOnly:
            def __init__(self, a):
                self.a = a
            def __getitem__(self, key):
                if not isinstance(key, slice) or key.start is not None or key.stop != 1:
                    raise AssertionError('Future source access')
                return self.a
        c, f, _ = self.check(args)
        args[4] = {key: PrefixOnly(a) for key, a in args[4].items()}
        self.assertTrue(ind.verify_selection(*args, c, f)['passed'])


class IndependenceTests(unittest.TestCase):
    def test_no_producer_import_and_declared_scope(self):
        text = Path(ind.__file__).read_text(); tree = ast.parse(text)
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in imports))
        self.assertIn('not a blind review', text)
        self.assertEqual(len(ind.WORK_FIELDS), 14)
        self.assertEqual(len(set(ind.WORK_FIELDS)), 14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
