"""Synthetic-only tests. No research cache, market source, or actual fit is read."""
import copy
import itertools
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_latent_state134_model as m


def model():
    return dict(available=True, status='available', reason=None, fit_attempted=True,
                scaler=dict(mean=0., sd=1.),
                parameters=dict(means=[-.2, .3], variances=[.5, 2.],
                                initial=[.4, .6], transition=[[.9, .1], [.2, .8]]),
                filtered_posterior=[.7, .3])


def brute(x, means, variances, pi, trans):
    paths = []
    for states in itertools.product(range(2), repeat=len(x)):
        p = pi[states[0]]
        for t, k in enumerate(states):
            if t:
                p *= trans[states[t-1]][k]
            p *= math.exp(-.5 * (x[t]-means[k])**2/variances[k]) / math.sqrt(2*math.pi*variances[k])
        paths.append((states, p))
    total = math.fsum(p for _, p in paths)
    gamma = np.zeros((len(x), 2)); counts = np.zeros((2, 2))
    for states, p in paths:
        for t, k in enumerate(states):
            gamma[t, k] += p / total
            if t:
                counts[states[t-1], k] += p / total
    return math.log(total), gamma, counts


class InferenceTests(unittest.TestCase):
    def test_forward_backward_matches_all_hidden_paths(self):
        x = np.array([-.8, .1, 1.3, -.2]); p = model()['parameters']
        args = [np.asarray(p[k]) for k in ('means', 'variances', 'initial', 'transition')]
        e = m._expectation(x, *args); ll, gamma, counts = brute(x, *args)
        self.assertAlmostEqual(e['log_likelihood'], ll, places=12)
        np.testing.assert_allclose(e['gamma'], gamma, atol=1e-13, rtol=0)
        np.testing.assert_allclose(e['counts'], counts, atol=1e-13, rtol=0)
        np.testing.assert_allclose(e['posterior'], gamma[-1], atol=1e-13, rtol=0)

    def test_filter_every_prefix_matches_enumeration(self):
        x = np.array([-.8, .1, 1.3, -.2]); d = model(); p = d['parameters']
        actual = m.filter_path(x, d)
        for n in range(1, len(x)+1):
            _, gamma, _ = brute(x[:n], p['means'], p['variances'], p['initial'], p['transition'])
            np.testing.assert_allclose(actual[n-1], gamma[-1], atol=1e-13, rtol=0)

    def test_state_label_permutation_is_equivariant(self):
        x = np.array([-.4, 1.1, .2]); d = model(); p = d['parameters']
        q = copy.deepcopy(d)
        for k in ('means', 'variances', 'initial'):
            q['parameters'][k] = p[k][::-1]
        q['parameters']['transition'] = np.asarray(p['transition'])[::-1, ::-1].tolist()
        np.testing.assert_allclose(m.filter_path(x, d), m.filter_path(x, q)[:, ::-1], atol=1e-14)
        np.testing.assert_allclose(m.memoryless(x, d), m.memoryless(x, q)[:, ::-1], atol=1e-14)

    def test_resume_from_previous_posterior_predicts_first(self):
        d = model(); x = np.array([-.4, .1, .8, -.2]); all_rows = m.filter_path(x, d)
        resumed = m.filter_path(x[2:], d, initial=all_rows[1], predict_first=True)
        np.testing.assert_allclose(resumed, all_rows[2:], atol=1e-14, rtol=0)

    def test_predict_first_is_not_initial_prior(self):
        d = model()
        self.assertFalse(np.allclose(m.filter_path([.2], d, [.99, .01], False),
                                     m.filter_path([.2], d, [.99, .01], True)))

    def test_stationary_equation(self):
        a = np.array([[.95, .05], [.2, .8]]); p = m.stationary(a)
        np.testing.assert_allclose(p, [.8, .2], atol=1e-14)
        np.testing.assert_allclose(p @ a, p, atol=1e-14)
        np.testing.assert_array_equal(m.stationary([[1., 0.], [.5, .5]]), [1., 0.])

    def test_nonunique_or_invalid_stationary_rejected(self):
        with self.assertRaises(m.NumericalError):m.stationary(np.eye(2))
        for bad in ([[.5, .4], [.1, .9]], [[-.1, 1.1], [.2, .8]], [[1, 0, 0]]):
            with self.assertRaises(ValueError):m.stationary(bad)

    def test_memoryless_equals_independent_bayes(self):
        d = model(); x = [-.3, 1.2, -.3]; p = d['parameters']; prior = m.stationary(p['transition'])
        result = m.memoryless(x, d)
        for t, value in enumerate(x):
            mass = [prior[k] * math.exp(-.5*(value-p['means'][k])**2/p['variances'][k]) /
                    math.sqrt(2*math.pi*p['variances'][k]) for k in range(2)]
            np.testing.assert_allclose(result[t], np.array(mass)/sum(mass), atol=1e-14)
        np.testing.assert_array_equal(result[0], result[2])

    def test_memoryless_equals_filter_for_iid_transition(self):
        d = model(); d['parameters']['transition'] = [[.6, .4], [.6, .4]]
        d['parameters']['initial'] = [.6, .4]
        x = [-.7, .4, .9, -.1]
        np.testing.assert_allclose(m.memoryless(x, d), m.filter_path(x, d), atol=1e-14)

    def test_empty_and_zero_prior(self):
        self.assertEqual(m.filter_path([], model()).shape, (0, 2))
        self.assertEqual(m.memoryless([], model()).shape, (0, 2))
        np.testing.assert_array_equal(m.filter_path([0.], model(), [1., 0.]), [[1., 0.]])

    def test_nonfinite_and_unavailable_are_not_inferred(self):
        for method in (m.filter_path, m.memoryless):
            with self.assertRaises(m.NumericalError):method([math.nan], model())
            with self.assertRaises(ValueError):method([.1], dict(available=False))
            with self.assertRaises(ValueError):method(['.1'], model())
            with self.assertRaises(ValueError):method([True], model())

    def test_probability_floor_is_after_count_normalization(self):
        z = np.array([-.1, .1]); e = dict(gamma=np.array([[.5, .5], [.5, .5]]),
                counts=np.array([[1e-30, 1e6], [2e6, 1e-30]]))
        _, _, pi, a = m._maximization(z, e)
        self.assertAlmostEqual(a[0, 0], 1e-8/(1+1e-8), places=16)
        self.assertAlmostEqual(a[1, 1], 1e-8/(1+1e-8), places=16)
        np.testing.assert_array_equal(pi, [.5, .5])


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(134)
        cls.x = np.concatenate([rng.normal(0, .004, 270), rng.normal(.001, .035, 270),
                                rng.normal(-.001, .004, 270), rng.normal(0, .035, 270)])
        cls.fitted = m.fit(cls.x)

    def test_fit_certifies_convergence_and_ordered_states(self):
        d = self.fitted; self.assertTrue(d['available']); self.assertTrue(d['fit_attempted'])
        self.assertTrue(d['certificate']['converged']); self.assertLessEqual(d['updates'], 300)
        self.assertGreaterEqual(d['variance_ratio'], 1.05); self.assertGreaterEqual(min(d['occupancy']), 20)
        self.assertAlmostEqual(sum(d['occupancy']), len(self.x), places=8)
        self.assertEqual(len(d['log_likelihood_history']), d['updates']+1)
        self.assertGreaterEqual(min(np.diff(d['log_likelihood_history'])) / len(self.x), -1e-7)
        self.assertLessEqual(abs(np.diff(d['log_likelihood_history'])[-1])/len(self.x), 1e-7)
        json.dumps(d, allow_nan=False)

    def test_final_posterior_replays_final_sorted_parameters(self):
        np.testing.assert_allclose(m.filter_path(self.x, self.fitted)[-1],
                                   self.fitted['filtered_posterior'], atol=1e-13, rtol=0)

    def test_scaler_population_and_affine_invariance(self):
        d = self.fitted; self.assertAlmostEqual(d['scaler']['mean'], float(self.x.mean()), places=15)
        self.assertAlmostEqual(d['scaler']['sd'], float(self.x.std(ddof=0)), places=15)
        other = m.fit(3*self.x+.2)
        for key in ('means','variances','initial','transition'):
            np.testing.assert_allclose(d['parameters'][key], other['parameters'][key], atol=2e-12, rtol=1e-11)

    def test_insufficient_and_constant_do_not_attempt_em(self):
        for x, reason in ((np.zeros(755), 'insufficient_observations'),
                          (np.full(756, .1), 'degenerate_training_scale')):
            d = m.fit(x); self.assertFalse(d['available']); self.assertFalse(d['fit_attempted'])
            self.assertEqual(d['reason'], reason); json.dumps(d, allow_nan=False)

    def test_nonfinite_fit_is_hard_failure(self):
        for value in (math.nan, math.inf):
            x = self.x.copy(); x[0] = value
            with self.assertRaises(m.NumericalError):m.fit(x)

    def test_nonconvergence_does_not_fallback(self):
        with patch.object(m, 'MAX_UPDATES', 1):
            with self.assertRaisesRegex(m.NumericalError, 'did not converge'):m.fit(self.x)

    def test_material_likelihood_decrease_is_failure(self):
        original = m._expectation; calls = [0]
        def poisoned(*args):
            e = original(*args); calls[0] += 1
            if calls[0] > 1:e['log_likelihood'] -= 1e9
            return e
        with patch.object(m, '_expectation', poisoned):
            with self.assertRaisesRegex(m.NumericalError, 'likelihood decrease'):m.fit(self.x)

    def test_postfit_occupancy_and_ratio_remain_attempted(self):
        # Test post-convergence gates without changing the mathematical fit.
        for gate, value, reason in (('MIN_OCCUPANCY', 2000., 'insufficient_effective_state_occupancy'),
                                    ('MIN_VARIANCE_RATIO', 1e6, 'unseparated_state_variances')):
            with patch.object(m, gate, value):d = m.fit(self.x)
            self.assertFalse(d['available']); self.assertTrue(d['fit_attempted'])
            self.assertEqual(d['reason'], reason); self.assertTrue(d['certificate']['converged'])


class AnnualTests(unittest.TestCase):
    def setUp(self):
        # Explicit synthetic calendar; no claim that bdate_range is an exchange calendar.
        self.sessions = pd.bdate_range('2009-01-01', '2017-12-29')
        x = np.arange(len(self.sessions)); self.price = pd.Series(100*np.exp(.0001*x+.01*np.sin(x/11)), self.sessions)
        self.signals = pd.DatetimeIndex(['2015-06-30','2015-12-31','2016-06-30'])

    def fake_fit(self, x):
        d = model(); d['n_observations'] = len(x); return d

    def run_annual(self, price=None, signals=None, sessions=None):
        with patch.object(m, 'fit', side_effect=self.fake_fit):
            return m.annual_probabilities(self.price if price is None else price,
                    self.sessions if sessions is None else sessions,
                    self.signals if signals is None else signals)

    def test_annual_window_preceding_price_and_single_shared_fit(self):
        calls=[]
        def capture(x):calls.append(x.copy());return self.fake_fit(x)
        with patch.object(m, 'fit', side_effect=capture):d=m.annual_probabilities(self.price,self.sessions,self.signals)
        self.assertEqual(len(calls),2); self.assertEqual(d['audit']['new_model_fits'],2)
        y=d['audit']['years'][0]; self.assertEqual(y['lower_bound_exclusive'],'2009-12-31')
        self.assertEqual(y['cutoff'],'2014-12-31'); self.assertEqual(y['preceding_price_date'],'2009-12-31')
        expected=self.price.pct_change().loc['2010-01-01':'2014-12-31'].to_numpy()
        np.testing.assert_array_equal(calls[0],expected)
        self.assertTrue(np.isfinite(d['filtered']).all());json.dumps(d['audit'],allow_nan=False)

    def test_all_current_year_returns_are_replayed(self):
        d=self.run_annual(); returns=self.price.pct_change().loc['2015-01-01':'2015-06-30'].to_numpy()
        expected=m.filter_path(returns,model(),model()['filtered_posterior'],True)[-1,0]
        self.assertAlmostEqual(d['filtered'][0],expected,places=14)
        self.assertAlmostEqual(d['memoryless'][0],m.memoryless(returns[-1:],model())[0,0],places=14)

    def test_future_poison_and_physically_truncated_prefix(self):
        original=self.run_annual(); short=self.run_annual(signals=self.signals[:1],
                                price=self.price.loc[:self.signals[0]],sessions=self.sessions[self.sessions<=self.signals[0]])
        poisoned=self.price.copy();poisoned.loc[poisoned.index>self.signals[0]]=np.nan
        poison=self.run_annual(price=poisoned,signals=self.signals[:1])
        np.testing.assert_array_equal(short['filtered'],original['filtered'][:1])
        np.testing.assert_array_equal(short['filtered'],poison['filtered'])
        self.assertEqual(short['audit']['signals'],original['audit']['signals'][:1])
        self.assertEqual(short['audit']['signals'],poison['audit']['signals'])

    def test_missing_query_poison_persists_and_does_not_leak_backwards(self):
        price=self.price.copy();price.loc['2015-09-01']=np.nan
        d=self.run_annual(price=price)
        self.assertTrue(np.isfinite(d['filtered'][0]));self.assertTrue(np.isnan(d['filtered'][1]))
        self.assertIsNone(d['audit']['signals'][0]['first_invalid_query_date'])
        self.assertEqual(d['audit']['signals'][1]['first_invalid_query_date'],'2015-09-01')
        self.assertEqual(d['audit']['signals'][1]['reason'],'invalid_query_history')
        self.assertEqual(d['audit']['signals'][2]['reason'],'incomplete_training_prices')

    def test_missing_training_is_unavailable_not_fit(self):
        price=self.price.drop(pd.Timestamp('2014-01-15'))
        with patch.object(m,'fit',side_effect=AssertionError('must not fit')):
            d=m.annual_probabilities(price,self.sessions,self.signals[:1])
        self.assertFalse(d['audit']['years'][0]['fit_attempted']);self.assertEqual(d['audit']['new_model_fits'],0)
        self.assertEqual(d['audit']['signals'][0]['reason'],'incomplete_training_prices')
        self.assertTrue(np.isnan(d['memoryless']).all())

    def test_incomplete_calendar_does_not_shorten_training(self):
        sessions=self.sessions[self.sessions>='2012-01-01']
        d=self.run_annual(sessions=sessions,signals=self.signals[:1])
        self.assertEqual(d['audit']['years'][0]['reason'],'incomplete_training_calendar')
        self.assertEqual(d['audit']['new_model_fits'],0)

    def test_nonpositive_training_and_query(self):
        for when, reason in [('2014-01-15','incomplete_training_prices'),('2015-03-02','invalid_query_history')]:
            price=self.price.copy();price.loc[when]=0.
            d=self.run_annual(price=price,signals=self.signals[:1])
            self.assertEqual(d['audit']['signals'][0]['reason'],reason)

    def test_calendar_holiday_and_signal_validation(self):
        sessions=self.sessions[self.sessions!=pd.Timestamp('2015-01-01')]
        d=self.run_annual(sessions=sessions,signals=self.signals[:1]);self.assertTrue(np.isfinite(d['filtered'][0]))
        with self.assertRaises(ValueError):self.run_annual(signals=['2015-07-04'])
        with self.assertRaises(ValueError):self.run_annual(signals=['2015-06-30','2015-06-30'])
        with self.assertRaises(ValueError):self.run_annual(sessions=self.sessions[::-1])
        with self.assertRaises(ValueError):self.run_annual(signals=['2015-06-30 12:00'])

    def test_solver_failure_is_not_yearly_unavailable(self):
        with patch.object(m,'fit',side_effect=m.NumericalError('failed')):
            with self.assertRaises(m.NumericalError):m.annual_probabilities(self.price,self.sessions,self.signals[:1])

    def test_empty_signals(self):
        d=self.run_annual(signals=[]);self.assertEqual(d['audit'],dict(years=[],signals=[],new_model_fits=0))
        self.assertEqual(d['filtered'].shape,(0,))


if __name__ == '__main__':
    unittest.main()
