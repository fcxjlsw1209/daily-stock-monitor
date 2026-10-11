import unittest
import numpy as np
import ridge_study as r


class RidgeTests(unittest.TestCase):
    def test_training_label_must_be_realized_by_cutoff(self):
        dates = ['2014-12-01', '2015-12-31', '2020-11-30', '2020-12-24', '2021-01-08']
        ends = ['2015-01-01', '2016-01-29', '2020-12-30', '2021-01-26', '2021-02-08']
        self.assertEqual(r.matured_rows(dates, ends, '2020-12-31'), [1, 2])

    def test_future_target_cannot_change_fit(self):
        rng = np.random.default_rng(4)
        X, y = rng.normal(size=(6, 4)), rng.normal(size=6)
        dates = ['2018-01-01', '2018-02-01', '2019-01-01', '2020-01-01', '2020-12-24', '2021-01-01']
        ends = ['2018-02-01', '2018-03-01', '2019-02-01', '2020-02-01', '2021-01-25', '2021-02-01']
        ids = r.matured_rows(dates, ends, '2020-12-31')
        a = r.fit_weighted_ridge(X[ids], y[ids], np.ones(len(ids)), 10.)
        y[4:] *= 1000
        b = r.fit_weighted_ridge(X[ids], y[ids], np.ones(len(ids)), 10.)
        np.testing.assert_array_equal(a['coef'], b['coef'])

    def test_forecast_is_separate_capped_group(self):
        X = np.zeros((2, 3, 5))
        spec = [('m', 1, 'core'), ('v', 1, 'vol'), ('i', 1, 'industry'), ('s', 1, 'shape'), ('a', 1, 'accel')]
        z, sp = r.append_forecast(X, spec, np.ones((6, 2, 3)), 2)
        self.assertEqual(z.shape, (2, 3, 6))
        self.assertEqual(sp[-1][2], 'ml')
        A, b, prior = r.e.limits(sp)
        self.assertEqual(b[4], .1)
        self.assertEqual(A[4, -1], 1.)

    @unittest.skipUnless((r.P / 'FORECAST_AUDIT.json').exists(), 'Run forecast first')
    def test_saved_models_all_obey_annual_cutoff(self):
        fits = r.read(r.P / 'FORECAST_AUDIT.json')
        self.assertEqual(len(fits), 36)
        for f in fits:
            self.assertLessEqual(f['latest_label_end'], f['cutoff'])
            self.assertGreater(f['rows'], 1000)
            self.assertLess(f['normal_equation_residual'], 1e-7)

    @unittest.skipUnless((r.P / 'FROZEN.json').exists(), 'Run train first')
    def test_frozen_core_and_ml_budgets(self):
        for row in r.read(r.P / 'FROZEN.json')['leaders'].values():
            spec = row['rule']['spec']
            for weights in row['rule']['weights'].values():
                core = sum(v for v, t in zip(weights, spec) if t[2] == 'core')
                ml = sum(v for v, t in zip(weights, spec) if t[2] == 'ml')
                self.assertGreaterEqual(core, .5 - 1e-7)
                self.assertLessEqual(ml, .1 + 1e-7)


if __name__ == '__main__':
    unittest.main()
