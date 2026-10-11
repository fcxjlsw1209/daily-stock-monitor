import copy
import unittest
import numpy as np
import pandas as pd
import style_study as s


class StyleStudyTests(unittest.TestCase):
    def test_split_is_disjoint_and_chronological(self):
        m = s.masks(69)
        self.assertEqual(m['training'].sum(), 35)
        self.assertEqual(m['validation'].sum(), 34)
        self.assertEqual(m['inner_fit'].sum(), 23)
        self.assertEqual(m['inner_validation'].sum(), 12)
        self.assertFalse((m['training'] & m['validation']).any())
        np.testing.assert_array_equal(m['inner_fit'] | m['inner_validation'], m['training'])

    def test_market_style_has_no_future_dependence(self):
        rng = np.random.default_rng(9)
        dates = pd.bdate_range('2015-01-01', periods=650)
        columns = ['SPY', 'QQQ', 'IWM', 'TLT', 'XLP', 'XLV', 'XLU', 'XLK', 'XLY', 'XLI']
        px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (650, 10)), axis=0)), index=dates, columns=columns)
        before = s.market_styles(px, dates[400:450])
        px.loc[dates[450]:] *= 10
        pd.testing.assert_frame_equal(before, s.market_styles(px, dates[400:450]))

    def test_holdout_labels_never_enter_fit(self):
        names = ['mom252_skip42', 'vol126', 'stock_sector_skip42_126', 'stock_sector_skip42_252',
                 'cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126', 'highvol_cmf63', 'trend_cmf63', 'acceleration_skip42']
        rng = np.random.default_rng(12)
        d = dict(names=names, X=rng.uniform(-.5, .5, (6, 4, len(names))),
                 eligible=np.ones((6, 4), bool), order=np.tile([0, 1, 2], (6, 1)))
        mask = np.arange(6) < 3
        before = s.fit_rule(d, ['mom252_skip42'], np.zeros(6, int), mask)
        other = copy.deepcopy(d)
        other['order'][3:] = [3, 2, 1]
        other['X'][3:] *= -3
        after = s.fit_rule(other, ['mom252_skip42'], np.zeros(6, int), mask)
        np.testing.assert_allclose(before['base'], after['base'], rtol=0, atol=1e-12)

    def test_unavailable_author_cannot_be_fixed_by_weights(self):
        ans = s.feasible_margin(np.zeros((4, 5)), np.ones(4, bool), [0, 1, -1], [])
        self.assertEqual(ans['status'], 'author_not_eligible_or_not_in_pool')
        self.assertIsNone(s.residual_needed(np.arange(4.), np.ones(4, bool), [0, 1, -1]))

    def test_single_month_constraints_preserve_core_budget(self):
        spec = [('m', 1, 'core'), ('v', 1, 'vol'), ('i', 1, 'industry'), ('p', 1, 'shape'), ('a', 1, 'accel')]
        X = np.tile(np.array([.4, .3, .2, .1])[:, None], (1, 5))
        ans = s.feasible_margin(X, np.ones(4, bool), [0, 1, 2], spec)
        self.assertEqual(ans['status'], 'strictly_feasible')
        self.assertGreaterEqual(ans['weights'][0], .5 - 1e-10)
        impossible = s.feasible_margin(X, np.ones(4, bool), [3, 2, 1], spec)
        self.assertEqual(impossible['status'], 'infeasible')

    def test_minimum_residual_is_not_a_predictive_model(self):
        value = s.residual_needed(np.array([.2, .1, .0, .4]), np.ones(4, bool), [0, 1, 2])
        self.assertAlmostEqual(value, .2000005)
        small = s.residual_needed(np.array([.4, .3, .2, .1]), np.ones(4, bool), [0, 1, 2])
        self.assertAlmostEqual(small, 0.)

    @unittest.skipUnless((s.P / 'FROZEN.json').exists(), 'Run research phases first')
    def test_saved_state_weights_keep_budgets_and_training_count(self):
        frozen = s.read(s.P / 'FROZEN.json')
        for row in frozen['leaders'].values():
            rule = row['rule']
            A, b, _ = s.e.limits(rule['spec'])
            base = np.array(rule['base'])
            for group, weights in rule['weights'].items():
                w = np.array(weights)
                self.assertAlmostEqual(w.sum(), 1.)
                self.assertGreaterEqual(w.min(), -1e-7)
                self.assertLessEqual(np.max(np.einsum('ij,j->i', A, w, optimize=False) - b), 1e-7)
                self.assertLessEqual(np.abs(w - base).sum(), .30 + 1e-7)
                self.assertLessEqual(rule['audit'][group]['months'], 35)
                if rule['audit'][group]['months'] < 8:
                    np.testing.assert_allclose(w, base, rtol=0, atol=1e-12)

    @unittest.skipUnless((s.P / 'STYLE_MONTHLY_EXPLANATIONS.json').exists(), 'Run diagnostics first')
    def test_every_month_and_stock_is_covered_and_contributions_add_up(self):
        for filename in ['MONTHLY_EXPLANATIONS.json', 'STYLE_MONTHLY_EXPLANATIONS.json']:
            months = s.read(s.P / filename)
            self.assertEqual(len(months), 69)
            previous = set()
            for m in months:
                names = {r['ticker'] for r in m['securities']}
                author = set(m['author'])
                self.assertEqual(names, previous | author | set(m['model']))
                self.assertEqual(set(m['entered']), author - previous)
                self.assertEqual(set(m['exited']), previous - author)
                for row in m['securities']:
                    if 'components' in row:
                        self.assertAlmostEqual(sum(row['components'].values()), row['score'])
                    if 'score_change_decomposition' in row:
                        dec = row['score_change_decomposition']
                        self.assertAlmostEqual(dec['feature_change'] + dec['market_style_weight_change'], dec['total'])
                previous = author


if __name__ == '__main__':
    unittest.main()
