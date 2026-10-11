import unittest
import numpy as np
import pandas as pd
from gate import contract, assess, admitted_transactions
from audit import line_envelope, panel_error
from boundary_audit import metrics_from_segments


class GateTests(unittest.TestCase):
    def test_high_return_or_local_fit_is_insufficient(self):
        self.assertFalse(assess({'cagr_terminal_wealth_maxdd': 'passed'})['admitted'])
        with self.assertRaises(ValueError):
            admitted_transactions([{'holdings': ['A', 'B', 'C']}], {})

    def test_scope_failure_blocks_even_perfect_metrics(self):
        e = {k: 'passed' for k in contract()['admission_requirements']}
        e['author_scope'] = 'failed'
        self.assertFalse(assess(e)['admitted'])

    def test_full_image_match_still_needs_rule_timing(self):
        e = {k: 'passed' for k in contract()['admission_requirements']}
        self.assertTrue(assess(e)['admitted'])
        self.assertFalse(assess(e, 'executable_rule_hypothesis')['admitted'])
        record = admitted_transactions([], e)
        self.assertFalse(record['author_transactions_verified'])

    def test_invalid_status_cannot_be_truthy(self):
        with self.assertRaises(ValueError):
            assess({'author_scope': 'yes'})

    def test_envelope_preserves_daily_peak_within_pixel_column(self):
        lo, hi = line_envelope(np.array([0., .25, 1.]), np.array([0., 10., 0.]), [.5], .5)
        self.assertEqual(lo[0], 0.)
        self.assertEqual(hi[0], 10.)

    def test_envelope_error_does_not_rescale_each_observation(self):
        x = np.arange(3, dtype=float)
        z = pd.DataFrame({'x': x, 'y': [0., 20., 0.], 'ymin': [0., 20., 0.], 'ymax': [0., 20., 0.]})
        r, _ = panel_error(x, np.zeros(3), np.zeros(3), z, 0., 2.)
        self.assertTrue(r['screen_rejected'])
        self.assertAlmostEqual(r['outside_fraction'], 1 / 3)

    def test_segment_metrics_equal_direct_fixed_path(self):
        prefixes = np.array([[1., 1.05, .90], [1., .8, 1.1]])
        suffix = np.array([1., 1.2, .9, 1.4])
        got = metrics_from_segments(prefixes, suffix)
        for i, prefix in enumerate(prefixes):
            nav = np.r_[prefix[:-1], prefix[-1] * suffix]
            r = nav[1:] / nav[:-1] - 1
            self.assertAlmostEqual(got['sharpe'][i], r.mean() / r.std(ddof=1) * np.sqrt(252))
            self.assertAlmostEqual(got['max_dd'][i], (nav / np.maximum.accumulate(nav) - 1).min())
            self.assertAlmostEqual(got['wealth'][i], nav[-1])


if __name__ == '__main__':
    unittest.main()
