import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('earnings_study_tests', P/'study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.a = dict(base_original65=np.array([[.12, .21, -.14, .08]]),
            core_original65=np.array([[.1, .15, -.1, .1]]), cw_original65=np.full((1, 1), .52),
            monthly_event10=np.array([[.02, np.nan, -.1, 0.]]),
            monthly_valid10=np.array([[True, False, True, True]]))
        self.row = dict(base='original65', family='event', key='event10', budget=.05)

    def test_control_is_exact_copy(self):
        row = dict(self.row, family='control')
        got = study.corrected_score(row, self.a, 1)
        np.testing.assert_array_equal(got, self.a['base_original65'])
        got[:] = 0
        self.assertNotEqual(float(self.a['base_original65'][0, 0]), 0.)

    def test_missing_and_zero_effect_exact_nochange(self):
        got = study.corrected_score(self.row, self.a, 1)
        np.testing.assert_array_equal(got[:, [1, 3]], self.a['base_original65'][:, [1, 3]])
        self.assertTrue(np.isnan(self.a['monthly_event10'][0, 1]))

    def test_manual_direction_and_bounded_support(self):
        got = study.corrected_score(self.row, self.a, 1)
        self.assertGreater(got[0, 0], self.a['base_original65'][0, 0])
        self.assertLess(got[0, 2], self.a['base_original65'][0, 2])
        self.assertLessEqual(np.abs(got-self.a['base_original65']).max(), .05)

    def test_zero_budget_cannot_change_decisions(self):
        np.testing.assert_array_equal(study.corrected_score(dict(self.row, budget=0.), self.a, 1), self.a['base_original65'])

    def test_declared_valid_nonfinite_rejected(self):
        self.a['monthly_valid10'][0, 1] = True
        with self.assertRaises(AssertionError): study.corrected_score(self.row, self.a, 1)

    def test_ridge_missing_neutral(self):
        self.a['ridge_earn'] = self.a['monthly_event10'].copy()
        self.a['ridge_available'] = self.a['monthly_valid10'].copy()
        got = study.corrected_score(dict(self.row, family='ridge', key='ridge_earn'), self.a, 1)
        np.testing.assert_array_equal(got, study.corrected_score(self.row, self.a, 1))

    def test_future_event_cannot_change_prefix(self):
        old = study.corrected_score(self.row, self.a, 1)
        for k, v in list(self.a.items()): self.a[k] = np.concatenate([v, v.copy()])
        self.a['monthly_event10'][1] = 1e6
        np.testing.assert_array_equal(old, study.corrected_score(self.row, self.a, 1))

    def test_grid_has_exact_budget_matched_controls(self):
        rows = study.grid()
        self.assertEqual(len(rows), 26)
        self.assertEqual([z['id'] for z in rows], list(range(26)))
        for row in rows:
            if row['family'] == 'event':
                key = row['key'].replace('event', 'matched_recent')
            elif row['key'] == 'ridge_earn': key = 'ridge_plain'
            else: continue
            match = [z for z in rows if z['base'] == row['base'] and z['key'] == key and z['budget'] == row['budget']]
            self.assertEqual(len(match), 1)


if __name__ == '__main__': unittest.main()
