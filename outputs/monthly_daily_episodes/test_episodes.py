import unittest
import numpy as np
import episodes as s


class EpisodeUnitTests(unittest.TestCase):
    def event_fixture(self):
        n = 8
        dates = np.array(['2020-01-02', '2020-02-03', '2020-03-04', '2020-04-06',
                          '2020-05-07', '2020-06-08', '2020-07-09', '2020-08-10'], dtype='datetime64[D]')
        e = dict(symbol=np.array(['A'] * n), dates=dates, signal=np.ones(n, int),
                 shape=np.zeros((n, 8)), market=np.zeros((n, 3)))
        for h in [5, 10, 21]:
            e['ends_' + str(h)] = dates + np.timedelta64(h + 8, 'D')
            e['excess_' + str(h)] = np.full(n, .08)
            e['utility_' + str(h)] = np.full(n, .06)
        q = dict(symbol='A', date=np.datetime64('2020-12-31'), signal=1, eligible=True,
                 shape=np.zeros(8), market=np.zeros(3), vol=.02)
        return e, q, s.models()[0]

    def test_grid_is_fixed_unique(self):
        self.assertEqual(len(s.models()), 24)
        self.assertEqual(len(s.grid()), 50)
        self.assertEqual(len({s.json.dumps({k: v for k, v in z.items() if k != 'id'}, sort_keys=True)
                              for z in s.grid()}), 50)

    def test_anchors_nonoverlapping_and_prefix_invariant(self):
        signal = np.ones(100, int)
        eligible = np.ones(100, bool)
        a = s.anchors(signal, eligible)
        np.testing.assert_array_equal(a, [0, 21, 42, 63, 84])
        np.testing.assert_array_equal(s.anchors(signal[:65], eligible[:65]), a[a < 65])

    def test_label_uses_next_open_and_adverse_close_path(self):
        px = np.array([100., 105., 90., 120.])
        opening = np.array([100., 110., 105., 90.])
        m = np.array([100., 102., 103., 104.])
        mo = np.array([100., 101., 102., 103.])
        y, u = s.label_path(px, opening, m, mo, 0, 3)
        self.assertAlmostEqual(y, 120 / 110 - 104 / 101)
        self.assertAlmostEqual(u, y - .5 * (1 - 90 / 110))

    def test_missing_label_path_is_not_filled(self):
        p = np.array([100., np.nan, 110.])
        y, u = s.label_path(p, np.ones(3) * 100, np.ones(3) * 100, np.ones(3) * 100, 0, 2)
        self.assertTrue(np.isnan(y) and np.isnan(u))
        self.assertTrue(np.isnan(s.label_path(p, p, p, p, 1, 5)[0]))

    def test_no_signal_no_history_and_ineligible_are_exact_zero(self):
        e, q, cfg = self.event_fixture()
        for update in [dict(signal=0), dict(symbol='B'), dict(eligible=False), dict(signal=2)]:
            p, d = s.estimate(e, dict(q, **update), cfg)
            self.assertEqual(p, 0.)
            self.assertEqual(d['own_weight'], 0.)

    def test_small_sample_rejected_then_shrunk(self):
        e, q, cfg = self.event_fixture()
        e2 = {k: v[:2] for k, v in e.items()}
        self.assertEqual(s.estimate(e2, q, cfg)[0], 0.)
        p, detail = s.estimate(e, q, cfg)
        self.assertGreater(p, 0.)
        self.assertGreater(detail['own_weight'], 0.)
        self.assertLess(detail['own_weight'], 1.)
        self.assertLess(detail['shrunk_effect'], .08)

    def test_magnitude_not_win_rate(self):
        e, q, cfg = self.event_fixture()
        p, _ = s.estimate(e, q, cfg)
        e['excess_5'] = np.full(8, .01)
        small, _ = s.estimate(e, q, cfg)
        self.assertGreater(p, small)

    def test_noise_deadzone_is_exact_zero(self):
        e, q, cfg = self.event_fixture()
        e['excess_5'] = np.array([.2, -.2] * 4)
        p, detail = s.estimate(e, q, cfg)
        self.assertEqual(p, 0.)
        self.assertEqual(detail['shrunk_effect'], 0.)

    def test_constant_outcome_has_no_conditional_increment(self):
        e, q, cfg = self.event_fixture()
        e['signal'][::2] = 2
        p, _ = s.estimate(e, q, dict(cfg, mode='increment'))
        self.assertAlmostEqual(p, 0.)

    def test_future_outcomes_cannot_change_forecast(self):
        e, q, cfg = self.event_fixture()
        q['date'] = np.datetime64('2020-05-31')
        p, d = s.estimate(e, q, cfg)
        mutated = {k: v.copy() for k, v in e.items()}
        future = e['ends_5'] > q['date']
        mutated['excess_5'][future] = 999.
        mutated['shape'][future] = 999.
        self.assertEqual((p, d), s.estimate(mutated, q, cfg))
        self.assertLessEqual(d['latest_label_end'], str(q['date']))

    def test_old_outcomes_outside_five_year_window_are_ignored(self):
        e, q, cfg = self.event_fixture()
        q['date'] = np.datetime64('2030-12-31')
        self.assertEqual(s.estimate(e, q, cfg)[0], 0.)


class EpisodeArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.d, cls.px, cls.op, cls.a = s.load()
        cls.e = dict(np.load(s.P / 'EVENTS.npz'))
        cls.pred = dict(np.load(s.P / 'FORECASTS.npz'))

    def test_original_controls_reproduce122_orders_and_scores(self):
        for row, name in [(s.grid()[0], 'baseline'), (s.grid()[1], 'primary')]:
            ch, sc = s.choices(row, self.m, self.d, self.a, self.pred, 122)
            ref = s.read(s.old.P / (name + '_ORDERS.json'))
            for i, ids in enumerate(ch):
                self.assertEqual([self.m['symbols'][i][j] for j in ids], ref[i]['symbols'])
                np.testing.assert_allclose(sc[i, ids], ref[i]['scores'], atol=1e-12)

    def test_archive_windows_do_not_overlap_within_security(self):
        for symbol in np.unique(self.e['symbol']):
            ids = self.e['anchor_index'][self.e['symbol'] == symbol]
            self.assertTrue((np.diff(ids) >= 21).all())

    def test_zero_effect_preserves_all_scores_and_choices(self):
        original, base = s.choices(s.grid()[1], self.m, self.d, self.a, self.pred, 122)
        choices, sc = s.select_boundary(base, self.d['eligible'], self.m['symbols'], np.zeros_like(base), self.a, s.grid()[2])
        self.assertEqual(original, choices)
        np.testing.assert_array_equal(base, sc)

    def test_top_two_locked_eligibility_and_budget_bound(self):
        original, base = s.choices(s.grid()[1], self.m, self.d, self.a, self.pred, 122)
        for row in s.grid()[2:]:
            ch, sc = s.choices(row, self.m, self.d, self.a, self.pred, 122)
            self.assertLessEqual(np.max(abs(sc - base)), row['budget'] * .5 + 1e-12)
            for i, ids in enumerate(ch):
                rank = sorted(np.flatnonzero(self.d['eligible'][i]), key=lambda j: (-base[i, j], self.m['symbols'][i][j]))
                self.assertIn(len(ids), [0, 2, 3])
                self.assertTrue(self.d['eligible'][i, ids].all())
                if len(rank) >= 3:
                    self.assertTrue(set(rank[:2]).issubset(ids))
                    for j in set(ids) - set(original[i]):
                        self.assertIn(j, rank[:5])
                        self.assertLessEqual(base[i, rank[2]] - base[i, j], .02 + 1e-12)

    def test_query_signal_absent_always_zero(self):
        signal = np.load(s.P / 'QUERIES.npz')['signal']
        for v in self.pred.values():
            self.assertTrue(np.isfinite(v).all())
            np.testing.assert_array_equal(v[signal <= 0], 0.)

    def test_later_forecasts_do_not_affect_training(self):
        mutated = {k: v.copy() for k, v in self.pred.items()}
        for v in mutated.values():
            v[88:] = 999.
        for row in s.grid()[::7]:
            self.assertEqual(s.choices(row, self.m, self.d, self.a, self.pred, 88)[0],
                             s.choices(row, self.m, self.d, self.a, mutated, 88)[0])

    def test_all_used_labels_mature_by_signal_date(self):
        details = s.read(s.P / 'BOUNDARY_DETAILS.json')
        for d in details:
            if d['latest_label_end'] is not None:
                self.assertLessEqual(d['latest_label_end'], d['date'])
            self.assertGreaterEqual(d['own_weight'], 0.)
            self.assertLess(d['own_weight'], 1.)
            self.assertLessEqual(abs(d['correction']), .5)
        self.assertTrue(s.read(s.P / 'DATA_AUDIT.json')['prefix_features_equal'])


if __name__ == '__main__':
    unittest.main()
