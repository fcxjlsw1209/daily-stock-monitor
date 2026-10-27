"""Synthetic admission tests, no market data or return computation."""
import copy
import unittest
import monthly_forecast_spacing269_admission as h


def fixture():
    g = dict(training=dict(cagr=.5, sharpe=1.7, max_drawdown=-.3),
             early=dict(cagr=.4), recent=dict(cagr=.6))
    m = {a: {p: copy.deepcopy(g) for p in ('same_close_zero', 'next_open_zero')} for a in h.ARMS}
    o = {a: [dict(symbols=['A', 'B', 'C']) for _ in range(122)] for a in h.ARMS}
    for a in h.ARMS[1:]:
        for p in m[a]: m[a][p]['training']['cagr'] = .52
        for i in (1, 54, 55): o[a][i]['symbols'] = ['A', 'B', 'D']
    return m, o


class AdmissionTests(unittest.TestCase):
    def test_both_eligible_tie_prefers_narrower_scope(self):
        m, o = fixture(); r = h.select_first88(m, o)
        self.assertEqual(r['qualified'], list(h.ARMS[1:])); self.assertEqual(r['primary'], h.ARMS[1])

    def test_best_qualified_training_return_wins(self):
        m, o = fixture(); m[h.ARMS[2]]['same_close_zero']['training']['cagr'] = .53
        self.assertEqual(h.select_first88(m, o)['primary'], h.ARMS[2])

    def test_failed_one_does_not_block_other(self):
        m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['training']['cagr'] = .1
        self.assertEqual(h.select_first88(m, o)['primary'], h.ARMS[2])

    def test_none_passing_retains_incumbent(self):
        m, o = fixture()
        for a in h.ARMS[1:]: m[a]['same_close_zero']['training']['cagr'] = .504
        r = h.select_first88(m, o); self.assertFalse(r['admitted']); self.assertEqual(r['primary'], h.ARMS[0])

    def test_open_zero_cannot_be_worse(self):
        m, o = fixture(); m[h.ARMS[1]]['next_open_zero']['training']['cagr'] = .49
        self.assertFalse(h.select_first88(m, o)['guards'][h.ARMS[1]]['open'])

    def test_each_path_risk_and_era_guards(self):
        for a in h.ARMS[1:]:
            for p in ('same_close_zero', 'next_open_zero'):
                for group, key, value in [('training', 'sharpe', 1.64), ('training', 'max_drawdown', -.34), ('early', 'cagr', .37), ('recent', 'cagr', .57)]:
                    m, o = fixture(); m[a][p][group][key] = value
                    self.assertNotIn(a, h.select_first88(m, o)['qualified'])

    def test_changes_need_both_eras_and_three_months(self):
        for indexes in ((1, 2, 3), (53, 54, 55), (1, 54)):
            m, o = fixture(); o[h.ARMS[1]] = copy.deepcopy(o[h.ARMS[0]])
            for i in indexes: o[h.ARMS[1]][i]['symbols'] = ['A', 'B', 'D']
            self.assertFalse(h.select_first88(m, o)['guards'][h.ARMS[1]]['support'])

    def test_order_only_not_change(self):
        m, o = fixture()
        for i in (1, 54, 55): o[h.ARMS[1]][i]['symbols'] = ['C', 'A', 'B']
        self.assertFalse(h.select_first88(m, o)['guards'][h.ARMS[1]]['support'])

    def test_later_orders_not_consumed(self):
        class First88(list):
            def __getitem__(self, i):
                if not isinstance(i, int) or i >= 88: raise AssertionError('Future order consumed')
                return super().__getitem__(i)
        m, o = fixture(); expected = h.select_first88(m, o)
        self.assertEqual(expected, h.select_first88(m, {a: First88(v) for a, v in o.items()}))

    def test_extra_cost_path_or_later_slice_rejected(self):
        m, o = fixture(); m[h.ARMS[1]]['next_open_cost'] = {'cagr': 10.}
        with self.assertRaises(ValueError): h.select_first88(m, o)
        m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['later34'] = {'cagr': 10.}
        with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_missing_roles_or_short_history_rejected(self):
        m, o = fixture(); del m[h.ARMS[1]]
        with self.assertRaises(ValueError): h.select_first88(m, o)
        m, o = fixture(); o[h.ARMS[1]] = o[h.ARMS[1]][:87]
        with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_first88_only_equal_to_full(self):
        m, o = fixture(); self.assertEqual(h.select_first88(m, o), h.select_first88(m, {a: v[:88] for a, v in o.items()}))

    def test_bad_required_metric_rejected(self):
        for value in (None, float('nan'), float('inf'), True):
            m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['training']['cagr'] = value
            with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_undefined_candidate_sharpe_rejects_only_that_arm(self):
        m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['training'].update(sharpe=None, daily_return_sd=0., sharpe_status='undefined_sharpe', sharpe_reason='zero_daily_return_sd')
        self.assertEqual(h.select_first88(m, o)['primary'], h.ARMS[2])

    def test_undefined_incumbent_sharpe_is_error(self):
        m, o = fixture(); m[h.ARMS[0]]['same_close_zero']['training'].update(sharpe=None, daily_return_sd=0., sharpe_status='undefined_sharpe', sharpe_reason='zero_daily_return_sd')
        with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_invalid_undefined_evidence_is_error(self):
        for sd in (None, 1e-30, -1., float('nan')):
            m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['training'].update(sharpe=None, daily_return_sd=sd, sharpe_status='undefined_sharpe', sharpe_reason='zero_daily_return_sd')
            with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_cagr_tolerance_only(self):
        m, o = fixture(); m[h.ARMS[1]]['same_close_zero']['training']['cagr'] = .505 - 5e-13
        self.assertTrue(h.select_first88(m, o)['guards'][h.ARMS[1]]['close'])
        m[h.ARMS[1]]['same_close_zero']['training']['sharpe'] = 1.65 - 5e-13
        self.assertFalse(h.select_first88(m, o)['guards'][h.ARMS[1]]['close'])

    def test_selection_tolerance_fixed_arm_order(self):
        m, o = fixture(); m[h.ARMS[2]]['same_close_zero']['training']['cagr'] = .52 + 5e-13
        self.assertEqual(h.select_first88(m, o)['primary'], h.ARMS[1])

    def test_malformed_decision_rejected(self):
        for names in (['A'], ['A', 'A'], ['A', 'B', 'C', 'D'], ['A', ''], 'ABC'):
            m, o = fixture(); o[h.ARMS[1]][1]['symbols'] = names
            with self.assertRaises(ValueError): h.select_first88(m, o)

    def test_inputs_unchanged(self):
        m, o = fixture(); before = copy.deepcopy((m, o)); h.select_first88(m, o)
        self.assertEqual((m, o), before)


if __name__ == '__main__': unittest.main()
