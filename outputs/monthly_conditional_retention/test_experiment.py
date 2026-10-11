import copy
import unittest
import numpy as np
import pandas as pd
import policy_study as x


class MonthlyPolicyTests(unittest.TestCase):
    def setup_context(self):
        return dict(symbols=['A', 'B', 'C', 'D'], ids=dict(A=0, B=1, C=2, D=3),
                    dates=['2020-12-31', '2021-01-29', '2021-02-26'],
                    opening_dates=['2021-01-04', '2021-02-01', '2021-03-01'],
                    eligible=np.ones((3, 4), bool),
                    close=np.array([[100.] * 4, [110., 105., 102., 115.], [112., 101., 104., 113.]]),
                    opening=np.array([[103.] * 4, [111., 106., 105., 116.], [113., 102., 105., 114.]]),
                    peak=np.array([[100.] * 4, [120., 110., 108., 118.], [119., 106., 111., 117.]]),
                    sigma=np.full((3, 4), .08), trend=np.ones(3, int), highvol=np.zeros(3, int),
                    periods=[dict(start=d, end=d, holdings=['A', 'B', 'C']) for d in ['2020-12-31', '2021-01-29', '2021-02-26']])

    def select(self, score, state, buffer=5, gap=0., policy=None, close=None, peak=None):
        n = len(score)
        return x.select_month(np.array(score), np.ones(n, bool), state,
                              np.full(n, 100.) if close is None else np.array(close),
                              np.full(n, 100.) if peak is None else np.array(peak),
                              np.full(n, .08), np.full(n, .8), 1, 0,
                              buffer, gap, policy or dict(name='none'))

    def test_buffer_keeps_incumbent_and_positive_margin_can_replace(self):
        state = {0: dict(units=10, cost=100, peak=100, entry_date='2020-12-31')}
        score = [.10, .14, .13, .12]
        kept, _ = self.select(score, state, buffer=4)
        self.assertIn(0, kept)
        fresh, _ = self.select(score, state, buffer=3)
        self.assertNotIn(0, fresh)
        small, _ = self.select(score, state, buffer=4, gap=.03)
        self.assertIn(0, small)
        replaced, _ = self.select(score, state, buffer=4, gap=.01)
        self.assertNotIn(0, replaced)

    def test_intramonth_loss_does_not_trigger_monthend_recovered_stop(self):
        # Intramonth low does not enter the execution engine. Only current close
        # relative to cost/peak can trigger a monthly decision.
        state = {0: dict(units=10, cost=100, peak=110, entry_date='2020-12-31')}
        picks, events = self.select([.4, .3, .2, .1], state, policy=dict(name='fixed', loss=.15, profit=.50))
        self.assertIn(0, picks)
        self.assertEqual(events[0]['reason'], 'retained')
        picks, events = self.select([.4, .3, .2, .1], state, policy=dict(name='fixed', loss=.15, profit=.50), close=[80, 100, 100, 100])
        self.assertNotIn(0, picks)
        self.assertEqual(events[0]['reason'], 'monthend_stop_loss')

    def test_exit_blocks_immediate_repurchase(self):
        state = {0: dict(units=10, cost=50, peak=100, entry_date='2020-12-31')}
        picks, events = self.select([.9, .3, .2, .1], state, policy=dict(name='fixed', loss=.15, profit=.5))
        self.assertEqual(picks, [1, 2, 3])
        self.assertEqual(events[0]['reason'], 'monthend_take_profit')

    def test_adaptive_bar_depends_on_volatility_regime_and_strength(self):
        policy = x.EXIT_POLICIES[3]
        normal = x.exit_thresholds(policy, .05, .6, 1, 0)
        volatile = x.exit_thresholds(policy, .08, .6, 1, 0)
        weak = x.exit_thresholds(policy, .05, .4, 0, 1)
        strong = x.exit_thresholds(policy, .05, .8, 1, 0)
        self.assertGreater(volatile['loss'], normal['loss'])
        self.assertLess(weak['loss'], normal['loss'])
        self.assertGreater(strong['profit'], normal['profit'])

    def test_average_cost_additions_and_trims(self):
        old = {0: dict(units=10., cost=100., peak=120., entry_date='2020-12-31')}
        add = x.rebalance_state(old, [0], np.array([200.]), 4000., '2021-01-29', np.array([180.]))
        self.assertEqual(add[0]['units'], 20)
        self.assertEqual(add[0]['cost'], 150)
        self.assertEqual(add[0]['peak'], 200)
        trimmed = x.rebalance_state(add, [0], np.array([250.]), 2500., '2021-02-26', np.array([240.]))
        self.assertEqual(trimmed[0]['units'], 10)
        self.assertEqual(trimmed[0]['cost'], 150)
        self.assertEqual(trimmed[0]['entry_date'], '2020-12-31')

    def test_no_author_state_or_future_prices_in_earlier_decisions(self):
        c = self.setup_context()
        score = np.array([[.4, .3, .2, .1], [.1, .2, .3, .4], [.4, .3, .2, .1]])
        strength = np.full_like(score, .8)
        config = dict(buffer=4, gap=0., exit=x.EXIT_POLICIES[3])
        before = x.simulate(c, score, strength, config, detailed=True)
        altered = copy.deepcopy(c)
        for p in altered['periods']:
            p['holdings'] = ['D', 'B', 'A']
        after = x.simulate(altered, score, strength, config, detailed=True)
        self.assertEqual(before, after)
        altered['close'][2] *= 10
        altered['peak'][2] *= 10
        changed = x.simulate(altered, score, strength, config, detailed=True)
        self.assertEqual(before['events'][:2], changed['events'][:2])

    def test_open_control_uses_actual_open_costs_not_primary_costs(self):
        c = self.setup_context()
        score = np.tile([.4, .3, .2, .1], (3, 1))
        config = dict(buffer=3, gap=0., exit=dict(name='none'))
        close = x.simulate(c, score, np.full_like(score, .8), config, count=1, detailed=True)
        opened = x.simulate(c, score, np.full_like(score, .8), config, count=1, execution='open', detailed=True)
        self.assertEqual(close['choices'], opened['choices'])
        self.assertEqual(close['state'][0]['cost'], 100.)
        self.assertEqual(opened['state'][0]['cost'], 103.)

    def test_fewer_than_two_eligible_means_cash(self):
        c = self.setup_context()
        c['eligible'][:] = False
        c['eligible'][:, 0] = True
        score = np.tile([.4, .3, .2, .1], (3, 1))
        result = x.simulate(c, score, np.full_like(score, .8), dict(buffer=3, gap=0., exit=dict(name='none')))
        self.assertEqual(result['choices'], [[], [], []])
        self.assertEqual(result['cash'], 100000.)


if __name__ == '__main__':
    unittest.main()
