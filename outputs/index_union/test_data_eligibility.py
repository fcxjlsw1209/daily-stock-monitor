import unittest
import numpy as np
import pandas as pd
from data_eligibility import entry_eligibility
from dynamic_features import features
from test_dynamic_features import fixture
from union_execution import engine, execute, ACCOUNTS


class AvailableDataTests(unittest.TestCase):
    def test_missing_session_excluded_from_every_strategy_rank(self):
        _, _, args = fixture()
        args['volume'].iloc[-1, 0] = np.nan
        for result in features(**args).values():
            self.assertFalse(result['mask'].iloc[-1, 0])
            self.assertTrue(pd.isna(result['score'].iloc[-1, 0]))
            self.assertTrue(result['score'].iloc[-1, 1:].notna().all())

    def test_infinite_or_inconsistent_open_is_not_eligible(self):
        _, _, args = fixture()
        args['opening'].iloc[-1, 0] = np.inf
        args['opening'].iloc[-1, 1] = args['high'].iloc[-1, 1] * 2
        for result in features(**args).values():
            self.assertFalse(result['mask'].iloc[-1, :2].any())
            self.assertTrue(result['score'].iloc[-1, :2].isna().all())

    def test_future_gap_does_not_change_past_signals(self):
        _, _, args = fixture()
        before = features(**args)
        args['volume'].iloc[-1, 0] = np.nan
        after = features(**args)
        for name in before:
            for field in before[name]:
                pd.testing.assert_frame_equal(before[name][field].iloc[:-1], after[name][field].iloc[:-1])

    def test_current_gap_does_not_delete_history_or_future_eligibility(self):
        _, _, a = fixture()
        a['volume'].iloc[-2, 0] = np.nan
        original = a['adj'].copy()
        eligible = entry_eligibility(a['eligible'], a['adj'], a['opening'], a['high'], a['low'], a['close'], a['volume'])
        self.assertFalse(eligible.iloc[-2, 0])
        self.assertTrue(eligible.iloc[-1, 0])
        pd.testing.assert_frame_equal(original, a['adj'])

    def decision(self, dates):
        return dict(session=str(dates[1].date()), signal_date=str(dates[0].date()), sells=[],
                    buys=[dict(symbol='A', budget=10000., score=.9, atr=2.)])

    def test_missing_open_cancels_entry_without_fees_or_replacement_all_six(self):
        dates = pd.bdate_range('2020-01-01', periods=3)
        for name in ACCOUNTS:
            state = engine(name).initial_state()
            decision = self.decision(dates)
            result = execute(name, state, decision, {}, dates)
            self.assertEqual(result[0]['cash'], 100000.)
            self.assertEqual(result[0]['positions'], {})
            self.assertEqual(result[0]['fees'], 0)
            self.assertEqual(len(result[-1]), 1)
            self.assertEqual(len(decision['buys']), 1)

    def test_missing_close_does_not_retroactively_cancel_valid_open(self):
        dates = pd.bdate_range('2020-01-01', periods=3)
        for name in ACCOUNTS:
            with self.assertRaises(ValueError):
                execute(name, engine(name).initial_state(), self.decision(dates),
                        {'A': dict(open=100., close=np.nan)}, dates)

    def test_missing_held_price_is_not_excluded_or_written_off(self):
        dates = pd.bdate_range('2020-01-01', periods=3)
        for name in ACCOUNTS:
            state, *_ = execute(name, engine(name).initial_state(), self.decision(dates),
                                {'A': dict(open=100., close=100.)}, dates)
            decision = dict(session=str(dates[2].date()), sells=[], buys=[])
            with self.assertRaises((KeyError, ValueError)):
                execute(name, state, decision, {}, dates)
            self.assertIn('A', state['positions'])
            self.assertAlmostEqual(state['equity'], 100000. - state['fees'])

if __name__ == '__main__':
    unittest.main()
