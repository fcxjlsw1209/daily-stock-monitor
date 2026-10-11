import unittest
import pandas as pd
from collect_azure_prices import regular_daily


class SessionAggregationTests(unittest.TestCase):
    def test_early_close_excludes_extended_hours(self):
        schedule = pd.DataFrame({'market_open': ['2010-11-26T14:30:00Z'], 'market_close': ['2010-11-26T18:00:00Z']}, index=pd.to_datetime(['2010-11-26']))
        raw = pd.DataFrame([['2010-11-26 09:00', 999, 999, 999, 999, 1], ['2010-11-26 09:30', 10, 12, 9, 11, 100], ['2010-11-26 12:30', 11, 13, 10, 12, 200], ['2010-11-26 13:00', 999, 999, 999, 999, 1]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        result = regular_daily(raw, schedule).iloc[0]
        self.assertEqual((result.open, result.high, result.low, result.close, result.volume), (10, 13, 9, 12, 300))
        self.assertFalse(result.official_open_close_verified)

    def test_summer_timezone_and_weekend(self):
        schedule = pd.DataFrame({'market_open': ['2010-03-15T13:30:00Z'], 'market_close': ['2010-03-15T20:00:00Z']}, index=pd.to_datetime(['2010-03-15']))
        raw = pd.DataFrame([['2010-03-15 09:30', 10, 11, 9, 10, 100], ['2010-03-14 09:30', 99, 99, 99, 99, 1]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        result = regular_daily(raw, schedule)
        self.assertEqual(result.date.tolist(), ['2010-03-15'])
        self.assertEqual(result.iloc[0].first_bar.hour, 13)

    def test_duplicate_bar_rejected(self):
        raw = pd.DataFrame([['2010-03-15 09:30', 10, 11, 9, 10, 100]] * 2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        with self.assertRaises(ValueError):
            regular_daily(raw, pd.DataFrame())
