import unittest
from io import StringIO
import pandas as pd
from normalize_new_prices import validate_frame, read_quantquote, session_mask


class Bars(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(date='2020-01-02',open=10,high=11,low=9,close=10,
                                  adj_close=8,volume=100)])

    def test_valid(self):
        self.assertTrue(validate_frame(self.frame()).iloc[0])

    def test_duplicate_dates_reject_both(self):
        d = self.frame()
        self.assertFalse(validate_frame(pd.concat([d,d], ignore_index=True)).any())

    def test_impossible_price_or_volume(self):
        for field, value in [('open',12),('low',11),('volume',-1),('adj_close',float('nan'))]:
            d=self.frame();d.loc[0,field]=value
            self.assertFalse(validate_frame(d).iloc[0], field)


class QuantQuoteFormat(unittest.TestCase):
    def test_preserves_fractional_adjusted_volume(self):
        d=read_quantquote(StringIO('20100104,0,10,11,9,10.5,123.456\n'))
        self.assertEqual(d.date.iloc[0], '2010-01-04')
        self.assertEqual(d.volume.iloc[0], 123.456)
    def test_rejects_intraday_and_extra_columns(self):
        for line in ['20100104,930,10,11,9,10,100', '20100104,0,10,11,9,10,100,5']:
            with self.assertRaises(ValueError):read_quantquote(StringIO(line))
    def test_rejects_impossible_calendar_date(self):
        with self.assertRaises(ValueError):read_quantquote(StringIO('20100230,0,10,11,9,10,100'))


class ExchangeSessions(unittest.TestCase):
    def test_single_day_and_holiday(self):
        self.assertTrue(session_mask(pd.Series(['2020-01-02'])).iloc[0])
        self.assertFalse(session_mask(pd.Series(['2020-01-01'])).iloc[0])
    def test_invalid_dates_fail_closed(self):
        self.assertFalse(session_mask(pd.Series(['not-a-date'])).any())


if __name__ == '__main__':
    unittest.main()
