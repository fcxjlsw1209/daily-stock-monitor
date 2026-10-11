import unittest
import pandas as pd
from audit_qstk_prices import clean


class ArchiveTests(unittest.TestCase):
    def row(self, **changes):
        row = {'Date': '2010-01-04', 'Open': 10, 'High': 12, 'Low': 9,
               'Close': 11, 'Volume': 100, 'Adj Close ': 8}
        row.update(changes)
        return row

    def test_empty_and_invalid_bars_not_filled(self):
        rows = pd.DataFrame([self.row(), self.row(Date='2010-01-05', Close=float('nan')),
                             self.row(Date='2010-01-06', Open=20)])
        d = clean(rows, {'2010-01-04','2010-01-05','2010-01-06'})
        self.assertEqual(d.quarantine.tolist(), [False, True, True])
        self.assertEqual(d.iloc[0].close, 11)
        self.assertEqual(d.iloc[0].adj_close, 8)
        self.assertFalse(d.identity_and_adjustments_verified.any())

    def test_non_session_and_terminal_quarantined(self):
        d = clean(pd.DataFrame([self.row(Date='2010-01-03'), self.row()]),
                  {'2010-01-04'}, '2010-01-04')
        self.assertTrue(d.quarantine.all())

    def test_duplicate_dates_not_silently_selected(self):
        d = clean(pd.DataFrame([self.row(),self.row(Close=10)]), {'2010-01-04'})
        self.assertTrue(d.quarantine.all())
