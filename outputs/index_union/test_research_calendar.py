import unittest
import pandas as pd
from research_calendar import sessions
from normalize_new_prices import session_mask
from membership import entry_session
class CalendarTests(unittest.TestCase):
    def test_carter_closure(self):
        self.assertEqual(list(sessions('2025-01-08','2025-01-10').strftime('%Y-%m-%d')),['2025-01-08','2025-01-10'])
    def test_delayed_entry_skips_closure(self):
        days=list(sessions('2025-01-06','2025-01-13').strftime('%Y-%m-%d'))
        self.assertEqual(entry_session('2025-01-08T09:30:00-05:00','2025-01-07T17:00:00-05:00',days),'2025-01-10')
    def test_closure_price_invalid(self):
        self.assertFalse(session_mask(pd.Series(['2025-01-09'])).iloc[0])
    def test_single_regular_session(self):
        self.assertEqual(len(sessions('2025-01-10','2025-01-10')),1)
if __name__=='__main__':unittest.main()
