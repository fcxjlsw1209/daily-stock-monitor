import unittest
import pandas as pd
import exchange_calendars as xc

from research_calendar import market_calendar, provenance
from event_schema import normalize_event, reaction_features


class CalendarTests(unittest.TestCase):
    def test_only_evidenced_session_removed_without_global_mutation(self):
        raw=xc.get_calendar('XNYS',start='2010-01-01',end='2030-12-31')
        before=raw.sessions.copy()
        cal=market_calendar(start='2010-01-01',end='2030-12-31')
        self.assertEqual(list(before.difference(cal.sessions)),[pd.Timestamp('2025-01-09')])
        self.assertEqual(len(cal.sessions.difference(before)),0)
        self.assertTrue(raw.sessions.equals(before))
        self.assertEqual(cal.session_close('2024-11-29'),pd.Timestamp('2024-11-29T18:00Z'))
        self.assertFalse(provenance()['global_registry_modified'])

    def event(self,day):
        return normalize_event(dict(symbol='TEST',official=dict(verified=True,
            source_url='https://example.com/release',actual_period_end='2024-12-31',
            publication_date=day,publication_timezone='America/New_York')))

    def test_date_only_before_or_on_closure_known_next_open_close(self):
        for day in ('2025-01-08','2025-01-09'):
            event=self.event(day)
            self.assertEqual(event['available_session'],'2025-01-10')
            self.assertEqual(pd.Timestamp(event['known_by_at']),pd.Timestamp('2025-01-10T21:00Z'))

    def test_reaction_counts_real_sessions_not_missing_holiday_bar(self):
        cal=market_calendar()
        event=self.event('2025-01-07')
        sessions=cal.sessions_in_range('2025-01-08','2025-01-14')
        bars=pd.DataFrame({'close':[100.,101.,102.,103.]},index=sessions)
        result=reaction_features(event,bars,'2025-01-14T21:00Z',horizon=3)
        self.assertEqual(result['status'],'ready')
        self.assertEqual(result['end_session'],'2025-01-14')
        self.assertAlmostEqual(result['post_known_close_to_close_return'],.03)
        early=reaction_features(event,bars,'2025-01-13T21:00Z',horizon=3)
        self.assertEqual(early['status'],'waiting_for_closed_window')


if __name__=='__main__':unittest.main()
