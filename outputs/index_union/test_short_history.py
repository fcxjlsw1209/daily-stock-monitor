import unittest
from audit_short_history import assess
from research_calendar import sessions

class ShortHistoryTests(unittest.TestCase):
    def test_exact_boundary(self):
        dates=sessions('2024-01-02','2025-03-01')
        self.assertTrue(assess(str(dates[0].date()),str(dates[251].date()))['new_entry_impact_proven_zero'])
        self.assertFalse(assess(str(dates[0].date()),str(dates[252].date()))['new_entry_impact_proven_zero'])
    def test_no_waiver_of_inherited_claims(self):
        self.assertFalse(assess('2026-06-15','2026-10-06')['corporate_action_entitlements_waived'])
    def test_non_sessions_are_not_observations(self):
        self.assertEqual(assess('2026-07-02','2026-07-06')['maximum_possible_observations'],2)

if __name__=='__main__':unittest.main()
