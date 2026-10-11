import unittest
from audit_primary_chain import audit

class ChainAuditTests(unittest.TestCase):
    def test_invalid_dates_fail_even_if_outside_window(self):
        for day in ['NaT', '0001-03-24', '', '2025-02-30']:
            with self.subTest(day=day), self.assertRaises(ValueError):
                audit(['A'], [dict(effective_session=day, action='addition', ticker_as_announced_or_effective='B')])

    def test_open_interval_uses_requested_end(self):
        result = audit(['A'], [], end='2020-01-07')
        self.assertEqual(result['provisional_intervals'][0]['end_exclusive'], '2020-01-08')

    def test_missing_delete_does_not_grant_admission(self):
        result = audit(['A'], [dict(effective_session='2020-01-02', action='deletion', ticker_as_announced_or_effective='B')])
        self.assertEqual(result['errors'][0]['error'], 'delete_absent')
        self.assertFalse(result['admission'])
