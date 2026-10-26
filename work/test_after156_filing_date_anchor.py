import unittest
from after156_filing_date_anchor import anchor, checked_calendar, clock, iso_day


class DateAnchorTests(unittest.TestCase):
    def setUp(self):
        # Deliberately excludes the weekend and a holiday, no fabricated fills.
        self.sessions = checked_calendar(['2020-07-01', '2020-07-02',
            '2020-07-06', '2020-07-07', '2020-07-08'])

    def row(self, filed='2020-07-02', accepted='2020-07-02T13:00:00Z'):
        return dict(filingDate=filed, acceptanceDateTime=accepted,
                    accessionNumber='0000000001-20-000001', form='10-Q', reportDate='2020-06-30')

    def test_preopen_still_delayed(self):
        self.assertEqual(anchor(self.row(), self.sessions)['proposed_anchor'], '2020-07-06')

    def test_postclose_holiday(self):
        x = anchor(self.row(accepted='2020-07-02T21:01:00Z'), self.sessions)
        self.assertEqual(x['proposed_anchor'], '2020-07-06')

    def test_utc_rollover_retains_both_dates(self):
        x = anchor(self.row(accepted='2020-07-03T02:00:00Z'), self.sessions)
        self.assertEqual(x['date_boundary'], '2020-07-03')
        self.assertEqual(x['raw_filing_date'], '2020-07-02')
        self.assertEqual(x['proposed_anchor'], '2020-07-06')

    def test_positive_offset_uses_raw_date_too(self):
        x = anchor(self.row('2020-07-06', '2020-07-07T01:00:00+14:00'), self.sessions)
        self.assertEqual(x['date_boundary'], '2020-07-07')
        self.assertEqual(x['proposed_anchor'], '2020-07-08')

    def test_negative_offset_uses_utc_date_too(self):
        x = anchor(self.row('2020-07-06', '2020-07-06T23:30:00-04:00'), self.sessions)
        self.assertEqual(x['date_boundary'], '2020-07-07')
        self.assertEqual(x['proposed_anchor'], '2020-07-08')

    def test_no_naive_timestamp(self):
        with self.assertRaises(ValueError): clock('2020-07-02T13:00:00')

    def test_no_large_clock_gap(self):
        with self.assertRaises(ValueError):
            anchor(self.row(accepted='2020-07-07T13:00:00Z'), self.sessions)

    def test_no_future_report(self):
        r=self.row();r['reportDate']='2020-07-06'
        with self.assertRaises(ValueError): anchor(r, self.sessions)

    def test_no_truncated_right_calendar(self):
        with self.assertRaises(ValueError):
            anchor(self.row('2020-07-08','2020-07-08T15:00:00Z'),self.sessions)

    def test_no_truncated_left_calendar(self):
        with self.assertRaises(ValueError):
            anchor(self.row('2020-06-30','2020-06-30T15:00:00Z'),self.sessions)

    def test_no_duplicate_sessions(self):
        with self.assertRaises(ValueError): checked_calendar(['2020-07-01','2020-07-01'])

    def test_no_unsorted_sessions(self):
        with self.assertRaises(ValueError): checked_calendar(['2020-07-02','2020-07-01'])

    def test_no_datetime_as_date(self):
        with self.assertRaises(ValueError): iso_day('2020-07-01T00:00:00')

    def test_winter_offset_bound(self):
        sessions=checked_calendar(['2020-01-30','2020-01-31','2020-02-03','2020-02-04'])
        r=self.row('2020-01-31','2020-02-01T01:00:00Z');r['reportDate']='2019-12-31'
        x=anchor(r,sessions)
        self.assertEqual(x['proposed_anchor'],'2020-02-03')
        self.assertFalse(x['exact_acceptance_instant_verified'])


if __name__ == '__main__': unittest.main()
