import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from audit_scope import candidate_issues, resolve, verify_resolution, verify_universe
from research_calendar import market_calendar


class CandidateScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = market_calendar(start='2010-01-01', end='2030-12-31')

    def fixture(self):
        old = dict(event_id='A_2024-01-25', symbol='A', status='ready',
                   known_by_at='2024-01-26T21:00:00+00:00', publication_date='2024-01-25',
                   fiscal_year=2023, fiscal_quarter=4, official_actual_period_end='2023-12-31')
        later = dict(event_id='A_2024-07-25', symbol='A', status='ready',
                     known_by_at='2024-07-26T20:00:00+00:00', publication_date='2024-07-25',
                     fiscal_year=2024, fiscal_quarter=2, official_actual_period_end='2024-06-30')
        row = dict(signal_date='2024-02-29', symbol='A', event_id=old['event_id'], status='verified_event_available')
        return row, old, later

    def test_recent_event_does_not_hide_missing_quarter(self):
        row, old, later = self.fixture()
        issues = candidate_issues('monthly', [row], [old, later], [], self.calendar)
        self.assertEqual([r['kind'] for r in issues], ['missing_intervening_quarter'])
        self.assertIsNone(resolve(issues, [], [])[0]['resolution'])

    def test_future_quarter_boundary_can_close_audit_but_not_after_publication(self):
        row, old, later = self.fixture()
        boundary = dict(symbol='A', previous_fiscal_year=2023, previous_fiscal_quarter=4,
                        next_fiscal_year=2024, next_fiscal_quarter=1, release_date='2024-04-25',
                        source_url='https://example.com/original', source_representation='test_fixture')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source-response.txt'
            path.write_text('Fixture first-quarter publication on April25,2024')
            boundary.update(reason='original release date', evidence=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            issues = candidate_issues('monthly', [row], [old, later], [], self.calendar)
            self.assertIsNotNone(resolve(issues, [], [boundary])[0]['resolution'])
            row['signal_date'] = '2024-04-30'
            issues = candidate_issues('monthly', [row], [old, later], [], self.calendar)
            self.assertTrue(any(r['resolution'] is None for r in resolve(issues, [], [boundary])))
            path.write_text('changed source')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                resolve(issues, [], [boundary])
        self.assertEqual(row['event_id'], old['event_id'])

    def test_boundary_without_source_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source evidence'):
            resolve([], [], [dict(reason='unsubstantiated future release')])

    def test_independent_frozen_universe_check_catches_synchronized_omission(self):
        meta = dict(dates=['2024-02-29'], symbols=[['A', 'QQQ']])
        rows = [dict(signal_date='2024-02-29', symbol='A', status='verified_event_available', event_id='A_1'),
                dict(signal_date='2024-02-29', symbol='QQQ', status='not_applicable_etf', event_id=None)]
        verify_universe(rows, meta, {'QQQ'})
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            verify_universe(rows[:1], meta, {'QQQ'})
        rows[0]['status'] = 'not_applicable_etf'
        with self.assertRaisesRegex(ValueError, 'ETF identity'):
            verify_universe(rows, meta, {'QQQ'})

    def test_wrong_or_old_selected_event_is_rejected(self):
        row, old, later = self.fixture()
        row['signal_date'] = '2024-07-31'
        with self.assertRaisesRegex(ValueError, 'latest actually known'):
            candidate_issues('monthly', [row], [old, later], [], self.calendar)

    def test_ipo_missing_and_etf_remain_separate(self):
        missing = dict(signal_date='2024-02-29', symbol='NEW', event_id=None, status='missing_event_history')
        etf = dict(missing, symbol='ETF', status='not_applicable_etf')
        issues = candidate_issues('monthly', [missing, etf], [], [], self.calendar)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['kind'], 'missing_prior_event')

    def test_vendor_hint_remains_review_not_an_event(self):
        row, old, later = self.fixture()
        hint = dict(symbol='A', provider_reported_date='2024-02-20')
        issues = candidate_issues('monthly', [row], [old, later], [hint], self.calendar)
        self.assertIn('unmatched_vendor_hint', [r['kind'] for r in issues])
        self.assertEqual(row['event_id'], old['event_id'])

    def test_explicit_resolution_fails_when_evidence_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text('actual date evidence')
            resolution = dict(reason='source-verified boundary', evidence=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
            verify_resolution(resolution)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_resolution(resolution)

    def test_resolution_cannot_waive_a_different_prior_or_vendor_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text('source-verified date')
            ref = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            issue = dict(issue_id='unmatched_vendor_hint|monthly|2024-02-29|A',
                         kind='unmatched_vendor_hint', signal_date='2024-02-29',
                         prior_release='2024-01-25', provider_dates=['2024-02-20'])
            resolution = dict(issue_id=issue['issue_id'], reason='wrong vendor date', evidence=[ref],
                              expected_prior_release='2024-01-25', expected_provider_dates=['2024-02-20'])
            self.assertIsNotNone(resolve([issue], [resolution], [])[0]['resolution'])
            with self.assertRaisesRegex(ValueError, 'different prior'):
                resolve([issue], [dict(resolution, expected_prior_release='2024-01-24')], [])
            with self.assertRaisesRegex(ValueError, 'different vendor'):
                resolve([issue], [dict(resolution, expected_provider_dates=['2024-02-21'])], [])

    def test_resolution_cannot_cross_publication_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text('future-quarter publication')
            ref = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            issue = dict(issue_id='long_reporting_interval|monthly|2024-03-01|A',
                         kind='long_reporting_interval', signal_date='2024-03-01',
                         prior_release='2023-10-25')
            resolution = dict(issue_id=issue['issue_id'], reason='not yet reported', evidence=[ref],
                              next_release='2024-03-01')
            with self.assertRaisesRegex(ValueError, 'next release boundary'):
                resolve([issue], [resolution], [])


if __name__ == '__main__':
    unittest.main()
