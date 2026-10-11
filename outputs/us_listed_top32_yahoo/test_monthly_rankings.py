import unittest
from pathlib import Path
from collect_monthly_rankings import parse_ranking, listing_status


class PublishedRankingTests(unittest.TestCase):
    def test_redirect_to_different_date_is_rejected(self):
        # Actual public request for 2020-12-31 returned 2023-01-04.
        body = (Path(__file__).parent/'monthly_rank_sources/cmc_2020_12_31.html').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Date mismatch'):
            parse_ranking(body, '2020-12-31')
        rows = parse_ranking(body, '2023-01-04')
        self.assertEqual(rows[0]['vendor_symbol'], 'AAPL')
        self.assertEqual(len(rows), 200)

    def test_ipo_and_explicit_delisting_are_respected(self):
        row = dict(symbol='X',exchange='NYSE',ipoDate='2023-02-01',delistingDate='2024-02-01',
                   evidence_files=['2024-01-04_active.csv'])
        self.assertFalse(listing_status('X','2023-01-31',[row])[0])
        self.assertTrue(listing_status('X','2023-02-28',[row])[0])
        self.assertFalse(listing_status('X','2024-02-02',[row])[0])

    def test_conflicting_registry_records_are_flagged(self):
        active = dict(symbol='X',exchange='NASDAQ',ipoDate='2023-01-01',delistingDate='null',
                      evidence_files=['2026-10-05_active.csv'])
        ended = dict(active,delistingDate='2023-01-01',evidence_files=['2023-01-04_delisted.csv'])
        listed, conflict = listing_status('X','2023-02-01',[active,ended])
        self.assertTrue(listed)
        self.assertTrue(conflict)

    def test_old_active_snapshot_does_not_prove_current_listing(self):
        row = dict(symbol='X',exchange='NYSE',ipoDate='2000-01-01',delistingDate='null',
                   evidence_files=['2020-01-04_active.csv'])
        self.assertFalse(listing_status('X','2023-01-31',[row])[0])


if __name__ == '__main__':
    unittest.main()
