"""Synthetic, in-memory tests; no observer/account entrypoints are executed."""
from copy import deepcopy
import json
import sqlite3
import unittest

import pandas as pd

from price_revision_guard import REGISTRY_KEY, validate_entry_basis


class PriceRevisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.position = dict(entry_date='2026-10-15', entry_price=243.61000061035156,
                             qty=41.47840440085118, cost=10109.60639846847)
        self.frame = pd.DataFrame({'Open': [243.11000061035156, 240., 228.],
                                   'Stock Splits': [0., 0., 0.]},
                                  index=pd.to_datetime(['2026-10-15', '2026-10-16', '2026-10-19']))
        self.manifest = dict(as_of='2026-10-19', snapshot_id='20261019T202830213985Z')
        self.record = dict(symbol='CRWD', entry_date='2026-10-15',
                           recorded_entry_price=243.61000061035156,
                           accepted_provider_open=243.11000061035156,
                           verified_snapshot='20261019T202830213985Z',
                           source_evidence_sha256='a' * 64)

    def tearDown(self):
        self.db.close()

    def register(self, records=None):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',
                        (REGISTRY_KEY, json.dumps([self.record] if records is None else records)))
        self.db.commit()

    def check(self):
        return validate_entry_basis(self.db, 'CRWD', self.position, self.frame, self.manifest)

    def test_unchanged_passes_without_registry(self):
        self.frame.iloc[0, 0] = self.position['entry_price']
        self.assertIsNone(self.check())

    def test_exact_verified_half_dollar_revision_and_no_mutation(self):
        self.register(); position = deepcopy(self.position); frame = self.frame.copy(deep=True)
        before = self.db.total_changes
        self.db.execute('PRAGMA query_only=ON')
        result = self.check()
        self.assertEqual(result['accepted_provider_open'], self.record['accepted_provider_open'])
        self.assertEqual(result['kind'], 'verified_entry_price_revision')
        self.assertEqual(self.position, position); pd.testing.assert_frame_equal(self.frame, frame)
        self.assertEqual(self.db.total_changes, before)
        json.dumps(result, allow_nan=False)

    def test_unregistered_half_dollar_revision_rejected(self):
        with self.assertRaises(ValueError): self.check()

    def test_new_snapshot_same_verified_price_passes(self):
        self.register(); self.manifest['snapshot_id'] = '20261019T210000000000Z'
        self.assertEqual(self.check()['observed_snapshot'], self.manifest['snapshot_id'])

    def test_other_entry_date_not_whitelisted(self):
        self.register(); self.position['entry_date'] = '2026-10-16'
        self.frame.loc[pd.Timestamp('2026-10-16'), 'Open'] = self.record['accepted_provider_open']
        with self.assertRaises(ValueError): self.check()

    def test_other_recorded_price_not_whitelisted(self):
        self.register(); self.position['entry_price'] += 1
        with self.assertRaises(ValueError): self.check()

    def test_another_small_revision_is_not_tolerated(self):
        self.register(); self.frame.iloc[0, 0] += .001
        with self.assertRaises(ValueError): self.check()

    def test_unregistered_tiny_difference_retains_legacy_tolerance(self):
        self.frame.iloc[0, 0] = self.position['entry_price'] + .000001
        self.assertIsNone(self.check())

    def test_unregistered_threshold_compatibility_in_both_directions(self):
        for scale in [.9991, 1.0009]:
            self.frame.iloc[0, 0] = self.position['entry_price'] * scale
            self.assertIsNone(self.check())
        for scale in [.9989, 1.0011]:
            self.frame.iloc[0, 0] = self.position['entry_price'] * scale
            with self.assertRaises(ValueError): self.check()

    def test_registered_entry_cannot_use_legacy_tolerance_for_another_price(self):
        self.register(); self.frame.iloc[0, 0] = self.position['entry_price'] + .000001
        with self.assertRaises(ValueError): self.check()

    def test_registered_provider_reversion_to_exact_original_is_allowed(self):
        self.register(); self.frame.iloc[0, 0] = self.position['entry_price']
        self.assertIsNone(self.check())

    def test_two_for_one_price_change_rejected(self):
        self.register(); self.frame.iloc[0, 0] = self.position['entry_price'] / 2
        with self.assertRaises(ValueError): self.check()

    def test_split_blocks_revision_exception_but_not_original_basis(self):
        self.register(); self.frame.iloc[1, 1] = 2
        with self.assertRaises(ValueError): self.check()
        self.frame.iloc[0, 0] = self.position['entry_price']
        self.assertIsNone(self.check())

    def test_nonfinite_nonpositive_or_boolean_values_rejected(self):
        self.register()
        for value in [float('nan'), float('inf'), 0., -1., True]:
            with self.subTest(value=value):
                self.position['entry_price'] = value
                with self.assertRaises(ValueError): self.check()
        self.position['entry_price'] = self.record['recorded_entry_price']
        self.frame.iloc[0, 0] = float('nan')
        with self.assertRaises(ValueError): self.check()

    def test_missing_or_nonfinite_split_metadata_rejected(self):
        self.register(); self.frame.iloc[1, 1] = float('nan')
        with self.assertRaises(ValueError): self.check()
        self.frame = self.frame.drop(columns=['Stock Splits'])
        with self.assertRaises(ValueError): self.check()

    def test_conflicting_registry_rejected(self):
        alternate = dict(self.record, accepted_provider_open=240.)
        self.register([self.record, alternate])
        with self.assertRaises(ValueError): self.check()

    def test_other_symbol_is_not_whitelisted(self):
        self.record['symbol'] = 'OTHER'; self.register()
        with self.assertRaises(ValueError): self.check()

    def test_registry_schema_and_evidence_hash_checked(self):
        self.record['source_evidence_sha256'] = 'not-a-hash'; self.register()
        with self.assertRaises(ValueError): self.check()
        self.record['source_evidence_sha256'] = 'a' * 64
        self.record['undocumented_override'] = True; self.register()
        with self.assertRaises(ValueError): self.check()

    def test_duplicate_json_keys_and_nonfinite_registry_rejected(self):
        self.register()
        for payload in ['[{"symbol":"CRWD","symbol":"OTHER"}]', '[NaN]']:
            self.db.execute('UPDATE meta SET value=? WHERE key=?', (payload, REGISTRY_KEY))
            with self.assertRaises(ValueError): self.check()

    def test_snapshot_dates_and_duplicates_checked(self):
        self.register(); self.manifest['as_of'] = '2026-10-16'
        with self.assertRaises(ValueError): self.check()
        self.manifest['as_of'] = '2026-10-19'
        self.frame.index = pd.to_datetime(['2026-10-15', '2026-10-15', '2026-10-19'])
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__':
    unittest.main()
