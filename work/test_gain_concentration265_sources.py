"""Synthetic-only checks of the265 metadata wrapper, with no real price reads."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gain_concentration265_sources as source


class MetadataTests(unittest.TestCase):
    def parent(self):
        return dict(schema=source._parent.SOURCE_SCHEMA, passed=True,
                    metadata_only=True, source_admitted=False, strict_pit=False,
                    prices_loaded=False, npz_decoded=False,
                    actual_features_orders_returns_computed=False,
                    registered_strategy=False,
                    actual253_slot_coverage='UNKNOWN_UNTIL_REGISTERED_PREPARE',
                    original_U0_and_core_parity='UNKNOWN_UNTIL_REGISTERED_PREPARE',
                    input_sha256={'synthetic.json': 'a' * 64},
                    source_policy={'synthetic': True}, clock_events=[{'q': 0}])

    def build(self, parent):
        with patch.object(source._parent, 'build_metadata_contract', return_value=parent):
            return source.build_metadata_contract()

    def test_adds_only_explicit_metadata_provenance(self):
        original = self.parent(); before = deepcopy(original)
        result = self.build(original)
        self.assertEqual(original, before)
        self.assertEqual(result['schema'], source.SOURCE_SCHEMA)
        self.assertEqual(result['clock_events'], original['clock_events'])
        self.assertFalse(result['parent_metadata_helper']['actual264_evidence_used'])
        self.assertEqual(result['prospective_proposal']['sha256'], source.PROPOSAL_SHA)
        self.assertEqual(len(result['input_sha256']), 3)
        self.assertFalse(result['source_admitted'])
        result['clock_events'][0]['q'] = 999
        self.assertEqual(original['clock_events'][0]['q'], 0)

    def test_rejects_admission_or_numerical_work_claims(self):
        for key in ('source_admitted', 'strict_pit', 'prices_loaded', 'npz_decoded',
                    'actual_features_orders_returns_computed', 'registered_strategy'):
            with self.subTest(key=key):
                parent = self.parent(); parent[key] = True
                with self.assertRaises(ValueError): self.build(parent)

    def test_rejects_claimed_coverage(self):
        for key in ('actual253_slot_coverage', 'original_U0_and_core_parity'):
            with self.subTest(key=key):
                parent = self.parent(); parent[key] = 'PASSED'
                with self.assertRaises(ValueError): self.build(parent)

    def test_rejects_wrong_schema_or_scope(self):
        for key, value in [('schema', 'other'), ('passed', False),
                           ('metadata_only', False)]:
            parent = self.parent(); parent[key] = value
            with self.assertRaises(ValueError): self.build(parent)

    def test_same_clock_and_skip_geometry(self):
        days=[]; day=date(2014,1,1)
        while day <= date(2016,3,1):
            if day.weekday() < 5: days.append(day.isoformat())
            day += timedelta(days=1)
        value=source.clock(0,'2016-02-29',['SYNTH'],days)
        self.assertEqual(value, source._parent.clock(0,'2016-02-29',['SYNTH'],days))
        t=days.index('2016-02-29')
        self.assertEqual(value['formation_positions'], list(range(t-294,t-41)))
        self.assertEqual((value['required_count'],value['return_count']),(253,252))

    def test_named_partial_not_arbitrary_first_row(self):
        value=source.clock(0,'2016-08-17',['SYNTH'],['2016-08-17','2016-08-18'])
        self.assertEqual(value['required_dates'], [])
        with self.assertRaises(ValueError):
            source.clock(0,'2016-08-16',['SYNTH'],['2016-08-16','2016-08-17'])

    def test_wrong_parent_hash_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fake.py'; p.write_text('raise RuntimeError("must not execute")\n')
            with self.assertRaisesRegex(ValueError,'Pinned metadata helper'):
                source._load_parent(p,'0'*64)

    def test_dependency_inventory_contains_original_and_proposal(self):
        paths=source.dependency_files()
        self.assertEqual(len(paths),len(set(paths)))
        self.assertIn(source.PARENT_PATH,paths)
        self.assertIn(source.PROPOSAL_PATH,paths)
        self.assertTrue(set(source._parent.dependency_files()).issubset(paths))


if __name__ == '__main__':
    unittest.main(verbosity=2)
