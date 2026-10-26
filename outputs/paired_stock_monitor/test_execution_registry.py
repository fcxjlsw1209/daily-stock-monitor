"""Routing checks do not open or write live account databases."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import execution_registry as registry


class ExecutionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paired = self.root / 'paired_stock_monitor'; self.paired.mkdir()
        self.active = self.paired / 'ACTIVE_EXECUTION.json'
        self.addCleanup(patch.stopall)
        patch.object(registry, 'ACTIVE_EXECUTION', self.active).start()
        patch.object(registry, 'P', self.paired).start()
        self.folder = self.root / 'daily_stock_monitor'; self.folder.mkdir()
        (self.folder / 'paper_ledger.sqlite').write_bytes(b'legacy untouched')

    def activate(self):
        data = dict(version='causal_replay_v3', accounts={name:dict(ledger='execution_v3/paper_ledger.sqlite', exports='execution_v3') for name in registry.ACCOUNT_NAMES})
        self.active.write_text(json.dumps(data))
        folder = self.folder / 'execution_v3'; folder.mkdir(exist_ok=True)
        (folder / 'paper_ledger.sqlite').write_bytes(b'new independent ledger')
        return data

    def test_absent_registry_preserves_old_paths_and_fixture_compatibility(self):
        for folder in [self.folder, self.root / 'synthetic_fixture']:
            route = registry.active_account_paths(folder)
            self.assertEqual(route, dict(version='v2', ledger=folder/'paper_ledger.sqlite', exports=folder, adapter=self.paired/'evening_adapter.py'))
        self.assertFalse(self.active.exists())

    def test_active_complete_registry_routes_independent_version(self):
        self.activate()
        route = registry.active_account_paths(self.folder)
        self.assertEqual(route['version'], 'causal_replay_v3')
        self.assertEqual(route['ledger'], self.folder/'execution_v3/paper_ledger.sqlite')
        self.assertEqual(route['exports'], self.folder/'execution_v3')
        self.assertEqual(route['adapter'], self.paired/'causal_replay_v3.py')
        self.assertEqual((self.folder/'paper_ledger.sqlite').read_bytes(), b'legacy untouched')

    def test_unknown_active_account_rejected(self):
        self.activate()
        with self.assertRaisesRegex(ValueError, 'Unknown execution account'):
            registry.active_account_paths(self.root/'unknown')

    def test_unknown_version_never_falls_back(self):
        data=self.activate(); data['version']='future'; self.active.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'version'):
            registry.active_account_paths(self.folder)

    def test_partial_registry_rejected(self):
        data=self.activate(); del data['accounts']['pressure_shadow_monitor']; self.active.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'all six'):
            registry.active_account_paths(self.folder)

    def test_path_escape_or_absolute_or_wrong_subtree_rejected(self):
        for value in ['../../elsewhere.sqlite', '/tmp/test.sqlite', 'execution_v3/../paper_ledger.sqlite', 'paper_ledger.sqlite', 'execution_v3\\paper_ledger.sqlite']:
            with self.subTest(path=value):
                data=self.activate(); data['accounts'][self.folder.name]['ledger']=value; self.active.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'paths'):
                    registry.active_account_paths(self.folder)

    def test_other_accounts_invalid_path_also_rejected(self):
        data=self.activate(); data['accounts']['accel_shadow_monitor']['exports']='../elsewhere'; self.active.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'paths'):
            registry.active_account_paths(self.folder)

    def test_symlink_subtree_cannot_point_outside_account(self):
        self.activate(); (self.folder/'execution_v3/paper_ledger.sqlite').unlink(); (self.folder/'execution_v3').rmdir()
        elsewhere=self.root/'elsewhere'; elsewhere.mkdir(); (elsewhere/'paper_ledger.sqlite').write_bytes(b'x')
        (self.folder/'execution_v3').symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            registry.active_account_paths(self.folder)

    def test_internal_alias_cannot_point_to_original(self):
        self.activate(); (self.folder/'execution_v3/paper_ledger.sqlite').unlink(); (self.folder/'execution_v3').rmdir()
        (self.folder/'execution_v3').symlink_to(self.folder, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'aliases'):
            registry.active_account_paths(self.folder)

    def test_hardlinked_legacy_ledger_rejected(self):
        self.activate(); (self.folder/'execution_v3/paper_ledger.sqlite').unlink()
        os.link(self.folder/'paper_ledger.sqlite', self.folder/'execution_v3/paper_ledger.sqlite')
        with self.assertRaisesRegex(ValueError, 'aliases original'):
            registry.active_account_paths(self.folder)

    def test_missing_active_database_never_falls_back(self):
        self.activate(); (self.folder/'execution_v3/paper_ledger.sqlite').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            registry.active_account_paths(self.folder)

    def test_duplicate_keys_and_malformed_json_rejected(self):
        self.activate(); self.active.write_text('{"version":"v2","version":"causal_replay_v3"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            registry.active_account_paths(self.folder)
        self.active.write_text('{')
        with self.assertRaises(ValueError):
            registry.active_account_paths(self.folder)

    def test_dangling_registry_symlink_never_falls_back(self):
        self.active.symlink_to(self.root/'missing.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            registry.active_account_paths(self.folder)


if __name__ == '__main__': unittest.main()
