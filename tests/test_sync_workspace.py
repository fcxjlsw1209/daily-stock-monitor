"""Exporter tests use temporary fixtures and never execute workspace code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'sync_workspace', Path(__file__).resolve().parents[1] / 'scripts/sync_workspace.py')
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class WorkspaceExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.research = self.base / 'research'
        self.observer = self.base / 'observer'
        self.repo = self.base / 'repo'
        for directory in (self.research / 'outputs', self.observer / 'outputs', self.repo):
            directory.mkdir(parents=True)

    def put(self, root, relative, content='VALUE = 1\n'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode('utf-8'))
        return path

    def export(self, **options):
        return sync.export_workspace(self.research, root=self.repo, **options)

    def allowlist(self, files):
        path = self.base / 'allowlist.json'
        path.write_text(json.dumps({'files': files}))
        return path

    def test_defaults_and_sensitive_exclusions(self):
        self.put(self.research, 'outputs/study/source.py')
        self.put(self.research, 'outputs/example_monitor/source.py')
        self.put(self.research, 'outputs/example_monitor/config.json', '{"allocation": 0.1}\n')
        for relative in ('outputs/study/vendor/copied.py', 'outputs/study/.env',
                         'outputs/study/cache/generated.py', 'outputs/study/plans/saved.py',
                         'outputs/study/raw/source.py',
                         'outputs/study/credentials.py', 'outputs/study/secrets/key.py',
                         'outputs/study/raw.csv', 'outputs/study/ledger.json',
                         'outputs/example_monitor/nested/config.json', 'work/unlisted.py'):
            self.put(self.research, relative)
        out = self.export()
        self.assertEqual(set(out['files']), {'outputs/study/source.py',
                         'outputs/example_monitor/source.py', 'outputs/example_monitor/config.json'})

    def test_explicit_work_allowlist_preserves_research_names_and_bytes(self):
        files = ['recovery165_factor.py', 'shell168_company.py', 'test_factor.py',
                 'archive_calendar_clock248_prefit_failure.py', 'backup_research.py',
                 'vendor_source_audit.py']
        for name in files:
            self.put(self.research, 'work/' + name, '# reviewed\r\nVALUE = 2\r\n')
        self.put(self.research, 'work/unlisted.py')
        self.put(self.research, 'work/venv/package.py')
        out = self.export(include_work_source=True, work_source_manifest=self.allowlist(files))
        self.assertEqual(set(out['files']), {'work/' + name for name in files})
        for name in files:
            self.assertEqual((self.repo / 'work' / name).read_bytes(),
                             (self.research / 'work' / name).read_bytes())
            self.assertEqual(out['sources']['work/' + name]['workspace'], 'research')

    def test_work_requires_both_flags(self):
        manifest = self.allowlist(['source.py'])
        for options in ({'include_work_source': True}, {'work_source_manifest': manifest}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.export(**options)
        self.assertFalse(any(self.repo.iterdir()))

    def test_work_rejects_unsafe_or_private_manifest_entries(self):
        for name in ('../escape.py', '/absolute.py', 'nested/source.py', './source.py',
                     'nested/../source.py', 'source\\escape.py', 'conversation_rebuild.py',
                     'chatlog_export.py', 'rollout_extract.py', 'restore_missing.py',
                     'migrate_archive.py', '.secret.py', 'secrets.py', 'file.json', 123):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.export(include_work_source=True, work_source_manifest=self.allowlist([name]))
        self.assertFalse(any(self.repo.iterdir()))

    def test_observer_is_the_only_monitor_source(self):
        self.put(self.research, 'outputs/example_monitor/source.py', 'VALUE = "stale"\n')
        self.put(self.research, 'outputs/research_only_monitor/source.py')
        self.put(self.research, 'outputs/study/source.py')
        runtime_source = self.put(self.observer, 'outputs/example_monitor/source.py', 'VALUE = "current"\n')
        self.put(self.observer, 'outputs/example_monitor/config.json', '{"allocation": 0.1}\n')
        self.put(self.observer, 'outputs/example_monitor/FREEZE.json', '{"frozen": true}\n')
        self.put(self.observer, 'outputs/example_monitor/RULES.md', 'Frozen rules\n')
        self.put(self.observer, 'outputs/example_monitor/state.json', '{"private": true}\n')
        self.put(self.observer, 'outputs/study/ignored.py')
        before = runtime_source.read_bytes()
        out = self.export(observer_workspace=self.observer)
        self.assertEqual((self.repo / 'outputs/example_monitor/source.py').read_bytes(), before)
        self.assertEqual(runtime_source.read_bytes(), before)
        self.assertNotIn('outputs/research_only_monitor/source.py', out['files'])
        self.assertNotIn('outputs/study/ignored.py', out['files'])
        self.assertNotIn('outputs/example_monitor/state.json', out['files'])
        self.assertEqual(out['sources']['outputs/example_monitor/source.py']['workspace'], 'observer')
        self.assertEqual(out['sources']['outputs/study/source.py']['workspace'], 'research')

    def test_launcher_allowlist_and_executable_mode(self):
        launcher = self.put(self.research, 'run_local.sh', '#!/bin/sh\nexec python3 local_run.py "$@"\n')
        launcher.chmod(0o755)
        self.put(self.observer, 'observer_run.py')
        self.put(self.research, 'CURRENT_RUNTIME.json', '{}')
        self.put(self.observer, 'AGENTS.md', 'Private instructions')
        self.put(self.observer, 'unlisted.py')
        out = self.export(observer_workspace=self.observer, include_local_launchers=True)
        self.assertEqual(set(out['files']), {'run_local.sh', 'observer_run.py'})
        exported = self.repo / 'run_local.sh'
        self.assertEqual(exported.read_bytes(), launcher.read_bytes())
        self.assertEqual(stat.S_IMODE(exported.stat().st_mode), 0o755)
        self.assertEqual(out['sources']['run_local.sh']['mode'], '0o755')
        self.assertEqual(out['sources']['observer_run.py']['workspace'], 'observer')

    def test_secret_literals_are_rejected_without_printing_values(self):
        cases = [
            ('source.py', 'API_KEY = "sensitive-value-123"\n'),
            ('source.py', 'ACCESS_TOKEN: str = "sensitive-value-123"\n'),
            ('source.py', 'cfg = {"api_key": "sensitive-value-123"}\n'),
            ('source.py', 'client(password="sensitive-value-123")\n'),
            ('source.py', 'client(clientSecret="sensitive-value-123")\n'),
            ('source.py', 'API_KEY = "sensitive-" + "value-123"\n'),
            ('source.py', 'cfg["api_key"] = "sensitive-value-123"\n'),
            ('source.py', 'cfg.password: str = "sensitive-value-123"\n'),
            ('source.py', 'API_KEY, count = "sensitive-value-123", 1\n'),
            ('source.py', 'URL = "https://example.test/?token=sensitive-value-123"\n'),
            ('source.py', 'URL = "https://unreviewed.test/?apikey=demo"\n'),
            ('source.py', 'KEY = "-----BEGIN ENCRYPTED PRIVATE KEY-----"\n'),
            ('config.json', '{"nested": {"token": "sensitive-value-123"}}\n'),
            ('run.sh', 'export API_KEY="sensitive-value-123"\n'),
        ]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content):
                with self.assertRaises(ValueError) as caught:
                    sync.inspect_source(Path(filename), content.encode())
                self.assertNotIn('sensitive-value-123', str(caught.exception))

    def test_environment_reads_are_not_secret_literals(self):
        content = '''import os
API_KEY = os.environ["API_KEY"]
ACCESS_TOKEN: str = os.getenv("ACCESS_TOKEN")
cfg = {"api_key": os.environ.get("API_KEY")}
client(password=os.environ["PASSWORD"])
URL = f"https://example.test/?token={ACCESS_TOKEN}"
INLINE_URL = f"https://example.test/?token={os.environ['ACCESS_TOKEN']}"
API_KEY_ENV = "PROVIDER_API_KEY"
args[1]['authorization_path'] = 'work/../outputs/unsafe.json'
metadata.authorization_path: str = 'work/source_grant.json'
DEMO_URL = "https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=demo"
'''
        sync.inspect_source(Path('source.py'), content.encode())
        sync.inspect_source(Path('run.sh'), b'export API_KEY="$PROVIDER_API_KEY"\n')
        sync.inspect_source(Path('run.sh'), b'URL="https://example.test/?token=$ACCESS_TOKEN"\n')

    def test_reviewed_negative_fixtures_do_not_bypass_other_checks(self):
        for filename, urls in sync.REVIEWED_TEST_URLS.items():
            for url in urls:
                content = ('URL = ' + repr(url) + '\n').encode()
                sync.inspect_source(Path(filename), content)
                with self.assertRaises(ValueError):
                    sync.inspect_source(Path('unreviewed_test.py'), content)
                with self.assertRaises(ValueError):
                    sync.inspect_source(Path(filename), content.replace(b'user:', b'other:'))
        filename = Path('test_verify_residual246_vox2014_source.py')
        sync.inspect_source(filename, b'update(Authorization="not-a-secret")\n')
        with self.assertRaises(ValueError):
            sync.inspect_source(filename, b'update(Authorization="actual-sensitive-value")\n')

    def test_reviewed_literal_hash_requires_exact_file_field_and_value(self):
        value = 'Permission to read this research source.'
        digest = hashlib.sha256(value.encode()).hexdigest()
        with patch.dict(sync.REVIEWED_LITERAL_HASHES, {'reviewed.py': {'authorization': digest}}):
            content = ('update(authorization=' + repr(value) + ')\n').encode()
            sync.inspect_source(Path('reviewed.py'), content)
            for filename, altered in (
                    ('other.py', content),
                    ('reviewed.py', content.replace(b'authorization=', b'token=')),
                    ('reviewed.py', content.replace(b'Permission', b'Other grant'))):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    sync.inspect_source(Path(filename), altered)

    def test_reviewed_query_fixture_requires_exact_file_query_and_full_line(self):
        filename = Path('collect_form4_source_probe.py')
        reviewed_line = ("                    raw.replace(b'/files/a.zip', b'/../a.zip'), "
                         "raw.replace(b'/files/a.zip', b'/a.zip?token=x')):")
        content = ('def synthetic_test():\n'
                   '    raw = b"synthetic"\n'
                   '    for bad in (raw,\n' + reviewed_line + '\n'
                   '        pass\n').encode('utf-8')
        sync.inspect_source(filename, content)
        invalid = [
            (Path('other.py'), content),
            (filename, content.replace(b'?token=x', b'?token=changed-value')),
            (filename, content.replace(b'?token=x', b'?password=x')),
            (filename, content.replace(reviewed_line.encode(),
                (reviewed_line + ' # https://example.test/?token=additional-value').encode())),
            (filename, content + b'URL = "https://example.test/?token=additional-value"\n'),
            (filename, content + b'PASSWORD = "additional-sensitive-value"\n'),
        ]
        for path, altered in invalid:
            with self.subTest(path=path), self.assertRaises(ValueError):
                sync.inspect_source(path, altered)

    def test_every_candidate_is_checked_before_any_copy(self):
        existing = self.put(self.repo, 'outputs/study/a.py', 'VALUE = "keep"\n')
        self.put(self.research, 'outputs/study/a.py', 'VALUE = "new"\n')
        self.put(self.research, 'outputs/study/z.py', 'TOKEN = "sensitive-value-123"\n')
        with self.assertRaises(ValueError):
            self.export()
        self.assertEqual(existing.read_text(), 'VALUE = "keep"\n')
        self.assertFalse((self.repo / 'outputs/study/z.py').exists())
        self.assertFalse((self.repo / sync.MANIFEST_PATH).exists())

    def test_source_file_and_directory_symlinks_are_rejected(self):
        outside = self.put(self.base, 'outside.py')
        for link_type in ('file', 'directory'):
            with self.subTest(link_type=link_type):
                link = self.research / 'outputs' / ('escape.py' if link_type == 'file' else 'escape')
                link.symlink_to(outside if link_type == 'file' else self.base, target_is_directory=link_type == 'directory')
                with self.assertRaises(ValueError):
                    self.export()
                link.unlink()
        self.assertFalse(any(self.repo.iterdir()))

    def test_destination_symlink_is_rejected_before_any_copy(self):
        outside = self.base / 'outside'
        outside.mkdir()
        self.put(self.research, 'outputs/a/source.py')
        self.put(self.research, 'outputs/z/source.py')
        (self.repo / 'outputs').mkdir()
        (self.repo / 'outputs/z').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse((self.repo / 'outputs/a').exists())
        self.assertFalse(any(outside.iterdir()))

    def test_work_symlink_and_invalid_parent_rejected_before_any_copy(self):
        self.put(self.research, 'outputs/study/source.py')
        outside = self.put(self.base, 'outside.py')
        (self.research / 'work').mkdir()
        (self.research / 'work/source.py').symlink_to(outside)
        with self.assertRaises(ValueError):
            self.export(include_work_source=True, work_source_manifest=self.allowlist(['source.py']))
        self.assertFalse(any(self.repo.iterdir()))
        self.put(self.repo, 'docs', 'not a directory')
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse((self.repo / 'outputs').exists())

    def test_fixture_adaptations_are_only_in_published_tests(self):
        content = '''import unittest
from pathlib import Path
P = Path(__file__).resolve().parent
class Example(unittest.TestCase):
    def test_feature_causality(self):
        pass
    def test_research_accounts(self):
        pass
'''
        relative = 'outputs/daily_stock_monitor/test_monitor.py'
        source = self.put(self.research, relative, content)
        unrelated = self.put(self.repo, 'unrelated.txt', 'keep me\n')
        out = self.export()
        exported = (self.repo / relative).read_text()
        self.assertEqual(exported.count('@unittest.skipUnless'), 2)
        self.assertEqual(source.read_text(), content)
        self.assertEqual(out['publication_adaptations'][relative]['source_sha256'],
                         hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(unrelated.read_text(), 'keep me\n')
        self.assertEqual(sync.fixture_guards(Path(relative), exported.encode()), exported.encode())


if __name__ == '__main__':
    unittest.main()
