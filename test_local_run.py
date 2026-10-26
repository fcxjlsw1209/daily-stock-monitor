"""Synthetic relocation/guard tests; no market imports, data, or backtests."""
from collections import OrderedDict
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import local_run as local


OLD = "/frozen/project"
NEW = "/local/project"


class RelocationTests(unittest.TestCase):
    def translate(self, value):
        return local.translate_paths(value, OLD, NEW)

    def test_exact_root_prefix_boundary_and_non_path_text(self):
        value = [OLD, OLD + "/work/a.json", OLD + "-backup/a", "note: " + OLD + "/a", "relative/a"]
        self.assertEqual(self.translate(value), [NEW, NEW + "/work/a.json", value[2], value[3], value[4]])

    def test_nested_keys_values_lists_tuples_and_idempotence(self):
        value = {OLD + "/a": [{"path": OLD + "/b"}, (OLD, 7)]}
        expected = {NEW + "/a": [{"path": NEW + "/b"}, (NEW, 7)]}
        self.assertEqual(self.translate(value), expected)
        self.assertEqual(self.translate(expected), expected)
        self.assertIn(OLD + "/a", value)

    def test_key_collision_rejected_even_when_values_equal(self):
        with self.assertRaisesRegex(local.LocalRunError, "collision"):
            self.translate({OLD + "/a": 1, NEW + "/a": 1})

    def test_unmapped_cloud_paths_fail_without_rewriting_text(self):
        for value in ("/x/CloudStorage/Drive/a", "/x/OneDrive-RokuInc/a", "C:\\Users\\x\\OneDrive\\a", "file:///x/CloudStorage/a", "~/OneDrive/a"):
            with self.subTest(value=value), self.assertRaisesRegex(local.LocalRunError, "Unmapped cloud"):
                self.translate(value)
        sentence = "Archived in OneDrive/CloudStorage; price remains 1.23"
        self.assertEqual(self.translate(sentence), sentence)

    def test_actual_old_cloud_prefix_is_allowed_but_sibling_is_rejected(self):
        self.assertEqual(local.translate_paths(local.DEFAULT_OLD_ROOT + "/a", local.DEFAULT_OLD_ROOT, NEW), NEW + "/a")
        with self.assertRaises(local.LocalRunError):
            local.translate_paths(local.DEFAULT_OLD_ROOT + "-backup/a", local.DEFAULT_OLD_ROOT, NEW)

    def test_relocated_parent_escape_rejected(self):
        with self.assertRaisesRegex(local.LocalRunError, "escapes"):
            self.translate(OLD + "/../other/a")

    def test_numeric_objects_preserved(self):
        values = [1, 2**100, -0.0, 1.2345678901234567, True, None, Decimal("0.01000"), float("nan"), float("inf")]
        mapped = self.translate(values)
        self.assertTrue(all(before is after for before, after in zip(values, mapped)))
        self.assertEqual(math.copysign(1, mapped[2]), -1)


class JsonWrapperTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads
        self.addCleanup(setattr, json, "loads", self.original)
        local.install_json_relocation(OLD, NEW)

    def test_load_delegates_and_file_bytes_hash_are_unchanged(self):
        raw = ('{ "path": "' + OLD + '/a", "value": 0.0100 }\n').encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PLAN.json"
            path.write_bytes(raw)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as source:
                result = json.load(source, parse_float=Decimal)
            self.assertEqual(result, {"path": NEW + "/a", "value": Decimal("0.0100")})
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_hooks_receive_original_values_and_result_is_relocated(self):
        seen = []
        def hook(value):
            seen.append(dict(value))
            return OrderedDict(value)
        value = json.loads('{"path": "/frozen/project/a", "n": 12}', object_hook=hook, parse_int=Decimal)
        self.assertEqual(seen[0]["path"], OLD + "/a")
        self.assertIsInstance(value, OrderedDict)
        self.assertEqual(value["path"], NEW + "/a")
        self.assertEqual(value["n"], Decimal(12))

    def test_pairs_hook_tuple_and_precedence_preserved(self):
        def forbidden(value):
            raise AssertionError("object_hook must not supersede object_pairs_hook")
        result = json.loads('{"/frozen/project/a": "/frozen/project/b"}', object_pairs_hook=tuple, object_hook=forbidden)
        self.assertEqual(result, ((NEW + "/a", NEW + "/b"),))

    def test_parse_constant_rejection_and_custom_decoder_preserved(self):
        def reject(value):
            raise ValueError("non-finite: " + value)
        with self.assertRaisesRegex(ValueError, "non-finite: NaN"):
            json.loads("NaN", parse_constant=reject)
        self.assertTrue(math.isnan(json.loads("NaN")))
        class Decoder(json.JSONDecoder):
            def __init__(self, marker, **kwargs):
                self.marker = marker
                super().__init__(**kwargs)
            def decode(self, value):
                self.assert_marker = self.marker
                return {"marker": self.marker, "path": super().decode(value)}
        self.assertEqual(json.loads('"/frozen/project/a"', cls=Decoder, marker=42), {"marker": 42, "path": NEW + "/a"})

    def test_install_is_idempotent_and_conflicting_mapping_rejected(self):
        first = json.loads
        self.assertIs(local.install_json_relocation(OLD, NEW), first)
        with self.assertRaisesRegex(local.LocalRunError, "different"):
            local.install_json_relocation(OLD, "/different")


class LocalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.script = self.root / "safe.py"
        self.script.write_text("pass\n")

    def test_script_must_be_relative_inside_root_and_python(self):
        self.assertEqual(local.resolve_script(self.root, "safe.py"), self.script)
        for path in (str(self.script), "../safe.py", "missing.py", "safe.txt"):
            with self.subTest(path=path), self.assertRaises(local.LocalRunError):
                local.resolve_script(self.root, path)
        link = self.root / "escape.py"
        link.symlink_to(Path(local.__file__).resolve())
        with self.assertRaisesRegex(local.LocalRunError, "outside"):
            local.resolve_script(self.root, "escape.py")

    def test_cloud_file_directory_process_and_network_events_denied(self):
        audit = local.make_audit_hook(OLD)
        for event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.remove"):
            for path in (OLD + "/a", "/x/CloudStorage/a", "/x/OneDrive-RokuInc/a"):
                with self.subTest(event=event, path=path), self.assertRaises(PermissionError):
                    audit(event, (path,))
        for event in ("socket.__new__", "socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork"):
            with self.subTest(event=event), self.assertRaises(PermissionError):
                audit(event, ())
        audit("open", (str(self.script), "r", 0))
        audit("os.listdir", (str(self.root),))

    def test_cloud_symlink_and_rename_destination_denied(self):
        link = self.root / "cloud_link"
        link.symlink_to("/nonexistent/CloudStorage/a")
        audit = local.make_audit_hook(OLD)
        with self.assertRaisesRegex(PermissionError, "symlink"):
            audit("open", (link, "r", 0))
        with self.assertRaises(PermissionError):
            audit("os.rename", (str(self.script), OLD + "/a"))

    def test_cli_executes_local_script_with_unchanged_arguments_and_guards(self):
        for child in ("work", "work/monthly-decision-deps", "work/research-plot-deps"):
            (self.root / child).mkdir(exist_ok=True)
        self.script.write_text('''import json, os, socket, subprocess, sys
from pathlib import Path
assert sys.dont_write_bytecode
assert Path.cwd() == Path(sys.argv[1])
assert sys.argv[2:] == ["--custom", "value"]
assert json.loads('"/frozen/project/a"') == str(Path.cwd()) + '/a'
actions = [lambda: open('/frozen/project/a'), lambda: os.listdir('/x/CloudStorage'),
           lambda: socket.socket(), lambda: subprocess.run(['/bin/true'])]
for action in actions:
    try: action()
    except PermissionError: pass
    else: raise AssertionError('guard did not reject operation')
print('LOCAL_SCRIPT_OK')
''')
        result = subprocess.run([sys.executable, "-I", "-B", str(Path(local.__file__).resolve()), "--root", str(self.root), "--old-root", OLD, "safe.py", str(self.root), "--custom", "value"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "LOCAL_SCRIPT_OK")
        self.assertFalse((self.root / "__pycache__").exists())


    def test_cli_imports_real_sibling_before_work_modules(self):
        for child in ("work", "work/monthly-decision-deps", "work/research-plot-deps"):
            (self.root / child).mkdir(exist_ok=True)
        folder = self.root / "outputs" / "nested_script"
        folder.mkdir(parents=True)
        (folder / "local_sibling_fixture.py").write_text("VALUE = 'sibling-ok'\n")
        (self.root / "work" / "local_sibling_fixture.py").write_text(
            "raise AssertionError('work module shadowed script sibling')\n")
        entry = folder / "entry.py"
        entry.write_text("""import sys
from pathlib import Path
from local_sibling_fixture import VALUE
assert Path(sys.path[0]) == Path(__file__).resolve().parent
assert sys.dont_write_bytecode
print(VALUE)
""")
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(local.__file__).resolve()),
             "--root", str(self.root), "--old-root", OLD,
             str(entry.relative_to(self.root))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "sibling-ok")
        self.assertFalse(list(self.root.rglob("__pycache__")))


class ProbeTests(unittest.TestCase):
    def test_probe_calls_only_constructor_cache_and_source_verifiers(self):
        calls = []
        source = "/local/reviewer.py"
        digest = "a" * 64
        pins = {"/local/dependency_" + str(i): digest for i in range(290)}
        pins[source] = digest
        plan = {"independent_prepare_source_hashes": {source: digest}, "runtime": {"python": "synthetic"}}
        class Runner:
            def __init__(self, root, plan_path, plan_sha, backend_factory):
                calls.append("construct")
                self.pins, self.plan = pins, plan
                with self_test.assertRaisesRegex(local.LocalRunError, "must not instantiate"):
                    backend_factory()
            def cache(self):
                calls.append("cache")
                return {str(i): digest for i in range(6)}
            def verify(self, value):
                calls.append("verify")
                return value
        self_test = self
        producer = SimpleNamespace(Runner=Runner, source_files=lambda root: {source: digest})
        reviewer = SimpleNamespace(source_files=lambda root: [source])
        def sha(path):
            return local.PLAN138_SHA256 if str(path).endswith("PLAN.json") else digest
        with patch.object(local.importlib, "import_module", side_effect=[producer, reviewer]), patch.object(local, "_sha256", side_effect=sha):
            result = local.probe138("/local")
        self.assertEqual(calls, ["construct", "cache", "verify"])
        self.assertEqual(result["registered_dependency_count"], 291)
        self.assertEqual(result["prepared_cache_file_count"], 6)
        self.assertFalse(result["returns_or_nav_computed"])
        self.assertFalse(result["independent_prepare_audit_executed"])


if __name__ == "__main__":
    unittest.main()
