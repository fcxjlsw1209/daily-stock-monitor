#!/usr/bin/env python3
"""Run a local research script without changing frozen JSON or source bytes.

Invoke via the accompanying sandbox wrapper. The Python audit hook supplements
the operating-system sandbox; it is not a security boundary for native code.
Only JSON decoding relocates paths. All digest checks still read original bytes.
"""
import argparse
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


DEFAULT_OLD_ROOT = (
    "/Users/shiweili/Library/CloudStorage/OneDrive-RokuInc/Documents/Codex/"
    "2026-09-18/5-grand-hyatt-fukuoka-5-14"
)
PLAN138_SHA256 = "6fa5d01b0911ac26d33a0834662e75ec674d0d9f1fc707bc5e7a8fceab3458fa"
PLAN138_DEPENDENCIES = 291
PLAN138_RELATIVE = "outputs/monthly_sector_asymmetry138/PLAN.json"


class LocalRunError(ValueError):
    """A relocation or local execution invariant failed."""


def _under(value, root):
    return value == root or value.startswith(root + "/")


def _cloud_component(value):
    parts = re.split(r"[\\/]", value.casefold())
    return any(p == "cloudstorage" or p.startswith("onedrive") for p in parts)


def _is_path_string(value):
    return value.startswith(("/", "file://", "~")) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def _root_string(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not os.path.isabs(value):
        raise LocalRunError("Root must be an absolute filesystem path")
    return os.path.normpath(value)


def translate_paths(value, old_root, root):
    """Relocate complete root/prefix strings, including dictionary keys.

    Numeric values and non-path text are returned unchanged. Decoder hooks run
    normally before this traversal, and list/tuple results are both supported.
    """
    old_root, root = _root_string(old_root), _root_string(root)

    def walk(item):
        if isinstance(item, str):
            if _under(item, old_root):
                relocated = root + item[len(old_root):]
                # Lexical '..' must not turn a registered path into an escape.
                if not _under(os.path.normpath(relocated), root):
                    raise LocalRunError("Relocated JSON path escapes local root: " + item)
                return relocated
            if _is_path_string(item) and _cloud_component(item):
                raise LocalRunError("Unmapped cloud path in JSON: " + item)
            return item
        if isinstance(item, dict):
            # copy/clear preserves standard dict subclasses returned by hooks.
            result = item.copy()
            result.clear()
            for key, child in item.items():
                mapped = walk(key)
                if mapped in result:
                    raise LocalRunError("JSON key collision after relocation: " + repr(mapped))
                result[mapped] = walk(child)
            return result
        if isinstance(item, list):
            return [walk(child) for child in item]
        if isinstance(item, tuple):
            return tuple(walk(child) for child in item)
        return item

    return walk(value)


def install_json_relocation(old_root, root):
    """Wrap loads once; stdlib json.load delegates to this exact function."""
    mapping = (_root_string(old_root), _root_string(root))
    current = json.loads
    installed = getattr(current, "_local_run_mapping", None)
    if installed is not None:
        if installed != mapping:
            raise LocalRunError("A different JSON relocation is already installed")
        return current

    @functools.wraps(current)
    def loads(*args, **kwargs):
        # Forward every original argument, including cls, numeric parsers and
        # hooks. Do not strip/replace parse_constant or non-finite validation.
        return translate_paths(current(*args, **kwargs), *mapping)

    loads._local_run_mapping = mapping
    json.loads = loads
    return loads


def _check_access_path(value, old_root):
    if value is None or isinstance(value, int):
        return
    try:
        path = os.fsdecode(os.fspath(value))
    except TypeError:
        return
    absolute = os.path.abspath(path)
    if _under(absolute, old_root) or _cloud_component(absolute):
        raise PermissionError("Local-only execution denied cloud path: " + absolute)
    # Block a local symlink that resolves into cloud storage as well.
    resolved = os.path.realpath(absolute)
    if _under(resolved, old_root) or _cloud_component(resolved):
        raise PermissionError("Local-only execution denied cloud symlink: " + absolute)


def make_audit_hook(old_root):
    """Return a testable Python guard; native calls require the OS sandbox."""
    old_root = _root_string(old_root)
    path_events = {
        "open", "os.listdir", "os.scandir", "os.chdir", "os.mkdir",
        "os.rmdir", "os.remove", "os.unlink", "os.truncate", "os.chmod",
        "os.chown", "os.utime", "os.readlink",
    }
    pair_events = {"os.rename", "os.replace", "os.link", "os.symlink"}
    process_events = {
        "os.system", "os.posix_spawn", "os.posix_spawnp", "os.spawn",
        "os.exec", "os.fork", "os.forkpty", "pty.spawn",
    }

    def audit(event, args):
        if event.startswith("socket.") or event.startswith("subprocess.") or event in process_events:
            raise PermissionError("Local-only execution denied operation: " + event)
        if event in path_events and args:
            _check_access_path(args[0], old_root)
        elif event in pair_events:
            for path in args[:2]:
                _check_access_path(path, old_root)

    return audit


def resolve_script(root, script):
    """Only accept an existing Python file given relative to the local root."""
    root = Path(root).resolve()
    relative = Path(script)
    if relative.is_absolute() or ".." in relative.parts:
        raise LocalRunError("Script must be a local relative path without '..'")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise LocalRunError("Script resolves outside local root: " + script) from None
    if path.suffix != ".py" or not path.is_file():
        raise LocalRunError("Script must be an existing local .py file: " + script)
    return path


def configure_runtime(root, old_root):
    """Set local imports and cwd, then install guards before research imports."""
    old_root = _root_string(old_root)
    _check_access_path(root, old_root)
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise LocalRunError("Local root is not a directory: " + str(root))
    imports = [root / "work", root / "work/monthly-decision-deps", root / "work/research-plot-deps"]
    for path in imports:
        _check_access_path(path, old_root)
        if not path.is_dir():
            raise LocalRunError("Required local import directory is missing: " + str(path))
    clean = []
    for entry in sys.path:
        if not entry:
            continue
        absolute = os.path.abspath(entry)
        if _under(absolute, old_root) or _cloud_component(absolute):
            continue
        resolved = os.path.realpath(absolute)
        if _under(resolved, old_root) or _cloud_component(resolved):
            continue
        if resolved not in {str(p) for p in imports}:
            clean.append(entry)
    sys.path[:] = [str(p) for p in imports] + clean
    sys.dont_write_bytecode = True
    os.chdir(root)
    install_json_relocation(old_root, str(root))
    sys.addaudithook(make_audit_hook(old_root))
    return root


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe138(root):
    """Verify only frozen inputs/cache/source contracts; never run a phase."""
    root = Path(root).resolve()
    plan_path = root / PLAN138_RELATIVE
    if _sha256(plan_path) != PLAN138_SHA256:
        raise LocalRunError("Round138 original PLAN digest mismatch")
    runner_module = importlib.import_module("monthly_sector_asymmetry138_runner")

    def no_backend(*args, **kwargs):
        raise LocalRunError("Metadata probe must not instantiate a research backend")

    runner = runner_module.Runner(root, plan_path, PLAN138_SHA256, backend_factory=no_backend)
    if len(runner.pins) != PLAN138_DEPENDENCIES:
        raise LocalRunError("Round138 must retain exactly 291 registered dependencies")
    cache = runner.cache()
    sources = runner_module.source_files(root)
    if not all(runner.pins.get(key) == value for key, value in sources.items()):
        raise LocalRunError("Runner source binding mismatch")
    reviewer = importlib.import_module("monthly_sector_asymmetry138_prepare_review")
    review_sources = {str(Path(path).resolve()): _sha256(path) for path in reviewer.source_files(root)}
    expected_review = runner.verify(runner.plan["independent_prepare_source_hashes"])
    if review_sources != expected_review:
        raise LocalRunError("Independent prepare reviewer source binding mismatch")
    if not all(runner.pins.get(key) == value for key, value in review_sources.items()):
        raise LocalRunError("Independent prepare reviewer has unregistered source files")
    if _sha256(plan_path) != PLAN138_SHA256:
        raise LocalRunError("Round138 PLAN changed during probe")
    return {
        "root": str(root), "round": 138, "plan_sha256": PLAN138_SHA256,
        "registered_dependency_count": len(runner.pins),
        "prepared_cache_file_count": len(cache),
        "runner_source_file_count": len(sources),
        "independent_prepare_source_file_count": len(review_sources),
        "runtime": runner.plan["runtime"],
        "source_and_runtime_bindings_verified": True,
        "independent_prepare_audit_executed": False,
        "returns_or_nav_computed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    parser.add_argument("--probe138", action="store_true", help="verify frozen round138 metadata and cache only")
    parser.add_argument("script", nargs="?", help="Python script relative to the local root")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.probe138 == bool(args.script):
        parser.error("choose exactly one of --probe138 or a local relative script")
    root = configure_runtime(args.root, args.old_root)
    if args.probe138:
        print(json.dumps(probe138(root), indent=2, allow_nan=False))
        return 0
    script = resolve_script(root, args.script)
    # Match direct Python script execution after validating the local path.
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + args.script_args
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LocalRunError, PermissionError, FileNotFoundError) as error:
        print("local_run: " + str(error), file=sys.stderr)
        raise SystemExit(2)
