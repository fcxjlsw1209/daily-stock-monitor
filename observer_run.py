#!/usr/bin/env python3
"""Observer-only launcher: unchanged scripts/JSON, local writes, network allowed.

Use run_observer.sh for the macOS sandbox. The Python audit guard is supplementary,
not a security boundary against native code. No monitor work occurs on import.
"""
import argparse
import functools
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


class ObserverRunError(ValueError):
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
        raise ObserverRunError("Root must be an absolute filesystem path")
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
                    raise ObserverRunError("Relocated JSON path escapes local root: " + item)
                return relocated
            if _is_path_string(item) and _cloud_component(item):
                raise ObserverRunError("Unmapped cloud path in JSON: " + item)
            return item
        if isinstance(item, dict):
            # copy/clear preserves standard dict subclasses returned by hooks.
            result = item.copy()
            result.clear()
            for key, child in item.items():
                mapped = walk(key)
                if mapped in result:
                    raise ObserverRunError("JSON key collision after relocation: " + repr(mapped))
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
    installed = getattr(current, "_observer_run_mapping", None)
    if installed is not None:
        if installed != mapping:
            raise ObserverRunError("A different JSON relocation is already installed")
        return current

    @functools.wraps(current)
    def loads(*args, **kwargs):
        # Forward every original argument, including cls, numeric parsers and
        # hooks. Do not strip/replace parse_constant or non-finite validation.
        return translate_paths(current(*args, **kwargs), *mapping)

    loads._observer_run_mapping = mapping
    json.loads = loads
    return loads


def check_path(value, old_root):
    if value is None or isinstance(value, int):
        return
    try:
        absolute = os.path.abspath(os.fsdecode(os.fspath(value)))
    except TypeError:
        return
    if _under(absolute, old_root) or _cloud_component(absolute):
        raise PermissionError('Observer denied cloud path: ' + absolute)
    resolved = os.path.realpath(absolute)
    if _under(resolved, old_root) or _cloud_component(resolved):
        raise PermissionError('Observer denied cloud symlink: ' + absolute)


def resolve_script(root, script, *, child=False):
    root = Path(root).resolve()
    candidate = Path(script)
    if '..' in candidate.parts or (candidate.is_absolute() and not child):
        raise ObserverRunError('Script must be local relative .py (child may use local absolute path)')
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ObserverRunError('Script escapes observer root: ' + str(script)) from None
    if path.suffix != '.py' or not path.is_file():
        raise ObserverRunError('Missing local Python script: ' + str(path))
    return path


def make_audit_hook(old_root, root, shim):
    old_root, root, shim = str(old_root), str(root), str(shim)
    paths = {'open', 'os.listdir', 'os.scandir', 'os.chdir', 'os.mkdir', 'os.rmdir',
             'os.remove', 'os.unlink', 'os.truncate', 'os.chmod', 'os.chown', 'os.utime', 'os.readlink'}
    pairs = {'os.rename', 'os.replace', 'os.link', 'os.symlink'}
    other_processes = {'os.system', 'os.posix_spawn', 'os.posix_spawnp', 'os.spawn',
                       'os.exec', 'os.fork', 'os.forkpty', 'pty.spawn'}

    def audit(event, args):
        if event == 'subprocess.Popen':
            executable, argv, cwd, env = args
            check_path(executable, old_root)
            if (os.path.abspath(os.fsdecode(executable)) != shim
                    or not isinstance(argv, (list, tuple)) or len(argv) < 2
                    or os.path.abspath(os.fsdecode(argv[0])) != shim):
                raise PermissionError('Observer permits only its wrapped Python child scripts')
            resolve_script(root, os.fsdecode(argv[1]), child=True)
            check_path(cwd, old_root)
            if cwd is not None and not _under(os.path.realpath(cwd), root):
                raise PermissionError('Observer child cwd escapes local runtime')
            if env is not None:
                for key in ('OBSERVER_RUNTIME_ROOT', 'OBSERVER_OLD_ROOT', 'OBSERVER_REAL_PYTHON'):
                    if env.get(key) != os.environ.get(key):
                        raise PermissionError('Observer child lost launcher environment: ' + key)
        elif event in other_processes:
            raise PermissionError('Observer denied unwrapped process: ' + event)
        elif event in paths and args:
            check_path(args[0], old_root)
        elif event in pairs:
            for path in args[:2]:
                check_path(path, old_root)
        # The observer intentionally permits sockets. run_observer.sh permits
        # outbound networking; no research launcher or sandbox is modified.
    return audit


def configure_runtime(root, old_root):
    old_root = _root_string(old_root)
    check_path(root, old_root)
    root = Path(root).resolve(strict=True)
    check_path(root / 'work', old_root)
    if not root.is_dir() or not (root / 'work').is_dir():
        raise ObserverRunError('Observer root and its work directory must exist')
    shim = Path(__file__).resolve().with_name('observer_python.sh')
    check_path(shim, old_root)
    if not shim.is_file() or not os.access(shim, os.X_OK):
        raise ObserverRunError('Executable observer_python.sh is required')
    real_python = str(Path(sys.executable).absolute())
    check_path(real_python, old_root)
    imports = [root / 'work', root / 'work/monthly-decision-deps', root / 'work/research-plot-deps']
    for path in imports:
        check_path(path, old_root)
    clean = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            check_path(entry, old_root)
        except PermissionError:
            continue
        if os.path.realpath(entry) not in {str(p) for p in imports}:
            clean.append(entry)
    sys.path[:] = [str(p) for p in imports if p.is_dir()] + clean
    sys.dont_write_bytecode = True
    os.environ.update(OBSERVER_RUNTIME_ROOT=str(root), OBSERVER_OLD_ROOT=old_root,
                      OBSERVER_REAL_PYTHON=real_python, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    private = root / '.observer-runtime'
    for name in ('tmp', 'cache'):
        check_path(private / name, old_root)
        (private / name).mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(private / 'tmp')
    os.environ['XDG_CACHE_HOME'] = str(private / 'cache')
    os.chdir(root)
    install_json_relocation(old_root, str(root))
    sys.executable = str(shim)
    sys.addaudithook(make_audit_hook(old_root, root, shim))
    # macOS platformdirs ignores XDG_CACHE_HOME; configure every yfinance
    # cache through the pinned library API before any observer imports run.
    quote_cache = private / 'cache' / 'yfinance'
    check_path(quote_cache, old_root)
    from yfinance import set_tz_cache_location
    set_tz_cache_location(str(quote_cache))
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--old-root', default=DEFAULT_OLD_ROOT)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('script')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = configure_runtime(args.root, args.old_root)
    script = resolve_script(root, args.script, child=args.child)
    check_path(script, args.old_root)
    # Match normal `python /path/script.py` sibling-import behavior.
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + args.script_args
    runpy.run_path(str(script), run_name='__main__')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ObserverRunError, PermissionError, FileNotFoundError) as error:
        print('observer_run: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
