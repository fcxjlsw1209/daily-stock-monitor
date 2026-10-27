"""Export authored source and explicit allowlists, never runtime data.

Usage: python scripts/sync_workspace.py /absolute/research/workspace
The destination is this checkout. Existing unrelated files are never removed.
Optional work manifests contain {"files": ["reviewed_top_level_source.py", ...]}
with paths relative to the research workspace's work/ directory.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'sources', 'additional_sources', 'vendor', 'vendor_netcdf', '__pycache__',
        'archives', 'archive', 'backups', 'backup', 'stock-signal-monitor', 'prices',
        'plans', 'snapshots', 'cache', 'venv', 'env', 'site-packages', 'node_modules',
        'credentials', 'secrets', 'logs', 'ledgers', 'paper_ledgers', 'raw'}
DOCS = {'RULES.md', 'POLICY.md', 'PROTOCOL.md', 'TIMESTAMP_FIX.md'}
META = {'PLAN.json', 'GRID.json', 'MODEL_CONFIGS.json', 'FROZEN.json'}
LAUNCHERS = {
    'research': ('local_run.py', 'run_local.sh', 'local.sandbox.sb', 'test_local_run.py'),
    'observer': ('observer_run.py', 'observer_python.sh', 'run_observer.sh', 'observer.sandbox.sb'),
}
PRIVATE_WORK = re.compile(r'(?:conversation|chatlog|rollout|restore_missing|migrate_archive)', re.I)
CREDENTIAL_NAME = re.compile(
    r'(?:^|[^a-z0-9])(?:api[_-]?key|password|passwd|secret|client[_-]?secret|'
    r'(?:access|refresh|auth|bearer)[_-]?token|token|authorization|private[_-]?key|'
    r'credentials?)(?:$|[^a-z0-9])', re.I)
CREDENTIAL_URL = re.compile(
    r'[?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)'
    r'=([^&#\s\"\'<>]+)', re.I)
# Reviewed negative-test fixtures, matched by both filename and exact literal.
# These intentionally exercise URL/header rejection; other literals in these
# files still receive the normal credential checks.
REVIEWED_TEST_URLS = {
    'test_after150_intc_extract.py': {'https://user:secret@www.intc.com/wrong'},
    'test_after151_lrcx_extract.py': {'https://user:pass@newsroom.lamresearch.com/x'},
    'test_after157_annual_body_collect.py': {'https://user:pass@www.microsoft.com/x'},
    'test_after157_msft_iframe_collect.py': {'https://user:pass@microsoft.gcs-web.com/x'},
    'test_after157_walmart_collect.py': {'https://user:pass@stock.walmart.com/x'},
    'test_roa_field_scope270_v2.py': {'https://user:pass@synthetic.example/r'},
}
REVIEWED_TEST_VALUES = {
    'test_verify_residual246_vox2014_source.py': {'not-a-secret'},
}
# Query rejection fixtures require the exact filename, matched query, and hash
# of the entire UTF-8 source line (without its newline). Extra literals on the
# same line invalidate the exception; every other line is scanned normally.
REVIEWED_QUERY_FIXTURE_LINES = {
    'collect_form4_source_probe.py': {
        '?token=x': 'ac02d02e3e99d7b6891cc970bd2ad02162b9daae99e48447bc2db63637f4afe9'},
}
# Independently reviewed research permission prose and synthetic leak-detection
# fixtures. Match the filename, field, and exact UTF-8 literal hash; never relax
# checks for every authorization field or every test TOKEN.
REVIEWED_LITERAL_HASHES = {
    'after155_cat_wire_source.py': {
        'authorization': '963f18f168ade599aab7a671b0df348c53723604d45b660b9b963454f1ac4a45'},
    'after155_tsla_wire_source.py': {
        'authorization': '86e8ef1c797090f8f936cf55262620552c51a17898f525117ff12947ae80942f'},
    'record_after148_next_source.py': {
        'source_followup_authorization': '9d9c4e31177a28335d73f83a8c24384bcdd4ad70c08e3d75cb58230b7ebdf8b5'},
    'test_collect_ownsector238_vox.py': {
        'TOKEN': 'aa93abca3a6b3c7c9ca218e04a75d1297875ebe1c76be7245f8b883737f9d148'},
    'test_collect_residual246_vox2014.py': {
        'TOKEN': '0d1d1ecba07d4281fdaaf8719c47ff58729bc5889a2e3b2503309a60d1889398'},
}
MANIFEST_PATH = Path('docs/WORKSPACE_SOURCE_MANIFEST.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def skipped(part):
    lower = part.lower()
    return (lower in SKIP or lower.startswith(('.', 'archive', 'backup', 'vendor'))
            or lower in {'credentials.py', 'secrets.py'})


def included(relative):
    parts = relative.parts
    if any(skipped(p) for p in parts):
        return False
    if relative.suffix == '.py' or relative.name in DOCS:
        return True
    if parts[0].endswith('_monitor'):
        return relative.name in {'config.json', 'FREEZE.json'} and len(parts) == 2
    return relative.name in META and len(parts) == 2


def safe_relative(value):
    # Do not let Path normalize away traversal or ambiguous manifest entries.
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(p in {'', '.', '..'} for p in value.split('/'))):
        raise ValueError('Invalid relative export path')
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError('Export paths must be relative')
    return relative


def checked_path(base, relative):
    relative = safe_relative(relative.as_posix())
    path = base
    for index, part in enumerate(relative.parts):
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed: {relative.as_posix()}')
        if index < len(relative.parts) - 1 and path.exists() and not path.is_dir():
            raise ValueError(f'Export path parent is not a directory: {relative.as_posix()}')
    try:
        path.resolve().relative_to(base)
    except ValueError:
        raise ValueError(f'Path escapes its workspace: {relative.as_posix()}') from None
    return path


def literal_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
        return ''.join(v.value for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_string(node.left), literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def credential_name(name):
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    # Environment-variable identifiers and file paths are not credential values.
    if re.search(r'_(?:env|env_var|environment_variable|path|file|filename|name)$', name, re.I):
        return False
    return bool(CREDENTIAL_NAME.search(name))


def assignment_name(target):
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        return literal_string(target.slice)
    return None


def placeholder(value):
    value = value.strip()
    return (not value or value.lower() in {'example', 'placeholder', 'changeme', 'dummy', 'test', 'fake'}
            or bool(re.fullmatch(r'\$?\{[^{}]+\}|\$[A-Za-z_]\w*|<[^<>]+>|YOUR_[A-Z_]+', value)))


def inspect_source(path, content=None):
    content = path.read_bytes() if content is None else content
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError(f'Export source is not UTF-8: {path.name}') from None
    if re.search(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----', text):
        raise ValueError(f'Private key marker: {path.name}')
    for match in CREDENTIAL_URL.finditer(text):
        value = match.group(1)
        expected = REVIEWED_QUERY_FIXTURE_LINES.get(path.name, {}).get(match.group(0))
        if expected is not None:
            line_start = text.rfind('\n', 0, match.start()) + 1
            line_end = text.find('\n', match.end())
            line = text[line_start:line_end if line_end >= 0 else len(text)]
            if hashlib.sha256(line.encode('utf-8')).hexdigest() == expected:
                continue
        # Alpha Vantage publishes this sample key. Restrict the exception to
        # its exact host and query value, not other providers or credentials.
        if match.group(0).lower() in {'?apikey=demo', '&apikey=demo'}:
            line_start = text.rfind('\n', 0, match.start()) + 1
            if re.search(r'https?://(?:www\.)?alphavantage\.co(?:/[^\s\"\'<>]*)?$',
                         text[line_start:match.start()], re.I):
                continue
        # A quoted environment lookup inside an f-string can terminate this
        # text match before the closing brace; it is still an expression.
        if not value.startswith(('{', '${', '$(')) and not placeholder(value):
            raise ValueError(f'Potential credential URL: {path.name}')
    for match in re.finditer(r'https?://[^/\s:@]+:[^/\s@]+@[^\s\"\'<>]+', text, re.I):
        if match.group(0) not in REVIEWED_TEST_URLS.get(path.name, set()):
            raise ValueError(f'Potential credential URL: {path.name}')

    def check(name, value, line=None):
        if (name is not None and isinstance(value, str) and credential_name(name) and not placeholder(value)
                and value not in REVIEWED_TEST_VALUES.get(path.name, set())):
            expected = REVIEWED_LITERAL_HASHES.get(path.name, {}).get(name)
            if expected == hashlib.sha256(value.encode('utf-8')).hexdigest():
                return
            location = f'{path.name}:{line}' if line else path.name
            raise ValueError(f'Credential-named literal: {location}')

    def check_assignment(target, value, line):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            for name, item in zip(target.elts, value.elts):
                check_assignment(name, item, line)
        else:
            check(assignment_name(target), literal_string(value), line)

    if path.suffix == '.py':
        try:
            tree = ast.parse(text, filename=path.name)
        except SyntaxError as exc:
            # SyntaxError's normal formatting can print source containing secrets.
            raise ValueError(f'Invalid Python syntax: {path.name}:{exc.lineno}') from None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    check_assignment(target, node.value, node.lineno)
            elif isinstance(node, ast.AnnAssign):
                check_assignment(node.target, node.value, node.lineno)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    name = literal_string(key)
                    if name is not None:
                        check(name, literal_string(value), node.lineno)
            elif isinstance(node, ast.keyword) and node.arg:
                check(node.arg, literal_string(node.value), node.lineno)
    elif path.suffix == '.json':
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON: {path.name}:{exc.lineno}') from None

        def inspect_json(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    check(name, item)
                    inspect_json(item)
            elif isinstance(value, list):
                for item in value:
                    inspect_json(item)
        inspect_json(data)
    else:
        # Launcher scripts may contain assignments; command/environment reads
        # are expressions, whereas quoted and bare literal values are checked.
        for number, line in enumerate(text.splitlines(), 1):
            match = re.match(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)', line)
            if match:
                value = match.group(2).strip().strip('\"\'')
                if not value.startswith(('$', '`')):
                    check(match.group(1), value, number)


def output_candidates(workspace, observer_only=False, exclude_monitors=False):
    source = checked_path(workspace, Path('outputs'))
    if not source.is_dir():
        raise ValueError('Workspace outputs directory is missing')
    def fail_walk(error):
        raise error

    for directory, dirs, files in os.walk(source, followlinks=False, onerror=fail_walk):
        parent = Path(directory)
        retained = []
        for name in sorted(dirs):
            if skipped(name):
                continue
            if parent == source:
                monitor = name.endswith('_monitor')
                if (observer_only and not monitor) or (exclude_monitors and monitor):
                    continue
            checked_path(workspace, (parent / name).relative_to(workspace))
            retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(source)
            if observer_only and (len(relative.parts) < 2 or not relative.parts[0].endswith('_monitor')):
                continue
            if exclude_monitors and relative.parts[0].endswith('_monitor'):
                continue
            if included(relative):
                yield Path('outputs') / relative


def work_candidates(workspace, manifest_path):
    if manifest_path.is_symlink():
        raise ValueError('Work source manifest must not be a symlink')
    try:
        manifest = json.loads(manifest_path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('Invalid work source manifest JSON') from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list):
        raise ValueError('Work source manifest must contain a files array')
    seen = set()
    for value in manifest['files']:
        relative = safe_relative(value)
        if (len(relative.parts) != 1 or relative.suffix != '.py'
                or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*\.py', value)
                or value.lower() in {'credentials.py', 'secrets.py'} or PRIVATE_WORK.search(value)):
            raise ValueError(f'Not an allowed top-level work source: {relative.as_posix()}')
        if value in seen:
            raise ValueError(f'Duplicate work source: {value}')
        seen.add(value)
        yield Path('work') / relative


def fixture_guards(relative, content):
    if relative.as_posix() not in {'outputs/daily_stock_monitor/test_monitor.py',
                                  'outputs/pressure_shadow_monitor/test_monitor.py'}:
        return content
    text = content.decode('utf-8')
    for method, fixture in [('test_feature_causality', 'stock_database/stocks.sqlite'),
                            ('test_research_accounts', 'strategy_selection/daily_equity.csv.gz')]:
        marker = '    def ' + method + '(self):'
        guard = f'    @unittest.skipUnless((P.parent / "{fixture}").exists(), "Requires unpublished historical research fixtures")\n'
        if marker not in text:
            raise ValueError(f'Expected private-fixture test is missing: {relative.as_posix()}')
        if guard + marker not in text:
            text = text.replace(marker, guard + marker)
    return text.encode('utf-8')


def export_workspace(workspace, observer_workspace=None, include_work_source=False,
                     work_source_manifest=None, include_local_launchers=False, root=None):
    root = Path(root or ROOT).resolve()
    workspace = Path(workspace).resolve()
    observer_workspace = Path(observer_workspace).resolve() if observer_workspace else None
    if workspace == root or observer_workspace == root:
        raise ValueError('Source workspace must differ from the destination checkout')
    if include_work_source != (work_source_manifest is not None):
        raise ValueError('--include-work-source and --work-source-manifest must be used together')
    candidates = [('research', workspace, rel) for rel in output_candidates(
        workspace, exclude_monitors=observer_workspace is not None)]
    if observer_workspace:
        candidates.extend(('observer', observer_workspace, rel)
                          for rel in output_candidates(observer_workspace, observer_only=True))
    if include_work_source:
        candidates.extend(('research', workspace, rel)
                          for rel in work_candidates(workspace, Path(work_source_manifest)))
    if include_local_launchers:
        for label, source in [('research', workspace), ('observer', observer_workspace)]:
            if source is not None:
                for name in LAUNCHERS[label]:
                    path = checked_path(source, Path(name))
                    if path.exists():
                        candidates.append((label, source, Path(name)))

    # Read, scan, and prepare every candidate before any destination mutation.
    # Hold checked bytes in memory so source changes cannot bypass this review.
    prepared, hashes, adaptations, sources = [], {}, {}, {}
    for label, source, relative in sorted(candidates, key=lambda item: item[2].as_posix()):
        path = checked_path(source, relative)
        if not path.is_file():
            raise ValueError(f'Export source is not a regular file: {relative.as_posix()}')
        destination = checked_path(root, relative)
        if destination.exists() and not destination.is_file():
            raise ValueError(f'Export destination is not a regular file: {relative.as_posix()}')
        if relative.as_posix() in hashes:
            raise ValueError(f'Duplicate export destination: {relative.as_posix()}')
        content = path.read_bytes()
        inspect_source(path, content)
        source_hash = hashlib.sha256(content).hexdigest()
        published = fixture_guards(relative, content)
        mode = stat.S_IMODE(path.stat().st_mode)
        prepared.append((destination, published, mode))
        key = relative.as_posix()
        hashes[key] = hashlib.sha256(published).hexdigest()
        sources[key] = {'workspace': label, 'path': key, 'sha256': source_hash,
                        'mode': oct(mode)}
        if published != content:
            adaptations[key] = {'source_sha256': source_hash,
                                'reason': 'Preserve missing-private-fixture test guards'}
    manifest_destination = checked_path(root, MANIFEST_PATH)
    if manifest_destination.exists() and not manifest_destination.is_file():
        raise ValueError('Export manifest destination is not a regular file')
    out = {'purpose': 'Source snapshot, not a complete data bundle or automatic account migration.',
           'files': hashes, 'sources': sources, 'publication_adaptations': adaptations,
           'excluded': ['credentials', 'raw/vendor data', 'third-party source',
                        'paper ledgers', 'saved trade plans', 'author-supplied ledger',
                        'generated forecasts and model fits', 'binary caches',
                        'unlisted work files', 'conversation recovery and private migration scripts']}
    for destination, content, mode in prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(mode)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--observer-workspace', type=Path,
                        help='Read *_monitor directories only from this independent workspace')
    parser.add_argument('--include-work-source', action='store_true',
                        help='Include explicitly reviewed top-level work/*.py files')
    parser.add_argument('--work-source-manifest', type=Path,
                        help='JSON object with a files array relative to research work/')
    parser.add_argument('--include-local-launchers', action='store_true',
                        help='Include only the explicit research and observer launcher allowlists')
    args = parser.parse_args()
    try:
        out = export_workspace(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Synchronized {len(out["files"])} source/config/protocol files; export hashes and test-only adaptations recorded.')


if __name__ == '__main__':
    main()
