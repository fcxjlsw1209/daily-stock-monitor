"""Export authored source and an explicit metadata allowlist, never runtime data.

Usage: python scripts/sync_workspace.py /absolute/research/workspace
The destination is this checkout. Existing unrelated files are never removed.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'sources', 'additional_sources', 'vendor', 'vendor_netcdf', '__pycache__',
        'archives', 'archive', 'backups', 'backup', 'stock-signal-monitor', 'prices',
        'plans', 'snapshots', 'cache'}
DOCS = {'RULES.md', 'POLICY.md', 'PROTOCOL.md', 'TIMESTAMP_FIX.md'}
META = {'PLAN.json', 'GRID.json', 'MODEL_CONFIGS.json', 'FROZEN.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(relative):
    parts = relative.parts
    if any(p in SKIP or p.startswith(('archive', 'backup', 'vendor')) for p in parts):
        return False
    if relative.suffix == '.py' or relative.name in DOCS:
        return True
    monitor = parts[0].endswith('_monitor')
    if monitor:
        return relative.name in {'config.json', 'FREEZE.json'} and len(parts) == 2
    return relative.name in META and len(parts) == 2


def inspect_source(path):
    text = path.read_text()
    if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', text):
        raise ValueError(f'Private key marker: {path.name}')
    if re.search(r'(?i)(?:api_?key|access_token|token)=[a-z0-9]{12,}', text):
        raise ValueError(f'Potential credential URL: {path.name}')
    if path.suffix == '.py':
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                name = ' '.join(ast.unparse(x) for x in node.targets)
                if re.search(r'(?i)(api_?key|password|secret|access_?token)', name) and len(node.value.value) > 10:
                    raise ValueError(f'Credential-named literal: {path.name}:{node.lineno}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    source = args.workspace.resolve() / 'outputs'
    assert source != ROOT / 'outputs' and source.is_dir()
    selected = sorted(f for f in source.rglob('*') if f.is_file() and not f.is_symlink() and included(f.relative_to(source)))
    # Review every selected file before making any destination changes.
    for f in selected:
        inspect_source(f)
    manifest = {}
    adaptations = {}
    for f in selected:
        rel = Path('outputs') / f.relative_to(source)
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        assert digest(dest) == digest(f)
        # Preserve the remote repository's existing source-only fixture guards.
        # These alter tests only, never frozen execution code or configurations.
        if str(rel) in {'outputs/daily_stock_monitor/test_monitor.py',
                         'outputs/pressure_shadow_monitor/test_monitor.py'}:
            content = dest.read_text()
            for method, fixture in [('test_feature_causality', 'stock_database/stocks.sqlite'),
                                    ('test_research_accounts', 'strategy_selection/daily_equity.csv.gz')]:
                marker = '    def ' + method + '(self):'
                guard = f'    @unittest.skipUnless((P.parent / "{fixture}").exists(), "Requires unpublished historical research fixtures")\n'
                assert marker in content
                content = content.replace(marker, guard + marker)
            dest.write_text(content)
            adaptations[str(rel)] = {'source_sha256': digest(f), 'reason': 'Preserve missing-private-fixture test guards'}
        manifest[str(rel)] = digest(dest)
    out = {'purpose': 'Source snapshot, not a complete data bundle or automatic account migration.',
           'files': manifest, 'publication_adaptations': adaptations,
           'excluded': ['credentials', 'raw/vendor data', 'third-party source',
                                           'paper ledgers', 'saved trade plans', 'author-supplied ledger',
                                           'generated forecasts and model fits', 'binary caches']}
    (ROOT / 'docs/WORKSPACE_SOURCE_MANIFEST.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'Synchronized {len(manifest)} source/config/protocol files; export hashes and test-only adaptations recorded.')


if __name__ == '__main__':
    main()
