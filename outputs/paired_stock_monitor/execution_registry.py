"""Resolve one atomically published execution route, without changing accounts.

An absent registry means the original v2 paths. A present but invalid registry
never falls back to v2: mixing or silently reviving execution ledgers is unsafe.
"""
import json
from pathlib import Path, PurePosixPath

P = Path(__file__).resolve().parent
ACTIVE_EXECUTION = P / 'ACTIVE_EXECUTION.json'
ACCOUNT_NAMES = frozenset({
    'daily_stock_monitor', 'pressure_shadow_monitor', 'resilience_shadow_monitor',
    'accel_shadow_monitor', 'high52_shadow_monitor', 'accel40_shadow_monitor',
})
VERSION = 'causal_replay_v3'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate execution registry key: ' + key)
        result[key] = value
    return result


def _relative_path(folder, value, label):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Invalid execution registry ' + label)
    relative = PurePosixPath(value)
    if relative.is_absolute() or '..' in relative.parts or '.' in value.split('/') or '//' in value:
        raise ValueError('Execution registry path escape: ' + label)
    target = folder / relative
    resolved = target.resolve()
    try:
        resolved.relative_to(folder.resolve())
    except ValueError:
        raise ValueError('Execution registry path escapes account: ' + label) from None
    # Even an internal alias can make a supposedly independent ledger point to
    # the original history. Require the declared v3 subtree itself, not aliases.
    if resolved != target.absolute():
        raise ValueError('Execution registry path aliases another location: ' + label)
    return target


def active_account_paths(folder):
    """Return the active ledger/export/adapter paths for a known daily account."""
    folder = Path(folder).absolute()
    legacy = folder / 'paper_ledger.sqlite'
    if ACTIVE_EXECUTION.is_symlink():
        raise ValueError('Execution registry must not be a symlink')
    if not ACTIVE_EXECUTION.exists():
        return dict(version='v2', ledger=legacy, exports=folder, adapter=P / 'evening_adapter.py')
    if folder.name not in ACCOUNT_NAMES:
        raise ValueError('Unknown execution account: ' + folder.name)
    registry = json.loads(ACTIVE_EXECUTION.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(registry, dict) or registry.get('version') != VERSION:
        raise ValueError('Unknown execution registry version')
    accounts = registry.get('accounts')
    if not isinstance(accounts, dict) or set(accounts) != ACCOUNT_NAMES:
        raise ValueError('Execution registry must contain exactly all six known accounts')
    for name, entry in accounts.items():
        if not isinstance(entry, dict) or set(entry) != {'ledger', 'exports'}:
            raise ValueError('Invalid execution account entry: ' + name)
        # Validate every entry, not only the requested account, before routing.
        if entry['ledger'] != 'execution_v3/paper_ledger.sqlite' or entry['exports'] != 'execution_v3':
            raise ValueError('Invalid independent v3 paths for account: ' + name)
    entry = accounts[folder.name]
    ledger = _relative_path(folder, entry['ledger'], 'ledger')
    exports = _relative_path(folder, entry['exports'], 'exports')
    if not ledger.is_file() or not exports.is_dir():
        raise ValueError('Active v3 execution ledger/export directory is missing')
    if legacy.exists() and ledger.samefile(legacy):
        raise ValueError('Active v3 ledger aliases original history')
    return dict(version=VERSION, ledger=ledger, exports=exports, adapter=P / 'causal_replay_v3.py')
