"""Read-only entry-price basis checks for explicitly verified quote revisions.

This exception does not restate fills, adjust positions, or process stock splits.
The caller verifies snapshot bytes and registers evidence before invoking it.
"""
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from numbers import Real
import re

import pandas as pd

REGISTRY_KEY = 'verified_entry_price_revisions_v1'
_REQUIRED = frozenset(('symbol', 'entry_date', 'recorded_entry_price',
                       'accepted_provider_open', 'verified_snapshot',
                       'source_evidence_sha256'))


def _day(value, label):
    if not isinstance(value, str):
        raise ValueError(label + ' must be an ISO calendar date')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(label + ' must be an ISO calendar date') from exc
    if parsed.isoformat() != value:
        raise ValueError(label + ' must be an ISO calendar date')
    return pd.Timestamp(parsed)


def _number(value, label, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(label + ' must be a finite number')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(label + ' must be a finite number') from exc
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(label + ' must be a finite positive number')
    return result


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(label + ' must be a nonempty identifier')
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate registry JSON key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonfinite registry JSON number: ' + value)


def _registry(db):
    row = db.execute('SELECT value FROM meta WHERE key=?', (REGISTRY_KEY,)).fetchone()
    if row is None:
        return []
    try:
        records = json.loads(row[0], object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid verified entry-price revision registry') from exc
    if not isinstance(records, list):
        raise ValueError('Verified entry-price revisions must be a JSON list')
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _REQUIRED:
            raise ValueError('Verified revision record has an invalid schema')
        _identifier(record['symbol'], 'revision symbol')
        _day(record['entry_date'], 'revision entry_date')
        old = _number(record['recorded_entry_price'], 'recorded_entry_price')
        accepted = _number(record['accepted_provider_open'], 'accepted_provider_open')
        if old == accepted:
            raise ValueError('Verified revision must describe an actual change')
        _identifier(record['verified_snapshot'], 'verified_snapshot')
        digest = record['source_evidence_sha256']
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('source_evidence_sha256 must be a lowercase SHA256')
        key = (record['symbol'], record['entry_date'], old)
        if key in seen:
            raise ValueError('Duplicate or conflicting verified entry-price revisions')
        seen.add(key)
    return records


def validate_entry_basis(db, symbol, position, frame, manifest):
    """Return matched evidence, or None for an unchanged price; never write state.

    A registered revision is reusable in a newer snapshot only when its provider
    Open is numerically identical. Any later change needs separate verification.
    Unregistered entries retain the legacy 0.1% tolerance. Once a revision is
    registered for an entry, only the exact recorded or accepted price is valid.
    Split checks apply only when using the revision exception, not the legacy path.
    """
    _identifier(symbol, 'symbol')
    entry = _day(position.get('entry_date'), 'position entry_date')
    recorded = _number(position.get('entry_price'), 'position entry_price')
    as_of = _day(manifest.get('as_of'), 'snapshot as_of')
    snapshot = _identifier(manifest.get('snapshot_id'), 'snapshot_id')
    if as_of < entry:
        raise ValueError('Snapshot precedes position entry')
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None:
        raise ValueError('Expected naive session-date index')
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered snapshot sessions')
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError('Expected normalized session dates')
    if entry not in frame.index or as_of not in frame.index:
        raise ValueError('Snapshot lacks required entry/as-of session')
    if (frame.index > as_of).any():
        raise ValueError('Snapshot contains sessions after as_of')
    if 'Open' not in frame:
        raise ValueError('Snapshot lacks Open')
    observed = _number(frame.at[entry, 'Open'], 'provider entry Open')
    matching = [record for record in _registry(db)
                if record['symbol'] == symbol and record['entry_date'] == position['entry_date']
                and _number(record['recorded_entry_price'], 'recorded_entry_price') == recorded]
    if not matching:
        # Preserve the frozen guard's tolerance and float comparison exactly.
        if abs(float(observed) / float(recorded) - 1) <= .001:
            return None
        raise ValueError('Unverified corporate action or entry-price revision: ' + symbol)
    if observed == recorded:
        return None
    record = matching[0]
    if _number(record['accepted_provider_open'], 'accepted_provider_open') != observed:
        raise ValueError('Registered entry-price revision changed again: ' + symbol)
    if 'Stock Splits' not in frame:
        raise ValueError('Revision exception requires Stock Splits metadata')
    for value in frame.loc[(frame.index > entry) & (frame.index <= as_of), 'Stock Splits']:
        split = _number(value, 'Stock Splits', positive=False)
        if split not in (Decimal(0), Decimal(1)):
            raise ValueError('Stock split after entry requires separate verification: ' + symbol)
    return dict(record, kind='verified_entry_price_revision', observed_snapshot=snapshot)
