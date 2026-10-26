"""Pure, unregistered peer-event aggregation; no I/O, orders, or returns.

Inputs are validated frozen latest-event fields, never a list of events from
which this module selects a replacement. Dates are official session labels;
the caller supplies the checksum-bound calendar and issuer/sector mappings.
"""
from datetime import date
import math

import numpy as np


RAW_NAMES = ('raw_peer_recent10', 'raw_negative_own_event10', 'raw_peer_event10')
HAND_NAMES = ('hand_peer_recent10', 'hand_negative_own_event10', 'hand_peer_event10')
EXTRA = RAW_NAMES + HAND_NAMES + ('source_available', 'common_available', 'peer_count')
METHOD = 'fixed_latest_event10_peer_predictive_aggregation_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _session(value, label):
    _require(isinstance(value, str), label + ' must be YYYY-MM-DD')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(label + ' must be YYYY-MM-DD') from exc
    _require(parsed.isoformat() == value, label + ' must be canonical YYYY-MM-DD')
    return value


def _identifier(value, label, optional=False):
    if optional and value is None:
        return None
    _require(isinstance(value, str) and bool(value) and value == value.strip(),
             label + ' must be a nonempty exact identifier or explicit None')
    return value


def _matrix(value, shape, label, numeric=False):
    raw = np.asarray(value)
    _require(raw.shape == shape, label + ' shape differs from original axes')
    if numeric:
        _require(all(isinstance(x, (int, float, np.integer, np.floating)) and
                     not isinstance(x, (bool, np.bool_)) for x in raw.flat),
                 label + ' must contain numeric nonboolean values')
        raw = np.asarray(raw, dtype=np.float64)
        _require(not np.isinf(raw).any(), label + ' contains infinity')
    return raw


def _mean(values):
    # Scale before summing to avoid overflow of a representable finite mean.
    count = len(values)
    terms = [float(x) / count for x in values]
    if any(x != 0. and y == 0. for x, y in zip(values, terms)):
        raise FloatingPointError('Peer mean underflowed a nonzero source')
    try:
        result = math.fsum(terms)
    except OverflowError as exc:
        raise FloatingPointError('Peer mean overflow') from exc
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite peer mean')
    return 0. if result == 0. else result


def _hand(raw):
    if raw == 0.:
        return 0.
    # tanh is exactly +/-1 at this scale in float64; avoid intermediate overflow.
    if abs(raw) > 50.:
        return math.copysign(.5, raw)
    result = .5 * math.tanh(raw / .05)
    if result == 0.:
        raise FloatingPointError('Hand transform underflowed a nonzero raw value')
    return result


def build_peer_features(meta, event10, matched_recent10, valid10, anchor_sessions,
                        event_ids, official_sessions, sector_benchmarks, issuer_keys):
    """Return ``{arrays, audit}`` on the unchanged monthly candidate axes.

    ``meta`` contains dates[M], symbols[M][W], etfs. All cell inputs are [M,W].
    event10/recent10 use NaN exactly where boolean valid10 is false. Anchors
    are YYYY-MM-DD or None; event IDs and mappings are exact strings or None.
    official_sessions is a sorted unique sequence of YYYY-MM-DD labels.
    The production caller enforces its frozen 122x49 scope; synthetic shapes
    are accepted here. No trading eligibility is an input or a peer filter.

    All unresolved repeated known non-ETF issuer keys in a monthly pool fail.
    Missing mappings never receive inferred identities. Raw and hand values
    are NaN outside one common mask; measured zero is positive exact zero.
    """
    _require(isinstance(meta, dict), 'meta must be a dictionary')
    dates, symbols, etfs = meta.get('dates'), meta.get('symbols'), meta.get('etfs')
    _require(isinstance(dates, list) and isinstance(symbols, list) and
             isinstance(etfs, (list, tuple)) and len(dates) == len(symbols),
             'Explicit original metadata axes and ETF list required')
    values = np.asarray(event10)
    _require(values.ndim == 2 and values.shape[0] == len(dates) and values.shape[1] > 0,
             'event10 must have rectangular original monthly axes')
    shape = values.shape
    for day in dates:
        _session(day, 'signal date')
    _require(dates == sorted(set(dates)), 'Signal dates must be unique and increasing')
    for names in symbols:
        _require(isinstance(names, list) and len(names) == shape[1], 'Candidate width mismatch')
        for symbol in names:
            _identifier(symbol, 'symbol')
        _require(len(set(names)) == len(names), 'Duplicate monthly ticker')
    for symbol in etfs:
        _identifier(symbol, 'ETF')
    _require(len(set(etfs)) == len(etfs), 'Duplicate ETF identifier')
    etfs = set(etfs)
    _require(isinstance(official_sessions, (list, tuple)) and bool(official_sessions),
             'Explicit official session sequence required')
    official = [_session(day, 'official session') for day in official_sessions]
    _require(official == sorted(set(official)), 'Official sessions must be unique and increasing')
    positions = {day: i for i, day in enumerate(official)}
    _require(all(day in positions for day in dates), 'Signal is not an official session')
    event = _matrix(event10, shape, 'event10', numeric=True)
    recent = _matrix(matched_recent10, shape, 'matched_recent10', numeric=True)
    valid = _matrix(valid10, shape, 'valid10')
    _require(valid.dtype == np.dtype(bool), 'valid10 must be a boolean matrix')
    _require(np.array_equal(valid, np.isfinite(event)) and
             np.array_equal(valid, np.isfinite(recent)), 'Exact event/recent valid10 masks required')
    anchors = _matrix(anchor_sessions, shape, 'anchor_sessions')
    ids = _matrix(event_ids, shape, 'event_ids')
    sectors = _matrix(sector_benchmarks, shape, 'sector_benchmarks')
    issuers = _matrix(issuer_keys, shape, 'issuer_keys')
    source = np.zeros(shape, dtype=bool)
    common = np.zeros(shape, dtype=bool)
    counts = np.zeros(shape, dtype=np.int64)
    arrays = {name: np.full(shape, np.nan, dtype=np.float64) for name in RAW_NAMES + HAND_NAMES}
    cells = []
    for i, (day, names) in enumerate(zip(dates, symbols)):
        seen = set()
        row_cells = []
        for j, symbol in enumerate(names):
            sector = _identifier(sectors[i, j], 'sector benchmark', optional=True)
            issuer = _identifier(issuers[i, j], 'issuer key', optional=True)
            event_id = _identifier(ids[i, j], 'event ID', optional=True)
            anchor = anchors[i, j]
            age = None
            if anchor is not None:
                _session(anchor, 'anchor session')
                _require(anchor in positions, 'Anchor is not an official session')
                age = positions[day] - positions[anchor]
            if valid[i, j]:
                _require(anchor is not None and event_id is not None, 'Valid event needs anchor and event ID')
                _require(age >= 0, 'Future valid event anchor')
            if symbol not in etfs and issuer is not None:
                _require(issuer not in seen, 'Unresolved duplicate issuer in original monthly pool')
                seen.add(issuer)
            if symbol in etfs:
                reason = 'etf'
            elif issuer is None:
                reason = 'unknown_issuer'
            elif sector is None:
                reason = 'unknown_sector'
            elif not valid[i, j]:
                reason = 'missing_event'
            elif age < 10:
                reason = 'immature_event'
            elif age > 63:
                reason = 'stale_event'
            else:
                reason = 'insufficient_peer_issuers'
                source[i, j] = True
            row_cells.append(dict(i=i, j=j, signal_date=day, symbol=symbol,
                sector_benchmark=sector, issuer_key=issuer, event_id=event_id,
                anchor_session=anchor, age_sessions=age, source_available=bool(source[i, j]),
                available=False, reason=reason, peer_count=0, peer_symbols=[],
                peer_issuer_keys=[], peer_event_ids=[], **{name: None for name in RAW_NAMES}))
        for j, cell in enumerate(row_cells):
            if not source[i, j]:
                continue
            peers = [k for k in range(shape[1]) if source[i, k]
                     and sectors[i, k] == sectors[i, j] and issuers[i, k] != issuers[i, j]]
            peers.sort(key=lambda k: (issuers[i, k], names[k]))
            count = len(peers)
            counts[i, j] = count
            cell.update(peer_count=count, peer_symbols=[names[k] for k in peers],
                        peer_issuer_keys=[issuers[i, k] for k in peers],
                        peer_event_ids=[ids[i, k] for k in peers])
            if count < 2:
                continue
            own_negative = -float(event[i, j]) / count
            if event[i, j] != 0. and own_negative == 0.:
                raise FloatingPointError('Own-event control underflowed a nonzero source')
            raw = (_mean([recent[i, k] for k in peers]),
                   0. if own_negative == 0. else own_negative,
                   _mean([event[i, k] for k in peers]))
            for name, hand_name, value in zip(RAW_NAMES, HAND_NAMES, raw):
                arrays[name][i, j] = value
                arrays[hand_name][i, j] = _hand(value)
                cell[name] = value
            common[i, j] = True
            cell.update(available=True, reason='available')
        cells.extend(row_cells)
    arrays.update(source_available=source, common_available=common, peer_count=counts)
    for name in RAW_NAMES + HAND_NAMES:
        _require(np.array_equal(np.isfinite(arrays[name]), common), 'Derived masks differ')
    return dict(arrays=arrays, audit=dict(method=METHOD, candidate_rows=int(np.prod(shape)),
        signal_dates=shape[0], candidates_per_date=shape[1], all_three_masks_equal=True,
        source_peer_trading_eligibility_used=False, cells=cells))
