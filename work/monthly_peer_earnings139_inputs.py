"""Read-only input adapter for a prospective peer earnings experiment.

This is not a reconstruction of repaired event returns from raw prices.
Call only after registration. The existing frozen event validator checks
availability and calculates the matched recent control from the old panel.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPAIR = 'outputs/monthly_earnings_repair'
FEATURES = 'outputs/monthly_earnings_factor_research/earnings_features.py'
CAPS = 'outputs/monthly_top35_extension/CAP_COLLECTION.json'
ALIASES = {'META': 'research_equity_FB', 'BRK-B': 'BRK.B'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def issuer_map(records):
    """Reuse the existing actual() aliases and vendor whole-issuer URLs."""
    result = {}
    for record in records:
        symbol, url = record['symbol'], record['url']
        require(isinstance(symbol, str) and isinstance(url, str) and bool(url),
                'Invalid registered issuer metadata')
        name = ALIASES.get(symbol, symbol)
        require(name not in result or result[name] == url,
                'Conflicting issuer identity for ' + name)
        result[name] = url
    return result


def validate_latest(meta, availability, events, document, calendar):
    """Validate the latest *known* event, not the latest completed reaction.

    A newer immature event must never be replaced with an older mature event.
    This check makes no completeness claim beyond the frozen scoped audit.
    """
    expected = {(day, name) for day, names in zip(meta['dates'], meta['symbols'])
                for name in names}
    require(len(expected) == sum(map(len, meta['symbols'])), 'Duplicate original candidates')
    rows = {(r['signal_date'], r['symbol']): r for r in document['rows']}
    available = {(r['signal_date'], r['symbol']): r for r in availability}
    require(set(rows) == set(available) == expected, 'Event candidate axes changed')
    require(len(rows) == len(document['rows']) and len(available) == len(availability),
            'Duplicate event/availability candidate')
    by_symbol, by_id = defaultdict(list), {}
    for event in events:
        key = event['event_id']
        require(key not in by_id, 'Duplicate event ID')
        by_id[key] = event
        require(event['status'] == 'ready' or event.get('known_by_at') is None,
                'Unready event has a knowledge timestamp')
        if event['status'] == 'ready':
            known = pd.Timestamp(event['known_by_at'])
            require(known.tz is not None, 'Event needs timezone')
            by_symbol[event['symbol']].append((known, key))
    for group in by_symbol.values():
        group.sort()
    etfs = set(meta['etfs'])
    for (day, name), row in rows.items():
        old = available[day, name]
        require(row['status'] == old['status'] and row['event_id'] == old['event_id'],
                'Frozen availability identity mismatch')
        close = calendar.session_close(pd.Timestamp(day))
        require(pd.Timestamp(row['signal_asof']) == close, 'Wrong signal close')
        if name in etfs:
            require(row['status'] == 'not_applicable_etf' and row['event_id'] is None
                    and not row['reactions'], 'ETF has event values')
            continue
        known = [(stamp, event_id) for stamp, event_id in by_symbol[name] if stamp <= close]
        if known:
            require(len(known) == 1 or known[-1][0] != known[-2][0],
                    'Ambiguous latest event timestamp')
        latest = known[-1][1] if known else None
        require(row['event_id'] == latest, 'Not the latest known event')
        if latest is not None:
            require(row['status'] == 'verified_event_available', 'Known event marked missing')
            for reaction in row['reactions'].values():
                require(reaction['known_by_at'] == by_id[latest]['known_by_at'],
                        'Reaction event knowledge mismatch')
        else:
            require(row['status'] != 'verified_event_available' and not row['reactions'],
                    'Unknown event scored')
    return rows


def load(root, reader, meta, panel, official_sessions, benchmark_map):
    """Use reader.json/sha/path so every consumed file belongs to PLAN pins."""
    root = Path(root).resolve()
    readiness = reader.json(REPAIR + '/SCOPED_READINESS.json')
    require(readiness['scoped_post_known_drift_data_ready'] is True
            and readiness['unresolved_count'] == 0 and not readiness['unresolved']
            and not readiness['invalid_candidate_price_windows']
            and readiness['protected_files_changed'] == 0, 'Frozen event readiness failed')
    for name, digest in readiness['inputs'].items():
        source = name if Path(name).is_absolute() else REPAIR + '/' + name
        require(reader.sha(source) == digest, 'Readiness binding mismatch: ' + name)
    panel_audit = reader.json(REPAIR + '/MONTHLY_PANEL_AUDIT.json')
    for name, digest in panel_audit['input_sha256'].items():
        require(reader.sha(REPAIR + '/' + name) == digest, 'Panel audit binding: ' + name)
    document = reader.json(REPAIR + '/MONTHLY_FEATURES.json')
    availability = reader.json(REPAIR + '/MONTHLY_AVAILABILITY.json')
    events = reader.json(REPAIR + '/EVENTS.json')

    path = reader.path(FEATURES)
    reader.sha(path)
    for relative in (REPAIR + '/research_calendar.py',
                     'outputs/index_union/CALENDAR_EXCEPTIONS.json',
                     'outputs/index_union/sources/closure_20250109.html'):
        reader.sha(relative)
    spec = importlib.util.spec_from_file_location('peer139_frozen_earnings_validator', path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    calendar = validator._calendar()
    observed = [str(s.date()) for s in calendar.sessions
                if official_sessions[0] <= str(s.date()) <= official_sessions[-1]]
    require(observed == list(official_sessions), 'Official event calendar differs')
    rows = validate_latest(meta, availability, events, document, calendar)
    event, event_audit = validator.build_features(document, meta, panel['adj'], meta['etfs'])
    issuers = issuer_map(reader.json(CAPS)['records'])
    shape = event['event10'].shape
    anchors = np.full(shape, None, dtype=object)
    ids = np.full(shape, None, dtype=object)
    sectors = np.full(shape, None, dtype=object)
    keys = np.full(shape, None, dtype=object)
    for i, (day, names) in enumerate(zip(meta['dates'], meta['symbols'])):
        mapping = benchmark_map(day, names, panel['sectors'], meta['etfs'])
        for j, symbol in enumerate(names):
            row = rows[day, symbol]
            ids[i, j] = row['event_id']
            if symbol not in meta['etfs']:
                sectors[i, j] = mapping.get(symbol)
                keys[i, j] = issuers.get(symbol)
            if event['valid10'][i, j]:
                anchors[i, j] = row['reactions']['10']['anchor_session']
    normalized = dict(meta=meta, event10=event['event10'],
        matched_recent10=event['matched_recent10'], valid10=event['valid10'],
        anchor_sessions=anchors, event_ids=ids, official_sessions=official_sessions,
        sector_benchmarks=sectors, issuer_keys=keys)
    provenance = dict(event_validator_audit=event_audit,
        original_event10_values_preserved=True, raw_event_prices_reconstructed=False,
        latest_known_event_independently_checked=True,
        unknown_issuer_symbols=sorted({s for names in meta['symbols'] for s in names
                                      if s not in meta['etfs'] and s not in issuers}),
        source_semantics='Frozen post-known-close drift; not immediate announcement surprise.',
        mapping_limitation='Inherited sector proxy and issuer URL identity, not new PIT classification proof.')
    return normalized, provenance
