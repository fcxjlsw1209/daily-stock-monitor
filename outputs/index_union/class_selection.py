"""Select one eligible common class per issuer using certified raw dollar volume.

Coverage certificates belong to a specific signal date. This function neither
certifies prices nor infers identities from ticker strings.
"""
import math
from collections import defaultdict


def select_classes(members, bars, signal_session, sessions, coverage):
    if sessions != sorted(set(sessions)) or signal_session not in sessions:
        raise ValueError('Ordered unique calendar including signal session required')
    window = set([d for d in sessions if d <= signal_session][-63:])
    by_security = defaultdict(list)
    for b in bars:
        if b['date'] in window:
            by_security[b['security_id']].append(b)
    selected = []
    seen_issuers = set()
    for member in members:
        issuer = member['issuer_id']
        if issuer in seen_issuers:
            raise ValueError('Duplicate issuer')
        seen_issuers.add(issuer)
        ids = member['security_ids']
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Invalid eligible classes')
        scores = {}
        if len(ids) > 1:
            for security in ids:
                certificate = coverage.get(security, {})
                if not (certificate.get('verified') and certificate.get('signal_session') == signal_session
                        and certificate.get('source_sha256')):
                    raise ValueError('Incomplete class price coverage: '+security)
                data = by_security[security]
                if len({b['date'] for b in data}) != len(data):
                    raise ValueError('Duplicate class bar')
                values = []
                for b in data:
                    if b.get('basis') != 'raw' or not b.get('verified'):
                        raise ValueError('Class selection requires verified raw units')
                    close, volume = float(b['close']), float(b['volume'])
                    if not (math.isfinite(close) and close > 0 and math.isfinite(volume) and volume >= 0):
                        raise ValueError('Invalid class bar')
                    values.append(close * volume)
                if len(values) < 20:
                    raise ValueError('At least 20 completed sessions required per class')
                scores[security] = sum(values) / len(values)
            chosen = min(ids, key=lambda s: (-scores[s], s))
        else:
            chosen = ids[0]
        selected.append(dict(issuer_id=issuer, security_id=chosen,
                             indexes=member['indexes'], mean_raw_dollar_volume=scores.get(chosen)))
    return sorted(selected, key=lambda r:r['issuer_id'])


def select_available_classes(members, bars, signal_session, sessions, coverage, entry_available):
    """Apply the v3 observed signal-day exclusion before representative selection.

    entry_available is produced from verified identities/units and observed
    signal-day inputs, never from future dates. Unknown availability raises;
    missing candidate classes are excluded, not holdings. Remaining multi-class
    comparisons still require the original evidence and 20-session minimum.
    """
    candidates, exclusions = [], []
    seen_issuers = set()
    for member in members:
        issuer = member['issuer_id']
        if issuer in seen_issuers:
            raise ValueError('Duplicate issuer')
        seen_issuers.add(issuer)
        ids = member['security_ids']
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Invalid eligible classes')
        remaining = []
        for security in ids:
            available = entry_available.get(security)
            if not isinstance(available, bool):
                raise ValueError('Explicit signal-day availability required: '+security)
            if available:
                remaining.append(security)
            else:
                exclusions.append(dict(issuer_id=issuer, security_id=security,
                    signal_session=signal_session, reason='signal_inputs_unavailable',
                    affects_existing_holdings=False))
        if remaining:
            candidates.append(dict(member, security_ids=remaining))
    return dict(selected=select_classes(candidates,bars,signal_session,sessions,coverage),
                exclusions=exclusions)
