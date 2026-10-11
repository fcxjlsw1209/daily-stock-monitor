"""Fail-closed, effective-dated index membership. No current-list backfilling."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json
INDEXES={'NDX','SP100'}
def timestamp(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('Timezone required')
    return t

def entry_session(effective_at, published_at, trading_sessions):
    """First supplied exchange session strictly after both NY calendar dates."""
    if not trading_sessions or trading_sessions != sorted(set(trading_sessions)):
        raise ValueError('Ordered unique exchange sessions required')
    ny = ZoneInfo('America/New_York')
    boundary = max(timestamp(effective_at).astimezone(ny).date().isoformat(),
                   timestamp(published_at).astimezone(ny).date().isoformat())
    if trading_sessions[0] > boundary:
        raise ValueError('Calendar must cover entry boundary')
    return next((day for day in trading_sessions if day > boundary), None)

def build_union(indexes,signal_at,known_at,identities,trading_sessions=None):
    signal=timestamp(signal_at);known=timestamp(known_at)
    if known<signal:raise ValueError('Knowledge cutoff precedes signal')
    if set(indexes)!=INDEXES:raise ValueError('Both historical indexes required')
    membership={}
    signal_day=signal.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    for name in sorted(INDEXES):
        data=indexes[name];base=data['baseline'];coverage=data['coverage']
        if not base.get('verified') or not coverage.get('verified') or not base.get('source_sha256') or not coverage.get('evidence_sha256'):
            raise ValueError('Unverified baseline or event coverage')
        start=timestamp(base['effective_at'])
        if not start<=signal<=timestamp(coverage['through']):raise ValueError('Outside certified coverage')
        if timestamp(coverage['from'])>start:raise ValueError('Coverage gap after baseline')
        if timestamp(base['known_at'])>signal:raise ValueError('Future baseline evidence')
        current=set(base['security_ids'])
        pending={}
        if len(current)!=len(base['security_ids']):raise ValueError('Duplicate baseline securities')
        events=sorted(data['events'],key=lambda e:timestamp(e['effective_at']))
        seen=set()
        for event in events:
            if event['event_id'] in seen:raise ValueError('Duplicate event')
            seen.add(event['event_id']);effective=timestamp(event['effective_at'])
            if effective<=start:raise ValueError('Event precedes baseline')
            if effective>signal:continue
            if not event.get('verified') or not event.get('source_sha256'):raise ValueError('Unverified event')
            add=set(event['add']);remove=set(event['remove'])
            if remove and timestamp(event['known_at'])>signal:
                # A never-eligible delayed addition cannot affect the candidate pool.
                # Still block unknown removals of baseline or already-eligible members.
                observable_remove = {s for s in remove if s not in pending or
                                     (pending[s] is not None and pending[s] <= signal_day)}
                if observable_remove:
                    raise ValueError('Removal not publicly known at signal cutoff')
            if add&remove or len(add)!=len(event['add']) or len(remove)!=len(event['remove']):raise ValueError('Malformed event')
            if not remove<=current or add&current:raise ValueError('Broken membership chain')
            current=(current-remove)|add
            for security in remove:pending.pop(security,None)
            for security in add:
                pending[security]=entry_session(event['effective_at'],event['known_at'],trading_sessions)
        signal_day=signal.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        eligible={s for s in current if s not in pending or (pending[s] is not None and pending[s]<=signal_day)}
        for security in sorted(eligible):
            matches=[r for r in identities if r['security_id']==security and timestamp(r['effective_from'])<=signal<timestamp(r['effective_to'])]
            if len(matches)!=1:raise ValueError('Missing or ambiguous identity')
            r=matches[0]
            if not r.get('verified') or not r.get('source_sha256') or timestamp(r['known_at'])>signal:raise ValueError('Unverified identity')
            item=membership.setdefault(r['issuer_id'],dict(security_ids=set(),indexes=set()))
            item['security_ids'].add(security);item['indexes'].add(name)
    rows=[dict(issuer_id=i,security_ids=sorted(v['security_ids']),indexes=sorted(v['indexes'])) for i,v in sorted(membership.items())]
    payload=dict(universe_version='historical_ndx_sp100_union_v3_available_data',signal_at=signal_at,known_at=known_at,members=rows)
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload
