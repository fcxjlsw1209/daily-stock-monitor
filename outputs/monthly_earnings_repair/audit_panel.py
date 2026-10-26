"""Reproducible offline checks and next-source queue; never changes research pools."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd
from research_calendar import market_calendar
import hashlib

P = Path(__file__).resolve().parent
RETURN_KEYS = ('post_known_close_to_close_return', 'market_adjusted_return', 'benchmark_return',
               'reaction_volume_sum', 'reaction_volume_mean')


def audit(rows, availability, events):
    calendar = market_calendar(start='2010-01-01', end='2030-12-31')
    pairs = [(r['signal_date'], r['symbol']) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate monthly signal/candidate')
    known = {(r['signal_date'], r['symbol']): r for r in availability}
    if set(known) != set(pairs):
        raise ValueError('Monthly feature pool changed')
    event_lookup = {r['event_id']: r for r in events}
    if len(event_lookup) != len(events):
        raise ValueError('Duplicate event ID')
    for row in rows:
        original = known[row['signal_date'], row['symbol']]
        if original['event_id'] != row['event_id']:
            raise ValueError('Latest event silently changed')
        if row['status'] != original['status'] and row['status'] != 'missing_cached_price_file':
            raise ValueError('Availability state changed')
        asof = pd.Timestamp(row['signal_asof'])
        if asof != calendar.session_close(pd.Timestamp(row['signal_date'])):
            raise ValueError('Signal timestamp is not the canonical XNYS session close')
        if row['event_id'] and row['status'] == 'verified_event_available':
            event = event_lookup[row['event_id']]
            if event['symbol'] != row['symbol'] or pd.Timestamp(event['known_by_at']) > asof:
                raise ValueError('Future or wrong-issuer event')
        if row['status'] != 'verified_event_available' and row['reactions']:
            raise ValueError('Missing/N/A event received reaction values')
        if row['status'] == 'verified_event_available' and set(row['reactions']) != {'3', '10', '21'}:
            raise ValueError('Verified candidate must retain all three reaction horizons')
        for horizon, reaction in row['reactions'].items():
            if reaction.get('horizon') != int(horizon):
                raise ValueError('Reaction horizon identity mismatch')
            if reaction['known_by_at'] != event_lookup[row['event_id']]['known_by_at']:
                raise ValueError('Reaction uses a different event knowledge time')
            if reaction['status'] == 'ready':
                if not (pd.Timestamp(reaction['known_by_at']) <= pd.Timestamp(reaction['anchor_close_at'])
                        < pd.Timestamp(reaction['end_close_at']) <= asof):
                    raise ValueError('Future or invalid reaction window')
                if reaction['post_known_close_to_close_return'] is None:
                    raise ValueError('Ready reaction lacks return')
            elif any(reaction.get(key) is not None for key in RETURN_KEYS):
                raise ValueError('Missing reaction silently scored or zero-filled')
    return dict(monthly_candidate_rows=len(rows), unique_signal_dates=len({p[0] for p in pairs}),
        duplicate_signal_candidate_pairs=0, future_reaction_values=0, zero_filled_missing_reactions=0,
        row_status_counts=dict(Counter(r['status'] for r in rows)), same_as_monthly_availability=True)


def main():
    read = lambda name: json.loads((P/name).read_text())
    results={}
    for scope in ('MONTHLY','TRAINING'):
        files=[scope+'_FEATURES.json',scope+'_AVAILABILITY.json','EVENTS.json','CALENDAR_PROVENANCE.json',
               'audit_panel.py', 'event_schema.py', 'research_calendar.py']
        result=audit(read(files[0])['rows'],read(files[1]),read('EVENTS.json'))
        result['scope']=scope.lower()
        result['input_sha256']={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files}
        (P/(scope+'_PANEL_AUDIT.json')).write_text(json.dumps(result,indent=2)+'\n')
        results[scope]=result
    coverage = read('COVERAGE.json')
    pending = sorted((r for r in coverage if r['verified_events'] == 0), key=lambda r: (-r['member_months'], r['symbol']))
    queue = dict(selection_policy='Uncollected issuers ordered by membership months in the frozen research universe, never by subsequent returns. Preserve all candidates and source limits.',
        partial_history_followup=[r['symbol'] for r in coverage if r['verified_events'] and r['status'] != 'verified_history_collected'],
        warmup_note='Even sequence-complete recent archives can require earlier 2011–2015 warmup; quarterly regular releases do not certify all preliminary updates.',
        issuers=[{k:r[k] for k in ('symbol', 'member_months', 'first_member_signal', 'last_member_signal')} for r in pending])
    (P/'NEXT_COLLECTION_QUEUE.json').write_text(json.dumps(queue, indent=2)+'\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
