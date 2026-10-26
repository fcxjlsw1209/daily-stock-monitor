"""Date-only filing-clock proposal diagnostic, never a strategy admission.

Keep raw SEC fields. Delay until the first session strictly after every
observed calendar date; do not relabel an acceptance timestamp as earnings.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from after156_event_scope_local_review import Sources, require

SOURCE = 'work/after156_filing_event_feasibility.json'
SOURCE_SHA = '1af27cbd0777d6e0542fe56c8306efc3b747e018ac00bd7e70d7400c99abc6de'
CALENDAR = 'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
OUT = 'work/after156_filing_date_anchor_diagnostic.json'
NY = ZoneInfo('America/New_York')


def iso_day(raw):
    require(isinstance(raw, str), 'Date must be a string')
    parsed = date.fromisoformat(raw)
    require(parsed.isoformat() == raw, 'Noncanonical date')
    return parsed


def clock(raw):
    require(isinstance(raw, str), 'Timestamp must be a string')
    parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
            'Offset required; no naive timestamps')
    return parsed


def checked_calendar(sessions):
    require(isinstance(sessions, (list, tuple)) and len(sessions) > 1,
            'Nonempty session calendar required')
    parsed = [iso_day(x) for x in sessions]
    require(parsed == sorted(set(parsed)), 'Calendar must be sorted and unique')
    return parsed


def next_session(sessions, boundary):
    require(sessions[0] <= boundary < sessions[-1], 'Boundary outside calendar')
    return sessions[bisect_right(sessions, boundary)]


def first_open_after(sessions, instant):
    """Only a comparison under an explicitly hypothetical timestamp reading."""
    ny = instant.astimezone(NY)
    require(sessions[0] <= ny.date() < sessions[-1], 'Instant outside calendar')
    if ny.date() in sessions and ny.timetz().replace(tzinfo=None) < time(9, 30):
        return ny.date()
    return next_session(sessions, ny.date())


def anchor(record, sessions):
    filing = iso_day(record['filingDate'])
    accepted = clock(record['acceptanceDateTime'])
    utc_day = accepted.astimezone(timezone.utc).date()
    raw_day = accepted.date()
    require(abs((utc_day - filing).days) <= 2, 'Clock/date gap needs source review')
    require(abs((raw_day - filing).days) <= 2, 'Raw clock/date gap needs source review')
    if record.get('reportDate'):
        require(iso_day(record['reportDate']) <= filing, 'Reporting period after filing')
    boundary = max(filing, raw_day, utc_day)
    chosen = next_session(sessions, boundary)
    # Both are diagnostics, not a claim either interpretation is correct.
    alternatives = {
        'encoded_offset_taken_literally': first_open_after(sessions, accepted),
        'raw_wall_time_interpreted_as_new_york': first_open_after(
            sessions, accepted.replace(tzinfo=NY)),
    }
    require(all(chosen >= x for x in alternatives.values()),
            'Conservative bound precedes one hypothetical interpretation')
    return dict(
        accession=record['accessionNumber'], form=record['form'],
        report_date=record.get('reportDate'), raw_filing_date=record['filingDate'],
        raw_acceptance=record['acceptanceDateTime'],
        date_boundary=boundary.isoformat(), proposed_anchor=chosen.isoformat(),
        hypothetical_first_opens={k: v.isoformat() for k, v in alternatives.items()},
        extra_sessions_vs_hypothetical={k: sessions.index(chosen)-sessions.index(v)
                                       for k, v in alternatives.items()},
        exact_acceptance_instant_verified=False,
        source_definition='original_SEC_periodic_filing_not_earnings_release')


def main():
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    src = Sources()
    data = src.read(SOURCE, SOURCE_SHA)
    # Verify the source closure rather than trusting a previously written flag.
    for path, digest in data['input_hashes'].items():
        src.read_bytes(path, digest)
    sessions = checked_calendar(src.read(CALENDAR, CALENDAR_SHA)['sessions'])
    src.read_bytes('work/after156_filing_date_anchor.py')
    src.read_bytes('work/after156_event_scope_local_review.py')
    rows = []
    for issuer in data['issuers']:
        for event in issuer['filing_events']:
            rows.append(dict(symbol=issuer['symbol'], **anchor(event, sessions)))
    summary = {}
    for symbol in sorted({x['symbol'] for x in rows}):
        own = [x for x in rows if x['symbol'] == symbol]
        summary[symbol] = dict(events=len(own),
            extra_sessions_vs_encoded_offset=dict(sorted(Counter(
                x['extra_sessions_vs_hypothetical']['encoded_offset_taken_literally']
                for x in own).items())))
    src.recheck()
    result = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
        status='DATE_ONLY_CLOCK_PROPOSAL_DIAGNOSED_NOT_ADMITTED',
        proposal='First session strictly after max(filingDate, raw timestamp date, UTC timestamp date)',
        scope='The fixed three-issuer clock-validation pilot only; not final experiment scope.',
        source_sha256=SOURCE_SHA, summary=summary, rows=rows,
        pending=['Independent SEC filing-date semantics and source receipt review',
                 'Historical identity, amendments, pagination and coverage over full candidate scope',
                 'Freeze one clock definition before any factor calculation; no result-based clock selection'],
        limitations=['The bound covers two stated timestamp interpretations, not arbitrary incorrect source data.',
                     'Later-retrieved data can contain corrections; not an exact original historical data vintage.',
                     'A delayed filing-event clock cannot be described as an exact acceptance or first-news clock.'],
        admission=dict(exact_clock=False, date_only_clock=False, factor=False, strategy=False),
        activity_counts=dict(price_values_read=0, factor_calculations=0, ranks=0,
                             orders=0, returns=0, network_requests=0, new_trials=0),
        input_hashes=src.hashes)
    if args.write:
        with src.path(OUT).open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(status=result['status'], summary=summary, events=len(rows))))


if __name__ == '__main__':
    main()
