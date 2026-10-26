"""Metadata-only earnings-clock census; no prices, financial values or models.

freeze binds a write-once specification before run computes coverage counts.
Observed quarterly continuity never certifies archive completeness.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_event_clock_census155'
REPAIR = ROOT / 'outputs/monthly_earnings_repair'
INPUTS = {
    'meta': 'outputs/monthly_local_linear70/META.json',
    'calendar': 'outputs/monthly_paired_target143/CALENDAR.json',
    'events': 'outputs/monthly_earnings_repair/EVENTS.json',
    'readiness': 'outputs/monthly_earnings_repair/SCOPED_READINESS.json',
    'event_schema': 'outputs/monthly_earnings_repair/event_schema.py',
    'scope_audit': 'outputs/monthly_earnings_repair/audit_scope.py',
    'boundaries': 'outputs/monthly_earnings_repair/BOUNDARY_OBSERVATIONS.json',
    'resolutions': 'outputs/monthly_earnings_repair/SCOPE_RESOLUTIONS.json',
    'hints': 'outputs/monthly_earnings_repair/VENDOR_HINTS.json',
    'prior_review': 'work/after152_additional_mechanism_review.json',
}
PINS = {
    'meta': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'calendar': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'events': '1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103',
    'readiness': '89cc0a0284128c827fd9e7b5e15244b952dc61f33ae12812e79e4c3e4e6a554a',
    'event_schema': 'cdac20a9dadd88dc7b7dffc7937651d0a8b7473b0255957a572b1ea561ccedff',
    'scope_audit': '3e57f7ba8d38998f837cbcb5cb8f913a195e5e5c95351c7bd21b0d1c39c11804',
    'boundaries': '9e1267ad97b602d435dea1f67e2f0132d3fa0f9834fda913132afefffead81f8',
    'resolutions': 'b70eb29b1312357afc88be1a37fb8aa0cb4c3fa709c5fbeef58ad21c359d1330',
    'hints': '0e5b5270953de96bec0a5729056b60ee148582cffd01c6017cedefc5b4b9e770',
    'prior_review': 'c5bab776403d7504a419a24bcacf5edeba8516220e279366f43c0c4f9bcfdf57',
}
ZERO = dict(price_values_read=0, financial_values_extracted=0, factor_calculations=0,
            model_fits=0, ranks=0, orders=0, returns=0, new_strategy_trials=0)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def local(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    # Legacy absolute references are translated in memory by run_local.sh.
    check('CloudStorage' not in p.parts and '..' not in p.parts, 'Forbidden path: ' + str(p))
    check(p.is_relative_to(ROOT), 'Outside local research root: ' + str(p))
    check(not p.is_symlink(), 'Symlink source: ' + str(p))
    check(p.resolve().is_relative_to(ROOT), 'Resolved source escapes root')
    return p


def rel(path):
    return str(local(path).relative_to(ROOT))


def write(path, value):
    blob = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(blob); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(temporary)


def day(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return None
    try: return date.fromisoformat(value).isoformat()
    except ValueError: return None


def quarter(event):
    y, q = event.get('fiscal_year'), event.get('fiscal_quarter')
    return y * 4 + q - 1 if type(y) is int and type(q) is int and 1900 < y < 2100 and 1 <= q <= 4 else None


def clock_issues(event, sessions):
    issues = []
    pub, avail, end = (day(event.get(k)) for k in ('publication_date', 'available_session', 'official_actual_period_end'))
    if not pub: issues.append('invalid_publication_date')
    if not avail or avail not in sessions: issues.append('invalid_available_session')
    if not end or (pub and end > pub): issues.append('invalid_period_end')
    if quarter(event) is None: issues.append('invalid_fiscal_quarter')
    try:
        known = datetime.fromisoformat(event['known_by_at'].replace('Z', '+00:00'))
        check(known.tzinfo is not None, 'aware known_by required')
        if avail and known.astimezone(timezone.utc).date().isoformat() > avail:
            issues.append('known_by_after_available_date')
    except (ValueError, TypeError, KeyError, AttributeError):
        issues.append('invalid_known_by_timestamp')
    if event.get('precision') == 'date_only' and pub:
        i = bisect_right(sessions, pub)
        if i >= len(sessions) or avail != sessions[i]: issues.append('not_first_session_after_publication_date')
    elif event.get('precision') != 'exact':
        issues.append('unknown_clock_precision')
    if pub and avail and avail < pub: issues.append('available_before_publication')
    if event.get('status') != 'ready': issues.append('source_status_not_ready')
    if not urlparse(event.get('source_url') or '').netloc: issues.append('missing_source_url')
    return sorted(set(issues))


def references(events, boundaries, resolutions):
    refs = []
    for e in events:
        refs.append(dict(owner=e['event_id'], role='primary', path=e.get('evidence_path'), sha256=e.get('evidence_sha256')))
        for r in e.get('additional_evidence') or []:
            refs.append(dict(owner=e['event_id'], role=r.get('role', 'additional'), path=r.get('evidence_path'), sha256=r.get('sha256')))
    for kind, records in (('boundary', boundaries), ('resolution', resolutions)):
        for i, e in enumerate(records):
            for r in e.get('evidence') or []:
                refs.append(dict(owner=kind + ':' + str(i), role=kind, path=r.get('path'), sha256=r.get('sha256')))
    for r in refs:
        if r['path']:
            r['path'] = rel(r['path'])
    return refs


def verify_refs(refs):
    cached, checked = {}, []
    for r in refs:
        row = dict(r); key = r['path']
        if not key or not isinstance(r['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', r['sha256']):
            row.update(status='invalid_reference', actual_sha256=None)
        else:
            if key not in cached:
                p = local(key)
                if not p.exists(): cached[key] = ('missing_file', None)
                elif not stat.S_ISREG(p.stat().st_mode): cached[key] = ('not_regular_file', None)
                elif getattr(p.stat(), 'st_flags', 0) & 0x40000000: cached[key] = ('dataless_file_not_opened', None)
                else: cached[key] = ('read', sha(p))
            status, actual = cached[key]
            row.update(status=('verified' if actual == r['sha256'] else 'hash_mismatch') if status == 'read' else status,
                       actual_sha256=actual)
        checked.append(row)
    return checked


def sequence_issues(events):
    result = []
    for a, b in zip(events, events[1:]):
        qa, qb = quarter(a), quarter(b)
        if qa is None or qb is None: kind = 'unresolved_quarter_label'
        elif qb == qa: kind = 'duplicate_quarter_observation'
        elif qb < qa: kind = 'nonmonotone_quarter_observation'
        elif qb > qa + 1: kind = 'unobserved_intervening_quarter_labels'
        else: continue
        result.append(dict(kind=kind, before=a['event_id'], after=b['event_id'],
                           quarter_step=None if qa is None or qb is None else qb - qa))
    return result


def self_test():
    # Metadata only. No real sources are read by these boundary checks.
    s = ['2020-01-02', '2020-01-03', '2020-01-06']
    e = dict(publication_date='2020-01-03', available_session='2020-01-06',
             official_actual_period_end='2019-12-31', known_by_at='2020-01-06T21:00:00+00:00',
             fiscal_year=2019, fiscal_quarter=4, precision='date_only', status='ready', source_url='https://example.org/a')
    check(clock_issues(e, s) == [], 'Synthetic weekend clock')
    check('not_first_session_after_publication_date' in clock_issues(dict(e, available_session='2020-01-03'), s), 'Synthetic future clock')
    a = dict(e, event_id='a'); b = dict(e, event_id='b', fiscal_year=2020, fiscal_quarter=1)
    check(sequence_issues([a, b]) == [], 'Synthetic annual rollover')
    check(sequence_issues([a, dict(b, fiscal_quarter=2)])[0]['quarter_step'] == 2, 'Synthetic missing quarter')
    check(sequence_issues([a, dict(a, event_id='repeat')])[0]['kind'] == 'duplicate_quarter_observation', 'Synthetic duplicate')
    check(day('2020-02-30') is None and day('2020-2-2') is None, 'Synthetic canonical date')
    return 6


def freeze():
    check(not (OUT / 'SPEC.json').exists(), 'SPEC already frozen')
    self_test()
    inputs = {}
    for key, name in INPUTS.items():
        p = local(name); actual = sha(p)
        check(actual == PINS[key], 'Input pin mismatch: ' + key)
        inputs[name] = actual
    events = load(local(INPUTS['events']))
    check(len(events) == 3771 and len({e['event_id'] for e in events}) == len(events), 'Expected unique3771 events')
    spec = dict(schema_version=1, stage='FROZEN_METADATA_ONLY_EVENT_CLOCK_CENSUS',
        created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
        source_hashes=inputs, source_code_hashes={rel(__file__): sha(Path(__file__))},
        horizons=dict(formation_return_start=-251, formation_return_end=-42, formation_sessions=210,
                      prior_event_warmup_sessions=20, event_anchor_start=-271, event_anchor_end=-42,
                      prospective_post_known_offsets=[2, 20], price_observations_read=0),
        scope=dict(query_months=122, query_width=49, stocks=73, stock_slots=3904, source_events=3771,
                   preserve_etf_identity_but_do_not_create_earnings_for_etfs=True),
        evidence_expectations=references(events, load(local(INPUTS['boundaries'])), load(local(INPUTS['resolutions']))),
        rules=['Only available_session <= formation end may enter per-slot observed sequences.',
               'Pre-warmup predecessor is a boundary observation, not part of the event count.',
               'Quarter-label gaps and 120-day intervals are diagnostic flags, never proof of missing releases.',
               'Continuous observed quarter labels do not prove complete original archives.',
               'Existing candidate-date readiness and resolutions cannot certify the new whole-window scope.',
               'Exact intraday first-public clocks are inherited claims, not newly verified here.',
               'Hash checking original bytes does not parse their financial contents or independently authenticate publication.',
               'All slots remain completeness_uncertified; no factor eligibility or strategy admission is created.'],
        self_tests_passed=6, **ZERO)
    write(OUT / 'SPEC.json', spec)
    print(json.dumps(dict(spec_sha256=sha(OUT / 'SPEC.json'), stage=spec['stage'])))


def validate_spec(expected):
    check(sha(OUT / 'SPEC.json') == expected, 'SPEC digest mismatch')
    spec = load(OUT / 'SPEC.json')
    check(spec['stage'] == 'FROZEN_METADATA_ONLY_EVENT_CLOCK_CENSUS' and spec['runtime_python'] == sys.version, 'SPEC runtime/stage')
    for path, digest in dict(spec['source_hashes'], **spec['source_code_hashes']).items():
        check(sha(local(path)) == digest, 'Frozen source changed: ' + path)
    check(all(spec[k] == v and type(spec[k]) is int for k, v in ZERO.items()), 'Forbidden activity')
    return spec


def run(expected):
    spec = validate_spec(expected)
    check(not any((OUT / n).exists() for n in ('EVENT_AUDIT.json', 'EVIDENCE_AUDIT.json', 'SLOTS.json', 'CENSUS.json', 'MANIFEST.json')), 'Census already started; preserve partial files')
    meta, calendar, events, readiness, hints, resolutions = [load(local(INPUTS[k])) for k in ('meta', 'calendar', 'events', 'readiness', 'hints', 'resolutions')]
    sessions = calendar['sessions']; positions = {d: i for i, d in enumerate(sessions)}
    check(sessions == sorted(set(sessions)) and all(day(d) for d in sessions), 'Calendar axis')
    etfs = set(meta['etfs']); stocks = sorted({s for row in meta['symbols'] for s in row if s not in etfs})
    check(len(meta['dates']) == 122 and all(len(row) == len(set(row)) == 49 for row in meta['symbols']) and len(stocks) == 73, 'Frozen query axis')
    refs = verify_refs(spec['evidence_expectations']); by_owner = defaultdict(list)
    for r in refs: by_owner[r['owner']].append(r)
    event_rows, groups = [], defaultdict(list)
    for e in events:
        checks = by_owner[e['event_id']]
        issues = clock_issues(e, sessions)
        failures = [r for r in checks if r['status'] != 'verified']
        row = {k: e.get(k) for k in ('event_id', 'symbol', 'status', 'fiscal_year', 'fiscal_quarter', 'official_actual_period_end',
               'publication_date', 'precision', 'known_by_at', 'available_session', 'source_kind', 'source_url',
               'first_public_time_verified', 'known_by_is_actual_publication', 'historical_archive_limitation')}
        row.update(source_references=checks, metadata_issues=issues, inherited_event_issues=e.get('issues', []),
            all_recorded_evidence_hashes_verified=bool(checks) and not failures,
            clock_metadata_consistent=not issues, intraday_clock_independently_reverified=False,
            full_calendar_completeness_verified=False, source_status_is_inherited=True)
        event_rows.append(row); groups[e['symbol']].append(row)
    for group in groups.values(): group.sort(key=lambda e: (e['available_session'] or '', e['event_id']))
    slot_rows = []
    for i, (signal, names) in enumerate(zip(meta['dates'], meta['symbols'])):
        check(signal in positions and positions[signal] >= 271, 'Insufficient frozen calendar for formation window')
        t = positions[signal]; warm, begin, end = sessions[t-271], sessions[t-251], sessions[t-42]
        for j, symbol in enumerate(names):
            if symbol in etfs: continue
            known = [e for e in groups[symbol] if day(e['available_session']) and e['available_session'] <= end]
            before = [e for e in known if e['available_session'] < warm]
            inside = [e for e in known if warm <= e['available_session']]
            chain = before[-1:] + inside
            seq_issues = sequence_issues(chain)
            applicable = [e for e in inside if e['available_session'] in positions and
                          positions[e['available_session']]+2 <= t-42 and positions[e['available_session']]+20 >= t-251]
            bad = [e['event_id'] for e in chain if e['metadata_issues'] or e['inherited_event_issues'] or not e['all_recorded_evidence_hashes_verified']]
            unresolved_dates = [e['event_id'] for e in groups[symbol] if not day(e['available_session'])]
            vendor = [h for h in hints['hints'] if h.get('symbol') == symbol and day(h.get('provider_reported_date')) and warm <= h['provider_reported_date'] <= end]
            unmatched = sorted({h['provider_reported_date'] for h in vendor if not any(e['publication_date'] and abs((date.fromisoformat(e['publication_date'])-date.fromisoformat(h['provider_reported_date'])).days) <= 3 for e in known)})
            local_res = [r['issue_id'] for r in resolutions if r.get('issue_id', '').endswith('|' + signal + '|' + symbol)]
            long_gaps = [dict(before=a['event_id'], after=b['event_id'], calendar_days=(date.fromisoformat(b['available_session'])-date.fromisoformat(a['available_session'])).days)
                         for a,b in zip(chain,chain[1:]) if (date.fromisoformat(b['available_session'])-date.fromisoformat(a['available_session'])).days > 120]
            blockers = ['whole_window_archive_completeness_not_certified', 'right_boundary_absence_not_proof_of_no_release']
            if not before: blockers.append('no_observed_pre_warmup_boundary_event')
            if not inside: blockers.append('no_observed_events_in_consumed_window')
            if seq_issues: blockers.append('observed_quarter_sequence_requires_review')
            if bad: blockers.append('source_or_clock_issues')
            if unmatched: blockers.append('unmatched_vendor_date_hints')
            if unresolved_dates: blockers.append('issuer_records_with_unresolved_anchor_dates')
            if long_gaps: blockers.append('long_observed_reporting_interval_requires_review')
            slot_rows.append(dict(signal_index=i, candidate_index=j, symbol=symbol, signal_date=signal,
                event_anchor_window=[warm,end], formation_window=[begin,end],
                predecessor_event_id=before[-1]['event_id'] if before else None,
                observed_event_ids=[e['event_id'] for e in inside], observed_event_count=len(inside),
                intersecting_post_known_window_event_ids=[e['event_id'] for e in applicable],
                quarter_sequence_issues=seq_issues, observed_quarter_labels_contiguous=bool(inside) and not seq_issues,
                long_reporting_intervals=long_gaps, evidence_or_clock_problem_event_ids=bad,
                unresolved_anchor_event_ids=unresolved_dates, unmatched_vendor_hint_dates=unmatched,
                existing_candidate_scope_resolution_ids=local_res, inherited_resolution_scope_not_extended=True,
                blockers=blockers, complete_event_calendar_verified=False, factor_admitted=False))
    check(len(slot_rows) == 3904, 'Every original stock slot retained')
    by_symbol = []
    for symbol in stocks:
        rows = [r for r in slot_rows if r['symbol'] == symbol]
        by_symbol.append(dict(symbol=symbol, query_slots=len(rows), catalog_events=len(groups[symbol]),
            slots_with_observed_contiguous_quarter_labels=sum(r['observed_quarter_labels_contiguous'] for r in rows),
            blocker_counts=dict(Counter(k for r in rows for k in r['blockers'])), certified_complete_slots=0))
    counts = dict(source_events=len(event_rows), source_symbols=len(groups), query_months=122, query_stocks=len(stocks),
        stock_slots=len(slot_rows), etf_slots=122*49-len(slot_rows),
        evidence_references=len(refs), unique_evidence_files=len({r['path'] for r in refs if r['path']}),
        evidence_reference_status=dict(Counter(r['status'] for r in refs)),
        event_clock_metadata_consistent=sum(r['clock_metadata_consistent'] for r in event_rows),
        event_precision=dict(Counter(r['precision'] for r in event_rows)),
        event_source_kind=dict(Counter(r['source_kind'] for r in event_rows)),
        slots_with_observed_contiguous_quarter_labels=sum(r['observed_quarter_labels_contiguous'] for r in slot_rows),
        slot_blockers=dict(Counter(k for r in slot_rows for k in r['blockers'])),
        complete_event_calendars_verified=0, factor_admitted_slots=0)
    census = dict(schema_version=1, stage='METADATA_ONLY_CENSUS_COMPLETE', generated_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=expected, counts=counts, by_symbol=by_symbol, source_hashes=spec['source_hashes'], source_code_hashes=spec['source_code_hashes'],
        inherited_readiness=dict(scope=readiness['scope'], scoped_post_known_drift_data_ready=readiness.get('scoped_post_known_drift_data_ready'),
            full_global_earnings_archive_complete=readiness.get('full_global_earnings_archive_complete'), limitations=readiness['limitations']),
        finding='Observed sequence continuity and matching source hashes are necessary diagnostics, not a proof that all regular releases are present.',
        next_feasible_action='Freeze an issuer/window evidence review of the listed quarter gaps, boundary cases and original archive/index completeness. Use existing local archives first; retain every unknown window as unavailable. No price-based selection and no strategy registration until that separate review.',
        limitations=['Source URLs and publication dates are inherited metadata, not newly authenticated original first-public clocks.',
            'Date-only next-session mapping is checked; exact intraday two-event timing inherits the prior audit.',
            '120-day gaps and unmatched vendor hints are leads, not proven missing events; nonquarterly issuers require source-specific cadence evidence.',
            'A row can be quarter-contiguous yet incomplete for preliminary releases, corrections, duplicates or boundary announcements.',
            'All per-slot observed sequences stop at formation end. Later announcements are not used to fill an earlier window.'], **ZERO)
    products = {'EVENT_AUDIT.json':event_rows, 'EVIDENCE_AUDIT.json':refs, 'SLOTS.json':slot_rows, 'CENSUS.json':census}
    for name, value in products.items(): write(OUT / name, value)
    write(OUT / 'MANIFEST.json', dict(spec_sha256=expected, output_hashes={name:sha(OUT/name) for name in products}, **ZERO))
    print(json.dumps(dict(counts=counts, manifest_sha256=sha(OUT/'MANIFEST.json')), sort_keys=True))


def verify(expected):
    validate_spec(expected)
    m = load(OUT/'MANIFEST.json'); check(m['spec_sha256'] == expected, 'Manifest SPEC')
    check(set(m['output_hashes']) == {'EVENT_AUDIT.json','EVIDENCE_AUDIT.json','SLOTS.json','CENSUS.json'}, 'Manifest outputs')
    for name, digest in m['output_hashes'].items(): check(sha(OUT/name) == digest, 'Output changed: '+name)
    c = load(OUT/'CENSUS.json')
    check(c['counts']['stock_slots'] == 3904 and c['counts']['source_events'] == 3771 and all(c[k] == 0 for k in ZERO), 'Census scope/counters')
    check(all(not r['complete_event_calendar_verified'] and not r['factor_admitted'] for r in load(OUT/'SLOTS.json')), 'Unjustified admission')
    print(json.dumps(dict(verified=True, census_sha256=sha(OUT/'CENSUS.json'), manifest_sha256=sha(OUT/'MANIFEST.json'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('self-test','freeze','run','verify')); p.add_argument('--spec-sha')
    args = p.parse_args()
    if args.phase == 'self-test': print(json.dumps(dict(synthetic_checks=self_test(), **ZERO)))
    elif args.phase == 'freeze': freeze()
    else:
        check(args.spec_sha is not None, '--spec-sha required')
        {'run':run, 'verify':verify}[args.phase](args.spec_sha)
