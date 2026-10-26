"""Independent, offline metadata reconciliation of the frozen event overlays.

No producer imports, financial extraction, market prices, factors or strategies.
This records local single-event repairs; quarterly continuity is not completeness.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'work/after156_event_scope_local_review.json'
OLD = 'outputs/monthly_event_overlay_scope156/'
SCOPE = 'outputs/monthly_event_scope156/'
META = 'outputs/monthly_local_linear70/META.json'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
LATEST = 'outputs/monthly_event_patch156_batch3/'
PATCHES = ('outputs/monthly_event_patch156/',
           'outputs/monthly_event_patch156_batch2/', LATEST)
PINS = {
    OLD+'MANIFEST.json': '926f869fb85cc423c60b9995a7fd7a60c07b7662409603eb04563c27006d67e6',
    LATEST+'CATALOG_MANIFEST.json': '6de0693859cf2cbba84ee477ca08695fbaea428f79776876c207be1f5e80558d',
    META: '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    CAL: '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
}
EXPECTED = ['CAT_2025-04-30', 'CAT_2025-08-05', 'CRM_2019-12-03',
            'HSBC_2016-08-03', 'HSBC_2017-02-21', 'SAP_2020-01-28']


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Sources:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else self.root / p
        require('..' not in p.parts and 'CloudStorage' not in p.parts and
                p.is_relative_to(self.root) and p.resolve() == p, 'Nonlocal source')
        return p

    def read_bytes(self, value, expected=None, size=None):
        p = self.path(value)
        before = p.stat()
        require(not getattr(before, 'st_flags', 0) & 0x40000000, 'Dataless source')
        blob = p.read_bytes()
        after = p.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), 'Source mutated')
        h = hashlib.sha256(blob).hexdigest()
        key = str(p.relative_to(self.root))
        require(expected is None or h == expected, 'Source SHA: '+key)
        require(size is None or len(blob) == size, 'Source size: '+key)
        require(key not in self.hashes or self.hashes[key] == h, 'Source changed: '+key)
        self.hashes[key] = h
        return blob

    def read(self, value, expected=None):
        return json.loads(self.read_bytes(value, expected))

    def refs(self, obj):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                self.read_bytes(obj['path'], obj['sha256'], obj.get('bytes'))
            for value in obj.values():
                self.refs(value)
        elif isinstance(obj, list):
            for value in obj:
                self.refs(value)

    def recheck(self):
        for p, h in list(self.hashes.items()):
            self.read_bytes(p, h)


def fiscal(record):
    year, quarter = record.get('fiscal_year'), record.get('fiscal_quarter')
    if type(year) is int and type(quarter) is int and 1 <= quarter <= 4:
        return 4*year+quarter-1
    return None


def sequence(records, sessions, lo, hi):
    """Independent date-sorted interval scan; predecessor is context only."""
    dates = set(sessions)
    require(lo in dates and hi in dates and lo <= hi, 'Bad interval')
    available = sorted((e for e in records if e.get('available_session') in dates),
                       key=lambda e: (e['available_session'], e['event_id']))
    inside = [e for e in available if lo <= e['available_session'] <= hi]
    previous = [e for e in available if e['available_session'] < lo][-1:]
    flags = []
    ordered = previous+inside
    for a, b in zip(ordered, ordered[1:]):
        qa, qb = fiscal(a), fiscal(b)
        delta = None if qa is None or qb is None else qb-qa
        if delta == 1:
            continue
        kind = ('unknown_quarter_label' if delta is None else 'duplicate_quarter'
                if delta == 0 else 'nonmonotone_quarter' if delta < 0 else
                'unobserved_quarter_labels')
        flags.append(dict(symbol=b['symbol'], before=a['event_id'], after=b['event_id'],
            before_publication=a['publication_date'], after_publication=b['publication_date'],
            before_anchor=a['available_session'], after_anchor=b['available_session'],
            kind=kind, quarter_step=delta,
            scope='left_predecessor_crossing_context' if a['available_session'] < lo
                  else 'inside_consumed_anchor_window',
            intermediate_fiscal_labels=['%04dQ%d' % (n//4, n%4+1)
                for n in range(qa+1, qb)] if delta is not None and delta > 1 else [],
            exact_missing_publication_dates_known=False, requires_source_review=True))
    unknown = [e['event_id'] for e in records if e.get('available_session') not in dates
               and (not isinstance(e.get('publication_date'), str) or e['publication_date'] <= hi)]
    return dict(event_ids=[e['event_id'] for e in inside], quarter_items=flags,
                unknown_anchor_ids=unknown,
                optional_predecessor=previous[0]['event_id'] if previous else None)


def summarize(rows):
    groups = {}
    for row in rows:
        for q in row['after']['quarter_items']:
            key = (q['symbol'], q['before'], q['after'], q['scope'])
            item = groups.setdefault(key, dict(q, slot_ids=[], signal_dates=[], interval_ids=[]))
            item['slot_ids'].append(row['slot_id'])
            item['signal_dates'].append(row['signal_date'])
            if row['interval_id'] not in item['interval_ids']:
                item['interval_ids'].append(row['interval_id'])
    return list(groups.values())


def counts(rows):
    flags = Counter(q['scope'] for r in rows for q in r['after']['quarter_items'])
    items = summarize(rows)
    return dict(slots=len(rows), slots_with_any_quarter_flag=sum(bool(r['after']['quarter_items']) for r in rows),
        slot_quarter_flags=dict(sorted(flags.items())),
        unique_inside_pairs=sum(q['scope']=='inside_consumed_anchor_window' for q in items),
        unique_left_pairs=sum(q['scope']=='left_predecessor_crossing_context' for q in items),
        consumed_unique=len({e for r in rows for e in r['after']['event_ids']}),
        unknown_anchor_slots=sum(bool(r['after']['unknown_anchor_ids']) for r in rows))


def source_closure(src):
    for p, h in PINS.items():
        src.read_bytes(p, h)
    for prefix in (OLD, SCOPE):
        manifest = src.read(prefix+'MANIFEST.json')
        spec = src.read(prefix+'SPEC.json', manifest['spec_sha256'])
        for section in ('input_hashes', 'source_hashes', 'source_code_hashes'):
            for p, h in spec.get(section, {}).items():
                src.read_bytes(p, h)
        for name, h in manifest['output_hashes'].items():
            src.read_bytes(prefix+name, h)
    manifests = []
    for prefix in PATCHES:
        m = src.read(prefix+'CATALOG_MANIFEST.json')
        spec = src.read(prefix+'PATCH_SPEC.json', m['spec']['sha256'])
        src.refs(m)
        src.refs(spec)
        src.refs(src.read(prefix+'INTEGRATION.json'))
        manifests.append(m)
    components = manifests[-1]['components']
    require([c['count'] for c in components] == [3771,4,4,2], 'Component scope')
    require(manifests[-1]['combined_event_count'] == 3781, 'Catalog count')
    for index, prior in enumerate(manifests):
        require([c['sha256'] for c in prior['components']] ==
                [c['sha256'] for c in components[:index+2]], 'Prior component replaced')
    return [src.read(c['path'], c['sha256']) for c in components]


def review_added(src, events, sessions):
    require(sorted(e['event_id'] for e in events) == EXPECTED, 'Only six reviewed additions')
    reports = []
    clock_keys = ('publication_date','publication_timezone','official_actual_period_end',
                  'available_session','known_by_at','precision')
    for e in events:
        require(e['single_event_source_admitted'] is True, 'Single event admission')
        require(all(e[k] is False for k in ('whole_window_archive_complete','factor_admitted','strategy_admitted')), 'Overclaim')
        src.read_bytes(e['evidence_path'], e['evidence_sha256'])
        src.refs(e['source_receipt'])
        receipt = src.read(e['source_receipt']['path'])
        require(receipt.get('http_status', receipt.get('status')) == 200, 'Source response')
        ref = e['independent_source_review']
        report = src.read(ref['path'], ref['sha256'])
        if e['symbol'] in ('CRM','SAP'):
            require(report['passed'] is True, 'CRM/SAP review')
            r = next(r for r in report['event_reviews'] if r.get('event_id')==e['event_id'])
            require(r['single_event_source_admitted'] is True, 'Event review')
            reviewed = r['verified_candidate']
            blob = src.read_bytes(e['evidence_path'])
            for quote in r['raw_evidence']:
                require(blob[quote['raw_byte_start']:quote['raw_byte_end']].decode('utf-8') == quote['raw_excerpt'], 'Raw locator')
        elif e['symbol'] == 'HSBC':
            r = next(r for r in report['checked_events'] if r['event_id']==e['event_id'])
            require(r['single_event_source_admitted_for_new_derived_catalog'] is True, 'HSBC review')
            require(e['quarter_is_period_endpoint_ordinal'] is True and
                    e['standalone_quarterly_release_verified'] is False and
                    e['reporting_period_type'] == r['period_type'], 'HSBC period semantics')
            reviewed = r['availability']
        else:
            require(report['admission_decision']['these_two_events_may_enter_new_derived_catalog'] is True, 'CAT review')
            reviewed = next(r for r in report['events'] if r['event_id']==e['event_id'])
            require(reviewed['single_event_source_admissible'] is True, 'CAT event')
        require(all(e[k] == reviewed[k] for k in clock_keys), 'Independent reviewed clock match')
        index = bisect_right(sessions, e['publication_date'])
        require(index < len(sessions) and sessions[index] == e['available_session'], 'Next-session clock')
        if e['symbol'] == 'SAP':
            require(e['release_kind']=='actual_preliminary_quarterly_and_annual_results', 'SAP preliminary semantics')
        reports.append(dict(event_id=e['event_id'], source_path=e['evidence_path'],
            source_sha256=e['evidence_sha256'], source_review=ref,
            publication_date=e['publication_date'], available_session=e['available_session'],
            reporting_period_type=e['reporting_period_type'], release_kind=e['release_kind'],
            source_and_receipt_bytes_verified=True, independent_review_identity_clock_verified=True,
            next_session_independently_verified=True,
            raw_document_reauthenticated_here=e['symbol'] in ('CRM','SAP'),
            limit='Inherited pinned independent source approval; no new original-vintage or archive-completeness claim.'))
    return reports


def tsla_hold(src):
    """Retain the separately inspected failed-source boundary without date inference."""
    base = 'outputs/monthly_earnings_repair/batch_technology_industrial/TSLA/'
    leaf = base+'raw/de691e5b9b29307fe7de'
    refs = {
        leaf+'.bin':'0efb0996be5525c1a4e116270edd2a25da37d672834b7b028be07234c733a810',
        leaf+'.meta.json':'48f379e0fe5e9df07b118029fe3373bc1f74306aae11d564c8e880927c0705e5',
        leaf+'.txt':'b1886de1e18aa2ec42c376d0eb39d484e0ea8605ffb55ada804de54e13cab4f4',
        leaf+'.text.meta.json':'d4ba42a78fe80f1f81db3439f92acd7d985068b84d472c31d0df77341869e533',
        base+'public_index_observations/search_release_2019.json':'f63678303cb9f8c9cb70303d3953f0830e8e5e2bd02277beff42d964029b56ea',
        'outputs/monthly_event_gap156_crm_sap_tsla/tsla.receipt.json':'ba486637f1a47346f7096a6c6f9a53b41cdf869695be613c6525f3eba8bfd3a6',
    }
    for p,h in refs.items():
        src.read_bytes(p,h)
    require(src.read_bytes(leaf+'.bin').startswith(b'%PDF-'), 'TSLA original PDF')
    receipt = src.read('outputs/monthly_event_gap156_crm_sap_tsla/tsla.receipt.json')
    require(receipt['http_status']==403, 'TSLA failed original release receipt')
    absent = 'outputs/monthly_event_gap156_crm_sap_tsla/tsla.html'
    require(not src.path(absent).exists(), 'TSLA source state changed: review new original')
    review = src.read('work/after155_crm_sap_independent_review.json')
    entry = next(e for e in review['event_reviews'] if e['symbol']=='TSLA')
    require(entry['single_event_source_admitted'] is False and entry['publication_date_verified'] is False, 'TSLA source hold')
    return dict(status='hold_no_verified_original_release_dateline',symbol='TSLA',fiscal_year=2019,fiscal_quarter=3,
        period_end_supported='2019-09-30',publication_date_admitted=None,
        original_pdf=leaf+'.bin',original_release_html_absent=absent,
        evidence_hashes=refs,existing_independent_review='work/after155_crm_sap_independent_review.json',
        additional_review='Separate read-only teammate inspection reconfirmed the existing source distinction; no new source admitted.',
        why='The cached Q3 update supports quarter/period. Its October23 mention schedules a webcast, which cannot establish the actual release date. Search-rendered actual-release text is not original release bytes. The attempted original release page returned403.',
        required_evidence='An admissible dated original actual-results release or equivalent direct dissemination evidence; do not fill a date from the webcast or fiscal-label gap.',
        bounded_search='Exact TSLA cache and prior source review; teammate searched TSLA plus earnings_repair text/JSON/HTML. No network or full archive completeness claim.')


def audit(src):
    components = source_closure(src)
    old_events = components[0]+components[1]
    new_events = old_events+components[2]+components[3]
    require(len({e['event_id'] for e in new_events})==len(new_events)==3781, 'Event identity collision')
    require(len({(e['symbol'],e.get('publication_date')) for e in new_events})==len(new_events), 'Issuer/date collision')
    sessions = src.read(CAL)['sessions']
    require(sessions == sorted(set(sessions)) and all(date.fromisoformat(d).isoformat()==d for d in sessions), 'Calendar axis')
    additions = review_added(src, components[2]+components[3], sessions)
    groups = [defaultdict(list), defaultdict(list)]
    for index, collection in enumerate((old_events,new_events)):
        for e in collection:
            groups[index][e['symbol']].append(e)
    slots = src.read(SCOPE+'SLOTS.json')
    intervals = src.read(SCOPE+'INTERVALS.json')
    old_rows = src.read(OLD+'SLOT_AUDIT.json')
    require(len(slots)==len(old_rows)==3904, 'Original stock slots')
    require(len(intervals)==99 and len({s['symbol'] for s in slots})==73, 'Original interval/issuer scope')
    require({s['interval_id'] for s in slots}=={r['interval_id'] for r in intervals}, 'Interval identities')
    for interval in intervals:
        own = [s for s in slots if s['interval_id']==interval['interval_id']]
        require([s['slot_id'] for s in own]==interval['source_slot_ids'], 'Interval slot ownership')
        require([min(s['required_anchor_indices'][0] for s in own),
                 max(s['required_anchor_indices'][1] for s in own)]==interval['required_anchor_indices'], 'Interval union endpoints')
    meta = src.read(META)
    expected_axis = {(i,j) for i,row in enumerate(meta['symbols']) for j,s in enumerate(row) if s not in meta['etfs']}
    require({(s['signal_index'],s['candidate_index']) for s in slots}==expected_axis, 'Original complete axis')
    require(len(meta['dates'])==122 and all(len(r)==49 for r in meta['symbols']), '122 by 49 grid')
    after_rows, changes = [], []
    for slot, old in zip(slots, old_rows):
        i,j = slot['signal_index'],slot['candidate_index']
        require(slot['slot_id']==old['slot_id'] and slot['symbol']==meta['symbols'][i][j]
                and slot['signal_date']==meta['dates'][i], 'Original identity')
        t = sessions.index(slot['signal_date'])
        lo,hi = sessions[t-271],sessions[t-44]
        require(slot['required_anchor_window']==[lo,hi] and slot['required_anchor_indices']==[t-271,t-44], 'Frozen interval')
        before = sequence(groups[0][slot['symbol']],sessions,lo,hi)
        require(before==old['after'], 'Independent old 51-slot reconciliation')
        after = sequence(groups[1][slot['symbol']],sessions,lo,hi)
        row = {k:old[k] for k in ('slot_id','symbol','signal_date','interval_id','anchor_window')}
        row['after'] = after
        after_rows.append(row)
        if before != after:
            changes.append(dict(slot_id=slot['slot_id'],symbol=slot['symbol'],signal_index=i,
                signal_date=slot['signal_date'],interval_id=slot['interval_id'],
                old_quarter_items=before['quarter_items'],new_quarter_items=after['quarter_items'],
                inserted_event_ids=sorted(set(after['event_ids'])-set(before['event_ids'])),
                predecessor_before=before['optional_predecessor'],predecessor_after=after['optional_predecessor']))
    before_count,after_count = counts(old_rows),counts(after_rows)
    require(before_count['slots_with_any_quarter_flag']==51 and before_count['unique_inside_pairs']==6, 'Requested baseline')
    old_items,new_items = summarize(old_rows),summarize(after_rows)
    def key(q):return(q['symbol'],q['before'],q['after'],q['scope'])
    remaining_keys = {key(q) for q in new_items}
    resolved = [q for q in old_items if key(q) not in remaining_keys]
    old_flagged = {r['slot_id'] for r in old_rows if r['after']['quarter_items']}
    new_flagged = {r['slot_id'] for r in after_rows if r['after']['quarter_items']}
    original_remaining = src.read(OLD+'REMAINING_ITEMS.json')
    old_lookup = {e['event_id']:e for e in old_events}
    internal_resolutions = []
    for q in old_items:
        if q['scope'] != 'inside_consumed_anchor_window':
            continue
        qa,qb = fiscal(old_lookup[q['before']]),fiscal(old_lookup[q['after']])
        matches = [e for e in components[2]+components[3] if e['symbol']==q['symbol'] and
                   qa < fiscal(e) < qb and q['before_anchor'] < e['available_session'] < q['after_anchor']]
        covered = {fiscal(e) for e in matches}
        fixed = covered == set(range(qa+1,qb))
        internal_resolutions.append(dict(symbol=q['symbol'],before=q['before'],after=q['after'],
            old_missing_labels=q['intermediate_fiscal_labels'],old_affected_slots=q['slot_ids'],
            source_event_ids=[e['event_id'] for e in matches],
            status='ordinal_gap_repaired_by_reviewed_events' if fixed else 'unresolved_no_admissible_dated_original',
            full_event_sequence_complete=False))
    require(sum(r['status']=='ordinal_gap_repaired_by_reviewed_events' for r in internal_resolutions)==5,
            'Five exact internal gap repairs')
    return dict(schema_version=1,stage='INDEPENDENT_LOCAL_EVENT_SCOPE_RECONCILIATION_AFTER156',
        passed=True,decision='Six already-admitted local additions repair five of six old internal quarter-label pairs; no full interval admission.',
        baseline=before_count,reconciled=after_count,reviewed_additions=additions,
        original_grid=dict(months=122,width=49,stock_slots=3904,issuers=73,intervals=99,preserved=True),
        changed_slots=len(changes),changed_slot_records=changes,
        changed_slots_by_period=dict(early53=sum(r['signal_index']<53 for r in changes),
            remaining_first88=sum(53<=r['signal_index']<88 for r in changes),later34=sum(r['signal_index']>=88 for r in changes)),
        slots_cleared_of_old_quarter_flags=sorted(old_flagged-new_flagged),
        internal_pair_resolutions=internal_resolutions,tsla_remaining_source_hold=tsla_hold(src),
        resolved_old_pair_records=resolved,remaining_quarter_items=new_items,
        unresolved_empty_intervals=original_remaining['empty_intervals'],
        unresolved_unknown_anchor_intervals=original_remaining['unknown_anchor_intervals'],
        boundary_coverage_pending=original_remaining['boundary_coverage_pending'],
        admission=dict(complete_intervals_certified=0,full_event_calendar_complete=False,
            historical_first_public_bytes_verified=False,quarter_continuity_proves_completeness=False,
            event_catalog_rewritten=False,new_events_admitted_here=0,factor_admitted=False,strategy_admitted=False),
        limits=['A repaired quarter ordinal is not an independently complete sequence of preliminary, final and corrective announcements.',
            'Left-predecessor flags are contextual; their missing fiscal labels are not automatically missing releases inside the consumed window.',
            'HSBC Q2/Q4 are half-year/annual endpoints; SAP is preliminary. Their original reporting classifications remain explicit.',
            'Six additions were already independently admitted in immutable batch2/batch3; this review newly reconciles their impact with the stale baseline.',
            'All99 original intervals still require bounded source enumeration and boundary evidence.'],
        activity_counts=dict(price_values_read=0,financial_values_extracted=0,factor_calculations=0,model_fits=0,
            ranks=0,orders=0,returns=0,new_strategy_trials=0,network_requests=0,frozen_files_modified=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', help='Publish the new review once, after all checks')
    args = p.parse_args()
    import test_after156_event_scope_local_review as synthetic
    suite = unittest.defaultTestLoader.loadTestsFromModule(synthetic)
    log = io.StringIO()
    tested = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(tested.wasSuccessful() and tested.testsRun==15, 'Synthetic tests failed: '+log.getvalue())
    src = Sources()
    result = audit(src)
    for name in ('work/after156_event_scope_local_review.py','work/test_after156_event_scope_local_review.py',
                 'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):
        src.read_bytes(name)
    src.recheck()
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
                  input_hashes=dict(sorted(src.hashes.items())),
                  synthetic_tests=dict(passed=True,tests_run=tested.testsRun,failures=len(tested.failures),
                                       errors=len(tested.errors),output=log.getvalue()))
    if args.write:
        with src.path(OUTPUT).open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','baseline','reconciled','changed_slots','changed_slots_by_period','admission')},sort_keys=True))


if __name__=='__main__':
    main()
