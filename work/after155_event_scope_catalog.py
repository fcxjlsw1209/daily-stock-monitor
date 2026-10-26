"""Write-once source-only catalog of consumed event-calendar intervals.

Uses only frozen calendar/candidate/event metadata. It does not load price data,
financial amounts, factors, model inputs or strategy outcomes. No source status
is promoted; new event supplements require a separate derived source binding.
"""
import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_event_scope156'
PINS = {
    'outputs/monthly_local_linear70/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'outputs/monthly_paired_target143/CALENDAR.json': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
    'outputs/monthly_earnings_repair/EVENTS.json': '1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103',
    'outputs/monthly_event_clock_census155/SPEC.json': '3a300b926650fff5f036a95be4f96d2542d38cc7c7cd56d9f9f91a8ce065991b',
    'outputs/monthly_event_clock_census155/CENSUS.json': '9248eea8dc84a92339db8e4e5c63c7e9024f0d8e13222c2a4ed03888edc116b5',
    'outputs/monthly_event_clock_census155/SLOTS.json': 'e190adc576ed22f36fd9dd2a169f0574b88bc9125994c020df87bc9013fbdf4e',
    'outputs/monthly_event_clock_census155/EVENT_AUDIT.json': '95df662030e0c7c241c9ee31cf1a66520ba377958dddc27c95dd4893ae0cce55',
    'outputs/monthly_event_clock_census155/EVIDENCE_AUDIT.json': '84a0292c18aa4edd6b3b79f0fe7f3b54e7d013f5e450a3bf8efd4226761a248c',
    'outputs/monthly_event_clock_census155/MANIFEST.json': '780a0409db9e849ea4768d4815f7a17cc1709439bb1f88aa0bb1a98838719e81',
    'work/after155_event_clock_method_review.json': '1cf1f9b2fa497d319dd54870c0d349b436bff9ab0415125019f061dc22a8458c',
}
META, CALENDAR, RAW = list(PINS)[:3]
OLD = 'outputs/monthly_event_clock_census155/'
FILES = ('INTERVALS.json', 'SLOTS.json', 'EVENTS.json', 'EVIDENCE_TASKS.json', 'SUMMARY.json')
ZERO = dict(price_values_read=0, financial_values_extracted=0, factor_calculations=0,
            model_fits=0, ranks=0, orders=0, returns=0, new_strategy_trials=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(name):
    p = Path(name)
    if not p.is_absolute(): p = ROOT / p
    require('..' not in p.parts and 'CloudStorage' not in p.parts and p.is_relative_to(ROOT), 'Nonlocal source')
    require(not p.is_symlink() and p.resolve().is_relative_to(ROOT), 'Source symlink/escape')
    return p


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(local(path).read_text())


def publish(path, value):
    path = local(path); path.parent.mkdir(exist_ok=True, parents=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.scope-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(name)


def canonical_day(value):
    if not isinstance(value, str): return False
    try: return date.fromisoformat(value).isoformat() == value
    except ValueError: return False


def union_ranges(ranges):
    result = []
    for lo, hi in sorted(ranges):
        require(type(lo) is int and type(hi) is int and lo <= hi, 'Invalid session interval')
        if result and lo <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], hi)
        else: result.append([lo, hi])
    return result


def intersects(anchor, begin, end):
    return anchor + 2 <= end and anchor + 20 >= begin


def potentially_unknown(event, end, positions):
    """Unknown or invalid anchor is relevant unless a publication lower bound excludes it."""
    if event.get('available_session') in positions: return False
    pub = event.get('publication_date')
    return not canonical_day(pub) or pub <= end


def classify_transition(item, positions, lo, hi, event_map):
    a = positions.get(event_map[item['before']].get('available_session'))
    b = positions.get(event_map[item['after']].get('available_session'))
    if a is None or b is None: kind = 'unresolved_endpoint_anchor'
    elif a < lo and lo <= b <= hi: kind = 'crosses_left_boundary_not_all_missing_quarters_are_in_scope'
    elif lo <= a <= b <= hi: kind = 'internal_observed_sequence_requires_source_review'
    else: kind = 'outside_or_crosses_right_boundary_context_only'
    return dict(item, scope_classification=kind, inferred_missing_in_window_count=None)


def self_test():
    checks = 0
    def ok(value):
        nonlocal checks
        require(value, 'Synthetic assertion ' + str(checks + 1)); checks += 1
    ok(union_ranges([]) == [])
    ok(union_ranges([(3, 5), (1, 2)]) == [[1, 5]])
    ok(union_ranges([(1, 8), (2, 3)]) == [[1, 8]])
    ok(union_ranges([(1, 2), (4, 5)]) == [[1, 2], [4, 5]])
    ok(intersects(80, 100, 200))
    ok(not intersects(79, 100, 200))
    ok(intersects(198, 100, 200))
    ok(not intersects(199, 100, 200))
    ok(potentially_unknown({}, '2020-01-01', {}))
    ok(potentially_unknown({'publication_date':'2019-01-01'}, '2020-01-01', {}))
    ok(not potentially_unknown({'publication_date':'2021-01-01'}, '2020-01-01', {}))
    ok(not potentially_unknown({'available_session':'2020-01-01'}, '2020-01-01', {'2020-01-01':0}))
    ok(canonical_day('2020-02-29') and not canonical_day('2020-02-30'))
    event = {'a':{'available_session':'a'}, 'b':{'available_session':'b'}}
    item = {'before':'a','after':'b','quarter_step':8}
    ok(classify_transition(item, {'a':1,'b':12}, 10, 20, event)['scope_classification'].startswith('crosses_left'))
    ok(classify_transition(item, {'a':11,'b':12}, 10, 20, event)['scope_classification'].startswith('internal'))
    ok(classify_transition(item, {'a':1,'b':2}, 10, 20, event)['scope_classification'].startswith('outside'))
    ok(classify_transition(item, {}, 10, 20, event)['inferred_missing_in_window_count'] is None)
    # Test index arithmetic, not calendar-day subtraction: a skipped holiday stays skipped.
    sessions = ['2020-11-23','2020-11-24','2020-11-25','2020-11-27']
    ok(sessions[3-2] == '2020-11-24')
    ok(all(intersects(a, 100, 200) == (80 <= a <= 198) for a in range(50, 230)))
    ok(union_ranges([(1, 2), (1, 2)]) == [[1, 2]])
    return checks


def validate_inputs():
    for path, digest in PINS.items(): require(sha(path) == digest, 'Frozen input changed: ' + path)
    old_manifest = read(OLD + 'MANIFEST.json')
    require(old_manifest['spec_sha256'] == PINS[OLD + 'SPEC.json'], 'Old SPEC binding')
    require(all(PINS[OLD + name] == value for name, value in old_manifest['output_hashes'].items()), 'Old output bindings')


def freeze():
    require(not (OUT / 'SPEC.json').exists(), 'Scope SPEC already frozen')
    checks = self_test(); validate_inputs()
    spec = dict(schema_version=1, stage='FROZEN_SOURCE_ONLY_CONSUMED_EVENT_SCOPE', created_utc=datetime.now(timezone.utc).isoformat(),
        runtime_python=sys.version, source_hashes=PINS, source_code_hashes={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        original_event_catalog_count=3771, source_catalog_supplements_included=False,
        scope=dict(query_months=122, width=49, stocks=73, stock_slots=3904, intervals=99, intersecting_existing_events=1677),
        offsets=dict(formation_start=-251, formation_end=-42, post_known_start=2, post_known_end=20, required_anchor_start=-271, required_anchor_end=-44),
        semantics=['Session-indexed inclusive return endpoint windows; each slot requires only intersecting event anchors.',
          'Merge overlapping or session-adjacent anchor intervals within each issuer; retain every original stock slot.',
          'Catalog only old3771 metadata; no supplemental event implicitly incorporated.',
          'Outside predecessor is optional boundary context, not a factor requirement or proof of missing in-window releases.',
          'Unresolved anchors are retained where source bounds cannot exclude the interval; no guessed dates.',
          'Every segment still requires source enumeration/clock/duplicate review; no calendar or strategy admission.'],
        synthetic_checks=checks, **ZERO)
    publish(OUT/'SPEC.json', spec)
    return dict(stage=spec['stage'], spec_sha256=sha(OUT/'SPEC.json'), synthetic_checks=checks)


def validate_spec(expected):
    require(sha(OUT/'SPEC.json') == expected, 'Exact scope SPEC required')
    spec = read(OUT/'SPEC.json')
    require(spec['stage']=='FROZEN_SOURCE_ONLY_CONSUMED_EVENT_SCOPE' and spec['runtime_python']==sys.version, 'Scope/runtime')
    require(spec['source_hashes']==PINS and spec['source_code_hashes']=={str(Path(__file__).relative_to(ROOT)):sha(__file__)}, 'Frozen scope bindings')
    require(all(type(spec[k]) is int and spec[k]==0 for k in ZERO), 'Source-only scope')
    validate_inputs()
    return spec


def products(spec_sha):
    meta, calendar = read(META), read(CALENDAR)
    sessions = calendar['sessions']; pos = {value:i for i,value in enumerate(sessions)}
    require(sessions == sorted(set(sessions)) and all(canonical_day(d) for d in sessions), 'Calendar axis')
    etfs = set(meta['etfs']); require(len(etfs)==17 and len(meta['dates'])==len(meta['symbols'])==122, 'Original dimensions')
    old_slots = read(OLD+'SLOTS.json'); old_map = {(r['signal_index'],r['candidate_index']):r for r in old_slots}
    events = read(OLD+'EVENT_AUDIT.json'); raw = read(RAW)
    event_map = {r['event_id']:r for r in events}; raw_map = {r['event_id']:r for r in raw}
    require(len(events)==len(event_map)==len(raw)==len(raw_map)==3771 and set(event_map)==set(raw_map), 'Old catalog identities')
    groups = defaultdict(list)
    for e in events:
        for key in ('symbol','publication_date','available_session','known_by_at','fiscal_year','fiscal_quarter'):
            require(e[key]==raw_map[e['event_id']].get(key), 'Old catalog metadata mismatch')
        groups[e['symbol']].append(e)
    for group in groups.values(): group.sort(key=lambda e:(e.get('available_session') or '',e['event_id']))
    slots, issuer_ranges = [], defaultdict(list)
    for i,(signal,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        require(len(names)==len(set(names))==49 and len(set(names)&etfs)==17 and signal in pos, 'Original pool')
        t=pos[signal]; require(t>=271, 'Calendar warmup')
        for j,symbol in enumerate(names):
            if symbol in etfs: continue
            lo,hi,begin,end=t-271,t-44,t-251,t-42
            old=old_map[(i,j)]
            require(old['symbol']==symbol and old['signal_date']==signal and old['formation_window']==[sessions[begin],sessions[end]], 'Original slot mapping')
            ids=[e['event_id'] for e in groups[symbol] if e.get('available_session') in pos and lo<=pos[e['available_session']]<=hi]
            require(ids==old['intersecting_post_known_window_event_ids'], 'Exact consumed event intersection')
            require(all(intersects(pos[event_map[e]['available_session']],begin,end) for e in ids), 'Intersection arithmetic')
            row=dict(slot_id='%03d:%02d'%(i,j),signal_index=i,candidate_index=j,symbol=symbol,signal_date=signal,
                formation_window=[sessions[begin],sessions[end]],required_anchor_window=[sessions[lo],sessions[hi]],
                required_anchor_indices=[lo,hi],existing_event_ids=ids,
                unresolved_anchor_event_ids=[e['event_id'] for e in groups[symbol] if potentially_unknown(e,sessions[hi],pos)],
                inherited_census_predecessor_event_id=old['predecessor_event_id'],predecessor_is_consumption_requirement=False,
                inherited_quarter_review_items=[classify_transition(x,pos,lo,hi,event_map) for x in old['quarter_sequence_issues']],
                inherited_long_interval_review_items=[classify_transition(x,pos,lo,hi,event_map) for x in old['long_reporting_intervals']],
                inherited_vendor_date_leads=old['unmatched_vendor_hint_dates'],vendor_dates_are_not_event_anchors=True,
                inherited_candidate_scope_resolution_ids=old['existing_candidate_scope_resolution_ids'],
                calendar_completeness_verified=False,factor_admitted=False)
            slots.append(row); issuer_ranges[symbol].append((lo,hi))
    require(len(slots)==len(old_slots)==len(old_map)==3904 and len(issuer_ranges)==73, 'Full stock axis')
    intervals=[]; by_symbol=defaultdict(list)
    for symbol in sorted(issuer_ranges):
        for lo,hi in union_ranges(issuer_ranges[symbol]):
            identifier=symbol+'__'+sessions[lo]+'__'+sessions[hi]
            row=dict(interval_id=identifier,symbol=symbol,required_anchor_window=[sessions[lo],sessions[hi]],
                required_anchor_indices=[lo,hi],source_slot_ids=[],existing_event_ids=[],unresolved_anchor_event_ids=[],
                source_enumeration_status='pending_bounded_issuer_archive_review',calendar_completeness_verified=False,factor_admitted=False)
            intervals.append(row); by_symbol[symbol].append(row)
    for slot in slots:
        lo,hi=slot['required_anchor_indices']
        owners=[r for r in by_symbol[slot['symbol']] if r['required_anchor_indices'][0]<=lo<=hi<=r['required_anchor_indices'][1]]
        require(len(owners)==1,'Unique interval ownership'); row=owners[0]
        slot['interval_id']=row['interval_id']; row['source_slot_ids'].append(slot['slot_id'])
    consumed=set(); tasks=[]; memberships=defaultdict(list)
    for row in intervals:
        symbol=row['symbol']; lo,hi=row['required_anchor_indices']
        selected=[e for e in groups[symbol] if e.get('available_session') in pos and lo<=pos[e['available_session']]<=hi]
        row['existing_event_ids']=[e['event_id'] for e in selected]
        row['unresolved_anchor_event_ids']=[e['event_id'] for e in groups[symbol] if potentially_unknown(e,sessions[hi],pos)]
        before=[e for e in groups[symbol] if e.get('available_session') in pos and pos[e['available_session']]<lo]
        row['optional_left_context_event_id']=before[-1]['event_id'] if before else None
        own=[s for s in slots if s['interval_id']==row['interval_id']]
        require(set(row['existing_event_ids'])=={e for s in own for e in s['existing_event_ids']}, 'Interval/slot event union')
        for e in selected: consumed.add(e['event_id']); memberships[e['event_id']].append(row['interval_id'])
        items={json.dumps(x,sort_keys=True):x for s in own for x in s['inherited_quarter_review_items']}
        long_items={json.dumps(x,sort_keys=True):x for s in own for x in s['inherited_long_interval_review_items']}
        references={x['path']:x['sha256'] for e in selected for x in e['source_references'] if x.get('path')}
        tasks.append(dict(interval_id=row['interval_id'],symbol=symbol,anchor_window=row['required_anchor_window'],status='pending_evidence_review_not_failed',
            existing_event_ids=row['existing_event_ids'],recorded_evidence_hashes=references,
            unresolved_anchor_records=[event_map[e] for e in row['unresolved_anchor_event_ids']],
            inherited_quarter_review_items=list(items.values()),
            inherited_long_interval_review_items=list(long_items.values()),
            inherited_vendor_date_leads=sorted({d for s in own for d in s['inherited_vendor_date_leads']}),
            evidence_or_clock_issue_event_ids=[e['event_id'] for e in selected if e['metadata_issues'] or e['inherited_event_issues'] or not e['all_recorded_evidence_hashes_verified']],
            required_checks=['Enumerate qualifying ordinary periodic-results releases over this bounded interval using dated issuer archive/index/notice evidence and full pagination.',
                'Reconcile each enumerated entry, not only existing catalog IDs; distinguish missing document from evidence that an actual release occurred.',
                'Validate source issuer/period/actual-release date/timezone and conservative session-close clock; exact first-public timestamp is not required for date-only post-known scope.',
                'Classify syndicated duplicates versus preliminary/formal/correction information events under a prospectively fixed taxonomy.',
                'Reconcile relevant quarter gaps, vendor hints and unknown dates against this interval only; do not infer every skipped quarter was released inside it.',
                'Support left and right interval coverage without requiring an arbitrarily old predecessor; a missing predecessor alone is not failure.',
                'Preserve unresolved bounded coverage as unavailable; do not replace unknown events with zero or exclude securities from the original pool.'],
            boundary_review=dict(left_status='pending_bounded_coverage_evidence',right_status='pending_bounded_coverage_evidence',
                optional_predecessor_event_id=row['optional_left_context_event_id'],missing_predecessor_is_failure=False),
            new_event_supplements_included=False,calendar_completeness_verified=False,factor_admitted=False))
    require(len(intervals)==99 and len(consumed)==1677,'Frozen expected consumed scope')
    catalog=[dict(event_id=e,source_event_audit=event_map[e],interval_ids=sorted(memberships[e]),
                  source_catalog='frozen_old_3771_only',calendar_completeness_verified=False,factor_admitted=False) for e in sorted(consumed)]
    count=dict(issuers=73,intervals=len(intervals),stock_slots=len(slots),etf_slots=2074,source_catalog_events=3771,
        existing_consumed_events=len(catalog),unique_recorded_evidence_files=len({r['path'] for e in consumed for r in event_map[e]['source_references'] if r.get('path')}),
        issuer_session_days=sum(hi-lo+1 for row in intervals for lo,hi in [row['required_anchor_indices']]),
        slots_with_no_existing_events=sum(not r['existing_event_ids'] for r in slots),
        slots_with_unresolved_anchors=sum(bool(r['unresolved_anchor_event_ids']) for r in slots),
        intervals_without_optional_predecessor=sum(r['optional_left_context_event_id'] is None for r in intervals))
    summary=dict(schema_version=1,stage='SOURCE_ONLY_CONSUMED_EVENT_SCOPE_CATALOG',spec_sha256=spec_sha,counts=count,
        source_supplements_included=False,calendar_completeness_verified=False,factor_admitted=False,
        limitations=['Only previously frozen3771 event catalog included; new AVGO/ABT or other supplements require separate derived binding.',
          'No new event truth, first-public timestamp, interval completeness or strategy eligibility certified.',
          'Existing evidence hashes are provenance pointers; this run verifies source manifests, not every underlying report again.',
          '99 merged intervals retain all3904 original stock slots; outside predecessors are optional context only.'],**ZERO)
    return dict(zip(FILES,(intervals,slots,catalog,tasks,summary)))


def run(expected):
    validate_spec(expected)
    require(not any((OUT/name).exists() for name in (*FILES,'MANIFEST.json')), 'Preserve existing/partial scope outputs')
    result=products(expected)
    for name,value in result.items(): publish(OUT/name,value)
    publish(OUT/'MANIFEST.json',dict(spec_sha256=expected,output_hashes={name:sha(OUT/name) for name in FILES},**ZERO))
    return dict(passed=True,counts=result['SUMMARY.json']['counts'],manifest_sha256=sha(OUT/'MANIFEST.json'))


def verify(expected):
    validate_spec(expected); manifest=read(OUT/'MANIFEST.json')
    require(manifest['spec_sha256']==expected and set(manifest['output_hashes'])==set(FILES),'Manifest scope')
    for name,digest in manifest['output_hashes'].items(): require(sha(OUT/name)==digest,'Changed scope output')
    # Reconstruct only metadata, never prices or factors, to verify all mappings.
    for name,wanted in products(expected).items(): require(read(OUT/name)==wanted,'Reconstructed scope mismatch '+name)
    return dict(verified=True,manifest_sha256=sha(OUT/'MANIFEST.json'),counts=read(OUT/'SUMMARY.json')['counts'])


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('self-test','freeze','run','verify'));p.add_argument('--spec-sha');a=p.parse_args()
    if a.phase=='self-test': result=dict(synthetic_checks=self_test(),**ZERO)
    elif a.phase=='freeze': result=freeze()
    else:
        require(a.spec_sha is not None,'Explicit frozen SPEC SHA required')
        result={'run':run,'verify':verify}[a.phase](a.spec_sha)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
