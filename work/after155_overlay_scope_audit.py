"""Independent source-only audit of a four-event overlay against consumed windows.

No producer imports, no prices, no financial values, no models and no strategy
registration. Freeze before deriving real metadata intersections. Old catalogs
and the original99-window scope remain unchanged.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/monthly_event_overlay_scope156'
PATCH = 'outputs/monthly_event_patch156/'
SCOPE = 'outputs/monthly_event_scope156/'
CAL = 'outputs/monthly_paired_target143/CALENDAR.json'
META = 'outputs/monthly_local_linear70/META.json'
PARENT = 'outputs/monthly_earnings_repair/EVENTS.json'
TOP = {
    PATCH+'PATCH_SPEC.json':'a4d54d9fa4c558e30127bc189e7c5bb1aa3c1ed801120c021e1bf3ef91bd171d',
    PATCH+'CATALOG_MANIFEST.json':'a591e765c3f5099ccb4a73a705db07a837f185b2d40eaa7941ec9d78e44fc722',
    PATCH+'INTEGRATION.json':'e1e7fb894039ca6700c7be1e82ca6e545aefb9282e410f63d18a7b948183fbc0',
    SCOPE+'SPEC.json':'098eaa6e519e49cd8bb26dd5e7e94d52f6fe1eb566d5a2b4e9c2b8388d10947e',
    SCOPE+'MANIFEST.json':'cd1c7135dfb0610b8c4bb46a7e10c8c94c4312ccda3033918073bb1b4de20876',
}
EXPECTED = (
 ('ABT_2020-04-16','ABT',2020,1,'2020-03-31','2020-04-16','2020-04-17'),
 ('ABT_2020-07-16','ABT',2020,2,'2020-06-30','2020-07-16','2020-07-17'),
 ('ABT_2020-10-21','ABT',2020,3,'2020-09-30','2020-10-21','2020-10-22'),
 ('AVGO_2021-06-03','AVGO',2021,2,'2021-05-02','2021-06-03','2021-06-04'))
FIELDS=('event_id','symbol','fiscal_year','fiscal_quarter','official_actual_period_end','publication_date','available_session')
FILES=('PATCH_VALIDATION.json','SLOT_AUDIT.json','INTERVAL_AUDIT.json','REMAINING_ITEMS.json','AUDIT.json')
ZERO=dict(price_values_read=0,financial_values_extracted=0,factor_calculations=0,model_fits=0,orders=0,returns=0,new_strategy_trials=0,network_requests=0)


def require(ok,message):
    if not ok: raise ValueError(message)


def path(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require('..' not in p.parts and 'CloudStorage' not in p.parts and p.is_relative_to(ROOT),'Outside research root')
    require(not p.is_symlink() and p.resolve().is_relative_to(ROOT),'Source symlink/escape')
    return p


def digest(value):
    h=hashlib.sha256()
    with path(value).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(value):return json.loads(path(value).read_text())


def write(value,content):
    p=path(value);p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(content,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.overlay-')
    try:
        with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        os.link(tmp,p)
    finally:os.unlink(tmp)


def source_closure():
    pins={};sizes={}
    def add(p,h,n=None):
        p=str(path(p).relative_to(ROOT));require(p not in pins or pins[p]==h,'Conflicting source hash')
        pins[p]=h
        if n is not None:
            require(p not in sizes or sizes[p]==n,'Conflicting byte size');sizes[p]=n
    for p,h in TOP.items():add(p,h);require(digest(p)==h,'Top-level pin mismatch')
    ps=read(PATCH+'PATCH_SPEC.json');pm=read(PATCH+'CATALOG_MANIFEST.json')
    for p,r in ps['inputs'].items():require(str(path(p))==str(path(r['path'])),'Patch input path');add(p,r['sha256'],r['bytes'])
    for r in pm['components']:add(r['path'],r['sha256'],r['bytes'])
    ss=read(SCOPE+'SPEC.json');sm=read(SCOPE+'MANIFEST.json')
    for p,h in {**ss['source_hashes'],**ss['source_code_hashes']}.items():add(p,h)
    for f,h in sm['output_hashes'].items():add(SCOPE+f,h)
    for p,h in pins.items():
        require(digest(p)==h,'Source changed '+p)
        if p in sizes:require(path(p).stat().st_size==sizes[p],'Source bytes changed '+p)
    require(pm['spec']['sha256']==TOP[PATCH+'PATCH_SPEC.json'] and sm['spec_sha256']==TOP[SCOPE+'SPEC.json'],'Parent SPEC linkage')
    return pins


def fiscal(e):
    y,q=e.get('fiscal_year'),e.get('fiscal_quarter')
    return y*4+q-1 if type(y) is int and type(q) is int and 1<=q<=4 else None


def label(n):return '%04dQ%d'%(n//4,n%4+1)


def scoped_sequence(rows,positions,lo,hi):
    known=[e for e in rows if e.get('available_session') in positions]
    known.sort(key=lambda e:(positions[e['available_session']],e['event_id']))
    inside=[e for e in known if lo<=positions[e['available_session']]<=hi]
    earlier=[e for e in known if positions[e['available_session']]<lo]
    flags=[]
    for a,b in zip(earlier[-1:]+inside,(earlier[-1:]+inside)[1:]):
        qa,qb=fiscal(a),fiscal(b)
        if qa is not None and qb==qa+1:continue
        step=None if qa is None or qb is None else qb-qa
        kind='unknown_quarter_label' if step is None else 'duplicate_quarter' if step==0 else 'nonmonotone_quarter' if step<0 else 'unobserved_quarter_labels'
        where='left_predecessor_crossing_context' if positions[a['available_session']]<lo else 'inside_consumed_anchor_window'
        flags.append(dict(symbol=b['symbol'],before=a['event_id'],after=b['event_id'],before_publication=a['publication_date'],after_publication=b['publication_date'],
            before_anchor=a['available_session'],after_anchor=b['available_session'],kind=kind,quarter_step=step,scope=where,
            intermediate_fiscal_labels=[label(x) for x in range(qa+1,qb)] if step is not None and step>1 else [],
            exact_missing_publication_dates_known=False,requires_source_review=True))
    unknown=[e['event_id'] for e in rows if e.get('available_session') not in positions and
             (not isinstance(e.get('publication_date'),str) or e['publication_date']<=list(positions)[hi])]
    return dict(event_ids=[e['event_id'] for e in inside],quarter_items=flags,unknown_anchor_ids=unknown,
        optional_predecessor=earlier[-1]['event_id'] if earlier else None)


def self_test():
    p={str(i):i for i in range(15)}
    def e(i,q):return dict(event_id='E'+str(i),symbol='X',available_session=str(i),publication_date=str(i),fiscal_year=2020,fiscal_quarter=q)
    a=scoped_sequence([e(1,1),e(5,3)],p,4,10)
    require(a['quarter_items'][0]['scope']=='left_predecessor_crossing_context','Outside predecessor')
    require(a['quarter_items'][0]['intermediate_fiscal_labels']==['2020Q2'],'Fiscal labels')
    b=scoped_sequence([e(4,1),e(5,3)],p,4,10)
    require(b['quarter_items'][0]['scope']=='inside_consumed_anchor_window','Internal pair')
    c=scoped_sequence([e(4,1),e(5,2),e(6,3)],p,4,10)
    require(c['quarter_items']==[] and len(c['event_ids'])==3,'Overlay gap repair')
    require(scoped_sequence([e(3,1),e(11,2)],p,4,10)['event_ids']==[],'Closed bounds')
    require(scoped_sequence([e(4,1),e(10,2)],p,4,10)['event_ids']==['E4','E10'],'Inclusive bounds')
    require(scoped_sequence([],p,4,10)['optional_predecessor'] is None,'Missing predecessor not failure')
    require(scoped_sequence([e(4,1),e(5,1)],p,4,10)['quarter_items'][0]['kind']=='duplicate_quarter','No quarter dedup')
    require(scoped_sequence([e(4,3),e(5,1)],p,4,10)['quarter_items'][0]['kind']=='nonmonotone_quarter','Nonmonotone')
    z=e(1,1);z['available_session']=None;z['publication_date']=None
    require(scoped_sequence([z],p,4,10)['unknown_anchor_ids']==['E1'],'Unknown not dropped')
    return 10


def freeze():
    require(not (OUT/'SPEC.json').exists(),'SPEC already frozen')
    checks=self_test();pins=source_closure()
    spec=dict(stage='FROZEN_INDEPENDENT_SOURCE_ONLY_OVERLAY_SCOPE_AUDIT',created_utc=datetime.now(timezone.utc).isoformat(),runtime_python=sys.version,
        input_hashes=pins,source_hashes={str(Path(__file__).relative_to(ROOT)):digest(__file__)},synthetic_checks=checks,
        expected_added_identities=[list(x) for x in EXPECTED],original_scope=dict(stocks=73,intervals=99,slots=3904,original_catalog=3771),
        scope_rule='For each original slot t, anchors in [t-271,t-44] inclusive; pair gaps inside this range separate from optional prior-event crossings.',
        completeness_rule='No absence is inferred from quarter continuity, repaired known gaps, or no unresolved data rows. Every interval retains archive-enumeration/boundary verification pending.',**ZERO)
    write(OUT/'SPEC.json',spec);return dict(spec_sha256=digest(OUT/'SPEC.json'),synthetic_checks=checks)


def check_spec(expected):
    require(digest(OUT/'SPEC.json')==expected,'Exact SPEC required');s=read(OUT/'SPEC.json')
    require(s['runtime_python']==sys.version and s['input_hashes']==source_closure(),'Runtime/input closure')
    require(s['source_hashes']=={str(Path(__file__).relative_to(ROOT)):digest(__file__)},'Frozen audit code')
    return s


def validate_patch(parent,added,positions):
    require(len(parent)==3771 and len({e['event_id'] for e in parent})==3771,'Parent identity count')
    require(tuple(tuple(e[k] for k in FIELDS) for e in added)==EXPECTED,'Four exact identities')
    ps=read(PATCH+'PATCH_SPEC.json');integration=read(PATCH+'INTEGRATION.json');manifest=read(PATCH+'CATALOG_MANIFEST.json')
    require(manifest['combined_event_count']==3775 and manifest['parent_saved_bytes_rewritten'] is False and manifest['parent_paths_rewritten'] is False,'Overlay composition')
    require(all(manifest[k] is False for k in ('factor_admitted','strategy_admitted','whole_window_archive_complete')),'No expanded admission')
    require(integration['status']=='PASS_FOUR_EVENT_SOURCE_OVERLAY' and integration['old_events_bytes_unchanged'] is True,'Patch integration')
    lookup={c['event_id']:c for c in ps['candidates']};seen=list(parent);validated=[]
    for e in added:
        c=lookup[e['event_id']]
        for old in seen:
            keys=[('event_id',),('symbol','fiscal_year','fiscal_quarter'),('symbol','publication_date'),('symbol','official_actual_period_end')]
            require(all(tuple(e.get(k) for k in key)!=tuple(old.get(k) for k in key) for key in keys),'Overlay duplicate/conflict')
        seen.append(e)
        sessions=list(positions);i=bisect_right(sessions,e['publication_date'])
        require(i<len(sessions) and e['available_session']==sessions[i],'Independent next-session mapping')
        require(e['precision']=='date_only' and e['known_by_is_actual_publication'] is False and e['first_public_time_verified'] is False,'Conservative clock policy')
        require(e['source_display_timestamp_used_as_first_public_time'] is False,'Displayed clock not promoted')
        for k,v in c['expected'].items():require(e[k]==v,'Independent reviewed clock '+k)
        require(e['evidence_path']==c['raw_path'] and digest(e['evidence_path'])==e['evidence_sha256'],'New raw source provenance')
        review=e['independent_source_review'];require(digest(review['path'])==review['sha256'],'Independent source receipt')
        receipt=read(review['path'])
        if e['symbol']=='ABT':
            require(receipt['admission_decision']['these_three_events_may_enter_new_derived_catalog'] is True,'ABT reviewed scope')
            match=[x for x in receipt['events'] if x['candidate_event_id']==e['event_id']]
            require(len(match)==1 and match[0]['single_event_admissible'] is True,'ABT exact event review')
        else:require(receipt['admission_decision']['single_event_may_enter_new_derived_catalog'] is True,'AVGO reviewed scope')
        validated.append(dict(event_id=e['event_id'],identity=tuple(e[k] for k in FIELDS),next_session_independently_verified=True,
            source_bytes_and_independent_review_pins_verified=True,clock_fields_match_independent_source_review=True,
            exact_intraday_clock_independently_recomputed=False,whole_window_completeness_certified=False))
    return dict(passed=True,parent_count=3771,added_count=4,combined_count=3775,new_conflicts=0,events=validated,
        limit='Independent metadata/hash/composition validation; relies on pinned independent raw-document source reviews for original publication facts. No new assertion of first-public time.',**ZERO)


def audit_products(expected):
    calendar=read(CAL);sessions=calendar['sessions'];positions={d:i for i,d in enumerate(sessions)}
    require(sessions==sorted(set(sessions)),'Calendar order')
    parent=read(PARENT);added=read(PATCH+'ADDED_EVENTS.json');proof=validate_patch(parent,added,positions)
    groups=[defaultdict(list),defaultdict(list)]
    for e in parent:groups[0][e['symbol']].append(e);groups[1][e['symbol']].append(e)
    for e in added:groups[1][e['symbol']].append(e)
    meta=read(META);slots=read(SCOPE+'SLOTS.json');intervals=read(SCOPE+'INTERVALS.json');tasks=read(SCOPE+'EVIDENCE_TASKS.json')
    require(len(slots)==3904 and len(intervals)==99 and len({s['symbol'] for s in slots})==73,'Original scope counts')
    pair_seen=set();audits=[];changes=[];allids=[set(),set()];seqcounts=[Counter(),Counter()]
    for slot in slots:
        i,j=slot['signal_index'],slot['candidate_index'];key=(i,j);require(key not in pair_seen,'Duplicate slot');pair_seen.add(key)
        require(slot['symbol']==meta['symbols'][i][j] and slot['signal_date']==meta['dates'][i] and slot['symbol'] not in meta['etfs'],'Full original stock identity')
        t=positions[slot['signal_date']];lo,hi=t-271,t-44
        require(slot['required_anchor_indices']==[lo,hi] and slot['required_anchor_window']==[sessions[lo],sessions[hi]],'Independent precise slot scope')
        before=scoped_sequence(groups[0][slot['symbol']],positions,lo,hi);after=scoped_sequence(groups[1][slot['symbol']],positions,lo,hi)
        require(before['event_ids']==slot['existing_event_ids'],'Old consumed membership independently checked')
        for z,r in enumerate((before,after)):
            allids[z].update(r['event_ids']);seqcounts[z].update(q['scope'] for q in r['quarter_items'])
        delta=[x for x in after['event_ids'] if x not in before['event_ids']]
        r=dict(slot_id=slot['slot_id'],interval_id=slot['interval_id'],symbol=slot['symbol'],signal_date=slot['signal_date'],anchor_window=slot['required_anchor_window'],
            before=before,after=after,added_event_ids=delta,calendar_completeness_verified=False,factor_admitted=False)
        audits.append(r)
        if delta:changes.append({k:r[k] for k in ('slot_id','interval_id','symbol','signal_date','anchor_window','added_event_ids')})
    require(pair_seen=={(i,j) for i,row in enumerate(meta['symbols']) for j,s in enumerate(row) if s not in meta['etfs']},'No original slot omitted')
    # Rebuild maximal connected intervals independently from individual slots.
    ranges=defaultdict(list)
    for slot in slots:ranges[slot['symbol']].append(slot['required_anchor_indices'])
    expected_intervals=[]
    for sym in sorted(ranges):
        merged=[]
        for lo,hi in sorted(ranges[sym]):
            if merged and lo<=merged[-1][1]+1:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        expected_intervals.extend((sym,lo,hi) for lo,hi in merged)
    require([(r['symbol'],*r['required_anchor_indices']) for r in intervals]==expected_intervals,'Independent99 maximal interval union')
    interval_audit=[];remaining={};hints=[]
    for interval,task in zip(intervals,tasks):
        require(interval['interval_id']==task['interval_id'],'Task/interval correspondence')
        lo,hi=interval['required_anchor_indices'];sym=interval['symbol'];key=interval['interval_id']
        own=[r for r in audits if r['interval_id']==key]
        require([r['slot_id'] for r in own]==interval['source_slot_ids'],'Interval owns original slots')
        before=scoped_sequence(groups[0][sym],positions,lo,hi);after=scoped_sequence(groups[1][sym],positions,lo,hi)
        require(before['event_ids']==interval['existing_event_ids'],'Old interval members')
        require(set(after['event_ids'])=={e for r in own for e in r['after']['event_ids']},'Merged interval/slot coverage')
        interval_audit.append(dict(interval_id=key,symbol=sym,anchor_window=interval['required_anchor_window'],before=before,after=after,
            added_event_ids=[e for e in after['event_ids'] if e not in before['event_ids']],
            archive_enumeration_verified=False,left_boundary_verified=False,right_boundary_verified=False,calendar_completeness_verified=False))
        for r in own:
            for q in r['after']['quarter_items']:
                k=(q['symbol'],q['before'],q['after'],q['scope'])
                if k not in remaining:remaining[k]=dict(q,slot_ids=[],signal_dates=[],interval_ids=[])
                entry=remaining[k];entry['slot_ids'].append(r['slot_id']);entry['signal_dates'].append(r['signal_date'])
                if key not in entry['interval_ids']:entry['interval_ids'].append(key)
        for d in task['inherited_vendor_date_leads']:
            matching=[e['event_id'] for e in added if e['symbol']==sym and e['publication_date']==d]
            hints.append(dict(symbol=sym,provider_date=d,interval_id=key,matched_new_event_ids=matching,
                status='exact_publication_date_matches_reviewed_added_event' if matching else 'unresolved_vendor_lead_not_proven_missing_release'))
    impacts=[dict(event_id=e['event_id'],symbol=e['symbol'],publication_date=e['publication_date'],available_session=e['available_session'],
        slots=[r['slot_id'] for r in changes if e['event_id'] in r['added_event_ids']],
        signal_dates=[r['signal_date'] for r in changes if e['event_id'] in r['added_event_ids']],
        interval_ids=sorted({r['interval_id'] for r in changes if e['event_id'] in r['added_event_ids']})) for e in added]
    allremaining=list(remaining.values())
    items=dict(quarter_items=allremaining,vendor_date_leads=hints,
        empty_intervals=[dict(interval_id=r['interval_id'],symbol=r['symbol'],anchor_window=r['anchor_window']) for r in interval_audit if not r['after']['event_ids']],
        unknown_anchor_intervals=[dict(interval_id=r['interval_id'],event_ids=r['after']['unknown_anchor_ids']) for r in interval_audit if r['after']['unknown_anchor_ids']],
        boundary_coverage_pending=[r['interval_id'] for r in interval_audit],
        warning='Exact missing release dates cannot be inferred from fiscal labels. Listed endpoints and labels are source-research tasks, not synthesized events.')
    report=dict(stage='INDEPENDENT_SOURCE_ONLY_OVERLAY_SCOPE_AUDIT',passed=True,spec_sha256=expected,
        counts=dict(original_catalog=3771,added_catalog=4,combined_catalog=3775,issuers=73,intervals=99,slots=3904,
            before_consumed_unique=len(allids[0]),after_consumed_unique=len(allids[1]),changed_slots=len(changes),
            changed_intervals=sum(bool(r['added_event_ids']) for r in interval_audit),
            before_slot_quarter_flags=dict(seqcounts[0]),after_slot_quarter_flags=dict(seqcounts[1]),
            before_slots_with_any_quarter_flag=sum(bool(r['before']['quarter_items']) for r in audits),
            after_slots_with_any_quarter_flag=sum(bool(r['after']['quarter_items']) for r in audits),
            remaining_unique_inside_pairs=sum(q['scope']=='inside_consumed_anchor_window' for q in allremaining),
            remaining_unique_left_crossing_pairs=sum(q['scope']=='left_predecessor_crossing_context' for q in allremaining),
            complete_intervals_certified=0),
        added_event_impacts=impacts,changed_slot_mapping=changes,
        decision='Known missing quarterly events are added under pinned single-event approvals. This cannot certify absence of other qualifying releases. No complete-event subset is certified by this overlay audit.',
        next_evidence='Review remaining inside-window pairs and unknown/empty intervals first, then reconcile archive enumeration and both boundaries for any prospectively selected issuer-window scope. Left-predecessor pairs are context and must not force repair outside the consumed window.',
        limitations=['No full original3775 catalog is rewritten. Original event dictionaries and 99 interval definitions remain frozen.',
            'A fiscal-label-contiguous interval is a candidate for evidence review, not a proven complete archive.',
            'No price performance or held positions were used to choose a research scope.'],**ZERO)
    return dict(zip(FILES,(proof,audits,interval_audit,items,report)))


def run(expected):
    check_spec(expected);require(not any((OUT/f).exists() for f in (*FILES,'MANIFEST.json')),'Preserve existing or partial audit')
    data=audit_products(expected)
    for f,value in data.items():write(OUT/f,value)
    write(OUT/'MANIFEST.json',dict(spec_sha256=expected,output_hashes={f:digest(OUT/f) for f in FILES},**ZERO))
    return dict(passed=True,counts=data['AUDIT.json']['counts'],manifest_sha256=digest(OUT/'MANIFEST.json'))


def verify(expected):
    check_spec(expected);m=read(OUT/'MANIFEST.json');require(m['spec_sha256']==expected and set(m['output_hashes'])==set(FILES),'Audit manifest')
    for f,h in m['output_hashes'].items():require(digest(OUT/f)==h,'Audit bytes changed')
    for f,value in audit_products(expected).items():require(read(OUT/f)==json.loads(json.dumps(value)),'Reconstructed audit '+f)
    return dict(verified=True,counts=read(OUT/'AUDIT.json')['counts'],manifest_sha256=digest(OUT/'MANIFEST.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('self-test','freeze','run','verify'));p.add_argument('--spec-sha');a=p.parse_args()
    if a.phase=='self-test':out=dict(synthetic_checks=self_test(),**ZERO)
    elif a.phase=='freeze':out=freeze()
    else:
        require(a.spec_sha is not None,'Frozen audit SPEC required');out={'run':run,'verify':verify}[a.phase](a.spec_sha)
    print(json.dumps(out,sort_keys=True))


if __name__=='__main__':main()
