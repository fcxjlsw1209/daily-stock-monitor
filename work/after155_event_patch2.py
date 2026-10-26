"""Second, immutable source-only event increment: CRM/SAP/two HSBC releases.

No whole-calendar or factor admission. Period-end and fiscal-quarter keys are
related-report diagnostics, never deduplication keys for preliminary/final news.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after155_event_patch2.py'
OUT='outputs/monthly_event_patch156_batch2'
SPEC=OUT+'/PATCH_SPEC.json'
PARENT='outputs/monthly_earnings_repair/EVENTS.json'
FIRST='outputs/monthly_event_patch156/ADDED_EVENTS.json'
FIRST_MANIFEST='outputs/monthly_event_patch156/CATALOG_MANIFEST.json'
FIRST_SPEC='outputs/monthly_event_patch156/PATCH_SPEC.json'
CALENDAR='outputs/monthly_paired_target143/CALENDAR.json'
SCOPE='outputs/monthly_event_scope156/'
REVIEWS={'crm_sap':'work/after155_crm_sap_independent_review.json','hsbc':'work/after155_hsbc_independent_review.json'}
PINS={PARENT:'1e027fc8b933af9753241c15a2e0dcc5a32daa13205ba4d74f90b64db6e2e103',
 FIRST:'1d107c783d93a78ced29572383c55f55c50d24108014aee417dc3772fef2a5e3',
 FIRST_MANIFEST:'a591e765c3f5099ccb4a73a705db07a837f185b2d40eaa7941ec9d78e44fc722',
 FIRST_SPEC:'a4d54d9fa4c558e30127bc189e7c5bb1aa3c1ed801120c021e1bf3ef91bd171d',
 REVIEWS['crm_sap']:'02c138c6d3f81f701c40109b1f85fdbe74d8558ad93449a47eca18a5ac104fc4',
 REVIEWS['hsbc']:'5e92978a2d3a5165eb1152b84397a9c99f9a4c53660b345dc9f141b1efe20316',
 SCOPE+'SPEC.json':'098eaa6e519e49cd8bb26dd5e7e94d52f6fe1eb566d5a2b4e9c2b8388d10947e',
 SCOPE+'SLOTS.json':'64515d1c5d81f4acd8c38fa6552c57ebdd7b4521769ddc13217b8aa0243a1022',
 SCOPE+'MANIFEST.json':'cd1c7135dfb0610b8c4bb46a7e10c8c94c4312ccda3033918073bb1b4de20876'}
EXPECTED=(('CRM_2019-12-03','CRM',2020,3,'2019-10-31','2019-12-03','2019-12-04'),
 ('HSBC_2016-08-03','HSBC',2016,2,'2016-06-30','2016-08-03','2016-08-04'),
 ('HSBC_2017-02-21','HSBC',2016,4,'2016-12-31','2017-02-21','2017-02-22'),
 ('SAP_2020-01-28','SAP',2019,4,'2019-12-31','2020-01-28','2020-01-29'))
ZERO={'network_requests':0,'price_values_read':0,'financial_values_extracted':0,'factor_calculations':0,'models_fit':0,'orders':0,'returns':0,'strategy_plans':0}
OUTPUT_NAMES=['ADDED_EVENTS.json','CONSUMPTION_IMPACT.json','CATALOG_MANIFEST.json','INTEGRATION.json']

def require(ok,msg):
    if not ok:raise ValueError(msg)
def relative(p):
    p=Path(p)
    if p.is_absolute():p=p.relative_to(ROOT)
    require('..' not in p.parts,'path traversal');return str(p)
def local(p):
    p=ROOT/relative(p);require(p.resolve()==p,'symlink redirect');return p
def digest(blob):return hashlib.sha256(blob).hexdigest()
def sha(p):return digest(local(p).read_bytes())
def read(p):return json.loads(local(p).read_bytes())
def binding(p):
    p=relative(p);raw=local(p).read_bytes();return dict(path=p,sha256=digest(raw),bytes=len(raw))
def checked(b):
    raw=local(b['path']).read_bytes();require(digest(raw)==b['sha256'] and len(raw)==b['bytes'],'binding changed '+b['path']);return raw
def write(p,value):
    path=local(p);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw)
    return binding(p)
def identity(c):
    e=c['approved'];return(c['event_id'],e['symbol'],c['fiscal_year'],c['fiscal_quarter'],e['official_actual_period_end'],e['publication_date'],e['available_session'])

def reviewed_candidates():
    first=read(REVIEWS['crm_sap']);second=read(REVIEWS['hsbc']);rows=[]
    require(first['passed'] is True and first['counters']['single_events_passed']==2,'CRM/SAP review approval')
    for r in first['event_reviews']:
        if r['symbol']=='TSLA':
            require(r['single_event_source_admitted'] is False,'TSLA may not enter');continue
        require(r['symbol'] in ('CRM','SAP') and r['single_event_source_admitted'] is True,'unexpected source candidate')
        e=deepcopy(r['verified_candidate']);kind=e['release_kind']
        rows.append(dict(event_id=r['event_id'],approved=e,fiscal_year=e['fiscal_year'],fiscal_quarter=e['fiscal_quarter'],
            review_key='crm_sap',review_event_container='event_reviews',raw_path=e['evidence_path'],raw_sha256=e['evidence_sha256'],
            receipt_path='outputs/monthly_event_gap156_crm_sap_tsla/'+r['symbol'].lower()+'.receipt.json',receipt_sha256=r['source_receipt_sha256'],
            source_kind='original_issuer_release_full_bytes',release_kind=kind,
            reporting_period_type='preliminary_quarterly_and_annual_results' if r['symbol']=='SAP' else 'quarterly_results',
            standalone_quarterly_release_verified=r['symbol']=='CRM',quarter_is_period_endpoint_ordinal=r['symbol']=='SAP',
            period_end_is_context_inference=r['period_end_is_context_inference'],period_basis=r['period_basis'],
            source_evidence=r['raw_evidence'],review_limitations=r['limitations']))
    require(second['review_status']=='APPROVE_TWO_SINGLE_EVENT_SOURCES_FOR_NEW_DERIVED_CATALOG_ONLY','HSBC review approval')
    for r in second['checked_events']:
        require(r['single_event_source_admitted_for_new_derived_catalog'] is True and r['availability']['symbol']=='HSBC','HSBC not approved')
        require(r['quarter_is_period_endpoint_ordinal'] is True and r['standalone_quarterly_release_verified'] is False,'HSBC duration classification lost')
        rows.append(dict(event_id=r['event_id'],approved=deepcopy(r['availability']),fiscal_year=r['fiscal_year'],fiscal_quarter=r['endpoint_quarter'],
            review_key='hsbc',review_event_container='checked_events',raw_path=r['pdf_path'],raw_sha256=r['pdf_sha256'],
            receipt_path=r['original_receipt_path'],receipt_sha256=r['original_receipt_sha256'],source_kind='issuer_original_earnings_release_pdf',
            release_kind='actual_'+r['period_type'],reporting_period_type=r['period_type'],standalone_quarterly_release_verified=False,
            quarter_is_period_endpoint_ordinal=True,period_end_is_context_inference=False,
            period_basis=second['conclusions']['period_label_qualification'],source_evidence=r['independently_extracted_evidence'],
            review_limitations=[second['conclusions'][k] for k in ['quarter_label_policy','archival_limit','gap_scope']]))
    rows.sort(key=lambda r:r['event_id']);require(tuple(map(identity,rows))==EXPECTED,'not the exact four reviewed identities')
    return rows


def input_closure():
    expected={}
    def add(p,h):
        p=relative(p);require(p not in expected or expected[p]==h,'conflicting pinned SHA '+p);require(sha(p)==h,'input SHA changed '+p);expected[p]=h
    for p,h in PINS.items():add(p,h)
    for name in REVIEWS.values():
        r=read(name)
        for p,h in r['input_hashes'].items():add(p,h)
    for b in read(FIRST_SPEC)['inputs'].values():checked(b);add(b['path'],b['sha256'])
    for b in read(FIRST_MANIFEST)['components']:checked(b);add(b['path'],b['sha256'])
    scope=read(SCOPE+'SPEC.json')
    for section in ('source_hashes','source_code_hashes'):
        for p,h in scope[section].items():add(p,h)
    sm=read(SCOPE+'MANIFEST.json')
    # Scope manifest binds all five original products.
    require(sm['spec_sha256']==PINS[SCOPE+'SPEC.json'],'scope manifest spec binding')
    for name,h in sm['output_hashes'].items():add(SCOPE+name,h)
    for p,h in read(CALENDAR)['provenance']['source_hashes'].items():add(p,h)
    for p in (SELF,'run_local.sh','local_run.py','local.sandbox.sb'):add(p,sha(p))
    return {p:binding(p) for p in sorted(expected)}


def collision_audit(parent,added):
    seen=list(parent);related=[]
    for new in added:
        require(all(new.get(k) is not None for k in ['event_id','symbol','publication_date','source_url','official_actual_period_end','fiscal_year','fiscal_quarter']),'missing new identity')
        for old in seen:
            same_symbol=old.get('symbol')==new['symbol']
            require(old.get('event_id')!=new['event_id'],'duplicate event ID')
            require(not(same_symbol and old.get('publication_date')==new['publication_date']),'same issuer/date needs distinct-event resolution')
            require(not(same_symbol and old.get('source_url')==new['source_url']),'duplicate original source URL')
            quarter=same_symbol and (old.get('fiscal_year'),old.get('fiscal_quarter'))==(new['fiscal_year'],new['fiscal_quarter'])
            period=same_symbol and old.get('official_actual_period_end')==new['official_actual_period_end']
            if quarter or period:related.append(dict(existing_event_id=old.get('event_id'),new_event_id=new['event_id'],same_fiscal_quarter=quarter,same_period_end=period,action='retained_distinct_release_not_merged'))
        seen.append(new)
    return related


def frozen_scope():
    rows=read(SCOPE+'SLOTS.json');sessions=read(CALENDAR)['sessions'];idx={d:i for i,d in enumerate(sessions)}
    require(len(rows)==3904 and len({r['slot_id'] for r in rows})==3904,'stock slots changed')
    require(len({r['symbol'] for r in rows})==73 and len({r['interval_id'] for r in rows})==99,'scope universe changed')
    for r in rows:
        t=idx[r['signal_date']];bounds=[t-271,t-44]
        require(r['required_anchor_indices']==bounds,'anchor index window changed')
        require(r['required_anchor_window']==[sessions[i] for i in bounds],'anchor date window changed')
    return rows


def freeze():
    require(not local(SPEC).exists(),'spec already exists')
    inputs=input_closure();candidates=reviewed_candidates();frozen_scope()
    require(len(read(PARENT))==3771 and len(read(FIRST))==4,'base component counts')
    # Perform collision checks on approved metadata, without normalized output creation.
    approved=[dict(c['approved'],event_id=c['event_id'],fiscal_year=c['fiscal_year'],fiscal_quarter=c['fiscal_quarter']) for c in candidates]
    related=collision_audit(read(PARENT)+read(FIRST),approved)
    s=dict(schema_version=1,kind='second_four_event_source_increment_spec',created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,candidates=candidates,parent_count=3771,first_increment_count=4,added_count=4,combined_count=3779,
        prior_manifest=inputs[FIRST_MANIFEST],review_paths=REVIEWS,python=sys.version,
        related_period_releases=related,consumption_scope=inputs[SCOPE+'SLOTS.json'],
        policy=dict(composition='References to original3771 + unchanged first4 + new4; no combined catalog serialization',
                    hard_conflicts=['event_id','issuer/publication_date','issuer/source_url'],
                    period_keys='Same fiscal-quarter or period-end records remain distinct; flag relationship, never deduplicate on those keys.',
                    scope='Original3904slots/99intervals with anchor[t-271,t-44] unchanged',
                    single_source_admission_only=True,whole_window_archive_complete=False,factor_admitted=False,strategy_admitted=False),activity=ZERO)
    print(json.dumps(write(SPEC,s)))


def load_spec(expected):
    require(sha(SPEC)==expected,'spec SHA mismatch');s=read(SPEC)
    for b in s['inputs'].values():checked(b)
    require(reviewed_candidates()==s['candidates'],'independent review metadata differs')
    require(len(read(PARENT))==3771 and len(read(FIRST))==4,'component count mismatch');frozen_scope()
    return s


def normalize(s):
    path=str(ROOT/'outputs/monthly_earnings_repair')
    if path not in sys.path:sys.path.insert(0,path)
    import event_schema
    import research_calendar
    frozen=read(CALENDAR);cal=research_calendar.market_calendar(frozen['start'],frozen['end'])
    require([str(d.date()) for d in cal.sessions]==frozen['sessions'],'constructed calendar differs from frozen')
    added=[]
    for c in s['candidates']:
        approved=c['approved'];raw=checked(s['inputs'][c['raw_path']]);require(digest(raw)==c['raw_sha256'],'original raw SHA mismatch')
        receipt=read(c['receipt_path']);require(sha(c['receipt_path'])==c['receipt_sha256'],'receipt SHA changed')
        require(receipt.get('http_status',receipt.get('status'))==200,'original receipt not HTTP200')
        if c['review_key']=='crm_sap':
            for e in c['source_evidence']:
                require(raw[e['raw_byte_start']:e['raw_byte_end']].decode('utf-8')==e['raw_excerpt'],'raw visible evidence locator changed')
        r=event_schema.normalize_event(dict(symbol=approved['symbol'],asset_type='equity',vendor={},official=dict(
            actual_period_end=approved['official_actual_period_end'],publication_date=approved['publication_date'],publication_at=None,
            publication_timezone=approved['publication_timezone'],source_url=approved['source_url'],verified=True,publication_kind='issuer_release')),calendar=cal)
        require(r['status']=='ready' and not r['issues'],'clock normalization failed')
        for key,value in r.items():
            if key=='status':continue
            require(key in approved and approved[key]==value,'approved normalized field mismatch '+c['event_id']+':'+key)
        r.update({k:deepcopy(c[k]) for k in ['event_id','fiscal_year','fiscal_quarter','source_kind','release_kind','reporting_period_type','standalone_quarterly_release_verified','quarter_is_period_endpoint_ordinal','period_end_is_context_inference','period_basis','source_evidence','review_limitations']})
        r.update(evidence_path=c['raw_path'],evidence_sha256=c['raw_sha256'],source_receipt=s['inputs'][c['receipt_path']],
            source_retrieved_at=receipt.get('attempted_utc',receipt.get('retrieved_at')),
            independent_source_review=s['inputs'][REVIEWS[c['review_key']]],review_event_container=c['review_event_container'],
            source_precision='date_only',first_public_time_verified=False,known_by_is_actual_publication=False,
            publication_version='later_retrieved_original_issuer_release',original_response_available=True,
            historical_archive_limitation='Later-retrieved issuer original; no contemporaneous immutable capture or exact first dissemination certified.',
            single_event_source_admitted=True,whole_window_archive_complete=False,factor_admitted=False,strategy_admitted=False,
            relation_policy='Retain any separately reviewed later formal/corrective release; same period is not duplicate-event identity.')
        added.append(r)
    require(tuple((r['event_id'],r['symbol'],r['fiscal_year'],r['fiscal_quarter'],r['official_actual_period_end'],r['publication_date'],r['available_session']) for r in added)==EXPECTED,'normalized identity scope changed')
    return added


def consumption(old,new):
    rows=frozen_scope();by_symbol=defaultdict(list)
    for e in old:by_symbol[e['symbol']].append(e)
    impacted=[];before_union=set();after_union=set();new_use={e['event_id']:[] for e in new}
    for r in rows:
        lo,hi=r['required_anchor_window']
        def belongs(e):return e.get('status')=='ready' and e.get('symbol')==r['symbol'] and lo<=e.get('available_session','')<=hi
        before=sorted(e['event_id'] for e in by_symbol[r['symbol']] if belongs(e))
        inserted=sorted(e['event_id'] for e in new if belongs(e));after=sorted(before+inserted)
        before_union.update(before);after_union.update(after)
        for event_id in inserted:new_use[event_id].append(r['slot_id'])
        if inserted:impacted.append(dict(slot_id=r['slot_id'],symbol=r['symbol'],signal_date=r['signal_date'],interval_id=r['interval_id'],required_anchor_window=r['required_anchor_window'],old_event_ids=before,new_event_ids=after,inserted_event_ids=inserted))
    return dict(kind='metadata_only_consumed_event_membership',slot_scope=binding(SCOPE+'SLOTS.json'),stock_slots=3904,stocks=73,intervals=99,
        anchor_offsets=[-271,-44],scope_definitions_changed=False,changed_slots=len(impacted),changed_intervals=len({r['interval_id'] for r in impacted}),
        consumed_unique_before=len(before_union),consumed_unique_after=len(after_union),per_new_event_slot_ids=new_use,changed_slot_records=impacted,
        predecessor_outside_scope_is_not_a_required_consumption_event=True,calendar_complete=False,factor_admitted=False,activity=ZERO)


def manifest(s,spec_sha,added):
    return dict(schema_version=1,kind='immutable_parent_plus_two_increments',spec=dict(path=SPEC,sha256=spec_sha),
        prior_manifest=s['prior_manifest'],components=[dict(s['inputs'][PARENT],count=3771,role='parent_original_bytes'),
        dict(s['inputs'][FIRST],count=4,role='unchanged_first_increment'),dict(added,count=4,role='second_increment_reviewed_sources')],
        combined_event_count=3779,logical_order='Parent original order, then first increment order, then second increment order',
        previous_component_bytes_rewritten=False,period_key_deduplication=False,whole_window_archive_complete=False,factor_admitted=False,strategy_admitted=False)


def build(spec_sha):
    s=load_spec(spec_sha)
    for name in OUTPUT_NAMES:require(not local(OUT+'/'+name).exists(),'output already exists '+name)
    added=normalize(s);old=read(PARENT)+read(FIRST);related=collision_audit(old,added);impact=consumption(old,added)
    ab=write(OUT+'/ADDED_EVENTS.json',added);ib=write(OUT+'/CONSUMPTION_IMPACT.json',impact);mb=write(OUT+'/CATALOG_MANIFEST.json',manifest(s,spec_sha,ab))
    for b in s['inputs'].values():checked(b)
    report=dict(schema_version=1,status='PASS_SECOND_FOUR_EVENT_SOURCE_INCREMENT',created_utc=datetime.now(timezone.utc).isoformat(),
       spec=dict(path=SPEC,sha256=spec_sha),added_events=ab,manifest=mb,consumption_impact=ib,
       parent_count=3771,first_increment_count=4,added_count=4,combined_count=3779,added_event_ids=[r['event_id'] for r in added],
       prior_component_hashes={p:s['inputs'][p]['sha256'] for p in [PARENT,FIRST,FIRST_MANIFEST,FIRST_SPEC]},prior_components_unchanged=True,
       independent_review_hashes={k:s['inputs'][p]['sha256'] for k,p in REVIEWS.items()},related_period_releases=related,
       clock_matches_independent_reviews=True,scope3904_unchanged=True,reporting_nature_preserved=True,
       cat_tsla_excluded=True,whole_window_archive_complete=False,factor_admitted=False,strategy_admitted=False,activity=ZERO)
    print(json.dumps(write(OUT+'/INTEGRATION.json',report)))


def verify(spec_sha):
    s=load_spec(spec_sha);report=read(OUT+'/INTEGRATION.json')
    added=json.loads(checked(report['added_events']));require(added==normalize(s),'new event records do not reproduce')
    require(json.loads(checked(report['manifest']))==manifest(s,spec_sha,report['added_events']),'manifest changed')
    old=read(PARENT)+read(FIRST);require(collision_audit(old,added)==report['related_period_releases'],'collision audit differs')
    impact=json.loads(checked(report['consumption_impact']));require(impact==consumption(old,added),'metadata impact differs')
    require(report['combined_count']==len(old)+len(added)==3779,'counts changed')
    print(json.dumps(dict(status='VERIFIED_SECOND_INCREMENT',integration=binding(OUT+'/INTEGRATION.json'),combined_count=3779,changed_slots=impact['changed_slots'],changed_intervals=impact['changed_intervals'],consumed_unique_before=impact['consumed_unique_before'],consumed_unique_after=impact['consumed_unique_after'],writes=0)))


class Tests(unittest.TestCase):
    def row(self,**kw):
        r=dict(event_id='X_2020-01-10',symbol='X',publication_date='2020-01-10',source_url='https://example.test/preliminary',official_actual_period_end='2019-12-31',fiscal_year=2019,fiscal_quarter=4);r.update(kw);return r
    def test_empty(self):self.assertEqual(collision_audit([], [self.row()]),[])
    def test_duplicate_id(self):
        with self.assertRaises(ValueError):collision_audit([self.row()],[self.row(publication_date='2020-02-10')])
    def test_same_day_needs_resolution(self):
        with self.assertRaises(ValueError):collision_audit([self.row()],[self.row(event_id='different',source_url='https://example.test/other')])
    def test_same_source_reject(self):
        with self.assertRaises(ValueError):collision_audit([self.row()],[self.row(event_id='later',publication_date='2020-02-10')])
    def test_later_final_retained_same_quarter(self):
        r=self.row(event_id='X_2020-02-10',publication_date='2020-02-10',source_url='https://example.test/final');a=collision_audit([self.row()],[r]);self.assertEqual(len(a),1);self.assertEqual(a[0]['action'],'retained_distinct_release_not_merged')
    def test_same_period_not_merged(self):
        r=self.row(event_id='X_2020-02-10',publication_date='2020-02-10',source_url='https://example.test/corrected',fiscal_quarter=3);self.assertTrue(collision_audit([self.row()],[r])[0]['same_period_end'])
    def test_other_issuer_allowed(self):self.assertEqual(collision_audit([self.row()],[self.row(event_id='Y_2020-01-10',symbol='Y')]),[])
    def test_missing_new_identity(self):
        r=self.row();del r['fiscal_quarter']
        with self.assertRaises(ValueError):collision_audit([], [r])
    def test_scope_exact_four(self):self.assertEqual(len(EXPECTED),4)
    def test_no_cat_tsla(self):self.assertEqual({r[1] for r in EXPECTED},{'CRM','SAP','HSBC'})
    def test_path_reject(self):
        with self.assertRaises(ValueError):relative('../x')
    def test_external_reject(self):
        with self.assertRaises(ValueError):relative('/tmp/x')

if __name__=='__main__':
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True)
    for n in ['self-test','freeze','build','verify']:m.add_argument('--'+n,action='store_true')
    ap.add_argument('--spec-sha');a=ap.parse_args()
    if a.self_test:
        t=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));sys.exit(not t.wasSuccessful())
    elif a.freeze:freeze()
    elif not a.spec_sha:ap.error('--spec-sha required')
    elif a.build:build(a.spec_sha)
    else:verify(a.spec_sha)
