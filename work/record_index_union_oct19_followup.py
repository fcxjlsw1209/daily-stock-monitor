"""Record bounded source recovery without mutating historical admission records."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/index_union'
OUT=BASE/'sources/oct19_dtcc_symbol_bridge'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
source=json.loads((OUT/'SOURCE_REVIEW.json').read_text())
for p,h in source['protected_inputs_sha256'].items():
    if sha(ROOT/p)!=h:raise ValueError('Historical input changed: '+p)
review={
 'reviewer_task':'/root/index_union_gap_review','mode':'independent_read_only_source_and_digest_review',
 'manifest_files_verified':10,'validator_hash_verified':True,'original_source_receipts_verified':2,
 'dtcc_404_responses_correctly_excluded':True,'claim_scope_approved':True,
 'extra_source_locator':{'signature_date':'2016-11-14','actual_signature_table_raw_html_line':620,'earlier_machine_locator_matches_page_title':True},
 'release_displayed_time':'2014-10-06T19:31:00-04:00',
 'release_displayed_time_is_not_independent_first_publication_timestamp_proof':True,
 'admission_or_strategy_approval':False,'source_review_sha256':sha(OUT/'SOURCE_REVIEW.json'),
 'manifest_sha256':sha(OUT/'MANIFEST.json')}
write(OUT/'INDEPENDENT_REVIEW.json',review)
followup={
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'automation_id':'top100','status':'supplemental_original_sources_recovered_strict_gates_unchanged',
 'sources_root':'sources/oct19_dtcc_symbol_bridge',
 'artifacts':{n:sha(OUT/n) for n in ['SOURCE_REVIEW.json','MANIFEST.json','INDEPENDENT_REVIEW.json']},
 'original_issuer_documents_downloaded':2,'source_integrity_and_literal_checks_passed':20,
 'existing_narrow_regression_tests_passed':8,
 'new_scope':'Recovered original issuer symbol-change announcement and later reporting-person filing with class-specific CUSIPs; 2016 mapping cannot be backdated to2014.',
 'dtcc_pdf_unavailable':True,'dtcc_responses':[404,404],'no_429_observed':True,'sec_requests_this_pass':0,
 'six_membership_disputes_remaining':6,'strict_backtest_admitted':False,
 'qualified_research_already_complete':True,'performance_rerun':False,
 'historical_gate_exclusion_and_fingerprint_files_unchanged':True,
 'observer_runtime_accessed':False,'cloud_storage_accessed':False,
 'remaining':['Contemporaneous identifier interval and complete Liberty corporate-action/price-unit chain.','Six disputed index events and full historical constituent evidence.','Permanent identity, historical sectors and generic cash/entitlement delivery evidence.'],
 'notification_decision':'DONT_NOTIFY','notification_reason':'Only supplemental evidence recovered; admission, performance and required user actions unchanged.',
 'automation_retained':True,
 'prior_source_validator_note':'Old work/validate_liberty_oct17_evidence.py referenced by Oct17 evidence is absent locally. This pass verified only new sources and existing narrow tests, not a rerun of that historical validator.',
 'record_script_sha256':sha(Path(__file__))}
write(BASE/'FOLLOWUP_2026-10-19_DTCC_QVC.json',followup)
print(json.dumps({'followup':'outputs/index_union/FOLLOWUP_2026-10-19_DTCC_QVC.json','sha256':sha(BASE/'FOLLOWUP_2026-10-19_DTCC_QVC.json'),'independent_review':True,'strict_gates_changed':False}))
