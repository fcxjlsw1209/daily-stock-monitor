"""Source collection receipt; no strategy or accounting admission."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name): return json.loads((OUT/name).read_text())
pages=load('FY2019.pages.json')
receipt=load('FY2019.receipt.json')
publisher=load('publisher.receipt.json')
assert receipt['http_status']==200 and receipt['raw_sha256']==sha(OUT/'FY2019.response.pdf')
assert publisher['http_status']==403
assert pages['source_sha256']==receipt['raw_sha256'] and pages['page_count']==101
pmap={p['page_1based']:p for p in pages['pages']}
def locate(page, start, end, interpretation):
    p=pmap[page]; t=p['text']; a=t.index(start); b=t.index(end,a)+len(end); excerpt=t[a:b]
    return dict(pdf_page_1based=page, printed_page=page-3, source_page_text_sha256=p['text_sha256'],
                character_start=a, character_end_exclusive=b, excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
                paraphrase=interpretation, exact_text_in='FY2019.pages.json')
findings=[
 locate(37,'In the first quarter of 2019, the Company adopted the Financial','impact on previously reported amounts.',
        'The report says Topic 606 was adopted in FY2019 Q1 using full retrospective transition, and prior total net sales were not restated because the standard had no material impact on previously reported amounts. No-material-impact is not a claim of exact zero impact.'),
 locate(37,'Additionally, beginning in the first quarter of 2019','reclassified to conform to the 2019 presentation.',
        'Amortization of the bundled Maps, Siri and free iCloud deferred value moved from Products to Services; the report identifies FY2018 and FY2017 component reclassification. This explains the components/presentation issue, while total-revenue edge admission still needs independent scope and provenance review.'),
 locate(37,'The Company’s fiscal year is the 52- or 53-week period','the associated quarters, months and periods of those fiscal years.',
        'FY2019 and FY2018 had 52 weeks, FY2017 53, including a 14th week in FY2017 Q1. This is an explicit duration statement in a later report, not evidence available retrospectively in FY2017.'),
 locate(39,'Note 2 – Revenue Recognition','services, as applicable.',
        'Note 2 describes control-based recognition, usually shipment for products and delivery over time for services.'),
 locate(40,'For the sale of third-party products','it retains.',
        'The policy uses gross presentation when Apple controls third-party products before transfer, and net commission presentation for specified app/digital-content store transactions where Apple does not obtain control. This is the FY2019 policy; its continuity with FY2018 has not been certified.')
]
paths=[p for p in OUT.iterdir() if p.is_file() and p.name!='COLLECTION.json']
paths += [ROOT/'work/after156_aapl_scope_review.json',ROOT/'outputs/monthly_aapl_annual_notes156/PLAN.json',ROOT/'outputs/monthly_aapl_annual_notes156/COLLECTION.json',ROOT/'outputs/monthly_aapl_annual_notes156/FY2019.receipt.json',ROOT/'outputs/dynamic_top100/registry_sec/0000320193_submissions.json',ROOT/'outputs/monthly_earnings_repair/apple/raw/02a47e6c4985a8a4.html']
inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
assert inputs['work/after156_aapl_scope_review.json']=='53a6e3af3911a3095b9e1ded7365f1a12939deb4f9c7da1e858ee06eb50f18e8'
result=dict(schema_version='after156_apple_2019_source_collection_v1',created_utc=datetime.now(timezone.utc).isoformat(),
 status='PDF_COLLECTED_CONTENT_DIAGNOSED_PUBLISHER_CHAIN_PENDING_NO_ADMISSION',
 input_hashes=inputs, direct_request_count=2, network_retries=0, redirects_followed=0, sec_requests=0,
 public_search_queries=3, search_results_used_for='Public URL discovery only; no search text establishes raw-byte provenance or accounting admission.',
 target=receipt, publisher_attempt=publisher,
 source_identity=dict(form='10-K',issuer='Apple Inc.',fiscal_year_end='2019-09-28',cover_pdf_page=1,cover_content_verified=True,
                      catalog_accession='0000320193-19-000119',catalog_identity_consistent=True,publisher_chain_verified=False,initial_publication_bytes_verified=False),
 publisher_chain=dict(local_official_link=dict(path='outputs/monthly_earnings_repair/apple/raw/02a47e6c4985a8a4.html',line_1based=580,target='http://investor.apple.com/'),
                      target_cdn='s2.q4cdn.com/470004039',exact_official_page_link_to_this_pdf_verified=False,
                      gap='Current official investor page returned 403. Public discoverability and matching document identity do not replace an exact publisher-link or equivalent independently verified provenance chain.'),
 stop_policy=dict(stopped_hosts=['www.sec.gov','investor.apple.com'],cdn_stopped=False,
                  rule='No more requests to the stopped hosts during 2026-10-20 America/Los_Angeles; no alternate identity or bypass. The supplied independent public CDN was separately authorized.'),
 source_inspection=dict(offline_wrapper=True,extracted_pages=101,visually_checked_pdf_pages=[1,22,37,39,40],
                        extraction_warnings=['pypdf emitted undefined object 0 0/cache-overwrite warnings; extraction completed with exit 0. Relevant source pages were independently rendered with PDFium and visually checked.'],
                        findings=findings),
 time_limitations=dict(cdn_last_modified_header='Thu, 31 Oct 2019 14:41:45 GMT',header_is_historical_availability_certificate=False,
                       catalog_acceptance_string='2019-10-31T02:12:36.000Z',catalog_acceptance_independently_verified=False,
                       rule='Bind future note availability separately; do not use retrieval date or unverified catalog strings as certified first-public history.'),
 diagnostic_effect=dict(prior_note='work/after156_aapl_scope_review.json',prior_note_preserved=True,
                        new_evidence='The newly obtained FY2019 report explicitly addresses the prior missing Topic 606 and component-reclassification statements.',
                        full_accounting_edge_admission=False,all_nine_edges_still_unadmitted=True,
                        remaining=['Exact publisher provenance chain for the collected PDF.','Independent original-note review and policy decision for reported totals, materiality, currency and 52/53-week duration.','FY2018 and other missing original notes where required to compare adjacent bases.','A future preregistered source scope with all note clocks; no retroactive round156 change.']),
 activity_counts=dict(new_financial_factors=0,new_orders=0,new_returns=0,new_models=0,new_strategy_trials=0,accounting_edges_admitted=0,round156_artifacts_changed=0,observer_writes=0))
with (OUT/'COLLECTION.json').open('x') as stream:
    json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
for rel,h in inputs.items(): assert sha(ROOT/rel)==h,rel
print(json.dumps(dict(path=str((OUT/'COLLECTION.json').relative_to(ROOT)),sha256=sha(OUT/'COLLECTION.json'),status=result['status'],input_files=len(inputs)),indent=2))
