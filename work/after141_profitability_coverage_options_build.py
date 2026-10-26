"""Source-only feasibility inventory. No monetary extraction or factor work."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parent.parent
OUT=R/'work/after141_profitability_coverage_options.json'

def bind(path, expected=None):
 p=R/path
 assert p.resolve()==p and p.is_file(),path
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if expected is not None:assert h==expected,path
 return {'path':path,'sha256':h}

def load(path):
 return json.loads((R/path).read_text())

ap='outputs/monthly_sec_apple_original_admission_v1/EVIDENCE.json'
a=load(ap);apple=[]
for release in a['releases'].values():
 if release['publication']['fiscal_quarter']!=4:continue
 current_end=release['parsed']['statement']['current_end']
 matches=[x for x in release['publication']['year_ends'] if x['end']==current_end]
 assert matches
 current=matches[0]
 year=current['fiscal_year'];receipt=release['receipt']
 raw='outputs/monthly_aapl_newsroom_financial_source_v1/'+receipt['response_path']
 pages=raw.removesuffix('.response')+'.pages.txt'
 assert receipt['status']==200 and receipt['decoder']=='identity'
 assert receipt['response_sha256']==receipt['decoded_sha256']
 apple.append({'fiscal_year':year,'fiscal_year_end':current['end'],'release_date':release['publication']['release_date'],
  'raw_pdf':bind(raw,receipt['response_sha256']),
  'receipt':bind(release['receipt_path'],release['receipt_sha256']),
  'parent_release':bind(release['parent_release_path'],release['parent_release_sha256']),
  'parent_receipt':bind(release['parent_receipt_path'],release['parent_receipt_sha256']),
  'existing_text_pages':bind(pages),
  'original_frozen_decoded_path':release['source_path'],'old_decoded_path_exists':(R/release['source_path']).exists(),
  'source_url':release['source_url'],'raw_response_equals_expected_identity_decoded_bytes':True,
  'numerator_label':'Gross margin','denominator_label':'Total assets','duration_label':'Twelve Months Ended',
  'unit':'millions; dollar reporting label, not independently ISO-certified here',
  'statement_context':'Condensed consolidated financial statements; unaudited; current annual column and same-year-end balance sheet',
  'amounts_extracted_by_this_inventory':False})
apple.sort(key=lambda r:r['fiscal_year'])
assert [r['fiscal_year'] for r in apple]==list(range(2016,2026))

amd_base='outputs/monthly_earnings_repair/batch_technology_industrial/AMD/'
md=load(amd_base+'events.json');amd=[]
for event in md:
 if event['fiscal_quarter']!=4:continue
 year=event['fiscal_year']
 if year not in [2015,2016,2017,2018,2020,2021,2022,2023,2024,2025]:continue
 raw=event['evidence_path'];meta=raw.removesuffix('.bin')+'.meta.json';receipt=load(meta)
 assert receipt.get('status',receipt.get('http_status'))==200
 expected=event.get('evidence_sha256',event.get('sha256'))
 raw_binding=bind(raw,expected)
 assert receipt['sha256']==raw_binding['sha256']
 amd.append({'fiscal_year':year,'fiscal_year_end':event['fiscal_period_end_actual'],'release_date':event['release_date'],
  'raw_html':raw_binding,'receipt':bind(meta),'source_url':event['source_url'],
  'numerator_label':'Gross margin' if year<=2018 else 'Gross profit','denominator_label':'Total Assets',
  'duration_label':'Year Ended','annual_amount_column_after_removing_label_currency_and_blanks':3 if year==2022 else 4,
  'annual_column_index_is_not_DOM_td_index':True,
  'unit':'Millions; dollar/$M reporting label, not independently ISO-certified here',
  'statement_context':'Condensed Consolidated Statements of Operations and same-release Condensed Consolidated Balance Sheets; unaudited',
  'first_public_time_verified':False,'publication_precision':event['precision'],
  'amounts_extracted_by_this_inventory':False})
amd.sort(key=lambda r:r['fiscal_year'])
assert len(amd)==10

adbe_base='outputs/monthly_earnings_repair/batch_technology_industrial/ADBE/'
adbe_events=load(adbe_base+'events.json');adbe=[]
for event in adbe_events:
 if event['fiscal_quarter']!=4:continue
 assert event['source_class']=='public_index_observation_of_original_issuer_release_not_full_bytes'
 adbe.append({'fiscal_year':event['fiscal_year'],'release_date':event['release_date'],'fiscal_year_end':event['fiscal_period_end_actual'],
  'index_observation_only':bind(event['evidence_path'],event['evidence_sha256']),'source_url':event['source_url'],
  'original_document_bytes_available_in_reviewed_cache':False})
assert len(adbe)==12

record={
 'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
 'status':'SOURCE_FEASIBILITY_ONLY_NOT_NEW_FACTOR_OR_SOURCE_ADMISSION',
 'scope':{'deeply_reviewed_issuers':['AAPL','AMD','ADBE'],'issuer_limit':3,'all_in_original_73_stock_pool':True,
  'network_requests':0,'SEC_requests':0,'CloudStorage_reads':0,'monetary_values_extracted_into_report':0,
  'factor_ratio_rank_order_return_computations':0,'round141_frozen_files_changed':0,'newly_admitted_fact_pairs':0},
 'unchanged_SEC_stop':{'local_date':'2026-10-19','timezone':'America/Los_Angeles','http_status':403,
  'global_stop':bind('work/SEC_REQUEST_STOP_20261019.json'),
  'source_stop':bind('outputs/monthly_profitability141_sources/STOPPED_2026-10-19.json')},
 'options':[
  {'symbol':'AAPL','status':'LOCAL_ORIGINAL_BYTES_SUPPORT_FUTURE_FIELD_ADMISSION',
   'index':bind(ap,'3ffee0f6fb016f68380e306d3e1f2e289c3b47c0dd1a65451f0bf72cffab1e35'),
   'continuous_current_annual_years':list(range(2016,2026)),'original_annual_documents':10,'sources':apple,
   'verified_by_local_readonly_review':['Ten original response PDFs match both response and identity-decoded SHA in existing receipt.','Ten receipt,parent HTML,parent receipt and text-page files are present and hash-checked.','Every annual PDF text has explicit monetary Gross margin, Twelve Months Ended current/prior annual columns and same-year-end Total assets.'],
   'remaining_gaps':['Prior frozen .decoded paths are absent: a NEW source plan must explicitly bind the existing .response paths and identity equivalence; never silently replace a frozen path.','FY2015 current release has only news/data-summary lead in reviewed evidence; the FY2015 comparison in the FY2016 PDF cannot be used before the 2016 release.','Gross margin/Assets require new field-level parser, locators and independent semantic/period review; previous Revenue admission does not cover these fields.','FY2022/FY2023 old text encodes dollar signs as /dollarsign: independently reconstruct from original PDF.','FY2024/FY2025 include separate non-GAAP reconciliations: restrict extraction to original consolidated statements.','Date-only publication and current-retrieval version limitation remain; require next-session availability and explicit causal identity policy.'],
   'could_expand_future_experiment_coverage':True,'currently_factor_admitted':False},
  {'symbol':'AMD','status':'LOCAL_ORIGINAL_BYTES_SUPPORT_PARTIAL_FUTURE_FIELD_ADMISSION',
   'index':bind(amd_base+'events.json','afe00f761ace64666bf5df44261b085dd2412c6fa1f5f13d980be09ca2b38892'),
   'discovery':bind(amd_base+'DISCOVERY.json','3cde1dfd45ab7eea202a2ea2984b912ef15632a17d04bf66806b00a573a790c1'),
   'pending':bind(amd_base+'PENDING.json','b0e570b2bcd7a750c9899e02d40ff6d91fdeb07530d2770f8b46d817c5208c31'),
   'archive':bind(amd_base+'raw/archive.bin','8adee128c946df8fb971bb5430083d119d7c04b171db3fdef5a6501a8c4f342b'),
   'continuous_explicit_Gross_profit_years':list(range(2020,2026)),
   'continuous_monetary_Gross_margin_semantic_candidate_years':[2015,2016,2017,2018],
   'missing_current_annual_years_in_reviewed_catalog':[2019], 'original_annual_documents':10,'sources':amd,
   'verified_by_local_readonly_review':['Ten raw HTML SHA values match event and HTTP200 receipts.','FY2020-FY2025 explicitly label Gross profit in the annual GAAP statements and same-release Total Assets.','FY2015-FY2018 Gross margin is monetary, with a separate Gross margin % row; it is not the percentage summary.','FY2016-FY2018 also identify GAAP Gross Margin in reconciliations; FY2015 remains a semantic review candidate.'],
   'remaining_gaps':['No FY2019 current annual release found in reviewed AMD events/manifest/raw; keep this gap and do not backfill from the FY2020 comparative column.','Annual current-column DOM layout varies, particularly FY2022; do not reuse a single hard-coded column index.','Need exact annual-start derivation, 52/53-week duration review, release-date/next-session rules and causal identity policy.','FY2025 separately presents discontinued operations relating to the sold ZT Systems manufacturing business: prospectively specify consolidated/current-operations measurement compatibility rather than infer operating improvement.','ISO USD evidence was not independently certified; retain the reported dollar label and verify cross-year currency convention before any growth control.','All pages were retrieved in2026; first-public-version and outside-catalog correction completeness are not certified.'],
   'could_expand_future_experiment_coverage':True,'currently_factor_admitted':False},
  {'symbol':'ADBE','status':'NO_COMPLETE_ANNUAL_ORIGINAL_BYTES_IN_REVIEWED_CACHE',
   'index':bind(adbe_base+'events.json','eb37eb7610e84b09cf2315bcbf0264b988c4db70d84f80a0c02251e1d6c2b301'),
   'continuous_index_only_years':list(range(2014,2026)),'original_annual_documents':0,'sources':adbe,
   'failed_home_response':bind(adbe_base+'raw/home.bin','ac54888c13b085eb13ce41c92ee27db39a8ec8830367c6d15990c1b501862976'),
   'failed_home_receipt':bind(adbe_base+'raw/home.meta.json','77bdcfd778187e8655a021e8a9105c5ba7b055bce5d5544c31a3b2efbc33e236'),
   'failed_home_http_status':403,'failed_home_bytes':398,
   'pending':bind(adbe_base+'PENDING.json','2e17586e5c0617a4ba590ffc0c10a86651d9e56fb133b3076fc8bab13c58cbb9'),
   'remaining_gaps':['All12 annual events are saved public-index observations, not original issuer PDFs. Their SHA identifies the observation JSON, not the official financial document.','The only reviewed raw homepage response is a398-byte403 page.','No ADBE-named complete annual PDF was found in the scoped local output filename inventory. This does not claim the entire machine has no Adobe document.','Requires a separately authorized future source acquisition or identified complete local original; no acquisition performed.'],
   'could_expand_future_experiment_coverage_from_current_original_bytes':False,'currently_factor_admitted':False}
 ],
 'recommended_next_source_step':'If pursuing a future coverage experiment, freeze a separate AAPL/AMD field-extraction plan over these exact20 original annual documents. Start with AAPL FY2016-FY2025 and AMD explicit-GP FY2020-FY2025; review AMD older monetary-Gross-margin years separately. Retain all other original securities and every known gap. This is source preparation only and neither reopens nor reverses round141.',
 'proposed_next_source_preparation_plan':{
  'status':'PROPOSED_SOURCE_ONLY_NOT_EXECUTED_NOT_AN_EXPERIMENT_REGISTRATION',
  'initial_scope':'AAPL FY2016-FY2025 ten already-local original annual PDFs; AMD can follow under a separate HTML field contract.',
  'freeze_before_extraction':'Bind every listed response PDF, receipt, parent HTML/receipt, this inventory, parser and tests, exact local runtime/dependency versions, and a local exchange-session calendar. Register the actual .response path with proof of identity-decoded SHA equality; leave old .decoded bindings unchanged.',
  'PDF_reconstruction':'Reconstruct the selected pages and table geometry directly from pinned original PDFs using the local pinned PDF parser. Existing pages.txt is corroboration only, especially where /dollarsign appears. Retain page number, bounding boxes, row labels, header cells, current annual column index and raw cell text hashes.',
  'numerator':'Exact monetary Gross margin row in the condensed consolidated statements of operations. Select only the Twelve Months Ended current fiscal-year column. Reject percentage margins, revenue-minus-cost reconstruction and non-GAAP reconciliations.',
  'denominator':'Exact Total assets row in the same original PDF consolidated balance sheet for precisely the same fiscal year-end; reject other totals, quarters or mismatched endpoints.',
  'raw_output_fields':['issuer','source event ID','current fiscal year label','period_end','prior fiscal year-end printed in same report','period_start derivation if any','inclusive duration days','publication date and precision','next_session_available_date','displayed Gross margin text','exact decimal gross amount','displayed Total assets text','exact decimal assets amount','separate scale for each field','literal currency evidence','source paths and SHA','cell/page locators','all review gaps'],
  'units':'Require explicit in-millions legend plus monetary dollar evidence; retain scale=1000000 separately, exact Decimal strings and original label. Do not invent ISO USD; perform no GP/Assets division or cross-year sign calculation in source preparation.',
  'periods':'Join current annual headers to the parent release current fiscal-end, not the first date mentioned in prose. For 52/53-week years derive start only as the day after the explicitly printed prior fiscal year-end, mark it derived, and require330-400 inclusive days. Keep same-release comparative values separate and never backfill earlier availability.',
  'availability':'Use the first pinned exchange session strictly later than the issuer-local dated release, preserving date-only precision. Do not inherit the old Revenue/SEC admission first_usable_date without reviewing its different filing dependency. Keep first_public_version_verified=false and current-retrieval limitation.',
  'identity':'Use dated historical anchors already public by the event; any continued AAPL security mapping is a separately declared finite causal assumption. Never use a future annual release to define an earlier identity interval.',
  'failure_policy':'Every expected annual event remains represented; changed hashes, duplicate/ambiguous rows or columns, unsupported unit, nonpositive assets, missing context or dates produce an explicit rejected candidate. No source substitution, no deletion of a pool member, no latest-valid-only fallback.',
  'independent_review':'Produce source-only candidate facts and a separate review receipt for semantics, annual period, units, publication timing and identity. New candidates start unadmitted; do not touch141 DATASET/RECEIPT or strategy state.',
  'synthetic_checks':['Quarter vs annual column exclusion','percentage and non-GAAP row rejection','currency-glyph and unit reconstruction','duplicate row/column conflicts','52/53-week dates and current-vs-comparative years','source hash mutation','strict next-session availability','future comparative append leaves earlier event unchanged','missing-latest event retained'],
  'success_criterion':'Ten exact AAPL current-year GP/Assets candidate pairs with independently reviewable source cells and explicit gaps/assumptions; still no ratios, ranks or returns.'
 },
 'required_before_any_future_factor':['New immutable source/field/availability/identity contract and extraction receipt.','Independent exact-cell/column, annual period, same-end assets and monetary unit verification.','Preserve event-vintage and invalid-latest-event barriers; do not use later comparative columns as earlier disclosures.','Record current-retrieval first-version limitation explicitly.','Separate prospective hypothesis and pre-return registration after source admission; no parameter or sign change selected from round141 results.'],
 'inventory_generator':bind('work/after141_profitability_coverage_options_build.py')}
with OUT.open('x')as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'path':str(OUT.relative_to(R)),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'reviewed_issuers':3,'candidate_original_annual_documents':20,'newly_admitted_fact_pairs':0}))
