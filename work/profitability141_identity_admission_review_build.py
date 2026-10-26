"""Offline identity/filing-metadata review. Never reads companyfacts values or market arrays."""
from pathlib import Path
import collections,datetime,hashlib,html,json,re
R=Path.cwd();out=R/'work/profitability141_identity_admission_review.json'
P=R/'outputs/monthly_profitability141_sources/DATA_PLAN.json';I=R/'work/monthly_profitability141_identity_review.json';C=R/'work/profitability141_migrated_cache_inventory_complete_20261019T202336581108Z.json';M=R/'migration/top100_local_completion_20261019/FULL_LOCAL_MANIFEST_20261019T202326644648Z.json';A=R/'work/monthly_profitability141_local_annual_feasibility.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rel(p):return str(p.relative_to(R))
def reference(p):return {'path':rel(p),'sha256':sha(p)}
plan=read(P);prior=read(I);cache=read(C);manifest=read(M);allowed=set(plan['allowed_ciks']);byname=collections.defaultdict(list);expected={x['path']:x['sha256'] for x in manifest};start='2016-08-17';end='2026-09-24'
for x in manifest:byname[Path(x['path']).name].append(x['path'])
source_checks={}
def source(p):
 rp=rel(p)
 if rp not in source_checks:
  actual=sha(p);check={'path':rp,'sha256':actual,'migration_manifest_hash_matches':expected.get(rp)==actual if rp in expected else None}
  sp=p.with_suffix('.provenance.json')
  if sp.exists():
   s=read(sp);check.update(receipt=reference(sp),original_url=s.get('source'),retrieved_at=s.get('retrieved_at'),original_receipt_hash_matches=s.get('sha256')==actual)
  else:check['original_receipt_hash_matches']=None
  source_checks[rp]=check
 return rp
# Reverify previously located original issuer statements, extracting only issuer/ticker identity strings.
annual_evidence=[]
for x in read(A)['annual_sources']:
 raw=(R/x['raw_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['raw_sha256']
 assert sha(R/x['receipt_path'])==x['receipt_sha256']
 t=raw.decode('utf8',errors='replace');t=re.sub(r'<script\b.*?</script>|<style\b.*?</style>',' ',t,flags=re.S|re.I);t=html.unescape(re.sub('<[^>]+>',' ',t));t=re.sub(r'\s+',' ',t)
 pattern=r'Microsoft \(Nasdaq[^)]*\)' if x['symbol']=='MSFT' else r'NVIDIA(?:\s*[’\']s)?\s*\(\s*NASDAQ\s*:\s*NVDA\s*\)'
 hit=re.search(pattern,t)
 assert hit,(x['raw_path'],'issuer ticker text missing')
 annual_evidence.append({'symbol':x['symbol'],'fiscal_year_label':x['fiscal_year'],'release_date':x['release_date'],'issuer_ticker_literal':hit.group(0),'raw':{'path':x['raw_path'],'sha256':x['raw_sha256']},'receipt':{'path':x['receipt_path'],'sha256':x['receipt_sha256']},'source_url':x['source_url'],'raw_and_receipt_hashes_reverified':True,'prior_provenance_checks':x['provenance_checks']})
meta=read(R/'outputs/monthly_earnings_repair/meta/IDENTITY.json');assert sha(R/meta['evidence_path'])==meta['evidence_sha256']
abbv_policy=next(x for x in read(R/'outputs/monthly_earnings_repair/batch_health_energy_foreign/IDENTITY_POLICY.json') if x['ticker']=='ABBV');abbv_ev=abbv_policy['evidence'][0];assert sha(R/abbv_ev['evidence_path'])==abbv_ev['sha256']
t=html.unescape(re.sub('<[^>]+>',' ',(R/abbv_ev['evidence_path']).read_bytes().decode('utf8',errors='replace')));t=re.sub(r'\s+',' ',t)
assert 'AbbVie will begin trading independently on the New York Stock Exchange today under the symbol "ABBV."' in t
rows=[]
for old in prior['rows']:
 candidates=set(old.get('plan_candidates',[]))&allowed
 if not candidates:continue
 assert len(candidates)==1;cik=next(iter(candidates));e=next(x for x in old['current_registrants'] if x['cik']==cik);p=R/e['evidence'];sub=read(p);assert str(sub['cik']).zfill(10)==cik
 datasets=[(source(p),sub['filings']['recent'])];missing=[];older=[]
 for older_ref in sub['filings']['files']:
  if older_ref['filingFrom']>end or older_ref['filingTo']<'2015-01-01':continue
  ps=byname[older_ref['name']]
  if not ps:missing.append(older_ref);continue
  # All duplicate copies must agree; the archive manifest provides their exact digests.
  assert len({expected[x] for x in ps})==1,(cik,'conflicting historical index copies')
  f=R/ps[0];datasets.append((source(f),read(f)));older.append({'path':rel(f),'declared_range':[older_ref['filingFrom'],older_ref['filingTo']]})
 annual={}
 for src,d in datasets:
  for k,form in enumerate(d.get('form',[])):
   if form!='10-K':continue
   filed=d['filingDate'][k]
   if filed<'2015-01-01' or filed>end:continue
   a={'accession':d['accessionNumber'][k],'filed':filed,'report_date':d.get('reportDate',['']*len(d['form']))[k],'primary_document':d.get('primaryDocument',['']*len(d['form']))[k],'source_index':src}
   if a['accession'] in annual:assert {k:v for k,v in a.items() if k!='source_index'}=={k:v for k,v in annual[a['accession']].items() if k!='source_index'}
   annual[a['accession']]=a
 seq=sorted(annual.values(),key=lambda x:(x['filed'],x['accession']));dates=sorted({x['filed'] for x in seq});pre=[x for x in seq if x['filed']<=start];post=[x for x in seq if x['filed']>start]
 anchors=([pre[-1]] if pre else [])+post
 adates=sorted({x['filed'] for x in anchors});gap=max(((datetime.date.fromisoformat(b)-datetime.date.fromisoformat(a)).days for a,b in zip(adates,adates[1:])),default=None);tail=(datetime.date.fromisoformat(end)-datetime.date.fromisoformat(adates[-1])).days if adates else None
 start_age=(datetime.date.fromisoformat(start)-datetime.date.fromisoformat(pre[-1]['filed'])).days if pre else None
 continuous=bool(pre and start_age<=400 and gap is not None and gap<=400 and tail<=400)
 matches=[x for x in cache['candidates'] if x['cik']==cik]
 cache_rows=[{'path':x['relative_path'],'sha256':x['sha256'],'entity_name':x['entity_name'],'origin_receipt_verified':x['origin_sidecar']['verified_url_hash_time'],'annual_10K_pair_metadata':[{'taxonomy':c['taxonomy'],**c.get('pair_coverage_by_form',{}).get('10-K',{})} for c in x['annual_concept_metadata'] if c.get('pair_coverage_by_form',{}).get('10-K')]} for x in matches]
 symbol=old['symbol'];status='same_cik_annual_index_continuity_only' if continuous else 'historical_index_window_incomplete'
 notes=['SEC current ticker/exchange arrays do not date earlier listing identity. Annual index rows establish filings attributed to this CIK, not original filing cover-page security-class evidence.','A 400-day filing-gap diagnostic is a transparent source-screening heuristic only; it is not a strategy parameter or financial-data admission rule.']
 if symbol=='DIS':status='entity_change_predecessor_bridge_missing';notes+=['Allowed CIK 0001744489 has former names TWDC Holdco 613 Corp/Corp.; indexed history starts in 2018 and 10-K reports start in 2019. Do not map pre-2019 DIS financials to it or import a predecessor CIK outside the frozen allowlist. Exact legal-security succession documents remain unreviewed.']
 if symbol in ['ADBE','KLAC','WMT']:notes+=['Former-name intervals show a name change within the research window; they are SEC metadata intervals, not certified legal effective dates. Same-CIK filings may be followed without creating a predecessor join, but historical stock/class identification still needs original document evidence.']
 if symbol=='BRK.B':notes+=['Current SEC metadata explicitly lists both BRK-B and BRK-A under CIK 0001067983. BRK.B is the internal Class B spelling; do not substitute Class A price/security history. Consolidated issuer facts alone do not certify the Class B listing through the entire window.']
 if symbol in ['GE','T','JNJ','PFE']:notes+=['Same registrant does not imply unchanged consolidated business scope. Any separation, discontinued-operation recast or comparative-period revision must retain its original accession and filing date; no retrospective backfill is admitted.']
 if symbol in ['ACN']:notes+=['Foreign domicile alone is not an ADR mapping. This CIK has 10-K index evidence; do not impose 20-F/ADR assumptions or map another foreign security without source evidence.']
 if symbol=='research_equity_FB':status='explicit_same_class_ticker_bridge_supported';notes+=['Original issuer announcement confirms FB since the 2012 IPO, META from 2022-06-09, unchanged Class A and CUSIP. Retain the frozen internal alias research_equity_FB.']
 if symbol in ['MSFT','NVDA']:status='dated_issuer_ticker_identity_supported';notes+=['Twelve original annual issuer releases for fiscal labels 2015–2026 explicitly pair the issuer with its Nasdaq symbol; support is at these dated observations, not an independently reconstructed daily securities master.']
 if symbol in ['ABT','ABBV']:notes+=['Original Abbott release dated 2013-01-02 confirms the completed separation and new independent ABBV trading. Treat ABT and ABBV as distinct issuers; do not splice pre-separation ABT financial history into ABBV. This pre-window anchor is not full-window stock-class certification.']
 rows.append({'symbol':symbol,'allowed_cik':cik,'current_sec_name':sub['name'],'current_tickers':sub['tickers'],'current_exchanges':sub['exchanges'],'former_names_as_reported':sub['formerNames'],'review_status':status,'full_historical_security_admission':False,'source_preparation_identity_evidence_supported':symbol in ['MSFT','NVDA','research_equity_FB'],'annual_index_screen':{'continuous_registered_cik_diagnostic':continuous,'first_pre_window_anchor':pre[-1] if pre else None,'latest_filing_before_window_end':seq[-1] if seq else None,'annual_filing_count_2015_to_cutoff':len(seq),'report_date_calendar_year_labels':sorted({x['report_date'][:4] for x in seq}),'largest_adjacent_filing_gap_days':gap,'latest_anchor_age_days_at_end':tail,'local_historical_indexes':older,'missing_relevant_declared_indexes':missing,'filings':seq},'current_submissions_source':rel(p),'local_companyfacts_candidates':cache_rows,'notes':notes})
assert len(rows)==len(allowed)==43 and {r['allowed_cik'] for r in rows}==allowed
stable=[r['symbol'] for r in rows if r['source_preparation_identity_evidence_supported']];continuous=[r['symbol'] for r in rows if r['annual_index_screen']['continuous_registered_cik_diagnostic']];neutral=[r['symbol'] for r in rows if not r['source_preparation_identity_evidence_supported']]
report={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'OFFLINE_IDENTITY_EVIDENCE_REVIEW_ONLY_NOT_FACTOR_ADMISSION','scope':{'allowed_ciks_reviewed':43,'original_pool_symbols_unchanged':73,'analysis_window':[start,end],'analysis_window_source':'parent task context','network_requests':0,'CloudStorage_reads':0,'companyfacts_values_read':0,'market_arrays_read':0,'ratios_scores_orders_returns_computed':0,'pool_exclusions':0,'frozen_mutations':0},'input_bindings':{k:reference(p) for k,p in [('data_plan',P),('prior_identity_review',I),('complete_200_file_metadata_inventory',C),('migration_manifest',M),('local_annual_feasibility',A)]},'decisions':{'dated_primary_issuer_ticker_supported_subset':stable,'same_cik_annual_index_continuity_subset':continuous,'must_remain_neutral_pending_historical_security_admission':neutral,'fully_certified_daily_security_histories':[],'financial_facts_or_strategy_candidates_admitted':[],'interpretation':['MSFT/NVDA have repeated dated primary issuer/ticker evidence; research_equity_FB has an explicit original same-Class-A/CUSIP ticker bridge. These are the defensible source-preparation identity subset, not blanket fact/date or strategy admission.','The larger continuous-CIK subset is a separate issuer-filing identity result. It cannot establish historical share classes, exchange transitions or legal-successor security joins from current names alone.','All remaining names stay in the original 73-security pool. An unadmitted auxiliary fact must remain neutral/missing under a future registered rule, never remove the security or substitute another company.','No lower annual-index year coverage is inferred from calendar report-year labels alone: 52/53-week fiscal years can skip a calendar label, as with JNJ.','The 30 original symbols outside the 43-CIK allowlist were not expanded or newly audited. Prior XOM/AVGO gaps remain outside this frozen scope.']},'counts':{'allowed_ciks':43,'dated_primary_supported':len(stable),'same_cik_annual_index_continuity':len(continuous),'neutral_pending_identity':len(neutral),'companyfacts_candidate_files_for_allowed':sum(len(r['local_companyfacts_candidates']) for r in rows),'allowed_ciks_with_companyfacts':sum(bool(r['local_companyfacts_candidates']) for r in rows),'current_and_historical_source_files_checked':len(source_checks)},'focused_original_identity_evidence':{'meta':{'identity_note':reference(R/'outputs/monthly_earnings_repair/meta/IDENTITY.json'),'original':{'path':meta['evidence_path'],'sha256':meta['evidence_sha256']},'raw_hash_reverified':True,'source_url':meta['source_url'],'ticker_effective_date':meta['ticker_effective_date'],'finding':'FB to META from 2022-06-09; same Class A, unchanged CUSIP; issuer announcement says FB had been used since 2012 IPO.','text_basis':'Previously reviewed original PDF text preserved in IDENTITY.json; this step reverified original byte hash.'},'abbott_abbvie':{'note':reference(R/'outputs/monthly_earnings_repair/batch_health_energy_foreign/IDENTITY_POLICY.json'),'original':abbv_ev,'raw_hash_reverified':True,'finding':'Original 2013-01-02 Abbott announcement identifies ABT and states independent NYSE trading of ABBV starts that day. Separate identities, not a rename or predecessor splice.'},'annual_issuer_ticker_observations':annual_evidence},'rows':rows,'source_checks':list(source_checks.values()),'validation':{'exact_allowed_cik_set':True,'all_migration_bound_source_hashes_match':all(x['migration_manifest_hash_matches'] is not False for x in source_checks.values()),'all_available_original_submission_receipt_hashes_match':all(x['original_receipt_hash_matches'] is not False for x in source_checks.values()),'primary_annual_raw_and_receipt_hashes':24,'runtime_guard_sha256':{n:sha(R/n) for n in ['local_run.py','run_local.sh','local.sandbox.sb']}}}
assert report['validation']['all_migration_bound_source_hashes_match'] and report['validation']['all_available_original_submission_receipt_hashes_match']
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'path':rel(out),'sha256':sha(out),'counts':report['counts'],'stable':stable,'continuity':continuous,'missing_indexes':{r['symbol']:[x['name'] for x in r['annual_index_screen']['missing_relevant_declared_indexes']] for r in rows if r['annual_index_screen']['missing_relevant_declared_indexes']}}))
