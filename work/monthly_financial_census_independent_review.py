"""Read-only verification of the frozen source-lead census; no producer import."""
from collections import Counter,defaultdict
from datetime import date,datetime,timezone
from pathlib import Path
import hashlib,json,re,sys
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
BASE='outputs/monthly_financial_coverage_census_v1/'
PLAN='04a019483b0dbdc967113573c36ec46c90f29e6468fe605318314916bc7345a9'
CENSUS='de95906ef3d752890d28027a9b52cfd7decc7aeb1bc0f09f2785f1826a3500fc'
TARGET=ROOT/'work/monthly_financial_census_independent_review.json'
seen={}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(path,expected=None):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 need(ROOT in p.parents and p.resolve()==p and not p.is_symlink(),'Canonical local path')
 before=p.stat();b=p.read_bytes();after=p.stat()
 need((before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns),'Concurrent mutation')
 h=hashlib.sha256(b).hexdigest();key=str(p.relative_to(ROOT))
 need(expected is None or h==expected,'Hash '+key);need(key not in seen or seen[key]==h,'Changed '+key)
 seen[key]=h;return b
def load(path,h=None):return json.loads(read(path,h))
def parsed(value):
 try:return date.fromisoformat(value)
 except (TypeError,ValueError):return None
need(not TARGET.exists(),'Preserve independent receipt')
plan=load(BASE+'PLAN.json',PLAN);report=load(BASE+'CENSUS.json',CENSUS)
need(report['plan_sha256']==PLAN,'Report PLAN')
for path,h in plan['code'].items():read(path,h)
data={k:load(v['path'],v['sha256']) for k,v in plan['inputs'].items()}
event_sha=plan['inputs']['events']['sha256']
need(data['event_validation']['output_sha256']['EVENTS.json']==event_sha==data['event_scope']['inputs']['EVENTS.json'],'Independent original catalog bindings')
meta=data['universe'];dates=meta['dates'];etfs=set(meta['etfs']);membership=defaultdict(list)
need(len(dates)==122 and dates==sorted(set(dates)) and len(meta['symbols'])==122 and len(etfs)==17,'Exact axes')
for d,names in zip(dates,meta['symbols']):
 need(parsed(d) and len(names)==len(set(names))==49 and etfs.issubset(names),'Each fixed pool')
 for s in names:
  if s not in etfs:membership[s].append(d)
need(len(membership)==73 and sum(map(len,membership.values()))==3904,'Full73/3904')
need(set(membership)==set(data['identity_leads']['candidate_universe']),'Identity locating universe')
events=defaultdict(list)
for e in data['events']:
 if e['symbol'] in membership:events[e['symbol']].append(e)
annual=[e for s in sorted(membership) for e in events[s] if e['fiscal_quarter']==4]
docs=report['annual_documents'];need(len(docs)==len(annual)==920,'Annual census')
patterns={
 'balance_statement':r'\b(?:consolidated\s+)?(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)\b',
 'cashflow_statement':r'\bstatements?\s+of\s+cash\s+flows?\b','total_assets':r'\btotal\s+assets\b',
 'operating_cashflow':r'\bnet\s+cash\b.{0,120}\boperating\s+activities\b','net_income':r'\bnet\s+(?:income|loss)\b',
 'stock_compensation':r'\b(?:stock|share)[ -]based\s+(?:compensation|payments?)\b',
 'annual_period':r'\b(?:(?:twelve|12)\s+months?\s+ended|(?:fiscal\s+)?years?\s+ended)\b'}
cache={};checked={};by_symbol=defaultdict(list)
for e,doc in zip(annual,docs):
 for key in ('event_id','symbol','fiscal_year','fiscal_quarter','publication_date','available_session','source_url','source_kind','first_public_time_verified','evidence_sha256'):
  need(doc[key]==e.get(key),'Event metadata '+key)
 key=(e['evidence_path'],e['evidence_sha256'])
 if key not in cache:
  raw=read(key[0],key[1])
  if raw[:4]==b'%PDF':cache[key]=('pdf',False,{})
  else:
   soup=BeautifulSoup(raw.decode('utf-8',errors='replace'),'html.parser')
   for n in soup.select('script,style,noscript'):n.decompose()
   text=' '.join(soup.get_text(' ',strip=True).split()).lower()
   cache[key]=('html_or_text',True,{k:bool(re.search(p,text)) for k,p in patterns.items()})
 fmt,scanned,labels=cache[key]
 need(doc['source_hash_verified'] is True and (doc['format'],doc['text_scanned'],doc['labels'])==(fmt,scanned,labels),'Source verification and seven exact label predicates')
 kind=e.get('source_kind');claim=isinstance(kind,str) and kind.startswith('issuer_') and not any(x in kind for x in ('excerpt','observation','not_full','notice_of_release_issuance')) and ('public_index' not in kind or 'full_bytes' in kind)
 complete=claim and all(labels.get(k,False) for k in patterns)
 need(doc['catalog_claims_issuer_full_body']==claim and doc['complete_text_label_lead']==complete and doc['financial_cells_admitted_by_census'] is False,'Lead classification')
 need(e['event_id'] not in checked,'Unique event identity');checked[e['event_id']]=doc;by_symbol[e['symbol']].append(e)
expected_months=[]
for s in sorted(membership):
 for d in membership[s]:
  now=parsed(d);prior=[];unknown=0;barrier=False
  for e in by_symbol[s]:
   available=parsed(e.get('available_session'))
   if available is None:
    unknown+=1;published=parsed(e.get('publication_date'));barrier|=published is None or published<=now
   elif available<=now:prior.append(e)
  latest=None
  if barrier:status='unknown_clock_barrier'
  elif not prior:status='no_prior_dated_annual_event'
  else:
   latest_available=max(parsed(e['available_session']) for e in prior)
   finalists=[e for e in prior if parsed(e['available_session'])==latest_available]
   status='ambiguous_latest_annual_event' if len(finalists)>1 else 'latest_selected_before_validity'
   if len(finalists)==1:latest=finalists[0]
  end=parsed(latest.get('official_actual_period_end')) if latest else None
  within=bool(latest and latest['status']=='ready' and end and 0<=(now-end).days<=550)
  doc=checked[latest['event_id']] if latest else {}
  expected_months.append(dict(symbol=s,session=d,latest_annual_date_lead=latest['event_id'] if latest else None,
   latest_status=status,annual_events_with_unknown_clocks=unknown,date_lead_within550=within,
   evidence_source_hash_verified=bool(latest),complete_text_label_lead=within and doc.get('complete_text_label_lead',False),financial_cells_admitted_by_census=False))
need(report['monthly_candidate_coverage']==expected_months,'Every latest-before-validity monthly cell')
prior_by=defaultdict(list)
for e in data['admitted_financial_sources']['events']:
 prior_by[e['symbol']].append(dict(event_id=e['event_id'],available_from=e.get('available_from'),cash_quality_admission=e.get('cash_quality_admission'),
  stock_compensation_admission=e.get('stock_compensation_admission'),fact_admitted=e.get('fact_admitted'),financial_attachment_source=e.get('source')))
need(len(report['stocks'])==73 and [s['symbol'] for s in report['stocks']]==sorted(membership),'Stock ordering')
ciks={x['symbol']:x['candidate_ciks'] for x in data['identity_leads']['candidate_cik_locating_evidence']};facts=defaultdict(list)
for x in data['companyfacts_inventory']['candidates']:facts[x['cik']].append(dict(path=x['relative_path'],sha256=x['sha256'],metadata_only_not_original_statement=True))
for row in report['stocks']:
 s=row['symbol'];dd=[checked[e['event_id']] for e in by_symbol[s]];complete=[d for d in dd if d['complete_text_label_lead']]
 want=dict(candidate_months=len(membership[s]),first_candidate_session=membership[s][0],last_candidate_session=membership[s][-1],
  locating_ciks=ciks.get(s,[]),companyfacts_metadata_leads=[x for cik in ciks.get(s,[]) for x in facts[cik]],event_records=len(events[s]),
  annual_q4_metadata_records=len(dd),annual_evidence_hash_verified=len(dd),catalog_claimed_full_bodies=sum(d['catalog_claims_issuer_full_body'] for d in dd),
  complete_text_label_leads=len(complete),complete_text_label_years=sorted({str(d['fiscal_year']) for d in complete}),
  candidate_months_with_complete_text_lead=sum(m['complete_text_label_lead'] for m in expected_months if m['symbol']==s),previous_admission_records=prior_by[s])
 for k,v in want.items():need(row[k]==v,'Stock total/provenance '+s+' '+k)
counts=dict(stocks=73,monthly_stock_slots=3904,etfs_excluded_from_accounting_census=17,all_event_records=sum(map(len,events.values())),
 annual_q4_metadata_records=920,annual_evidence_hash_verified=920,annual_evidence_read_or_hash_failures=0,
 catalog_claimed_full_bodies=sum(d['catalog_claims_issuer_full_body'] for d in docs),catalog_not_full_body=sum(not d['catalog_claims_issuer_full_body'] for d in docs),
 distinct_verified_primary_evidence_path_hashes=len(cache),stocks_with_complete_text_label_leads=sum(r['complete_text_label_leads']>0 for r in report['stocks']),
 complete_text_label_leads=sum(d['complete_text_label_lead'] for d in docs),monthly_stock_slots_with_complete_text_leads=sum(m['complete_text_label_lead'] for m in expected_months),
 new_financial_admissions=0,new_strategy_trials=0,source_kind_counts=dict(Counter(d['source_kind'] for d in docs)))
need(report['counts']==counts,'All counts exactly reconstructed')
need(sum(len(v) for v in prior_by.values())==44,'Preserve all44 original source records')
prior_summary={s:dict(records=len(rows),cash_quality_admitted=sum(x['cash_quality_admission']['admitted'] for x in rows),
 stock_compensation_admitted=sum(x['stock_compensation_admission']['admitted'] for x in rows)) for s,rows in prior_by.items()}
for path,h in list(seen.items()):read(path,h)
seen['work/monthly_financial_census_independent_review.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt=dict(schema_version=1,passed=True,stage='independent_source_lead_census_review',plan_sha256=PLAN,census_sha256=CENSUS,
 reviewed_utc=datetime.now(timezone.utc).isoformat(),scope='Independent source-only reconstruction: complete candidate axes, primary evidence hashes and seven text predicates, latest annual selection before validity, all aggregate counts, and old admission preservation.',
 counters=counts,checks=dict(fixed73_stocks_3904_slots=True,all920_primary_evidence_hashes_verified=True,all_seven_label_predicates_rechecked=True,
 all3904_latest_event_rows_reconstructed=True,period_end_550_day_age=True,unknown_clock_barrier=True,full_catalog_vs_asof_coverage_separated=True,
 all44_old_admission_records_deep_equal=True,companyfacts_and_cik_remain_locators=True,producer_imported=False),
 preserved_old_admissions=prior_summary,input_hashes=seen,
 limitations=['Source-lead census only, not financial source admission or statement-context verification.',
 '94 is a count of annual-event rows with all seven broad label hits, not 94 independent authenticated financial documents; shared primary evidence may be reused across events.',
 '13 stocks is full-catalog label coverage; 280 is dated candidate-slot coverage. Neither measures investable holdings or strategy efficacy.',
 'AAPL and MSFT have zero all-seven-label primary-body hits, while all10 and12 previous admission records remain preserved; attachment omission/title wording does not revoke old admission.',
 'The catalog full-body flag is a claim, not independently certified HTTP receipt completeness, original publication version, accounting scope, identity, currency or matching annual columns.',
 'No new financial values, ratios, factors, orders, returns or model fits were computed. No frozen files were modified.'],
 financial_values_extracted=0,factor_calculations=0,return_calculations=0,new_admissions=0)
with TARGET.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=True,receipt_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),counters=counts,preserved_old_admissions=prior_summary)))
