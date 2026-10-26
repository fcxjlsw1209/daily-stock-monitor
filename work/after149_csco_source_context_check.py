"""Read-only CSCO statement metadata; never parse or emit financial amounts."""
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
import json, hashlib, re, argparse
from bs4 import BeautifulSoup
import after141_amd_source_extract as tablelib
import cashflow_html_repair_v2 as locator
import stock_comp_source_v1 as sbc
ROOT=Path(__file__).resolve().parents[1]
CENSUS='outputs/monthly_financial_coverage_census_v1/CENSUS.json'
BASE='outputs/monthly_earnings_repair/batch_technology_industrial/CSCO/'
TITLE=re.compile(r'(?:condensed\s+)?consolidated\s+(?:statements?\s+of\s+cash\s+flows?|balance\s+sheets?)',re.I)
LABEL=re.compile(r'^(?:.*(?:share|stock)[ -]based\s+compensation.*|net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?|(?:net\s+)?cash.*(?:operating\s+activities|operations)|total\s+assets|(?:stock|share)[ -]based\s+compensation(?:\s+expense)?|adjustments?\s+to\s+reconcile.*)$',re.I)
BODY=re.compile(r'^(?:net\s+(?:income|loss).*|assets:?|current\s+assets:?|cash(?:\s+and\s+cash\s+equivalents)?|total\s+assets)$',re.I)
HEADER=re.compile(r'^(?:(?:in\s+)?millions(?:,.*)?|unaudited|fiscal\s+years?.*|(?:twelve|12|three|3|six|6|nine|9)\s+months?\s+ended.*|years?\s+ended.*|(?:July|August)\s+\d{1,2},?\s+\d{4}|(?:19|20)\d{2}|\$)$',re.I)
PINS={}
def raw(p):
 b=(ROOT/p).read_bytes();PINS[p]=hashlib.sha256(b).hexdigest();return b
def obj(p):return json.loads(raw(p))
def stable():
 for p,h in PINS.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p

def inspect(doc):
 p=doc['evidence_path'];b=raw(p);assert PINS[p]==doc['evidence_sha256']
 receiptp=p.replace('.bin','.meta.json');receipt=obj(receiptp)
 assert receipt['sha256']==PINS[p] and receipt['status']==200
 soup=BeautifulSoup(b,'html.parser')
 for n in soup(['script','style']):n.decompose()
 tables=[]
 for ti,t in enumerate(soup.find_all('table')):
  try:rows,grid,phys=tablelib.table_grid(t)
  except ValueError:continue
  targets=[{'row':ri,'physical_label_cell':pi,'label':v['text']} for (ri,pi),v in phys.items() if LABEL.fullmatch(v['text'])]
  if not targets:continue
  # Never emit cell contents from a value-bearing row; labels only above.
  starts=[ri for (ri,pi),v in phys.items() if BODY.fullmatch(v['text'])]
  stop=min(starts) if starts else min(x['row'] for x in targets)
  heads=[{'row':ri,'physical_cell':pi,'columns':v['columns'],'text':v['text']} for (ri,pi),v in phys.items() if ri<stop and (HEADER.fullmatch(v['text'].strip('()')) or TITLE.fullmatch(v['text']) or ('millions' in v['text'].lower() and not re.search(r'\d',v['text'])))]
  ext=tablelib.external_context(t)
  formal=[m.group(0) for text in ext+[v['text'] for (ri,_),v in phys.items() if ri<stop] for m in TITLE.finditer(text)]
  dates=tablelib.dates([h['text'] for h in heads])
  columns=[]
  for col in sorted({c for r,c in grid}):
   hv=tablelib.header_evidence(grid,stop,col)
   dates_col=tablelib.dates([x['text'] for x in hv])
   annual=bool(re.search(r'(?:twelve|12)\s+months?\s+ended|years?\s+ended',' '.join(x['text'] for x in hv),re.I))
   quarter=bool(re.search(r'(?:three|3|six|6|nine|9)\s+months?\s+ended',' '.join(x['text'] for x in hv),re.I))
   if dates_col:columns.append(dict(column=col,dates=dates_col,annual=annual,quarter=quarter))
  if formal:
   for target in targets:
    target['physical_row_geometry']=[{'physical_cell':pi,'columns':v['columns'],'rowspan':v['rowspan'],'contains_letters':bool(re.search('[A-Za-z]',v['text'])),'is_empty':not bool(v['text']),'cell_markup_sha256':v['parsed_cell_markup_sha256']} for (ri,pi),v in phys.items() if ri==target['row']]
   tables.append(dict(table=ti,nested_tables=len(t.find_all('table')),title=formal,headers=heads,header_end_row=stop,dates=dates,columns=columns,labels=targets,dollar_literal=('$' in t.get_text()),scale_words=sorted(set(re.findall(r'\b(?:millions|thousands)\b',' '.join(x['text'] for x in heads)+ ' '+' '.join(ext),re.I)))))
 ends=sorted({d for t in tables for d in t['dates']})
 prior=[d for d in ends if d<doc['official_actual_period_end'] and (date.fromisoformat(doc['official_actual_period_end'])-date.fromisoformat(d)).days in range(330,401)]
 period={'current':dict(period_start=None,period_end=doc['official_actual_period_end'],days_inclusive=None),'comparison':dict(period_start=None,period_end=prior[-1] if prior else '1900-01-01',days_inclusive=None)}
 if prior:
  period['current']['period_start']=(date.fromisoformat(prior[-1])+timedelta(days=1)).isoformat();period['current']['days_inclusive']=(date.fromisoformat(period['current']['period_end'])-date.fromisoformat(prior[-1])).days
 meta=locator.inventory(b,period)
 # Private metadata helper deliberately bypasses period-completeness admission;
 # this phase only locates source columns and cannot admit any cell.
 smeta=sbc._html_inventory(b,period)
 return dict(event_id=doc['event_id'],fiscal_year=doc['fiscal_year'],source=dict(path=p,sha256=PINS[p],url=doc['source_url'],receipt_path=receiptp),release_date=doc['publication_date'],available_session=doc['available_session'],periods=period,tables=tables,cashflow_asset_inventory=meta,stock_compensation_inventory=smeta,source_issuer_name_found=bool(re.search(r'CISCO\s+SYSTEMS[,. ]+INC',soup.get_text(' ',strip=True),re.I)),source_listing_reference=next((m.group(0) for m in re.finditer(r'NASDAQ\s*[:(]\s*CSCO',soup.get_text(' ',strip=True),re.I)),None),accounting_context=[{'dom_tag':n.name,'fiscal_year_references':re.findall(r'fiscal\s+(20\d{2})',n.get_text(' ',strip=True),re.I),'standard_references':re.findall(r'(?:ASC|ASU)\s+[0-9-]+',n.get_text(' ',strip=True),re.I),'sha256':hashlib.sha256(str(n).encode()).hexdigest(),'redacted_text':re.sub(r'\d[\d,.]*','[number]',tablelib.norm(n.get_text(' ',strip=True)))[:1200]} for n in soup.find_all(['p','li']) if not n.find('table') and re.search(r'retrospect|reclassif|new accounting standard',n.get_text(' ',strip=True),re.I) and len(n.get_text(' ',strip=True))<2000])

def forbid_amounts(*args, **kwargs):
 raise AssertionError('Monetary extraction is outside the context-audit scope')

def main():
 tablelib.exact_amount=forbid_amounts
 locator.extract=forbid_amounts
 sbc.extract_current=forbid_amounts
 c=obj(CENSUS);events=obj(BASE+'events.json');docs=[x for x in c['annual_documents'] if x['symbol']=='CSCO'];assert len(docs)==16
 records=[inspect(d) for d in docs]
 byend={e['periods']['current']['period_end']:e for e in records}
 for e in records:
  prev=byend.get(e['periods']['comparison']['period_end'])
  if prev:
   e['periods']['comparison']=dict(prev['periods']['current'])
   e['comparison_start_evidence_event']=prev['event_id']
  e['release_metadata']=next(x for x in events if x['release_date']==e['release_date'])
 sub=obj('outputs/dynamic_top100/registry_sec/0000858877_submissions.json')
 ident={k:sub.get(k) for k in ('cik','name','tickers','exchanges','formerNames')}
 ident['current_metadata_not_historical_security_master']=True
 pins=obj('outputs/dynamic_top100/registry_sec/0000858877_submissions.provenance.json')
 ident['receipt']=pins
 for p in (BASE+'DISCOVERY.json',BASE+'PENDING.json',BASE+'JOIN_PENDING.json','outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json', 'work/after149_csco_source_context_check.py','work/cashflow_html_repair_v2.py','work/cashflow_source_preflight_v1.py','work/stock_comp_source_v1.py','work/after141_amd_source_extract.py'):
  raw(p)
 stable()
 result=dict(stage='csco_metadata_context_only',symbol='CSCO',events=records,identity=ident,source_hashes=PINS,financial_amounts_extracted=0,financial_calculations=0,factor_calculations=0,returns_computed=0,fact_admissions=0)
 # Verify source metadata links; financial values remain unparsed.
 cp=obj('outputs/monthly_financial_coverage_census_v1/PLAN.json')
 assert c['plan_sha256']==PINS['outputs/monthly_financial_coverage_census_v1/PLAN.json']
 session_obj=obj('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')
 sessions=session_obj['sessions']
 filing_path='outputs/dynamic_top100/registry_sec/filing_indexes/CIK0000858877-submissions-001.json'
 hist=obj(filing_path)
 hist_prov=obj(filing_path.replace('.json','.provenance.json'))
 assert hist_prov['sha256']==PINS[filing_path]
 assert pins['sha256']==PINS['outputs/dynamic_top100/registry_sec/0000858877_submissions.json']
 facts_meta=obj('outputs/dynamic_top100/registry_sec/0000858877_facts.json')
 assert facts_meta['cik']==858877 and facts_meta['entityName']=='CISCO SYSTEMS, INC.'
 filing_anchors=[]
 for container in (sub['filings']['recent'],hist):
  for i,form in enumerate(container['form']):
   if form=='10-K':
    filing_anchors.append({key:container[key][i] for key in ('form','reportDate','filingDate','accessionNumber','primaryDocument')})
 for e in records:
  assert e['source_issuer_name_found'] and e['source_listing_reference']
  assert date.fromisoformat(e['periods']['current']['period_end'])<date.fromisoformat(e['release_date'])
  assert len(e['tables'])==2 and all(t['nested_tables']==0 for t in e['tables'])
  assert all(t['scale_words']==['millions'] and t['dollar_literal'] for t in e['tables'])
  assert all(e['periods'][side]['period_end'] in t['dates'] for t in e['tables'] for side in ('current','comparison'))
  assert all(len(e['cashflow_asset_inventory']['candidates'][side][field])==1 for side in ('current','comparison') for field in ('operating_cash_flow','net_income'))
  assert len(e['stock_compensation_inventory']['candidates'])==1
  e['calendar_verification']={'source':'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json','status':'outside_pinned_calendar_range_existing_catalog_date_only'}
  if sessions[0]<=e['release_date']<sessions[-1]:
   assert min(d for d in sessions if d>e['release_date'])==e['available_session']
   e['calendar_verification']['status']='first_strictly_later_pinned_session_verified'
  e['sec_annual_filing_anchors']=[x for x in filing_anchors if x['reportDate']==e['periods']['current']['period_end']]
  e['period_basis']='Current annual start is prior fiscal-end +1 day using this report annual columns; comparison start is the predecessor report annual start where available. No day-count estimate or financial value is used.'
  e['context_status']='ready_for_prospective_raw_extraction_with_explicit_context_conditions'
  e['context_conditions']=['All new monetary cells still require a frozen extraction plan, signed-amount parsing and independent physical-cell review.','Literal dollar and millions only; ISO currency and historical first-public version are not newly certified.','Same-current-report columns only; do not splice values from predecessor reports.']
  if e['fiscal_year']<=2018:
   assert e['stock_compensation_inventory']['gaps']
   e['context_conditions'].append('Old SBC parser rejects the separate excess-tax-benefits line. A prospective parser must distinguish the standalone share-based compensation expense line from the separately labelled tax line; never sum or net them.')
  if e['fiscal_year']==2011:
   e['context_conditions'].extend(['Comparison FY2010 annual start remains unknown in this bounded source set; preserve a sentinel and do not assert complete comparative duration.','Old cashflow locator returns no assets although table7 row23 has separate exact-date column groups. A source-only adapter or independent manual locator must explicitly verify physical cells2,3 current and5,6 comparison before extraction.'])
  if e['calendar_verification']['status'].startswith('outside'):
   e['context_conditions'].append('Historical availability before2015 is inherited from the event catalog, not independently validated against this pinned2015+ session calendar.')
  if e['accounting_context']:
   e['context_conditions'].append('Retain report-specific reclassification/adoption notes; context detection is not an accounting-comparability clearance.')
  e['fact_admitted']=False
 result['source_hashes']=PINS
 result['created_utc']=datetime.now(timezone.utc).isoformat()
 result['identity']['annual_filing_scope']='Cached SEC10-K index links CIK and annual reportDate; current submissions metadata plus each original release Nasdaq:CSCO reference are identity leads, not a complete historical security master.'
 result['identity']['annual_filings_count']=len(filing_anchors)
 result['passed']=True
 result['scope']='16 CSCO annual original reports; amount-free DOM table, header, label and provenance context audit only. Existing helpers were used only for inventories; their monetary extraction entrypoints were disabled.'
 result['summary']={'annual_reports':16,'same_report_current_comparison_contexts':16,'standalone_sbc_rows':16,'old_all_field_inventory_without_context_gaps_years':list(range(2019,2027)),'prospective_parser_tax_label_disambiguation_years':list(range(2011,2019)),'asset_locator_adapter_years':[2011],'missing_comparison_start_years':[2011],'pinned_calendar_verified_years':list(range(2015,2027)),'pre2015_calendar_validation_pending_years':list(range(2011,2015)),'new_financial_admissions':0}
 result['accounting_findings']=[
  {'years':list(range(2011,2019)),'finding':'Standalone Share-based compensation expense is followed by separately labelled Excess tax benefits from share-based compensation inside the operating reconciliation and a similarly named financing line. The SBC addback is not the tax line. Any CFO-SBC transformation remains a separate, unregistered accounting/model decision.'},
  {'years':[2011,2012,2014,2015,2016],'finding':'Source explicitly states prior-period amounts were reclassified to conform to current presentation. Retain flags; no cross-report numeric equality or revised-value inference was performed.'},
  {'years':[2018],'finding':'Revenue-recognition adoption appears in the forward-looking guidance context; fiscal-year and ASC references are retained in per-event redacted notes. Do not treat the mere word adoption as proof that the just-ended annual CFO/assets use a changed basis.'},
  {'years':[2019],'finding':'Source states prior-period information retrospectively adjusted for Restricted Cash cash-flow standard adoption. Preserve scope flag for comparative flows; no conclusion that CFO or total assets changed numerically is made.'}]
 result['next_step']='Freeze a separate metadata-bound raw extraction plan. FY2019–2026 have immediate unchanged-inventory paths. FY2012–2018 need explicit separate-tax-label handling; FY2011 additionally needs an asset-row locator adapter and retained missing comparative-start sentinel. All16 may be listed with gaps in the prospective source census; none are fact/model/strategy admitted by this review.'
 stable()
 output=ROOT/'work/after149_csco_source_context.json'
 with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'passed':True,'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'summary':result['summary'],'amounts_extracted':0},indent=2))

if __name__=='__main__':main()
