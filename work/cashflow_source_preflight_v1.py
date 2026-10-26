"""Prospective source-only CFO/NI/assets extraction; no financial arithmetic.

freeze-plan records metadata-only locators. build is a separate explicit command
after that plan is reviewed; missing/ambiguous cells remain gaps, never estimates.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import io
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup
import same_report144_admission as safe
import after141_amd_source_extract as html
import after141_aapl_source_extract as pdf
import profitability141_issuer_statements as legacy

ROOT=Path(__file__).resolve().parents[1]
SELF='work/cashflow_source_preflight_v1.py'
TEST='work/test_cashflow_source_preflight_v1.py'
OUT='outputs/monthly_cashflow_source_preflight_v1'
UNION='outputs/monthly_same_report145_union'
PDF_PLAN='outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
HTML_PLAN='outputs/monthly_msft_nvda_same_report_preflight_v1/PLAN.json'
AMD_DATA='outputs/monthly_amd_comparative_source_v1/CANDIDATES.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
FIELDS=('operating_cash_flow','net_income','total_assets')
SIDES=('current','comparison')
PINS={
 UNION+'/DATA.json':'59af87e1493f01cd2779153be772532905efd857a78b413aa0494cc1eb68ecca',
 UNION+'/AUDIT.json':'fc3b22e2379c6f17ad37ebc4453e6f509ae27c4ee947dbc844894cab38467d9c',
 UNION+'/PLAN.json':'925a0ff2d59ae0676fadb77d6a2f0344bf0748296b189365a6588cbc4ce44ea4',
 UNION+'/MANIFEST.json':'6cfc61431eb5405ace1266774b58b5ec8911dd15aab39c050fa825a4a3135d17',
 PDF_PLAN:'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50',
 HTML_PLAN:'33024f402deb6dbc0c36ea8b2b1219cf6c69036b417a7865b128345e574882b2',
 AMD_DATA:'5ff80e9efddc9b5f1203f0299415065b6218349dc6c69587f845af3f71ffe38e',
 SESSIONS:'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4',
 'work/after141_aapl_source_extract.py':'a4e18ca94928e4924e8e6a40c8863e184c5fc42bf92a5da7cd9213313d496fda',
 'work/after141_amd_source_extract.py':'868cae81055318b9df934f4bfe2849c4971f861c4b480942139c8c7606ae4766',
 'work/profitability141_issuer_statements.py':'d6fea019c3ae155ddb61234b0ccf920092ab642de0a1fce166a24b87d29443de',
 'work/same_report144_admission.py':'afb62d9286327d9849d9c42ca579c4566cbdc3bfdf7bcdc2274bfd8b30fd41c1',
 'work/after145_cashflow_hypothesis.json':'ca3bb044c49e36a319e1f3289d33ee0a19ea92c16f722942af8c29e7cb64353f'}
CF_TITLE=re.compile(r'(?:consolidated\s+statements?\s+of\s+cash\s+flows?|cash\s+flows?\s+statements?)',re.I)
BS_TITLE=re.compile(r'(?:consolidated\s+)?balance\s+sheets?',re.I)
BAD=re.compile(r'non[\s-]*gaap|free\s+cash\s+flow|pro\s+forma|as\s+adjusted',re.I)
NI=re.compile(r'^net\s+(?:income|loss)(?:\s*\((?:loss|income)\))?$',re.I)
CFO=re.compile(r'^(?:net\s+)?cash\s+(?:provided\s+by(?:\s*\(used\s+(?:in|for)\))?|used\s+(?:in|for)(?:\s*\(provided\s+by\))?|generated\s+by|generated\s+from|from)\s+(?:operating\s+activities|operations)$',re.I)
ANNUAL=re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b|\byears?\s+ended\b',re.I)
QUARTER=re.compile(r'\b(?:three|six|nine|3|6|9)\s+months?\s+ended\b',re.I)
ADJUSTMENT=re.compile(r'adjustments?\s+to\s+reconcile\s+(?:net\s+)?(?:income|loss)',re.I)
require=safe.require

def sha(root,path):return safe.hash_file(root,path)
def read(root,path,expected=None):return safe.read(root,path,expected)
def publish(root,path,value):return safe.publish(root,path,value)
def norm(s):return html.norm(s).rstrip(':').strip()
def code_hashes(root):return {p:sha(root,p)for p in (SELF,TEST)}
def expect_sha(value):require(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'Explicit SHA256 required')

def schema():
 return dict(schema_version=1,stage='source_preflight_only',fields=list(FIELDS),sides=list(SIDES),events=44,maximum_cells=264,
  net_income='Exact consolidated net income/loss starting line in formal indirect-method operating cash flow statement, before attribution. No parent/common/continuing-only substitution.',
  extraction='Select unique formal annual row/date locator; ambiguous/missing/invalid units or cells remain null plus explicit gap. Never quarterly subtraction, TTM, sums, ratios, signs or scores.',
  availability='All six cells use current release first strictly later pinned session. Original event/period/identity metadata retained only as existing research assumptions.',
  identity='No new security-master or first-public-version certification.',
  accounting='Every new field requires fresh independent source review. Recast/restatement words create new flags; prior GP/assets clearance never clears cashflow flags.',
  admission='Every event and cell fact_admitted/model_admitted/strategy_admitted false; source audit passed means process/hash verification only.',
  pdf='PDF layout text candidates require same-page annual headers and exact row; build also requires PDFium same-line text/geometry agreement. Independent visual review remains pending.')

def period_specs(e):
 require(set(e['same_report'])==set(SIDES),'Both original periods required')
 p={s:{k:e['same_report'][s][k]for k in ('period_start','period_end','days_inclusive')}for s in SIDES}
 require(safe.day(p['current']['period_end'])<safe.day(e['release_date']),'Release precedes period end')
 for side,x in p.items():
  end=safe.day(x['period_end'])
  if x['period_start']is None:require(side=='comparison' and x['days_inclusive']is None,'Invalid missing period')
  else:require(type(x['days_inclusive'])is int and (end-safe.day(x['period_start'])).days+1==x['days_inclusive']and 330<=x['days_inclusive']<=400,'Annual period metadata mismatch')
 return p

def units(parts):
 found=set(re.findall(r'\b(?:in\s+)?(millions|thousands)\b',' '.join(parts),re.I))
 found={x.lower()for x in found}
 require(len(found)==1,'missing_or_ambiguous_scale')
 return str({'millions':1000000,'thousands':1000}[next(iter(found))])

def label_matches(field,label):
 return bool(CFO.fullmatch(label)if field=='operating_cash_flow'else NI.fullmatch(label)if field=='net_income'else re.fullmatch(r'total\s+assets',label,re.I))

def accounting_flags(text):
 words=sorted({m.lower()for m in re.findall(r'\b(?:recast\w*|restat\w*|reclassif\w*|retrospect\w*|adopt\w*|accounting\s+standards?)\b',text,re.I)})
 flags=[dict(code='new_fields_independent_source_review_required')]
 if words:flags.append(dict(code='cashflow_accounting_change_context_unreviewed',matched_terms=words))
 return flags

def empty_candidates():return {s:{f:[]for f in FIELDS}for s in SIDES}

def html_inventory(raw,periods):
 soup=BeautifulSoup(raw,'html.parser')
 for node in soup(['script','style']):node.decompose()
 candidates=empty_candidates();rejected=[]
 for ti,table in enumerate(soup.find_all('table')):
  try:rows,grid,physical=html.table_grid(table)
  except ValueError:
   rejected.append(dict(table_index=ti,reason='invalid_table_structure'));continue
  labels={ri:[norm(x['text'])for (r,_),x in physical.items()if r==ri and re.search('[A-Za-z]',x['text'])]for ri in range(len(rows))}
  nis=[ri for ri,ls in labels.items()if any(NI.fullmatch(t)for t in ls)]
  asset_rows=[ri for ri,ls in labels.items()if any(re.fullmatch('assets|cash(?: and cash equivalents)?|total assets',t,re.I)for t in ls)]
  starts=nis+asset_rows
  if not starts:continue
  stop=min(starts)
  heads=[x['text']for (ri,_),x in physical.items()if ri<stop and x['text']]
  if table.caption:heads.insert(0,norm(table.caption.get_text(' ',strip=True)))
  external=html.external_context(table)
  # Microsoft div/c-table headings are siblings rather than table cells.
  if not CF_TITLE.search(' '.join(heads+external))and not BS_TITLE.search(' '.join(heads+external)):
   external=[legacy.statement_context(table)]
  context=' '.join(heads+external)
  cf=bool(CF_TITLE.search(context));bs=bool(BS_TITLE.search(context))
  if not(cf or bs)or BAD.search(context):continue
  if cf and bs:
   rejected.append(dict(table_index=ti,reason='ambiguous_statement_scope'));continue
  # Never save body text/amounts in the locator inventory.
  title=(CF_TITLE if cf else BS_TITLE).search(context).group(0)
  try:scale=units(heads+external)
  except ValueError:
   rejected.append(dict(table_index=ti,reason='missing_or_ambiguous_scale'));continue
  ni_scope=(len(nis)==1 and not any(re.match(r'^net\s+(?:income|loss)\b',t,re.I)for ri,ls in labels.items()if ri<nis[0]for t in ls)
   and any(ADJUSTMENT.search(t)for ri,ls in labels.items()if nis[0]<ri for t in ls))
  for field in FIELDS:
   if(field=='total_assets')!=bs:continue
   target_rows=[(ri,t)for ri,ls in labels.items()for t in ls if label_matches(field,t)]
   if field=='net_income'and not ni_scope:
    rejected.append(dict(table_index=ti,field=field,reason='scope_mismatch_or_missing_indirect_start'));continue
   for ri,label in target_rows:
    for side in SIDES:
     wanted=periods[side]['period_end'];columns=[];header=[]
     for col in sorted({c for r,c in grid}):
      pieces=[x['text']for x in html.header_evidence(grid,stop,col)]
      try:column_dates=html.dates(pieces)
      except ValueError:continue
      if column_dates==[wanted]and(bs or(ANNUAL.search(' '.join(pieces))and not QUARTER.search(' '.join(pieces)))):
       columns.append(col);header.extend(pieces)
     if not columns or columns!=list(range(min(columns),max(columns)+1)):continue
     selected=sorted({grid[(ri,c)]['physical_cell']for c in columns if(ri,c)in grid})
     selected=[pi for pi in selected if not re.search('[A-Za-z]',physical[(ri,pi)]['text'])]
     if not selected:continue
     if any(physical[(ri,pi)]['rowspan']!=1 or not set(physical[(ri,pi)]['columns']).issubset(columns) for pi in selected):
      rejected.append(dict(table_index=ti,field=field,side=side,reason='cell_spans_multiple_periods_or_rows'));continue
     if re.search(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b',table.get_text(' ',strip=True),re.I):
      rejected.append(dict(table_index=ti,field=field,side=side,reason='conflicting_currency'));continue
     candidates[side][field].append(dict(format='html',table_index_zero_based=ti,row_index_zero_based=ri,
      physical_cell_indices_zero_based=selected,logical_columns=columns,row_label=label,header_literals=list(dict.fromkeys(header)),
      statement_title=title,scale=scale,currency_literal='$'if '$'in table.get_text()else None,
      period_end=wanted,ni_scope='indirect_method_consolidated_start'if field=='net_income'else None))
 return dict(candidates=candidates,metadata_gaps=rejected,accounting_review_flags=accounting_flags(soup.get_text(' ',strip=True)))

PDF_LABELS={'operating_cash_flow':['Cash generated by operating activities','Net cash provided by operating activities','Net cash provided by (used in) operating activities','Net cash from operating activities'],
 'net_income':['Net income','Net loss','Net income (loss)','Net loss (income)'],'total_assets':['Total assets']}

def pdf_row_match(line,label):
 pattern=r'^\s*'+r'\s*'.join(re.escape(c)for c in label if not c.isspace())+r'\s+(.+)$'
 match=re.match(pattern,line,re.I)
 # Boundary is syntactic only: attribution/prose after a short label is not
 # mistaken for an amount, and the longer income(loss) labels remain distinct.
 return match if match and not re.search('[A-Za-z]',match[1].replace('/dollarsign','')) else None

def pdf_inventory(texts,periods):
 candidates=empty_candidates();gaps=[]
 for page,text in enumerate(texts):
  lines=text.splitlines();head=' '.join(lines[:12]);cf=bool(CF_TITLE.search(head));bs=bool(BS_TITLE.search(head))
  if not(cf or bs)or cf==bs or BAD.search(head):continue
  if 'Apple Inc.'not in text:
   gaps.append(dict(page_1based=page+1,reason='missing_issuer_heading'));continue
  try:
   scale=units(lines[:12]);headers=[]
   for count in (2,3,4):
    try:headers.append(pdf.headers(lines,count))
    except ValueError:pass
   require(len(headers)==1,'ambiguous_PDF_dates');dates,_,h1,h2=headers[0]
   if cf:require(ANNUAL.search(head)and not QUARTER.search(head),'nonannual_PDF_header')
  except ValueError as ex:
   gaps.append(dict(page_1based=page+1,reason=str(ex)));continue
  ni_rows=[i for i,line in enumerate(lines)if any(pdf_row_match(line,label)for label in PDF_LABELS['net_income'])]
  adjustment_rows=[i for i,line in enumerate(lines)if ADJUSTMENT.search(norm(line))]
  ni_scope=len(ni_rows)==1 and any(i>ni_rows[0]for i in adjustment_rows)
  for field in FIELDS:
   if(field=='total_assets')!=bs:continue
   if field=='net_income'and not ni_scope:
    gaps.append(dict(page_1based=page+1,field=field,reason='scope_mismatch_or_missing_indirect_start'));continue
   for i,line in enumerate(lines):
    for label in PDF_LABELS[field]:
     if not pdf_row_match(line,label):continue
     for side in SIDES:
      wanted=periods[side]['period_end']
      if dates.count(wanted)!=1:continue
      candidates[side][field].append(dict(format='pdf',page_1based=page+1,line_1based=i+1,financial_column_index_0based=dates.index(wanted),
       row_label=label,header_literals=[h1,h2],header_dates=dates,statement_title=(CF_TITLE if cf else BS_TITLE).search(head).group(0),
       scale=scale,currency_literal='$'if '$'in text or '/dollarsign'in text else None,period_end=wanted,
       ni_scope='indirect_method_consolidated_start'if field=='net_income'else None))
 return dict(candidates=candidates,metadata_gaps=gaps,accounting_review_flags=accounting_flags(' '.join(texts)))

def configure_pdf(plan):
 for p,h in plan['pdf_runtime_hashes'].items():
  path=Path(p);require(path.is_absolute()and 'CloudStorage'not in p and 'OneDrive'not in p and path.resolve()==path,'Local pinned PDF runtime required')
  require(safe.digest(path.read_bytes())==h,'PDF runtime SHA mismatch')
 directory=plan['pdf_runtime']['bundle_site_packages']
 if directory not in sys.path:sys.path.append(directory)
 from pypdf import PdfReader
 import pypdfium2
 return PdfReader,pypdfium2

def pdf_texts(raw,pdf_plan):
 reader,_=configure_pdf(pdf_plan);doc=reader(io.BytesIO(raw))
 require(1<=len(doc.pages)<=12,'Bounded annual release PDF only')
 return [p.extract_text(extraction_mode='layout')for p in doc.pages]

def inventory(raw,periods,pdf_plan):
 return pdf_inventory(pdf_texts(raw,pdf_plan),periods)if raw.startswith(b'%PDF-')else html_inventory(raw,periods)

def source_metadata(root,events):
 pp=read(root,PDF_PLAN);hp=read(root,HTML_PLAN);ad=read(root,AMD_DATA);lookup={}
 for x in pp['sources']:
  if x['symbol']=='AAPL':lookup[x['raw_path']]=dict(url=x['source_url'],receipt=x['receipt'])
 for x in hp['sources']:lookup[x['raw_path']]=dict(url=x['source_url'],receipt=dict(path=x['receipt_path'],sha256=x['receipt_sha256']),receipt_entry_key=x['receipt_entry_key'])
 for x in ad['rows']:lookup[x['source']['path']]=dict(url=x['source']['url'],receipt=x['source']['receipt'])
 result=[];axis=safe.session_axis(read(root,SESSIONS)['sessions'])
 require(len(events)==44 and len({e['event_id']for e in events})==44,'All44 events required')
 require({s:sum(e['symbol']==s for e in events)for s in ('AAPL','AMD','MSFT','NVDA')}==dict(AAPL=10,AMD=10,MSFT=12,NVDA=12),'Exact issuer census')
 for e in events:
  p=period_specs(e);require(safe.next_session(e['release_date'],axis)==e['available_from'],'Original next-session mismatch')
  source=dict(deepcopy(e['source']),**deepcopy(lookup[e['source']['path']]))
  require(source['url'].startswith('https://'),'Original HTTPS source URL required')
  result.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],release_date=e['release_date'],available_from=e['available_from'],
   periods=p,source=source,identity={k:deepcopy(e[k])for k in ('issuer_id','security_id','identity_valid_from','identity_valid_through')},
   prior_field_admitted=e['admitted'],inherited_rejection_codes=deepcopy(e['rejection_codes']),prior_identity_evidence=deepcopy(e['evidence']['identity_anchors'])))
 return result

def freeze(root):
 require(not safe.local(root,OUT+'/PLAN.json').exists(),'Plan already exists')
 safe.verify_hashes(root,PINS);union=read(root,UNION+'/DATA.json');ua=read(root,UNION+'/AUDIT.json')
 sources=dict(PINS)
 for key,h in ua['source_hashes'].items():
  p=Path(key);key=str(p.relative_to(root))if p.is_absolute()else key
  require(key not in sources or sources[key]==h,'Conflicting source pin');sources[key]=h
 safe.verify_hashes(root,sources)
 tests=read(root,OUT+'/TESTS.json');code=code_hashes(root)
 require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('source_code_hashes')==code,'Current passing synthetic tests required')
 for k in ('failures','errors','real_source_cells_extracted'):require(type(tests.get(k))is int and tests[k]==0,'Clean synthetic test scope')
 pp=read(root,PDF_PLAN);specs=source_metadata(root,union['events'])
 # This phase calls only locator inventory, never monetary parsing/extraction.
 for spec in specs:
  raw=safe.local(root,spec['source']['path']).read_bytes();require(safe.digest(raw)==spec['source']['sha256'],'Source SHA mismatch')
  spec['locator_inventory']=inventory(raw,spec['periods'],pp)
 publish(root,OUT+'/SCHEMA.json',schema())
 for f in ('SCHEMA.json','TESTS.json'):sources[OUT+'/'+f]=sha(root,OUT+'/'+f)
 safe.verify_hashes(root,sources)
 plan=dict(schema_version=1,stage='frozen_cashflow_source_preflight_plan',created_utc=datetime.now(timezone.utc).isoformat(),runtime=sys.version,
  source_hashes=sources,source_code_hashes=code,pdf_runtime=pp['pdf_runtime'],pdf_runtime_hashes=pp['pdf_runtime_hashes'],events=specs,
  expected_events=44,maximum_cells=264,metadata_only_locators=True,financial_calculations_permitted=False,admission_permitted=False,network_requests_permitted=False)
 publish(root,OUT+'/PLAN.json',plan)
 return dict(plan_sha256=sha(root,OUT+'/PLAN.json'),events=44,maximum_cells=264,actual_source_cells_extracted=0)

def extract_html(raw,loc):
 soup=BeautifulSoup(raw,'html.parser');table=soup.find_all('table')[loc['table_index_zero_based']]
 rows,_,physical=html.table_grid(table);ri=loc['row_index_zero_based']
 cells=[physical[(ri,pi)]['text']for pi in loc['physical_cell_indices_zero_based']]
 display=' '.join(x for x in cells if x).strip();value=html.exact_amount(display)
 markups=[str(rows[ri].find_all(['td','th'],recursive=False)[pi])for pi in loc['physical_cell_indices_zero_based']]
 return dict(displayed_amount=display,value_decimal=value,parsed_cell_sha256=[safe.digest(x.encode())for x in markups])

def extract_pdf(raw,texts,loc,pdf_plan):
 page=loc['page_1based']-1;cells,line,_=pdf.amount_row(texts[page].splitlines(),loc['row_label'],len(loc['header_dates']))
 require(line==loc['line_1based'],'PDF line changed');index=loc['financial_column_index_0based']
 _,pdfium=configure_pdf(pdf_plan);doc=pdfium.PdfDocument(raw);pg=doc[page];tp=pg.get_textpage()
 row=dict(measure_label=loc['row_label'],all_row_cells=cells,locator=dict(financial_column_index_0based=index))
 try:pdf.attach_geometry(row,tp)
 finally:tp.close();pg.close();doc.close()
 return dict(displayed_amount=cells[index],value_decimal=pdf.decimal_cell(cells[index]),pdfium_geometry=row['locator'],pdfium_text_agrees=True)

def map_event(spec,raw,pdf_plan):
 current=inventory(raw,spec['periods'],pdf_plan)
 require(current==spec['locator_inventory'],'Frozen metadata locator inventory changed')
 result={k:deepcopy(v)for k,v in spec.items()if k!='locator_inventory'}
 result.update(fields={s:{}for s in SIDES},gaps=[],accounting_review_flags=deepcopy(current['accounting_review_flags']),
  fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
 texts=pdf_texts(raw,pdf_plan)if raw.startswith(b'%PDF-')else None
 for side in SIDES:
  for field in FIELDS:
   locs=current['candidates'][side][field];result['fields'][side][field]=None
   reason='missing_formal_annual_cell'if not locs else'context_or_column_ambiguous'if len(locs)!=1 else None
   if reason is None:
    loc=locs[0]
    try:
     require(loc['currency_literal']=='$','missing_literal_currency')
     value=extract_pdf(raw,texts,loc,pdf_plan)if texts is not None else extract_html(raw,loc)
     result['fields'][side][field]=dict(value,scale=loc['scale'],currency_literal=loc['currency_literal'],currency_iso=None,
      period=deepcopy(spec['periods'][side]),available_from=spec['available_from'],source=deepcopy(spec['source']),locator=deepcopy(loc),fact_admitted=False)
    except (ValueError,IndexError,KeyError)as ex:reason='invalid_raw_cell: '+str(ex)
   if reason is not None:result['gaps'].append(dict(side=side,field=field,reason=reason,candidate_count=len(locs)))
 result['metadata_gaps']=deepcopy(current['metadata_gaps'])
 return result

def products(root,plan_sha):
 expect_sha(plan_sha);plan=read(root,OUT+'/PLAN.json',plan_sha)
 require(plan['runtime']==sys.version and plan['stage']=='frozen_cashflow_source_preflight_plan','Frozen plan/runtime mismatch')
 require(all(plan[k]is False for k in ('financial_calculations_permitted','admission_permitted','network_requests_permitted')),'Source-only plan required')
 require(plan['source_code_hashes']==code_hashes(root),'Frozen source code changed')
 safe.verify_hashes(root,plan['source_hashes']);safe.verify_hashes(root,plan['source_code_hashes'])
 require(len(plan['events'])==44 and plan['maximum_cells']==264,'Fixed full census')
 rows=[]
 for spec in plan['events']:
  raw=safe.local(root,spec['source']['path']).read_bytes();require(safe.digest(raw)==spec['source']['sha256'],'Source changed')
  rows.append(map_event(spec,raw,plan))
 count=sum(e['fields'][s][f]is not None for e in rows for s in SIDES for f in FIELDS)
 data=dict(schema_version=1,stage='raw_source_candidates_not_admitted',plan_sha256=plan_sha,events=rows,
  counts=dict(events=44,requested_cells=264,extracted_cells=count,gap_cells=264-count),fact_admissions=0,model_admissions=0,strategy_admissions=0,
  financial_calculations=0,ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0)
 data.update(model_fits=0,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
 audit=dict(schema_version=1,stage='source_preflight_audit',passed=True,meaning='Input hashes and deterministic source extraction verified; missing cells are explicit gaps, not admission.',
  plan_sha256=plan_sha,data_sha256=safe.digest(safe.encoded(data)),counts=data['counts'],complete_cell_coverage=count==264,
  source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],pdf_runtime_hashes=plan['pdf_runtime_hashes'],
  all44_events_preserved=True,all_rejected_prior_events_preserved=True,independent_new_field_review_pending=True,
  financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,network_requests=0,fact_admissions=0)
 safe.verify_hashes(root,plan['source_hashes']);return data,audit

def build(root,plan_sha,verify=False):
 data,audit=products(root,plan_sha);values={'RAW_FACTS.json':data,'AUDIT.json':audit}
 manifest=dict(plan_sha256=plan_sha,output_hashes={k:safe.digest(safe.encoded(v))for k,v in values.items()});values['MANIFEST.json']=manifest
 for name,value in values.items():
  if verify:require(read(root,OUT+'/'+name)==value,'Reproduced output mismatch')
  else:publish(root,OUT+'/'+name,value)
 return dict(passed=True,verified=verify,counts=data['counts'],output_hashes=manifest['output_hashes'])

def main():
 parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
 for flag in ('freeze-plan','build','verify'):mode.add_argument('--'+flag,action='store_true')
 parser.add_argument('--plan-sha');args=parser.parse_args()
 print(safe.json.dumps(freeze(ROOT)if args.freeze_plan else build(ROOT,args.plan_sha,args.verify),sort_keys=True))
if __name__=='__main__':main()
