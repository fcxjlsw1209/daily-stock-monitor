"""Bounded original-report annual basic EPS denominator extraction, no factors."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup
import after158_basic_shares_inventory as inv
import after141_amd_source_extract as hd
import after141_aapl_source_extract as pd

ROOT=Path(__file__).resolve().parents[1]
SELF='work/after158_basic_shares_extract.py'
TEST='work/after158_basic_shares_extract_tests.py'
CONTRACT='work/after158_basic_shares_extraction_contract.json'
OUT='outputs/after158_basic_shares_extraction_v1'
PLAN=inv.PLAN
INVENTORY=inv.OUT+'/INVENTORY.json'
PDF_PLAN='outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'
PINS={PLAN:'3ff000c9a1b9c17d04d2a2edf2f79f21b18811f4e32263b38b768ace805ca60b',INVENTORY:'3620ca72d412624c5cc7f0167d9988292f5fb01f66a618d45f385dbcfd379121',inv.OUT+'/MANIFEST.json':'b27b3d6c4c49615bb3d65d6a9ea4f13147b7cd4c7dec27b0ce61a52bda8683ca',inv.OUT+'/PDF_TEXT_ADDENDUM.json':'120addc442c4d8ffe9271ada884064f79f1cb8fb0d048e1ea09ee6bfc5326af6',PDF_PLAN:'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50'}
SIDES=('current','comparison')
require=inv.require
norm=inv.norm
sha=inv.sha
read=inv.read
verify=inv.verify
write_once=inv.write_once
UNIT=re.compile(r'\b(thousands|millions|billions)\b',re.I)
BAD=re.compile(r'non[\s\-]*gaap|pro\s*forma|reconciliation|as\s+adjusted',re.I)
SPLIT=re.compile(r'\b(?:stock[\s-]+split|share[\s-]+split|split[\s-]+adjust\w*|reverse[\s-]+split)\b',re.I)


def rules():
 return dict(scope='All 44 original events, original order and bytes; no substitutes, later filings or outcome selection.',field='Current and same-report prior annual BASIC earnings-per-share denominator.',semantics='Explicit weighted-average basic-share wording is distinct from implicit weighted-average meaning of Basic below Shares used in computing EPS; the latter is retained with literal semantics, not certified as explicit weighted-average wording.',annual='Exact original current/comparative dates under Year Ended or Twelve Months Ended; reject interim-only or ambiguous mapping. Do not annualize 52/53-week denominators.',units='Only exact original share units or unambiguous formal table scale, with any share-specific exception taking precedence. Never use monetary scale when shares have an exception. Unknown/conflicting units remain missing.',amounts='Parse original nonnegative decimal share counts, preserve genuine zero; unit normalization only. No NI/EPS reverse calculation, diluted/outstanding substitution, ratio, change, rank, return or imputation.',html='Use every frozen candidate and revalidate original table, row label, column header and formal statement context. Deduplicate logical colspan columns by physical cell. Never select a candidate by its number.',pdf='Reconstruct original page layout and separately verify complete Basic row and date geometry with PDFium. Retain the annual right-hand two columns of the four-column Apple operations statement only after exact dates and duration mapping.',split='Keep original same-report split and retrospective evidence separately; do not adjust numbers or equate no keyword with certified absence. Prior holdings, identity and timing holds remain unchanged.',failures='Keep all 44 events with null cells and explicit reasons on unknown/ambiguous input. A parsed raw pair is not new source/strategy admission.',limits={'max_bytes':33554432,'max_tables':2000,'max_rows_per_table':4000,'max_span':100,'max_pdf_pages':12},activity={'factors':0,'scores':0,'returns':0,'orders':0})


def implementation_hashes():
 paths=[SELF,TEST,inv.SELF,'work/after141_amd_source_extract.py','work/after141_aapl_source_extract.py','work/cashflow_pdf_repair_v4.py','work/cashflow_source_preflight_v1.py','work/same_report144_admission.py','work/profitability141_issuer_statements.py','AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb']
 result={p:sha(ROOT/p)for p in paths}
 import bs4,soupsieve
 for m in(bs4,soupsieve):
  for p in sorted(Path(m.__file__).parent.rglob('*.py')):result[str(p.resolve())]=sha(p)
 return result


def freeze():
 verify(PINS);plan=read(PLAN);verify(plan['source_hashes']);pdf=read(PDF_PLAN);verify(pdf['pdf_runtime_hashes'])
 require(len(plan['events'])==44,'exact_44_required')
 doc=dict(schema='after158_basic_shares_extraction_contract_v1',created_utc=datetime.now(timezone.utc).isoformat(),rules=rules(),events=deepcopy(plan['events']),input_hashes={**PINS,**plan['source_hashes']},pdf_runtime=pdf['pdf_runtime'],pdf_runtime_hashes=pdf['pdf_runtime_hashes'],financial_values_parsed_under_this_contract=False,source_admission_changed=False,implementation_policy='Execution manifest pins final implementation and dependencies; this contract freezes source scope and extraction rules before parsing values.')
 write_once(CONTRACT,doc);print(json.dumps({'contract':CONTRACT,'sha256':sha(ROOT/CONTRACT),'events':44}))


def amount(text):
 t=norm(text)
 require(re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',t) is not None,'ambiguous_or_non_share_amount')
 d=Decimal(t.replace(',',''));require(d.is_finite() and d>=0,'invalid_share_count');return format(d,'f')


def unit_scale(evidence):
 """Original local table/row scales, with explicit share exception precedence."""
 found=[]
 for e in evidence:
  text=norm(e['text']);low=text.lower();units=UNIT.findall(low)
  if not units:continue
  explicit=re.search(r'(?:number\s+of\s+shares|shares)(?:\s*,?\s*which\s+are)?\s+(?:reflected\s+)?in\s+(thousands|millions|billions)',low)
  if explicit:found.append((2,explicit[1],e));continue
  if re.search(r'\bshares\b.*?\(in\s+(thousands|millions|billions)\)',low):
   found.append((2,units[-1],e));continue
  if re.search(r'\bin\s+(?:thousands|millions|billions)\b',low) or re.search(r'(?:\(\s*)?(?:amounts\s+in\s+)?(?:thousands|millions|billions)\s*,?\s*except',low):
   # A different unspecified share exception makes the general scale unusable.
   if re.search(r'except[^)]*(?:number\s+of\s+shares|share\s+counts)',low):continue
   if len(set(units))==1:found.append((1,units[0],e))
 require(found,'explicit_share_units_missing')
 priority=max(x[0]for x in found);best=[x for x in found if x[0]==priority]
 require(len({x[1]for x in best})==1,'conflicting_share_units')
 u=best[0][1];return {'scale':{'thousands':1000,'millions':1000000,'billions':1000000000}[u],'unit':u,'evidence':[x[2]for x in best],'share_specific':priority==2}


def cell(value,event,side,semantic,units,locator):
 d=amount(value)
 return dict(displayed_value=norm(value),value_decimal=d,value_in_shares=format(Decimal(d)*units['scale'],'f'),unit=deepcopy(units),period_end=event['periods'][side]['period_end'],semantic_basis=semantic,weighted_average_explicit=semantic=='explicit_basic_weighted_average_shares',locator=locator,source_sha256=event['source']['sha256'],source_admission_changed=False)


def split_evidence(parts):
 found=[]
 for loc,text in parts:
  text=norm(text)
  for match in SPLIT.finditer(text):
   # Preserve the paragraph including following retroactive-basis sentences.
   snippet=text if len(text)<=1400 else text[max(0,match.start()-450):match.end()+850]
   record={'locator':loc,'literal':snippet}
   if record not in found:found.append(record)
 return dict(status='same_report_split_text_present_review_scope_required'if found else'no_split_keyword_found_not_certified_absence',evidence=found,automatic_adjustment_applied=False,complete_split_history_certified=False)


def local_context(table,rows,ri):
 result=[]
 cap=table.find('caption',recursive=False)
 if cap:result.append({'locator':{'caption':True},'text':norm(cap.get_text(' ',strip=True))})
 # Headers before first reported revenue row only; prevents unrelated later tables.
 for j,cells in enumerate(rows[:ri]):
  text=norm(' '.join(c['text']for c in cells))
  if re.match(r'^(?:net\s+|total\s+)?(?:revenue|sales)',text,re.I):break
  if text:result.append({'locator':{'row_index':j},'text':text})
 # Nearest bounded preceding non-table blocks supports captions outside a table.
 seen=set()
 for step,node in enumerate(table.previous_elements):
  if step>=150:break
  if not getattr(node,'name',None):continue
  if node.name=='table' or node.find_parent('table')is not None:break
  if node.name not in ('p','div','h1','h2','h3','h4','strong','b')or node.find('table'):continue
  text=norm(node.get_text(' ',strip=True))
  if 0<len(text)<=300 and text not in seen:
   result.append({'locator':{'preceding_table_context':True,'tag':node.name,'source_line_1based':node.sourceline},'text':text});seen.add(text)
  if len(seen)>=8:break
 return result


def headers_for(grid,ri,col,event):
 years={event['periods'][s]['period_end'][:4]for s in SIDES}
 seen=set();out=[]
 for j in range(ri):
  c=grid.get((j,col))
  if c and (c['row'],c['physical'])not in seen and inv.header_like(c['text'],years):
   out.append({'row_index':c['row'],'physical_cell':c['physical'],'text':c['text']});seen.add((c['row'],c['physical']))
 return out


def validate_annual(heads,wanted):
 text=' '.join(x['text']for x in heads)
 require(inv.ANNUAL.search(text) and not inv.INTERIM.search(text),'annual_column_ambiguous_or_interim')
 parsed=hd.dates([x['text']for x in heads])
 # A split month/day row plus year row is supported by joined header text.
 require(parsed==[wanted],'exact_annual_date_header_mismatch')


def html_extract(raw,event,entry):
 soup=BeautifulSoup(raw,'html.parser')
 for n in soup(['script','style']):n.decompose()
 tables=soup.find_all('table');require(len(tables)<=2000,'table_bound')
 candidates=entry['inventory']['candidates'];require(len(candidates)==1,'unique_frozen_basic_candidate_required')
 c=candidates[0];ti=c['table_index'];ri=c['row_index'];table=tables[ti];_,grid,rows=inv.table_grid(table)
 labels=[norm(' '.join(x['text']for x in rr if re.search('[A-Za-z]',x['text'])))for rr in rows]
 semantic=inv.label_type(labels[ri],labels[:ri]);require(semantic==c['semantic_kind'],'basic_semantics_changed')
 ctx=local_context(table,rows,ri);inside=[x for x in ctx if not x['locator'].get('preceding_table_context')]
 # A formal title in the table/caption dominates unrelated preceding narrative.
 formal=[x for x in inside if inv.TITLE.search(x['text'])]
 if not formal:formal=[x for x in ctx if inv.TITLE.search(x['text'])]
 require(formal and not any(BAD.search(x['text'])for x in formal),'formal_gaap_statement_title_required')
 unit_context=inside if any(UNIT.search(x['text'])for x in inside)else ctx
 units=unit_scale(unit_context)
 result={};selected=[]
 for side in SIDES:
  locs=c['side_locators'][side];phys={(ri,x['physical_cell'])for x in locs};require(len(phys)==1,'unique_physical_annual_cell_required')
  pi=next(iter(phys))[1];source=rows[ri][pi];head_options=[]
  for item in locs:
   col=item['logical_column'];require(grid[(ri,col)]is source,'physical_logical_mapping_mismatch');heads=headers_for(grid,ri,col,event)
   validate_annual(heads,event['periods'][side]['period_end']);head_options.append({'logical_column':col,'headers':heads})
  selected.append(pi)
  raw_row=table.find_all('tr')[ri];raw_cell=raw_row.find_all(['td','th'],recursive=False)[pi]
  locator=dict(format='html',table_index=ti,row_index=ri,physical_cell=pi,source_line_1based=raw_cell.sourceline,logical_columns=[x['logical_column']for x in locs],row_label=labels[ri],header_evidence=head_options,statement_context=formal,parsed_cell_sha256=hashlib.sha256(str(raw_cell).encode()).hexdigest(),parsed_table_sha256=hashlib.sha256(str(table).encode()).hexdigest())
  result[side]=cell(source['text'],event,side,semantic,units,locator)
 require(len(set(selected))==2,'current_prior_share_same_cell')
 parts=[]
 for j,n in enumerate(soup.find_all(['p','div','li','td','caption'])):
  if n.find(['p','div','li','td','table']):continue
  parts.append(({'dom_leaf_index':j,'tag':n.name,'source_line_1based':n.sourceline},n.get_text(' ',strip=True)))
 return result,split_evidence(parts)


def pdf_extract(raw,event,contract):
 path=contract['pdf_runtime']['bundle_site_packages']
 if path not in sys.path:sys.path.append(path)
 from pypdf import PdfReader
 import pypdfium2 as pdfium
 import cashflow_pdf_repair_v4 as geo
 require(raw.startswith(b'%PDF-'),'original_pdf_required')
 doc=PdfReader(io.BytesIO(raw));require(1<=len(doc.pages)<=12,'pdf_page_bound')
 texts=[p.extract_text(extraction_mode='layout')for p in doc.pages];lines=texts[0].splitlines();head=norm(' '.join(lines[:12]));compact=geo.compact(head)
 require('appleinc.'in compact and 'consolidatedstatementsofoperations'in compact,'formal_apple_statement_required')
 require(not BAD.search(head),'non_gaap_statement')
 require('threemonthsended'in compact and 'twelvemonthsended'in compact and compact.index('threemonthsended')<compact.index('twelvemonthsended'),'duration_order_ambiguous')
 dates,hl,h1,h2=pd.headers(lines,4)
 require(dates[0]==dates[2]==event['periods']['current']['period_end']and dates[1]==dates[3]==event['periods']['comparison']['period_end'],'pdf_date_mismatch')
 headings=[i for i,t in enumerate(lines)if re.search(r'shares\s+used\s+in\s+computing\s+earnings\s+per\s+share',t,re.I)]
 require(len(headings)==1,'unique_share_denominator_heading_required');start=headings[0]
 values,rel,line=pd.amount_row(lines[start+1:],'Basic',4);line_no=start+1+rel
 units=unit_scale([{'locator':{'page_1based':1,'line_1based':i+1},'text':t}for i,t in enumerate(lines[:8])]);result={}
 d=pdfium.PdfDocument(raw);p=d[0];tp=p.get_textpage()
 try:
  rows=geo.physical_rows(tp);hh=[r for r in rows if 'sharesusedincomputingearningspershare'in geo.compact(geo.row_text(r))]
  require(len(hh)==1,'unique_geometry_share_heading_required');hy=geo.bbox(hh[0])[1]
  # Remove only the earlier EPS-value Basic row; all headers remain for geometry.
  rows=[r for r in rows if not(geo.row_text(r).lower().startswith('basic')and geo.bbox(r)[1]>hy)]
  for side,col in zip(SIDES,(2,3)):
   loc={'row_label':'Basic','header_dates':dates,'header_literals':[h1,h2],'financial_column_index_0based':col}
   geometry=geo.geometry_rows(rows,loc,values);geometry.pop('pdfium_all_row_cells')
   loc.update(format='pdf',page_1based=1,line_1based=line_no,share_heading_line_1based=start+1,share_heading_literal=norm(lines[start]),unit_scope='original_operations_statement',duration_literal='Twelve Months Ended',row_sha256=hashlib.sha256(line.encode()).hexdigest(),geometry=geometry,independent_PDFium_text_match=True)
   result[side]=cell(values[col],event,side,'basic_eps_denominator_weighting_implicit',units,loc)
 finally:tp.close();p.close();d.close()
 return result,split_evidence([({'page_1based':i+1,'line_1based':j+1},t)for i,text in enumerate(texts)for j,t in enumerate(text.splitlines())])


def run(contract_sha,write=False):
 require(re.fullmatch('[0-9a-f]{64}',contract_sha or''),'explicit_contract_sha_required');require(sha(ROOT/CONTRACT)==contract_sha,'contract_hash_mismatch');contract=read(CONTRACT)
 require(contract['rules']==rules(),'rules_changed');verify(contract['input_hashes']);verify(contract['pdf_runtime_hashes']);code=implementation_hashes();inventory=read(INVENTORY)
 require([e['event_id']for e in contract['events']]==[e['event_id']for e in inventory['events']],'original_order_changed')
 out=[]
 for e,i in zip(contract['events'],inventory['events']):
  raw=(ROOT/e['source']['path']).read_bytes();require(len(raw)<=33554432 and hashlib.sha256(raw).hexdigest()==e['source']['sha256'],'raw_source_changed')
  record={k:deepcopy(e[k])for k in('original_index','event_id','symbol','fiscal_year','source','release_date','available_from','identity','periods')}
  record.update(inherited_holds=deepcopy(i['inherited_holds']),cells={s:None for s in SIDES},failures=[],source_admission_changed=False,fact_admitted=False,model_admitted=False,strategy_admitted=False)
  try:
   cells,split=pdf_extract(raw,e,contract)if e['format']=='pdf'else html_extract(raw,e,i)
   record.update(cells=cells,split_evidence=split)
  except(ValueError,IndexError,KeyError)as ex:
   record['failures'].append(type(ex).__name__+': '+str(ex));record['split_evidence']={'status':'not_assessed_extraction_blocked','evidence':[],'automatic_adjustment_applied':False,'complete_split_history_certified':False}
  record['raw_annual_pair_extracted']=all(record['cells'][s]is not None for s in SIDES);out.append(record)
 verify(contract['input_hashes']);verify(contract['pdf_runtime_hashes']);verify(code)
 summary={'events':44,'raw_annual_pairs_extracted':sum(e['raw_annual_pair_extracted']for e in out),'failures':dict(Counter(x for e in out for x in e['failures'])),'by_symbol':{s:{'events':sum(e['symbol']==s for e in out),'pairs':sum(e['symbol']==s and e['raw_annual_pair_extracted']for e in out)}for s in sorted({e['symbol']for e in out})},'explicit_weighted_average_pairs':sum(e['raw_annual_pair_extracted']and e['cells']['current']['weighted_average_explicit']for e in out),'implicit_basic_eps_denominator_pairs':sum(e['raw_annual_pair_extracted']and not e['cells']['current']['weighted_average_explicit']for e in out)}
 result={'schema':'after158_basic_shares_raw_extraction_v1','contract_sha256':contract_sha,'events':out,'summary':summary,'source_admission_changed':False,'activity':{'source_cells_extracted':2*summary['raw_annual_pairs_extracted'],'factors':0,'scores':0,'returns':0,'orders':0},'limitations':['Basic EPS denominator implicit semantics is labeled separately.','Same-report split text is evidence, not complete comparability certification.','Old identity, first-public-version, correction-history and source admission holds remain.','This smoothed share-base measure is not fiscal-year-end shares, actual net issuance or the published NI factor.']}
 if write:
  tests=read(OUT+'/TESTS.json');require(tests.get('passed')is True and tests.get('synthetic_only')is True and tests.get('implementation_hashes')==code,'current_passing_synthetic_tests_required')
  write_once(OUT+'/RAW_SHARES.json',result)
  manifest={'schema':'after158_basic_shares_extraction_manifest_v1','created_utc':datetime.now(timezone.utc).isoformat(),'contract':CONTRACT,'contract_sha256':contract_sha,'input_hashes':{CONTRACT:contract_sha,**contract['input_hashes'],**contract['pdf_runtime_hashes']},'implementation_hashes':code,'outputs':{OUT+'/'+name:sha(ROOT/(OUT+'/'+name))for name in ('RAW_SHARES.json','TESTS.json')},'summary':summary,'source_admission_changed':False,'activity':result['activity']}
  write_once(OUT+'/MANIFEST.json',manifest)
 print(json.dumps({'written':write,'summary':summary,'failed_events':[{'event_id':x['event_id'],'failures':x['failures']}for x in out if x['failures']],**({'output_sha256':sha(ROOT/(OUT+'/RAW_SHARES.json')),'manifest_sha256':sha(ROOT/(OUT+'/MANIFEST.json'))}if write else{})},indent=2))
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze-contract',action='store_true');p.add_argument('--contract-sha');p.add_argument('--write',action='store_true');a=p.parse_args()
 if a.freeze_contract:require(not a.contract_sha and not a.write,'freeze_is_metadata_only');freeze()
 else:run(a.contract_sha,a.write)
if __name__=='__main__':main()
