"""194 source-only signed trade operating-cashflow cells, offline and write-once.

No component aggregation, financial ratios, factors, orders or returns. The
current CFO locator is a pinned reference and its column/amount are reverified.
"""
from pathlib import Path
from datetime import datetime, timezone, date
from decimal import Decimal
import argparse, hashlib, json, re
from bs4 import BeautifulSoup, Tag

ROOT=Path(__file__).resolve().parents[1]
SCOPE='work/working_capital194_source_scope.json'
SCOPE_SHA='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
CONCEPT='work/working_capital194_experiment_concept.json'
CONCEPT_SHA='380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
DATA='outputs/monthly_cash_quality_source_admission_v1/DATASET.json'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
OUT='outputs/monthly_working_capital194_source/nvda_amd/EXTRACTION.json'
KEYS=('accounts_receivable','inventories','accounts_payable')
EXACT={
 'accounts_receivable':re.compile(r'^(?:trade\s+)?accounts? receivable(?:,?\s+net)?$',re.I),
 'inventories':re.compile(r'^inventor(?:y|ies)$',re.I),
 'accounts_payable':re.compile(r'^(?:trade\s+)?accounts? payable$',re.I)}
CANDIDATES={
 'accounts_receivable':re.compile(r'receivable',re.I),
 'inventories':re.compile(r'inventor',re.I),
 'accounts_payable':re.compile(r'payable',re.I)}
CF=re.compile(r'consolidated\s+statements?\s+of\s+cash\s+flows?',re.I)
CHANGES=re.compile(r'^changes in operating assets and liabilities',re.I)
MONTHS={n.lower():i for i,n in enumerate(('January','February','March','April','May','June','July','August','September','October','November','December'),1)}
DATE=re.compile(r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2}),?\s+(\d{4})\b',re.I)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def norm(text):return re.sub(r'\s+',' ',text).strip()
def source_bytes(root,item):
 p=(root/item['path']).resolve()
 if root.resolve() not in p.parents:raise ValueError('source path escape')
 raw=p.read_bytes()
 if sha(raw)!=item['sha256']:raise ValueError('SHA mismatch: '+item['path'])
 return raw

def validate_receipt(source,receipt):
 e=receipt[source['receipt_entry_key']] if 'receipt_entry_key' in source else receipt
 if e.get('sha256')!=source['sha256'] or e.get('url')!=source['url']:raise ValueError('receipt source mismatch')
 if e.get('status',e.get('http_status'))!=200:raise ValueError('non200 receipt')
 return {k:e.get(k) for k in ('url','final_url','status','http_status','retrieved_at','content_type','bytes','sha256')}

def table_grid(table):
 rows=[r for r in table.find_all('tr') if r.find_parent('table') is table];grid={};physical={}
 for ri,row in enumerate(rows):
  col=0
  for pi,cell in enumerate(row.find_all(['td','th'],recursive=False)):
   while (ri,col) in grid:col+=1
   rs,cs=int(cell.get('rowspan',1)),int(cell.get('colspan',1))
   if not 1<=rs<=100 or not 1<=cs<=100:raise ValueError('invalid HTML span')
   text=norm(cell.get_text(' ',strip=True));html=str(cell)
   item={'text':text,'row':ri,'physical_cell':pi,'logical_columns':list(range(col,col+cs)),
         'cell_html':html,'cell_html_sha256':sha(html.encode())}
   physical[(ri,pi)]=item
   for rr in range(ri,ri+rs):
    for cc in item['logical_columns']:
     if (rr,cc) in grid:raise ValueError('overlapping spans')
     grid[(rr,cc)]=item
   col+=cs
 return rows,grid,physical

def context_before(table):
 parts=[];seen=set()
 for step,node in enumerate(table.previous_elements):
  if step>250:break
  if not isinstance(node,Tag):continue
  if node.name=='table' or node.find_parent('table') is not None:break
  if node.name not in ('div','p','font','strong','b','h1','h2','h3','h4') or node.find('table'):continue
  text=norm(node.get_text(' ',strip=True))
  if not text or len(text)>350 or text in seen:continue
  seen.add(text);parts.append(text)
  if CF.search(text) or len(parts)>=12:break
 return list(reversed(parts))

def exact_signed(literal):
 text=norm(literal).replace('\u2212','-')
 # A dash is zero only after a unique annual numeric cell has been located.
 if re.fullmatch(r'\$?\s*[-\u2013\u2014]',text):return '0'
 m=re.fullmatch(r'\$?\s*(\()?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?',text)
 if not m or bool(m[1])!=bool(m[3]) or (m[1] and m[2].startswith('-')):raise ValueError('missing_or_ambiguous_signed_literal')
 value=Decimal(m[2].replace(',',''))
 if m[1]:value=-value
 return format(value,'f')

def row_evidence(rows,physical,ti,ri):
 row=rows[ri];items=[v for (r,_),v in physical.items() if r==ri]
 nonempty=[v for v in items if v['text']];text=norm(row.get_text(' ',strip=True));html=str(row)
 return {'row_label':nonempty[0]['text'] if nonempty else '', 'row_text':text,'row_text_sha256':sha(text.encode()),
         'row_html':html,'row_html_sha256':sha(html.encode()),'table_index_zero_based':ti,
         'row_index_zero_based':ri,'cells':items}

def annual_cell(table,rows,grid,physical,ri,inherited):
 body=next((i for i,r in enumerate(rows) if re.search(r'^cash flows from operating activities|^net (?:income|loss)',norm(r.get_text(' ',strip=True)),re.I)),None)
 if body is None or body>ri:raise ValueError('missing_cashflow_header_boundary')
 context=context_before(table)+[v['text'] for (r,_),v in physical.items() if r<body and v['text']]
 if not CF.search(' '.join(context)):raise ValueError('missing_formal_cashflow_title')
 units=[x for x in context if re.search(r'\bmillions\b',x,re.I)]
 if not units or any(re.search(r'\bthousands\b',x,re.I) for x in context):raise ValueError('missing_or_ambiguous_million_units')
 if inherited['scale']!='1000000':raise ValueError('inherited_scale_mismatch')
 cols=list(inherited['logical_columns']);headers=[];seen=set()
 for rr in range(body):
  for cc in cols:
   item=grid.get((rr,cc))
   if item and item['text'] and (item['row'],item['physical_cell']) not in seen:
    headers.append(item);seen.add((item['row'],item['physical_cell']))
 htext=' '.join(v['text'] for v in headers)
 if not re.search(r'(?:year|twelve months|12 months)\s+ended',htext,re.I):raise ValueError('not_current_annual_column')
 found={f'{int(y):04d}-{MONTHS[m.lower()]:02d}-{int(d):02d}' for m,d,y in DATE.findall(htext)}
 if found!={inherited['period_end']}:raise ValueError('annual_period_header_mismatch')
 cells=[];seen=set()
 for cc in cols:
  item=grid.get((ri,cc))
  if item and (item['row'],item['physical_cell']) not in seen:
   if item['row']!=ri:raise ValueError('annual_amount_inherited_from_other_row')
   cells.append(item);seen.add((item['row'],item['physical_cell']))
 literal=norm(' '.join(v['text'] for v in cells))
 if '(' in literal and ')' not in literal:
  tail=grid.get((ri,max(cols)+1))
  if tail and tail['row']==ri and tail['text']==')' and (tail['row'],tail['physical_cell']) not in seen:
   cells.append(tail);cols+=tail['logical_columns'];literal=norm(literal+' )')
 signed=exact_signed(literal)
 currency=[]
 for rr in range(body,len(rows)):
  rowtext=norm(rows[rr].get_text(' ',strip=True))
  if '$' in rowtext:
   currency.append({'row_index_zero_based':rr,'row_text':rowtext,'row_html_sha256':sha(str(rows[rr]).encode())})
 if not currency:raise ValueError('cashflow_statement_currency_missing')
 return {'signed_literal':literal,'signed_value_decimal':signed,'scale':'1000000',
  'currency_literal':'$' if '$' in literal else None,'currency_iso':None,
  'statement_currency_evidence':{'format':'html','table_index_zero_based':inherited['table_index_zero_based'],
      'literal':'$','rows':currency,'original_cfo_locator':inherited},
  'locator':{'format':'html','table_index_zero_based':inherited['table_index_zero_based'],'row_index_zero_based':ri,
      'period_end':inherited['period_end'],'logical_columns':sorted(set(cols)),
      'physical_cell_indices_zero_based':[v['physical_cell'] for v in cells],
      'cell_html_sha256':[v['cell_html_sha256'] for v in cells],'selected_cells':cells,
      'statement_title_evidence':[x for x in context if CF.search(x)],'unit_literals':sorted(set(units)),
      'annual_header_evidence':headers,'context_evidence':context,
      'annual_column_inherited_from_cfo_but_reverified':True,'original_cfo_locator':inherited}}

def extract(raw,target,old):
 soup=BeautifulSoup(raw,'html.parser');tables=soup.find_all('table'); inherited=target.get('current_cfo_locator')
 fields={k:None for k in KEYS};candidates={k:[] for k in KEYS};reasons={k:[] for k in KEYS}
 cfo_evidence=None;table_failure=None;formal_index=None if inherited is None else inherited['table_index_zero_based']
 formal_scope_rows=[];change_rows=[];cfo_index=None
 if inherited is not None:
  if not 0<=formal_index<len(tables):raise ValueError('inherited_table_index_invalid')
  table=tables[formal_index];rows,grid,physical=table_grid(table);cfo_index=inherited['row_index_zero_based']
  if not 0<=cfo_index<len(rows):raise ValueError('inherited_cfo_index_invalid')
  cfo_evidence=row_evidence(rows,physical,formal_index,cfo_index)
  if cfo_evidence['row_label']!=inherited['row_label']:raise ValueError('inherited_cfo_label_mismatch')
  cf=old.get('fields',{}).get('current',{}).get('operating_cash_flow')
  if not cf or cf['locator']!=inherited:raise ValueError('original_cfo_locator_mismatch')
  try:
   cfo_evidence['annual_cell']=annual_cell(table,rows,grid,physical,cfo_index,inherited)
   if Decimal(cfo_evidence['annual_cell']['signed_value_decimal'])!=Decimal(cf['value_decimal']):raise ValueError('original_cfo_amount_mismatch')
   if cf['scale']!=cfo_evidence['annual_cell']['scale'] or cf['period']!=old['periods']['current']:raise ValueError('original_cfo_period_scale_mismatch')
  except ValueError as exc:table_failure=str(exc)
  change_rows=[i for i,r in enumerate(rows[:cfo_index]) if CHANGES.search(norm(r.get_text(' ',strip=True)))]
  formal_scope_rows=[row_evidence(rows,physical,formal_index,i) for i,r in enumerate(rows[:cfo_index+1]) if re.search(r'continuing operations|discontinued operations',norm(r.get_text(' ',strip=True)),re.I)]
 for ti,table in enumerate(tables):
  rows,grid,physical=table_grid(table)
  for ri,row in enumerate(rows):
   ev=row_evidence(rows,physical,ti,ri);label=ev['row_label']
   relevant=[k for k in KEYS if CANDIDATES[k].search(label)]
   if not relevant:continue
   for key in relevant:
    item=dict(ev);item['annual_cell']=None;item['hold_reasons']=[]
    item['in_inherited_formal_cashflow_table']=ti==formal_index
    item['exact_component_label']=EXACT[key].fullmatch(label) is not None
    if ti!=formal_index:item['hold_reasons'].append('not_inherited_formal_annual_cashflow_table')
    else:
     if table_failure:item['hold_reasons'].append(table_failure)
     if not item['exact_component_label']:item['hold_reasons'].append('combined_or_other_component_label')
     if not change_rows or not any(i<ri<cfo_index for i in change_rows):item['hold_reasons'].append('outside_operating_asset_liability_reconciliation')
     if formal_scope_rows:item['hold_reasons'].append('continuing_discontinued_scope_not_combined_or_substituted')
     try:item['annual_cell']=annual_cell(table,rows,grid,physical,ri,inherited)
     except ValueError as exc:item['hold_reasons'].append(str(exc))
    item['hold_reasons']=list(dict.fromkeys(item['hold_reasons']));candidates[key].append(item)
 for key in KEYS:
  valid=[e for e in candidates[key] if e['annual_cell'] is not None and not e['hold_reasons']]
  if len(valid)==1:
   row=valid[0];field=dict(row['annual_cell'])
   field.update({k:row[k] for k in ('row_label','row_text','row_text_sha256','row_html','row_html_sha256')})
   field.update({'source':target['source'],'period':old['periods']['current'],
      'available_from':target['inherited_available_from'],'release_date':old['release_date'],
      'component':key,'fact_admitted':False})
   field['locator']['operating_reconciliation_heading_rows']=change_rows
   field['locator']['cashflow_table_html_sha256']=sha(str(tables[formal_index]).encode())
   fields[key]=field
  else:
   if inherited is None:reasons[key].append('missing_inherited_formal_annual_cashflow_locator')
   if len(valid)>1:reasons[key].append('multiple_component_rows_ambiguous_scope')
   else:
    relevant=[e for e in candidates[key] if e['in_inherited_formal_cashflow_table']]
    reasons[key]+=[r for e in relevant for r in e['hold_reasons']]
    reasons[key].append('no_unique_verified_current_annual_component_cell')
   reasons[key]=list(dict.fromkeys(reasons[key]))
 inherited_reasons=list(old.get('cash_quality_admission',{}).get('reasons',[]))+list(old.get('inherited_rejection_codes',[]))
 hold=[]
 if not target['inherited_admitted']:hold=['preserved_inherited_source_hold']+inherited_reasons
 hold += [k+':'+r for k in KEYS for r in reasons[k]]
 verified=all(fields[k] is not None for k in KEYS)
 return {'event_id':target['event_id'],'symbol':target['symbol'],'fiscal_year':old['fiscal_year'],
  'release_date':old['release_date'],'available_from':target['inherited_available_from'],'source':target['source'],
  'inherited_admitted':target['inherited_admitted'],'inherited_hold_reasons':inherited_reasons,
  'inherited_accounting_review_flags':old.get('accounting_review_flags',[]),'inherited_gaps':old.get('gaps',[]),
  'inherited_metadata_gaps':old.get('metadata_gaps',[]),'inherited_identity':old.get('identity'),
  'period':old['periods']['current'],'extraction_verified':verified,
  'status':'verified' if verified and target['inherited_admitted'] else 'held',
  'hold_reasons':list(dict.fromkeys(hold)),'component_hold_reasons':reasons,'fields':fields,
  'candidate_evidence':candidates,'formal_cashflow_locator_inherited':inherited,
  'original_cfo_evidence':cfo_evidence,'scope_conflict_evidence':formal_scope_rows,'source_table_count':len(tables),
  'component_sum_computed':False,'fact_admitted':False,'model_admitted':False,'strategy_admitted':False}

def build(root=ROOT):
 scope=json.loads(source_bytes(root,{'path':SCOPE,'sha256':SCOPE_SHA}))
 source_bytes(root,{'path':CONCEPT,'sha256':CONCEPT_SHA})
 for path,digest in scope['input_sha256'].items():source_bytes(root,{'path':path,'sha256':digest})
 old={e['event_id']:e for e in json.loads((root/DATA).read_bytes())['events']}
 sessions=json.loads((root/SESSIONS).read_bytes())['sessions']
 targets=[e for e in scope['prospective_targets'] if e['symbol'] in ('NVDA','AMD')]
 if len(targets)!=22 or len({e['event_id'] for e in targets})!=22:raise ValueError('scope census mismatch')
 records=[]
 for target in targets:
  o=old[target['event_id']]
  if target['inherited_admitted']!=o['cash_quality_admission']['admitted'] or target['inherited_available_from']!=o['available_from']:raise ValueError('inheritance mismatch')
  first=next((s for s in sessions if s>o['release_date']),None)
  if first!=target['inherited_available_from']:raise ValueError('release next-session mismatch')
  source=target['source'];raw=source_bytes(root,source);receipt=json.loads(source_bytes(root,source['receipt']))
  receipt_evidence=validate_receipt(source,receipt);record=extract(raw,target,o)
  record['receipt_verified']=receipt_evidence;record['availability_clock_verified']=True;records.append(record)
 pins={SCOPE:SCOPE_SHA,CONCEPT:CONCEPT_SHA,**scope['input_sha256']}
 for t in targets:
  for item in (t['source'],t['source']['receipt']):pins[item['path']]=item['sha256']
 code=['work/wc194_nvda_amd_extract.py','work/wc194_nvda_amd_tests.py','work/wc194_nvda_amd_probe.py']
 return {'schema':'working_capital194_nvda_amd_source_extraction_v1','created_utc':datetime.now(timezone.utc).isoformat(),
  'scope_sha256':SCOPE_SHA,'concept_sha256':CONCEPT_SHA,'input_sha256':pins,
  'code_sha256':{p:sha((root/p).read_bytes()) for p in code},'records':records,
  'parser':'BeautifulSoup html.parser; deterministic logical grid; exact Decimal signed literal',
  'summary':{'records':len(records),'AMD':sum(r['symbol']=='AMD' for r in records),'NVDA':sum(r['symbol']=='NVDA' for r in records),
   'inherited_admitted':sum(r['inherited_admitted'] for r in records),'inherited_held':sum(not r['inherited_admitted'] for r in records),
   'extraction_verified':sum(r['extraction_verified'] for r in records),'verified':sum(r['status']=='verified' for r in records),
   'held':sum(r['status']=='held' for r in records),'component_cells':{k:sum(r['fields'][k] is not None for r in records) for k in KEYS}},
  'component_sum_computed':False,'ratios_computed':False,'factors_computed':False,'returns_computed':False,'formal_admission':False,
  'limitations':scope['source_limitations']+['Inherited holds are retained; no later filing or balance-sheet difference substitutes for unavailable cashflow rows.']}

def write_once(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=build()
 if args.write:write_once(ROOT/OUT,result)
 print(json.dumps({'summary':result['summary'],'records':[{'event_id':r['event_id'],'status':r['status'],
  'signed_values':{k:None if v is None else v['signed_value_decimal'] for k,v in r['fields'].items()},'holds':r['hold_reasons']} for r in result['records']]},indent=2))
if __name__=='__main__':main()
