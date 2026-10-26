"""Independent raw-source WC194 audit. Never sums components or computes ratios."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import argparse, hashlib, importlib, io, json, re
ROOT=Path(__file__).resolve().parents[1]
BASE='work/capex193_independent.py'
BASE_SHA='7ff6d145c56a6f0e66b40f569dc34cb10ad5dcba32b834c9376af2a0e17ed698'
assert hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest()==BASE_SHA
b=importlib.import_module('capex193_independent')
OUT=ROOT/'outputs/monthly_working_capital194_source/independent'
SCOPE='work/working_capital194_source_scope.json'
SCOPE_SHA='f2dda7ce4d3d61084bfbbb6e3fabca2388250575588054a41bd33890dc4948be'
CONCEPT='work/working_capital194_experiment_concept.json'
CONCEPT_SHA='380aae3cf18288dd668cefb3a9be94d0e9c90b76e8d085f6d59ff0ddafed9a83'
KEYS=('accounts_receivable','inventories','accounts_payable')
LABELS={'accounts_receivable':r'accounts receivable(?:,\s*net)?', 'inventories':r'inventories(?:,\s*net)?', 'accounts_payable':r'accounts payable'}
PAIRS=(('current','operating_cash_flow'),('current','total_assets'),('comparison','total_assets'))

def publish(name,obj):
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def scope_inputs():
 scope=json.loads(b.pinned(SCOPE,SCOPE_SHA));b.pinned(CONCEPT,CONCEPT_SHA)
 for p,h in scope['input_sha256'].items():b.pinned(p,h)
 events={e['event_id']:e for e in b.load(b.DATA)['events']}
 b.need(len(events)==44 and len(scope['prospective_targets'])==44,'scope44')
 b.need(set(events)=={t['event_id'] for t in scope['prospective_targets']},'exact original event identities')
 b.need(sum(t['inherited_admitted'] for t in scope['prospective_targets'])==31,'inherited31')
 for key in ('component_sum_permitted','new_financial_ratios_permitted','factor_computation_permitted','return_computation_permitted','observer_changes_permitted'):
  b.need(scope[key] is False,'source-only permission '+key)
 return scope,events

def preliminary_status(e):
 s,y=e['symbol'],int(e['fiscal_year'])
 if s in ('AAPL','MSFT'):return 'three_separate_annual_row_leads'
 if s=='NVDA':return 'three_separate_annual_row_leads' if y>=2020 else 'annual_operating_reconciliation_not_found_in_cached_release'
 if y in (2023,2024,2025):return 'three_separate_annual_row_leads'
 if y in (2015,2016,2017):return 'accounts_payable_combined_with_accrued_liabilities_and_other'
 return 'annual_operating_reconciliation_not_found_in_cached_release'

def preflight():
 scope,events=scope_inputs();rows=[]
 for t in scope['prospective_targets']:
  e=events[t['event_id']];s=t['source'];b.pinned(s['path'],s['sha256']);b.pinned(s['receipt']['path'],s['receipt']['sha256'])
  rows.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source=s,inherited_admitted=t['inherited_admitted'],inherited_holds=e['cash_quality_admission']['reasons'],row_availability_lead=preliminary_status(e),annual_cfo_locator=t['current_cfo_locator'],amounts_extracted=False,source_admission=False))
 refs={}
 for p in ('work/monthly_cash_quality146_factor.py','work/monthly_asset_growth147_factor.py','work/monthly_stock_comp149_factor.py','work/capex193_experiment_concept.json','work/after148_next_mechanism_review.json'):
  refs[p]=b.sha(b.path(p).read_bytes())
 publish('PREFLIGHT.json',dict(schema='wc194_source_inventory_not_admission_v1',created_utc=datetime.now(timezone.utc).isoformat(),scope_sha256=SCOPE_SHA,concept_sha256=CONCEPT_SHA,source_only=True,component_sum_computed=False,ratios_factors_orders_returns_computed=False,events=rows,novelty=dict(evidence_sha256=refs,readable_distinction=['146 measures CFO/assets, NI/assets and CFO-minus-NI/assets; it does not identify the three separate cash-flow contributions.','147 uses total-asset growth;149 subtracts stock compensation;193 subtracts cash PP&E. None identifies separate AR/inventory/AP signed cash-flow rows.','Bounded review of the latest50 completed PLAN contracts (141-192) found no explicit AR/inventory/AP measurement contract. This inventory is not a proof against every historical experiment.'],unknown=['Some earlier source implementations are missing; missing files are not evidence of novelty.']),limits=['32 three-row source leads and at most29 inherited-qualified leads are preliminary, not enforced extraction/admission counts.','Only three named trade working-capital contributions, not complete working capital or GAAP/organic CFO.','Separate supplier/related-party/vendor nontrade balances and flows are excluded; combined AP/accrual rows cannot substitute for pure AP.','No balance-sheet differences or missing-to-zero substitution.']))

def cell_label(t,r):
 nonblank=[b.norm(c['text']) for c in t['rows'][r] if b.norm(c['text'])]
 return nonblank[0] if nonblank else ''

def exact_component_label(label,key):return bool(re.fullmatch(LABELS[key],b.norm(label),re.I))

def signed_cell(t,r,columns):
 # Preserve explicitly printed sign/currency tokens split from the numeric cell.
 # Never extend into any neighbouring numeric value or infer an absent sign.
 cols=list(columns);g=b.grid(t)
 for c,allowed in ((min(cols)-1,{'(','-','−','$'}),(max(cols)+1,{')'})):
  x=g.get((r,c))
  if x and x['row']==r and x['columns']==[c] and b.norm(x['text']) in allowed:cols.append(c)
 cols.sort();literal,cells=b.selected(t,r,cols);value=b.number(literal)
 return literal,value,cells,cols

def html_components(raw,e):
 h=b.Tables();h.feed(raw.decode('utf-8',errors='strict'))
 pairs={}
 for period,name in PAIRS:
  f=e['fields'][period].get(name)
  pairs[period+'_'+name]=b.html_pair(h,f) if f and f['source']['sha256']==b.sha(raw) else {'checked':False,'reason':'field_missing_or_other_pinned_source'}
 fields={k:None for k in KEYS};holds=[];candidates=[]
 for t in h.tables:
  for r in range(len(t['rows'])):
   label=cell_label(t,r)
   if re.match(r'accounts receivable|inventories|accounts payable',label,re.I):
    candidates.append(dict(table=t['index'],row=r,row_label=label,row_text=b.rowtext(t,r),row_text_sha256=b.sha(b.rowtext(t,r).encode()),context=t['context']))
 cfo=e['fields']['current'].get('operating_cash_flow')
 if cfo is None:return fields,['missing_formal_current_annual_cashflow_column'],candidates,pairs
 loc=cfo['locator'];b.need(loc['format']=='html','HTML CFO locator')
 b.need(cfo['source']['sha256']==b.sha(raw),'CFO source mismatch')
 t=h.tables[loc['table_index_zero_based']];start=loc.get('statement_segment',{}).get('start_row',0);end=loc['row_index_zero_based']
 marker=[r for r in range(start,end) if re.search(r'changes in operating assets and liabilities',b.rowtext(t,r),re.I)]
 b.need(len(marker)==1,'unique operating-asset/liability reconciliation section')
 start=marker[0]+1;heads=b.header_evidence(t,loc)
 b.need(loc['period_end']==e['periods']['current']['period_end'],'current annual period identity')
 for key in KEYS:
  hits=[r for r in range(start,end) if exact_component_label(cell_label(t,r),key)]
  if len(hits)!=1:
   related=[x for x in candidates if x['table']==t['index'] and start<=x['row']<end and re.match('accounts payable',x['row_label'],re.I)] if key=='accounts_payable' else []
   holds.append(key+('_combined_scope_not_separable' if related else '_missing_or_ambiguous'))
   continue
  r=hits[0];lit,value,cells,cols=signed_cell(t,r,loc['logical_columns'])
  fields[key]=dict(component=key,signed_literal=lit,signed_value_decimal=str(value),scale=loc['scale'],annual_period_end=loc['period_end'],row_label=cell_label(t,r),row_text=b.rowtext(t,r),row_text_sha256=b.sha(b.rowtext(t,r).encode()),statement_title=loc['statement_title'],statement_currency_literal=cfo['currency_literal'],currency_iso=None,source=cfo['source'],period=e['periods']['current'],locator=dict(format='html',table=t['index'],row=r,logical_columns=cols,original_annual_anchor_columns=loc['logical_columns'],cells=cells,header_evidence=heads,operating_changes_marker_row=marker[0],original_cfo_locator=loc))
 # A continuing-operations reconciliation cannot be paired with total CFO.
 subtotal=[r for r in range(start,end) if re.search(r'cash.*operating activities.*(?:continuing|discontinued) operations',b.rowtext(t,r),re.I)]
 if subtotal:
  candidates.append(dict(scope_conflict='component_rows_in_continuing_operations_but_original_cfo_total',subtotal_rows=[{'row':r,'text':b.rowtext(t,r)} for r in subtotal],parsed_fields=fields))
  return {k:None for k in KEYS},['current_cfo_total_vs_continuing_component_scope_conflict'],candidates,pairs
 return fields,holds,candidates,pairs

def pdf_components(raw,e,pdfium,pypdf):
 cfo=e['fields']['current'].get('operating_cash_flow');b.need(cfo is not None,'PDF CFO source')
 loc=cfo['locator'];b.need(loc['format']=='pdf' and cfo['source']['sha256']==b.sha(raw),'PDF exact source')
 doc=pdfium.PdfDocument(raw)
 try:
  page=doc[loc['page_1based']-1];tp=page.get_textpage();text=tp.get_text_range();tp.close();page.close()
  normalized=b.norm(text)
  b.need('Twelve Months Ended' in normalized and '(In millions)' in normalized and 'CONSOLIDATED STATEMENTS OF CASH FLOWS' in normalized.upper(),'PDF annual statement/title/units')
  begin=normalized.index('Changes in operating assets and liabilities:');end=normalized.index(loc['row_label'])
  b.need(begin<end,'PDF operating reconciliation section')
  heads=normalized[:begin];fields={}
  for key,label in [('accounts_receivable','Accounts receivable, net'),('inventories','Inventories'),('accounts_payable','Accounts payable')]:
   b.need(begin<normalized.index(label,begin)<end,'PDF exact section '+key)
   ans=b.pdf_row(doc,loc['page_1based']-1,label,loc['financial_column_index_0based'],loc['header_dates'])
   fields[key]=dict(ans,component=key,signed_value_decimal=ans['value_decimal'],scale=cfo['scale'],annual_period_end=loc['period_end'],row_label=label,statement_title=loc['statement_title'],statement_currency_literal=cfo['currency_literal'],currency_iso=None,source=cfo['source'],period=e['periods']['current'],locator=dict(format='pdf',page=loc['page_1based'],financial_column_index=loc['financial_column_index_0based'],header_dates=loc['header_dates'],header_evidence=heads,original_cfo_locator=loc))
  pairs={period+'_'+name:b.pdf_pair(doc,e['fields'][period].get(name)) for period,name in PAIRS}
  # Independent render generated from this source, not a producer image.
  render=OUT/'renders'/('AAPL_FY'+str(e['fiscal_year'])+'_cashflow.png');render.parent.mkdir(parents=True,exist_ok=True)
  page=doc[loc['page_1based']-1];bitmap=page.render(scale=1.7);image=bitmap.to_pil();buffer=io.BytesIO();image.save(buffer,format='PNG');image.close();bitmap.close();page.close()
  if render.exists():b.need(render.read_bytes()==buffer.getvalue(),'existing independent render differs')
  else:
   with render.open('xb') as output:output.write(buffer.getvalue())
  rd=pypdf.PdfReader(io.BytesIO(raw));layout=rd.pages[loc['page_1based']-1].extract_text(extraction_mode='layout')
  evidence=dict(page=loc['page_1based'],pdfium_text=text,pypdf_layout=layout,render_path=str(render.relative_to(ROOT)),render_sha256=b.sha(render.read_bytes()),visual_review_pending=True)
  return fields,[],[evidence],pairs
 finally:doc.close()

def complete_pairs(e,pairs,pdfium,source_pins):
 for period,name in PAIRS:
  key=period+'_'+name;f=e['fields'][period].get(name)
  if f and not pairs[key]['checked']:
   s=f['source'];raw=b.pinned(s['path'],s['sha256']);source_pins[s['path']]=s['sha256'];b.pinned(s['receipt']['path'],s['receipt']['sha256']);source_pins[s['receipt']['path']]=s['receipt']['sha256']
   if raw.startswith(b'%PDF'):
    doc=pdfium.PdfDocument(raw)
    try:pairs[key]=b.pdf_pair(doc,f)
    finally:doc.close()
   else:
    h=b.Tables();h.feed(raw.decode('utf-8'));pairs[key]=b.html_pair(h,f)
 return pairs

def expected():
 scope,events=scope_inputs();sessions=json.loads(b.pinned(b.SESSIONS,b.SESSIONS_SHA))['sessions'];b.need(sessions==sorted(set(sessions)),'calendar order')
 pypdf,pdfium,rt=b.runtime();rows=[];pins=dict(scope['input_sha256']);pins[SCOPE]=SCOPE_SHA;pins[CONCEPT]=CONCEPT_SHA;pins[BASE]=BASE_SHA
 for target in scope['prospective_targets']:
  e=events[target['event_id']];s=target['source'];raw=b.pinned(s['path'],s['sha256']);pins[s['path']]=s['sha256'];b.pinned(s['receipt']['path'],s['receipt']['sha256']);pins[s['receipt']['path']]=s['receipt']['sha256']
  cfo=e['fields']['current'].get('operating_cash_flow')
  b.need(target['current_cfo_locator']==(cfo['locator'] if cfo else None),'scope CFO locator')
  b.need(target['inherited_admitted']==e['cash_quality_admission']['admitted'],'scope admission')
  avail=b.next_session(e['release_date'],sessions);b.need(avail==target['inherited_available_from']==e['available_from'],'strict next-session clock')
  if raw.startswith(b'%PDF'):fields,holds,ev,pairs=pdf_components(raw,e,pdfium,pypdf)
  else:fields,holds,ev,pairs=html_components(raw,e)
  complete_pairs(e,pairs,pdfium,pins)
  verified=all(fields[k] is not None for k in KEYS)
  if verified:b.need(330<=e['periods']['current']['days_inclusive']<=400,'annual duration')
  inherited=e['cash_quality_admission']['reasons']
  rows.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],release_date=e['release_date'],available_from=avail,source=s,source_sha256=s['sha256'],inherited_admitted=target['inherited_admitted'],inherited_hold_reasons=inherited,review_passed=True,components={k:dict(verified=fields[k] is not None,signed_value_decimal=fields[k]['signed_value_decimal'] if fields[k] else None,scale=fields[k]['scale'] if fields[k] else None,annual_period_end=fields[k]['annual_period_end'] if fields[k] else None) for k in KEYS},components_verified={k:fields[k] is not None for k in KEYS},extraction_verified=verified,three_signed_rows_verified=verified,fields=fields,source_hold_reasons=holds,hold_reasons=inherited+holds,candidate_evidence=ev,paired_original_fields=pairs,paired_cfo_verified=pairs['current_operating_cash_flow']['checked'],paired_assets_verified=pairs['current_total_assets']['checked'] and pairs['comparison_total_assets']['checked'],fact_admitted=False,strategy_admitted=False))
  print(e['event_id'],{k:fields[k] is not None for k in KEYS},holds,flush=True)
 b.need(len(rows)==44,'exact coverage44')
 runtime_manifest=b.load(b.RUNTIME);pins.update(runtime_manifest['pdf_runtime_hashes']);pins[b.RUNTIME]=b.RUNTIME_SHA
 acq=runtime_manifest['acquisition'];pins[acq['path']]=acq['sha256'];pins.update(b.load(acq['path'])['members'])
 old=runtime_manifest['old_plan'];pins[old['path']]=old['sha256']
 for p,h in pins.items():b.pinned(p,h)
 publish('EXPECTED.json',dict(schema='wc194_independent_raw_signed_components_v1',created_utc=datetime.now(timezone.utc).isoformat(),scope_sha256=SCOPE_SHA,concept_sha256=CONCEPT_SHA,source_only=True,component_sum_computed=False,ratios_factors_orders_returns_computed=False,code_sha256={'work/wc194_independent.py':b.sha(Path(__file__).read_bytes()),BASE:BASE_SHA},runtime=rt,input_sha256=pins,events=rows,summary=dict(events=len(rows),three_signed_rows_verified=sum(r['extraction_verified'] for r in rows),inherited_admitted=sum(r['inherited_admitted'] for r in rows),source_qualified_leads=sum(r['extraction_verified'] and r['inherited_admitted'] and r['paired_cfo_verified'] and r['paired_assets_verified'] for r in rows))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','expected']);args=ap.parse_args()
 {'preflight':preflight,'expected':expected}[args.action]()
