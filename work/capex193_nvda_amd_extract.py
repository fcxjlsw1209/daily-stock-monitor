"""CAPEX193: pinned, offline HTML evidence extraction; no financial ratios.

This conservatively verifies only unique, signed annual cash PP&E purchase rows
in an inherited formal cash-flow statement. Marketing and combined rows remain
inspectable evidence, never replacements. All inherited holds are retained.
"""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import argparse, hashlib, json, re
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'work/capex193_source_scope.json'
SCOPE_SHA = '71ccee9c200ae377010fb03a155bc7a6976f7f3f139db70a0a0ebdf49098fe08'
DATA = 'outputs/monthly_cash_quality_source_admission_v1/DATASET.json'
OUT = 'outputs/monthly_capex193_source/nvda_amd/EXTRACTION.json'
CF = re.compile(r'consolidated\s+statements?\s+of\s+cash\s+flows?', re.I)
PPE = re.compile(r'^(?:purchases?(?:\s+related)?\s+(?:of|to)|additions?\s+to)\s+property(?:,?\s+plant)?\s+and\s+equipment$', re.I)
CANDIDATE = re.compile(r'^(?:(?:purchases?|additions?|principal payments|payments)\b.*(?:property|equipment)|capital expenditures\b)', re.I)
COMBINED = re.compile(r'intangible|acquisition|businesses', re.I)
MONTHS = {n.lower():i for i,n in enumerate(('January','February','March','April','May','June','July','August','September','October','November','December'),1)}
DATE = re.compile(r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2}),?\s+(\d{4})\b',re.I)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def norm(text): return re.sub(r'\s+',' ',text).strip()
def source_bytes(root, item):
    path=(root/item['path']).resolve()
    if root.resolve() not in path.parents: raise ValueError('source path escape')
    raw=path.read_bytes()
    if sha(raw)!=item['sha256']: raise ValueError('SHA mismatch: '+item['path'])
    return raw

def validate_receipt(source, receipt):
    entry=receipt[source['receipt_entry_key']] if 'receipt_entry_key' in source else receipt
    if entry.get('sha256')!=source['sha256'] or entry.get('url')!=source['url']:
        raise ValueError('receipt source mismatch')
    if entry.get('status',entry.get('http_status'))!=200: raise ValueError('non200 receipt')
    return {k:entry.get(k) for k in ('url','final_url','status','http_status','retrieved_at','content_type','bytes','sha256')}

def table_grid(table):
    rows=[r for r in table.find_all('tr') if r.find_parent('table') is table]
    grid={}; physical={}
    for ri,row in enumerate(rows):
        col=0
        for pi,cell in enumerate(row.find_all(['td','th'],recursive=False)):
            while (ri,col) in grid: col+=1
            rs,cs=int(cell.get('rowspan',1)),int(cell.get('colspan',1))
            if not 1<=rs<=100 or not 1<=cs<=100: raise ValueError('invalid HTML span')
            item={'text':norm(cell.get_text(' ',strip=True)),'row':ri,'physical_cell':pi,
                  'logical_columns':list(range(col,col+cs)),'parsed_cell_sha256':sha(str(cell).encode())}
            physical[(ri,pi)]=item
            for rr in range(ri,ri+rs):
                for cc in item['logical_columns']:
                    if (rr,cc) in grid: raise ValueError('overlapping spans')
                    grid[(rr,cc)]=item
            col+=cs
    return rows,grid,physical

def context_before(table):
    parts=[];seen=set()
    for step,node in enumerate(table.previous_elements):
        if step>250: break
        if not isinstance(node,Tag): continue
        if node.name=='table' or node.find_parent('table') is not None: break
        if node.name not in ('div','p','font','strong','b','h1','h2','h3','h4') or node.find('table'): continue
        text=norm(node.get_text(' ',strip=True))
        if not text or len(text)>350 or text in seen: continue
        seen.add(text);parts.append(text)
        if CF.search(text) or len(parts)>=12: break
    return list(reversed(parts))

def exact_signed(literal):
    text=norm(literal).replace('\u2212','-')
    m=re.fullmatch(r'\$?\s*(\()?\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(\))?',text)
    if not m or bool(m[1])!=bool(m[3]) or (m[1] and m[2].startswith('-')):
        raise ValueError('missing or ambiguous signed monetary literal')
    value=Decimal(m[2].replace(',',''))
    if m[1]: value=-value
    return format(value,'f')

def row_evidence(rows,grid,physical,ti,ri):
    row=rows[ri]; items=[v for (r,_),v in physical.items() if r==ri]
    nonempty=[v for v in items if v['text']]
    label=nonempty[0]['text'] if nonempty else ''
    text=norm(row.get_text(' ',strip=True))
    return {'row_label':label,'row_text':text,'row_text_sha256':sha(text.encode()),
            'row_html_sha256':sha(str(row).encode()),'table_index_zero_based':ti,
            'row_index_zero_based':ri,'cells':items}

def annual_amount(table, rows, grid, physical, ri, inherited):
    """Verify inherited annual column against raw headers; locate new row itself."""
    body=next((i for i,r in enumerate(rows) if re.search(r'^cash flows from operating activities|^net (?:income|loss)',norm(r.get_text(' ',strip=True)),re.I)),None)
    if body is None or body>=ri: raise ValueError('missing cash-flow header boundary')
    header_items=[v for (r,_),v in physical.items() if r<body and v['text']]
    context=context_before(table)+[v['text'] for v in header_items]
    if not CF.search(' '.join(context)): raise ValueError('missing formal cash-flow title')
    units=[x for x in context if re.search(r'\bmillions\b',x,re.I)]
    if not units or any(re.search(r'\bthousands\b',x,re.I) for x in context): raise ValueError('missing or ambiguous million units')
    cols=list(inherited['logical_columns'])
    evidence=[]; seen=set()
    for rr in range(body):
        for cc in cols:
            item=grid.get((rr,cc))
            if item and item['text'] and (item['row'],item['physical_cell']) not in seen:
                evidence.append(item);seen.add((item['row'],item['physical_cell']))
    htext=' '.join(v['text'] for v in evidence)
    if not re.search(r'(?:year|twelve months|12 months)\s+ended',htext,re.I): raise ValueError('not annual column')
    found={f'{int(y):04d}-{MONTHS[m.lower()]:02d}-{int(d):02d}' for m,d,y in DATE.findall(htext)}
    if found!={inherited['period_end']}: raise ValueError('annual period header mismatch')
    selected=[];seen=set()
    for cc in cols:
        item=grid.get((ri,cc))
        if item and (item['row'],item['physical_cell']) not in seen:
            selected.append(item);seen.add((item['row'],item['physical_cell']))
    literal=norm(' '.join(v['text'] for v in selected))
    if '(' in literal and ')' not in literal:
        tail=grid.get((ri,max(cols)+1))
        if tail and tail['text']==')' and (tail['row'],tail['physical_cell']) not in seen:
            selected.append(tail);cols+=tail['logical_columns'];literal=norm(literal+' )')
    signed=exact_signed(literal)
    if Decimal(signed)>0: raise ValueError('purchase lacks explicit negative cash-outflow sign')
    currency='$' if any('$' in norm(r.get_text(' ',strip=True)) for r in rows[:ri]) else None
    if currency is None: raise ValueError('cash-flow currency literal missing')
    return {'signed_literal':literal,'signed_value_decimal':signed,
            'positive_outflow_decimal':format(-Decimal(signed),'f'),'scale':'1000000',
            'currency_literal':currency,'currency_iso':None,
            'locator':{'format':'html','page_1based':None,'period_end':inherited['period_end'],
                       'logical_columns':sorted(set(cols)),
                       'physical_cell_indices_zero_based':[v['physical_cell'] for v in selected],
                       'parsed_cell_sha256':[v['parsed_cell_sha256'] for v in selected],
                       'statement_title_evidence':[x for x in context if CF.search(x)],
                       'unit_literals':sorted(set(units)),'annual_header_evidence':evidence,
                       'annual_column_inherited_from_cfo_but_reverified':True,
                       'context_evidence':context}}

def extract(raw, target, old):
    soup=BeautifulSoup(raw,'html.parser'); tables=soup.find_all('table')
    inherited=target.get('current_cfo_locator'); candidates=[]; acceptable=[]; fail=[]
    formal_index=None if inherited is None else inherited['table_index_zero_based']
    for ti,table in enumerate(tables):
        rows,grid,physical=table_grid(table)
        for ri,row in enumerate(rows):
            ev=row_evidence(rows,grid,physical,ti,ri)
            if not CANDIDATE.search(ev['row_label']): continue
            ev['in_inherited_formal_cashflow_table']=ti==formal_index
            ev['classification']=('combined_pp_and_e_and_other_assets' if COMBINED.search(ev['row_label']) else
                                  'principal_payment_not_total_purchases' if re.match(r'principal',ev['row_label'],re.I) else
                                  'cash_pp_and_e_purchase_label' if PPE.fullmatch(ev['row_label']) else 'other_capital_candidate')
            ev['annual_amount']=None;ev['annual_amount_hold_reason']=None
            if ti==formal_index:
                try: ev['annual_amount']=annual_amount(table,rows,grid,physical,ri,inherited)
                except ValueError as exc: ev['annual_amount_hold_reason']=str(exc)
                if ev['classification']=='cash_pp_and_e_purchase_label' and ev['annual_amount'] is not None:
                    acceptable.append(ev)
            else: ev['annual_amount_hold_reason']='not_an_inherited_formal_annual_cashflow_statement'
            context=[norm(r.get_text(' ',strip=True)) for r in rows[max(0,ri-4):ri+1]]
            ev['nearby_rows']=context
            candidates.append(ev)
    capex=None
    if len(acceptable)==1:
        selected=acceptable[0];capex=dict(selected['annual_amount'])
        capex.update({k:selected[k] for k in ('row_label','row_text','row_text_sha256','row_html_sha256')})
        capex['locator'].update({k:selected[k] for k in ('table_index_zero_based','row_index_zero_based')})
        capex['period']=old['periods']['current']
    elif len(acceptable)>1: fail.append('multiple_scope_pp_and_e_rows_no_single_consolidated_literal')
    else:
        if inherited is None: fail.append('missing_inherited_formal_annual_cashflow_locator')
        if any(e['classification']=='combined_pp_and_e_and_other_assets' for e in candidates): fail.append('pp_and_e_combined_with_intangibles_or_other_assets')
        fail.append('no_unique_verified_current_annual_cash_pp_and_e_purchase_cell')
    inherited_reasons=list(old.get('cash_quality_admission',{}).get('reasons',[]))
    inherited_reasons+=list(old.get('inherited_rejection_codes',[]))
    if not target['inherited_admitted']: fail=['preserved_inherited_source_hold']+fail+inherited_reasons
    fail=list(dict.fromkeys(fail))
    return {'event_id':target['event_id'],'symbol':target['symbol'],'fiscal_year':old['fiscal_year'],
            'release_date':old['release_date'],'source':target['source'],
            'inherited_admitted':target['inherited_admitted'],'available_from':target['inherited_available_from'],
            'inherited_hold_reasons':inherited_reasons,'inherited_accounting_review_flags':old.get('accounting_review_flags',[]),
            'inherited_gaps':old.get('gaps',[]),'inherited_metadata_gaps':old.get('metadata_gaps',[]),
            'inherited_identity':old.get('identity'),'period':old['periods']['current'],
            'extraction_verified':capex is not None,'status':'verified' if capex is not None and target['inherited_admitted'] else 'held',
            'hold_reasons':fail,'capex':capex,'candidate_evidence':candidates,
            'formal_cashflow_locator_inherited':inherited,'source_table_count':len(tables),
            'fact_admitted':False,'model_admitted':False,'strategy_admitted':False}

def build(root=ROOT):
    scope_raw=source_bytes(root,{'path':SCOPE,'sha256':SCOPE_SHA});scope=json.loads(scope_raw)
    for path,digest in scope['input_sha256'].items(): source_bytes(root,{'path':path,'sha256':digest})
    old={e['event_id']:e for e in json.loads((root/DATA).read_bytes())['events']}
    targets=[e for e in scope['prospective_targets'] if e['symbol'] in ('NVDA','AMD')]
    if len(targets)!=22 or len({e['event_id'] for e in targets})!=22: raise ValueError('scope census mismatch')
    events=[]
    for target in targets:
        o=old[target['event_id']]
        if target['inherited_admitted']!=o['cash_quality_admission']['admitted'] or target['inherited_available_from']!=o['available_from']: raise ValueError('inheritance mismatch')
        source=target['source'];raw=source_bytes(root,source)
        receipt=json.loads(source_bytes(root,source['receipt']))
        receipt_evidence=validate_receipt(source,receipt)
        event=extract(raw,target,o);event['receipt_verified']=receipt_evidence;events.append(event)
    pins={SCOPE:SCOPE_SHA,**scope['input_sha256']}
    for t in targets:
        for item in (t['source'],t['source']['receipt']): pins[item['path']]=item['sha256']
    code=['work/capex193_nvda_amd_extract.py','work/test_capex193_nvda_amd_extract.py','work/capex193_nvda_amd_probe.py']
    return {'schema':'capex193_nvda_amd_source_extraction_v1','created_utc':datetime.now(timezone.utc).isoformat(),
            'scope_sha256':SCOPE_SHA,'input_sha256':pins,'code_sha256':{p:sha((root/p).read_bytes()) for p in code},
            'parser':'BeautifulSoup html.parser; local deterministic colspan grid; Decimal exact displayed amount',
            'definition':'current annual consolidated cash PP&E purchase outflow, positive; no ratios',
            'events':events,'summary':{'events':len(events),'AMD':sum(e['symbol']=='AMD' for e in events),
             'NVDA':sum(e['symbol']=='NVDA' for e in events),'extraction_verified':sum(e['extraction_verified'] for e in events),
             'verified':sum(e['status']=='verified' for e in events),'held':sum(e['status']=='held' for e in events)},
            'ratios_computed':False,'factors_computed':False,'returns_computed':False,'formal_admission':False,
            'limitations':['inherited holds preserved even when the CAPEX cell is independently verified',
              'combined intangible purchases never treated as pure PP&E',
              'FCF/selected corporate data never replaces missing formal annual cash-flow statement',
              'currency literal $ does not independently certify an ISO currency',
              'historical publication clock inherited; original first-public bytes/revisions not certified',
              'multiple continuing/discontinued purchase rows preserved without choosing or summing a consolidated value']}

def write_once(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    result=build()
    if args.write: write_once(ROOT/OUT,result)
    print(json.dumps({'summary':result['summary'],'events':[{'event_id':e['event_id'],'status':e['status'],'capex':None if e['capex'] is None else e['capex']['positive_outflow_decimal'],'holds':e['hold_reasons']} for e in result['events']]},indent=2))
if __name__=='__main__': main()
