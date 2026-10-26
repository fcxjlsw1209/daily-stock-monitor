"""Independent source-cell audit after the exact HTML repair build, no factors."""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
BASE='outputs/monthly_cashflow_html_repair_v2/'
OLD='outputs/monthly_cashflow_source_preflight_v1/'
SELF='work/cashflow_html_repair_v2_review_actual.py'
OUT='work/cashflow_html_repair_v2_actual_review.json'
V1_PLAN_SHA='4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472'
V1_DATA_SHA='4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87'
SESSIONS='outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SESSIONS_SHA='c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p,h=None):
 if h:assert sha(p)==h,(p,'sha mismatch')
 return json.loads((ROOT/p).read_bytes())
def norm(s):return re.sub(r'\s+',' ',s).strip()
def expand(table):
 rows=[r for r in table.find_all('tr')if r.find_parent('table')is table]
 grid={};physical={}
 for ri,row in enumerate(rows):
  col=0
  for pi,cell in enumerate(row.find_all(['td','th'],recursive=False)):
   while(ri,col)in grid:col+=1
   rs=int(cell.get('rowspan',1));cs=int(cell.get('colspan',1));assert rs>0 and cs>0
   entry=dict(row=ri,physical=pi,text=norm(cell.get_text(' ',strip=True)),markup=str(cell),start_col=col,end_col=col+cs)
   physical[ri,pi]=entry
   for rr in range(ri,ri+rs):
    for cc in range(col,col+cs):assert(rr,cc)not in grid;grid[rr,cc]=entry
   col+=cs
 return rows,grid,physical

def independent_cell(raw,year,field):
 soup=BeautifulSoup(raw,'html.parser');found=[]
 for ti,table in enumerate(soup.find_all('table')):
  rows,grid,physical=expand(table)
  cfo_rows=sorted({r for(r,_),c in physical.items()if c['text'].rstrip(':').lower()=='net cash from operations'})
  for cfor in cfo_rows:
   nirows=sorted({r for(r,_),c in physical.items()if r<cfor and c['text'].rstrip(':').lower()=='net income'})
   if not nirows:continue
   nir=nirows[-1]
   formal=sorted({r for(r,_),c in physical.items()if r<nir and re.fullmatch(r'(?:consolidated )?(?:cash flows statements|statements of cash flows)',c['text'],re.I)})
   start=formal[-1]if formal else 0
   between=' '.join(c['text']for(r,_),c in physical.items()if nir<r<cfor)
   if not re.search(r'Adjustments to reconcile net income',between,re.I):continue
   headers=[c['text']for(r,_),c in physical.items()if start<=r<nir]
   if table.caption is not None:headers.append(norm(table.caption.get_text(' ',strip=True)))
   if not any(re.search(r'\bIn millions\b',h,re.I)for h in headers):continue
   maxcol=max(cc for(_,cc)in grid)
   cols=[];proof={}
   for cc in range(maxcol+1):
    seen=[]
    for rr in range(start,nir):
     c=grid.get((rr,cc))
     if c is not None and c['text']and c['text']not in seen:seen.append(c['text'])
    joined=' '.join(seen)
    if re.search(r'Twelve Months Ended June 30',joined,re.I)and not re.search(r'Three Months Ended',joined,re.I)and str(year)in seen:
     cols.append(cc);proof[cc]=seen
   if not cols:continue
   ri=nir if field=='net_income'else cfor
   unique={grid[ri,c]['physical']:grid[ri,c]for c in cols if(ri,c)in grid}
   candidates=[c for c in unique.values()if re.search(r'\d',c['text'])and not re.search(r'[A-Za-z]',c['text'])]
   if len(candidates)!=1:continue
   cell=candidates[0]
   # Only syntax normalization: no cross-field arithmetic.
   literal=re.sub(r'\s*,\s*',',',cell['text']).replace('$','').strip()
   assert re.fullmatch(r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)',literal),literal
   plain=literal.replace(',','');plain='-'+plain[1:-1]if plain.startswith('(')else plain
   found.append(dict(value_decimal=str(Decimal(plain)),scale='1000000',table_index_zero_based=ti,row_index_zero_based=ri,
    physical_cell_index_zero_based=cell['physical'],raw_literal=cell['text'],raw_markup_sha256=hashlib.sha256(cell['markup'].encode()).hexdigest(),
    header_evidence=proof,cf_segment_start_row=start,ni_start_row=nir,cfo_total_row=cfor,period_end='%d-06-30'%year))
 assert len(found)==1,('independent cell not unique',year,field,len(found))
 return found[0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);p.add_argument('--data-sha',required=True);a=p.parse_args()
 assert re.fullmatch('[0-9a-f]{64}',a.plan_sha)and re.fullmatch('[0-9a-f]{64}',a.data_sha)
 plan=read(BASE+'PLAN.json',a.plan_sha);data=read(BASE+'RAW_FACTS.json',a.data_sha);old=read(OLD+'RAW_FACTS.json',V1_DATA_SHA);oldplan=read(OLD+'PLAN.json',V1_PLAN_SHA)
 audit=read(BASE+'AUDIT.json');manifest=read(BASE+'MANIFEST.json')
 assert plan['stage']=='frozen_HTML_repair_plan'and data['plan_sha256']==a.plan_sha
 assert audit['data_sha256']==a.data_sha and manifest['output_hashes']['RAW_FACTS.json']==a.data_sha
 assert manifest['output_hashes']['AUDIT.json']==sha(BASE+'AUDIT.json')
 sessions=read(SESSIONS,SESSIONS_SHA)['sessions'];assert sessions==sorted(set(sessions))
 olds={e['event_id']:e for e in old['events']};ops={e['event_id']:e for e in oldplan['events']}
 expected=[e['event_id']for e in old['events']if e['symbol']in ('MSFT','AMD','NVDA')]
 assert[e['event_id']for e in data['events']]==expected and len(expected)==34
 source_hashes={BASE+'PLAN.json':a.plan_sha,BASE+'RAW_FACTS.json':a.data_sha,BASE+'AUDIT.json':sha(BASE+'AUDIT.json'),BASE+'MANIFEST.json':sha(BASE+'MANIFEST.json'),OLD+'PLAN.json':V1_PLAN_SHA,OLD+'RAW_FACTS.json':V1_DATA_SHA,SESSIONS:SESSIONS_SHA,SELF:sha(SELF)}
 positions=[];proof=[];preserved=0
 for e in data['events']:
  prior=olds[e['event_id']];spec=ops[e['event_id']]
  for key in ('periods','available_from','release_date','identity','inherited_rejection_codes','source'):
   assert e[key]==prior[key]==spec[key],(e['event_id'],key)
  assert all(e[k]is False for k in ('fact_admitted','model_admitted','strategy_admitted','first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'))
  for side in ('current','comparison'):
   for field in ('net_income','operating_cash_flow','total_assets'):
    before=prior['fields'][side][field];after=e['fields'][side][field]
    if before is not None:
     assert after is not None
     for key in ('value_decimal','displayed_amount','scale','currency_literal','period','available_from','source','parsed_cell_sha256'):
      assert after[key]==before[key],(e['event_id'],side,field,key)
     preserved+=1
    elif after is not None:
     pos=(e['event_id'],side,field);positions.append(pos)
     assert e['symbol']=='MSFT'and e['fiscal_year']in (2018,2021)
     source=e['source'];assert sha(source['path'])==source['sha256'];source_hashes[source['path']]=source['sha256']
     year=int(e['periods'][side]['period_end'][:4]);ref=independent_cell((ROOT/source['path']).read_bytes(),year,field)
     assert(ref['value_decimal'],ref['scale'])==(after['value_decimal'],after['scale']),pos
     assert ref['period_end']==after['period']['period_end']
     assert after['period']==spec['periods'][side]
     assert after['available_from']==sessions[bisect_right(sessions,spec['release_date'])]==spec['available_from']
     assert after['source']==source and after['fact_admitted']is False
     assert ref['raw_markup_sha256']in after['parsed_cell_sha256']
     proof.append(dict(event_id=e['event_id'],side=side,field=field,source=source,independent_reference=ref,
      current_release=spec['release_date'],available_from=after['available_from'],period=after['period'],matched=True))
 assert set(positions)=={('MSFT:FY2018:2018-07-19','current','net_income'),('MSFT:FY2021:2021-07-27','current','net_income'),('MSFT:FY2021:2021-07-27','comparison','net_income'),('MSFT:FY2021:2021-07-27','current','operating_cash_flow'),('MSFT:FY2021:2021-07-27','comparison','operating_cash_flow')}
 assert preserved==157 and len(positions)==5
 result=dict(schema_version=1,stage='independent_actual_HTML_representation_cell_review',created_utc=datetime.now(timezone.utc).isoformat(),
  passed=True,plan_sha256=a.plan_sha,data_sha256=a.data_sha,events=34,old_nonnull_cells_exactly_preserved=preserved,new_cells_independently_verified=len(proof),
  method='Independent BeautifulSoup physical-row/column expansion and exact annual-header search; no producer code imported. Syntax-only decimal parsing and literal equality checks.',
  checked_positions=proof,all_original_identity_period_and_availability_preserved=True,all_rejections_preserved=True,
  first_public_version_verified=False,complete_correction_history_verified=False,source_admissions=0,model_admissions=0,strategy_admissions=0,
  financial_calculations=0,factors_computed=0,returns_computed=0,input_hashes=source_hashes,source_hashes=source_hashes)
 encoded=(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
 with(ROOT/OUT).open('xb')as f:f.write(encoded)
 print(json.dumps(dict(path=OUT,sha256=hashlib.sha256(encoded).hexdigest(),passed=True,new_cells=5,preserved=157)))
if __name__=='__main__':main()
