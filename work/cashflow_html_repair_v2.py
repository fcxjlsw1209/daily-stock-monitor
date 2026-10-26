"""Two bounded HTML representation repairs; no new source or financial model.

Draft/freeze inspect only table metadata. Build is separately hash-authorized.
V1 and original source bytes are read-only throughout.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from bs4 import BeautifulSoup
import cashflow_source_preflight_v1 as v1

io=v1.safe
ROOT=Path(__file__).resolve().parents[1]
SELF='work/cashflow_html_repair_v2.py'
TEST='work/test_cashflow_html_repair_v2.py'
OUT='outputs/monthly_cashflow_html_repair_v2'
OLD=v1.OUT
PINS={
 OLD+'/PLAN.json':'4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472',
 OLD+'/RAW_FACTS.json':'4f6d6e694195a4986f911f7ba1defc24efc515b49aa510b5c4be61df78c45c87',
 OLD+'/AUDIT.json':'43e908b15cf4ceb5fdfce700d74c6ff3788b906cff8f485c8c5e55026161c143',
 OLD+'/HTML_INDEPENDENT_REVIEW.json':'7fd96c4f899ca482682ff70ffa9da3ee61daf6823b878b320899e0650ac955e2',
}
TITLE=re.compile(r'^(?:condensed\s+)?(?:consolidated\s+)?(?:statements?\s+of\s+cash\s+flows?|cash\s+flows?\s+statements?|balance\s+sheets?)$',re.I)
require=io.require

def schema():
 return dict(schema_version=1,stage='HTML_representation_repair_only',census=dict(AMD=10,MSFT=12,NVDA=12),events=34,maximum_cells=204,
  repairs=['Split one HTML table at explicit formal statement title rows; each segment needs its own units/annual headers.',
   'Normalize whitespace immediately around thousands-separator commas, then apply the unchanged strict monetary grammar.'],
  exclusions='No new sources, inferred values, changed NI scope, income-statement fallback, selected-cashflow-summary admission, OCR, ratios, signs, ranks or returns.',
  status='All candidates remain unadmitted; existing rejection and accounting flags remain. No first-public-version, correction-history or daily-security-master certification.')

def code_hashes(root):return {p:io.hash_file(root,p)for p in (SELF,TEST)}

def segment_ranges(table):
 rows,_,physical=v1.html.table_grid(table)
 titles=[]
 for ri in range(len(rows)):
  texts=[c['text']for (r,_),c in physical.items()if r==ri and c['text']]
  prior_body=any(v1.NI.fullmatch(v1.norm(c['text']))or re.fullmatch(r'assets|total assets|cash(?: and cash equivalents)?',v1.norm(c['text']),re.I)for (r,_),c in physical.items()if r<ri)
  if prior_body and len(texts)==1 and TITLE.fullmatch(v1.norm(texts[0])):
   titles.append((ri,v1.norm(texts[0])))
 starts=sorted({0,*[ri for ri,_ in titles]})
 if len(starts)==1:return []
 spans=[]
 for i,start in enumerate(starts):
  end=starts[i+1]if i+1<len(starts)else len(rows)
  # Splitting across a rowspan would silently change cell identity.
  require(not any(r<start<r+c['rowspan']or r<end<r+c['rowspan']for (r,_),c in physical.items()),'Statement boundary crosses rowspan')
  spans.append(dict(start=start,end=end,title=next((t for r,t in titles if r==start),None)))
 return spans

def inventory(raw,periods):
 original=v1.html_inventory(raw,periods)
 soup=BeautifulSoup(raw,'html.parser')
 for node in soup(['script','style']):node.decompose()
 repaired=deepcopy(original);segments=[]
 for ti,table in enumerate(soup.find_all('table')):
  try:ranges=segment_ranges(table)
  except ValueError:
   repaired['metadata_gaps'].append(dict(table_index=ti,reason='segmentation_rejected_cross_boundary_span'))
   continue
  if not ranges:continue
  rows,_,_=v1.html.table_grid(table)
  for side in v1.SIDES:
   for field in v1.FIELDS:
    repaired['candidates'][side][field]=[x for x in repaired['candidates'][side][field]if x['table_index_zero_based']!=ti]
  for part in ranges:
   virtual=BeautifulSoup('<table></table>','html.parser');target=virtual.table
   if part['start']==0:
    context=v1.html.external_context(table)
    if not(v1.CF_TITLE.search(' '.join(context))or v1.BS_TITLE.search(' '.join(context))):
     context=[v1.legacy.statement_context(table)]
    for text in context:
     node=virtual.new_tag('p');node.string=text;target.insert_before(node)
    if table.caption:target.append(deepcopy(table.caption))
   for row in rows[part['start']:part['end']]:target.append(deepcopy(row))
   result=v1.html_inventory(str(virtual).encode(),periods)
   segments.append(dict(table_index_zero_based=ti,start_row=part['start'],end_row_exclusive=part['end'],title=part['title']))
   for side in v1.SIDES:
    for field in v1.FIELDS:
     for loc in result['candidates'][side][field]:
      loc=deepcopy(loc);loc['table_index_zero_based']=ti;loc['row_index_zero_based']+=part['start']
      loc['statement_segment']=deepcopy(segments[-1])
      repaired['candidates'][side][field].append(loc)
   for gap in result['metadata_gaps']:
    repaired['metadata_gaps'].append(dict(gap,original_table_index=ti,segment_start_row=part['start']))
 repaired['statement_segments']=segments
 return repaired

def normalize_commas(display):
 # Only comma-adjacent whitespace is removed. The unchanged strict grammar
 # rejects malformed grouping, a naked digit gap, percentages and footnotes.
 return re.sub(r'\s*,\s*',',',display)

def extract(raw,loc):
 soup=BeautifulSoup(raw,'html.parser');table=soup.find_all('table')[loc['table_index_zero_based']]
 rows,_,physical=v1.html.table_grid(table);ri=loc['row_index_zero_based']
 cells=[physical[(ri,pi)]['text']for pi in loc['physical_cell_indices_zero_based']]
 display=' '.join(x for x in cells if x).strip();normalized=normalize_commas(display)
 value=v1.html.exact_amount(normalized)
 markups=[str(rows[ri].find_all(['td','th'],recursive=False)[pi])for pi in loc['physical_cell_indices_zero_based']]
 return dict(displayed_amount=display,normalized_monetary_literal=normalized,value_decimal=value,
  parsed_cell_sha256=[io.digest(x.encode())for x in markups],comma_whitespace_normalized=normalized!=display)

def map_event(spec,raw):
 observed=inventory(raw,spec['periods']);require(observed==spec['locator_inventory'],'Frozen V2 locators changed')
 out={k:deepcopy(value)for k,value in spec.items()if k!='locator_inventory'}
 out.update(fields={s:{}for s in v1.SIDES},gaps=[],accounting_review_flags=observed['accounting_review_flags'],metadata_gaps=observed['metadata_gaps'],
  fact_admitted=False,model_admitted=False,strategy_admitted=False,first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False)
 for side in v1.SIDES:
  for field in v1.FIELDS:
   locs=observed['candidates'][side][field];out['fields'][side][field]=None
   reason='missing_formal_annual_cell'if not locs else'context_or_column_ambiguous'if len(locs)!=1 else None
   if reason is None:
    loc=locs[0]
    try:
     require(loc['currency_literal']=='$','missing_literal_currency');item=extract(raw,loc)
     out['fields'][side][field]=dict(item,scale=loc['scale'],currency_literal=loc['currency_literal'],currency_iso=None,
      period=deepcopy(spec['periods'][side]),available_from=spec['available_from'],source=deepcopy(spec['source']),locator=loc,fact_admitted=False)
    except (ValueError,IndexError,KeyError)as exc:reason='invalid_raw_cell: '+str(exc)
   if reason:out['gaps'].append(dict(side=side,field=field,reason=reason,candidate_count=len(locs)))
 return out

def closure(root):
 io.verify_hashes(root,PINS);plan=io.read(root,OLD+'/PLAN.json');sources=dict(PINS)
 review=io.read(root,OLD+'/HTML_INDEPENDENT_REVIEW.json')
 require(review['passed']is True and review['plan_sha256']==PINS[OLD+'/PLAN.json']and review['data_sha256']==PINS[OLD+'/RAW_FACTS.json'],'Exact V1 independent review binding')
 sources[OLD+'/MANIFEST.json']=review['source_manifest_sha256']
 for group in (plan['source_hashes'],plan['source_code_hashes'],review['review_code_hashes']):
  for p,h in group.items():
   require(p not in sources or sources[p]==h,'Conflicting frozen pin');sources[p]=h
 io.verify_hashes(root,sources)
 return plan,sources

def validate_specs(events,original):
 expected=[e for e in original['events']if e['symbol']in ('AMD','MSFT','NVDA')]
 require(len(expected)==34 and len({e['event_id']for e in expected})==34,'Canonical34 reference events required')
 require([e['event_id']for e in events]==[e['event_id']for e in expected],'Exact unique ordered canonical34 events required')
 for actual,prior in zip(events,expected):
  require(set(actual)==set(prior),'Unexpected V2 event metadata keys')
  for key in prior:
   if key!='locator_inventory':require(actual[key]==prior[key],'V2 changed canonical metadata: '+key)

def draft(root):
 old,sources=closure(root);tests=io.read(root,OUT+'/TESTS.json')
 require(tests['passed']is True and tests['synthetic_only']is True and tests['source_code_hashes']==code_hashes(root),'Current passing synthetic tests required')
 events=[]
 for spec in old['events']:
  if spec['symbol']not in ('AMD','MSFT','NVDA'):continue
  item=deepcopy(spec);raw=io.local(root,spec['source']['path']).read_bytes()
  require(io.digest(raw)==spec['source']['sha256'],'Source SHA mismatch')
  item['locator_inventory']=inventory(raw,spec['periods']);events.append(item)
 require(len(events)==34 and len({e['event_id']for e in events})==34,'All34 HTML events required')
 validate_specs(events,old)
 sources[OUT+'/TESTS.json']=io.hash_file(root,OUT+'/TESTS.json')
 result=dict(schema_version=1,stage='metadata_only_HTML_repair_draft',schema=schema(),runtime=sys.version,
  source_hashes=sources,source_code_hashes=code_hashes(root),events=events,real_new_source_cells_extracted=0,
  financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,fact_admissions=0)
 io.verify_hashes(root,sources);io.publish(root,OUT+'/DRAFT.json',result)
 return dict(draft_sha256=io.hash_file(root,OUT+'/DRAFT.json'),events=34,new_cells_extracted=0,segmented_events=[e['event_id']for e in events if e['locator_inventory']['statement_segments']])

def freeze(root,draft_sha):
 v1.expect_sha(draft_sha);data=io.read(root,OUT+'/DRAFT.json',draft_sha)
 require(data['stage']=='metadata_only_HTML_repair_draft'and data['schema']==schema(),'Exact metadata-only V2 draft required')
 require(data['source_code_hashes']==code_hashes(root)and data['runtime']==sys.version,'Draft code/runtime changed')
 io.verify_hashes(root,data['source_hashes'])
 validate_specs(data['events'],io.read(root,OLD+'/PLAN.json'))
 plan=dict(data,stage='frozen_HTML_repair_plan',draft_sha256=draft_sha,created_utc=datetime.now(timezone.utc).isoformat())
 io.publish(root,OUT+'/PLAN.json',plan)
 return dict(plan_sha256=io.hash_file(root,OUT+'/PLAN.json'),new_cells_extracted=0)

def products(root,plan_sha):
 v1.expect_sha(plan_sha);plan=io.read(root,OUT+'/PLAN.json',plan_sha)
 require(plan['source_code_hashes']==code_hashes(root)and plan['runtime']==sys.version,'Frozen code/runtime changed')
 require(plan['stage']=='frozen_HTML_repair_plan'and len(plan['events'])==34,'Frozen full HTML plan required')
 io.verify_hashes(root,plan['source_hashes']);old=io.read(root,OLD+'/RAW_FACTS.json')
 validate_specs(plan['events'],io.read(root,OLD+'/PLAN.json'))
 old_events={e['event_id']:e for e in old['events']}
 events=[];preserved=0;repaired=[]
 for spec in plan['events']:
  raw=io.local(root,spec['source']['path']).read_bytes();require(io.digest(raw)==spec['source']['sha256'],'Source SHA mismatch')
  event=map_event(spec,raw);previous=old_events[event['event_id']]
  for side in v1.SIDES:
   for field in v1.FIELDS:
    before=previous['fields'][side][field];after=event['fields'][side][field]
    if before is not None:
     require(after is not None,'V2 dropped an existing nonnull value')
     for key in ('value_decimal','displayed_amount','scale','currency_literal','period','available_from','source','parsed_cell_sha256'):
      require(before[key]==after[key],'V2 changed frozen existing value/evidence')
     preserved+=1
    elif after is not None:repaired.append(dict(event_id=event['event_id'],side=side,field=field))
  require(event['inherited_rejection_codes']==previous['inherited_rejection_codes'],'V2 dropped inherited rejection')
  events.append(event)
 count=sum(e['fields'][s][f]is not None for e in events for s in v1.SIDES for f in v1.FIELDS)
 data=dict(schema_version=1,stage='HTML_repair_candidates_not_admitted',plan_sha256=plan_sha,events=events,
  counts=dict(events=34,requested_cells=204,extracted_cells=count,gap_cells=204-count,existing_v1_cells_preserved=preserved,newly_parsed_cells=len(repaired)),
  financial_calculations=0,model_fits=0,ratios_computed=0,signs_computed=0,ranks_computed=0,returns_computed=0,fact_admissions=0,model_admissions=0,strategy_admissions=0)
 audit=dict(schema_version=1,stage='HTML_repair_source_audit',passed=True,meaning='Representation repairs only; all new cells require separate independent audit.',
  plan_sha256=plan_sha,data_sha256=io.digest(io.encoded(data)),source_hashes=plan['source_hashes'],source_code_hashes=plan['source_code_hashes'],
  counts=data['counts'],newly_parsed_positions=repaired,all_original_nonnull_cells_exactly_preserved=True,
  all_inherited_rejections_preserved=True,independent_new_cell_audit_required=True,financial_calculations=0,model_fits=0,ranks_computed=0,returns_computed=0,fact_admissions=0)
 io.verify_hashes(root,plan['source_hashes']);return data,audit

def build(root,plan_sha,verify=False):
 data,audit=products(root,plan_sha);outputs={'RAW_FACTS.json':data,'AUDIT.json':audit}
 outputs['MANIFEST.json']=dict(plan_sha256=plan_sha,output_hashes={name:io.digest(io.encoded(value))for name,value in outputs.items()})
 for name,value in outputs.items():
  if verify:require(io.read(root,OUT+'/'+name)==value,'Reproduced V2 output mismatch')
  else:io.publish(root,OUT+'/'+name,value)
 return dict(passed=True,counts=data['counts'],newly_parsed_positions=audit['newly_parsed_positions'])

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 for key in ('draft','freeze-plan','build','verify'):g.add_argument('--'+key,action='store_true')
 p.add_argument('--draft-sha');p.add_argument('--plan-sha');a=p.parse_args()
 result=draft(ROOT)if a.draft else freeze(ROOT,a.draft_sha)if a.freeze_plan else build(ROOT,a.plan_sha,a.verify)
 print(io.json.dumps(result,sort_keys=True))
