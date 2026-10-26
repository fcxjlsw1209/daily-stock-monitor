"""AMD same-release annual comparison candidates. Offline, source-only, never admission.

Reuse the frozen original-current-year statement parser, then independently locate
its comparison-year groups. A separate stdlib HTML parser audits the four cells.
No financial ratios, factors, scores, trading eligibility, or returns are computed.
"""
import argparse
import bisect
import copy
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
SELF = 'work/after142_amd_comparative_source.py'
TEST = 'work/after142_amd_comparative_tests.py'
OUT = 'outputs/monthly_amd_comparative_source_v1'
PINNED = {
 'work/after141_amd_source_extract.py':'868cae81055318b9df934f4bfe2849c4971f861c4b480942139c8c7606ae4766',
 'work/after141_amd_source_extract_tests.py':'675903274089939066dd3d2848682f64c0ed8c3c83199c4105785c9f2639276b',
 'work/after142_profitability_admission_gaps.json':'1ac5a2fef5e586de5f98e56c5b92d02efe258479fbe999719258cd7485716405',
 'outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json':'cc6f7b11ce8a4f80a72fab27bb6ce7676f0dd893bfe92adb9d0de5513d01ae50',
 'outputs/monthly_profitability_extension_sources_v1/RAW_FACTS.json':'4e88dce4be908a6059481ca383d868422ae0698811b235963e260027617777c6',
 'outputs/monthly_earnings_repair/batch_technology_industrial/AMD/events.json':'afe00f761ace64666bf5df44261b085dd2412c6fa1f5f13d980be09ca2b38892',
 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json':'c05c15f19bc1f7a3428e456a71199d64f05799ef3c432fa45f7bc485d0ff3cf4',
 'outputs/monthly_profitability141_issuer_admission/ADMISSION_POLICY.json':'73fc02e55c00557a7f2f72336e3909385e8a0c1f08366d54dbc843df16cc2d7d',
}
YEARS = (2015,2016,2017,2018,2020,2021,2022,2023,2024,2025)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def local(path):
 p=Path(path)
 if p.is_absolute() or '..' in p.parts: raise ValueError('Local relative path required')
 p=ROOT/p
 if p.resolve()!=p: raise ValueError('Symlinks prohibited')
 return p

# Validate dependency bytes before import, including when synthetic tests import us.
if sha(local('work/after141_amd_source_extract.py').read_bytes()) != PINNED['work/after141_amd_source_extract.py']:
 raise ValueError('Frozen parser dependency changed')
import after141_amd_source_extract as base


def canonical_day(value):
 if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value: raise ValueError('Canonical date required')
 return date.fromisoformat(value)


def comparative_cell(soup, current, prior, kind):
 """Locate an annual prior group by its own header, not by column offset."""
 ti=current['locator']['table_index_zero_based']; table=soup.find_all('table')[ti]
 rows, grid, physical=base.table_grid(table)
 starts=[i for i,row in enumerate(rows) if any(base.BODY_START.fullmatch(base.norm(c.get_text(' ',strip=True))) for c in row.find_all(['td','th'],recursive=False))]
 if not starts: raise ValueError('No financial body')
 stop=min(starts); cols=sorted({c for _,c in grid}); evidence={c:base.header_evidence(grid,stop,c) for c in cols}
 target=[]
 for c in cols:
  texts=[x['text'] for x in evidence[c]]
  annual=any(re.fullmatch(r'year\s+ended(?:\s+.*)?',s,re.I) for s in texts)
  quarterly=any(re.search(r'\b(?:three|six|nine|3|6|9)\s+months?\s+ended',s,re.I) for s in texts)
  if base.dates(texts)==[prior] and (kind=='total_assets' or annual and not quarterly): target.append(c)
 if not target or target!=list(range(min(target),max(target)+1)): raise ValueError('Ambiguous comparison year column group')
 ri=current['locator']['row_index_zero_based']; selected=[x for (r,_),x in physical.items() if r==ri and set(target).intersection(x['columns'])]
 if not selected or any(x['rowspan']!=1 or not set(x['columns'])<=set(target) for x in selected): raise ValueError('Comparison cell crosses date groups')
 display=base.norm(' '.join(x['text'] for x in selected if x['text']))
 value=base.exact_amount(display)
 if kind=='total_assets' and Decimal(value)<=0: raise ValueError('Nonpositive comparison assets')
 dollars=[x for (r,_),x in physical.items() if r>=stop and set(target).intersection(x['columns']) and '$' in x['text']]
 if not dollars: raise ValueError('Comparison currency evidence missing')
 headers=[];seen=set()
 for c in target:
  for x in evidence[c]:
   key=(x['row'],x['physical_cell'])
   if key not in seen:headers.append(x);seen.add(key)
 return {'measure_label':current['measure_label'],'displayed_amount':display,'amount_decimal':value,
  'date':prior,'scale':current['scale'],'scale_evidence':copy.deepcopy(current['scale_evidence']),
  'currency_literal':'$','currency_iso':None,'locator':{'table_index_zero_based':ti,'row_index_zero_based':ri,'logical_columns_zero_based':target},
  'amount_cells':selected,'header_cells':headers,'currency_evidence_cells':dollars,
  'statement_title_and_unit_context':current['statement_title_and_unit_context'],'fact_admitted':False}


def publication_anchor(soup, spec):
 text=base.norm(soup.get_text(' ',strip=True))
 pattern=(r'(?P<anchor>(?:SUNNYVALE|SANTA CLARA),\s*Calif\.\s*[,\-\u2013\u2014]?\s*'
          r'(?P<day>'+base.DATE.pattern+r').{0,180}?AMD\s*\(NASDAQ\s*:\s*AMD\))')
 matches=list(re.finditer(pattern,text,re.I))
 if not matches: raise ValueError('Original dateline plus AMD ticker anchor absent')
 values={base.dates([m.group('day')])[0] for m in matches}
 if values!={spec['release_date']}: raise ValueError('Conflicting original release dateline')
 if not re.search(r'ADVANCED\s+MICRO\s+DEVICES',text,re.I): raise ValueError('Issuer legal name absent')
 return {'literal':matches[0].group('anchor'),'release_date':spec['release_date'],
  'timezone':'America/Los_Angeles','precision':'date_only','original_exact_clock_verified':False,
  'issuer_name':'Advanced Micro Devices, Inc.','ticker_literal':'AMD (NASDAQ:AMD)',
  'continuous_security_identity_verified':False}


def verify_provenance(raw, spec, receipt, events, sessions):
 if sha(raw)!=spec['raw_sha256']:raise ValueError('Raw SHA mismatch')
 url=spec['source_url'];parsed=urlsplit(url)
 if parsed.scheme!='https' or parsed.netloc!='ir.amd.com':raise ValueError('Unapproved issuer host')
 if receipt.get('status')!=200 or receipt.get('sha256')!=sha(raw):raise ValueError('Receipt status or SHA mismatch')
 if receipt.get('url')!=url or receipt.get('final_url')!=url:raise ValueError('Receipt URL mismatch')
 if receipt.get('public_unauthenticated') is not True or not receipt.get('retrieved_at'):raise ValueError('Receipt provenance incomplete')
 if 'text/html' not in receipt.get('content_type',''):raise ValueError('Non-HTML receipt')
 es=[e for e in events if e.get('ticker')=='AMD' and e.get('fiscal_year')==spec['fiscal_year'] and e.get('fiscal_quarter')==4]
 if len(es)!=1:raise ValueError('Missing or duplicate annual catalog event')
 event=es[0]
 for key,want in [('source_url',url),('evidence_path',spec['raw_path']),('evidence_sha256',sha(raw)),('release_date',spec['release_date']),('fiscal_period_end_actual',spec['fiscal_year_end'])]:
  if event.get(key)!=want:raise ValueError('Catalog event mismatch: '+key)
 if event.get('source_retrieved_at')!=receipt['retrieved_at']:raise ValueError('Retrieval evidence mismatch')
 if not sessions or sessions!=sorted(set(sessions)):raise ValueError('Invalid session sequence')
 for day in sessions:canonical_day(day)
 release=canonical_day(spec['release_date']);i=bisect.bisect_right(sessions,release.isoformat())
 if i==len(sessions):raise ValueError('No later session available')
 return sessions[i]


def scope_review(year, text):
 """Fixed source-only disposition, not learned from factors or returns."""
 notes=[];blockers=[]
 if year==2016:
  for term in ['FASB ASU 2015-17','FASB ASU 2015-03']:
   if term not in text:raise ValueError('Required FY2016 accounting scope note absent')
   i=text.index(term);notes.append(text[max(0,i-30):i+170])
  notes.append('Use FY2016 release comparison assets 3084, never overwrite FY2015 original 3109. Same-report presentation is a proposed accounting-scope assumption.')
 if year==2018:
  blockers.append('FY2017 comparison gross/assets differ from original FY2017; required accounting restatement explanation not established. Cash-flow reclassification note does not resolve these measures.')
 if year in (2022,2023):notes.append('As-reported company scope includes the Xilinx acquisition; this is not organic or constant-scope growth.')
 if year==2025 or 'discontinued operations' in text.lower():
  blockers.append('Continuing-operations gross profit versus post-divestiture assets requires explicit scope admission; do not infer comparability solely because prior comparison amounts match.')
  m=re.search(r'Loss\s*\(Income\) from discontinued operations relates to.{0,200}',text,re.I)
  if m:notes.append(m.group(0))
 if year==2020:notes.append('FY2019 comparison values are available here only after FY2020 release; do not create/backfill a missing FY2019 release event.')
 return {'comparison_candidate_status':'EXCLUDED_PENDING_COMPARABILITY' if blockers else 'SOURCE_CANDIDATE_PENDING_INDEPENDENT_ADMISSION',
  'eligible_for_candidate_review':not blockers,'admitted_for_factor_use':False,'blockers':blockers,'scope_notes':notes,
  'required_assumptions':['Current official retrieval assumed to preserve original release values; first-version authenticity unverified.',
  'Same issuer dollar-reporting labels assumed stable; ISO USD remains unresolved.',
  'As-reported GAAP-style consolidated statement scope, not constant-business/organic scope.',
  'Annual 52/53-week periods may differ in length; no annualization adjustment is performed.']}


def extract_candidate(raw, spec, receipt, events, sessions):
 first=verify_provenance(raw,spec,receipt,events,sessions)
 original=base.extract(raw,spec)
 soup=base.BeautifulSoup(raw,'html.parser')
 for node in soup(['script','style']):node.decompose()
 anchor=publication_anchor(soup,spec);prior=original['annual_period']['prior_year_end']
 current={k:copy.deepcopy(original[k]) for k in ('gross_amount','total_assets')}
 if Decimal(current['total_assets']['amount_decimal'])<=0:raise ValueError('Nonpositive current assets')
 comparison={k:comparative_cell(soup,current[k],prior,k) for k in current}
 for part in (current,comparison):
  for fact in part.values():fact['available_from']=first;fact['source_raw_sha256']=sha(raw)
 review=scope_review(spec['fiscal_year'],base.norm(soup.get_text(' ',strip=True)))
 return {'schema_version':1,'symbol':'AMD','fiscal_year':spec['fiscal_year'],
  'source':{'path':spec['raw_path'],'sha256':sha(raw),'url':spec['source_url'],'receipt':copy.deepcopy(spec['receipt'])},
  'publication_anchor':anchor,'available_from':first,'availability_rule':'All four values first usable on first pinned session strictly after THIS release; no backdated prior-year event.',
  'current_period':copy.deepcopy(original['annual_period']),
  'comparison_period':{'end':prior,'start':None,'duration_label':'Year Ended','start_not_inferred':True},
  'current':current,'comparison':comparison,'scope_review':review,
  'fact_admitted':False,'strategy_admitted':False,'historical_first_version_verified':False}


class PhysicalTables(HTMLParser):
 """Separate physical-cell reconstruction for concordance with BeautifulSoup."""
 def __init__(self):super().__init__(convert_charrefs=True);self.tables=[];self.stack=[]
 def handle_starttag(self,tag,attrs):
  if tag=='table':self.tables.append([]);self.stack.append(self.tables[-1])
  elif tag=='tr' and self.stack:self.stack[-1].append([])
  elif tag in ('td','th') and self.stack and self.stack[-1]:self.stack[-1][-1].append({'text':'','attrs':dict(attrs)})
 def handle_endtag(self,tag):
  if tag=='table' and self.stack:self.stack.pop()
 def handle_data(self,data):
  if self.stack and self.stack[-1] and self.stack[-1][-1]:self.stack[-1][-1][-1]['text']+=data+' '


def audit_candidate(raw, candidate):
 """Audit source SHA, physical cells, annual headers, dates and as-of guards.

A second parser concordance check, not a substitute for another human's semantic
source admission. It does not import the old selector or financial models.
"""
 if sha(raw)!=candidate['source']['sha256']:raise ValueError('Audit source mismatch')
 parser=PhysicalTables();parser.feed(raw.decode('utf-8'))
 checked=[]
 for part in ('current','comparison'):
  for key,fact in candidate[part].items():
   loc=fact['locator'];table=parser.tables[loc['table_index_zero_based']]
   def physical(item):
    got=re.sub(r'\s+',' ',table[item['row']][item['physical_cell']]['text']).strip()
    if got!=item['text']:raise ValueError('Second-parser physical cell differs')
    return got
   values=[physical(c) for c in fact['amount_cells']]
   joined=' '.join(x for x in values if x).strip(); normalized=re.sub(r'\s+',' ',joined)
   if normalized!=fact['displayed_amount']:raise ValueError('Audit displayed amount differs')
   # Independent integer/decimal lexical check; source numbers may be parentheses-negative.
   n=normalized.replace('$','').replace(',','').replace(' ','');negative=n.startswith('(') and n.endswith(')');n=n.strip('()')
   if not re.fullmatch(r'-?\d+(?:\.\d+)?',n):raise ValueError('Audit nonmonetary number')
   result=format(-Decimal(n) if negative else Decimal(n),'f')
   if result!=fact['amount_decimal']:raise ValueError('Audit numerical value differs')
   headers=[physical(c) for c in fact['header_cells']]
   dates=base.dates(headers)
   if dates!=[fact['date']]:raise ValueError('Audit period header mismatch')
   if key=='gross_amount' and not any(re.fullmatch(r'year\s+ended(?:\s+.*)?',s,re.I) for s in headers):raise ValueError('Audit requires annual header')
   for c in fact['currency_evidence_cells']:physical(c)
   if fact['available_from']!=candidate['available_from'] or fact['available_from']<=candidate['publication_anchor']['release_date']:raise ValueError('Comparison incorrectly backdated')
   if fact['source_raw_sha256']!=sha(raw):raise ValueError('Mixed source vintages')
   checked.append(part+'.'+key)
 if candidate['scope_review']['admitted_for_factor_use'] or candidate['fact_admitted'] or candidate['strategy_admitted']:raise ValueError('Source package cannot admit factors')
 expected=scope_review(candidate['fiscal_year'],base.norm(base.BeautifulSoup(raw,'html.parser').get_text(' ',strip=True)))
 if candidate['scope_review']!=expected:raise ValueError('Unresolved scope blocker was removed or changed')
 return {'symbol':'AMD','fiscal_year':candidate['fiscal_year'],'source_sha256':sha(raw),'four_physical_cells_and_headers_verified':checked,
  'comparison_not_backdated':True,'same_source_vintage':True,'excluded_scope_preserved':True,'passed':True}


def bundle():
 bindings={}
 def read(path,expected=None):
  data=local(path).read_bytes();digest=sha(data)
  if expected is not None and digest!=expected:raise ValueError('Pinned input changed: '+path)
  bindings[path]=digest;return data
 docs={path:read(path,s) for path,s in PINNED.items()}
 plan=json.loads(docs['outputs/monthly_profitability_extension_sources_v1/EXTRACTION_PLAN.json'])
 events=json.loads(docs['outputs/monthly_earnings_repair/batch_technology_industrial/AMD/events.json'])
 sessions=json.loads(docs['outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'])['sessions']
 specs=[s for s in plan['sources'] if s['symbol']=='AMD']
 if tuple(sorted(s['fiscal_year'] for s in specs))!=YEARS:raise ValueError('Unexpected annual source coverage')
 original=json.loads(docs['outputs/monthly_profitability_extension_sources_v1/RAW_FACTS.json'])
 old={r['fiscal_year']:r for r in original['rows'] if r['symbol']=='AMD'}
 rows=[];audits=[]
 for spec in sorted(specs,key=lambda x:x['fiscal_year']):
  raw=read(spec['raw_path'],spec['raw_sha256']);receipt=json.loads(read(spec['receipt']['path'],spec['receipt']['sha256']))
  row=extract_candidate(raw,spec,receipt,events,sessions)
  for k in ('gross_amount','total_assets'):
   if row['current'][k]['amount_decimal']!=old[spec['fiscal_year']][k]['amount_decimal']:raise ValueError('Original current-year extraction changed')
  rows.append(row);audits.append(audit_candidate(raw,row))
 for path in (SELF,TEST,'run_local.sh','local_run.py','local.sandbox.sb',OUT+'/TESTS.log'):read(path)
 for path,s in bindings.items():
  if sha(local(path).read_bytes())!=s:raise ValueError('Input changed during source review: '+path)
 data={'schema_version':1,'status':'SOURCE_ONLY_CANDIDATES_NOT_ADMISSION','source_scope':'10 original AMD annual earnings releases; same-release current and comparative columns',
  'rows':rows,'missing_original_releases':[{'fiscal_year':2019,'available_from':None,'reason':'Original FY2019 release missing from reviewed local catalog. FY2020 comparative is not an earlier event.'}],
  'candidate_years':[r['fiscal_year'] for r in rows if r['scope_review']['eligible_for_candidate_review']],
  'excluded_years':[r['fiscal_year'] for r in rows if not r['scope_review']['eligible_for_candidate_review']],
  'new_factor_admissions':0,'strategy_changes':0,'ratios_factors_scores_orders_returns_computed':0,'network_requests':0}
 audit={'schema_version':1,'status':'PASS_REPRODUCIBLE_EXTRACTION_CONCORDANCE_NOT_SEMANTIC_ADMISSION','rows':audits,
  'current_values_match_prior_extraction':True,'raw_statement_cells_checked':40,'input_files_unchanged':len(bindings),
  'independent_human_admission_still_required':True,'limitations':['Second parser shares the explicit date grammar with the frozen parser; it is not an independent semantic source reviewer.',
   'ISO currency and historical first-public-version remain unverified.','No synthetic historical FY2019 event; eventual event-selection policy must handle missing chronology and disallow stale carry.',
   'FY2018 and FY2025 remain excluded until their accounting-scope blockers are separately resolved.']}
 return data,audit,bindings


def execute(verify=False):
 data,audit,bindings=bundle();folder=local(OUT)
 payloads={'CANDIDATES.json':data,'AUDIT.json':audit}
 if verify:
  manifest=json.loads((folder/'MANIFEST.json').read_text())
  if manifest['input_sha256']!=bindings:raise ValueError('Manifest input mismatch')
  for name,obj in payloads.items():
   raw=(folder/name).read_bytes()
   if sha(raw)!=manifest['output_sha256'][name] or json.loads(raw)!=obj:raise ValueError('Reproduced candidate output differs: '+name)
  print(json.dumps({'status':'REPRODUCED_EXACT_CANDIDATE_DATA','pairs':len(data['rows']),'candidates':data['candidate_years'],'excluded':data['excluded_years']}));return
 if any((folder/name).exists() for name in (*payloads,'MANIFEST.json')):raise ValueError('Refusing to overwrite source package')
 folder.mkdir(parents=True,exist_ok=True);out_hashes={}
 for name,obj in payloads.items():
  raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();(folder/name).write_bytes(raw);out_hashes[name]=sha(raw)
 manifest={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'SOURCE_EXTRACTION_ONLY_NOT_ADMISSION','input_sha256':bindings,'output_sha256':out_hashes,'network_requests':0,'factor_or_return_computations':0}
 (folder/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':manifest['status'],'pairs':len(data['rows']),'candidates':data['candidate_years'],'excluded':data['excluded_years'],'manifest_sha256':sha((folder/'MANIFEST.json').read_bytes())}))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args();execute(args.verify)
