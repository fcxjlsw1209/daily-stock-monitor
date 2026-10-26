"""Source-only three-own-annual-report feasibility. No ratios or strategy work.

Only unit-normalized equality of already reported monetary cells is computed.
No monthly query grid, price, model, financial ratio or portfolio is loaded.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
from after149_batch1_independent_review import grid_from_html, parsed_dates, signed_text, norm, ANNUAL, SHORT, header_facts, external_heading

ROOT = Path(__file__).resolve().parents[1]
SELF = 'work/after219_roa_stability_source_preflight_v2.py'
OUT = 'work/after219_roa_stability_source_preflight_v2.json'
CONTEXT = 'work/after219_roa_stability_context_v2.json'
CONTEXT_SHA = 'a713c6ce62c8a6ec9608b495d8cbfbe19b474f4d9d9f40a3d3dd7acd27cf5e55'
BASE = 'outputs/monthly_cash_quality_source_admission_v1'
DATASETS = [BASE+'/DATASET.json'] + [f'outputs/monthly_financial_coverage_admission_batch{i}_v1/DATASET.json' for i in range(1,6)]
EXPECTED = ['3725b384ab4f48a3cabee1fc1adb1acd896be3f4967e8b6cd649a9a690c953d5', 'b9096e587cc6460b4007fe85cddd6a0ebfbbd211736a31fb899405b5f02fb4f3','04a2d2410a35db0a02034fab75b781cd43e5f52f569fb5272f34f3219cee2b33','7a74e1d529ea0f2c36d73dead551582cb6dc157373dec6369ecd6576b5e6c719','8605fc0e0520f39591ed9ff3aa82a75abfbb14f6f57d3cf865e83e5a853db324','8358d49a9d18a2fd072746cfafdecb63eaa7e6013b3a625ca59a210b06073711']
ACCOUNTING = 'work/cashflow_accounting_review_v1.json'
PUNCT = re.compile(r'[\s$()\-\u2212]+')
# Evidence-specific source findings, not rules inferred from a term search.
KNOWN_EDGE_ISSUES = {
 'MSFT:FY2018:2018-07-19': 'Explicit retrospective revenue-recognition and lease adoption; do not mix FY2017 original NI/assets with this restated presentation.',
 'TMO_2019-01-30': 'Current release explicitly restates FY2017 for pension accounting. Existing field admission did not certify unchanged historical measurement/assets.',
 'PEP_2016-02-11': 'Venezuela deconsolidation and impairment change consolidated business perimeter. Single-report admission retained this economics; a stable cross-report perimeter is not established.',
 'PYPL_2016-01-27': 'FY2015 combined/carve-out basis following eBay separation needs explicit cross-report treatment.',
}


def require(x, message):
 if not x: raise ValueError(message)

class Reader:
 def __init__(self): self.hashes={}; self.raw={}
 def read(self,path,expected=None):
  p=Path(path); p=p if p.is_absolute() else ROOT/p
  require('CloudStorage' not in p.parts and not p.is_symlink(),'Forbidden source location')
  p=p.resolve(); key=str(p.relative_to(ROOT)); raw=p.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
  require(expected is None or sha==expected,'SHA mismatch '+key)
  require(key not in self.hashes or self.hashes[key]==sha,'Input changed '+key)
  self.hashes[key]=sha; self.raw[key]=raw; return raw
 def load(self,path,expected=None): return json.loads(self.read(path,expected))


def money(cell):
 """Unit-normalized source amount, solely for equality; not a financial ratio."""
 require(cell is not None,'Missing source amount')
 v,s=cell['value_decimal'],cell['scale']
 require(type(v) is str and type(s) is str and Decimal(v).is_finite() and Decimal(s).is_finite() and Decimal(s)>0,'Explicit decimal/scale')
 return Fraction(v)*Fraction(s)


def brief_cell(cell, field_path, source):
 if cell is None:return None
 return dict(field_path=field_path,value_decimal=cell['value_decimal'],scale=cell['scale'],
  currency_literal=cell.get('currency_literal'),currency_iso=cell.get('currency_iso'),
  source=source,as_of=cell.get('as_of'),period=cell.get('period'),
  locator=cell.get('locator'),row_text=cell.get('row_text'),displayed_amount=cell.get('displayed_amount'))


def normalized(event,batch,review,reader):
 src=event['source'];raw=reader.read(src['path'],src['sha256'])
 receipt=src.get('receipt',{})
 rp=receipt if isinstance(receipt,str) else receipt.get('path',src.get('receipt_path'));rs=src.get('receipt_sha256') if isinstance(receipt,str) else receipt.get('sha256',src.get('receipt_sha256'))
 if rp:reader.read(rp,rs)
 if batch==0:
  f=event['fields']; ni=f['current']['net_income']; ca=f['current']['total_assets']; pa=f['comparison']['total_assets']; comp=f['comparison']['net_income']; period=event['periods']['current']
  keys=['fields.current.net_income','fields.current.total_assets','fields.comparison.total_assets']; gate=event['cash_quality_admission']
 else:
  f=event['fields'];ni=f['annual_net_income'];ca=f['current_total_assets'];pa=f['prior_total_assets'];comp=None;period=ni['period'];keys=['fields.annual_net_income','fields.current_total_assets','fields.prior_total_assets'];gate=event['ni_assets_admission']
 identity=event['identity'];require(gate['admitted'] is True and gate['reasons']==[],'Only existing admitted facts')
 require(identity['security_id']==event['symbol'],'Source security mismatch')
 release=event.get('release_date',event.get('publication_date')); known=event['available_from']
 ni_review=(review or {}).get('fields',{}).get('annual_net_income',{})
 r=dict(event_id=event['event_id'],symbol=event['symbol'],issuer_id=identity['issuer_id'],security_id=identity['security_id'],
  identity_evidence=identity,source_dataset=DATASETS[batch],source=src,batch=batch,fiscal_year=event['fiscal_year'],
  release_date=release,assumed_known_session=known,known_at_scope='original_document_date_plus_first_pinned_next_session_assumption_not_certified_first_public_time',
  period=period,first_public_version_verified=False,complete_correction_history_verified=False,
  declared_field_admission=gate,ni=brief_cell(ni,keys[0],src),assets_current=brief_cell(ca,keys[1],src),assets_prior=brief_cell(pa,keys[2],src),
  stored_comparative_ni=brief_cell(comp,'fields.comparison.net_income',src),
  original_document_retrieved_later=True,source_bytes_verified=True,
  same_report_accounting_review=(review or {}).get('accounting_review'),accounting_review_flags=event.get('accounting_review_flags',[]),
  field_review_source='work/cashflow_accounting_review_v1.json' if batch==0 else f'work/after{148+batch}_batch{batch}_independent_review.json',
  ni_scope='Source-specific GAAP total NI selected by inherited single-report field review, not a new cross-report attribution certification',
  field_scope_status='inherited_single_report_NI_and_asset_review_preserved; cross_report_scope_not_admitted',
  source_review_passed=None if review is None else review.get('passed'),
  current_ni_reading=ni_review.get('independent_source_reading'))
 require(all(x['source']['sha256']==src['sha256'] and x['source']['path']==src['path'] for x in (ni,ca,pa)),'Cross-source field')
 return r


def comparative_html(record,reader,prior_end):
 """Read only a prior-year monetary witness from the same admitted NI row.

 Date-group geometry is bound to the existing independent review; batch1
 rediscovery uses exact annual/date headers. A failed check remains unknown.
 """
 rd=record['current_ni_reading'];require(isinstance(rd,dict),'No prior independent NI row geometry')
 raw=reader.read(record['source']['path'],record['source']['sha256'])
 soup=BeautifulSoup(raw,'html.parser');tables=soup.find_all('table');table=tables[rd['table']]
 require(not table.find('table'),'Nested HTML statement')
 cells,grid=grid_from_html(table);row=rd['row'];ordered=sorted((c for c in cells.values() if c['row']==row),key=lambda c:c['physical_cell'])
 groups=rd.get('all_date_groups')
 if groups is None:
  stop=rd['header_end_row'];headers=[c for c in cells.values() if c['row']<stop];groups={}
  for col in sorted({x for c in headers for x in c['columns']}):
   txt=' '.join(c['text'] for c in headers if col in c['columns'])
   if not ANNUAL.search(txt) or SHORT.search(txt):continue
   ds=parsed_dates(txt)
   if len(ds)==1:groups.setdefault(next(iter(ds)),[]).append(col)
  require(prior_end in groups,'Prior full-year date not uniquely discoverable')
  _,_,_,scale=header_facts(cells,row,groups[prior_end],'annual_net_income',prior_end,external_heading(table))
  require(scale==rd['scale'],'Comparative units differ from independent row')
 require(prior_end in groups,'No prior annual date group')
 cols=groups[prior_end];require(cols,'No prior columns')
 ownership={}
 domain=set(range(min(min(v) for v in groups.values()),max(max(v) for v in groups.values())+1))
 # The existing batch5 auditor excludes adjacent percent-of-revenue columns.
 header_end=max((c['row'] for c in rd.get('header_cells',rd.get('headers',[]))),default=-1)
 excluded_columns=set()
 for c in cells.values():
  if c['row']<=header_end and (c['text'].startswith('%') or c['text'].lower()=='revenues'):
   excluded_columns.update(c['columns'])
 require(not any(excluded_columns & set(v) for v in groups.values()),'Nonmonetary header overlaps dated amount')
 domain.difference_update(excluded_columns)
 for col in domain:
  ds={d:min(abs(col-x) for x in cs) for d,cs in groups.items()};near=[d for d,v in ds.items() if v==min(ds.values())];ownership[col]=near[0] if len(near)==1 else None
 chosen=[c for c in ordered if set(c['columns'])&set(cols)];require(chosen,'No comparative cells')
 lo,hi=ordered.index(chosen[0]),ordered.index(chosen[-1])
 for ix in (lo-1,hi+1):
  if 0<=ix<len(ordered):
   c=ordered[ix]
   if c['text'] and PUNCT.fullmatch(c['text']) and any(ownership.get(x)==prior_end for x in c['columns']):
    require(all(ownership.get(x)==prior_end for x in c['columns']),'Ambiguous adjacent punctuation');chosen.append(c)
 chosen.sort(key=lambda c:c['physical_cell'])
 for c in chosen:require(c['rowspan']==1 and all(ownership.get(x)==prior_end for x in c['columns']),'Cross-year/row source witness')
 for c in ordered:
  if c not in chosen and c['text'] and any(ownership.get(x)==prior_end for x in c['columns']):raise ValueError('Unconsumed same-year monetary cell')
 displayed=' '.join(c['text'] for c in chosen if c['text']);value=signed_text(displayed)
 return dict(value_decimal=value,scale=rd['scale'],currency_literal='$',as_of=prior_end,
  source=record['source'],locator=dict(table=rd['table'],row=row,date_columns=cols,physical_cells=chosen),
  source_review=record['field_review_source'],displayed_amount=displayed,
  annual_column_scope='same_existing_reviewed_consolidated_NI_row_and_annual_date_group',
  reextracted_for_source_equality_only=True,monetary_witness_not_new_feature=True)


def compare(a,b):
 if a is None or b is None:return dict(status='unknown',reason='missing_original_or_comparative_cell')
 require(a.get('currency_literal')==b.get('currency_literal')=='$','Inconsistent within-report literal currency')
 return dict(status='equal' if money(a)==money(b) else 'different',original=dict(value_decimal=a['value_decimal'],scale=a['scale'],source=a['source']),
  later_comparative=dict(value_decimal=b['value_decimal'],scale=b['scale'],source=b['source']),
  method='exact_scaled_decimal_source_amount_equality_no_ratio')


def document_notes(record,reader):
 if record['batch']==0:return []
 raw=reader.read(record['source']['path'],record['source']['sha256']);soup=BeautifulSoup(raw,'html.parser')
 for tag in soup(['script','style']):tag.decompose()
 text=norm(soup.get_text(' ',strip=True));pat=re.compile(r'restat(?:e|ed|ement)|retrospect|recast|deconsolidat|carve.out|spin.off|spun.off|change.{0,35}accounting|adopt.{0,80}(?:revenue|lease|pension)',re.I)
 notes=[]
 for m in pat.finditer(text):
  lo=max(0,m.start()-150);hi=min(len(text),m.end()+300);quote=text[lo:hi]
  if any(abs(lo-x['start'])<180 for x in notes):continue
  notes.append(dict(start=lo,end=hi,quote=quote,source=record['source'],
   interpretation='context_lead_only_not_automatic_rejection_or_clearance',
   assumed_known_session=record['assumed_known_session']))
 return notes[:20]


def main():
 reader=Reader();reader.load(CONTEXT,CONTEXT_SHA);reader.read(SELF);reader.read('work/after219_roa_stability_source_preflight.py','a8406aa86556ba881382c0a830275ed0a592b7a50e8f9bfdf8259e7656e8e068');reader.read('work/after219_roa_stability_source_preflight.json','e34d6d3ec81a39dae589c67c1751f194a82a4edab835d3b973e9c0298ac20cec');reader.read('work/after219_roa_stability_source_preflight_initial_context_v1.json');reader.read('work/after219_roa_stability_source_preflight_initial_context_v1.py');reader.read('work/after149_batch1_independent_review.py')
 sessions=reader.load('outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json')['sessions'];require(sessions==sorted(set(sessions)),'Pinned calendar')
 accounting=reader.load(ACCOUNTING);cashreviews={e['event_id']:e for e in accounting['events']};records=[];raw_catalogs=[];saved_counts=[]
 for batch,(path,digest) in enumerate(zip(DATASETS,EXPECTED)):
  data=reader.load(path,digest);base=str(Path(path).parent);admission=reader.load(base+'/ADMISSION.json')
  if batch==0:
   reader.load(base+'/POLICY.json');reviews=cashreviews;gate='cash_quality_admission'
  else:
   reviewpath=f'work/after{148+batch}_batch{batch}_independent_review.json';r=reader.load(reviewpath,data['independent_review_sha256']);reviews={e['event_id']:e for e in r['event_reviews']}
   reader.load(f'work/after{148+batch}_batch{batch}_admission_policy.json',data['policy_sha256']);reader.load(base+'/MANIFEST.json');gate='ni_assets_admission'
  raw_catalogs.append(dict(source_dataset=path,catalog=data.get('latest_event_catalog') if batch else [{k:e.get(k) for k in ('event_id','symbol','fiscal_year','release_date','available_from','periods','cash_quality_admission','inherited_rejection_codes')} for e in data['events']],scope='complete_existing_package_event_catalog_including_held_and_unprocessed_not_only91; no_monthly_signal_matching', worldwide_exhaustive_release_catalog_certified=False))
  valid=[e for e in data['events'] if e.get(gate,{}).get('admitted') is True]
  saved_counts.append(dict(dataset=path,existing_admitted_events=len(valid)))
  for e in valid:
   print('source',e['event_id'],flush=True)
   record=normalized(e,batch,reviews.get(e['event_id']),reader)
   release=record['release_date'];pos=bisect_right(sessions,release)
   require(0<pos<len(sessions) and sessions[pos]==record['assumed_known_session'],'Clock not first pinned next session')
   require(record['period']['period_end']<release<record['assumed_known_session'],'Source chronology')
   record['next_session_clock_rechecked']=True
   record['same_report_accounting_findings']=[x for x in accounting['findings'] if e['event_id'] in x['event_ids']]
   record['document_accounting_context_leads']=document_notes(record,reader)
   records.append(record)
 require(len(records)==91,'Expected existing91 admitted events')
 identities=defaultdict(list)
 for record in records:identities[record['issuer_id'],record['security_id']].append(record)
 require(len(identities)==15,'Expected15 distinct source issuer/security identity pairs')
 edges=[];chains=[];gaps=[];issuers=[]
 for (issuer,security),rows in sorted(identities.items()):
  rows.sort(key=lambda r:r['fiscal_year']);years={}
  for r in rows:require(r['fiscal_year'] not in years,'Ambiguous own-year source');years[r['fiscal_year']]=r
  local_edges={}
  for year,later in sorted(years.items()):
   if year-1 not in years:continue
   earlier=years[year-1];prior_end=earlier['period']['period_end'];start=later['period']['period_start']
   contiguous=date.fromisoformat(prior_end)+timedelta(days=1)==date.fromisoformat(start)
   eid=f'{issuer}|{security}|FY{year-1}->FY{year}'
   ni_witness=None;ni_error=None
   if later['batch']==0:ni_witness=later['stored_comparative_ni']
   else:
    try:ni_witness=comparative_html(later,reader,prior_end)
    except (ValueError,KeyError,TypeError,IndexError) as exc:ni_error=type(exc).__name__+': '+str(exc)
   ni_comparison=compare(earlier['ni'],ni_witness);assets_comparison=compare(earlier['assets_current'],later['assets_prior'])
   problems=[]
   if not contiguous:problems.append('noncontiguous_annual_periods')
   if ni_comparison['status']=='different':problems.append('reported_NI_comparative_differs_from_original')
   if assets_comparison['status']=='different':problems.append('reported_asset_comparative_differs_from_original')
   if later['event_id'] in KNOWN_EDGE_ISSUES:problems.append('explicit_accounting_or_perimeter_issue_requires_cross_report_resolution')
   numeric_checked=contiguous and ni_comparison['status']=='equal' and assets_comparison['status']=='equal'
   edge=dict(edge_id=eid,issuer_id=issuer,security_id=security,earlier_event_id=earlier['event_id'],later_event_id=later['event_id'],
    earlier_fiscal_year=year-1,later_fiscal_year=year,identity_exact_match=True,annual_boundaries_contiguous=contiguous,
    assumed_known_session=max(earlier['assumed_known_session'],later['assumed_known_session']),
    known_at_scope='not_before_both_original_dated_sources; retrospective_collection_time_preserved_in_source_receipts',
    ni_original_vs_later_comparison=ni_comparison,assets_original_vs_later_comparison=assets_comparison,comparative_ni_witness=ni_witness,comparative_read_error=ni_error,
    known_accounting_issue=KNOWN_EDGE_ISSUES.get(later['event_id']),
    numeric_boundary_status='checked_reported_NI_and_assets_equality' if numeric_checked else 'mismatch' if problems else 'unknown',
    source_chain_status='ineligible_pending_cross_report_resolution' if problems else 'unknown_cross_report_scope_not_independently_admitted',
    reasons=problems or ['same_report_field_admission_is_not_three_report_accounting_scope_admission'],
    source_report_dates_not_backdated=True,first_public_version_verified=False,complete_correction_history_verified=False,
    can_use_for_factor=False,required_remaining_evidence=['Independent cross-report total-NI attribution and consolidated-perimeter review tied to both report SHA values.',
     'Explicit disposition of source accounting notes; equality alone does not prove unchanged method or economic perimeter.',
     'A source-use policy must retain first-public-version/correction-history uncertainty; no certified PIT claim.'])
   # A/B/C is a source-preflight diagnosis, never a change in admission.
   explicit_conflicts=[x for x in problems if x != 'explicit_accounting_or_perimeter_issue_requires_cross_report_resolution']
   classification=('A_explicit_source_conflict' if explicit_conflicts else
    'B_reported_fields_match_conditional_scope_only' if numeric_checked else
    'C_missing_or_unresolved_source_evidence')
   scope_alerts=[dict(event_id=r['event_id'],description=KNOWN_EDGE_ISSUES[r['event_id']],
    assumed_known_session=r['assumed_known_session'],source=r['source']) for r in (earlier,later) if r['event_id'] in KNOWN_EDGE_ISSUES]
   edge.update(preflight_class=classification,explicit_conflicts=explicit_conflicts,
    broad_scope_alerts=scope_alerts,
    accounting_admission_known_at=None,accounting_admission_completed=False,
    numeric_comparison_evidence=[dict(event_id=r['event_id'],source=r['source'],
     available_from=r['assumed_known_session']) for r in (earlier,later)],
    numeric_comparison_assumed_known_session=max(earlier['assumed_known_session'],later['assumed_known_session']),
    numeric_comparison_known_at_scope='Only own-year monetary cells and later same-report comparative cells. Footnote/correction leads are separately retained; they do not clear the edge.',
    evidence_version_created_utc=datetime.now(timezone.utc).isoformat(),
    historical_admission_timeline_constructed=False,
    inherited_same_report_NI_attribution_checks_preserved=True,
    NI_cross_report_scope_admitted=False,asset_cross_report_scope_admitted=False,
    complete_first_publication_vintage_verified=False,
    next_action=('Resolve exact source conflicts before any factor use.' if classification.startswith('A_') else
     'Eligible only for future independent consideration of explicitly limited as-reported same-company research; not admitted by this preflight.' if classification.startswith('B_') else
     'Obtain missing or unambiguous source evidence; no stale or inferred replacement.'))
   edges.append(edge);local_edges[year]=edge
  for year,last in sorted(years.items()):
   if year<min(years)+2:continue
   needed=[year-2,year-1,year]
   if not all(y in years for y in needed):
    gaps.append(dict(issuer_id=issuer,security_id=security,ending_fiscal_year=year,required_years=needed,missing_admitted_years=[y for y in needed if y not in years],status='ineligible_missing_contiguous_admitted_own_year',fallback_permitted=False));continue
   members=[years[y] for y in needed];es=[local_edges[year-1],local_edges[year]]
   blocked=any(e['source_chain_status'].startswith('ineligible') for e in es)
   chains.append(dict(chain_id=f'{issuer}|{security}|FY{year-2}-{year}',issuer_id=issuer,security_id=security,
    fiscal_years=needed,event_ids=[r['event_id'] for r in members],edge_ids=[e['edge_id'] for e in es],
    sources=[r['source'] for r in members],earliest_assumed_known_session=max(r['assumed_known_session'] for r in members),
    literal_comparison_evidence_not_after_chain_assumed_known_session=True,
    accounting_admission_evidence_complete=False,accounting_admission_known_at=None,
    candidate_structural_three_report_chain=True,numeric_edges_checked=all(e['numeric_boundary_status']=='checked_reported_NI_and_assets_equality' for e in es),
    status='ineligible_pending_cross_report_resolution' if blocked else 'unknown_cross_report_review_required',can_use_for_factor=False,
    actual_query_support_not_computed=True,known_future_restated_numbers_never_substituted=True,
    missing_evidence=['Independent NI/assets edge-specific accounting admission; existing one-report admission does not establish this.',
     'Decision-date latest-catalog blocking and age test must later be performed; no monthly candidate matching was run.']))
   cls=('A_explicit_source_conflict' if any(e['preflight_class'].startswith('A_') for e in es) else
    'C_missing_or_unresolved_source_evidence' if any(e['preflight_class'].startswith('C_') for e in es) else
    'B_reported_fields_match_conditional_scope_only')
   chains[-1].update(preflight_class=cls,edge_preflight_classes=[e['preflight_class'] for e in es],
    broader_scope_alert_event_ids=sorted(set(a['event_id'] for e in es for a in e['broad_scope_alerts'])),
    can_be_considered_for_limited_as_reported_research=cls.startswith('B_'),
    admission_decision_deferred=True)

  issuers.append(dict(issuer_id=issuer,security_id=security,symbol_aliases=sorted(set(r['symbol'] for r in rows)),
   admitted_fiscal_years=sorted(years),admitted_events=len(rows),identity_scope='exact_source_issuer_id_plus_security_id_not_ticker_only; no_alias_inference',
   source_identity_certified=False))
 result=dict(schema_version=2,stage='SOURCE_ONLY_THREE_REPORT_ROA_STABILITY_PREFLIGHT_NOT_STRATEGY_ADMISSION',created_utc=datetime.now(timezone.utc).isoformat(),
  context=dict(path=CONTEXT,sha256=CONTEXT_SHA),preserved_previous_run=dict(path='work/after219_roa_stability_source_preflight.json',sha256=reader.hashes['work/after219_roa_stability_source_preflight.json'],implementation_path='work/after219_roa_stability_source_preflight.py',implementation_sha256=reader.hashes['work/after219_roa_stability_source_preflight.py'],run_session='90624',run_exit_code=0,relation='Earlier context-v2-bound source-only run preserved in place. This explicit v2 adds actionable A/B/C and separates literal-evidence clocks from unresolved accounting admission. No prior run is claimed to validate this script.'),superseded_preflight=dict(path='work/after219_roa_stability_source_preflight_initial_context_v1.json',sha256=reader.hashes['work/after219_roa_stability_source_preflight_initial_context_v1.json'],reason='Preserved initial context v1 run; v2 adds full existing catalogs and explicit source clocks; no original records overwritten'),source_scope='Existing15issuers91admittedannualevents only; no newly admitted reports',
  checks=dict(source_amount_equalities_only=True,ratio_calculations=0,ROA_calculations=0,means=0,standard_deviations=0,
   factor_calculations=0,ranks=0,orders=0,returns=0,monthly_support_matches=0,network_requests=0,observer_changes=0),
  implementation_sha256=reader.hashes[SELF],input_sha256=dict(sorted(reader.hashes.items())),existing_package_counts=saved_counts,
  counts=dict(issuer_identity_pairs=len(issuers),reports=len(records),consecutive_year_edges=len(edges),candidate_three_report_chains=len(chains),missing_year_chains=len(gaps),
   edge_numeric_status=dict(Counter(e['numeric_boundary_status'] for e in edges)),chain_status=dict(Counter(c['status'] for c in chains)),
   numerically_checked_chains=sum(c['numeric_edges_checked'] for c in chains),factor_admitted_chains=0,
   edge_preflight_classes=dict(Counter(e['preflight_class'] for e in edges)),
   chain_preflight_classes=dict(Counter(c['preflight_class'] for c in chains)),
   catalog_rows_including_held_and_unprocessed=sum(len(x['catalog']) for x in raw_catalogs)),
  issuers=issuers,reports=records,edges=edges,candidate_chains=chains,missing_chain_members=gaps,source_catalogs=raw_catalogs,
  retained_limitations=['Annual-release documents retrieved retrospectively, not certified first-public bytes.',
   'Never treat current cached companyfacts or later comparative NI as an older original-year observation.',
   'A later disclosed correction may block only once that evidence is assumed known; it cannot be applied backward.',
   'Same-report consolidated total NI/average assets admission does not establish cross-report invariance.',
   'This preflight does not register an experiment or certify source/support eligibility.'],
  class_definition=dict(A='Literal source amounts or annual boundaries explicitly conflict. No attribution/asset correction is guessed.',B='NI and asset comparative amounts match across contiguous exact-identity annual records, with inherited single-report field interpretation; no claim of constant economic perimeter, unchanged global accounting policy, or first-publication PIT. Can be considered later for limited as-reported research only.',C='At least one necessary source witness is missing, ambiguous, or unparsed; no amount inferred.'),
  catalog_policy='Preserve all existing package catalog records including held/unprocessed/clock-unverified. A later consumer must select latest before value validity, block fallback, and test550days from latest fiscal end; this source-only preflight does not run signal eligibility.',
  recommended_next_step='Independent edge-specific NI/assets accounting review of numerically checked chains; resolve or exclude mismatch/basis-break edges before prospective factor registration.')
 out=ROOT/OUT;require(not out.exists(),'Write-once preflight already exists')
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(output=OUT,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),counts=result['counts']),ensure_ascii=False))

if __name__=='__main__':main()
