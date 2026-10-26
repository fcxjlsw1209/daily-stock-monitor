"""Record a source-only accounting review; no monetary arithmetic or admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
OUT='work/cashflow_accounting_review_v1.json'
BASE='outputs/monthly_cashflow_source_preflight_v1/'
PLAN_SHA='4161a752cfac3039ea733b000cab647b8c77c48562ef6f497d109aefc7be1472'
APPLE_SHA='1610b77c1ffa81ceb9727eeaf70b84a1b13feb4370fd47eb93f793142b79a26f'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p,expected=None):
 if expected:assert sha(p)==expected,(p,'hash')
 return json.loads((ROOT/p).read_bytes())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--html-review',required=True);ap.add_argument('--html-sha',required=True);a=ap.parse_args()
 plan=read(BASE+'PLAN.json',PLAN_SHA);facts=read(BASE+'RAW_FACTS.json');apple=read(BASE+'APPLE_VISUAL_REVIEW.json',APPLE_SHA);html=read(a.html_review,a.html_sha)
 assert facts['plan_sha256']==PLAN_SHA
 assert len(plan['events'])==len(facts['events'])==44
 assert not any(facts[k]for k in ('fact_admissions','model_admissions','strategy_admissions','financial_calculations','ratios_computed','signs_computed','ranks_computed','returns_computed'))
 sources={p:sha(p)for p in [BASE+'PLAN.json',BASE+'SCHEMA.json',BASE+'RAW_FACTS.json',BASE+'AUDIT.json',BASE+'MANIFEST.json',BASE+'APPLE_VISUAL_REVIEW.json',a.html_review,'work/after145_cashflow_experiment_concept.json','work/after144_msft_nvda_source_review.json','work/after142_amd_comparative_independent_review.json','work/cashflow_accounting_review_v1_probe.py','work/cashflow_accounting_review_v1_record.py']}
 events={e['event_id']:e for e in plan['events']};fe={e['event_id']:e for e in facts['events']};text={}
 for e in plan['events']:
  s=e['source'];assert sha(s['path'])==s['sha256'];sources[s['path']]=s['sha256']
  raw=(ROOT/s['path']).read_bytes()
  if not raw.startswith(b'%PDF-'):
   soup=BeautifulSoup(raw,'html.parser')
   for n in soup(['script','style']):n.decompose()
   text[e['event_id']]=' '.join(soup.get_text(' ',strip=True).split())
 def ids(symbol,*years):return [e['event_id']for e in plan['events']if e['symbol']==symbol and e['fiscal_year']in years]
 def ev(event_id,quote,occurrence=0):
  body=text[event_id];start=-1
  for _ in range(occurrence+1):start=body.find(quote,start+1);assert start>=0,(event_id,quote)
  s=events[event_id]['source'];return dict(path=s['path'],sha256=s['sha256'],locator='normalized visible text chars %d..%d'%(start,start+len(quote)),quote=quote)
 def evi(symbol,year,quote):return ev(ids(symbol,year)[0],quote)
 def parity(symbol,newyear,oldyear):
  later=ids(symbol,newyear)[0];earlier=ids(symbol,oldyear)[0];result={}
  for field in ('operating_cash_flow','net_income','total_assets'):
   l=fe[later]['fields']['comparison'][field];r=fe[earlier]['fields']['current'][field]
   result[field]=dict(status='not_comparable_missing_candidate'if l is None or r is None else 'identical_literal_and_scale'if(l['value_decimal'],l['scale'])==(r['value_decimal'],r['scale'])else'different_literal_or_scale',later_comparison=None if l is None else dict(value=l['value_decimal'],scale=l['scale'],locator=l['locator']),earlier_current=None if r is None else dict(value=r['value_decimal'],scale=r['scale'],locator=r['locator']))
  return dict(later_event=later,earlier_event=earlier,fields=result,method='Equality of independently physically reviewed source candidates only; no subtraction, ratio or economic normalization.')
 findings=[]
 def add(key,which,status,scope,current,comparison,remaining,evidence):
  findings.append(dict(id=key,event_ids=which,accounting_status=status,scope=scope,current_concept_fields_effect=current,comparison_flow_effect=comparison,remaining_limitations=remaining,evidence=evidence,source_admission_granted=False))
 add('MSFT2018_FULL_RETROSPECTIVE',ids('MSFT',2018),'same_report_basis_explicitly_reconciled',
  'Revenue-recognition and lease standards adopted July 1, 2017. The release explicitly restates the prior periods presented; this supports comparing current and prior columns on this report basis, including assets and cash-flow reconciliation NI.',
  'Current CFO/NI and both assets endpoints can be considered together under the disclosed same-current-report basis once all exact cells are available. Current NI is still null in frozen v1 due to displayed internal spaces; no substitute is made here.',
  'Comparative CFO is identical to the earlier original while comparative NI and assets differ. This demonstrates that unchanged CFO does not prove all measures were unchanged.',
  ['Do not mix 2017 original NI/assets with 2018 restated columns.','Do not backdate revised comparisons before the current release.','Same-report comparability is not first-public-version certification.'],
  [evi('MSFT',2018,'We adopted new accounting standards related to revenue recognition and leases effective July 1, 2017. The prior periods presented here have been restated to reflect adoption of these new standards.')])
 add('MSFT2023_SEGMENT_RECAST_AND_ESTIMATE',ids('MSFT',2023),'segment_recast_limited_clearance_with_estimate_warning',
  'The recast footnote is attached to segment presentation; all three selected comparative consolidated totals equal their prior-report current totals. A separate useful-life estimate change affects depreciation and reported earnings.',
  'No selected-total mismatch from the segment recast is observed. Current NI and year-end assets still embody the depreciation estimate; current CFO is a reconciliation including noncash adjustments, not a guarantee of an estimate-free metric.',
  'Selected prior consolidated CFO/NI/assets match older printed totals; segment-component continuity is not certified.',
  ['Do not infer that CFO or a future CFO-minus-NI quantity is unaffected by estimates.','No constant-useful-life normalization is supported or performed.'],
  [evi('MSFT',2023,'We have recast certain prior period amounts to conform to the way we internally manage and monitor our business.'),evi('MSFT',2023,'change in accounting estimate for the useful lives of our server and network equipment')])
 add('MSFT2025_SEGMENT_RECAST',ids('MSFT',2025),'segment_recast_limited_clearance',
  'The recast footnote concerns internally managed business presentation. Selected consolidated prior CFO/NI/assets match the earlier current-report totals.',
  'No numeric conflict is observed for the current-concept fields because of this segment note; source-version uncertainty remains.',
  'Selected totals match; no segment history or component-series certification.',
  ['A note about business segments cannot be generalized to an assurance about every cash-flow component.'],
  [evi('MSFT',2025,'We have recast certain prior period amounts to conform to the way we internally manage and monitor our business.')])
 add('MSFT2026_CASHFLOW_RECAST',ids('MSFT',2026),'selected_totals_limited_clearance',
  'A recast note follows the cash-flow table. Current-release comparative CFO, reconciliation NI and total assets all equal the same fiscal-year current totals printed in the prior release.',
  'The presentation-recast flag alone does not invalidate these selected totals. Current CFO/NI plus same-report assets remain candidates subject to raw-cell and version gates.',
  'No mismatch in these selected totals; unselected cash-flow components may have changed.',
  ['Do not claim no restatement occurred: the source explicitly says recast.','Do not certify component totals, complete correction history, or original first-published bytes.'],
  [evi('MSFT',2026,'We have recast certain prior period amounts to conform to the current period presentation.')])
 add('MSFT2021_2022_ESTIMATE_CONTEXT',ids('MSFT',2021,2022),'economic_interpretation_warning',
  'Useful-life/depreciation effects are also discussed in FY2021 and FY2022, not only FY2023.',
  'NI and assets are reported under contemporaneous estimates. A cash-versus-income hypothesis can respond to noncash estimates, not only recurring earnings quality.',
  'Do not read the FY2022 statement as proof of an additional new adoption in that year; it discusses the impact being compared.',
  ['No amount, date-of-adoption or total-CFO neutrality is inferred from these short passages.'],
  [evi('MSFT',2021,'change in estimated useful lives of our server and network equipment'),evi('MSFT',2022,'change in accounting estimate for the useful lives of our server and network equipment')])
 add('MSFT2016_2017_FUTURE_ADOPTION',ids('MSFT',2016,2017),'future_adoption_not_current_restatement',
  'The FY2016/FY2017 releases discuss anticipated or elected future adoption; the FY2017 effective date July 1, 2017 is after its June 30 fiscal close.',
  'The generic occurrence of adoption/restatement words is not enough to declare current columns restated.',
  'Use FY2018 explicit restatement for FY2018 comparisons; do not relabel FY2017 original amounts.',
  ['Site-shell statement that information is current as of earnings date is not historical-byte certification.'],
  [evi('MSFT',2017,'We elected to early adopt the standard effective July 1, 2017, using the full retrospective method')])
 add('AMD2016_BALANCE_PRESENTATION',ids('AMD',2016),'disclosed_balance_sheet_reclassification',
  'Balance-sheet footnotes identify deferred-tax classification and debt-issuance-cost presentation adoption. Prior assets need not equal the preceding release.',
  'Use only the same current-report assets endpoints; amounts remain reported presentation, not economically constant classification. Comparative CFO/NI are missing from the formal statement in v1.',
  'Lack of comparative flow columns is an extraction/coverage gap distinct from this asset presentation note.',
  ['No inheritance of old GP/assets admission into CFO/NI.'],
  [evi('AMD',2016,'Amounts reflected adoption of FASB ASU 2015-17, Balance Sheet Classification of Deferred Taxes beginning in the first quarter of 2016.'),evi('AMD',2016,'Amounts reflected adoption of FASB ASU 2015-03, Simplifying the Presentation of Debt Issuance Costs beginning in the first quarter of 2016.')])
 add('AMD2018_CFO_CLASSIFICATION_VS_OLD_ASSET_HOLD',ids('AMD',2018),'cashflow_scope_explained_but_legacy_hold_retained',
  'ASU2016-15 reclassified cash receipts from operating to investing retrospectively, and the release says all periods presented conform. This explains CFO classification, not the old unresolved GP/assets differences.',
  'The CFO classification is explicitly current-report consistent, but the concept needs assets and NI too; the old asset-comparability rejection remains and formal NI/CFO extraction gaps are not repaired here.',
  'Same-report CFO comparisons may be evaluated under the disclosed classification. Old-report CFO must not be spliced in. The no-material-impact wording is not treated as zero effect.',
  ['Revenue adoption discussion in FY2017 does not by itself map the FY2018 restated assets.','No inherited rejection is removed.'],
  [evi('AMD',2018,'which reclassified certain cash receipts from operating activities to investing activities, with retrospective application. All periods presented conform to the classification requirements of the standard.')])
 add('AMD2021_TAX_ADOPTION',ids('AMD',2021),'disclosed_modified_retrospective_tax_change',
  'The balance-sheet footnote discloses modified-retrospective ASU2019-12 adoption and a deferred-tax liability recognized in accumulated deficit.',
  'Retain tax/opening-equity comparability caveat for reported income and asset-based interpretation; this is not evidence of fabricated or invalid annual statements.',
  'Do not assume the prior year was restated under a modified-retrospective adoption.',
  ['No reported monetary effect is recalculated or normalized.'],
  [evi('AMD',2021,'the Company adopted ASU 2019-12, Income Taxes (Topic 740): Simplifying the Accounting for Income Taxes, using the modified retrospective adoption method')])
 add('AMD2025_CONSOLIDATED_VS_CONTINUING',ids('AMD',2025),'current_flow_scope_resolved_but_legacy_hold_retained',
  'Cash-flow reconciliation starts with consolidated Net income including discontinued operations. Separate continuing and discontinued CFO rows precede the final Net cash provided by operations total. The selected candidate is that final total.',
  'Final total CFO matches consolidated NI scope. This resolves the specific total-versus-continuing selection issue, not the inherited cross-year/asset scope hold. A divested business contributes annual income/cash but is not a year-end owned asset.',
  'No claim of constant-business or organic comparability between annual totals; original continuing-only GP rejection is not silently repurposed into an admission.',
  ['All six inherited rejected events stay rejected until a distinct prospective source decision.','Do not replace total NI with continuing NI or total CFO with continuing CFO.'],
  [evi('AMD',2025,"Loss (Income) from discontinued operations relates to ZT Systems' manufacturing business which was divested in the fourth quarter of 2025."),evi('AMD',2025,'Net cash provided by operating activities of continuing operations'),evi('AMD',2025,'Net cash provided by operating activities of discontinued operations'),evi('AMD',2025,'Net cash provided by operations')])
 add('NVDA2017_STOCK_COMPENSATION_TAX',ids('NVDA',2017),'annual_comparison_basis_warning',
  'The ASU2016-09 note says adjustments are reflected beginning in fiscal2017 including all fiscal quarters within that year. It does not establish identical treatment of FY2016.',
  'Current reported tax/NI/CFO interpretation needs this adoption caveat. Frozen v1 has missing formal CF fields for these early releases; do not use non-GAAP reconciliation or free-cash-flow numbers instead.',
  'Fiscal2016 comparability is not cleared by the statement about fiscal2017 quarters.',
  ['Later non-GAAP tax-adjustment explanations do not authorize substitution for GAAP cash-flow starting NI.'],
  [evi('NVDA',2017,'In third quarter of fiscal 2017, NVIDIA adopted an accounting standard (ASU 2016-09), which requires adjustments to be reflected beginning in fiscal 2017, including all fiscal quarters within the year.')])
 # Independent Apple pixel evidence belongs to the root reviewer, not this script.
 add('AAPL2018_PRIOR_CFO_REVISION',ids('AAPL',2017,2018),'unresolved_comparison_revision',
  'Independent pixel review observes FY2017 CFO 63,598 in FY2017 versus 64,225 as the FY2018 comparison. The original financing section has an excess-tax-benefit line; this is a clue, not a proven explanation.',
  'An unexplained change in the unused comparative CFO alone does not establish that FY2018 current CFO/NI or its same-report asset endpoints are wrong. Those current-concept fields need their own explicit review; no automatic blanket rejection or automatic clearance is granted.',
  'Comparative-flow equivalence and cross-report growth remain unresolved. Keep both originals and do not splice, backdate, or compute a correction.',
  ['Find a directly applicable original-dated adoption/reclassification explanation before claiming the cause.','The concept does not use comparative CFO/NI for growth; preserve the flag without inventing additional dependence.'],
  [dict(path=BASE+'APPLE_VISUAL_REVIEW.json',sha256=APPLE_SHA,locator='observations plus FY2017/FY2018 CF page3 entries',quote='Independent pixel review; actual raw amounts preserved in that receipt.')])
 # Inspect all events, keeping global source assumptions and the prior masks explicit.
 rows=[]
 for e in plan['events']:
  associated=[f['id']for f in findings if e['event_id']in f['event_ids']]
  missing=[s+'.'+f for s in ('current','comparison')for f,v in fe[e['event_id']]['fields'][s].items()if v is None]
  rows.append(dict(event_id=e['event_id'],symbol=e['symbol'],fiscal_year=e['fiscal_year'],source=e['source'],
   prose_review='local_html_prose_review'if e['event_id']in text else'independent_root_pdf_visual_review_bound',
   finding_ids=associated,prior_field_admitted=e['prior_field_admitted'],inherited_rejection_codes=e['inherited_rejection_codes'],
   inherited_rejections_unchanged=True,periods=e['periods'],current_release_available_from=e['available_from'],
   missing_frozen_v1_fields=missing,
   non_finding_meaning='No additional flagged issue in bounded release review; not proof of absence of other accounting changes.',
   fact_admitted=False,model_admitted=False,strategy_admitted=False))
 rejected=[r for r in rows if r['inherited_rejection_codes']]
 assert len(rejected)==6
 inherited=[]
 for r in rejected:
  unknown=any('unknown_comparison' in c for c in r['inherited_rejection_codes'])
  inherited.append(dict(event_id=r['event_id'],codes=r['inherited_rejection_codes'],retained=True,
    reason='Exact comparative annual start/duration is unknown; formal annual labeling and an asset endpoint do not fill it. The current unregistered concept expressly retains both-period boundaries, so the hold still applies.'if unknown else'Old field-specific comparability exclusion stays unchanged. New cash-flow observations narrow the issue, but do not grant source eligibility for the current concept.',
    accounting_finding_ids=r['finding_ids']))
 output=dict(schema_version=1,stage='source_only_field_scoped_accounting_review',created_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,
  conclusion='Complete bounded accounting review of the 44-event source census; field-specific issues separated from raw-parser coverage and old masks. No source or strategy admission.',
  source_only=True,source_admissions=0,factor_calculations=0,ratios_computed=0,financial_subtractions=0,returns_computed=0,ranks_or_orders=0,
  first_public_version_verified=False,complete_correction_history_verified=False,certified_daily_security_master=False,
  concept_status='Unregistered and not executed; exact definition bound only to explain whether a source issue touches used fields.',
  concept_minimum_fields=['current consolidated GAAP CFO','current reconciliation-start consolidated GAAP NI','current year-end assets','comparison year-end assets','both annual boundaries','current release availability and issuer scope'],
  comparison_flows_are_audited_but_not_used_by_bound_concept=True,
  physical_cell_review_delegation=dict(html=dict(path=a.html_review,sha256=a.html_sha),apple=dict(path=BASE+'APPLE_VISUAL_REVIEW.json',sha256=APPLE_SHA)),
  findings=findings,selected_total_literal_parity=[parity('MSFT',y,y-1)for y in (2018,2023,2025,2026)],
  inherited_rejection_review=inherited,events=rows,
  general_limits=['The 44 selected release documents are not a complete accounting-note archive or correction history.','An earnings-release footer or official current URL does not establish original first-public bytes.','Contiguous annual starts and next-session availability remain declared existing source assumptions, not newly certified event times.','Consolidated acquisition/disposal effects, working-capital timing, tax effects, stock compensation and depreciation estimates can alter reported cash-versus-income metrics without indicating superior investment quality.','No valid new source value is substituted into frozen v1 nulls; raw candidate repair requires separate versioned work.','Any later source admission needs its own prospective policy and exact used-field gates; all prior rejected sentinels remain present.'],
  source_hashes=sources,other_files_modified=False)
 encoded=(json.dumps(output,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
 with (ROOT/OUT).open('xb')as f:f.write(encoded)
 print(json.dumps(dict(path=OUT,sha256=hashlib.sha256(encoded).hexdigest(),events=len(rows),findings=len(findings),inherited_rejections_retained=len(inherited),source_admissions=0)))
if __name__=='__main__':main()
