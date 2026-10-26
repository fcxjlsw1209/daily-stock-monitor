"""Source-only literal XML/report-anchor audit. No strategy imports or flow calculations."""
import datetime as dt
import hashlib
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'outputs/monthly_etf_primary_flow_sources'
OUT=BASE/'ssga_actions_v2'
REVIEW=ROOT/'work/etf_flow_ssga_actions_v2_review.json'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
SYMS=['SPY','GLD','XLE','XLF','XLI','XLK','XLV','XLY']
D=Decimal

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write_once(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def xml_rows(p):
 with zipfile.ZipFile(p) as z:
  ss=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml'))]
  rows=[]
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',NS):
   vals={}
   for c in row:
    v=c.find('m:v',NS)
    if v is not None: vals[c.get('r')]=ss[int(v.text)] if c.get('t')=='s' else v.text
   rows.append(vals)
  return rows

def nav_rows(symbol):
 p=BASE/'ssga'/(symbol+'_navhistory.xlsx')
 source=read(BASE/'ssga'/(symbol+'_SOURCE_EXTRACT.json'))
 assert sha(p)==source['metadata']['source_sha256']
 extracts={r['date']:r for r in source['rows']}
 result={}
 for row in xml_rows(p):
  a=next((v for k,v in row.items() if k.startswith('A')),None)
  if not a:continue
  try:date=dt.datetime.strptime(a,'%d-%b-%Y').date().isoformat()
  except ValueError:continue
  vals={re.sub(r'\d','',k):v for k,v in row.items()}
  if not all(k in vals for k in ['B','C','D']):continue
  if date in extracts:
   for k,col in [('B','nav'),('C','shares_outstanding'),('D','net_assets')]:
    try:assert float(vals[k])==extracts[date][col]
    except ValueError:pass
  result[date]={'source_file':str(p.relative_to(ROOT)),'source_row':row,'nav':vals['B'],'shares':vals['C'],'net_assets':vals['D']}
 return result

pins={}
for name,expected in {
 'SOURCE_FEASIBILITY_SUMMARY.json':'67fe2492cb9561dec5700566310fae39f1ca04de46c8e2f92e47e708d0da39ab',
 'CALENDAR_ACCOUNTING_AUDIT.json':'2cf542a5755cab8e99100b36fcf04ca60dd60efc4d3feeaea9a37b559399f482',
}.items():assert sha(BASE/name)==expected
for name,expected in {
 'GLD_2025_annual.pdf':'ef08b1182863ba103428a755537096e2f00f06382bde607535846bbd6e2227d8',
 'GLD_2026_Q3.pdf':'6a2f13b07f889064bdae2c3e4b8f95e0d698384fbed5ff6aa90f54f148a36fcf',
 'Sector_annual_financial.pdf':'1f439a3f08e1f6d4eab5844955215108239feb690571943651a7c229606931a4',
 'Sector_semi_financial.pdf':'53218b0b917b94210bd9a8ba1e131ffaba4c6166d5c4c06db8d83b5e041c2d7e',
}.items():assert sha(OUT/name)==expected
retrieval=[]
for p in sorted(OUT.glob('*_RETRIEVAL.json')):
 j=read(p);retrieval.append({'file':str(p.relative_to(ROOT)),'url':j['url'],'status':j.get('status'),'sha256':j.get('sha256'),'requested_utc':j['requested_utc'],'error':j.get('error')})
 if j.get('status')==200:
  raw=OUT/p.name.removesuffix('_RETRIEVAL.json');assert sha(raw)==j['sha256']
 assert not any(h in j['url'] for h in ['www.sec.gov','investors.statestreet.com'])
calendar=read(ROOT/'outputs/monthly_paired_target143/CALENDAR.json')['sessions']
raw={sym:nav_rows(sym) for sym in SYMS}
annual={'XLE':('601348400','44.67'),'XLF':('1004995427','53.88'),'XLI':('155026000','154.29'),'XLK':('639511794','141.01'),'XLV':('247115324','139.23'),'XLY':('205606504','119.83')}
semi={'XLE':('714448400','61.24'),'XLF':('976445427','49.38'),'XLI':('173526000','161.75'),'XLK':('633561794','132.96'),'XLV':('264465324','146.61'),'XLY':('194656504','108.97')}
anchors=[]
for date,nxt,report,values in [('2025-09-30','2025-10-01','Sector_annual_financial.pdf',annual),('2026-03-31','2026-04-01','Sector_semi_financial.pdf',semi)]:
 assert calendar[calendar.index(date)+1]==nxt
 for sym,(shares,nav) in values.items():
  factor=2 if date=='2025-09-30' and sym in ['XLE','XLK','XLY'] else 1
  same=raw[sym][date];after=raw[sym][nxt]
  a={'symbol':sym,'report_date':date,'report_file':str((OUT/report).relative_to(ROOT)),'report_pdf_page':28 if sym=='XLY' else 29,'report_column_label':{'XLE':'Energy','XLF':'Financial','XLI':'Industrial','XLK':'Technology','XLV':'Health Care','XLY':'Consumer Discretionary'}[sym], 'report_shares':shares,'report_nav':nav,'report_share_units_per_native_share':factor,'same_date_source':same,'next_official_session':nxt,'next_date_source':after}
  a['report_shares_match_same_date_in_report_units']=D(shares)==D(same['shares'])*factor
  a['report_shares_match_next_date_in_report_units']=D(shares)==D(after['shares'])*factor
  a['report_nav_matches_same_date_to_cent_in_report_units']=(D(same['nav'])/factor).quantize(D('.01'),rounding=ROUND_HALF_UP)==D(nav)
  assert not a['report_shares_match_same_date_in_report_units']
  assert a['report_shares_match_next_date_in_report_units']
  assert a['report_nav_matches_same_date_to_cent_in_report_units']
  anchors.append(a)
for date,nxt,shares,nav,report,page in [('2024-09-30','2024-10-01','303300000','243.01','GLD_2025_annual.pdf',49),('2025-09-30','2025-10-01','353700000','352.09','GLD_2025_annual.pdf',49),('2026-06-30','2026-07-01','352000000','369.47','GLD_2026_Q3.pdf',3),('2026-08-03','2026-08-04','352400000',None,'GLD_2026_Q3.pdf',1)]:
 assert calendar[calendar.index(date)+1]==nxt
 same=raw['GLD'][date];after=raw['GLD'][nxt]
 a={'symbol':'GLD','report_date':date,'report_file':str((OUT/report).relative_to(ROOT)),'report_pdf_page':page,'report_shares':shares,'report_nav':nav,'report_share_units_per_native_share':1,'same_date_source':same,'next_official_session':nxt,'next_date_source':after,'report_shares_match_same_date_in_report_units':D(shares)==D(same['shares']),'report_shares_match_next_date_in_report_units':D(shares)==D(after['shares'])}
 a['report_nav_matches_same_date_to_cent_in_report_units']=None if nav is None else D(same['nav']).quantize(D('.01'),rounding=ROUND_HALF_UP)==D(nav)
 assert a['report_shares_match_same_date_in_report_units'] and not a['report_shares_match_next_date_in_report_units']
 assert nav is None or a['report_nav_matches_same_date_to_cent_in_report_units']
 anchors.append(a)
html=(BASE/'ssga/GLD_product.html').read_text()
m=re.search(r'"field_admin_title":"Trust Information - Total Shares Outstanding"(?P<context>.{0,1300}?)"field_title":"Total Shares Outstanding"',html)
assert m
clock=re.search(r'"field_delay":"([^"]+)"',m.group(0)).group(1)
assert clock=='Updated between 10:30 a.m. and 10:45 a.m. and 4.00 p.m. and 4:45 p.m. New York time'
clock_context=json.loads('{'+m.group(0)+'}')
# Literal distribution metadata. No share-growth, flow, NAV return or rank calculations.
dist={s:[] for s in SYMS}
for row in xml_rows(OUT/'historical_distributions.xlsx'):
 vals={re.sub(r'\d','',k):v for k,v in row.items()}
 sym=vals.get('B')
 if sym in dist and vals.get('D'):
  try:d=dt.datetime.strptime(vals['D'],'%m/%d/%Y').date().isoformat()
  except ValueError:continue
  dist[sym].append({'ex_date':d,'source_row':row})
dist_scope={sym:{'rows':len(rows),'first_ex_date':min((r['ex_date']for r in rows),default=None),'last_ex_date':max((r['ex_date']for r in rows),default=None),'research_window_rows':sum('2016-05-11'<=r['ex_date']<='2026-09-24' for r in rows),'complete_split_ledger':False}for sym,rows in dist.items()}
method=read(ROOT/'work/etf_flow_ssga_method_review.json')
event_rows={sym:{date:raw[sym][date] for date in ['2025-12-04','2025-12-05']} for sym in ['XLE','XLK','XLY']}
for f in [ROOT/'AGENTS.md',ROOT/'LOCAL_SETUP.md',ROOT/'outputs/monthly_paired_target143/CALENDAR.json',ROOT/'work/etf_flow_ssga_method_review.json',ROOT/'work/etf_reported_share_growth_feasibility.json',BASE/'SOURCE_FEASIBILITY_SUMMARY.json',BASE/'CALENDAR_ACCOUNTING_AUDIT.json',BASE/'LAGGED_COVERAGE_SUPPLEMENT.json',BASE/'ssga/FEASIBILITY.json']:
 pins[str(f.relative_to(ROOT))]=sha(f)
for sym in SYMS:
 for suffix in ['_SOURCE_EXTRACT.json','_navhistory.xlsx']:
  f=BASE/'ssga'/(sym+suffix);pins[str(f.relative_to(ROOT))]=sha(f)
for f in (BASE/'ssga').glob('*product.html'):pins[str(f.relative_to(ROOT))]=sha(f)
for f in sorted(OUT.iterdir()):
 if f.is_file():pins[str(f.relative_to(ROOT))]=sha(f)
for f in sorted((OUT/'rendered').glob('*.png')):pins[str(f.relative_to(ROOT))]=sha(f)
for f in [BASE/'ssga_method/STOP_investors.statestreet.com_2026-10-23.json',BASE/'ishares_actions/STOP_www.sec.gov_2026-10-23.json',BASE/'ssga_method/XLF_Form8937.pdf']+[BASE/'ssga_method'/(s+'_2025_Form8937.pdf')for s in ['XLE','XLK','XLY']]:pins[str(f.relative_to(ROOT))]=sha(f)
summary={'schema':'ssga_actions_v2_source_anchor_audit','passed':True,'purpose':'source semantics and unit/date comparisons only; not feature admission or strategy evaluation','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'input_sha256':pins,'anchors':anchors,'sector_report_anchors':12,'sector_same_date_shares_matches':0,'sector_next_session_shares_matches':12,'sector_same_date_nav_matches_to_cent':12,'gld_share_anchors':4,'gld_same_date_shares_matches':4,'gld_next_session_shares_matches':0,'gld_nav_anchors':3,'gld_same_date_nav_matches_to_cent':3,'distribution_index_scope':dist_scope,'gld_current_share_publication_component':clock_context,'known_native_split_rows':event_rows,'retrievals':retrieval,'actual_flow_computed':False,'actual_share_growth_computed':False,'actual_new_features_computed':False,'actual_orders_computed':False,'actual_portfolio_returns_computed':False}
write_once(OUT/'SOURCE_ANCHOR_AUDIT.json',summary)
pins2={**pins,str((OUT/'SOURCE_ANCHOR_AUDIT.json').relative_to(ROOT)):sha(OUT/'SOURCE_ANCHOR_AUDIT.json')}
review={
 'schema':'etf_flow_ssga_actions_v2_review','stage':'SOURCE_ONLY_UNIT_ACTION_AND_CLOCK_REVIEW_NOT_REGISTERED','passed_source_checks':True,'fully_admitted_research_intervals':[], 'strict_historical_PIT_established':False,
 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'symbols':SYMS,'research_source_window':['2016-05-11','2026-09-24'],'coverage_sessions':2608,'coverage_reused_not_recomputed':True,
 'input_sha256':pins2,
 'material_findings':[
  {'id':'sector_dates','finding':'For all six sector ETFs, both 2025-09-30 and2026-03-31 statement share balances match the next official-session workbook row exactly, after applying the explicit annual-report split presentation factor where required. None matches the same-date row; NAV matches the same-date row to the published cent.','evidence':'SOURCE_ANCHOR_AUDIT.json,12anchors','limit':'Two report-date anchors per ETF do not prove a universal one-session shift or trade-versus-settlement convention. No shift is applied.'},
  {'id':'annual_restatement','finding':'Sector2025 annual financial statements Note13 retroactively restate the September30 statement and historical per-share figures for the December4 after-close2:1 action. The downloaded daily workbook preserves native units around December4/5. Report values and raw rows cannot be compared without the declared report scale.','evidence':'Sector_annual_financial.pdf pp28-29,63; source table records XLE601348400,XLK639511794,XLY205606504 report shares, matched against twice October1 native rows.','limit':'This is report-specific restatement evidence, not proof that the whole daily archive is unadjusted.'},
  {'id':'gld_anchors','finding':'Four GLD statement/cover anchors match same-date workbook shares exactly:2024-09-30,2025-09-30,2026-06-30,2026-08-03. All four differ from next-session shares; three associated NAVs match same date at report precision.','limit':'Four anchors do not prove full-window action completeness or immutable historical vintages.'},
  {'id':'gld_clock','finding':'The current issuer page contains an explicit share-component update schedule.','field_delay':clock,'raw_component_context':clock_context,'scope':'Current downloaded page only; not evidence of2016-2026 publication deadlines or first-public versions. Website-gold trade-date versus bar-list settlement-date wording does not establish the share column clock.'},
  {'id':'indices','finding':'Downloaded2025 ICI tax summaries concern distribution tax classification. The historical distribution XLSX spans1993/1999 onward forSPY/six sectors, with2026 latest rows, but contains dividends/capital-gain/ex-record-pay fields and no split ledger. GLD is absent from this distribution table.','limit':'No official exhaustive2016-2026 share-action index has been established. No-event search results are never used as absence proof.'}
 ],
 'action_admission':{
  'known_action_facts_reused':method['corporate_actions'],
  'known_native_bridge':{'symbols':['XLE','XLK','XLY'],'ratio_new_per_old':'2:1','raw_boundary':['2025-12-04','2025-12-05'],'confirmed_scope':'The issuer action and local native-unit break are established. A proposed comparison spanning this boundary must put endpoints into one unit; no factor or share-growth value was calculated.','full_history_actions_complete':False},
  'xlf_non_split':{'dates':['2016-09-19','2016-09-21','2016-09-22'],'treatment':'XLRE in-kind distribution is not an XLF split. Do not create a share multiplier or label its NAV drop as cash outflow. Portfolio exposure changes are a separate comparability limitation.'},
  'official_exhaustive_action_ledger':'UNKNOWN for all8 across full research window',
  'unrelated_products_excluded':['GLDM2022reverse split is not GLD','EMEA SPDR UCITS share-class action pages are not the US funds','XLB/XLU2025split evidence is not evidence thatXLF/XLI/XLV never split']
 },
 'asset_date_assessment':[
  {'symbol':'SPY','raw_row_coverage':'full2608,reused','independent_share_anchor_status':'NOT_OBTAINED in this boundedreview; official linked annual viewer resolved to current client application, no report bytes claimed','admission':'OBSERVATION_ARCHIVE_ONLY','full_interval_action_and_unit_admission':False},
  {'symbol':'GLD','verified_dates':['2024-09-30','2025-09-30','2026-06-30','2026-08-03'],'admission':'Four physical-share/date anchors verified; current publication-component schedule documented; no continuous63-session interval certified solely from point anchors','full_interval_action_and_unit_admission':False},
  *[{'symbol':s,'verified_report_dates':['2025-09-30','2026-03-31'],'matched_share_row_dates':['2025-10-01','2026-04-01'],'known_split_bridge':s in ['XLE','XLK','XLY'],'admission':'Unit/date diagnostics passed with stated report-scale conversion; same-row accounting pairing disproven at anchors; no continuous feature interval admitted','full_interval_action_and_unit_admission':False}for s in ['XLE','XLF','XLI','XLK','XLV','XLY']]
 ],
 'research_interpretation':{
  'nav_weighted_same_row_flow_admissible':False,
  'reported_share_growth_potential':'A separately registered latest-vintage reported-row share-growth study may avoid NAV pairing. It must freeze an evidenced common-unit/action policy and explicitly accepted report-date/publication assumptions first; this review does not grant that registration.',
  'non_crossing_known_action_windows':'Not crossing the known2025boundary removes that particular split correction, but does not establish absence of other actions or prove historical publication availability.',
  'fixed_five_session_lag':'Declared assumption only. Does not establish strictPIT or repair restated units or date-label inconsistencies.',
  'publication_and_revision_unknown':'Daily workbooks first-public timestamps and historical revision policy remainUNKNOWN; current download cannot recreate archival vintages.',
  'neutral_policy':'If later preregistered, unsupported asset-dates must remain neutral in both arms without changing universe or baseline. No coverage chosen using returns.'
 },
 'remaining_minimum_conditions':['Complete or explicitly bounded issuer action coverage for each admitted window, with report-row effective boundaries and no negative-proof inference.','Verify native versus restated share basis at relevant action boundaries; do not use retrospectively restated report shares as native values.','Predeclare report-date semantics and publication/revision assumptions; no globaldate shift inferred from12anchors.','Freeze supported asset-date subset independently of outcomes, then register before any factor/rank/order/return calculation.'],
 'source_collection':{'successful_responses':sum(x['status']==200 for x in retrieval),'http404':sum(x['status']==404 for x in retrieval),'http403_or429':sum(x['status']in(403,429)for x in retrieval),'stopped_hosts_requested':False,'search_only_uncollected_leads':['2025sector_splitFAQ404','2025APnotice404','GLD2016Q3historicalURL404'],'no_unretrieved_bytes_claimed':True},
 'review_correction_log':[{'temporary_message_error':'486648320 was tentatively attributed toXLE during text extraction.','correction':'It belongs toUtilities/XLU, outsidethis8fundscope. It was excluded before final audited values. FinalXLE2025reportshares601348400 visually confirmed onPDFp29.','all_final_sector_columns_visually_checked':True}],
 'scope_flags':{'existing_sources_unchanged':True,'new_source_documents_collected':True,'literal_source_units_and_dates_compared':True,'split_share_unit_conversion_only':True,'actual_flow_computed':False,'actual_share_growth_computed':False,'actual_factors_computed':False,'model_fits_computed':False,'rankings_computed':False,'orders_computed':False,'returns_computed':False,'strategy_registered':False,'central_modified':False,'observer_modified':False,'frozen_strategy_modified':False},
 'output_paths':['outputs/monthly_etf_primary_flow_sources/ssga_actions_v2/SOURCE_ANCHOR_AUDIT.json','work/etf_flow_ssga_actions_v2_review.json']
}
write_once(REVIEW,review)
for rel,digest in pins2.items():assert sha(ROOT/rel)==digest
print(json.dumps({'passed':True,'review':str(REVIEW.relative_to(ROOT)),'review_sha256':sha(REVIEW),'audit_sha256':sha(OUT/'SOURCE_ANCHOR_AUDIT.json'),'inputs':len(pins2),'sector_anchors':12,'gld_anchors':4,'strict_PIT':False,'new_factors':0},indent=2))
