"""Offline source-unit inspection only. No strategy inputs or calculations."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal as D
import csv, hashlib, io, json, re, xml.etree.ElementTree as ET
from pypdf import PdfReader
ROOT=Path('/Users/shiweili/code/monthly-stock-research')
BASE=ROOT/'outputs/monthly_etf_primary_flow_sources'
OUT=BASE/'ishares_actions_v2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
pins={
'ishares/IGV_fund_download_20261023.xls':'59ad4dee0e055accb0c9ddabed22a7ed6373bed48b820ee3f9852af65e740a31',
'ishares/IWM_fund_download_20261023.xls':'ee7639510361746b09a61bf4e6bf705e6bcd665d7f7e509ef745d49ecae1f66d',
'ishares/MTUM_fund_download_20261023.xls':'2caa81f02e796961bcde4107d80db66c778de4705d34c8ac03f9b92579cf1860',
'ishares/SOXX_fund_download_20261023.xls':'3589740ab872898c28c949c009f788a48617de2733290be8a01eee71a562706f',
'ishares/TLT_fund_download_20261023.xls':'305be64ebaa59bf1ca455d91b04c87dcf319bdcc6c9ea0ef12f5e0c2ec356804',
'ishares/FEASIBILITY.json':'ea307ef6b297b290c6cd4c339f97a3cdfb86e08ccf8cd122ed6b5970fc07d51a',
'ishares_actions/SPLIT_CONVENTION_AUDIT.json':'80c22271244c10fc12b297a30578006e423689acb5a71e01da43264e969a5d8f',
'ishares_actions/annual_2026_pdf_from_viewer_override.pdf':'d6e66e38115d84822d03c350ba158ea0a600e7d8091e1b7af800895d5a5e8ec4'}
for p,h in pins.items(): assert sha(BASE/p)==h,p
# Read-only SpreadsheetML parsing: XML ampersands are escaped in memory, never in source.
NS={'s':'urn:schemas-microsoft-com:office:spreadsheet'}
def history(ticker):
 p=BASE/'ishares'/(ticker+'_fund_download_20261023.xls')
 x=ET.fromstring(re.sub(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;',p.read_text()))
 sheet=next(w for w in x.findall('s:Worksheet',NS) if w.attrib['{'+NS['s']+'}Name']=='Historical')
 out={}
 for row in sheet.findall('.//s:Row',NS):
  vals=[''.join(c.itertext()) for c in row.findall('s:Cell/s:Data',NS)]
  if len(vals)==4:
   try: d=datetime.strptime(vals[0],'%b %d, %Y').date().isoformat()
   except ValueError: continue
   assert d not in out
   out[d]={'nav':vals[1],'shares':vals[3]}
 return out
hist={t:history(t) for t in ('IGV','SOXX')}
# Verify every new network source against its independently saved HTTP receipt.
network=[]
for p in sorted(OUT.glob('*_RETRIEVAL.json')):
 r=json.loads(p.read_text()); raw=OUT/r['raw_file']
 assert sha(raw)==r['sha256'] and raw.stat().st_size==r['bytes']
 network.append({'receipt':ref(p),'raw':ref(raw),'url':r['requested_url'],'status':r['status'],'retrieved_utc':r['retrieved_utc'],'content_type':r['response_headers']['Content-Type']})
# The API's dates and endpoints are observed in public scripts, not guessed.
contract=(OUT/'official_download_contract_js.html').read_text()
hostcontract=(OUT/'official_api_host_contract_js.html').read_text()
assert 'asOfDate:s||""' in contract and 'r.Holdings="holdings"' in contract
assert 'v1/get-fund-document?' in hostcontract
assert 'v2/get-product-data?' in hostcontract
holdings=[]
for t,vals in {'IGV':{'20240305':89250000,'20240306':88000000,'20240307':87500000},'SOXX':{'20240305':56550000,'20240306':57150000,'20240307':55950000}}.items():
 for ds,expected in vals.items():
  p=OUT/(t+'_holdings_'+ds+'_contract.html')
  rows=list(csv.reader(io.StringIO(p.read_text(encoding='utf-8-sig'))))
  d=datetime.strptime(ds,'%Y%m%d').date().isoformat()
  assert rows[1][0]=='Fund Holdings as of'
  assert datetime.strptime(rows[1][1],'%b %d, %Y').date().isoformat()==d
  assert rows[3][0]=='Shares Outstanding'
  q=D(rows[3][1].replace(',','')); assert q==expected and q==D(hist[t][d]['shares'])
  holdings.append({'ticker':t,'as_of':d,'shares_outstanding':str(q),'historical_same_date_shares':hist[t][d]['shares'],'same_date_exact':True,'source':ref(p),'physical_native_units_proved':False})
factsheets=[]
for t,expected,cusip in [('IGV',130750000,'464287515'),('SOXX',85250000,'464287523')]:
 p=OUT/(t+'_factsheet.pdf'); text=PdfReader(p).pages[0].extract_text()
 assert 'Fact Sheet as of September 30, 2026' in text
 m=re.search(r'Shares Outstanding\s*:\s*([\d,]+)',text); assert m
 q=int(m.group(1).replace(',','')); assert q==expected and str(q)==hist[t]['2026-09-30']['shares']
 assert cusip in text
 factsheets.append({'ticker':t,'as_of':'2026-09-30','cusip':cusip,'shares_outstanding':q,'historical_same_date_shares':int(hist[t]['2026-09-30']['shares']),'next_date_shares':int(hist[t]['2026-10-01']['shares']),'same_date_exact':True,'next_date_equal':q==int(hist[t]['2026-10-01']['shares']),'source':ref(p),'pdf_page':1})
# Accounting unit reconciliation only: no share-growth, cash-flow, return or feature.
# Rounded AUM ($000) and explicitly split-adjusted year-end NAV imply a bounded
# split-adjusted outstanding-unit count. These are not directly printed native shares.
annual=BASE/'ishares_actions/annual_2026_pdf_from_viewer_override.pdf'
r=PdfReader(annual)
anchors=[]
for t,page,factor,data in [('IGV',34,5,[('2022-03-31','2022-04-01','5403390','68.83'),('2023-03-31','2023-04-03','5387680','60.88')]),('SOXX',36,3,[('2022-03-31','2022-04-01','8952849','157.48'),('2023-03-31','2023-04-03','7785915','148.30')])]:
 text=r.pages[page-1].extract_text()
 note=('five' if factor==5 else 'three')+'-for-one stock split'
 assert note in text and 'March 6, 2024' in text and 'Net assets, end of year (000)' in text
 for d,nextd,aum000,nav in data:
  assert f'{int(aum000):,}' in text and nav in text
  a=D(aum000)*1000;n=D(nav)
  lo=(a-D('500'))/(n+D('.005')); hi=(a+D('500'))/(n-D('.005'))
  q=D(hist[t][d]['shares']); qnext=D(hist[t][nextd]['shares'])
  next_in=(lo<=qnext<=hi); native_lo=lo/factor;native_hi=hi/factor
  assert next_in and not(native_lo<=q<=native_hi) and not(native_lo<=qnext<=native_hi)
  anchors.append({'ticker':t,'financial_date':d,'next_trading_date_checked':nextd,'pdf_page':page,'net_assets_rounded_thousands':aum000,'explicitly_split_adjusted_nav_rounded':nav,'forward_split_factor':factor,'unit_bound_rounding_assumption':'nearest $1000 for net assets and nearest $0.01 for NAV; not a source-stated precision guarantee','implied_split_adjusted_shares_interval':[str(lo),str(hi)],'implied_native_pre_split_shares_interval':[str(native_lo),str(native_hi)],'historical_same_date_shares':str(q),'historical_next_trading_date_shares':str(qnext),'same_date_in_financial_interval':lo<=q<=hi,'next_date_in_financial_interval':next_in,'same_date_in_native_interval':native_lo<=q<=native_hi,'next_date_in_native_interval':native_lo<=qnext<=native_hi,'inference':'reported historical shares are compatible with the post-split share unit at this anchor, while exact financial-clock alignment remains unresolved','direct_native_shares_printed_in_source':False,'source':ref(annual)})
report={
 'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'issuer source-only share-unit and reporting-date inspection; not backtest admission',
 'decision':'PARTIAL_RESOLUTION_NO_GLOBAL_SHIFT_NO_NATIVE_SHARE_CERTIFICATION',
 'new_factor_values':0,'new_model_fits':0,'new_orders':0,'new_strategy_returns':0,'new_flow_calculations':0,'new_strategy_trials':0,
 'protected_old_sources_verified':[{**ref(BASE/p),'expected_sha256':h} for p,h in pins.items()],
 'network_sources':network,'historical_download_contract':{'observed_chain':['IGV_product_20261023.html -> official_product_component_js.html','official_product_component_js.html -> official_holdings_wrapper_js.html','official_holdings_wrapper_js.html -> official_download_contract_js.html','official_download_contract_js.html -> official_api_host_contract_js.html'],'parameters':{'appType':'PRODUCT_PAGE','appSubType':'ISHARES','targetSite':'us-ishares','locale':'en_US','userType':'individual','portfolioId':'observed fund ID','asOfDate':'YYYYMMDD','component':'holdings'},'unadvertised_old_date_probe':'Requested one real historical date via the observed public contract; response 200 with matching as-of date, then five bounded action-window checks. No authentication or endpoint guessing.','headers_only_inspected':True,'csv_raw_suffix_note':'Frozen collector used .html suffix for non-PDF/XLS response; HTTP Content-Type is text/csv. Raw bytes unchanged.'},
 'split_window_download_checks':holdings,'official_factsheet_same_date_checks':factsheets,'financial_rounding_unit_reconciliations':anchors,
 'conclusions':[
  'The official historical holdings download accepts explicit pre-2024 dates and preserves the requested date in the CSV header. Its Shares Outstanding agrees exactly with Historical for both funds on March 5, 6 and 7, 2024.',
  'Two current-vintage issuer historical downloads agreeing is not contemporaneous native-share evidence: both may be restated.',
  'At four 2022/2023 financial-year anchors, the next observed trading-date Historical share value is inside the interval implied by the audited split-restated NAV and net assets. The same observed values exclude native pre-split units under the stated rounding assumptions. This strengthens post-split-unit compatibility at these anchors, not whole-history certification.',
  'The 2026-09-30 issuer factsheets agree with Historical shares on the SAME date and differ from the next date. Therefore a universal one-trading-day shift chosen from the annual-report anchors is unsupported.',
  'The March 2026 factsheet values seen in search indexing were not obtained as raw issuer PDFs: the live originals now contain September 2026. Indexed stale content is not admitted as a saved March source.',
  'A 2024 annual financial statements link discovered in official search returned HTTP404; its raw response and retrieval receipt are retained. No SEC request was made in this v2 phase because the October23 LA provider stop remains active.'
 ],
 'per_fund_status':{
  'IGV':{'shares_basis':'post_split_units_compatible_at_2022_2023_financial_anchors; direct native-share convention and full-history adjustment unknown','action_window':'2024-03-05 through 2024-03-07 current-vintage CSV/Historical equality established; no factor or time shift applied'},
  'SOXX':{'shares_basis':'post_split_units_compatible_at_2022_2023_financial_anchors; direct native-share convention and full-history adjustment unknown','action_window':'2024-03-05 through 2024-03-07 current-vintage CSV/Historical equality established; 2026-11-05 split is after research cutoff and not applied'},
  'IWM':{'shares_basis':'unknown','actions':'2005 split outside requested 2016+ window known from prior official CME/CFTC source; no complete official in-window no-split certificate found'},
  'MTUM':{'shares_basis':'unknown','actions':'2026-11-05 split is after research cutoff and not applied; no complete official 2016+ no-split certificate found'},
  'TLT':{'shares_basis':'unknown','actions':'no in-window split discovered, which is not evidence of completeness'}},
 'remaining_gaps':['Direct pre-2024 native physical shares or explicit whole-history issuer shares adjustment policy.','Historical Shares Outstanding observation/reporting/publication clock, including treatment of settlement and pending issuance.','Complete corporate-action coverage for IWM/MTUM/TLT during required history.','Native units and action-date mapping for the full evaluation history, without arbitrary shifts or future-data assumptions.'],
 'validation':{'passed':True,'old_sources_verified':len(pins),'new_http_receipts_verified':len(network),'csv_header_crosschecks':len(holdings),'factsheet_crosschecks':len(factsheets),'financial_unit_anchor_checks':len(anchors),'selected_financial_pages_rendered_and_visually_checked':[34,36],'source_only':True,'all_sources_kept_raw':True,'no_alignment_correction_applied':True,'no_new_admission':True}
}
path=OUT/'SHARE_UNIT_REVIEW.json'
assert not path.exists(), 'write once'
path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'path':str(path),'sha256':sha(path),'validation':report['validation']}))
