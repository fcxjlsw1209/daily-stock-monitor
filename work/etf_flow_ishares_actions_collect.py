"""Collect official iShares raw sources separately from offline research.

No credentials, derived flow calculations, signals, forecasts, or orders.
One request per invocation; no retries or automatic redirect following.
403/429 blocks this host for the rest of the America/Los_Angeles day.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib import error, parse, request
from zoneinfo import ZoneInfo

ROOT = Path('/Users/shiweili/code/monthly-stock-research')
OUT = ROOT / 'outputs/monthly_etf_primary_flow_sources/ishares_actions'
ALLOWED_HOSTS = ('www.ishares.com', 'www.blackrock.com', 'www.sec.gov', 'www.cftc.gov', 'www.businesswire.com', 'www.nasdaq.com')


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def collect(label, url):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('Unsafe source label')
    parsed = parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
        raise ValueError('Only unauthenticated issuer or official publication HTTPS sources are allowed')
    OUT.mkdir(parents=True, exist_ok=True)
    day = datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    receipt = OUT / (label + '_RETRIEVAL.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        raw = OUT / prior['raw_file']
        if prior['requested_url'] != url or hashlib.sha256(raw.read_bytes()).hexdigest() != prior['sha256']:
            raise ValueError('Existing source does not match request or original SHA')
        return {'reused': True, 'receipt': str(receipt), 'status': prior['status']}
    stop = OUT / ('STOP_' + parsed.hostname + '_' + day + '.json')
    other_stop = ROOT / 'outputs/monthly_etf_primary_flow_sources/ishares' / stop.name
    if stop.exists() or other_stop.exists():
        raise RuntimeError('Host stopped for this local day after 403/429; do not retry or bypass')
    record = {'requested_url': url, 'requested_utc': datetime.now(timezone.utc).isoformat(),
              'local_date': day, 'local_timezone': 'America/Los_Angeles',
              'source_role': 'raw issuer source only; no factor or backtest admission',
              'automatic_retries': 0, 'automatic_redirects': False}
    opener = request.build_opener(NoRedirect())
    req = request.Request(url, headers={'User-Agent': 'LocalResearch/1.0', 'Accept': '*/*'})
    try:
        response = opener.open(req, timeout=40)
    except error.HTTPError as exc:
        response = exc
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc))
        with (OUT / (label + '_TRANSPORT_ERROR.json')).open('x') as handle:
            json.dump(record, handle, indent=2)
        raise
    with response:
        body = response.read(40_000_001)
        if len(body) > 40_000_000:
            raise ValueError('Unexpected source size')
        record.update(status=response.code, final_url=response.url,
                      response_headers={key: response.headers.get(key) for key in
                                        ('Content-Type', 'Content-Disposition', 'Last-Modified', 'ETag',
                                         'Date', 'Retry-After', 'Location')},
                      retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    content = (record['response_headers']['Content-Type'] or '').lower()
    suffix = '.pdf' if body[:4] == b'%PDF' else ('.xlsx' if body[:2] == b'PK' else ('.xls' if 'excel' in content else '.html'))
    raw = OUT / (label + suffix)
    with raw.open('xb') as handle:
        handle.write(body)
    record['raw_file'] = raw.name
    with receipt.open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    if record['status'] in (403, 429):
        with stop.open('x') as handle:
            json.dump({'status': record['status'], 'host': parsed.hostname, 'local_date': day,
                       'receipt': receipt.name, 'retry_after': record['response_headers']['Retry-After'],
                       'policy': 'Stop requests to this host for the local day; no bypass.'}, handle, indent=2)
    return {'reused': False, 'status': record['status'], 'bytes': len(body), 'raw_file': str(raw),
            'receipt': str(receipt), 'sha256': record['sha256']}



def inspect_actions():
    """Read saved documents only. Check units/identities; never calculate flows or returns."""
    from decimal import Decimal, ROUND_HALF_UP
    import xml.etree.ElementTree as ET
    from pypdf import PdfReader
    prior = ROOT / 'outputs/monthly_etf_primary_flow_sources/ishares'
    expected = {
      'IGV':'59ad4dee0e055accb0c9ddabed22a7ed6373bed48b820ee3f9852af65e740a31',
      'IWM':'ee7639510361746b09a61bf4e6bf705e6bcd665d7f7e509ef745d49ecae1f66d',
      'MTUM':'2caa81f02e796961bcde4107d80db66c778de4705d34c8ac03f9b92579cf1860',
      'SOXX':'3589740ab872898c28c949c009f788a48617de2733290be8a01eee71a562706f',
      'TLT':'305be64ebaa59bf1ca455d91b04c87dcf319bdcc6c9ea0ef12f5e0c2ec356804'}
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    ns={'s':'urn:schemas-microsoft-com:office:spreadsheet'}
    histories={}; source_files={}
    for ticker,sha in expected.items():
        p=prior/(ticker+'_fund_download_20261023.xls')
        assert digest(p)==sha
        raw=p.read_text()
        safe=re.sub(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)', '&amp;',raw)
        xml=ET.fromstring(safe)
        rows={}
        for sheet in xml.findall('s:Worksheet',ns):
            if sheet.attrib.get('{'+ns['s']+'}Name')!='Historical': continue
            for row in sheet.findall('s:Table/s:Row',ns)[1:]:
                vals=[c.find('s:Data',ns).text if c.find('s:Data',ns) is not None else '' for c in row.findall('s:Cell',ns)]
                if not vals: continue
                try: day=datetime.strptime(vals[0],'%b %d, %Y').date().isoformat()
                except ValueError: continue
                rows[day]={'NAV per Share':vals[1],'Ex-Dividends':vals[2],'Shares Outstanding':vals[3]}
        histories[ticker]=rows
        source_files[ticker]={'file':str(p.relative_to(ROOT)),'sha256':sha,'raw_unchanged':True}
    receipts={}
    for p in sorted(OUT.glob('*_RETRIEVAL.json')):
        rec=json.loads(p.read_text())
        assert digest(OUT/rec['raw_file'])==rec['sha256']
        receipts[p.stem]={'file':str(p.relative_to(ROOT)),'sha256':digest(p),
          'raw_file':rec['raw_file'],'raw_sha256':rec['sha256'],'status':rec['status'],
          'url':rec['requested_url'],'retrieved_utc':rec['retrieved_utc']}
    release=(OUT/'BlackRock_2023_split_Nasdaq_release.html').read_text()
    assert '401.11' in release and '552.87' in release and 'March 7, 2024' in release
    annual=PdfReader(OUT/'annual_2026_pdf_from_viewer_override.pdf')
    pages={n:annual.pages[n-1].extract_text() for n in (25,26,34,36)}
    assert '134,300,000' in pages[25] and '61,500,000' in pages[26]
    assert 'five-for-one' in pages[34] and 'three-for-one' in pages[36]
    sai=PdfReader(OUT/'current_SAI_actual_viewer_link.pdf')
    assert 'three-for-one' in sai.pages[0].extract_text()
    assert 'MTUM 4:1' in sai.pages[1].extract_text()
    nav_checks=[]; date_checks=[]
    for ticker,ratio,press_nav,page,annual_shares,annual_nav in [('IGV',5,'401.11',25,134300000,'80.03'),('SOXX',3,'552.87',26,61500000,'328.96')]:
        historical_nav=Decimal(histories[ticker]['2023-12-20']['NAV per Share'])
        native_reexpression=(historical_nav*ratio).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
        assert native_reexpression==Decimal(press_nav)
        nav_checks.append({'ticker':ticker,'date':'2023-12-20','historical_nav':str(historical_nav),
            'official_contemporaneous_native_nav':press_nav,'official_split_ratio':ratio,
            'historical_nav_times_official_ratio_at_published_cent_precision':str(native_reexpression),
            'classification':'restated','scope':'this pre-split NAV anchor, not a full-history certification',
            'source':'BlackRock_2023_split_Nasdaq_release.html',
            'source_publication':'2023-12-21 13:45 EST; issuer-authored BusinessWire release published by Nasdaq',
            'method':'same-date official native NAV anchor and announced ratio; no price-jump inference'})
        same=histories[ticker]['2026-03-31']; following=histories[ticker]['2026-04-01']
        assert Decimal(same['NAV per Share']).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)==Decimal(annual_nav)
        assert int(same['Shares Outstanding'])!=annual_shares
        assert int(following['Shares Outstanding'])==annual_shares
        date_checks.append({'ticker':ticker,'official_statement_date':'2026-03-31','official_pdf_page':page,
            'official_nav':annual_nav,'official_shares_outstanding':annual_shares,
            'historical_same_date':same,'historical_following_business_date':'2026-04-01',
            'historical_following_date':following,'nav_same_date_matches_to_cents':True,
            'shares_same_date_matches':False,'shares_following_date_matches':True,
            'conclusion':'observed date-label inconsistency; a global one-session shift is NOT established or applied'})
    classifications={}
    for ticker in expected:
        classifications[ticker]={'nav_basis':'restated' if ticker in ('IGV','SOXX') else 'unknown',
            'nav_scope':'same-date pre-2024-split anchor verified; annual comparative per-share footnote confirms restatement' if ticker in ('IGV','SOXX') else 'no independent pre-event native NAV anchor verified in this audit',
            'shares_outstanding_basis':'unknown','shares_scope':'no independent same-date pre-split physical share-count anchor verified; no global restatement claim',
            'native_classification_proven':False,'all_daily_history_verified':False}
    result={'schema':'ishares_split_convention_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'purpose':'source units and date semantics only; not factor/backtest admission',
      'raw_histories':source_files,'retrieval_receipts':receipts,
      'classifications':classifications,'nav_native_vs_restated_checks':nav_checks,
      'annual_report_date_alignment_checks':date_checks,
      'official_events':[
        {'ticker':'IGV','ratio':'5:1','record_date':'2024-03-04','payable_after_close':'2024-03-06','ex_date':'2024-03-07','source':'BlackRock_2023_split_Nasdaq_release.html'},
        {'ticker':'SOXX','ratio':'3:1','record_date':'2024-03-04','payable_after_close':'2024-03-06','ex_date':'2024-03-07','source':'BlackRock_2023_split_Nasdaq_release.html'},
        {'ticker':'IWM','ratio':'2:1','ex_date':'2005-06-09','source':'IWM_2005_CME_CFTC_notice.pdf','page':2,'publication':'CME submission 05-58 dated 2005-05-27; CFTC-hosted scan','verification':'page 2 rendered and visually read; not inferred from price'},
        {'ticker':'MTUM','ratio':'4:1','record_date':'2026-11-03','payable_after_close':'2026-11-04','ex_date':'2026-11-05','source':'current_SAI_actual_viewer_link.pdf','page':2,'announcement':'2026-10-05','after_raw_history_cutoff':True,'not_applied_early':True},
        {'ticker':'SOXX','ratio':'3:1','record_date':'2026-11-03','payable_after_close':'2026-11-04','ex_date':'2026-11-05','source':'current_SAI_actual_viewer_link.pdf','pages':[1,9],'announcement':'2026-08-21; reiterated 2026-10-23','after_raw_history_cutoff':True,'not_applied_early':True}],
      'annual_report_split_footnotes':{'file':'annual_2026_pdf_from_viewer_override.pdf','IGV_page':34,'SOXX_page':36,'scope':'comparative per-share financial highlights are explicitly adjusted; does not prove download share-count timing or all-history units'},
      'outstanding':[
        'Historical shares-outstanding date semantics and original availability/revision times are unverified.',
        'Two year-end share anchors occur in the next business-day row while same-date NAV matches; no silent shift permitted.',
        'No complete official per-ticker action ledger or same-date pre-split physical share-count anchor was obtained.',
        'IWM pre-2005 split units remain unknown despite the verified corporate action.',
        'MTUM and TLT lack independent historical unit-restatement anchors; absence of a discovered split is not proof of none.',
        'The 2026-11-05 MTUM/SOXX announced actions occur after the downloaded history cutoff and must not be applied early.'],
      'access_limits':{'sec_host':'www.sec.gov','first_request_status':403,'stop_receipt':'STOP_www.sec.gov_2026-10-23.json','further_sec_requests':0,'BusinessWire':'one timeout; no retry','old_ishares_literature':'two HTTP404 preserved; current actual product-link PDF retrieved separately'},
      'scope_assertions':{'no_flow_calculation':True,'no_features_or_predictions':True,'no_orders_or_returns':True,'no_raw_rewrites':True,'no_date_shift':True,'no_global_split_factor_application':True},
      'collector_sha256':digest(Path(__file__))}
    out=OUT/'SPLIT_CONVENTION_AUDIT.json'
    with out.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    assert all(digest(ROOT/v['file'])==v['sha256'] for v in source_files.values())
    return {'audit':str(out),'sha256':digest(out),'nav_restated_anchors':2,'date_alignment_mismatches':2,'share_basis_unknown':5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label',nargs='?')
    parser.add_argument('url',nargs='?')
    parser.add_argument('--audit',action='store_true',help='Offline read-only document/unit inspection; requires pypdf')
    args=parser.parse_args()
    if args.audit:
        if args.label or args.url: parser.error('--audit takes no URL or label')
        print(json.dumps(inspect_actions()))
    else:
        if not args.label or not args.url: parser.error('Collection requires label and URL')
        print(json.dumps(collect(args.label,args.url)))

if __name__=='__main__':
    main()
