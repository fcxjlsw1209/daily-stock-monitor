"""Repair exceptional early original releases without changing archived collectors."""
import datetime as dt
import json
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from collect import P, BASE, module, fetch, dump, sha

DATE = r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\s+\d{1,2},?\s+\d{4})'


def date(s):
    s=s.replace(',', '').replace('.', '').replace('Sept ', 'Sep ')
    for fmt in ('%B %d %Y','%b %d %Y'):
        try:return dt.datetime.strptime(s,fmt).date().isoformat()
        except ValueError:pass
    raise ValueError('Unrecognized date '+s)


def main():
    output = P / 'extra_events.json'
    events = json.loads(output.read_text()) if output.exists() else []
    bykey = {(e['ticker'], e['release_date']): e for e in events}
    errors = []
    apple = module('extend_apple', BASE / 'apple/collect.py')
    targets = {t['reported_date']: t for t in json.loads(apple.VENDOR.read_text())}
    for day, url in [('2012-01-24','https://www.apple.com/newsroom/2012/01/24Apple-Reports-First-Quarter-Results/'),
                     ('2013-01-23','https://www.apple.com/newsroom/2013/01/23Apple-Reports-Record-Results/')]:
        try:
            file,m=fetch(url)
            assert m['http_status']==200
            body=file.read_bytes()
            if day=='2012-01-24':
                # Only normalize the explicitly observed intervening phrase for parser;
                # checksum/excerpt below always refer to original bytes, not transformation.
                modified=body.replace(b'first quarter which spanned 14 weeks and ended',b'first quarter ended')
                e=apple.parse_html(modified,targets[day],url,str(file))
                e['evidence_sha256']=sha(body)
                e['evidence_excerpt']='fiscal 2012 first quarter which spanned 14 weeks and ended December 31, 2011'
                e['parser_normalization']='Accept intervening explicit 14-week fiscal-period phrase; original bytes preserved.'
            else: e=apple.parse_html(body,targets[day],url,str(file))
            e['retrieved_at']=m['retrieved_at'];bykey[(e['ticker'],day)]=e
        except Exception as exc: errors.append(dict(ticker='AAPL',url=url,error=str(exc)))
    source=BASE/'nvidia';catalog=json.loads((source/'release_catalog.json').read_text());links={}
    for row in catalog['archives']:
        soup=BeautifulSoup((source/row['evidence_path']).read_text(),'html.parser')
        for a in soup.select('a[href]'):
            url=urljoin('https://nvidianews.nvidia.com',a['href'])
            if re.search(r'/news/nvidia-(?:reports|announces|financial).*results',url) and re.search(r'201[1-5]',url):
                links[url]=row['url']
    versions=[]
    for url,discovery in links.items():
        if (url.endswith('fiscal-2015') and any(x in url for x in ['fourth-quarter','third-quarter','second-quarter','first-quarter'])):
            existing=[e for e in json.loads((source/'outside_scope_events.json').read_text()) if e['source_url']==url]
        else: existing=[]
        try:
            if existing:
                e=existing[0];e['evidence_path']=str(source/e['evidence_path'])
            else:
                file,m=fetch(url)
                if m['http_status']!=200:raise ValueError('HTTP '+str(m['http_status']))
                soup=BeautifulSoup(file.read_text(),'html.parser')
                title=soup.select_one('.article-title').get_text(' ',strip=True)
                pub=date(soup.select_one('.article-date').get_text(' ',strip=True))
                if not '2011-01-01'<=pub<'2015-01-01':continue
                text=re.sub(r'\s+',' ',soup.select_one('.article-body').get_text(' ',strip=True))
                fiscal=re.search(r'(first|second|third|fourth)\s+quarter.*?fiscal\s+(?:year\s+)?(\d{4})',title,re.I)
                ended=re.search(r'\bended\s+'+DATE,text[:6000],re.I)
                if not fiscal or not ended:raise ValueError('Missing fiscal identity or actual quarter end')
                period=date(ended[1]);q=['first','second','third','fourth'].index(fiscal[1].lower())+1
                if not 0<(dt.date.fromisoformat(pub)-dt.date.fromisoformat(period)).days<100:raise ValueError('Period lag invalid')
                e=dict(ticker='NVDA',release_date=pub,publication_timestamp=None,precision='date_only',
                    publication_timezone='America/Los_Angeles',
                    publication_timezone_basis='Issuer Santa Clara dateline; conservative issuer-local date, exact time not inferred.',
                    fiscal_period_end_actual=period,fiscal_year=int(fiscal[2]),fiscal_quarter=q,
                    source_url=url,evidence_path=str(file),evidence_sha256=m['sha256'],retrieved_at=m['retrieved_at'],
                    source_kind='issuer_original_newsroom',source_title=title,
                    fiscal_period_evidence=ended[0],archive_source_url=discovery,first_public_time_verified=False)
            if not '2011-01-01'<=e['release_date']<'2015-01-01':continue
            if e['release_date']=='2014-05-08':
                e['publication_version']='Repeated May 6 results unchanged, as explicitly stated; archived separately and not a new event.'
                versions.append(e);continue
            if e['release_date']=='2014-05-06':e['publication_version']='Early full results released after an inadvertent internal email; May 8 repetition confirms unchanged figures.'
            bykey[(e['ticker'],e['release_date'])]=e
            print('NVDA',e['release_date'],e['fiscal_period_end_actual'],flush=True)
        except Exception as exc:errors.append(dict(ticker='NVDA',url=url,error=str(exc)))
        dump('extra_events.json',list(bykey.values()));dump('EXTRA_PENDING.json',errors)
    dump('extra_events.json',list(bykey.values()));dump('EXTRA_PENDING.json',errors);dump('REPEATED_VERSIONS.json',versions)


if __name__=='__main__':main()
