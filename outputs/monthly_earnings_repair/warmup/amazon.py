"""Earlier Amazon releases: actual Seattle body dates, never migrated CMS dates."""
import datetime as dt
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from collect import P, fetch, dump
from extend import DATE, date


def parse(file, meta, archive):
    soup=BeautifulSoup(file.read_text(),'html.parser')
    node=soup.select_one('.PressReleasePage-articleBody')
    title=soup.select_one('.PressReleasePage-headline')
    if not node or not title:raise ValueError('Missing original body/title')
    text=re.sub(r'\s+',' ',node.get_text(' ',strip=True))
    if not re.search(r'Amazon\.com Announces',title.get_text(),re.I):raise ValueError('Unexpected issuer/title')
    # Legacy releases place publication date before Business Wire, newer ones after.
    opening=text[:2200]
    hit=re.search(r'SEATTLE[ ,—–-]*'+DATE,opening,re.I)
    if not hit:hit=re.search(r'SEATTLE[ ,—–-]*\(BUSINESS WIRE\)[ ,—–-]*'+DATE,opening,re.I)
    end=re.search(r'(first|second|third|fourth) quarter ended '+DATE,opening,re.I)
    if not hit or not end:raise ValueError('Original publication dateline/period missing')
    day,period=date(hit[1]),date(end[2]);q=['first','second','third','fourth'].index(end[1].lower())+1
    if not '2011-01-01'<=day<'2015-01-01':raise ValueError('Outside earlier scope')
    if int(period[5:7])!=q*3 or not 0<(dt.date.fromisoformat(day)-dt.date.fromisoformat(period)).days<100:raise ValueError('Fiscal identity invalid')
    return dict(ticker='AMZN',release_date=day,publication_timestamp=None,precision='date_only',
        publication_timezone='America/Los_Angeles',publication_timezone_basis='Original SEATTLE dateline, conservative issuer-local date; no exact timestamp inferred.',
        fiscal_period_end_actual=period,fiscal_year=int(period[:4]),fiscal_quarter=q,
        source_url=meta['url'],evidence_path=str(file),evidence_sha256=meta['sha256'],retrieved_at=meta['retrieved_at'],
        source_kind='issuer_original_press_release',release_date_evidence=hit[0],fiscal_period_evidence=end[0],
        discovery_archive=archive,modified_time_used_as_publication=False,first_public_time_verified=False)


def main():
    links={};archives=[];events=[];errors=[]
    for year in range(2011,2015):
        url='https://press.aboutamazon.com/press-release-archive?q=quarter&f2='+str(year)
        seen=set()
        while url and url not in seen:
            seen.add(url)
            try:
                f,m=fetch(url);archives.append(m)
                if m['http_status']!=200:raise ValueError('Archive HTTP '+str(m['http_status']))
                soup=BeautifulSoup(f.read_text(),'html.parser')
                for h in soup.select('h2'):
                    a=h.find('a',href=True)
                    if a and re.search(r'^Amazon\.com Announces (First|Second|Third|Fourth) Quarter (Results|Sales)',h.get_text(' ',strip=True),re.I):links[urljoin(url,a['href'])]=url
                nxt=[a['href'] for a in soup.select('a[href]') if a.get_text(' ',strip=True)=='Load more']
                url=urljoin(url,nxt[0]) if nxt else None
            except Exception as exc:errors.append(dict(url=url,error=str(exc)));break
    dump('AMZN_CATALOG.json',dict(links=links,archives=archives))
    for url,archive in links.items():
        try:
            f,m=fetch(url)
            if m['http_status']!=200:raise ValueError('Release HTTP '+str(m['http_status']))
            e=parse(f,m,archive);events.append(e)
            print(e['release_date'],e['fiscal_period_end_actual'],flush=True)
        except Exception as exc:errors.append(dict(url=url,error=str(exc)))
        dump('amazon_events.json',events);dump('AMZN_PENDING.json',errors)
    print('events',len(events),'errors',len(errors),flush=True)


if __name__=='__main__':main()
