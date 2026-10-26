"""Use dated original earnings bulletins/releases, never webcast clock times."""
import calendar
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from budvz_io import P,dump

MONTHS='January February March April May June July August September October November December'.split()
DATE='('+'|'.join(MONTHS)+r')\s+(\d{1,2})\s*,?\s+(20\d\d)'

def clean(text):
    text=' '.join(text.split())
    # Known extraction letter spacing, no inference of numerical dates.
    for word in MONTHS+['Quarterly','Investor']:
        text=re.sub('\\s*'.join(word),word,text,flags=re.I)
    return text

def publication(link):
    if 'path' not in link or 'meta' not in link:return None
    p=Path(link['path']);meta=link['meta']
    if meta['http_status']!=200: return None
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    if p.read_bytes().startswith(b'%PDF'):
        t=p.with_suffix('.txt')
        if not t.exists():return None
        text=clean(t.read_text())
        if not re.search(r'Investor Quarterly',text[:300],re.I):
            m=re.search(r'NEW YORK\s*(?:[-–—,]|\s)*\s*'+DATE,text[:2500],re.I)
            if not m and 'FOR IMMEDIATE RELEASE' in text[:300] and 'News Release' in text[:100]:
                m=re.search(DATE,text[:500],re.I)
            if not m or not re.search('Verizon',text[:2500],re.I):return None
            date=dt.date(int(m[3]),MONTHS.index(m[1].title())+1,int(m[2]))
            return date,text,'issuer_original_dated_earnings_release_pdf'
        m=re.search(DATE,text[:300],re.I)
        if not m:return None
        if not re.search('Verizon',text[:500],re.I) or not re.search(r'earnings\s*r\s*elease',text,re.I):return None
        date=dt.date(int(m[3]),MONTHS.index(m[1].title())+1,int(m[2]))
        return date,text,'issuer_original_dated_earnings_bulletin_pdf'
    if '/news/' in link['url']:
        s=BeautifulSoup(p.read_bytes(),'html.parser')
        text=clean(s.get_text(' ',strip=True));h=s.select_one('h1')
        if not h or not re.search(r'(Verizon.*(?:report|deliver|growth|result|momentum|earnings|network expansion|financial guidance)|(?:quarter|result).*Verizon)',h.get_text(),re.I):return None
        # Date visible on original news article, not mutable CMS dateModified.
        title=h.get_text(' ',strip=True);before=text[:text.find(title)]
        dates=list(re.finditer(r'\b(\d{2})/(\d{2})/(20\d\d)\b',before[-500:]))
        if dates:
            m=dates[-1];date=dt.date(int(m[3]),int(m[1]),int(m[2]))
        else:
            m=re.search(r'Published\s+([A-Za-z]{3,9})\s+(\d{1,2}),\s+(20\d\d)',text)
            if not m:return None
            month=next((i for i,v in enumerate(MONTHS,1) if v.lower().startswith(m[1].lower())),None)
            if month is None:return None
            date=dt.date(int(m[3]),month,int(m[2]))
        return date,text,'issuer_original_news_release_html'
    return None

def main():
    events=[];pending=[]
    supplemental=P/'vz/SUPPLEMENTAL_LINKS.json'
    extras=json.loads(supplemental.read_text()) if supplemental.exists() else []
    for row in json.loads((P/'vz/MANIFEST.json').read_text()):
        y,q=row['year'],row['quarter'];month=q*3
        period=dt.date(y,month,calendar.monthrange(y,month)[1])
        links=row.get('links',[])+[x for x in extras if (x['year'],x['quarter'])==(y,q)];pub=None
        for a in links:
            value=publication(a)
            if value and 0<(value[0]-period).days<100:
                pub=(a,value);break
        if not pub and (y,q)==(2016,4):
            observed=P/'vz/indexed_2016Q4.json'
            if observed.exists():
                o=json.loads(observed.read_text())
                assert o['original_response_available'] is False
                assert (o['observed_report_year'],o['observed_report_quarter'])==(y,q)
                a=dict(url=o['source_url'],path=str(observed),meta=dict(
                    sha256=hashlib.sha256(observed.read_bytes()).hexdigest(),retrieved_at=o['retrieved_at']))
                pub=(a,(dt.date.fromisoformat(o['observed_publication_date']),o['quoted_excerpt'],
                    'public_index_observation_not_original_bytes'))
        if not pub:
            pending.append(dict(year=y,quarter=q,reason='dated original earnings release/bulletin needed'));continue
        a,(release,text,kind)=pub;period_source=None
        for b in links:
            if 'path' not in b or 'meta' not in b:continue
            p=Path(b['path']);txt=p.with_suffix('.txt')
            if b['meta']['http_status']!=200 or not txt.exists():continue
            t=clean(txt.read_text())
            # Require the actual date in the associated current-quarter statement,
            # with an ended/statement context. Calendar value only builds a matcher.
            date_pattern=rf'\b{month}/{period.day}/(?:{y}|{str(y)[2:]})\b'
            m=re.search(r'(?:Mos\. Ended|months ended).{0,120}?'+date_pattern,t,re.I)
            if not m:
                m=re.search(r'As of '+period.strftime('%B')+rf' {period.day}, {y}.{{0,70}}Verizon Communications',t[:500],re.I)
            if m:
                assert hashlib.sha256(p.read_bytes()).hexdigest()==b['meta']['sha256']
                period_source=(b,m.group());break
        if period_source is None:
            pending.append(dict(year=y,quarter=q,reason='actual period end needs statement evidence',release_date=str(release)));continue
        b,excerpt=period_source
        e=dict(ticker='VZ',release_date=str(release),publication_timestamp=None,precision='date_only',
            publication_timezone='America/New_York',publication_timezone_basis='New York issuer-local dated original earnings document; no conference-call time used.',
            fiscal_period_end_actual=str(period),fiscal_year=y,fiscal_quarter=q,source_url=a['url'],
            evidence_path=a['path'],evidence_sha256=a['meta']['sha256'],source_kind=kind,
            retrieved_at=a['meta']['retrieved_at'],first_public_time_verified=False,
            evidence_excerpt=text[:250] if 'bulletin' in kind else f'Original article displayed publication date {release}',
            additional_evidence=[dict(evidence_path=b['path'],sha256=b['meta']['sha256'],source_url=b['url'],excerpt=excerpt)],
            period_basis='Actual ended/as-of date printed in the quarter-specific original statement',
            evidence_limitation='Document date is observed; exact first-public clock time is not independently verified.')
        events.append(e)
        if kind=='public_index_observation_not_original_bytes':
            e['evidence_excerpt']=text
            e['evidence_limitation']='Publication date from explicit public-index observation; primary SHA covers observation, not original HTML. Actual period from separately hashed original statement.'
    dump(P/'vz/events.json',events);dump(P/'vz/PENDING.json',pending)
    print('VZ verified',len(events),'pending',len(pending))

if __name__=='__main__':main()
