"""Cache original PG earnings material; date-only facts require body evidence."""
import datetime as dt
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'
CUTOFF = '2026-09-24'
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
def dump(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(b): return hashlib.sha256(b).hexdigest()
def rel(p): return str(p.relative_to(ROOT))
def fetch(url,stem,suffix='.html'):
    RAW.mkdir(exist_ok=True,parents=True)
    p=RAW/(stem+suffix); mp=RAW/(stem+'.meta.json')
    if p.exists() and mp.exists():
        b=p.read_bytes(); m=json.loads(mp.read_text())
        if m['url'] != url or m['sha256'] != sha(b): raise ValueError('Cache mismatch '+str(p))
        if m['status'] != 200: raise ValueError('Cached HTTP '+str(m['status']))
        return b,p,m
    if mp.exists(): raise ValueError('Cache stem already reserved for a different source/format: '+stem)
    stops_path=HERE/'SOURCE_STOPS.json'; stops=json.loads(stops_path.read_text()) if stops_path.exists() else {}
    host=urllib.parse.urlsplit(url).netloc
    if host in stops: raise RuntimeError('Stopped source '+host)
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public earnings research)'}),timeout=40) as r:
            b=r.read(); m=dict(url=url,final_url=r.geturl(),status=r.status,content_type=r.headers.get('Content-Type'))
    except urllib.error.HTTPError as e:
        b=e.read(); m=dict(url=url,status=e.code,retry_after=e.headers.get('Retry-After'))
    m.update(retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=sha(b),bytes=len(b),public_unauthenticated=True)
    p.write_bytes(b);dump(mp,m)
    if m['status'] in [403,429]:
        stops[host]=dict(m,policy='Stop source for this task; no bypass or automated retries');dump(stops_path,stops)
    if m['status']!=200: raise ValueError('HTTP '+str(m['status'])+' '+url)
    time.sleep(.2)
    return b,p,m

def pdf_text(path):
    out=path.with_suffix('.txt'); mp=path.with_suffix('.text.meta.json'); original=sha(path.read_bytes())
    if out.exists() and mp.exists():
        m=json.loads(mp.read_text())
        if m['source_sha256']==original and m['text_sha256']==sha(out.read_bytes()): return out.read_text()
    code='import sys; from pypdf import PdfReader; print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
    text=subprocess.check_output([PDF_PYTHON,'-c',code,str(path)],text=True)
    out.write_text(text);dump(mp,dict(source_sha256=original,text_sha256=sha(out.read_bytes()),parser='bundled pypdf'))
    return text

from bs4 import BeautifulSoup
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sept?\\.?|Oct\\.?|Nov\\.?|Dec\\.?'
DATE = r'(?:'+MONTHS+r')\s+\d{1,2}\s*,?\s*\d{4}'
QWORDS={'first':1,'second':2,'third':3,'fourth':4}
def clean(t): return re.sub(r'\s+',' ',t.replace('\u00a0',' ').replace('–','-').replace('—','-'))
def iso_date(t):
    t=clean(t).replace(',',' ').replace('.','')
    parts=t.split(); parts[0]=parts[0][:3]
    return dt.datetime.strptime(' '.join(parts),'%b %d %Y').date().isoformat()
def html_text(b):
    s=BeautifulSoup(b,'html.parser')
    for x in s(['script','style']): x.decompose()
    return clean(s.get_text(' ',strip=True))
def parse_release(text):
    t=clean(text)
    dateline=re.search(r'\bCINCINNATI\s*,\s*('+DATE+r')\s*[-:]?',t,re.I)
    if not dateline: raise ValueError('Original release Cincinnati calendar-date dateline absent')
    release_date=iso_date(dateline.group(1)); body=t[dateline.start():]
    if not '2011-01-01'<=release_date<=CUTOFF: raise ValueError('Original dateline outside requested release window')
    # Company identity and observed results must be in the same original release paragraph.
    release_start=t.rfind('News Release',0,dateline.start())
    if release_start<0 or dateline.start()-release_start>900: release_start=dateline.start()
    identity_text=t[release_start:dateline.end()+1200]
    intro=body[:1200]
    if not re.search(r'Procter\s*(?:&|and)\s*Gamble\s+Company',intro,re.I): raise ValueError('Issuer identity missing from original release')
    qm=re.search(r'\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter\b',identity_text,re.I)
    if not qm: raise ValueError('Actual quarterly release identity absent; may be another corporate announcement')
    if not re.search(r'\b(reported|announced|announces|delivered|achieved|increased)\b',intro,re.I): raise ValueError('Actual reported results context absent')
    if re.search(r'\b(to webcast|will webcast|plans to release|will release)\b',intro,re.I): raise ValueError('Scheduled event is not an actual release')
    q=QWORDS[qm.group(1).lower()]
    expected_month={1:'September',2:'December',3:'March',4:'June'}[q]
    expected_day=30 if q in (1,4) else 31
    # Require an actual current-quarter financial statement period in original exhibit.
    periods=[]
    for m in re.finditer(r'\bThree\s+Months\s+Ended\s+('+DATE+r')',body,re.I):
        actual=iso_date(m.group(1))
        if actual < release_date: periods.append((actual,m.group(0)))
    periods.sort(reverse=True)
    period_basis='Explicit Three Months Ended date in original release exhibit'
    if not periods:
        wrapper=re.search(r'On\s+('+DATE+r'),?\s+The Procter & Gamble Company.{0,50}?issued a news release with respect to earnings for the quarter ended\s+('+DATE+r')',t,re.I)
        if wrapper and iso_date(wrapper.group(1))==release_date:
            periods.append((iso_date(wrapper.group(2)),wrapper.group(0)))
            period_basis='Explicit issuer statement linking the same-dated earnings release to its actual quarter end in the same full filing; original exhibit dateline independently supplies publication date'
    if not periods: raise ValueError('Original release lacks explicit Three Months Ended calendar date')
    actual,period_excerpt=periods[0]
    end=dt.date.fromisoformat(actual)
    if (end.month,end.day)!=({1:9,2:12,3:3,4:6}[q],expected_day): raise ValueError('Original fiscal ordinal and actual period disagree')
    if not 0<(dt.date.fromisoformat(release_date)-end).days<100: raise ValueError('Stale period: no current quarterly actual end')
    fy=end.year+(1 if q in (1,2) else 0)
    # Corroborate June-ending fiscal-year label in original body, never archive calendar year.
    fym=re.search(r'\bfiscal\s+(?:year\s+)?'+str(fy)+r'\b',body,re.I)
    if not fym: raise ValueError('Original release does not corroborate inferred June-ending fiscal year')
    return dict(release_date=release_date,release_date_excerpt=dateline.group(0),
        fiscal_year=fy,fiscal_quarter=q,fiscal_period_end_actual=actual,
        fiscal_period_excerpt=period_excerpt,fiscal_period_basis=period_basis,fiscal_identity_excerpt=identity_text[max(0,qm.start()-50):qm.end()+120],
        fiscal_year_excerpt=body[max(0,fym.start()-55):fym.end()+90])

def evidence(p,u,excerpt=None):
    result=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=u)
    if excerpt: result['excerpt']=excerpt
    return result

def build():
    discovery=json.loads((HERE/'DISCOVERY.json').read_text()); events=[]; rejected=[]
    feed_path=RAW/'issuer_8k_feed_cik.json'; feed_meta=json.loads(feed_path.with_suffix('.meta.json').read_text())
    for item in discovery['selected']:
        p=RAW/('filing_'+str(item['FilingId'])+'.html')
        if not p.exists():
            rejected.append(dict(filing_id=item['FilingId'],status='not_downloaded',reason='Pending raw source collection'));continue
        meta=json.loads(p.with_suffix('.meta.json').read_text());url=meta['url']
        if meta['status']!=200:
            rejected.append(dict(filing_id=item['FilingId'],status='pending_review',reason='HTTP '+str(meta['status'])));continue
        raw=p.read_bytes()
        if sha(raw)!=meta['sha256']: raise ValueError('Raw source checksum mismatch')
        try: result=parse_release(html_text(raw))
        except ValueError as err:
            rejected.append(dict(filing_id=item['FilingId'],filing_date_discovery_only=item['FilingDate'],source_url=url,evidence_path=rel(p),status='not_admitted',reason=str(err)));continue
        proof=evidence(p,url)
        events.append(dict(ticker='PG',release_date=result['release_date'],publication_timestamp=None,
            fiscal_period_end_actual=result['fiscal_period_end_actual'],fiscal_year=result['fiscal_year'],fiscal_quarter=result['fiscal_quarter'],
            source_url=url,source_class='issuer_linked_original_earnings_release_exhibit_html',
            evidence_path=rel(p),evidence_sha256=sha(raw),source_retrieved_at=meta['retrieved_at'],
            precision='date_only',publication_timezone='America/New_York',
            publication_timezone_basis='Conservative issuer-local calendar-date research policy: original release Cincinnati, Ohio dateline maps to America/New_York. No source clock or CMS timezone asserted.',
            publication_timezone_evidence=dict(proof,kind='issuer_local_calendar_date_policy',observed_dateline=result['release_date_excerpt'],geographic_mapping='Cincinnati, Ohio -> America/New_York',cms_timezone_observed=False,exact_publication_time_inferred=False),
            first_public_time_verified=False,official_date_published_raw=result['release_date_excerpt'],
            release_date_basis='Actual Cincinnati dateline inside original issuer earnings news release exhibit, not SEC filing/acceptance date.',
            release_date_evidence=[dict(proof,kind='original_release_dateline',excerpt=result['release_date_excerpt'])],
            fiscal_period_evidence=dict(proof,excerpt=result['fiscal_period_excerpt'],basis=result['fiscal_period_basis']),
            fiscal_identity_evidence=dict(proof,quarter_excerpt=result['fiscal_identity_excerpt'],year_excerpt=result['fiscal_year_excerpt'],basis='Original quarter ordinal and fiscal-year wording cross-checked with actual fiscal end; fiscal year ends June 30.'),
            official_archive_evidence=dict(evidence(feed_path,feed_meta['url']),filing_id=item['FilingId'],role='Issuer-maintained link provenance only; all filing/received/create timestamps excluded from release facts'),
            additional_evidence=[dict(evidence_path=rel(feed_path),sha256=sha(feed_path.read_bytes()),role='official_issuer_archive_original_document_link')],
            timestamp_note='No archive PressReleaseDate clock, PDF metadata, SEC FilingDate/ReceivedDate/CreateDate, or conference-call clock used as actual publication instant.',
            historical_limitation='Full issuer-linked Q4 CDN archival rendering of issuer filing including original release exhibit; retrieved currently, not an immutable contemporaneous news feed.',
            in_study_window=result['release_date']>='2016-01-01'))
    events.sort(key=lambda r:r['release_date'])
    keys=[(x['fiscal_year'],x['fiscal_quarter']) for x in events]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate fiscal events need manual review')
    expected=[(fy,q) for fy in range(2011,2027) for q in range(1,5) if (fy,q)>=(2011,2)]
    missing=['%dQ%d'%x for x in expected if x not in keys]
    dump(HERE/'events.json',events)
    dump(HERE/'PENDING.json',dict(missing_quarters=missing,candidates_not_admitted=rejected,
        note='Discovery filing dates are neither publication evidence nor fiscal dates. Missing quarters stay missing; same-day nonrelease supplements are excluded.'))
    dump(HERE/'STATUS.json',dict(ticker='PG',generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),verified_events=len(events),requested_release_window=['2011-01-01',CUTOFF],
        earliest_release=events[0]['release_date'] if events else None,latest_release=events[-1]['release_date'] if events else None,
        missing_quarters=missing,all_events_date_only=True,exact_publication_timestamps=0,
        source='Official PG investor archive -> full Q4 CDN HTML including original earnings release exhibits',
        source_stops=json.loads((HERE/'SOURCE_STOPS.json').read_text()) if (HERE/'SOURCE_STOPS.json').exists() else {},
        no_yahoo_av_sec_requests=True,no_strategies_prices_or_ledgers_modified=True))
    print('events',len(events),'missing',missing)
    for r in rejected: print('NOT_ADMITTED',r['filing_id'],r['reason'])

if __name__=='__main__': build()
