"""Issuer-original Walmart quarterly earnings evidence, with offline replay.

Walmart fiscal years end January 31; the separately disclosed 13-week comp-
sales reporting period is NOT its consolidated financial reporting quarter.
Only original release display dates/datelines establish calendar publication.
"""
import datetime as dt
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'
ARCHIVE = 'https://stock.walmart.com/financial-information/financial-results'
PDF_PYTHON = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
CUTOFF = '2026-09-24'
START = '2011-01-01'

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def sha(value):
    return hashlib.sha256(value).hexdigest()

def rel(path):
    return str(path.relative_to(ROOT))

def clean(text):
    return re.sub(r'\s+', ' ', text.replace('\u00a0',' ').replace('–','-').replace('—','-'))

def fetch(url, stem, suffix):
    path = RAW / (stem + suffix)
    meta_path = RAW / (stem + '.meta.json')
    if path.exists() and meta_path.exists():
        body = path.read_bytes(); meta = json.loads(meta_path.read_text())
        if meta['url'] != url or meta['sha256'] != sha(body):
            raise ValueError('Cached source or checksum changed')
        return body, path, meta
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    stop_path = HERE / 'SOURCE_STOPS.json'
    stops = json.loads(stop_path.read_text()) if stop_path.exists() else {}
    host = urlparse(url).hostname
    if host in stops and stops[host]['utc_date'] == today:
        raise RuntimeError('Source stopped for UTC day after 403/429: ' + host)
    try:
        req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 (public earnings research)'})
        with urllib.request.urlopen(req,timeout=40) as r:
            body=r.read(); meta=dict(url=url,final_url=r.geturl(),status=r.status,
                sha256=sha(body),retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),bytes=len(body),public_unauthenticated=True)
    except urllib.error.HTTPError as e:
        if e.code in (403,429):
            stops[host]=dict(utc_date=today,status=e.code,url=url,retry_after=e.headers.get('Retry-After'),policy='Stop host for rest of UTC day')
            dump(stop_path,stops)
        raise
    if suffix == '.pdf' and not body.startswith(b'%PDF'):
        raise ValueError('Expected original PDF bytes')
    path.write_bytes(body);dump(meta_path,meta);time.sleep(.15)
    return body,path,meta

def pdf_text(path):
    target = path.with_suffix('.txt'); sidecar = path.with_suffix('.text.meta.json')
    source_sha=sha(path.read_bytes())
    if target.exists() and sidecar.exists():
        info=json.loads(sidecar.read_text())
        if info['source_sha256']==source_sha and info['text_sha256']==sha(target.read_bytes()):
            return target.read_text()
    code='import sys;from pypdf import PdfReader; print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
    text=subprocess.check_output([PDF_PYTHON,'-c',code,str(path)],text=True)
    target.write_text(text);dump(sidecar,dict(source_sha256=source_sha,text_sha256=sha(target.read_bytes()),parser='bundled pypdf'))
    return text

def archive_items(body):
    soup=BeautifulSoup(body,'html.parser'); items=[]
    for info in soup.select('.results-info'):
        label=info.h3.get_text(' ',strip=True)
        match=re.fullmatch(r'(FY|Q[1-3]) (\d{4})',label)
        if not match: continue
        year=int(match[2]);quarter=4 if match[1]=='FY' else int(match[1][1])
        if (year,quarter)<(2011,4) or (year,quarter)>(2027,2):continue
        period_raw=info.select_one('.date').get_text(' ',strip=True)
        period=dt.datetime.strptime(re.sub(r'^(?:Quarter|Fiscal Year) Ended ','',period_raw),'%b %d, %Y').date().isoformat()
        box=info.parent
        links=box.select('a[href]')
        release=next((a['href'] for a in links if '/earnings_release/' in a['href']),None)
        html=next((a['href'] for a in links if 'HTML Release' in a.get_text()),None)
        items.append(dict(fiscal_year=year,fiscal_quarter=quarter,label=label,period_raw=period_raw,
            fiscal_period_end_actual=period,pdf_url=release,html_url=html))
    return sorted(items,key=lambda z:(z['fiscal_year'],z['fiscal_quarter']))

def download_all():
    RAW.mkdir(parents=True,exist_ok=True)
    body,_,_=fetch(ARCHIVE,'archive','.html')
    items=archive_items(body);dump(HERE/'archive_items.json',items)
    for item in items:
        label=f"{item['fiscal_year']}Q{item['fiscal_quarter']}"
        try:
            if not item['pdf_url']: raise ValueError('No original PDF linked')
            _,path,_=fetch(item['pdf_url'],label,'.pdf')
            text=pdf_text(path)
            print(label,len(text),clean(text)[:150],flush=True)
        except Exception as e:
            print(label,'ERROR',e,flush=True)
            if isinstance(e,urllib.error.HTTPError) and e.code in (403,429): break
            if isinstance(e,RuntimeError) and 'Source stopped' in str(e):break


MONTHS = {name.lower():i for i,name in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
MONTHS.update({k[:3]:v for k,v in list(MONTHS.items())})
DATE_RE = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}\s*,\s*\d{4})'
QWORDS = {1:'first',2:'second',3:'third',4:'fourth'}

def iso_date(text):
    m=re.fullmatch(r'([A-Za-z]+)\.?\s+(\d{1,2})\s*,\s*(\d{4})',text)
    if not m:raise ValueError('Invalid original calendar date')
    return dt.date(int(m[3]),MONTHS[m[1].lower()],int(m[2])).isoformat()

def parse_release(text,item):
    text=clean(text)
    year,quarter=item['fiscal_year'],item['fiscal_quarter']
    # Quarter identity must occur in the release's opening headline/highlights;
    # forward guidance or previous-year comparatives later in the PDF cannot
    # establish the current reporting quarter.
    qpattern=rf'(?:\bQ\s*{quarter}\b|\b{QWORDS[quarter]}[- ]quarter\b)'
    identity=re.search(qpattern,text[:1800],re.I)
    if not identity:raise ValueError('Original headline does not confirm expected fiscal quarter')
    # FY identity is backed by the issuer archive label AND matching original
    # financial reporting end. FY guidance appearing elsewhere is not used.
    direct=list(re.finditer(r'BENTONVILLE,\s*Ark\.?\s*,\s*'+DATE_RE,text,re.I))
    footers=list(re.finditer(r'(?:NYSE|NASDAQ):\s*WMT\s+'+DATE_RE+r'\s+stock\.walmart\.com',text,re.I))
    matches=direct+footers
    if not matches:raise ValueError('No original release dateline or publication-date footer; CMS/call dates are insufficient')
    dates={iso_date(m[1]) for m in matches}
    if len(dates)!=1:raise ValueError('Conflicting original release publication dates')
    release_date=next(iter(dates))
    period=item['fiscal_period_end_actual']
    p=dt.date.fromisoformat(period)
    # Independently validate archive fiscal labels against Walmart consolidated
    # reporting months. This is NOT the separate 13-week comparable-sales date.
    expected_month={1:4,2:7,3:10,4:1}[quarter]
    expected_year=year if quarter==4 else year-1
    expected_day=30 if quarter==1 else 31
    if (p.year,p.month,p.day)!=(expected_year,expected_month,expected_day):
        raise ValueError('Issuer archive fiscal label/end conflict')
    month_name=next(k.title() for k,v in MONTHS.items() if v==p.month and len(k)>3)
    if p.month==5:month_name='May'
    month_rx=rf'(?:{month_name}|{month_name[:3]}\.?)\s+{p.day}'
    # Older PDF extraction reorders table headings; its original prose directly
    # states the quarter and year ended. Require the financial-results context.
    narrative=re.search(r'(?:financial results|results) for (?:the )?[^.]{0,100}quarter[^.]{0,100}ended\s+'+month_rx+rf'\s*,\s*{p.year}\b',text,re.I)
    if narrative:
        period_excerpt=narrative[0]
        period_basis='Explicit financial-results current quarter end in original release body; cross-checked against official fiscal-quarter archive'
    else:
        # Only the current consolidated income-statement heading is eligible;
        # never scan an arbitrary comp-sales date or a future guidance date.
        statement=re.search(r'(?:Condensed )?Consolidated Statements of Income',text,re.I)
        if not statement:raise ValueError('Original consolidated income statement absent')
        head=text[statement.start():statement.start()+600]
        head=head.split('Revenues:')[0]
        if not re.search(r'(?:Three Months|Quarters) Ended',head,re.I):
            raise ValueError('Current quarter statement heading absent')
        if not re.search(month_rx+r'\s*,',head,re.I) or not re.search(rf'\b{p.year}\s+{p.year-1}\b',head):
            raise ValueError('Original current/prior year financial statement columns do not confirm actual fiscal end')
        period_excerpt=head
        period_basis='Original consolidated income statement quarter-end month/day and current/prior year columns; cross-checked against official fiscal-quarter archive'
    if not START<=release_date<=CUTOFF:raise ValueError('Release outside requested evidence window')
    if not period<release_date:raise ValueError('Fiscal period must end before earnings publication')
    return dict(release_date=release_date,release_excerpt=matches[0][0],
        release_basis='Original issuer release Bentonville dateline' if direct else 'Publication date printed in original issuer release NYSE/NASDAQ WMT footer',
        location_excerpt=direct[0][0] if direct else None,
        identity_excerpt=text[max(0,identity.start()-65):identity.end()+150],
        period_excerpt=period_excerpt,period_basis=period_basis)

def proof(path,url,excerpt=None):
    out=dict(evidence_path=rel(path),sha256=sha(path.read_bytes()),source_url=url)
    if excerpt is not None:out['excerpt']=excerpt
    return out

def collect_events():
    archive_path=RAW/'archive.html'
    archive_body=archive_path.read_bytes()
    archive_meta=json.loads((RAW/'archive.meta.json').read_text())
    if sha(archive_body)!=archive_meta['sha256']:raise ValueError('Archive checksum mismatch')
    soup=BeautifulSoup(archive_body,'html.parser')
    org=next(json.loads(tag.string) for tag in soup.select('script[type="application/ld+json"]') if tag.string and '"@type":"Corporation"' in tag.string)
    if org['address']['addressLocality']!='Bentonville, AR':raise ValueError('Issuer-local date timezone mapping needs review')
    archive_proof=proof(archive_path,ARCHIVE)
    location_proof=proof(archive_path,ARCHIVE,json.dumps(org['address']))
    events=[];pending=[]
    for item in archive_items(archive_body):
        label=f"{item['fiscal_year']}Q{item['fiscal_quarter']}"
        path=RAW/(label+'.pdf');txtpath=path.with_suffix('.txt')
        try:
            body=path.read_bytes();meta=json.loads(path.with_suffix('.meta.json').read_text())
            if meta['url']!=item['pdf_url'] or sha(body)!=meta['sha256']:raise ValueError('Release checksum/source mismatch')
            parsed=parse_release(pdf_text(path),item)
            release_proof=proof(path,item['pdf_url'],parsed['release_excerpt'])
            dateline=parsed['location_excerpt']
            row=dict(ticker='WMT',release_date=parsed['release_date'],publication_timestamp=None,
                precision='date_only',publication_timezone='America/Chicago',
                publication_timezone_basis='Conservative issuer-local calendar-date policy: Bentonville, Arkansas -> America/Chicago; not an observed website timezone or publication instant.',
                publication_timezone_evidence=dict(kind='issuer_local_calendar_date_policy',
                    observed_dateline=dateline,geographic_mapping='Bentonville, Arkansas -> America/Chicago',
                    evidence_path=rel(path) if dateline else rel(archive_path),sha256=sha(body) if dateline else archive_proof['sha256'],
                    issuer_address_fallback=None if dateline else location_proof,
                    cms_timezone_observed=False,exact_publication_time_inferred=False),
                first_public_time_verified=False,official_date_published_raw=parsed['release_excerpt'],
                fiscal_period_end_actual=item['fiscal_period_end_actual'],fiscal_year=item['fiscal_year'],fiscal_quarter=item['fiscal_quarter'],
                source_url=item['pdf_url'],source_class='issuer_original_press_release',
                evidence_path=rel(path),evidence_sha256=sha(body),evidence_excerpt=parsed['identity_excerpt'],
                source_retrieved_at=meta['retrieved_at'],body_text_evidence_path=rel(txtpath),body_text_sha256=sha(txtpath.read_bytes()),
                release_date_basis=parsed['release_basis'],release_date_evidence=[release_proof],
                fiscal_period_evidence=dict(proof(path,item['pdf_url'],parsed['period_excerpt']),basis=parsed['period_basis']),
                official_archive_evidence=dict(archive_proof,quarter_label=item['label'],actual_period_end_label=item['period_raw'],original_pdf_link=item['pdf_url']),
                additional_evidence=[dict(evidence_path=archive_proof['evidence_path'],sha256=archive_proof['sha256'],role='official_quarter_identity_period_end_pdf_link_and_issuer_location')],
                fiscal_calendar_note='Fiscal year ends January 31: FY Q1/Q2/Q3 calendar years are FY minus one. Separate 13-week comp-sales dates are not consolidated fiscal period ends.',
                timestamp_note='No PDF creation/modification metadata, CMS update time, scheduled call time or SEC acceptance time is treated as actual publication time.',
                historical_limitation='Current issuer-hosted original release archive, not a contemporaneously captured immutable historical news feed.',
                in_study_window=parsed['release_date']>='2016-01-01')
            events.append(row)
            print(label,parsed['release_date'],'ready',flush=True)
        except Exception as exc:
            pending.append(dict(item,status='pending_review',reason=str(exc)))
            print(label,'PENDING',str(exc),flush=True)
    events.sort(key=lambda r:r['release_date'])
    dump(HERE/'events.json',events);dump(HERE/'pending.json',pending)
    expected=[(y,q) for y in range(2011,2028) for q in range(1,5) if (2011,4)<=(y,q)<=(2027,2)]
    observed={(e['fiscal_year'],e['fiscal_quarter']) for e in events}
    missing=[f'{y}Q{q}' for y,q in expected if (y,q) not in observed]
    status=dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),ticker='WMT',requested_release_window=[START,CUTOFF],
        expected_quarterly_events=len(expected),verified_events=len(events),pending_count=len(pending),missing_quarters=missing,
        earliest_release=min((e['release_date'] for e in events),default=None),latest_release=max((e['release_date'] for e in events),default=None),
        all_events_date_only=True,approximate_times_inferred=False,full_original_pdf_bytes_downloaded=True,full_archive_html_downloaded=True,
        fiscal_quarter_coverage_complete=not missing,source='Walmart official investor-relations archive and original release PDFs',
        source_stops=json.loads((HERE/'SOURCE_STOPS.json').read_text()) if (HERE/'SOURCE_STOPS.json').exists() else {},
        scope='Quarterly results only; interim business updates/preannouncements are not claimed complete.',unrelated_accounts_or_prices_modified=False)
    dump(HERE/'STATUS.json',status)
    dump(HERE/'PENDING.json',dict(missing_quarters=missing,pending=pending,
        limitations=['Exact first-public publication timestamps remain unverified; all events are date-only.',
          'Current issuer archive is not an immutable contemporaneous archive.',
          'Out-of-cycle business updates and preannouncements are outside the regular-quarterly-results collection.']))
    return events,pending

if __name__=='__main__':
    download_all()
    collect_events()
