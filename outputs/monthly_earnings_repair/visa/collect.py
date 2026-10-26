"""Visa original earnings-release PDFs. Offline reproducible admission, no vendor calls.

Public indexed IR rendering is used only to discover original document links.
Publication evidence must be a dateline within the downloaded release itself.
Visa FY Q1 ends December of the prior calendar year; Q2/Q3/Q4 end Mar/Jun/Sep.
"""
import calendar
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / 'raw'
CUTOFF = '2026-09-24'
PDF_PYTHON = '/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'
QWORDS = {1:'first',2:'second',3:'third',4:'fourth'}
PDF_TEXT_PARSER = 'bundled pypdf layout extraction v2; financial table headings retain visual order'

def sha(body): return hashlib.sha256(body).hexdigest()
def dump(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rel(path): return str(path.relative_to(ROOT))
def clean(text): return re.sub(r'\s+', ' ',text.replace('\xa0',' ').replace('–','-').replace('—','-'))

def fetch(url, stem, suffix=".pdf"):
    path, meta_path = RAW/(stem+suffix), RAW/(stem+'.meta.json')
    if path.exists() and meta_path.exists():
        body=path.read_bytes(); meta=json.loads(meta_path.read_text())
        if meta['url'] != url or meta['sha256'] != sha(body): raise ValueError('Cache hash/URL mismatch')
        return body,path,meta
    if not (url.startswith('https://s1.q4cdn.com/050606653/') or url.startswith('https://usa.visa.com/about-visa/newsroom/press-releases.releaseId.')): raise ValueError('Only observed official Visa CDN paths permitted')
    stop_path=HERE/'SOURCE_STOPS.json'; stops=json.loads(stop_path.read_text()) if stop_path.exists() else {}
    day=dt.datetime.now(dt.timezone.utc).date().isoformat(); host=urlsplit(url).netloc
    if stops.get(host,{}).get('utc_date') == day: raise RuntimeError('Source stopped today: '+host)
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public earnings research)'}),timeout=40) as response:
            body=response.read()
            meta=dict(url=url,final_url=response.geturl(),status=response.status,retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=sha(body),bytes=len(body),public_unauthenticated=True)
    except urllib.error.HTTPError as err:
        if err.code in (403,429):
            stops[host]=dict(utc_date=day,status=err.code,url=url,retry_after=err.headers.get('Retry-After'),policy='Stop this source for the rest of the UTC day')
            dump(stop_path,stops)
        raise
    if suffix == '.pdf' and not body.startswith(b'%PDF'): raise ValueError('Expected original PDF bytes')
    path.write_bytes(body); dump(meta_path,meta); time.sleep(.15)
    return body,path,meta

def pdf_text(path):
    target=path.with_suffix('.txt'); sidecar=path.with_suffix('.text.meta.json'); source_hash=sha(path.read_bytes())
    if target.exists() and sidecar.exists():
        m=json.loads(sidecar.read_text())
        if m['source_sha256']==source_hash and m['text_sha256']==sha(target.read_bytes()) and m.get('parser') == PDF_TEXT_PARSER: return target.read_text()
    code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sys.argv[1]).pages))'
    text=subprocess.check_output([PDF_PYTHON,'-c',code,str(path)],text=True)
    target.write_text(text);dump(sidecar,dict(source_sha256=source_hash,text_sha256=sha(target.read_bytes()),parser=PDF_TEXT_PARSER))
    return text

class BodyParser(HTMLParser):
    """Original visible body only: exclude scripts, styles and metadata."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_body=False; self.ignored=0; self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='body': self.in_body=True
        if tag in ('script','style'): self.ignored += 1
    def handle_endtag(self,tag):
        if tag=='body': self.in_body=False
        if tag in ('script','style'): self.ignored=max(0,self.ignored-1)
    def handle_data(self,data):
        if self.in_body and not self.ignored: self.parts.append(data)

def html_text(path):
    parser=BodyParser();parser.feed(path.read_text());text=' '.join(parser.parts)
    target=path.with_suffix('.txt');target.write_text(text)
    dump(path.with_suffix('.text.meta.json'),dict(source_sha256=sha(path.read_bytes()),text_sha256=sha(target.read_bytes()),parser='stdlib HTMLParser visible body, scripts/styles/head excluded'))
    return text

def fiscal_period_evidence(text, year, quarter):
    """Require company financial reporting context, never a bare date match.

    Consolidated income-statement columns use the quarter-ending calendar year;
    a Q1 FY label belongs to December of the preceding calendar year. Date
    strings in volume metrics, buybacks, call schedules or comparison prose do
    not establish the issuer reporting period.
    """
    py,pm=(year-1,12) if quarter==1 else (year,(quarter-1)*3)
    pd=calendar.monthrange(py,pm)[1]
    period='%04d-%02d-%02d'%(py,pm,pd)
    date=calendar.month_name[pm]+r'\s+'+str(pd)+r'\s*,?\s*'+str(py)+r'\b'
    # This must be an actual financial-results statement for the same FY/Q.
    narrative=re.search(r"financial results for (?:the )?(?:Company[’']s\s+)?fiscal\s+"+
        QWORDS[quarter]+r'\s+quarter\s+(?:and\s+full.year\s+)?'+str(year)+
        r'\s*,?\s+end(?:ed|ing)\s+'+date,text,re.I)
    if narrative:
        return dict(date=period,excerpt=narrative.group(0),
            basis='Explicit company financial-results statement for the same fiscal year and quarter, with actual period end')
    # Layout extraction retains header/column ordering even for older original
    # PDFs whose default extraction places the caption after the table rows.
    title=r'Consolidated Statements? of Operations\s*\(Unaudited\)'
    for heading in re.finditer(title,text,re.I):
        head=text[heading.start():heading.end()+500]
        revenues=re.search(r'(?:Operating Revenues|Net revenues?|Service revenues)',head,re.I)
        if revenues:head=head[:revenues.start()]
        month_day=calendar.month_name[pm]+r'\s+'+str(pd)+r'\s*,'
        column_years=rf'\b{py}\s*(?:\(\d+\)\s*)?{py-1}\b'
        if (re.search(r'Three Months Ended',head,re.I)
                and re.search(month_day,head,re.I)
                and re.search(column_years,head)):
            return dict(date=period,excerpt=head.strip(),
                basis='Original consolidated statement of operations: three-month period-end heading and current/prior-year columns')
    raise ValueError('Actual fiscal-period end absent from company financial-results statement or consolidated operations quarter header')

def parse_release(text, year, quarter):
    text=clean(text)
    identity=re.search(r'\bfiscal\s+'+QWORDS[quarter]+r'\s+quarter\s+(?:and\s+full.year\s+)?'+str(year)+r'\b',text,re.I)
    if not identity: raise ValueError('Fiscal year/quarter not confirmed by original body')
    if not re.search(r'\b(?:R\s*epor\s*ts|Posts)\s+(?:Strong\s+)?Fiscal\b', text, re.I):
        raise ValueError('Actual results release headline absent; schedules are not releases')
    dateline=re.search(r'\b(S\s*an Francisco|Foster City)\s*,?\s*(?:CA\s*,?\s*)?('+MONTHS+r')\s+(\d{1,2})\s*,\s*(\d{4})',text,re.I)
    date_format='%B %d %Y'
    if not dateline:
        dateline=re.search(r'\b(SAN FRANCISCO)\s*-+\s*\(BUSINESS WIRE\)\s*-+\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})\s*,\s*(\d{4})',text,re.I)
        date_format='%b %d %Y'
    if not dateline: raise ValueError('Original California publication dateline absent')
    release=dt.datetime.strptime(' '.join(dateline.groups()[1:]),date_format).date().isoformat()
    fiscal=fiscal_period_evidence(text,year,quarter)
    period=fiscal['date']
    if period >= release: raise ValueError('Fiscal end must precede release date')
    if not (0 < (dt.date.fromisoformat(release)-dt.date.fromisoformat(period)).days < 100): raise ValueError('Release period is not current quarter')
    return dict(release_date=release,identity_excerpt=identity.group(0),dateline_excerpt=dateline.group(0),fiscal_period_end_actual=period,period_excerpt=fiscal['excerpt'],period_basis=fiscal['basis'],location=re.sub(r'^S\s+an', 'San', dateline.group(1), flags=re.I))

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    sources=json.loads((HERE/'sources.json').read_text()); events=[]; pending=[]
    for s in sorted(sources,key=lambda x:(x['year'],x['quarter'])):
        year,quarter=s['year'],s['quarter']; label=f'{year}Q{quarter}'
        try:
            body,path,meta=fetch(s['url'],label,'.html' if s.get('format')=='html' else '.pdf')
            parsed=parse_release(html_text(path) if s.get('format')=='html' else pdf_text(path),year,quarter)
            if not '2011-01-01' <= parsed['release_date'] <= CUTOFF: continue
            proof=dict(evidence_path=rel(path),sha256=sha(body),source_url=s['url'])
            event=dict(ticker='V',release_date=parsed['release_date'],fiscal_period_end_actual=parsed['fiscal_period_end_actual'],fiscal_year=year,fiscal_quarter=quarter,
                publication_timestamp=None,precision='date_only',publication_timezone='America/Los_Angeles',
                publication_timezone_basis='Conservative issuer-local calendar-date policy: original release California dateline maps to America/Los_Angeles. No exact release instant or CMS timezone asserted.',
                publication_timezone_evidence=dict(proof,kind='issuer_local_calendar_date_policy',observed_dateline=parsed['dateline_excerpt'],geographic_mapping=parsed['location']+', CA -> America/Los_Angeles',cms_timezone_observed=False,exact_publication_time_inferred=False),
                first_public_time_verified=False,official_date_published_raw=parsed['dateline_excerpt'],source_url=s['url'],source_class='issuer_original_press_release',
                evidence_path=rel(path),evidence_sha256=sha(body),evidence_excerpt=parsed['identity_excerpt'],retrieved_at=meta['retrieved_at'],
                release_date_basis='Actual publication date in original release California dateline; conference-call and schedule times excluded.',
                release_date_evidence=[dict(proof,kind='original_release_dateline',excerpt=parsed['dateline_excerpt'])],
                fiscal_period_evidence=dict(proof,excerpt=parsed['period_excerpt'],basis=parsed['period_basis']),
                identity_evidence=dict(proof,excerpt=parsed['identity_excerpt']),
                timestamp_note='No CMS, PDF metadata, SEC acceptance, conference-call, or scheduled-release timestamp used.',
                historical_limitation='Current issuer archive, not immutable contemporaneously captured historical feed.',in_study_window=parsed['release_date']>='2016-01-01',additional_evidence=[])
            events.append(event);print(label,parsed['release_date'],'ready',flush=True)
        except Exception as error:
            pending.append(dict(label=label,year=year,quarter=quarter,source_url=s['url'],reason=str(error)));print(label,'pending',str(error),flush=True)
    events.sort(key=lambda e:e['release_date']); dump(HERE/'events.json',events)
    got={(x['fiscal_year'],x['fiscal_quarter']) for x in events}
    missing=[f'{y}Q{q}' for y in range(2011,2027) for q in range(1,5) if (y,q) not in got and (y<2026 or q<=3)]
    dump(HERE/'PENDING.json',dict(requested_release_window=['2011-01-01',CUTOFF],missing_quarters=missing,failures=pending,unresolved_precision='All dates remain date_only; exact first-publication timestamps unverified.',out_of_scope='FY2026 Q4 ends after cutoff.'))
    dump(HERE/'STATUS.json',dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),ticker='V',verified_events=len(events),expected_quarters=63,missing_quarters=missing,earliest_release=min((e['release_date'] for e in events),default=None),latest_release=max((e['release_date'] for e in events),default=None),requested_release_window=['2011-01-01',CUTOFF],all_date_only=True,unrelated_accounts_or_prices_modified=False))
if __name__=='__main__':main()
