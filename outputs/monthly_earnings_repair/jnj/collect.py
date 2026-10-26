"""JNJ official original evidence collector. No prices, vendor APIs or trading.

Direct hosts are stopped for the UTC day on the first 403/429. The public
search index is labeled as an index representation, never original HTML.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=HERE/'raw'
CDN='https://s203.q4cdn.com/636242992'
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
CUTOFF='2026-09-24'
WORDS={1:'first',2:'second',3:'third',4:'fourth'}
MONTHS={s.lower():i for i,s in enumerate(['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
MONTHS.update({s[:3]:i for s,i in list(MONTHS.items())})
DATE=r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)\s+(\d{1,2})\s*,?\s*(20\d{2})'
def sha(body): return hashlib.sha256(body).hexdigest()
def dump(path,data): path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rel(path): return str(path.relative_to(ROOT))
def clean(text): return re.sub(r'\s+',' ',text.replace('\u00a0',' ').replace('–','-').replace('—','-'))
def date_groups(match): return dt.date(int(match[3]),MONTHS[match[1].rstrip('.').lower()[:3]],int(match[2])).isoformat()
def fetch(url,stem,suffix='.pdf'):
 path=RAW/(stem+suffix);meta_path=RAW/(stem+'.meta.json');today=dt.datetime.now(dt.timezone.utc).date().isoformat()
 if path.exists() and meta_path.exists():
  meta=json.loads(meta_path.read_text());body=path.read_bytes()
  if meta['url']!=url or meta['sha256']!=sha(body): raise ValueError('cache source/checksum mismatch')
  return path,meta
 host=urlparse(url).hostname
 stops_path=HERE/'DIRECT_SOURCE_STOPS.json';stops=json.loads(stops_path.read_text()) if stops_path.exists() else []
 if host in ('www.investor.jnj.com','investor.jnj.com','www.sec.gov') or any(x['host']==host and x['utc_date']==today for x in stops):
  raise RuntimeError('direct source stopped today: '+host)
 failures_path=HERE/'DOWNLOAD_FAILURES.json';failures=json.loads(failures_path.read_text()) if failures_path.exists() else []
 if any(x['url']==url and x['utc_date']==today for x in failures): raise RuntimeError('cached failed URL: '+url)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public historical earnings research)'}),timeout=40) as r:
   body=r.read();meta=dict(url=url,final_url=r.geturl(),status=r.status,sha256=sha(body),retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),bytes=len(body),content_type=r.headers.get('Content-Type'),source_representation='original_http_bytes')
 except urllib.error.HTTPError as error:
  failure=dict(url=url,host=host,http_status=error.code,retry_after=error.headers.get('Retry-After'),utc_date=today,retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat())
  failures.append(failure);dump(failures_path,failures)
  if error.code in (403,429): stops.append(failure);dump(stops_path,stops)
  raise
 if suffix=='.pdf' and not body.startswith(b'%PDF'): raise ValueError('response is not PDF')
 path.write_bytes(body);dump(meta_path,meta);time.sleep(.15)
 return path,meta

def pdf_text(path):
 target=path.with_suffix('.txt');sidecar=path.with_suffix('.text.meta.json');checksum=sha(path.read_bytes())
 if target.exists() and sidecar.exists():
  info=json.loads(sidecar.read_text())
  if info['source_sha256']==checksum and info['text_sha256']==sha(target.read_bytes()): return target.read_text()
 code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
 text=subprocess.check_output([PDF_PYTHON,'-c',code,str(path)],text=True);target.write_text(text)
 dump(sidecar,dict(source_sha256=checksum,text_sha256=sha(target.read_bytes()),parser='bundled pypdf extraction'))
 return text

def identity(text,year,quarter):
 t=clean(re.sub(r'L\d+:\s*','',text))
 patterns=[r'Johnson\s*(?:&|and)\s*Johnson\s+Reports?\s+'+str(year)+r'\s+'+WORDS[quarter]+r'[- ]quarter',r'Johnson\s*(?:&|and)\s*Johnson\s+Reports?\s+Q'+str(quarter)+r'(?:\s*(?:and|&)\s*Full[- ]Year)?\s+'+str(year)]
 patterns.append(r'Johnson\s*&\s*Johnson\s*\(NYSE:\s*JNJ\)\s*today\s+announced\s+results\s+for\s+'+WORDS[quarter]+r'\s*-?\s*quarter(?:\s+and\s+full\s*year)?\s+'+str(year))
 for pat in patterns:
  m=re.search(pat,t,re.I)
  if m:return m.group(0)
 raise ValueError('release FY/Q headline identity absent')

def parse_release(text,year,quarter):
 t=clean(re.sub(r'L\d+:\s*','',text));ident=identity(t,year,quarter)
 locpat=r'NEW\s+BRUNSWICK,?\s*N\s*\.?\s*J\s*\.?'
 loc=re.search(locpat,t,re.I)
 if not loc:raise ValueError('issuer dateline absent')
 m=re.search(locpat+r'\s*,?\s*\(?(?:-+\s*\(BUSINESS WIRE\)\s*-+\s*)?'+DATE,t,re.I)
 if m:
  date=date_groups(m);excerpt=m.group(0);basis='Original release body New Brunswick dateline'
 else:
  # Q4-generated original NEWS RELEASE PDFs put an ISO date immediately after
  # their title. Do not pick the nearest date before the dateline: bullets can
  # announce unrelated future meetings (e.g. December 8 in the Q1 2026 release).
  m=re.match(r'NEWS RELEASE\s+(Johnson\s*&\s*Johnson\s+reports?.{0,180}?)\s+(20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d{2})\b',t,re.I)
  if not m:raise ValueError('original publication date absent; webpage/CMS heading alone not admitted')
  date=m.group(2);date=dt.datetime.strptime(date,'%m/%d/%Y').date().isoformat() if '/' in date else date;excerpt=m.group(0);basis='Original NEWS RELEASE PDF date directly following its title'
 if int(date[:4]) not in (year,year+1):raise ValueError('release date not compatible with FY')
 if not re.search(r'today\s+announced',t,re.I):raise ValueError('actual results announcement absent')
 return dict(release_date=date,dateline_excerpt=excerpt,identity_excerpt=ident,release_date_basis=basis)

def release_tasks():
 urls=set()
 for p in RAW.glob('*discovery*.json'):
  for u in re.findall(r'https://[^\s)]+',json.loads(p.read_text()).get('response','')):
   if ('investor-news/news-details/' in u or '/media-center/press-releases/' in u) and re.search('reports-',u,re.I):urls.add(u.split('?')[0])
 tasks=[]
 for year in range(2010,2027):
  for quarter in range(1,5):
   if year==2010 and quarter!=4 or year==2026 and quarter>2:continue
   have=False
   for p in RAW.glob('*.txt'):
    try:parse_release(p.read_text()[:20000],year,quarter);have=True;break
    except ValueError:pass
   if have:continue
   candidates=[]
   for u in urls:
    slug=u.lower().rsplit('/default.aspx',1)[0].rstrip('/').split('/')[-1]
    if re.search(r'reports-'+str(year)+'-'+WORDS[quarter]+'-quarter',slug) or re.search(r'reports-q'+str(quarter)+r'(?:-and-full-year)?-'+str(year),slug):candidates.append(u)
   candidates.sort(key=lambda u:('investor-news' not in u,u))
   tasks.append(dict(year=year,quarter=quarter,url=candidates[0] if candidates else None))
 return tasks

def proof(path,url,excerpt=None):
 d=dict(evidence_path=rel(path),sha256=sha(path.read_bytes()),source_url=url)
 if excerpt is not None:d['excerpt']=excerpt
 return d

def period_catalog():
 catalog={}
 for meta_path in sorted(RAW.glob('*.meta.json')):
  meta=json.loads(meta_path.read_text());url=meta.get('url','')
  p=meta_path.with_suffix('').with_suffix('.pdf')
  if not p.exists():continue
  text=clean(pdf_text(p))
  if not re.search(r'Johnson\s*&\s*Johnson',text,re.I):continue
  label=re.search(r'/doc_financials/(20\d{2})/q([1-4])/',url)
  if not label:continue
  year,quarter=map(int,label.groups())
  # An explicitly quarter-labeled financial reconciliation date in the original
  # earnings attachment is also direct actual-period evidence.
  supplement=re.search(WORDS[quarter]+r'\s+Quarter\s+'+DATE,text,re.I)
  if supplement and int(supplement[3])==year:
   catalog.setdefault(f'{year}Q{quarter}',dict(date=date_groups(supplement),**proof(p,url,supplement.group(0)),basis='Explicit quarter name and period date in original earnings financial reconciliation schedule'))
  # Only a real Form 10-Q/K can establish period metadata. Release boilerplate
  # also mentions prior 10-Ks; it is not a substitute for the original filing.
  if not re.search(r'UNITED\s+STATES\s*SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION',text[:5000],re.I) or not re.search(r'FORM\s+10[- ]?[QK]',text[:5000],re.I):continue
  def add(y,q,date,excerpt,basis):
   key=f'{y}Q{q}'
   d=dt.date.fromisoformat(date)
   if q<4 and (d.year!=y or not q*3-1<=d.month<=q*3+1):raise ValueError('inconsistent quarter end')
   if q==4 and not(d.year==y and d.month==12 or d.year==y+1 and d.month==1):raise ValueError('inconsistent FY end')
   item=dict(date=date,**proof(p,url,excerpt),basis=basis)
   if key in catalog and catalog[key]['date']!=date:raise ValueError('conflicting original fiscal ends: '+key)
   catalog.setdefault(key,item)
  cover=re.search(r'for\s+the\s+(?:quarterly period|fiscal year)\s+ended\s+'+DATE,text[:20000],re.I)
  if not cover:continue
  date=date_groups(cover)
  add(year,quarter,date,cover.group(0),'Explicit original Form 10-Q/10-K cover date; current issuer archive labels FY/Q, original body confirms fiscal period')
  if quarter<4:
   m=re.search(r'fiscal\s+'+WORDS[quarter]+r'\s+quarters\s+ended\s+('+DATE+r')\s+and\s+('+DATE+r')',text,re.I)
   if m:
    matches=list(re.finditer(DATE,m.group(0),re.I))
    if date_groups(matches[0])!=date:raise ValueError('cover/comparative current date conflict')
    add(year-1,quarter,date_groups(matches[1]),m.group(0),'Explicit same fiscal quarter comparative period in original filing; one prior fiscal year, not a calendar-month approximation')
   # This header in the quarterly filing identifies the immediately preceding
   # annual report. It is metadata only, never used as release availability.
   m=re.search(r'for\s+the\s+fiscal\s+year\s+ended\s+'+DATE,text,re.I)
   if m:add(year-1,4,date_groups(m),m.group(0),'Explicit previous fiscal-year end identified by the current quarterly filing')
  else:
   m=re.search(r'fiscal\s+years\s+ended\s+('+DATE+r')(?:,\s*|\s+and\s+)('+DATE+r')(?:\s+and\s+('+DATE+r'))?',text,re.I)
   if m:
    dates=list(re.finditer(DATE,m.group(0),re.I))
    if date_groups(dates[0])==date:
     for i,z in enumerate(dates):add(year-i,4,date_groups(z),m.group(0),'Explicit fiscal years ended comparative statement in the original annual report')
 return catalog

def build():
 periods=period_catalog();dump(HERE/'FISCAL_PERIODS.json',periods)
 candidates=[]
 for meta_path in sorted(RAW.glob('*.meta.json')):
  meta=json.loads(meta_path.read_text());p=meta_path.with_suffix('').with_suffix('.pdf')
  if not p.exists():continue
  candidates.append((p,meta['url'],pdf_text(p)[:25000],meta.get('retrieved_at'), 'original_pdf_bytes'))
 events=[];pending=[]
 for year in range(2010,2027):
  for quarter in range(1,5):
   if year==2010 and quarter!=4 or year==2026 and quarter>2:continue
   label=f'{year}Q{quarter}';selected=None;failures=[]
   source_list=list(candidates)
   indexed=RAW/f'indexed_release_{year}q{quarter}.json'
   if indexed.exists():
    d=json.loads(indexed.read_text());source_list.append((indexed,d['url'],d['response'],d['retrieved_at'],'public_search_index_excerpt_not_original_html'))
   for path,url,text,retrieved,representation in source_list:
    try:parsed=parse_release(text,year,quarter)
    except ValueError as e:
     if path==indexed:failures.append(str(e))
     continue
    selected=(path,url,retrieved,representation,parsed);break
   missing=[]
   if not selected:missing.append('original release publication evidence not yet verified'+(': '+', '.join(failures) if failures else ''))
   if label not in periods:missing.append('actual fiscal-period end not yet verified from issuer filing')
   if missing:pending.append(dict(ticker='JNJ',fiscal_year=year,fiscal_quarter=quarter,status='pending_review',reasons=missing));continue
   path,url,retrieved,representation,parsed=selected;period=periods[label]
   if not '2011-01-01'<=parsed['release_date']<=CUTOFF:continue
   if not period['date']<parsed['release_date']:raise ValueError('release before fiscal period end: '+label)
   p=proof(path,url,parsed['dateline_excerpt'])
   row=dict(ticker='JNJ',release_date=parsed['release_date'],publication_timestamp=None,precision='date_only',
    publication_timezone='America/New_York',publication_timezone_basis='Conservative issuer-local calendar-date mapping: original New Brunswick, New Jersey dateline -> America/New_York. No exact release time or CMS timezone inferred.',
    publication_timezone_evidence=dict(kind='issuer_local_calendar_date_policy',observed_dateline='New Brunswick, N.J.',geographic_mapping='New Brunswick, NJ -> America/New_York',evidence_path=rel(path),sha256=sha(path.read_bytes()),cms_timezone_observed=False,exact_publication_time_inferred=False),
    first_public_time_verified=False,fiscal_year=year,fiscal_quarter=quarter,fiscal_period_end_actual=period['date'],source_url=url,
    source_class='issuer_original_press_release' if representation=='original_pdf_bytes' else 'issuer_original_release_public_index_excerpt',
    source_representation=representation,evidence_path=rel(path),evidence_sha256=sha(path.read_bytes()),retrieved_at=retrieved,
    raw_original_bytes_sha256=sha(path.read_bytes()) if representation=='original_pdf_bytes' else None,
    raw_original_bytes_available=representation=='original_pdf_bytes',evidence_excerpt=parsed['identity_excerpt'],
    release_date_basis=parsed['release_date_basis'],release_date_evidence=[dict(p,kind='issuer_release_body_dateline')],
    fiscal_period_evidence=period,additional_evidence=[dict(evidence_path=period['evidence_path'],sha256=period['sha256'],role='actual_fiscal_period_end_in_official_filing')],
    timestamp_note='No CMS migration date, PDF creation/modification time, preannouncement schedule, conference-call time or filing acceptance date is treated as actual publication time.',
    historical_limitation='Current official archive or public index representation; not a contemporaneous immutable historical feed. Public-index evidence is not a download of complete original HTML.',in_study_window=parsed['release_date']>='2016-01-01')
   events.append(row)
 events.sort(key=lambda x:x['release_date']);dump(HERE/'events.json',events);dump(HERE/'PENDING.json',pending)
 dump(HERE/'STATUS.json',dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),ticker='JNJ',requested_release_window=['2011-01-01',CUTOFF],verified_events=len(events),missing_quarters=[f"{x['fiscal_year']}Q{x['fiscal_quarter']}" for x in pending],pending_count=len(pending),earliest_release=events[0]['release_date'] if events else None,latest_release=events[-1]['release_date'] if events else None,original_pdf_events=sum(x['raw_original_bytes_available'] for x in events),public_index_events=sum(not x['raw_original_bytes_available'] for x in events),all_events_date_only=True,approximate_times_inferred=False,source='Issuer original PDFs and transparently labeled public-index original-release excerpts; fiscal ends from issuer-hosted filings',unrelated_accounts_or_prices_modified=False))
 print(json.dumps(json.loads((HERE/'STATUS.json').read_text()),indent=2))
if __name__=='__main__':build()
