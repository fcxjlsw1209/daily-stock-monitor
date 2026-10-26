"""ExxonMobil candidate-history supplement; original release dates, never filing clocks."""
from source_io import *
import subprocess,sys
sys.path.insert(0,str(HERE.parent.parent))
from event_schema import normalize_event
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE=MONTH+r'\s+\d{1,2},?\s+20\d{2}'
QMAP=dict(first=1,second=2,third=3,fourth=4)
def clean(t):return re.sub(r'\s+',' ',t).replace('\xa0',' ').replace('—','-').replace('–','-')
def iso(s):return dt.datetime.strptime(s.replace(',','').title(),'%B %d %Y').date().isoformat()
def extract(p):
 raw=p.read_bytes()
 if raw.startswith(b'%PDF'):
  dst=p.with_suffix('.txt');mp=p.with_suffix('.text.meta.json')
  if dst.exists() and mp.exists():
   m=json.load(open(mp))
   if sha(raw)==m['source_sha256'] and sha(dst.read_bytes())==m['text_sha256']:return dst.read_text()
  code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sys.argv[1]).pages))'
  t=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True,stderr=subprocess.DEVNULL);dst.write_text(t);dump(mp,dict(source_sha256=sha(raw),text_sha256=sha(dst.read_bytes()),parser='bundled pypdf layout'));return t
 s=BeautifulSoup(raw,'html.parser');a=s.select_one('article.full-news-article') or s
 for n in a(['script','style']):n.decompose()
 return a.get_text(' ',strip=True)

def parse(raw):
 t=clean(raw)
 if not re.search(r'Exxon\s*Mobil',t,re.I):raise ValueError('Issuer identity absent')
 dm=re.search(r'FOR IMMEDIATE RELEASE\s*(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY)?\s*,?\s*('+DATE+r')',t,re.I)
 if dm:
  date_basis='Dated FOR IMMEDIATE RELEASE header in issuer-hosted original earnings exhibit; filing/acceptance dates ignored'
  body=t[dm.start():]
 else:
  dm=re.search(r'\bReleased\s+('+DATE+r')',t,re.I)
  if not dm:raise ValueError('Explicit issuer release-publication notice absent; CMS metadata not substituted')
  date_basis='Explicit Released notice on complete issuer earnings original; date-only, ignoring datetime/time-zone clock attributes and modified metadata'
  body=t
 release=iso(dm.group(1))
 if not '2010-10-01'<=release<='2016-03-01':raise ValueError('Outside assigned early-candidate supplement window')
 head=body[:2600]
 if re.search(r'earnings conference call|to release|will release',head[:500],re.I):raise ValueError('Conference or future notice is not actual financial results')
 qm=re.search(r'ExxonMobil Earns.{0,70}?in (20\d{2}).{0,80}?Fourth Quarter',head,re.I)
 if qm:fy=int(qm.group(1));q=4;identity=qm.group()
 else:
  qm=re.search(r'(First|Second|Third|Fourth)\s+Quarter\s+(?:of\s+)?(20\d{2})',head,re.I)
  if not qm:raise ValueError('Explicit current fiscal identity absent')
  q=QMAP[qm.group(1).lower()];fy=int(qm.group(2));identity=qm.group()
 # Attachment I is the actual current-quarter financial statement. It explicitly
 # labels current and comparative years and the date of shares/equity/capital.
 am=re.search(r'Attachment(?:\s+I)?\s+(Exxon\s+Mobil.*?)Attachment\s+II\b',body,re.I)
 if not am:raise ValueError('Original Attachment I financial statement absent')
 table=am.group(1)
 if not re.search(r'(?:'+list(QMAP)[q-1]+r'\s+Quarter\s+'+str(fy)+r'|'+str(fy)+r'\s+'+str(fy-1)+r')',table,re.I):raise ValueError('Current fiscal year column absent from period table')
 ends=[]
 for pm in re.finditer(r'(?:At|as of)\s+('+MONTH+r')\s+(\d{1,2})\b(?:\s*,\s*(20\d{2}))?',table,re.I):
  month=dt.datetime.strptime(pm.group(1).title(),'%B').month;day=int(pm.group(2));year=int(pm.group(3)) if pm.group(3) else fy
  if year!=fy:continue
  pd=dt.date(year,month,day);rd=dt.date.fromisoformat(release)
  if 0<(rd-pd).days<100:ends.append((pd.isoformat(),pm.group()))
 if len({v[0] for v in ends})!=1:raise ValueError('Observed current period date missing or conflicting; calendar quarter not invented')
 end,period_label=ends[0]
 if (int(end[5:7])-1)//3+1!=q:raise ValueError('FY/Q disagrees with explicit period')
 return dict(release_date=release,fiscal_year=fy,fiscal_quarter=q,fiscal_period_end_actual=end,release_excerpt=dm.group(),release_basis=date_basis,identity_excerpt=identity,period_excerpt=table,period_label=period_label)

def event(p,m,r):
 proof=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=m['url'])
 e=dict(ticker='XOM',release_date=r['release_date'],publication_timestamp=None,precision='date_only',publication_timezone='America/Chicago',publication_timezone_basis='Conservative issuer-local Irving, Texas publication date; no exact clock inferred',first_public_time_verified=False,fiscal_year=r['fiscal_year'],fiscal_quarter=r['fiscal_quarter'],fiscal_period_end_actual=r['fiscal_period_end_actual'],source_url=m['url'],source_class='issuer_hosted_original_earnings_release_exhibit_full_bytes' if '/sec-filings/' in m['url'] else 'issuer_original_earnings_release_html_full_bytes',evidence_path=rel(p),evidence_sha256=proof['sha256'],source_retrieved_at=m['retrieved_at'],release_date_basis=r['release_basis'],release_date_evidence=[dict(proof,excerpt=r['release_excerpt'])],fiscal_identity_evidence=dict(proof,excerpt=r['identity_excerpt']),fiscal_period_evidence=dict(proof,excerpt=r['period_excerpt'],basis='Original Attachment I current-year financial statement explicitly reports '+r['period_label']+'; fiscal year taken from its labeled current-year column, not a vendor/calendar guess'),historical_limitation='Current issuer archival original; historical first-publication instant not verified',event_scope='regular quarterly results; historical issuer Estimated wording denotes its standard full earnings announcement, not an outlook-only notice',in_study_window=r['release_date']>='2016-01-01')
 if r['release_date']<'2011-01-01':e['training_predecessor_only']=True
 return e

def build():
 by={};pending=[]
 for mp in sorted((HERE/'XOM/raw').glob('*.meta.json')):
  if '.text.' in mp.name:continue
  m=json.load(open(mp));p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200:continue
  try:r=parse(extract(p));e=event(p,m,r)
  except ValueError as ex:pending.append(dict(source_url=m['url'],evidence_path=rel(p),reason=str(ex)));continue
  key=e['fiscal_year'],e['fiscal_quarter']
  if key in by:
   old=by[key];assert all(e[k]==old[k] for k in ['release_date','fiscal_period_end_actual']),(e,old)
   if '/sec-filings/' in m['url']:by[key]=e
  else:by[key]=e
 oldpath=HERE.parent.parent/'integrated_batches/health_energy_foreign_866/events.json';old=[e for e in json.load(open(oldpath)) if e['ticker']=='XOM'];oldkeys={(e['fiscal_year'],e['fiscal_quarter']) for e in old};es=[e for k,e in sorted(by.items()) if k not in oldkeys]
 dump(HERE/'events.json',es);dump(HERE/'PENDING.json',dict(unadmitted=pending,baseline=rel(oldpath),baseline_sha256=sha(oldpath.read_bytes()),baseline_event_count=len(old),source_stops_path=rel(HERE/'SOURCE_STOPS.json')))
 print('XOM',len(es),'new; pending',len(pending));print([(e['fiscal_year'],e['fiscal_quarter'],e['release_date'],e['fiscal_period_end_actual']) for e in es]);print(pending)
if __name__=='__main__':build()
