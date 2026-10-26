"""Offline admission: original dateline + observed financial period. Unknowns remain pending."""
from common import *
import subprocess,calendar
PDF_PYTHON='/Users/shiweili/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
MONTH=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)'
DATE=MONTH+r'\.?\s+\d{1,2}\s*,?\s*\d{4}'
ORD=r'(first|second|third|fourth)'
QMAP=dict(first=1,second=2,third=3,fourth=4)
def clean(t):
 t=re.sub(r'\s+',' ',re.sub(r'L\d+(?:@P[\d-]+)?:\s*',' ',t)).replace('\xa0',' ').replace('–','-').replace('—','-').replace('‑','-').replace('‐','-')
 t=re.sub(r'(?<=\w)\s*-\s*(?=\w)','-',t)
 return re.sub(r'\b(20\d)\s+(\d)\b',r'\1\2',t)
def iso(t):
 parts=t.replace(',',' ').replace('.','').split();parts[0]=parts[0][:3]
 return dt.datetime.strptime(' '.join(parts),'%b %d %Y').date().isoformat()
def extract(p):
 raw=p.read_bytes()
 if raw.startswith(b'%PDF'):
  dest=p.with_suffix('.txt');mp=p.with_suffix('.text.meta.json')
  if dest.exists() and mp.exists():
   m=json.loads(mp.read_text())
   if m['source_sha256']==sha(raw) and m['text_sha256']==sha(dest.read_bytes()):return dest.read_text()
  code='import sys;from pypdf import PdfReader;print("\\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sys.argv[1]).pages))'
  t=subprocess.check_output([PDF_PYTHON,'-c',code,str(p)],text=True,stderr=subprocess.DEVNULL)
  dest.write_text(t);dump(mp,dict(source_sha256=sha(raw),text_sha256=sha(dest.read_bytes()),parser='bundled pypdf layout'))
  return t
 s=BeautifulSoup(raw,'html.parser')
 for a in s(['script','style']):a.decompose()
 x=s.select_one('.main-content') or s.select_one('.wd_news_body') or s.select_one('main') or s
 return x.get_text(' ',strip=True)
def parse(k,rawtext):
 t=clean(rawtext)
 if re.search(r'\bPreliminary\b',t[:1800],re.I) or re.search(r'\bAMD Updates? [^.]{0,150}(?:Outlook|Expected)',t[:500],re.I):raise ValueError('Preliminary/outlook update retained separately; not regular quarterly results')
 if not re.search(r'\b(?:Reports?|Announces?|Reported|Announced|Releases|Achieves|Delivers|Posts|Maintains|Sets|Outpaces|Caps|Raises|Celebrates|Overtakes)\b',t[:2500],re.I):raise ValueError('Actual results headline absent')
 if re.search(r'\b(?:to Report|will release|Announces? (?:the )?Date|financial conference call)\b',t[:300],re.I):raise ValueError('Scheduled announcement is not actual results')
 ident={'INTC':r'Intel','AMD':r'AMD|Advanced Micro Devices','KLAC':r'KLA','LRCX':r'Lam Research','AVGO':r'Broadcom|Avago','ADBE':r'Adobe','MU':r'Micron','PDD':r'PDD|Pinduoduo','PLTR':r'Palantir','AMAT':r'Applied Materials','CSCO':r'Cisco','TMUS':r'T-Mobile'}
 if not re.search(ident.get(k,re.escape(CONFIG[k][0])),t[:20000],re.I):raise ValueError('Original issuer identity absent')
 # The location + date must occur in the release body, never CMS or filing metadata.
 city=r'(SANTA CLARA|SUNNYVALE|MILPITAS|FREMONT|PALO ALTO|SAN JOSE|SAN FRANCISCO|NEW YORK|DUBLIN|BOISE|DENVER|MIAMI|SINGAPORE|AUSTIN|BELLEVUE|PEORIA|DEERFIELD|IRVING|SHANGHAI|DUBLIN|CAMBRIDGE|SEOUL|ARMONK|SAN DIEGO)'
 dm=re.search(r'\b'+city+r'[A-Za-z .;,&()-]{0,70}('+DATE+r')',t,re.I)
 if dm:release=iso(dm.group(2));de=dm.group(0)
 else:
  dm=re.search(r'\b(FREMONT|PALO ALTO)[^/]{0,50}\b(\d{2}/\d{2}/\d{2})\b',t,re.I)
  if dm:release=dt.datetime.strptime(dm.group(2),'%m/%d/%y').date().isoformat();de=dm.group(0)
  else:raise ValueError('Original release dateline date absent')
 if not ('2010-10-01' if k in ['HPQ','CSCO'] else '2011-01-01')<=release<=CUTOFF:raise ValueError('Publication date outside requested window')
 # Match original FY/Q wording near the actual release headline/intro.
 q=fy=None;qe=None
 pats=[ORD+r'[- ]quarter(?:\s+and\s+(?:full[- ]year|fiscal[- ]year|full fiscal(?: year)?))?\s+(?:of\s+)?(?:the\s+)?(?:(?:fiscal|financial)\s+)?(?:year\s+)?(?:(?:ended|ending)\s+)?(?:FY\s*)?(20\d{2})',r'(?:fiscal\s+(?:year\s+)?|FY\s*)(20\d{2})\s+'+ORD+r'[- ]quarter',r'\bQ([1-4])\s*(?:FY\s*)?(20\d{2})',ORD+r'[- ]quarter\s+FY\s*(\d{2})\b',r'\bQ([1-4])\s*FY\s*(\d{2})\b']
 matches=[(m.start(),i,m) for i,pat in enumerate(pats) for m in re.finditer(pat,t[:4500],re.I)]
 for _,i,m in sorted(matches,key=lambda v:v[0]):
  if m:
   if i==0:q=QMAP[m.group(1).lower()];fy=int(m.group(2))
   elif i==1:fy=int(m.group(1));q=QMAP[m.group(2).lower()]
   elif i==2:q=int(m.group(1));fy=int(m.group(2))
   elif i==3:q=QMAP[m.group(1).lower()];fy=2000+int(m.group(2))
   else:q=int(m.group(1));fy=2000+int(m.group(2))
   qe=m.group(0);break
 if k=='CSCO':
  cm=re.search(r'GAAP Results Q([1-4])\s*(?:FY\s*)?(20\d{2})',t,re.I)
  if cm:q=int(cm.group(1));fy=int(cm.group(2));qe=cm.group(0)
 # Exact current financial period from narrative or a consolidated statement header.
 periods=[]
 for m in re.finditer(r'(?:quarter(?:\s+and\s+(?:fiscal\s+)?year)?\s+(?:(?:of\s+)?(?:fiscal\s+)?(?:year\s+)?20\d{2}\s*,?\s*)?(?:which\s+)?|three\s+months\s+(?:and\s+year\s+)?)(?:end(?:ed|ing)\s*(?:on\s*)?)('+DATE+r')',t,re.I):
  v=iso(m.group(1))
  if 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(v)).days<100:periods.append((v,m.group(0)))
 # Body intro may say "FY2026 ended ...".
 for m in re.finditer(r'quarter.{0,90}?\bend(?:ed|ing)\s+(?:on\s+)?('+DATE+r')',t[:5000],re.I):
  v=iso(m.group(1))
  if 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(v)).days<100:periods.append((v,m.group(0)))
 if not periods:
  for m in re.finditer(r'(?:Consolidated (?:Condensed )?Statements? of (?:Income|Operations|Earnings)|Condensed Consolidated Statements? of (?:Income|Operations|Earnings))',t,re.I):
   head=t[m.start():m.end()+650]
   if not re.search(r'(?:Three Months|Quarter|Quarters) Ended',head,re.I):continue
   dates=list(re.finditer(DATE,head,re.I))
   for d in dates[:4]:
    v=iso(d.group())
    if 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(v)).days<100:periods.append((v,head[:max(300,d.end())]));break
 if not periods:
  from special_collect import period_header
  try:
   v,ev=period_header(t)
   if 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(v)).days<100:periods.append((v,ev))
  except ValueError:pass
 if not periods:raise ValueError('Actual current fiscal period absent from results narrative/statement header')
 periods.sort(reverse=True);period,pe=periods[0];pd=dt.date.fromisoformat(period)
 if k=='TMUS' and fy is None:
  if pd.month in [3,6,9,12] and pd.day==calendar.monthrange(pd.year,pd.month)[1]:
   fy=pd.year;q=pd.month//3;qe='Calendar fiscal-year quarter mapping of original financial statement quarter end '+period
 if k=='LRCX':
  # Lam quarter labels in release are calendar-end months; June-ending FY confirmed in annual archive.
  nearest=min([(abs((pd-dt.date(y,m,calendar.monthrange(y,m)[1])).days),y,m) for y in [pd.year-1,pd.year,pd.year+1] for m in [3,6,9,12]])
  _,cy,cm=nearest;fy=cy+(cm>6);q={9:1,12:2,3:3,6:4}[cm];qe='June-ending fiscal-year mapping of original quarter end '+period
 if fy is None:raise ValueError('Fiscal year/quarter identity absent')
 # Quarter order and actual fiscal end must agree, permitting 52/53-week boundary shifts.
 if k in ['INTC','AMD','CAT','IBM']:
  target_year=fy+(q==4 and pd.month==1)
  month={1:3,2:6,3:9,4:12}[q]
  ref=dt.date(fy,month,calendar.monthrange(fy,month)[1])
  if abs((pd-ref).days)>10:raise ValueError('Fiscal label disagrees with actual end')
 if k in ['KLAC','LRCX']:
  year=fy-(q in [1,2]);month={1:9,2:12,3:3,4:6}[q];ref=dt.date(year,month,calendar.monthrange(year,month)[1])
  if abs((pd-ref).days)>10:raise ValueError('June FY label disagrees with actual end')
 if k in ['CSCO','CRM','ACN','MU','AMAT','AVGO','ADBE','ARM','PDD','TMUS','QCOM','HPQ']:
  end_month={'CSCO':7,'CRM':1,'ACN':8,'MU':8,'AMAT':10,'AVGO':10,'ADBE':11,'ARM':3,'PDD':12,'TMUS':12,'QCOM':9,'HPQ':10}[k]
  month=(end_month-3*(4-q)-1)%12+1
  year=fy-(month>end_month)
  ref=dt.date(year,month,calendar.monthrange(year,month)[1])
  if abs((pd-ref).days)>10:raise ValueError('Issuer fiscal-year/quarter label disagrees with actual observed period end')
 return dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=de,period_excerpt=pe,identity_excerpt=qe)
def event(k,p,m,r,source_class='issuer_original_release_full_bytes'):
 proof=dict(evidence_path=rel(p),sha256=sha(p.read_bytes()),source_url=m['url'])
 return dict(ticker=k,release_date=r['release_date'],publication_timestamp=None,precision='date_only',publication_timezone=CONFIG[k][2],publication_timezone_basis='Conservative issuer-local calendar-date policy based on original release dateline; no exact clock, CMS or SEC acceptance time inferred.',first_public_time_verified=False,fiscal_period_end_actual=r['fiscal_period_end_actual'],fiscal_year=r['fiscal_year'],fiscal_quarter=r['fiscal_quarter'],source_url=m['url'],source_class=source_class,evidence_path=rel(p),evidence_sha256=proof['sha256'],source_retrieved_at=m['retrieved_at'],release_date_basis='Original actual-results publication dateline',release_date_evidence=[dict(proof,excerpt=r['dateline_excerpt'])],fiscal_period_evidence=dict(proof,excerpt=r['period_excerpt']),fiscal_identity_evidence=dict(proof,excerpt=r['identity_excerpt']),historical_limitation='Current issuer archive retrieval; exact historical first-publication instant unverified',in_study_window=r['release_date']>='2016-01-01')
def build(tickers=None):
 all_events=[]
 for k in tickers or ['INTC','AMD','KLAC','LRCX','MU']:
  events=[];pending=[]
  for mp in sorted((HERE/k/'raw').glob('*.meta.json')):
   if mp.name.endswith('.text.meta.json'):continue
   m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'));u=m['url']
   if m['status']!=200 or not any(x in u for x in ['press-releases/detail/','image/','newsroom.lamresearch.com/','/earnings_release/','news-releases/news-release-details/','cloudfront.net/CIK-0001321655/','newsroom.cisco.com/','newsroom.accenture.com/content/','newsroom.accenture.com/subjects/','/news/press-releases/','t-mobile.com/news/','/sec-filings/content/','/all-sec-filings/content/']):continue
   if urllib.parse.urlsplit(u).path.rstrip('/')=='/press-releases' or m.get('content_type','').startswith('image/'):
    pending.append(dict(source_url=u,evidence_path=rel(p),reason='Discovery archive or image, not standalone textual release'));continue
   assert sha(p.read_bytes())==m['sha256']
   try:r=parse(k,extract(p));events.append(event(k,p,m,r))
   except ValueError as e:pending.append(dict(source_url=u,evidence_path=rel(p),reason=str(e)))
  unique={}
  for e in events:
   key=(e['fiscal_year'],e['fiscal_quarter'])
   if key in unique:
    prev=unique[key]
    if prev['fiscal_period_end_actual']!=e['fiscal_period_end_actual'] or prev['release_date']!=e['release_date']:raise ValueError('Conflicting original releases '+str(key))
    pending.append(dict(source_url=e['source_url'],reason='Duplicate original source for same event; retained raw bytes'));continue
   unique[key]=e
  events=sorted(unique.values(),key=lambda x:x['release_date']);dump(HERE/k/'events.json',events);dump(HERE/k/'PENDING.json',dict(unadmitted=pending));print(k,len(events),'pending',len(pending));all_events+=events
 return all_events
if __name__=='__main__':build()
