"""Offline, evidence-first parsing of issuer earnings originals."""
import collect as c
import calendar
import re
from collections import Counter
MONTHS='January February March April May June July August September October November December'.split()
MP={n[:3].lower():i+1 for i,n in enumerate(MONTHS)}
MN=r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep(?:t)?\.?|Oct\.?|Nov\.?|Dec\.?)'
MDY=MN+r'\s+\d{1,2}\s*,?\s*20\d{2}'
DMY=r'\d{1,2}\s+'+MN+r'\s*,?\s*20\d{2}'
DP=r'(?:'+MDY+'|'+DMY+')'
WORDS={1:'first',2:'second',3:'third',4:'fourth'}
def date(s):
 m=re.search(r'('+MN+r')\s+(\d{1,2})\s*,?\s*(20\d{2})',s,re.I)
 if m:month,day,year=MP[m[1][:3].lower()],int(m[2]),int(m[3])
 else:
  m=re.search(r'(\d{1,2})\s+('+MN+r')\s*,?\s*(20\d{2})',s,re.I)
  if not m:raise ValueError('unsupported full date: '+s)
  day,month,year=int(m[1]),MP[m[2][:3].lower()],int(m[3])
 return c.dt.date(year,month,day).isoformat()
def proof(p,url,excerpt,basis):return dict(evidence_path=c.rel(p),sha256=c.sha(p.read_bytes()),source_url=url,excerpt=excerpt,basis=basis)
def text(p,t):
 if p.suffix=='.json' and c.load(p,{}).get('indexed_source_text'):
  tx=c.load(p,{})['indexed_source_text'];tx=re.sub(r'L\d+(?:@P\d+(?:-\d+)?)?:\s*',' ',tx)
  return c.clean(tx.replace('\x02','-'))
 if t=='BABA' and not p.read_bytes().startswith(b'%PDF'):
  d=c.load(p,{}).get('content',{});body=d.get('documentContentEnUS','')
  if not body:body=d.get('documentContent',{});body=body.get('en_US','') if isinstance(body,dict) else body
  return c.clean(c.BeautifulSoup(body,'html.parser').get_text(' ',strip=True))
 tx=c.clean(c.text_of(p))
 if t=='TM' and p.with_suffix('.visual_review.json').exists():
  review=c.load(p.with_suffix('.visual_review.json'),{})
  if review['source_sha256']!=c.sha(p.read_bytes()):raise ValueError('visual review source changed')
  return review['transcribed_header']
 if t=='TM' and p.with_suffix('.alternate.txt').exists():
  # Independent extraction of the same PDF's first three pages preserves the
  # header reading order; append the original full extraction for statements.
  tx=c.clean(p.with_suffix('.alternate.txt').read_text())+' '+tx
 if t=='XOM' and not p.read_bytes().startswith(b'%PDF'):
  s=c.BeautifulSoup(p.read_bytes(),'html.parser');article=s.select_one('article.full-news-article')
  if article:tx=c.clean(article.get_text(' ',strip=True))
 if t=='NVS' and not p.read_bytes().startswith(b'%PDF'):
  article=c.BeautifulSoup(p.read_bytes(),'html.parser').find('article')
  if article:tx=c.clean(article.get_text(' ',strip=True))
 # Repair only observed PDF letter-spacing in month tokens, not values.
 for month in MONTHS:
  tx=re.sub(r'\b'+r'\s*'.join(month)+r'\b',month,tx,flags=re.I)
 for city in ('MINNETONKA','NEW YORK','KENILWORTH','HONG KONG'):
  tx=re.sub(r'\b'+r'\s*'.join(city.replace(' ',''))+r'\b',city,tx,flags=re.I)
 return tx
def identity(t,tx,task):
 head=tx[:6000];label=task.get('label','')
 if t in ('BA','NVS','MRK','SHEL'):
  for qq,word in WORDS.items():
   m=re.search(r'\b'+word+r'[- ]+Quarter\s+(20\d{2})|\bQ'+str(qq)+r'\s*(20\d{2})',label,re.I)
   if m:
    yy=int(m[1] or m[2])
    if str(yy) in head and re.search(word+r'[- ]+quarter|\bQ'+str(qq),head,re.I):return yy,qq,label+'; original body corroborates year and quarter'
 if t=='ABBV':
  m=re.search(r'AbbVie Reports Full[- ]Year\s+(20\d{2})\s+and Fourth[- ]Quarter',head[:1500],re.I)
  if m:return int(m[1]),4,m[0]
 if t=='GE':
  m=re.search(r'GE\s*([1-4])Q\s*(20\d{2})\s*EARNIN\s*GS',head,re.I)
  if m:return int(m[2]),int(m[1]),m[0]
 if t=='XOM':
  m=re.search(r'ExxonMobil Earns.{0,60}?in (20\d{2}).{0,80}?Fourth Quarter',head,re.I)
  if m:return int(m[1]),4,m[0]
 if t=='TM':
  m=re.search(r'FY\s*(20\d{2})\s*(First|Second|Third|Semi-Annual)?',head,re.I)
  if not m:raise ValueError('Toyota fiscal year absent')
  return int(m[1]),{None:4,'first':1,'second':2,'third':3,'semi-annual':2}[m[2].lower() if m[2] else None],m[0]
 if t=='NVO':
  full=re.search(r'Financial statement for\s+(20\d{2})',head[:1000],re.I)
  if full:return int(full[1]),4,full[0]
  m=re.search(r'Financial report for the period\s+'+DP+r'\s+to\s+('+DP+')',head,re.I)
  if not m:raise ValueError('Novo report period absent')
  end=date(m[1]);return int(end[:4]),(int(end[5:7])+2)//3,m[0]
 if t=='BABA':
  m=re.search(r'(?:quarter|three months|six months|period).{0,25}?ended\s+('+DP+')',head,re.I)
  if not m:raise ValueError('Alibaba reporting period absent')
  end=date(m[1]);y=int(end[:4]);mo=int(end[5:7]);return y+(mo>3),((mo-4)%12)//3+1,m[0]
 if t=='CHL':
  if not re.search(r'CHINA MOBILE(?:\s+LIMITED)?',head,re.I):raise ValueError('original document belongs to a different issuer, despite archive link')
  m=re.search(r'KEY PERFORMANCE INDICATORS FOR THE FIRST (THREE QUARTERS|QUARTER) OF\s+(20\d{2})',head,re.I)
  if m:return int(m[2]),3 if m[1].upper()=='THREE QUARTERS' else 1,m[0]
  m=re.search(r'(20\d{2})\s+(?:ANNOUNCEMENT OF\s+)?(Annual|Interim)\s+Results',head,re.I)
  if m:return int(m[1]),4 if m[2].lower()=='annual' else 2,m[0]
  m=re.search(r'(?:three|nine) months ended\s+('+DP+')',tx,re.I)
  if m:
   d=date(m[1]);return int(d[:4]),int(d[5:7])//3,m[0]
 if t=='ASML':
  m=re.search(r'has published its (20\d{2}) fourth[- ]quarter|has published its (20\d{2}) full[- ]year',head,re.I)
  if m:return int(m[1] or m[2]),4,m[0]
  m=re.search(r'has published its\s+(20\d{2})\s+(first|second|third|fourth)[-\s]+quarter',head,re.I)
  if m:return int(m[1]),next(q for q,w in WORDS.items() if w==m[2].lower()),m[0]
 if t=='AZN':
  m=re.search(r'(FIRST|SECOND|THIRD|FOURTH)[- ]QUARTER(?: AND (?:FULL[- ]YEAR|HALF[- ]YEAR|NINE[- ]MONTHS))? (?:RESULTS|INTERIM MANAGEMENT STATEMENT) (20\d{2})',head,re.I)
  if m:return int(m[2]),next(q for q,w in WORDS.items() if w==m[1].lower()),m[0]
 if t=='ORCL':
  for qq,word in WORDS.items():
   m=re.search(r'(?:'+word+r'\s+Quarter\s+(?:and\s+Fiscal\s+Full\s+Year\s+)?Fiscal\s+(?:Year\s+)?(20\d{2})|fiscal\s+(20\d{2})\s+Q'+str(qq)+r'\b)',head,re.I)
   if m:return int(m[1] or m[2]),qq,m[0]
 matches=[]
 for q,word in WORDS.items():
  pats=[r'\b'+word+r'[-\s]+quarter(?:\s+and\s+(?:full[-\s]*year|half[-\s]*year|year[-\s]*end))?\s+(?:of\s+)?(20\d{2})',r'\bQ'+str(q)+r'\s*[/& -]*(?:and\s*)?(?:(?:Full[- ]Year|Half[- ]Year|HY|FY)\s*)?(20\d{2})',r'\b(20\d{2})\s+'+word+r'[-\s]+quarter']
  for pat in pats:
   m=re.search(pat,head,re.I)
   if m:matches.append((m.start(),int(m[1]),q,m[0]))
 if t=='UNH':
  m=re.search(r'UnitedHealth Group\s+Reports\s+(20\d{2})\s+Results',head,re.I)
  if m:matches.append((m.start(),int(m[1]),4,m[0]))
 if matches:
  _,y,q,ident=min(matches);return y,q,ident
 # These original labels are corroborated by the same FY in the body. Release
 # calendars and URL years are not substituted for a document identity.
 for q,word in WORDS.items():
  m=re.search(r'\b'+word+r'[-\s]+quarter\s+(20\d{2})|\bQ'+str(q)+r'\s*(20\d{2})',label,re.I)
  if m:
   y=int(m[1] or m[2])
   if str(y) in head and re.search(word+r'[-\s]+quarter|\bQ'+str(q),head,re.I):return y,q,m[0]
 if t in ('UNH','MRK','NVS','AZN','BA','GE'):
  m=re.search(r'(?:full[-\s]*year|year[-\s]*end|annual|reports)\s+(20\d{2})\s*(?:results)?',head,re.I)
  if m and re.search('fourth[- ]quarter|Q4|full[- ]year|annual results',head,re.I):return int(m[1]),4,m[0]
 raise ValueError('explicit reported FY/Q absent')
def release(t,tx,y,q):
 patterns={
  'UNH':[r'(?:MINNETONKA\s*,?\s*Minn\.?|NEW YORK,\s*N\.?Y\.?|MINNEAPOLIS)\s*\(?('+MDY+')',r'('+MDY+r')\s*\)?\s+UnitedHealth Group\s*\(NYSE'],
  'XOM':[r'(?:IRVING|SPRING),\s*Texas\s*[-,]*\s*('+MDY+')',r'FOR IMMEDIATE RELEASE\s*(?:[A-Z]+,\s*)?('+MDY+')'],
  'MRK':[r'(?:RAHWAY|KENILWORTH|WHITEHOUSE STATION),\s*N\.?\s*J\.?\s*[,(-]*\s*('+MDY+')'],
  'NVS':[r'Basel,\s*('+MDY+')'],
  'DHR':[r'WASHINGTON\s*,\s*(?:D\.C\.\s*,\s*)?('+MDY+')'],
  'BA':[r'(?:CHICAGO|ARLINGTON),?(?:\s*(?:Va\.?|Virginia)\s*,)?\s*[,(-]*\s*('+MDY+')'],
  'GE':[r'GE\s*[1-4]Q\s*20\d{2}\s*EARNIN\s*GS\s*('+MDY+')',r'(?:BOSTON|EVENDALE|FAIRFIELD|CINCINNATI)\s*[, -].{0,30}?('+MDY+')'],
  'NVO':[r'('+DMY+r')\s*-?\s*Financial report for the period',r'Financial report for the period\s+'+DP+r'\s+to\s+'+DP+r'\s*,?\s*('+DMY+')'],
  'BABA':[r'(?:Hangzhou|Hong Kong).{0,90}?('+MDY+')'],
  'SAP':[r'Walldorf,\s*Germany\s*[-,]+\s*('+MDY+')',r'Press Release by SAP(?: News)?\s+('+MDY+')'],
  'ABBV':[r'NORTH CHICAGO,\s*Ill\.?\s*,\s*('+MDY+')'],
  'PFE':[r'NEW YORK\s*,?\s*(?:N\.?Y\.?\s*,?\s*)?(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s*)?('+MDY+')'],
  'LLY':[r'('+MDY+r')\s+Eli Lilly and Company',r'INDIANAPOLIS\s*,\s*('+MDY+')'],
  'ASML':[r'VELDHOVEN,\s*the Netherlands,\s*('+MDY+')',r'Press release\s*-\s*('+MDY+')'],
  'ORCL':[r'(?:AUSTIN,\s*Texas|REDWOOD SHORES,\s*(?:California|Calif\.))\s*[, -]+\s*('+MDY+')'],
  'CHL':[r'^.{0,30}?('+DMY+r')\s+For Immediate Release',r'Hong Kong,\s*('+DMY+')',r'('+DMY+r'),\s*Hong Kong'],
  'SHEL':[r'(?:London|The Hague),?\s*('+DP+')',r'date of this (?:announcement|document),\s*('+DP+')',r'('+DP+r')\s+The information in this Report',r'^.{0,150}?('+DMY+r')\s+Royal Dutch Shell'],
 }
 if t=='NVO' and q==4:
  m=re.search(r'Financial statement for\s+'+str(y)+r'\s+('+DMY+')',tx[:1500],re.I)
  if m:return date(m[1]),m[0]
 if t=='XOM':
  m=re.search(r'('+MDY+r')\s+\d{1,2}:\d\d\s*[AP]M\s*(?:C[DS]T|E[DS]T)',tx[:1100],re.I)
  if m:return date(m[1]),m[0]
 if t=='TM':
  m=re.search(r'English translation from the original Japanese-language document\s+('+MDY+')',tx[:3000],re.I)
  if not m:raise ValueError('Toyota original publication header absent')
  return date(m[1]),m[0]
 if t=='AZN':
  # Original announcement header, including old PLC date/time formatting.
  m=re.search(r'(?:AstraZeneca(?:\s+PLC)?\s+)?('+DMY+r')(?:\s+\d\d:\d\d)?',tx[:350],re.I)
  if not m:raise ValueError('AZN original header date absent')
  return date(m[1]),m[0]
 for pat in patterns.get(t,[]):
  m=re.search(pat,tx if t in ('CHL','SHEL') else tx[:12000],re.I)
  if m:return date(m[1]),m[0]
 raise ValueError('original publication dateline/header absent')
def expected_period(t,y,q):
 if t in ('TM','BABA'):mo=[6,9,12,3][q-1];yy=y-1 if q<4 else y
 elif t=='ORCL':mo=[8,11,2,5][q-1];yy=y-1 if q<3 else y
 else:mo=q*3;yy=y
 return c.dt.date(yy,mo,calendar.monthrange(yy,mo)[1]).isoformat()
def period(t,tx,y,q,task,p,allow_support=True):
 if t=='ASML':
  for m in re.finditer(r'(?:quarter and (?:three|six|nine|twelve)(?: months|-month period)|three months|quarter) ended\s+('+DP+')',tx,re.I):
   d=date(m[1])
   if int(d[:4])==y and abs(int(d[5:7])-q*3)<=1:return d,proof(p,task['url'],m[0],'Explicit non-calendar ASML fiscal reporting period in release')
  raise ValueError('ASML actual fiscal end absent; no calendar substitution')
 if t=='PFE':
  for m in re.finditer(r'The financial statements present the three(?: and (?:six|nine|twelve))? months ended\s+('+DP+r').{0,90}?Subsidiaries operating outside the U.S.',tx,re.I):
   d=date(m[1])
   if int(d[:4])==y and abs(int(d[5:7])-q*3)<=1:return d,proof(p,task['url'],m[0],'Original financial-statement note states consolidated current reporting end and separately distinguishes overseas reporting lag')
  for m in re.finditer(r'(?:U\.?S\.? subsidiaries|domestic (?:operations|subsidiaries)).{0,160}?(?:ended on|ended|ending on|ending|was)\s+('+DP+')',tx,re.I):
   d=date(m[1])
   if int(d[:4])==y and abs(int(d[5:7])-q*3)<=1:return d,proof(p,task['url'],m[0],"Pfizer explicitly stated U.S. subsidiaries reporting-period end; overseas subsidiaries use a different reporting lag")
  raise ValueError('Pfizer U.S. reporting period end absent; no calendar substitution')
 if t in ('TM','BABA','NVO','CHL'):
  pats=[r'(?:through|to)\s+('+DP+')'] if t!='BABA' else [r'(?:quarter|three months|six months|period).{0,25}?ended\s+('+DP+')']
  expected=expected_period(t,y,q)
  for pat in pats:
   for m in re.finditer(pat,tx[:10000],re.I):
    if date(m[1])==expected:return expected,proof(p,task['url'],m[0],'Explicit actual reporting-period end in original release')
 if t=='XOM':
  label='FY' if q==4 else 'Q'+str(q)
  m=re.search(label+r'\s+'+str(y)+r'\s+(?:Fiscal Year|Quarter) Ended\s+('+MDY+')',task.get('archive_context',''),re.I)
  if m:return date(m[1]),proof(c.ROOT/task['archive_path'],task['archive_url'],m[0],'Exact fiscal-period end beside this release in original issuer financial-results archive')
 if t=='DHR':
  for m in re.finditer(r'(?:quarter|year|three months|twelve months|Three-Month Period|Six-Month Period).{0,30}?ended\s+('+DP+')',tx,re.I):
   d=date(m[1]);dd=c.dt.date.fromisoformat(d)
   if dd.year==y and abs(dd.month-q*3)<=1:return d,proof(p,task['url'],m[0],'Explicit non-calendar fiscal quarter/year ended in issuer release')
  raise ValueError('actual Danaher fiscal end not explicit; calendar substitute forbidden')
 expected=expected_period(t,y,q);d=c.dt.date.fromisoformat(expected);month=MONTHS[d.month-1]
 # A current-period complete date in a financial statement or reporting-period
 # sentence directly establishes the endpoint, including a balance-sheet date.
 for m in re.finditer(DP,tx,re.I):
  try:found_date=date(m[0])
  except ValueError:continue
  if found_date!=expected:continue
  ctx=tx[max(0,m.start()-240):m.end()+80]
  if re.search(r'\b(?:ended|ending|period to|months to|year to|balance sheet|as (?:at|of)|net debt|financial position)\b',ctx,re.I):return expected,proof(p,task['url'],ctx,'Explicit current-period date in original financial statement/reporting text')
 # PDF tables can place month/day columns above a separate year row. Preserve
 # the complete table header, require both the target month/day AND year.
 md=month+r'\s+'+str(d.day)+r'\s*,?'
 dm=str(d.day)+r'\s+'+month
 for m in re.finditer(r'(?:Three|Six|Nine|Twelve|3|6|9|12)\s*[- ]?Months\s+Ended|Year\s+Ended|CONSOLIDATED\s+(?:STATEMENTS OF (?:OPERATIONS|INCOME)|BALANCE SHEETS)',tx,re.I):
  ctx=tx[m.start():m.end()+330]
  if re.search(md+'|'+dm,ctx,re.I) and re.search(r'\b'+str(d.year)+r'\b',ctx):return expected,proof(p,task['url'],ctx,'Original statement header explicitly supplies month/day and current fiscal-year column; PDF column order retained in excerpt')
 if allow_support and t in ('MRK','NVS','ORCL','SHEL','SAP'):
  if t in ('MRK','ORCL','SHEL','SAP'):supports=c.load(c.HERE/(t+'_supports.json'),[])
  else:supports=[x for x in candidates(t) if 'financial-report' in x['url']]+c.load(c.HERE/'NVS_supports.json',[])
  for support in supports:
   label=support.get('label','')+' '+support['url']
   same=(support.get('release_url')==task['url'] or (str(y) in label and (re.search(WORDS[q]+r'[- ]+Quarter',label,re.I) or re.search(r'(?:q|quarter[-_/ ]?)'+str(q)+r'\b',label,re.I))))
   if not same:continue
   sp=c.ROOT/support['evidence_file'] if support.get('evidence_file') else c.RAW/t/(c.sha(support['url'].encode())[:20]+'.bin')
   if not sp.exists():continue
   try:return period(t,text(sp,t),y,q,support,sp,False)
   except ValueError:pass
 raise ValueError('actual fiscal period end evidence absent')
def candidates(t):return c.load(c.HERE/(t+'_candidates.json'),[])
def build():
 events={};rejected=[]
 for t in c.TICKERS:
  for task in candidates(t):
   p=c.ROOT/task['evidence_file'] if task.get('evidence_file') else c.RAW/t/(c.sha(task['url'].encode())[:20]+'.bin')
   if not p.exists():rejected.append(dict(ticker=t,url=task['url'],reason='original download absent'));continue
   m=c.load(p,{}) if task.get('indexed_original') else c.load(p.with_suffix('.meta.json'),{})
   if task.get('indexed_original'):m['sha256']=c.sha(p.read_bytes())
   try:
    tx=text(p,t);y,q,ident=identity(t,tx,task);rd,release_excerpt=release(t,tx,y,q)
    if not '2011-01-01'<=rd<=c.CUTOFF:continue
    end,ep=period(t,tx,y,q,task,p)
    if not end<rd:raise ValueError('release does not follow reported period')
    if (c.dt.date.fromisoformat(rd)-c.dt.date.fromisoformat(end)).days>120:raise ValueError('release more than 120 days after actual period')
    rep='public_search_index_excerpt' if task.get('indexed_original') else ('original_pdf_bytes' if p.read_bytes().startswith(b'%PDF') else ('original_public_api_json_bytes' if t=='BABA' else 'original_html_bytes'))
    zone=c.ZONES[t]
    if t=='SHEL' and (rd<'2022-01-21' or 'HAGUE' in release_excerpt.upper()):zone='Europe/Amsterdam'
    if t=='BA' and 'ARLINGTON' in release_excerpt.upper():zone='America/New_York'
    if t=='BABA' and 'HONG KONG' in release_excerpt.upper():zone='Asia/Hong_Kong'
    if t=='UNH' and 'NEW YORK' in release_excerpt.upper():zone='America/New_York'
    if t=='ORCL' and 'AUSTIN' in release_excerpt.upper():zone='America/Chicago'
    ev=dict(ticker=t,release_date=rd,publication_timestamp=None,precision='date_only',publication_timezone=zone,publication_timezone_basis='Conservative issuer-local calendar-date policy; geographic issuer dateline mapping. No CMS timezone or exact publication instant inferred.',first_public_time_verified=False,fiscal_year=y,fiscal_quarter=q,fiscal_period_end_actual=end,source_url=task['url'],source_class='issuer_original_release',source_representation=rep,evidence_path=c.rel(p),evidence_sha256=m['sha256'],raw_original_bytes_sha256=m['sha256'],raw_original_bytes_available=True,retrieved_at=m['retrieved_at'],evidence_excerpt=ident,release_date_evidence=[proof(p,task['url'],release_excerpt,'Original issuer dateline or release-document publication header')],fiscal_period_evidence=ep,additional_evidence=[dict(evidence_path=ep['evidence_path'],sha256=ep['sha256'],role='actual_fiscal_period_end')],historical_limitation='Current issuer archive is not a contemporaneous immutable feed. First-public time not established.',timestamp_note='No call time, publication schedule, CMS dateModified, PDF file time, or SEC acceptance substituted.',in_study_window=rd>='2016-01-01')
    if t=='SHEL':
     ip=c.HERE/'SHEL_IDENTITY.json';iid=c.load(ip,{})
     ev['issuer_identity_mapping']=iid['mapping'];ev['issuer_name_at_release']='Royal Dutch Shell plc' if rd<'2022-01-21' else 'Shell plc';ev['historical_ticker_context']='RDS.A and RDS.B (issuer-wide earnings, class not selected)' if rd<'2022-01-31' else 'SHEL'
     ev['additional_evidence'].append(dict(evidence_path=c.rel(ip),sha256=c.sha(ip.read_bytes()),role='issuer_identity_continuity'))
    if task.get('indexed_original'):
     ev.update(raw_original_bytes_available=False,raw_original_bytes_sha256=None,historical_limitation='Official original accessed as public search-index excerpt. Original HTTP HTML/PDF bytes were NOT downloaded; evidence SHA hashes the saved index capture only. First-public instant not established.')
    if t=='TM' and p.with_suffix('.visual_review.json').exists():
     vp=p.with_suffix('.visual_review.json');ev['additional_evidence'].append(dict(evidence_path=c.rel(vp),sha256=c.sha(vp.read_bytes()),role='visual_transcription_of_original_pdf_header'));ev['additional_evidence']+=c.load(vp,{})['images']
    k=(t,y,q)
    if k in events:
     old=events[k]
     if (old['release_date'],old['fiscal_period_end_actual'])!=(rd,end):raise ValueError('conflicting duplicate fiscal event')
     if old['source_representation']=='original_pdf_bytes':continue
    events[k]=ev
   except Exception as e:rejected.append(dict(ticker=t,url=task['url'],label=task.get('label'),reason=str(e)))
 out=sorted(events.values(),key=lambda r:(r['ticker'],r['release_date']));c.dump(c.HERE/'events.json',out);c.dump(c.HERE/'PARSE_PENDING.json',rejected)
 pending=[]
 for t in c.TICKERS:
  for y in range(2010,2028):
   for q in range(1,5):
    end=expected_period(t,y,q)
    if not '2010-12-01'<=end<='2026-06-30':continue
    if (t,y,q) not in events:pending.append(dict(ticker=t,fiscal_year=y,fiscal_quarter=q,status='pending_review',reason='Release date and actual fiscal end not both verified; see PARSE_PENDING and SOURCE_STOPS. Target cells are search targets, never assumed actual fiscal ends.'))
 c.dump(c.HERE/'PENDING.json',pending)
 counts=Counter(r['ticker'] for r in out)
 c.dump(c.HERE/'STATUS.json',dict(generated_utc=c.utc(),requested_release_window=['2011-01-01',c.CUTOFF],tickers=c.TICKERS,verified_events=len(out),events_by_ticker={t:counts[t] for t in c.TICKERS},pending_count=len(pending),parse_pending_count=len(rejected),all_events_date_only=True,source_stops=c.load(c.HERE/'SOURCE_STOPS.json',[]),complete=False))
 from finalize import run
 run(tested=False)
 print(c.json.dumps(dict(counts=counts,total=len(out),rejected=len(rejected)),indent=2))
if __name__=='__main__':build()
