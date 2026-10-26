"""Strict original-release fact extraction. Date-only clocks remain null."""
import sys,json,re,datetime as dt,collections
from bs4 import BeautifulSoup
import core as c
M=r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
DATE=rf'({M})\.?\s+(\d{{1,2}}),?\s+(20\d{{2}})'
DATE_RE=re.compile(DATE,re.I)
MONTHS={m:i for i,m in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],1)}
QMAP={'first':1,'second':2,'third':3,'fourth':4,'1st':1,'2nd':2,'3rd':3,'4th':4}
IDENT={'GS':r'Goldman Sachs','KO':r'Coca.Cola','NKE':r'NIKE','HD':r'Home Depot','WFC':r'Wells Fargo','BAC':r'Bank of America','T':r'AT&T','MA':r'Mastercard','C':r'Citigroup|Citi','PM':r'Philip Morris','PYPL':r'PayPal','HSBC':r'HSBC','VZ':r'Verizon','BUD':r'Anheuser.Busch|AB InBev','MCD':r'McDonald','COST':r'Costco','CMCSA':r'Comcast','PEP':r'PepsiCo','ABT':r'Abbott','TMO':r'Thermo Fisher','DIS':r'Walt Disney|Disney'}
GEO={'GS':('NEW YORK','America/New_York'),'KO':('ATLANTA','America/New_York'),'HD':('A\s*T\s*LANTA','America/New_York'),'NKE':('BEAVERTON','America/Los_Angeles'),'T':('DA\s*LLAS','America/Chicago'),'WFC':('SAN FRANCISCO','America/Los_Angeles'),'BAC':('CHARLOTTE','America/New_York'),'MA':('PURCHASE','America/New_York'),'C':('NEW YORK','America/New_York'),'PM':('STAMFORD|NEW YORK|LAUSANNE','America/New_York'),'PYPL':('SAN JOSE','America/Los_Angeles'),'VZ':('NEW YORK','America/New_York'),'HSBC':('LONDON','Europe/London'),'BUD':('LEUVEN|BRUSSELS','Europe/Brussels'),'MCD':('OAK BROOK|CHICAGO','America/Chicago'),'COST':('ISSAQUAH','America/Los_Angeles'),'CMCSA':('PHILADELPHIA','America/New_York'),'PEP':('PURCHASE','America/New_York'),'ABT':('ABBOTT PARK','America/Chicago'),'TMO':('WALTHAM','America/New_York'),'DIS':('BURBANK','America/Los_Angeles')}
def date(m):return dt.date(int(m[3]),MONTHS[m[1][:3].lower()],int(m[2]))
NOTICE_RE=re.compile(r'(?:announces?|confirms?|updates?)\s+(?:the\s+)?(?:timing|logistics|schedule)|timing.{0,100}(?:release|results)|announces[^.]{0,100}earnings and conference call|(?:will|to) (?:report|release|announce)[^.]{0,80}(?:results|earnings)|schedules[^.]{0,80}(?:earnings|results)',re.I)
def notice_heading(b,t,index_block=None):
 if index_block is not None:return index_block.strip().splitlines()[0]
 if b.startswith(b'%PDF'):return t[:300].split('Consolidated Results')[0]
 if b.lstrip().startswith(b'{'):
  payload=json.loads(b)
  if 'title' in payload:return payload['title']
 title=BeautifulSoup(b,'html.parser').title
 return title.get_text(' ',strip=True)if title else t.split('.')[0]
def reject_notice(a,t):
 heading=a.get('notice_heading',t.split('.')[0][:400]);url=re.sub(r'[-_/]+',' ',a['url'])
 if NOTICE_RE.search(url)or NOTICE_RE.search(heading)or re.search(r'conference.call|earnings.call|webcast|updates.logistics',url,re.I):raise ValueError('scheduled earnings notice or conference logistics is not an actual results release')

def extract(a,s):
 if s=='BAC' and a.get('issuer_filing_mirror'):
  import bac_mirror_parser
  return bac_mirror_parser.extract(a)
 if s=='HSBC':
  from hsbc_parser import extract as parse_hsbc
  return parse_hsbc(a)
 p=c.ROOT/a['evidence_path'];b=p.read_bytes();mp=p.with_name(p.name+'.meta.json');meta=json.loads(mp.read_text())
 assert c.sha(b)==a['evidence_sha256']==meta['sha256']
 isindexed='index_block'in a;ispdf=b.startswith(b'%PDF');isapi=s=='PM' and not isindexed and b.lstrip().startswith(b'{');t=c.clean(c.pdf_text(p)) if ispdf else c.text(b)
 if isapi:
  payload=json.loads(b);body=payload['content'].replace('\\r\\n',' ').replace('\\n',' ').replace('\\\"','\"');t=c.clean(payload['title']+' '+c.text(body))
 if isindexed:
  ib=a['index_block']
  if s=='WFC' and '\n\n'in ib:ib=ib.split('\n\n',1)[1]
  t=c.clean(re.sub(r'\s*\|\s*',' ',ib))
 if not re.search(IDENT[s],t[:8000],re.I):raise ValueError('issuer identity absent')
 if s=='WFC' and isindexed and not re.search(r'Wells Fargo Reports?.{0,150}(?:Income|Quarter|Results|Earnings)',a['index_block'].strip().splitlines()[0]+' '+t[:1600],re.I):raise ValueError('not a quarterly financial-results announcement')
 reject_notice(dict(a,notice_heading=notice_heading(b,t,a.get('index_block'))),t)
 if re.search(r'announces timing|announces[^.]{0,100}earnings and conference call|will (?:report|release|announce)[^.]{0,80}(?:results|earnings)',t[:800],re.I):raise ValueError('scheduled earnings notice is not an actual results release')
 if not ispdf and not isapi and not isindexed:
  bs=BeautifulSoup(b,'html.parser');article=bs.select_one('article.full-news-article'); dateel=bs.select_one('time.date, .wd_release_date, .wd_date, .wd_article-info')
  if article:t=c.clean(article.get_text(' ',strip=True))
  if s=='BAC' and not dateel:
   node=bs.find(string=re.compile(DATE+r' at ',re.I));dateel=node.parent if node else None
  if s=='TMO' and not dateel:
   node=bs.find(string=re.compile(r'(?:Monday|Tuesday|Wednesday|Thursday|Friday), '+DATE,re.I));dateel=node.parent if node else None
  if dateel:release_match=DATE_RE.search(dateel.get_text(' ',strip=True));release_excerpt=dateel.get_text(' ',strip=True);basis='Issuer publication-date header on original release; CMS modification fields excluded'
  else:release_match=None
 else:release_match=None
 geo,zone=GEO[s]
 if isindexed and not release_match:
  im=re.search(r'^\s*(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*)?'+DATE+r'(?:\s+\d.*)?\s*$',a['index_block'],re.I|re.M)
  if im:release_match=DATE_RE.search(im.group());release_excerpt=im.group().strip();basis='Publication-date header explicitly present in public index of original issuer release; original bytes unavailable'
  if not release_match:
   nm=re.search(r'^\s*(\d{1,2})/(\d{1,2})/(20\d{2})\s*$',a['index_block'],re.M)
   if nm:
    formatted=dt.date(int(nm[3]),int(nm[1]),int(nm[2])).strftime('%B %d, %Y');release_match=DATE_RE.search(formatted);release_excerpt=nm.group().strip();basis='Explicit numeric publication-date header on indexed original release'
 region=r'(?:CT|Conn\.?|Connecticut|Mass\.?|Massachusetts|N\.?Y\.?|New York|N\.?C\.?|North Carolina|Calif\.?|California|Ill\.?|Illinois|Ore\.?|Oregon|Texas|Penn\.?|Pa\.?|Washington|Wash\.?|Switzerland|USA)'
 dm=re.search(rf'(?:{geo})(?:\s*,\s*{region})?[,\s.:\-()]*(?:(?:BUSINESS WIRE|PRNewswire|PR Newswire)[,\s.:\-()]*)?({DATE})',t[:18000],re.I)
 header_publication_excerpt=release_excerpt if release_match else None
 if dm and not re.search(r'conference|webcast|will report|scheduled',dm.group(),re.I):
  release_match=DATE_RE.search(dm.group());release_excerpt=dm.group();basis='Actual original body geographic dateline; takes precedence over regional republication/CMS page header'
 if not release_match:
  if s=='WFC':
   dm=re.search(rf'News Release[^A-Za-z]{{0,10}}({DATE})',t[:400],re.I)
   if dm:release_match=DATE_RE.search(dm.group());release_excerpt=dm.group();basis='Original issuer News Release publication-date header'
  elif s=='C':
   dm=re.search(rf'For Immediate Release.{{0,75}}?({DATE})',t[:5000],re.I)
   if dm:release_match=DATE_RE.search(dm.group());release_excerpt=dm.group();basis='Original For Immediate Release dated header'
 if not release_match:raise ValueError('no actual publication dateline/header')
 rd=date(release_match)
 if not(c.START<=rd.isoformat()<=c.CUTOFF):raise ValueError('outside requested release-date window')
 # Current quarter identity must occur in the release heading or opening result paragraph.
 qi=re.search(r'\b(first|second|third|fourth|1st|2nd|3rd|4th)[ -]quarter\b',t[:5000],re.I)
 if s=='C':qi=re.search(r'\b(FIRST|SECOND|THIRD|FOURTH) QUARTER.{0,25}RESULTS',t[:5000]) or qi
 if not qi:raise ValueError('quarter identity absent in opening')
 q=QMAP[qi[1].lower()];qexcerpt=t[max(0,qi.start()-80):qi.end()+150]
 # Dates are only extracted from explicit ended-period contexts, including tables.
 candidates=[]
 for ended in re.finditer(r'(?:(?:three|thirteen|twelve|six|nine|fifty.two|fifty.three|quarter|year|months|weeks|period)[^.!?]{0,90}?ended)',t,re.I):
  chunk=t[ended.start():ended.end()+240]
  for dm in DATE_RE.finditer(chunk):
   d=date(dm)
   if 0<(rd-d).days<=115:candidates.append((d,chunk,'Explicit ended-period wording in original release'))
  for dm in re.finditer(r'\b(\d{1,2})/(\d{1,2})/(20\d{2})\b',chunk):
   d=dt.date(int(dm[3]),int(dm[1]),int(dm[2]))
   if 0<(rd-d).days<=115:candidates.append((d,chunk,'Explicit ended-period table in original release'))
  # Month/day headers followed by a current/prior-year row.
  for dm in re.finditer(rf'({M})\.?\s+(\d{{1,2}}),\s+(?:{M})\.?\s+\d{{1,2}},\s+(20\d{{2}})\s+20\d{{2}}',chunk,re.I):
   d=date(dm)
   if 0<(rd-d).days<=115:candidates.append((d,chunk,'Explicit ended-period table, split current and comparative column headers'))
  # Multiple comparative date columns in an explicitly ended-period table.
  split=re.search(rf'((?:{M}\.?\s+\d{{1,2}},\s*){{2,6}})(?:\(in millions\)\s*)?((?:20\d{{2}}\s*){{2,6}})',chunk,re.I)
  if split:
   monthdays=list(re.finditer(rf'({M})\.?\s+(\d{{1,2}}),',split[1],re.I));years=re.findall(r'20\d{2}',split[2])
   if len(monthdays)==len(years):
    for md,yr in zip(monthdays,years):
     d=dt.date(int(yr),MONTHS[md[1][:3].lower()],int(md[2]))
     if 0<(rd-d).days<=115:candidates.append((d,chunk,'Explicit ended-period table, matched month/day and year columns'))
 period_doc=None
 if s in ['BAC','T','C']:
  from period_sources import candidates as supplement_candidates
  extra=supplement_candidates(a,s,q,rd)
  if extra:
   pe,pexcerpt,pebasis,period_doc,period_header=max(extra,key=lambda x:x[0]);candidates.append((pe,pexcerpt,pebasis))
 if not candidates:raise ValueError('actual fiscal period-end missing')
 pe,pexcerpt,pebasis=max(candidates,key=lambda x:x[0])
 if period_doc:pe,pexcerpt,pebasis,period_doc,period_header=max(extra,key=lambda x:x[0])
 fy=pe.year+(1 if s=='NKE' and pe.month>5 else 0)+(1 if s=='COST' and pe.month>9 else 0)+(1 if s=='DIS' and q==1 else 0)-(1 if s=='HD' and pe.month<3 else 0)-(1 if s=='KO' and q==4 and pe.month==1 else 0)
 # Year cross-check against source text, close to quarter/fiscal identity.
 ymp=[m for m in re.finditer(rf'\b{fy}\b',t[:6000])]
 if not ymp:raise ValueError('fiscal year not corroborated in opening')
 yi=min(ymp,key=lambda m:abs(m.start()-qi.start()));yexcerpt=t[max(0,yi.start()-100):yi.end()+100]
 if s not in ['HD','NKE','COST','DIS'] and not (pe.month==q*3 or (pe.month==(q*3)%12+1 and pe.day<=7)):raise ValueError('quarter identity mismatches explicit actual end')
 if s=='NKE' and {2:3,5:4,8:1,11:2}.get(pe.month)!=q:raise ValueError('Nike fiscal quarter mismatch')
 if s=='HD' and ((q==4 and pe.month not in [1,2]) or (q<4 and pe.month not in [q*3+1,q*3+2])):raise ValueError('Home Depot fiscal quarter mismatch')
 ge=re.search(geo,t,re.I)
 if s=='PM' and ge and 'LAUSANNE'in ge.group().upper():zone='Europe/Zurich'
 if not ge and s not in ['WFC','BAC','PYPL']:raise ValueError('issuer-local geography absent')
 evidence=lambda ex: c.proof(p,a['url'],ex)
 ev=dict(ticker=s,release_date=rd.isoformat(),publication_timestamp=None,fiscal_period_end_actual=pe.isoformat(),fiscal_year=fy,fiscal_quarter=q,source_url=a['url'],source_class='issuer_original_earnings_release_api_payload' if isapi else ('issuer_original_earnings_release_pdf' if ispdf else 'issuer_original_earnings_release_html'),evidence_path=c.rel(p),evidence_sha256=c.sha(b),source_retrieved_at=meta['retrieved_at'],precision='date_only',publication_timezone=zone,publication_timezone_basis='Conservative issuer-local calendar-date research policy; no publication clock inferred.',first_public_time_verified=False,official_date_published_raw=release_excerpt,release_date_basis=basis,release_date_evidence=[dict(evidence(release_excerpt),kind='original_release_publication_date')],fiscal_period_evidence=dict(evidence(pexcerpt),basis=pebasis),fiscal_identity_evidence=dict(evidence(qexcerpt+' | '+yexcerpt),quarter_excerpt=qexcerpt,year_excerpt=yexcerpt,basis='Original quarter/year wording cross-checked against explicit actual fiscal period-end; issuer fiscal-year convention applies'),publication_timezone_evidence=dict(evidence(t[max(0,ge.start()-30):ge.end()+80] if ge else 'Issuer-local San Francisco calendar-date policy'),kind='issuer_local_calendar_date_policy',geographic_mapping=geo+' -> '+zone,cms_timezone_observed=False,exact_publication_time_inferred=False),timestamp_note='No SEC filing/acceptance date, CMS modification timestamp, PDF metadata clock or conference-call time treated as release instant.',historical_limitation='Original issuer archive retrieved currently; not contemporaneous immutable capture.',in_study_window=rd.isoformat()>='2015-01-01')
 if header_publication_excerpt:ev['page_publication_header_raw']=header_publication_excerpt
 if s in ['BAC','PYPL'] and not ge:
  for ga in json.loads((c.HERE/s/'release_queue.json').read_text()):
   if 'newsroom' not in ga['url'] or 'evidence_path'not in ga:continue
   gp=c.ROOT/ga['evidence_path'];gt=c.text(gp.read_bytes());gg=re.search(geo,gt,re.I)
   if gg:
    ev['publication_timezone_evidence']=dict(c.proof(gp,ga['url'],gt[max(0,gg.start()-40):gg.end()+100]),kind='same_issuer_geographic_policy_evidence',geographic_mapping=geo+' -> '+zone,cms_timezone_observed=False,exact_publication_time_inferred=False);break
  else:raise ValueError('Issuer geography evidence absent')
 if s=='WFC' and not ge:
  gp=c.HERE/'WFC/indexed_2015_2018.json';gv=json.loads(gp.read_text());block=next(x for x in gv['response'].split('--------------------------------------------------------------------------------')if 'SAN FRANCISCO' in x);gu=re.search(r'\((https://[^\s]+)\)',block)[1];gm=re.search(r'SAN FRANCISCO.{0,150}',block);ev['publication_timezone_evidence']=dict(c.proof(gp,gu,gm.group()),kind='indexed_issuer_local_geography_policy',geographic_mapping='San Francisco -> America/Los_Angeles',cms_timezone_observed=False,exact_publication_time_inferred=False,evidence_class='public_search_index_response_not_original_file')
 if isindexed:
  ev['source_class']='public_indexed_issuer_release_excerpt';ev['indexed_source_only']=True;ev['original_source_fully_cached']=False;ev['original_source_sha256']=None;ev['index_query']=a['index_query'];ev['historical_limitation']='Cached public search-index response only. SHA covers the saved response, not original issuer file bytes; indexed excerpts can be incomplete and stale.';ev['release_date_basis']='Indexed primary issuer release actual dateline/publication header. Relative crawl/publication labels and conference times excluded.'
 if period_doc:
  pp=c.ROOT/period_doc['evidence_path'];ev['fiscal_period_evidence']=dict(c.proof(pp,period_doc['url'],pexcerpt),basis=pebasis,original_statement_title=period_header)
  ev['additional_evidence']=[c.proof(pp,period_doc['url'])]
  if period_doc.get('quarter_package_evidence'):ev['quarter_package_evidence']=period_doc['quarter_package_evidence']
 if a.get('archive'):ev['official_archive_evidence']=a['archive']
 return ev

def main():
 selected=set(sys.argv[1:]);events=[];pending=[]
 if selected and (c.HERE/'original_events.json').exists():events=[r for r in json.loads((c.HERE/'original_events.json').read_text())if r['ticker']not in selected]
 if selected and (c.HERE/'SOURCE_PENDING.json').exists():pending=[r for r in json.loads((c.HERE/'SOURCE_PENDING.json').read_text())if r['ticker']not in selected]
 for qp in sorted(c.HERE.glob('*/release_queue.json')):
  s=qp.parent.name
  if selected and s not in selected:continue
  for a in json.loads(qp.read_text()):
   if 'evidence_path' not in a:pending.append(dict(ticker=s,url=a['url'],reason=a.get('error','not downloaded')));continue
   try:events.append(extract(a,s))
   except Exception as e:pending.append(dict(ticker=s,url=a['url'],evidence_path=a['evidence_path'],reason=str(e)))
 unique={}
 for x in events:
  k=(x['ticker'],x['fiscal_year'],x['fiscal_quarter'])
  if k in unique and unique[k]['release_date']!=x['release_date']:raise ValueError(('conflicting release dates',k))
  unique[k]=x
 events=sorted(unique.values(),key=lambda x:(x['ticker'],x['release_date']))
 c.dump(c.HERE/'original_events.json',events);c.dump(c.HERE/'SOURCE_PENDING.json',pending)
 print('EVENTS',len(events),dict(collections.Counter(x['ticker'] for x in events)))
 print('PENDING',dict(collections.Counter((x['ticker']+': '+x['reason'].split(' https:')[0]) for x in pending)))
if __name__=='__main__':main()
