"""Indexed original issuer announcement clauses, never SEC header dates."""
import re,json,datetime as dt
import core as c
from build_events import DATE_RE,DATE,date,QMAP

def facts(t):
 if not re.search(r'AT&T\s+INC\.',t,re.I):raise ValueError('Explicit issuer identity absent')
 pattern=rf'Press release(?: and Investor Briefing)? dated\s+({DATE})\s+reporting financial results for the (first|second|third|fourth) quarter ended\s+({DATE})'
 pm=re.search(pattern,t,re.I)
 if not pm:raise ValueError('Explicit dated earnings release and actual quarter-ended clause absent')
 dates=list(DATE_RE.finditer(pm.group()));rd,pe=map(date,dates);q=QMAP[re.search(r'(first|second|third|fourth) quarter',pm.group(),re.I)[1].lower()]
 announced=[]
 for am in re.finditer(r'(?:registrant|AT&T|We) announced on\s+',t,re.I):
  dm=DATE_RE.match(t,am.end())
  if dm and date(dm)==rd and re.search('results of operations|net income|results for',t[dm.end():dm.end()+140],re.I):announced.append(t[am.start():dm.end()+180])
 if not announced:raise ValueError('No matching explicit past announcement statement')
 if pe.month!=q*3 or not 0<(rd-pe).days<=115:raise ValueError('Quarter end conflicts with original quarter identity')
 geo=re.search(r'Dallas,?\s+Texas',t,re.I)
 if not geo:raise ValueError('Original issuer location absent')
 return dict(release_date=rd.isoformat(),fiscal_period_end_actual=pe.isoformat(),fiscal_quarter=q,fiscal_year=pe.year,release_quote=announced[0],period_quote=pm.group(),geo_quote=t[max(0,geo.start()-60):geo.end()+80])

def extract(p,block,v):
 m=re.match(r'^\s*(.*?) \((https://[^\s]+)\)',block);url=m[2]
 if not url.startswith('https://www.sec.gov/Archives/edgar/data/732717/'):raise ValueError('Not the target issuer filing path')
 t=c.clean(re.sub(r'\s*\|\s*',' ',block));f=facts(t)
 if not c.START<=f['release_date']<=c.CUTOFF:raise ValueError('Outside requested window')
 release=f.pop('release_quote');period=f.pop('period_quote');geo=f.pop('geo_quote');meta=json.loads(p.with_name(p.name+'.meta.json').read_text());proof=lambda ex:c.proof(p,url,ex)
 return dict(f,ticker='T',precision='date_only',publication_timestamp=None,publication_timezone='America/Chicago',source_url=url,source_class='public_indexed_issuer_announcement_statement',evidence_path=c.rel(p),evidence_sha256=c.sha(p.read_bytes()),source_retrieved_at=meta['retrieved_at'],indexed_source_only=True,original_source_fully_cached=False,original_source_sha256=None,index_query=v.get('queries',v.get('query')),first_public_time_verified=False,official_date_published_raw=release,release_date_basis='Public index reproduces explicit past announcement statement and matching dated earnings-release exhibit description; SEC filing/header/acceptance date excluded.',release_date_evidence=[dict(proof(release),kind='indexed_explicit_original_announcement_statement'),dict(proof(period),kind='indexed_original_earnings_release_description')],fiscal_period_evidence=dict(proof(period),basis='Explicit current quarter ended in original earnings-release description'),fiscal_identity_evidence=dict(proof(period),quarter_excerpt=period,year_excerpt=period,basis='Explicit ordinal quarter and actual reporting year'),publication_timezone_evidence=dict(proof(geo),kind='indexed_issuer_local_geography_policy',geographic_mapping='Dallas, Texas -> America/Chicago',cms_timezone_observed=False,exact_publication_time_inferred=False),publication_timezone_basis='Conservative issuer-local calendar-date policy from original address, not a publication clock.',historical_limitation='Public search-index excerpt only; checksum is the response cache, not original SEC document bytes. No direct SEC request was made.',timestamp_note='No SEC filing/acceptance, CMS, conference or PDF metadata instant used.',in_study_window=f['release_date']>='2015-01-01')

def run():
 out=[];pending=[]
 for p in (c.HERE/'T').glob('indexed*.json'):
  if p.name.endswith('.meta.json'):continue
  v=json.loads(p.read_text())
  for block in v['response'].split('--------------------------------------------------------------------------------'):
   if 'https://www.sec.gov/Archives/edgar/data/732717/'not in block[:350]:continue
   try:out.append(extract(p,block,v))
   except Exception as e:pending.append(dict(evidence_path=c.rel(p),error=str(e),source=block.splitlines()[0:2]))
 unique={}
 for e in out:
  k=e['fiscal_year'],e['fiscal_quarter']
  if k in unique:assert e['release_date']==unique[k]['release_date']
  else:unique[k]=e
 c.dump(c.HERE/'T/announcement_events.json',list(unique.values()));c.dump(c.HERE/'T/ANNOUNCEMENT_PENDING.json',pending);print('T indexed past-announcement events',len(unique),sorted(unique))
if __name__=='__main__':run()
