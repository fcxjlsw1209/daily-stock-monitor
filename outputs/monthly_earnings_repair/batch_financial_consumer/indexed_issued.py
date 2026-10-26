"""Explicit past publication statements from indexed issuer originals; not filing dates."""
import re,json
import core as c
from build_events import DATE,DATE_RE,date
CONFIG={'WFC':('72971',r'Wells Fargo\s*&\s*Company','San Francisco','America/Los_Angeles'),'PM':('1413329',r'Philip Morris International Inc\.',r'New York|Stamford','America/New_York')}
def facts(text,ticker):
 cik,name,geo,tz=CONFIG[ticker]
 # The clause must assert an already issued earnings release, directly tied to the actual quarter end.
 rx=rf'On\s+({DATE}),\s*{name}\s*(?:\([^)]{{0,60}}\))?\s*issued a press release\s+(?:regarding|announcing)\s+its\s+(?:results of operations and financial condition|financial results)\s+for the quarter(?: and (?:nine months|six months|three months|year))?\s+ended\s+({DATE})'
 m=re.search(rx,text,re.I)
 if not m:raise ValueError('No explicit past issuer earnings-publication clause with quarter end')
 ds=list(DATE_RE.finditer(m.group()));rd,pe=map(date,ds);tail=text[m.end():m.end()+400]
 if not 0<(rd-pe).days<=115 or pe.month not in [3,6,9,12]:raise ValueError('Reporting period inconsistent')
 q=pe.month//3;fy=pe.year
 if ticker=='WFC':
  qm=re.search(r'([1-4])Q(\d{2})\s+Quarterly Supplement',tail,re.I)
  if not qm or (int(qm[1]),2000+int(qm[2]))!=(q,fy):raise ValueError('Explicit contemporaneous quarterly supplement identity missing')
  identity=m.group()+tail[:qm.end()]
 else:identity=m.group()
 gm=re.search(geo,text,re.I)
 if not gm:raise ValueError('Issuer geography absent in indexed source')
 return dict(release_date=rd.isoformat(),fiscal_period_end_actual=pe.isoformat(),fiscal_year=fy,fiscal_quarter=q,publication_timezone=tz,release_quote=m.group(),identity_quote=identity,geo_quote=text[max(0,gm.start()-70):gm.end()+100])
def extract(p,block,v,ticker):
 um=re.match(r'^\s*(.*?) \((https://[^\s]+)\)',block);url=um[2];cik=CONFIG[ticker][0]
 if not url.startswith('https://www.sec.gov/Archives/edgar/data/'+cik+'/'):raise ValueError('Wrong issuer original path')
 # Search snippets may repeat an original passage; admission requires it in the indexed body after the prelude.
 body=block.split('\n\n',1)[1] if '\n\n'in block else ''
 t=c.clean(re.sub(r'\s*\|\s*',' ',body));f=facts(t,ticker)
 if not '2010-10-01'<=f['release_date']<=c.CUTOFF:raise ValueError('Outside event window including training predecessor')
 release=f.pop('release_quote');ident=f.pop('identity_quote');geo=f.pop('geo_quote');m=json.loads(p.with_name(p.name+'.meta.json').read_text());proof=lambda x:c.proof(p,url,x)
 return dict(f,ticker=ticker,precision='date_only',publication_timestamp=None,source_url=url,source_class='public_indexed_issuer_announcement_statement',evidence_path=c.rel(p),evidence_sha256=c.sha(p.read_bytes()),source_retrieved_at=m['retrieved_at'],indexed_source_only=True,original_source_fully_cached=False,original_source_sha256=None,index_query=v.get('queries',v.get('query')),first_public_time_verified=False,official_date_published_raw=release,release_date_basis='Original issuer body explicitly says On DATE issuer issued a press release reporting that quarter; filing date and acceptance time excluded.',release_date_evidence=[dict(proof(release),kind='indexed_explicit_original_announcement_statement')],fiscal_period_evidence=dict(proof(release),basis='Actual quarter ended date in same issued-earnings-release statement'),fiscal_identity_evidence=dict(proof(ident),quarter_excerpt=ident,year_excerpt=ident,basis='Explicit issuer quarter-ended date and fiscal year; WFC additionally requires matching nQyy Quarterly Supplement identity; PM calendar fiscal year.'),publication_timezone_evidence=dict(proof(geo),kind='indexed_issuer_local_geography_policy',cms_timezone_observed=False,exact_publication_time_inferred=False),publication_timezone_basis='Conservative issuer-local calendar date from original issuer address; no publication clock inferred.',historical_limitation='Public index only; SHA covers cached response, not original document bytes; no direct SEC requests.',timestamp_note='No filing/acceptance, CMS, PDF metadata or conference time used.',in_study_window=f['release_date']>='2015-01-01')
def run(ticker):
 out=[];pending=[]
 for p in (c.HERE/ticker).glob('indexed*.json'):
  if p.name.endswith('.meta.json'):continue
  v=json.loads(p.read_text())
  for block in v.get('response','').split('--------------------------------------------------------------------------------'):
   if '/Archives/edgar/data/'+CONFIG[ticker][0]+'/'not in block[:500]:continue
   try:out.append(extract(p,block,v,ticker))
   except Exception as e:pending.append(dict(evidence_path=c.rel(p),reason=str(e),header=block.splitlines()[0:2]))
 unique={}
 for e in out:
  k=e['fiscal_year'],e['fiscal_quarter']
  if k in unique:assert (e['release_date'],e['fiscal_period_end_actual'])==(unique[k]['release_date'],unique[k]['fiscal_period_end_actual'])
  else:unique[k]=e
 if ticker=='PM':
  cp=c.HERE/'PM/indexed_issued2011.json';cv=json.loads(cp.read_text());cu='https://www.sec.gov/Archives/edgar/data/1413329/000119312510028437/d8k.htm'
  cb=next(z for z in cv['response'].split('--------------------------------------------------------------------------------')if '('+cu+')'in z)
  ct=c.clean(re.sub(r'\s*\|\s*',' ',cb.split('\n\n',1)[1]));cm=re.search(r'On February 11, 2010.{0,300}fiscal year ended December 31, 2009',ct);assert cm
  for e in unique.values():e['fiscal_calendar_evidence']=dict(c.proof(cp,cu,cm.group()),kind='pre_training_original_issuer_calendar_fiscal_year_definition',basis='Original issuer statement explicitly identifies December 31 as fiscal year-end; actual quarter-ended dates map to calendar fiscal quarters, no release or end date imputed.')
 c.dump(c.HERE/ticker/'issued_events.json',list(unique.values()));c.dump(c.HERE/ticker/'ISSUED_PENDING.json',pending);print(ticker,len(unique),sorted(unique))
if __name__=='__main__':
 for ticker in ['WFC','PM']:run(ticker)
