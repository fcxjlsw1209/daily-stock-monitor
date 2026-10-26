"""Independent original release/date/period proof joins; no filing timestamps."""
from join_collect import *
def tesla_notices():
 out=[];pending=[]
 for p in (HERE/'TSLA/public_index_observations').glob('notice_*.json'):
  j=json.loads(p.read_text());t=clean(j['rendering']);um=re.search(r'\((https://ir\.tesla\.com/[^\s)]+)\)',t)
  dm=re.search(r'On ('+DATE+r'), Tesla, Inc\. released its financial results for the quarter ended ('+DATE+r') by posting its '+ORD+r' Quarter (20\d{2}) Update',t,re.I)
  if not dm or not um:pending.append(dict(evidence_path=rel(p),reason='Explicit actual release issuance statement absent'));continue
  r=dict(release_date=iso(dm.group(1)),fiscal_period_end_actual=iso(dm.group(2)),fiscal_quarter=QMAP[dm.group(3).lower()],fiscal_year=int(dm.group(4)),dateline_excerpt=dm.group(),period_excerpt=dm.group(),identity_excerpt=dm.group())
  e=event('TSLA',p,dict(url=um.group(1),retrieved_at=j['observed_at']),r,'public_index_observation_of_issuer_explicit_actual_release_notice_not_full_bytes')
  e['release_date_basis']='Original issuer Item 2.02 explicitly states the actual financial-release publication date and period; no SEC filing/acceptance timestamp used.'
  e['publication_timezone']='America/Chicago' if r['fiscal_year']>=2022 else 'America/Los_Angeles'
  e['evidence_limitation']='SHA-256 identifies saved public-index observation of original issuer document, not the original HTML/PDF bytes.';out.append(e)
 merge('TSLA',out,pending)
def tmobile():
 reports={};out=[];pending=[]
 for mp in (HERE/'TMUS/raw').glob('*.meta.json'):
  if 'text.meta' in mp.name:continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or 's29.q4cdn.com/310188824/' not in m['url'] or not p.read_bytes().startswith(b'%PDF'):continue
  t=clean(extract(p));dm=re.search(r'For the (?:quarterly period|fiscal year) ended ('+DATE+r')',t[:65000],re.I)
  if not dm or not re.search(r'T-MOBILE US, INC',t[:65000],re.I):continue
  period=iso(dm.group(1));pd=dt.date.fromisoformat(period)
  reports[(pd.year,pd.month//3)]=dict(evidence_path=rel(p),sha256=m['sha256'],source_url=m['url'],excerpt=dm.group(),period=period)
 for mp in (HERE/'TMUS/raw').glob('*.meta.json'):
  if 'text.meta' in mp.name:continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or not m.get('content_type','').startswith('text/html') or '/news/business/' not in m['url']:continue
  t=clean(extract(p));dm=re.search(r'Bellevue[A-Za-z ,.-]{0,25}('+DATE+r')',t,re.I)
  qm=re.search(r'reported (?:its )?'+ORD+r' quarter(?: and (?:full[- ]year|full year))? (20\d{2}) results',t,re.I)
  if not dm or not qm:continue
  fy=int(qm.group(2));q=QMAP[qm.group(1).lower()];proof=reports.get((fy,q))
  if not proof:continue
  release=iso(dm.group(1));period=proof['period'];assert 0<(dt.date.fromisoformat(release)-dt.date.fromisoformat(period)).days<100
  r=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=dm.group(),period_excerpt=proof['excerpt'],identity_excerpt=qm.group())
  e=event('TMUS',p,m,r,'issuer_original_release_full_bytes_with_original_financial_report_period_proof')
  e['fiscal_period_evidence']={k:v for k,v in proof.items() if k!='period'};e['additional_evidence']=[e['fiscal_period_evidence']]
  e['release_date_basis']='Actual results dateline of original corporate newsroom release. Actual period independently confirmed by original issuer financial statement cover; filing date not used.';out.append(e)
 merge('TMUS',out,pending)
if __name__=='__main__':tesla_notices();tmobile()
