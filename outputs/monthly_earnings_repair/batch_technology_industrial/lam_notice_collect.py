"""Issuer-linked complete originals: Item 2.02 actual issuance statement, no filing clock."""
from join_collect import *
def build():
 index=None
 for mp in (HERE/'LRCX/raw').glob('*.meta.json'):
  if '.text.' in mp.name:continue
  m=json.load(open(mp))
  if 'investor.lamresearch.com/sec-filings?' in m['url'] and 'year=2011' in m['url']:
   p=mp.with_name(mp.name.replace('.meta.json','.bin'));index=(p,m);break
 assert index
 ip,im=index;links={a['href'] for a in BeautifulSoup(ip.read_bytes(),'html.parser').select('a[href]')}
 out=[];pending=[]
 for mp in (HERE/'LRCX/raw').glob('*.meta.json'):
  if '.text.' in mp.name:continue
  m=json.load(open(mp))
  if m['status']!=200 or 'app.quotemedia.com' not in m['url']:continue
  assert m['url'] in links
  p=mp.with_name(mp.name.replace('.meta.json','.bin'));t=clean(extract(p))
  dm=re.search(r'On ('+DATE+r'), Lam Research Corporation.{0,70}?issued a press release announcing its financial results for the fiscal quarter ended ('+DATE+r')',t,re.I)
  if not dm:pending.append(dict(source_url=m['url'],reason='Explicit actual issuance statement absent'));continue
  release=iso(dm.group(1));period=iso(dm.group(2));pd=dt.date.fromisoformat(period)
  _,cy,cm=min((abs((pd-dt.date(y,mo,calendar.monthrange(y,mo)[1])).days),y,mo) for y in [pd.year-1,pd.year,pd.year+1] for mo in [3,6,9,12])
  fy=cy+(cm>6);q={9:1,12:2,3:3,6:4}[cm]
  r=dict(release_date=release,fiscal_period_end_actual=period,fiscal_year=fy,fiscal_quarter=q,dateline_excerpt=dm.group(),period_excerpt=dm.group(),identity_excerpt='June-ending fiscal-year mapping of exact original quarter end '+period)
  e=event('LRCX',p,m,r,'issuer_linked_regulatory_original_full_bytes');e['release_date_basis']='Original Item 2.02 explicitly states actual press-release issuance date; neither filing date nor acceptance clock used.';e['issuer_source_url']=im['url'];e['additional_evidence']=[dict(evidence_path=rel(ip),sha256=im['sha256'],source_url=im['url'],excerpt='Official Lam IR archive directly links this QuoteMedia-hosted original under its 2011 8-K results files')];out.append(e)
 merge('LRCX',out,pending)
if __name__=='__main__':build()
