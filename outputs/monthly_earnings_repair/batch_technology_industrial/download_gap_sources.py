from common import *
import concurrent.futures
def palantir():
 rows=json.loads((HERE/'PLTR/raw/8k_feed.bin').read_text())['GetEdgarFilingListResult']
 for x in rows:
  u=next((z['Url'] for z in x['DocumentList'] if z['DocumentType']=='HTML'),None)
  if u:
   try:b,p,m=fetch('PLTR',u,'filing_'+str(x['FilingId']))
   except Exception as e:print('PLTR',str(e),flush=True)
def tmus():
 for mp in sorted((HERE/'TMUS/raw').glob('*.meta.json')):
  if mp.name.endswith('.text.meta.json'):continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or 'html' not in m.get('content_type',''):continue
  for a in BeautifulSoup(p.read_bytes(),'html.parser').find_all('a',href=True):
   u=urllib.parse.urljoin(m['url'],a['href'])
   if '.pdf' not in u or not re.search('earn|release|result',u+' '+a.get_text(),re.I):continue
   if 'sec.gov' in u or 'investor.t-mobile.com' in u:continue
   try:b,p,m=fetch('TMUS',u);print('TMUS PDF',len(b),flush=True)
   except Exception as e:print('TMUS PDF',str(e),flush=True)
def mirrors():
 rows=json.loads((HERE.parent/'SEC_REVIEW_QUEUE.json').read_text())
 print('SEC QUEUE TYPE',type(rows).__name__,flush=True)
 if isinstance(rows,dict):rows=rows['candidates']
 for k,base in [('INTC','https://www.intc.com/filings-reports/all-sec-filings/content/'),('AMD','https://ir.amd.com/financial-information/sec-filings/content/')]:
  for r in rows:
   if r['symbol']!=k or r['filing_date']>='2020-01-01':continue
   url=base+r['accession']+'/'+r['document_url'].rsplit('/',1)[-1]
   try:
    b,p,m=fetch(k,url);print(k,'MIRROR',len(b),flush=True)
    for a in BeautifulSoup(b,'html.parser').find_all('a',href=True):
     if not re.search('99.1|press release|earnings',a.get_text(),re.I):continue
     u=urllib.parse.urljoin(url,a['href'])
     if urllib.parse.urlsplit(u).netloc!=urllib.parse.urlsplit(base).netloc:continue
     try:b,p,m=fetch(k,u);print(k,'EXHIBIT',len(b),flush=True)
     except Exception as ex:print(k,'EXHIBIT',str(ex),flush=True)
   except Exception as e:print(k,'MIRROR',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda f:f(),[palantir,tmus,mirrors]))
