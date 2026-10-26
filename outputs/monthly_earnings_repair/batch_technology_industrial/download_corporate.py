from common import *
import concurrent.futures
def corp(k):
 urls=set()
 for p in (HERE/k/'public_index_observations').glob('search_corp_*.json'):
  s=json.loads(p.read_text())['rendering']
  for u in re.findall(r'\((https?://[^\s)]+)\)',s):
   host=urllib.parse.urlsplit(u).netloc
   if host not in {'www.salesforce.com','www.t-mobile.com','www.caterpillar.com'}:continue
   if not re.search(r'earnings|results|release|financial',u,re.I):continue
   urls.add(u.split('?')[0])
 dump(HERE/k/'DISCOVERY.json',sorted(urls))
 for u in sorted(urls):
  try:b,p,m=fetch(k,u);print(k,len(b),u,flush=True)
  except Exception as e:print(k,str(e),flush=True)
def acn_pdf():
 for mp in sorted((HERE/'ACN/raw').glob('*.meta.json')):
  if mp.name.endswith('.text.meta.json'):continue
  m=json.loads(mp.read_text());p=mp.with_name(mp.name.replace('.meta.json','.bin'))
  if m['status']!=200 or '/news/' not in m['url']:continue
  for a in BeautifulSoup(p.read_bytes(),'html.parser').find_all('a',href=True):
   u=urllib.parse.urljoin(m['url'],a['href'])
   if '.pdf' not in u or 'results' not in u:continue
   try:b,p,m=fetch('ACN',u);print('ACN PDF',len(b),flush=True)
   except Exception as e:print('ACN PDF',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda f:f(),[lambda:corp('CRM'),lambda:corp('TMUS'),lambda:corp('CAT'),acn_pdf]))
