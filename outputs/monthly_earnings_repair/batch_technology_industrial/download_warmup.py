from common import *
import concurrent.futures

def one(k):
 p=HERE/k/'warmup_discovery.json'
 if not p.exists():return
 rows=json.loads(p.read_text());urls=set()
 for row in rows:
  for u in re.findall(r'^.*?\((https://[^\s)]+)\)',row['rendering'],re.M):
   if k=='INTC' and 'www.intc.com/' in u and ('/press-releases/detail/' in u or '/all-sec-filings/content/' in u):urls.add(u)
   if k=='AMD' and 'ir.amd.com/' in u and ('/press-releases/detail/' in u or '/all-sec-filings/content/' in u):urls.add(u)
 for u in sorted(urls):
  try:b,p,m=fetch(k,u);print(k,len(b),u,flush=True)
  except Exception as e:print(k,e,flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,['INTC','AMD']))
