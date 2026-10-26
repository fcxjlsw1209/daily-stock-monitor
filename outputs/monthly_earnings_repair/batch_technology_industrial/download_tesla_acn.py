from common import *
import concurrent.futures
def acn():
 rows=json.loads((HERE/'ACN/raw/query_index.bin').read_text())['data']
 for off in [1000,2000,3000]:
  try:b,p,m=fetch('ACN','https://newsroom.accenture.com/query-index.json?offset='+str(off),'query_'+str(off));rows+=json.loads(b)['data']
  except Exception as e:print('ACN INDEX',e,flush=True)
 urls=sorted(set('https://newsroom.accenture.com'+x['path'] for x in rows if re.search(r'Accenture Reports.*(?:Quarter|quarter).*Results',x['title']) and '/news/' in x['path']))
 dump(HERE/'ACN/DISCOVERY.json',urls)
 for u in urls:
  try:b,p,m=fetch('ACN',u);print('ACN',u,len(b),flush=True)
  except Exception as e:print('ACN',str(e),flush=True)
def tesla():
 for p in sorted((HERE/'TSLA/public_index_observations').glob('deck_*.json')):
  t=json.loads(p.read_text())['rendering'];urls=re.findall(r'https://(?:assets-ir.tesla.com|digitalassets.tesla.com|www.tesla.com)/[^\s)";]+',t)
  if not urls:continue
  try:b,p,m=fetch('TSLA',urls[0]);print('TSLA',len(b),flush=True)
  except Exception as e:print('TSLA',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda f:f(),[acn,tesla]))
