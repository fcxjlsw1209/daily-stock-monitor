from common import *
import concurrent.futures
def csco():
 for u in json.loads((HERE/'CSCO/DISCOVERY.json').read_text()):
  try:b,p,m=fetch('CSCO',u);print('CSCO',len(b),flush=True)
  except Exception as e:print('CSCO ERROR',str(e),flush=True)
def acn():
 for name,u in [('query_index','https://newsroom.accenture.com/query-index.json'),('site_map','https://newsroom.accenture.com/sitemap.xml'),('q326','https://newsroom.accenture.com/content/3qfy26-earnings/accenture-reports-third-quarter-fiscal-2026-results.pdf'),('q226','https://newsroom.accenture.com/content/2qfy26-earnings/accenture-reports-second-quarter-fiscal-2026-results.pdf'),('q425','https://newsroom.accenture.com/content/4q-full-fy25-earnings/accenture-reports-fourth-quarter-and-full-year-fiscal-2025-results.pdf')]:
  try:b,p,m=fetch('ACN',u,name);print('ACN',name,len(b),flush=True)
  except Exception as e:print('ACN ERROR',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda f:f(),[csco,acn]))
