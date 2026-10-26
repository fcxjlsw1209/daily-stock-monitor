from common import *
import concurrent.futures
def crm():
 urls=set()
 for p in (HERE/'CRM/public_index_observations').glob('search_frozen_*.json'):
  urls.update(re.findall(r'\((https://www.salesforce.com/news/press-releases/[^\s)]+)\)',json.loads(p.read_text())['rendering']))
 for u in sorted(urls):
  try:b,p,m=fetch('CRM',u.split('?')[0]);print('CRM',len(b),flush=True)
  except Exception as e:print('CRM',str(e),flush=True)
def intel():
 rows=json.loads((HERE.parent/'SEC_REVIEW_QUEUE.json').read_text())['candidates']
 for r in rows:
  if r['symbol']!='INTC' or r['filing_date']<'2020-01-01':continue
  u='https://www.intc.com/filings-reports/all-sec-filings/content/'+r['accession']+'/'+r['document_url'].rsplit('/',1)[-1]
  try:b,p,m=fetch('INTC',u);print('INTC',len(b),flush=True)
  except Exception as e:print('INTC',str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda f:f(),[crm,intel]))
