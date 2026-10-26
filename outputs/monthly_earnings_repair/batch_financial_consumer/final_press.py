import re,json
import core as c
from build_events import extract
TICKER_PATH={'WFC':'72971','PM':'1413329'}
def run(s):
 out=[];pending=[]
 for p in (c.HERE/s).glob('indexed*.json'):
  if p.name.endswith('.meta.json'):continue
  if not any(z in p.name for z in ['table201','period2012q2','last2014q3']):continue
  v=json.loads(p.read_text())
  for block in v['response'].split('--------------------------------------------------------------------------------'):
   m=re.match(r'^\s*(.*?) \((https://[^\s]+)\)',block)
   if not m:continue
   url=m[2]
   if '/Archives/edgar/data/'+TICKER_PATH[s]+'/'not in url or not re.search(r'dex991\.htm$',url):continue
   a=dict(url=url,evidence_path=c.rel(p),evidence_sha256=c.sha(p.read_bytes()),index_block=block,index_query=v['queries'])
   try:out.append(extract(a,s))
   except Exception as e:pending.append(dict(url=url,evidence_path=c.rel(p),error=str(e)))
 unique={}
 for e in out:
  k=e['fiscal_year'],e['fiscal_quarter']
  if k in unique:assert (e['release_date'],e['fiscal_period_end_actual'])==(unique[k]['release_date'],unique[k]['fiscal_period_end_actual'])
  else:unique[k]=e
 c.dump(c.HERE/s/'final_press_events.json',list(unique.values()));c.dump(c.HERE/s/'FINAL_PRESS_PENDING.json',pending);print(s,[(e['release_date'],e['fiscal_period_end_actual'],e['fiscal_year'],e['fiscal_quarter'])for e in unique.values()]);print('PENDING',pending)
if __name__=='__main__':
 for s in ['WFC','PM']:run(s)
