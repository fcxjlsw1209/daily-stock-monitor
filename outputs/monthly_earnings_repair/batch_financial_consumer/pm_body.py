import json,re
import core as c
from expand_download import saveq
out=[]
for off in range(0,800,100):
 u=f'https://www.pmi.com/api/web/v1/investor-relation/press-releases?offset={off}&limit=100&sort=desc'
 try:b,p,m=c.fetch(u,'PM','news_page_'+str(off));j=json.loads(b)
 except Exception as e:print(off,e,flush=True);break
 for x in j['data']:
  if x['date'][:10]<'2011-01-01' or x['date'][:10]>'2026-09-24':continue
  if re.search(r'Reports .*Quarter',x['title'],re.I) and re.search('Results',x['title'],re.I):
   out.append(dict(url='https://www.pmi.com/api/web/v1/investor-relation/press-releases/'+str(x['id']),text=x['title'],archive=c.proof(p,u)))
 c.dump(c.HERE/'PM/original_body_queue_discovery.json',out)
saveq('PM',out)
