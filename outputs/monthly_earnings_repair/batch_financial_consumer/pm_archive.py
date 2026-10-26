import json,sys
import core as c
from expand_download import saveq
index=json.loads((c.HERE/'PM/raw/quarterly_reports.html').read_text());items=[]
for x in index['data']:
 if x['year']<2010 or x['year']==2010 and x['quarter']<4:continue
 y,q=x['year'],x['quarter'];u=f'https://www.pmi.com/api/web/v1/investor-relation/quarterly-reports/{y}-{q}'
 try:
  b,p,m=c.fetch(u,'PM',f'quarter_{y}_{q}');v=json.loads(b)
  for a in v['documents']:
   if a['title']=='Press Release':items.append(dict(url=a['url'],text=a['title'],archive=c.proof(p,u)))
 except Exception as e:print(y,q,str(e),flush=True)
 c.dump(c.HERE/'PM/discovered_releases.json',items)
saveq('PM',items)
