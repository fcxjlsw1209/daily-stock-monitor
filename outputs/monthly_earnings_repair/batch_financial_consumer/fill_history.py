import sys,json,re
import core as c
from archive_download import links
from expand_download import saveq
# Historical Thermo Fisher pagination, linked directly in its archive.
ls=[]
for off in range(700,1300,100):
 u='https://india.newsroom.thermofisher.com/press-releases?l=100&o='+str(off)
 try:ls+=links(u,'TMO','history_'+str(off))
 except Exception as e:print('TMO',off,e,flush=True);break
saveq('TMO',[a for a in ls if re.search(r'Thermo Fisher Scientific Reports .*Quarter',a['text'],re.I)])
# Include annual releases whose titles do not explicitly mention fourth quarter.
ls=[]
for p in (c.HERE/'PM/raw').glob('news_page_*.html'):
 j=json.loads(p.read_text());m=json.loads(p.with_name(p.name+'.meta.json').read_text())
 for x in j['data']:
  if c.START<=x['date'][:10]<=c.CUTOFF and re.search(r'Reports .*Results',x['title'],re.I):ls.append(dict(url='https://www.pmi.com/api/web/v1/investor-relation/press-releases/'+str(x['id']),text=x['title'],archive=c.proof(p,m['url'])))
for p in (c.HERE/'PM/raw').glob('quarter_*.html'):
 j=json.loads(p.read_text());m=json.loads(p.with_name(p.name+'.meta.json').read_text())
 for x in j['documents']:
  if re.search('Earnings Results',x['title'],re.I) and 'gcs-web.com'not in x['url']:ls.append(dict(url=x['url'],text=x['title'],archive=c.proof(p,m['url'])))
saveq('PM',ls)
