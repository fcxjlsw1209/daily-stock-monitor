import sys,json,re,urllib.parse
from bs4 import BeautifulSoup
import core as c
from archive_download import links
from expand_download import saveq

def run(s):
 if s in ['TMO','ABT']:
  base='https://india.newsroom.thermofisher.com/press-releases?l=100' if s=='TMO' else 'https://abbott.mediaroom.com/press-releases?category=783&l=100'
  ls=[]
  for off in range(0,601,100):
   u=base+'&o='+str(off)
   try:ls+=links(u,s,'history_'+str(off))
   except Exception as e:print(s,off,e,flush=True);break
  c.dump(c.HERE/s/'history_links.json',ls)
  select=[a for a in ls if re.search(r'(Thermo Fisher Scientific|Abbott) Reports .*Quarter',a['text'],re.I)]
  return saveq(s,select)
 if s=='BAC':
  ls=[]
  for page in range(0,30):
   u='https://newsroom.bankofamerica.com/content/newsroom/press-releases.html?categTitle=Corporate+and+Financial+News&category=press-release-categories%2Fcorporate-and-financial-news&page='+str(page)+'&year=all'
   try:ls+=links(u,s,'news_page_'+str(page))
   except Exception as e:print(s,page,e,flush=True);break
  c.dump(c.HERE/s/'history_links.json',ls);return saveq(s,[a for a in ls if re.search(r'Bank of America Reports .*Quarter.*Financial Results',a['text'],re.I)])
 if s=='VZ':
  u='https://www.verizon.com/about/investors/quarterly-reports';ls=links(u,s,'quarter_index');sub=[]
  for a in ls:
   if re.search('quarterly-reports/.*earnings',a['url']):
    try:sub+=links(a['url'],s)
    except Exception as e:print(s,e,flush=True)
  c.dump(c.HERE/s/'history_links.json',sub);return saveq(s,[a for a in sub if re.search('earnings.release|press.release',a['text']+' '+a['url'],re.I) and not re.search('webcast|conference',a['url'],re.I)])
 if s in ['T','C']:
  items=[]
  if s=='T':
   for p in (c.HERE/s/'raw').glob('archive_*.html'):
    mp=p.with_name(p.name+'.meta.json');m=json.loads(mp.read_text())
    if m['status']!=200:continue
    for a in links(m['url'],s,p.stem):
     if '.pdf'in a['url'].lower() and re.search('financial.{0,8}operat|consolidated|financial.statement',a['url'],re.I):items.append(a)
  else:
   p=c.HERE/s/'raw/expanded_archive.html';u='https://www.citigroup.com/global/investors/quarterly-earnings';bs=BeautifulSoup(p.read_bytes(),'html.parser')
   for img in bs.find_all('img',alt=True):
    a=img.find_parent('a',href=True)
    if a and 'Financial Supplement PDF'in img['alt']:items.append(dict(url=urllib.parse.urljoin(u,a['href']),text=img['alt'],archive=c.proof(p,u)))
  out=[];seen=set()
  for a in items:
   a['url']=a['url'].replace(' ','%20')
   if a['url']in seen:continue
   seen.add(a['url'])
   try:
    b,p,m=c.fetch(a['url'],s);a.update(evidence_path=c.rel(p),evidence_sha256=c.sha(b),meta_path=c.rel(p.with_name(p.name+'.meta.json')));c.pdf_text(p)
   except Exception as e:a['error']=str(e)
   out.append(a);c.dump(c.HERE/s/'supplement_queue.json',out)
  print(s,'SUPPLEMENTS',len(out),flush=True)
if __name__=='__main__':
 for s in sys.argv[1:]:
  try:run(s)
  except Exception as e:print(s,'FAILED',str(e),flush=True)
