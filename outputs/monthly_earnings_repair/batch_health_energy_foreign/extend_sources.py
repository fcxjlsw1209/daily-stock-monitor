"""Discover observed official archive links and cache each original."""
import collect as c
import re
from concurrent.futures import ThreadPoolExecutor

def save_tasks(t,arr):
 old=c.load(c.HERE/(t+'_candidates.json'),[]);arr=list({a['url']:a for a in old+arr}.values());c.dump(c.HERE/(t+'_candidates.json'),arr)
 for a in arr:
  try:p,m=c.fetch(a['url'],t,'earnings_candidate');c.text_of(p)
  except Exception as e:
   print(t,a['url'],str(e),flush=True)
   if getattr(e,'code',None) in (403,429):break
 print(t,'candidates',len(arr),flush=True)

def abbv():
 t='ABBV';arr=[]
 for start in (0,100,200):
  u='https://news.abbvie.com/index.php?s=search_results&query=AbbVie%20Reports&l=100'+('&o='+str(start) if start else '')
  p,m=c.fetch(u,t)
  for l in c.links(p,u):
   if re.match(r'AbbVie Reports.*(?:Quarter|Year).*Financial Results',l['text'],re.I):arr.append(dict(ticker=t,url=l['url'],label=l['text'],archive_url=u,archive_path=c.rel(p),archive_sha256=m['sha256']))
 save_tasks(t,arr)
def chl():
 t='CHL';arr=[]
 for year in range(2011,2021):
  for typ in ['media/press','ir/hkex_announcements']:
   u=f'https://www.chinamobileltd.com/en/{typ}.php?year={year}';p,m=c.fetch(u,t)
   for l in c.links(p,u):
    label=l['text']
    if (re.search(r'(?:annual|interim|quarter|[13]Q|three months|nine months).*results',label,re.I) or re.search(r'(?:three|nine) months ended',label,re.I)) and not re.search(r'202[1-6]|webcast|A Shares',label,re.I):
     arr.append(dict(ticker=t,url=l['url'],label=label,archive_url=u,archive_path=c.rel(p),archive_sha256=m['sha256']))
 save_tasks(t,arr)
def azn():
 t='AZN';arr=[]
 for mp in (c.RAW/t).glob('*.meta.json'):
  m=c.load(mp,{})
  if m.get('role')!='discovery':continue
  p=mp.with_suffix('').with_suffix('.bin')
  for l in c.links(p,m['url']):
   if '.pdf' in l['url'].lower() and re.search(r'(?:results?|financial).*(?:press release|announcement|release)',l['text'],re.I):arr.append(dict(ticker=t,url=l['url'],label=l['text'],archive_url=m['url'],archive_path=c.rel(p),archive_sha256=m['sha256']))
 save_tasks(t,arr)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in [abbv,chl,azn]:pool.submit(f)
