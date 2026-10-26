import collect as c,re
from extend_sources import save_tasks
from concurrent.futures import ThreadPoolExecutor

def run(t):
 arr=[]
 for p in c.RAW.glob('*.json'):
  if p.name not in ['shell_sources.json','shell_more.json','alternate_discovery2.json','asml_discovery.json']:continue
  d=c.load(p,{});s=d.get('response','');s=s if isinstance(s,str) else c.json.dumps(s)
  for u in set(re.findall(r'https://[^\s)"<>]+',s)):
   if t=='SHEL' and 'www.shell.com/' in u and re.search(r'q[1-4]-20\d\d-quarterly-press-release.pdf',u):arr.append(dict(ticker=t,url=u,label='Original quarterly press release'))
   if t=='ASML' and 'ourbrand.asml.com/' in u and 'Press-Release' in u:arr.append(dict(ticker=t,url=u,label='Original quarterly press release'))
 save_tasks(t,arr)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['SHEL','ASML']))
