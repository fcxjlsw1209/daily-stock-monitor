import collect as c,re
from concurrent.futures import ThreadPoolExecutor

def run(t):
 arr=c.load(c.HERE/(t+'_candidates.json'),[]);urls=[]
 if t=='NVS':
  for y in range(2019,2010,-1):
   for q in range(1,5):urls.append((f'https://www.novartis.com/sites/novartis_com/files/q{q}-{y}-media-release-en.pdf',f'Q{q} {y} press release'))
 if t=='PFE':
  for y in range(2020,2010,-1):
   for q in range(1,5):urls.append((f'https://s21.q4cdn.com/317678438/files/doc_financials/'+(f'Quarterly/{y}/q{q}/' if y<2020 else f'{y}/q{q}/')+f'Q{q}-{y}-PFE-Earnings-Release.pdf',f'Q{q} {y} candidate'))
 for u,l in urls:
  try:
   p,m=c.fetch(u,t,'earnings_candidate');c.text_of(p);arr.append(dict(ticker=t,url=u,label=l))
  except Exception as e:
   if getattr(e,'code',None) in [403,429] or 'source stopped' in str(e):print(t,'STOP',str(e),flush=True);break
 c.dump(c.HERE/(t+'_candidates.json'),list({a['url']:a for a in arr}.values()));print(t,len(arr),flush=True)
with ThreadPoolExecutor(max_workers=2) as p:list(p.map(run,['NVS','PFE']))
