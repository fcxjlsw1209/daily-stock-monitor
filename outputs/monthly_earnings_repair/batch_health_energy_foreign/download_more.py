import collect as c, re
from extend_sources import save_tasks
from concurrent.futures import ThreadPoolExecutor

def observed(t):
 arr=[]
 for p in c.RAW.glob(t.lower()+'_discovery_*.json'):
  s=c.load(p,{})['response']
  for u in set(re.findall(r'\((https://[^)]+)\)',s)):
   if t=='LLY' and re.search(r'prnewswire.com/news-releases/lilly-(?:reports|delivers).*(?:quarter|q[1-4]).*financial-results',u):arr.append(dict(ticker=t,url=u,label='Issuer release distributed by PRNewswire'))
   if t=='ORCL' and 'www.oracle.com' in u and (('/news/announcement/' in u and 'earnings' in u) or ('/corporate/pressrelease/q' in u)):
    # Localized copies of the same English issuer original are duplicates. Keep
    # discovery URL unaltered, and exclude Japan subsidiary investor documents.
    arr.append(dict(ticker=t,url=u,label='Oracle Corporation earnings original'))
 save_tasks(t,arr)
def extra_pdf():
 t='PFE';arr=[]
 # Observed current CDN path family; a successful body, not the guessed path,
 # must establish issuer, release date and fiscal endpoint in offline parsing.
 for y in range(2026,2014,-1):
  for q in range(1,5):
   if y==2026 and q>2:continue
   for folder in [f'{y}/q{q}',f'Quarterly/{y}/q{q}']:
    success=False
    for suffix in ['-FINAL','']:
     u=f'https://s206.q4cdn.com/795948973/files/doc_financials/{folder}/Q{q}-{y}-PFE-Earnings-Release{suffix}.pdf'
     try:
      p,m=c.fetch(u,t,'earnings_candidate');c.text_of(p);arr.append(dict(ticker=t,url=u,label=f'Q{q} {y} original candidate'));success=True;break
     except Exception as e:
      if getattr(e,'code',None) in (403,429) or 'source stopped' in str(e):c.dump(c.HERE/'PFE_candidates.json',list({a['url']:a for a in c.load(c.HERE/'PFE_candidates.json',[])+arr}.values()));print(t,'STOP',str(e),flush=True);return
    if success:break
 c.dump(c.HERE/'PFE_candidates.json',list({a['url']:a for a in c.load(c.HERE/'PFE_candidates.json',[])+arr}.values()));print(t,len(arr),flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:
  fut=[pool.submit(observed,t) for t in ['LLY','ORCL']]+[pool.submit(extra_pdf)]
  for f in fut:
   try:f.result()
   except Exception as e:print(type(e).__name__,str(e),flush=True)
