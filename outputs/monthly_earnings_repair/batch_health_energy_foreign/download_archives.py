"""Fetch observed earnings links; sequential requests per issuer host."""
import collect as c
import re
import concurrent.futures
def tasks(ticker):
 out=[]
 for mp in (c.RAW/ticker).glob('*.meta.json'):
  meta=c.load(mp,{})
  if meta.get('role')!='discovery':continue
  p=mp.with_suffix('').with_suffix('.bin')
  if p.read_bytes().startswith(b'%PDF'):continue
  soup=c.BeautifulSoup(p.read_bytes(),'html.parser')
  for a in soup.find_all('a',href=True):
   u=c.urljoin(meta['url'],a['href']);label=c.clean(a.get_text(' ',strip=True));good=False
   if ticker=='UNH':good=label.lower() in ('release','earnings release') and '.pdf' in u.lower()
   if ticker=='XOM':good=('/earnings_release/' in u and '.pdf' in u.lower()) or (label=='Earnings Release' and '/press-releases/detail/' in u)
   if ticker=='NVS':good='-en.pdf' in u and ('media-release' in u or 'financial-report' in u)
   if ticker=='AZN':good='.pdf' in u.lower() and bool(re.search(r'(?:results?|financial).*announcement|results? release',label,re.I))
   if ticker=='TM':good=('summary_en.pdf' in u or ('Financial Summary' in label and '.pdf' in u))
   if ticker=='MRK':good='Earnings Announcement' in label
   if ticker=='NVO':good='company announcement' in label.lower()
   if ticker=='DHR':good=label=='Press Release' and re.search(r'Reports-.+Results',u,re.I)
   if u.startswith('http://investors.danaher.com/'):u=u.replace('http://','https://',1)
   if not good:continue
   # Archive context is retained as provenance, not blindly made into a date.
   node=a
   for parent in a.parents:
    if parent.name in ('body','html'):break
    txt=c.clean(parent.get_text(' ',strip=True))
    if len(txt)>2500:break
    node=parent
   out.append(dict(ticker=ticker,url=u,label=label,archive_context=c.clean(node.get_text(' ',strip=True)),archive_path=c.rel(p),archive_sha256=meta['sha256'],archive_url=meta['url']))
 return list({x['url']:x for x in out}.values())
def run(ticker):
 arr=tasks(ticker);c.dump(c.HERE/(ticker+'_candidates.json'),arr);done=0;errors=[]
 for task in arr:
  try:
   p,m=c.fetch(task['url'],ticker,'earnings_candidate');c.text_of(p);done+=1
  except Exception as e:
   errors.append(dict(url=task['url'],error=str(e)))
   if 'source stopped' in str(e) or getattr(e,'code',0) in (403,429):break
 c.dump(c.HERE/(ticker+'_download_status.json'),dict(candidates=len(arr),downloaded=done,errors=errors))
 print(ticker,len(arr),done,len(errors),flush=True)
if __name__=='__main__':
 tickers=c.sys.argv[1:] or ['UNH','XOM','NVS','AZN','TM','MRK','NVO']
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(tickers)) as pool:list(pool.map(run,tickers))
