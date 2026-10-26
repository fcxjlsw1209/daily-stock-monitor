import collect as c,re
u='https://www.shell.com/investors/results-and-reporting/quarterly-results.model.json';p,m=c.fetch(u,'SHEL','official page model discovery');arr=c.load(c.HERE/'SHEL_candidates.json',[]);sup=c.load(c.HERE/'SHEL_supports.json',[])
def walk(o):
 if isinstance(o,dict):
  mo=o.get('model',{});title=mo.get('title','')
  if re.search(r'Q[1-4] 20\d{2} Results',title):
   for l in mo.get('links',[]):
    if re.search(r'(?:press release|unaudited results|results announcement)',l.get('name',''),re.I):
     url=c.urljoin(u,l['value']);a=dict(ticker='SHEL',url=url,label=title+' '+l['name'],archive_url=u,archive_path=c.rel(p),archive_sha256=m['sha256']);(sup if 'unaudited' in l['name'].lower() else arr).append(a)
  for v in o.values():walk(v)
 elif isinstance(o,list):
  for v in o:walk(v)
walk(c.load(p,{}));arr=list({a['url']:a for a in arr}.values());sup=list({a['url']:a for a in sup}.values());c.dump(c.HERE/'SHEL_candidates.json',arr);c.dump(c.HERE/'SHEL_supports.json',sup);print('SHEL',len(arr),'support',len(sup),flush=True)
for a in arr+sup:
 try:p,m=c.fetch(a['url'],'SHEL','earnings original');print(m['bytes'],a['label'],flush=True)
 except Exception as e:print(str(e),flush=True)
