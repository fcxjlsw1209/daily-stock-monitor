import collect as c,re
arr=c.load(c.HERE/'GE_candidates.json',[])
existing={(e['fiscal_year'],e['fiscal_quarter']) for e in c.load(c.HERE/'events.json',[]) if e['ticker']=='GE'}
for y in range(2011,2022):
 for q,ord in [(1,'1st'),(2,'2nd'),(3,'3rd'),(4,'4th')]:
  if (y,q) in existing:continue
  u=f'https://www.ge.com/investor-relations/events-reports/ge-{ord}-quarter-{y}-earnings-webcast'
  try:
   p,m=c.fetch(u,'GE','official historical results event index')
   for a in c.links(p,u):
    if 'Press Release' in a['text'] and '.pdf' in a['url']:
     try:c.fetch(a['url'],'GE','original quarterly press release');arr.append(dict(ticker='GE',url=a['url'],label=f'Q{q} {y} official press release'));print(y,q,a['url'],flush=True)
     except Exception as e:print(e,flush=True)
  except Exception as e:print(y,q,e,flush=True)
  c.dump(c.HERE/'GE_candidates.json',list({a.get('evidence_file',a['url']):a for a in arr}.values()))
