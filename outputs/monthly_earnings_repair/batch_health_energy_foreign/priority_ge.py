import collect as c,re
arr=c.load(c.HERE/'GE_candidates.json',[])
for y in [2015,2016,2017]:
 for q,ord in [(1,'1st'),(2,'2nd'),(3,'3rd'),(4,'4th')]:
  u=f'https://www.ge.com/investor-relations/events-reports/ge-{ord}-quarter-{y}-earnings-webcast'
  try:
   p,m=c.fetch(u,'GE','official historical results event index')
   for a in c.links(p,u):
    if 'Press Release' in a['text'] and '.pdf' in a['url']:
     try:c.fetch(a['url'],'GE','original quarterly press release');arr.append(dict(ticker='GE',url=a['url'],label=f'Q{q} {y} official press release'));print(y,q,a['url'],flush=True)
     except Exception as e:print(e,flush=True)
  except Exception as e:print(y,q,e,flush=True)
c.dump(c.HERE/'GE_candidates.json',list({a['url']:a for a in arr}.values()))
p=c.HERE/'build.py';s=p.read_text().replace("if t=='XOM':\n  m=re.search", "if t=='GE':\n  m=re.search(r'GE\\s*([1-4])Q\\s*(20\\d{2})\\s*EARNINGS',head,re.I)\n  if m:return int(m[2]),int(m[1]),m[0]\n if t=='XOM':\n  m=re.search",1)
s=s.replace("'GE':[r'(?:BOSTON", "'GE':[r'GE\\s*[1-4]Q\\s*20\\d{2}\\s*EARNINGS\\s*('+MDY+')',r'(?:BOSTON")
p.write_text(s)
