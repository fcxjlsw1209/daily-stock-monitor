import collect as c,re
p=c.HERE/'build.py';s=p.read_text()
s=s.replace("# Repair only observed PDF", "if t=='NVS' and not p.read_bytes().startswith(b'%PDF'):\n  article=c.BeautifulSoup(p.read_bytes(),'html.parser').find('article')\n  if article:tx=c.clean(article.get_text(' ',strip=True))\n # Repair only observed PDF")
s=s.replace("if t=='TM':\n  m=re.search", "if t=='XOM':\n  m=re.search(r'ExxonMobil Earns.{0,60}?in (20\\d{2}).{0,80}?Fourth Quarter',head,re.I)\n  if m:return int(m[1]),4,m[0]\n if t=='TM':\n  m=re.search",1)
s=s.replace("if t=='PFE':\n  for m", "if t=='PFE':\n  for m in re.finditer(r'The financial statements present the three(?: and (?:six|nine|twelve))? months ended\\s+('+DP+r').{0,90}?Subsidiaries operating outside the U.S.',tx,re.I):\n   d=date(m[1])\n   if int(d[:4])==y and abs(int(d[5:7])-q*3)<=1:return d,proof(p,task['url'],m[0],'Original financial-statement note states consolidated current reporting end and separately distinguishes overseas reporting lag')\n  for m")
p.write_text(s)
for t in ['GE','UNH','NVO']:
 arr=c.load(c.HERE/(t+'_candidates.json'),[]);response=c.load(c.RAW/'priority_discovery.json',{})['response'];domain={'GE':'www.ge.com','UNH':'www.unitedhealthgroup.com','NVO':'www.novonordisk.com'}[t]
 urls=list(dict.fromkeys(re.findall(r'\((https://'+re.escape(domain)+r'/[^)]+)\)',response)))
 if t=='NVO':urls += [a['url'] for a in arr if any(i in a['url'] for i in ['112995','143008'])]
 for u in urls:
  try:
   pp,m=c.fetch(u,t,'priority original discovery');arr.append(dict(ticker=t,url=u,label='Official earnings original; body validation required'))
   if t=='NVO':
    for pdf in set(re.findall(r'https://ml-eu.globenewswire.com/Resource/Download/[a-f0-9-]+',pp.read_text(errors='replace'))):
     try:c.fetch(pdf,t,'linked issuer release attachment');arr.append(dict(ticker=t,url=pdf,label='Official issuer release attachment'))
     except Exception as e:print(e,flush=True)
   print(t,u,m['bytes'],flush=True)
  except Exception as e:print(t,e,flush=True)
 c.dump(c.HERE/(t+'_candidates.json'),list({a.get('evidence_file',a['url']):a for a in arr}.values()))
