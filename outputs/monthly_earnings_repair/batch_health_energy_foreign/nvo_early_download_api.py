import collect as c,re
m=c.load(c.HERE/'NVO_early_archive_api.json',{});d=c.load(c.ROOT/m['path'],{});a=c.load(c.HERE/'NVO_candidates.json',[])
for x in d['data']['resultBeanList']:
 if 'General Meeting' in x['title']:continue
 u='https://www.novonordisk.com/news-and-media/news-and-ir-materials/news-details.html?id='+str(x['id'])
 try:
  p,meta=c.fetch(u,'NVO','Official original historical release detail');a.append(dict(ticker='NVO',url=u,label=x['title']))
  links=re.findall(r'https?://hugin.info/2013/R/\d+/\d+\.pdf',p.read_text())
  links+=re.findall(r'https://ml-eu.globenewswire.com/[Rr]esource/[Dd]ownload/[a-f0-9-]+',p.read_text())
  print(x['date'],x['id'],links,flush=True)
  for v in dict.fromkeys(links):
   v=v.replace('http:','https:')
   try:
    pdf,pm=c.fetch(v,'NVO','Historical official release attachment linked from issuer original page');a.append(dict(ticker='NVO',url=v,label=x['title'],archive_path=c.rel(p),archive_url=u));print(pm['bytes'],v,flush=True)
   except Exception as ex:print(ex,flush=True)
 except Exception as ex:print(ex,flush=True)
 c.dump(c.HERE/'NVO_candidates.json',list({z.get('evidence_file',z['url']):z for z in a}.values()))
