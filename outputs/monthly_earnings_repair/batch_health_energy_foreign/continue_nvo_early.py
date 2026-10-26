import collect as c,re
src=c.RAW/'nvo_early2.json';d=c.load(src,{})
s='\n'.join(r['value'] for r in d['results'] if r['status']=='fulfilled')
urls=list(dict.fromkeys(re.findall(r'\((https://(?:www.novonordisk.com|www.globenewswire.com)/[^)]+)\)',s)))
a=c.load(c.HERE/'NVO_candidates.json',[])
for u in urls:
 if 'annual_report' in u or u.endswith('financial-results.html'):continue
 try:
  p,m=c.fetch(u,'NVO','original early quarterly release');a.append(dict(ticker='NVO',url=u,label='Original Novo financial report'));print(m['bytes'],u,flush=True)
  links=[v['url'] for v in c.links(p,u)] if not p.read_bytes().startswith(b'%PDF') else []
  links+=re.findall(r'https://ml-eu.globenewswire.com/Resource/Download/[a-f0-9-]+',p.read_text(errors='ignore')) if not p.read_bytes().startswith(b'%PDF') else []
  for v in dict.fromkeys(links):
   if ('.pdf' in v or 'Resource/Download' in v) and not any(x in v for x in ['cookie','privacy']):
    try:c.fetch(v,'NVO','original release attachment');a.append(dict(ticker='NVO',url=v,label='Original Novo financial report attachment'));print(v,flush=True)
    except Exception as e:print(e,flush=True)
 except Exception as e:print(e,flush=True)
 c.dump(c.HERE/'NVO_candidates.json',list({x.get('evidence_file',x['url']):x for x in a}.values()))
