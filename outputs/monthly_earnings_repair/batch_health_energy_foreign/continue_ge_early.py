import collect as c,re
src=c.RAW/'early_ge_nvo1.json';d=c.load(src,{})
s='\n'.join(r['value'] for r in d['results'][:2] if r['status']=='fulfilled')
urls=list(dict.fromkeys(re.findall(r'\((https://www.ge.com/[^)]+)\)',s)))
a=c.load(c.HERE/'GE_candidates.json',[])
for u in urls:
 if not ('webcast_press' in u or '/news/press-releases/' in u):continue
 if any(x in u for x in ['annual-report','1q-10','2q-10']):continue
 try:
  p,m=c.fetch(u,'GE','official original early quarterly release');a.append(dict(ticker='GE',url=u,label='Original historical GE earnings release'));print(m['bytes'],u,flush=True)
  for v in c.links(p,u) if not p.read_bytes().startswith(b'%PDF') else []:
   if '.pdf' in v['url'] and re.search('release|earnings|financial',v['text']+' '+v['url'],re.I):
    try:c.fetch(v['url'],'GE','original release attachment');a.append(dict(ticker='GE',url=v['url'],label='Original historical GE earnings release attachment'));print(v['url'],flush=True)
    except Exception as e:print(e,flush=True)
 except Exception as e:print(e,flush=True)
 c.dump(c.HERE/'GE_candidates.json',list({x.get('evidence_file',x['url']):x for x in a}.values()))
