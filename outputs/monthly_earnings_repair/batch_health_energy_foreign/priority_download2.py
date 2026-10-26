import collect as c,re
s='\n'.join(x['value'] for x in c.load(c.RAW/'priority_search2.json',{})['results'] if x['status']=='fulfilled')
urls=list(dict.fromkeys(re.findall(r'\((https://[^)]+)\)',s)))
for t,fragment in [('GE','ge.com/'),('UNH','unitedhealthgroup.com/'),('NVO','novonordisk.com/'),('PFE','q4cdn.com/795948973/')]:
 arr=c.load(c.HERE/(t+'_candidates.json'),[])
 for u in urls:
  if fragment not in u:continue
  try:
   p,m=c.fetch(u,t,'priority original release');arr.append(dict(ticker=t,url=u,label='Original issuer earnings candidate'));print(t,m['bytes'],u,flush=True)
   if t=='GE':
    for a in c.links(p,u):
     if re.search('press.release|financial.results|earnings.release',a['text']+' '+a['url'],re.I) and '.pdf' in a['url']:
      try:c.fetch(a['url'],t,'linked earnings PDF');arr.append(dict(ticker=t,url=a['url'],label='Original earnings release attachment'))
      except Exception as e:print(e,flush=True)
   if t=='NVO':
    for pdf in set(re.findall(r'https://ml-eu.globenewswire.com/Resource/Download/[a-f0-9-]+',p.read_text(errors='replace'))):
     try:c.fetch(pdf,t,'linked issuer release attachment');arr.append(dict(ticker=t,url=pdf,label='Original earnings release attachment'))
     except Exception as e:print(e,flush=True)
  except Exception as e:print(t,e,flush=True)
 c.dump(c.HERE/(t+'_candidates.json'),list({a.get('evidence_file',a['url']):a for a in arr}.values()))
# Save independently indexed ASML original only if actual release body exists.
arr=c.load(c.HERE/'ASML_candidates.json',[]);src=c.RAW/'priority_asml_open.json';d=c.load(src,{})
for r in d['results']:
 if r['status']!='fulfilled':continue
 v=r['value'];tx=v['response']
 if 'Press release -' not in tx:continue
 p=c.RAW/'ASML'/('index_priority_'+c.sha(v['url'].encode())[:20]+'.json');c.dump(p,dict(url=v['url'],retrieved_at=d['retrieved_at'],source_representation='public_search_index_not_original_bytes',original_bytes_available=False,indexed_source_text=tx,upstream_capture_path=c.rel(src),upstream_capture_sha256=c.sha(src.read_bytes())));arr.append(dict(ticker='ASML',url=v['url'],label='Indexed official earnings original',evidence_file=c.rel(p),indexed_original=True))
c.dump(c.HERE/'ASML_candidates.json',list({a.get('evidence_file',a['url']):a for a in arr}.values()))
