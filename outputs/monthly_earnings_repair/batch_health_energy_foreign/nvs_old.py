import collect as c,re
src=c.RAW/'nvs_early_search.json';d=c.load(src,{});s='\n'.join(r['value'] for r in d['results'] if r['status']=='fulfilled');urls=list(dict.fromkeys(re.findall(r'\((https://www.novartis.com/news/media-releases/[^)]+)\)',s)));arr=c.load(c.HERE/'NVS_candidates.json',[])
print('NVS release candidates',len(urls),flush=True)
for u in urls:
 try:
  p,m=c.fetch(u,'NVS','earnings release discovery');tx=c.clean(c.text_of(p));arr.append(dict(ticker='NVS',url=u,label='Original media release'))
  links=c.links(p,u);english=[]
  for a in links:
   if ('hugin.info/134323/' in a['url'] and a['url'].endswith('.pdf')) or ('media-release' in a['url'] and a['url'].endswith('.pdf')):
    english.append(a['url'])
  # Only first attachment immediately following full media release in English;
  # other languages are retained downloads but not used as English candidates.
  for pdf in english[:1]:
   try:
    pp,mm=c.fetch(pdf,'NVS','linked original English media release');arr.append(dict(ticker='NVS',url=pdf,label='Original full media release',parent_url=u));print('PDF',mm['bytes'],pdf,flush=True)
   except Exception as e:print(e,flush=True)
 except Exception as e:print(e,flush=True)
c.dump(c.HERE/'NVS_candidates.json',list({a.get('evidence_file',a['url']):a for a in arr}.values()))
