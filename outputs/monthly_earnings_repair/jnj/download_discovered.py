import collect as c
import json,re
urls={}
for p in c.RAW.glob('*discovery*.json'):
 t=json.loads(p.read_text()).get('response','')
 for u in re.findall(r'https://(?:www\.)?(?:investor\.jnj\.com|s203.q4cdn.com/636242992)/files/[^\s)]+\.pdf',t):
  url=re.sub(r'https://(?:www\.)?investor\.jnj\.com',c.CDN,u)
  if '/doc_financials/' in url or re.search('Reports-|JNJ_News|Annual-Report',url,re.I): urls.setdefault(url,[]).append(c.rel(p))
manifest=[]
for url,provenance in sorted(urls.items()):
 stem='document_'+c.sha(url.encode())[:16]
 try:
  p,m=c.fetch(url,stem);text=c.pdf_text(p)
  manifest.append(dict(url=url,evidence_path=c.rel(p),sha256=c.sha(p.read_bytes()),discovery_index_paths=provenance,text_path=c.rel(p.with_suffix('.txt'))))
  print(url.rsplit('/',1)[-1],len(p.read_bytes()),flush=True)
 except Exception as e:
  print('ERROR',url,str(e),flush=True)
  if isinstance(e,RuntimeError) and 'source stopped' in str(e) or getattr(e,'code',None) in (403,429): break
c.dump(c.HERE/'DOWNLOAD_MANIFEST.json',manifest)
