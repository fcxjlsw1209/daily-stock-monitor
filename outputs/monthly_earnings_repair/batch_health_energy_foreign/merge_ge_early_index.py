import collect as c,re
from collections import defaultdict
parts=defaultdict(list)
for name in ['ge_early_sec.json','ge_early_open.json','early_more.json']:
 src=c.RAW/name;d=c.load(src,{})
 for r in d.get('results',[]):
  if r['status']!='fulfilled':continue
  for chunk in re.split(r'-{50,}',r['value']):
   m=re.search(r'\((https://[^\s)]+)\)',chunk[:900])
   if not m or not '/data/40545/' in m[1]:continue
   parts[m[1]].append((chunk,src,d['retrieved_at']))
a=c.load(c.HERE/'GE_candidates.json',[])
for u,ps in parts.items():
 if not ('GE-2013-01-18-Exhibit99' in u or 'ge3q14_ex99' in u):continue
 p=c.RAW/'GE'/('index_early_merged_'+c.sha(u.encode())[:20]+'.json')
 c.dump(p,dict(url=u,retrieved_at=max(x[2] for x in ps),source_representation='public_search_index_not_original_bytes',original_bytes_available=False,indexed_source_text='\n\n'.join(x[0] for x in ps),upstream_captures=[dict(path=c.rel(src),sha256=c.sha(src.read_bytes())) for src in dict.fromkeys(x[1] for x in ps)]))
 a.append(dict(ticker='GE',url=u,label='Original GE earnings release and same-original financial table excerpts',evidence_file=c.rel(p),indexed_original=True))
c.dump(c.HERE/'GE_candidates.json',list({z.get('evidence_file',z['url']):z for z in a}.values()))
