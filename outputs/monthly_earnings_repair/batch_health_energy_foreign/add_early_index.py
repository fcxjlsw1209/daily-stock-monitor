import collect as c,re

def add(t,srcname,pattern,label):
 src=c.RAW/srcname;d=c.load(src,{})
 responses=[d['response']] if 'response' in d else [r['value'] for r in d['results'] if r['status']=='fulfilled']
 a=c.load(c.HERE/(t+'_candidates.json'),[])
 for response in responses:
  for chunk in re.split(r'-{50,}',response):
   m=re.search(r'\((https://[^\s)]+)\)',chunk[:900])
   if not m or not re.search(pattern,m[1]):continue
   u=m[1];p=c.RAW/t/('index_early_'+c.sha((u+srcname+chunk).encode())[:20]+'.json')
   c.dump(p,dict(url=u,retrieved_at=d['retrieved_at'],source_representation='public_search_index_not_original_bytes',original_bytes_available=False,indexed_source_text=chunk,upstream_captures=[dict(path=c.rel(src),sha256=c.sha(src.read_bytes()))]))
   a.append(dict(ticker=t,url=u,label=label,evidence_file=c.rel(p),indexed_original=True))
 c.dump(c.HERE/(t+'_candidates.json'),list({x.get('evidence_file',x['url']):x for x in a}.values()))
if __name__=='__main__':
 for src in ['ge_early_open.json','ge_early_sec.json']:add('GE',src,r'/data/40545/.*(?:exhibit99|Exhibit99|ex99|e991)','Original GE earnings release')
 add('NVO','nvo_sec_open1.json',r'/353278/','Original Novo earnings release')
