"""Preserve human-readable review/transcription and indexed-original evidence."""
import collect as c,re
reviews=[('700a1fc7a3411469ebb3',2024,1,'August 1, 2023','April 1, 2023','June 30, 2023'),('d59820ec7935ce13336c',2012,1,'August 2, 2011','April 1, 2011','June 30, 2011'),('baaf07efa6ee9e2de0ba',2011,4,'May 11, 2011','April 1, 2010','March 31, 2011'),('7b53cec0967329bdd561',2011,3,'February 8, 2011','April 1, 2010','December 31, 2010')]
for stem,y,q,rd,start,end in reviews:
 p=c.RAW/'TM'/(stem+'.bin');m=c.load(p.with_suffix('.meta.json'),{})
 title=f'FY{y} '+{1:'First Quarter',3:'Third Quarter',4:''}[q]
 tx=f'FINANCIAL SUMMARY {title} ({start} through {end}) English translation from the original Japanese-language document {rd} Company name: Toyota Motor Corporation'
 c.dump(p.with_suffix('.visual_review.json'),dict(source_url=m['url'],source_sha256=m['sha256'],reviewed_at=c.utc(),method='Visual reading of rendered original PDF cover and page 2; publication header above Company name, not later Filing date',transcribed_header=tx,images=[dict(evidence_path=c.rel(p.with_suffix(f'.page{i}.png')),sha256=c.sha(p.with_suffix(f'.page{i}.png').read_bytes())) for i in (1,2)]))
arr=c.load(c.HERE/'ASML_candidates.json',[])
for src in list(c.RAW.glob('asml_index_*.json'))+list(c.RAW.glob('asml_single_*.json')):
 d=c.load(src,{})
 for block in d['response'].split('--------------------------------------------------------------------------------'):
  if 'Press release -' not in block or 'has published its' not in block:continue
  m=re.search(r'\((https://[^)]+)\)',block)
  if not m:continue
  url=m[1];key=c.sha((url+src.name).encode())[:20];p=c.RAW/'ASML'/('index_'+key+'.json')
  c.dump(p,dict(url=url,retrieved_at=d['retrieved_at'],source_representation='public_search_index_not_original_bytes',indexed_source_text=block,upstream_capture_path=c.rel(src),upstream_capture_sha256=c.sha(src.read_bytes()),original_bytes_available=False))
  arr.append(dict(ticker='ASML',url=url,label='Publicly indexed official issuer earnings release; original bytes unavailable',evidence_file=c.rel(p),indexed_original=True))
c.dump(c.HERE/'ASML_candidates.json',list({a.get('evidence_file',a['url']):a for a in arr}.values()))
