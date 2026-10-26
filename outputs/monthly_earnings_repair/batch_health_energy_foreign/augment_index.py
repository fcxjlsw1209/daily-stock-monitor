"""Join separately cached publication-header and statement-table index views."""
import collect as c
for t,pattern in [('LLY','lly_period*.json'),('SAP','sap_supports.json')]:
 arr=c.load(c.HERE/(t+('_supports.json' if t=='SAP' else '_candidates.json')),[])
 for src in c.RAW.glob(pattern):
  d=c.load(src,{})
  for item in d.get('results',[]):
   if item['status']!='fulfilled':continue
   v=item['value'];url=v['url'];body=v['response'];parents=[dict(path=c.rel(src),sha256=c.sha(src.read_bytes()))]
   if t=='LLY':
    old=c.ROOT/v['evidence_file'];body=c.load(old,{})['indexed_source_text']+'\n\nStatement table view of same original URL:\n'+body;parents.append(dict(path=c.rel(old),sha256=c.sha(old.read_bytes())))
   p=c.RAW/t/('index_augmented_'+c.sha((url+src.name).encode())[:20]+'.json');c.dump(p,dict(url=url,retrieved_at=d['retrieved_at'],source_representation='public_search_index_not_original_bytes',original_bytes_available=False,indexed_source_text=body,upstream_captures=parents))
   arr.append(dict(ticker=t,url=url,label=f"Q{v['fiscal_quarter']} {v['fiscal_year']} official original indexed tables",evidence_file=c.rel(p),indexed_original=True))
 c.dump(c.HERE/(t+('_supports.json' if t=='SAP' else '_candidates.json')),list({a.get('evidence_file',a['url']):a for a in arr}.values()))
p=c.HERE/'build.py';s=p.read_text().replace("('MRK','NVS','ORCL','SHEL')","('MRK','NVS','ORCL','SHEL','SAP')").replace("('MRK','ORCL','SHEL')","('MRK','ORCL','SHEL','SAP')").replace("sp=c.RAW/t/(c.sha(support['url'].encode())[:20]+'.bin')","sp=c.ROOT/support['evidence_file'] if support.get('evidence_file') else c.RAW/t/(c.sha(support['url'].encode())[:20]+'.bin')")
p.write_text(s)
