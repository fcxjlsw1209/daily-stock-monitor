"""Public-index observations are evidence of original text, not downloaded originals."""
from collect import *
def build_public(tickers=['ADBE','AVGO','PDD']):
 for k in tickers:
  events=[];pending=[]
  for p in sorted((HERE/k/'public_index_observations').glob('*.json')):
   if p.name=='archive.json' or p.name.startswith('search_'):continue
   j=json.loads(p.read_text());t=j['rendering'];m=re.search(r'\((https?://[^\s)]+)\)',t)
   if not m:pending.append(dict(evidence_path=rel(p),reason='No original URL in public-index response'));continue
   u=m.group(1)
   try:
    r=parse(k,t);e=event(k,p,dict(url=u,retrieved_at=j['observed_at']),r,'public_index_observation_of_original_issuer_release_not_full_bytes')
    e['historical_limitation']='Original issuer text observed through public search index; cached evidence is the indexed rendering, not downloaded original HTML/PDF bytes. Source clocks and CMS dates not used.'
    e['evidence_limitation']='SHA-256 identifies this saved public-index observation JSON, not the original issuer document.'
    if k=='PDD':
     e['publication_timezone']='Europe/Dublin' if r['dateline_excerpt'].startswith('DUBLIN') else 'Asia/Shanghai'
    if k=='AVGO' and r['dateline_excerpt'].startswith('SINGAPORE'):e['publication_timezone']='Asia/Singapore'
    events.append(e)
   except ValueError as err:pending.append(dict(source_url=u,evidence_path=rel(p),reason=str(err)))
  events.sort(key=lambda x:x['release_date']);keys=[(x['fiscal_year'],x['fiscal_quarter']) for x in events]
  unique={}
  for e in events:
   key=(e['fiscal_year'],e['fiscal_quarter'])
   if key in unique:
    if any(unique[key][f]!=e[f] for f in ['release_date','fiscal_period_end_actual']):raise ValueError('Conflicting original public events '+k)
    pending.append(dict(source_url=e['source_url'],evidence_path=e['evidence_path'],reason='Official archive link resolves to duplicate original quarter, not the archive-labelled quarter; retained, not double counted'));continue
   unique[key]=e
  events=list(unique.values())
  dump(HERE/k/'events.json',events);dump(HERE/k/'PENDING.json',dict(unadmitted=pending));print(k,len(events),'pending',len(pending),flush=True)
  for r in pending[:8]:print('PENDING',r['reason'],r.get('source_url',''))
if __name__=='__main__':build_public()
