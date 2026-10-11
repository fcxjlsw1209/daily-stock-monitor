"""Dated sector observations from primary reports, with ambiguous names excluded."""
from pathlib import Path
from collections import defaultdict
import json,re,hashlib
from extract_rydex_sectors import clean
P=Path(__file__).resolve().parent
# Reused names must not be assigned to today's ticker. These remain unresolved.
AMBIGUOUS={'broadcom','constellation energy','constellation energy group','kraft foods','kraft foods group','news','news corporation','altera','allergan','dow','dow chemical','dupont','du pont','monster worldwide'}
def main():
 names=defaultdict(set)
 for f in ['RYDEX_2009_SECTOR_OBSERVATIONS.json','SPDR_SECTOR_CANDIDATES.json','SP_HISTORICAL_SECTOR_OBSERVATIONS.json']:
  for r in json.loads((P/f).read_text())['observations']:names[clean(r['company'])].add(r['symbol'])
 for r in json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text()):
  for n in r['names']+[x['name'] for x in r.get('name_evidence',[])]:names[clean(n)].add(r['symbol'])
 members={r['symbol'] for r in json.loads((P/'input_diagnostics/MANIFEST.json').read_text())['symbol_gaps']}
 out=[];unmapped=[];conflicts=[]
 for f in sorted((P/'sources/spdr_sectors').glob('*_holdings_extracted.json')):
  doc=json.loads(f.read_text());dated=defaultdict(list)
  for r in doc['rows']:
   name=clean(re.sub(r'\([a-z]+\)$','',r['company'],flags=re.I))
   matches=sorted(names[name]&members)
   if name in AMBIGUOUS or len(matches)!=1:
    unmapped.append(dict(report=f.name,company=r['company'],candidates=matches));continue
   dated[matches[0]].append(r)
  for s,rows in dated.items():
   if len({r['sector'] for r in rows})!=1:
    conflicts.append(dict(report=f.name,symbol=s,sectors=sorted({r['sector'] for r in rows})));continue
   r=rows[0]
   out.append(dict(symbol=s,company=r['company'],sector=r['sector'],effective_session=doc['observed_as_of'],known_at=doc['known_at'],source_file=doc['source_file'],source_sha256=doc['source_sha256'],source_url=doc['source_url'],classification_observation_verified=True,permanent_identity_verified=False,report=f.name))
 result=dict(observations=out,unmapped=unmapped,conflicts=conflicts,warning='Unique historical name proposals; ambiguous/reused company names excluded. Publication date gates use. These do not certify permanent identities.')
 (P/'SPDR_ANNUAL_SECTOR_OBSERVATIONS.json').write_text(json.dumps(result,indent=2)+'\n')
 print('observations',len(out),'symbols',len({r['symbol'] for r in out}),'conflicts',len(conflicts),'unmapped',len(unmapped))
if __name__=='__main__':main()
