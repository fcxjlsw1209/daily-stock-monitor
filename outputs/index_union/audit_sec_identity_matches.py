"""Review queue from primary SEC issuer tickers, retaining alias/epoch limits."""
from pathlib import Path
from collections import defaultdict
import json,hashlib
P=Path(__file__).resolve().parent

def main():
 bysymbol=defaultdict(list)
 files=list((P.parent/'dynamic_top100').rglob('*submissions*.json'))+list((P/'sources/sec_identity').glob('*_submissions.json'))
 for f in files:
  try:d=json.loads(f.read_text())
  except (ValueError,OSError):continue
  if not isinstance(d,dict):continue
  for s in d.get('tickers',[]):
   bysymbol[s.replace('-','.')].append(dict(cik=str(d['cik']).zfill(10),name=d['name'],source_file=str(f.relative_to(P.parent)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),former_names=d.get('formerNames',[])))
 rows=[]
 for r in json.loads((P/'input_diagnostics/MANIFEST.json').read_text())['price_sources']:
  s=r['symbol'];matches=bysymbol.get(s,[])
  rows.append(dict(symbol=s,primary_current_matches=matches,unique_current_issuer=len({x['cik'] for x in matches})==1,historical_security_continuity_verified=False,price_source=r))
 doc=dict(records=rows,unique_current_issuer_count=sum(r['unique_current_issuer'] for r in rows),missing_current_ticker=[r['symbol'] for r in rows if not r['primary_current_matches']],warning='SEC current ticker is identity corroboration, not historical sector or proof of equity continuity. Explicit dated aliases/epochs are still required.')
 (P/'SEC_IDENTITY_REVIEW.json').write_text(json.dumps(doc,indent=2)+'\n');print({k:v for k,v in doc.items() if k not in ['records','warning']})
if __name__=='__main__':main()
