"""Build raw-unit execution candidates from existing Yahoo caches, offline.

No candidate is promoted to historical identity/action coverage certification.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
from price_units import split_adjusted_to_raw
P=Path(__file__).resolve().parent


def main():
    inventory=json.loads((P/'NEW_PRICE_VALIDATION.json').read_text())['files']
    out=P/'execution_candidates';out.mkdir(exist_ok=True)
    results=[]
    for row in inventory:
        if row['provider']!='yahoo':continue
        source=P/row['normalized_file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['normalized_sha256']:
            raise ValueError('Normalized input checksum mismatch: '+row['symbol'])
        data=pd.read_csv(source,index_col='date',parse_dates=True)
        raw=split_adjusted_to_raw(data)
        dest=out/(row['symbol']+'_raw.csv.gz')
        raw.to_csv(dest,compression={'method':'gzip','mtime':0})
        results.append(dict(symbol=row['symbol'],rows=len(raw),split_events=int(raw.split.ne(1).sum()),
          input_file=row['normalized_file'],input_sha256=row['normalized_sha256'],
          file=str(dest.relative_to(P)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
          identity_verified=False,action_completeness_verified=False))
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),files=results,
      file_count=len(results),rows=sum(r['rows'] for r in results),
      interpretation='Raw-unit execution candidates; adj_close stays in source total-return signal units.',
      limitations=['All future splits through the vendor adjustment anchor must be present.',
                   'Retains existing row quarantine and identity warnings; this is unit conversion only.',
                   'Cash dividends converted to raw per-share units; spin-off/merger entitlements require separate evidence.'])
    (P/'EXECUTION_CANDIDATES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
