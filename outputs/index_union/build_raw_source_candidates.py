"""Integrate documented Tiingo/WIKI raw units, preserving all quarantines.

Does not concatenate providers, certify identities or write paper accounts.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
from price_units import raw_to_split_adjusted
P=Path(__file__).resolve().parent

def main():
    results=[];out=P/'raw_source_candidates';out.mkdir(exist_ok=True)
    for r in json.loads((P/'NEW_PRICE_VALIDATION.json').read_text())['files']:
        if r['provider'] not in ['tiingo','wiki_union_archive']:continue
        source=P/r['normalized_file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=r['normalized_sha256']:
            raise ValueError('Source changed: '+r['symbol'])
        d=pd.read_csv(source,index_col='date',parse_dates=True)
        if r['provider']=='wiki_union_archive':d=d.rename(columns={'ex-dividend':'dividend','split_ratio':'split'})
        signal,raw=raw_to_split_adjusted(d)
        row=dict(symbol=r['symbol'],provider=r['provider'],rows=len(d),
             input_file=r['normalized_file'],input_sha256=r['normalized_sha256'],
             identity_verified=False,action_completeness_verified=False)
        for kind,frame in [('signal',signal),('raw',raw)]:
            dest=out/(r['symbol']+'_'+r['provider']+'_'+kind+'.csv.gz')
            frame.to_csv(dest,compression={'method':'gzip','mtime':0})
            row[kind+'_file']=str(dest.relative_to(P));row[kind+'_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        results.append(row)
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),files=results,
        file_count=len(results),rows=sum(r['rows'] for r in results),admission=False,
        limitations=['Uses explicit raw share events; no adj_close ratio treated as stock split.',
          'Provider anchor is local to each file; no cross-provider splice or retroactive ticker merge.',
          'Preserves all row quarantine and identity warnings.'])
    (P/'RAW_SOURCE_CANDIDATES.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))
if __name__=='__main__':main()
