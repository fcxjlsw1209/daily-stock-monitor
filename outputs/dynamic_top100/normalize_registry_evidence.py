"""Separate registry candidates from established issuer joins and existing accounts."""
import hashlib,json
from pathlib import Path
import pandas as pd
from normalize_expanded_evidence import normalize,TAGS
P=Path(__file__).resolve().parent

def main():
    progress=json.loads((P/'registry_sec/progress.json').read_text());rows=[]
    for r in progress:
        if not r.get('facts_file'):continue
        f=P/r['facts_file'];j=json.loads(f.read_text())
        if str(j['cik']).zfill(10)!=r['cik']:raise ValueError('CIK mismatch')
        for tax,concepts in j.get('facts',{}).items():
            for tag,item in concepts.items():
                if tag not in TAGS:continue
                rows.extend(dict(v,cik=r['cik'],taxonomy=tax,concept=tag,source_file=r['facts_file']) for v in item.get('units',{}).get('shares',[]))
    out=normalize(pd.DataFrame(rows))
    old=pd.read_csv(P/'expanded_sec/disclosure_events_unverified.csv.gz',dtype={'cik':str})
    old_ids=set(old.cik.str.zfill(10));new_ids=set(out.cik)-old_ids
    out.to_csv(P/'registry_sec/disclosure_events_unverified.csv.gz',index=False,compression='gzip')
    stats=dict(candidates_processed=len(progress),raw_share_rows=len(rows),normalized_events=len(out),issuers_with_share_observations=int(out.cik.nunique()),
               additional_ciks_with_share_observations=len(new_ids),additional_ciks=sorted(new_ids),conflicts=int(out.conflict.sum()),
               pre_2010_share_issuers=int(out.loc[out.filed<'2010-01-01','cik'].nunique()),identity_verified=False,usable_for_backtest=False,
               note='New share evidence is separate. CIK/name hints are not permanent security identities or historical US listing proof.',
               source_progress_sha256=hashlib.sha256((P/'registry_sec/progress.json').read_bytes()).hexdigest())
    (P/'REGISTRY_SHARE_AUDIT.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
