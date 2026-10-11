"""Compare reviewed evidence across collections; raw archives remain unchanged."""
from pathlib import Path
import json, hashlib, datetime
import pandas as pd
from resolve_share_amendments import reviewed_events
from reconcile_share_evidence import reconcile
P=Path(__file__).resolve().parent
KEY=['cik','taxonomy','concept','filed','reported_as_of']

def main():
    fixes=json.loads((P/'pilot_identity/share_corrections.json').read_text())
    base,_=reviewed_events()
    base=reconcile(base,fixes);base['evidence_collection']='reviewed_expanded'
    frames=[base];hashes={}
    for rel in ['pilot_identity/disclosure_events_unverified.csv.gz','registry_sec/disclosure_events_unverified.csv.gz','legacy_sec/disclosure_events_unverified.csv.gz']:
        f=P/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
        d=pd.read_csv(f,dtype={'cik':str});d.cik=d.cik.str.zfill(10)
        d=reconcile(d,fixes);d['evidence_collection']=rel;frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    conflicts=[]
    for key,g in d.groupby(KEY,dropna=False):
        values=sorted(g.shares.dropna().unique().tolist())
        if g.conflict.any() or len(values)>1:
            conflicts.append(dict(zip(KEY,key),values=values,collections=sorted(g.evidence_collection.unique())))
    d.to_csv(P/'reviewed_disclosures_UNVERIFIED.csv.gz',index=False)
    out=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_rows=len(d),distinct_keys=len(d.groupby(KEY)),issuer_ciks=int(d.cik.nunique()),conflicting_keys=len(conflicts),conflicts=conflicts,remaining_scale_keys=len(d.loc[d.scale_review_required,KEY].drop_duplicates()),input_sha256=hashes,usable_for_backtest=False,raw_preserved=True)
    (P/'REVIEWED_DISCLOSURE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['conflicts','input_sha256']},indent=2))
if __name__=='__main__':main()
