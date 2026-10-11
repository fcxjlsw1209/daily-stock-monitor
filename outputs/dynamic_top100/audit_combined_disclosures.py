"""Audit all disclosure evidence together without promoting CIK hints to securities."""
from pathlib import Path
import hashlib,json
import pandas as pd
P=Path(__file__).resolve().parent

def main():
    paths=['expanded_sec/disclosure_events_unverified.csv.gz','pilot_identity/disclosure_events_unverified.csv.gz','registry_sec/disclosure_events_unverified.csv.gz','legacy_sec/disclosure_events_unverified.csv.gz']
    frames=[];hashes={}
    for rel in paths:
        f=P/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
        d=pd.read_csv(f,dtype={'cik':str});d['cik']=d.cik.str.zfill(10);d['evidence_collection']=rel;frames.append(d)
    d=pd.concat(frames,ignore_index=True);keys=['cik','taxonomy','concept','filed','reported_as_of'];groups=d.groupby(keys,dropna=False)
    conflicts=[]
    for key,g in groups:
        values=sorted(g.shares.dropna().unique().tolist())
        if g.conflict.any() or len(values)>1:
            conflicts.append(dict(zip(keys,key),values=values,collections=sorted(g.evidence_collection.unique().tolist()),existing_conflict=bool(g.conflict.any())))
    base=set(frames[0].cik)|set(frames[1].cik);new=set(d.cik)-base
    result=dict(source_observations=len(d),distinct_disclosure_keys=len(groups),issuer_ciks=int(d.cik.nunique()),additional_ciks_beyond_expanded_and_pilot=len(new),additional_ciks=sorted(new),
                conflicting_keys=len(conflicts),conflicts=conflicts,input_sha256=hashes,usable_for_backtest=False,
                note='Coverage audit only. Class units, issuance events, identity and known-at listing status remain separate admission requirements.')
    (P/'COMBINED_DISCLOSURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['conflicts','input_sha256','additional_ciks']},indent=2))
if __name__=='__main__':main()
