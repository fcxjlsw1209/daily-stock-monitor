"""SEC-derived name matches for discovery; never certify listing or market cap."""
import csv, gzip, hashlib, json, re
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent

def name_key(name):
    # Do not strip industry terms or collapse issuer/successor names.
    s=name.casefold().replace('&',' and ')
    s=re.sub(r'\b(incorporated|inc|corporation|corp|limited|ltd|plc|company|co)\b',' ',s)
    return re.sub(r'[^a-z0-9]','',s)

def candidate_matches(episode, lookup):
    matches={}
    for name in episode['names']:
        key=name_key(name)
        if not key: continue
        for record in lookup.get(key,[]):
            item=dict(record,matched_vendor_name=name)
            matches[(record['cik'],record['name'],record.get('start_date'),record.get('end_date'))]=item
    return sorted(matches.values(), key=lambda x:(x['cik'],x['name'],x.get('start_date','')))

def main():
    folder=P/'additional_sources/sec_filer_registry'; lookup=defaultdict(list); hashes={}; count=0
    for f in sorted(folder.glob('*_names.csv.gz')):
        m=json.loads(f.with_suffix('.provenance.json').read_text())
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=m['sha256']:raise ValueError('Registry checksum mismatch: '+f.name)
        hashes[f.name]=digest
        with gzip.open(f,'rt',newline='') as h:
            for r in csv.DictReader(h):
                if not r['cik'].isdigit():raise ValueError('Invalid CIK')
                r['cik']=r['cik'].zfill(10);r['source_file']=f.name
                lookup[name_key(r['name'])].append(r);count+=1
    episodes=json.loads((P/'listing_episodes_unverified.json').read_text());out=[]
    for e in episodes:
        hits=candidate_matches(e,lookup);ids=sorted({r['cik'] for r in hits})
        out.append(dict(evidence_id=e['evidence_id'],symbol=e['symbol'],names=e['names'],
                        candidate_ciks=ids,name_evidence=hits,previous_possible_ciks=e['possible_ciks'],
                        unique_name_match=len(ids)==1,identity_verified=False,usable_for_rank=False))
    (P/'registry_identity_hints.json').write_text(json.dumps(out,indent=2)+'\n')
    summary=dict(registry_name_rows=count,episodes=len(out),with_name_candidates=sum(bool(x['candidate_ciks']) for x in out),
                 unique_name_matches=sum(x['unique_name_match'] for x in out),ambiguous_name_matches=sum(len(x['candidate_ciks'])>1 for x in out),
                 new_unique_hints=sum(x['unique_name_match'] and not x['previous_possible_ciks'] for x in out),
                 input_sha256=hashes,identity_certified=0,complete_eligible_coverage=False,
                 caveats=['SEC registration is not US exchange listing.','Name interval dates are SEC filer metadata, not exchange admission dates.',
                          'Names and current metadata are discovery aids; match original historical filings before joining shares or prices.'])
    (P/'REGISTRY_IDENTITY_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
