"""Cross-check a public history table as discovery, never historical sector input.

Upstream _merge_stocks overwrites nonempty fields with the latest values while
retaining created_at. That date must NOT date the sector or company name.
"""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent

def main():
    source=P/'sources/lawcal_sp500/data/components_history.csv'
    with source.open() as f: rows=list(csv.DictReader(f))
    wanted={r['symbol'] for r in json.loads((P/'UNION_PRICE_DISCOVERY_AUDIT.json').read_text())['rows']}
    hints={r['symbol']:r for r in json.loads((P.parent/'dynamic_top100/registry_identity_hints.json').read_text())}
    matches=[]
    for r in rows:
        if r['symbol'] not in wanted:continue
        def cik(x):
            try:return str(int(x)).zfill(10)
            except (TypeError,ValueError):return None
        old={cik(x) for x in hints.get(r['symbol'],{}).get('candidate_ciks',[])}
        old.discard(None)
        matches.append(dict(symbol=r['symbol'],candidate_cik=cik(r['cik']),upstream_name=r['name'],reported_added=r['date_added'],reported_removed=r['date_removed'],registry_candidate_ciks=sorted(old),cik_agrees_with_registry=cik(r['cik']) in old,identity_verified=False,sector_usable_as_historical=False))
    result=dict(source_file=str(source.relative_to(P)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),total_source_rows=len(rows),union_candidate_rows=len(matches),union_candidate_symbols=len({r['symbol'] for r in matches}),agreement_rows=sum(r['cik_agrees_with_registry'] for r in matches),records=matches,admission=False,warning='Discovery only. Latest sector/name overwrites historical values. created_at is first symbol observation, not field knowledge time. CIK equality does not prove same equity through reorganizations.')
    (P/'FREE_IDENTITY_SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
