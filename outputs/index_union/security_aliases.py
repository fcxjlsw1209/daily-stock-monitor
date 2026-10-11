"""Date-restricted research IDs for reviewed same-equity ticker changes.

Mergers, spin-offs, tracking stocks and share-class conversions are not aliases.
This resolves symbol identity only, never certifies provider prices or sectors.
"""
from pathlib import Path
from datetime import datetime,time
from zoneinfo import ZoneInfo
import hashlib,json
P=Path(__file__).resolve().parent
ALLOWED={'ticker_change','ticker_change_same_security','ticker_exchange_change','name_ticker_change'}


def compile_aliases(documents):
    edges=[]
    for filename,document,sha in documents:
        for e in document['aliases']:
            # HANS/MNST is specifically reviewed continuation despite CUSIP/name change.
            if e['kind'] not in ALLOWED and (e['old'],e['new'])!=('HANS','MNST'):continue
            if e['published']>e['effective']:raise ValueError('Rename not known by effective date')
            edges.append(dict(e,evidence_file=filename,evidence_sha256=sha))
    by_old={e['old']:e for e in edges};by_new={e['new']:e for e in edges}
    if len(by_old)!=len(edges) or len(by_new)!=len(edges):raise ValueError('Ambiguous alias branch')
    roots=sorted(set(by_old)-set(by_new));seen=set();rows=[]
    for root in roots:
        symbol=root;start='2008-01-01';chain=[]
        while symbol in by_old:
            e=by_old[symbol]
            if symbol in seen or e['effective']<=start:raise ValueError('Cyclic or reversed aliases')
            seen.add(symbol);chain.append(e);symbol=e['new'];start=e['effective']
        symbols=[root]+[e['new'] for e in chain]
        starts=['2008-01-01']+[e['effective'] for e in chain]
        ends=[e['effective'] for e in chain]+['9999-12-31']
        for i,symbol in enumerate(symbols):
            # First old symbol is confirmed retrospectively by this rename proof;
            # do not assert its historical public known_at earlier than the evidence.
            e=chain[max(0,i-1)]
            rows.append(dict(symbol=symbol,effective_from=starts[i],effective_to=ends[i],
                security_id='research_equity_'+root,issuer_id='research_issuer_'+root,
                evidence_known_date=e['published'],evidence_file=e['evidence_file'],
                evidence_sha256=e['evidence_sha256'],source_url=e['source'],
                retrospective_identity_link_verified=True,point_in_time_identity_certified=False))
    if len(seen)!=len(edges):raise ValueError('Unresolved alias cycle')
    return rows


def resolve(rows,symbol,session):
    matching=[r for r in rows if r['symbol']==symbol and r['effective_from']<=session<r['effective_to']]
    if len(matching)>1:raise ValueError('Ambiguous dated identity')
    return matching[0] if matching else None


def main():
    docs=[]
    for path in sorted(P.glob('SECURITY_ALIAS_EVIDENCE*.json')):
        data=path.read_bytes();docs.append((path.name,json.loads(data),hashlib.sha256(data).hexdigest()))
    rows=compile_aliases(docs)
    result=dict(rows=rows,security_chains=len({r['security_id'] for r in rows}),symbol_intervals=len(rows),
                scope='Same-equity ticker continuity only; not a complete security master',
                excluded='Mergers, spin-offs, tracking stocks, ACT/AGN and class conversions',
                warning='Research IDs are assigned, not vendor identifiers. Earlier baseline identity still requires contemporaneous evidence; prices require independent lineage checks.')
    (P/'DATED_SYMBOL_IDENTITIES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
