"""Audit cached vendor listings. Episodes are evidence, never certified issuer IDs."""
import csv, hashlib, io, json, sqlite3
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=ROOT/'additional_sources'/'alpha_vantage'
def audit():
    episodes=defaultdict(lambda: {'active_dates':set(),'delisted_dates':set(),'names':set(),'sources':set(),'delisting_dates':set()})
    snapshots=[]
    for meta in sorted(P.glob('????-??-??_*.json')):
        m=json.loads(meta.read_text()); data=meta.with_suffix('.csv').read_bytes()
        if not m.get('validated') or hashlib.sha256(data).hexdigest()!=m['sha256']:
            raise ValueError('Listing checksum/validation failure: '+meta.name)
        rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        stocks=[r for r in rows if r['assetType']=='Stock']
        snapshots.append({'date':m['date'],'state':m['state'],'rows':len(rows),'stock_rows':len(stocks)})
        for r in stocks:
            # IPO and exchange distinguish some reused symbols, but are not permanent identifiers.
            key=(r['symbol'],r['exchange'],r['ipoDate'])
            e=episodes[key];e[m['state']+'_dates'].add(m['date']);e['names'].add(r['name']);e['sources'].add(meta.name)
            if r['delistingDate'] not in ('','null','None'):e['delisting_dates'].add(r['delistingDate'])
    symbol_keys=defaultdict(list)
    for k in episodes:symbol_keys[k[0]].append(k)
    candidates={r['symbol'] for r in csv.DictReader((ROOT/'historical_candidate_inventory.csv').open())}
    identity=defaultdict(set)
    for r in csv.DictReader((ROOT/'expanded_sec/candidate_identity_evidence.csv').open()):identity[r['symbol']].add(r['cik'])
    normalized=[]
    for k,e in sorted(episodes.items()):
        symbol,exchange,ipo=k
        conflicts=[]
        if len(symbol_keys[symbol])>1:conflicts.append('multiple_symbol_episodes')
        if len(e['names'])>1:conflicts.append('multiple_names')
        if len(e['delisting_dates'])>1:conflicts.append('conflicting_delisting_dates')
        if e['active_dates'] and e['delisting_dates'] and max(e['active_dates'])>=min(e['delisting_dates']):conflicts.append('active_at_or_after_delisting')
        normalized.append(dict(evidence_id=hashlib.sha256('|'.join(k).encode()).hexdigest()[:20],symbol=symbol,exchange=exchange,ipo_date=ipo,names=sorted(e['names']),active_snapshot_dates=sorted(e['active_dates']),delisted_snapshot_dates=sorted(e['delisted_dates']),vendor_delisting_dates=sorted(e['delisting_dates']),source_files=sorted(e['sources']),possible_ciks=sorted(identity[symbol]),in_previous_candidates=symbol in candidates,conflicts=conflicts,issuer_identity_verified=False,eligible_security_verified=False,usable_for_rank=False))
    out=ROOT/'listing_episodes_unverified.json';out.write_text(json.dumps(normalized,indent=2))
    summary=dict(snapshots=snapshots,stock_evidence_episodes=len(normalized),distinct_symbols=len(symbol_keys),symbols_with_multiple_episodes=sum(len(v)>1 for v in symbol_keys.values()),episodes_outside_prior_candidates=sum(not r['in_previous_candidates'] for r in normalized),episodes_with_conflicts=sum(bool(r['conflicts']) for r in normalized),rank_ready=0,complete_historical_universe_verified=False,notes=['Yearly snapshots cannot resolve all intra-year renames or listing interruptions.','Possible CIK matches are discovery hints only; no ticker-only identity joins authorized.','Vendor delisting dates are not executable terminal-price dates.'])
    (ROOT/'listing_audit.json').write_text(json.dumps(summary,indent=2))
    with sqlite3.connect(ROOT/'free_evidence.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS listing_evidence_unverified (evidence_id TEXT PRIMARY KEY, symbol TEXT, exchange TEXT, ipo_date TEXT, evidence_json TEXT, usable_for_rank INTEGER CHECK(usable_for_rank=0))')
        for r in normalized:db.execute('INSERT OR REPLACE INTO listing_evidence_unverified VALUES (?,?,?,?,?,0)',(r['evidence_id'],r['symbol'],r['exchange'],r['ipo_date'],json.dumps(r)))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('snapshots','notes')},indent=2))
if __name__=='__main__':audit()
