"""Broaden company coverage using US listing evidence, without an index gate.

Public chart snapshots are retrospective research data, not certified PIT vintages.
All downloaded bodies are cached. HTTP429 stops this source for the UTC day.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, re, hashlib, time
import pandas as pd
import requests

P = Path(__file__).resolve().parent
O = P.parent
D = P / 'broad_caps'
D.mkdir(exist_ok=True)
(D / 'raw').mkdir(exist_ok=True)


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f, obj):
    f.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def parse(body):
    m = re.search(r'\bdata\s*=\s*(\[\{"d":.*?\]);', body.decode(), re.S)
    if not m: raise ValueError('No public historical chart')
    z = pd.DataFrame(json.loads(m.group(1)))
    z['date'] = pd.to_datetime(z.d, unit='s', utc=True).dt.strftime('%Y-%m-%d')
    z['market_cap_usd'] = z.m * 100000
    z = z[['date','market_cap_usd']].sort_values('date').drop_duplicates('date', keep='last')
    if not z.market_cap_usd.gt(0).all(): raise ValueError('Nonpositive cap')
    return z


def catalog():
    directory = json.loads((O/'four_factor_rebound/vendor_cap/DIRECTORY.json').read_text())
    listings = json.loads((P/'LISTING_EVIDENCE.json').read_text())
    by_symbol = {}
    for e in listings: by_symbol.setdefault(e['symbol'], []).append(e)
    # The existing source alias map preserves earlier names without adding a
    # security because it later appeared in a winning portfolio.
    aliases = {'GOOG':'GOOGL', 'BRK.A':'BRK-B', 'BRK.B':'BRK-B', 'BRK-A':'BRK-B'}
    out = {}
    for s, item in directory.items():
        canonical = aliases.get(s, s)
        ls = by_symbol.get(s, []) + ([] if canonical==s else by_symbol.get(canonical, []))
        if not ls: continue
        if re.search(r'-P[A-Z]+$', s) or s == 'PBR-A': continue
        out[canonical] = dict(symbol=canonical, url=item['url'], vendor_name=item['name'],
            listing_evidence=ls, discovery='full_cached_US_listing_registry_intersect_vendor_directory')
    # SKHY is a new US ADR; local Korean ticker is used only to identify its
    # issuer-wide USD cap. Its US eligibility starts with the US listing.
    if 'SKHY' in by_symbol and '000660.KS' in directory:
        item=directory['000660.KS']
        out['SKHY']=dict(symbol='SKHY', url=item['url'], vendor_name=item['name'],
            listing_evidence=by_symbol['SKHY'], discovery='new_US_ADR_issuer_mapping')
    for r in json.loads((O/'four_factor_rebound/vendor_cap/COLLECTION.json').read_text())['records']:
        s = aliases.get(r['vendor_ticker'], r['vendor_ticker'])
        if s in out: continue
        ls = by_symbol.get(s, []) + by_symbol.get(r['symbol'], [])
        out[s] = dict(symbol=s, url=r['url'], vendor_name=r['name'], listing_evidence=ls,
            legacy_security_id=r['symbol'], discovery='previous_historical_alias_cap_archive')
    dump(D/'CATALOG.json', out)
    return out


def main():
    cats = catalog()
    existing = {f.name:f for f in (O/'four_factor_rebound/vendor_cap/sources').glob('*.html')}
    for s in ['TSM','BABA','NVO','SAP','TM']:
        f=O/'monthly_reconstruction_v2/foreign/sources'/f'{s}_cap.html'
        if f.exists() and s in cats:
            slug=cats[s]['url'].rstrip('/').split('/')[-2]
            existing[slug+'.html']=f
    saved = D/'COLLECTION.json'
    results=json.loads(saved.read_text())['records'] if saved.exists() else []
    completed={r['symbol'] for r in results}
    failures=json.loads(saved.read_text()).get('failures',[]) if saved.exists() else []
    failed={r['symbol'] for r in failures}
    limit=D/'RATE_LIMIT.json'
    now=datetime.now(timezone.utc)
    throttled=limit.exists() and json.loads(limit.read_text())['utc_date']==now.date().isoformat()
    # Prefer larger current candidates only for download scheduling. Every
    # mapped historical candidate remains in the queue regardless of rank.
    priority=[r['symbol'] for r in json.loads((P/'CURRENT_RANKED_CANDIDATES.json').read_text())]
    order=sorted(cats, key=lambda s:(priority.index(s) if s in priority else 999, s))
    downloaded=0
    with requests.Session() as session:
        session.headers.update({'User-Agent':'Mozilla/5.0'})
        for s in order:
            if s in completed or s in failed: continue
            item=cats[s];slug=item['url'].rstrip('/').split('/')[-2]
            f=existing.get(slug+'.html', D/'raw'/(slug+'.html'))
            try:
                if not f.exists():
                    if throttled: continue
                    response=session.get(item['url'],timeout=25)
                    if response.status_code==429:
                        throttled=True
                        dump(limit,dict(utc_date=now.date().isoformat(),url=item['url'],
                            retry_after=response.headers.get('Retry-After'),time_utc=datetime.now(timezone.utc).isoformat()))
                        print('HTTP429: stop new source requests today; process cache only',flush=True)
                        continue
                    response.raise_for_status();f.write_bytes(response.content);downloaded+=1
                    time.sleep(.15)
                frame=parse(f.read_bytes())
                target=D/(s+'.csv');frame.to_csv(target,index=False)
                within=frame[(frame.date>='2015-01-01')&(frame.date<='2026-09-24')]
                results.append(dict(symbol=s,url=item['url'],raw_file=str(f),raw_sha256=sha(f),
                    data_file=str(target),sha256=sha(target),rows=len(frame),first=frame.date.min(),last=frame.date.max(),
                    maximum_cap_2015_to_cutoff=float(within.market_cap_usd.max()) if len(within) else None,
                    historical_publication_vintage_verified=False))
            except (requests.RequestException,ValueError,KeyError) as e:
                failures.append(dict(symbol=s,url=item['url'],error_type=type(e).__name__,error=str(e)[:180]))
            if (len(results)+len(failures))%20==0:
                dump(saved,dict(records=results,failures=failures,total_candidates=len(cats),throttled=throttled))
                print('processed',len(results), 'failed',len(failures), 'downloaded',downloaded,'of',len(cats),flush=True)
    dump(saved,dict(records=results,failures=failures,total_candidates=len(cats),throttled=throttled))
    print('COMPLETE',len(results),'failed',len(failures),'downloaded',downloaded,'throttled',throttled,flush=True)


if __name__=='__main__':main()
