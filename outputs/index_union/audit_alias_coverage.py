"""Estimate alias-related date coverage; never certify provider security identity.

Restrict to documented ticker changes or one-for-one holding-company transitions. Reused tickers, spun-out securities,
class conversions and merger successors are deliberately not concatenated.
"""
from pathlib import Path
import gzip
import hashlib
import json
import pandas as pd
from research_calendar import sessions as research_sessions

P = Path(__file__).resolve().parent
GROUPS = [('WFMI','WFM'), ('ERTS','EA'), ('HANS','MNST'), ('JOYG','JOY'),
          ('RIMM','BBRY','BB'), ('CTRP','TCOM'), ('FB','META'), ('PCLN','BKNG'),
          ('LINTA','QVCA','QRTEA'), ('BK','BNY'), ('WAG','WBA'), ('WLTW','WTW')]


def main():
    observed = {k:set(v) for k,v in json.loads(gzip.decompress((P/'OBSERVED_PRICE_DATES.json.gz').read_bytes())).items()}
    receipts = []
    for f in sorted(P.glob('SECURITY_ALIAS_EVIDENCE*.json')):
        receipts.extend(json.loads(f.read_text())['aliases'])
    links = {(r['old'],r['new']):r for r in receipts}
    groups = {}
    for group in GROUPS:
        for a,b in zip(group,group[1:]):
            if (a,b) not in links:
                raise ValueError('Missing reviewed alias evidence: '+a+' '+b)
        for symbol in group:
            groups[symbol] = group
    sessions = list(research_sessions('2008-01-01','2026-10-07').strftime('%Y-%m-%d'))
    # Source- and date-restricted alias. Never union all modern NWSA/FOXA data.
    news = json.loads((P/'NEWS_LINEAGE_OVERLAP_AUDIT.json').read_text())
    news_dates = set()
    if news['candidate_date_alias_supported']:
        for file, sha in news['source_files'].items():
            if hashlib.sha256((P/file).read_bytes()).hexdigest() != sha:
                raise ValueError('Dated alias evidence changed')
        d = pd.read_csv(P/'sources/wiki_union/FOXA.csv')
        news_dates = set(d.loc[d.date.ge(news['applicable_from']) & d.date.lt(news['applicable_to_exclusive']), 'date'])
    fox_file = P/'sources/quantiacs/extracted/TFCFA.csv'
    fox_dates = set()
    if fox_file.exists():
        receipt=json.loads(fox_file.with_suffix('.provenance.json').read_text())
        if receipt['provider_identity']['id'] != 'tts-161284560' or receipt['provider_identity']['cik'] != '1308161':
            raise ValueError('Unexpected old Fox identity')
        if hashlib.sha256(fox_file.read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('Old Fox source changed')
        d=pd.read_csv(fox_file)
        fox_dates=set(d.loc[d.date.ge('2013-07-01') & d.date.lt('2019-03-19'), 'date'])
    liberty=json.loads((P/'LIBERTY_2013_DATED_ALIAS.json').read_text())
    for file_key,hash_key in [('source_file','source_sha256'),('price_file','price_sha256')]:
        if hashlib.sha256((P/liberty[file_key]).read_bytes()).hexdigest()!=liberty[hash_key]:
            raise ValueError('Liberty dated alias source changed')
    ld=pd.read_csv(P/liberty['price_file'])
    liberty_dates=set(ld.loc[ld.date.ge(liberty['applicable_from']) & ld.date.lt(liberty['applicable_to_exclusive']), 'date'])
    results = []
    for index in ['NDX','SP100']:
        data = json.loads((P/(index+'_PRIMARY_CHAIN_AUDIT.json')).read_text())
        for interval in data['provisional_intervals']:
            sym = interval['symbol']
            required = [d for d in sessions if interval['start'] <= d < interval['end_exclusive']]
            direct = observed.get(sym,set())
            aliases = groups.get(sym,(sym,))
            possible = set().union(*(observed.get(s,set()) for s in aliases))
            dated_sources = []
            if sym == 'LMCA' and interval['end_exclusive'] <= '2013-01-14':
                possible |= liberty_dates
                dated_sources.append('wiki_union_archive:STRZA old LMCA through2013-01-11 only')
            if sym == 'FOXA':
                possible |= fox_dates
                dated_sources.append('quantiacs:tts-161284560; old21CF only2013-07-01..2019-03-18')
            if sym == 'NWSA' and interval['end_exclusive'] <= '2013-07-01':
                possible |= news_dates
                dated_sources.append('wiki_union_archive:FOXA before2013-07-01; NEWS_LINEAGE_OVERLAP_AUDIT.json')
            results.append(dict(index=index, **interval, alias_candidates=list(aliases), dated_alias_sources=dated_sources,
                required_sessions=len(required), same_ticker_missing=sum(d not in direct for d in required),
                missing_dates_after_alias_candidates=[d for d in required if d not in possible],
                remaining_date_gaps_after_alias_candidates=sum(d not in possible for d in required),
                identity_and_units_certified=False))
    report = dict(admission=False, scope='Provisional intervals, potential alias date coverage only; overlapping index sessions counted separately',
        same_ticker_missing=sum(r['same_ticker_missing'] for r in results),
        missing_after_alias_candidates=sum(r['remaining_date_gaps_after_alias_candidates'] for r in results),
        intervals=results,
        warning='No alias dates are promoted to executable bars. Provider lineage and units must be checked before use; this only prioritizes investigation.')
    (P/'ALIAS_COVERAGE_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='intervals'},indent=2))
    for r in sorted(results,key=lambda r:r['remaining_date_gaps_after_alias_candidates'],reverse=True)[:20]:
        print(r['index'],r['symbol'],r['start'],r['end_exclusive'],r['remaining_date_gaps_after_alias_candidates'])

if __name__ == '__main__':
    main()
