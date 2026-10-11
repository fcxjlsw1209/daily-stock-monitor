"""Compute unverified cap candidates to focus source work; never emit memberships.

Only vendor raw closes are used. Multiple classes, tickers and share concepts
remain alternative scenarios, not automatically certified or combined.
"""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd
from reconcile_share_evidence import reconcile

P = Path(__file__).resolve().parent


def extract_prices():
    dest = P / 'pilot_identity/wiki_march2010_raw.csv.gz'
    if dest.exists():
        return pd.read_csv(dest)
    pieces = []
    with zipfile.ZipFile(P / 'additional_sources/quandl_wiki.zip') as archive:
        with archive.open('WIKI_PRICES.csv') as stream:
            for chunk in pd.read_csv(stream, chunksize=500000, usecols=['ticker', 'date', 'close']):
                part = chunk[chunk.date.between('2010-03-01', '2010-03-31')]
                if len(part):
                    pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    if frame.duplicated(['ticker', 'date']).any():
        raise ValueError('Ambiguous raw prices')
    frame.to_csv(dest, index=False, compression='gzip')
    dest.with_suffix('.provenance.json').write_text(json.dumps(dict(source='additional_sources/quandl_wiki.zip', price_field='close (not adj_close)', sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), usable_for_backtest=False), indent=2))
    return frame


def main():
    prices = extract_prices()
    mappings = pd.read_csv(P / 'expanded_sec/candidate_identity_evidence.csv', dtype={'cik': str})[['cik', 'symbol']].dropna().drop_duplicates()
    # Original-report historical identities; price aliases still require validation.
    explicit = [('0001288776', 'GOOGL'), ('0001308161', 'FOXA'), ('0001308161', 'FOX'), ('0001285785', 'MOS'), ('0001377013', 'TWC'), ('0001465112', 'DTV')]
    mappings = pd.concat([mappings, pd.DataFrame(explicit, columns=['cik', 'symbol'])], ignore_index=True).drop_duplicates()
    mappings.cik = mappings.cik.str.zfill(10)
    facts = pd.concat([pd.read_csv(P / 'expanded_sec/disclosure_events_unverified.csv.gz', dtype={'cik': str}), pd.read_csv(P / 'pilot_identity/disclosure_events_unverified.csv.gz', dtype={'cik': str})], ignore_index=True).drop_duplicates()
    facts.cik = facts.cik.str.zfill(10)
    facts = reconcile(facts, json.loads((P / 'pilot_identity/share_corrections.json').read_text()))
    facts = facts[~facts.conflict & facts.shares.gt(0)].copy()
    facts['available'] = pd.to_datetime(facts.available_no_earlier_than, utc=True)
    scenarios = []
    for day, dayprices in prices.groupby('date'):
        decision = pd.Timestamp(day + ' 20:20', tz='America/Los_Angeles').tz_convert('UTC')
        known = facts[facts.available <= decision]
        # Keep independent concepts. Selecting a single one would hide class conflicts.
        latest = known.sort_values(['reported_as_of', 'available']).drop_duplicates(['cik', 'taxonomy', 'concept'], keep='last')
        joined = mappings.merge(latest, on='cik').merge(dayprices, left_on='symbol', right_on='ticker')
        joined['candidate_cap_usd'] = joined.shares * joined.close
        joined.loc[joined.scale_review_required, 'candidate_cap_usd'] = float('nan')
        joined['shares_age_days'] = (pd.Timestamp(day) - pd.to_datetime(joined.reported_as_of)).dt.days
        joined['usable_for_rank'] = False
        joined['reason'] = 'Unverified identity/classes/units; scenario only, not a sourced cap bound'
        scenarios.append(joined[['date', 'cik', 'symbol', 'taxonomy', 'concept', 'reported_as_of', 'filed', 'available_no_earlier_than', 'shares', 'original_shares', 'correction_source', 'scale_review_required', 'shares_age_days', 'close', 'candidate_cap_usd', 'accessions', 'source_file', 'usable_for_rank', 'reason']])
    result = pd.concat(scenarios, ignore_index=True)
    result.to_csv(P / 'pilot_identity/cap_scenarios_UNVERIFIED.csv.gz', index=False, compression='gzip')
    last = result[result.date.eq('2010-03-31')]
    report = dict(price_rows=len(prices), price_symbols=prices.ticker.nunique(), scenario_rows=len(result), days=result.date.nunique(), covered_issuer_hints=result.cik.nunique(), month_end_issuer_hints=last.cik.nunique(), certified_cap_intervals=0, can_start_backtest=False, purpose='Prioritize economic-unit, stale-share and identity checks. This is NOT an investable ranking and cannot feed the strategy engine.')
    (P / 'PILOT_CAP_DIAGNOSTICS.json').write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == '__main__':
    main()
