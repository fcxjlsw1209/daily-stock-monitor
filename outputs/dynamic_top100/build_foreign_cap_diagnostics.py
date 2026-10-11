"""Source-aware foreign issuer cap scenarios; never certifies membership.

Only explicitly extracted ordinary share totals and raw USD daily prices are
combined. Publication dates are conservatively available next UTC day. Historical
alias/action continuity is still an independent unresolved requirement.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parent
SYMBOLS = {'0001089113':'HSBC','0000067088':'MUFG','0001046179':'TSM',
           '0000818686':'TEVA','0001000275':'RY','0000839923':'VOD'}


def known_observation(rows, session):
    cutoff = pd.Timestamp(session, tz='UTC') + pd.Timedelta(hours=23)
    eligible = []
    for row in rows:
        available = pd.Timestamp(row['filed'], tz='UTC') + pd.Timedelta(days=1)
        if row.get('adr_ratio_published'):
            available = max(available, pd.Timestamp(row['adr_ratio_published'], tz='UTC') + pd.Timedelta(days=1))
        if available <= cutoff and row['as_of'] <= session:
            eligible.append((row['as_of'], available, row))
    return max(eligible, key=lambda x:(x[0],x[1]))[2] if eligible else None


def candidate_cap(observation, close, direct_common=False):
    ratio = 1 if direct_common else observation.get('ordinary_shares_per_ads')
    if ratio is None or ratio <= 0 or close <= 0:
        raise ValueError('Missing or invalid economic unit')
    return observation['total_common_share_count'] * close / ratio


def main():
    observations = json.loads((P/'pilot_identity/VERIFIED_CLASS_COUNTS.json').read_text())
    output = []
    for cik, symbol in SYMBOLS.items():
        file = P/'additional_sources/tiingo'/f'{symbol}_prices.json'
        meta = json.loads(file.with_name(file.stem+'.provenance.json').read_text())
        assert hashlib.sha256(file.read_bytes()).hexdigest() == meta['sha256']
        counts = [r for r in observations if r['cik']==cik]
        for bar in json.loads(file.read_text()):
            day = bar['date'][:10]
            if not '2010-03-01' <= day <= '2010-03-31': continue
            row = known_observation(counts, day)
            if row is None: continue
            cap = candidate_cap(row, bar['close'], direct_common=(symbol=='RY'))
            output.append(dict(session=day,cik=cik,vendor_symbol=symbol,
                historical_symbol=row.get('historical_us_symbol',symbol),raw_close=bar['close'],
                shares=row['total_common_share_count'],shares_as_of=row['as_of'],
                published=row['filed'],ordinary_per_price_unit=row.get('ordinary_shares_per_ads',1),
                candidate_cap_usd=cap,source=row['source'],price_sha256=meta['sha256'],
                usable_for_rank=False,reason='Stale shares, alias, actions and daily continuity not certified'))
    frame=pd.DataFrame(output)
    frame.to_csv(P/'pilot_identity/foreign_cap_scenarios_UNVERIFIED.csv.gz',index=False,compression='gzip')
    report={'scenarios':len(frame),'issuers':frame.cik.nunique(),'certified_intervals':0,
            'latest_scenarios':frame[frame.session=='2010-03-31'].to_dict('records'),
            'usable_for_backtest':False}
    (P/'FOREIGN_CAP_DIAGNOSTICS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'scenarios':len(frame),'issuers':frame.cik.nunique(),'certified_intervals':0}))

if __name__=='__main__': main()
