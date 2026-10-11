"""Recover an explicitly identified retired security, without reusing its ticker feed."""
import datetime,hashlib,json
from pathlib import Path
import pandas as pd
import exchange_calendars as xc
from audit_price_archives import audit
P=Path(__file__).resolve().parent
IDENTIFIER='US000000013059'
def select_identity(rows):
    selected=[r for r in rows if r.get('permaTicker')==IDENTIFIER]
    if len(selected)!=1 or selected[0].get('name')!='Paramount Global - Class B' or selected[0].get('ticker')!='PARA' or selected[0].get('isActive') is not False:
        raise ValueError('Retired security identity mismatch')
    return selected[0]
def eligible_dates(dates):
    # Before this date the series needs CBS/VIAC lifecycle mapping. Merger date
    # has a new security trading; no terminal liquidation is inferred here.
    return dates.ge('2022-02-17') & dates.lt('2025-08-07')
def checked(path):
    raw=path.read_bytes();meta=json.loads(path.with_suffix('.provenance.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('Checksum mismatch')
    return json.loads(raw)
def main():
    identity=select_identity(checked(P/'additional_sources/tiingo_search/PARA_including_delisted.json'))
    rows=checked(P/'additional_sources/tiingo'/f'{IDENTIFIER}_prices.json')
    df=pd.DataFrame(rows);df['symbol']='PARA'
    out=audit(df,'tiingo_perma_PARAMOUNT_B')
    sessions=set(xc.get_calendar('XNYS',start='2008-01-01',end='2026-10-05').sessions.strftime('%Y-%m-%d'))-{'2025-01-09'}
    out['lifecycle_quarantine']=~eligible_dates(out.date)
    out['quarantine']=out.quarantine|out.lifecycle_quarantine|~out.date.isin(sessions)
    out['vendor_permaticker']=IDENTIFIER
    out.to_csv(P/'recovered_prices/tiingo_perma_PARAMOUNT_B.csv.gz',index=False,compression='gzip')
    good=out[~out.quarantine]
    viac=out.copy()
    viac['symbol']='VIAC'
    viac['lifecycle_quarantine']=~(viac.date.ge('2019-12-05') & viac.date.lt('2022-02-17'))
    # Re-run structural validation, rather than clearing another period's mask.
    viac=audit(viac.drop(columns=['quarantine']), 'tiingo_perma_VIAC_B')
    viac['quarantine']=viac.quarantine|viac.lifecycle_quarantine|~viac.date.isin(sessions)
    viac.to_csv(P/'recovered_prices/tiingo_perma_VIAC_B.csv.gz',index=False,compression='gzip')
    report=dict(viac_candidate_rows=int((~viac.quarantine).sum()),checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity=identity,raw_rows=len(out),candidate_rows=len(good),quarantined_rows=int(out.quarantine.sum()),first_candidate=good.date.min(),last_candidate=good.date.max(),sources=['https://ir.paramount.com/news-releases/news-release-details/viacomcbs-announces-completion-merger-cbs-and-viacom','https://www.sec.gov/Archives/edgar/data/813828/000081382822000011/viac-20220216.htm','https://ir.paramount.com/news-releases/news-release-details/skydance-media-and-paramount-global-complete-merger-creating'],usable_for_backtest=False,remaining=['Earlier CBS mapping and merger conversion','Corporate action adjustment validation','Merger election and settlement','Company capitalization/class aggregation'])
    (P/'PERMATICKER_RECOVERY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def audit_peabody():
    identifier='US000000095151'
    hits=checked(P/'additional_sources/tiingo_search/Peabody_including_delisted.json')
    matches=[r for r in hits if r.get('permaTicker')==identifier]
    if len(matches)!=1 or matches[0].get('name')!='Peabody Energy' or matches[0].get('isActive') is not False:raise ValueError('Peabody identity mismatch')
    rows=checked(P/'additional_sources/tiingo'/f'{identifier}_prices.json')
    df=pd.DataFrame(rows);df['symbol']='BTU'
    out=audit(df,'tiingo_perma_PEABODY_OLD')
    out['lifecycle_quarantine']=out.date.ge('2016-04-13')
    sessions=set(xc.get_calendar('XNYS',start='2008-01-01',end='2026-10-05').sessions.strftime('%Y-%m-%d'))-{'2025-01-09'}
    out['quarantine']=out.quarantine|out.lifecycle_quarantine|~out.date.isin(sessions)
    out['vendor_permaticker']=identifier
    out['security_episode']='Peabody_pre_2017_bankruptcy_equity'
    out.to_csv(P/'recovered_prices/tiingo_perma_PEABODY_OLD.csv.gz',index=False,compression='gzip')
    good=out[~out.quarantine]
    report=dict(identity=matches[0],raw_rows=len(out),candidate_rows=len(good),quarantined_rows=int(out.quarantine.sum()),last_candidate=good.date.max(),usable_for_backtest=False,sources=['https://www.sec.gov/Archives/edgar/data/1064728/000106472816000173/peabodyenergy-8xknyse.htm','https://www.sec.gov/Archives/edgar/data/1064728/000106472815000088/btu_8-kx20150930xpostxsplit.htm'],remaining=['Issuer continuity must not imply old equity survives bankruptcy','OTC disposal/zero-value terminal treatment','Full action and session validation'])
    (P/'PEABODY_PERMA_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
    main()
    audit_peabody()
