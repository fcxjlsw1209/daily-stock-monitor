"""Materialize source-hashed real input candidates, without granting admission.

This is a data integration diagnostic, not a performance backtest. Every excluded
field is explicit. No current sector or later close is used to repair an input.
All reused/complex ticker identities stay out until a dated mapping is supplied.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
import pandas as pd
from research_calendar import sessions
from dynamic_features import features
from sector_history import benchmark_returns
P=Path(__file__).resolve().parent
START='2010-01-04';END='2026-10-06'
# These mappings concern unambiguous report group definitions, not a claim that
# the original 2009 report used GICS. Other groups need individual review.
GROUPS={
 'AEROSPACE & DEFENSE':'Industrials','APPAREL':'Consumer Cyclical',
 'AUTO MANUFACTURERS':'Consumer Cyclical','BANKS':'Financial Services',
 'BEVERAGES':'Consumer Defensive','BIOTECHNOLOGY':'Healthcare','CHEMICALS':'Basic Materials',
 'COMPUTERS':'Technology','COSMETICS & PERSONAL CARE':'Consumer Defensive',
 'DIVERSIFIED FINANCIAL SERVICES':'Financial Services','ELECTRIC':'Utilities',
 'FOOD':'Consumer Defensive','HEALTH CARE - PRODUCTS':'Healthcare',
 'HEALTH CARE - SERVICES':'Healthcare','INSURANCE':'Financial Services',
 'MACHINERY':'Industrials','MANUFACTURING':'Industrials','MINING':'Basic Materials',
 'OFFICE & BUSINESS EQUIPMENT':'Technology','OIL & GAS':'Energy',
 'OIL & GAS SERVICES':'Energy','PHARMACEUTICALS':'Healthcare','PIPELINES':'Energy',
 'SEMICONDUCTORS':'Technology','TRANSPORTATION':'Industrials'}
ETFS={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE',
 'Financial Services':'XLF','Industrials':'XLI','Technology':'XLK','Consumer Defensive':'XLP',
 'Real Estate':'XLRE','Utilities':'XLU','Healthcare':'XLV','Consumer Cyclical':'XLY'}
# Not a list of poor performers: these labels require distinct dated securities,
# holdings reorganizations or share-class entitlements. Never infer from ticker.
COMPLEX=set('AA ACT DD DELL DOW GOLD MICC SLE SGP SNDK VIP GOOG LBTYA LBTYK LILA LILAK FWONA FWONK BATRA BATRK QVCAQ QVCGQ'.split())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(path,digest):
    if sha(path)!=digest:raise ValueError('Source changed: '+str(path))

def add_date(row,calendar):
    known=pd.Timestamp(row['published_at'])
    if known.tzinfo is None:raise ValueError('Naive announcement')
    day=max(row['effective_session'],str(known.tz_convert('America/New_York').date()))
    later=calendar[calendar>pd.Timestamp(day)]
    return str(later[0].date()) if len(later) else '9999-12-31'

def provisional_members(calendar):
    names=['ndx_primary_reviewed_rows.json','sp100_modern_event_rows.json',
           'sp100_legacy_reviewed_rows.json','sp100_supplemental_reviewed_rows.json']
    events={k:[] for k in ['NDX','SP100']}
    for filename in names:
        for r in json.loads((P/filename).read_text()):
            checked(P/r['source_file'],r['source_sha256'])
            events['NDX' if filename.startswith('ndx') else 'SP100'].append(r)
    seeds={k:json.loads((P/f).read_text())['constructed_members'] for k,f in
           [('NDX','NDX_INITIAL_RECONSTRUCTION.json'),('SP100','SP100_INITIAL_RECONSTRUCTION.json')]}
    symbols=sorted(set(sum(seeds.values(),[]))|{r.get('ticker_as_announced_or_effective',r.get('ticker')) for v in events.values() for r in v})
    masks={k:pd.DataFrame(False,index=calendar,columns=symbols) for k in events}
    from audit_primary_chain import audit
    for k,mask in masks.items():
        chain=audit(seeds[k],events[k],start=START,end=END)
        if chain['errors']:raise ValueError('Broken provisional member chain')
        additions={(r.get('ticker_as_announced_or_effective',r.get('ticker')),r['effective_session']):r
                   for r in events[k] if r['action']=='addition'}
        for interval in chain['provisional_intervals']:
            s=interval['symbol'];begin=interval['start'];end=interval['end_exclusive']
            if begin!=START:begin=add_date(additions[(s,begin)],calendar)
            # A late-publicized short-lived addition may have no eligible day.
            if begin<end:
                mask.loc[(calendar>=begin)&(calendar<end),s]=True
    for r in json.loads((P/'LISTING_ELIGIBILITY_CONSTRAINTS.json').read_text()):
        checked(P/r['source_file'],r['source_sha256'])
        known=pd.Timestamp(r['known_at']).tz_convert('America/New_York')
        if str(known.date())>=r['ineligible_from']:raise ValueError('Listing constraint not known before effect')
        if r['symbol'] in masks[r['index']]:masks[r['index']].loc[r['ineligible_from']:,r['symbol']]=False
    return masks['NDX']|masks['SP100'],masks

def sector_candidates(calendar,symbols):
    result=pd.DataFrame(None,index=calendar,columns=symbols,dtype=object)
    rows=[];unmapped=[]
    for r in json.loads((P/'HISTORICAL_INDUSTRY_OBSERVATIONS.json').read_text())['observations']:
        sector=GROUPS.get(r['observed_industry'])
        if sector:
            rows.append(dict(symbol=r['symbol_as_of_report'],sector=sector,
             effective_session=r['observed_as_of'],known_at=r['known_at'],
             source_file=r['source_file'],source_sha256=r['source_sha256'],inferred_crosswalk=True))
        else:unmapped.append(dict(symbol=r['symbol_as_of_report'],industry=r['observed_industry']))
    rows+=json.loads((P/'SP_HISTORICAL_SECTOR_OBSERVATIONS.json').read_text())['observations']
    rows+=json.loads((P/'BIOTECH_2009_SECTOR_OBSERVATIONS.json').read_text())['observations']
    rows+=json.loads((P/'SPDR_SECTOR_CANDIDATES.json').read_text())['observations']
    rows+=json.loads((P/'RYDEX_2009_SECTOR_OBSERVATIONS.json').read_text())['observations']
    rows+=json.loads((P/'SPDR_ANNUAL_SECTOR_OBSERVATIONS.json').read_text())['observations']
    gics={'Financials':'Financial Services','Consumer Staples':'Consumer Defensive',
          'Consumer Discretionary':'Consumer Cyclical','Information Technology':'Technology',
          'Health Care':'Healthcare','Materials':'Basic Materials'}
    for r in json.loads((P/'HISTORICAL_SECTOR_TRANSITIONS.json').read_text())['records']:
        for symbol in r['symbols_at_event']:
            rows.append(dict(symbol=symbol,sector=gics.get(r['new_gics_sector'],r['new_gics_sector']),
               effective_session=r['effective_session'],known_at=r['known_date']+'T23:59:59-04:00',
               source_file=r['source_file'],source_sha256=r['source_sha256']))
    for r in rows:
        checked(P/r['source_file'],r['source_sha256'])
        known=pd.Timestamp(r['known_at']).tz_convert('America/New_York')
        # End-of-day date-only evidence cannot be used before the next session.
        first=max(pd.Timestamp(r['effective_session']),known.tz_localize(None).normalize())
        if known.hour>=16:first=max(first,known.tz_localize(None).normalize()+pd.Timedelta(days=1))
        r['usable_from']=str(first.date())
    for r in sorted(rows,key=lambda x:(x['usable_from'],x['known_at'])):
        if r['symbol'] in result:result.loc[r['usable_from']:,r['symbol']]=r['sector']
    # Forward observation persistence is only a diagnostic assumption, never a
    # certificate that unobserved GICS transitions did not occur.
    return result,rows,unmapped

def main():
    out=P/'input_diagnostics';out.mkdir(exist_ok=True)
    dates=sessions('2008-01-02',END)
    membership,indexes=provisional_members(dates);symbols=membership.columns
    panels={k:pd.DataFrame(np.nan,index=dates,columns=symbols) for k in
      ['adj','close','opening','high','low','volume','dividends','splits','raw_per_signal']}
    inventory=json.loads((P/'EXECUTION_CANDIDATES.json').read_text())['files'];used=[]
    selected={r['symbol']:dict(r,signal_file=r['input_file'],signal_sha256=r['input_sha256'],provider='yahoo') for r in inventory}
    # Fixed provider priority; never select a provider from strategy returns.
    extra=json.loads((P/'RAW_SOURCE_CANDIDATES.json').read_text())['files']
    for r in sorted(extra,key=lambda r:r['provider']!='tiingo'):
        if r['symbol'] not in selected:
            selected[r['symbol']]=dict(r,file=r['raw_file'],sha256=r['raw_sha256'])
    # Retired constituents whose modern Yahoo label has been reused. The
    # archive predates the reuse; modern label history cannot identify the old stock.
    for r in extra:
        if r['provider']=='wiki_union_archive' and r['symbol'] in {'APC','BBBY','EMC','SHLD','SPLS'}:
            selected[r['symbol']]=dict(r,file=r['raw_file'],sha256=r['raw_sha256'])
    for s,r in selected.items():
        if s not in symbols or s in COMPLEX:continue
        checked(P/r['file'],r['sha256']);checked(P/r['signal_file'],r['signal_sha256'])
        d=pd.read_csv(P/r['signal_file'],index_col='date',parse_dates=True)
        raw=pd.read_csv(P/r['file'],index_col='date',parse_dates=True)
        good=d.bar_valid & d.quarantine_reason.fillna('').eq('')
        for target,src in [('adj','adj_close'),('opening','open'),('close','close'),('high','high'),('low','low'),('volume','volume')]:
            panels[target][s]=d[src].where(good).reindex(dates)
        for target,src in [('dividends','dividend'),('splits','split'),('raw_per_signal','raw_per_signal')]:
            panels[target][s]=raw[src].where(good).reindex(dates)
        used.append(dict(symbol=s,provider=r['provider'],input_file=r['signal_file'],input_sha256=r['signal_sha256'],raw_file=r['file'],raw_sha256=r['sha256']))
    sectors,observations,unmapped=sector_candidates(dates,symbols)
    etf=pd.DataFrame(index=dates);etf_sources=[]
    for symbol in list(ETFS.values())+['SPY']:
        path=P/'sources/yahoo_prices'/(symbol+'.csv')
        prov=json.loads(path.with_suffix('.provenance.json').read_text());checked(path,prov['sha256'])
        d=pd.read_csv(path,index_col='Date',parse_dates=True)
        etf[symbol]=d['Adj Close'].reindex(dates)
        etf_sources.append(dict(file=str(path.relative_to(P)),sha256=prov['sha256']))
        if symbol=='SPY':
            market_close=d['Close'].reindex(dates);market_dividends=d['Dividends'].reindex(dates)
    sector60=benchmark_returns(sectors,etf,ETFS)
    eligible=membership.copy()
    for s in COMPLEX & set(symbols):eligible[s]=False
    # Diagnostic ticker IDs, not certified permanent identities. No multiple
    # share classes are admitted as independent issuer positions here.
    panels.update(sectors=sectors,sector60=sector60,eligible=eligible,
                  market_close=market_close,market_dividends=market_dividends)
    calculated=features(**{k:panels[k] for k in ['adj','close','opening','high','low','volume','sector60','market_close','market_dividends','eligible']})
    counts=pd.DataFrame(index=dates)
    counts['provisional_members']=membership.sum(axis=1)
    counts['observed_close']=(membership&panels['close'].notna()).sum(axis=1)
    counts['dated_sector_observation']=(membership&sectors.notna()).sum(axis=1)
    counts['valid_253_session_history']=(membership&panels['adj'].notna().rolling(253).sum().eq(253)).sum(axis=1)
    counts['sector_and_price']=(membership&sectors.notna()&panels['close'].notna()).sum(axis=1)
    for name,f in calculated.items():counts[name+'_entry_signals']=f['mask'].sum(axis=1)
    counts=counts.loc[START:END]
    counts.to_csv(out/'coverage_by_session.csv')
    sources=[]
    for key,value in panels.items():
        path=out/(key+'.csv.gz');value.to_csv(path,compression={'method':'gzip','mtime':0})
        sources.append(dict(field=key,file=str(path.relative_to(P)),sha256=sha(path)))
    gaps=[]
    for s in symbols:
        member=membership[s].loc[START:END]
        gaps.append(dict(symbol=s,member_sessions=int(member.sum()),
            missing_price=int((member&panels['close'][s].loc[START:END].isna()).sum()),
            missing_sector=int((member&sectors[s].loc[START:END].isna()).sum()),
            complex_identity_excluded=s in COMPLEX))
    report=dict(generated_at=datetime.now(timezone.utc).isoformat(),start=START,end=END,
      status='actual_input_integration_diagnostic_not_admitted',performance_publishable=False,
      real_sessions=len(counts),candidate_ticker_columns=len(symbols),price_inputs_loaded=len(used),yahoo_inputs_loaded=sum(r['provider']=='yahoo' for r in used),
      fields=sources,price_sources=used,etf_sources=etf_sources,
      sector_sources=[dict(file=f,sha256=sha(P/f)) for f in sorted({r['source_file'] for r in observations})],
      yearly_coverage=counts.groupby(counts.index.year).mean().round(2).to_dict(orient='index'),
      symbol_gaps=gaps,unmapped_2009_industries=unmapped,
      assumptions=['Ticker-level member chain is provisional, source hashed; addition delayed after known and effective dates.',
       'Unambiguous report industries crosswalked for diagnosis only; unobserved sector changes may remain.',
       'Complex/reused ticker histories explicitly excluded; remaining ticker identity is still not certified.',
       'No holdings or performance calculated; no paper account modified.'])
    (out/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','real_sessions','candidate_ticker_columns','yahoo_inputs_loaded','yearly_coverage']},indent=2))
if __name__=='__main__':main()
