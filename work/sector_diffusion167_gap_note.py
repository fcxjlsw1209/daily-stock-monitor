"""Source-gap diagnosis only: no factor, correlation, rank, order or return computation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_sector_diffusion167'
DEST=ROOT/'work/sector_diffusion167_gap_note.json'
seen={}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path,expected=None):
    p=Path(path).resolve();require(p.is_relative_to(ROOT),'Local research files only')
    h=digest(p);require(expected is None or h==expected,'Changed source '+str(p))
    require(str(p) not in seen or seen[str(p)]==h,'Concurrent source change');seen[str(p)]=h
    return p
def read(path,expected=None):return json.loads(record(path,expected).read_text())
def bad_dates(values,dates):
    arr=np.asarray(values,dtype=float)
    return [str(d.date()) for d,b in zip(dates,~(np.isfinite(arr)&(arr>0))) if b]
def summary_bad(days):return dict(count=len(days),first=days[0] if days else None,last=days[-1] if days else None,dates=days)
def main():
    require(not DEST.exists(),'Write-once gap note')
    plan=read(OUT/'PLAN.json');cache=read(OUT/'CACHE.json')
    audit=read(OUT/'FEATURE_AUDIT.json',cache['FEATURE_AUDIT.json'])
    meta=read(OUT/'META.json',cache['META.json'])
    calendar=read(plan['inputs']['calendar']['path'],plan['inputs']['calendar']['sha256'])['sessions']
    pos={d:i for i,d in enumerate(calendar)}
    price_path=record(plan['inputs']['prices']['path'],plan['inputs']['prices']['sha256'])
    with np.load(price_path,allow_pickle=False) as z:
        prices=pd.DataFrame(z['close'],index=pd.to_datetime(z['dates']),columns=z['symbols'])
    with np.load(record(OUT/'DATA.npz',cache['DATA.npz']),allow_pickle=False) as z:
        deltas={k:z[k] for k in ('delta_followthrough','delta_control')}
    source={};source_meta={}
    for sector in ('XLC','XLK'):
        path=ROOT/'outputs/index_union/sources/yahoo_prices'/f'{sector}.csv'
        require(str(path) in plan['raw_source_hashes'],'Exact raw sector source required')
        f=record(path,plan['raw_source_hashes'][str(path)])
        frame=pd.read_csv(f,index_col=0,parse_dates=True)
        require(frame.index.is_unique,'Unique raw sector dates')
        source[sector]=frame
        source_meta[sector]=dict(path=str(f),sha256=digest(f),first=str(frame.index[0].date()),last=str(frame.index[-1].date()),rows=len(frame))
    skhy=ROOT/'outputs/monthly_published44_reconstruction/prices/SKHY.csv'
    skhy_frame=pd.read_csv(record(skhy,plan['raw_source_hashes'][str(skhy)]),index_col=0,parse_dates=True)
    skhy_info=read(skhy.with_suffix('.source.json'),plan['raw_source_hashes'][str(skhy.with_suffix('.source.json'))])
    source_meta['SKHY']=dict(path=str(skhy),sha256=digest(skhy),first=str(skhy_frame.index[0].date()),last=str(skhy_frame.index[-1].date()),rows=len(skhy_frame),source_receipt=skhy_info)
    rows=[]
    for i,row in enumerate(audit['event_history']):
        if row['initial_forced_neutral']:continue
        t=pos[row['date']];pdts=pd.to_datetime(calendar[t-253:t-41]);vdts=pd.to_datetime(calendar[t-273:t-42])
        require(len(pdts)==212 and len(vdts)==231,'Frozen window sizes')
        require(str(pdts[0].date())==row['window_start'] and str(pdts[-1].date())==row['window_end'],'Price windows match audit')
        require(str(vdts[0].date())==row['volume_window_start'] and str(vdts[-1].date())==row['volume_window_end'],'Volume windows match audit')
        for j,(symbol,direct,valid) in enumerate(zip(row['symbols'],row['eligible_direct_stock'],row['common_support'])):
            if not direct or valid:continue
            sector=row['sector_symbols'][j];require(sector in source,'Unexpected gap sector')
            stock_bad=bad_dates(prices[symbol].reindex(pdts),pdts)
            sector_bad=bad_dates(prices[sector].reindex(pdts),pdts)
            volume_bad=bad_dates(source[sector]['Volume'].reindex(vdts),vdts)
            raw_sector_bad=bad_dates(source[sector]['Adj Close'].reindex(pdts),pdts)
            require(raw_sector_bad==sector_bad,'Raw/cached sector missing masks agree')
            if symbol=='SKHY':require(bad_dates(skhy_frame['Adj Close'].reindex(pdts),pdts)==stock_bad,'Raw/cached SKHY missing masks agree')
            neutral=all(deltas[k][i,j]==0. for k in deltas)
            require(neutral and row['raw_followthrough'][j] is None and row['raw_control'][j] is None,'Missing cells must be shared-neutral')
            causes=[]
            if stock_bad:causes.append('missing_or_nonpositive_stock_adjusted_price')
            if sector_bad:causes.append('missing_or_nonpositive_sector_adjusted_price')
            if volume_bad:causes.append('missing_or_nonpositive_sector_volume')
            rows.append(dict(signal_date=row['date'],symbol=symbol,display_symbol='FB' if symbol=='research_equity_FB' else symbol,sector=sector,
                stock_price=summary_bad(stock_bad),sector_price=summary_bad(sector_bad),sector_volume=summary_bad(volume_bad),
                classification=causes or ['source_complete_numerical_degeneracy_unresolved'],
                source_failure_sufficient_to_explain_neutral=bool(causes),both_deltas_zero=neutral,
                numerical_rank_or_residual_not_evaluated=True))
    require(len(rows)==95,'Expected95 existing mapped unavailable cells')
    by_symbol={}
    causes={}
    for symbol in sorted({r['symbol'] for r in rows}):
        rr=[r for r in rows if r['symbol']==symbol]
        by_symbol[symbol]=dict(cells=len(rr),first_signal=rr[0]['signal_date'],last_signal=rr[-1]['signal_date'],signals=[r['signal_date'] for r in rr],
            all_have_proven_source_gap=all(r['source_failure_sufficient_to_explain_neutral'] for r in rr))
    for row in rows:
        key='+'.join(row['classification']);causes[key]=causes.get(key,0)+1
    unresolved=[dict(signal_date=r['signal_date'],symbol=r['symbol']) for r in rows if not r['source_failure_sufficient_to_explain_neutral']]
    require(all(digest(p)==h for p,h in seen.items()),'Sources changed during diagnosis')
    result=dict(schema='sector_diffusion167_postprepare_source_gap_note_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only diagnosis of already registered and prepared missing cells. Only positive/finite source support, masks and neutral deltas checked. No new factor, SVD, correlation, ranking, orders or return paths calculated.',
        plan_sha256=seen[str(OUT/'PLAN.json')],audit_sha256=seen[str(OUT/'FEATURE_AUDIT.json')],mapped_unavailable_cells=95,
        by_symbol=by_symbol,causes=causes,unresolved_numerical_or_other=unresolved,source_files=source_meta,rows=rows,input_hashes=seen,
        raw_cached_sector_and_skhy_missing_masks_match=True,all95_shared_neutral=all(r['both_deltas_zero'] for r in rows),
        conclusions=['The XLC archive begins2018-06-19. Existing source windows before that date cannot meet full212price/231volume support; neither price nor volume is backfilled.',
                     'SKHY archive begins2026-07-10, after both affected old-window endpoints; all212 required old stock prices are unavailable for each affected signal.',
                     'A demonstrated source failure suffices to explain neutral treatment. Numerical degeneracy is not evaluated on incomplete windows, so no claim is made about latent rank or residual behavior after hypothetical data repair.',
                     'These diagnostics neither alter the frozen source, mapping, eligibility nor source-support rules. ETF history is not extended or replaced.'],
        actual_new_features_computed=0,actual_new_orders_computed=0,actual_returns_computed=0,numerical_projections_computed=0,frozen_files_changed=0,
        generator=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)))
    with DEST.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(DEST),sha256=digest(DEST),by_symbol=by_symbol,causes=causes,unresolved=unresolved)))
if __name__=='__main__':main()
