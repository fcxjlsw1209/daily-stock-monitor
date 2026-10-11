from pathlib import Path
from datetime import datetime,timezone
import json,sqlite3,math
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
DB=P.parents[1]/'stock_database'/'stocks.sqlite'
SECTOR_ETF={'Technology':'XLK','Communication Services':'XLC','Consumer Cyclical':'XLY','Consumer Defensive':'XLP','Financial Services':'XLF','Healthcare':'XLV','Industrials':'XLI','Energy':'XLE','Basic Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE'}

def read_bars(path):
    h=pd.read_csv(path,index_col=0)
    h.index=pd.to_datetime(h.index,utc=True).tz_convert('America/New_York')
    if h.index.duplicated().any():raise ValueError('Duplicate timestamps '+str(path))
    return h.sort_index()

def snapshot(h):
    """Only completed bars BEFORE 15:40. No dependence on later bars."""
    rows=[]
    for day,g in h.groupby(h.index.date):
        prefix=g.between_time('09:30','15:35'); times=set(prefix.index.strftime('%H:%M'))
        required=set(pd.date_range('2000-01-01 09:30','2000-01-01 15:35',freq='5min').strftime('%H:%M'))
        if times!=required:continue
        if prefix[['Open','High','Low','Close','Volume']].isna().any().any():continue
        if (prefix.Close<=0).any() or (prefix.Volume<0).any():continue
        early=prefix.between_time('09:30','15:05');tail=prefix.between_time('15:10','15:35')
        if len(tail)!=6 or tail.Volume.sum()<=0:continue
        vw=(((tail.High+tail.Low+tail.Close)/3)*tail.Volume).sum()/tail.Volume.sum()
        price=float(prefix.Close.iloc[-1])
        rows.append({'date':str(day),'price1540':price,'early_low':early.Low.min(),'tail_low':tail.Low.min(),'vwap_proxy':vw,
          'stabilized':bool(tail.Low.min()>=early.Low.min() and price>=vw),'last_signal_bar':str(prefix.index[-1])})
    return pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()

def choose(f,flag):
    chosen=[];counts={}
    for r in f[f[flag]].sort_values(['z3','symbol']).itertuples():
        if counts.get(r.sector,0)>=2:continue
        chosen.append(r.symbol);counts[r.sector]=counts.get(r.sector,0)+1
        if len(chosen)==5:break
    return chosen if len(chosen)>=3 else []

def main():
    manifest=json.loads((P/'download_manifest.json').read_text());meta={r['symbol']:r for r in manifest}
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:d=pd.read_sql_query('SELECT * FROM daily_prices',db)
    def wide(k):return d.pivot(index='trade_date',columns='symbol',values=k).sort_index()
    c,o,div,volume=[wide(k) for k in ['close','open','dividends','volume']]
    dates=c.index;symbols=list(c.columns);day_next=dict(zip(dates[:-1],dates[1:]))
    ma100=c.rolling(100).mean().shift(1);r60=c.pct_change(60,fill_method=None).shift(1)
    sd=c.pct_change(fill_method=None).rolling(20).std().shift(1)
    eligible=((c.rolling(200).count()>=200)&((c*volume).rolling(20).mean()>20e6)).shift(1).fillna(False)
    rank=r60.where(eligible).rank(axis=1,pct=True)
    strong=eligible&(c.shift(1)>ma100)&(r60>0)&(rank>=.75)
    bars={};snap={};etfdaily={};coverage=[];qc=[]
    for symbol,m in meta.items():
        path=P/'raw'/f'{symbol}_5m.csv.gz'
        if not path.exists():continue
        h=read_bars(path);bars[symbol]=h;snap[symbol]=snapshot(h)
        coverage.append({'symbol':symbol,'rows':len(h),'sessions':len(set(h.index.date)),'usable_1540_sessions':len(snap[symbol]),'first':str(h.index.min()),'last':str(h.index.max()),'sector':m.get('sector')})
        if symbol in set(SECTOR_ETF.values())|{'SPY'}:
            ed=pd.read_csv(P/'raw'/f'{symbol}_daily.csv',index_col=0)
            ed.index=pd.to_datetime(ed.index,utc=True).tz_convert('America/New_York').strftime('%Y-%m-%d')
            etfdaily[symbol]=ed.reindex(dates)
        elif symbol in c:
            for day,g in h.groupby(h.index.strftime('%Y-%m-%d')):
                if day in dates and '15:55' in set(g.index.strftime('%H:%M')):
                    last=g.between_time('15:55','15:55').Close.iloc[0]
                    if abs(last/c.at[day,symbol]-1)>.03:qc.append({'symbol':symbol,'date':day,'reason':'intraday/daily close difference >3%'})
    pd.DataFrame(coverage).to_csv(P/'intraday_coverage.csv',index=False)
    sample=sorted(set(snap['SPY'].index)&set(dates[:-1]))
    earnings={};earnings_coverage={}
    for symbol in symbols:
        path=P/'raw'/f'{symbol}_earnings.csv'
        if path.exists():
            ed=pd.read_csv(path,index_col=0);idx=pd.to_datetime(ed.index,utc=True).tz_convert('America/New_York')
            earnings[symbol]=set(idx.strftime('%Y-%m-%d'))
            earnings_coverage[symbol]=bool(min(earnings[symbol])<=sample[0] and max(earnings[symbol])>=sample[-1])
        else:earnings[symbol]=set();earnings_coverage[symbol]=False
    prior3=c.shift(3)
    etfprior3={s:e.Close.shift(3) for s,e in etfdaily.items()}
    feats=[];spyma=etfdaily['SPY'].Close.rolling(200).mean().shift(1)
    for day in sample:
        next_day=day_next[day];sp=snap['SPY'].loc[day].price1540
        market_good=sp>spyma.loc[day]
        spret=sp/etfprior3['SPY'].loc[day]-1
        for symbol in symbols:
            sector=meta.get(symbol,{}).get('sector');etf=SECTOR_ETF.get(sector)
            if symbol not in snap or day not in snap[symbol].index or etf not in snap or day not in snap[etf].index:continue
            v=snap[symbol].loc[day];r3=v.price1540/prior3.at[day,symbol]-1
            scale=sd.at[day,symbol]*np.sqrt(3);z=r3/scale if scale>0 else np.nan
            er=snap[etf].loc[day].price1540/etfprior3[etf].loc[day]-1
            a=bool(strong.at[day,symbol] and market_good and -2.5<=z<=-.75 and r3<0)
            b=bool(a and (er<0 or spret<0) and (r3-er)>=-scale)
            cc=bool(b and v.stabilized)
            excluded=day in earnings[symbol] or next_day in earnings[symbol]
            dd=bool(cc and earnings_coverage[symbol] and not excluded)
            feats.append({'date':day,'next_date':next_day,'symbol':symbol,'sector':sector,'sector_etf':etf,'z3':z,'ret3':r3,
              'sector_ret3':er,'spy_ret3':spret,'price1540':v.price1540,'tail_low':v.tail_low,'early_low':v.early_low,
              'vwap_proxy':v.vwap_proxy,'last_signal_bar':v.last_signal_bar,'A_pullback':a,'B_market_linked':b,'C_stabilized':cc,
              'D_earnings_diagnostic':dd,'earnings_coverage':earnings_coverage[symbol],'earnings_excluded':excluded})
    features=pd.DataFrame(feats);features.to_csv(P/'signal_features.csv.gz',index=False,compression='gzip')
    daily=[];trades=[];exclusions=[];picks=[]
    variants=['A_pullback','B_market_linked','C_stabilized','D_earnings_diagnostic']
    for day in sample:
        f=features[features.date==day];nxt=day_next[day]
        for model in variants:
            selected=choose(f,model)
            picks.append({'date':day,'strategy':model,'candidates':int(f[model].sum()),'selected':','.join(selected),'count':len(selected)})
            for exitmode in ['open','09:45','10:00']:
                rows=[];missing=[]
                for s in selected:
                    entry=c.at[day,s]
                    if exitmode=='open':sell=o.at[nxt,s]
                    else:
                        hh=bars[s];bar=hh[(hh.index.strftime('%Y-%m-%d')==nxt)&(hh.index.strftime('%H:%M')==exitmode)]
                        sell=bar.Open.iloc[0] if len(bar)==1 else np.nan
                    if not np.isfinite(entry) or not np.isfinite(sell) or entry<=0 or sell<=0:missing.append(s);continue
                    gross=(sell+div.at[nxt,s])/entry-1
                    rows.append({'entry_date':day,'exit_date':nxt,'strategy':model,'exit_mode':exitmode,'symbol':s,'weight':.2,'entry':entry,'exit':sell,'dividend':div.at[nxt,s],'gross':gross})
                if missing:
                    exclusions.append({'date':day,'strategy':model,'exit_mode':exitmode,'missing':missing});continue
                trades.extend(rows)
                for cost in [0,5,10,20]:
                    net=sum(.2*(r['gross']-cost/1e4) for r in rows)
                    daily.append({'entry_date':day,'exit_date':nxt,'strategy':model,'exit_mode':exitmode,'cost_bps':cost,'picks':len(selected),'exposure':.2*len(selected),'net':net})
    daily=pd.DataFrame(daily);trades=pd.DataFrame(trades)
    daily.to_csv(P/'daily_portfolios.csv',index=False);trades.to_csv(P/'trades.csv',index=False);pd.DataFrame(picks).to_csv(P/'daily_selection.csv',index=False)
    summaries=[]
    for (model,exitmode,cost),f in daily.groupby(['strategy','exit_mode','cost_bps']):
        active=f[f.picks>0];eq=(1+f.net).cumprod();peak=np.maximum.accumulate(np.r_[1.,eq])[1:]
        t=trades[(trades.strategy==model)&(trades.exit_mode==exitmode)] if len(trades) else pd.DataFrame()
        n=len(active);wins=int((active.net>0).sum());phat=wins/n if n else np.nan
        if n:
            denom=1+1.96**2/n;mid=(phat+1.96**2/(2*n))/denom;rad=1.96*math.sqrt(phat*(1-phat)/n+1.96**2/(4*n*n))/denom
        else:mid=rad=np.nan
        summaries.append({'strategy':model,'exit_mode':exitmode,'cost_bps':cost,'sessions':len(f),'active_days':n,'trades':len(t),'win_days':wins,
          'portfolio_win':phat,'trade_win':float((t.gross-cost/1e4>0).mean()) if len(t) else None,
          'cumulative_return':float(eq.iloc[-1]-1),'max_drawdown':float((eq/peak-1).min()),'worst_day':float(f.net.min()),
          'mean_active_day_bps':float(active.net.mean()*1e4) if n else None,'win_wilson_low':mid-rad,'win_wilson_high':mid+rad})
    summary=pd.DataFrame(summaries);summary.to_csv(P/'summary.csv',index=False)
    audit={'generated_utc':datetime.now(timezone.utc).isoformat(),'downloaded_instruments':len(bars),'minute_rows':sum(len(h) for h in bars.values()),
      'entry_start':sample[0],'entry_end':sample[-1],'last_exit':day_next[sample[-1]],'entry_sessions':len(sample),
      'source_quality_warnings':qc,'selected_exit_exclusions':exclusions,'earnings_coverage_symbols':sum(earnings_coverage.values()),
      'limitations':['~60 recent sessions only','Current-universe/sector hindsight','Earnings dates retrospective, not point-in-time','No negative-news exclusion','5m VWAP approximation','Auction fill/slippage not observed','Wilson interval treats portfolio nights as independent; serial correlation can widen uncertainty']}
    (P/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2));print(summary.query("cost_bps==5 and exit_mode=='open'").to_string(index=False))
if __name__=='__main__':main()
