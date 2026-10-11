"""Exploratory close-to-next-open tests; rules frozen in protocol.md."""
from pathlib import Path
import sqlite3,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
DB=P.parent/'stock_database'/'stocks.sqlite'
COSTS=[0,5,10,20]
SPLITS={'development':('2011-01-01','2016-12-31'),'validation':('2017-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18'),'all':('2011-01-01','2026-09-18')}

def pick(score,eligible,k=5):
    score=score.where(eligible)
    ranks=score.rank(axis=1,ascending=False,method='first')
    chosen=(ranks<=k)
    chosen.loc[chosen.sum(axis=1)<3,:]=False
    return chosen.astype(float)/k

def metrics(r,active,trades):
    eq=np.cumprod(1+r.to_numpy());peaks=np.maximum.accumulate(np.r_[1.,eq])[1:]
    net=trades.net.to_numpy() if len(trades) else np.array([])
    return {'sessions':len(r),'active_days':int(active.sum()),'active_fraction':float(active.mean()),
      'trades':len(trades),'avg_picks_active':len(trades)/max(active.sum(),1),
      'portfolio_win_rate':float((r[active]>0).mean()) if active.any() else None,
      'trade_win_rate':float((net>0).mean()) if len(net) else None,
      'avg_trade_net_bps':float(net.mean()*1e4) if len(net) else None,
      'avg_active_day_net_bps':float(r[active].mean()*1e4) if active.any() else None,
      'cagr':float(eq[-1]**(252/len(r))-1) if len(eq) else None,
      'total_return':float(eq[-1]-1) if len(eq) else None,
      'max_drawdown':float((eq/peaks-1).min()) if len(eq) else None,
      'worst_day':float(r.min()),'worst_trade':float(net.min()) if len(net) else None,
      'profit_factor':float(net[net>0].sum()/-net[net<0].sum()) if (net<0).any() else None,
      'annualized_vol':float(r.std()*np.sqrt(252))}

def make_signals(c,o,div,vol,spy):
    r1=c.pct_change(fill_method=None);r3=c.pct_change(3,fill_method=None);r60=c.pct_change(60,fill_method=None)
    eligible=(c.rolling(200).count()>=200)&((c*vol).rolling(20).mean()>20e6)
    sd=r1.rolling(20).std();z3=r3/(sd*np.sqrt(3))
    trend=(c>c.rolling(100).mean())&(r60>0)
    strong=trend&(r60.where(eligible).rank(axis=1,pct=True)>=.5)
    pullback=strong&(r3<0)&(z3>=-2.5)&(z3<=-.25)
    overnight=(o+div)/c.shift(1)-1
    overnight_sd=overnight.rolling(63).std()
    risk_on=spy.Close>spy.Close.rolling(200).mean()
    models={
      'equal_universe':eligible.div(eligible.sum(axis=1).replace(0,np.nan),axis=0).fillna(0),
      'momentum':pick(r60,eligible),
      'reversal':pick(-r3,eligible),
      'strong_pullback':pick(-z3,eligible&pullback),
      'strong_pullback_market':pick(-z3,(eligible&pullback).mul(risk_on,axis=0)),
      'overnight_persistence':pick(overnight.rolling(63).mean()/overnight_sd,eligible),
      'defensive_trend':pick(-overnight_sd,eligible&trend)}
    return models,overnight,r60,r3,z3

def main():
    con=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    d=pd.read_sql_query('SELECT symbol,trade_date,open,close,dividends,volume FROM daily_prices',con)
    issues=pd.read_sql_query('SELECT * FROM quality_issues',con);con.close()
    d['trade_date']=pd.to_datetime(d.trade_date)
    for row in issues.itertuples():
        if row.reason=='OHLC/volume':
            # Preserve row but do not use anomalous source bar for signals/P&L.
            d.loc[(d.symbol==row.symbol)&(d.trade_date==row.trade_date),['open','close','dividends','volume']]=np.nan
    spy=pd.read_csv(P/'spy_source.csv',index_col=0,parse_dates=True)
    dates=spy.index
    def wide(key):return d.pivot(index='trade_date',columns='symbol',values=key).reindex(dates)
    c,o,div,vol=map(wide,['close','open','dividends','volume'])
    models,overnight,r60,r3,z3=make_signals(c,o,div,vol,spy)
    results=[];yearly=[];daily=[];trade_out=[];selections=[];invalid_executions=[]
    future_return=overnight.shift(-1)
    future_price=o.shift(-1)/c-1
    exit_dates=pd.Series(dates,index=dates).shift(-1)
    for name,signal_weights in models.items():
        # All selection inputs known before the entry session begins.
        w=signal_weights.shift(1).fillna(0)
        valid=(dates>=pd.Timestamp('2011-01-01'))&exit_dates.notna()
        w=w.loc[valid];f=future_return.loc[valid];fp=future_price.loc[valid]
        invalid=(w>0)&f.isna()
        if invalid.any().any():
            for dt,sym in invalid.stack()[lambda x:x].index:invalid_executions.append({'strategy':name,'entry_date':str(dt.date()),'symbol':sym})
        # A source-invalid selected position is not replaced with a hindsight alternative.
        # Whole affected portfolio day omitted; report explicitly.
        keep=~invalid.any(axis=1)
        w=w.loc[keep];f=f.loc[keep];fp=fp.loc[keep]
        assert (w.sum(axis=1)<=1.00000001).all()
        gross=(w*f.fillna(0)).sum(axis=1)
        inds=np.where(w.to_numpy()>0)
        t=pd.DataFrame({'entry_date':w.index[inds[0]],'symbol':w.columns[inds[1]],'weight':w.to_numpy()[inds],
          'gross':f.to_numpy()[inds],'price_only':fp.to_numpy()[inds]})
        t['exit_date']=t.entry_date.map(exit_dates);t['strategy']=name
        for cost in COSTS:
            net=gross-w.sum(axis=1)*cost/1e4
            frame=pd.DataFrame({'entry_date':w.index,'exit_date':exit_dates.loc[w.index].values,'strategy':name,'cost_bps':cost,
                'gross':gross.values,'net':net.values,'exposure':w.sum(axis=1).values,'picks':(w>0).sum(axis=1).values})
            tt=t.copy();tt['net']=tt.gross-cost/1e4
            for split,(a,b) in SPLITS.items():
                sub=frame[(frame.exit_date>=a)&(frame.exit_date<=b)]
                st=tt[(tt.exit_date>=a)&(tt.exit_date<=b)]
                results.append({'strategy':name,'cost_bps':cost,'split':split,**metrics(sub.net,sub.exposure>0,st)})
            if cost==10:
                daily.append(frame);trade_out.append(tt)
                for year,sub in frame.groupby(frame.exit_date.dt.year):
                    st=tt[tt.exit_date.dt.year==year]
                    yearly.append({'strategy':name,'year':int(year),**metrics(sub.net,sub.exposure>0,st)})
        # Latest available signal, for next session's close, not a recommendation to trade.
        last=signal_weights.iloc[-1];chosen=last[last>0]
        if name!='equal_universe':
            for sym,weight in chosen.items():selections.append({'strategy':name,'signal_as_of':str(dates[-1].date()),'symbol':sym,'weight':weight,'ret60':r60.iloc[-1][sym],'ret3':r3.iloc[-1][sym],'z3':z3.iloc[-1][sym]})
    # Same close/open execution and costs for SPY benchmark.
    spy_ret=(spy.Open.shift(-1)+spy.Dividends.shift(-1))/spy.Close-1
    good=(dates>=pd.Timestamp('2011-01-01'))&spy_ret.notna()
    for cost in COSTS:
        frame=pd.DataFrame({'entry_date':dates[good],'exit_date':exit_dates[good].values,'strategy':'SPY_overnight','cost_bps':cost,
          'gross':spy_ret[good].values,'net':spy_ret[good].values-cost/1e4,'exposure':1.,'picks':1})
        for split,(a,b) in SPLITS.items():
            sub=frame[(frame.exit_date>=a)&(frame.exit_date<=b)]
            results.append({'strategy':'SPY_overnight','cost_bps':cost,'split':split,**metrics(sub.net,sub.exposure>0,sub)})
        if cost==10:
            daily.append(frame)
            for year,sub in frame.groupby(frame.exit_date.dt.year):yearly.append({'strategy':'SPY_overnight','year':int(year),**metrics(sub.net,sub.exposure>0,sub)})
    pd.DataFrame(results).to_csv(P/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    pd.concat(daily).to_csv(P/'daily_portfolios.csv',index=False)
    # Full records reproducible; avoid huge CSV of the 100-name diagnostic benchmark.
    pd.concat([t for t in trade_out if t.strategy.iloc[0]!='equal_universe']).to_csv(P/'trades.csv.gz',index=False,compression='gzip')
    pd.DataFrame(selections).to_csv(P/'latest_research_signals.csv',index=False)
    (P/'excluded_source_days.json').write_text(json.dumps(invalid_executions,indent=2))
    cols=['strategy','split','cagr','portfolio_win_rate','trade_win_rate','avg_trade_net_bps','max_drawdown','active_fraction','worst_day']
    print(pd.DataFrame(results).query('cost_bps==10')[cols].to_string(index=False))
    print('Source-invalid portfolio days:',invalid_executions)
if __name__=='__main__':main()
