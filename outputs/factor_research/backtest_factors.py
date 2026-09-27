"""Frozen, daily individual-stock factor study; reuses audited portfolio engine."""
from pathlib import Path
import sys, sqlite3, json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'swing_research'))
from backtest_swing import load as load_portfolio, signals, simulate, summarize, MAP, DB, DATA
FACTORS=['long_momentum','relative_strength','pullback','stabilization','stability']
WEIGHTS=pd.Series([.30,.20,.25,.15,.10],index=FACTORS)
SPLITS={'all':('2019-01-01','2026-09-18'),'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18')}


def load_inputs():
    data=load_portfolio()
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
        rows=pd.read_sql_query('SELECT symbol,trade_date,adj_close,high,low FROM daily_prices',db)
    rows.trade_date=pd.to_datetime(rows.trade_date)
    raw={k:rows.pivot(index='trade_date',columns='symbol',values=k).sort_index() for k in ['adj_close','high','low']}
    sectors={}
    for e in MAP.values():
        h=pd.read_csv(DATA/'raw'/f'{e}_daily.csv',index_col=0)
        h.index=pd.to_datetime(h.index,utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        sectors[e]=h['Adj Close'].reindex(data.close.index)
    raw['sector_adjusted']=pd.DataFrame(sectors)
    return data,raw


def build(data,raw):
    a=raw['adj_close'];cols=a.columns;c=data.close[cols]
    r=a.pct_change(fill_method=None);r3=a.pct_change(3,fill_method=None);r60=a.pct_change(60,fill_method=None)
    z=r3/(r.rolling(20).std()*np.sqrt(3))
    mom=a.shift(21)/a.shift(252)-1;r12=a/a.shift(252)-1
    sector60=raw['sector_adjusted'].pct_change(60,fill_method=None)
    relative=pd.DataFrame({s:(1+r60[s])/(1+sector60[MAP[data.sector[s]]])-1 for s in cols})
    spread=raw['high']-raw['low']
    location=((c-raw['low'])/spread).where(spread!=0,.5)
    lr=np.log(a/a.shift(1));smooth=lr.rolling(60).sum()/lr.abs().rolling(60).sum()
    factors=dict(long_momentum=mom,relative_strength=relative,pullback=-abs(z+1.25),stabilization=location,stability=smooth)
    liquid=(c*data.volume[cols]).rolling(20).mean()>20e6
    valid=((a>0)&np.isfinite(a)).rolling(253).sum().eq(253)&liquid
    valid &= np.isfinite(raw['high'])&np.isfinite(raw['low'])&np.isfinite(c)
    valid &= (raw['low']>0)&(raw['high']>=raw['low'])&(c>=raw['low'])&(c<=raw['high'])
    valid &= np.isfinite(data.volume[cols])&(data.volume[cols]>0)&(data.opening[cols]>0)
    for v in factors.values():valid &= np.isfinite(v)
    ranks={k:v.where(valid).rank(axis=1,pct=True) for k,v in factors.items()}
    equal=sum(ranks.values())/len(FACTORS)
    weighted=sum(ranks[k]*WEIGHTS[k] for k in FACTORS)
    r12rank=r12.where(valid).rank(axis=1,pct=True)
    alternative=weighted-WEIGHTS.long_momentum*ranks['long_momentum']+WEIGHTS.long_momentum*r12rank
    regime=data.close.SPY>data.close.SPY.rolling(200).mean()
    gate=(valid&(a>a.rolling(100).mean())&(mom>0)&(r60>0)&(z>=-3)&(z<0)).mul(regime,axis=0)
    old_mask,old_score=signals(data)['stock_base']
    common=old_mask&valid
    models={'legacy':(old_mask,old_score),'legacy_common':(common,old_score),'legacy_equal':(common,equal),
      'legacy_weighted':(common,weighted),'soft_equal':(gate&(equal>=.65),equal),
      'soft_weighted':(gate&(weighted>=.65),weighted),'soft_weighted_r12':(gate&(alternative>=.65),alternative)}
    return dict(factors=factors,r12=r12,valid=valid,ranks=ranks,gate=gate,z=z,models=models)


def forward_returns(data,cols,horizon,cost_bps=5):
    """At signal t, entry t+1 open; exit t+1+h open; div t+2..t+1+h."""
    fee=cost_bps/20000
    entry=data.opening[cols].shift(-1);exit_price=data.opening[cols].shift(-(horizon+1))
    entitled=data.dividends[cols].rolling(horizon,min_periods=horizon).sum().shift(-(horizon+1))
    return (exit_price*(1-fee)+entitled)/(entry*(1+fee))-1


def diagnostics(data,built):
    gate=built['gate'];cols=gate.columns;records=[]
    values={**built['factors'],'r12':built['r12']}
    for horizon in [3,5,10]:
        target=forward_returns(data,cols,horizon)
        for name,factor in values.items():
            eligible=gate&np.isfinite(target)
            x=factor.where(eligible);y=target.where(eligible)
            n=x.count(axis=1);xr=x.rank(axis=1,pct=True);yr=y.rank(axis=1,pct=True)
            ic=xr.corrwith(yr,axis=1)
            top=y.where(xr>.8).mean(axis=1);bottom=y.where(xr<=.2).mean(axis=1)
            selected=(n>=10)&(x.index>='2019-01-01')&(x.index<='2026-09-18')
            table=pd.DataFrame({'date':x.index.strftime('%Y-%m-%d'),'factor':name,'horizon':horizon,'stocks':n.values,
              'ic':ic.values,'top_net':top.values,'bottom_net':bottom.values,'spread':(top-bottom).values})
            records.append(table.loc[selected.values])
    daily=pd.concat(records,ignore_index=True);summary=[]
    for (factor,h),g in daily.groupby(['factor','horizon']):
        for split,(a,b) in SPLITS.items():
            sub=g[(g.date>=a)&(g.date<=b)]
            summary.append(dict(factor=factor,horizon=h,split=split,days=len(sub),stock_labels=int(sub.stocks.sum()),
              mean_ic=sub.ic.mean(),positive_ic_fraction=(sub.ic>0).mean(),top_mean_net=sub.top_net.mean(),
              bottom_mean_net=sub.bottom_net.mean(),top_minus_bottom=sub.spread.mean()))
    daily.to_csv(P/'factor_diagnostics_daily.csv.gz',index=False,compression='gzip')
    pd.DataFrame(summary).to_csv(P/'factor_diagnostics_summary.csv',index=False)
    # Average within-day Spearman correlation in the same gate, avoiding pooled time trends.
    corr=[]
    for j,k1 in enumerate(values):
        for k2 in list(values)[j+1:]:
            x=values[k1].where(gate);y=values[k2].where(gate)
            z=x.rank(axis=1).corrwith(y.rank(axis=1),axis=1)
            z=z[(gate.sum(axis=1)>=10)&(z.index>='2019-01-01')]
            corr.append(dict(factor1=k1,factor2=k2,days=int(z.count()),mean_daily_spearman=z.mean()))
    pd.DataFrame(corr).to_csv(P/'factor_correlations.csv',index=False)


def turnover(d,o,a,b):
    sample=d[(d.date>=a)&(d.date<=b)]
    selected=o[(o.date>=a)&(o.date<=b)]
    years=((pd.Timestamp(sample.date.iloc[-1])-pd.Timestamp(sample.date.iloc[0])).days+1)/365.25
    value=float((selected.price*selected.quantity).sum())
    return {'annual_one_way_turnover':value/(2*sample.equity.mean()*years)}


def run():
    data,raw=load_inputs();built=build(data,raw)
    diagnostics(data,built)
    metrics=[];yearly=[];daily=[];trades=[];orders=[];opens=[];counts=[]
    for name,(mask,score) in built['models'].items():
        count=mask.loc['2019-01-01':'2026-09-18'].sum(axis=1)
        counts.append(dict(strategy=name,mean_daily_candidates=count.mean(),zero_candidate_days=int((count==0).sum()),
          signal_days=len(count),candidate_stock_days=int(count.sum())))
        for cost in [5,10,20]:
            d,t,o,u=simulate(data,mask,score,'fixed5',cost)
            for split,(a,b) in SPLITS.items():metrics.append(dict(strategy=name,cost_bps=cost,split=split,**summarize(d,t,a,b),**turnover(d,o,a,b)))
            for year in range(2019,2027):yearly.append(dict(strategy=name,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31'),**turnover(d,o,f'{year}-01-01',f'{year}-12-31')))
            for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
            daily.append(d);trades.append(t);orders.append(o);opens.extend(dict(v,strategy=name,cost_bps=cost) for v in u)
            print(name,cost,'trades',len(t),'ending equity',round(d.equity.iloc[-1],2),flush=True)
    pd.DataFrame(metrics).to_csv(P/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    pd.concat(daily).to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'trades.csv.gz',index=False,compression='gzip')
    pd.concat(orders).to_csv(P/'orders.csv.gz',index=False,compression='gzip')
    pd.DataFrame(opens).to_csv(P/'open_positions.csv',index=False)
    pd.DataFrame(counts).to_csv(P/'candidate_counts.csv',index=False)
    latest=pd.DataFrame({k:r.iloc[-1] for k,r in built['ranks'].items()})
    for k,(m,s) in built['models'].items():latest[k+'_score']=s.iloc[-1];latest[k+'_eligible']=m.iloc[-1]
    latest['signal_date']=str(data.close.index[-1].date());latest['sector']=pd.Series(data.sector)
    latest.sort_values('soft_weighted_score',ascending=False).to_csv(P/'latest_research_scores.csv',index_label='symbol')
    sources=[DB,P/'protocol.md',P/'backtest_factors.py',P.parent/'swing_research'/'backtest_swing.py',
      P.parent/'overnight_research'/'spy_source.csv',P.parent/'swing_research'/'qqq_source.csv',DATA/'download_manifest.json']
    sources+=sorted((DATA/'raw').glob('*_daily.csv'))
    hashes={str(x.relative_to(P.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    (P/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    audit={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'start':'2019-01-02','end':'2026-09-18',
      'symbols':len(raw['adj_close'].columns),'weights':WEIGHTS.to_dict(),'score_floor':.65,'max_positions':5,'sector_cap':2,
      'learned_weights':False,'universe_point_in_time':False,'heldout_test':False,'diagnostics_min_stocks':10,
      'rankable_stock_days':int(built['valid'].loc['2019':].sum().sum()),'candidate_gate_stock_days':int(built['gate'].loc['2019':].sum().sum())}
    (P/'audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(metrics).query("cost_bps==5 and split=='all'")[['strategy','closed_trades','cagr','trade_win','max_drawdown','average_exposure']].to_string(index=False))

if __name__=='__main__':run()
