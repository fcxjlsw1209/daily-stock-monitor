"""Six frozen incremental hypotheses, R12-1 preserved, no parameter fitting."""
from pathlib import Path
from functools import lru_cache
from datetime import datetime,timezone
import sys,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'factor_research'))
from backtest_factors import load_inputs,build,forward_returns,simulate,summarize,turnover,SPLITS,FACTORS
NEW=['high_proximity','low_downside','quiet_volume','range_contraction','accumulation','beta_adjusted_strength']


def load():
    data,raw=load_inputs()
    market=pd.read_csv(P.parent/'overnight_research'/'spy_source.csv',index_col=0,parse_dates=True)['Adj Close']
    raw['market_adjusted']=market.reindex(data.close.index)
    return data,raw


def new_factors(data,raw):
    a=raw['adj_close'];c=data.close[a.columns];v=data.volume[a.columns]
    r=a.pct_change(fill_method=None)
    spread=raw['high']-raw['low']
    norm_range=spread/c.shift(1)
    multiplier=((2*c-raw['high']-raw['low'])/spread).where(spread!=0,0.)
    lr=np.log(a/a.shift(1));m=np.log(raw['market_adjusted']/raw['market_adjusted'].shift(1))
    beta=lr.rolling(126).cov(m).div(m.rolling(126).var(),axis=0).shift(1)
    residual=lr-beta.mul(m,axis=0)
    factors={
      'high_proximity':a/a.rolling(252).max(),
      'low_downside':-np.sqrt(r.clip(upper=0).pow(2).rolling(63).mean()),
      'quiet_volume':-v.rolling(3).mean()/v.shift(3).rolling(20).mean(),
      'range_contraction':-norm_range.rolling(5).mean()/norm_range.shift(5).rolling(20).mean(),
      'accumulation':(multiplier*v).rolling(21).sum()/v.rolling(21).sum(),
      'beta_adjusted_strength':residual.rolling(60).sum()/(residual.rolling(60).std()*np.sqrt(60))}
    return factors,beta,residual


def combine_score(old,momentum,new):
    return .30*momentum+(6./7.)*(old-.30*momentum)+.10*new


def prepare(data,raw):
    old=build(data,raw);factors,beta,residual=new_factors(data,raw)
    common=old['valid'].copy()
    for f in factors.values():common &= np.isfinite(f)
    ranks={k:v.where(common).rank(axis=1,pct=True) for k,v in factors.items()}
    old_mask,old_score=old['models']['soft_weighted'];mask=old_mask&common
    momentum=old['ranks']['long_momentum']
    models={'baseline_original':(old_mask,old_score),'baseline_common':(mask,old_score)}
    for k in NEW:models[k]=(mask,combine_score(old_score,momentum,ranks[k]))
    models['blend_all']=(mask,combine_score(old_score,momentum,sum(ranks.values())/len(NEW)))
    return dict(old=old,factors=factors,beta=beta,residual=residual,common=common,ranks=ranks,models=models)


def residualize(values,controls):
    values=np.asarray(values,float);controls=np.asarray(controls,float)
    x=np.column_stack([np.ones(len(controls)),controls])
    return values-x@np.linalg.lstsq(x,values,rcond=None)[0]


def correlate(x,y):
    x=x-np.mean(x);y=y-np.mean(y)
    den=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/den) if den>1e-12 else np.nan


@lru_cache(maxsize=4)
def bootstrap_indices(n,reps=1000,block=20):
    rng=np.random.default_rng(20260918);block=min(block,n)
    start=rng.integers(0,n-block+1,size=(reps,int(np.ceil(n/block))))
    return (start[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:n]


def bootstrap_interval(series):
    v=np.asarray(series,float)
    if not np.isfinite(v).any():return np.nan,np.nan
    samples=v[bootstrap_indices(len(v))]
    means=np.nanmean(samples,axis=1)
    return tuple(np.nanquantile(means,[.025,.975]))


def diagnostic(data,b):
    gate=b['old']['gate']&b['common'];cols=gate.columns
    dates=gate.loc['2019-01-01':'2026-09-18'].index
    controls={k:b['old']['factors'][k] for k in FACTORS}
    records=[]
    for h in [3,5,10]:
        labels=forward_returns(data,cols,h)
        for day in dates:
            take=gate.loc[day]&np.isfinite(labels.loc[day]);symbols=cols[take]
            if len(symbols)<15:continue
            control=pd.DataFrame({k:v.loc[day,symbols] for k,v in controls.items()}).rank(pct=True).to_numpy()
            x=pd.DataFrame({k:v.loc[day,symbols] for k,v in b['factors'].items()}).rank(pct=True)
            y=labels.loc[day,symbols];yr=y.rank(pct=True).to_numpy()
            xr=residualize(x.to_numpy(),control);target_residual=residualize(yr,control)
            for j,k in enumerate(NEW):
                top=y[x[k]>.8].mean();bottom=y[x[k]<=.2].mean()
                records.append(dict(date=str(day.date()),factor=k,horizon=h,stocks=len(symbols),
                  ic=correlate(x[k].to_numpy(),yr),partial_ic=correlate(xr[:,j],target_residual),
                  top_net=top,bottom_net=bottom,spread=top-bottom))
    daily=pd.DataFrame(records);summary=[]
    for (factor,h),g in daily.groupby(['factor','horizon']):
        for split,(a,z) in SPLITS.items():
            s=g[(g.date>=a)&(g.date<=z)]
            lo,hi=(np.nan,np.nan)
            if split=='all' and h==5:
                spread=s.set_index(pd.to_datetime(s.date)).spread.reindex(dates)
                lo,hi=bootstrap_interval(spread)
            summary.append(dict(factor=factor,horizon=h,split=split,days=len(s),stock_labels=int(s.stocks.sum()),
              mean_ic=s.ic.mean(),mean_partial_ic=s.partial_ic.mean(),top_mean_net=s.top_net.mean(),bottom_mean_net=s.bottom_net.mean(),
              spread=s.spread.mean(),spread_bootstrap_low=lo,spread_bootstrap_high=hi))
    daily.to_csv(P/'diagnostic_daily.csv.gz',index=False,compression='gzip')
    pd.DataFrame(summary).to_csv(P/'diagnostic_summary.csv',index=False)
    correlations=[]
    all_factors={**controls,**b['factors']}
    for k in NEW:
        for other,v in all_factors.items():
            if other==k:continue
            r1=b['factors'][k].where(gate).rank(axis=1);r2=v.where(gate).rank(axis=1)
            corr=r1.corrwith(r2,axis=1).reindex(dates).where(gate.loc[dates].sum(axis=1)>=15)
            correlations.append(dict(factor=k,other=other,days=int(corr.count()),mean_spearman=corr.mean()))
    pd.DataFrame(correlations).to_csv(P/'correlations.csv',index=False)
    print('Factor diagnostics complete',flush=True)


def run():
    data,raw=load();b=prepare(data,raw);diagnostic(data,b)
    metrics=[];yearly=[];days=[];trades=[];orders=[];opens=[];counts=[]
    for name,(mask,score) in b['models'].items():
        count=mask.loc['2019-01-01':'2026-09-18'].sum(axis=1)
        counts.append(dict(strategy=name,candidate_stock_days=int(count.sum()),zero_candidate_days=int((count==0).sum())))
        for cost in [5,10,20]:
            d,t,o,u=simulate(data,mask,score,'fixed5',cost)
            for split,(a,z) in SPLITS.items():metrics.append(dict(strategy=name,cost_bps=cost,split=split,**summarize(d,t,a,z),**turnover(d,o,a,z)))
            for year in range(2019,2027):yearly.append(dict(strategy=name,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
            for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
            days.append(d);trades.append(t);orders.append(o);opens.extend(dict(v,strategy=name,cost_bps=cost) for v in u)
            print(name,cost,'trades',len(t),'final equity',round(d.equity.iloc[-1],2),flush=True)
    daily=pd.concat(days,ignore_index=True)
    pd.DataFrame(metrics).to_csv(P/'metrics.csv',index=False);pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    daily.to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'trades.csv.gz',index=False,compression='gzip')
    pd.concat(orders).to_csv(P/'orders.csv.gz',index=False,compression='gzip')
    pd.DataFrame(opens).to_csv(P/'open_positions.csv',index=False);pd.DataFrame(counts).to_csv(P/'candidate_counts.csv',index=False)
    paired=[]
    daily5=daily[daily.cost_bps==5].pivot(index='date',columns='strategy',values='return')
    for name in b['models']:
        excess=daily5[name]-daily5.baseline_common;lo,hi=bootstrap_interval(excess)
        paired.append(dict(strategy=name,mean_daily_excess=excess.mean(),bootstrap_low=lo,bootstrap_high=hi))
    pd.DataFrame(paired).to_csv(P/'paired_daily_return_intervals.csv',index=False)
    latest=pd.DataFrame({k:v.iloc[-1] for k,v in b['ranks'].items()})
    for k,(mask,score) in b['models'].items():latest[k+'_score']=score.iloc[-1];latest[k+'_eligible']=mask.iloc[-1]
    latest['signal_date']=str(data.close.index[-1].date());latest.to_csv(P/'latest_research_scores.csv',index_label='symbol')
    hashes=json.loads((P.parent/'factor_research'/'source_hashes.json').read_text())
    for k,v in hashes.items():
        assert hashlib.sha256((P.parent/k).read_bytes()).hexdigest()==v, 'Upstream source changed: '+k
    for path in [P/'protocol.md',P/'research.py']:
        hashes[str(path.relative_to(P.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (P/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    audit={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'cutoff':'2026-09-18','new_factors':NEW,
      'r12_minus_1_weight':.30,'new_factor_weight':.10,'new_blend_factor_weight':.10/6,'original_candidates':int(b['models']['baseline_original'][0].loc['2019':].sum().sum()),
      'common_candidates':int(b['models']['baseline_common'][0].loc['2019':].sum().sum()),'trained_weights':False,'heldout_test':False,
      'bootstrap_reps':1000,'bootstrap_block':20,'bootstrap_seed':20260918,'confidence_intervals_multiple_test_adjusted':False}
    (P/'audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(metrics).query("cost_bps==5 and split=='all'")[['strategy','closed_trades','cagr','trade_win','max_drawdown','average_exposure']].to_string(index=False))
if __name__=='__main__':run()
