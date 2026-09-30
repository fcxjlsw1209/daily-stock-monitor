"""Read-only imports of frozen calculations; all outputs remain in this folder."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import hashlib
import json
from dataclasses import replace
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
SRC=Path(os.environ.get('STOCK_RESEARCH_OUTPUTS', str(P.parents[2] / 'outputs'))).expanduser().resolve()
sys.path.insert(0,str(SRC/'daily_stock_monitor'))
from strategy import features,load_inputs
from engine import initial_state,plan,execute
from backtest_factors import summarize,forward_returns,turnover


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def components(data,raw,base):
    c=data.close[raw['adj_close'].columns]
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    hist=macd-macd.ewm(span=9,adjust=False).mean()
    f=(hist-hist.shift(2))/base['atr']
    valid=base['base']['valid']&np.isfinite(base['cmf'])
    rank=f.where(valid).rank(axis=1,pct=True)
    assert np.isfinite(f.where(base['masks']['broad']).stack()).all()
    assert not (base['masks']['broad']&~np.isfinite(f)).any().any()
    mom=base['base']['ranks']['long_momentum']
    score=.30*mom+(6/7)*(base['score']-.30*mom)+.10*rank
    return f,hist,{**base,'score':score}


def adverse(trades,data,raw):
    out=[]
    for row in trades.itertuples():
        entry=pd.Timestamp(row.entry_date);end=pd.Timestamp(row.exit_date)
        lows=raw['low'].loc[(raw['low'].index>=entry)&(raw['low'].index<end),row.symbol]
        low=min(float(lows.min()),row.exit_price,row.entry_price)
        out.append(low/row.entry_price-1)
    return out


def simulate(data,raw,f,cost):
    state=initial_state();ds=[];ts=[];os=[];dates=data.close.index
    cfg={'exit':'guarded10','allocation':.10,'max_positions':5,'sector_cap':2}
    for day in dates[(dates>='2019-01-01')&(dates<='2026-09-18')]:
        q=plan(state,day,dates,data.close,f['ma5'],f['atr'],f['masks']['broad'],f['score'],data.sector,cfg)
        needed=set(state['positions'])|{x['symbol'] for x in q['buys']}
        bars={s:dict(open=data.opening.at[day,s],close=data.close.at[day,s],dividends=data.dividends.at[day,s]) for s in needed}
        state,d,t,o,_=execute(state,q,bars,dates,cost,data.sector)
        ds.append(d);ts.extend(t);os.extend(o)
    t=pd.DataFrame(ts)
    t['mae_price']=adverse(t,data,raw)
    unrealized=sum(p['qty']*data.close.at[dates[-1],s]+p['dividends']-p['cost'] for s,p in state['positions'].items())
    assert abs(state['equity']-100000-t.net_pnl.sum()-unrealized)<1e-5
    return pd.DataFrame(ds),t,pd.DataFrame(os),state


def metrics(d,t,o,a,b):
    result={**summarize(d,t,a,b),**turnover(d,o,a,b)}
    q=t[(t.exit_date>=a)&(t.exit_date<=b)]
    result.update(stop_rate=float((q.reason=='close_below_2atr').mean()),
                  stop_trades=int((q.reason=='close_below_2atr').sum()),
                  mean_mae=float(q.mae_price.mean()),p05_mae=float(q.mae_price.quantile(.05)))
    return result


def label_valid(index,horizon,end):
    exit_day=pd.Series(index,index=index).shift(-(horizon+1))
    return exit_day.notna()&(exit_day<=pd.Timestamp(end))


def block_interval(delta,block=20,reps=2000):
    arr=np.asarray(delta,float);rng=np.random.default_rng(20260929)
    n=len(arr);draw=[]
    for _ in range(reps):
        starts=rng.integers(0,n-block+1,int(np.ceil(n/block)))
        sample=np.concatenate([arr[s:s+block] for s in starts])[:n]
        draw.append(sample.mean()*252)
    return dict(annual_mean_log_advantage=float(arr.mean()*252),
                low95=float(np.quantile(draw,.025)),high95=float(np.quantile(draw,.975)),block=block,reps=reps)


def diagnose(data,raw,base,f,hist):
    mask=base['masks']['broad']
    r=raw['adj_close'].pct_change(fill_method=None);v=data.volume[r.columns]
    press=(-r.clip(upper=0))*v/v.shift().rolling(20).mean()
    press=press.shift().rolling(3).mean()-press
    market=(data.close.SPY+data.dividends.SPY)/data.close.SPY.shift()-1
    beta=r.shift().rolling(60).cov(market.shift()).div(market.shift().rolling(60).var(),axis=0)
    residual=r-beta.mul(market,axis=0)
    resil=residual.where(np.repeat((market<0).to_numpy()[:,None],len(r.columns),axis=1)).rolling(20,min_periods=3).mean()
    comp={**{k:base['base']['factors'][k] for k in ['long_momentum','relative_strength','pullback','stability']},
          'CMF21':base['cmf'],'pressure_decay':press,'resilience':resil}
    corr=[]
    for name,y in comp.items():
        ok=mask&np.isfinite(f)&np.isfinite(y)
        a=f.where(ok).rank(axis=1);b=y.where(ok).rank(axis=1)
        cr=a.corrwith(b,axis=1).where(ok.sum(axis=1)>=5).loc['2019-01-01':'2026-09-18'].dropna()
        corr.append(dict(factor=name,days=len(cr),mean_daily_spearman=cr.mean(),median_daily_spearman=cr.median()))
    pd.DataFrame(corr).to_csv(P/'correlations.csv',index=False)
    rows=[];daily=[]
    windows={'all':('2019-01-01','2026-09-18'),'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18')}
    target=forward_returns(data,mask.columns,5,cost_bps=10)
    for label,(a,b) in windows.items():
        for sample,extra in [('all_candidates',mask),('negative_hist',mask&(hist<0))]:
            eligible=extra&np.isfinite(target)&np.isfinite(f)
            valid_dates=label_valid(mask.index,5,b)&(mask.index>=a)&(mask.index<=b)
            x=f.where(eligible);y=target.where(eligible)
            ranks=x.rank(axis=1,pct=True)
            n=x.count(axis=1);ic=x.rank(axis=1).corrwith(y.rank(axis=1),axis=1)
            top=y.where(ranks>2/3).mean(axis=1);bottom=y.where(ranks<=1/3).mean(axis=1)
            z=pd.DataFrame(dict(ic=ic,top=top,bottom=bottom,spread=top-bottom,stocks=n))
            z=z.loc[valid_dates&(n>=5)].dropna()
            rows.append(dict(window=label,sample=sample,days=len(z),candidate_labels=int(z.stocks.sum()),
                             mean_ic=z.ic.mean(),top_mean_net=z.top.mean(),bottom_mean_net=z.bottom.mean(),
                             top_minus_bottom=z.spread.mean()))
            daily.append(z.assign(window=label,sample=sample).rename_axis('date').reset_index())
    pd.DataFrame(rows).to_csv(P/'factor_diagnostics.csv',index=False)
    pd.concat(daily).to_csv(P/'factor_diagnostics_daily.csv.gz',index=False,compression='gzip')


def main():
    # Files imported/read by the pure original loaders and calculation engines.
    paths=[SRC/'stock_database/stocks.sqlite',SRC/'daily_stock_monitor/strategy.py',SRC/'daily_stock_monitor/engine.py',
           SRC/'daily_stock_monitor/config.json',SRC/'pressure_shadow_monitor/strategy.py',SRC/'resilience_shadow_monitor/strategy.py',
           SRC/'factor_research/backtest_factors.py',SRC/'swing_research/backtest_swing.py',
           SRC/'overnight_research/spy_source.csv',SRC/'swing_research/qqq_source.csv',
           SRC/'overnight_research/intraday_validation/download_manifest.json']
    paths+=sorted((SRC/'overnight_research/intraday_validation/raw').glob('*_daily.csv'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (P/'input_hashes.json').write_text(json.dumps(before,indent=2))
    data,raw=load_inputs();base=features(data,raw)
    cfg=json.loads((SRC/'daily_stock_monitor/config.json').read_text())
    assert set(cfg['symbols'])==set(raw['adj_close'].columns)
    assert all(cfg['sectors'][s]==data.sector[s] for s in cfg['symbols'])
    f,hist,new=components(data,raw,base)
    pd.testing.assert_frame_equal(base['masks']['broad'],new['masks']['broad'])
    # Check all new scores using only prefixes; includes indicators and cross-sectional ranking.
    for n in [2300,3100,4100]:
        short=replace(data,**{k:getattr(data,k).iloc[:n] for k in ['close','opening','dividends','volume','sector_prices']})
        sr={k:v.iloc[:n] for k,v in raw.items()}
        bf=features(short,sr);ff,hh,nn=components(short,sr,bf)
        pd.testing.assert_frame_equal(f.iloc[:n],ff)
        pd.testing.assert_frame_equal(new['score'].iloc[:n],nn['score'])
    models={'baseline':base,'momentum':new}
    for name,directory in [('pressure','pressure_shadow_monitor'),('resilience','resilience_shadow_monitor')]:
        models[name]=module('offline_'+name,SRC/directory/'strategy.py').features(data,raw)
    diagnose(data,raw,base,f,hist)
    records=[];yearly=[];daily=[];trades=[];orders=[];states=[];results={}
    for name,feat in models.items():
        for cost in [10,20]:
            d,t,o,state=simulate(data,raw,feat,cost)
            results[(name,cost)]=(d,t,o)
            for split,(a,b) in {'all':('2019-01-01','2026-09-18'),'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18')}.items():
                records.append(dict(model=name,cost_bps=cost,window=split,**metrics(d,t,o,a,b)))
            for y in range(2019,2027):
                yearly.append(dict(model=name,cost_bps=cost,year=y,**metrics(d,t,o,f'{y}-01-01',f'{y}-12-31')))
            for frame,collection in [(d,daily),(t,trades),(o,orders)]:
                collection.append(frame.assign(model=name,cost_bps=cost))
            states.append(dict(model=name,cost_bps=cost,state=state))
            print(name,cost,'trades',len(t),'equity',round(state['equity'],2),flush=True)
    old=pd.read_csv(SRC/'strategy_selection/metrics.csv')
    for cost in [10,20]:
        a=next(x for x in records if x['model']=='baseline' and x['cost_bps']==cost and x['window']=='all')
        b=old.query("strategy=='broad_guarded10' and split=='all' and cost_bps==@cost").iloc[0]
        for k in ['cagr','max_drawdown','trade_win','closed_trades']:
            np.testing.assert_allclose(a[k],b[k],rtol=1e-10,atol=1e-12)
    pd.DataFrame(records).to_csv(P/'metrics.csv',index=False)
    pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
    pd.concat(daily).to_csv(P/'daily.csv.gz',index=False,compression='gzip')
    pd.concat(trades).to_csv(P/'trades.csv.gz',index=False,compression='gzip')
    pd.concat(orders).to_csv(P/'orders.csv.gz',index=False,compression='gzip')
    (P/'final_states.json').write_text(json.dumps(states,indent=2))
    intervals=[];overlap=[]
    for cost in [10,20]:
        d0,_,o0=results[('baseline',cost)];d1,_,o1=results[('momentum',cost)]
        delta=np.log1p(d1['return'].to_numpy())-np.log1p(d0['return'].to_numpy())
        intervals.append(dict(cost_bps=cost,**block_interval(delta)))
        neworders=set(map(tuple,o1.loc[o1.side=='buy',['date','symbol']].to_numpy()))
        for other in ['baseline','pressure','resilience']:
            _,_,oo=results[(other,cost)]
            others=set(map(tuple,oo.loc[oo.side=='buy',['date','symbol']].to_numpy()))
            overlap.append(dict(cost_bps=cost,other=other,new_buys=len(neworders),other_buys=len(others),
                                common=len(neworders&others),jaccard=len(neworders&others)/len(neworders|others)))
    pd.DataFrame(overlap).to_csv(P/'buy_overlap.csv',index=False)
    (P/'bootstrap.json').write_text(json.dumps(intervals,indent=2))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    validation=dict(original_baseline_reproduced=True,baseline_candidate_mask_identical=True,
                    prefix_checks=[2300,3100,4100],source_hashes_unchanged=True,
                    portfolios=8,no_live_ledger_access=True,new_factor='MACD histogram 2-session improvement / existing SMA ATR14',
                    universe_point_in_time=False,heldout=False)
    (P/'validation.json').write_text(json.dumps(validation,indent=2))
    print(pd.DataFrame(records).query("window=='all'")[['model','cost_bps','cagr','max_drawdown','trade_win','stop_rate','mean_mae']].round(5).to_string(index=False),flush=True)


if __name__=='__main__':main()
