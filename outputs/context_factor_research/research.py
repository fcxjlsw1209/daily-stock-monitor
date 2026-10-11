from pathlib import Path
import sys,json,hashlib
from dataclasses import replace
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'strategy_selection'))
from research import simulate,load_inputs,features,summarize,SPLITS

def candidates(data,raw,f):
    c=data.close[raw['adj_close'].columns];v=data.volume[c.columns]
    r=raw['adj_close'].pct_change(fill_method=None)
    pressure=(-r.clip(upper=0))*v/v.shift(1).rolling(20).mean()
    decay=pressure.shift(1).rolling(3).mean()-pressure
    priorlow=raw['low'].shift(1).rolling(20).min()
    reclaim=((c-priorlow)/f['atr']).where((raw['low']<priorlow)&(c>priorlow),0.)
    reclaim=reclaim.where(priorlow.notna()&np.isfinite(f['atr'])&(f['atr']>0))
    sr=data.close.SPY.pct_change(fill_method=None);oldvol=sr.shift(1).rolling(20).std()
    stress=(sr.rolling(5).std()/oldvol<=1.5)&(sr>=-2*oldvol)
    sectors=raw['sector_adjusted'];breadth=(sectors>sectors.rolling(50).mean()).sum(axis=1)>=6
    breadth &= sectors.rolling(50).count().eq(50).all(axis=1)
    base=f['masks']['broad'];models={'baseline':(base,f['score'])}
    for name,val in {'pressure_decay':decay,'support_reclaim':reclaim}.items():
        valid=f['base']['valid']&np.isfinite(f['cmf'])&np.isfinite(val)
        rank=val.where(valid).rank(axis=1,pct=True)
        mom=f['base']['ranks']['long_momentum']
        score=.30*mom+(6/7)*(f['score']-.30*mom)+.10*rank
        models[name+'_control']=(base&valid,f['score'])
        models[name]=(base&valid,score)
    models['stress_filter']=(base.mul(stress,axis=0),f['score'])
    models['breadth_filter']=(base.mul(breadth,axis=0),f['score'])
    return models,{'pressure_decay':decay,'support_reclaim':reclaim},pressure,priorlow

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    protected=[p for p in (P.parent/'daily_stock_monitor').iterdir() if p.suffix=='.py' or p.name in ['config.json','paper_ledger.sqlite']]
    before={str(p):digest(p) for p in protected}
    data,raw=load_inputs();f=features(data,raw);models,vals,pressure,priorlow=candidates(data,raw,f)
    n=len(data.close)-150
    short=replace(data,**{k:getattr(data,k).iloc[:n] for k in ['close','opening','dividends','volume','sector_prices']})
    rawshort={k:v.iloc[:n] for k,v in raw.items()};sf=features(short,rawshort);sm,*_=candidates(short,rawshort,sf)
    for name,(mask,score) in models.items():
        pd.testing.assert_frame_equal(mask.iloc[:n],sm[name][0]);pd.testing.assert_frame_equal(score.iloc[:n],sm[name][1])
    for name in vals:
        assert not (models[name][0]&~models['baseline'][0]).any().any()
        pd.testing.assert_frame_equal(models[name][0],models[name+'_control'][0])
    pd.testing.assert_frame_equal(vals['pressure_decay'],pressure.shift(1).rolling(3).mean()-pressure)
    j=raw['low'].index.get_loc(priorlow.dropna(how='all').index[0]);pd.testing.assert_series_equal(priorlow.iloc[j],raw['low'].iloc[j-20:j].min(),check_names=False)
    metrics=[];daily=[];trades=[];orders=[];years=[];cache={}
    original=pd.read_csv(P.parent/'strategy_selection/metrics.csv')
    for name,(mask,score) in models.items():
      for cost in [10,20]:
        ff={**f,'masks':{'trial':mask},'score':score}
        d,t,o,state=simulate(data,ff,'trial','guarded10',cost)
        assert d.positions.max()<=5 and d.cash.min()>=-1e-7
        assert t.sessions.max()<=10
        movement=pd.Series(np.where(o.side=='buy',-o.quantity*o.price-o.fee,o.quantity*o.price-o.fee),index=o.date).groupby(level=0).sum().reindex(d.date,fill_value=0)
        np.testing.assert_allclose(d.cash,100000+movement.cumsum().to_numpy()+d.dividends_cumulative,atol=1e-6)
        np.testing.assert_allclose(d.equity,d.cash+d.invested,atol=1e-6)
        for split,(lo,hi) in SPLITS.items():
            row=dict(strategy=name,cost_bps=cost,split=split,**summarize(d,t,lo,hi));metrics.append(row)
            if name=='baseline':
                ref=original[(original.strategy=='broad_guarded10')&(original.cost_bps==cost)&(original.split==split)].iloc[0]
                for key in ['closed_trades','cagr','trade_win','max_drawdown']:np.testing.assert_allclose(row[key],ref[key],atol=1e-10)
        for year in range(2019,2027):years.append(dict(strategy=name,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
        cache[(name,cost)]=d['return'].to_numpy()
        for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
        daily.append(d);trades.append(t);orders.append(o)
        print(name,cost,len(t),round(state['equity'],2),flush=True)
    m=pd.DataFrame(metrics);m.to_csv(P/'metrics.csv',index=False);pd.DataFrame(years).to_csv(P/'yearly.csv',index=False)
    for name,frames in [('daily_equity',daily),('trades',trades),('orders',orders)]:pd.concat(frames).to_csv(P/(name+'.csv.gz'),index=False,compression='gzip')
    rng=np.random.default_rng(42);n=len(cache[('baseline',10)]);idx=(rng.integers(0,n-20+1,size=(1000,int(np.ceil(n/20))))[:,:,None]+np.arange(20)).reshape(1000,-1)[:,:n]
    intervals=[]
    for (name,cost),ret in cache.items():
        diff=ret-cache[('baseline',cost)];means=diff[idx].mean(axis=1)*252
        intervals.append(dict(strategy=name,cost_bps=cost,annualized_arithmetic_difference=diff.mean()*252,lower95=np.quantile(means,.025),upper95=np.quantile(means,.975)))
    pd.DataFrame(intervals).to_csv(P/'paired_intervals.csv',index=False)
    checks=[]
    for name in ['pressure_decay','support_reclaim','stress_filter','breadth_filter']:
        passed=True
        for cost in [10,20]:
            a=m[(m.strategy==name)&(m.cost_bps==cost)].set_index('split');b=m[(m.strategy=='baseline')&(m.cost_bps==cost)].set_index('split')
            passed &= bool(a.loc['all','cagr']>b.loc['all','cagr'] and a.loc['all','max_drawdown']>b.loc['all','max_drawdown'] and a.loc['all','closed_trades']>=.7*b.loc['all','closed_trades'] and (a.loc[['earlier','recent'],'cagr']>=b.loc[['earlier','recent'],'cagr']).all())
        checks.append(dict(strategy=name,passes_historical_screen=passed))
    (P/'screen.json').write_text(json.dumps(checks,indent=2))
    assert before=={str(p):digest(p) for p in protected}
    (P/'validation.json').write_text(json.dumps({'prefix_causality':True,'baseline_reproduced':True,'cash_equity_reconciled':True,'capacity_and_max_hold':True,'formula_checks':True,'original_monitor_unchanged':True,'source_hashes':before},indent=2))
    print(m.query("cost_bps==10 and split=='all'")[['strategy','cagr','trade_win','max_drawdown','closed_trades','average_exposure']].to_string(index=False))
if __name__=='__main__':run()
