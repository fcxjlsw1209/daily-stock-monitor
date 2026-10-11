from pathlib import Path
import sys,json
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]/'factor_research'))
from backtest_factors import load_inputs,summarize
from features import all_features
from engine import initial_state,plan,execute
PERIODS={'all':('2019-01-01','2026-09-18'),'early':('2019-01-01','2021-12-31'),'middle':('2022-01-01','2023-12-31'),'late':('2024-01-01','2026-09-18')}
def backtest(data,f,rule,cost):
 state=initial_state();daily=[];trades=[];cfg={'exit':rule,'allocation':.2,'max_positions':5,'sector_cap':2};dates=data.close.index
 for day in dates[(dates>='2019-01-01')&(dates<='2026-09-18')]:
  decision=plan(state,day,dates,data.close,f['ma5'],f['atr'],f['masks']['trend'],f['score'],data.sector,cfg)
  needed=set(state['positions'])|{b['symbol'] for b in decision['buys']}
  bars={s:{'open':data.opening.at[day,s],'close':data.close.at[day,s],'dividends':data.dividends.at[day,s]} for s in needed}
  state,d,t,o,act=execute(state,decision,bars,dates,cost,data.sector);daily.append(d);trades.extend(t)
 d=pd.DataFrame(daily);t=pd.DataFrame(trades)
 residual=state['equity']-100000-sum(x['net_pnl'] for x in trades)-sum(p['qty']*data.close.at[day,s]+p['dividends']-p['cost'] for s,p in state['positions'].items())
 assert abs(residual)<.001
 return d,t
if __name__=='__main__':
 data,raw=load_inputs();f=all_features(data,raw)
 from dataclasses import replace
 cut='2023-12-29';small=replace(data,**{k:getattr(data,k).loc[:cut] for k in ['close','opening','dividends','volume','sector_prices']})
 short=all_features(small,{k:v.loc[:cut] for k,v in raw.items()})
 for n in f:
  for k in ['score','atr','ma5']:pd.testing.assert_frame_equal(f[n][k].loc[:cut],short[n][k])
  pd.testing.assert_frame_equal(f[n]['masks']['trend'].loc[:cut],short[n]['masks']['trend'])
 rows=[];years=[]
 for n in f:
  for rule in ['fixed5','fixed20','trend20']:
   for cost in [10,20]:
    d,t=backtest(data,f[n],rule,cost)
    for split,(a,b) in PERIODS.items():rows.append(dict(family=n,exit=rule,cost_bps=cost,split=split,**summarize(d,t,a,b)))
    for y in range(2019,2027):years.append(dict(family=n,exit=rule,cost_bps=cost,year=y,**summarize(d,t,f'{y}-01-01',f'{y}-12-31')))
    d.to_csv(P/f'{n}_{rule}_{cost}_daily.csv',index=False)
    t.to_csv(P/f'{n}_{rule}_{cost}_trades.csv',index=False)
    pd.DataFrame(rows).to_csv(P/'metrics.csv',index=False);pd.DataFrame(years).to_csv(P/'yearly.csv',index=False)
    print(n,rule,cost,rows[-4]['cagr'],rows[-4]['max_drawdown'],flush=True)
 (P/'validation.json').write_text(json.dumps({'prefix_invariance':'passed','accounting':'passed','trials':30,'point_in_time':False,'pristine_holdout':False}))
