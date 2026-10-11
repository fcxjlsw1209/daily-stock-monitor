from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'round3'))
PERIODS={'all':('2019-01-01','2026-09-18'),'early':('2019-01-01','2021-12-31'),'middle':('2022-01-01','2023-12-31'),'late':('2024-01-01','2026-09-18')}
from features import all_features
from engine import initial_state,plan,execute
from backtest_factors import load_inputs,summarize

def adjust(decision,state,prev,close,ma,vol,variant):
 import copy
 d=copy.deepcopy(decision)
 if 'fast' in variant:
  sold={x['symbol'] for x in d['sells']}
  for s in state['positions']:
   if s not in sold and close.at[prev,s]<ma.at[prev,s]:
    d['sells'].append({'symbol':s,'reason':'close_below_sma10'})
  # Early exits free capacity on the following decision, not retroactively in today's ranking.
  d['buys']=[b for b in d['buys'] if close.at[prev,b['symbol']]>=ma.at[prev,b['symbol']]]
 if 'vol' in variant:
  for b in d['buys']:
   sigma=float(vol.at[prev,b['symbol']])
   if not np.isfinite(sigma) or sigma<=0:raise ValueError('Invalid lagged volatility')
   b['budget']*=min(1.,.25/sigma)
 return d

def backtest(data,f,raw,variant,cost):
 state=initial_state();daily=[];trades=[];dates=data.close.index
 ma=raw['adj_close'].rolling(10).mean();vol=raw['adj_close'].pct_change(fill_method=None).rolling(60).std()*np.sqrt(252)
 cfg={'exit':'fixed20','allocation':.2,'max_positions':5,'sector_cap':2}
 for day in dates[(dates>='2019-01-01')&(dates<='2026-09-18')]:
  prev=dates[dates.get_loc(day)-1]
  d=plan(state,day,dates,data.close,f['ma5'],f['atr'],f['masks']['trend'],f['score'],data.sector,cfg)
  d=adjust(d,state,prev,raw['adj_close'],ma,vol,variant)
  needed=set(state['positions'])|{b['symbol'] for b in d['buys']}
  bars={s:{'open':data.opening.at[day,s],'close':data.close.at[day,s],'dividends':data.dividends.at[day,s]} for s in needed}
  state,di,t,_,_=execute(state,d,bars,dates,cost,data.sector);daily.append(di);trades.extend(t)
 residual=state['equity']-100000-sum(x['net_pnl'] for x in trades)-sum(p['qty']*data.close.at[day,s]+p['dividends']-p['cost'] for s,p in state['positions'].items())
 assert abs(residual)<.001
 return pd.DataFrame(daily),pd.DataFrame(trades)

if __name__=='__main__':
 data,raw=load_inputs();features=all_features(data,raw);rows=[]
 for family in ['accel','high52']:
  for variant in ['baseline','vol','fast','vol_fast']:
   for cost in [10,20]:
    d,t=backtest(data,features[family],raw,variant,cost)
    if variant=='baseline':
     old=pd.read_csv(P.parent/'round3'/f'{family}_fixed20_{cost}_daily.csv')
     np.testing.assert_allclose(d.equity,old.equity,rtol=1e-12)
    for split,(a,b) in PERIODS.items():
     rows.append(dict(family=family,variant=variant,cost_bps=cost,split=split,**summarize(d,t,a,b),mean_exposure=d.loc[d.date.between(a,b),'exposure'].mean()))
    d.to_csv(P/f'{family}_{variant}_{cost}_daily.csv',index=False);t.to_csv(P/f'{family}_{variant}_{cost}_trades.csv',index=False)
    pd.DataFrame(rows).to_csv(P/'metrics.csv',index=False)
    print(family,variant,cost,rows[-4]['cagr'],rows[-4]['max_drawdown'],flush=True)
 (P/'validation.json').write_text(json.dumps({'baseline_reproduction':'passed','accounting':'passed','trials':16,'new_variants':12,'pristine_holdout':False}))
