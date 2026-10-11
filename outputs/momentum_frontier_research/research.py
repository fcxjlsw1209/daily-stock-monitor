from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'factor_research'))
from backtest_factors import load_inputs,build,simulate,summarize

def candidates(data,raw):
 b=build(data,raw);a=raw['adj_close'];r=a.pct_change(fill_method=None)
 market=data.close.SPY.pct_change(fill_method=None)
 beta=r.rolling(126).cov(market).div(market.rolling(126).var(),axis=0).shift(1)
 residual=r.sub(beta.mul(market,axis=0))
 residual_mom=residual.shift(21).rolling(126).mean()/residual.shift(21).rolling(126).std()
 continuous=(r.gt(0).astype(float)-r.lt(0).astype(float)).shift(21).rolling(126).mean()
 downside=-r.clip(upper=0).rolling(60).std()
 jump=-r.abs().rolling(60).max()/r.abs().rolling(60).sum()
 spread=raw['high']-raw['low'];loc=((2*data.close[a.columns]-raw['high']-raw['low'])/spread).where(spread!=0,0)
 cmf=(loc*data.volume[a.columns]).rolling(21).sum()/data.volume[a.columns].rolling(21).sum()
 values={'mom':b['factors']['long_momentum'],'residual':residual_mom,'continuous':continuous,'downside':downside,'jump':jump,'cmf':cmf}
 valid=b['valid'].copy()
 for v in values.values():valid &= np.isfinite(v)
 ranks={k:v.where(valid).rank(axis=1,pct=True) for k,v in values.items()}
 recipes={'quality':{'mom':.5,'continuous':.2,'downside':.15,'cmf':.15},'residual':{'mom':.3,'residual':.2,'continuous':.2,'downside':.15,'cmf':.15},'jumps':{'mom':.5,'jump':.2,'downside':.15,'cmf':.15},'defensive':{'mom':.5,'downside':.3,'cmf':.2}}
 trend=(valid&(values['mom']>0)&(a>a.rolling(100).mean())).mul(data.close.SPY>data.close.SPY.rolling(200).mean(),axis=0)
 models={}
 for name,w in recipes.items():
  assert sum(w.get(k,0) for k in ['mom','residual'])<=.5+1e-9
  score=sum(ranks[k]*v for k,v in w.items())
  for gate,mask in [('pullback',b['gate']&valid),('trend',trend)]:models[name+'_'+gate]=(mask&(score>=.65),score)
 return models

def main():
 data,raw=load_inputs();models=candidates(data,raw)
 # Prefix invariance: candidate signals must not use future observations.
 cut=pd.Timestamp('2023-12-29')
 from dataclasses import replace
 small=replace(data,**{k:getattr(data,k).loc[:cut] for k in ['close','opening','dividends','volume','sector_prices']})
 prefix=candidates(small,{k:v.loc[:cut] for k,v in raw.items()})
 for k,(mask,score) in models.items():
  pd.testing.assert_frame_equal(mask.loc[:cut],prefix[k][0]);pd.testing.assert_frame_equal(score.loc[:cut],prefix[k][1])
 metrics=[];curves=[]
 periods={'all':('2019-01-01','2026-09-18'),'early':('2019-01-01','2021-12-31'),'middle':('2022-01-01','2023-12-31'),'late':('2024-01-01','2026-09-18')}
 for name,(mask,score) in models.items():
  for cost in [10,20]:
   d,t,o,u=simulate(data,mask,score,'fixed5',cost)
   for split,(start,end) in periods.items():metrics.append(dict(strategy=name,cost_bps=cost,split=split,**summarize(d,t,start,end)))
   d['strategy']=name;d['cost_bps']=cost;curves.append(d)
   pd.DataFrame(metrics).to_csv(P/'metrics.csv',index=False)
   print(name,cost,metrics[-4]['cagr'],metrics[-4]['max_drawdown'],flush=True)
 for cost in [10,20]:
  mask=pd.DataFrame(True,index=data.close.index,columns=['QQQ'])
  d,t,o,u=simulate(data,mask,mask.astype(float),'buyhold',cost,etf=True,buyhold=True)
  for split,(start,end) in periods.items():metrics.append(dict(strategy='QQQ_buyhold',cost_bps=cost,split=split,**summarize(d,t,start,end)))
 pd.DataFrame(metrics).to_csv(P/'metrics.csv',index=False)
 pd.concat(curves).to_csv(P/'daily_equity.csv.gz',index=False,compression='gzip')
 (P/'audit.json').write_text(json.dumps({'prefix_invariance':'passed','untouched_holdout':False,'point_in_time_universe':False,'trials':16,'leverage':1,'max_position_budget':.2,'momentum_family_weight_cap':.5,'protocol_sha256':hashlib.sha256((P/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
