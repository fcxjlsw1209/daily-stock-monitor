from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'factor_research'))
from backtest_factors import build
RECIPES={
'jump':{'mom':.5,'jump':.2,'downside':.15,'cmf':.15},
'residual':{'mom':.3,'residual':.2,'jump':.2,'downside':.15,'cmf':.15},
'high52':{'mom':.3,'high52':.2,'jump':.2,'downside':.15,'cmf':.15},
'accel':{'mom':.3,'accel':.2,'jump':.2,'downside':.15,'cmf':.15},
'volume':{'mom':.5,'contraction':.2,'downside':.15,'cmf':.15}}
def all_features(data,raw):
 b=build(data,raw);a=raw['adj_close'];c=data.close[a.columns];v=data.volume[a.columns];r=a.pct_change(fill_method=None)
 market=(data.close.SPY+data.dividends.SPY)/data.close.SPY.shift(1)-1
 beta=r.rolling(126).cov(market).div(market.rolling(126).var(),axis=0).shift(1)
 residual=r.sub(beta.mul(market,axis=0));spread=raw['high']-raw['low']
 loc=((2*c-raw['high']-raw['low'])/spread).where(spread!=0,0)
 values={'mom':a.shift(21)/a.shift(252)-1,'residual':residual.shift(21).rolling(126).mean()/residual.shift(21).rolling(126).std(),
 'high52':a/a.rolling(252).max(),'accel':a/a.shift(21)-a.shift(21)/a.shift(42),
 'jump':-r.abs().rolling(60).max()/r.abs().rolling(60).sum(),'downside':-r.clip(upper=0).rolling(60).std(),
 'cmf':(loc*v).rolling(21).sum()/v.rolling(21).sum(),'contraction':-v.rolling(5).mean()/v.rolling(20).mean()}
 tr=pd.DataFrame(np.maximum.reduce([spread.to_numpy(),abs(raw['high']-c.shift(1)).to_numpy(),abs(raw['low']-c.shift(1)).to_numpy()]),index=c.index,columns=c.columns)
 result={}
 for name,w in RECIPES.items():
  valid=b['valid'].copy()
  for k in w:valid &= np.isfinite(values[k])
  score=sum(values[k].where(valid).rank(axis=1,pct=True)*weight for k,weight in w.items())
  mask=(valid&(values['mom']>0)&(a>a.rolling(100).mean())&(score>=.65)).mul(data.close.SPY>data.close.SPY.rolling(200).mean(),axis=0)
  result[name]={'score':score,'masks':{'trend':mask},'ma5':c.rolling(20).mean(),'atr':tr.rolling(14).mean()}
 return result
