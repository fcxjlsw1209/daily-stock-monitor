from pathlib import Path
import sys,json,importlib.util
import numpy as np
import pandas as pd
from dataclasses import replace
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'round3'))
spec=importlib.util.spec_from_file_location('prior_run',P.parent/'round3'/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from backtest_factors import build,load_inputs

def make(data,raw):
 b=build(data,raw);a=raw['adj_close'];r=a.pct_change(fill_method=None);c=data.close[a.columns];v=data.volume[a.columns]
 spread=raw['high']-raw['low'];loc=((2*c-raw['high']-raw['low'])/spread).where(spread!=0,0)
 vals={'mom':a.shift(21)/a.shift(252)-1,'accel':a/a.shift(21)-a.shift(21)/a.shift(42),'high52':a/a.rolling(252).max(),'relative':b['factors']['relative_strength'],'smooth':b['factors']['stability'],
 'jump':-r.abs().rolling(60).max()/r.abs().rolling(60).sum(),'downside':-r.clip(upper=0).rolling(60).std(),'cmf':(loc*v).rolling(21).sum()/v.rolling(21).sum(),
 'contraction':-v.rolling(5).mean()/v.rolling(20).mean(),'tail':r.rolling(60).quantile(.05),'volume_stability':-v.rolling(20).std()/v.rolling(20).mean()}
 supports={'balanced':{'jump':.2,'downside':.15,'cmf':.15},'flow':{'jump':.1,'downside':.1,'cmf':.3},'quiet':{'contraction':.2,'volume_stability':.15,'cmf':.15},'tail':{'tail':.25,'jump':.15,'cmf':.1}}
 recipes={}
 for secondary in ['accel','high52','relative','smooth']:
  for mw in [.2,.4]:
   for support,w in supports.items():recipes[f'{secondary}_r{int(mw*100)}_{support}']={'mom':mw,secondary:.5-mw,**w}
 for n in ['accel','high52']:recipes[n+'_control']={'mom':.3,n:.2,**supports['balanced']}
 out={};tr=pd.DataFrame(np.maximum.reduce([spread.to_numpy(),abs(raw['high']-c.shift(1)).to_numpy(),abs(raw['low']-c.shift(1)).to_numpy()]),index=c.index,columns=c.columns)
 for n,w in recipes.items():
  assert abs(sum(w.values())-1)<1e-10
  assert sum(w.get(k,0) for k in ['mom','accel','high52','relative','smooth'])<=.50000001
  valid=b['valid'].copy()
  for k in w:valid &= np.isfinite(vals[k])
  score=sum(vals[k].where(valid).rank(axis=1,pct=True)*weight for k,weight in w.items())
  mask=(valid&(vals['mom']>0)&(a>a.rolling(100).mean())&(score>=.65)).mul(data.close.SPY>data.close.SPY.rolling(200).mean(),axis=0)
  out[n]={'score':score,'masks':{'trend':mask},'ma5':c.rolling(20).mean(),'atr':tr.rolling(14).mean()}
 return out,recipes

if __name__=='__main__':
 data,raw=load_inputs();fs,recipes=make(data,raw);(P/'recipes.json').write_text(json.dumps(recipes,indent=2))
 cut='2023-12-29';small=replace(data,**{k:getattr(data,k).loc[:cut] for k in ['close','opening','dividends','volume','sector_prices']});short,_=make(small,{k:v.loc[:cut] for k,v in raw.items()})
 for n in fs:
  pd.testing.assert_frame_equal(fs[n]['score'].loc[:cut],short[n]['score']);pd.testing.assert_frame_equal(fs[n]['masks']['trend'].loc[:cut],short[n]['masks']['trend'])
 rows=[];years=[]
 for n,f in fs.items():
  for cost in [10,20]:
   d,t=prior.backtest(data,f,'fixed20',cost)
   if n.endswith('_control'):
    old=pd.read_csv(P.parent/'round3'/f'{n.split("_")[0]}_fixed20_{cost}_daily.csv');np.testing.assert_allclose(d.equity,old.equity,rtol=1e-12)
   for split,(a,b) in prior.PERIODS.items():rows.append(dict(recipe=n,cost_bps=cost,split=split,**prior.summarize(d,t,a,b)))
   for y in range(2019,2027):years.append(dict(recipe=n,cost_bps=cost,year=y,**prior.summarize(d,t,f'{y}-01-01',f'{y}-12-31')))
   d.to_csv(P/f'{n}_{cost}_daily.csv',index=False);t.to_csv(P/f'{n}_{cost}_trades.csv',index=False)
   pd.DataFrame(rows).to_csv(P/'metrics.csv',index=False);pd.DataFrame(years).to_csv(P/'yearly.csv',index=False)
   print(n,cost,round(rows[-4]['cagr'],4),flush=True)
 (P/'validation.json').write_text(json.dumps({'runs':68,'recipes':34,'controls':2,'prefix_invariance':'passed','baseline_reproduction':'passed','accounting':'passed','momentum_cap':.5,'pristine_holdout':False},indent=2))
