from pathlib import Path
import sys,json,hashlib
from dataclasses import replace
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'strategy_selection'))
from research import simulate,load_inputs,features,summarize,SPLITS

def models_for(data,raw):
 f=features(data,raw);r=raw['adj_close'].pct_change(fill_method=None)
 c=data.close[raw['adj_close'].columns];span=raw['high']-raw['low']
 clv=((2*c-raw['high']-raw['low'])/span).where(span!=0,0.)
 signals={'clv':clv.diff(),'momentum':r/r.shift(1).rolling(20).std()}
 filters={'clv':(clv>0)&(clv.diff()>0),'momentum':r>0}
 pressure=(-r.clip(upper=0))*data.volume[c.columns]/data.volume[c.columns].shift(1).rolling(20).mean()
 decay=pressure.shift(1).rolling(3).mean()-pressure
 valid=f['base']['valid']&np.isfinite(f['cmf']);mom=f['base']['ranks']['long_momentum']
 shadow=.30*mom+(6/7)*(f['score']-.30*mom)+.10*decay.where(valid&np.isfinite(decay)).rank(axis=1,pct=True)
 parents={'baseline':(f['masks']['broad'],f['score']),'shadow':(f['masks']['broad']&np.isfinite(decay),shadow)}
 models={};lost=[]
 for parent,(mask,score) in parents.items():
  models[parent]=(mask,score,parent)
  for signal,values in signals.items():
   good=valid&np.isfinite(values)
   lost.append(dict(parent=parent,signal=signal,lost_candidate_days=int((mask&~good).to_numpy().sum())))
   ranked=.30*mom+(6/7)*(score-.30*mom)+.10*values.where(good).rank(axis=1,pct=True)
   models[parent+'_'+signal+'_rank']=(mask&good,ranked,parent)
   models[parent+'_'+signal+'_filter']=(mask&good&filters[signal],score,parent)
 return f,models,signals,filters,lost,decay

def hashes():
 paths=[]
 for name in ['daily_stock_monitor','pressure_shadow_monitor','paired_stock_monitor']:
  paths.extend(p for p in (P.parent/name).iterdir() if p.is_file())
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def run():
 before=hashes();data,raw=load_inputs();f,models,signals,filters,lost,decay=models_for(data,raw)
 for end in ['2021-12-31','2024-12-31']:
  short=replace(data,**{k:getattr(data,k).loc[:end] for k in ['close','opening','dividends','volume','sector_prices']})
  sr={k:v.loc[:end] for k,v in raw.items()};_,sm,*_=models_for(short,sr)
  for name,(mask,score,parent) in models.items():
   pd.testing.assert_frame_equal(mask.loc[:end],sm[name][0]);pd.testing.assert_frame_equal(score.loc[:end],sm[name][1])
 reference=pd.read_csv(P.parent/'context_factor_research/metrics.csv')
 metrics=[];yearly=[];daily=[];trades=[];orders=[];cache={};diag=[]
 for name,(mask,score,parent) in models.items():
  assert not (mask&~models[parent][0]).any().any()
  for cost in [10,20]:
   d,t,o,state=simulate(data,{**f,'masks':{'trial':mask},'score':score},'trial','guarded10',cost)
   assert d.positions.max()<=5 and d.cash.min()>=-1e-7 and t.sessions.max()<=10
   cashflow=pd.Series(np.where(o.side=='buy',-o.quantity*o.price-o.fee,o.quantity*o.price-o.fee),index=o.date).groupby(level=0).sum().reindex(d.date,fill_value=0)
   np.testing.assert_allclose(d.cash,100000+cashflow.cumsum().to_numpy()+d.dividends_cumulative,atol=1e-6)
   np.testing.assert_allclose(d.equity,d.cash+d.invested,atol=1e-6)
   for order in o.itertuples():np.testing.assert_allclose(order.price,data.opening.at[pd.Timestamp(order.date),order.symbol])
   for split,(lo,hi) in SPLITS.items():
    row=dict(strategy=name,parent=parent,cost_bps=cost,split=split,**summarize(d,t,lo,hi));metrics.append(row)
    if name in ['baseline','shadow']:
     refname='baseline' if name=='baseline' else 'pressure_decay'
     ref=reference[(reference.strategy==refname)&(reference.cost_bps==cost)&(reference.split==split)].iloc[0]
     for key in ['closed_trades','cagr','trade_win','max_drawdown']:np.testing.assert_allclose(row[key],ref[key],atol=1e-10)
   for year in range(2019,2027):yearly.append(dict(strategy=name,parent=parent,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
   cache[(name,cost)]=(d.copy(),t.copy())
   for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
   daily.append(d);trades.append(t);orders.append(o)
   print(name,cost,len(t),round(state['equity'],2),flush=True)
 m=pd.DataFrame(metrics);m.to_csv(P/'metrics.csv',index=False);pd.DataFrame(yearly).to_csv(P/'yearly.csv',index=False)
 for name,frames in [('daily_equity',daily),('trades',trades),('orders',orders)]:pd.concat(frames).to_csv(P/(name+'.csv.gz'),index=False,compression='gzip')
 rng=np.random.default_rng(42);n=len(cache[('baseline',10)][0]);idx=(rng.integers(0,n-19,size=(1000,int(np.ceil(n/20))))[:,:,None]+np.arange(20)).reshape(1000,-1)[:,:n]
 intervals=[];screen=[]
 for name,(mask,score,parent) in models.items():
  if name==parent:continue
  passed=True
  for cost in [10,20]:
   diff=cache[(name,cost)][0]['return'].to_numpy()-cache[(parent,cost)][0]['return'].to_numpy();boot=diff[idx].mean(axis=1)*252
   intervals.append(dict(strategy=name,parent=parent,cost_bps=cost,annualized_arithmetic_difference=diff.mean()*252,lower95=np.quantile(boot,.025),upper95=np.quantile(boot,.975)))
   a=m[(m.strategy==name)&(m.cost_bps==cost)].set_index('split');b=m[(m.strategy==parent)&(m.cost_bps==cost)].set_index('split')
   passed &= bool(a.loc['all','cagr']>b.loc['all','cagr'] and a.loc['all','max_drawdown']>b.loc['all','max_drawdown'] and a.loc['all','closed_trades']>=.7*b.loc['all','closed_trades'] and (a.loc[['earlier','recent'],'cagr']>=b.loc[['earlier','recent'],'cagr']).all())
   if name.endswith('_filter'):
    t=cache[(parent,cost)][1];rejected=[]
    for tr in t.itertuples():
     prev=data.close.index[data.close.index.get_loc(pd.Timestamp(tr.entry_date))-1]
     rejected.append(not bool(mask.at[prev,tr.symbol]))
    rejected=np.array(rejected);tt=t.loc[rejected]
    diag.append(dict(strategy=name,cost_bps=cost,parent_trades=len(t),rejected=len(tt),rejected_winners=int((tt.net_pnl>0).sum()),rejected_losses=int((tt.net_pnl<0).sum()),rejected_mean_return=tt.net_return.mean(),retained_parent_win=(t.loc[~rejected,'net_pnl']>0).mean()))
  screen.append(dict(strategy=name,passes_historical_screen=passed))
 pd.DataFrame(intervals).to_csv(P/'paired_intervals.csv',index=False);pd.DataFrame(diag).to_csv(P/'filter_diagnostics.csv',index=False)
 corr=[]
 for name,val in signals.items():
  for othername,other in [('pressure_decay',decay),('cmf',f['cmf'])]:
   a=val.where(models['baseline'][0]).rank(axis=1);b=other.where(models['baseline'][0]).rank(axis=1)
   corr.append(dict(signal=name,other=othername,mean_daily_spearman=a.corrwith(b,axis=1).mean()))
 pd.DataFrame(corr).to_csv(P/'correlations.csv',index=False)
 (P/'screen.json').write_text(json.dumps(screen,indent=2));assert before==hashes()
 (P/'validation.json').write_text(json.dumps({'prefix_causality':True,'both_parents_reproduced':True,'next_open_fills':True,'cash_equity':True,'capacity_and_max_hold':True,'live_files_unchanged':True,'lost_candidates':lost,'protected_hashes':before},indent=2))
 print(m.query("cost_bps==10 and split=='all'")[['strategy','cagr','trade_win','max_drawdown','closed_trades','average_exposure']].to_string(index=False),flush=True)
if __name__=='__main__':run()
