from pathlib import Path
import sys,json
from dataclasses import replace
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'recovery_confirmation_research'))
from research_recovery import models_for,hashes,load_inputs,summarize,SPLITS
from engine import initial_state,plan,execute

def resistance(data,raw,f):
 r=raw['adj_close'].pct_change(fill_method=None)
 # Parent pipeline uses split-adjusted SPY; obtain total-return market via reconstructed adjusted sector source when available.
 market=data.close.SPY.pct_change(fill_method=None)
 # Include known cash dividends for SPY total-return estimate; both terms known by close.
 market=(data.close.SPY+data.dividends.SPY)/data.close.SPY.shift(1)-1
 beta=r.shift(1).rolling(60).cov(market.shift(1)).div(market.shift(1).rolling(60).var(),axis=0)
 residual=r-beta.mul(market,axis=0)
 return residual.where(pd.DataFrame(np.repeat((market<0).to_numpy()[:,None],len(r.columns),axis=1),index=r.index,columns=r.columns)).rolling(20,min_periods=3).mean()

def hold_plan(state,day,dates,data,f,mask,score):
 i=dates.get_loc(day);prev=dates[i-1];sells=[];remaining=set(state['positions'])
 for symbol,p in state['positions'].items():
  age=i-dates.get_loc(pd.Timestamp(p['entry_date']));close=data.close.at[prev,symbol];reason=None
  if p.get('reclaimed',False) and close<f['ma5'].at[prev,symbol]:reason='lost_sma5_after_reclaim'
  if close<p['entry_price']-2*p['atr']:reason='close_below_2atr'
  if age>=10:reason=reason or 'maximum_sessions'
  if reason:sells.append(dict(symbol=symbol,reason=reason));remaining.remove(symbol)
 sold={x['symbol'] for x in sells};buys=[]
 for symbol,value in score.loc[prev].where(mask.loc[prev]).dropna().sort_values(ascending=False,kind='stable').items():
  if len(remaining)>=5:break
  if symbol in remaining or symbol in sold:continue
  if sum(data.sector[s]==data.sector[symbol] for s in remaining)>=2:continue
  buys.append(dict(symbol=symbol,budget=state['equity']*.1,score=float(value),atr=float(f['atr'].at[prev,symbol]),reference_close=float(data.close.at[prev,symbol])));remaining.add(symbol)
 return dict(session=str(day.date()),signal_date=str(prev.date()),sells=sells,buys=buys)

def gap_buys(buys,bars,adverse=False):
 accepted=[];skipped=[]
 for buy in buys:
  symbol=buy['symbol'];cap=buy['reference_close']+.5*buy['atr']
  if bars[symbol]['open']>=cap:skipped.append(dict(symbol=symbol,open=bars[symbol]['open'],cap=cap));continue
  if adverse:bars[symbol]['open']=cap
  accepted.append(buy)
 return accepted,skipped

def simulation(data,f,mask,score,cost,mode):
 state=initial_state();ds=[];ts=[];os=[];skips=[];dates=data.close.index
 for day in dates[(dates>='2019-01-01')&(dates<='2026-09-18')]:
  if mode=='hold_exit':q=hold_plan(state,day,dates,data,f,mask,score)
  else:q=plan(state,day,dates,data.close,f['ma5'],f['atr'],mask,score,data.sector,{'exit':'guarded10','allocation':.1,'max_positions':5,'sector_cap':2})
  symbols=set(state['positions'])|{x['symbol'] for x in q['buys']}
  bars={s:dict(open=float(data.opening.at[day,s]),close=float(data.close.at[day,s]),dividends=float(data.dividends.at[day,s])) for s in symbols}
  if mode.startswith('gap'):
   q['buys'],rejected=gap_buys(q['buys'],bars,mode=='gap_adverse');skips.extend(dict(date=str(day.date()),**x) for x in rejected)
  state,d,t,o,_=execute(state,q,bars,dates,cost,data.sector)
  if mode=='hold_exit':
   for s,p in state['positions'].items():p['reclaimed']=p.get('reclaimed',False) or bool(data.close.at[day,s]>f['ma5'].at[day,s])
  ds.append(d);ts.extend(t);os.extend(o)
 d=pd.DataFrame(ds);t=pd.DataFrame(ts);o=pd.DataFrame(os)
 assert t.sessions.max()<=10 and d.positions.max()<=5 and d.cash.min()>=-1e-7
 flow=pd.Series(np.where(o.side=='buy',-o.quantity*o.price-o.fee,o.quantity*o.price-o.fee),index=o.date).groupby(level=0).sum().reindex(d.date,fill_value=0)
 np.testing.assert_allclose(d.cash,100000+flow.cumsum().to_numpy()+d.dividends_cumulative,atol=1e-6)
 np.testing.assert_allclose(d.equity,d.cash+d.invested,atol=1e-6)
 unrealized=sum(p['qty']*data.close.at[dates[-1],s]+p['dividends']-p['cost'] for s,p in state['positions'].items())
 np.testing.assert_allclose(state['equity']-100000,t.net_pnl.sum()+unrealized,atol=1e-6)
 return d,t,o,skips

def run():
 before=hashes();data,raw=load_inputs();f,parents,*_=models_for(data,raw);val=resistance(data,raw,f)
 for end in ['2021-12-31','2024-12-31']:
  short=replace(data,**{k:getattr(data,k).loc[:end] for k in ['close','opening','dividends','volume','sector_prices']})
  pd.testing.assert_frame_equal(val.loc[:end],resistance(short,{k:v.loc[:end] for k,v in raw.items()},f))
 # Known synthetic auction boundary: equality and above skipped, below filled; no backfill.
 buys=[dict(symbol='A',reference_close=100,atr=2),dict(symbol='B',reference_close=100,atr=2)]
 accepted,rejected=gap_buys(buys,{'A':{'open':101},'B':{'open':100.5}});assert [x['symbol'] for x in accepted]==['B'] and len(rejected)==1
 models={};lost=[]
 for parent in ['baseline','shadow']:
  mask,score,_=parents[parent];good=f['base']['valid']&np.isfinite(f['cmf'])&np.isfinite(val);mom=f['base']['ranks']['long_momentum']
  models[parent]=(mask,score,parent,'parent')
  models[parent+'_residual_rank']=(mask&good,.30*mom+(6/7)*(score-.30*mom)+.10*val.where(good).rank(axis=1,pct=True),parent,'rank')
  n=int((mask&~good).to_numpy().sum());lost.append(dict(parent=parent,lost=n))
  if n:models[parent+'_residual_control']=(mask&good,score,parent,'parent')
  for mode in ['gap_limit','gap_adverse','hold_exit']:models[parent+'_'+mode]=(mask,score,parent,mode)
 metrics=[];years=[];daily=[];trades=[];orders=[];skiprows=[];cache={}
 ref=pd.read_csv(P.parent/'recovery_confirmation_research/metrics.csv')
 for name,(mask,score,parent,mode) in models.items():
  for cost in [10,20]:
   d,t,o,skips=simulation(data,f,mask,score,cost,mode);cache[(name,cost)]=d['return'].to_numpy()
   for split,(lo,hi) in SPLITS.items():
    row=dict(strategy=name,parent=parent,cost_bps=cost,split=split,**summarize(d,t,lo,hi));metrics.append(row)
    if name==parent:
     old=ref[(ref.strategy==parent)&(ref.cost_bps==cost)&(ref.split==split)].iloc[0]
     for k in ['cagr','trade_win','max_drawdown','closed_trades']:np.testing.assert_allclose(row[k],old[k],atol=1e-10)
   for year in range(2019,2027):years.append(dict(strategy=name,cost_bps=cost,year=year,**summarize(d,t,f'{year}-01-01',f'{year}-12-31')))
   for frame in [d,t,o]:frame['strategy']=name;frame['cost_bps']=cost
   daily.append(d);trades.append(t);orders.append(o);skiprows.extend(dict(strategy=name,cost_bps=cost,**x) for x in skips)
   print(name,cost,len(t),round(d.equity.iloc[-1],2),flush=True)
 m=pd.DataFrame(metrics);m.to_csv(P/'metrics.csv',index=False);pd.DataFrame(years).to_csv(P/'yearly.csv',index=False);pd.DataFrame(skiprows).to_csv(P/'skipped_gaps.csv',index=False)
 for name,frames in [('daily_equity',daily),('trades',trades),('orders',orders)]:pd.concat(frames).to_csv(P/(name+'.csv.gz'),index=False,compression='gzip')
 n=len(cache[('baseline',10)]);rng=np.random.default_rng(42);idx=(rng.integers(0,n-19,size=(1000,int(np.ceil(n/20))))[:,:,None]+np.arange(20)).reshape(1000,-1)[:,:n]
 intervals=[];screens=[]
 for name,(mask,score,parent,mode) in models.items():
  if name==parent:continue
  passed=True
  for cost in [10,20]:
   delta=cache[(name,cost)]-cache[(parent,cost)];boot=delta[idx].mean(axis=1)*252
   intervals.append(dict(strategy=name,cost_bps=cost,annualized_arithmetic_difference=delta.mean()*252,lower95=np.quantile(boot,.025),upper95=np.quantile(boot,.975)))
   a=m[(m.strategy==name)&(m.cost_bps==cost)].set_index('split');b=m[(m.strategy==parent)&(m.cost_bps==cost)].set_index('split')
   passed &= bool(a.loc['all','cagr']>b.loc['all','cagr'] and a.loc['all','max_drawdown']>b.loc['all','max_drawdown'] and (a.loc[['earlier','recent'],'cagr']>=b.loc[['earlier','recent'],'cagr']).all())
   if mode=='rank':passed &= bool(a.loc['all','closed_trades']>=.7*b.loc['all','closed_trades'])
   if mode=='gap_limit':
    stress=m[(m.strategy==parent+'_gap_adverse')&(m.cost_bps==cost)&(m.split=='all')].iloc[0]
    passed &= bool(stress.cagr>b.loc['all','cagr'] and stress.max_drawdown>b.loc['all','max_drawdown'])
  screens.append(dict(strategy=name,historical_screen=passed,execution_provisional=mode.startswith('gap')))
 pd.DataFrame(intervals).to_csv(P/'paired_intervals.csv',index=False);(P/'screen.json').write_text(json.dumps(screens,indent=2))
 assert before==hashes();(P/'validation.json').write_text(json.dumps({'both_parents_reproduced':True,'prefix_causality':True,'cash_equity_pnl':True,'capacity_sector_max_hold':True,'auction_boundary':True,'live_files_unchanged':True,'lost_candidates':lost,'protected_hashes':before},indent=2))
 print(m.query("cost_bps==10 and split=='all'")[['strategy','cagr','trade_win','max_drawdown','closed_trades','mean_hold','average_exposure']].to_string(index=False),flush=True)
if __name__=='__main__':run()
