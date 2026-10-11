from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import exchange_calendars as xc
P=Path(__file__).resolve().parent
NAMES=['QQQ_hold','QLD_hold','QQQ_TQQQ_annual','MA200_QLD_QQQ','MA200_QLD_cash','MA200_QQQ_cash']

def load():
 out={s:pd.read_csv(P/'data'/f'{s}.csv',index_col=0,parse_dates=True) for s in ['QQQ','QLD','TQQQ','IRX']}
 cal=xc.get_calendar('XNYS',start='1999-01-01',end='2026-12-31').sessions
 cal=cal.drop(pd.Timestamp('2025-01-09'),errors='ignore')
 for s,h in out.items():
  assert h.index.is_unique and h.index.is_monotonic_increasing
  if s=='IRX':continue
  needed=cal[(cal>=h.index[0])&(cal<=h.index[-1])]
  missing=needed.difference(h.index)
  if len(missing):raise ValueError(f'{s} missing sessions {list(missing)}')
  assert ((h[['Open','High','Low','Close','Adj Close']]>0).all().all())
  assert (h.Low<=h[['Open','Close']].min(axis=1)+1e-4).all()
  assert (h.High>=h[['Open','Close']].max(axis=1)-1e-4).all()
 return out

def factors(data,spread=.005,zero_cash=False):
 q=data['QQQ'];ix=q.index;days=ix.to_series().diff().dt.days
 qc=q['Adj Close'];qo=q.Open*qc/q.Close
 quoted=data['IRX'].Close/100
 rate=quoted/(1-quoted*91/360)*365/360
 # Use only yield observations strictly earlier than the equity session.
 rate=rate.reindex(rate.index.union(ix)).sort_index().ffill().reindex(ix).shift(1)
 seen=pd.Series(data['IRX'].index,index=data['IRX'].index).reindex(data['IRX'].index.union(ix)).ffill().reindex(ix).shift(1)
 assert ((ix.to_series()-seen).dt.days.dropna()<=10).all()
 on=pd.DataFrame(index=ix);intra=pd.DataFrame(index=ix)
 on['QQQ']=qo/qc.shift(1);intra['QQQ']=qc/qo
 for s,L in [('QLD',2),('TQQQ',3)]:
  cost=(L-1)*(rate+spread)+.0095
  end=1+L*(qc/qc.shift(1)-1)-cost*days/365
  opening=1+L*(qo/qc.shift(1)-1)-cost*(days-6.5/24)/365
  on[s]=opening;intra[s]=end/opening
  h=data[s].reindex(ix);c=h['Adj Close'];o=h.Open*c/h.Close
  actual=c.notna()&c.shift(1).notna()
  on.loc[actual,s]=(o/c.shift(1))[actual]
  intra.loc[actual,s]=(c/o)[actual]
 r=rate*0 if zero_cash else rate
 on['cash']=1+r*(days-6.5/24)/365;intra['cash']=1+r*(6.5/24)/365
 ma=qc.rolling(200,min_periods=200).mean()
 sig=(qc>ma).where(ma.notna()).where(qc!=ma).ffill().shift(1)
 return on,intra,sig

def rebalance(values,target,fee):
 total=values.sum();newtotal=total
 # Fee charged on actual stock dollar purchases/sales, including entry.
 for _ in range(30):
  new=total-fee*np.abs(target[:3]*newtotal-values[:3]).sum()
  if abs(new-newtotal)<1e-9:break
  newtotal=new
 return target*new, total-new

def run(on,intra,sig,name,start='2000-01-03',fee=.0005):
 ix=on.index[on.index>=pd.Timestamp(start)];v=np.array([0.,0.,0.,100000.]);rows=[];trades=[];prev=None;year=None
 for day in ix:
  assert pd.notna(sig.loc[day]),'Insufficient SMA warmup'
  if prev is not None:v*=on.loc[day,['QQQ','QLD','TQQQ','cash']].to_numpy()
  high=bool(sig.loc[day]);asset={'QQQ_hold':0,'QLD_hold':1,'MA200_QLD_QQQ':1 if high else 0,'MA200_QLD_cash':1 if high else 3,'MA200_QQQ_cash':0 if high else 3}.get(name)
  target=np.array([.5,0,.5,0]) if name=='QQQ_TQQQ_annual' else np.eye(4)[asset]
  change=prev is None or (day.year!=year if name=='QQQ_TQQQ_annual' else not np.array_equal(target,prev))
  cost=0.
  if change:
   before=v.copy();v,cost=rebalance(v,target,fee)
   trades.append({'date':str(day.date()),'target':target.tolist(),'fee':cost,'equity_at_open_before':before.sum(),'equity_at_open_after':v.sum()})
  v*=intra.loc[day,['QQQ','QLD','TQQQ','cash']].to_numpy()
  assert np.isfinite(v).all() and (v>=0).all()
  rows.append({'date':day,'equity':v.sum(),'stock_fraction':v[:3].sum()/v.sum(),'fee':cost,'asset':int(asset) if asset is not None else 4})
  prev=target;year=day.year
 return pd.DataFrame(rows).set_index('date'),trades

def drawdown(eq):
 peak=eq.cummax().clip(lower=100000);dd=eq/peak-1;trough=dd.idxmin();prior=eq.loc[:trough]
 peakdate=prior.idxmax() if prior.max()>=100000 else None;level=peak.loc[trough]
 recovered=eq.loc[trough:];recovered=recovered[recovered>=level]
 return {'max_drawdown':float(dd.min()),'peak_date':str(peakdate.date()) if peakdate is not None else 'initial','trough_date':str(trough.date()),'recovery_date':str(recovered.index[0].date()) if len(recovered) else None}

def metrics(df,trades):
 eq=df.equity;years=((eq.index[-1]-eq.index[0]).days+6.5/24)/365.25
 m={'cagr':float((eq.iloc[-1]/100000)**(1/years)-1),'ending_equity':float(eq.iloc[-1]),'rebalances':len(trades),'fees':float(df.fee.sum()),'stock_days':float((df.stock_fraction>0).mean()),**drawdown(eq)}
 for n in [5,10]:
  r=(eq/eq.shift(n*252))**(1/n)-1;r=r.dropna()
  m[f'rolling_{n}y']={'worst':float(r.min()),'median':float(r.median()),'negative_fraction':float((r<0).mean())}
 for label,a,b in [('dotcom','2000-01-01','2003-12-31'),('gfc','2007-01-01','2009-12-31')]:
  e=eq.loc[a:b]
  if not len(e):continue
  # Normalize local equity so initial-capital anchor does not contaminate the crisis peak.
  scale=100000/e.iloc[0];d=drawdown(e*scale);trough=pd.Timestamp(d['trough_date']);level=(e*scale).loc[:trough].max()
  rec=eq.loc[trough:]*scale;rec=rec[rec>=level]
  d['recovery_date']=str(rec.index[0].date()) if len(rec) else None;m[label]=d
 return m

def main():
 data=load();on,intra,sig=factors(data);results={};equities={}
 for start in ['2000-01-03','2007-01-03','2011-01-03']:
  results[start]={}
  for n in NAMES:
   df,t=run(on,intra,sig,n,start);results[start][n]=metrics(df,t)
   if start=='2000-01-03':
    equities[n]=df.equity;df.to_csv(P/(n+'_daily.csv'));(P/(n+'_trades.json')).write_text(json.dumps(t,indent=2))
 sensitivities={}
 for label,spread,fee,zero in [('double_fee',.005,.001,False),('funding_low',0,.0005,False),('funding_high',.01,.0005,False),('zero_cash',.005,.0005,True)]:
  a,b,c=factors(data,spread,zero);sensitivities[label]={}
  for n in NAMES:
   df,t=run(a,b,c,n,fee=fee);sensitivities[label][n]=metrics(df,t)
 pd.DataFrame(equities).to_csv(P/'equity_curves.csv')
 (P/'results.json').write_text(json.dumps({'base':results,'sensitivity':sensitivities},indent=2))
 print(json.dumps(results['2000-01-03'],indent=2))
if __name__=='__main__':main()
