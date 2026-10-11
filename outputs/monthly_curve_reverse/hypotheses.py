"""Small declared rule family to explain screenshot, not optimize for 80% CAGR.
All decisions use month-end observations only; same-close zero-cost execution is
an optimistic author-convention diagnostic, NOT proof of executable fills.
"""
import json
import numpy as np,pandas as pd
from extract import P,O,START,END,read_prices
from inverse import run_month,load_observations,interpolation_matrix,mul,display

RULES={
 'r12_2':{'r12_2':1},
 'risk12_2':{'risk12_2':1},
 'r6_2':{'r6_2':1},
 'r12_1_timing_control':{'r12_1':1},
 'mix12_6':{'r12_2':.7,'r6_2':.3},
 'mix_raw_risk':{'r12_2':.5,'risk12_2':.5},
 'momentum_highvol':{'r12_2':.75,'vol63':.25},
 'momentum_high52':{'r12_2':.75,'near_high':.25},
}

def metrics(series):
 r=series.pct_change().dropna();wealth=float(series.iloc[-1]/series.iloc[0]);n=len(r)
 return dict(start=str(series.index[0].date()),end=str(series.index[-1].date()),wealth=wealth,cagr_252=float(wealth**(252/n)-1),max_dd=float((series/series.cummax()-1).min()),sharpe=float(r.mean()/r.std()*np.sqrt(252)))

def simulate(px,member,score,positive,buffer):
 idx=px.index;inds=np.flatnonzero((idx>='2016-08-31')&(idx<=END));dates=idx[inds];wealth=pd.Series(index=dates,dtype=float);wealth.iloc[0]=1;positions={};cash=1.0;orders=[]
 for i in inds:
  day=idx[i];prev=idx[i-1];vals={s:px.at[day,s] for s in positions}
  if any(not np.isfinite(v) or v<=0 for v in vals.values()):raise ValueError(f'Cannot value held position {day} {positions}')
  nav=cash+sum(q*vals[s] for s,q in positions.items());wealth.at[day]=nav
  endmonth=(i+1<len(idx) and idx[i+1].to_period('M')!=day.to_period('M'))
  if day==dates[0] or endmonth:
   eligible=member.loc[day]&positive.loc[day]&px.iloc[max(0,i-252):i+1].notna().all()
   sc=score.loc[day].where(eligible).dropna().sort_values(ascending=False,kind='stable');pool=sc.index.tolist()
   chosen=[]
   if buffer:chosen=[s for s in pool[:5] if s in positions][:3]
   chosen+= [s for s in pool if s not in chosen][:3-len(chosen)]
   if len(chosen)<2:chosen=[]
   old=set(positions);positions={s:nav/len(chosen)/px.at[day,s] for s in chosen};cash=0 if chosen else nav
   orders.append(dict(date=str(day.date()),holdings=[display(s) for s in chosen],new=[display(s) for s in chosen if s not in old],retained=[display(s) for s in chosen if s in old]))
 return wealth,orders

def fit_errors(series,cal,observed):
 x=cal['wealth']['x0']+cal['wealth']['width']*((series.index-START).days/(END-START).days).to_numpy();z=observed[(observed.x>=x[0])&(observed.x<=x[-1])].copy();xp=z.x.to_numpy();model=np.interp(xp,x,np.log(series.to_numpy()));target=z.log_wealth.to_numpy()
 # Both target and hypothesis rebased at 2016-08-31; target initial screenshot value approximate.
 ref=observed[observed.panel=='wealth'];target0=np.interp(x[0],ref.x,ref.log_wealth);r=model-(target-target0)
 date=START+pd.to_timedelta((xp-cal['wealth']['x0'])/cal['wealth']['width']*(END-START).days,unit='D')
 out={}
 for label,m in [('development',date<'2022-01-01'),('later_diagnostic',date>='2022-01-01'),('full',np.ones(len(z),bool))]:
  out[label+'_global_log_rmse']=float(np.sqrt(np.mean(r[m]**2)))
 # Monthwise normalized shape removes accumulated level but is reported separately.
 z['residual']=r;z['month']=date.to_period('M');z['is_development']=date<'2022-01-01'
 centered=z.residual-z.groupby('month').residual.transform('mean')
 out['development_month_shape_log_rmse']=float(np.sqrt(np.mean(centered[z.is_development]**2)))
 out['later_month_shape_log_rmse']=float(np.sqrt(np.mean(centered[~z.is_development]**2)))
 return out

def main():
 config=dict(rules=RULES,retention=[False,True],eligibility='Historical price-scaled index-union Top32 +17 assumed ETFs; positive R12-2, 253 continuous completed price observations. No future-completeness filter for simulated rules.',execution='Month-end same-close zero-cost diagnostic; equal value 3 (or2), fixed adjusted-price units until month end. Dividend reinvestment approximation.',selection='Lowest 2016-2021 month-normalized screenshot log-shape error. Later 2022-2026 diagnostic checks without selecting on those metrics. Both periods have been seen in earlier research, not independent OOS.',no_target_return_optimization=True,no_model_fit=True)
 (P/'HYPOTHESES_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 px,member,_=read_prices();ret=px.pct_change(fill_method=None);vol=ret.rolling(63).std();raw={'r12_2':px.shift(42)/px.shift(252)-1,'r6_2':px.shift(42)/px.shift(126)-1,'r12_1':px.shift(21)/px.shift(252)-1,'vol63':vol,'near_high':px/px.rolling(252).max()};raw['risk12_2']=raw['r12_2']/vol
 rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()};positive=raw['r12_2']>0
 cal=json.loads((P/'CALIBRATION.json').read_text());obs=pd.read_csv(P/'DIGITIZED_WEALTH.csv');results={};curves={};orders={}
 for rule,weights in RULES.items():
  score=sum(rank[k]*v for k,v in weights.items())
  for buffer in [False,True]:
   name=rule+('_keep5' if buffer else '_replace');
   try:
    nav,o=simulate(px,member,score,positive,buffer);curves[name]=nav;orders[name]=o
    results[name]=dict(**metrics(nav),**fit_errors(nav,cal,obs),rebalance_events=len(o),holdings_replaced=sum(len(q['new']) for q in o))
   except ValueError as e:results[name]=dict(failed=str(e))
   print(name,json.dumps(results[name]),flush=True)
 successful={k:v for k,v in results.items() if 'failed'not in v};selected=min(successful,key=lambda n:successful[n]['development_month_shape_log_rmse'])
 (P/'RULE_COMPARISON.json').write_text(json.dumps(dict(selected_by_development_shape=selected,results=results),indent=2)+'\n');(P/'RULE_HOLDINGS.json').write_text(json.dumps(orders,indent=2)+'\n');pd.DataFrame(curves).to_csv(P/'RULE_NAV.csv')
 # Screenshot monthly return estimates with bands, never claimed exact realized returns.
 ref=obs[obs.panel=='wealth'];axis=cal['wealth'];rows=[]
 for period in pd.period_range('2016-09','2026-09',freq='M'):
  days=px.index[(px.index.to_period('M')==period)&(px.index<=END)];s=px.index[px.index.get_loc(days[0])-1];e=days[-1];xx=axis['x0']+axis['width']*(pd.DatetimeIndex([s,e])-START).days.to_numpy()/(END-START).days
  logv=np.interp(xx,ref.x,ref.log_wealth);r=np.exp(logv[1]-logv[0])-1
  rows.append(dict(month=str(period),estimated_return=float(r),illustrative_low=float(np.exp(logv[1]-logv[0]-2/axis['log_scale'])-1),illustrative_high=float(np.exp(logv[1]-logv[0]+2/axis['log_scale'])-1),note='Two-boundary one-pixel sensitivity only, not statistical CI; occlusion/time ambiguity can be larger'))
 (P/'APPROX_MONTHLY_RETURNS.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('SELECTED',selected)
if __name__=='__main__':main()
