"""Coherent whole-curve hindsight attribution. Explicitly non-causal oracle search.
Used to assess whether known price paths can explain chart, NEVER a strategy.
"""
import json,sys
from pathlib import Path
from itertools import combinations
import numpy as np,pandas as pd
from research import P,O,load,raw_features,dump,START,END,metrics,fit_errors
from inverse import load_observations,interpolation_matrix,mul

BEAM=32;SHORTLIST=1536;PER_NODE=48

def prep_month(px,member,raw,period,cal,obs,gate_quantile):
 days=px.index[(px.index.to_period('M')==period)&(px.index<=END)];s=px.index[px.index.get_loc(days[0])-1];e=days[-1];frame=px.loc[s:e];i=px.index.get_loc(s);elig=member.loc[s]&raw['m252_42'].loc[s].gt(0)&px.iloc[i-252:i+1].notna().all()
 if gate_quantile:elig &=raw['m252_42'].loc[s].where(member.loc[s]).rank(pct=True).ge(gate_quantile)
 cols=sorted(t for t in px if elig[t] and frame[t].notna().all() and frame[t].gt(0).all());k=min(3,len(cols));combs=np.asarray(list(combinations(range(len(cols)),k)),dtype=int) if k>=2 else np.zeros((1,0),dtype=int)
 ratios=(frame[cols]/frame[cols].iloc[0]).to_numpy();paths=np.mean(ratios[:,combs],axis=2).T if k>=2 else np.ones((1,len(frame)))
 axis=cal['wealth'];x=axis['x0']+axis['width']*((frame.index-START).days/(END-START).days).to_numpy();targets=[];projections=[];weights=[]
 for panel in ['wealth','relative']:
  z=obs[panel];z=z[(z.x>x[0]+.6)&(z.x<x[-1]-.6)];projections.append(np.log(mul(paths,interpolation_matrix(x,z.x.to_numpy()))));targets.append(z.log_wealth.to_numpy());weights.append(1/(np.maximum(1,(z.ymax-z.ymin).to_numpy()/2+.5)/cal[panel]['log_scale'])**2)
 pred=np.concatenate(projections,axis=1);target=np.concatenate(targets);w=np.concatenate(weights);center=((target-pred)*w).sum(axis=1)/w.sum();res=pred+center[:,None]-target;base=(res**2*w).sum(axis=1)/w.sum()*axis['log_scale']**2
 dd=obs['dd'];dd=dd[(dd.x>x[0]+.6)&(dd.x<x[-1]-.6)];grid=(dd.x.to_numpy()[:,None]+np.linspace(-.45,.45,7)).ravel();interp=interpolation_matrix(x,grid)
 # Endpoint term is weak because month boundary shares pixels with adjacent month.
 ref=obs['wealth'];endpoint=float(np.interp(x[-1],ref.x,ref.log_wealth));startpoint=float(np.interp(x[0],ref.x,ref.log_wealth));peakref=max(0,float(ref.loc[ref.x<=x[0],'log_wealth'].max()))
 return dict(period=str(period),start=s,end=e,cols=cols,combos=combs,paths=paths,peaks=np.maximum.accumulate(paths,axis=1),logends=np.log(paths[:,-1]),center=center,base=base,dd=dd,interp=interp,endpoint=endpoint,startpoint=startpoint,peakref=peakref,x=x,frameindex=frame.index)

def drawloss(z,pids,prior_peak,cal):
 paths=z['paths'][pids];peak=z['peaks'][pids];dd=paths/np.maximum(prior_peak,peak)-1;d=z['dd']
 if len(d)<3:return np.zeros(len(pids))
 draw=(cal['dd']['y0']-cal['dd']['scale']*mul(dd,z['interp'])).reshape(len(paths),len(d),7);lo=draw.min(2);hi=draw.max(2);mid=draw.mean(2);ymin=d.ymin.to_numpy();ymax=d.ymax.to_numpy();gap=np.maximum(np.maximum(lo-ymax-1,ymin-hi-1),0);centergap=np.maximum(np.maximum(mid-ymax-2,ymin-mid-2),0)
 return (gap**2+.15*centergap**2).mean(axis=1)/9

def run(px,raw,member,pool,q,cal,obs):
 tag=pool+f'_q{q}';allz=[];layers=[];beam=[]
 for count,period in enumerate(pd.period_range('2016-09','2026-09',freq='M')):
  z=prep_month(px,member,raw,period,cal,obs,q);allz.append(z)
  if not beam:beam=[dict(cost=0.,level=z['startpoint'],peak=max(z['peakref'],z['startpoint']),parent=None,combo=None)]
  proposals=[]
  for state_i,state in enumerate(beam):
   l=state['level'];score=z['base']+(l-z['center'])**2*cal['wealth']['log_scale']**2
   boundary=(l+z['logends']-z['endpoint'])**2*cal['wealth']['log_scale']**2*.25;rough=score+boundary
   n=min(SHORTLIST,len(score));pids=np.argpartition(rough,n-1)[:n];total=rough[pids]+drawloss(z,pids,np.exp(state['peak']-l),cal);take=np.argsort(total,kind='stable')[:PER_NODE]
   for a in take:
    ci=int(pids[a]);end=l+z['logends'][ci];peak=max(state['peak'],l+np.log(z['paths'][ci].max()));proposals.append(dict(cost=state['cost']+float(total[a]),level=float(end),peak=float(peak),parent=state_i,combo=ci,month_loss=float(total[a])))
  # Preserve different endpoint/peak states, reducing collapse to one local optimum.
  proposals.sort(key=lambda n:n['cost']);seen=set();new=[]
  for item in proposals:
   key=(round(item['level']/.004),round(item['peak']/.006))
   if key in seen:continue
   seen.add(key);new.append(item)
   if len(new)==BEAM:break
  beam=new;layers.append(beam)
  print(tag,str(period),'combinations',len(z['paths']),'loss',round(beam[0]['cost'],2),flush=True)
 last=min(range(len(beam)),key=lambda j:beam[j]['cost']);chosen=[]
 for t in range(len(layers)-1,-1,-1):
  state=layers[t][last];chosen.append((t,state));last=state['parent']
 chosen.reverse();nav=pd.Series(dtype=float);orders=[]
 initial=allz[0]['startpoint']
 for t,state in chosen:
  z=allz[t];startlevel=initial if t==0 else chosen[t-1][1]['level'];ci=state['combo'];vals=np.exp(startlevel-initial)*z['paths'][ci];s=pd.Series(vals,index=z['frameindex']);nav=pd.concat([nav,s[~s.index.isin(nav.index)]])
  holdings=[z['cols'][j] for j in z['combos'][ci]];orders.append(dict(month=z['period'],date=str(z['start'].date()),holdings=holdings,month_loss=state['month_loss'],terminal_relative_wealth=float(vals[-1]),selection_uses_future_month_prices=True))
 mm=metrics(nav);err=fit_errors(nav,cal,pd.read_csv(O/'monthly_curve_reverse/DIGITIZED_WEALTH.csv'))
 result=dict(tag=tag,**mm,**err,beam_width=BEAM,candidates_per_state=SHORTLIST,path_score=chosen[-1][1]['cost'],mode='NON_CAUSAL_HINDSIGHT_ATTRIBUTION',not_trading_strategy=True,holdings_verified=False,orders=orders)
 dump(P/f'INVERSE_{tag}.json',result);nav.to_csv(P/f'INVERSE_{tag}_NAV.csv');return result

def main():
 px,fs,pools=load();raw=raw_features(px,fs);cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();cfg=dict(beam=BEAM,shortlist=SHORTLIST,variants=[['core17',0],['theme17',0],['theme17',.5]],future_returns_used=True,purpose='Can a globally continuous monthly fixed-share path approximately explain chart, before trying to infer causal selection rules?',limitations='Beam search is approximate; a poor fit does not prove no path exists. Missing-month paths excluded only for attribution, not deployable eligibility. Pool remains our proxy; no causal alpha claim.');dump(P/'INVERSE_CONFIG.json',cfg);out={}
 for pool,q in cfg['variants']:
  r=run(px,raw,pools[pool],pool,q,cal,obs);out[r['tag']]={k:v for k,v in r.items() if k!='orders'};dump(P/'INVERSE_SUMMARY.json',out)
if __name__=='__main__':main()
