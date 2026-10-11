"""Retrospective portfolio attribution to screenshot, NEVER a trading signal.
Per-month NAV scale and prior peak are nuisance parameters. Month fits are
independent; a low image loss is not evidence of an investable selection rule.
"""
from pathlib import Path
from itertools import combinations
import json,hashlib
import numpy as np,pandas as pd
from extract import P,O,START,END,read_prices,sha

def combo_paths(prices,combos):
 # Equal value at entry, fixed adjusted-price units afterward, not daily reweighting.
 z=prices/prices[0]
 return (z[:,combos[:,0]]+z[:,combos[:,1]]+z[:,combos[:,2]]).T/3

def candidates(n):
 rows=list(combinations(range(n),3));rows.extend((a,b,n) for a,b in combinations(range(n),2));rows.append((n,n,n))
 return np.asarray(rows,dtype=int)

def portfolio_paths(prices):
 c=candidates(prices.shape[1]);paths=combo_paths(np.column_stack([prices,np.ones(len(prices))]),c)
 # The padding sentinel represents a missing slot: 2 assets are 50/50, not thirds cash.
 two=(c[:,2]==prices.shape[1])&(c[:,0]!=prices.shape[1]);paths[two]=(paths[two]*3-1)/2
 return c,paths

def interpolation_matrix(x,xobs):
 m=np.zeros((len(x),len(xobs)))
 for k,z in enumerate(xobs):
  i=np.searchsorted(x,z,side='right')-1;i=np.clip(i,0,len(x)-2);t=np.clip((z-x[i])/(x[i+1]-x[i]),0,1);m[i,k]=1-t;m[i+1,k]=t
 return m

def mul(a,b):return np.einsum('ij,jk->ik',a,b,optimize=False)

def score_paths(paths,x,observations,cal,xshift=0,ddweight=1.0,constraints=None):
 # One pixel is several dates. All observations and uncertainty stay in pixel space.
 npth=len(paths); peaks=np.maximum.accumulate(paths,axis=1)
 logobs=[];logpred=[];precision=[];details={}
 for panel in ['wealth','relative']:
  z=observations[panel];xs=z.x.to_numpy()-xshift;mat=interpolation_matrix(x,xs);v=mul(paths,mat);logpred.append(np.log(v));sc=cal[panel]['log_scale'];logobs.append(z.log_wealth.to_numpy());precision.append(1/(np.maximum(1.0,(z.ymax-z.ymin).to_numpy()/2+.5)/sc)**2)
 obs=np.concatenate(logobs);pred=np.concatenate(logpred,axis=1);weight=np.concatenate(precision)
 if len(obs)<4:return None
 anchor=((obs[None,:]-pred)*weight).sum(axis=1)/weight.sum()
 if constraints is not None:
  anchor=np.clip(anchor,constraints['start_log_nav']-constraints['log_tolerance'],constraints['start_log_nav']+constraints['log_tolerance'])
 wealth_res=pred+anchor[:,None]-obs;wealth_loss=(wealth_res**2*weight).sum(axis=1)/weight.sum()*cal['wealth']['log_scale']**2
 # Prior peak estimated analytically, refine where past peak exceeds month peak.
 d=observations['dd'];ddloss=np.zeros(npth);prior_peak=np.ones(npth)
 if len(d)>=3:
  xs=d.x.to_numpy()-xshift;mid=interpolation_matrix(x,xs);v=mul(paths,mid);peak=mul(peaks,mid)
  scale=cal['dd']['scale'];zero=cal['dd']['y0'];target_q=1-(d.y.to_numpy()-zero)/scale
  w=1/np.maximum(2,(d.ymax-d.ymin).to_numpy()/2)**2
  # Full median fit is an initializer only; envelope loss below respects raster compression.
  if constraints is None:
   invlow=np.full(npth,1/1.8);invhigh=np.ones(npth)
  else:
   # A monthly fit cannot invent a prior global NAV peak absent from the screenshot.
   lowpeak=np.maximum(1,np.exp(constraints['prior_log_peak']-constraints['log_tolerance']-anchor))
   highpeak=np.maximum(1,np.exp(constraints['prior_log_peak']+constraints['log_tolerance']-anchor))
   invlow=1/highpeak;invhigh=1/lowpeak
  invp=np.clip((v*target_q*w).sum(axis=1)/(v*v*w).sum(axis=1),invlow,invhigh)
  for _ in range(6):
   elig=peak*invp[:,None]<1.002;den=(v*v*w*elig).sum(axis=1);num=(v*target_q*w*elig).sum(axis=1)
   candidate=np.divide(num,den,out=invp.copy(),where=den>1e-9);invp=np.clip(candidate,invlow,invhigh)
  prior_peak=1/invp
  # Include uncertainty in horizontal position through each column's raster footprint.
  sub=np.linspace(-.45,.45,7);grid=(xs[:,None]+sub).ravel();interp=interpolation_matrix(x,grid)
  dd=paths/np.maximum(prior_peak[:,None],peaks)-1;draw=zero-scale*mul(dd,interp);draw=draw.reshape(npth,len(d),len(sub));lo=draw.min(axis=2);hi=draw.max(axis=2);ctr=draw.mean(axis=2)
  ymin=d.ymin.to_numpy();ymax=d.ymax.to_numpy()
  # One-way coverage avoids treating line occlusion as observed NAV. Weak center penalty
  # stops arbitrary excessively wide envelopes winning; this is a diagnostic score.
  gap=np.maximum(np.maximum(lo-ymax-1,ymin-hi-1),0)
  ctr_gap=np.maximum(np.maximum(ctr-ymax-2,ymin-ctr-2),0)
  ddloss=(gap**2+.15*ctr_gap**2).mean(axis=1)/9
 total=wealth_loss+ddweight*ddloss
 return total,anchor,prior_peak,wealth_loss,ddloss

def load_observations():
 z=pd.read_csv(P/'DIGITIZED_WEALTH.csv');return {panel:z[z.panel==panel].copy() for panel in ['wealth','relative']}|{'dd':pd.read_csv(P/'purple_dd_pixels.csv')}

def display(s):return 'META/FB' if s=='research_equity_FB' else s

def run_month(month,px,member,obs,cal,shift=0,wide=False):
 period=pd.Period(month,freq='M');days=px.index[(px.index.to_period('M')==period)&(px.index<=END)];start=px.index[px.index.get_loc(days[0])-1];end=days[-1]
 if start<START:return None
 f=px.loc[start:end];eligible=member.loc[start];cols=sorted(s for s in px if bool(eligible.get(s,False)) and f[s].notna().all() and f[s].gt(0).all())
 # Full-month completeness is retrospective identification feasibility only, not an entry rule.
 c,paths=portfolio_paths(f[cols].to_numpy());axis=cal['wealth'];x=axis['x0']+axis['width']*((f.index-START).days/(END-START).days).to_numpy()
 # Exclude boundary pixels mixing different months' portfolios.
 ob={p:z[(z.x>x[0]+.6)&(z.x<x[-1]-.6)].copy() for p,z in obs.items()}
 reference=obs['wealth'];prior=reference[reference.x<=x[0]+shift]
 constraint=dict(start_log_nav=float(np.interp(x[0]+shift,reference.x,reference.log_wealth)),prior_log_peak=max(0,float(prior.log_wealth.max())) if len(prior) else 0,log_tolerance=1.5/axis['log_scale'])
 res=score_paths(paths,x,ob,cal,xshift=shift,constraints=constraint)
 if res is None:return dict(month=month,status='insufficient_visible_pixels')
 total,anchor,peak,wealth,dd=res;order=np.argsort(total,kind='stable');best=order[0]
 near=np.flatnonzero(total<=total[best]+1);names=lambda i:[display(cols[a]) for a in c[i] if a<len(cols)]
 counts={}
 for i in near:
  for s in names(i):counts[s]=counts.get(s,0)+1
 top=[]
 for i in order[:12]:top.append(dict(holdings=names(i),score=float(total[i]),wealth_rmse_px=float(np.sqrt(wealth[i])),dd_scaled_rmse=float(np.sqrt(dd[i])),return_pct=float(100*(paths[i,-1]-1)),start_wealth=float(np.exp(anchor[i])),prior_peak_over_start_nav=float(peak[i])))
 ranking=[]
 # Look only at features known at start; these describe inferred choices, not actual choices.
 ix=px.index.get_loc(start)
 if ix>=252:
  ratios=px.loc[start,cols]/px.iloc[ix-252][cols]-1
  m122=px.iloc[ix-42][cols]/px.iloc[ix-252][cols]-1
  m62=px.iloc[ix-42][cols]/px.iloc[ix-126][cols]-1
  vol=px[cols].iloc[ix-251:ix+1].pct_change(fill_method=None).std()*np.sqrt(252)
  m3=px.loc[start,cols]/px.iloc[ix-63][cols]-1
  feat=pd.DataFrame({'r12':ratios,'r12_2':m122,'r6_2':m62,'risk_r12_2':m122/vol,'vol':vol,'r3':m3}).rank(pct=True)
  for s in names(best):
   original='research_equity_FB' if s=='META/FB' else s
   ranking.append(dict(symbol=s,**{k:float(v) for k,v in feat.loc[original].items()}))
 out=dict(month=month,status='retrospective_candidates_only',start=str(start.date()),end=str(end.date()),candidate_symbols=len(cols),combinations=len(c),visible_columns={p:len(v) for p,v in ob.items()},global_level_constraints=constraint,best=top[0],top_candidates=top,near_score_plus1_count=len(near),near_ticker_fractions={s:round(v/len(near),5) for s,v in sorted(counts.items(),key=lambda a:-a[1])[:15]},best_holdings_pre_entry_percentiles=ranking,best_second_margin=float(total[order[1]]-total[best]),universe='Our approximate historical index union Top32 +17 unlevered ETFs; not verified author pool')
 return out

def main():
 px,member,sources=read_prices();cal=json.loads((P/'CALIBRATION.json').read_text());obs=load_observations();months=pd.period_range('2016-09','2026-09',freq='M')
 config=dict(task='Reverse attribute author monthly screenshot, then investigate ranking. NOT performance backtest.',image_hash=cal['image_sha256'],monthly_assumptions=['Last trading day close rebalance, fixed adjusted-price units until next rebalance','Equal initial weights; 3 positions, 2 positions, or cash','Independent per-month inverse fitting uses month future returns and fits NAV scale/prior peak; no concatenated performance is tradable','Our historical index-union Top32 +17 ETF list is a hypothesis. Author exact list unknown.'],score='Image wealth/ratio precision-weighted fit with free monthly NAV scale; compressed drawdown envelope loss with fitted prior peak constrained by all previously digitized wealth peaks; monthly starting NAV constrained to digitized boundary +/-1.5 pixels. Ranking score is not a probability.',stability='Repeat with x-axis offset +/-0.35 pixels (~1.24 calendar days). A score within best+1 defines an illustrative ambiguity set, not a confidence interval.',no_strategy_optimization=True,no_live_changes=True)
 (P/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 results=[]
 for month in months:
  r=run_month(str(month),px,member,obs,cal)
  if r.get('best'):
   perturb=[run_month(str(month),px,member,obs,cal,shift=s) for s in [-.35,.35]]
   r['xshift_best_holdings']=[v['best']['holdings'] for v in perturb];r['stable_exact_triple_under_xshift']=all(set(v['best']['holdings'])==set(r['best']['holdings']) for v in perturb)
   r['persistent_tickers_under_xshift']=sorted(set(r['best']['holdings']).intersection(*[set(v['best']['holdings']) for v in perturb]))
  results.append(r)
  (P/'MONTHLY_CANDIDATES.json').write_text(json.dumps(results,indent=2)+'\n')
  print(month,','.join(r.get('best',{}).get('holdings',[])),round(r.get('best',{}).get('score',0),3),'ambiguous',r.get('near_score_plus1_count'),flush=True)
 summary=dict(months=len(results),fitted=sum('best'in r for r in results),stable_combinations=sum(r.get('stable_exact_triple_under_xshift',False) for r in results),median_ambiguity=float(np.median([r['near_score_plus1_count'] for r in results if 'best'in r])),median_best_score=float(np.median([r['best']['score'] for r in results if 'best'in r])),inferred_holdings_are_verified=False)
 (P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
