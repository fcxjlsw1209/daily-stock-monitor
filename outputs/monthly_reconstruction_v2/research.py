"""Monthly author-model reconstruction, isolated research with explicit clocks.
No production imports/mutations. All tried hypotheses retained, no hidden trials.
"""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_curve_reverse'))
from extract import read_prices,START,END,sha
from hypotheses import metrics,fit_errors
BASE_ETFS=['SPY','QQQ','IWM','DIA','TLT','GLD','XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
THEME_ETFS=[s for s in BASE_ETFS if s not in ['DIA','XLB','XLC','XLRE','XLU','XLP']]+['SMH','SOXX','IGV','VGT','MTUM','ARKK']
LEVER_ETFS=[s for s in BASE_ETFS if s not in ['IWM','DIA','XLK']]+['TQQQ','QLD','SOXL']
POOLS={'core17':BASE_ETFS,'theme17':THEME_ETFS,'leveraged17_DIAGNOSTIC':LEVER_ETFS}
REBUILD=O/'monthly_curve_reverse';FRAME=O/'index_union/alias_resolved_inputs'

def dump(f,obj):Path(f).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load():
 px,member,sources=read_prices();fs={k:pd.read_csv(FRAME/f'{k}.csv.gz',index_col=0,parse_dates=True) for k in ['close','opening','high','low','volume','sector60']}
 sources +=[FRAME/f'{k}.csv.gz' for k in fs]
 manifest=json.loads((O/'four_factor_rebound/ETF_SOURCES.json').read_text())+json.loads((P/'ETF_MANIFEST.json').read_text())
 for symbol in ['QLD','TQQQ']:
  f=O/'qqq_ma200_research/data'/f'{symbol}.csv';manifest.append(dict(symbol=symbol,file=str(f),sha256=sha(f)))
 for r in manifest:
  file=Path(r['file']);assert sha(file)==r['sha256'];d=pd.read_csv(file,index_col=0,parse_dates=True);s=r['symbol'];px[s]=d['Adj Close'].reindex(px.index)
  for k,col in [('close','Close'),('opening','Open'),('high','High'),('low','Low'),('volume','Volume')]:fs[k][s]=d[col].reindex(px.index)
  fs['sector60'][s]=px.SPY.pct_change(60,fill_method=None);sources.append(file)
 cols=px.columns;fs={k:v.reindex(index=px.index,columns=cols) for k,v in fs.items()};member=member.reindex(columns=cols,fill_value=False)
 pools={}
 stocks=member.copy();stocks.loc[:,sorted(set(sum(POOLS.values(),[])))]=False
 for name,etfs in POOLS.items():
  assert len(etfs)==17 and len(set(etfs))==17
  mask=stocks.copy();mask[etfs]=px[etfs].notna();pools[name]=mask
 sources +=[P/'ETF_MANIFEST.json',REBUILD/'CALIBRATION.json',REBUILD/'DIGITIZED_WEALTH.csv']
 dump(P/'SOURCE_HASHES.json',{str(f):sha(f) for f in sources})
 return px,fs,pools

def raw_features(px,fs):
 ret=px.pct_change(fill_method=None);vol=ret.rolling(63).std();a={}
 for span in [126,189,252,294,378]:
  for lag in [21,42,63]:a[f'm{span}_{lag}']=px.shift(lag)/px.shift(span)-1
 core=a['m252_42'];a.update(risk=core/vol,risk_half=core/np.sqrt(vol),vol=vol,lowvol=-vol,r63=px.pct_change(63,fill_method=None),r21=px.pct_change(21,fill_method=None),r5=px.pct_change(5,fill_method=None),accel=px.shift(42)/px.shift(126)-px.shift(126)/px.shift(252),near252=px/px.rolling(252).max(),downside=-ret.clip(upper=0).pow(2).rolling(63).mean().pow(.5),relative=(1+px.pct_change(60,fill_method=None))/(1+fs['sector60'])-1)
 c,h,l,v=(fs[k] for k in ['close','high','low','volume']);flow=((2*c-h-l)/(h-l).replace(0,np.nan)).fillna(0);a['cmf21']=(flow*v).rolling(21).sum()/v.rolling(21).sum();a['volume_ratio']=v.rolling(21).mean()/v.rolling(126).mean();a['trend_gap']=px/px.rolling(200).mean()-1;a['draw63']=px/px.rolling(63).max()-1
 return a

def fee_adjusted_nav_fraction(old_weights,target_weights,cost_bps):
 half=cost_bps/20000
 if half==0:return 1.
 symbols=set(old_weights)|set(target_weights)
 lo,hi=0.,1.
 for _ in range(55):
  z=(lo+hi)/2;trade=sum(abs(z*target_weights.get(s,0)-old_weights.get(s,0)) for s in symbols)
  if z+half*trade>1:hi=z
  else:lo=z
 return (lo+hi)/2

class Simulator:
 def __init__(self,px,fs,member):
  self.px=px;self.fs=fs;self.member=member;self.idx=px.index;self.dates=px.loc['2016-08-31':END].index
  self.events=[self.idx.get_loc(self.dates[0])]+[i for i in range(self.idx.get_loc(self.dates[0])+1,self.idx.get_loc(END)+1) if i+1<len(self.idx) and self.idx[i].to_period('M')!=self.idx[i+1].to_period('M')]
  self.valid=member&px.notna().rolling(253,min_periods=253).sum().eq(253)&px.gt(0)
  self.ret=px.pct_change(fill_method=None)
 def run(self,score,positive,buffer=0,cost_bps=0,signal_lag=0,market_gate=None):
  nav=pd.Series(index=self.dates,dtype=float);level=1.;prev_hold=[];orders=[];end_i=self.idx.get_loc(END)
  for e,i in enumerate(self.events):
   j=self.events[e+1] if e+1<len(self.events) else end_i;day=self.idx[i];signal=self.idx[i-signal_lag]
   sc=score.loc[signal].where(self.valid.loc[signal]&positive.loc[signal]).dropna().sort_values(ascending=False,kind='stable');chosen=[]
   if market_gate is not None and not bool(market_gate.loc[signal]):sc=sc.iloc[:0]
   if buffer:chosen=[s for s in sc.index[:buffer] if s in prev_hold][:3]
   chosen += [s for s in sc.index if s not in chosen][:3-len(chosen)]
   if len(chosen)<2:chosen=[]
   # Net-difference turnover uses previous market weights at the actual rebalance price.
   old_weights={}
   if e and prev_hold:
    previous_i=self.events[e-1];rr=self.px.iloc[i][prev_hold]/self.px.iloc[previous_i][prev_hold];old_weights=(rr/rr.sum()).to_dict()
   target={s:1/len(chosen) for s in chosen};turnover=sum(abs(target.get(s,0)-old_weights.get(s,0)) for s in set(target)|set(old_weights));fraction=fee_adjusted_nav_fraction(old_weights,target,cost_bps);fee=1-fraction
   # Solve transaction fee and net equal-target allocation self-consistently.
   if fee>=1:raise ValueError('Fee invalid')
   initial=level*(1-fee);frame=self.px.iloc[i:j+1][chosen]
   if chosen:
    if not frame.notna().all().all() or not frame.gt(0).all().all():raise ValueError(f'Unpriceable held path {day} {chosen}')
    path=(frame/frame.iloc[0]).mean(axis=1)
   else:path=pd.Series(1.,index=self.idx[i:j+1])
   segment=initial*path
   # At a rebalance day's close, new fees occur after old position marking.
   nav.loc[segment.index]=segment;level=float(segment.iloc[-1]);prev_hold=chosen
   orders.append(dict(date=str(day.date()),signal_date=str(signal.date()),holdings=chosen,turnover=float(turnover),fee_fraction=fee,entry_equity=initial))
  assert nav.notna().all() and nav.gt(0).all()
  return nav,orders

# Targets are mature at each fit date; never labels extending into the forecast month.
MODEL_FEATURES=['m252_42','m126_42','m294_42','risk','r63','r21','r5','accel','relative','vol','downside','near252','cmf21','volume_ratio','trend_gap','draw63']
def train_models(px,raw,member,window_months,target_kind):
 idx=px.index;ends=np.array([i for i in range(252,len(idx)-1) if idx[i].to_period('M')!=idx[i+1].to_period('M')]);features={k:raw[k].where(member).rank(axis=1,pct=True) for k in MODEL_FEATURES};X=np.stack([features[k].to_numpy() for k in MODEL_FEATURES],axis=2)
 # Context interactions can alter volatility/momentum effects in weak markets.
 market=(px.SPY/px.SPY.rolling(200).mean()-1).clip(-.3,.3).to_numpy();X=np.concatenate([X,X[:,:,[0,1,4,9]]*market[:,None,None]],axis=2)
 valid=member.to_numpy()&np.isfinite(X).all(axis=2)&(px.notna().rolling(253,min_periods=253).sum().eq(253)).to_numpy()
 y=np.full(px.shape,np.nan);label_end={}
 for k,i in enumerate(ends[:-1]):
  j=ends[k+1];y[i]=(px.iloc[j]/px.iloc[i]-1).to_numpy();label_end[i]=j
 if target_kind=='rank':y=pd.DataFrame(y,index=idx,columns=px.columns).where(member).rank(axis=1,pct=True).to_numpy()-.5
 elif target_kind=='risk':y=y/(raw['vol'].to_numpy()*np.sqrt(21));y=np.clip(y,-5,5)
 elif target_kind=='excess':y=y-y[:,list(px.columns).index('SPY'),None];y=np.clip(y,-.75,.75)
 else:raise ValueError(target_kind)
 pred=np.full(px.shape,np.nan);stump=pred.copy();audit=[]
 for i in ends:
  if not pd.Timestamp('2016-08-31')<=idx[i]<=END:continue
  trainends=[t for t in ends if t<i and label_end.get(t,len(idx))<=i and idx[t]>=pd.Timestamp('2010-01-01')][-window_months:]
  if len(trainends)<24:continue
  dataX=X[trainends];dataY=y[trainends];ok=valid[trainends]&np.isfinite(dataY);rr,cc=np.where(ok);xx=dataX[rr,cc];yy=dataY[rr,cc]
  count=np.bincount(rr,minlength=len(trainends));weights=np.exp2(-(len(trainends)-1-rr)/24)/count[rr];weights*=len(weights)/weights.sum();mu=np.average(xx,axis=0,weights=weights);sd=np.sqrt(np.average((xx-mu)**2,axis=0,weights=weights));sd[sd<1e-6]=1;z=(xx-mu)/sd;ym=np.average(yy,weights=weights)
  coef=np.linalg.solve(np.einsum('ni,nj,n->ij',z,z,weights)+100*np.eye(z.shape[1]),np.einsum('ni,n,n->i',z,weights,yy-ym));test=valid[i];block=(X[i,test]-mu)/sd;pp=np.einsum('ni,i->n',block,coef)+ym;pred[i,test]=pp
  # Single training-only residual stump, fixed15% contribution.
  resid=yy-(np.einsum('ni,i->n',z,coef)+ym);best=None
  for k in range(len(MODEL_FEATURES)):
   for split in [.25,.5,.75]:
    left=xx[:,k]<=split
    if min(left.sum(),(~left).sum())<50:continue
    a=np.average(resid[left],weights=weights[left]);b=np.average(resid[~left],weights=weights[~left]);err=np.average((resid-np.where(left,a,b))**2,weights=weights)
    if best is None or err<best[0]:best=(err,k,split,a,b)
  tp=pp.copy()
  if best:
   _,k,split,a,b=best;tp+=.15*np.where(X[i,test,k]<=split,a,b)
  stump[i,test]=tp
  audit.append(dict(signal_date=str(idx[i].date()),last_train_signal=str(idx[trainends[-1]].date()),latest_label_end=str(idx[max(label_end[t] for t in trainends)].date()),rows=len(yy),window_months=window_months,target=target_kind))
 return pd.DataFrame(pred,index=idx,columns=px.columns),pd.DataFrame(stump,index=idx,columns=px.columns),audit

def fixed_rules(rank,raw):
 rules={}
 for span in [126,189,252,294,378]:
  for lag in [21,42,63]:rules[f'm{span}_{lag}']=rank[f'm{span}_{lag}']
 for f in ['risk','risk_half']:rules[f]=rank[f]
 rules['balanced']=.5*rank['m252_42']+.2*rank['risk']+.15*rank['relative']+.1*rank['cmf21']+.05*rank['near252']
 rules['acceleration']=.5*rank['m252_42']+.2*rank['accel']+.15*rank['relative']+.15*rank['vol']
 rules['relative']=.6*rank['m252_42']+.25*rank['relative']+.15*rank['cmf21']
 rules['short_blend']=.6*rank['m252_42']+.2*rank['m126_42']+.1*rank['r21']+.1*rank['cmf21']
 return rules

def main():
 cfg=dict(author_goal=dict(cagr_252=.8064,max_dd=-.3446,sharpe=1.867,wealth=387.72),goals='Reproduce curve AND risk AND a chronological rule. Matching CAGR alone is insufficient. Author exact model/universe remain unknown.',primary_pools=['core17','theme17'],leveraged_pool='Diagnostic only, different embedded exposure; may not count as reproducing unlevered assumptions.',pools=POOLS,selection='Keep all trials. Select by2016-2021 global log-NAV discrepancy +2*month-shape discrepancy against digitized chart; evaluate2022-2026 separately. Already viewed periods, not independent OOS.',execution='Month-end close; same-close full bars zero-cost reference. No intramonth trades. 3 equal or2 equal, otherwisecash; fixed adjusted-price units.',eligibility='Original current research historical index-union price-scaled Top32;253 consecutive historical prices;positive P[t-42]/P[t-252]-1. Not certified all-US Top32 or author ETF17.',rules=dict(spans=[126,189,252,294,378],skip=[21,42,63],risk=['vol63','sqrt(vol63)'],blend_names=['balanced','acceleration','relative','short_blend']),models=dict(features=MODEL_FEATURES,window_months=[36,72],targets=['excess','rank','risk'],alpha=100,model_rank_weights=[.15,.3],residual_stump_weight=.15,label='Actual next calendar-month close return; mature before or at model signalclose',sample_weight='Each month equal total before24-month half-life'),no_production_changes=True)
 dump(P/'CONFIG.json',cfg);px,fs,pools=load();raw=raw_features(px,fs);cal=json.loads((REBUILD/'CALIBRATION.json').read_text());obs=pd.read_csv(REBUILD/'DIGITIZED_WEALTH.csv');results={};orderlog={};curves={};audits={};(P/'predictions').mkdir(exist_ok=True)
 def save(pool,name,sc,gate,sim,buffer=0):
  key=pool+'__'+name;case=dict(pool=pool,name=name,buffer=buffer)
  try:
   nav,orders=sim.run(sc,gate,buffer=buffer);curve=nav/float(nav.iloc[0]);mm=metrics(curve);err=fit_errors(curve,cal,obs);case.update(**mm,**err,development_selection_error=err['development_global_log_rmse']+2*err['development_month_shape_log_rmse'],new_holdings=sum(len(set(v['holdings'])-set(orders[i-1]['holdings'] if i else [])) for i,v in enumerate(orders)))
   curves[key]=curve;orderlog[key]=orders
  except Exception as e:case['failure']=type(e).__name__+': '+str(e)
  results[key]=case;dump(P/'RESULTS.json',results);print(key,round(case.get('cagr_252',0)*100,2),round(case.get('max_dd',0)*100,2),case.get('failure',''),flush=True)
 for pool,member in pools.items():
  rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()};sim=Simulator(px,fs,member);gate=raw['m252_42']>0;rule=fixed_rules(rank,raw)
  for name,sc in rule.items():save(pool,name,sc,gate,sim)
  for name in ['m252_42','balanced','relative']:save(pool,name+'_keep5',rule[name],gate,sim,5)
  if pool.startswith('leveraged'):continue
  for window in [36,72]:
   for target in ['excess','rank','risk']:
    key=f'{pool}_{window}_{target}';pred,tree,audit=train_models(px,raw,member,window,target);audits[key]=audit
    pred.to_csv(P/'predictions'/f'{key}.csv.gz');tree.to_csv(P/'predictions'/f'{key}_tree.csv.gz');dump(P/'TRAINING_AUDIT.json',audits)
    for name,model in [('ridge',pred),('ridge_tree',tree)]:
     pr=model.where(member).rank(axis=1,pct=True)
     for w in [.15,.3]:
      score=(1-w)*rule['balanced']+w*pr
      save(pool,f'{name}_{window}_{target}_w{w}',score,gate,sim)
      # These only use current-month prediction rows; no forward/backward filling.
  dump(P/'ORDERS.json',orderlog);pd.DataFrame(curves).to_csv(P/'CURVES.csv.gz')
 usable={k:v for k,v in results.items() if 'failure'not in v};primary={k:v for k,v in usable.items() if not k.startswith('leveraged')};selected=min(primary,key=lambda k:primary[k]['development_selection_error']);best_cagr=max(primary,key=lambda k:primary[k]['cagr_252']);best_late=min(primary,key=lambda k:primary[k]['later_diagnostic_global_log_rmse'])
 dump(P/'SELECTION.json',dict(development_selected=selected,best_full_cagr_retrospective=best_cagr,best_later_fit_retrospective=best_late,all_trials=len(results),primary_trials=len(primary),reproduced=False));print('SELECTION',selected,'CAGR_BEST',best_cagr,flush=True)
if __name__=='__main__':main()
