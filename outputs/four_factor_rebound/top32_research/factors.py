"""Four equally weighted groups; event outcomes are delayed until observable."""
import numpy as np,pandas as pd

def matured_average(value,event,horizon=6,window=756,prior=20):
 x=value.where(event).shift(horizon);n=event.where(value.notna(),False).astype(float).shift(horizon)
 count=n.rolling(window,min_periods=20).sum();total=x.rolling(window,min_periods=20).sum()
 pooled_total=x.sum(axis=1,min_count=1).rolling(window,min_periods=20).sum()
 pooled_n=n.sum(axis=1,min_count=1).rolling(window,min_periods=20).sum()
 mean=pooled_total/pooled_n
 return total.add(mean*prior,axis=0)/(count+prior)

def build(d):
 a=d['adj'];c=d['close'];r=a.pct_change(fill_method=None);v=d['volume'];sig=r.rolling(20).std()
 r63=a.pct_change(63,fill_method=None);r126=a.pct_change(126,fill_method=None)
 location=((c-d['low'])/(d['high']-d['low'])).fillna(.5);z=a.pct_change(3,fill_method=None)/(sig*np.sqrt(3))
 ma100=a.rolling(100).mean();valid=d['eligible']&a.notna().rolling(253).sum().eq(253)&((c*v).rolling(20).mean()>20e6)&d['sectors'].notna()
 valid &= d['opening'].gt(0)&d['high'].ge(c)&d['low'].le(c)&v.gt(0)
 strong=(r126>0)&(a>ma100);event=valid&strong&((z<0)|(location<.2))&(z>-3.5)
 ao=d['opening']*a/c;ah=d['high']*a/c;al=d['low']*a/c
 outcome=ao.shift(-6)/ao.shift(-1)-1-.001
 hi=pd.concat([ah.shift(-i) for i in range(1,6)],keys=range(5)).groupby(level=1).max().reindex(a.index)
 lo=pd.concat([al.shift(-i) for i in range(1,6)],keys=range(5)).groupby(level=1).min().reindex(a.index)
 mfe=hi/ao.shift(-1)-1;mae=lo/ao.shift(-1)-1
 hist=matured_average(outcome,event);path=matured_average(mfe+mae,event)
 rank=lambda x:x.where(valid).rank(axis=1,pct=True)
 momentum=(rank(r63/sig)+rank(r126/sig)+rank((1+r63)/(1+d['sector60'])-1))/3
 rebound=rank(hist)
 route=.7*rank(path)+.3*rank(sig)
 depth=a/a.rolling(20).max()-1;cmf=(((2*c-d['high']-d['low'])/(d['high']-d['low'])).fillna(0)*v).rolling(21).sum()/v.rolling(21).sum()
 shape=(rank(-abs(z+1.25))+rank(location)+rank(cmf)+rank(-abs(depth/sig+2)))/4
 score=(momentum+rebound+route+shape)/4
 tr=pd.DataFrame(np.maximum.reduce([(d['high']-d['low']).to_numpy(),(d['high']-c.shift()).abs().to_numpy(),(d['low']-c.shift()).abs().to_numpy()]),index=c.index,columns=c.columns)
 x=[momentum,rebound,route,shape,rank(z),rank(depth),rank(cmf),rank(sig)]
 return dict(score=score,event=event,valid=valid,groups=[momentum,rebound,route,shape],X=x,y=outcome,atr=tr.rolling(14).mean(),ma5=c.rolling(5).mean(),market=d['market_close']>d['market_close'].rolling(200).mean())

def ridge_predictions(f,first='2019-01-01',alpha=10.):
 """Monthly refit, trailing 756 sessions, six-session label maturation embargo."""
 idx=f['score'].index;X=np.stack([v.to_numpy() for v in f['X']],axis=2);y=f['y'].to_numpy();event=f['event'].to_numpy();pred=np.full(y.shape,np.nan);audit=[]
 months=idx[idx>=first].to_period('M').unique()
 for month in months:
  target=np.flatnonzero(idx.to_period('M')==month);t=target[0];end=t-6 # exclusive: last training outcome known before first prediction session
  start=max(0,end-756);xx=X[start:end].reshape(-1,X.shape[2]);yy=y[start:end].ravel();ok=np.isfinite(xx).all(axis=1)&np.isfinite(yy)&event[start:end].ravel()
  xx=xx[ok];yy=yy[ok]
  if len(yy)<500:continue
  mu=xx.mean(axis=0);sd=xx.std(axis=0);sd[sd<1e-8]=1;zz=(xx-mu)/sd;ym=yy.mean();coef=np.linalg.solve(np.einsum('ni,nj->ij',zz,zz)+alpha*np.eye(zz.shape[1]),np.einsum('ni,n->i',zz,yy-ym))
  if not np.isfinite(coef).all():raise ValueError('Nonfinite Ridge coefficients')
  block=(X[target]-mu)/sd;finite=np.isfinite(block).all(axis=2);values=np.full(finite.shape,np.nan)
  values[finite]=np.einsum('ni,i->n',block[finite],coef)+ym
  if not np.isfinite(values[finite]).all():raise ValueError('Nonfinite Ridge predictions')
  pred[target]=values
  audit.append(dict(prediction_month=str(month),training_last_signal=str(idx[end-1].date()),latest_label_open=str(idx[end+5].date()),first_prediction=str(idx[t].date()),samples=len(yy),alpha=alpha))
 return pd.DataFrame(pred,index=idx,columns=f['score'].columns),audit
