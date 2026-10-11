"""Differentiable inverse learning of ONE monthly ranking/portfolio rule.

The soft mixture is an optimization surrogate, never a reported trading result.
Actual replays select one equal-weight triple using signal-date features only.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent;OLD=O/'monthly_inverse_rules'
sys.path.insert(0,str(OLD))
from common import base,START,END,FEATURES,features,summary
from coherent_fit import build_cache
from fit import project_l1
from inverse import load_observations,interpolation_matrix,mul
from itertools import combinations

EXTRA=['diversification','vol_dispersion']

def dump(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def combination_features(px,signal,names,combos):
    # Complete253-session qualification implies complete63-session returns.
    i=px.index.get_loc(signal);ret=px[names].iloc[i-63:i+1].pct_change(fill_method=None).iloc[1:].to_numpy()
    cov=np.cov(ret,rowvar=False,ddof=1);vol=np.sqrt(np.maximum(np.diag(cov),1e-16));a,b,c=combos.T
    variance=(vol[a]**2+vol[b]**2+vol[c]**2+2*(cov[a,b]+cov[a,c]+cov[b,c]))/9
    pvol=np.sqrt(np.maximum(variance,0));meanvol=vol[combos].mean(axis=1)
    diversification=1-pvol/meanvol
    dispersion=np.log(vol[combos].max(axis=1)/vol[combos].min(axis=1))
    return pd.DataFrame(dict(diversification=diversification,vol_dispersion=dispersion)).rank(pct=True).to_numpy()-.5


def prepare(px,meta,cal,obs):
    stock=build_cache(px,meta,cal,obs);M=len(meta);K=stock['E'].shape[1];T=stock['L'].shape[2]
    F=np.zeros((M,K,len(FEATURES)+len(EXTRA)));C=np.zeros((M,K));mask=np.zeros((M,K),bool)
    rr=[];yy=[];ww=[]
    axis=cal['wealth'];first=pd.Timestamp(meta[0]['signal_date']);x0=axis['x0']+axis['width']*(first-START).days/(END-START).days
    offset=np.interp(x0,obs['wealth'].x,obs['wealth'].log_wealth)
    for j,m in enumerate(meta):
        z=np.load(m['file']);h=z['combos'];n=len(h);mask[j,:n]=True
        F[j,:n,:len(FEATURES)]=z['X'][h].mean(axis=1);C[j,:n]=z['C'][h,2].mean(axis=1)
        F[j,:n,len(FEATURES):]=combination_features(px,pd.Timestamp(m['signal_date']),m['symbols'],h)
        frame=px.loc[m['signal_date']:m['label_month_end'],m['symbols']];path=np.mean((frame/frame.iloc[0]).to_numpy()[:,h],axis=2).T
        x=axis['x0']+axis['width']*((frame.index-START).days/(END-START).days).to_numpy()
        paths=[];target=[];weight=[]
        for panel in ['wealth','relative']:
            ob=obs[panel];ob=ob[(ob.x>x[0]+.8)&(ob.x<x[-1]-.8)]
            paths.append(mul(path,interpolation_matrix(x,ob.x.to_numpy())));target.append(ob.log_wealth.to_numpy()-offset)
            weight.append(1/(np.maximum(1,(ob.ymax-ob.ymin).to_numpy()/2+.5)/cal[panel]['log_scale'])**2)
        rr.append(np.concatenate(paths,axis=1));yy.append(np.concatenate(target));w=np.concatenate(weight);ww.append(w/w.sum())
    J=max(map(len,yy));R=np.ones((M,K,J));Y=np.zeros((M,J));W=np.zeros((M,J))
    for m in range(M):
        n,j=rr[m].shape;R[m,:n,:j]=rr[m];Y[m,:j]=yy[m];W[m,:j]=ww[m]
    return dict(F=F,C=C,mask=mask,R=R,Y=Y,W=W,paths=np.exp(stock['L']),month_weight=stock['month_weight'],targetdd=stock['targetdd'],A=stock['A'],B=stock['B'],E=stock['E'])


def path_loss(path,observed,d,gradient=False):
    """Image loss and analytic adjoint through continuous compounded wealth."""
    mw=d['month_weight']/d['month_weight'].sum();ow=d['W'];M,T=path.shape
    end=np.log(path[:,-1]);level=np.r_[0,np.cumsum(end)[:-1]]
    residual=np.log(observed)+level[:,None]-d['Y'];mean=np.sum(ow*residual,axis=1)
    global_rmse=np.sqrt(max(1e-24,np.sum(mw[:,None]*ow*residual**2)))
    centered=residual-mean[:,None];shape_rmse=np.sqrt(max(1e-24,np.sum(mw[:,None]*ow*centered**2)))
    logpath=np.log(path)+level[:,None];flat=logpath.ravel();peaks=np.maximum.accumulate(np.r_[0.,flat])
    resets=np.r_[True,flat>=peaks[:-1]];peak_ids=np.maximum.accumulate(np.where(resets,np.arange(len(flat)+1),0))[1:]-1
    draw=np.exp(flat-peaks[1:])-1;worst=np.argmin(draw.reshape(M,T),axis=1)+np.arange(M)*T;dd=draw[worst]
    valid=np.isfinite(d['targetdd']);ddres=dd[valid]-d['targetdd'][valid];ddrmse=np.sqrt(max(1e-24,np.mean(ddres**2)))
    loss=global_rmse+4*shape_rmse+.5*ddrmse
    details=dict(global_log_rmse=float(global_rmse),month_shape_log_rmse=float(shape_rmse),monthly_dd_rmse=float(ddrmse))
    if not gradient:return float(loss),details
    gh=mw[:,None]*ow*(residual/global_rmse+4*centered/shape_rmse)
    gd=np.zeros_like(flat);g=.5*ddres/(valid.sum()*ddrmse)*(1+dd[valid]);current=worst[valid];previous=peak_ids[current]
    np.add.at(gd,current,g);ok=previous>=0;np.add.at(gd,previous[ok],-g[ok]);gd=gd.reshape(M,T)
    level_gradient=gh.sum(axis=1)+gd.sum(axis=1)
    future=np.cumsum(level_gradient[::-1])[::-1]-level_gradient
    gp=gd/path;gp[:,-1]+=future/path[:,-1]
    return float(loss),details,gh/observed,gp


def soft_objective(coef,d,temperature):
    score=d.get('core_weight',.5)*d['C']+np.einsum('mkd,d->mk',d['F'],coef)
    score=np.where(d['mask'],score/temperature,-np.inf);peak=score.max(axis=1);q=np.exp(score-peak[:,None]);q/=q.sum(axis=1)[:,None]
    path=np.einsum('mk,mkt->mt',q,d['paths']);observed=np.einsum('mk,mkj->mj',q,d['R'])
    loss,detail,go,gp=path_loss(path,observed,d,True)
    gq=np.einsum('mkt,mt->mk',d['paths'],gp)+np.einsum('mkj,mj->mk',d['R'],go)
    gs=q*(gq-np.sum(gq*q,axis=1)[:,None])/temperature
    grad=np.einsum('mk,mkd->d',gs,d['F'])
    return loss,grad


def hard_objective(coefs,d):
    values=[];choices=[]
    for coef in coefs:
        score=np.where(d['mask'],d.get('core_weight',.5)*d['C']+np.einsum('mkd,d->mk',d['F'],coef),-np.inf)
        c=np.argmax(score,axis=1);mi=np.arange(len(c));loss,_=path_loss(d['paths'][mi,c],d['R'][mi,c],d)
        values.append(loss);choices.append(c)
    return np.array(values),np.array(choices)


def project(w,mode,core_weight=.5):
    result=w.copy()
    if mode=='stock_only':result[:15]=project_l1(result[:15],1-core_weight);result[15:]=0
    else:result[:15]=project_l1(result[:15],.9-core_weight);result[15:]=project_l1(result[15:],.1)
    return result


def optimize(d,warm,mode,seed):
    rng=np.random.default_rng(seed);archive=[];losses=[];steps=[];best=(np.inf,None);cw=d.get('core_weight',.5)
    for restart,start in enumerate([warm,np.zeros_like(warm)]):
        w=project(start,mode,cw);first=np.zeros_like(w);second=np.zeros_like(w);step=0
        for temperature in [.08,.04,.02,.01]:
            for iteration in range(40):
                step+=1;loss,grad=soft_objective(w,d,temperature)
                assert np.isfinite(loss) and np.isfinite(grad).all()
                archive.append(w.copy());losses.append(loss);steps.append((restart,step,temperature))
                if iteration%5==0:
                    value=hard_objective(w[None,:],d)[0][0]
                    if value<best[0]:best=(float(value),w.copy())
                first=.9*first+.1*grad;second=.999*second+.001*grad**2
                update=(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
                w=project(w-.008*update,mode,cw)
        value=hard_objective(w[None,:],d)[0][0]
        archive.append(w.copy());losses.append(soft_objective(w,d,.01)[0]);steps.append((restart,step+1,.01))
        if value<best[0]:best=(float(value),w.copy())
    # Bounded hard-selection refinement removes dependence on the soft surrogate.
    candidates=[best[1].copy()];hard_values=[best[0]]
    for scale in [.02,.01,.005]:
        trial=np.array([project(best[1]+rng.normal(0,scale,len(warm)),mode,cw) for _ in range(96)])
        value,_=hard_objective(trial,d);candidates.extend(trial);hard_values.extend(value);k=int(np.argmin(value))
        if value[k]<best[0]:best=(float(value[k]),trial[k].copy())
    return best,dict(soft_coefficients=np.array(archive),soft_losses=np.array(losses),soft_steps=np.array(steps),hard_coefficients=np.array(candidates),hard_losses=np.array(hard_values))


def policy_scores(px,fs,member,coef,mode,signal_lag=0,gate_quantile=.5):
    raw,core,f,gate=features(px,fs,member)
    if gate_quantile!=.5:gate=(raw['m252_42']>0)&raw['m252_42'].where(member).rank(axis=1,pct=True).ge(gate_quantile)
    sim=base.Simulator(px,fs,member);out=pd.DataFrame(np.nan,index=px.index,columns=px.columns);details=[]
    for i in sim.events:
        signal=px.index[i-signal_lag];eligible=sim.valid.loc[signal]&gate.loc[signal];names=sorted(px.columns[eligible]);k=min(3,len(names))
        if k<2:chosen=[];cid=None
        else:
            h=np.array(list(combinations(range(len(names)),k)),dtype=int)
            X=np.array([[float(f[z].at[signal,s]) for z in FEATURES] for s in names]);C=core['mixed'].loc[signal,names].to_numpy()
            stock=.5*C+np.einsum('nd,d->n',X,coef[:15]);score=stock[h].mean(axis=1)
            if mode=='portfolio' and k==3:score+=np.einsum('kd,d->k',combination_features(px,signal,names,h),coef[15:])
            cid=int(np.argmax(score));chosen=[names[j] for j in h[cid]]
        if chosen:out.loc[signal,chosen]=np.arange(len(chosen),0,-1)
        details.append(dict(signal_date=str(signal.date()),holdings=chosen,eligible_symbols=names))
    return sim,out,gate,details
