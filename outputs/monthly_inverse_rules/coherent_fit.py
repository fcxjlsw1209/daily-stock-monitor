"""Refine inferred rules against one continuous curve, not isolated month hits.

Weights are constant across the replay. Search criteria are curve discrepancy and
drawdown discrepancy, never a target CAGR. Full-sample fits remain explanatory.
Every evaluated coefficient vector is retained for the multiple-testing audit.
"""
import json
import numpy as np
import pandas as pd
from common import P,O,base,START,END,FEATURES,features,score_frame,summary,dump
from fit import pack,project_l1,match_orders
from inverse import load_observations,interpolation_matrix,mul


def build_cache(px,meta,cal,obs):
    d=pack(meta);M=len(meta);N=d['X'].shape[1];K=max(len(np.load(m['file'])['combos']) for m in meta)
    T=max(len(px.loc[m['signal_date']:m['label_month_end']]) for m in meta)
    lookup=np.full((M,N,N,N),-1,dtype=np.int32);A=np.zeros((M,K));B=A.copy();E=A.copy();L=np.zeros((M,K,T));length=[];targetdd=[]
    ref=obs['wealth'];axis=cal['wealth'];first=pd.Timestamp(meta[0]['signal_date']);x0=axis['x0']+axis['width']*(first-START).days/(END-START).days
    target0=np.interp(x0,ref.x,ref.log_wealth)
    for m,z in enumerate(meta):
        q=np.load(z['file']);combos=q['combos'];frame=px.loc[z['signal_date']:z['label_month_end'],z['symbols']]
        path=np.mean((frame/frame.iloc[0]).to_numpy()[:,combos],axis=2).T
        x=axis['x0']+axis['width']*((frame.index-START).days/(END-START).days).to_numpy()
        pred=[];target=[];weights=[]
        for panel in ['wealth','relative']:
            ob=obs[panel];ob=ob[(ob.x>x[0]+.8)&(ob.x<x[-1]-.8)]
            pred.append(np.log(mul(path,interpolation_matrix(x,ob.x.to_numpy()))))
            target.append(ob.log_wealth.to_numpy()-target0)
            weights.append(1/(np.maximum(1,(ob.ymax-ob.ymin).to_numpy()/2+.5)/cal[panel]['log_scale'])**2)
        delta=np.concatenate(pred,axis=1)-np.concatenate(target)[None,:];w=np.concatenate(weights);w/=w.sum()
        n=len(combos);A[m,:n]=np.sum(delta**2*w,axis=1);B[m,:n]=np.sum(delta*w,axis=1);E[m,:n]=np.log(path[:,-1]);L[m,:n,:len(frame)]=np.log(path);L[m,:n,len(frame):]=np.log(path[:,-1,None])
        lookup[m,combos[:,0],combos[:,1],combos[:,2]]=np.arange(n)
        ob=obs['dd'];ob=ob[(ob.x>x[0]+.8)&(ob.x<x[-1]-.8)]
        targetdd.append(float((cal['dd']['y0']-ob.ymax.max())/cal['dd']['scale']) if len(ob) else np.nan)
        length.append(len(frame))
    d.update(A=A,B=B,E=E,L=L,lookup=lookup,length=np.array(length),targetdd=np.array(targetdd))
    return d


def evaluate(coefs,d,core_id):
    sc=.5*d['C'][None,:,:,core_id]+np.einsum('mnd,bd->bmn',d['X'],coefs)
    sc=np.where(d['mask'][None,:,:],sc,-np.inf)
    chosen=np.sort(np.argsort(-sc,axis=2,kind='stable')[:,:,:3],axis=2)
    mi=np.arange(sc.shape[1])[None,:];cid=d['lookup'][mi,chosen[:,:,0],chosen[:,:,1],chosen[:,:,2]];assert (cid>=0).all()
    end=d['E'][mi,cid];level=np.pad(np.cumsum(end,axis=1)[:,:-1],((0,0),(1,0)))
    a=d['A'][mi,cid];b=d['B'][mi,cid]
    weight=d['month_weight'];weight=weight/weight.sum()
    globalrmse=np.sqrt(np.maximum(0,np.sum(weight*(a+2*level*b+level**2),axis=1)))
    shapermse=np.sqrt(np.maximum(0,np.sum(weight*(a-b*b),axis=1)))
    paths=d['L'][mi,cid]+level[:,:,None]
    peakmonth=paths.max(axis=2);prior=np.pad(np.maximum.accumulate(peakmonth,axis=1)[:,:-1],((0,0),(1,0)))
    peaks=np.maximum(np.maximum.accumulate(paths,axis=2),prior[:,:,None])
    monthlydd=(np.exp(paths-peaks)-1).min(axis=2)
    valid=np.isfinite(d['targetdd']);ddrmse=np.sqrt(np.mean((monthlydd[:,valid]-d['targetdd'][None,valid])**2,axis=1))
    loss=globalrmse+4*shapermse+.5*ddrmse
    return loss,dict(global_log_rmse=globalrmse,month_shape_log_rmse=shapermse,monthly_dd_rmse=ddrmse,chosen=chosen,combo_ids=cid)


def search(d,core_id,warm,seed):
    rng=np.random.default_rng(seed);initial=np.vstack([warm,np.zeros(len(FEATURES))]+[project_l1(rng.normal(0,.06,len(FEATURES))) for _ in range(30)])
    archive=[initial];loss,_=evaluate(initial,d,core_id);values=[loss];order=np.argsort(loss);elite=initial[order[:4]];best=elite[0].copy();bestloss=float(loss[order[0]]);trace=[]
    for iteration in range(24):
        scale=[.08,.04,.02,.01][iteration//6]
        around=best+rng.normal(0,scale,(32,len(FEATURES)))
        diverse=elite[rng.integers(0,len(elite),16)]+rng.normal(0,scale*1.5,(16,len(FEATURES)))
        random=rng.normal(0,.12,(16,len(FEATURES)))
        trial=np.vstack([project_l1(w) for w in np.vstack([around,diverse,random])]);scores,_=evaluate(trial,d,core_id)
        archive.append(trial);values.append(scores)
        combined=np.vstack([elite,trial]);combinedloss,_=evaluate(combined,d,core_id);elite=combined[np.argsort(combinedloss)[:4]]
        if float(combinedloss.min())<bestloss:bestloss=float(combinedloss.min());best=elite[0].copy()
        trace.append(dict(iteration=iteration+1,loss=bestloss,scale=scale))
    return best,bestloss,trace,np.vstack(archive),np.concatenate(values)


def run():
    cfg=dict(reason='Local candidate matches can coexist with a large cumulative wealth gap. Infer weights against the continuously compounded curve and monthly drawdowns.',
             objective='Global logNAVRMSE +4*month-centered logshapeRMSE +.5*monthly worst-drawdownRMSE; confidence-weighted months.',
             variables='One fixed15-coefficient ranking vector; L1<=.5, coremomentumweight=.5. No per-month fitted weights.',
             core_families=['raw','risk','mixed'],fit_eras=['through2021','all_retrospective'],generations=24,evaluations_per_generation=64,
             audit='All tested coefficient vectors and losses are saved. Full-sample fits are retrospective explanations, not independent validation. Optimizer does not consume CAGR or Sharpe.',
             choice='Select per pool by early-only training curve objective, not full-period return. ReportSharpe/CAGR/DD for all final models.')
    dump('COHERENT_CONFIG.json',cfg);px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();previous=json.loads((P/'MODELS.json').read_text())
    models={};results={};orders_out={};matches={};curves={};(P/'coherent_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        meta=json.loads((P/f'BAGS_{pool}.json').read_text());member=pools[pool];raw,core,f,gate=features(px,fs,member);sim=base.Simulator(px,fs,member)
        cache=build_cache(px,meta,cal,obs)
        for era in cfg['fit_eras']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            for ci,family in enumerate(cfg['core_families']):
                tag=f'{pool}__{era}__{family}';warm=np.array([previous[tag]['coefficients'][k] for k in FEATURES])
                coef,loss,trace,trials,losses=search(d,ci,warm,20261009+ci)
                np.savez_compressed(P/'coherent_trials'/f'{tag}.npz',coefficients=trials,loss=losses)
                models[tag]=dict(coefficients={k:float(w) for k,w in zip(FEATURES,coef)},objective=loss,trace=trace,evaluations=len(trials),
                                 latest_fit_label=meta[np.flatnonzero(use)[-1]]['label_month_end'],fit_months=int(use.sum()),full_sample_explanatory_fit=era=='all_retrospective')
                dump('COHERENT_MODELS.json',models);sc=score_frame(core[family],f,coef)
                for lag,cost in [(0,0),(1,10)]:
                    nav,oo=sim.run(sc,gate,cost_bps=cost,signal_lag=lag);key=tag+f'__lag{lag}_cost{cost}';match,rows=match_orders(oo,meta)
                    later=nav.loc['2021-12-31':];later=later/later.iloc[0]
                    results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(later),fit_era=era,
                                      objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=match,independent_oos=False)
                    curves[key]=nav;orders_out[key]=oo;matches[key]=rows
                    if lag==0:
                        _,detail=evaluate(coef[None,:],cache,ci)
                        expected=[[meta[i]['symbols'][j] for j in row] for i,row in enumerate(detail['chosen'][0])]
                        assert [set(h) for h in expected]==[set(o['holdings']) for o in oo]
                        end=cache['E'][np.arange(len(meta)),detail['combo_ids'][0]]
                        assert np.isclose(np.exp(end.sum()),nav.iloc[-1],rtol=1e-12)
                    print(key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),flush=True)
                dump('COHERENT_RESULTS.json',results);dump('COHERENT_ORDERS.json',orders_out);dump('COHERENT_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'COHERENT_CURVES.csv.gz')
    selected={pool:min((k for k in results if k.startswith(pool+'__through2021') and results[k]['signal_lag']==0),key=lambda k:results[k]['objective']) for pool in ['core17','theme17']}
    dump('COHERENT_SELECTION.json',dict(selected_by_early_curve_fit=selected,model_count=len(models),coefficient_evaluations=sum(v['evaluations'] for v in models.values()),
                                      replay_count=len(results),best_full_cagr_retrospective=max(results,key=lambda k:results[k]['cagr_252']),author_strategy_reproduced=False))
    print('COHERENT_SELECTED',selected,flush=True)


if __name__=='__main__':run()
