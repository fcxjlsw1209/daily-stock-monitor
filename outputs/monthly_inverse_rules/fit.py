"""Learn one ranking function from alternative image-consistent portfolios.

Multiple-instance likelihood keeps the monthly alternatives intact. It does not
pretend that the average of incompatible portfolios is the author's holding.
No optimization term uses a strategy's CAGR or Sharpe.
"""
import json
import numpy as np
import pandas as pd
from common import P,O,OLD,base,FEATURES,features,score_frame,summary,dump


def shift_cumsum(a):
    return np.pad(np.cumsum(a,axis=1)[:,:-1],((0,0),(1,0)))


def triple_partition(logit,mask):
    """Log sum exp over all unordered triples and each name's inclusion prob."""
    peak=np.max(np.where(mask,logit,-np.inf),axis=1)
    value=np.exp(np.where(mask,logit-peak[:,None],-np.inf))
    left1=shift_cumsum(value);left2=shift_cumsum(value*left1)
    rev=value[:,::-1];right1=shift_cumsum(rev)[:,::-1];right2=shift_cumsum(rev*shift_cumsum(rev))[:,::-1]
    z=np.sum(value*left2,axis=1)
    assert np.all(z>0)
    inclusion=value*(left2+right2+left1*right1)/z[:,None]
    return np.log(z)+3*peak,inclusion


def objective(coef,d,core_id,temperature=.04,penalty=.2):
    score=.5*d['C'][:,:,core_id]+np.einsum('mnd,d->mn',d['X'],coef)
    logden,inclusion=triple_partition(score/(3*temperature),d['mask'])
    bagscore=.5*d['BC'][:,:,core_id]+np.einsum('mkd,d->mk',d['BX'],coef)
    a=bagscore/temperature+d['logprior'];peak=a.max(axis=1);q=np.exp(a-peak[:,None]);q/=q.sum(axis=1)[:,None]
    lognum=peak+np.log(np.exp(a-peak[:,None]).sum(axis=1))
    neg=np.einsum('mn,mnd->md',inclusion,d['X'])/3
    pos=np.einsum('mk,mkd->md',q,d['BX'])
    w=d['month_weight'];w=w/w.sum()
    loss=float(np.sum(w*(logden-lognum))+penalty*np.dot(coef,coef))
    grad=np.einsum('m,md->d',w,(neg-pos)/temperature)+2*penalty*coef
    return loss,grad


def project_l1(x,radius=.5):
    if np.abs(x).sum()<=radius:return x
    u=np.sort(np.abs(x))[::-1];s=np.cumsum(u)
    take=np.flatnonzero(u>(s-radius)/np.arange(1,len(x)+1))[-1]
    threshold=(s[take]-radius)/(take+1)
    return np.sign(x)*np.maximum(np.abs(x)-threshold,0)


def pack(meta):
    M=len(meta);N=max(len(v['symbols']) for v in meta);K=max(len(v['candidate_triples']) for v in meta);D=len(FEATURES)
    X=np.zeros((M,N,D));C=np.zeros((M,N,3));mask=np.zeros((M,N),bool)
    BX=np.zeros((M,K,D));BC=np.zeros((M,K,3));logprior=np.full((M,K),-np.inf);confidence=[]
    for i,m in enumerate(meta):
        assert base.sha(m['file'])==m['sha256'];z=np.load(m['file']);n=len(m['symbols']);k=len(z['bag'])
        X[i,:n]=z['X'];C[i,:n]=z['C'];mask[i,:n]=True
        BX[i,:k]=z['X'][z['bag']].mean(axis=1);BC[i,:k]=z['C'][z['bag']].mean(axis=1)
        logprior[i,:k]=np.log(z['weights']);confidence.append(.1+.9*m['confidence']**2)
    return dict(X=X,C=C,mask=mask,BX=BX,BC=BC,logprior=logprior,month_weight=np.array(confidence))


def optimize(d,core_id,seed,iterations=350):
    rng=np.random.default_rng(seed);coef=np.zeros(len(FEATURES)) if seed==0 else project_l1(rng.normal(0,.05,len(FEATURES)))
    first=np.zeros_like(coef);second=np.zeros_like(coef);best=(np.inf,coef.copy());trace=[]
    for step in range(1,iterations+1):
        loss,grad=objective(coef,d,core_id)
        if loss<best[0]:best=(loss,coef.copy())
        first=.9*first+.1*grad;second=.999*second+.001*grad**2
        update=(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        coef=project_l1(coef-.01*update)
        if step==1 or step%50==0:trace.append(dict(iteration=step,loss=loss,l1=float(np.abs(coef).sum())))
    loss,_=objective(coef,d,core_id)
    if loss<best[0]:best=(loss,coef.copy())
    return best[1],dict(seed=seed,loss=best[0],trace=trace)


def match_orders(orders,meta):
    rows=[]
    for order,m in zip(orders,meta):
        assert order['date']==m['signal_date']
        chosen=set(order['holdings']);bags=m['candidate_triples'];hit=any(chosen==set(r['holdings']) for r in bags)
        nearest=max(bags,key=lambda r:(len(chosen&set(r['holdings'])),-r['robust_loss']))
        z=np.load(m['file']);mapping={s:i for i,s in enumerate(m['symbols'])};loss=None;percentile=None
        if len(chosen)==3 and all(s in mapping for s in chosen):
            ids=np.array(sorted(mapping[s] for s in chosen));where=np.flatnonzero(np.all(z['combos']==ids,axis=1))
            if len(where):loss=float(z['robust'][where[0]]);percentile=float(np.mean(z['robust']<=loss))
        rows.append(dict(month=m['month'],signal_date=order['signal_date'],holdings=order['holdings'],
                         anchor=m['anchor'],confidence=m['confidence'],candidate_hit=hit,
                         nearest_candidate=nearest['holdings'],common_names=sorted(chosen&set(nearest['holdings'])),
                         image_loss=loss,image_loss_percentile=percentile,verified_author_trade=False))
    result={}
    for label,subset in [('early',[r for r in rows if r['month']<'2022-01']),('later',[r for r in rows if r['month']>='2022-01']),('all',rows)]:
        anchors=[r for r in subset if r['anchor']]
        result[label]=dict(months=len(subset),candidate_hits=sum(r['candidate_hit'] for r in subset),
                           anchors=len(anchors),anchor_hits=sum(r['candidate_hit'] for r in anchors),
                           mean_image_percentile=float(np.mean([r['image_loss_percentile'] for r in subset if r['image_loss_percentile'] is not None])))
    return result,rows


def run():
    cfg=json.loads((P/'CONFIG.json').read_text());px,fs,pools=base.load();models={};results={};curves={};allorders={};allmatches={}
    def replay(key,sim,sc,gate,buffer,lag,cost,meta,era,model_key=None):
        nav,orders=sim.run(sc,gate,buffer=buffer,signal_lag=lag,cost_bps=cost)
        mm,mrows=match_orders(orders,meta);early=nav.loc[:'2021-12-31'];late=nav.loc['2021-12-31':];late=late/late.iloc[0]
        results[key]=dict(**summary(nav),early=summary(early),later_diagnostic=summary(late),fit_era=era,
                          model=model_key,buffer=buffer,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,
                          independent_oos=False,uses_full_sample_fit=era=='all_retrospective')
        curves[key]=nav;allorders[key]=orders;allmatches[key]=mrows
        dump('RESULTS.json',results)
        print('REPLAY',key,round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),flush=True)
    for pool in cfg['pools']:
        meta=json.loads((P/f'BAGS_{pool}.json').read_text());member=pools[pool];raw,core,f,gate=features(px,fs,member);sim=base.Simulator(px,fs,member)
        for lag,cost in [(0,0),(1,10)]:
            replay(f'{pool}__baseline_lag{lag}_cost{cost}',sim,core['raw'],gate,0,lag,cost,meta,'baseline')
        for era in cfg['inference']['fit_eras']:
            training=[m for m in meta if era=='all_retrospective' or m['month']<'2022-01'];d=pack(training)
            for core_id,family in enumerate(cfg['inference']['core_families']):
                tries=[];best=None
                for seed in [0,11,29]:
                    coef,audit=optimize(d,core_id,seed,cfg['inference']['iterations']);tries.append(audit)
                    if best is None or audit['loss']<best[0]:best=(audit['loss'],coef)
                    print('FIT',pool,era,family,seed,round(audit['loss'],4),flush=True)
                model_key=f'{pool}__{era}__{family}';coef=best[1]
                models[model_key]=dict(coefficients={k:float(w) for k,w in zip(FEATURES,coef)},fixed_strength_weight=.5,l1=float(np.abs(coef).sum()),
                                       loss=best[0],fit_months=len(training),latest_fit_label=training[-1]['label_month_end'],
                                       model_type='multiple-instance linear ranking with L2 regularization',restarts=tries,
                                       labels_available_only_after_author_image=True,full_sample_explanatory_fit=era=='all_retrospective')
                dump('MODELS.json',models);sc=score_frame(core[family],f,coef)
                for buffer in [0,5]:
                    for lag,cost in [(0,0),(1,10)]:
                        key=f'{model_key}__keep{buffer}_lag{lag}_cost{cost}'
                        replay(key,sim,sc,gate,buffer,lag,cost,meta,era,model_key)
                dump('ORDERS.json',allorders);dump('MONTH_MATCHES.json',allmatches);pd.DataFrame(curves).to_csv(P/'CURVES.csv.gz')
    # Selection never optimizes full-period annual return or Sharpe.
    selection={}
    for pool in cfg['pools']:
        possible={k:v for k,v in results.items() if k.startswith(pool) and v['fit_era']=='through2021' and v['signal_lag']==0}
        selected=min(possible,key=lambda k:(-possible[k]['matching']['early']['anchor_hits'],possible[k]['matching']['early']['mean_image_percentile']))
        selection[pool]=selected
    dump('SELECTION.json',dict(selected_by_early_candidate_match=selection,restarts_run=sum(len(v['restarts']) for v in models.values()),
                              model_count=len(models),replay_count=len(results),highest_full_period_cagr_retrospective=max(results,key=lambda k:results[k]['cagr_252']),
                              no_independent_oos=True,author_strategy_reproduced=False))
    print('SELECTION',json.dumps(selection),flush=True)


if __name__=='__main__':run()
