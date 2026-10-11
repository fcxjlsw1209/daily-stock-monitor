"""Small conditional correction to the inferred monthly ranking, not daily trading.

Three depth-two regression trees learn residual membership in alternative image
portfolios. Tree score is capped at10%; price features still use signal-date data.
Image-derived fit labels are retrospective. All model rules and trials are saved.
"""
import json
import numpy as np
import pandas as pd
from common import P,O,base,FEATURES,features,score_frame,summary,dump
from coherent_fit import build_cache,evaluate
from fit import project_l1,triple_partition,match_orders
from inverse import load_observations


def predict(tree,X):
    if 'value' in tree:return np.full(X.shape[:-1],tree['value'])
    return np.where(X[...,tree['feature']]<=tree['threshold'],predict(tree['left'],X),predict(tree['right'],X))


def fit_tree(X,y,w,depth=2,min_leaf=60):
    mean=float(np.sum(w*y)/w.sum());best=float(np.sum(w*(y-mean)**2));split=None
    if not depth or len(y)<2*min_leaf:return dict(value=mean,n=len(y))
    for j in range(X.shape[1]):
        for threshold in np.unique(np.quantile(X[:,j],[.2,.4,.6,.8])):
            left=X[:,j]<=threshold
            if min(left.sum(),(~left).sum())<min_leaf:continue
            error=0.
            for mask in [left,~left]:
                avg=np.sum(w[mask]*y[mask])/w[mask].sum();error+=np.sum(w[mask]*(y[mask]-avg)**2)
            if error<best-1e-12:best=error;split=j,float(threshold),left
    if split is None:return dict(value=mean,n=len(y))
    j,t,l=split
    return dict(feature=j,feature_name=FEATURES[j],threshold=t,left=fit_tree(X[l],y[l],w[l],depth-1,min_leaf),right=fit_tree(X[~l],y[~l],w[~l],depth-1,min_leaf))


def learn(d,coef,ci):
    score=.5*d['C'][:,:,ci]+np.einsum('mnd,d->mn',d['X'],coef)
    _,prior=triple_partition(score/.12,d['mask'])
    target=np.zeros_like(prior)
    # Build expected stock inclusion in each month's whole alternative bag.
    # BX is insufficient to recover identities: raw bag indices are supplied.
    for i,(triples,weight) in enumerate(d['bags']):
        for triple,q in zip(triples,weight):target[i,triple]+=q
    mask=d['mask'];X=d['X'][mask];y=(target-prior)[mask]
    w=np.broadcast_to(d['month_weight'][:,None]/mask.sum(axis=1)[:,None],mask.shape)[mask]
    trees=[];fit=np.zeros_like(y)
    for _ in range(3):
        tree=fit_tree(X,y-fit,w);trees.append(tree);fit+=.5*predict(tree,X)
    return trees


def correction(trees,X,mask):
    value=sum(.5*predict(t,X) for t in trees)
    return pd.DataFrame(value).where(mask).rank(axis=1,pct=True).fillna(.5).to_numpy()-.5


def refine(d,ci,warm,seed):
    rng=np.random.default_rng(seed);best=project_l1(warm,.4);archive=[best[None,:]]
    def ev(v):return evaluate(np.column_stack([v,np.full(len(v),.1)]),d,ci)[0]
    loss=float(ev(best[None,:])[0]);values=[np.array([loss])];trace=[]
    for step in range(12):
        trial=np.vstack([best+rng.normal(0,[.06,.03,.015][step//4],(48,len(FEATURES))),rng.normal(0,.08,(16,len(FEATURES)))])
        trial=np.array([project_l1(q,.4) for q in trial]);value=ev(trial);archive.append(trial);values.append(value);k=int(np.argmin(value))
        if value[k]<loss:best=trial[k].copy();loss=float(value[k])
        trace.append(dict(step=step+1,loss=loss))
    return best,loss,trace,np.vstack(archive),np.concatenate(values)


def run():
    dump('TREE_CONFIG.json',dict(core='Mixed raw and risk-adjusted skip-two-month momentum; fixedweight50%',
        linear_correction_l1=.4,tree_weight=.1,trees=3,depth=2,min_leaf=60,threshold_quantiles=[.2,.4,.6,.8],
        target='Residual inclusion in image-compatible candidate portfolios. Not future realized returns.',
        feature_selection='All15 preexisting features; no additional technical factors.',
        search='12*64 fixed-family coefficient trials per model plus initial; same curve discrepancy objective.',
        eras=['through2021','all_retrospective'],pools=['core17','theme17'],no_independent_oos=True))
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();old=json.loads((P/'POLISH_MODELS.json').read_text())
    results={};models={};curves={};orders={};matches={};(P/'tree_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        meta=json.loads((P/f'BAGS_{pool}.json').read_text());cache=build_cache(px,meta,cal,obs);raw,core,f,gate=features(px,fs,pools[pool]);sim=base.Simulator(px,fs,pools[pool])
        Xfull=np.stack([f[k].fillna(0).to_numpy() for k in FEATURES],axis=-1)
        for era in ['through2021','all_retrospective']:
            tag=f'{pool}__{era}__mixed';warm=np.array([old[tag]['coefficients'][k] for k in FEATURES]);use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            d['bags']=[(np.load(m['file'])['bag'],np.load(m['file'])['weights']) for m,u in zip(meta,use) if u];trees=learn(d,project_l1(warm,.4),2)
            ext=correction(trees,cache['X'],cache['mask']);expanded=dict(cache,X=np.concatenate([cache['X'],ext[:,:,None]],axis=-1));d={k:v[use] for k,v in expanded.items()}
            coef,loss,trace,trials,values=refine(d,2,warm,202610091)
            np.savez_compressed(P/'tree_trials'/f'{tag}.npz',coefficients=trials,loss=values)
            models[tag]=dict(coefficients=dict(zip(FEATURES,map(float,coef))),objective=loss,trees=trees,tree_weight=.1,trace=trace,evaluations=len(trials),
                fit_months=int(use.sum()),latest_fit_label=old[tag]['latest_fit_label'],full_sample_explanatory_fit=era=='all_retrospective')
            # Rank tree correction over the SAME qualified candidates as cached.
            eligible=sim.valid&gate
            treeframe=pd.DataFrame(correction(trees,Xfull,eligible.to_numpy()),index=px.index,columns=px.columns)
            sc=score_frame(core['mixed'],f,coef)+.1*treeframe
            for lag,cost in [(0,0),(1,10)]:
                key=tag+f'__lag{lag}_cost{cost}';nav,oo=sim.run(sc,gate,cost_bps=cost,signal_lag=lag);mm,rows=match_orders(oo,meta);late=nav.loc['2021-12-31':].copy();late=late/late.iloc[0]
                results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(late),fit_era=era,objective=loss,
                    signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,independent_oos=False)
                curves[key]=nav;orders[key]=oo;matches[key]=rows
                if lag==0:
                    _,detail=evaluate(np.r_[coef,.1][None,:],expanded,2);want=[[meta[i]['symbols'][j] for j in q] for i,q in enumerate(detail['chosen'][0])]
                    assert list(map(set,want))==[set(o['holdings']) for o in oo]
                    e=cache['E'][np.arange(len(meta)),detail['combo_ids'][0]];assert np.isclose(np.exp(e.sum()),nav.iloc[-1],rtol=1e-12)
                print(key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),flush=True)
            dump('TREE_MODELS.json',models);dump('TREE_RESULTS.json',results);dump('TREE_ORDERS.json',orders);dump('TREE_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'TREE_CURVES.csv.gz')
    print('DONE',flush=True)


if __name__=='__main__':run()
