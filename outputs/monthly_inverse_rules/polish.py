"""Bounded numerical refinement of the SAME fixed-weight inverse rules.

No new factors, month-specific choices, CAGR target, or portfolio leverage.
Coordinate and pairwise moves improve the previously specified image objective.
All evaluated coefficients, including failures, are preserved.
"""
import json
import numpy as np
import pandas as pd
from common import P,O,base,FEATURES,features,score_frame,summary,dump
from coherent_fit import build_cache,evaluate
from fit import project_l1,match_orders
from inverse import load_observations


def refine(coef,d,ci,seed):
    rng=np.random.default_rng(seed);best=coef.copy();loss=float(evaluate(best[None,:],d,ci)[0][0]);archive=[best[None,:]];losses=[np.array([loss])];trace=[]
    def accept(trial):
        nonlocal best,loss
        trial=np.unique(trial,axis=0);scores=evaluate(trial,d,ci)[0];archive.append(trial);losses.append(scores)
        k=int(np.argmin(scores))
        if scores[k]<loss-1e-12:best=trial[k].copy();loss=float(scores[k])
    for sweep in range(6):
        initial=loss
        for j in rng.permutation(len(FEATURES)):
            radius=max(0,.5-(abs(best).sum()-abs(best[j])))
            trial=np.repeat(best[None,:],65,axis=0);trial[:,j]=np.linspace(-radius,radius,65);accept(trial)
        for _ in range(12):
            a,b=rng.choice(len(FEATURES),2,replace=False);direction=np.zeros(len(FEATURES));direction[a]=1;direction[b]=rng.choice([-1.,1.])
            width=[.08,.04,.02,.01,.005,.0025][sweep]
            trial=np.array([project_l1(best+t*direction) for t in np.linspace(-width,width,65)]);accept(trial)
        trace.append(dict(sweep=sweep+1,loss=loss,improvement=initial-loss))
    return best,loss,trace,np.vstack(archive),np.concatenate(losses)


def run():
    dump('POLISH_CONFIG.json',dict(purpose='Refine numerical search within the same15-feature constant-weight family; do not add model complexity.',
         objective='Same frozen continuous-curve objective as COHERENT_CONFIG.json; CAGR and Sharpe are not inputs.',sweeps=6,
         per_sweep='15 coordinate grids plus12 pair directions,65 points each; exact unchanged coefficients may be deduplicated.',
         independent_oos=False,full_fits='Retrospective explanatory calibration, never forward evidence.'))
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations()
    old=json.loads((P/'COHERENT_MODELS.json').read_text());models={};results={};orders={};matches={};curves={};(P/'polish_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        meta=json.loads((P/f'BAGS_{pool}.json').read_text());cache=build_cache(px,meta,cal,obs);member=pools[pool];raw,core,f,gate=features(px,fs,member);sim=base.Simulator(px,fs,member)
        for era in ['through2021','all_retrospective']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            for ci,family in enumerate(['raw','risk','mixed']):
                tag=f'{pool}__{era}__{family}';warm=np.array([old[tag]['coefficients'][k] for k in FEATURES])
                coef,loss,trace,trials,values=refine(warm,d,ci,202610090+ci)
                assert abs(coef).sum()<=.5+1e-12
                np.savez_compressed(P/'polish_trials'/f'{tag}.npz',coefficients=trials,loss=values)
                models[tag]=dict(coefficients=dict(zip(FEATURES,map(float,coef))),objective=loss,initial_objective=old[tag]['objective'],
                    trace=trace,evaluations=len(values),latest_fit_label=old[tag]['latest_fit_label'],fit_months=int(use.sum()),
                    full_sample_explanatory_fit=era=='all_retrospective')
                sc=score_frame(core[family],f,coef)
                for lag,cost in [(0,0),(1,10)]:
                    key=tag+f'__lag{lag}_cost{cost}';nav,oo=sim.run(sc,gate,cost_bps=cost,signal_lag=lag);mm,rr=match_orders(oo,meta)
                    later=nav.loc['2021-12-31':];later=later/later.iloc[0]
                    results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(later),fit_era=era,
                        objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,independent_oos=False)
                    curves[key]=nav;orders[key]=oo;matches[key]=rr
                    if lag==0:
                        _,detail=evaluate(coef[None,:],cache,ci);want=[[meta[i]['symbols'][j] for j in q] for i,q in enumerate(detail['chosen'][0])]
                        assert list(map(set,want))==[set(o['holdings']) for o in oo]
                        e=cache['E'][np.arange(len(meta)),detail['combo_ids'][0]];assert np.isclose(np.exp(e.sum()),nav.iloc[-1],rtol=1e-12)
                    print(key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),flush=True)
                dump('POLISH_MODELS.json',models);dump('POLISH_RESULTS.json',results);dump('POLISH_ORDERS.json',orders);dump('POLISH_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'POLISH_CURVES.csv.gz')
    selected={pool:min((k for k in results if k.startswith(pool+'__through2021') and results[k]['signal_lag']==0),key=lambda k:results[k]['objective']) for pool in ['core17','theme17']}
    dump('POLISH_SELECTION.json',dict(selected_by_early_curve_fit=selected,model_count=len(models),coefficient_evaluations=sum(v['evaluations'] for v in models.values()),
        replay_count=len(results),best_full_cagr_retrospective=max(results,key=lambda k:results[k]['cagr_252']),author_strategy_reproduced=False))
    print('DONE',flush=True)


if __name__=='__main__':run()
