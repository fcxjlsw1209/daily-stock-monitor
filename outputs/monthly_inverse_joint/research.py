import json
import numpy as np
import pandas as pd
from engine import *
from fit import match_orders
from validate import independent_nav,strict_subset


def run():
    cfg=dict(purpose='Continue reconstructing the monthly strategy from alternative monthly holdings and the entire image.',
        variants=['stock_only: unchanged15stockfeatures, better joint optimizer','portfolio: stockcorrection40% plus10% portfolio diversification/volatility dispersion'],
        constant_core='50% mixed raw and risk-adjusted R12-2/R6-2 momentum; positive R12-2 in top half.',
        portfolio_hypothesis='Three stocks may be selected jointly, rather than each independently. This is a hypothesis, not a claim about the author.',
        training='Analytic gradient through soft portfolio mixtures and continuous compounded wealth; hard one-triple replay only for reported results.',
        objective='Same globallogNAVRMSE +4*monthlyshapelogRMSE +.5*monthlyDDRMSE as previous study. No return or Sharpe target supplied to optimizer.',
        soft_temperatures=[.08,.04,.02,.01],iterations_per_temperature=40,restarts=2,hard_refinement=288,
        eras=['through2021','all_retrospective'],pools=['core17','theme17'],no_independent_oos=True,
        no_future_month_rankings=True,original_accounts_unchanged=True)
    dump('CONFIG.json',cfg)
    for f in ['PROTECTED_BEFORE.json','SOURCE_HASHES.json']:dump(f,json.loads((OLD/f).read_text()))
    dump('PREVIOUS_STUDY_HASHES.json',{str(f):base.sha(f) for f in OLD.glob('*') if f.is_file() and f.suffix in ['.py','.json','.txt','.gz','.png']})
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();old=json.loads((OLD/'POLISH_MODELS.json').read_text())
    models={};results={};curves={};orders={};matches={};(P/'trials').mkdir(exist_ok=True)
    for pool in cfg['pools']:
        meta=json.loads((OLD/f'BAGS_{pool}.json').read_text());print('CACHE',pool,flush=True);cache=prepare(px,meta,cal,obs)
        for era in cfg['eras']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            warm=np.r_[[old[f'{pool}__{era}__mixed']['coefficients'][k] for k in FEATURES],0.,0.]
            initial=hard_objective(warm[None,:],d)[0][0];assert np.isclose(initial,old[f'{pool}__{era}__mixed']['objective'],rtol=1e-10)
            for mode in ['stock_only','portfolio']:
                tag=f'{pool}__{era}__{mode}';print('FIT_START',tag,flush=True)
                best,trials=optimize(d,warm,mode,202610092);loss,coef=best;np.savez_compressed(P/'trials'/f'{tag}.npz',**trials)
                models[tag]=dict(coefficients=dict(zip(FEATURES+EXTRA,map(float,coef))),objective=loss,starting_objective=float(initial),
                    fit_months=int(use.sum()),latest_fit_label=meta[np.flatnonzero(use)[-1]]['label_month_end'],
                    soft_evaluations=len(trials['soft_losses']),hard_refinement_evaluations=len(trials['hard_losses']),
                    full_sample_explanatory_fit=era=='all_retrospective')
                dump('MODELS.json',models)
                for lag,cost in [(0,0),(0,10),(1,10)]:
                    sim,sc,gate,decisions=policy_scores(px,fs,pools[pool],coef,mode,lag);nav,oo=sim.run(sc,gate,signal_lag=lag,cost_bps=cost)
                    rebuilt=independent_nav(px,oo,nav.index,cost);delta=float(((rebuilt-nav)/nav).abs().max());assert delta<1e-11
                    if lag==0:
                        _,ids=hard_objective(coef[None,:],cache)
                        want=[]
                        for i,(m,cid) in enumerate(zip(meta,ids[0])):
                            z=np.load(m['file']);h=z['combos'][cid];want.append([m['symbols'][j] for j in h]);assert decisions[i]['eligible_symbols']==m['symbols']
                        assert list(map(set,want))==[set(o['holdings']) for o in oo]
                        if cost==0:assert np.isclose(np.prod(cache['paths'][np.arange(len(meta)),ids[0],-1]),nav.iloc[-1],rtol=1e-12)
                    key=tag+f'__lag{lag}_cost{cost}';mm,rows=match_orders(oo,meta);strict=[]
                    for o,m in zip(oo,meta):strict.append(any(set(o['holdings'])==set(q['holdings']) for q in strict_subset(m['candidate_triples'])))
                    late=nav.loc['2021-12-31':].copy();late=late/late.iloc[0]
                    results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(late),
                        fit_era=era,mode=mode,objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,
                        strict_hits=sum(strict),strict_anchor_hits=sum(s and m['anchor'] for s,m in zip(strict,meta)),
                        independent_accounting_error=delta,independent_oos=False)
                    curves[key]=nav;orders[key]=oo;matches[key]=rows
                    print('RESULT',key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),'strict',sum(strict),flush=True)
                dump('RESULTS.json',results);dump('ORDERS.json',orders);dump('MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'CURVES.csv.gz')
    development=min((k for k in results if results[k]['fit_era']=='through2021' and results[k]['signal_lag']==0 and results[k]['cost_bps_roundtrip']==0),key=lambda k:results[k]['objective'])
    explanation=min((k for k in results if results[k]['fit_era']=='all_retrospective' and results[k]['signal_lag']==0 and results[k]['cost_bps_roundtrip']==0),key=lambda k:results[k]['objective'])
    dump('SELECTION.json',dict(early_curve_selected=development,full_curve_selected=explanation,best_full_cagr_retrospective=max(results,key=lambda k:results[k]['cagr_252']),
        best_full_sharpe_retrospective=max(results,key=lambda k:results[k]['sharpe']),models=len(models),replays=len(results),author_strategy_reproduced=False))
    print('DONE',flush=True)


if __name__=='__main__':run()
