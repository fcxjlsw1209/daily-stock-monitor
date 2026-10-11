"""Test30/40% explicit core weights: prior user limit was AT MOST50%, not fixed50%."""
from engine import *
from fit import match_orders
from validate import independent_nav,strict_subset


def run():
    dump('CORE_WEIGHT_CONFIG.json',dict(reason='Previous fits assumed50% fixed core although user specified a maximum. Test30% and40% without changing momentum definitions.',
        explicit_core_weights=[.3,.4],secondary_l1='1-coreweight',secondary_features=FEATURES,
        qualification='Same positiveR12-2 and top-half,253 completed prices.',
        caveat='Some secondary features also measure momentum; explicit core weight is not an economic exposure cap.',
        unchanged='Same monthly clock, original candidate bags, pools, optimizer, image objective, no leverage.',no_independent_oos=True))
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();old=json.loads((OLD/'POLISH_MODELS.json').read_text())
    results={};models={};orders={};curves={};matches={};(P/'core_weight_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        meta=json.loads((OLD/f'BAGS_{pool}.json').read_text());cache=prepare(px,meta,cal,obs);_,core,f,gate=features(px,fs,pools[pool]);sim=base.Simulator(px,fs,pools[pool])
        for era in ['through2021','all_retrospective']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()};warm=np.r_[[old[f'{pool}__{era}__mixed']['coefficients'][k] for k in FEATURES],0.,0.]
            for cw in [.3,.4]:
                tag=f'{pool}__{era}__core{int(cw*100)}';d['core_weight']=cw;print('FIT_START',tag,flush=True);best,trials=optimize(d,warm,'stock_only',202610095);loss,coef=best
                np.savez_compressed(P/'core_weight_trials'/f'{tag}.npz',**trials)
                models[tag]=dict(coefficients=dict(zip(FEATURES+EXTRA,map(float,coef))),core_weight=cw,objective=loss,fit_months=int(use.sum()),latest_fit_label=meta[np.flatnonzero(use)[-1]]['label_month_end'],
                    soft_evaluations=len(trials['soft_losses']),hard_refinement_evaluations=len(trials['hard_losses']),full_sample_explanatory_fit=era=='all_retrospective')
                sc=cw*core['mixed']+sum(float(coef[i])*f[k] for i,k in enumerate(FEATURES))
                for lag,cost in [(0,0),(0,10),(1,10)]:
                    nav,oo=sim.run(sc,gate,signal_lag=lag,cost_bps=cost);rebuilt=independent_nav(px,oo,nav.index,cost);delta=float(((rebuilt-nav)/nav).abs().max());assert delta<1e-11
                    if lag==0:
                        cache['core_weight']=cw;_,ids=hard_objective(coef[None,:],cache);del cache['core_weight'];want=[]
                        for m,cid in zip(meta,ids[0]):
                            z=np.load(m['file']);want.append([m['symbols'][j] for j in z['combos'][cid]])
                        assert list(map(set,want))==[set(o['holdings']) for o in oo]
                        if cost==0:assert np.isclose(np.prod(cache['paths'][np.arange(len(meta)),ids[0],-1]),nav.iloc[-1],rtol=1e-12)
                    key=tag+f'__lag{lag}_cost{cost}';mm,rows=match_orders(oo,meta);strict=[any(set(o['holdings'])==set(q['holdings']) for q in strict_subset(m['candidate_triples'])) for o,m in zip(oo,meta)]
                    late=nav.loc['2021-12-31':].copy();late=late/late.iloc[0]
                    results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(late),fit_era=era,core_weight=cw,objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,
                        strict_hits=sum(strict),strict_anchor_hits=sum(s and m['anchor'] for s,m in zip(strict,meta)),independent_accounting_error=delta,independent_oos=False)
                    curves[key]=nav;orders[key]=oo;matches[key]=rows
                    print('RESULT',key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),'strict',sum(strict),flush=True)
                dump('CORE_WEIGHT_MODELS.json',models);dump('CORE_WEIGHT_RESULTS.json',results);dump('CORE_WEIGHT_ORDERS.json',orders);dump('CORE_WEIGHT_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'CORE_WEIGHT_CURVES.csv.gz')
    print('CORE_WEIGHT_DONE',flush=True)


if __name__=='__main__':run()
