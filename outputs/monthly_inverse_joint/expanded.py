"""Test an unsupported eligibility assumption: top-half vs positive momentum.

Rebuild alternative portfolios before fitting: never reuse labels constructed
under the old narrower gate as though the eligibility assumptions were equal.
"""
from engine import *
from inverse import score_paths
from fit import match_orders
from validate import independent_nav,strict_subset


def bags(px,fs,member,pool,cal,obs):
    folder=P/'expanded_bags';folder.mkdir(exist_ok=True);target=P/f'EXPANDED_BAGS_{pool}.json'
    if target.exists():
        data=json.loads(target.read_text())
        if len(data)==121 and all(base.sha(m['file'])==m['sha256'] for m in data):return data
    raw,core,ff,_=features(px,fs,member);records=[]
    for month in pd.period_range('2016-09','2026-09',freq='M'):
        days=px.index[(px.index.to_period('M')==month)&(px.index<=END)];s=px.index[px.index.get_loc(days[0])-1];e=days[-1];i=px.index.get_loc(s);frame=px.loc[s:e]
        eligible=member.loc[s]&(raw['m252_42'].loc[s]>0)&px.iloc[i-252:i+1].notna().all()
        names=sorted(t for t in px if eligible[t] and frame[t].notna().all() and frame[t].gt(0).all());assert len(names)>=3
        h=np.asarray(list(combinations(range(len(names)),3)),dtype=int);paths=np.mean((frame[names]/frame[names].iloc[0]).to_numpy()[:,h],axis=2).T
        a=cal['wealth'];x=a['x0']+a['width']*((frame.index-START).days/(END-START).days).to_numpy();observed={k:v[(v.x>x[0]+.8)&(v.x<x[-1]-.8)].copy() for k,v in obs.items()}
        loss=[];best=[]
        for shift in [-.2,0,.2]:
            ref=obs['wealth'];prior=ref[ref.x<=x[0]+shift];constraint=dict(start_log_nav=float(np.interp(x[0]+shift,ref.x,ref.log_wealth)),prior_log_peak=max(0,float(prior.log_wealth.max())) if len(prior) else 0,log_tolerance=1.5/a['log_scale'])
            value=score_paths(paths,x,observed,cal,xshift=shift,constraints=constraint)[0];loss.append(value);best.append(set(names[j] for j in h[int(np.argmin(value))]))
        loss=np.stack(loss);relative=loss-loss.min(axis=1,keepdims=True);robust=np.max(relative,axis=0)+.25*loss.mean(axis=0);order=np.argsort(robust,kind='stable')[:64];hh=h[order]
        weights=np.exp(-(robust[order]-robust[order[0]])/.25);weights/=weights.sum();inclusion=np.zeros(len(names))
        for triple,w in zip(hh,weights):inclusion[triple]+=w
        share=inclusion/3;effective=float(np.exp(-np.sum(share[share>0]*np.log(share[share>0]))));concentration=float(np.clip((len(names)-effective)/(len(names)-3),0,1))
        X=np.array([[ff[k].at[s,t] for k in FEATURES] for t in names]);C=np.array([[core[k].at[s,t] for k in ['raw','risk','mixed']] for t in names]);assert np.isfinite(X).all() and np.isfinite(C).all()
        file=folder/f'{pool}_{month}.npz';np.savez_compressed(file,X=X,C=C,combos=h,bag=hh,weights=weights,robust=robust)
        record=dict(pool=pool,month=str(month),signal_date=str(s.date()),label_month_end=str(e.date()),symbols=names,
            confidence=concentration,anchor=concentration>=.65,stable_best_names=sorted(set.intersection(*best)),file=str(file),sha256=base.sha(file),
            candidate_triples=[dict(holdings=[names[j] for j in triple],image_weight=float(w),robust_loss=float(robust[ci])) for triple,w,ci in zip(hh,weights,order)],verified_author_trades=False)
        records.append(record);dump(target.name,records)
        if len(records)%12==0:print('EXPANDED_BAGS',pool,len(records),'names',len(names),flush=True)
    return records


def run():
    dump('EXPANDED_CONFIG.json',dict(reason='Author monthly description does not explicitly require top-half R12-2. Existing q0 attribution has81/363 inferred name-months below this gate in the theme pool.',
        qualification='Historical pool membership, positive R12-2,253 completed observations; no percentile cutoff.',
        unchanged='Same17ETFs, core50%,15secondaryfeaturesL1<=.5, three equal fixed-share positions, identical image objective.',
        mode='stock_only',pools=['core17','theme17'],eras=['through2021','all_retrospective'],no_independent_oos=True,
        label_construction='Re-enumerate all complete-path positive-momentum triples for image attribution. Actual entry selection never checks future path completeness.',
        why_not_proof='An image-consistent hindsight path is not confirmed author holdings; changing the gate tests a hypothesis only.'))
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();old=json.loads((OLD/'POLISH_MODELS.json').read_text())
    results={};models={};orders={};curves={};matches={};(P/'expanded_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        meta=bags(px,fs,pools[pool],pool,cal,obs);print('EXPANDED_CACHE',pool,flush=True);cache=prepare(px,meta,cal,obs)
        for era in ['through2021','all_retrospective']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            tag=f'{pool}__{era}__positive_only';warm=np.r_[[old[f'{pool}__{era}__mixed']['coefficients'][k] for k in FEATURES],0.,0.]
            print('FIT_START',tag,flush=True);best,trials=optimize(d,warm,'stock_only',202610093);loss,coef=best
            np.savez_compressed(P/'expanded_trials'/f'{tag}.npz',**trials)
            models[tag]=dict(coefficients=dict(zip(FEATURES+EXTRA,map(float,coef))),objective=loss,fit_months=int(use.sum()),latest_fit_label=meta[np.flatnonzero(use)[-1]]['label_month_end'],
                soft_evaluations=len(trials['soft_losses']),hard_refinement_evaluations=len(trials['hard_losses']),full_sample_explanatory_fit=era=='all_retrospective')
            for lag,cost in [(0,0),(0,10),(1,10)]:
                sim,sc,gate,decisions=policy_scores(px,fs,pools[pool],coef,'stock_only',lag,gate_quantile=0);nav,oo=sim.run(sc,gate,signal_lag=lag,cost_bps=cost)
                rebuilt=independent_nav(px,oo,nav.index,cost);delta=float(((rebuilt-nav)/nav).abs().max());assert delta<1e-11
                if lag==0:
                    _,ids=hard_objective(coef[None,:],cache);want=[]
                    for i,(m,cid) in enumerate(zip(meta,ids[0])):
                        z=np.load(m['file']);want.append([m['symbols'][j] for j in z['combos'][cid]]);assert decisions[i]['eligible_symbols']==m['symbols']
                    assert list(map(set,want))==[set(o['holdings']) for o in oo]
                    if cost==0:assert np.isclose(np.prod(cache['paths'][np.arange(len(meta)),ids[0],-1]),nav.iloc[-1],rtol=1e-12)
                key=tag+f'__lag{lag}_cost{cost}';mm,rows=match_orders(oo,meta);strict=[any(set(o['holdings'])==set(q['holdings']) for q in strict_subset(m['candidate_triples'])) for o,m in zip(oo,meta)]
                late=nav.loc['2021-12-31':].copy();late=late/late.iloc[0]
                results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(late),fit_era=era,objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,
                    strict_hits=sum(strict),strict_anchor_hits=sum(s and m['anchor'] for s,m in zip(strict,meta)),independent_accounting_error=delta,independent_oos=False)
                curves[key]=nav;orders[key]=oo;matches[key]=rows
                print('RESULT',key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),'strict',sum(strict),flush=True)
            dump('EXPANDED_MODELS.json',models);dump('EXPANDED_RESULTS.json',results);dump('EXPANDED_ORDERS.json',orders);dump('EXPANDED_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'EXPANDED_CURVES.csv.gz')
        del cache,d
    print('EXPANDED_DONE',flush=True)


if __name__=='__main__':run()
