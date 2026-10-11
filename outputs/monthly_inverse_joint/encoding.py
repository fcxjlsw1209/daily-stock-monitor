"""Check whether percentile compression loses economically large factor gaps.

Compare the SAME factors with robust cross-sectional scaling. The author has not
specified rank vs raw feature scaling. This does not add any date/ticker features.
"""
from engine import *
from fit import match_orders
from validate import independent_nav,strict_subset


def normalized(value,member):
    z=value.where(member);median=z.median(axis=1);deviation=z.sub(median,axis=0)
    scale=1.4826*deviation.abs().median(axis=1);scale=scale.where(scale>1e-12,z.std(axis=1)).where(lambda s:s>1e-12,1.)
    return (deviation.div(scale,axis=0).clip(-3,3)/6).fillna(0).where(member)


def encoded_features(px,fs,member):
    raw,_,_,gate=features(px,fs,member)
    norm={k:normalized(raw[k],member) for k in set(FEATURES[:-2])|{'m252_42','m126_42','risk'}}
    core={'raw':.8*(norm['m252_42']+.5)+.2*(norm['m126_42']+.5),
          'risk':.8*(norm['risk']+.5)+.2*(norm['m252_42']+.5)}
    core['mixed']=.5*core['raw']+.5*core['risk'];f={k:norm[k] for k in FEATURES[:-2]}
    regime=(px.SPY/px.SPY.rolling(200).mean()-1).clip(-.2,.2)/.2
    f['vol_market']=f['vol'].mul(regime,axis=0);f['extension_market']=f['near252'].mul(regime,axis=0)
    return core,f,gate


def run():
    dump('ENCODING_CONFIG.json',dict(reason='Rank percentiles compress the distance between exceptional momentum and ordinary strength; author did not disclose factor scaling.',
        transform='For each date and eligible universe: subtract median, divide by1.4826*MAD, fallback to standard deviation when MAD=0,clip[-3,3],divide6. Missing secondary features neutral.',
        core='Same50% mixed core; same15secondaryfeatures andL1<=.5. No new ticker,calendar,or future-return input.',
        qualification='Keep previous positiveR12-2 plus top-half gate, isolating scaling from eligibility.',
        parameters='Identical constant-coefficient joint optimizer and image objective; same unleveredETF hypotheses.',no_independent_oos=True))
    px,fs,pools=base.load();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());obs=load_observations();old=json.loads((OLD/'POLISH_MODELS.json').read_text())
    results={};models={};orders={};curves={};matches={};folder=P/'encoding_bags';folder.mkdir(exist_ok=True);(P/'encoding_trials').mkdir(exist_ok=True)
    for pool in ['core17','theme17']:
        original=json.loads((OLD/f'BAGS_{pool}.json').read_text());core,f,gate=encoded_features(px,fs,pools[pool]);meta=[]
        for m in original:
            n=dict(m);signal=m['signal_date'];names=m['symbols'];z=dict(np.load(m['file']))
            z['X']=np.array([[f[k].at[signal,t] for k in FEATURES] for t in names]);z['C']=np.array([[core[k].at[signal,t] for k in ['raw','risk','mixed']] for t in names])
            assert np.isfinite(z['X']).all() and np.isfinite(z['C']).all();file=folder/f"{pool}_{m['month']}.npz";np.savez_compressed(file,**z);n.update(file=str(file),sha256=base.sha(file));meta.append(n)
        dump(f'ENCODING_BAGS_{pool}.json',meta);print('ENCODING_CACHE',pool,flush=True);cache=prepare(px,meta,cal,obs)
        for era in ['through2021','all_retrospective']:
            use=np.array([era=='all_retrospective' or m['month']<'2022-01' for m in meta]);d={k:v[use] for k,v in cache.items()}
            tag=f'{pool}__{era}__robust_scale';warm=np.r_[[old[f'{pool}__{era}__mixed']['coefficients'][k] for k in FEATURES],0.,0.]
            print('FIT_START',tag,flush=True);best,trials=optimize(d,warm,'stock_only',202610094);loss,coef=best;np.savez_compressed(P/'encoding_trials'/f'{tag}.npz',**trials)
            models[tag]=dict(coefficients=dict(zip(FEATURES+EXTRA,map(float,coef))),objective=loss,fit_months=int(use.sum()),latest_fit_label=meta[np.flatnonzero(use)[-1]]['label_month_end'],
                soft_evaluations=len(trials['soft_losses']),hard_refinement_evaluations=len(trials['hard_losses']),full_sample_explanatory_fit=era=='all_retrospective')
            score=.5*core['mixed']+sum(float(coef[i])*f[k] for i,k in enumerate(FEATURES));sim=base.Simulator(px,fs,pools[pool])
            for lag,cost in [(0,0),(0,10),(1,10)]:
                nav,oo=sim.run(score,gate,signal_lag=lag,cost_bps=cost);rebuilt=independent_nav(px,oo,nav.index,cost);delta=float(((rebuilt-nav)/nav).abs().max());assert delta<1e-11
                if lag==0:
                    _,ids=hard_objective(coef[None,:],cache);want=[]
                    for m,cid in zip(meta,ids[0]):
                        z=np.load(m['file']);want.append([m['symbols'][j] for j in z['combos'][cid]])
                    assert list(map(set,want))==[set(o['holdings']) for o in oo]
                    if cost==0:assert np.isclose(np.prod(cache['paths'][np.arange(len(meta)),ids[0],-1]),nav.iloc[-1],rtol=1e-12)
                key=tag+f'__lag{lag}_cost{cost}';mm,rows=match_orders(oo,meta);strict=[any(set(o['holdings'])==set(q['holdings']) for q in strict_subset(m['candidate_triples'])) for o,m in zip(oo,meta)]
                late=nav.loc['2021-12-31':].copy();late=late/late.iloc[0]
                results[key]=dict(**summary(nav),early=summary(nav.loc[:'2021-12-31']),later_diagnostic=summary(late),fit_era=era,objective=loss,signal_lag=lag,cost_bps_roundtrip=cost,matching=mm,
                    strict_hits=sum(strict),strict_anchor_hits=sum(s and m['anchor'] for s,m in zip(strict,meta)),independent_accounting_error=delta,independent_oos=False)
                curves[key]=nav;orders[key]=oo;matches[key]=rows
                print('RESULT',key,'CAGR',round(results[key]['cagr_252']*100,2),'Sharpe',round(results[key]['sharpe'],3),'DD',round(results[key]['max_dd']*100,2),'loss',round(loss,3),'strict',sum(strict),flush=True)
            dump('ENCODING_MODELS.json',models);dump('ENCODING_RESULTS.json',results);dump('ENCODING_ORDERS.json',orders);dump('ENCODING_MATCHES.json',matches);pd.DataFrame(curves).to_csv(P/'ENCODING_CURVES.csv.gz')
    print('ENCODING_DONE',flush=True)


if __name__=='__main__':run()
