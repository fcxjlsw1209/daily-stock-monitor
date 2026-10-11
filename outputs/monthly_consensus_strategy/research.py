"""Candidate combinations motivate rules; mature market returns train the model.

This is a new, implementable monthly hypothesis study, not an assertion that
inverse holdings were the author's trades. No inverse labels enter training.
"""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent;O=P.parent;OLD=O/'monthly_reconstruction_v2'
# Avoid importing this same file under its sibling's generic module name.
spec=importlib.util.spec_from_file_location('monthly_base',OLD/'research.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
END=base.END


def dump(name,obj):
    (P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def stats(nav,initial=1.):
    n=len(nav)-1;returns=nav.pct_change().dropna()
    wealth=float(nav.iloc[-1]/initial)
    # Initial acquisition fee is not normalized away.
    return dict(start=str(nav.index[0].date()),end=str(nav.index[-1].date()),wealth=wealth,
                cagr_252=float(wealth**(252/n)-1),max_dd=float((nav/np.maximum(initial,nav.cummax())-1).min()),
                sharpe=float(returns.mean()/returns.std()*np.sqrt(252)))


FEATURES=['m252_42','m126_42','risk','r63','r21','r5','relative','vol','downside',
          'near252','cmf21','volume_ratio','trend_gap','draw63','accel']


def ranks_and_rules(px,raw,member):
    rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()}
    # Optional secondary factors missing for an eligible name are neutral, not
    # silently used to exclude that name. Model also records missing indicators.
    for name in ['relative','cmf21','volume_ratio']:
        rank[name]=rank[name].fillna(.5).where(member)
    room=.5*((-raw['near252']).where(member).rank(axis=1,pct=True)+(-raw['trend_gap']).where(member).rank(axis=1,pct=True))
    momentum=.4*rank['m252_42']+.1*rank['m126_42']
    market_up=px.SPY.gt(px.SPY.rolling(200).mean())
    vol_state=rank['vol'].where(market_up,1-rank['vol'],axis=0)
    common=momentum+.15*rank['relative']+.10*rank['cmf21']
    rules={
        'factor_control':momentum+.20*rank['relative']+.15*rank['cmf21']+.15*rank['vol'],
        'strength_room':common+.15*room+.10*rank['vol'],
        'strength_room_regime':common+.15*room+.10*vol_state,
    }
    return rank,room,momentum,rules


def monthly_examples(px,raw,member,rank):
    idx=px.index
    ends=np.array([i for i in range(253,len(idx)-1) if idx[i].to_period('M')!=idx[i+1].to_period('M')])
    sig=ends-1
    frames=[rank[k].iloc[sig].to_numpy() for k in FEATURES]
    missing=np.stack([raw[k].iloc[sig].isna().to_numpy().astype(float) for k in ['relative','cmf21','volume_ratio']],axis=2)
    X=np.stack(frames,axis=2)
    regime=(px.SPY/px.SPY.rolling(200).mean()-1).clip(-.3,.3).iloc[sig].to_numpy()
    # Fixed interactions allow nonlinear ranking without unconstrained tree search.
    m=X[:,:,0]-.5;v=X[:,:,7]-.5;near=X[:,:,9]-.5;r21=X[:,:,4]-.5
    extras=np.stack([m*v,m*near,m*r21,v*regime[:,None],near*regime[:,None],r21**2],axis=2)
    X=np.concatenate([X,missing,extras],axis=2)
    valid=(member & px.notna().rolling(253,min_periods=253).sum().eq(253) & raw['m252_42'].gt(0)).iloc[sig].to_numpy() & np.isfinite(X).all(axis=2)
    eligible_strong=valid & (rank['m252_42'].iloc[sig].to_numpy()>=.5)
    # Each label corresponds exactly to month-end close entry and next month-end
    # close exit; features were captured the session before entry.
    y=np.full((len(ends),len(px.columns)),np.nan);maturity=np.full(len(ends),len(idx),dtype=int)
    for k,(i,j) in enumerate(zip(ends[:-1],ends[1:])):
        y[k]=(np.log(px.iloc[j]/px.iloc[i])-np.log(px.SPY.iloc[j]/px.SPY.iloc[i])).clip(-1,1).to_numpy()
        maturity[k]=j
    return dict(ends=ends,sig=sig,X=X,valid=valid,strong=eligible_strong,y=y,maturity=maturity)


def fit_models(px,raw,member,rank,training_scope='strong_half',reference_matmul=False):
    d=monthly_examples(px,raw,member,rank);idx=px.index
    pred=pd.DataFrame(np.nan,index=idx,columns=px.columns);audit=[]
    use=d['strong'] if training_scope=='strong_half' else d['valid']
    for row,signal in enumerate(d['sig']):
        entry=d['ends'][row]
        if not pd.Timestamp('2016-08-31')<=idx[entry]<=END:continue
        train=np.flatnonzero((d['maturity']<=signal)&(d['sig']<signal))[-60:]
        if len(train)<24:raise ValueError('Insufficient mature monthly training observations')
        ok=use[train]&np.isfinite(d['y'][train]);rr,cc=np.where(ok)
        X=d['X'][train][rr,cc];y=d['y'][train][rr,cc]
        counts=np.bincount(rr,minlength=len(train));weights=np.exp2(-(len(train)-1-rr)/36)/counts[rr];weights*=len(weights)/weights.sum()
        mu=np.average(X,axis=0,weights=weights);sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=weights));sd[sd<1e-8]=1
        z=(X-mu)/sd;ym=np.average(y,weights=weights)
        coef=np.linalg.solve(np.einsum('ni,nj,n->ij',z,z,weights)+100*np.eye(z.shape[1]),np.einsum('ni,n,n->i',z,weights,y-ym))
        test=d['valid'][row];block=(d['X'][row,test]-mu)/sd
        # Explicit reduction avoids spurious macOS NumPy/BLAS matmul FP flags.
        assert np.isfinite(block).all() and np.isfinite(coef).all()
        if reference_matmul:
            # Validation-only reference to the original reduction. Inputs and
            # outputs must still be finite; warning suppression is not a repair.
            with np.errstate(all='ignore'):
                pp=block@coef+ym
        else:
            pp=np.einsum('ni,i->n',block,coef)+ym
        assert np.isfinite(pp).all()
        pred.iloc[signal,np.flatnonzero(test)]=pp
        audit.append(dict(signal_date=str(idx[signal].date()),entry_date=str(idx[entry].date()),
                          latest_label_end=str(idx[d['maturity'][train].max()].date()),
                          training_scope=training_scope,months=len(train),samples=len(y),
                          missing_secondary_factors='neutral percentile + indicator',uses_inverse_labels=False,
                          coefficients=coef.tolist()))
    return pred,audit


def model_rule(rank,room,momentum,pred,member):
    modelrank=pred.where(member).rank(axis=1,pct=True)
    return momentum+.2*modelrank+.1*room+.1*rank['relative']+.05*rank['cmf21']+.05*rank['vol']


def run():
    cfg=dict(task='Build a usable monthly strategy from common conceptual clues, without needing exact author trades.',
             causal_training='Never train on inferred image holdings. Ridge targets only completed future-month returns from earlier dates.',
             signal_clock='Prior session close; month-end close execution; fixed adjusted-price units in between.',
             cost_roundtrip_bps=10,position_rules='3 equal;2 equal if only2 qualify; otherwise cash. No leverage or intramonth exits.',
             weights=dict(momentum12_2=.40,momentum6_2=.10,strength_room=dict(relative=.15,room=.15,cmf=.10,vol=.10),
                          model=dict(ridge=.20,room=.10,relative=.10,cmf=.05,vol=.05)),
             room='Mean percentile of negative proximity to252-day high and negative distance to200-day mean. A preference among strong stocks, not a mandatory pullback.',
             ridge=dict(window_months=60,half_life_months=36,alpha=100,target='log next-month return minusSPY',scopes=['all_positive','strong_half']),
             eligibility=['positive R12-2, top half in R12-2,253 historical observations','same plus positive last63dayreturn'],
             exit_variants=['Monthly reselect3','Keep existing positions while within top5 of eligible score; fill remaining slots by score'],
             pools=['core17','theme17'],planned_trials=42,
             selection='Highest2016-2021 net Sharpe among new rules, tie break CAGR. Later period2022-2026 reported separately. Previously researched dates, not independent OOS.',
             source_clues='Previous multi-path feature associations suggest intermediate momentum and less extension. These only motivate feature definitions; no inferred holdings enter model training.',
             market_cap_limit='Existing historicalindex-union Top32proxy; not fully certified PITall-US pool or verified authorETFlist.',
             original_accounts_unchanged=True)
    dump('CONFIG.json',cfg)
    px,fs,pools=base.load();raw=base.raw_features(px,fs);results={};curves={};orders={};audits={}
    dump('PROTECTED_BEFORE.json',json.loads((O/'monthly_curve_reverse/PROTECTED_BEFORE.json').read_text()))
    def save(key,sim,score,gate,buffer,is_baseline=False):
        nav,oo=sim.run(score,gate,buffer=buffer,cost_bps=10,signal_lag=1)
        development=nav.loc[:'2021-12-31'];later=nav.loc['2021-12-31':]
        yearend=nav.groupby(nav.index.year).last();previous=np.r_[1.,yearend.to_numpy()[:-1]]
        results[key]=dict(**stats(nav),development=stats(development),later_diagnostic=stats(later,initial=float(later.iloc[0])),
                          calendar_returns={str(y):float(v/p-1) for (y,v),p in zip(yearend.items(),previous)},
                          average_one_way_turnover_per_month=float(np.mean([r['turnover']/2 for r in oo])),
                          cash_months=sum(not r['holdings'] for r in oo),baseline=is_baseline,
                          entry_cash_weight=0 if all(r['holdings'] for r in oo) else None)
        orders[key]=oo;curves[key]=nav;dump('RESULTS.json',results)
        print(key,round(results[key]['cagr_252']*100,2),round(results[key]['max_dd']*100,2),'DEV',round(results[key]['development']['sharpe'],3),flush=True)
    for pool in cfg['pools']:
        member=pools[pool];rank,room,mom,rules=ranks_and_rules(px,raw,member);sim=base.Simulator(px,fs,member)
        for scope in cfg['ridge']['scopes']:
            pred,audit=fit_models(px,raw,member,rank,scope);audits[f'{pool}_{scope}']=audit
            dump('TRAINING_AUDIT.json',audits);pred.to_csv(P/f'predictions_{pool}_{scope}.csv.gz')
            rules['ridge_'+scope]=model_rule(rank,room,mom,pred,member)
        save(pool+'__R12_2_baseline',sim,rank['m252_42'],raw['m252_42']>0,0,True)
        strong=(raw['m252_42']>0)&(rank['m252_42']>=.5)
        for name,score in rules.items():
            for gn,gate in [('strong',strong),('strong_recovering',strong&(raw['r63']>0))]:
                for buffer in [0,5]:save(f'{pool}__{name}__{gn}__keep{buffer}',sim,score,gate,buffer)
        dump('ORDERS.json',orders);pd.DataFrame(curves).to_csv(P/'CURVES.csv.gz')
    assert len(results)==cfg['planned_trials']
    candidates={k:v for k,v in results.items() if not v['baseline']}
    selected=max(candidates,key=lambda k:(candidates[k]['development']['sharpe'],candidates[k]['development']['cagr_252']))
    best=max(candidates,key=lambda k:candidates[k]['cagr_252'])
    dump('SELECTION.json',dict(development_selected=selected,best_full_cagr_retrospective=best,all_trials=len(results),
                              author_strategy_reproduced=False,independent_oos=False,not_added_to_paper_accounts=True))
    print('SELECTED',selected,'BEST_FULL',best,flush=True)


if __name__=='__main__':
    run()
