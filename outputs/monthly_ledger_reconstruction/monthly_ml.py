from reconstruct import *

def fit_ridge_asof(samples, cutoff, alpha):
    window=str((pd.Timestamp(cutoff)-pd.DateOffset(years=5)).date())
    train=[r for r in samples if r['end']<=cutoff and r['start']>=window]
    assert train and max(r['end'] for r in train)<=cutoff
    X=np.vstack([r['X'] for r in train]);y=np.concatenate([r['y'] for r in train])
    assert np.isfinite(X).all() and np.isfinite(y).all()
    mean=float(y.mean())
    gram=np.einsum('ni,nj->ij',X,X,optimize=False)+alpha*np.eye(X.shape[1])
    rhs=np.einsum('ni,n->i',X,y-mean,optimize=False)
    coef=np.linalg.solve(gram,rhs)
    audit=dict(train_start=min(r['start'] for r in train),train_end=max(r['end'] for r in train),months=len(train),examples=len(y),excluded_incomplete_labels=sum(r['excluded'] for r in train))
    return coef,mean,audit


def main():
    px,raw,members,periods,sources=load_inputs();f=feature_set(px,raw)
    f['mom14_2']=px.shift(42)/px.shift(294)-1;f['mom11_2']=px.shift(42)/px.shift(231)-1
    states=market_states(px)*2-1
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    dates=px.index.to_series().groupby(px.index.to_period('M')).last().tolist()
    # Last partial source month is not considered a completed monthly label.
    if pd.Timestamp(dates[-1]).month==px.index[-1].month:dates=dates[:-1]
    needed=FEATURES+['mom14_2','mom11_2'];trials=[];audits=[];predictions=[]
    for pool,(member,etfs) in members.items():
        def snapshot(d):
            universe=sorted(px.columns[member.loc[d]])
            valid=[s for s in universe if warm.at[d,s] and f['mom12_2'].at[d,s]>0]
            vals=pd.DataFrame({k:f[k].loc[d,universe] for k in needed})
            q=vals.rank(pct=True).fillna(.5)-.5
            base=q.loc[valid,FEATURES].to_numpy()
            x=np.column_stack([base]+[base*states.at[d,k] for k in states])
            return dict(date=str(d.date()),symbols=valid,q=q.loc[valid],X=x,missing=int(vals.loc[valid,FEATURES].isna().sum().sum()))
        samples=[]
        for start,end in zip(dates,dates[1:]):
            if start<pd.Timestamp('2014-01-01'):continue
            r=snapshot(start)
            if not r['symbols']:continue
            y=px.loc[end,r['symbols']]/px.loc[start,r['symbols']]-1-(px.at[end,'SPY']/px.at[start,'SPY']-1)
            keep=np.isfinite(y.to_numpy());samples.append(dict(start=str(start.date()),end=str(end.date()),X=r['X'][keep],y=np.clip(y.to_numpy()[keep],-1,1),excluded=int((~keep).sum())))
        events=[dict(snapshot(pd.Timestamp(e['start'])),end=e['end'],target=sorted(actual(s) for s in e['holdings'])) for e in periods]
        for alpha in [.1,1.,10.]:
            forecast={};years=sorted({pd.Timestamp(r['end']).year for r in events})
            for year in years:
                ev=[r for r in events if pd.Timestamp(r['end']).year==year]
                cutoff=min(r['date'] for r in ev)
                coef,mean,meta=fit_ridge_asof(samples,cutoff,alpha)
                assert meta['examples']>500 and meta['months']>=36
                audit=dict(pool=pool,alpha=alpha,holding_year=year,fit_cutoff=cutoff,**meta)
                audits.append(audit)
                for r in ev:
                    assert audit['train_end']<=r['date']
                    pred=np.einsum('ij,j->i',r['X'],coef,optimize=False)+mean
                    score=pd.Series(pred).rank(pct=True).to_numpy()-.5
                    assert abs(score).max()<=.5+1e-12
                    forecast[r['date']]=score
                    predictions.append(dict(pool=pool,alpha=alpha,date=r['date'],cutoff=cutoff,symbols=r['symbols'],predicted_excess_return=pred.tolist(),centered_prediction_rank=score.tolist()))
            for core in ['mom12_2','mom14_2','mom11_2']:
                for style in ['pure','space']:
                    details=[]
                    for r in events:
                        q=r['q'];base=q[core].to_numpy() if style=='pure' else (.7*q[core]+.15*q.vol63+.1*q.industry_relative60+.05*q.cmf21).to_numpy()
                        sc=.9*base+.1*forecast[r['date']]
                        chosen=sorted(r['symbols'][i] for i in np.argsort(-sc,kind='stable')[:3]);target=set(r['target'])
                        details.append(dict(date=r['date'],end=r['end'],target=r['target'],diagnostic_selection=chosen,shared=len(set(chosen)&target),exact=set(chosen)==target))
                    dev=[r for r in details if r['end']<'2025-01-01'];tail=[r for r in details if r['end']>='2025-01-01']
                    res=dict(pool=pool,alpha=alpha,core=core,style=style,small_ml_weight=.1,exact_months=sum(r['exact'] for r in details),mean_shared=float(np.mean([r['shared'] for r in details])),development_exact=sum(r['exact'] for r in dev),development_mean=float(np.mean([r['shared'] for r in dev])),frozen_exact=sum(r['exact'] for r in tail),details=details)
                    trials.append(res)
                    print('ML',pool,alpha,core,style,res['exact_months'],res['mean_shared'],flush=True)
    dump('MONTHLY_ML_TRIALS.json',trials);dump('MONTHLY_ML_TRAINING_AUDIT.json',audits);dump('MONTHLY_ML_PREDICTIONS.json',predictions)
    dump('MONTHLY_ML_SELECTION.json',dict(development_selected=max(trials,key=lambda r:(r['development_exact'],r['development_mean'])),descriptive_best=max(trials,key=lambda r:(r['exact_months'],r['mean_shared']))))
    assert all(sha(f)==h for f,h in sources.items())
if __name__=='__main__':main()
