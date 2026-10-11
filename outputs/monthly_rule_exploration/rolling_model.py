from explore import *

def training_data(env,family):
    data=[]
    for ev in env['meta']:
        if ev['start'] not in env['rows']:continue
        row=env['rows'][ev['start']];names=row['symbols'];ret=env['px'].loc[ev['end'],names]/env['px'].loc[ev['start'],names]-1
        valid=ret.notna().to_numpy() & np.isfinite(ret.to_numpy());assert valid.all(),ev['start']
        y=ret.rank(pct=True).to_numpy()-.5;data.append(dict(start=ev['start'],end=ev['end'],X=row['X'][family],y=y))
    return data

def prediction(row,data,family):
    past=[d for d in data if d['end']<=row['date'] and d['start']<row['date']][-36:]
    assert len(past)>=24 and all(p['end']<=row['date'] for p in past)
    x=np.vstack([p['X'] for p in past]);y=np.concatenate([p['y'] for p in past]);xm=x.mean(axis=0);ym=y.mean();xc=x-xm;yc=y-ym
    gram=np.einsum('ni,nj->ij',xc,xc,optimize=False)+10*np.eye(x.shape[1]);rhs=np.einsum('ni,n->i',xc,yc,optimize=False)
    coef=np.linalg.solve(gram,rhs);pred=np.einsum('ij,j->i',row['X'][family]-xm,coef,optimize=False)+ym
    assert np.isfinite(pred).all()
    correction=.1*(pd.Series(pred).rank(pct=True).to_numpy()-.5);assert abs(correction).max()<=.05+1e-12
    return correction,dict(decision=row['date'],first_training_month=past[0]['start'],last_training_month_start=past[-1]['start'],last_training_label_end=past[-1]['end'],months=len(past),samples=len(y),alpha=10.,coef=coef.tolist())

def main():
    env=prepare();windows=[('2020-06-30','2020-12-31'),('2022-07-29','2022-11-30'),('2025-07-31','2025-11-28'),('2026-01-30','2026-05-29')]
    trials=[];cutoffs=[];cache={}
    for family in FAMILIES:
        data=training_data(env,family)
        for a,b in windows:
            for ev in [m for m in env['meta'] if a<=m['start']<b]:
                key=(family,ev['start'])
                if key not in cache:
                    cache[key],audit=prediction(env['rows'][ev['start']],data,family);cutoffs.append(dict(family=family,**audit))
    for core,family in product(['mom12_2','mom6_2','equal_6_12'],FAMILIES):
        results=[]
        for a,b in windows:
            es=[m for m in env['meta'] if a<=m['start']<b];targets=[]
            for ev in es:
                row=env['rows'][ev['start']];x=row['X'][family]
                c=x[:,0] if core=='mom12_2' else x[:,2] if core=='mom6_2' else .5*(x[:,0]+x[:,2])
                score=.5*c+.24*x[:,FEATURES.index('vol63')]-.16*x[:,FEATURES.index('volume_ratio21_126')]+cache[(family,ev['start'])]
                targets.append([row['symbols'][i] for i in np.argsort(-score,kind='stable')[:3]])
            nav,orders,check=curve_check(env,es,targets,a);monthly=[];capital=float(nav.iloc[0]);peak=float(env['base'].loc[:a].max())
            for ev,ts in zip(es,targets):
                nv,no,nc=curve_check(env,[ev],[ts],ev['start'],initial=capital,peak=peak);capital=float(nv.iloc[-1]);peak=max(peak,float(nv.max()))
                monthly.append(dict(order=no[0],check=nc))
            results.append(dict(start=a,end=b,orders=orders,check=check,monthly=monthly,
                all_months_not_rejected=all(m['check']['not_rejected'] for m in monthly)))
        t=dict(id=len(trials),core=core,family=family,windows=results,all_windows_not_rejected=all(r['check']['not_rejected'] and r['all_months_not_rejected'] for r in results))
        trials.append(t);print('ROLLING',core,family,[r['all_months_not_rejected'] for r in results],flush=True)
    dump('ROLLING_MODEL_TRIALS.json',trials);dump('ROLLING_MODEL_CUTOFFS.json',cutoffs)
    dump('ROLLING_MODEL_STATUS.json',dict(methods=len(trials),training_audits=len(cutoffs),all_window_passes=sum(t['all_windows_not_rejected'] for t in trials),
        windows=[dict(start=a,end=b,passes=sum(t['windows'][i]['all_months_not_rejected'] and t['windows'][i]['check']['not_rejected'] for t in trials)) for i,(a,b) in enumerate(windows)],
        original_account_changes=False,full_period_backtest_run=False,author_model_known=False))
    for f,h in read(P/'PROTECTED_SOURCES.json').items():assert sha(f)==h,f
if __name__=='__main__':main()
