from reconstruct import *
from rank_audit import cap_panel,ordered

def main():
    px,raw,members,periods,sources=load_inputs();f=feature_set(px,raw);ret=px.pct_change(fill_method=None)
    for n in [168,231,294]:
        m=px.shift(42)/px.shift(n)-1;v=ret.rolling(n-42).std().shift(42)
        f[f'mom{n//21}_2']=m
        f[f'full{(n-42)//21}_sharpe']=m/v.replace(0,np.nan)
        if n==294:f['full12_half_risk']=m/np.sqrt(v.replace(0,np.nan))
    cores=['mom12_2','mom14_2','mom11_2','mom8_2','full12_sharpe','full9_sharpe','full6_sharpe','full12_half_risk','aligned_risk12_2','arithmetic_sharpe12_2']
    caps,eligible,issuers=cap_panel(px,raw,sources)
    warm=px.notna().rolling(295,min_periods=295).sum().eq(295)&px.gt(0)
    trial=[];rows_by_config={};universe_summary=[]
    for pool,(member,etfs) in members.items():
        for boundary in ['top32','cap_band10pct_diagnostic']:
            for lag in [0,1]:
                events=[]
                for event in periods:
                    d=pd.Timestamp(event['start']);asof=px.index[px.index.get_loc(d)-lag]
                    ranked=ordered(asof,caps,eligible,issuers)
                    stocks=ranked[:32] if boundary=='top32' else [s for s in ranked if caps.at[asof,s]>=.9*caps.at[asof,ranked[31]]]
                    universe=sorted(set(stocks)|{s for s in etfs if pd.notna(px.at[asof,s])})
                    vals={k:f[k].loc[asof,universe] for k in cores+['vol63','industry_relative60','cmf21']}
                    # Actual month ends: the same known calendar anchor is used across every month.
                    if lag==0:
                        def at_month_offset(n):
                            month=d.to_period('M')-n;available=px.index[px.index.to_period('M')==month]
                            assert len(available)
                            return px.loc[available[-1],universe]
                        vals['calendar12_2']=at_month_offset(2)/at_month_offset(12)-1
                        vals['calendar14_2']=at_month_offset(2)/at_month_offset(14)-1
                    tab=pd.DataFrame(vals)
                    valid=[s for s in universe if warm.at[asof,s] and f['mom12_2'].at[asof,s]>0]
                    q=tab.rank(pct=True).fillna(.5)-.5
                    # Only names already eligible at the signal cutoff can be selected.
                    target={actual(s) for s in event['holdings']}
                    events.append(dict(date=event['start'],end=event['end'],asof=str(asof.date()),symbols=valid,target=target,q=q.loc[valid],universe=universe))
                universe_summary.append(dict(pool=pool,boundary=boundary,lag=lag,unavailable_target_periods=sum(not z['target'].issubset(z['symbols']) for z in events),mean_size=float(np.mean([len(z['universe']) for z in events]))))
                choices=list(events[0]['q'].columns[:len(cores)])+(['calendar12_2','calendar14_2'] if lag==0 else [])
                for core in choices:
                    for style in ['pure','space']:
                        for bonus in [0.,.03,.06]:
                            previous=set();details=[]
                            for ev in events:
                                q=ev['q'];base=q[core].to_numpy() if style=='pure' else (.7*q[core]+.15*q.vol63+.1*q.industry_relative60+.05*q.cmf21).to_numpy()
                                sc=base+np.array([bonus if s in previous else 0 for s in ev['symbols']])
                                chosen={ev['symbols'][i] for i in np.argsort(-sc,kind='stable')[:3]};previous=chosen
                                details.append(dict(date=ev['date'],end=ev['end'],asof=ev['asof'],target=sorted(ev['target']),diagnostic_selection=sorted(chosen),exact=chosen==ev['target'],shared=len(chosen&ev['target'])))
                            train=[z for z in details if z['end']<'2025-01-01'];test=[z for z in details if z['end']>='2025-01-01']
                            trial.append(dict(pool=pool,boundary=boundary,lag=lag,core=core,style=style,holding_bonus=bonus,
                                              exact_months=sum(z['exact'] for z in details),mean_shared=float(np.mean([z['shared'] for z in details])),
                                              development_exact=sum(z['exact'] for z in train),development_mean=float(np.mean([z['shared'] for z in train])),
                                              frozen_exact=sum(z['exact'] for z in test),frozen_mean=float(np.mean([z['shared'] for z in test])),details=details))
    dump('DEFINITION_TRIALS.json',trial);dump('DEFINITION_UNIVERSE_AUDIT.json',universe_summary);dump('DEFINITION_SOURCES.json',sources)
    for boundary in ['top32','cap_band10pct_diagnostic']:
        group=[t for t in trial if t['boundary']==boundary]
        dev=max(group,key=lambda t:(t['development_exact'],t['development_mean']))
        allbest=max(group,key=lambda t:(t['exact_months'],t['mean_shared']))
        dump('DEFINITION_'+boundary+'_SELECTED.json',dict(development_selected=dev,descriptive_best=allbest))
        print(boundary,'DEV', {k:v for k,v in dev.items() if k!='details'},flush=True)
        print(boundary,'ALL', {k:v for k,v in allbest.items() if k!='details'},flush=True)
    assert all(sha(f)==h for f,h in sources.items())
if __name__=='__main__':main()
