"""Explicit monotone core-summary hypothesis, preserving array dimensions only."""
from joint_explore import *
CORE_NAMES=['mom12_2','mom9_2','mom6_2','mean_momentum_horizons','strongest_momentum_horizon','weakest_momentum_horizon']

def transformed(row):
    r=dict(row);xx={}
    for family,x in row['X'].items():
        z=x.copy();v=x[:,:3];z[:,3]=v.mean(axis=1);z[:,4]=v.max(axis=1);z[:,5]=v.min(axis=1)
        assert np.isfinite(z).all() and abs(z).max()<=.5+1e-12;xx[family]=z
    r['X']=xx;return r

def main():
    env=prepare();env['rows']={d:transformed(r) for d,r in env['rows'].items()};summary=[]
    for label,starts in [('2020_and_2025',['2020-06-30','2025-07-31']),('all_four',[a for a,b in JOINT_WINDOWS])]:
        rows=[];cs=[];groups=[]
        for start,end in JOINT_WINDOWS:
            if start not in starts:continue
            audit=read(P/f'JOINT_{start}_AMBIGUITY.json');events=[m for m in env['meta'] if start<=m['start']<end]
            ids=list(range(len(rows),len(rows)+len(events)));rows.extend(env['rows'][ev['start']] for ev in events);cs.extend(a['candidates'] for a in audit)
            groups.append(dict(start=start,end=end,events=events,ids=ids))
        trials=[];local=[]
        for kind,family in product(KINDS,FAMILIES):
            r=mixed_fit(rows,cs,family,kind,seconds=20);r['id']=len(trials)
            if r['has_solution']:
                r['fit']['feature_names']=CORE_NAMES+list(r['fit']['feature_names'][6:]);r['fit']['input_transform']='positive_monotone_momentum_horizon_summary_v1'
                if r['fit']['margin']>STRICT:
                    r['stability']=stability(rows,r['targets'],r['fit']);checks=[]
                    for g in groups:
                        ts=[r['targets'][i] for i in g['ids']];nav,orders,check=curve_check(env,g['events'],ts,g['start'])
                        checks.append(dict(start=g['start'],end=g['end'],orders=orders,check=check))
                    r['windows']=checks;r['all_local_pass']=r['stability']['local_robustness_pass'] and all(c['check']['not_rejected'] for c in checks)
                    if r['all_local_pass']:local.append(r)
            trials.append(r);dump(f'MONOTONE_{label}_TRIALS.json',trials)
            print('MONOTONE FIT',label,kind,family,r['status'],r.get('fit',{}).get('margin'),r.get('all_local_pass'),round(r['elapsed'],2),flush=True)
        status=dict(group=label,windows=starts,trials=len(trials),local_passes=len(local),timed_out=sum(r['status']==1 for r in trials))
        if local:
            best=min(local,key=lambda r:(KINDS.index(r['fit']['kind']),r['fit']['nonzero'],-r['fit']['margin'],r['id']))
            future=[]
            for g in groups:
                ts=[best['targets'][i] for i in g['ids']];nav,_,_=curve_check(env,g['events'],ts,g['start'])
                ne=next(m for m in env['meta'] if m['start']==g['end']);nt=choose(env['rows'][g['end']],best['fit'])
                _,no,nc=curve_check(env,[ne],[nt],g['end'],initial=float(nav.iloc[-1]),peak=max(float(env['base'].loc[:g['start']].max()),float(nav.max())))
                future.append(dict(order=no[0],check=nc,weights_refitted=False))
            dump(f'MONOTONE_{label}_SELECTED.json',dict(**best,next_months=future,selection_without_next_results=True))
            status.update(selected_id=best['id'],next_months_not_rejected=[f['check']['not_rejected'] for f in future])
        summary.append(status);dump('MONOTONE_STATUS.json',summary)
    for f,h in read(P/'PROTECTED_SOURCES.json').items():assert sha(f)==h,f
if __name__=='__main__':main()
