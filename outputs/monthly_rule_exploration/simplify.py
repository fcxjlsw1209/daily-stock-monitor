from explore import *
from scipy.optimize import linprog

def sparse_fit(rows,targets,family,cores):
    ids=list(cores)+[FEATURES.index('vol63'),FEATURES.index('volume_ratio21_126')];sign=np.ones(len(ids));sign[-1]=-1
    ds=[]
    for r,t in zip(rows,targets):
        x=r['X'][family];y=np.array([s in t for s in r['symbols']]);ds.append((x[y,None]-x[~y][None,:]).reshape(-1,len(FEATURES)))
    D=np.vstack(ds);d=D[:,ids]*sign;n=len(ids)
    a=np.r_[[-1.]*len(cores),[0.]*(n-len(cores)+1)]
    f=linprog(np.r_[np.zeros(n),-1.],A_ub=np.vstack([np.column_stack([-d,np.ones(len(d))]),a]),b_ub=np.r_[np.zeros(len(d)),-.5],
        A_eq=[np.r_[np.ones(n),0.]],b_eq=[1.],bounds=[(0,None)]*n+[(-1,1)],method='highs')
    assert f.success
    w=np.zeros(len(FEATURES));w[ids]=f.x[:-1]*sign
    margin=float(f.x[-1]);actual=np.einsum('ij,j->i',D,w,optimize=False);assert actual.min()>=margin-1e-7
    return dict(kind='linear',family=family,weights=w.tolist(),feature_names=FEATURES,core_mass=float(w[:len(CORE)].sum()),model_l1=0.,margin=margin,nonzero=int((abs(w)>1e-7).sum()))

def main():
    env=prepare();src=read(P/'JOINT_2020-06-30_SELECTED.json');events=[m for m in env['meta'] if src['start']<=m['start']<src['end']]
    rows=[env['rows'][e['start']] for e in events];targets=[o['holdings'] for o in src['orders']];trials=[];local=[]
    for nc in [1,2]:
        for ids,family in product(combinations(range(len(CORE)),nc),FAMILIES):
            fit=sparse_fit(rows,targets,family,ids);trial=dict(id=len(trials),fit=fit)
            if fit['margin']>STRICT:
                trial['stability']=stability(rows,targets,fit)
                if trial['stability']['local_robustness_pass']:local.append(trial)
            trials.append(trial)
    dump('SIMPLIFY_TRIALS.json',trials)
    status=dict(trials=len(trials),positive=sum(t['fit']['margin']>STRICT for t in trials),local_robust=len(local))
    if local:
        best=min(local,key=lambda r:(r['fit']['nonzero'],-r['fit']['margin'],r['id']));fit=best['fit']
        nav,orders,check=curve_check(env,events,targets,src['start']);assert check['not_rejected']
        capital=float(nav.iloc[-1]);peak=max(src['prior_peak'],float(nav.max()));later=[];parts=[nav]
        for ev in [m for m in env['meta'] if m['start']>=src['end']][:3]:
            nv,no,nc=curve_check(env,[ev],[choose(env['rows'][ev['start']],fit)],ev['start'],initial=capital,peak=peak)
            capital=float(nv.iloc[-1]);peak=max(peak,float(nv.max()));parts.append(nv);later.append(dict(order=no[0],check=nc))
        cross=[]
        for a,b in WINDOWS:
            if a==src['start']:continue
            es=[m for m in env['meta'] if a<=m['start']<b];ts=[choose(env['rows'][e['start']],fit) for e in es]
            _,co,cc=curve_check(env,es,ts,a);cross.append(dict(start=a,end=b,orders=co,check=cc))
        full=pd.concat(parts);full=full[~full.index.duplicated(keep='last')]
        dump('SIMPLIFY_SELECTED.json',dict(**best,start=src['start'],end=src['end'],orders=orders,check=check,extensions=later,cross_windows=cross,
            initial_nav=src['initial_nav'],prior_peak=src['prior_peak'],nav=[dict(date=str(d.date()),value=float(v)) for d,v in full.items()],
            selection_without_extension_outcomes=True,author_rule_verified=False))
        status.update(selected=best['id'],frozen_next_months_not_rejected=[r['check']['not_rejected'] for r in later],cross_pass=[r['check']['not_rejected'] for r in cross])
        print('SELECTED',fit,'EXT',status['frozen_next_months_not_rejected'],flush=True)
    dump('SIMPLIFY_STATUS.json',status)
if __name__=='__main__':main()
