"""Short-window inverse-choice exploration with bounded nonlinear side factors."""
from pathlib import Path
from itertools import combinations,product
from copy import deepcopy
import sys,json,argparse
import numpy as np
import pandas as pd
from scipy.optimize import linprog
P=Path(__file__).resolve().parent;O=P.parent
sys.path.insert(0,str(O/'monthly_confident_window'))
from infer_window import replay,image_check
from decisions import CORE,SIDE,FEATURES,STRICT
from frontier import load_first_months
from study import load,read,sha,xcoord,I
from inverse import score_paths,load_observations

WINDOWS=[('2022-07-29','2022-10-31'),('2025-07-31','2025-10-31'),('2026-01-30','2026-04-30'),('2020-06-30','2020-09-30')]
FAMILIES=['universe_percentile','raw_magnitude','asset_class_percentile']
KINDS=['linear','quadratic','threshold']

def dump(n,z): (P/n).write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def feature_names(kind):
    if kind=='linear':return list(FEATURES)
    if kind=='quadratic':return FEATURES+[f'squared_{k}' for k in SIDE]
    return FEATURES+[f'{k}>{t:+.2f}' for k,t in product(SIDE,[-.25,0.,.25])]

def design(X,kind):
    if kind=='linear':return X
    if kind=='quadratic':z=4*X[:,len(CORE):]**2-.5
    else:z=np.column_stack([(X[:,len(CORE)+i]>t).astype(float)-.5 for i,t in product(range(len(SIDE)),[-.25,0.,.25])])
    assert np.isfinite(z).all() and abs(z).max()<=.5+1e-10
    return np.column_stack([X,z])

def choose(row,fit):
    x=design(row['X'][fit['family']],fit['kind']);w=np.array(fit['weights'])
    score=np.einsum('ij,j->i',x,w,optimize=False)
    assert np.isfinite(score).all()
    if len(w)>len(FEATURES):
        correction=np.einsum('ij,j->i',x[:,len(FEATURES):],w[len(FEATURES):],optimize=False)
        assert abs(correction).max()<=.05+1e-7
    return [row['symbols'][i] for i in np.argsort(-score,kind='stable')[:3]]

def fit_rule(rows,targets,family,kind):
    nc=len(CORE);nf=len(feature_names(kind));ns=nf-nc;n=nc+2*ns+1;ds=[]
    for row,target in zip(rows,targets):
        X=design(row['X'][family],kind);y=np.array([s in target for s in row['symbols']]);assert y.sum()==3
        ds.append((X[y,None]-X[~y][None,:]).reshape(-1,nf))
    D=np.vstack(ds);A=np.column_stack([-D[:,:nc],-D[:,nc:],D[:,nc:],np.ones(len(D))]);b=np.zeros(len(D))
    z=np.zeros(n);z[:nc]=-1;A=np.vstack([A,z]);b=np.r_[b,-.5]
    if kind!='linear':
        z=np.zeros(n);z[nc+len(SIDE):nc+ns]=1;z[nc+ns+len(SIDE):nc+2*ns]=1
        A=np.vstack([A,z]);b=np.r_[b,.1]
    eq=np.ones((1,n));eq[0,-1]=0;c=np.zeros(n);c[-1]=-1
    f=linprog(c,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=[(0,None)]*(n-1)+[(-1,1)],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert f.success,f.message
    v=f.x;w=np.r_[v[:nc],v[nc:nc+ns]-v[nc+ns:nc+2*ns]]
    margin=float(v[-1]);actual=np.einsum('ij,j->i',D,w,optimize=False)
    assert np.isfinite(actual).all() and actual.min()>=margin-1e-7
    primal=float(np.einsum('i,i->',c,v,optimize=False))
    dual=float(np.einsum('i,i->',b,f.ineqlin.marginals,optimize=False)+f.eqlin.marginals.sum()-f.lower.marginals[-1]+f.upper.marginals[-1])
    assert abs(primal-dual)<1e-7
    assert w[:nc].min()>=-1e-8 and w[:nc].sum()>=.5-1e-8
    assert abs(w).sum()<=1+1e-7 and abs(w[len(FEATURES):]).sum()<=.1+1e-7
    out=dict(kind=kind,family=family,weights=w.tolist(),feature_names=feature_names(kind),margin=margin,
        nonzero=int((abs(w)>1e-7).sum()),core_mass=float(w[:nc].sum()),model_l1=float(abs(w[len(FEATURES):]).sum()),dual_gap=abs(primal-dual))
    if margin>STRICT:
        for row,target in zip(rows,targets):assert set(choose(row,out))==set(target)
    return out

def perturb_weights(w,v):
    # Preserve actual L1 masses in the core, linear sides, and nonlinear model.
    out=v.copy()
    for ids in [slice(0,len(CORE)),slice(len(CORE),len(FEATURES)),slice(len(FEATURES),None)]:
        a=abs(w[ids]).sum();b=abs(out[ids]).sum()
        if b>0:out[ids]*=a/b
    return out

def stability(rows,targets,fit):
    w=np.array(fit['weights']);kind=fit['kind'];family=fit['family'];X=[design(r['X'][family],kind) for r in rows]
    idx=[set(i for i,s in enumerate(r['symbols']) if s in t) for r,t in zip(rows,targets)]
    def exact(v):return all(set(np.argsort(-np.einsum('ij,j->i',x,v,optimize=False),kind='stable')[:3])==target for x,target in zip(X,idx))
    rounded=perturb_weights(w,np.round(w,3));round_pass=exact(rounded)
    rng=np.random.default_rng(20261009)
    count=sum(exact(perturb_weights(w,w*rng.uniform(.99,1.01,len(w)))) for _ in range(500))
    return dict(round_3_decimals_preserves=round_pass,weight_1pct_preserve_fraction=count/500,
        local_robustness_pass=bool(round_pass and count>=475))

def prepare():
    px,_,sources=load();rows={r['date']:r for r in load_first_months(121)}
    mf=O/'monthly_path_constraints/META_theme17.json';meta=read(mf)
    nf=O/'monthly_path_constraints/theme17__prior__draw_v2__blocks_NAV.csv';base=pd.read_csv(nf,index_col=0,parse_dates=True).iloc[:,0]
    cal=read(I/'CALIBRATION.json');obs=load_observations();cf=O/'monthly_reconstruction_contract/CONTRACT.json';cfg=read(cf)['image_audit']
    for f in [mf,nf,cf,O/'monthly_decision_reconstruction/SNAPSHOTS.json',O/'monthly_confident_window/WINDOW_SELECTION.json']:sources[str(f)]=sha(f)
    return dict(px=px,rows=rows,meta=meta,base=base,cal=cal,obs=obs,cfg=cfg,sources=sources)

def audit_window(env,start,end):
    key=start;events=[m for m in env['meta'] if start<=m['start']<end];assert len(events)==3
    audits=[];allowed=[]
    for ev in events:
        row=env['rows'][ev['start']];names=row['symbols'];p=env['px'].loc[ev['start']:ev['end'],names]
        assert p.notna().all().all() and p.gt(0).all().all()
        combo=np.array(list(combinations(range(len(names)),3)),dtype=int)
        paths=np.mean(p.div(p.iloc[0]).to_numpy()[:,combo],axis=2).T
        x=xcoord(p.index,env['cal']);ob={k:z[(z.x>x[0]+.35)&(z.x<x[-1]-.35)] for k,z in env['obs'].items()}
        level=float(env['base'].at[pd.Timestamp(ev['start'])]);peak=float(env['base'].loc[:ev['start']].max())
        anchor=dict(start_log_nav=np.log(level),prior_log_peak=np.log(peak),log_tolerance=0.)
        L=np.array([score_paths(paths,x,ob,env['cal'],xshift=shift,constraints=anchor)[0] for shift in [-.35,0.,.35]])
        assert np.isfinite(L).all();excess=L-L.min(axis=1)[:,None]
        near_all=np.all(excess<=1,axis=0);ids=np.flatnonzero(near_all)
        ids=sorted(ids,key=lambda j:(float(excess[:,j].max()),float(excess[:,j].sum()),int(j)))
        candidates=[dict(index=int(j),holdings=[names[k] for k in combo[j]],worst_excess=float(excess[:,j].max()),sum_excess=float(excess[:,j].sum())) for j in ids]
        audits.append(dict(start=ev['start'],end=ev['end'],eligible=len(names),all_triples=len(combo),near_all_count=len(ids),
            kept_count=min(len(ids),12),candidates=candidates))
        allowed.append(candidates[:12]);print('AUDIT',start,ev['start'],len(ids),'kept',min(len(ids),12),flush=True)
    dump(f'{key}_AMBIGUITY.json',audits)
    paths=list(product(*allowed));paths.sort(key=lambda z:(sum(c['worst_excess'] for c in z),sum(c['sum_excess'] for c in z)))
    shortlist=paths[:30]
    dump(f'{key}_SEARCH_SCOPE.json',dict(total_shortlisted_cartesian_paths=len(paths),evaluated_paths=len(shortlist),
        exhaustive_over_all_triples=False,ambiguity_alternatives_truncated=any(a['near_all_count']>12 for a in audits) or len(paths)>30))
    return events,[[c['holdings'] for c in z] for z in shortlist]

def curve_check(env,events,targets,start,initial=None,peak=None):
    if initial is None:initial=float(env['base'].at[pd.Timestamp(start)])
    if peak is None:peak=float(env['base'].loc[:start].max())
    nav,orders=replay(env['px'],events,targets,initial)
    check=image_check(nav,peak,env['px'],env['cal'],env['obs'],env['cfg'])
    return nav,orders,check

def window_run(env,start,end):
    events,paths=audit_window(env,start,end);rows=[env['rows'][m['start']] for m in events]
    checks=[];trials=[];local=[]
    for pi,targets in enumerate(paths):
        nav,orders,check=curve_check(env,events,targets,start)
        checks.append(dict(path_id=pi,orders=orders,check=check))
        for kind,family in product(KINDS,FAMILIES):
            fit=fit_rule(rows,targets,family,kind)
            trial=dict(id=len(trials),path_id=pi,fit=fit,positive=bool(fit['margin']>STRICT),image_not_rejected=check['not_rejected'])
            if trial['positive']:
                trial['stability']=stability(rows,targets,fit)
                if check['not_rejected'] and trial['stability']['local_robustness_pass']:local.append(trial)
            trials.append(trial)
        if pi%5==0:print('FIT',start,pi,'/',len(paths),'local pass',len(local),flush=True)
        dump(f'{start}_TRIALS.json',trials)
    dump(f'{start}_PATH_CHECKS.json',checks)
    best=None
    if local:
        best=min(local,key=lambda t:(KINDS.index(t['fit']['kind']),t['fit']['nonzero'],-t['fit']['margin'],t['id']))
        fit=best['fit'];case=checks[best['path_id']];targets=[o['holdings'] for o in case['orders']]
        nav,orders,check=curve_check(env,events,targets,start)
        ne=next(m for m in env['meta'] if m['start']==end);nr=env['rows'][end];nt=choose(nr,fit)
        peak=max(float(env['base'].loc[:start].max()),float(nav.max()))
        nv,no,nc=curve_check(env,[ne],[nt],end,initial=float(nav.iloc[-1]),peak=peak)
        cross=[]
        for a,b in WINDOWS:
            if a==start:continue
            es=[m for m in env['meta'] if a<=m['start']<b]
            ts=[choose(env['rows'][e['start']],fit) for e in es]
            cv,co,cc=curve_check(env,es,ts,a)
            cross.append(dict(start=a,end=b,orders=co,check=cc,weights_refitted=False))
        payload=dict(**best,start=start,end=end,orders=orders,check=check,
            next_month=dict(order=no[0],check=nc,weights_refitted=False),cross_windows=cross,
            nav=[dict(date=str(d.date()),value=float(v)) for d,v in nav.items()],
            next_nav=[dict(date=str(d.date()),value=float(v)) for d,v in nv.items()],
            initial_nav=float(nav.iloc[0]),prior_peak=float(env['base'].loc[:start].max()),
            retrospective=True,global_author_strategy_reproduced=False)
        dump(f'{start}_SELECTED.json',payload)
        print('SELECT',start,fit['kind'],fit['margin'],'NEXT',nc['not_rejected'],'CROSS',[r['check']['not_rejected'] for r in cross],flush=True)
    status=dict(start=start,end=end,paths=len(paths),trials=len(trials),positive=sum(t['positive'] for t in trials),
        local_robust_pass=len(local),selected_trial=best['id'] if best else None,
        next_month_not_rejected=payload['next_month']['check']['not_rejected'] if best else None)
    dump(f'{start}_STATUS.json',status);return status

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start');args=parser.parse_args()
    env=prepare();dump('SOURCE_HASHES.json',env['sources'])
    statuses=[]
    for start,end in WINDOWS:
        if args.start and args.start!=start:continue
        statuses.append(window_run(env,start,end));dump('RUN_STATUS.json',statuses)
    protected=read(O/'monthly_confident_window/PROTECTED_SOURCES.json');protected.update(env['sources'])
    for f,h in protected.items():assert sha(f)==h,f
    dump('PROTECTED_SOURCES.json',protected)
    dump('VALIDATION.json',dict(source_files_unchanged=len(protected),full_period_return_backtest_run=False,
        core_monotone_by_construction=True,nonlinear_model_max_per_security=.05,original_accounts_unchanged=True,
        actual_author_holdings_verified=False))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})

if __name__=='__main__':main()
