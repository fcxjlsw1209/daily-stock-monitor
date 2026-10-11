"""Bounded factor-threshold correction, chronological choice explanations only."""
from itertools import product
import numpy as np
from scipy.optimize import linprog
from frontier import load_first_months
from decisions import P, read, dump, sha, CORE, SIDE, FEATURES, STRICT, display

STUMPS=list(product(range(len(FEATURES)),[-.25,0.,.25]))
NAMES=FEATURES+[f'{FEATURES[j]}>{t:+.2f}' for j,t in STUMPS]


def design(X):
    return np.column_stack([X]+[(X[:,j]>t).astype(float)-.5 for j,t in STUMPS])


def solve(rows,family,dominance):
    nc=len(CORE);ns=len(NAMES)-nc;n=nc+2*ns+1;all_d=[]
    for row in rows:
        x=design(row['X'][family]);yes,no=np.flatnonzero(row['y']),np.flatnonzero(~row['y'])
        all_d.append((x[yes,None]-x[no][None,:]).reshape(-1,len(NAMES)))
    D=np.vstack(all_d)
    A=np.column_stack([-D[:,:nc],-D[:,nc:],D[:,nc:],np.ones(len(D))]);b=np.zeros(len(D))
    extra=[];rhs=[]
    if dominance=='half':
        z=np.zeros(n);z[:nc]=-1;extra.append(z);rhs.append(-.5)
    else:
        for group in [[0,1,2,3],[4],[5,6,7,8,9]]:
            z=np.zeros(n);z[:nc]=-1
            for j in group:z[nc+j]=1;z[nc+ns+j]=1
            extra.append(z);rhs.append(0.)
    z=np.zeros(n);z[nc+len(SIDE):nc+ns]=1;z[nc+ns+len(SIDE):nc+2*ns]=1
    extra.append(z);rhs.append(.1)
    A=np.vstack([A,extra]);b=np.r_[b,rhs]
    eq=np.ones((1,n));eq[0,-1]=0;c=np.zeros(n);c[-1]=-1
    fit=linprog(c,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=[(0,None)]*(n-1)+[(-1,1)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert fit.success,fit.message
    v=fit.x;w=np.r_[v[:nc],v[nc:nc+ns]-v[nc+ns:nc+2*ns]]
    m=np.einsum('ij,j->i',D,w,optimize=False)
    assert np.isfinite(m).all() and m.min()>=v[-1]-1e-7
    dual=float(np.einsum('i,i->',b,fit.ineqlin.marginals,optimize=False)+fit.eqlin.marginals.sum()-fit.lower.marginals[-1]+fit.upper.marginals[-1])
    primal=float(np.einsum('i,i->',c,v,optimize=False));assert abs(primal-dual)<1e-7
    assert abs(w[len(FEATURES):]).sum()<=.1+1e-8
    return dict(weights=w.tolist(),feature_names=NAMES,margin=float(v[-1]),core_mass=float(w[:nc].sum()),model_l1=float(abs(w[len(FEATURES):]).sum()),primal_dual_gap=abs(primal-dual))


def choices(row,fit,family):
    x=design(row['X'][family]);w=np.array(fit['weights'])
    score=np.einsum('ij,j->i',x,w,optimize=False)
    correction=np.einsum('ij,j->i',x[:,len(FEATURES):],w[len(FEATURES):],optimize=False)
    assert np.isfinite(score).all() and abs(correction).max()<=.05+1e-8
    ids=np.argsort(-score,kind='stable')[:3]
    return [row['symbols'][i] for i in ids],score,correction


def main():
    rows=load_first_months(6);results={}
    for family,dominance in product(['universe_percentile','raw_magnitude','asset_class_percentile'],['half','largest_group']):
        name=f'{family}__{dominance}';history=[];prior=None;exact=0
        for i,row in enumerate(rows):
            prediction=None
            if prior:
                p,_,_=choices(row,prior,family)
                prediction=dict(picks=p,overlap=len(set(p)&set(np.array(row['symbols'])[row['y']])),trained_through=rows[i-1]['date'])
            fit=solve(rows[:i+1],family,dominance)
            matches=[]
            for earlier in rows[:i+1]:
                p,score,correction=choices(earlier,fit,family)
                actual=set(np.array(earlier['symbols'])[earlier['y']])
                matches.append(dict(date=earlier['date'],picks=p,exact=set(p)==actual,
                    all_candidates=[dict(symbol=s,selected=bool(y),score=float(sc),small_model=float(co)) for s,y,sc,co in zip(earlier['symbols'],earlier['y'],score,correction)]))
            history.append(dict(date=row['date'],prior_prediction=prediction,fit=fit,reconstructed_decisions=matches))
            if fit['margin']<=STRICT:break
            assert all(m['exact'] for m in matches)
            prior=fit;exact=i+1
        results[name]=dict(exact_prefix_months=exact,history=history)
        print(name,exact,fit['margin'],flush=True)
    dump('THRESHOLD_RESULTS.json',results)
    protected=read(P/'SOURCE_HASHES.json')
    for f,h in protected.items():assert sha(f)==h,f
    dump('THRESHOLD_STATUS.json',dict(maximum_common_prefix=max(r['exact_prefix_months'] for r in results.values()),full_period_backtest_run=False,sources_unchanged=len(protected),small_model_bound=.05))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})


if __name__=='__main__':main()
