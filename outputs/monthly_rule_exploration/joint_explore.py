"""Joint discrete portfolio hypotheses and shared score constraints; retrospective research."""
import sys, json, argparse, time
from itertools import combinations,product
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from explore import *

JOINT_WINDOWS=[('2022-07-29','2022-11-30'),('2020-06-30','2020-09-30'),('2025-07-31','2025-10-31'),('2026-01-30','2026-04-30')]

def candidates_for(env,start,end):
    evs=[m for m in env['meta'] if start<=m['start']<end];choices=[];audit=[]
    for ev in evs:
        row=env['rows'][ev['start']];names=row['symbols'];p=env['px'].loc[ev['start']:ev['end'],names]
        assert p.notna().all().all() and p.gt(0).all().all()
        combos=np.array(list(combinations(range(len(names)),3)),dtype=int)
        paths=np.mean(p.div(p.iloc[0]).to_numpy()[:,combos],axis=2).T
        x=xcoord(p.index,env['cal']);obs={k:z[(z.x>x[0]+.35)&(z.x<x[-1]-.35)] for k,z in env['obs'].items()}
        anchor=dict(start_log_nav=np.log(env['base'].at[pd.Timestamp(ev['start'])]),prior_log_peak=np.log(env['base'].loc[:ev['start']].max()),log_tolerance=0.)
        loss=np.array([score_paths(paths,x,obs,env['cal'],xshift=dx,constraints=anchor)[0] for dx in [-.35,0.,.35]])
        assert np.isfinite(loss).all()
        excess=loss-loss.min(axis=1)[:,None];all_mask=np.all(excess<=1,axis=0);any_mask=np.any(excess<=1,axis=0)
        mask=all_mask if all_mask.any() else any_mask;ids=np.flatnonzero(mask)
        cs=[[names[i] for i in combos[j]] for j in ids];choices.append(cs)
        audit.append(dict(start=ev['start'],end=ev['end'],eligible=len(names),triple_count=len(combos),intersection_count=int(all_mask.sum()),union_count=int(any_mask.sum()),
            basis='intersection' if all_mask.any() else 'union_only_empty_intersection',retained_count=len(cs),candidates=cs))
        print('JOINT CANDIDATES',start,ev['start'],audit[-1]['basis'],len(cs),flush=True)
    return evs,choices,audit

def mixed_fit(rows,candidates,family,kind,excluded=(),seconds=20):
    nc=len(CORE);nf=len(feature_names(kind));ns=nf-nc;nw=nc+2*ns;mi=nw;ti=nw+1
    offsets=[];offset=ti+len(rows)
    for cs in candidates:offsets.append(offset);offset+=len(cs)
    n=offset;lo=np.zeros(n);hi=np.ones(n);lo[mi]=-1;lo[ti:ti+len(rows)]=-1
    A=[];lb=[];ub=[]
    def add(a,l,h):A.append(a);lb.append(l);ub.append(h)
    a=np.zeros(n);a[:nw]=1;add(a,1,1)
    a=np.zeros(n);a[:nc]=1;add(a,.5,np.inf)
    if kind!='linear':
        a=np.zeros(n);a[nc+len(SIDE):nc+ns]=1;a[nc+ns+len(SIDE):nw]=1;add(a,-np.inf,.1)
    for m,(row,cs) in enumerate(zip(rows,candidates)):
        X=design(row['X'][family],kind);off=offsets[m]
        a=np.zeros(n);a[off:off+len(cs)]=1;add(a,1,1)
        for i,s in enumerate(row['symbols']):
            a=np.zeros(n);a[:nc]=X[i,:nc];a[nc:nc+ns]=X[i,nc:];a[nc+ns:nw]=-X[i,nc:]
            a[ti+m]=-1;a[off:off+len(cs)]=-2*np.array([s in c for c in cs]);a[mi]=-.5
            add(a,-2,np.inf)
            b=a.copy();b[mi]=.5;add(b,-np.inf,0)
    for path in excluded:
        a=np.zeros(n)
        for m,j in enumerate(path):a[offsets[m]+j]=1
        add(a,-np.inf,len(rows)-1)
    integer=np.zeros(n);integer[ti+len(rows):]=1;obj=np.zeros(n);obj[mi]=-1
    then=time.monotonic()
    f=milp(obj,integrality=integer,bounds=Bounds(lo,hi),constraints=LinearConstraint(np.array(A),lb,ub),
        options=dict(time_limit=seconds,mip_rel_gap=0.0,presolve=True))
    result=dict(status=int(f.status),message=f.message,elapsed=time.monotonic()-then,
        has_solution=f.x is not None,proven_optimal=bool(f.status==0),family=family,kind=kind,
        variables=n,binaries=int(integer.sum()),constraints=len(A))
    for name in ['mip_gap','mip_node_count','mip_dual_bound']:
        v=getattr(f,name,None)
        if v is not None and np.isfinite(v):result[name]=float(v)
    if f.x is None:return result
    v=f.x;activity=np.einsum('ij,j->i',np.array(A),v,optimize=False)
    assert np.min(activity-np.array(lb))>=-2e-6 and np.min(np.array(ub)-activity)>=-2e-6
    assert abs(v[integer.astype(bool)]-np.round(v[integer.astype(bool)])).max()<1e-6
    path=[int(np.argmax(v[off:off+len(cs)])) for off,cs in zip(offsets,candidates)]
    targets=[cs[j] for cs,j in zip(candidates,path)]
    result.update(path=path,targets=targets,mixed_margin=float(v[mi]))
    # Recompute the chosen-label optimum independently and verify strict rankings.
    fit=fit_rule(rows,targets,family,kind)
    assert fit['margin']>=v[mi]-2e-6
    if f.status==0:assert abs(fit['margin']-v[mi])<2e-6
    result['fit']=fit
    return result

def run(env,start,end):
    events,cs,audit=candidates_for(env,start,end);rows=[env['rows'][m['start']] for m in events]
    dump(f'JOINT_{start}_AMBIGUITY.json',audit)
    trials=[];local=[]
    for kind,family in product(KINDS,FAMILIES):
        excluded=[]
        for attempt in range(3):
            r=mixed_fit(rows,cs,family,kind,excluded)
            r.update(id=len(trials),attempt=attempt,start=start,end=end)
            if r['has_solution']:
                excluded.append(r['path'])
                if r['fit']['margin']>STRICT:
                    nav,orders,check=curve_check(env,events,r['targets'],start)
                    r.update(orders=orders,check=check,stability=stability(rows,r['targets'],r['fit']))
                    if check['not_rejected'] and r['stability']['local_robustness_pass']:local.append(r)
            trials.append(r);dump(f'JOINT_{start}_TRIALS.json',trials)
            print('JOINT FIT',start,kind,family,attempt,r['status'],r.get('fit',{}).get('margin'),r.get('check',{}).get('not_rejected'),'seconds',round(r['elapsed'],2),flush=True)
            if not r['has_solution'] or (r['proven_optimal'] and r['fit']['margin']<=STRICT):break
    status=dict(start=start,end=end,trial_count=len(trials),local_robust_count=len(local),selected_id=None,
        global_author_rule_reproduced=False,candidate_paths_total=int(np.prod([len(c) for c in cs])),timed_out=sum(r['status']==1 for r in trials))
    if local:
        best=min(local,key=lambda r:(KINDS.index(r['fit']['kind']),r['fit']['nonzero'],-r['fit']['margin'],r['id']))
        fit=best['fit'];nav,orders,check=curve_check(env,events,best['targets'],start)
        ne=next(m for m in env['meta'] if m['start']==end)
        nv,no,nc=curve_check(env,[ne],[choose(env['rows'][end],fit)],end,initial=float(nav.iloc[-1]),peak=max(float(env['base'].loc[:start].max()),float(nav.max())))
        cross=[]
        for a,b in JOINT_WINDOWS:
            if a==start:continue
            es=[m for m in env['meta'] if a<=m['start']<b];ts=[choose(env['rows'][e['start']],fit) for e in es]
            cv,co,cc=curve_check(env,es,ts,a)
            cross.append(dict(start=a,end=b,orders=co,check=cc,weights_refitted=False))
        payload=dict(**best,next_month=dict(order=no[0],check=nc,weights_refitted=False),cross_windows=cross,
            nav=[dict(date=str(d.date()),value=float(v)) for d,v in nav.items()],next_nav=[dict(date=str(d.date()),value=float(v)) for d,v in nv.items()],
            selection_without_adjacent_outcome=True,initial_nav=float(nav.iloc[0]),prior_peak=float(env['base'].loc[:start].max()))
        dump(f'JOINT_{start}_SELECTED.json',payload);status.update(selected_id=best['id'],next_month_not_rejected=nc['not_rejected'])
        print('JOINT SELECT',start,fit['kind'],'NEXT',nc['not_rejected'],'CROSS',[c['check']['not_rejected'] for c in cross],flush=True)
    dump(f'JOINT_{start}_STATUS.json',status);return status

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start');args=ap.parse_args()
    env=prepare();statuses=[]
    for a,b in JOINT_WINDOWS:
        if args.start and a!=args.start:continue
        statuses.append(run(env,a,b));dump('JOINT_RUN_STATUS.json',statuses)
    source=read(P/'PROTECTED_SOURCES.json')
    for f,h in source.items():assert sha(f)==h,f
    dump('JOINT_VALIDATION.json',dict(protected_unchanged=len(source),full_period_backtest_run=False,all_labels_retrospective=True))
if __name__=='__main__':main()
