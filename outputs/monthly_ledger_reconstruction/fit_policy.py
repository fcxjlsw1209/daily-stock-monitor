from reconstruct import *
from scipy.optimize import minimize
MODELS={'linear70':(.7,.3,0),'linear85':(.85,.15,0),'regime70':(.7,.2,.1),'shape70':(.7,.2,.1)}

def get_rows():
    rows={}
    ends={r['start']:r['end'] for r in read(O/'monthly_supplied_ledger_audit/PERIODS.json')}
    for z in read(P/'SNAPSHOTS.json'):
        y=np.array([s in z['target'] for s in z['symbols']])
        rows.setdefault(z['pool'],[]).append(dict(z, end=ends[z['date']], y=y, X={k:np.array(v) for k,v in z['X'].items()}))
    return rows

def design(row,family,model):
    X=row['X'][family]
    names=list(FEATURES)
    if model=='regime70':
        # Fixed, observed market-state interaction; core's main coefficients remain positive.
        weak=1-row['regime'];ids=[0,1,2,FEATURES.index('vol63'),FEATURES.index('industry_relative60'),FEATURES.index('cmf21')]
        z=X[:,ids]*weak;names+=['weak_x_'+FEATURES[j] for j in ids]
    elif model=='shape70':
        ids=[FEATURES.index(s) for s in ['vol63','industry_relative60','cmf21','volume_ratio21_126','near_high252','ma126_distance']]
        z=np.maximum(X[:,ids],0);names+=['positive_part_'+FEATURES[j] for j in ids]
    else:z=np.empty((len(X),0))
    return np.column_stack([X,z]),names

def pairs(rows,family,model):
    allD=[];weights=[]
    for r in rows:
        if not r['valid']:continue
        x,_=design(r,family,model);y=r['y'];D=(x[y,None]-x[~y][None,:]).reshape(-1,x.shape[1])
        allD.append(D);weights.extend([1/len(D)]*len(D))
    assert allD
    weights=np.array(weights)/len(allD)
    return np.vstack(allD),weights

def fit(rows,family,model):
    D,weights=pairs(rows,family,model);c=len(CORE);s=len(SIDE);m=D.shape[1]-len(FEATURES)
    core,side,small=MODELS[model]
    B=np.column_stack([D[:,:c],D[:,c:c+s],-D[:,c:c+s],D[:,c+s:],-D[:,c+s:]])
    def collapse(v):return np.r_[v[:c],v[c:c+s]-v[c+s:c+2*s],v[c+2*s:c+2*s+m]-v[c+2*s+m:]]
    def fun(v):
        w=collapse(v);diff=.03-np.einsum('ij,j->i',B,v,optimize=False);hinge=np.maximum(diff,0)
        loss=float(np.einsum('i,i,i->',weights,hinge,hinge,optimize=False)+.001*np.einsum('i,i->',w,w,optimize=False))
        g=-2*np.einsum('ij,i,i->j',B,weights,hinge,optimize=False)
        g+=.002*np.r_[w[:c],w[c:c+s],-w[c:c+s],w[c+s:],-w[c+s:]]
        return loss,g
    n=B.shape[1];v=np.zeros(n);v[:c]=core/c
    eq=np.zeros(n);eq[:c]=1
    ss=np.zeros(n);ss[c:c+2*s]=1
    zz=np.zeros(n);zz[c+2*s:]=1
    dot=lambda a,b:float(np.einsum('i,i->',a,b,optimize=False))
    cons=[dict(type='eq',fun=lambda v:dot(v,eq)-core,jac=lambda v:eq),dict(type='ineq',fun=lambda v:side-dot(v,ss),jac=lambda v:-ss)]
    if m:cons.append(dict(type='ineq',fun=lambda v:small-dot(v,zz),jac=lambda v:-zz))
    res=minimize(fun,v,jac=True,bounds=[(0,None)]*n,constraints=cons,method='SLSQP',options={'maxiter':1000,'ftol':1e-11})
    assert res.success,res.message
    w=collapse(res.x)
    assert abs(w[:c].sum()-core)<1e-7 and np.min(w[:c])>=-1e-8
    assert abs(w[c:c+s]).sum()<=side+1e-7 and abs(w[c+s:]).sum()<=small+1e-7
    names=design(rows[0],family,model)[1]
    return w,dict(weights=dict(zip(names,w.tolist())),loss=float(res.fun),iterations=int(res.nit),
                  first_label_start=min(r['date'] for r in rows if r['valid']),last_label_start=max(r['date'] for r in rows if r['valid']),
                  last_label_end=max(r['end'] for r in rows if r['valid']),valid_training_months=sum(r['valid'] for r in rows))

def evaluate(rows,w,family,model):
    details=[]
    for r in rows:
        x,_=design(r,family,model);sc=np.einsum('ij,j->i',x,w,optimize=False);idx=np.argsort(-sc,kind='stable')[:3]
        chosen=[r['symbols'][j] for j in idx]
        ranks={s:int(np.flatnonzero(np.argsort(-sc,kind='stable')==j)[0])+1 for j,s in enumerate(r['symbols']) if s in r['target']}
        details.append(dict(date=r['date'],end=r['end'],target=r['target'],diagnostic_selection=chosen,
                            target_ranks=ranks,shared=len(set(chosen)&set(r['target'])),exact=set(chosen)==set(r['target']),proxy_valid=r['valid']))
    return dict(months=len(rows),exact_months=sum(z['exact'] for z in details),mean_shared=float(np.mean([z['shared'] for z in details])),details=details)

def main():
    rows=get_rows();trials=[];annual=[]
    for pool,rr in rows.items():
        for family in FAMILIES:
            for model in MODELS:
                w,f=fit(rr,family,model);fit_all=evaluate(rr,w,family,model)
                train=[r for r in rr if r['end']<'2025-01-01'];test=[r for r in rr if r['end']>='2025-01-01']
                wt,ft=fit(train,family,model);assert ft['last_label_end']<=min(r['date'] for r in test)
                development=evaluate(train,wt,family,model);frozen=evaluate(test,wt,family,model)
                trials.append(dict(pool=pool,family=family,model=model,all_fit=f,all_agreement=fit_all,training_fit=ft,training_agreement=development,frozen_agreement=frozen))
                print('FIT',pool,family,model,fit_all['exact_months'],round(fit_all['mean_shared'],3),'FROZEN',frozen['exact_months'],round(frozen['mean_shared'],3),flush=True)
                year_results=[]
                for year in range(2022,2027):
                    test_year=[r for r in rr if pd.Timestamp(r['end']).year==year]
                    if not test_year:continue
                    cutoff=min(r['date'] for r in test_year);past=[r for r in rr if r['end']<=cutoff]
                    wy,fy=fit(past,family,model);assert fy['last_label_end']<=cutoff and fy['last_label_start']<cutoff
                    year_results.append(dict(year=year,fit=fy,evaluation=evaluate(test_year,wy,family,model)))
                details=[d for z in year_results for d in z['evaluation']['details']]
                annual.append(dict(pool=pool,family=family,model=model,months=len(details),exact_months=sum(d['exact'] for d in details),mean_shared=float(np.mean([d['shared'] for d in details])),years=year_results))
                dump('FITTED_TRIALS.json',trials);dump('ANNUAL_FROZEN_TRIALS.json',annual)
    chosen=max(trials,key=lambda t:(t['training_agreement']['exact_months'],t['training_agreement']['mean_shared']))
    dump('DEVELOPMENT_SELECTED.json',chosen)
    descriptive=max(trials,key=lambda t:(t['all_agreement']['exact_months'],t['all_agreement']['mean_shared']))
    dump('DESCRIPTIVE_SELECTED.json',descriptive)
if __name__=='__main__':main()
