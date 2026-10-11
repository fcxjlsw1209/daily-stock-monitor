from fit_policy import *


def solve(D):
    c=len(CORE);s=len(SIDE);m=D.shape[1]-c-s
    B=np.column_stack([D[:,:c],D[:,c:c+s],-D[:,c:c+s],D[:,c+s:],-D[:,c+s:]])
    n=B.shape[1]+1
    A=np.column_stack([-B,np.ones(len(B))]);b=np.zeros(len(B))
    cr=np.zeros(n);cr[:c]=-1;sr=np.zeros(n);sr[c:c+2*s]=1;mr=np.zeros(n);mr[c+2*s:-1]=1
    A=np.vstack([A,cr,sr,mr]);b=np.r_[b,-.6,.3,.1]
    eq=np.ones((1,n));eq[0,-1]=0;obj=np.zeros(n);obj[-1]=-1
    sol=linprog(obj,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1],bounds=[(0,None)]*(n-1)+[(-1,1)],method='highs')
    assert sol.success
    v=sol.x;w=np.r_[v[:c],v[c:c+s]-v[c+s:c+2*s],v[c+2*s:c+2*s+m]-v[c+2*s+m:-1]]
    assert np.einsum('ij,j->i',D,w,optimize=False).min()>=v[-1]-1e-7
    assert w[:c].sum()>=.6-1e-7 and abs(w[c:c+s]).sum()<=.3+1e-7 and abs(w[c+s:]).sum()<=.1+1e-7
    return w,float(v[-1])

def main():
    px,_,_,_,sources=load_inputs();rows=get_rows()
    states=market_states(px)
    specs={s:[s] for s in states};specs['all_four']=list(states)
    allres=[]
    for pool,rr in rows.items():
        segment=[r for r in rr if '2022-11-30'<=r['date']<='2023-12-29'];assert len(segment)==14 and all(r['valid'] for r in segment)
        future=[r for r in rr if '2023-12-29'<r['date']][:3]
        for family in FAMILIES:
            for label,state_names in specs.items():
                def design_state(r):
                    x=r['X'][family]
                    # Center state to avoid arbitrary constant core redefinition.
                    return np.column_stack([x]+[x*(2*states.at[pd.Timestamp(r['date']),s]-1) for s in state_names])
                ds=[]
                for r in segment:
                    x=design_state(r);y=r['y'];ds.append((x[y,None]-x[~y][None,:]).reshape(-1,x.shape[1]))
                w,margin=solve(np.vstack(ds));details=[]
                for r in segment+future:
                    x=design_state(r);sc=np.einsum('ij,j->i',x,w,optimize=False);chosen=[r['symbols'][i] for i in np.argsort(-sc,kind='stable')[:3]]
                    details.append(dict(date=r['date'],target=r['target'],diagnostic_selection=chosen,exact=set(chosen)==set(r['target']),shared=len(set(chosen)&set(r['target'])),proxy_valid=r['valid']))
                names=FEATURES+[s+'__'+k for s in state_names for k in FEATURES]
                allres.append(dict(pool=pool,family=family,state=label,margin=margin,local_exact=sum(d['exact'] for d in details[:14]),future_exact=sum(d['exact'] for d in details[14:]),weights=dict(zip(names,w.tolist())),details=details))
                print(pool,family,label,margin,'exact',allres[-1]['local_exact'],'next',allres[-1]['future_exact'],flush=True)
    dump('TRANSITION_FEASIBILITY.json',allres)
    assert all(sha(f)==h for f,h in sources.items())
if __name__=='__main__':main()
