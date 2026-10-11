from fit_policy import *

def main():
    rows=get_rows();leaders=read(P/'UNIFORM_WINDOW_LEADERS.json')
    targets=[leaders[0],next(r for r in leaders if r['start']=='2023-06-30')]
    results=[]
    for z in targets:
        segment=[r for r in rows[z['pool']] if z['start']<=r['date'] and r['end']<=z['end']]
        D=pair_matrix(segment,z['family']);active=list(range(len(FEATURES)));history=[]
        while len(active)>1:
            candidates=[]
            for j in active:
                rest=[k for k in active if k!=j]
                if not any(k<len(CORE) for k in rest):continue
                w,fit=max_margin(D,active=rest)
                history.append(dict(removed=FEATURES[j],active=[FEATURES[k] for k in rest],margin=fit['margin']))
                if fit['margin']>1e-5:candidates.append((fit['margin'],rest,w,fit))
            if not candidates:break
            _,active,w,fit=max(candidates,key=lambda t:t[0])
        w,fit=max_margin(D,active=active)
        rr=np.round(w,2)
        if rr[:len(CORE)].sum()<.6: rr[np.argmax(rr[:len(CORE)])]+=.6-rr[:len(CORE)].sum()
        full=evaluate(segment,w,z['family'],'linear70')
        rounded=evaluate(segment,rr,z['family'],'linear70')
        future=[r for r in rows[z['pool']] if r['date']>=z['end']][:3]
        details=[]
        for r in segment+future:
            x=r['X'][z['family']];score=np.einsum('ij,j->i',x,rr,optimize=False)
            order=np.argsort(-score,kind='stable')
            details.append(dict(date=r['date'],end=r['end'],target=r['target'],top6=[dict(symbol=r['symbols'][i],score=float(score[i]),contributions={FEATURES[j]:float(x[i,j]*rr[j]) for j in active}) for i in order[:6]],fit_period=r in segment))
        result=dict(start=z['start'],end=z['end'],pool=z['pool'],family=z['family'],greedy_not_global_minimal=True,
                    active=[FEATURES[j] for j in active],fit=fit,rounded_weights=dict(zip(FEATURES,rr.tolist())),
                    full=full,rounded=rounded,subsequent_frozen=evaluate(future,rr,z['family'],'linear70'),
                    elimination_history=history,decisions=details)
        results.append(result)
        print('SPARSE',z['start'],z['end'],result['active'],'rounded',rounded['exact_months'],'future',result['subsequent_frozen']['exact_months'],flush=True)
    dump('SPARSE_WINDOW_EXPLANATIONS.json',results)
if __name__=='__main__':main()
