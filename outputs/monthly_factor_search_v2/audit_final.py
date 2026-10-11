"""Causality, repeatability, cost and parameter-neighborhood diagnostics."""
from validate import *

def main():
    meta=read(P/'INPUTS.json');px,raw,members,periods,sources=load_inputs();co,si=features(px,raw);fs={**co,**si};opening=load_open(px,raw)
    checks=[]
    for date in ['2020-12-31','2022-06-30','2023-12-29','2024-12-31','2026-08-31']:
        cut=pd.Timestamp(date);cc,ss=features(px.loc[:cut],{k:z.loc[:cut] for k,z in raw.items()});prefix={**cc,**ss}
        for name in fs:
            a=fs[name].loc[cut].to_numpy();b=prefix[name].iloc[-1].to_numpy()
            assert np.array_equal(np.isnan(a),np.isnan(b)),(date,name,'NA pattern')
            assert np.allclose(a,b,rtol=1e-8,atol=1e-8,equal_nan=True),(date,name)
            checks.append(dict(date=date,factor=name,passed=True))
        np.testing.assert_allclose(market_states(px).loc[cut],market_states(px.loc[:cut]).iloc[-1])
        # Verify the new image-free loader reproduces feature packs used for the search.
        for key,m in meta.items():
            i=m['dates'].index(date);symbols=m['symbols'][i];tab=pd.DataFrame({f:fs[f].loc[cut,symbols] for f in m['feature_names']})
            q=normalize(tab,m['norm']).to_numpy();z=np.load(m['file'])['X'][i,:len(symbols)]
            np.testing.assert_allclose(q,z,atol=1e-12,rtol=1e-12)
    dump('PREFIX_CHECKS.json',checks)
    leaders=read(P/'SELECTED_CASES.json');neighborhood=[];extra_checks=[]
    for spec in leaders[:2]:
        orders=orders_from_pack(spec)
        for cost in [.001,.0025]:
            res,eq,_,_=replay(px,opening,orders,cost,'open');extra_checks.append(dict(case=spec['name'],control='cost_stress_next_open',**res))
        # Earlier years use precisely the same frozen rule, not newly selected weights.
        m=meta[spec['key']];data=np.load(m['file']);extra=None
        if 'correction' in spec:
            state,side=spec['correction'].split('__',1)
            ex=data['X'][:,:,m['feature_names'].index(side)]*(market_states(px).loc[m['dates'],state].to_numpy()[:,None]*2-1)
            extra=ex
        scores=score_of(spec,data['X'],extra);old=[]
        for i,(d,e) in enumerate(zip(m['dates'],m['ends'])):
            if d>='2020-12-31':continue
            ids=np.flatnonzero(data['eligible'][i]);chosen=sorted(ids,key=lambda j:(-scores[i,j],j))[:3]
            if len(chosen)<2:chosen=[]
            assert all(data['valid'][i,j] for j in chosen)
            old.append(dict(date=d,end=e,signal_date=d,symbols=[m['symbols'][i][j] for j in chosen],scores=[float(scores[i,j]) for j in chosen]))
        for execution in ['close','open']:
            res,eq,_,_=replay(px,opening,old,.0005,execution,start=START,end=LABEL_START)
            extra_checks.append(dict(case=spec['name'],control='earlier_history_'+execution,**res))
        # Local perturbation never reselects the winning rule.
        mask=data['label_mask'];d={k:data[k][mask] for k in ['X','eligible','ratios','valid','labels','label_mask']};ds=np.array(m['dates'])[mask].tolist();ends=np.array(m['ends'])[mask].tolist()
        correction=extra[mask];rng=np.random.default_rng(40997)
        results=[];seen=set()
        c,a,b,wc,wa,wb=spec['recipe'];g=spec['gamma'];final=np.array([wc,wa*(1-g/(1-wc)),wb*(1-g/(1-wc)),g])
        for _ in range(300):
            w=final*rng.uniform(.95,1.05,4);w/=w.sum()
            if w[0]<.5 or w[3]>.1:continue
            score=d['X'][:,:,c]*w[0]+d['X'][:,:,a]*w[1]+d['X'][:,:,b]*w[2]+correction*w[3]
            z=screen({**d,'X':score[:,:,None]},[(0,-1,-1,1.,0.,0.)],ds,ends)
            assert z['valid'][0]
            results.append(float(np.expm1(z['log_later'][0]/((END-LABEL_START).days/365.2425))));seen.add(z['picks'][0].tobytes())
        neighborhood.append(dict(case=spec['name'],method='Multiply each final score weight independently by Uniform(.95,1.05), normalize; retain core>=.50,correction<=.10; fixed seed40997; zero-cost same-close full ledger interval',draws=300,valid_draws=len(results),distinct_holdings_paths=len(seen),min_cagr=min(results),median_cagr=float(np.median(results)),max_cagr=max(results),fraction_at_least70=float(np.mean(np.array(results)>=.70)),p10_cagr=float(np.quantile(results,.1)),p90_cagr=float(np.quantile(results,.9))))
    protected=read(P/'PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(f)!=h];assert not changed,changed
    assert all(sha(f)==h for f,h in sources.items())
    dump('FINAL_AUDIT.json',dict(prefix_factor_checks=len(checks),market_state_prefix_checks=5,packed_feature_checks=20,protected_files_checked=len(protected),protected_changed=changed,neighborhood=neighborhood,extra_checks=extra_checks,source_hashes=sources))
    print('AUDIT_OK',len(checks),'factor prefix checks; protected',len(protected),flush=True)
    for z in neighborhood:print(z,flush=True)
if __name__=='__main__':main()
