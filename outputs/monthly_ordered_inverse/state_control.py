"""Small observable regime adjustments; no month IDs or ticker bias terms."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ordered_study import (P,read,dump,pack,basis,pairs,limits,objective,metrics,
    load_prices,market_states,ranks_and_picks,engine,display,independent,sha)

def fit_delta(X,d,mask,spec,base,budget):
    if mask.sum()<8:return base.copy(),dict(fallback='fewer_than8_calibration_months',months=int(mask.sum()))
    D,pw,g,ids=pairs(X,d,mask);A,b,bounds,prior=limits(spec);n=len(base)
    B=np.column_stack([np.eye(n),-np.eye(n)])
    def effective(z):return base+z[:n]-z[n:]
    def fun(z):
        w=effective(z);val,grad=objective(w,D,pw,.01,base)
        # Tiny penalty makes cancelling +/- coordinates unhelpful.
        return val+1e-10*z.sum(),np.r_[grad,-grad]+1e-10
    lower=np.array([x[0] for x in bounds]);upper=np.array([x[1] for x in bounds])
    constraints=[{'type':'eq','fun':lambda z:effective(z).sum()-1.,'jac':lambda z:np.r_[np.ones(n),-np.ones(n)]},
      {'type':'ineq','fun':lambda z:2*budget-z.sum(),'jac':lambda z:-np.ones(2*n)},
      {'type':'ineq','fun':lambda z:effective(z)-lower,'jac':lambda z:B},
      {'type':'ineq','fun':lambda z:upper-effective(z),'jac':lambda z:-B},
      {'type':'ineq','fun':lambda z:b-np.einsum('ij,j->i',A,effective(z),optimize=False),'jac':lambda z:-np.einsum('ij,jk->ik',A,B,optimize=False)}]
    z=minimize(fun,np.zeros(2*n),jac=True,method='SLSQP',bounds=[(0.,2*budget)]*(2*n),constraints=constraints,options={'maxiter':500,'ftol':1e-11})
    assert z.success,z.message;w=effective(z.x)
    assert np.abs(w-base).sum()<=2*budget+1e-8
    return w,dict(months=int(mask.sum()),objective=float(z.fun),l1_delta=float(np.abs(w-base).sum()))

def main():
    dump('STATE_CONTROL_PLAN.json',dict(reason='Per-month feasibility suggests nonconstant factor tradeoffs. Test only observable market states with explicitly small score corrections.',
      grid='3 partitions(SPY aboveMA200,SPY vol63 above trailing252median,their4-way cross) x2 score-move budgets5%/10%=6 rules.',
      constraints='Keep selected architecture and global frozen48 weights; state-specific weights learned on first48 only; same group/core bounds, coefficient L1 change<=2*budget so bounded features imply score change<=budget. No ticker/month identity. Fewer8 calibration months in state uses global rule.',
      timing='Follow-up iteration after original later21 result was viewed, explicitly not pristine OOS. No parameter/rule selected using later21 or return.',
      selection='Same calibration similarity metric as initial study. Zero cost same-close research only. No account promotion.'))
    m=read(P/'INPUTS.json');w=read(P/'FROZEN.json')['winner'];d=pack(w['key'],m);X,spec=basis(d,w['core'],w['vol_sign'],w['acceleration'])
    px,raw,members,periods,sources,pools=load_prices();dates=[p['start'] for p in periods]
    st=market_states(px).loc[dates].astype(int);dev=np.array(m['development']);base=np.array(w['weights'])
    partitions=dict(trend=st.spy_trend.to_numpy(),volatility=st.high_market_vol.to_numpy(),joint=(st.spy_trend*2+st.high_market_vol).to_numpy())
    trials=[];scorepacks=[]
    for name,groups in partitions.items():
        for budget in [.05,.10]:
            score=np.zeros(X.shape[:2]);coefs={};audit={}
            for group in np.unique(groups):
                wg,details=fit_delta(X,d,dev&(groups==group),spec,base,budget)
                score[groups==group]=np.einsum('ijk,k->ij',X[groups==group],wg,optimize=False)
                coefs[int(group)]=wg.tolist();audit[int(group)]=details
            trials.append(dict(id=len(trials),partition=name,budget=budget,weights=coefs,audit=audit,calibration=metrics(score,d,dev)));scorepacks.append(score)
    key=lambda r:(r['calibration']['selection_score'],r['calibration']['order_exact'],r['calibration']['set_exact'],-r['id'])
    winner=max(trials,key=key);dump('STATE_FROZEN.json',dict(trials=trials,winner=winner))
    score=scorepacks[winner['id']];ranks,picks=ranks_and_picks(score,d['eligible']);orders=[];rows=[]
    for i,p in enumerate(periods):
        sy=[d['symbols'][i][j] for j in picks[i]]
        orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=sy,scores=[float(score[i,j]) for j in picks[i]]))
        rows.append(dict(date=p['start'],end=p['end'],author=p['holdings'],ours=list(map(display,sy)),author_ranks=[int(ranks[i,j]) if j>=0 else None for j in d['order'][i]],state=int(partitions[winner['partition']][i])))
    met,eq,trades,holds=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0,execution='close',start=pd.Timestamp(dates[0]),end=pd.Timestamp(periods[-1]['end']))
    np.testing.assert_allclose(eq,independent(px,orders),rtol=1e-11,atol=1e-6)
    results=dict(rule=winner,metrics=met,matching={n:metrics(score,d,mask) for n,mask in [('all',np.ones(len(dev),bool)),('calibration48',dev),('later21',~dev)]},months=rows,
       validation=dict(independent_replay=True,fit_masks='first48 only',budget_and_group_constraints=True))
    dump('STATE_CONTROL.json',results);dump('state_control_HOLDINGS.json',holds);dump('state_control_TRADES.json',trades)
    print('State control',winner['partition'],winner['budget'],results['matching'],dict(cagr=met['cagr'],sharpe=met['sharpe']),flush=True)

if __name__=='__main__':main()
