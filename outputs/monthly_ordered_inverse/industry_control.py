"""A documented ASML missing-industry control; fixed rule, no new rule selection."""
import numpy as np
import pandas as pd
from ordered_study import (P,read,dump,load_prices,panels,make_pack,basis,fit,metrics,
                           ranks_and_picks,engine,display,independent)

def main():
    dump('INDUSTRY_CONTROL_PLAN.json',dict(reason='ASML industry_relative60 is missing in64 pool months, including6 author-selected months; old code neutralized missing values.',
      assumption='Use XLK60-session return as a broad technology proxy for ASML, consistent with other semiconductor foreign issuers in inherited code. This is a researcher assumption, NOT recovered author industry definition or certified historical classification.',
      source='https://www.asml.com/en/company/about-asml',source_fact='ASML designs lithography machines for semiconductor manufacturing.',
      scope='Only isolated in-memory ASML sector60 substitution. Freeze the selected factor architecture; compare old weights and refit first48 only. No new hyperparameter search; old inputs and simulation unchanged.'))
    px,raw,members,periods,sources,pools=load_prices();raw['sector60']=raw['sector60'].copy()
    before_missing=int(raw['sector60'].ASML.reindex([p['start'] for p in periods]).isna().sum())
    raw['sector60']['ASML']=px.XLK.pct_change(60,fill_method=None)
    feat,cores,states=panels(px,raw);w=read(P/'FROZEN.json')['winner']
    d=make_pack(px,raw,periods,pools,members,32,'theme17',feat,cores,states,w['key'].split('_')[-1])
    X,spec=basis(d,w['core'],w['vol_sign'],w['acceleration']);dev=np.array(read(P/'INPUTS.json')['development'])
    wf,obj=fit(X,d,dev,spec,w['regularization']);results={}
    for name,weights in [('frozen_weights',np.array(w['weights'])),('refit_first48',wf)]:
        scores=np.einsum('ijk,k->ij',X,weights,optimize=False);rank,picks=ranks_and_picks(scores,d['eligible']);orders=[];rows=[]
        for i,p in enumerate(periods):
            sy=[d['symbols'][i][j] for j in picks[i]]
            orders.append(dict(date=p['start'],end=p['end'],signal_date=p['start'],symbols=sy,scores=[float(scores[i,j]) for j in picks[i]]))
            rows.append(dict(date=p['start'],author=p['holdings'],ours=list(map(display,sy)),author_ranks=[int(rank[i,j]) if j>=0 else None for j in d['order'][i]]))
        met,eq,trades,holds=engine.replay(px,raw['opening']*px/raw['close'],orders,cost=0,execution='close',start=pd.Timestamp(periods[0]['start']),end=pd.Timestamp(periods[-1]['end']))
        np.testing.assert_allclose(eq,independent(px,orders),rtol=1e-11,atol=1e-6)
        results[name]=dict(weights=weights.tolist(),metrics=met,matching={n:metrics(scores,d,mask) for n,mask in [('all',np.ones(len(dev),bool)),('calibration48',dev),('later21',~dev)]},months=rows)
    dump('INDUSTRY_CONTROL.json',dict(before_missing_dates=before_missing,results=results))
    print({k:dict(cagr=v['metrics']['cagr'],sharpe=v['metrics']['sharpe'],matches=v['matching']['all']) for k,v in results.items()})

if __name__=='__main__':main()
