"""Separate the effect of fees from moving monthly signal date by one session."""
import json
import numpy as np
import pandas as pd
from common import P,base,FEATURES,features,score_frame,summary,dump
from validate import independent_nav


def run():
    selection=json.loads((P/'POLISH_SELECTION.json').read_text());models=json.loads((P/'POLISH_MODELS.json').read_text())
    chosen=list(selection['selected_by_early_curve_fit'].values())+[selection['best_full_cagr_retrospective']]
    px,fs,pools=base.load();results={};orders={};curves={};maxerr=0.
    for key in chosen:
        tag=key.split('__lag')[0];pool,era,family=tag.split('__');coef=np.array([models[tag]['coefficients'][f] for f in FEATURES])
        _,core,f,gate=features(px,fs,pools[pool]);sc=score_frame(core[family],f,coef);sim=base.Simulator(px,fs,pools[pool])
        for lag,cost in [(0,0),(0,10),(1,0),(1,10)]:
            name=tag+f'__lag{lag}_cost{cost}';nav,oo=sim.run(sc,gate,signal_lag=lag,cost_bps=cost)
            rebuild=independent_nav(px,oo,nav.index,cost);error=float(((rebuild-nav)/nav).abs().max());assert error<1e-11;maxerr=max(maxerr,error)
            results[name]=dict(**summary(nav),signal_lag=lag,cost_bps_roundtrip=cost,fit_era=era)
            orders[name]=oo;curves[name]=nav
            print(name,round(results[name]['cagr_252']*100,2),round(results[name]['sharpe'],3),flush=True)
    dump('EXECUTION_RESULTS.json',results);dump('EXECUTION_ORDERS.json',orders);pd.DataFrame(curves).to_csv(P/'EXECUTION_CURVES.csv.gz')
    dump('EXECUTION_VALIDATION.json',dict(cases=len(results),max_independent_nav_relative_error=maxerr,no_refitting=True,
        parameters='Exactly the fixed polished model coefficients; no date-specific coefficient changes.',
        interpretation='Both signals select for the FOLLOWING month. A one-session lag tests timing sensitivity, not a future-month ranking allegation.'))


if __name__=='__main__':run()
