"""Shared code for inverse learning of a monthly ranking rule."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;O=P.parent;OLD=O/'monthly_reconstruction_v2'
spec=importlib.util.spec_from_file_location('inverse_monthly_base',OLD/'research.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
START,END=base.START,base.END


def dump(name,obj):
    (P/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


FEATURES=['vol','downside','relative','cmf21','volume_ratio','near252','trend_gap',
          'draw63','r63','r21','r5','accel','path_efficiency','vol_market','extension_market']


def features(px,fs,member):
    raw=base.raw_features(px,fs)
    logret=np.log(px).diff()
    raw['path_efficiency']=np.log(px.shift(42)/px.shift(252)).abs()/logret.abs().rolling(210).sum().shift(42)
    rank={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()}
    core={'raw':.8*rank['m252_42']+.2*rank['m126_42'],
          'risk':.8*rank['risk']+.2*rank['m252_42']}
    core['mixed']=.5*core['raw']+.5*core['risk']
    f={k:rank[k].fillna(.5).where(member)-.5 for k in FEATURES[:-2]}
    regime=(px.SPY/px.SPY.rolling(200).mean()-1).clip(-.2,.2)/.2
    f['vol_market']=f['vol'].mul(regime,axis=0)
    f['extension_market']=f['near252'].mul(regime,axis=0)
    gate=(raw['m252_42']>0)&(rank['m252_42']>=.5)
    return raw,core,f,gate


def score_frame(core,f,coef):
    return .5*core+sum(float(w)*f[k] for k,w in zip(FEATURES,coef))


def summary(nav):
    r=nav.pct_change().dropna();wealth=float(nav.iloc[-1]);n=len(nav)-1
    return dict(start=str(nav.index[0].date()),end=str(nav.index[-1].date()),wealth=wealth,
                cagr_252=float(wealth**(252/n)-1),max_dd=float((nav/np.maximum(1.,nav.cummax())-1).min()),
                sharpe=float(r.mean()/r.std()*np.sqrt(252)))
