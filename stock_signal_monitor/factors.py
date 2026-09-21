from dataclasses import dataclass
import numpy as np
import pandas as pd

MAP = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Financial Services': 'XLF', 'Healthcare': 'XLV', 'Industrials': 'XLI', 'Energy': 'XLE', 'Basic Materials': 'XLB', 'Utilities': 'XLU', 'Real Estate': 'XLRE'}

@dataclass
class Inputs:
    close:pd.DataFrame
    opening:pd.DataFrame
    dividends:pd.DataFrame
    volume:pd.DataFrame
    sector:dict
    sector_prices:pd.DataFrame

def signals(data):
    c=data.close;stocks=[s for s in c if s not in ['SPY','QQQ']]
    r60=c.pct_change(60,fill_method=None);r3=c.pct_change(3,fill_method=None)
    scale=c.pct_change(fill_method=None).rolling(20).std()*np.sqrt(3);z=r3/scale
    eligible=(c.rolling(200).count()>=200)&((c*data.volume).rolling(20).mean()>20e6)
    rank=r60[stocks].where(eligible[stocks]).rank(axis=1,pct=True)
    common=eligible&(c>c.rolling(100).mean())&(r60>0)&(r3<0)&(z>=-2.5)&(z<=-.75)
    common=common.mul(c.SPY>c.SPY.rolling(200).mean(),axis=0)
    a=common[stocks]&(rank>=.75)
    er=pd.DataFrame({s:data.sector_prices[MAP[data.sector[s]]].pct_change(3,fill_method=None) for s in stocks})
    linked=a&((er<0).add(r3.SPY<0,axis=0)>0)&((r3[stocks]-er)>=-scale[stocks])
    return {'stock_base':(a,-z[stocks]),'stock_industry':(linked,-z[stocks]),
      'SPY_pullback':(common[['SPY']],-z[['SPY']]),'QQQ_pullback':(common[['QQQ']],-z[['QQQ']])}

WEIGHTS = pd.Series([.30,.20,.25,.15,.10], index=["long_momentum","relative_strength","pullback","stabilization","stability"])
FACTORS = list(WEIGHTS.index)

def build(data,raw):
    a=raw['adj_close'];cols=a.columns;c=data.close[cols]
    r=a.pct_change(fill_method=None);r3=a.pct_change(3,fill_method=None);r60=a.pct_change(60,fill_method=None)
    z=r3/(r.rolling(20).std()*np.sqrt(3))
    mom=a.shift(21)/a.shift(252)-1;r12=a/a.shift(252)-1
    sector60=raw['sector_adjusted'].pct_change(60,fill_method=None)
    relative=pd.DataFrame({s:(1+r60[s])/(1+sector60[MAP[data.sector[s]]])-1 for s in cols})
    spread=raw['high']-raw['low']
    location=((c-raw['low'])/spread).where(spread!=0,.5)
    lr=np.log(a/a.shift(1));smooth=lr.rolling(60).sum()/lr.abs().rolling(60).sum()
    factors=dict(long_momentum=mom,relative_strength=relative,pullback=-abs(z+1.25),stabilization=location,stability=smooth)
    liquid=(c*data.volume[cols]).rolling(20).mean()>20e6
    valid=((a>0)&np.isfinite(a)).rolling(253).sum().eq(253)&liquid
    valid &= np.isfinite(raw['high'])&np.isfinite(raw['low'])&np.isfinite(c)
    valid &= (raw['low']>0)&(raw['high']>=raw['low'])&(c>=raw['low'])&(c<=raw['high'])
    valid &= np.isfinite(data.volume[cols])&(data.volume[cols]>0)&(data.opening[cols]>0)
    for v in factors.values():valid &= np.isfinite(v)
    ranks={k:v.where(valid).rank(axis=1,pct=True) for k,v in factors.items()}
    equal=sum(ranks.values())/len(FACTORS)
    weighted=sum(ranks[k]*WEIGHTS[k] for k in FACTORS)
    r12rank=r12.where(valid).rank(axis=1,pct=True)
    alternative=weighted-WEIGHTS.long_momentum*ranks['long_momentum']+WEIGHTS.long_momentum*r12rank
    regime=data.close.SPY>data.close.SPY.rolling(200).mean()
    gate=(valid&(a>a.rolling(100).mean())&(mom>0)&(r60>0)&(z>=-3)&(z<0)).mul(regime,axis=0)
    old_mask,old_score=signals(data)['stock_base']
    common=old_mask&valid
    models={'legacy':(old_mask,old_score),'legacy_common':(common,old_score),'legacy_equal':(common,equal),
      'legacy_weighted':(common,weighted),'soft_equal':(gate&(equal>=.65),equal),
      'soft_weighted':(gate&(weighted>=.65),weighted),'soft_weighted_r12':(gate&(alternative>=.65),alternative)}
    return dict(factors=factors,r12=r12,valid=valid,ranks=ranks,gate=gate,z=z,models=models)
