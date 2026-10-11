"""Frozen six-strategy factor equations with daily eligible-union ranking.

Input columns are permanent security IDs, not reused ticker strings. sector60
must be the security's contemporaneously known sector ETF 60-session return.
This pure calculation does not admit data or write any simulation ledger.
"""
import numpy as np
import pandas as pd
from data_eligibility import entry_eligibility


def features(adj, close, opening, high, low, volume, sector60, market_close,
             market_dividends, eligible):
    frames=[close,opening,high,low,volume,sector60,eligible]
    if not adj.index.is_unique or not adj.index.is_monotonic_increasing or not adj.columns.is_unique:
        raise ValueError('Unique ordered sessions and security IDs required')
    for f in frames:
        if not adj.index.equals(f.index) or not adj.columns.equals(f.columns):
            raise ValueError('Exact input alignment required')
    if eligible.isna().any().any() or not all(t==bool for t in eligible.dtypes):
        raise ValueError('Complete boolean membership mask required')
    for x in [market_close,market_dividends]:
        if not adj.index.equals(x.index):raise ValueError('Market input misaligned')
    eligible=entry_eligibility(eligible,adj,opening,high,low,close,volume)
    a=adj;c=close;v=volume;r=a.pct_change(fill_method=None)
    r3=a.pct_change(3,fill_method=None);r60=a.pct_change(60,fill_method=None)
    z=r3/(r.rolling(20).std()*np.sqrt(3));mom=a.shift(21)/a.shift(252)-1
    relative=(1+r60)/(1+sector60)-1
    spread=high-low;location=((c-low)/spread).where(spread!=0,.5)
    lr=np.log(a/a.shift(1));smooth=lr.rolling(60).sum()/lr.abs().rolling(60).sum()
    base_values=dict(long_momentum=mom,relative_strength=relative,pullback=-abs(z+1.25),stabilization=location,stability=smooth)
    valid=((a>0)&np.isfinite(a)).rolling(253).sum().eq(253)&((c*v).rolling(20).mean()>20e6)
    valid &= np.isfinite(high)&np.isfinite(low)&np.isfinite(c)&(low>0)&(high>=low)&(c>=low)&(c<=high)
    valid &= np.isfinite(v)&(v>0)&(opening>0)
    for value in base_values.values():valid &= np.isfinite(value)
    valid &= eligible
    rank={k:x.where(valid).rank(axis=1,pct=True) for k,x in base_values.items()}
    cmf=(((2*c-high-low)/spread).where(spread!=0,0)*v).rolling(21).sum()/v.rolling(21).sum()
    base_valid=valid&np.isfinite(cmf)
    score=.30*rank['long_momentum']+.20*rank['relative_strength']+.25*rank['pullback']+.10*rank['stability']+.15*cmf.where(base_valid).rank(axis=1,pct=True)
    regime=market_close>market_close.rolling(200).mean()
    gate=(valid&(a>a.rolling(100).mean())&(mom>0)&(r60>0)&(z>=-3)&(z<0)).mul(regime,axis=0)
    base_mask=gate&base_valid&(score>=.65)
    tr=pd.DataFrame(np.maximum.reduce([spread.to_numpy(),abs(high-c.shift(1)).to_numpy(),abs(low-c.shift(1)).to_numpy()]),index=c.index,columns=c.columns)
    atr=tr.rolling(14).mean()
    result={'baseline':dict(score=score,mask=base_mask,ma5=c.rolling(5).mean(),atr=atr)}
    pressure=(-r.clip(upper=0))*v/v.shift(1).rolling(20).mean()
    decay=pressure.shift(1).rolling(3).mean()-pressure
    market=(market_close+market_dividends)/market_close.shift(1)-1
    beta=r.shift(1).rolling(60).cov(market.shift(1)).div(market.shift(1).rolling(60).var(),axis=0)
    residual=r-beta.mul(market,axis=0)
    down=pd.DataFrame(np.repeat((market<0).to_numpy()[:,None],len(r.columns),axis=1),index=r.index,columns=r.columns)
    resilience=residual.where(down).rolling(20,min_periods=3).mean()
    for name,value in [('pressure',decay),('resilience',resilience)]:
        okay=base_valid&np.isfinite(value)
        s=.30*rank['long_momentum']+(6/7)*(score-.30*rank['long_momentum'])+.10*value.where(okay).rank(axis=1,pct=True)
        # These shadows change sorting only, not the original baseline threshold.
        result[name]=dict(score=s,mask=base_mask&okay,ma5=c.rolling(5).mean(),atr=atr)
    values=dict(mom=mom,accel=a/a.shift(21)-a.shift(21)/a.shift(42),high52=a/a.rolling(252).max(),
      jump=-r.abs().rolling(60).max()/r.abs().rolling(60).sum(),downside=-r.clip(upper=0).rolling(60).std(),cmf=cmf)
    for name,second,mw,sw in [('accel','accel',.3,.2),('high52','high52',.3,.2),('accel40','accel',.4,.5-.4)]:
        weights={'mom':mw,second:sw,'jump':.2,'downside':.15,'cmf':.15}
        okay=valid.copy()
        for k in weights:okay &= np.isfinite(values[k])
        s=sum(values[k].where(okay).rank(axis=1,pct=True)*w for k,w in weights.items())
        mask=(okay&(mom>0)&(a>a.rolling(100).mean())&(s>=.65)).mul(regime,axis=0)
        result[name]=dict(score=s,mask=mask,ma5=c.rolling(20).mean(),atr=atr)
    return result
