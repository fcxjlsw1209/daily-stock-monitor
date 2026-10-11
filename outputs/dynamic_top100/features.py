"""Six strategy signals with daily membership applied BEFORE percentile ranks."""
import numpy as np
import pandas as pd
from universe import rank_factor

def six_features(data,raw,membership,sector_etfs):
    a=raw['adj_close'];cols=a.columns;c=data.close[cols];v=data.volume[cols];h=raw['high'];l=raw['low']
    if not a.index.equals(membership.index) or not a.columns.equals(membership.columns):raise ValueError('Membership not aligned to price panel')
    if not sector_etfs.index.equals(a.index) or not sector_etfs.columns.equals(a.columns):raise ValueError('Effective-dated sectors not aligned')
    if sector_etfs.where(membership).isna().to_numpy()[membership.to_numpy()].any():raise ValueError('Missing sector for member')
    r=a.pct_change(fill_method=None);r3=a.pct_change(3,fill_method=None);r60=a.pct_change(60,fill_method=None)
    z=r3/(r.rolling(20).std()*np.sqrt(3));mom=a.shift(21)/a.shift(252)-1
    etf60=raw['sector_adjusted'].pct_change(60,fill_method=None)
    sector60=pd.DataFrame(np.nan,index=a.index,columns=cols)
    for symbol in etf60:
        sector60=sector60.where(~sector_etfs.eq(symbol),pd.DataFrame(np.repeat(etf60[symbol].to_numpy()[:,None],len(cols),axis=1),index=a.index,columns=cols))
    relative=(1+r60)/(1+sector60)-1
    spread=h-l;location=((c-l)/spread).where(spread!=0,.5)
    lr=np.log(a/a.shift(1));smooth=lr.rolling(60).sum()/lr.abs().rolling(60).sum()
    vals={'mom':mom,'relative':relative,'pullback':-abs(z+1.25),'stabilization':location,'stability':smooth}
    valid=((a>0)&np.isfinite(a)).rolling(253).sum().eq(253)&((c*v).rolling(20).mean()>20e6)
    valid &= np.isfinite(h)&np.isfinite(l)&np.isfinite(c)&(l>0)&(h>=l)&(c>=l)&(c<=h)
    valid &= np.isfinite(v)&(v>0)&(data.opening[cols]>0)
    for x in vals.values():valid &= np.isfinite(x)
    valid &= membership
    ranks={k:rank_factor(x,membership,valid) for k,x in vals.items()}
    cmf=(((2*c-h-l)/spread).where(spread!=0,0)*v).rolling(21).sum()/v.rolling(21).sum()
    cv=valid&np.isfinite(cmf)
    score=.30*ranks['mom']+.20*ranks['relative']+.25*ranks['pullback']+.10*ranks['stability']+.15*rank_factor(cmf,membership,cv)
    regime=data.close.SPY>data.close.SPY.rolling(200).mean()
    gate=(valid&(a>a.rolling(100).mean())&(mom>0)&(r60>0)&(z>=-3)&(z<0)).mul(regime,axis=0)
    base_mask=gate&cv&(score>=.65)
    tr=pd.DataFrame(np.maximum.reduce([spread.to_numpy(),abs(h-c.shift(1)).to_numpy(),abs(l-c.shift(1)).to_numpy()]),index=c.index,columns=c.columns)
    common={'ma5':c.rolling(5).mean(),'atr':tr.rolling(14).mean()}
    result={'baseline':dict(common,score=score,mask=base_mask)}
    pressure=(-r.clip(upper=0))*v/v.shift(1).rolling(20).mean();decay=pressure.shift(1).rolling(3).mean()-pressure
    market=(data.close.SPY+data.dividends.SPY)/data.close.SPY.shift(1)-1
    beta=r.shift(1).rolling(60).cov(market.shift(1)).div(market.shift(1).rolling(60).var(),axis=0)
    residual=r-beta.mul(market,axis=0)
    down=pd.DataFrame(np.repeat((market<0).to_numpy()[:,None],len(cols),axis=1),index=r.index,columns=cols)
    resilience=residual.where(down).rolling(20,min_periods=3).mean()
    for name,value in [('shadow',decay),('resilience',resilience)]:
        good=cv&np.isfinite(value)
        ranked=.30*ranks['mom']+(6/7)*(score-.30*ranks['mom'])+.10*rank_factor(value,membership,good)
        result[name]=dict(common,score=ranked,mask=base_mask&good)
    vals.update(cmf=cmf,accel=a/a.shift(21)-a.shift(21)/a.shift(42),high52=a/a.rolling(252).max(),jump=-r.abs().rolling(60).max()/r.abs().rolling(60).sum(),downside=-r.clip(upper=0).rolling(60).std())
    for name,mw,secondary in [('accel',.3,'accel'),('high52',.3,'high52'),('accel40',.4,'accel')]:
        weights={'mom':mw,secondary:.5-mw,'jump':.2,'downside':.15,'cmf':.15};good=valid.copy()
        for k in weights:good &= np.isfinite(vals[k])
        ranked=sum(rank_factor(vals[k],membership,good)*w for k,w in weights.items())
        mask=(good&(mom>0)&(a>a.rolling(100).mean())&(ranked>=.65)).mul(regime,axis=0)
        result[name]=dict(common,score=ranked,mask=mask)
    return result
