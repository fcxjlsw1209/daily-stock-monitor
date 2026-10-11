"""Frozen 65.3395543% research candidate; prospective price-only selection."""
import numpy as np
import pandas as pd
from base_strategy import factors as base_factors, WEIGHTS

SECTOR={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE','Financial Services':'XLF','Industrials':'XLI','Technology':'XLK','Consumer Defensive':'XLP','Real Estate':'XLRE','Utilities':'XLU','Healthcare':'XLV','Consumer Cyclical':'XLY'}
FOREIGN={'ASML':'XLK','ARM':'XLK','TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY','SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF','NVS':'XLV','SNDK':'XLK'}


def factors(panel):
    px=panel['adj'];r=px.pct_change(fill_method=None)
    vals,eligible,state=base_factors(px,panel['close'],panel['high'],panel['low'],panel['volume'])
    corr={b:(r.rolling(63).corr(r[b]),r.rolling(252).corr(r[b])) for b in SECTOR.values()}
    beta=r.rolling(63).cov(r.SPY).div(r.SPY.rolling(63).var(),axis=0).shift(1)
    residual=r-beta.mul(r.SPY,axis=0)
    extreme=r.SPY.lt(r.SPY.rolling(252).quantile(.1).shift(1))
    tail=residual.where(extreme,axis=0).rolling(252,min_periods=10).mean()
    mom={n:px.shift(42)/px.shift(n)-1 for n in [126,252]}
    efficiency=np.log(px.shift(42)/px.shift(294))/np.log(px/px.shift(1)).abs().rolling(252).sum().shift(42).replace(0,np.nan)
    calm=r.SPY.rolling(63).std()*np.sqrt(252)<.10
    return vals,eligible,state,corr,tail,mom,efficiency,calm


def selection(panel,day,etfs):
    day=pd.Timestamp(day);member=panel['member'].loc[day]
    stocks=sorted(member.index[member]);ss=sorted(set(stocks)|set(etfs))
    if len(stocks)!=32 or len(ss)!=49:raise ValueError('Need exact current32issuer+17ETF pool')
    vals,eligible,state,corr,tail,mom,efficiency,calm=factors(panel)
    tab=pd.DataFrame({k:v.loc[day,ss] for k,v in vals.items()})
    q=tab.rank(pct=True).fillna(.5)-.5;q['regime_cmf']*=state.at[day]
    rel={126:{},252:{}};stable={};sectors=panel['sectors']
    for s in ss:
        b=None if s in etfs else FOREIGN.get(s,SECTOR.get(sectors.at[day,s]) if s in sectors else None)
        for n in rel:
            if b is None:rel[n][s]=np.nan;continue
            a,z=mom[n].at[day,s],mom[n].at[day,b]
            rel[n][s]=(1+a)/(1+z)-1 if np.isfinite(a) and np.isfinite(z) and z>-1 else np.nan
        stable[s]=-abs(corr[b][0].at[day,s]-corr[b][1].at[day,s]) if b is not None else np.nan
    extra=pd.DataFrame({'sector126':pd.Series(rel[126]),'sector252':pd.Series(rel[252]),
        'stability':pd.Series(stable),'tail':tail.loc[day,ss],
        'raw_momentum':mom[252].loc[day,ss],'path':efficiency.loc[day,ss]})
    ranks=extra.replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5)-.5
    if calm.at[day]:
        parts=dict(core=.5*ranks.raw_momentum,volatility=.2*q.volatility,
            sector=.0875*(ranks.sector126+ranks.sector252),path=.05*ranks.path,cmf=.075*q.regime_cmf)
        budget=0.
    else:
        support=sum(q[k]*w for k,w in WEIGHTS.items() if k!='momentum')
        budget=.10 if state.at[day]>0 else 0.
        parts=dict(core=.52*q.momentum,support=(1-budget/.48)*support,
            industry=budget*(.25*ranks.sector126+.5*ranks.stability+.25*ranks['tail']))
    scores=sum(parts.values())
    good=[s for s in ss if eligible.at[day,s]]
    chosen=sorted(good,key=lambda s:(-scores[s],s))[:3]
    if len(chosen)<2:chosen=[]
    return dict(symbols=chosen,scores={s:float(scores[s]) for s in chosen},
        target_weights={s:1/len(chosen) for s in chosen},candidate_count=len(good),universe=ss,
        calm=bool(calm.at[day]),high_vol=bool(state.at[day]>0),correction_budget=budget,
        contributions={s:{k:float(v[s]) for k,v in parts.items()} for s in chosen})
