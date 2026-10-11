"""Frozen industry candidate; no research labels or prospective data used."""
import numpy as np
import pandas as pd
from base_strategy import factors as base_factors,WEIGHTS
SECTOR={'Basic Materials':'XLB','Communication Services':'XLC','Energy':'XLE','Financial Services':'XLF','Industrials':'XLI','Technology':'XLK','Consumer Defensive':'XLP','Real Estate':'XLRE','Utilities':'XLU','Healthcare':'XLV','Consumer Cyclical':'XLY'}
FOREIGN={'ASML':'XLK','ARM':'XLK','TSM':'XLK','BABA':'XLY','NVO':'XLV','SAP':'XLK','TM':'XLY','SHEL':'XLE','SKHY':'XLK','BHP':'XLB','HSBC':'XLF','SMFG':'XLF','NVS':'XLV','SNDK':'XLK'}

def factors(panel):
    px=panel['adj'];r=px.pct_change(fill_method=None)
    vals,eligible,state=base_factors(px,panel['close'],panel['high'],panel['low'],panel['volume'])
    corr={b:(r.rolling(63).corr(r[b]),r.rolling(252).corr(r[b])) for b in SECTOR.values()}
    beta=r.rolling(63).cov(r.SPY).div(r.SPY.rolling(63).var(),axis=0).shift(1)
    residual=r-beta.mul(r.SPY,axis=0)
    extreme=r.SPY.lt(r.SPY.rolling(252).quantile(.1).shift(1))
    tail=residual.where(extreme,axis=0).rolling(252,min_periods=10).mean().replace([np.inf,-np.inf],np.nan)
    return vals,eligible,state,corr,tail,px.shift(42)/px.shift(252)-1

def selection(panel,day,etfs):
    day=pd.Timestamp(day);member=panel['member'].loc[day];stocks=sorted(member.index[member]);ss=sorted(set(stocks)|set(etfs))
    if len(stocks)!=32 or len(ss)!=49:raise ValueError('Need exact current32issuer+17ETF pool')
    vals,eligible,state,corr,tail,mom=factors(panel)
    tab=pd.DataFrame({k:v.loc[day,ss] for k,v in vals.items()});q=tab.rank(pct=True).fillna(.5)-.5;q['regime_cmf']*=state.at[day]
    core=.52*q.momentum;support=sum(q[k]*w for k,w in WEIGHTS.items() if k!='momentum')
    rel={};stable={};sectors=panel['sectors']
    for s in ss:
        b=None if s in etfs else FOREIGN.get(s,SECTOR.get(sectors.at[day,s]) if s in sectors else None)
        if b is None:rel[s]=stable[s]=np.nan;continue
        a,z=mom.at[day,s],mom.at[day,b]
        rel[s]=(1+a)/(1+z)-1 if np.isfinite(a) and np.isfinite(z) and z>-1 else np.nan
        stable[s]=-abs(corr[b][0].at[day,s]-corr[b][1].at[day,s])
    extra=pd.DataFrame({'sector_relative':pd.Series(rel),'sector_stability':pd.Series(stable),'tail_resilience':tail.loc[day,ss]})
    extra=extra.replace([np.inf,-np.inf],np.nan);ranks=extra.rank(pct=True).fillna(.5)-.5
    correction=.5*ranks.sector_relative+.25*ranks.sector_stability+.25*ranks.tail_resilience
    budget=.15 if state.at[day]>0 else 0.;scores=core+(1-budget/.48)*support+budget*correction
    good=[s for s in ss if eligible.at[day,s]];chosen=sorted(good,key=lambda s:(-scores[s],s))[:3]
    if len(chosen)<2:chosen=[]
    return dict(symbols=chosen,scores={s:float(scores[s]) for s in chosen},target_weights={s:1/len(chosen) for s in chosen},candidate_count=len(good),universe=ss,high_vol=bool(budget),correction_budget=budget,missing_factors={k:int(extra[k].isna().sum()) for k in extra},contributions={s:dict(core=float(core[s]),support=float((1-budget/.48)*support[s]),industry=float(budget*correction[s])) for s in chosen})
