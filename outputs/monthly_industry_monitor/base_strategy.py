"""Exact frozen four-factor monthly candidate. No training, curve or ledger inputs."""
import numpy as np
import pandas as pd
WEIGHTS={'momentum':.52,'volatility':.219375,'acceleration':.185625,'regime_cmf':.075}

def factors(adj,close,high,low,volume):
    returns=adj.pct_change(fill_method=None)
    sigma=returns.rolling(63).std();spy=sigma['SPY']
    regime=pd.Series(np.where(spy>spy.rolling(252).median(),1.,-1.),index=adj.index)
    spread=high-low;loc=((2*close-high-low)/spread.replace(0,np.nan)).where(spread!=0,0.)
    cmf=(loc*volume).rolling(63).sum()/volume.rolling(63).sum()
    values=dict(momentum=(adj.shift(42)/adj.shift(294)-1)/sigma.replace(0,np.nan),
        volatility=returns.rolling(126).std(),acceleration=adj.shift(42)/adj.shift(126)-adj.shift(126)/adj.shift(210),
        regime_cmf=-cmf)
    values={k:v.replace([np.inf,-np.inf],np.nan) for k,v in values.items()}
    eligible=adj.notna().rolling(295,min_periods=295).sum().eq(295)&adj.gt(0)&(adj.shift(42)/adj.shift(252)-1).gt(0)
    return values,eligible,regime

def selection(panel,day,etfs):
    adj=panel['adj'];day=pd.Timestamp(day)
    members=panel['member'].loc[day];stocks=sorted(members.index[members])
    if len(stocks)!=32:raise ValueError('Need exactly32cached issuer representatives; no substitute pool')
    symbols=sorted(set(stocks)|set(etfs))
    if len(symbols)!=49:raise ValueError('Expected32stocks+17distinctETFs')
    vals,eligible,state=factors(adj,panel['close'],panel['high'],panel['low'],panel['volume'])
    table=pd.DataFrame({k:z.loc[day,symbols] for k,z in vals.items()})
    ranks=table.rank(pct=True).fillna(.5)-.5
    ranks['regime_cmf']*=state.at[day]
    scores=sum(ranks[k]*w for k,w in WEIGHTS.items())
    qualified=[s for s in symbols if eligible.at[day,s]]
    chosen=sorted(qualified,key=lambda s:(-scores[s],s))[:3]
    if len(chosen)<2:chosen=[]
    return dict(symbols=chosen,scores={s:float(scores[s]) for s in chosen},target_weights={s:1/len(chosen) for s in chosen},candidate_count=len(qualified),universe=symbols)
