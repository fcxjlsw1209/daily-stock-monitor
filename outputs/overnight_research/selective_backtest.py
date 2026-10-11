from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import backtest as base
ROOT=Path(__file__).resolve().parent
ORIGINAL=base.make_signals

def selective_signals(c,o,div,vol,spy):
    models,on,r60,r3,z3=ORIGINAL(c,o,div,vol,spy)
    orig=models['strong_pullback_market']
    eligible=(c.rolling(200).count()>=200)&((c*vol).rolling(20).mean()>20e6)
    rank=r60.where(eligible).rank(axis=1,pct=True)
    risk_on=spy.Close>spy.Close.rolling(200).mean()
    pullback=eligible&(c>c.rolling(100).mean())&(r60>0)&(rank>=.5)&(z3>=-2.5)&(z3<=-.25)&(r3<0)
    sma50=spy.Close.rolling(50).mean()
    better_market=risk_on&(spy.Close>sma50)&(sma50>sma50.shift(20))
    realized_weights=orig.shift(2).fillna(0)
    valid=~((realized_weights>0)&on.isna()).any(axis=1)
    exposure=realized_weights.sum(axis=1)
    realized=((realized_weights*on.fillna(0)).sum(axis=1)-.001*exposure).where(valid)
    active=(exposure>0)&valid
    count=active.astype(float).rolling(126,min_periods=126).sum()
    wins=((realized>0)&active).astype(float).rolling(126,min_periods=126).sum()/count
    avg=realized.where(active,0).rolling(126,min_periods=126).sum()/count
    gate=(count>=30)&(wins>=.55)&(avg>=.0005)
    variants={
      'original':orig,
      'market_trend':base.pick(-z3,pullback.mul(better_market,axis=0)),
      'deeper_pullback':base.pick(-z3,(pullback&(z3<=-.75)).mul(risk_on,axis=0)),
      'strong_and_deep':base.pick(-z3,(pullback&(z3<=-.75)&(rank>=.75)).mul(risk_on,axis=0)),
      'rolling_edge':orig.mul(gate,axis=0)}
    return variants,on,r60,r3,z3

if __name__=='__main__':
    base.P=ROOT/'selective'
    shutil.copy(ROOT/'spy_source.csv',base.P/'spy_source.csv')
    base.make_signals=selective_signals
    base.main()
