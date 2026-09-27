from pathlib import Path
import sys
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'factor_research'))
from backtest_factors import load_inputs,build


def features(data,raw):
    b=build(data,raw);cols=raw['adj_close'].columns;c=data.close[cols];v=data.volume[cols]
    high=raw['high'];low=raw['low'];spread=high-low
    multiplier=((2*c-high-low)/spread).where(spread!=0,0.)
    cmf=(multiplier*v).rolling(21).sum()/v.rolling(21).sum()
    valid=b['valid']&np.isfinite(cmf);cmfr=cmf.where(valid).rank(axis=1,pct=True)
    r=b['ranks'];score=.30*r['long_momentum']+.20*r['relative_strength']+.25*r['pullback']+.10*r['stability']+.15*cmfr
    from backtest_swing import signals
    old=signals(data)
    masks={'strict':old['stock_base'][0]&valid&(b['factors']['long_momentum']>0),
      'industry':old['stock_industry'][0]&valid&(b['factors']['long_momentum']>0),
      'broad':b['gate']&valid&(score>=.65)}
    tr=pd.DataFrame(np.maximum.reduce([spread.to_numpy(),abs(high-c.shift(1)).to_numpy(),abs(low-c.shift(1)).to_numpy()]),index=c.index,columns=c.columns)
    return dict(score=score,masks=masks,ma5=c.rolling(5).mean(),atr=tr.rolling(14).mean(),cmf=cmf,base=b)
