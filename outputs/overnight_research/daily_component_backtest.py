from pathlib import Path
import json,shutil
import pandas as pd
import numpy as np
import backtest as base
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'intraday_validation'
MAP={'Technology':'XLK','Communication Services':'XLC','Consumer Cyclical':'XLY','Consumer Defensive':'XLP','Financial Services':'XLF','Healthcare':'XLV','Industrials':'XLI','Energy':'XLE','Basic Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE'}
ORIGINAL=base.make_signals

def make_signals(c,o,div,vol,spy):
    _,on,r60,r3,z3=ORIGINAL(c,o,div,vol,spy)
    meta={x['symbol']:x.get('sector') for x in json.loads((DATA/'download_manifest.json').read_text())}
    eligible=(c.rolling(200).count()>=200)&((c*vol).rolling(20).mean()>20e6)
    rank=r60.where(eligible).rank(axis=1,pct=True)
    candidate=eligible&(c>c.rolling(100).mean())&(r60>0)&(rank>=.75)&(r3<0)&(z3>=-2.5)&(z3<=-.75)
    candidate=candidate.mul(spy.Close>spy.Close.rolling(200).mean(),axis=0)
    sectret={}
    for etf in set(MAP.values()):
        e=pd.read_csv(DATA/'raw'/f'{etf}_daily.csv',index_col=0)
        e.index=pd.to_datetime(e.index,utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        sectret[etf]=e.Close.reindex(c.index).pct_change(3,fill_method=None)
    etfr=pd.DataFrame({s:sectret[MAP[meta[s]]] for s in c.columns},index=c.index)
    scale=c.pct_change(fill_method=None).rolling(20).std()*np.sqrt(3)
    linked=candidate&((etfr<0).add(spy.Close.pct_change(3,fill_method=None)<0,axis=0)>0)&((r3-etfr)>=-scale)
    def capped(mask):
        w=pd.DataFrame(0.,index=c.index,columns=c.columns)
        scores=(-z3).where(mask)
        for day,row in scores.iterrows():
            chosen=[];counts={}
            for s in row.dropna().sort_values(ascending=False,kind='stable').index:
                sector=meta[s]
                if counts.get(sector,0)>=2:continue
                chosen.append(s);counts[sector]=counts.get(sector,0)+1
                if len(chosen)==5:break
            if len(chosen)>=3:w.loc[day,chosen]=.2
        return w
    return {'daily_pullback':capped(candidate),'daily_market_linked':capped(linked)},on,r60,r3,z3
if __name__=='__main__':
    base.P=DATA/'daily_component';base.P.mkdir(exist_ok=True)
    shutil.copy(ROOT/'spy_source.csv',base.P/'spy_source.csv')
    base.SPLITS={'earlier':('2019-01-01','2021-12-31'),'recent':('2022-01-01','2026-09-18'),'all':('2019-01-01','2026-09-18')}
    base.make_signals=make_signals
    base.main()
