from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
P=Path(__file__).resolve().parent;U=P.parent/'index_union';sys.path.insert(0,str(U))
from run_available_research import load
from price_units import split_adjusted_to_raw
ETFS=['SPY','QQQ','IWM','DIA','TLT','GLD','XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
def prepare():
 m,d,_=load(U/'alias_resolved_inputs/MANIFEST.json',3);sources=[]
 for s in ETFS:
  f=U/'sources/yahoo_prices'/(s+'.csv')
  if not f.exists():
   f=P/'sources'/(s+'.csv')
   if not f.exists():
    import yfinance as yf
    x=yf.Ticker(s).history(start='2008-01-01',end='2026-10-07',auto_adjust=False,actions=True)
    if x.empty:raise ValueError('No ETF data '+s)
    x.index=x.index.tz_localize(None).normalize();x.to_csv(f,index_label='Date')
  x=pd.read_csv(f,index_col='Date',parse_dates=True).rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume','Dividends':'dividend','Stock Splits':'split'})
  if not x.index.is_unique:raise ValueError('Duplicate ETF sessions')
  raw=split_adjusted_to_raw(x)
  good=x[['open','high','low','close','adj_close','volume']].apply(np.isfinite).all(axis=1)&x.close.gt(0)&x.volume.gt(0)
  good&=x.high.ge(x[['open','close','low']].max(axis=1)-1e-5)&x.low.le(x[['open','close']].min(axis=1)+1e-5)
  for k,c in [('adj','adj_close'),('close','close'),('opening','open'),('high','high'),('low','low'),('volume','volume')]:d[k][s]=x[c].where(good).reindex(d['close'].index)
  for k,c in [('dividends','dividend'),('splits','split'),('raw_per_signal','raw_per_signal')]:d[k][s]=raw[c].where(good).reindex(d['close'].index)
  d['eligible'][s]=d['adj'][s].notna();d['issuers'][s]='FUND_'+s
  d['sectors'][s]={'SPY':'Broad Equity','QQQ':'Technology','IWM':'Small Cap','DIA':'Broad Equity','TLT':'Treasury','GLD':'Gold','XLB':'Basic Materials','XLC':'Communication Services','XLE':'Energy','XLF':'Financial Services','XLI':'Industrials','XLK':'Technology','XLP':'Consumer Defensive','XLRE':'Real Estate','XLU':'Utilities','XLV':'Healthcare','XLY':'Consumer Cyclical'}[s]
  d['sector60'][s]=d['market_close'].pct_change(60,fill_method=None)
  sources.append({'symbol':s,'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 cols=d['close'].columns
 for k,v in d.items():
  if isinstance(v,pd.DataFrame):d[k]=v.reindex(columns=cols)
 (P/'ETF_SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n')
 return m,d
if __name__=='__main__':
 m,d=prepare();print('Prepared',len(d['close']),len(d['close'].columns),'columns')
