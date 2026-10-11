"""Audit archive coverage; never fill missing bars or silently merge providers."""
from pathlib import Path
from io import StringIO
import pandas as pd,numpy as np,json,zipfile,hashlib
P=Path(__file__).resolve().parent;A=P/'additional_sources';D=P/'recovered_prices';D.mkdir(exist_ok=True);summary=[]
def audit(df,source,dayfirst=False):
 df=df.copy();df.columns=[str(c).lower().replace(' ','_') for c in df.columns]
 if 'date_(edt)' in df:df=df.rename(columns={'date_(edt)':'date'})
 if dayfirst:df['date']=pd.to_datetime(df.date,format='%d-%m-%Y',errors='coerce').dt.strftime('%Y-%m-%d')
 else:df['date']=pd.to_datetime(df.date,format='mixed',errors='coerce').dt.strftime('%Y-%m-%d')
 prices=['open','high','low','close']
 for col in prices+['volume']:df[col]=pd.to_numeric(df[col],errors='coerce')
 missing=df[prices+['volume']].isna().any(axis=1)|df.date.isna()
 invalid=(df[prices]<=0).any(axis=1)|(df.low>df.high)|(df.open<df.low-1e-5)|(df.open>df.high+1e-5)|(df.close<df.low-1e-5)|(df.close>df.high+1e-5)|(df.volume<0)
 dup=df.duplicated(['symbol','date'],keep=False)
 # Verified exchange trading suspension; empty post-suspension rows remain quarantined.
 post=((df.symbol=='TWTR')&(df.date>='2022-10-28'))|((df.symbol=='ATVI')&(df.date>='2023-10-13'))
 df['quarantine']=missing|invalid|dup|post;df['source']=source;df['identity_and_adjustments_verified']=False
 df.to_csv(D/(source+'.csv.gz'),index=False,compression='gzip')
 for symbol,g in df.groupby('symbol'):
  good=g[~g.quarantine]
  summary.append(dict(source=source,symbol=symbol,rows=len(g),quarantined=int(g.quarantine.sum()),first_valid=good.date.min() if len(good) else None,last_valid=good.date.max() if len(good) else None,identity_and_adjustments_verified=False))
 return df
def main():
    with zipfile.ZipFile(A/'kaggle_nyse.zip') as z:audit(pd.read_csv(z.open('prices.csv')),'kaggle_2016_raw')
    for name,symbol in [('github_twtr','TWTR'),('github_twtr2','TWTR'),('yhoo_archive','YHOO')]:
     df=pd.read_csv(A/(name+'.txt'));df['symbol']=symbol;audit(df,name,dayfirst=name=='github_twtr')
    for name in ['chart_atvi','stockanalysis_atvi']:
     df=pd.read_html(StringIO((A/(name+'.txt')).read_text()))[0];df['symbol']='ATVI';audit(df,name)
    pd.DataFrame(summary).to_csv(D/'coverage.csv',index=False)
    print('sources',len(set(r['source'] for r in summary)),'rows',sum(r['rows'] for r in summary),'quarantined',sum(r['quarantined'] for r in summary),flush=True)

if __name__=='__main__':main()
