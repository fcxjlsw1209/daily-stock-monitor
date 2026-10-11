"""Read existing audited research inputs. No remote download or synthetic ranking refresh."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent.parent

def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())

def load_panel(etfs):
    source={}
    def frame(path):
        source[str(path)]=sha(path);return pd.read_csv(path,index_col=0,parse_dates=True,low_memory=False)
    p=O/'index_union/alias_resolved_inputs';panel={k:frame(p/(filename+'.csv.gz')) for k,filename in [('adj','adj'),('open','opening'),('close','close'),('high','high'),('low','low'),('volume','volume')]}
    manifest_files=[O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']
    manifest_dates=[]
    for mf in manifest_files:
        source[str(mf)]=sha(mf)
        for entry in read(mf):
            assert sha(entry['file'])==entry['sha256'],'Price source fingerprint mismatch'
            z=frame(Path(entry['file']))
            if entry['symbol'] in etfs or entry['symbol'] in ['TSM','BABA','NVO','SAP','TM','XLB','XLC','XLRE','XLU','XLP']:manifest_dates.append(z.index[-1])
            for k,col in [('adj','Adj Close'),('open','Open'),('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:
                panel[k][entry['symbol']]=z[col].reindex(panel[k].index)
    # Exact source universe and additional prices of guidance_tweaks, never infer
    # today's membership by forwarding an old monthly observation.
    for folder in [O/'monthly_published44_reconstruction/prices',O/'monthly_top35_extension/prices']:
        for f in folder.glob('*.csv'):
            mf=f.with_suffix('.source.json');source[str(mf)]=sha(mf);md=read(mf)
            if sha(f)!=md['sha256']:raise ValueError('Additional price fingerprint mismatch')
            z=frame(f);symbol=f.stem
            if symbol in panel['adj'] and panel['adj'][symbol].notna().any():raise ValueError('Unexpected duplicate archive security')
            for k,col in [('adj','Adj Close'),('open','Open'),('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:panel[k][symbol]=z[col].reindex(panel['adj'].index)
            manifest_dates.append(z.index[-1])
    # Reproduce the frozen early extension without rewriting its source files.
    for f in (O/'monthly_2016_extension/prices').glob('*.csv'):
        mf=f.with_suffix('.source.json');source[str(mf)]=sha(mf)
        if sha(f)!=read(mf)['sha256']:raise ValueError('Early price fingerprint mismatch')
        z=frame(f).reindex(panel['adj'].index);s=f.stem
        if s not in panel['adj']:
            for v in panel.values():v[s]=np.nan
        a=panel['adj'][s];overlap=a.notna()&z['Adj Close'].notna()
        first=a.index[overlap][0] if overlap.any() else a.index[-1]+pd.Timedelta(days=1)
        ratio=float(a.at[first]/z.at[first,'Adj Close']) if overlap.any() else 1.
        gap=a.isna()&(a.index<first)
        panel['adj'].loc[gap,s]=z.loc[gap,'Adj Close']*ratio
        for k,c in [('open','Open'),('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:panel[k].loc[gap,s]=z.loc[gap,c]
    for s,cache in [('CHL','CHL'),('PTR','PTRCY')]:
        f=O/'dynamic_top100/additional_sources/tiingo'/f'{cache}_prices.json';source[str(f)]=sha(f)
        z=pd.DataFrame(read(f));z.index=pd.to_datetime(z.date,utc=True).dt.tz_localize(None)
        if not z.index.is_unique:raise ValueError('Duplicate historical sessions')
        z=z.reindex(panel['adj'].index);ok=z.index<pd.Timestamp('2021-01-01')
        for k,c in [('adj','adjClose'),('open','open'),('close','close'),('high','high'),('low','low'),('volume','volume')]:panel[k][s]=z[c].where(ok)
    sectors=frame(O/'index_union/alias_resolved_inputs/sectors.csv.gz');panel['sectors']=sectors
    mf=O/'monthly_top35_extension/POOLS.json';source[str(mf)]=sha(mf);pools=read(mf)['32']
    early=O/'monthly_2016_extension/POOLS.json';source[str(early)]=sha(early)
    pools={**{day:ss for day,ss in read(early).items() if day<'2020-12-31'},**pools}
    actual=lambda s:{'META':'research_equity_FB','BRK-B':'BRK.B'}.get(s,s)
    member=pd.DataFrame(False,index=pd.to_datetime(sorted(pools)),columns=panel['adj'].columns)
    for date,ss in pools.items():
        names=[actual(s) for s in ss]
        if len(set(names))!=32:raise ValueError('Invalid monthly Top32 archive')
        member.loc[pd.Timestamp(date),names]=True
    panel['member']=member
    dates=[sectors.index[-1],panel['adj'].index[-1],*manifest_dates]
    info=dict(data_as_of=str(min(dates).date()),price_panel_end=str(panel['adj'].index[-1].date()),membership_end=str(member.index[-1].date()),membership_frequency='monthend archive only; no forward fill',source_hashes=source)
    # Reindex only columns. Never create a fresh date in membership by filling old rows.
    cols=panel['adj'].columns
    for k in ['open','close','high','low','volume']:panel[k]=panel[k].reindex(index=panel['adj'].index,columns=cols)
    return panel,info

def normalized_prices(panel,anchors,start,required):
    adj=panel['adj'];close=panel['close'];out_close={};out_open={};anchors=dict(anchors)
    for s in sorted(required):
        if s not in anchors:
            hist=adj.loc[:start,s].dropna()
            if hist.empty:hist=adj[s].dropna()
            if hist.empty:raise ValueError('No normalization anchor:'+s)
            d=hist.index[-1] if hist.index[-1]<=pd.Timestamp(start) else hist.index[0]
            raw=float(close.at[d,s])
            if not np.isfinite(raw) or raw<=0:raise ValueError('Invalid anchor price:'+s)
            anchors[s]=dict(date=str(d.date()),reference_price=raw)
        anchor=anchors[s];base=float(adj.at[pd.Timestamp(anchor['date']),s])
        if not np.isfinite(base) or base<=0:raise ValueError('Lost normalization anchor:'+s)
        out_close[s]=adj[s]/base*anchor['reference_price']
        out_open[s]=out_close[s]*panel['open'][s]/close[s]
    return pd.DataFrame(out_close,index=adj.index),pd.DataFrame(out_open,index=adj.index),anchors
