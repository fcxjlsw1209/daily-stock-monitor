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
        source[str(path)]=sha(path);return pd.read_csv(path,index_col=0,parse_dates=True)
    p=O/'index_union/alias_resolved_inputs';panel={k:frame(p/(filename+'.csv.gz')) for k,filename in [('adj','adj'),('open','opening'),('close','close'),('high','high'),('low','low'),('volume','volume')]}
    manifest_files=[O/'four_factor_rebound/ETF_SOURCES.json',O/'monthly_reconstruction_v2/ETF_MANIFEST.json',O/'monthly_reconstruction_v2/foreign/ETF_MANIFEST.json']
    manifest_dates=[]
    for mf in manifest_files:
        source[str(mf)]=sha(mf)
        for entry in read(mf):
            assert sha(entry['file'])==entry['sha256'],'Price source fingerprint mismatch'
            z=frame(Path(entry['file']))
            if entry['symbol'] in etfs or entry['symbol'] in ['TSM','BABA','NVO','SAP','TM']:manifest_dates.append(z.index[-1])
            for k,col in [('adj','Adj Close'),('open','Open'),('close','Close'),('high','High'),('low','Low'),('volume','Volume')]:
                panel[k][entry['symbol']]=z[col].reindex(panel[k].index)
    member=frame(O/'monthly_reconstruction_v2/foreign/membership.csv.gz').fillna(False).astype(bool)
    panel['member']=member
    dates=[member.index[-1],panel['adj'].index[-1],*manifest_dates]
    info=dict(data_as_of=str(min(dates).date()),price_panel_end=str(panel['adj'].index[-1].date()),membership_end=str(member.index[-1].date()),source_hashes=source)
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
