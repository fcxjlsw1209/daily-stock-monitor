"""Read the existing research price caches without network or source mutation."""
import hashlib
import json
from pathlib import Path
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate_frame(df):
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError('Cached prices have duplicate or unordered sessions')
    # No fill, scaling, or substitution of dates. Bad bars stay missing.
    return df.where(df['close'].gt(0) & df['volume'].ge(0))

def load_supplemental(symbol, outputs):
    outputs=Path(outputs)
    folders={'BUD':'monthly_2016_extension/prices','HSBC':'monthly_2016_extension/prices',
             'NVS':'monthly_2016_extension/prices','SHEL':'monthly_published44_reconstruction/prices',
             'SKHY':'monthly_signal_20260930/prices'}
    if symbol in folders:
        p=outputs/folders[symbol]/(symbol+'.csv');meta=p.with_suffix('.source.json')
        if not p.exists() or not meta.exists():return None,{}
        m=json.loads(meta.read_text())
        if m['symbol']!=symbol or m['sha256']!=sha(p):raise ValueError('Supplemental price identity/hash mismatch')
        z=pd.read_csv(p,index_col='Date',parse_dates=['Date'])
        return validate_frame(pd.DataFrame({'close':z['Adj Close'],'volume':z['Volume']})),{str(p):sha(p),str(meta):sha(meta)}
    if symbol in {'TSM','TM','NVO','SAP','BABA'}:
        p=outputs/'monthly_reconstruction_v2/foreign/sources'/(symbol+'.csv');raw=p.with_suffix('.json')
        if not p.exists() or not raw.exists():return None,{}
        j=json.loads(raw.read_text())['chart']['result'][0]
        if j['meta']['symbol']!=symbol:raise ValueError('Foreign chart identity mismatch')
        z=pd.read_csv(p,index_col='Date',parse_dates=['Date'])
        return validate_frame(pd.DataFrame({'close':z['Adj Close'],'volume':z['Volume']})),{str(p):sha(p),str(raw):sha(raw)}
    if symbol=='CHL':
        folder=outputs/'dynamic_top100/additional_sources/tiingo'
        p=folder/'CHL_prices.json';provenance=folder/'CHL_prices.provenance.json';identity=folder/'CHL_metadata.json'
        m=json.loads(provenance.read_text());i=json.loads(identity.read_text())
        if m['sha256']!=sha(p) or i['ticker']!='CHL' or i['name']!='China Mobile Ltd':
            raise ValueError('CHL cached price identity/hash mismatch')
        z=pd.DataFrame(json.loads(p.read_text()));z.index=pd.to_datetime(z['date'],utc=True).dt.tz_localize(None)
        # Match the existing monthly research's pre-2021 US-listed segment.
        # Later vendor bars are not accepted as a continued NYSE listing.
        z=z.loc[z.index<pd.Timestamp('2021-01-01')]
        return validate_frame(pd.DataFrame({'close':z['adjClose'],'volume':z['adjVolume']})),{str(p):sha(p),str(provenance):sha(provenance),str(identity):sha(identity)}
    return None,{}
