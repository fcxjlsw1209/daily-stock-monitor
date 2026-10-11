"""Read-only, retrospective QQQ total-return reference; never touches ledgers."""
import hashlib,json
import numpy as np
import pandas as pd

def calculate(frame,start,end):
    bars=frame.loc[start:end].copy()
    if bars.empty or str(bars.index[0].date())!=start or str(bars.index[-1].date())!=end:
        raise ValueError('QQQ benchmark endpoints missing')
    values=bars[['Open','Close','Adj Close']].to_numpy()
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('Invalid QQQ benchmark prices')
    first=bars.iloc[0]
    adjusted_open=first['Open']*first['Adj Close']/first['Close']
    # Adj Close ratios approximate dividend reinvestment; exclude entry-day dividend.
    curve=bars['Adj Close']/adjusted_open/1.0005
    peak=np.maximum.accumulate(np.r_[1.,curve.to_numpy()])
    drawdown=np.r_[1.,curve.to_numpy()]/peak-1
    return {'start':start,'as_of':end,'return':float(curve.iloc[-1]-1),
            'equity':float(100000*curve.iloc[-1]),'max_drawdown':float(drawdown.min()),
            'entry_open':float(first['Open']),'last_close':float(bars.iloc[-1]['Close'])}

def summarize(snapshot,end):
    manifest=json.loads((snapshot/'manifest.json').read_text())
    path=snapshot/'QQQ.csv.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['hashes']['QQQ']:
        raise ValueError('QQQ snapshot hash mismatch')
    frame=pd.read_csv(path,index_col=0,parse_dates=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Invalid QQQ date order')
    return {'symbol':'QQQ','snapshot_id':snapshot.name,
            'method':'Retrospective reference: $100000 fully invested at start-session open, 0.05% entry cost, no sale cost while held; adjusted-close dividend-reinvestment approximation. Not a paper ledger or volatility-matched portfolio.',
            'periods':{start:calculate(frame,start,end) for start in ['2026-09-22','2026-09-28','2026-10-05'] if start<=end}}
