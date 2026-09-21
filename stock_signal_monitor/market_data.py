"""Free-data snapshots. Fail closed on stale or incomplete latest-session data."""
from pathlib import Path
import json,hashlib,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import exchange_calendars as xc
from .paths import runtime_dir
P=runtime_dir()
CACHE=P/'cache'


def schedule(now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:raise ValueError('Timezone required')
    cal=xc.get_calendar('XNYS',start='2010-01-01',end=f'{now.year+2}-12-31')
    return cal.schedule.drop(pd.Timestamp('2025-01-09'),errors='ignore')


def completed_session(now):
    now=pd.Timestamp(now);s=schedule(now)
    return s.index[s['close']+pd.Timedelta(minutes=15)<=now][-1]


def download(cutoff):
    import yfinance as yf
    cfg=json.loads((P/'config.json').read_text());symbols=cfg['symbols']+['SPY','QQQ']+sorted(set(cfg['sector_etfs'].values()))
    symbols=list(dict.fromkeys(symbols));cutoff=pd.Timestamp(cutoff)
    start=str((cutoff-pd.Timedelta(days=600)).date());end=str((cutoff+pd.Timedelta(days=1)).date())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=CACHE/stamp;folder.mkdir(parents=True)
    failures=[];hashes={}
    def fetch(symbol):
        error=None
        for attempt in range(2):
            try:
                h=yf.Ticker(symbol).history(start=start,end=end,interval='1d',auto_adjust=False,actions=True,timeout=25,raise_errors=True)
                if h.empty:raise ValueError('No history')
                h.index=pd.to_datetime(h.index).tz_convert('America/New_York').tz_localize(None).normalize()
                h=h.loc[:cutoff]
                if h.index.duplicated().any() or h.index[-1]!=cutoff:raise ValueError('Stale/duplicate history')
                for field in ['Open','High','Low','Close','Adj Close','Volume']:
                    if field not in h or not np.isfinite(h.iloc[-1][field]) or h.iloc[-1][field]<=0:raise ValueError('Invalid latest '+field)
                if h.Close.iloc[-1]<h.Low.iloc[-1]-1e-6 or h.Close.iloc[-1]>h.High.iloc[-1]+1e-6:raise ValueError('OHLC inconsistency')
                for field in ['Dividends','Stock Splits']:
                    if field not in h:h[field]=0.
                path=folder/(symbol+'.csv.gz');h.to_csv(path,compression='gzip')
                return symbol,hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception as exc:
                error=str(exc)
                if attempt==0:time.sleep(.5)
        raise RuntimeError(symbol+': '+error)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,s):s for s in symbols}
        for future in as_completed(futures):
            try:s,digest=future.result();hashes[s]=digest
            except Exception as exc:failures.append(str(exc))
    manifest={'snapshot_id':stamp,'as_of':str(cutoff.date()),'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'hashes':hashes,'failures':failures}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if failures:raise RuntimeError('Incomplete market snapshot; no new plan: '+'; '.join(failures))
    (CACHE/'latest.json').write_text(json.dumps({'folder':str(folder),'as_of':str(cutoff.date())}))
    return folder


def load_snapshot(folder):
    from .factors import Inputs
    cfg=json.loads((P/'config.json').read_text());folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['failures']:raise ValueError('Incomplete snapshot')
    frames={}
    for s,digest in manifest['hashes'].items():
        path=folder/(s+'.csv.gz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Snapshot modified: '+s)
        frames[s]=pd.read_csv(path,index_col=0,parse_dates=True)
    ix=frames['SPY'].index;stocks=cfg['symbols'];allstocks=stocks+['SPY','QQQ']
    def matrix(field,which):return pd.DataFrame({s:frames[s][field].reindex(ix) for s in which})
    sectors={**cfg['sectors'],'SPY':'ETF','QQQ':'ETF'}
    data=Inputs(matrix('Close',allstocks),matrix('Open',allstocks),matrix('Dividends',allstocks),matrix('Volume',allstocks),sectors,matrix('Close',list(set(cfg['sector_etfs'].values()))))
    raw={'adj_close':matrix('Adj Close',stocks),'high':matrix('High',stocks),'low':matrix('Low',stocks),'sector_adjusted':matrix('Adj Close',list(set(cfg['sector_etfs'].values())))}
    return data,raw,frames,manifest


def session_bar(history,day):
    """Undo future split adjustments for immutable, actual-session share units."""
    day=pd.Timestamp(day);r=history.loc[day]
    later=history.loc[history.index>day,'Stock Splits'];factor=float(later.where(later>0,1.).prod())
    return {'open':float(r.Open)*factor,'close':float(r.Close)*factor,'dividends':float(r.Dividends)*factor,'split':float(r['Stock Splits'])}
