"""Cache only the extra prices needed by the widened historical research pool."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import yfinance as yf

P=Path(__file__).resolve().parent

def main():
    d=P/'prices';d.mkdir(exist_ok=True)
    limit=d/'RATE_LIMIT.json'
    today=datetime.now(timezone.utc).date().isoformat()
    if limit.exists() and json.loads(limit.read_text())['day']==today:
        raise RuntimeError('Source already rate limited today; no more requests')
    for symbol in ['NVS','HSBC','BHP','SMFG','SNDK']:
        f=d/(symbol+'.csv')
        if f.exists():
            print('REUSE',symbol,flush=True);continue
        try:
            z=yf.Ticker(symbol).history(start='2018-01-01',end='2026-09-25',auto_adjust=False,actions=True,raise_errors=True)
            if z.empty or 'Adj Close' not in z:raise ValueError('Empty or incomplete response')
            z.index=z.index.tz_localize(None).normalize()
            assert z.index.is_unique and z.index.is_monotonic_increasing
            assert (z['Adj Close'].dropna()>0).all()
            z.to_csv(f)
            meta={'symbol':symbol,'source':'Yahoo via yfinance history','start_requested':'2018-01-01','end_exclusive':'2026-09-25','retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'first':str(z.index[0].date()),'last':str(z.index[-1].date()),'rows':len(z)}
            f.with_suffix('.source.json').write_text(json.dumps(meta,indent=2)+'\n')
            print('SAVED',symbol,len(z),flush=True)
        except Exception as exc:
            if '429' in str(exc) or 'RateLimit' in type(exc).__name__:
                limit.write_text(json.dumps({'day':today,'source':'Yahoo','symbol':symbol,'type':type(exc).__name__})+'\n')
                raise
            print('FAILED',symbol,type(exc).__name__,str(exc)[:160],flush=True)

if __name__=='__main__':main()
