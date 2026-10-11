from pathlib import Path
import concurrent.futures,sqlite3,time,json
from datetime import datetime,timezone
import pandas as pd
import yfinance as yf
P=Path(__file__).resolve().parent
DB=P.parents[1]/'stock_database'/'stocks.sqlite'
ETFS=['SPY','XLK','XLC','XLY','XLP','XLF','XLV','XLI','XLE','XLB','XLU','XLRE']

def fetch(sym,etf=False):
    dest=P/'raw';dest.mkdir(exist_ok=True)
    result={'symbol':sym,'fetched_at_utc':datetime.now(timezone.utc).isoformat(),'errors':[]}
    t=yf.Ticker(sym)
    for attempt in range(3):
        try:
            h=t.history(period='60d',interval='5m',auto_adjust=False,prepost=False,timeout=25)
            if h.empty:raise ValueError('empty intraday')
            h.to_csv(dest/f'{sym}_5m.csv.gz',compression='gzip')
            result.update(intraday_rows=len(h),first=str(h.index.min()),last=str(h.index.max()))
            break
        except Exception as e:
            if attempt==2:result['errors'].append('intraday: '+str(e))
            else:time.sleep(2)
    if etf:
        try:
            d=t.history(start='2010-01-01',end='2026-09-19',interval='1d',auto_adjust=False,timeout=25)
            if d.empty:raise ValueError('empty daily')
            d.to_csv(dest/f'{sym}_daily.csv');result['daily_rows']=len(d)
        except Exception as e:result['errors'].append('daily: '+str(e))
    else:
        try:
            info=t.info
            result.update(sector=info.get('sector'),industry=info.get('industry'),exchange=info.get('exchange'))
        except Exception as e:result['errors'].append('sector: '+str(e))
        try:
            ed=t.get_earnings_dates(limit=12)
            if ed is not None and len(ed):
                ed.to_csv(dest/f'{sym}_earnings.csv');result['earnings_rows']=len(ed)
            else:result['errors'].append('earnings: no dates')
        except Exception as e:result['errors'].append('earnings: '+str(e))
    return result

def main():
    with sqlite3.connect(DB) as db:syms=[r[0] for r in db.execute('SELECT symbol FROM companies ORDER BY market_cap_rank')]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(fetch,s,s in ETFS):s for s in syms+ETFS}
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            r=f.result();rows.append(r)
            print(i,len(fs),r['symbol'],r.get('intraday_rows'),r.get('sector'),r['errors'],flush=True)
            (P/'download_manifest.json').write_text(json.dumps(rows,indent=2))
    print('COMPLETE',len(rows))
if __name__=='__main__':main()
