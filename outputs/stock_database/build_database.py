#!/usr/bin/env python3
"""Fixed-universe Yahoo daily history -> SQLite. Re-running refreshes all adjusted history."""
import argparse, concurrent.futures, csv, hashlib, json, math, sqlite3, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import exchange_calendars as xc
import yfinance as yf

ROOT = Path(__file__).resolve().parent
# NYSE/Nasdaq exceptional closure missing from older exchange_calendars releases.
# https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2024-87
EXTRA_CLOSURES = {'2025-01-09'}
SCHEMA = '''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS companies(
 symbol TEXT PRIMARY KEY, source_symbol TEXT NOT NULL, company_name TEXT NOT NULL UNIQUE,
 market_cap_rank INTEGER NOT NULL UNIQUE, market_cap_usd_approx REAL NOT NULL,
 market_cap_display TEXT NOT NULL, ranking_retrieved_at_utc TEXT NOT NULL,
 ranking_source_url TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS daily_prices(
 symbol TEXT NOT NULL REFERENCES companies(symbol), trade_date TEXT NOT NULL,
 open REAL, high REAL, low REAL, close REAL NOT NULL CHECK(close>0), adj_close REAL,
 volume INTEGER, dividends REAL, stock_splits REAL,
 currency TEXT NOT NULL, source TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL,
 PRIMARY KEY(symbol,trade_date));
CREATE INDEX IF NOT EXISTS prices_by_date ON daily_prices(trade_date,symbol);
CREATE TABLE IF NOT EXISTS fetch_log(
 run_id TEXT NOT NULL,symbol TEXT NOT NULL,status TEXT NOT NULL,row_count INTEGER,
 first_date TEXT,last_date TEXT,message TEXT, metadata_json TEXT,
 PRIMARY KEY(run_id,symbol));
CREATE TABLE IF NOT EXISTS quality_issues(
 symbol TEXT NOT NULL,trade_date TEXT NOT NULL,reason TEXT NOT NULL,
 PRIMARY KEY(symbol,trade_date,reason));
CREATE VIEW IF NOT EXISTS screened_closing_prices AS
 SELECT p.* FROM daily_prices p WHERE NOT EXISTS
 (SELECT 1 FROM quality_issues q WHERE q.symbol=p.symbol AND q.trade_date=p.trade_date);
CREATE VIEW IF NOT EXISTS closing_prices AS
 SELECT c.market_cap_rank,c.company_name,p.symbol,p.trade_date,p.close,p.adj_close,p.currency
 FROM daily_prices p JOIN companies c USING(symbol);
CREATE VIEW IF NOT EXISTS latest_prices AS
 SELECT p.* FROM daily_prices p WHERE p.trade_date=(SELECT max(q.trade_date) FROM daily_prices q WHERE q.symbol=p.symbol);
'''

def fetch(symbol, start, end, cutoff):
    errors=[]
    for attempt in range(3):
        try:
            ticker=yf.Ticker(symbol)
            h=ticker.history(start=start,end=end,interval='1d',auto_adjust=False,
                back_adjust=False,actions=True,repair=False,raise_errors=True,timeout=30)
            m=ticker.get_history_metadata()
            if h.empty: raise ValueError('No history returned')
            if m.get('currency')!='USD': raise ValueError('Unexpected currency: '+str(m.get('currency')))
            if m.get('instrumentType')!='EQUITY': raise ValueError('Not EQUITY: '+str(m.get('instrumentType')))
            # Exchange-local date; discard incomplete trading days and empty quote rows.
            h=h.copy(); h['trade_date']=[x.date().isoformat() for x in h.index]
            h=h[(h.trade_date>=start)&(h.trade_date<=cutoff)&h.Close.notna()]
            if h.empty: raise ValueError('No completed daily bars')
            if h.trade_date.duplicated().any(): raise ValueError('Duplicate session dates')
            if not all(math.isfinite(float(v)) and float(v)>0 for v in h.Close):
                raise ValueError('Invalid closing prices')
            if h['Adj Close'].isna().any() or (h['Adj Close']<=0).any():
                raise ValueError('Invalid adjusted close')
            return h,m
        except Exception as exc:
            errors.append(str(exc))
            if attempt<2: time.sleep(2*(attempt+1))
    raise RuntimeError(' | '.join(errors))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db',type=Path,default=ROOT/'stocks.sqlite')
    ap.add_argument('--as-of',default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--validate-only',action='store_true',help='Validate stored data without downloading')
    args=ap.parse_args()
    start='2010-01-01'; now=datetime.now(timezone.utc); run=now.isoformat()
    cal=xc.get_calendar('XNYS',start=start,end=str(int(args.as_of[:4])+1)+'-12-31')
    sched=cal.schedule
    sched=sched[~sched.index.strftime('%Y-%m-%d').isin(EXTRA_CLOSURES)]
    cutoff_rows=sched[(sched.index.strftime('%Y-%m-%d')<=args.as_of)&(sched['close']<=pd.Timestamp(now)-pd.Timedelta(minutes=30))]
    cutoff=cutoff_rows.index[-1].date().isoformat()
    end=(datetime.fromisoformat(cutoff)+timedelta(days=1)).date().isoformat()
    companies=list(csv.DictReader((ROOT/'companies.csv').open()))
    assert len(companies)==100 and len({c['company_name'] for c in companies})==100
    db=sqlite3.connect(args.db); db.executescript(SCHEMA)
    for c in companies:
        db.execute('INSERT OR IGNORE INTO companies VALUES(?,?,?,?,?,?,?,?)',
          tuple(c[k] for k in ['symbol','source_symbol','company_name','market_cap_rank','market_cap_usd_approx','market_cap_display','ranking_retrieved_at_utc','ranking_source_url']))
    if not args.validate_only:
        db.executemany('INSERT OR REPLACE INTO metadata VALUES(?,?)',[
      ('requested_start',start),('requested_as_of',args.as_of),('last_attempt_utc',run),
      ('target_last_completed_session',cutoff),('universe','Fixed current top 100 US-listed companies; foreign issuers included; one share class per company'),
      ('price_semantics','Yahoo Close is split-adjusted, not dividend-adjusted; Adj Close incorporates split/dividend adjustments. USD per US-listed share/ADR. Not raw historical tape prices.'),
      ('history_semantics','Yahoo ticker history may include predecessor securities. No synthetic pre-listing prices or forward filling. Current-universe selection has survivorship bias.'),
      ('yfinance_version',yf.__version__)])
    db.commit(); failures=[]
    if args.validate_only:
        run=db.execute("SELECT value FROM metadata WHERE key='last_attempt_utc'").fetchone()[0]
        cutoff=db.execute("SELECT value FROM metadata WHERE key='target_last_completed_session'").fetchone()[0]
        failures=[r[0] for r in db.execute("SELECT symbol FROM fetch_log WHERE run_id=? AND status='error'",(run,))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch,c['symbol'],start,end,cutoff):c for c in ([] if args.validate_only else companies)}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            c=futures[f]; symbol=c['symbol']
            try:
                h,m=f.result(); fetched=datetime.now(timezone.utc).isoformat()
                def val(row,key):
                    v=row.get(key); return None if pd.isna(v) else float(v)
                rows=[(symbol,r.trade_date,val(r,'Open'),val(r,'High'),val(r,'Low'),val(r,'Close'),val(r,'Adj Close'),
                    None if pd.isna(r.Volume) else int(r.Volume),val(r,'Dividends'),val(r,'Stock Splits'),'USD','Yahoo Finance',fetched) for _,r in h.iterrows()]
                with db:
                    db.execute('DELETE FROM daily_prices WHERE symbol=?',(symbol,))
                    db.executemany('INSERT INTO daily_prices VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
                    db.execute('INSERT INTO fetch_log VALUES(?,?,?,?,?,?,?,?)',(run,symbol,'ok',len(rows),h.trade_date.min(),h.trade_date.max(),'',json.dumps(m,default=str)))
                print(f'{i}/100 {symbol}: {len(rows)} rows {h.trade_date.min()}..{h.trade_date.max()}',flush=True)
            except Exception as exc:
                failures.append(symbol)
                with db: db.execute('INSERT INTO fetch_log VALUES(?,?,?,?,?,?,?,?)',(run,symbol,'error',0,None,None,str(exc),'{}'))
                print(f'{i}/100 {symbol}: ERROR {exc}',flush=True)
    coverage=[]; missing=[]; invalid=[]
    for c in companies:
        symbol=c['symbol']; h=pd.read_sql_query('SELECT * FROM daily_prices WHERE symbol=? ORDER BY trade_date',db,params=(symbol,))
        if h.empty: coverage.append({'symbol':symbol,'rows':0,'status':'no_data'});continue
        expected={d.date().isoformat() for d in cal.sessions_in_range(h.trade_date.min(),cutoff)}-EXTRA_CLOSURES
        dates=set(h.trade_date); gaps=sorted(expected-dates)
        for d in gaps: missing.append({'symbol':symbol,'trade_date':d})
        extra=dates-expected
        for d in extra: invalid.append({'symbol':symbol,'trade_date':d,'reason':'non-session'})
        bad=h[(h.low>h.high)|(h.close<h.low-0.01)|(h.close>h.high+0.01)|(h.volume<0)]
        for _,r in bad.iterrows(): invalid.append({'symbol':symbol,'trade_date':r.trade_date,'reason':'OHLC/volume'})
        coverage.append({'symbol':symbol,'company_name':c['company_name'],'rows':len(h),'first_date':h.trade_date.min(),
          'last_date':h.trade_date.max(),'missing_sessions_since_first_observation':len(gaps),'status':'review' if gaps or extra or len(bad) else 'ok'})
    pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False)
    pd.DataFrame(missing,columns=['symbol','trade_date']).to_csv(ROOT/'missing_sessions.csv',index=False)
    with db:
        db.execute('DELETE FROM quality_issues')
        db.executemany('INSERT INTO quality_issues VALUES(?,?,?)',[(r['symbol'],r['trade_date'],r['reason']) for r in invalid]+[(r['symbol'],r['trade_date'],'missing_session') for r in missing])
    report={'run_id':run,'target_last_session':cutoff,'company_count':db.execute('SELECT count(*) FROM companies').fetchone()[0],
      'price_rows':db.execute('SELECT count(*) FROM daily_prices').fetchone()[0],
      'symbols_with_prices':db.execute('SELECT count(DISTINCT symbol) FROM daily_prices').fetchone()[0],
      'fetch_failures':failures,'missing_session_count':len(missing),'source_anomalies':invalid,
      'symbols_at_target_date':db.execute('SELECT count(*) FROM daily_prices WHERE trade_date=?',(cutoff,)).fetchone()[0],
      'integrity_check':db.execute('PRAGMA integrity_check').fetchone()[0],
      'foreign_key_errors':db.execute('PRAGMA foreign_key_check').fetchall()}
    if not failures:
        db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('last_successful_refresh_utc',run));db.commit()
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2))
    db.close();print(json.dumps(report,indent=2))
    if failures or report['integrity_check']!='ok' or report['foreign_key_errors']: raise SystemExit(1)
    if invalid or missing: print('WARNING: source data issues retained and flagged; see quality_issues and validation.json')

if __name__=='__main__': main()
