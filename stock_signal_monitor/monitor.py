"""Pre-open immutable plans + post-close forward-paper reconciliation. No broker API."""
from pathlib import Path
import argparse,hashlib,json,sqlite3
from datetime import datetime,timezone
import pandas as pd
from .strategy import features
from .engine import initial_state,plan,execute
from .market_data import schedule,completed_session,download,load_snapshot,session_bar,CACHE
from .paths import runtime_dir
P=runtime_dir()
PACKAGE=Path(__file__).resolve().parent


def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)


def fingerprint():
    paths=[P/'config.json']+sorted(PACKAGE.glob('*.py'))
    return hashlib.sha256(''.join(hashlib.sha256(x.read_bytes()).hexdigest() for x in paths).encode()).hexdigest()


def connect(path=None):
    db=sqlite3.connect(path or P/'paper_ledger.sqlite');db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS plans(session TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL,snapshot TEXT NOT NULL,code_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS sessions(date TEXT PRIMARY KEY,payload TEXT NOT NULL,state TEXT NOT NULL,status TEXT NOT NULL,snapshot TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY,data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY,data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS corporate_actions(id INTEGER PRIMARY KEY,session TEXT NOT NULL,data TEXT NOT NULL);''')
    return db


def init(db,cfg):
    found=db.execute("SELECT value FROM meta WHERE key='code_hash'").fetchone();digest=fingerprint()
    if found and found[0]!=digest:raise RuntimeError('Frozen strategy/code changed. Do not rewrite forward ledger; review and create a new version.')
    if not found:
        with db:
            db.execute('INSERT INTO meta VALUES (?,?)',('code_hash',digest))
            db.execute('INSERT INTO meta VALUES (?,?)',('config',dumps(cfg)))
            db.execute('INSERT INTO meta VALUES (?,?)',('initial_state',dumps(initial_state(cfg['initial_capital']))))
            db.execute('INSERT INTO meta VALUES (?,?)',('created_utc',datetime.now(timezone.utc).isoformat()))


def latest_state(db):
    row=db.execute('SELECT state FROM sessions ORDER BY date DESC LIMIT 1').fetchone()
    return json.loads(row[0] if row else db.execute("SELECT value FROM meta WHERE key='initial_state'").fetchone()[0])


def save_plan(db,decision,created,snapshot,market_open):
    created=pd.Timestamp(created);market_open=pd.Timestamp(market_open)
    if created>=market_open-pd.Timedelta(minutes=1):raise RuntimeError('Too late to issue pre-open plan; no retroactive recommendation.')
    old=db.execute('SELECT payload FROM plans WHERE session=?',(decision['session'],)).fetchone()
    if old:
        if json.loads(old[0])!=decision:raise RuntimeError('Existing plan is immutable; refusing revised recommendation.')
        return False
    with db:db.execute('INSERT INTO plans VALUES (?,?,?,?,?)',(decision['session'],created.isoformat(),dumps(decision),str(snapshot),fingerprint()))
    return True


def reconcile(db,cfg,frames,manifest,now):
    state=latest_state(db);s=schedule(now);cutoff=pd.Timestamp(manifest['as_of'])
    days=s.index[(s.index>=cfg['paper_start'])&(s.index<=cutoff)]
    if state['last_date']:days=days[days>pd.Timestamp(state['last_date'])]
    count=0
    for day in days:
        ds=str(day.date());record=db.execute('SELECT created_utc,payload FROM plans WHERE session=?',(ds,)).fetchone()
        if record:
            if pd.Timestamp(record[0])>=s.at[day,'open']:raise RuntimeError('Plan was recorded after open')
            decision=json.loads(record[1]);status='recorded_plan_executed'
        else:
            decision=dict(session=ds,buys=[],sells=[]);status='missed_plan_no_retroactive_trades'
        unit_adjustments=[]
        for buy in decision['buys']:
            if 'reference_close' in buy and 'signal_date' in decision:
                original_close=session_bar(frames[buy['symbol']],decision['signal_date'])['close']
                scale=buy['reference_close']/original_close
                if scale<=0:raise RuntimeError('Invalid plan price basis')
                buy['atr']/=scale
                if abs(scale-1)>1e-8:unit_adjustments.append(dict(symbol=buy['symbol'],plan_atr_unit_scale=scale))
        needed=set(state['positions'])|{r['symbol'] for r in decision['buys']}
        bars={symbol:session_bar(frames[symbol],day) for symbol in needed}
        new,daily,trades,orders,actions=execute(state,decision,bars,s.index,cfg['cost_bps'],cfg['sectors'])
        actions.extend(unit_adjustments)
        with db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',(ds,dumps(daily),dumps(new),status,manifest['snapshot_id']))
            db.executemany('INSERT INTO trades(data) VALUES (?)',[(dumps(t),) for t in trades])
            db.executemany('INSERT INTO orders(data) VALUES (?)',[(dumps(o),) for o in orders])
            db.executemany('INSERT INTO corporate_actions(session,data) VALUES (?,?)',[(ds,dumps(a)) for a in actions])
        state=new;count+=1
    return count


def statistics(db):
    state=latest_state(db);trades=[json.loads(r[0]) for r in db.execute('SELECT data FROM trades ORDER BY id')]
    daily=[json.loads(r[0]) for r in db.execute('SELECT payload FROM sessions ORDER BY date')]
    equity=[state['initial']]+[x['equity'] for x in daily];peak=equity[0];dd=0.
    for value in equity:peak=max(peak,value);dd=min(dd,value/peak-1)
    realized=sum(t['net_pnl'] for t in trades);unrealized=state['equity']-state['initial']-realized
    return dict(mode='forward_paper_only',as_of=state['last_date'],initial_capital=state['initial'],equity=state['equity'],cash=state['cash'],
      cumulative_return=state['equity']/state['initial']-1,closed_trades=len(trades),wins=sum(t['net_pnl']>0 for t in trades),
      win_rate=sum(t['net_pnl']>0 for t in trades)/len(trades) if trades else None,realized_pnl=realized,unrealized_pnl=unrealized,
      max_close_drawdown=dd,open_positions=len(state['positions']),positions=state['positions'],
      missed_sessions=db.execute("SELECT count(*) FROM sessions WHERE status='missed_plan_no_retroactive_trades'").fetchone()[0],
      fees=state['fees'],dividends=state['dividends'])


def exports(db,report):
    rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM sessions ORDER BY date')]
    pd.DataFrame(rows,columns=['date','equity','cash','invested','exposure','positions','return','fees_cumulative','dividends_cumulative']).to_csv(P/'daily_results.csv',index=False)
    trades=[json.loads(r[0]) for r in db.execute('SELECT data FROM trades ORDER BY id')]
    pd.DataFrame(trades,columns=['symbol','entry_date','exit_date','entry_price','exit_price','quantity','entry_outlay','dividends','entry_fee','exit_fee','net_pnl','net_return','sessions','reason']).to_csv(P/'closed_trades.csv',index=False)
    state=latest_state(db)
    pd.DataFrame([dict(symbol=s,**p) for s,p in state['positions'].items()],columns=['symbol','entry_date','entry_price','qty','cost','entry_fee','dividends','atr']).to_csv(P/'open_positions.csv',index=False)
    (P/'status.json').write_text(dumps(report))
    st=report['statistics'];win='尚无已平仓交易' if st['win_rate'] is None else f"{st['win_rate']:.1%}（{st['wins']}/{st['closed_trades']}）"
    text=f"# 每日模拟跟踪\n\n模式：事前记录的模拟账户；不是实际成交或历史回测。\n\n状态：{report['status']}\n\n行情截至：{report.get('data_as_of','未刷新')}；账本截至：{st['as_of'] or '尚未开始'}。\n\n- 初始模拟资金：${st['initial_capital']:,.2f}\n- 当前净值：${st['equity']:,.2f}\n- 累计净收益：{st['cumulative_return']:.2%}\n- 已平仓胜率：{win}\n- 已实现盈亏：${st['realized_pnl']:,.2f}\n- 未实现盈亏（含在持仓计提分红）：${st['unrealized_pnl']:,.2f}\n- 最大日终回撤：{-st['max_close_drawdown']:.2%}\n- 当前持仓：{st['open_positions']}只\n- 错过事前计划的交易日：{st['missed_sessions']}\n"
    if report.get('plan'):
        text+='\n## 下一开盘计划\n\n```json\n'+json.dumps(report['plan'],ensure_ascii=False,indent=2)+'\n```\n'
    (P/'DAILY_STATUS.md').write_text(text)


def run(phase='auto',refresh=True,folder=None):
    now=pd.Timestamp.now(tz='UTC');cfg=json.loads((P/'config.json').read_text());db=connect();init(db,cfg)
    calendar=schedule(now);today=pd.Timestamp(now.tz_convert('America/New_York').date());cutoff=completed_session(now)
    if phase=='auto':phase='morning' if now.tz_convert('America/Los_Angeles').hour<10 else 'close'
    if folder is not None:folder=Path(folder)
    elif refresh:folder=download(cutoff)
    else:folder=Path(json.loads((CACHE/'latest.json').read_text())['folder'])
    data,raw,frames,manifest=load_snapshot(folder)
    if pd.Timestamp(manifest['as_of'])!=cutoff:raise RuntimeError('Stale data; expected completed session '+str(cutoff.date()))
    reconciled=reconcile(db,cfg,frames,manifest,now)
    report=dict(status='updated_paper_ledger',phase=phase,data_as_of=manifest['as_of'],snapshot_id=manifest['snapshot_id'],reconciled_sessions=reconciled,plan=None)
    if today not in calendar.index:report['status']='market_closed'
    elif phase=='morning':
        market_open=calendar.at[today,'open']
        if now>=market_open-pd.Timedelta(minutes=1):report['status']='missed_preopen_window_no_new_plan'
        elif today<pd.Timestamp(cfg['paper_start']):report['status']='paper_start_pending'
        else:
            record=db.execute('SELECT payload FROM plans WHERE session=?',(str(today.date()),)).fetchone()
            if record:decision=json.loads(record[0]);report['status']='existing_immutable_plan'
            else:
                state=latest_state(db)
                for symbol,position in state['positions'].items():
                    prior_open=float(data.opening.at[pd.Timestamp(position['entry_date']),symbol])
                    if abs(prior_open/position['entry_price']-1)>.001:
                        raise RuntimeError('Unprocessed corporate action or historical price revision for '+symbol+'; no new pre-open plan')
                f=features(data,raw);decision=plan(state,today,calendar.index,data.close,f['ma5'],f['atr'],f['masks'][cfg['admission']],f['score'],cfg['sectors'],cfg)
                decision['strategy_version']=cfg['version'];decision['data_as_of']=manifest['as_of']
                decision['cost_bps']=cfg['cost_bps'];decision['fill_model']='next_session_open_proxy_not_actual_broker_fill'
                save_plan(db,decision,pd.Timestamp.now(tz='UTC'),folder,market_open);report['status']='new_preopen_plan'
            report['plan']=decision
    elif phase=='close' and now<calendar.at[today,'close']+pd.Timedelta(minutes=15):report['status']='waiting_for_completed_session'
    report['statistics']=statistics(db);exports(db,report);db.close();return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['auto','morning','close','status'],default='auto');parser.add_argument('--no-refresh',action='store_true');parser.add_argument('--snapshot')
    args=parser.parse_args()
    try:
        if args.phase=='status':
            cfg=json.loads((P/'config.json').read_text());db=connect();init(db,cfg);result={'status':'status_only','statistics':statistics(db)};db.close()
        else:result=run(args.phase,not args.no_refresh,args.snapshot)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except Exception as exc:
        failure={'status':'ERROR_NO_NEW_PLAN','error':str(exc),'at_utc':datetime.now(timezone.utc).isoformat()}
        (P/'last_error.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure));raise SystemExit(1)
