"""Independent monthly paper observer. Only pre-saved plans can create trades."""
from pathlib import Path
from datetime import datetime, timezone
import sys, json, hashlib, sqlite3, copy, fcntl
import numpy as np
import pandas as pd
import exchange_calendars as xc

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'strategy_improvements'))
import research as strategy
r=strategy.r
h=r.h
ETFS=list(r.prepare.__globals__['ETFS'])
CAL=xc.get_calendar('XNYS')


def dump(value):
 return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect(path):
 db=sqlite3.connect(path)
 db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);'
  'CREATE TABLE IF NOT EXISTS plans(session TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);'
  'CREATE TABLE IF NOT EXISTS daily(session TEXT PRIMARY KEY,payload TEXT NOT NULL);'
  'CREATE TABLE IF NOT EXISTS fills(id TEXT PRIMARY KEY,payload TEXT NOT NULL);')
 return db


def immutable(db,table,key,payload):
 column='id' if table=='fills' else 'session'
 assert table in ['daily','fills']
 old=db.execute(f'SELECT payload FROM {table} WHERE {column}=?',(key,)).fetchone()
 encoded=dump(payload)
 if old:
  if old[0]!=encoded:raise ValueError(f'Past {table} changed: {key}; preserved existing record')
 else:db.execute(f'INSERT INTO {table} VALUES (?,?)',(key,encoded))


def save_plan(db,decision,now,source):
 day=decision['session'];old=db.execute('SELECT payload FROM plans WHERE session=?',(day,)).fetchone()
 if old:return json.loads(old[0])
 if pd.Timestamp(now)>=CAL.session_close(day)-pd.Timedelta(minutes=1):
  raise ValueError('Missed month-end plan deadline; no retrospective entry')
 if decision['signal_date']!=str(CAL.previous_session(day).date()):raise ValueError('Wrong monthly signal date')
 if pd.Timestamp(now)<CAL.session_close(decision['signal_date'])+pd.Timedelta(minutes=15):
  raise ValueError('Signal session not complete; no early plan')
 payload=dict(decision=copy.deepcopy(decision),origin='prospective_month_end',created_utc=now.isoformat(),source=copy.deepcopy(source))
 db.execute('INSERT INTO plans VALUES (?,?,?)',(day,now.isoformat(),dump(payload)))
 return payload


def replay_planner(plans):
 def plan(name,state,day,dates,close,ma5,atr,mask,score,sectors,reps,issuers):
  key=str(day.date())
  if key in plans:return copy.deepcopy(plans[key]['decision'])
  return dict(session=key,signal_date=str(dates[dates.get_loc(day)-1].date()),sells=[],buys=[],candidate_count=0,equity_basis=state['equity'])
 return plan


def schedule(now):
 today=pd.Timestamp(now).tz_convert('America/New_York').tz_localize(None).normalize()
 sessions=CAL.sessions_in_range(today-pd.Timedelta(days=14),today+pd.Timedelta(days=70))
 complete=[s for s in sessions if CAL.session_close(s)+pd.Timedelta(minutes=15)<=pd.Timestamp(now)]
 latest=complete[-1]
 ends=[s for s in sessions if CAL.next_session(s).month!=s.month and CAL.session_close(s)>pd.Timestamp(now)]
 return latest,ends[0]


def initialize():
 config=json.loads((P/'config.json').read_text())
 freeze=P/'FREEZE.json'
 if not freeze.exists():
  raise ValueError('Missing frozen source manifest; explicit registration required')
 frozen=json.loads(freeze.read_text())
 for name,sha in frozen.items():
  if digest(P.parent.parent/name)!=sha:raise ValueError('Frozen code/config changed: '+name)
 db=connect(P/'paper_ledger.sqlite')
 old=db.execute('SELECT value FROM meta WHERE key=?',('strategy_id',)).fetchone()
 if old and old[0]!=config['strategy_id']:raise ValueError('Ledger strategy mismatch')
 if not old:
  with db:db.execute('INSERT INTO meta VALUES (?,?)',('strategy_id',config['strategy_id']))
 return config,db


def run(now=None):
 now=now or datetime.now(timezone.utc);config,db=initialize();latest,nextend=schedule(now)
 result=dict(strategy_id=config['strategy_id'],registered_utc=config['registered_utc'],generated_utc=now.isoformat(),
  status='REGISTERED_WAITING_CURRENT_PANEL',account_type='independent_forward_paper',data_as_of=None,
  next_rebalance=str(nextend.date()),required_signal_date=str(CAL.previous_session(nextend).date()),
  state=strategy.ENGINE.initial_state(),statistics=None,plan=None,orders=[],missed_rebalances=[],data_certified=False)
 manifest=json.loads((r.U/'alias_resolved_inputs/MANIFEST.json').read_text())
 result['data_as_of']=manifest['end']
 previous=db.execute('SELECT value FROM meta WHERE key=?',('last_result',)).fetchone()
 if previous:
  old=json.loads(previous[0]);result['state']=old['state'];result['statistics']=old['statistics']
  result['valuation_as_of']=old.get('valuation_as_of')
 if latest<pd.Timestamp(config['start']):
  result['status']='WAITING_FIRST_COMPLETE_SESSION'
  result['reason']='Registered during the first session; waiting for its completed daily bar.'
  db.close();return result
 if pd.Timestamp(manifest['end'])<latest:
  result['status']='WAITING_CURRENT_PANEL'
  result['reason']=f'Need complete historical-union / Top32 / ETF panel through {latest.date()}; cached panel ends {manifest["end"]}. No new plan or fill.'
  db.close();return result
 # Only this account uses these research panels; never substitute the old static100 snapshot.
 m,d=r.prepare();d=r.apply(m,d,P/'pool');_,mf=r.features(d)
 for k,v in list(d.items()):
  if isinstance(v,(pd.DataFrame,pd.Series)):d[k]=v.loc[:latest]
 for k,v in list(mf.items()):
  if isinstance(v,(pd.DataFrame,pd.Series)):mf[k]=v.loc[:latest]
 if not d['close'].index[-1]==latest:raise ValueError('Current panel session missing')
 if not d['adj'].loc[latest,ETFS].notna().all():raise ValueError('Current ETF data missing')
 dates=d['close'].index
 if pd.Timestamp(config['start']) not in dates:raise ValueError('Observation start missing from panel')
 plans={day:json.loads(payload) for day,payload in db.execute('SELECT session,payload FROM plans')}
 h.plan=replay_planner(plans);h.execute=strategy.net_execute
 md=dict(d);md['opening']=d['close'];ff=dict(score=mf['base'],mask=mf['valid']&mf['positive'],atr=mf['atr'],ma5=mf['ma5'])
 output=h.run_account('baseline',md,config['start'],str(latest.date()),{'baseline':ff})
 if output['status']!='complete':raise ValueError(f'Monthly replay blocked: {output.get("error")}')
 result.update(status='OBSERVING',state=output['state'],statistics=output['metrics'],valuation_as_of=str(latest.date()))
 monthends=[s for s in dates if s>=pd.Timestamp(config['start']) and CAL.next_session(s).month!=s.month]
 result['missed_rebalances']=[str(s.date()) for s in monthends if str(s.date()) not in plans]
 # Save a new plan only after the exact prior session closes and before target close.
 if latest==CAL.previous_session(nextend):
  future_dates=dates.append(pd.DatetimeIndex([nextend]));sectors=d['sectors'].loc[latest].to_dict()
  reps=list(d['eligible'].columns[d['eligible'].loc[latest]&mf['base'].loc[latest].notna()])
  planner=strategy.monthly_plan('buffer5',{nextend},d['adj'].pct_change(fill_method=None),mf['valid']&False)
  decision=planner('baseline',output['state'],nextend,future_dates,d['close'],mf['ma5'],mf['atr'],ff['mask'],mf['base'],sectors,reps,d['issuers'])
  source=dict(data_as_of=str(latest.date()),manifest_sha256=digest(r.U/'alias_resolved_inputs/MANIFEST.json'),
    pool_audit_sha256=digest(P/'pool/POOL_AUDIT.json'))
 else:decision=None
 # Validate all old rows before committing any new ones. Revised past data is not rewritten.
 with db:
  for row in output['daily']:immutable(db,'daily',row['date'],row)
  old_fills={key for key, in db.execute('SELECT id FROM fills')}
  for i,fill in enumerate(output['orders']):
   key=f'{fill["date"]}:{i}'
   immutable(db,'fills',key,fill)
   if key not in old_fills:result['orders'].append(fill)
  if decision:result['plan']=save_plan(db,decision,now,source)
  elif str(nextend.date()) in plans:result['plan']=plans[str(nextend.date())]
  db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('last_result',dump(result)))
 db.close();return result


def publish(result):
 (P/'STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 stat=result.get('statistics');state=result.get('state',{})
 lines=['月策略观察：差额调仓＋前五名保留',f'状态：{result["status"]}',
  f'注册：{result.get("registered_utc")}；输入行情截至：{result.get("data_as_of")}。',
  f'下次月末调仓：{result.get("next_rebalance")}；所需信号日：{result.get("required_signal_date")}。',
  f'模拟净值：{state.get("equity",100000):,.2f}美元；持仓数：{len(state.get("positions",{}))}。',
  result.get('reason',''),
  '独立前向模拟；不带入研究期持仓或收益。严格历史数据认证未完成，后续验证并不消除此局限。',
  '首次交易须有月末前已冻结计划。缺少当前动态股票池数据时等待，不借用旧六账户的静态池或补造成交。',
  '当前计划：'+json.dumps(result.get('plan'),ensure_ascii=False),
  '累计统计：'+json.dumps(stat,ensure_ascii=False)]
 (P/'STATUS.txt').write_text('\n'.join(lines)+'\n')


def main():
 with (P/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:result=run()
  except Exception as exc:
   result=dict(status='ERROR_NO_NEW_PLAN',reason=str(exc),generated_utc=datetime.now(timezone.utc).isoformat(),plan=None)
   publish(result);print(json.dumps(result,ensure_ascii=False));raise SystemExit(1)
  publish(result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
