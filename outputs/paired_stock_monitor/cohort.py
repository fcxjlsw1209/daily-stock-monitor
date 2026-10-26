"""Read-only common-window comparison. Do not blend pre-start wins into cohort."""
import json,sqlite3
from execution_registry import active_account_paths

REPLAY_STATUSES = frozenset({
 'user_authorized_retrospective_executed',
 'authorized_causal_replay_executed',
})

def summarize_cohort(accounts,cutoff):
 configs={n:json.loads((p/'config.json').read_text()) for n,p in accounts.items()}
 start=max(c['paper_start'] for c in configs.values());result={};sets=[];replay_sets=[]
 routes={n:active_account_paths(p) for n,p in accounts.items()}
 if len({route['version'] for route in routes.values()})!=1:raise ValueError('Mixed active execution versions in cohort')
 for name,folder in accounts.items():
  route=routes[name]
  db=sqlite3.connect(route['ledger'].resolve().as_uri()+'?mode=ro',uri=True)
  rows=[(d,json.loads(p),s) for d,p,s in db.execute('SELECT date,payload,status FROM sessions ORDER BY date')]
  earlier=[v for d,v,s in rows if d<start];base=earlier[-1]['equity'] if earlier else configs[name]['initial_capital']
  window=[(d,v,s) for d,v,s in rows if start<=d<=cutoff]
  values=[base]+[v['equity'] for d,v,s in window];peak=base;dd=0.
  for value in values:peak=max(peak,value);dd=min(dd,value/peak-1)
  trades=[json.loads(x[0]) for x in db.execute('SELECT data FROM trades')];db.close()
  closed=[t for t in trades if start<=t['entry_date'] and t['exit_date']<=cutoff]
  inherited=[t for t in trades if t['entry_date']<start<=t['exit_date']<=cutoff]
  complete={d for d,v,s in window if s=='recorded_plan_executed'};sets.append(complete)
  replayed={d for d,v,s in window if s in REPLAY_STATUSES};replay_sets.append(replayed)
  result[name]={'base_equity':base,'return':values[-1]/base-1,'max_drawdown':dd,'days':len(window),'closed_trades':len(closed),'win_rate':sum(t['net_pnl']>0 for t in closed)/len(closed) if closed else None,'mean_trade_return':sum(t['net_return'] for t in closed)/len(closed) if closed else None,'average_exposure':sum(v['exposure'] for d,v,s in window)/len(window) if window else None,'inherited_closed_trades':len(inherited),'replayed_days':len(replayed),'replayed_days_by_status':{status:sum(s==status for d,v,s in window) for status in sorted(REPLAY_STATUSES)},'execution_version':route['version'],'ledger_path':str(route['ledger'])}
 common=len(set.intersection(*sets)) if sets else 0
 common_replayed=len(set.intersection(*replay_sets)) if replay_sets else 0
 return {'start':start,'as_of':cutoff,'common_complete_days':common,'common_replayed_days':common_replayed,'accounts':result,'initial_review_ready':common>=60 and all(x['closed_trades']>=100 for x in result.values()),'note':'Returns include inherited positions; win rates include only entries on/after common start. Not equal-start portfolios. Authorized replay days are separate from prospective complete days.'}
