"""Extend the already chosen m_base_positive unchanged; no parameter selection."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, functools
import numpy as np
import pandas as pd
import research as r
P=Path(__file__).resolve().parent
OUT=P/'monthly_2010'

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(daily):
 idx=pd.Index([v['date'] for v in daily]);start=idx[0];end=idx[-1]
 intervals={'full':(start,end),'2010_2018':(start,'2018-12-31'),'2019_onward':('2019-01-01',end),'excluding_last252':(start,idx[-253]),'last252':(idx[-252],end)}
 result={name:dict(start=a,end=b,**r.period_metrics(daily,a,b)) for name,(a,b) in intervals.items()}
 result['annual']={str(y):r.period_metrics(daily,f'{y}-01-01',f'{y}-12-31') for y in range(int(start[:4]),int(end[:4])+1)}
 return result

def run():
 OUT.mkdir(exist_ok=True)
 cfg=json.loads((P/'CONFIG.json').read_text());assert next(c for c in cfg['monthly'] if c['name']=='m_base_positive')=={'name':'m_base_positive','mode':'base','gate':'positive'}
 frozen={str(path.relative_to(P.parent.parent)):sha(path) for path in [P/'CONFIG.json',P/'research.py',r.F/'factors.py',r.F/'vendor_pool.py',r.U/'alias_resolved_inputs/MANIFEST.json']}
 config=dict(strategy='m_base_positive',start='2010-01-04',end='2026-10-06',initial=100000,roundtrip_cost_bps=[10,30],selection='None; reuse original development-selected monthly recipe unchanged',execution=cfg['monthly_execution'],model='No Ridge/tree: selected base monthly score is fixed weighted ranks',source_sha256=frozen,strict_historical_admission=False,original_accounts_modified=False,created_at=datetime.now(timezone.utc).isoformat())
 (OUT/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
 m,d=r.prepare();assert m['end']==config['end'];d=r.apply(m,d,OUT/'pool');_,mf=r.features(d)
 dates=d['close'].index;monthends=set(dates[:-1][dates[:-1].to_period('M')!=dates[1:].to_period('M')]);md=dict(d);md['opening']=d['close']
 ff=dict(score=mf['base'],mask=mf['valid']&mf['positive'],atr=mf['atr'],ma5=mf['ma5'])
 r.h.plan=r.monthly_plan(monthends)
 # Independent restart reproduces original2019 run: verifies scope change only.
 r.h.execute=r.execute_equal_close
 repeat=r.h.run_account('baseline',md,'2019-01-02',m['end'],{'baseline':ff})
 original=json.loads((P/'m_base_positive.json').read_text());assert repeat['status']=='complete'
 assert [x['date'] for x in repeat['daily']]==[x['date'] for x in original['daily']]
 np.testing.assert_allclose([x['equity'] for x in repeat['daily']],[x['equity'] for x in original['daily']],rtol=1e-12,atol=1e-7)
 assert repeat['orders']==original['orders']
 print('Original2019 run reproduced: identical orders and NAV',flush=True)
 results={};checks={};status={}
 for cost in config['roundtrip_cost_bps']:
  r.h.execute=functools.partial(r.execute_equal_close,cost_bps=cost);ot=r.h.terminal_baskets.apply;os=r.h.spin_distributions.apply
  try:
   r.h.terminal_baskets.apply=functools.partial(ot,cost_bps=cost);r.h.spin_distributions.apply=functools.partial(os,cost_bps=cost)
   output=r.h.run_account('baseline',md,config['start'],m['end'],{'baseline':ff})
  finally:r.h.terminal_baskets.apply=ot;r.h.spin_distributions.apply=os
  (OUT/f'cost{cost}.json').write_text(json.dumps(output,allow_nan=False)+'\n');status[str(cost)]={k:output.get(k) for k in ['status','blocked_session','error','performance_publishable']}
  (OUT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print('Run',cost,json.dumps(status[str(cost)]),flush=True)
  if output['status']!='complete':continue
  assert all(pd.Timestamp(o['date']) in monthends for o in output['orders'])
  assert all(o['signal_date']<o['session'] for o in output['decisions'])
  assert all(x['cash']>=-1e-5 for x in output['daily'])
  for o in output['orders']:
   px=d['close'].at[pd.Timestamp(o['date']),o['symbol']]*d['raw_per_signal'].at[pd.Timestamp(o['date']),o['symbol']]
   assert np.isclose(o['price'],px,rtol=1e-10),o
  results[str(cost)]=dict(metrics=output['metrics'],periods=summarize(output['daily']))
  checks[str(cost)]=dict(orders_only_month_end=True,signals_before_fills=True,prices_match_raw_close=True,nonnegative_cash=True,first_fill=output['orders'][0]['date'],orders=len(output['orders']),corporate_actions=len(output['actions']),cancellations=len(output['cancellations']))
  print('Metrics',cost,json.dumps(results[str(cost)]),flush=True)
 # Fully invested QQQ, first regular open, 5bps buy fee, adjusted-price reinvestment proxy.
 runidx=dates[(dates>=config['start'])&(dates<=m['end'])];qadj=d['adj']['QQQ'];adjusted_open=d['opening'].at[runidx[0],'QQQ']*qadj.at[runidx[0]]/d['close'].at[runidx[0],'QQQ']
 qrows=[dict(date=str(day.date()),equity=float(100000/(1+.0005)*qadj.at[day]/adjusted_open)) for day in runidx]
 assert all(np.isfinite(x['equity']) and x['equity']>0 for x in qrows)
 coverage=pd.read_csv(OUT/'pool/coverage.csv');coverage=coverage[coverage.date.between(config['start'],m['end'])];coverage['year']=coverage.date.str[:4]
 coverage_summary=coverage.groupby('year').agg(sessions=('date','size'),min_members=('members','min'),min_cap_covered=('covered','min'),min_selected=('selected','min'),mean_cap_covered=('covered','mean')).reset_index().to_dict('records')
 baseline=json.loads((OUT/'cost10.json').read_text());signals=[v['signal_date'] for v in baseline['decisions'] if v['buys']];counts=ff['mask'].sum(axis=1)
 audit=dict(original_2019_reproduced=True,checks=checks,min_qualified_on_buy_signal_dates=int(counts.loc[signals].min()) if signals else None,coverage_by_year=coverage_summary,snapshot_vintage_certified=False,cap_frequency='Monthly/weekly/mixed retrospective snapshots; past-only45day maximumage',strict_historical_admission=False)
 (OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 (OUT/'SUMMARY.json').write_text(json.dumps(dict(strategy='m_base_positive',complete=all(v['status']=='complete' for v in status.values()),costs=results,qqq_buy_hold=dict(assumption='First-session open;5bpsbuyfee;adjustedprice approximates dividends reinvested;no terminal sale',periods=summarize(qrows))),indent=2)+'\n')
 print('DONE',flush=True)
if __name__=='__main__':run()
