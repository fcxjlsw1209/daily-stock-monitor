import unittest, tempfile, json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import observe as o
from unittest.mock import patch


class ObserveTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.db=o.connect(Path(self.tmp.name)/'ledger.sqlite')
 def tearDown(self):self.db.close();self.tmp.cleanup()
 def test_month_end_exchange_calendar(self):
  latest,target=o.schedule(datetime(2026,10,8,19,tzinfo=timezone.utc))
  self.assertEqual(str(latest.date()),'2026-10-07');self.assertEqual(str(target.date()),'2026-10-30')
 def test_plan_is_immutable(self):
  d=dict(session='2026-10-30',signal_date='2026-10-29',buys=[],sells=[])
  now=datetime(2026,10,29,23,tzinfo=timezone.utc)
  a=o.save_plan(self.db,d,now,{});d['buys']=[{'symbol':'DO_NOT_REPLACE'}]
  b=o.save_plan(self.db,d,now,{})
  self.assertEqual(a,b);self.assertFalse(b['decision']['buys'])
 def test_late_plan_rejected(self):
  d=dict(session='2026-10-30',signal_date='2026-10-29',buys=[],sells=[])
  with self.assertRaises(ValueError):o.save_plan(self.db,d,datetime(2026,10,30,20,tzinfo=timezone.utc),{})
  self.assertEqual(self.db.execute('SELECT count(*) FROM plans').fetchone()[0],0)
 def test_unfinished_signal_session_rejected(self):
  d=dict(session='2026-10-30',signal_date='2026-10-29',buys=[],sells=[])
  with self.assertRaises(ValueError):o.save_plan(self.db,d,datetime(2026,10,29,19,tzinfo=timezone.utc),{})
 def test_repeated_rows_not_duplicated_or_rewritten(self):
  o.immutable(self.db,'daily','x',dict(equity=100000));o.immutable(self.db,'daily','x',dict(equity=100000))
  self.assertEqual(self.db.execute('SELECT count(*) FROM daily').fetchone()[0],1)
  with self.assertRaises(ValueError):o.immutable(self.db,'daily','x',dict(equity=100001))
 def test_unplanned_month_end_has_no_trade(self):
  dates=pd.DatetimeIndex(['2026-10-29','2026-10-30']);plan=o.replay_planner({})
  d=plan('baseline',dict(equity=100000),dates[-1],dates,None,None,None,None,None,None,None,None)
  self.assertFalse(d['buys']);self.assertFalse(d['sells']);self.assertNotIn('target_weights',d)
 def test_full_plan_fill_and_idempotent_replay(self):
  root=Path(self.tmp.name);u=root/'inputs';(u/'alias_resolved_inputs').mkdir(parents=True)
  (root/'pool').mkdir();(root/'pool/POOL_AUDIT.json').write_text('{}')
  manifest=u/'alias_resolved_inputs/MANIFEST.json';manifest.write_text(json.dumps({'end':'2026-10-29'}))
  dates=o.CAL.sessions_in_range('2026-09-01','2026-10-30');cols=['A','B','C']+o.ETFS
  close=pd.DataFrame(100.,index=dates,columns=cols);close.loc['2026-10-30','A']=125.
  sectors=pd.DataFrame('fund',index=dates,columns=cols)
  for s in ['A','B','C']:sectors[s]=s
  data={k:close.copy() for k in ['close','adj','opening','high','low','volume','sector60']}
  data.update(eligible=close.gt(0),dividends=close*0,splits=close*0+1,raw_per_signal=close*0+1,
   sectors=sectors,issuers={s:s for s in cols},market_close=close['SPY'],market_dividends=close['SPY']*0,
   signal_basis='split_adjusted',execution_basis='raw_with_explicit_actions')
  score=close*0+.1;score['A']=.9;score['B']=.8;score['C']=.7
  mf=dict(base=score,valid=close.gt(0),positive=close.gt(0),atr=close*0+1,ma5=close)
  config=dict(start='2026-10-08',strategy_id='test',registered_utc='2026-10-08T19:00:00+00:00')
  def init():return config,o.connect(root/'full.sqlite')
  with patch.object(o,'P',root),patch.object(o.r,'U',u),patch.object(o,'initialize',init),patch.object(o.r,'prepare',lambda:({},data.copy())),patch.object(o.r,'apply',lambda m,d,p:d),patch.object(o.r,'features',lambda d:({},dict(mf))):
   first=o.run(datetime(2026,10,29,23,tzinfo=timezone.utc))
   self.assertEqual(first['plan']['decision']['session'],'2026-10-30');self.assertFalse(first['orders'])
   manifest.write_text(json.dumps({'end':'2026-10-30'}))
   second=o.run(datetime(2026,10,30,21,tzinfo=timezone.utc))
   self.assertEqual(len(second['orders']),3)
   a=next(v for v in second['orders'] if v['symbol']=='A');self.assertEqual(a['price'],125.)
   third=o.run(datetime(2026,10,30,22,tzinfo=timezone.utc))
   self.assertFalse(third['orders']);self.assertEqual(second['state'],third['state'])


if __name__=='__main__':unittest.main()
