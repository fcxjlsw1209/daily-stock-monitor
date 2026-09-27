import unittest
import pandas as pd
from evening_adapter import target_session,plan_mode
class EveningTests(unittest.TestCase):
 def test_weekend_target(self):
  calendar=pd.DataFrame(index=pd.to_datetime(['2026-09-18','2026-09-21','2026-09-22']))
  self.assertEqual(target_session(calendar,'2026-09-18'),pd.Timestamp('2026-09-21'))
 def test_evening_and_late(self):
  opening='2026-09-23T13:30Z'
  self.assertEqual(plan_mode('2026-09-22T20:20Z',opening),'prospective')
  self.assertEqual(plan_mode('2026-09-23T14:00Z',opening),'retrospective')
  self.assertEqual(plan_mode('2026-09-23T13:29Z',opening),'retrospective')

class AuthorizedReconcileTests(unittest.TestCase):
 def test_authorized_late_plan_and_idempotence(self):
  import sys,json
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'daily_stock_monitor'))
  import monitor as m
  from reconciliation_v2 import reconcile
  db=m.connect(':memory:');cfg={'initial_capital':100000.,'paper_start':'2026-09-22','cost_bps':10,'sectors':{'X':'test'}};m.init(db,cfg)
  q={'session':'2026-09-22','origin':'user_authorized_retrospective','buys':[{'symbol':'X','budget':10000,'atr':2}],'sells':[]}
  db.execute('INSERT INTO plans VALUES (?,?,?,?,?)',('2026-09-22','2026-09-22T17:00Z',json.dumps(q),'test','test'));db.commit()
  h=pd.DataFrame({'Open':[100.],'Close':[101.],'Dividends':[0.],'Stock Splits':[0.]},index=pd.to_datetime(['2026-09-22']))
  args=(m,db,cfg,{'X':h},{'as_of':'2026-09-22','snapshot_id':'test'},pd.Timestamp('2026-09-22T21:00Z'))
  with self.assertRaises(RuntimeError):reconcile(*args)
  db.execute('INSERT INTO meta VALUES (?,?)',('authorized_retrospective_2026-09-22','explicit user authorization'));db.commit()
  self.assertEqual(reconcile(*args),1);self.assertEqual(reconcile(*args),0)
  self.assertEqual(db.execute('SELECT count(*) FROM orders').fetchone()[0],1)
  self.assertEqual(db.execute('SELECT status FROM sessions').fetchone()[0],'user_authorized_retrospective_executed')
  db.close()
