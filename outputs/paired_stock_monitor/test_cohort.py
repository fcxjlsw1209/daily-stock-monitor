import json, sqlite3, tempfile, unittest
from pathlib import Path
from cohort import summarize_cohort
import run_pair

class CohortTests(unittest.TestCase):
 def test_common_window_excludes_inherited_wins_and_missed_days(self):
  with tempfile.TemporaryDirectory() as tmp:
   accounts={}
   for name,start in [('baseline','2026-09-22'),('shadow','2026-09-22'),('resilience','2026-09-28')]:
    p=Path(tmp)/name;p.mkdir();accounts[name]=p
    (p/'config.json').write_text(json.dumps({'paper_start':start,'initial_capital':100000}))
    db=sqlite3.connect(p/'paper_ledger.sqlite');db.executescript('CREATE TABLE sessions(date TEXT,payload TEXT,status TEXT); CREATE TABLE trades(data TEXT);')
    for date,equity,status in [('2026-09-25',110000,'recorded_plan_executed'),('2026-09-28',111100,'recorded_plan_executed'),('2026-09-29',108900,'missed')]:
     if date>=start:db.execute('INSERT INTO sessions VALUES(?,?,?)',(date,json.dumps({'equity':equity,'exposure':.3}),status))
    for entry,pnl in [('2026-09-24',100),('2026-09-28',-10)]:
     db.execute('INSERT INTO trades VALUES(?)',(json.dumps({'entry_date':entry,'exit_date':'2026-09-29','net_pnl':pnl,'net_return':pnl/10000}),))
    db.commit();db.close()
   r=summarize_cohort(accounts,'2026-09-29');a=r['accounts']['baseline']
   self.assertEqual(r['common_complete_days'],1);self.assertFalse(r['initial_review_ready'])
   self.assertAlmostEqual(a['return'],-.01);self.assertEqual(a['closed_trades'],1)
   self.assertEqual(a['win_rate'],0);self.assertEqual(a['inherited_closed_trades'],1)
 def test_third_snapshot_mismatch_rejected(self):
  reports={n:{'data_as_of':'2026-09-25','snapshot_id':'same','phase':'close'} for n in run_pair.ACCOUNTS}
  reports['resilience']['snapshot_id']='different'
  with self.assertRaisesRegex(RuntimeError,'mismatch'):run_pair.publish(reports)
