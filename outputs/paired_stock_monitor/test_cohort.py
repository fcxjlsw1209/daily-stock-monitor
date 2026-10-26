import json, sqlite3, tempfile, unittest
from pathlib import Path
from cohort import summarize_cohort
from unittest.mock import patch
import execution_registry
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
   with patch.object(execution_registry,'ACTIVE_EXECUTION',Path(tmp)/'absent_registry.json'):
    r=summarize_cohort(accounts,'2026-09-29')
   a=r['accounts']['baseline']
   self.assertEqual(r['common_complete_days'],1);self.assertFalse(r['initial_review_ready'])
   self.assertAlmostEqual(a['return'],-.01);self.assertEqual(a['closed_trades'],1)
   self.assertEqual(a['win_rate'],0);self.assertEqual(a['inherited_closed_trades'],1)
 def test_third_snapshot_mismatch_rejected(self):
  reports={n:{'data_as_of':'2026-09-25','snapshot_id':'same','phase':'close'} for n in run_pair.ACCOUNTS}
  reports['resilience']['snapshot_id']='different'
  with self.assertRaisesRegex(RuntimeError,'mismatch'):run_pair.publish(reports)

 def test_active_v3_routing_and_replay_counts_preserve_forward_definition(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);accounts={};mapping={}
   for name in execution_registry.ACCOUNT_NAMES:
    mapping[name]={'ledger':'execution_v3/paper_ledger.sqlite','exports':'execution_v3'}
   active=root/'ACTIVE_EXECUTION.json';active.write_text(json.dumps({'version':'causal_replay_v3','accounts':mapping}))
   for key,name in [('baseline','daily_stock_monitor'),('shadow','pressure_shadow_monitor')]:
    p=root/name;p.mkdir();(p/'execution_v3').mkdir();accounts[key]=p
    (p/'config.json').write_text(json.dumps({'paper_start':'2026-10-01','initial_capital':100000}))
    for file,is_v3 in [(p/'paper_ledger.sqlite',False),(p/'execution_v3/paper_ledger.sqlite',True)]:
     db=sqlite3.connect(file);db.executescript('CREATE TABLE sessions(date TEXT,payload TEXT,status TEXT);CREATE TABLE trades(data TEXT);')
     rows=[('2026-10-01',101000,'recorded_plan_executed'),('2026-10-02',102000,'authorized_causal_replay_executed'),('2026-10-05',103000,'user_authorized_retrospective_executed'),('2026-10-06',104000,'missed_plan_no_retroactive_trades')]
     for date,equity,status in rows:
      if not is_v3:equity=90000
      db.execute('INSERT INTO sessions VALUES(?,?,?)',(date,json.dumps({'equity':equity,'exposure':.3}),status))
     db.commit();db.close()
   before={str(p/'paper_ledger.sqlite'):(p/'paper_ledger.sqlite').read_bytes() for p in accounts.values()}
   with patch.object(execution_registry,'ACTIVE_EXECUTION',active):
    result=summarize_cohort(accounts,'2026-10-06')
   self.assertEqual(result['common_complete_days'],1)
   self.assertEqual(result['common_replayed_days'],2)
   self.assertFalse(result['initial_review_ready'])
   for item in result['accounts'].values():
    self.assertAlmostEqual(item['return'],.04)
    self.assertEqual(item['replayed_days'],2)
    self.assertEqual(item['replayed_days_by_status'],{'authorized_causal_replay_executed':1,'user_authorized_retrospective_executed':1})
    self.assertEqual(item['execution_version'],'causal_replay_v3')
    self.assertIn('execution_v3',item['ledger_path'])
   self.assertEqual(before,{str(p/'paper_ledger.sqlite'):(p/'paper_ledger.sqlite').read_bytes() for p in accounts.values()})

 def test_missing_active_ledger_does_not_read_legacy(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'daily_stock_monitor';p.mkdir()
   (p/'config.json').write_text(json.dumps({'paper_start':'2026-10-01','initial_capital':100000}))
   active=root/'ACTIVE_EXECUTION.json';active.write_text(json.dumps({'version':'causal_replay_v3','accounts':{name:{'ledger':'execution_v3/paper_ledger.sqlite','exports':'execution_v3'} for name in execution_registry.ACCOUNT_NAMES}}))
   with patch.object(execution_registry,'ACTIVE_EXECUTION',active),self.assertRaisesRegex(ValueError,'missing'):
    summarize_cohort({'baseline':p},'2026-10-06')

if __name__=='__main__':unittest.main()
