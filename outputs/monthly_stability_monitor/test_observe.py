import copy,json,sqlite3,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import observe,inputs,strategy

UTC=timezone.utc
ETFS=json.loads((observe.P/'config.json').read_text())['etfs']

class LedgerTests(unittest.TestCase):
    def decision(self):
        return dict(signal_date='2026-10-30',session='2026-11-02',target_weights={'A':.5,'B':.5})

    def test_schedule_and_freeze_deadline(self):
        now=datetime(2026,10,30,21,tzinfo=UTC)
        self.assertEqual([str(x.date()) for x in observe.schedule(now)],['2026-10-30','2026-10-30','2026-11-02'])
        db=observe.connect(':memory:')
        with self.assertRaisesRegex(ValueError,'not complete'):
            observe.save_plan(db,self.decision(),datetime(2026,10,30,19,tzinfo=UTC),{},'2026-10-09')
        with self.assertRaisesRegex(ValueError,'deadline'):
            observe.save_plan(db,self.decision(),datetime(2026,11,2,15,tzinfo=UTC),{},'2026-10-09')
        p=observe.save_plan(db,self.decision(),now,{},'2026-10-09')
        changed=self.decision();changed['target_weights']={}
        self.assertEqual(p,observe.save_plan(db,changed,now,{},'2026-10-09'))
        self.assertEqual(db.execute('select count(*) from plans').fetchone()[0],1)
        db.close()

    def test_fills_use_open_and_cost_then_fixed_units(self):
        close=pd.Series({'A':120.,'B':200.});opening=pd.Series({'A':100.,'B':200.})
        s,f,c=observe.day_step(observe.initial(),pd.Timestamp('2026-11-02'),close,opening,self.decision())
        self.assertAlmostEqual(s['positions']['A']['qty'],100000/1.0005/2/100)
        self.assertAlmostEqual(s['fees'],100000-100000/1.0005)
        old=copy.deepcopy(s['positions'])
        nexts,f,c=observe.day_step(s,pd.Timestamp('2026-11-03'),close*1.1,opening,None)
        self.assertEqual(nexts['positions'],old);self.assertEqual(f,[])
        self.assertAlmostEqual(nexts['equity'],s['equity']*1.1)

    def test_cash_exit_realized_and_difference_rebalance(self):
        prices=pd.Series({'A':100.,'B':100.})
        s,_,_=observe.day_step(observe.initial(),pd.Timestamp('2026-11-02'),prices,prices,self.decision())
        s,orders,closed=observe.day_step(s,pd.Timestamp('2026-12-01'),prices,prices,self.decision())
        self.assertEqual(orders,[])
        s,orders,closed=observe.day_step(s,pd.Timestamp('2027-01-04'),prices*1.1,prices*1.1,{'target_weights':{}})
        self.assertEqual(len(closed),2);self.assertEqual(s['positions'],{})
        self.assertAlmostEqual(s['realized_pnl'],s['equity']-100000)

    def test_missing_open_and_single_name_are_rejected(self):
        prices=pd.Series({'A':100.,'B':np.nan})
        with self.assertRaisesRegex(ValueError,'Missing execution'):
            observe.day_step(observe.initial(),pd.Timestamp('2026-11-02'),prices,prices,self.decision())
        with self.assertRaisesRegex(ValueError,'Invalid equal'):
            observe.day_step(observe.initial(),pd.Timestamp('2026-11-02'),prices,prices,{'target_weights':{'A':1.}})

    def test_immutable_rows_reject_revision(self):
        db=observe.connect(':memory:');row={'equity':100000.}
        self.assertTrue(observe.immutable(db,'daily','2026-10-09',row))
        self.assertFalse(observe.immutable(db,'daily','2026-10-09',row))
        with self.assertRaisesRegex(ValueError,'revised'):observe.immutable(db,'daily','2026-10-09',{'equity':99000.})
        self.assertEqual(json.loads(db.execute('select payload from daily').fetchone()[0]),row);db.close()

    def test_normalization_resists_vendor_adjustment_rebase(self):
        ds=pd.to_datetime(['2026-10-09','2026-11-02'])
        panel={'adj':pd.DataFrame({'A':[90.,110.]},index=ds),'close':pd.DataFrame({'A':[100.,110.]},index=ds),'open':pd.DataFrame({'A':[99.,108.]},index=ds)}
        c,o,a=inputs.normalized_prices(panel,{},'2026-10-09',{'A'})
        panel['adj']*=.5
        c2,o2,a2=inputs.normalized_prices(panel,a,'2026-10-09',{'A'})
        pd.testing.assert_frame_equal(c,c2);pd.testing.assert_frame_equal(o,o2);self.assertEqual(a,a2)
        self.assertAlmostEqual(o.iloc[1,0],108/90*100)

    def test_stale_panel_produces_no_plan_or_fills(self):
        cfg=json.loads((observe.P/'config.json').read_text());db=observe.connect(':memory:')
        with patch.object(observe,'initialize',return_value=(cfg,db)),patch.object(observe,'load_panel',return_value=({},dict(data_as_of='2026-09-24',source_hashes={}))),patch.object(observe,'save_plan') as save:
            r=observe.run(datetime(2026,10,9,22,tzinfo=UTC))
        self.assertEqual(r['status'],'WAITING_CURRENT_PANEL');self.assertEqual(r['orders'],[]);self.assertIsNone(r['plan']);save.assert_not_called()

    def test_fingerprint_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'config.json').write_text('{}');(p/'FREEZE.json').write_text(json.dumps({'config.json':'wrong'}))
            with patch.object(observe,'P',p),patch.object(observe,'ROOT',p):
                with self.assertRaisesRegex(ValueError,'Frozen code'):observe.initialize()
            self.assertFalse((p/'paper_ledger.sqlite').exists())

    def test_prospective_plan_fill_and_repeat_lifecycle(self):
        cfg=json.loads((observe.P/'config.json').read_text())
        ds=observe.CAL.sessions_in_range(cfg['start'],'2026-11-03')
        px=pd.DataFrame({'A':100.,'B':200.},index=ds)
        panel={k:px.copy() for k in ['adj','close','open','high','low','volume']}
        panel['member']=px.notna();panel['sectors']=pd.DataFrame('Technology',index=ds,columns=px.columns)
        chosen={'symbols':['A','B'],'scores':{'A':.4,'B':.3},'target_weights':{'A':.5,'B':.5},'candidate_count':2,'universe':['A','B']}
        info={'data_as_of':'2026-11-03','source_hashes':{'synthetic_fixture':'fixture'}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'config.json').write_text(json.dumps(cfg));(p/'FREEZE.json').write_text(json.dumps({'config.json':inputs.sha(p/'config.json')}))
            with patch.object(observe,'P',p),patch.object(observe,'ROOT',p),patch.object(observe,'load_panel',return_value=(panel,info)),patch.object(observe,'selection',return_value=chosen):
                r=observe.run(datetime(2026,10,30,21,tzinfo=UTC))
                self.assertEqual(r['plan']['decision']['session'],'2026-11-02');self.assertEqual(r['orders'],[])
                r=observe.run(datetime(2026,11,2,22,tzinfo=UTC))
                self.assertEqual(len(r['orders']),2)
                r2=observe.run(datetime(2026,11,2,23,tzinfo=UTC))
                self.assertEqual(r2['orders'],[]);self.assertEqual(r['statistics'],r2['statistics'])
                r3=observe.run(datetime(2026,11,3,22,tzinfo=UTC))
                self.assertEqual(r3['state']['positions'],r2['state']['positions'])
                with sqlite3.connect(p/'paper_ledger.sqlite') as db:
                    self.assertEqual(db.execute('select count(*) from plans').fetchone()[0],1)
                    self.assertEqual(db.execute('select count(*) from fills').fetchone()[0],2)
                panel['adj'].loc['2026-11-02','A']*=1.1
                with self.assertRaisesRegex(ValueError,'Historical daily revised'):
                    observe.run(datetime(2026,11,3,23,tzinfo=UTC))

class HistoricalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel,cls.info=inputs.load_panel(ETFS)
        cls.precomputed=strategy.factors(cls.panel)

    def test_exact122_selections_and_scores(self):
        rows=json.loads((observe.P.parent/'monthly_stability_research/robust_control_ORDERS.json').read_text())
        with patch.object(strategy,'factors',return_value=self.precomputed):
            for row in rows:
                r=strategy.selection(self.panel,row['date'],ETFS)
                self.assertEqual(r['symbols'],row['symbols'],row['date'])
                expected=dict(zip(row['symbols'],row['scores']))
                for s,v in r['scores'].items():self.assertAlmostEqual(v,expected[s],places=12)

    def test_prefix_no_future_dependency(self):
        for date in ['2017-06-30','2022-12-30','2024-12-31']:
            p={k:v.loc[:date] for k,v in self.panel.items()}
            prefix=strategy.selection(p,date,ETFS)
            with patch.object(strategy,'factors',return_value=self.precomputed):full=strategy.selection(self.panel,date,ETFS)
            self.assertEqual(prefix['symbols'],full['symbols'])
            np.testing.assert_allclose(list(prefix['scores'].values()),list(full['scores'].values()),atol=1e-12)

if __name__=='__main__':unittest.main()
