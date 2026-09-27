import unittest,sys,json,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from engine import initial_state,plan,execute
from strategy import load_inputs,features
from market_data import session_bar,schedule,completed_session
from monitor import connect,init,save_plan,reconcile,statistics,latest_state
P=Path(__file__).resolve().parent

class MonitorTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2026-09-01',periods=20);c=pd.DataFrame({'X':100.},index=dates)
        return dates,c,c+1,c*0+2,pd.DataFrame(True,index=dates,columns=['X']),c*0+1,{'X':'test'},{'exit':'guarded10','allocation':.1}

    def test_prior_close_signal_and_gap_stop(self):
        dates,c,ma,atr,mask,score,sec,cfg=self.fixture();state=initial_state(1000)
        decision=plan(state,dates[1],dates,c,ma,atr,mask,score,sec,cfg)
        self.assertEqual(decision['buys'][0]['budget'],100)
        state,d,t,o,a=execute(state,decision,{'X':{'open':100,'close':95,'dividends':10}},dates,0,sec)
        self.assertEqual(state['dividends'],0)
        c.loc[dates[1],'X']=95
        decision=plan(state,dates[2],dates,c,ma,atr,mask,score,sec,cfg)
        self.assertEqual(decision['sells'][0]['reason'],'close_below_2atr');self.assertFalse(decision['buys'])
        state,d,t,o,a=execute(state,decision,{'X':{'open':70,'close':80,'dividends':1}},dates,0,sec)
        self.assertAlmostEqual(t[0]['net_return'],-.29);self.assertAlmostEqual(state['equity'],971)

    def test_rebound_and_maximum_holding(self):
        dates,c,ma,atr,mask,score,sec,cfg=self.fixture();state=initial_state(1000)
        q=plan(state,dates[1],dates,c,ma,atr,mask,score,sec,cfg)
        state,*_=execute(state,q,{'X':{'open':100,'close':100}},dates,0,sec)
        self.assertFalse(plan(state,dates[10],dates,c,ma,atr,mask,score,sec,cfg)['sells'])
        self.assertEqual(plan(state,dates[11],dates,c,ma,atr,mask,score,sec,cfg)['sells'][0]['reason'],'maximum_sessions')
        c.loc[dates[1],'X']=102
        self.assertEqual(plan(state,dates[2],dates,c,ma,atr,mask,score,sec,cfg)['sells'][0]['reason'],'close_above_sma5')

    def test_split_and_dividend_units(self):
        dates,c,ma,atr,mask,score,sec,cfg=self.fixture();state=initial_state(1000)
        state,*_=execute(state,plan(state,dates[1],dates,c,ma,atr,mask,score,sec,cfg),{'X':{'open':100,'close':100}},dates,0,sec)
        no_orders={'session':str(dates[2].date()),'buys':[],'sells':[]}
        state,*_=execute(state,no_orders,{'X':{'open':50,'close':50,'split':2,'dividends':.5}},dates,0,sec)
        self.assertEqual(state['positions']['X']['qty'],2);self.assertEqual(state['positions']['X']['atr'],1)
        self.assertEqual(state['equity'],1001)
        h=pd.DataFrame({'Open':[50,50],'Close':[50,50],'Dividends':[.5,0],'Stock Splits':[0,2]},index=dates[:2])
        self.assertEqual(session_bar(h,dates[0])['open'],100);self.assertEqual(session_bar(h,dates[0])['dividends'],1)
        self.assertEqual(session_bar(h,dates[1])['open'],50)

    def test_immutable_plan_and_late_rejection(self):
        db=connect(':memory:');decision={'session':'2026-09-22','buys':[],'sells':[]}
        self.assertTrue(save_plan(db,decision,'2026-09-22T13:20Z','test','2026-09-22T13:30Z'))
        self.assertFalse(save_plan(db,decision,'2026-09-22T13:21Z','test','2026-09-22T13:30Z'))
        with self.assertRaises(RuntimeError):save_plan(db,{**decision,'new':1},'2026-09-22T13:21Z','test','2026-09-22T13:30Z')
        with self.assertRaises(RuntimeError):save_plan(db,decision,'2026-09-22T13:30Z','test','2026-09-22T13:30Z')
        db.close()

    def test_reconcile_idempotent_and_no_backdated_trade(self):
        db=connect(':memory:');cfg={'initial_capital':1000,'paper_start':'2026-09-22','cost_bps':10,'sectors':{'X':'test'}};init(db,cfg)
        q={'session':'2026-09-22','buys':[{'symbol':'X','budget':100.,'atr':2.}],'sells':[]}
        save_plan(db,q,'2026-09-22T13:20Z','test','2026-09-22T13:30Z')
        h=pd.DataFrame({'Open':[100,102],'Close':[102,103],'Dividends':[0,0],'Stock Splits':[0,0]},index=pd.to_datetime(['2026-09-22','2026-09-23']))
        args=(db,cfg,{'X':h},{'as_of':'2026-09-23','snapshot_id':'test'},pd.Timestamp('2026-09-23T21:00Z'))
        self.assertEqual(reconcile(*args),2);before=latest_state(db);self.assertEqual(reconcile(*args),0);self.assertEqual(before,latest_state(db))
        st=statistics(db);self.assertEqual(st['closed_trades'],0);self.assertIsNone(st['win_rate']);self.assertEqual(st['missed_sessions'],1)
        self.assertEqual(db.execute('SELECT count(*) FROM orders').fetchone()[0],1);db.close()

    def test_entry_day_split_preserves_plan_atr_units(self):
        for reference,planned_atr in [(100.,2.),(50.,1.)]:
            db=connect(':memory:');cfg={'initial_capital':1000,'paper_start':'2026-09-22','cost_bps':0,'sectors':{'X':'test'}};init(db,cfg)
            q={'session':'2026-09-22','signal_date':'2026-09-21','buys':[{'symbol':'X','budget':100.,'atr':planned_atr,'reference_close':reference}],'sells':[]}
            save_plan(db,q,'2026-09-22T13:20Z','test','2026-09-22T13:30Z')
            h=pd.DataFrame({'Open':[50,50],'Close':[50,50],'Dividends':[0,0],'Stock Splits':[0,2]},index=pd.to_datetime(['2026-09-21','2026-09-22']))
            reconcile(db,cfg,{'X':h},{'as_of':'2026-09-22','snapshot_id':'test'},pd.Timestamp('2026-09-22T21:00Z'))
            self.assertEqual(latest_state(db)['positions']['X']['atr'],1.)
            db.close()

    def test_holidays_and_early_close(self):
        s=schedule(pd.Timestamp('2026-09-21T18:00Z'))
        self.assertNotIn(pd.Timestamp('2025-01-09'),s.index);self.assertNotIn(pd.Timestamp('2026-11-26'),s.index)
        self.assertEqual(completed_session(pd.Timestamp('2026-11-27T18:20Z')),pd.Timestamp('2026-11-27'))
        self.assertEqual(completed_session(pd.Timestamp('2026-11-27T18:10Z')),pd.Timestamp('2026-11-25'))

    @unittest.skipUnless((P.parent/"stock_database"/"stocks.sqlite").exists(), "Requires unpublished historical research fixtures")
    def test_feature_causality(self):
        data,raw=load_inputs();f=features(data,raw);n=3100
        short=replace(data,**{k:getattr(data,k).iloc[:n] for k in ['close','opening','dividends','volume','sector_prices']})
        g=features(short,{k:v.iloc[:n] for k,v in raw.items()})
        for k in ['score','ma5','atr','cmf']:pd.testing.assert_frame_equal(f[k].iloc[:n],g[k])
        for k in f['masks']:pd.testing.assert_frame_equal(f['masks'][k].iloc[:n],g['masks'][k])

    @unittest.skipUnless((P.parent/"strategy_selection"/"daily_equity.csv.gz").exists(), "Requires unpublished research results")
    def test_research_accounts(self):
        root=P.parent/'strategy_selection';d=pd.read_csv(root/'daily_equity.csv.gz');o=pd.read_csv(root/'orders.csv.gz')
        for (name,cost),g in o.groupby(['strategy','cost_bps']):
            dd=d[(d.strategy==name)&(d.cost_bps==cost)].set_index('date')
            delta=np.where(g.side=='buy',-g.quantity*g.price-g.fee,g.quantity*g.price-g.fee)
            movement=pd.Series(delta,index=g.date).groupby(level=0).sum().reindex(dd.index,fill_value=0)
            np.testing.assert_allclose(dd.cash,100000+movement.cumsum()+dd.dividends_cumulative,atol=1e-6)
            np.testing.assert_allclose(dd.equity,dd.cash+dd.invested,atol=1e-6)
            self.assertLessEqual(dd.positions.max(),5);self.assertTrue((dd.cash>=-1e-7).all())
        trades=pd.read_csv(root/'trades.csv.gz')
        for name,g in trades.groupby('strategy'):self.assertLessEqual(g.sessions.max(),10 if name.endswith('guarded10') else 5)

if __name__=='__main__':unittest.main()
