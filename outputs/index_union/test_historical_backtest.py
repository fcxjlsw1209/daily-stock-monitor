import unittest
import numpy as np
import pandas as pd
from historical_backtest import run_all, run_account, metrics
from test_dynamic_features import fixture
from union_execution import engine


def inputs():
    _, _, data = fixture()
    dates, cols = data['close'].index, data['close'].columns
    data.update(dividends=data['close']*0,splits=data['close']*0+1,
                sectors=pd.DataFrame({s:'sector_'+str(i//2) for i,s in enumerate(cols)},index=dates),
                issuers={s:'issuer_'+s for s in cols},execution_basis='raw_with_explicit_actions',
                signal_basis='split_adjusted',raw_per_signal=data['close']*0+1)
    return data

class HistoricalBacktestTests(unittest.TestCase):
    def test_all_six_complete_end_to_end_without_changing_inputs(self):
        d=inputs();before=d['close'].copy();dates=d['close'].index
        results=run_all(d,dates[300],dates[-1])
        self.assertEqual(len(results),6)
        for name,r in results.items():
            self.assertEqual(r['status'],'complete',(name,r.get('error')))
            self.assertEqual(len(r['daily']),100)
            self.assertGreater(len(r['orders']),0)
            self.assertGreater(len(r['trades']),0)
            self.assertAlmostEqual(r['metrics']['realized_pnl']+r['metrics']['unrealized_pnl'],r['metrics']['equity']-100000)
            self.assertFalse(r['performance_publishable'])
        pd.testing.assert_frame_equal(d['close'],before)

    def test_no_future_data_changes_earlier_orders(self):
        d=inputs();dates=d['close'].index
        a=run_all(d,dates[300],dates[349])
        for key in ['adj','opening','close','high','low']:
            d[key].iloc[350:]*=2
        b=run_all(d,dates[300],dates[349])
        for name in a:
            self.assertEqual(a[name]['orders'],b[name]['orders'])
            self.assertEqual(a[name]['daily'],b[name]['daily'])

    def test_all_empty_eligible_days_keep_cash_not_forced_picks(self):
        d=inputs();dates=d['close'].index;d['eligible'].loc[:,:]=False
        for r in run_all(d,dates[300],dates[-1]).values():
            self.assertEqual(r['metrics']['equity'],100000)
            self.assertIsNone(r['metrics']['net_win_rate'])
            self.assertEqual(r['orders'],[])

    def test_actual_held_gap_stops_without_publishing_partial_performance(self):
        d=inputs();dates=d['close'].index
        normal=run_account('accel',d,dates[300],dates[-1]);buy=next(o for o in normal['orders'] if o['side']=='buy')
        day=pd.Timestamp(buy['date']);next_day=dates[dates.get_loc(day)+1]
        d['close'].at[next_day,buy['symbol']]=np.nan
        result=run_account('accel',d,dates[300],dates[-1])
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(result['blocked_session'],str(next_day.date()))
        self.assertIn(buy['symbol'],result['last_valid_state']['positions'])
        self.assertIsNone(result['metrics'])

    def test_missing_open_cancelled_in_full_run(self):
        d=inputs();dates=d['close'].index
        normal=run_account('accel',d,dates[300],dates[-1]);buy=next(o for o in normal['orders'] if o['side']=='buy')
        d['opening'].at[pd.Timestamp(buy['date']),buy['symbol']]=np.nan
        result=run_account('accel',d,dates[300],dates[-1])
        self.assertEqual(result['status'],'complete')
        self.assertTrue(any(x['symbol']==buy['symbol'] and x['session']==buy['date'] for x in result['cancellations']))
        self.assertFalse(any(o['symbol']==buy['symbol'] and o['date']==buy['date'] for o in result['orders']))

    def test_missing_action_cannot_silently_be_no_action(self):
        d=inputs();dates=d['close'].index
        normal=run_account('accel',d,dates[300],dates[-1]);buy=normal['orders'][0]
        d['splits'].at[pd.Timestamp(buy['date']),buy['symbol']]=np.nan
        r=run_account('accel',d,dates[300],dates[-1])
        self.assertEqual(r['status'],'blocked')
        self.assertIn('action coverage',r['error'])

    def test_adjusted_execution_basis_rejected(self):
        d=inputs();d['execution_basis']='unknown';dates=d['close'].index
        with self.assertRaises(ValueError):run_all(d,dates[300],dates[-1])

    def test_split_units_do_not_create_profit_or_change_signal_timing(self):
        d=inputs();dates=d['close'].index
        original=run_all(d,dates[300],dates[-1])
        d['raw_per_signal'].iloc[:330]=2.
        d['splits'].iloc[330]=2.
        split=run_all(d,dates[300],dates[-1])
        for name in original:
            self.assertEqual(split[name]['status'],'complete',split[name].get('error'))
            self.assertAlmostEqual(original[name]['metrics']['equity'],split[name]['metrics']['equity'],places=6)
            self.assertEqual([(r['date'],r['symbol'],r['side']) for r in original[name]['orders']],
                             [(r['date'],r['symbol'],r['side']) for r in split[name]['orders']])

    def test_drawdown_includes_initial_capital(self):
        s=engine('baseline').initial_state()
        r=metrics(s,[dict(date='2020-01-01',equity=99000.)],[])
        self.assertAlmostEqual(r['maximum_drawdown'],-.01)
        self.assertIsNone(r['net_win_rate'])

if __name__=='__main__':unittest.main()
