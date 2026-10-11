import unittest
import pandas as pd
from audit_genzyme_recovery import compare_quarters, audit_window
import terminal_baskets


class GenzymeRecoveryTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(date='2010-01-04',open=10.,high=11.,low=9.,close=10.,adj_close=10.,volume=100)])

    def test_wrong_issuer_prices_fail_quarterly_match(self):
        result=compare_quarters(self.frame(),[dict(quarter='2010Q1',high=60.15,low=48.18)])
        self.assertFalse(result[0]['matched'])

    def test_complete_single_session(self):
        _,report=audit_window(self.frame(),'2010-01-04','2010-01-04')
        self.assertTrue(report['price_window_checks_pass'])

    def test_missing_session_is_not_filled(self):
        d,report=audit_window(self.frame(),'2010-01-04','2010-01-05')
        self.assertEqual(len(d),1)
        self.assertEqual(report['missing_sessions'],['2010-01-05'])
        self.assertFalse(report['price_window_checks_pass'])

    def test_adjustment_mismatch_fails(self):
        frame=self.frame();frame['adj_close']=5.
        _,report=audit_window(frame,'2010-01-04','2010-01-04')
        self.assertFalse(report['price_window_checks_pass'])

    def test_duplicate_quote_fails(self):
        frame=pd.concat([self.frame(),self.frame()],ignore_index=True)
        _,report=audit_window(frame,'2010-01-04','2010-01-04')
        self.assertFalse(report['price_window_checks_pass'])

    def test_genzyme_basket_keeps_cash_claim_and_cvr_proceeds_separate(self):
        dates=pd.bdate_range('2011-04-01','2011-04-15')
        state=dict(cash=0.,fees=0.,positions={'GENZ':dict(qty=10.,cost=760.,
            dividends=0.,entry_fee=0.,entry_date='2011-04-01',entry_price=76.)})
        event=dict(event_id='genz',security_id='GENZ',effective_session='2011-04-11',
            execution_session='2011-04-11',known_at='2011-04-08T23:59:59-04:00',
            terms_verified=True,source_sha256='test',policy='liquidate_first_tradable_open',
            price_basis='raw_with_explicit_actions',cash_per_share=74.,payment_session=None,
            components=[dict(security_id='GCVRZ',ratio=1.)])
        output,_,trades,_,_=terminal_baskets.apply(state,dict(buys=[],sells=[]),
            [event],'2011-04-11',dates,{'GCVRZ':2.45})
        self.assertAlmostEqual(output['cash'],24.5*(1-.0005))
        self.assertEqual(output['cash_receivables'][0]['amount'],740.)
        self.assertIsNone(output['cash_receivables'][0]['payment_session'])
        self.assertAlmostEqual(trades[0]['exit_price'],76.45)
        self.assertNotIn('GENZ',output['positions'])


if __name__=='__main__':
    unittest.main()
