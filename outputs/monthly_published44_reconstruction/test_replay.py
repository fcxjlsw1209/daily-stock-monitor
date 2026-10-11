import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run import corrected_stocks, select, independent_equity


class ReplayTests(unittest.TestCase):
    def test_delisted_issuer_does_not_consume_top32_slot(self):
        p=Path(__file__).parent.parent/'us_listed_top32_yahoo/PUBLISHED_MONTHLY_RANKINGS.json'
        event=next(e for e in json.loads(p.read_text())['periods'] if e['date']=='2023-02-28')
        self.assertIn('LTM',[r['symbol'] for r in event['top32']])
        corrected=corrected_stocks(event)
        self.assertEqual(len(corrected),32)
        self.assertNotIn('LTM',[r['symbol'] for r in corrected])
        self.assertEqual(corrected[-1]['symbol'],event['boundary33_40'][0]['symbol'])

    def test_scores_cannot_override_eligibility_or_minimum_two(self):
        scores=np.array([[100.,2.,1.],[100.,2.,1.]])
        eligible=np.array([[False,True,False],[False,True,True]])
        choices=select(scores,eligible,[['A','B','C']]*2)
        self.assertEqual(choices[0][0],[])
        self.assertEqual(choices[1][0],['B','C'])

    def test_fixed_shares_are_not_daily_equal_weight_rebalanced(self):
        dates=pd.date_range('2023-01-01',periods=3)
        px=pd.DataFrame({'A':[100.,200.,100.],'B':[100.,100.,100.]},index=dates)
        eq=independent_equity(px,[dict(date='2023-01-01',end='2023-01-03',symbols=['A','B'])])
        np.testing.assert_allclose(eq,[100000.,150000.,100000.])

    def test_cash_month_retains_nav(self):
        dates=pd.date_range('2023-01-01',periods=3)
        px=pd.DataFrame({'A':[100.,200.,100.]},index=dates)
        eq=independent_equity(px,[dict(date='2023-01-01',end='2023-01-03',symbols=[])])
        np.testing.assert_allclose(eq,[100000.]*3)


if __name__=='__main__':
    unittest.main()
