import unittest
import pandas as pd
from qqq_benchmark import calculate
class BenchmarkTests(unittest.TestCase):
    def test_entry_cost_and_dividend_adjustment(self):
        f=pd.DataFrame({'Open':[100.,99.],'Close':[100.,100.],'Adj Close':[99.,100.]},index=pd.to_datetime(['2026-09-28','2026-09-29']))
        r=calculate(f,'2026-09-28','2026-09-29')
        self.assertAlmostEqual(r['return'],100/99/1.0005-1)
        self.assertAlmostEqual(r['max_drawdown'],1/1.0005-1)
    def test_missing_endpoint(self):
        f=pd.DataFrame(index=pd.to_datetime([]))
        with self.assertRaises(ValueError):calculate(f,'2026-09-28','2026-09-29')
