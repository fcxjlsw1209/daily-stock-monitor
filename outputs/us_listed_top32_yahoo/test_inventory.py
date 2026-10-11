import json
import unittest
from pathlib import Path

from build_inventory import current_pool, security_exclusion


def quote(symbol, cap, **extra):
    return dict(symbol=symbol, marketCap=cap, exchange='NYQ', currency='USD',
                quoteType='EQUITY', regularMarketTime=1791576000, **extra)


class InventoryTests(unittest.TestCase):
    def test_issuer_class_dedup_and_cap_not_summed(self):
        rows = [quote('GOOG', 3000), quote('GOOGL', 2900),
                quote('BRK-A', 2100), quote('BRK-B', 2000)]
        rows += [quote('S'+str(i), 1000-i) for i in range(40)]
        ranked, excluded = current_pool(rows)
        self.assertEqual([r['symbol'] for r in ranked[:2]], ['GOOGL', 'BRK-B'])
        self.assertEqual(ranked[0]['marketCap'], 2900)
        self.assertEqual(len(excluded), 2)

    def test_foreign_ADR_allowed_and_preferred_excluded(self):
        self.assertIsNone(security_exclusion(quote('TSM', 3000)))
        self.assertEqual(security_exclusion(quote('JPM-PC', 2000)), 'preferred_equity_excluded')
        self.assertEqual(security_exclusion(quote('PBR-A', 2000)), 'preferred_equity_excluded')

    def test_etf_otc_nonusd_not_ranked(self):
        for field, value in [('quoteType','ETF'), ('exchange','PNK'), ('currency','CAD')]:
            q = quote('TEST', 1000); q[field] = value
            self.assertIsNotNone(security_exclusion(q))

    def test_missing_common_representative_fails(self):
        with self.assertRaises(ValueError):
            current_pool([quote('BRK-A', 2000)])

    def test_never_certifies_historical_from_current_snapshot(self):
        root = Path(__file__).resolve().parent
        config = json.loads((root/'CONFIG.json').read_text())
        self.assertFalse(config['historical_pool_ready'])
        self.assertTrue(config['model_search_paused'])
        self.assertEqual(config['reconstruction_cost_bps'], 0)


if __name__ == '__main__':
    unittest.main()
