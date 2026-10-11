import unittest
import copy
import numpy as np
import adaptive as a


class AdaptiveTests(unittest.TestCase):
    def pack(self):
        names = list(dict.fromkeys(k for menu in a.CORE_MENUS.values() for k in menu))
        names += ['vol126', 'downside63', 'idio63', 'vol_ratio21_126', 'stock_sector_skip42_126',
                  'stock_sector_skip42_252', 'cmf21', 'cmf63', 'signed_volume63', 'volume_ratio21_126',
                  'highvol_cmf63', 'trend_cmf63', 'high252_distance', 'close_location21',
                  'jump_concentration63', 'acceleration_skip42']
        return dict(names=names, X=np.random.default_rng(3).uniform(-.5, .5, (20, 4, len(names))),
                    eligible=np.ones((20, 4), bool), order=np.tile([0, 1, 2], (20, 1)))

    def test_all_core_menus_exclude_latest_two_months(self):
        for core in a.CORE_MENUS.values():
            for name in core:
                self.assertTrue('skip42' in name or 'skip2' in name or 'risk_' in name)
        self.assertEqual(a.FOLDS, [(17, 23), (23, 35)])

    def test_holdout_does_not_enter_joint_core_fit(self):
        d = self.pack()
        X, spec = a.design(d, a.CORE_MENUS['durations'], 'risk_path')
        mask = np.arange(20) < 17
        groups = np.zeros(20, int)
        before = a.fit(X, spec, d, groups, mask)
        d2 = copy.deepcopy(d)
        d2['order'][17:] = [3, 2, 1]
        X2 = X.copy()
        X2[17:] *= -2
        after = a.fit(X2, spec, d2, groups, mask)
        np.testing.assert_allclose(before['base'], after['base'], atol=1e-12, rtol=0)
        A, b, prior = a.e.limits(spec)
        self.assertLessEqual(np.max(np.einsum('ij,j->i', A, before['base'], optimize=False) - b), 1e-7)

    def test_sparse_state_falls_back_to_global(self):
        d = self.pack()
        X, spec = a.design(d, a.CORE_MENUS['risk_methods'], 'legacy')
        groups = np.zeros(20, int)
        groups[15:] = 1
        rule = a.fit(X, spec, d, groups, np.arange(20) < 17)
        np.testing.assert_allclose(rule['base'], rule['weights']['1'], rtol=0, atol=1e-12)
        self.assertEqual(rule['audit']['1']['months'], 2)

    @unittest.skipUnless((a.P / 'INPUTS.json').exists(), 'Run prepare first')
    def test_pool_extension_uniform_and_no_author_injection(self):
        m = a.read(a.P / 'INPUTS.json')
        pools = a.read(a.O / 'monthly_top35_extension/POOLS.json')
        for key, d in m['packs'].items():
            for i, ss in enumerate(d['symbols']):
                date = m['periods'][i]['start']
                stock = {a.e.actual(s) for s in pools[str(d['pool'])][date]}
                self.assertTrue(stock.issubset(set(ss)))
                self.assertEqual(len(ss), d['pool'] + 17)
                if date == '2022-05-31':
                    self.assertNotIn('COST', ss)

    @unittest.skipUnless((a.P / 'FROZEN.json').exists(), 'Run train first')
    def test_frozen_budgets(self):
        for r in a.read(a.P / 'FROZEN.json')['leaders'].values():
            rule = r['rule']
            A, b, _ = a.e.limits(rule['spec'])
            for w in rule['weights'].values():
                self.assertAlmostEqual(sum(w), 1.)
                self.assertLessEqual(np.max(np.einsum('ij,j->i', A, w, optimize=False) - b), 1e-7)
                self.assertLessEqual(np.abs(np.array(w) - rule['base']).sum(), .30 + 1e-7)


if __name__ == '__main__':
    unittest.main()
