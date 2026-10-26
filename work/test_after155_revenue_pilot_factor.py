"""Synthetic-only tests; no historical financial records or outcomes."""
import unittest
from after155_revenue_pilot_factor import measure_levels, NAMES


class RevenuePilotFactorTest(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(measure_levels(['10']*4), dict.fromkeys(NAMES, 0.))

    def test_constant_growth(self):
        values=measure_levels(['100','200','400','800'])
        for value in values.values(): self.assertAlmostEqual(value, .5, places=14)

    def test_weakest_not_latest(self):
        values=measure_levels(['100','110','99','198'])
        self.assertAlmostEqual(values[NAMES[0]], .5)
        self.assertAlmostEqual(values[NAMES[2]], -1/11)
        self.assertAlmostEqual(values[NAMES[1]], ((1.98**(1/3))-1)/(1.98**(1/3)))

    def test_identical_ending_total_different_path(self):
        a=measure_levels(['100','200','300','400'])
        b=measure_levels(['100','50','200','400'])
        self.assertEqual(a[NAMES[1]],b[NAMES[1]])
        self.assertNotEqual(a[NAMES[2]],b[NAMES[2]])

    def test_same_currency_scale_invariance(self):
        self.assertEqual(measure_levels(['1','2','3','4']),
                         measure_levels(['1000000','2000000','3000000','4000000']))

    def test_decline(self):
        for value in measure_levels(['800','400','200','100']).values():
            self.assertAlmostEqual(value,-1/3)

    def test_no_input_mutation(self):
        values=['100','110','121','133.1']; before=values[:]
        measure_levels(values)
        self.assertEqual(values,before)

    def test_invalid(self):
        for bad in ([],['1']*3,['1']*5,[1,2,3,4],['1','2','0','4'],
                    ['1','2','-1','4'],['1','2','NaN','4'],['1','2','Infinity','4']):
            with self.subTest(bad=bad), self.assertRaises(ValueError): measure_levels(bad)

    def test_large_exact_amounts(self):
        values=measure_levels(['1e100','2e100','4e100','8e100'])
        self.assertEqual(values,dict.fromkeys(NAMES,.5))


if __name__=='__main__': unittest.main()
