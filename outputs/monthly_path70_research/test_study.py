import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('path70_test_driver',Path(__file__).with_name('study.py'))
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Driver(unittest.TestCase):
    def rows(self):
        def metric(cagr):return dict(cagr=cagr,sharpe=1.6,max_drawdown=-.3,total_return=1.)
        return [dict(spec=z,training=metric(.5 if z['id']==1 else .49),early=metric(.4),recent=metric(.6)) for z in s.configs()]

    def test_grid_two_exact_controls_and_matched_bases(self):
        rows=s.configs()
        self.assertEqual(len(rows),26)
        self.assertEqual([z['id'] for z in rows],list(range(26)))
        self.assertEqual([z['base'] for z in rows[:2]],['original65','incumbent66'])
        for factor in s.FACTORS:
            for budget in s.BUDGETS:
                self.assertEqual({z['base'] for z in rows if z['factor']==factor and z['budget']==budget},{'original65','incumbent66'})

    def test_fallback_and_all_frozen_family_diagnostics(self):
        selected=s.select(self.rows())
        self.assertEqual(selected['primary']['spec']['id'],1)
        self.assertEqual(len(selected),12)
        self.assertTrue(all(not selected[b+'__'+f]['admitted'] for b in ('original65','incumbent66') for f in s.FACTORS))

    def test_select_uses_training_admission_not_full_period_field(self):
        rows=self.rows()
        rows[2]['training']['cagr']=.51
        rows[3]['full_cagr']=10.
        selected=s.select(rows)
        self.assertEqual(selected['primary']['spec']['id'],2)

    def test_weak_early_segment_cannot_win_by_training_cagr(self):
        rows=self.rows()
        rows[2]['training']['cagr']=.9
        rows[2]['early']['cagr']=.3
        self.assertEqual(s.select(rows)['primary']['spec']['id'],1)


if __name__=='__main__':unittest.main()
