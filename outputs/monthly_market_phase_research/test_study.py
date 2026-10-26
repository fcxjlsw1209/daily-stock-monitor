import unittest
from pathlib import Path
import importlib.util
import numpy as np

sp=importlib.util.spec_from_file_location('marketphase_driver_test',Path(__file__).with_name('study.py'))
study=importlib.util.module_from_spec(sp);sp.loader.exec_module(study)


class ScoringTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(17);core=rng.uniform(-.25,.25,(8,49));support=rng.uniform(-.5,.5,(8,49))
        a=dict(base_original65=core+.48*support,core_original65=core,cw_original65=np.full((8,49),.52),eligible=np.ones((8,49),bool),state_valid=np.ones(8,bool))
        for state,factor in study.PROFILES.values():
            a['state_'+state]=rng.uniform(0,1,8);a[factor]=rng.uniform(-.5,.5,(8,49));a['valid_'+factor]=np.ones((8,49),bool)
        for factor in study.PEERS+list(study.PEER_CONTROLS.values()):
            a[factor]=rng.uniform(-.5,.5,(8,49));a['valid_'+factor]=np.ones((8,49),bool)
        return a

    def test_zero_intensity_and_missing_exact_nochange(self):
        a=self.fixture();a['state_pressure'][:]=0.
        row=dict(base='original65',profile='distress_lowdown',mode='phase',budget=.075)
        np.testing.assert_array_equal(study.score(row,a,8),a['base_original65'])
        a['state_pressure'][:]=np.nan;a['state_valid'][:]=False
        np.testing.assert_array_equal(study.score(row,a,8),a['base_original65'])
        row.update(mode='static')
        np.testing.assert_array_equal(study.score(row,a,8),a['base_original65'])

    def test_phase_one_matches_static_and_support_bound(self):
        a=self.fixture()
        for profile,(state,factor) in study.PROFILES.items():
            a['state_'+state][:]=1.
            row=dict(base='original65',profile=profile,mode='phase',budget=.075)
            sc=study.score(row,a,8)
            np.testing.assert_array_equal(sc,study.score({**row,'mode':'static'},a,8))
            self.assertTrue((np.abs((sc-a['core_original65'])/.48)<=.5+1e-12).all())

    def test_future_and_row_prefix_invariance(self):
        a=self.fixture();row=dict(base='original65',profile='repair_volatility',mode='phase',budget=.075)
        prefix=study.score(row,a,4).copy()
        a['state_repair'][4:]=0.;a['volatility'][4:]*=-1
        np.testing.assert_array_equal(study.score(row,a,8)[:4],prefix)

    def test_peer_missing_does_not_change_eligibility(self):
        a=self.fixture();before=a['eligible'].copy();a['valid_peer_participation'][:]=False
        row=dict(base='original65',profile='peer_participation',mode='peer',budget=.075)
        np.testing.assert_array_equal(study.score(row,a,8),a['base_original65'])
        np.testing.assert_array_equal(before,a['eligible'])

    def test_fixed_grid_and_training_guards(self):
        rows=study.grid();self.assertEqual(len(rows),42);self.assertEqual(len({r['id'] for r in rows}),42)
        z={k:dict(cagr=.2,sharpe=1.,max_drawdown=-.2) for k in ['training','early','recent']}
        self.assertTrue(study.admitted(z,z))


if __name__=='__main__':unittest.main()
