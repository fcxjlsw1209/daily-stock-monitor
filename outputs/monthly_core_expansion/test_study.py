import unittest
import importlib.util
from pathlib import Path
import numpy as np
_sp=importlib.util.spec_from_file_location('monthly_core_driver_tests',Path(__file__).with_name('study.py'))
study=importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(study)

class ScoringTests(unittest.TestCase):
    def setUp(self):
        core=np.array([[.1,-.1,.15],[-.05,.03,.02]])
        support=np.array([[.2,-.1,.5],[-.2,.1,.05]])
        cw=np.full((2,1),.5)
        self.a=dict(base_original65=core+(1-cw)*support,core_original65=core,cw_original65=cw,
                    pred_return=np.zeros((2,3)),pred_ordinal=np.zeros((2,3)),pred_available=np.ones((2,3),bool))
        self.a['factor_hac_sharpe']=np.array([[.4,-.2,.1],[-.3,.2,.4]])
        self.a['valid_hac_sharpe']=np.ones((2,3),bool)
        self.row=dict(base='original65',core='original',blend=0.,arm='none',budget=0.)

    def test_grid_unique_fixed90(self):
        rows=study.grid()
        self.assertEqual(len(rows),90)
        self.assertEqual([x['id'] for x in rows],list(range(90)))
        self.assertEqual(len({tuple(sorted((k,v) for k,v in x.items() if k!='id')) for x in rows}),90)

    def test_original_exact_and_zero_ml_exact(self):
        np.testing.assert_array_equal(study.score(self.row,self.a,2),self.a['base_original65'])
        for arm in ['ordinal','return']:
            np.testing.assert_array_equal(study.score(dict(self.row,arm=arm,budget=.05),self.a,2),self.a['base_original65'])

    def test_core_half_full_exact(self):
        for blend in [.5,1.]:
            expected=self.a['base_original65']+blend*(.5*self.a['factor_hac_sharpe']-self.a['core_original65'])
            actual=study.score(dict(self.row,core='hac_sharpe',blend=blend),self.a,2)
            np.testing.assert_allclose(actual,expected,atol=1e-15)

    def test_missing_core_exact_fallback(self):
        self.a['valid_hac_sharpe'][:]=False
        self.a['factor_hac_sharpe'][:]=np.nan
        np.testing.assert_allclose(study.score(dict(self.row,core='hac_sharpe',blend=1.),self.a,2),self.a['base_original65'],atol=1e-15)

    def test_ml_only_support_bounded_and_unavailable_zero(self):
        self.a['pred_return'][:]=100.
        row=dict(self.row,core='hac_sharpe',blend=1.,arm='return',budget=.05)
        plain=study.score(dict(row,arm='none',budget=0.),self.a,2)
        actual=study.score(row,self.a,2)
        self.assertTrue(np.all(actual-plain<=.05+1e-15))
        self.a['pred_available'][:]=False
        np.testing.assert_array_equal(study.score(row,self.a,2),plain)

    def test_prefix_unchanged_by_future(self):
        row=dict(self.row,core='hac_sharpe',blend=.5,arm='ordinal',budget=.05)
        expected=study.score(row,self.a,1)
        for key,value in self.a.items():
            if value.dtype!=bool:value[1]=10.
        np.testing.assert_array_equal(study.score(row,self.a,1),expected)

if __name__=='__main__':unittest.main()
