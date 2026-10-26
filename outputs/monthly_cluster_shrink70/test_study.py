"""Placement and gate tests do not access historical outcomes."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

p=Path(__file__).with_name('study.py')
s=importlib.util.spec_from_file_location('test_cluster_driver',p)
study=importlib.util.module_from_spec(s);s.loader.exec_module(study)


class Placement(unittest.TestCase):
    def data(self):
        return dict(forecast_available=np.ones((3,2),bool),eligible=np.ones((3,2),bool),
           state_weak_narrow=np.array([True,False,False]),state_calm10=np.array([False,True,False]),
           extension_available=np.ones((3,2),bool),rank_log_excess=np.zeros((3,2)),
           rank_cluster_sensitivity_shrink=np.full((3,2),.4))

    def test_declared_slot_and_regime(self):
        a=self.data()
        with patch.object(study.local,'modified_pre_score',return_value=np.zeros((3,2))):
            got=study.pre_score('cluster_sensitivity_shrink',a,3)
        np.testing.assert_allclose(got,[[.005,.005],[.001,.001],[0,0]],atol=1e-15)

    def test_no_change_ineligible_unavailable(self):
        a=self.data();a['eligible'][0,0]=False;a['forecast_available'][0,1]=False
        original=np.arange(6,dtype=float).reshape(3,2)
        with patch.object(study.local,'modified_pre_score',return_value=original.copy()):
            got=study.pre_score('cluster_sensitivity_shrink',a,3)
        np.testing.assert_array_equal(got[0],original[0])

    def test_identical_rank_identity(self):
        a=self.data();a['rank_cluster_sensitivity_shrink'][:]=0
        original=np.arange(6,dtype=float).reshape(3,2)
        with patch.object(study.local,'modified_pre_score',return_value=original.copy()):
            got=study.pre_score('cluster_sensitivity_shrink',a,3)
        self.assertEqual(got.tobytes(),original.tobytes())

    def test_prefix_does_not_consume_future(self):
        a=self.data();a['rank_cluster_sensitivity_shrink'][1:]=np.nan
        with patch.object(study.local,'modified_pre_score',return_value=np.zeros((3,2))):
            got=study.pre_score('cluster_sensitivity_shrink',a,1)
        self.assertTrue(np.isfinite(got).all())

    def test_tie_retains_control(self):
        metric=dict(cagr=.5,sharpe=1.5,max_drawdown=-.3)
        rows=[dict(id=i,name=study.NAMES[i],training=metric.copy(),early=metric.copy(),recent=metric.copy()) for i in range(len(study.NAMES))]
        self.assertEqual(study.select(rows)['id'],0)
        rows[1]['training']['cagr']=.7;rows[1]['early']['cagr']=.4
        self.assertEqual(study.select(rows)['id'],0)


if __name__=='__main__':unittest.main()
