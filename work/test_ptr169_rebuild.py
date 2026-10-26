"""Synthetic acceptance-gate tests; no market fits or returns are evaluated."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import ptr169_rebuild as subject


class RebuildGateTests(unittest.TestCase):
    def test_missing_array_is_rejected(self):
        with self.assertRaises(AssertionError):
            subject.numeric_parity({'x': np.array([1.])},
                                   {'x': np.array([1.]), 'mask': np.array([True])})

    def test_boolean_type_is_not_interchangeable(self):
        with self.assertRaises(AssertionError):
            subject.numeric_parity({'mask': np.array([1])}, {'mask': np.array([True])})

    def test_missing_mask_is_exact(self):
        with self.assertRaises(AssertionError):
            subject.numeric_parity({'x': np.array([np.nan, 1.])},
                                   {'x': np.array([0., 1.])})

    def test_declared_numeric_tolerance(self):
        subject.numeric_parity({'x': np.array([1. + 1e-13, np.nan])},
                               {'x': np.array([1., np.nan])})
        with self.assertRaises(AssertionError):
            subject.numeric_parity({'x': np.array([1. + 1e-10])}, {'x': np.array([1.])})

    def control(self):
        names = ['DATA.npz', 'META.json', 'ORDERS.json', 'TRAIN_INPUTS.npz',
                 'WEEKLY_INPUTS.npz', 'MONTHLY_META.json', 'WEEKLY_META.json',
                 'BASE_DATA.npz', 'AUDIT.json', 'BASE_AUDIT.json',
                 'LABEL_AUDIT.json', 'MODEL_AUDIT.json']
        return dict(passed=True, arm='control', source_plan_sha256='registered',
                    artifacts_sha256={name: 'digest' for name in names})

    def test_control_binds_current_plan(self):
        data = self.control()
        def sha(path):
            return 'registered' if path.name == 'PLAN.json' else 'digest'
        with patch.object(subject, 'read', return_value=data), patch.object(subject, 'sha', side_effect=sha):
            self.assertEqual(subject.validated_control(), 'digest')
            data['source_plan_sha256'] = 'other-plan'
            with self.assertRaises(AssertionError): subject.validated_control()

    def test_control_binds_complete_outputs(self):
        data = self.control()
        def sha(path):
            return 'registered' if path.name == 'PLAN.json' else 'digest'
        with patch.object(subject, 'read', return_value=data), patch.object(subject, 'sha', side_effect=sha):
            data['artifacts_sha256']['DATA.npz'] = 'changed'
            with self.assertRaises(AssertionError): subject.validated_control()
            del data['artifacts_sha256']['DATA.npz']
            with self.assertRaises(AssertionError): subject.validated_control()


if __name__ == '__main__':
    unittest.main()
