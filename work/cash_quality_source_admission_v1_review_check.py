"""Run only synthetic admission tests without invoking the producer report writer."""
import unittest
import test_cash_quality_source_admission_v1 as suite

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(suite.AdmissionTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
