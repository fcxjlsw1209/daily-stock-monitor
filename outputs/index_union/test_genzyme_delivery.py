import unittest

from audit_genzyme_delivery import assess_receipt


class DeliveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.exact = dict(holder_scope='tendered_plan', cvr_received_date='2011-04-07')
        self.bound = dict(holder_scope='untendered_plan', cvr_received_date=None,
                          receipt_after_date_exclusive='2011-05-28')

    def test_tender_receipt_does_not_certify_generic_holder(self):
        self.assertEqual(assess_receipt(self.exact, '2011-04-11', 'generic_holder'),
                         'different_holder_scope_not_applicable')

    def test_same_day_does_not_prove_opening_availability(self):
        self.assertEqual(assess_receipt(self.exact, '2011-04-07', 'tendered_plan'),
                         'same_day_receipt_time_unknown')

    def test_exact_receipt_rejects_earlier_liquidation(self):
        self.assertEqual(assess_receipt(self.exact, '2011-04-04', 'tendered_plan'),
                         'liquidation_precedes_observed_receipt')

    def test_exact_receipt_only_establishes_date_order(self):
        self.assertEqual(assess_receipt(self.exact, '2011-04-11', 'tendered_plan'),
                         'receipt_date_precedes_liquidation_date_only')

    def test_untendered_lower_bound_rejects_april11(self):
        self.assertEqual(assess_receipt(self.bound, '2011-04-11', 'untendered_plan'),
                         'liquidation_precedes_observed_receipt')

    def test_exclusive_bound_rejects_same_date(self):
        self.assertEqual(assess_receipt(self.bound, '2011-05-28', 'untendered_plan'),
                         'liquidation_precedes_observed_receipt')

    def test_bound_does_not_invent_next_session_receipt(self):
        self.assertEqual(assess_receipt(self.bound, '2011-05-31', 'untendered_plan'),
                         'exact_receipt_date_unknown')

    def test_conflicting_precision_rejected(self):
        self.exact['receipt_after_date_exclusive'] = '2011-05-28'
        with self.assertRaises(ValueError):
            assess_receipt(self.exact, '2011-06-01', 'tendered_plan')


if __name__ == '__main__':
    unittest.main()
