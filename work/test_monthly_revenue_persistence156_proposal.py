"""Synthetic proposal construction: no actual sources, features or state writes."""
from copy import deepcopy
import unittest
import prepare_revenue_persistence156_proposal as p


def fixture():
    c = p.lifecycle
    design = dict(parent_completed_round=155, parent_total_trials=978,
                  proposed_mechanism=dict(dose=.01, arms=[dict(name=n, primary=i == 3) for i, n in enumerate(c.NAMES)]))
    refs = {k: dict(path=v, sha256='a'*64) for k, v in [('source_manifest', c.SOURCE_MANIFEST),
        ('source_grid_review', c.SOURCE_REVIEW), ('source_plan', c.SOURCE_PLAN), ('source_calendar', c.SOURCE_CALENDAR)]}
    refs['source_plan']['sha256'] = c.SOURCE_PLAN_SHA; refs['source_calendar']['sha256'] = c.SOURCE_CALENDAR_SHA
    return design, refs


class ProposalTests(unittest.TestCase):
    def build(self, design, refs):
        return p.build_proposal(design, refs, '2020-01-01T00:00:00+00:00', {'fixed': True}, {'train': 88})

    def test_fixed_four_arms_and_no_calculation(self):
        design, refs = fixture(); before = deepcopy((design, refs)); result = self.build(design, refs)
        self.assertEqual(result['arms'], list(p.lifecycle.NAMES))
        self.assertEqual(result['primary_challenger'], p.lifecycle.NAMES[3])
        self.assertEqual(result['fixed']['execution_paths'], 12)
        self.assertEqual(result['fixed']['monthly_orders'], 488)
        self.assertEqual(result['actual_new_factors'], 0)
        self.assertEqual((design, refs), before)

    def test_missing_source_review_fails(self):
        design, refs = fixture(); refs.pop('source_grid_review')
        with self.assertRaises(ValueError): self.build(design, refs)

    def test_changed_dose_or_arm_order_fails(self):
        for mutate in [lambda d: d['proposed_mechanism'].update(dose=.02),
                       lambda d: d['proposed_mechanism']['arms'].reverse()]:
            design, refs = fixture(); mutate(design)
            with self.assertRaises(ValueError): self.build(design, refs)

    def test_alternate_source_package_fails(self):
        design, refs = fixture(); refs['source_manifest']['path'] = 'other/package.json'
        with self.assertRaises(ValueError): self.build(design, refs)

    def test_source_plan_hash_cannot_change(self):
        design, refs = fixture(); refs['source_plan']['sha256'] = 'b'*64
        with self.assertRaises(ValueError): self.build(design, refs)

    def test_changed_parent_fails(self):
        design, refs = fixture(); design['parent_completed_round'] = 154
        with self.assertRaises(ValueError): self.build(design, refs)


if __name__ == '__main__':
    unittest.main()
