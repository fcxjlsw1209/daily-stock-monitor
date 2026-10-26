"""Only fabricated design/reference fixtures; no market data or source grid."""
from copy import deepcopy
import unittest
import prepare_filing_association157_proposal as p


class ProposalTests(unittest.TestCase):
    def setUp(self):
        c = p.lifecycle
        self.design = dict(predecessor=dict(completed_round=156, completed_trials=982),
            arms=[dict(name=n) for n in c.NAMES], clock=dict(id='second_XNYS_session_strictly_after_filingDate_v1'),
            fixed_transform=dict(score_dose=.01), hypothesis={'fixed': True},
            explicit_reconstruction_assumptions=['Not certified exact acceptance or original source vintage.'])
        self.refs = {k:dict(path=v, sha256='a'*64) for k,v in dict(source_manifest=c.SOURCE_MANIFEST,
            source_grid_review=c.SOURCE_REVIEW, source_plan=c.DESIGN, source_calendar=c.SOURCE_CALENDAR).items()}
        self.refs['source_plan']['sha256'] = c.DESIGN_SHA
        self.refs['source_calendar']['sha256'] = c.SOURCE_CALENDAR_SHA

    def build(self):
        return p.build_proposal(self.design, self.refs, '2026-10-20T21:00:00+00:00', {'fixed':'measurement'}, {'fixed':'selection'})

    def test_exact_scope_and_counts(self):
        v = self.build()
        self.assertEqual(v['fixed']['period'], ['2016-08-17','2026-09-24'])
        self.assertEqual(v['fixed']['execution_paths'], 12)
        self.assertEqual(v['fixed']['monthly_orders'], 488)
        self.assertEqual(v['actual_new_nuisance_estimates'], 0)
        self.assertNotIn('model_fits', v['fixed'])

    def test_input_objects_unchanged(self):
        a,b=deepcopy(self.design),deepcopy(self.refs); self.build()
        self.assertEqual((a,b),(self.design,self.refs))

    def test_unexpected_clock_fails(self):
        self.design['clock']['id']='exact_acceptance';
        with self.assertRaises(ValueError):self.build()

    def test_arm_order_fails(self):
        self.design['arms'].reverse()
        with self.assertRaises(ValueError):self.build()

    def test_dose_change_fails(self):
        self.design['fixed_transform']['score_dose']=.02
        with self.assertRaises(ValueError):self.build()

    def test_bad_reference_or_missing_reference_fails(self):
        self.refs['source_plan']['sha256']='b'*64
        with self.assertRaises(ValueError):self.build()
        self.refs.pop('source_grid_review')
        with self.assertRaises(ValueError):self.build()

    def test_wrong_predecessor_fails(self):
        self.design['predecessor']['completed_trials']=978
        with self.assertRaises(ValueError):self.build()


if __name__=='__main__':unittest.main(verbosity=2)
