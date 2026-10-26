"""Synthetic-only regression checks for the precisely bounded cell repair."""
import ast
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
import annual_lag158_cell_repair as r
import test_monthly_annual_lag158_model as frozen_tests
from test_monthly_annual_lag158_model import fixture


def early_row(data, i=0, j=2):
    overlay = data['rows'][i*49+j]
    row = overlay['original_row']
    reason = r.ETF_REASON if row['is_etf'] else r.UNAVAILABLE_REASON
    row.update(selected_snapshot=None, cik=None, selected_event_ids=[], period_hint_ids=[],
               tier1=False, tier2=False, tier3=False, normal_period_admission=False, reasons=[reason])
    row.pop('known_amendment_event_ids', None)
    overlay.update(conditional_annual_pair_admitted=False, body_pair_context_verified=False,
                   body_context_source=None, old_admission_flags_changed=False, remaining_row_reasons=[reason])
    data['source_mask'][i,j] = False
    return overlay


def cell_args(data, i=0, j=2):
    signal = data['meta']['dates'][i]
    return [data['rows'][i*49+j], data['source_mask'][i,j], data['eligible'][i,j],
            data['meta']['symbols'][i][j], signal, data['calendar']['sessions'].index(signal),
            data['calendar']['sessions'], data['calendar']['closes'], data['event_metadata']]


class CellRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.template = fixture()
    def setUp(self): self.data = deepcopy(self.template)

    def test_only_amendment_branch_differs_from_frozen_cell_ast(self):
        old = ast.parse(inspect.getsource(r.original._cell)).body[0]
        new = ast.parse(inspect.getsource(r.repaired_cell)).body[0]
        k = next(i for i,node in enumerate(old.body) if isinstance(node,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='amendments' for t in node.targets))
        old.body[k:k+2] = []
        new.body[k:k+1] = []
        self.assertEqual(ast.dump(ast.Module(body=old.body,type_ignores=[])),
                         ast.dump(ast.Module(body=new.body,type_ignores=[])))
        self.assertEqual(ast.dump(old.args),ast.dump(new.args))

    def test_original_failure_reproduced_without_actual_sources(self):
        early_row(self.data)
        with self.assertRaisesRegex(ValueError,'known amendments'):
            r.original._cell(*cell_args(self.data))

    def test_both_early_return_rows_remain_unavailable_with_blockers(self):
        for j in (2,32):
            overlay=early_row(self.data,j=j); before=deepcopy(overlay)
            values,reasons,detail=r.repaired_cell(*cell_args(self.data,j=j))
            self.assertIsNone(values)
            self.assertIn('SOURCE_NOT_CONDITIONALLY_ADMITTED',reasons)
            self.assertIn('SOURCE_BLOCKER:'+overlay['original_row']['reasons'][0],reasons)
            self.assertIn(r.ABSENT_REASON,reasons)
            self.assertEqual(detail['information_cutoff_date'],overlay['original_row']['information_cutoff_date'])
            self.assertEqual(overlay,before)

    def test_exception_requires_entire_exact_early_return_shape(self):
        invalid = [('row','selected_snapshot','other'),('row','cik','0000000001'),
            ('row','selected_event_ids',['event']),('row','period_hint_ids',['hint']),
            ('row','tier1',True),('row','tier2',True),('row','tier3',True),
            ('row','normal_period_admission',True),('row','is_etf',True),
            ('row','reasons',['OTHER']),('row','reasons',[r.UNAVAILABLE_REASON,'OTHER']),
            ('overlay','body_pair_context_verified',True),('overlay','body_context_source','KO'),
            ('overlay','old_admission_flags_changed',True)]
        for section,key,value in invalid:
            overlay=early_row(self.data)
            target=overlay['original_row'] if section=='row' else overlay
            old=deepcopy(target[key]);target[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'outside frozen'):
                r.repaired_cell(*cell_args(self.data))
            target[key]=old

    def test_required_explicit_fields_cannot_be_omitted(self):
        for key in ('selected_snapshot','cik','selected_event_ids','period_hint_ids',
                    'tier1','tier2','tier3','normal_period_admission'):
            overlay=early_row(self.data);del overlay['original_row'][key]
            with self.subTest(key=key),self.assertRaises(ValueError):r.repaired_cell(*cell_args(self.data))

    def test_existing_bad_types_are_not_treated_as_absence(self):
        for value in (None,{},'',False,(),[1],['']):
            overlay=early_row(self.data);overlay['original_row']['known_amendment_event_ids']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'known amendments'):
                r.repaired_cell(*cell_args(self.data))

    def test_admitted_missing_key_still_fails_before_ineligible_or_initial_fallback(self):
        row=self.data['rows'][0]['original_row'];del row['known_amendment_event_ids']
        for eligible,signal in ((True,self.data['meta']['dates'][0]),(False,self.data['meta']['dates'][0]),
                                (False,r.INITIAL_PARTIAL)):
            args=cell_args(self.data,j=0);args[2]=eligible;args[4]=signal
            with self.subTest(eligible=eligible,signal=signal),self.assertRaisesRegex(ValueError,'outside frozen'):
                r.repaired_cell(*args)

    def test_source_flag_mismatch_not_covered(self):
        early_row(self.data);args=cell_args(self.data);args[1]=True
        with self.assertRaisesRegex(ValueError,'Source mask'):r.repaired_cell(*args)

    def test_present_amendments_and_bad_latest_match_frozen_cell(self):
        for j in (0,2,32):
            args=cell_args(self.data,j=j)
            self.assertEqual(r.repaired_cell(*args),r.original._cell(*args))
        self.data['rows'][0]['original_row']['known_amendment_event_ids']=['amendment']
        self.assertEqual(r.repaired_cell(*cell_args(self.data,j=0)),r.original._cell(*cell_args(self.data,j=0)))
        self.data['rows'][0]['original_row']['known_amendment_event_ids']=[]
        event=self.data['rows'][0]['original_row']['selected_event_ids'][0]
        self.data['event_metadata'][event]['filingDate']='invalid'
        self.assertEqual(r.repaired_cell(*cell_args(self.data,j=0)),r.original._cell(*cell_args(self.data,j=0)))

    def test_cutoff_checks_retained_for_early_return_row(self):
        row=early_row(self.data)['original_row'];row['information_cutoff_ns']+=1
        values,reasons,_=r.repaired_cell(*cell_args(self.data))
        self.assertIsNone(values);self.assertIn('ORIGINAL_CUTOFF_MISMATCH',reasons)

    def test_full_grid_arrays_identical_and_all123_prefixes_with_missing_rows(self):
        expected=r.original.build(**self.data)
        for i in range(122):
            early_row(self.data,i,2);early_row(self.data,i,32)
        before=json.dumps({k:self.data[k] for k in ('meta','rows','event_metadata','calendar')},sort_keys=True)
        with patch.object(r.original,'_cell',r.repaired_cell):
            full=r.original.build(**self.data)
            for key in r.original.EXTRA:
                self.assertEqual(full['arrays'][key].tobytes(),expected['arrays'][key].tobytes())
            self.assertEqual(full['audit']['rows'],5978)
            self.assertEqual(full['audit']['source_conditionally_admitted_rows'],244)
            self.assertEqual(full['audit']['reason_counts'][r.ABSENT_REASON],244)
            for n in range(123):
                got=r.original.build(**self.data,n=n)
                for key in r.original.EXTRA:
                    self.assertEqual(got['arrays'][key].tobytes(),full['arrays'][key][:n].tobytes())
            for key in r.original.RAW_NAMES:self.assertTrue(np.isnan(full['arrays'][key][:,[2,32]]).all())
            for key in r.original.RANK_NAMES:self.assertTrue((full['arrays'][key][:,[2,32]]==0).all())
            self.assertFalse(full['arrays']['common_available'][:,[2,32]].any())
        self.assertEqual(before,json.dumps({k:self.data[k] for k in ('meta','rows','event_metadata','calendar')},sort_keys=True))

    def test_future_missing_shape_is_not_read_for_short_prefix(self):
        early_row(self.data,1,2)['original_row']['reasons']=['future invalid']
        with patch.object(r.original,'_cell',r.repaired_cell):
            r.original.build(**self.data,n=1)
            with self.assertRaises(ValueError):r.original.build(**self.data,n=2)


class FrozenModelRegression(frozen_tests.ModelTests):
    def setUp(self):
        super().setUp()
        substitution=patch.object(r.original,'_cell',r.repaired_cell)
        substitution.start();self.addCleanup(substitution.stop)


class FrozenRankRegression(frozen_tests.RankTests):
    pass


if __name__=='__main__': unittest.main(verbosity=2)
