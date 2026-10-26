"""Synthetic frozen-slot, source-identity and registration tests; no outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

p = Path(__file__).with_name('study.py')
spec = importlib.util.spec_from_file_location('test_cluster_budget_driver', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture():
    shape = (4, 6)
    cw = np.array([.52, .5, .52, .5])[:, None]
    support = np.tile([.1, -.1, .05, 0., -.05, .1], (4, 1))
    core = cw*np.tile([.45, .4, .3, .29, .28, .27], (4, 1))
    a = dict(baseline=core+(1-cw)*support, core=core, core_weight=cw,
        eligible=np.ones(shape, bool), state_calm10=np.array([False, True, False, True]),
        state_weak_narrow=np.array([False, False, True, True]),
        pre_score=np.tile([.30, .25, .20, .19, .181, .1805], (4, 1)),
        effects=np.tile([0., 0., -.10, .05, .15, .50], (4, 1)),
        forecast_available=np.ones(shape, bool), extension_available=np.ones(shape, bool),
        old_ml4=np.zeros(shape), rank_log_excess=np.tile([-.4, -.3, -.2, .1, .3, .5], (4, 1)),
        rank_cluster_sensitivity_shrink=np.tile([.5, .3, .1, -.2, -.3, -.4], (4, 1)),
        model_available=np.ones(shape, bool), local_available=np.ones(shape, bool),
        raw_source_log=np.tile([-.06, -.05, -.04, .02, .04, .06], (4, 1)),
        raw_local_constant=np.tile([5., 3., 1., -2., -3., -4.], (4, 1)),
        raw_local_linear=np.tile([-4., -3., -2., 1., 3., 5.], (4, 1)))
    m = dict(symbols=[list('ABCDEF') for _ in range(4)],
        dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'],
        ends=['2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30'])
    return a, m


class Placement(unittest.TestCase):
    def setUp(self):
        self.a, self.m = fixture()

    def test_registered_arms_and_maximum_budget(self):
        self.assertEqual(s.design()['configs'], s.configs())
        self.assertEqual(s.FRACTIONS, (0., .5, .75))
        self.assertTrue(all(0. <= 1.-.25-f <= 1. for f in s.FRACTIONS))

    def test_exact_half_and_three_quarter_placement_preserves_local_correction(self):
        base = s.local.modified_pre_score(s.LOCAL_SPEC, self.a, 4)
        for name, fraction in zip(s.NAMES[1:], s.FRACTIONS[1:]):
            weight = fraction*np.array([0., .01, .05, .05])[:, None]
            delta = weight*(self.a['rank_cluster_sensitivity_shrink']-self.a['rank_log_excess'])
            want = base.copy()
            want[delta != 0.] += delta[delta != 0.]
            np.testing.assert_array_equal(s.pre_score(name, self.a, 4), want)
        # The simultaneous weak/calm state gets only the weak coefficient.
        np.testing.assert_array_equal(delta[3], delta[2])

    def test_no_change_ineligible_unavailable_or_calm_without_extension(self):
        self.a['eligible'][2, 0] = False
        self.a['forecast_available'][3, 1] = False
        self.a['extension_available'][1, 2] = False
        self.a['extension_available'][2, 3] = False
        base = s.local.modified_pre_score(s.LOCAL_SPEC, self.a, 4)
        for name in s.NAMES[1:]:
            got = s.pre_score(name, self.a, 4)
            for row, col in ((2, 0), (3, 1), (1, 2)):
                self.assertEqual(got[row, col], base[row, col])
            self.assertNotEqual(got[2, 3], base[2, 3])

    def test_fraction_zero_bypasses_cluster_and_exact_source_orders(self):
        del self.a['rank_cluster_sensitivity_shrink']
        want = s.local.choices(s.LOCAL_SPEC, self.a, self.m, 4)
        got = s.choices(s.NAMES[0], self.a, self.m, 4)
        self.assertEqual(got[0], want[0])
        self.assertEqual(got[1].tobytes(), want[1].tobytes())
        self.assertEqual(s.rp.make_orders(*got, self.m), s.rp.make_orders(*want, self.m))

    def test_identical_rank_identity_all_scores_and_orders(self):
        self.a['rank_cluster_sensitivity_shrink'] = self.a['rank_log_excess'].copy()
        want = s.choices(s.NAMES[0], self.a, self.m, 4)
        for name in s.NAMES[1:]:
            got = s.choices(name, self.a, self.m, 4)
            self.assertEqual(got[0], want[0])
            self.assertEqual(got[1].tobytes(), want[1].tobytes())

    def test_one_original_selector_receives_original_payload(self):
        for name in s.NAMES[1:]:
            with patch.object(s.local.ep, 'select_boundary', wraps=s.local.ep.select_boundary) as selector:
                s.choices(name, self.a, self.m, 4)
            self.assertEqual(selector.call_count, 1)
            pre, eligible, symbols, effects, original, policy = selector.call_args.args
            np.testing.assert_array_equal(pre, s.pre_score(name, self.a, 4))
            np.testing.assert_array_equal(eligible, self.a['eligible'])
            np.testing.assert_array_equal(effects, self.a['effects'])
            self.assertEqual(symbols, self.m['symbols'])
            for key in ('baseline', 'core', 'state_calm10', 'state_weak_narrow'):
                np.testing.assert_array_equal(original[key], self.a[key])
            self.assertTrue(all(policy[k] == v for k, v in s.local.POLICY_FIELDS.items()))

    def test_prefix_future_poison_and_input_immutability(self):
        before = copy.deepcopy(self.a)
        for name in s.NAMES:
            full = s.choices(name, self.a, self.m, 4)
            short = s.choices(name, self.a, self.m, 2)
            self.assertEqual(short[0], full[0][:2])
            np.testing.assert_array_equal(short[1], full[1][:2])
            poisoned = copy.deepcopy(self.a)
            for value in poisoned.values():
                if value.dtype.kind == 'f':
                    value[2:] = np.nan
                else:
                    value[2:] = ~value[2:]
            got = s.choices(name, poisoned, self.m, 2)
            self.assertEqual(got[0], short[0])
            np.testing.assert_array_equal(got[1], short[1])
        for key in before:
            self.assertEqual(before[key].tobytes(), self.a[key].tobytes())

    def test_unknown_arm_rejected_before_scoring(self):
        for fn, args in ((s.pre_score, ('unregistered', {}, 1)), (s.choices, ('unregistered', {}, {}, 1))):
            with self.assertRaises(ValueError):
                fn(*args)

    def test_active_nonfinite_rank_rejected_but_unused_rank_ignored(self):
        self.a['rank_cluster_sensitivity_shrink'][2, 0] = np.nan
        for name in s.NAMES[1:]:
            with self.assertRaises(ValueError):
                s.pre_score(name, self.a, 4)
        self.a['forecast_available'][2, 0] = False
        for name in s.NAMES[1:]:
            self.assertTrue(np.isfinite(s.pre_score(name, self.a, 4)).all())

    def test_less_than_two_cash_and_two_half_preserved(self):
        for count in (0, 1, 2):
            self.a['eligible'][:] = False
            self.a['eligible'][:, :count] = True
            for name in s.NAMES:
                chosen, _ = s.choices(name, self.a, self.m, 4)
                self.assertEqual([len(row) for row in chosen], [count if count == 2 else 0]*4)


class Selection(unittest.TestCase):
    def rows(self):
        metric = dict(cagr=.5, sharpe=1.5, max_drawdown=-.3)
        return [dict(id=i, name=name, training=metric.copy(), early=metric.copy(), recent=metric.copy())
                for i, name in enumerate(s.NAMES)]

    def test_first88_gates_each_reject_a_higher_cagr(self):
        for section, field, delta in (('training', 'sharpe', .05001),
                ('training', 'max_drawdown', .03001), ('early', 'cagr', .02501), ('recent', 'cagr', .02501)):
            rows = self.rows()
            rows[1]['training']['cagr'] = .7
            rows[1][section][field] -= delta
            self.assertEqual(s.select(rows)['id'], 0)
        rows = self.rows()
        rows[1]['training']['cagr'] -= 2e-12
        self.assertFalse(s.local.admitted(rows[1], rows[0]))

    def test_gate_boundaries_and_highest_admitted_cagr(self):
        rows = self.rows()
        candidate = rows[2]
        candidate['training'].update(cagr=.6, sharpe=1.45, max_drawdown=-.329999999)
        candidate['early']['cagr'] = .475
        candidate['recent']['cagr'] = .475
        self.assertEqual(s.select(rows)['id'], 2)
        rows[1]['training']['cagr'] = .7
        self.assertEqual(s.select(rows)['id'], 1)

    def test_ties_lower_id_and_default_control(self):
        rows = self.rows()
        self.assertEqual(s.select(rows)['id'], 0)
        rows[1]['training']['cagr'] += .5e-12
        self.assertEqual(s.select(rows)['id'], 0)
        rows[1]['training']['cagr'] = .6
        rows[2]['training']['cagr'] = .6+.5e-12
        self.assertEqual(s.select(rows)['id'], 1)
        rows[1]['training']['cagr'] = rows[2]['training']['cagr'] = .4
        self.assertEqual(s.select(rows)['id'], 0)

    def test_later_values_cannot_influence_selection_and_unknown_rows_rejected(self):
        rows = self.rows()
        rows[1]['later34'] = dict(cagr=999.)
        self.assertEqual(s.select(rows)['id'], 0)
        with self.assertRaises(AssertionError):
            s.select(rows[::-1])
        rows[1]['name'] = 'unknown'
        with self.assertRaises(AssertionError):
            s.select(rows)


class Provenance(unittest.TestCase):
    def arrays(self):
        return {str(i): np.array([np.nan, -0., float(i)]) for i in range(29)}

    def test_all29_byte_identity_detects_membership_dtype_shape_and_signed_zero(self):
        original = self.arrays()
        s.assert_arrays_identical(copy.deepcopy(original), original)
        for kind in ('missing', 'extra', 'dtype', 'shape', 'bytes'):
            changed = copy.deepcopy(original)
            if kind == 'missing': del changed['0']
            if kind == 'extra': changed['extra'] = changed['0']
            if kind == 'dtype': changed['0'] = changed['0'].astype(np.float32)
            if kind == 'shape': changed['0'] = changed['0'].reshape(1, 3)
            if kind == 'bytes': changed['0'][1] = 0.
            with self.assertRaises(AssertionError, msg=kind):
                s.assert_arrays_identical(changed, original)

    def test_source_data_hash_and_array_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root/'source'
            source_dir.mkdir()
            a = self.arrays()
            np.savez(source_dir/'DATA.npz', **a)
            (source_dir/'META.json').write_text('{}')
            fixed = dict(source_hashes={str(Path('source')/name): s.sha(source_dir/name)
                                      for name in ('DATA.npz', 'META.json')})
            with patch.object(s, 'ROOT', root), patch.object(s, 'design', return_value=fixed), \
                    patch.object(s.source, 'P', source_dir), patch.object(s.source, 'load', return_value=({}, a, None, None)):
                s.source_data()
                a['0'][1] = 0.
                with self.assertRaises(AssertionError): s.source_data()
                (source_dir/'META.json').write_text('{"changed":true}')
                with self.assertRaises(AssertionError): s.source_data()

    def test_registration_binds_manifest_self_digest_and_all21444_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'protected.json'
            members = {str(root/str(i)): '0'*64 for i in range(21444)}
            manifest.write_text(json.dumps(members))
            fixed = dict(protected_manifest=dict(path='protected.json', sha256=s.sha(manifest)), source_hashes={})
            plan = dict(design_sha256=s.DESIGN_SHA, configs=s.configs(), actual_fits_before_registration=0,
                protected_manifest=fixed['protected_manifest'], input_hashes={str(manifest): s.sha(manifest)})
            (root/'PLAN.json').write_text(json.dumps(plan))
            with patch.object(s, 'P', root), patch.object(s, 'ROOT', root), \
                    patch.object(s, 'design', return_value=fixed), patch.object(s.old, 'assert_hashes') as verify:
                s.registered()
                s.protected(fixed, members=True)
                self.assertEqual(verify.call_args.args[0], members)
                bad = copy.deepcopy(plan)
                bad['input_hashes'] = {}
                (root/'PLAN.json').write_text(json.dumps(bad))
                with self.assertRaises(AssertionError): s.registered()
                (root/'PLAN.json').write_text(json.dumps(plan))
                manifest.write_text('{}')
                with self.assertRaises(AssertionError): s.registered()


if __name__ == '__main__':
    unittest.main()
