"""Synthetic spacing tests; no NPZ, prices, fits, orders or returns."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import monthly_forecast_spacing269_kernel as k


SOURCE = Path(__file__).resolve().parents[1] / 'outputs/monthly_guidance_tweaks/tweaks.py'
SOURCE_SHA = '0664f5b7de4f5141fab7e11cba70f4f616aec44b1bb400703a127ed4da6a35a3'


def source_ranks():
    """Compile only the pinned pure rank function, not its research driver."""
    content = SOURCE.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA:
        raise AssertionError('Frozen source rank bytes changed')
    node = next(x for x in ast.parse(content).body
                if isinstance(x, ast.FunctionDef) and x.name == 'ranks')
    scope = {'pd': pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['ranks']


def ranks(values):
    values = list(values)
    n = len(values)
    return np.array([((float(2*sum(y < x for y in values)
                                    + sum(y == x for y in values)+1)/2.)/n)-.5
                     for x in values], dtype=np.float64)


def refresh(a):
    for i in range(len(a['dates'])):
        ids = np.flatnonzero(a['use'][i])
        if len(ids):
            for role in ('pooled', 'local'):
                a[role+'_rank'][i, ids] = ranks(a[role+'_raw'][i, ids])
    return a


def fixture(n=4, width=6):
    pool = np.array([-.04, -.01, -.01, .02, .08, .10])[:width]
    local = np.array([.02, -.02, -.01, .05, .07, .13])[:width]
    a = dict(pre=np.tile(np.linspace(-.2, .3, width), (n, 1)),
        pooled_raw=np.tile(pool, (n, 1)), pooled_rank=np.zeros((n, width)),
        local_raw=np.tile(local, (n, 1)), local_rank=np.zeros((n, width)),
        use=np.ones((n, width), bool), ready=np.ones(n, bool), weak=np.ones(n, bool),
        dates=['2016-08-17', '2016-08-31', '2016-09-30', '2016-10-31'][:n])
    return refresh(a)


def call(a, mode=k.ARMS[2]):
    return k.transform(**a, mode=mode)


def prefix(a, n):
    return {key: value[:n] for key, value in a.items()}


def exact(test, x, y):
    test.assertEqual(x.shape, y.shape)
    test.assertEqual(x.dtype, y.dtype)
    test.assertEqual(x.tobytes(), y.tobytes())


class CanonicalTests(unittest.TestCase):
    def test_source_rank_convention_ties_and_mask(self):
        old = source_ranks()
        rng = np.random.default_rng(269)
        for n in range(1, 50):
            values = rng.integers(-4, 5, n).astype(float)
            use = rng.random(n) > .25
            exact(self, old(values, use)[use], ranks(values[use]))

    def test_literal_affine_and_weights(self):
        a = fixture(); out, audit = call(a)
        i = 1; pieces = []
        for role, weight in (('pooled', .0375), ('local', .0125)):
            x, r = a[role+'_raw'][i], a[role+'_rank'][i]
            y = ((x-x.min())/(x.max()-x.min()))*(r.max()-r.min())+r.min()
            y[x == x.min()] = r[x == x.min()]; y[x == x.max()] = r[x == x.max()]
            exact(self, np.array(audit['events'][i][role]['new_rank']), y)
            pieces.append(weight*(y-r))
        expected = a['pre'][i].copy(); delta = pieces[0]+pieces[1]
        expected[delta != 0.] += delta[delta != 0.]
        exact(self, out[i], expected)

    def test_pooled_does_not_consume_local(self):
        a = fixture(); a['local_raw'][:] = np.inf; a['local_rank'][:] = np.nan
        _, audit = call(a, k.ARMS[1])
        self.assertTrue(all(x['local'] is None for x in audit['events']))
        self.assertEqual(audit['counts']['local_rank_checks'], 0)

    def test_incumbent_all_pre_bytes(self):
        a = fixture(); a['pre'][0] = [np.nan, np.inf, -np.inf, -0., 0., 1.]
        a['pooled_raw'][:] = np.inf; a['local_rank'][:] = np.nan
        out, audit = call(a, k.ARMS[0]); exact(self, out, a['pre'])
        self.assertEqual(audit['counts']['active_cells'], 0)

    def test_partial_exact_without_consuming_forecast(self):
        a = fixture(1); a['pooled_raw'][:] = np.inf; a['pooled_rank'][:] = np.nan
        a['local_raw'][:] = np.inf; a['pre'][0, 1] = -0.
        for mode in k.ARMS:
            out, audit = call(a, mode); exact(self, out, a['pre'])
            self.assertEqual(audit['events'][0]['branch'], 'initial_partial')

    def test_nonweak_and_not_ready_no_arithmetic(self):
        a = fixture(); a['weak'][1] = False; a['ready'][2] = False
        for name in ('pooled_raw', 'pooled_rank', 'local_raw', 'local_rank', 'pre'):
            a[name][1:3] = np.nan
        out, audit = call(a); exact(self, out[1:3], a['pre'][1:3])
        self.assertEqual([x['branch'] for x in audit['events'][1:3]], ['nonweak', 'not_ready'])

    def test_empty_use(self):
        a = fixture(); a['use'][1] = False; a['pooled_raw'][1] = np.inf
        out, audit = call(a); exact(self, out[1], a['pre'][1])
        self.assertEqual(audit['events'][1]['branch'], 'empty_use')

    def test_outside_u_preserves_nan_payload_and_zero(self):
        a = fixture(); a['use'][:, -2:] = False
        a['pre'][:, -2] = np.array([0x7ff8000000000017], np.uint64).view(float)[0]
        a['pre'][:, -1] = -0.
        for name in ('pooled_raw', 'pooled_rank', 'local_raw', 'local_rank'):
            a[name][:, -2:] = np.inf
        refresh(a); out, _ = call(a); exact(self, out[:, -2:], a['pre'][:, -2:])

    def test_constant_rank_copy(self):
        a = fixture(); a['pooled_raw'][:] = 9.; a['local_raw'][:] = -3.; refresh(a)
        out, audit = call(a); exact(self, out, a['pre'])
        self.assertEqual(audit['counts']['pooled_identity_rows'], 3)
        self.assertEqual(audit['counts']['local_identity_rows'], 3)

    def test_two_levels_copy_even_extreme_span(self):
        a = fixture(); a['pooled_raw'][:] = [-1e308]*2+[1e308]*4
        a['local_raw'][:] = [-1e308]*4+[1e308]*2; refresh(a)
        out, audit = call(a); exact(self, out, a['pre'])
        self.assertEqual(audit['counts']['pooled_affine_rows'], 0)

    def test_single_member_rank_is_positive_half(self):
        a = fixture(); a['use'][:] = False; a['use'][:, 3] = True; refresh(a)
        out, audit = call(a); exact(self, out, a['pre'])
        self.assertEqual(audit['events'][1]['pooled']['old_rank'], [.5])

    def test_endpoints_and_ties_exact(self):
        a = fixture(); a['pooled_raw'][:] = [-2., -2., -1., 0., 2., 2.]; refresh(a)
        _, audit = call(a)
        m = audit['events'][1]['pooled']
        for j in (0, 1, 4, 5):
            self.assertEqual(np.float64(m['old_rank'][j]).tobytes(), np.float64(m['new_rank'][j]).tobytes())

    def test_signed_zero_extremum_first_occurrence(self):
        a = fixture(); a['pooled_raw'][:] = [-0., 0., 1., 2., 3., 4.]; refresh(a)
        _, audit = call(a)
        self.assertEqual(np.float64(audit['events'][1]['pooled']['minimum']).tobytes(), np.float64(-0.).tobytes())

    def test_zero_delta_never_adds_positive_zero(self):
        a = fixture(); a['pre'][:] = -0.; a['pooled_raw'][:] = 1.; a['local_raw'][:] = 2.; refresh(a)
        out, audit = call(a); exact(self, out, a['pre'])
        self.assertEqual(audit['counts']['changed_pre_cells'], 0)

    def test_cancellation_does_not_write(self):
        a = fixture(); a['pre'][:] = -0.
        # Same endpoints and reversed spacing give exactly opposite map shifts;
        # pool 1/4 shift versus local 3/4 shift, with 3:1 fixed slot weights.
        a['pooled_raw'][:] = [0., .375, .375, .625, .625, 1.]
        a['local_raw'][:] = [0., .125, .125, .875, .875, 1.]
        refresh(a); out, audit = call(a)
        self.assertTrue(any(x != 0. for x in audit['events'][1]['pooled']['delta']))
        # Independently honor exact total-zero cells, whether all cells cancel
        # or only endpoints do under the specified binary64 arithmetic.
        zero = np.array(audit['events'][1]['delta']) == 0.
        self.assertTrue(zero.any()); exact(self, out[1, zero], a['pre'][1, zero])

    def test_counter_schema_and_json(self):
        a = fixture(); _, audit = call(a)
        self.assertEqual(tuple(audit), k.AUDIT_KEYS)
        self.assertEqual(tuple(audit['counts']), k.COUNT_FIELDS)
        self.assertEqual(audit['counts']['active_events'], 3)
        self.assertEqual(audit['counts']['pooled_mapped_cells'], 18)
        self.assertEqual(audit['counts']['local_rank_checks'], 3)
        for event in audit['events']:
            self.assertEqual(tuple(event), k.EVENT_KEYS)
            for role in ('pooled', 'local'):
                if event[role] is not None:self.assertEqual(tuple(event[role]), k.MAP_KEYS)
        json.dumps(audit, allow_nan=False)

    def test_inputs_readonly_and_immutable(self):
        a = fixture(); snap = {name: v.tobytes() for name, v in a.items() if isinstance(v, np.ndarray)}
        for v in a.values():
            if isinstance(v, np.ndarray):v.setflags(write=False)
        for mode in k.ARMS:call(a, mode)
        for name, before in snap.items():self.assertEqual(before, a[name].tobytes())

    def test_every_prefix_matches_and_future_is_irrelevant(self):
        a = fixture()
        for mode in k.ARMS:
            full = call(a, mode)
            for n in range(5):
                small = prefix(a, n); actual = call(small, mode)
                exact(self, actual[0], full[0][:n]); self.assertEqual(actual[1]['events'], full[1]['events'][:n])
                poisoned = deepcopy(a)
                for name in ('pre', 'pooled_raw', 'pooled_rank', 'local_raw', 'local_rank'):
                    poisoned[name][n:] = np.inf
                exact(self, call(prefix(poisoned, n), mode)[0], actual[0])

    def test_n0_no_forecast_access(self):
        class NoRead(np.ndarray):
            def __getitem__(self, key):raise AssertionError('numeric body accessed')
        a = fixture(0)
        for name in ('pooled_raw', 'pooled_rank', 'local_raw', 'local_rank'):
            a[name] = a[name].view(NoRead)
        for mode in k.ARMS:
            out, audit = call(a, mode); self.assertEqual(out.shape, (0, 6))
            self.assertEqual(audit['counts']['transform_calls'], 1)

    def test_exact_representable_affine_raw_invariance(self):
        a = fixture(); a['pooled_raw'][:] = [0., 1., 1., 2., 4., 8.]
        a['local_raw'][:] = [1., 2., 4., 4., 8., 16.]; refresh(a)
        b = deepcopy(a); b['pooled_raw'] = 2*b['pooled_raw']+16; b['local_raw'] = 4*b['local_raw']-32
        exact(self, call(a)[0], call(b)[0])


class InvalidTests(unittest.TestCase):
    def test_wrong_rank_one_ulp(self):
        a = fixture(); a['pooled_rank'][1, 1] = np.nextafter(a['pooled_rank'][1, 1], np.inf)
        with self.assertRaisesRegex(ValueError, 'ranks must match'):call(a)

    def test_wrong_rank_population(self):
        a = fixture(); a['use'][:, -1] = False
        with self.assertRaisesRegex(ValueError, 'ranks must match'):call(a)

    def test_consumed_nonfinite(self):
        for name in ('pre', 'pooled_raw', 'pooled_rank', 'local_raw', 'local_rank'):
            for bad in (np.nan, np.inf, -np.inf):
                a = fixture(); a[name][1, 2] = bad
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):call(a)

    def test_overflowing_nonconstant_span(self):
        a = fixture(); a['pooled_raw'][:] = [-1e308, -1e308, 0., 1., 1e308, 1e308]; refresh(a)
        with self.assertRaises(FloatingPointError):call(a)

    def test_numeric_shapes_and_dtypes(self):
        for name in ('pre', 'pooled_raw', 'pooled_rank', 'local_raw', 'local_rank'):
            for bad in (np.ones((4, 6), np.float32), np.ones((4, 5)), np.ones(24), [[0.]*6]*4):
                a = fixture(); a[name] = bad
                with self.subTest(name=name), self.assertRaises(ValueError):call(a)

    def test_boolean_shapes_and_dtypes(self):
        for name in ('use', 'ready', 'weak'):
            a = fixture(); a[name] = a[name].astype(int)
            with self.assertRaises(ValueError):call(a)
        a = fixture(); a['ready'] = np.ones((4, 1), bool)
        with self.assertRaises(ValueError):call(a)

    def test_strict_dates(self):
        for ds in (['2016-8-17', '2016-08-31', '2016-09-30', '2016-10-31'],
                   ['2016-08-17']*4, ['2016-07-31', '2016-08-17', '2016-08-31', '2016-09-30'],
                   ['2016-08-17', '2016-08-31', '2016-02-30', '2016-10-31'], [1, 2, 3, 4], []):
            a = fixture(); a['dates'] = ds
            with self.subTest(dates=ds), self.assertRaises(ValueError):call(a)

    def test_invalid_mode_and_width(self):
        for mode in ('unknown', None, 0, True):
            with self.assertRaises(ValueError):call(fixture(), mode)
        a = fixture(0, 0)
        with self.assertRaises(ValueError):call(a)


if __name__ == '__main__':
    unittest.main()
