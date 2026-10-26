"""Fabricated cache-only checks for explicit stock/ETF support calibration."""
import copy
import unittest

import numpy as np

import monthly_asset_aux210_kernel as k


def make_arrays(support, calm=None, weak=None):
    support = np.asarray(support, dtype=float)
    rows, width = support.shape
    calm = np.ones(rows, bool) if calm is None else np.asarray(calm, bool)
    weak = np.zeros(rows, bool) if weak is None else np.asarray(weak, bool)
    cw = np.where(calm, .5, .52)[:, None]
    core = np.zeros((rows, width))
    return dict(baseline=core+(1.-cw)*support, core=core, core_weight=cw,
        state_calm10=calm, state_weak_narrow=weak,
        extension_available=np.ones((rows, width), bool),
        forecast_available=np.ones((rows, width), bool), eligible=np.ones((rows, width), bool),
        rank_log_excess=np.zeros((rows, width)))


def metadata(rows, width, etfs=None):
    names = ['S%d' % j for j in range(width)]
    return dict(symbols=[names.copy() for _ in range(rows)],
                etfs=names[width//2:] if etfs is None else list(etfs))


def fixture():
    a = make_arrays([[-.25, -.125, .375, -.375, .0625, .25],
                    [-.25, -.25, .125, .125, .375, .375],
                    [.125, .125, .125, -.25, 0., .25],
                    [-.25, -.125, .375, -.375, .0625, .25]],
                   calm=[True, False, True, True], weak=[False, True, False, False])
    a['extension_available'][3, 0] = False
    a['forecast_available'][3, 1] = False
    a['eligible'][3, 2] = False
    a['rank_log_excess'][3, 3] = np.nan
    a['rank_log_excess'][3, 4] = np.inf
    return a, metadata(4, 6)


class AssetAuxTests(unittest.TestCase):
    def test_explicit_membership_and_group_rank_not_global(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        np.testing.assert_array_equal(v['candidate'][0], [-.5, 0., .5, -.5, 0., .5])
        self.assertEqual(z['rank_rows'], 8)
        self.assertEqual(z['grouped_rank_rows'], 8)
        self.assertEqual(z['stock_cells'], 12); self.assertEqual(z['etf_cells'], 12)
        self.assertEqual([g['asset_type'] for g in z['event_history'][0]['groups']], ['stock', 'etf'])
        self.assertEqual(z['event_history'][0]['groups'][1]['indices'], [3, 4, 5])
        self.assertFalse(np.array_equal(v['candidate'][0], np.argsort(np.argsort(v['support'][0]))/5.-.5))

    def test_exact_support_and_effective_weight_order(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        np.testing.assert_array_equal(v['support'], (a['baseline']-a['core'])/(1.-a['core_weight']))
        np.testing.assert_array_equal(v['weight'][0], [.5-(.025+.01)]*6)
        np.testing.assert_array_equal(v['weight'][1], [.48-.05]*6)
        np.testing.assert_array_equal(v['weight'][3], [.5-.025]*5+[.5-(.025+.01)])
        np.testing.assert_array_equal(v['candidate_delta'][3], v['weight'][3]*(v['candidate'][3]-v['support'][3]))
        self.assertGreater(abs(v['candidate_delta'][3]).max(), .025)
        self.assertEqual(z['active_forecast_cells'], 13)

    def test_affine_matched_tied_endpoints(self):
        s = [-.25, -.25, 0., .125, .5, .5, -.25, .25]
        a = make_arrays([s, s]); m = metadata(2, 8, ['S6', 'S7'])
        v, z = k.build_features(a, m)
        q = (np.array([1.5, 1.5, 3., 4., 5.5, 5.5])-1.)/5.-.5
        np.testing.assert_array_equal(v['candidate'][1, :6], q)
        support = v['support'][1, :6]
        expected = q.min()+(support-support.min())*(q.max()-q.min())/(support.max()-support.min())
        expected[support==support.min()] = q.min(); expected[support==support.max()] = q.max()
        np.testing.assert_array_equal(v['control'][1, :6], expected)
        self.assertFalse(np.array_equal(q, expected))

    def test_two_levels_exact_copy_per_group(self):
        a = make_arrays([[-.25, -.25, .125, .125, .125, -.375, .25]]*2)
        v, z = k.build_features(a, metadata(2, 7, ['S5', 'S6']))
        np.testing.assert_array_equal(v['candidate'], v['control'])
        np.testing.assert_array_equal(v['candidate_delta'], v['control_delta'])
        self.assertEqual(z['two_level_rows'], 4)

    def test_constant_group_only_is_noop_even_nonzero_support(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        for key in ('candidate', 'control', 'candidate_delta', 'control_delta'):
            np.testing.assert_array_equal(v[key][2, :3], np.zeros(3))
        self.assertTrue((v['support'][2, :3] != 0).all())
        self.assertTrue(np.any(v['candidate_delta'][2, 3:] != 0))
        self.assertEqual(z['constant_rows'], 1)

    def test_singleton_and_empty_groups_neutral(self):
        a = make_arrays([[.125], [-.25]])
        for etfs in ([], ['S0'], ['UNOBSERVED']):
            v, z = k.build_features(a, metadata(2, 1, etfs))
            self.assertEqual(z['rank_rows'], 2); self.assertEqual(z['constant_rows'], 2)
            self.assertEqual(z['empty_groups'], 2)
            for key in ('candidate', 'control', 'candidate_delta', 'control_delta'):
                np.testing.assert_array_equal(v[key], np.zeros((2, 1)))
            for event in z['event_history']:
                empty = next(g for g in event['groups'] if not g['size'])
                self.assertIsNone(empty['support_min']); self.assertIsNone(empty['candidate_max'])
                self.assertEqual(empty['distinct_levels'], 0); self.assertFalse(empty['constant'])

    def test_single_group_exact_204_transform(self):
        import monthly_aux_rank204_kernel as old
        a, m = fixture(); expected, _ = old.build_features(a)
        for etfs in ([], m['symbols'][0]):
            mm = dict(m, etfs=etfs)
            actual, audit = k.build_features(a, mm)
            for key in expected: np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(audit['rank_rows'], 4)

    def test_first_partial_targets_computed_deltas_zero(self):
        a, m = fixture(); v, _ = k.build_features(a, m)
        self.assertTrue(np.any(v['candidate'][0] != 0))
        self.assertTrue(np.any(v['control'][0] != 0))
        np.testing.assert_array_equal(v['candidate_delta'][0], np.zeros(6))
        np.testing.assert_array_equal(v['control_delta'][0], np.zeros(6))

    def test_rank_unfiltered_by_eligibility_forecast_or_extension(self):
        a, m = fixture(); v, _ = k.build_features(a, m)
        for key in ('eligible', 'extension_available', 'forecast_available'):
            b = copy.deepcopy(a); b[key][:] = False
            w, z = k.build_features(b, m)
            for field in ('support', 'candidate', 'control'): np.testing.assert_array_equal(v[field], w[field])
            self.assertFalse(z['ranking_eligibility_filtered']); self.assertFalse(z['new_availability_filter'])
        b = copy.deepcopy(a); b['eligible'][:] = False
        w, _ = k.build_features(b, m)
        self.assertTrue(np.any(w['candidate_delta'][3] != 0))

    def test_all_funding_states_and_each_active_mask(self):
        s = [[-.25, 0., .25]]*4
        a = make_arrays(s, calm=[False, False, True, True], weak=[False, True, False, True])
        m = metadata(4, 3, []); v, _ = k.build_features(a, m)
        expected = [.48, .48-.05, .5-(.025+.01), .5-(.025+.05)]
        np.testing.assert_array_equal(v['weight'], np.repeat(np.array(expected)[:, None], 3, axis=1))
        for key in ('extension_available', 'forecast_available', 'eligible', 'rank_log_excess'):
            b = copy.deepcopy(a); b[key][2, 1] = np.nan if key=='rank_log_excess' else False
            w, _ = k.build_features(b, m); self.assertEqual(w['weight'][2, 1], .5-.025)

    def test_cw_three_shapes_and_state_parity(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        for cw in (a['core_weight'][:, 0], np.broadcast_to(a['core_weight'], (4, 6)).copy()):
            b = dict(a, core_weight=cw); w, zz = k.build_features(b, m)
            self.assertEqual(z, zz)
            for key in v: np.testing.assert_array_equal(v[key], w[key])
        a['core_weight'][0, 0] = np.nextafter(.5, 1.)
        with self.assertRaisesRegex(ValueError, 'calm'): k.build_features(a, m)

    def test_metadata_invalid_and_no_ticker_inference(self):
        a, m = fixture()
        bad = [None, {}, dict(m, etfs='S3'), dict(m, etfs=['S3', 'S3']),
               dict(m, etfs=[1]), dict(m, etfs=['']), dict(m, symbols=m['symbols'][:-1])]
        for value in bad:
            with self.assertRaises(ValueError): k.build_features(a, value)
        for row in (['S0']*6, ['S0']*5, [1, 'B', 'C', 'D', 'E', 'F']):
            mm = copy.deepcopy(m); mm['symbols'][0] = row
            with self.assertRaises(ValueError): k.build_features(a, mm)
        mm = metadata(4, 6, ['S0', 'NOT_IN_POOL'])
        _, z = k.build_features(a, mm)
        self.assertEqual(z['etf_cells'], 4); self.assertEqual(z['stock_cells'], 20)

    def test_all_prefixes_physical_cache_and_metadata_truncation(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        for n in range(5):
            w, zz = k.build_features(a, m, limit=n)
            small = {key:value[:n].copy() for key,value in a.items()}
            mm = dict(m, symbols=m['symbols'][:n])
            x, zx = k.build_features(small, mm)
            self.assertEqual(zz, zx); self.assertEqual(zz['event_history'], z['event_history'][:n])
            for key in v:
                np.testing.assert_array_equal(w[key], v[key][:n]); np.testing.assert_array_equal(w[key], x[key])
                self.assertEqual(w[key].dtype, np.float64)

    def test_future_numeric_and_symbol_poison_not_consumed(self):
        a, m = fixture(); v, z = k.build_features(a, m, 2)
        b = {key:value.astype(object) for key,value in a.items()}
        for value in b.values(): value[2:] = 'FUTURE'
        mm = copy.deepcopy(m); mm['symbols'][2:] = [None, 'FUTURE']
        w, zz = k.build_features(b, mm, 2); self.assertEqual(z, zz)
        for key in v: np.testing.assert_array_equal(v[key], w[key])
        with self.assertRaises(ValueError): k.build_features(b, mm)
        for value in b.values(): value[:] = 'FUTURE'
        mm['symbols'] = [None]*4
        w, zz = k.build_features(b, mm, 0)
        self.assertEqual(zz['events'], 0); self.assertEqual(zz['grouped_rank_rows'], 0)
        self.assertTrue(all(x.shape == (0, 6) for x in w.values()))

    def test_permutation_with_symbol_identity_equivariance(self):
        a, m = fixture(); v, _ = k.build_features(a, m); order = [4, 0, 5, 2, 1, 3]
        b = {key:(value[:, order] if value.ndim==2 and value.shape[1]==6 else value.copy()) for key,value in a.items()}
        mm = dict(m, symbols=[[row[j] for j in order] for row in m['symbols']])
        w, _ = k.build_features(b, mm)
        for key in v: np.testing.assert_array_equal(w[key], v[key][:, order])

    def test_independent_group_location_scale_changes_keep_targets(self):
        a, m = fixture(); v, _ = k.build_features(a, m)
        s = v['support'].copy(); s[:, :3] = s[:, :3]*.5+.0625
        b = make_arrays(s, a['state_calm10'], a['state_weak_narrow'])
        w, _ = k.build_features(b, m)
        for key in ('candidate', 'control'): np.testing.assert_array_equal(w[key], v[key])

    def test_exact_ties_and_unequal_spacing_not_epsilon(self):
        s = [.125, np.nextafter(.125, 1.), .25, -.25, 0., .25]
        a = make_arrays([s, s]); v, z = k.build_features(a, metadata(2, 6))
        self.assertEqual(z['event_history'][1]['groups'][0]['distinct_levels'], 3)
        self.assertLess(v['candidate'][1, 0], v['candidate'][1, 1])
        self.assertNotEqual(v['control'][1, 1], v['candidate'][1, 1])

    def test_shape_limit_numeric_boolean_and_bounds_fail_closed(self):
        a, m = fixture()
        for limit in (-1, 5, True, 1., '1'):
            with self.assertRaises(ValueError): k.build_features(a, m, limit)
        for key in k.REQUIRED:
            b = copy.deepcopy(a); del b[key]
            with self.assertRaises(ValueError): k.build_features(b, m)
            b = copy.deepcopy(a); b[key] = np.array([1])
            with self.assertRaises(ValueError): k.build_features(b, m)
        for key in ('baseline', 'core', 'core_weight', 'rank_log_excess'):
            for invalid in (True, '0.1', 1+0j, None):
                b = copy.deepcopy(a); b[key] = b[key].astype(object); b[key].flat[0] = invalid
                with self.assertRaises(ValueError): k.build_features(b, m)
        for key in ('state_calm10', 'state_weak_narrow', 'eligible', 'extension_available', 'forecast_available'):
            b = copy.deepcopy(a); b[key] = b[key].astype(object); b[key].flat[0] = 1
            with self.assertRaises(ValueError): k.build_features(b, m)
        for invalid in (np.nan, np.inf, -.50001):
            b = copy.deepcopy(a); b['baseline'][0, 0] = invalid
            with self.assertRaises(ValueError): k.build_features(b, m)
        b = copy.deepcopy(a); b['baseline'][0, 0] = np.finfo(float).max; b['core'][0, 0] = -np.finfo(float).max
        with self.assertRaises(FloatingPointError): k.build_features(b, m)

    def test_original_arrays_metadata_unmodified_outputs_detached(self):
        a, m = fixture(); before, before_m = copy.deepcopy(a), copy.deepcopy(m)
        a['unused'] = object(); v, _ = k.build_features(a, m)
        for value in v.values(): value[:] = 999.
        for key in before: np.testing.assert_array_equal(a[key], before[key])
        self.assertEqual(m, before_m)

    def test_synthetic_49_stock_etf_domains_not_eligibility(self):
        s = np.arange(49, dtype=float)/64.-.375
        a = make_arrays([s, s]); m = metadata(2, 49, ['S%d' % j for j in range(32, 49)])
        a['eligible'][:, 3:] = False; v, z = k.build_features(a, m)
        np.testing.assert_array_equal(v['candidate'][1, :32], np.arange(32)/31.-.5)
        np.testing.assert_array_equal(v['candidate'][1, 32:], np.arange(17)/16.-.5)
        self.assertEqual(z['stock_cells'], 64); self.assertEqual(z['etf_cells'], 34)
        self.assertEqual(z['checked_cells'], 98)


if __name__ == '__main__':
    unittest.main(verbosity=2)
