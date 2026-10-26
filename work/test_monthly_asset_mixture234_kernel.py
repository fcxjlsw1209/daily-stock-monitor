"""Round234 fabricated-cache tests; never load actual feature or return files."""
import copy
import json
import unittest

import numpy as np

import monthly_asset_mixture234_kernel as k


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


class AssetMixtureTests(unittest.TestCase):
    def test_exact_support_and_effective_weight_order(self):
        a, m = fixture(); v, z = k.build_features(a, m)
        np.testing.assert_array_equal(v['support'], (a['baseline']-a['core'])/(1.-a['core_weight']))
        np.testing.assert_array_equal(v['weight'][0], [.5-(.025+.01)]*6)
        np.testing.assert_array_equal(v['weight'][1], [.48-.05]*6)
        np.testing.assert_array_equal(v['weight'][3], [.5-.025]*5+[.5-(.025+.01)])
        np.testing.assert_array_equal(v['candidate_delta'][3], v['weight'][3]*(v['candidate'][3]-v['support'][3]))
        wide=make_arrays([[-.5,-.375,-.25,-.125]]*2)
        remapped,_=k.build_features(wide,metadata(2,4,['S2','S3']))
        self.assertGreater(abs(remapped['candidate_delta'][1]).max(), .025)
        self.assertEqual(z['active_forecast_cells'], 13)


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
        self.assertEqual(zz['events'], 0); self.assertEqual(zz['both_group_rows'], 0)
        self.assertTrue(all(x.shape == (0, 6) for x in w.values()))


    def test_permutation_with_symbol_identity_equivariance(self):
        a, m = fixture(); v, _ = k.build_features(a, m); order = [4, 0, 5, 2, 1, 3]
        b = {key:(value[:, order] if value.ndim==2 and value.shape[1]==6 else value.copy()) for key,value in a.items()}
        mm = dict(m, symbols=[[row[j] for j in order] for row in m['symbols']])
        w, _ = k.build_features(b, mm)
        for key in v: np.testing.assert_array_equal(w[key], v[key][:, order])


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


    def test_common_score_mixture_not_separate_group_rank(self):
        s=[-.375,-.125,.125,.375]
        a=make_arrays([s,s]);m=metadata(2,4,['S3']);v,z=k.build_features(a,m)
        expected=[.5*(.5/3)+.5*0.-.5,.5*(1.5/3)+.5*0.-.5,
                  .5*(2.5/3)+.5*0.-.5,.5*1.+.5*.5-.5]
        np.testing.assert_array_equal(v['candidate'][1],expected)
        np.testing.assert_array_equal(v['control'][1],[-.375,-.125,.125,.375])
        self.assertLess(v['candidate'][1,2],0.)
        self.assertEqual(v['candidate'][1,3],.25)
        self.assertNotEqual(float(v['candidate'][1].mean()),0.)
        row=z['event_history'][1]
        self.assertEqual(row['stock_indices'],[0,1,2]);self.assertEqual(row['etf_indices'],[3])
        self.assertEqual(row['stock_less'],[0,1,2,3]);self.assertEqual(row['stock_equal'],[1,1,1,0])
        self.assertEqual(row['etf_less'],[0,0,0,0]);self.assertEqual(row['etf_equal'],[0,0,0,1])
        self.assertEqual(row['pooled_less'],[0,1,2,3]);self.assertEqual(row['pooled_equal'],[1]*4)
        self.assertEqual(row['branch'],'equal_mixture')

    def test_full_row_constant_preserves_nonzero_support(self):
        a=make_arrays([[.125]*4,[-.25]*4]);m=metadata(2,4,['S3']);v,z=k.build_features(a,m)
        for key in ('candidate','control','candidate_delta','control_delta'):
            np.testing.assert_array_equal(v[key],np.zeros((2,4)))
        self.assertTrue((v['support']!=0).all())
        self.assertEqual(z['constant_rows'],2);self.assertEqual(z['both_group_rows'],2)
        self.assertEqual(z['pooled_cdf_evaluations'],0);self.assertEqual(z['group_cdf_evaluations'],0)
        for row in z['event_history']:
            self.assertEqual(row['branch'],'constant_row')
            for key in ('stock_less','stock_equal','etf_less','etf_equal','pooled_less','pooled_equal'):
                self.assertEqual(row[key],[])

    def test_constant_subgroups_in_varied_row_are_not_exempt(self):
        s=[.125,.125,.375,.375]
        a=make_arrays([s,s]);v,z=k.build_features(a,metadata(2,4,['S2','S3']))
        np.testing.assert_array_equal(v['candidate'][1],[-.25,-.25,.25,.25])
        np.testing.assert_array_equal(v['control'][1],[-.25,-.25,.25,.25])
        self.assertTrue((v['candidate_delta'][1]!=0).all())
        self.assertEqual(z['constant_rows'],0);self.assertEqual(z['group_cdf_evaluations'],16)
        self.assertEqual(z['event_history'][1]['stock_equal'],[2,2,0,0])
        self.assertEqual(z['event_history'][1]['etf_equal'],[0,0,2,2])

    def test_absent_group_exact_control_copy_with_normal_delta(self):
        s=[-.25,-.125,.375]
        for etfs in ([],['S0','S1','S2'],['NOT_PRESENT']):
            a=make_arrays([s,s]);v,z=k.build_features(a,metadata(2,3,etfs))
            np.testing.assert_array_equal(v['candidate'],v['control'])
            np.testing.assert_array_equal(v['candidate_delta'],v['control_delta'])
            self.assertTrue(np.any(v['candidate_delta'][1]!=0))
            self.assertEqual(z['single_group_rows'],2);self.assertEqual(z['both_group_rows'],0)
            self.assertEqual(z['group_cdf_evaluations'],0);self.assertEqual(z['pooled_cdf_evaluations'],6)
            self.assertEqual(z['event_history'][1]['branch'],'single_group_copy')

    def test_singleton_pool_constant_precedes_single_group_copy(self):
        a=make_arrays([[.125],[-.25]])
        for etfs in ([],['S0'],['NOT_PRESENT']):
            v,z=k.build_features(a,metadata(2,1,etfs))
            for key in ('candidate','control','candidate_delta','control_delta'):
                np.testing.assert_array_equal(v[key],np.zeros((2,1)))
            self.assertEqual(z['constant_rows'],2);self.assertEqual(z['single_group_rows'],2)
            self.assertTrue(all(e['single_group'] and e['branch']=='constant_row' for e in z['event_history']))

    def test_group_distribution_replication_keeps_candidate_not_pooled_control(self):
        s=[-.375,-.125,.125,.375]
        a=make_arrays([s,s]);v,_=k.build_features(a,metadata(2,4,['S3']))
        repeated=[-.375,-.375,-.125,-.125,.125,.125,.375]
        b=make_arrays([repeated,repeated]);w,_=k.build_features(b,metadata(2,7,['S6']))
        ids=[0,2,4,6]
        np.testing.assert_array_equal(w['candidate'][1,ids],v['candidate'][1])
        self.assertFalse(np.array_equal(w['control'][1,ids],v['control'][1]))

    def test_equal_group_counts_exact_on_binary_fraction_fixture(self):
        s=[-.375,.125,-.125,.375]
        a=make_arrays([s,s]);v,_=k.build_features(a,metadata(2,4,['S2','S3']))
        np.testing.assert_array_equal(v['candidate'],v['control'])
        np.testing.assert_array_equal(v['candidate_delta'],v['control_delta'])

    def test_group_label_swap_keeps_equal_mixture(self):
        a,m=fixture();v,_=k.build_features(a,m)
        swapped=dict(m,etfs=[s for s in m['symbols'][0] if s not in m['etfs']])
        w,_=k.build_features(a,swapped)
        for key in v:np.testing.assert_array_equal(v[key],w[key])

    def test_global_positive_affine_map_preserves_cdf_targets(self):
        a,m=fixture();v,_=k.build_features(a,m)
        b=make_arrays(v['support']*.5+.0625,a['state_calm10'],a['state_weak_narrow'])
        w,_=k.build_features(b,m)
        for key in ('candidate','control'):np.testing.assert_array_equal(v[key],w[key])
        self.assertFalse(np.array_equal(v['candidate_delta'],w['candidate_delta']))

    def test_separate_group_translation_changes_common_x_cdfs(self):
        s=[-.25,-.125,0.,.125]
        a=make_arrays([s,s]);m=metadata(2,4,['S2','S3']);v,_=k.build_features(a,m)
        moved=[0.,.125,0.,.125]
        b=make_arrays([moved,moved]);w,_=k.build_features(b,m)
        self.assertFalse(np.array_equal(v['candidate'],w['candidate']))
        self.assertFalse(np.array_equal(v['control'],w['control']))

    def test_exact_cross_group_ties_and_signed_zeros(self):
        s=[-.25,0.,-.0,.25,0.]
        a=make_arrays([s,s]);v,z=k.build_features(a,metadata(2,5,['S3','S4']))
        np.testing.assert_array_equal(v['candidate'][:,1],v['candidate'][:,2])
        np.testing.assert_array_equal(v['candidate'][:,1],v['candidate'][:,4])
        self.assertEqual(z['event_history'][1]['pooled_equal'],[1,3,3,1,3])
        self.assertEqual(z['event_history'][1]['stock_equal'],[1,2,2,0,2])
        self.assertEqual(z['event_history'][1]['etf_equal'],[0,1,1,1,1])

    def test_adjacent_binary64_values_not_epsilon_ties(self):
        x=.125;y=np.nextafter(x,1.)
        s=[x,y,.25,-.25]
        a=make_arrays([s,s]);v,z=k.build_features(a,metadata(2,4,['S2','S3']))
        self.assertEqual(z['event_history'][1]['distinct_levels'],4)
        self.assertLess(v['candidate'][1,0],v['candidate'][1,1])
        self.assertLess(v['control'][1,0],v['control'][1,1])
        self.assertEqual(z['event_history'][1]['pooled_equal'],[1]*4)

    def test_canonical_pooled_counts_do_not_use_endpoint_rank(self):
        s=[-.25,-.25,0.,.125,.5]
        a=make_arrays([s,s]);v,z=k.build_features(a,metadata(2,5,['S4']))
        expected=[(0+.5*2)/5-.5,(0+.5*2)/5-.5,(2+.5*1)/5-.5,
                  (3+.5*1)/5-.5,(4+.5*1)/5-.5]
        np.testing.assert_array_equal(v['control'][1],expected)
        self.assertNotEqual(v['control'][1,-1],.5)
        self.assertEqual(z['event_history'][1]['pooled_less'],[0,0,2,3,4])

    def test_synthetic49_domain_uses32stocks17etfs_and_all_ineligible_names(self):
        s=np.arange(49,dtype=float)/64.-.375
        a=make_arrays([s,s]);m=metadata(2,49,['S%d'%j for j in range(32,49)])
        a['eligible'][:,3:]=False;v,z=k.build_features(a,m)
        self.assertEqual(z['stock_cells'],64);self.assertEqual(z['etf_cells'],34)
        self.assertEqual(z['checked_cells'],98);self.assertEqual(z['group_cdf_evaluations'],196)
        row=z['event_history'][1]
        self.assertEqual(row['stock_less'],list(range(32))+[32]*17)
        self.assertEqual(row['etf_less'],[0]*32+list(range(17)))
        expected=[(j+.5)/49-.5 for j in range(49)]
        np.testing.assert_array_equal(v['control'][1],expected)
        self.assertTrue(np.any(v['candidate_delta'][1,3:]!=0))

    def test_monotone_targets_and_bounds(self):
        s=[-.5,.25,-.125,.5,0.,.25,-.375]
        a=make_arrays([s,s]);v,_=k.build_features(a,metadata(2,7,['S0','S3']))
        ids=np.argsort(v['support'][1],kind='stable')
        for key in ('candidate','control'):
            self.assertTrue((np.diff(v[key][1,ids])>=0).all())
            self.assertTrue((np.abs(v[key])<=.5).all())
        self.assertTrue((np.abs(v['candidate_delta'])<=.5).all())

    def test_compact_audit_exact_counts_json_and_no_duplicated_targets(self):
        a,m=fixture();v,z=k.build_features(a,m)
        self.assertEqual(set(v),set(k.VALUE_KEYS));self.assertEqual(z['schema'],'asset_mixture234_features_v1')
        self.assertEqual(z['both_group_rows'],4);self.assertEqual(z['single_group_rows'],0)
        self.assertEqual(z['constant_rows'],0);self.assertEqual(z['group_cdf_evaluations'],48)
        self.assertEqual(z['pooled_cdf_evaluations'],24)
        self.assertTrue(all(type(z[key]) is int for key in k.COUNT_FIELDS))
        self.assertEqual(json.loads(json.dumps(z,allow_nan=False)),z)
        self.assertFalse(z['portfolio_returns_calculated'])
        for event in z['event_history']:
            self.assertTrue(set(v).isdisjoint(event))
            for prefix,size in (('stock',event['stock_count']),('etf',event['etf_count']),('pooled',6)):
                self.assertTrue(all(0<=x<=size for x in event[prefix+'_less']))
                self.assertTrue(all(0<=x<=size for x in event[prefix+'_equal']))

    def test_meta_missing_dates_allowed_and_invalid_global_etfs_not_ignored_at_zero(self):
        a,m=fixture();self.assertNotIn('dates',m)
        v,z=k.build_features(a,m,0);self.assertTrue(all(x.shape==(0,6) for x in v.values()))
        self.assertEqual(z['event_history'],[])
        with self.assertRaises(ValueError):k.build_features(a,dict(m,etfs=['S0','S0']),0)
        with self.assertRaises(ValueError):k.build_features(a,dict(m,symbols=[]),0)

    def test_additive_support_component_preserves_core_not_modified_prescore(self):
        a,m=fixture();v,_=k.build_features(a,m)
        b=copy.deepcopy(a);offset=np.full_like(a['core'],.03125)
        b['core']=a['core']+offset;b['baseline']=a['baseline']+offset
        w,_=k.build_features(b,m)
        # This exactly representable offset leaves original B-C unchanged.
        for key in v:np.testing.assert_array_equal(v[key],w[key])
        b['modified_pre']=object()
        x,_=k.build_features(b,m)
        for key in v:np.testing.assert_array_equal(x[key],w[key])


if __name__=='__main__':
    unittest.main(verbosity=2)
