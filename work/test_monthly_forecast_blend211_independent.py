"""Independent211 synthetic-only identity, mature-loss and score-slot tests."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_forecast_blend211_independent as oracle


def fixture(end='2019-02-28'):
    months = pd.date_range('2016-08-31', end, freq='ME')
    names = ['ONE', 'FUND', 'TWO']; width = len(names)
    tm = dict(dates=[str(d.date()) for d in months],
        ends=[str((d+pd.offsets.MonthEnd(1)).date()) for d in months],
        symbols=[names.copy() for _ in months])
    qm = dict(dates=['2016-08-17']+tm['dates'], ends=['2016-08-31']+tm['ends'],
              symbols=[names.copy()]+copy.deepcopy(tm['symbols']), etfs=['FUND'])
    own_pattern, peer_pattern = np.array([.25, -.25, 0.]), np.array([-.25, 0., .25])
    target = peer_pattern+.25*(own_pattern-peer_pattern)
    train = dict(offsets=np.arange(len(months)+1, dtype=int)*width,
        dates=np.repeat(np.array(tm['dates'], dtype='datetime64[D]'), width),
        ends=np.repeat(np.array(tm['ends'], dtype='datetime64[D]'), width),
        train_plain=np.zeros((len(months)*width, 10)), common_good=np.ones(len(months)*width, bool))
    for key in oracle.TARGETS: train['target_'+key] = np.tile(target, len(months))
    q = len(qm['dates'])*width
    source = dict(raw_candidate=np.tile(peer_pattern, len(qm['dates'])),
        raw_control=np.tile(own_pattern, len(qm['dates'])), raw_source_log=np.zeros(q),
        available=np.ones(q, bool), local_available=np.zeros(q, bool))
    return train, tm, qm, source, np.ones((len(qm['dates']), width), bool)


def prefix(inputs, n, physical=True):
    train, tm, qm, source, eligible = copy.deepcopy(inputs)
    width = eligible.shape[1]
    for key in ('dates', 'ends', 'symbols'): qm[key] = qm[key][:n]
    source = {k: v[:n*width] for k, v in source.items()}
    eligible = eligible[:n]
    if physical:
        cutoff = '%04d-12-31' % (int(qm['dates'][-1][:4])-1) if n else None
        months = [i for i, end in enumerate(tm['ends']) if cutoff is not None and end <= cutoff]
        flat = [j for i in months for j in range(int(train['offsets'][i]), int(train['offsets'][i+1]))]
        for key in ('dates', 'ends', 'symbols'): tm[key] = [tm[key][i] for i in months]
        for key in train:
            if key != 'offsets': train[key] = train[key][flat]
        train['offsets'] = np.r_[0, np.cumsum([len(ss) for ss in tm['symbols']], dtype=int)]
    return train, tm, qm, source, eligible


def latest(built):
    return built['audit']['annual'][-1]


def rank_fixture():
    source = np.array([[0., .1, .2, .3], [.2, .1, 0., -.1],
                       [0., .2, .1, .3], [0., .1, .2, .3]])
    shape = source.shape
    arrays = dict(baseline=np.zeros(shape), eligible=np.ones(shape, bool),
        forecast_available=np.ones(shape, bool), model_available=np.ones(shape, bool),
        local_available=np.zeros(shape, bool), raw_source_log=source.copy(),
        raw_local_constant=source+.02, raw_local_linear=source[:, ::-1].copy(),
        rank_log_excess=np.vstack([oracle.centered_rank(x) for x in source]),
        state_calm10=np.array([True, True, False, True]),
        state_weak_narrow=np.array([False, False, True, False]),
        extension_available=np.ones(shape, bool))
    forecast = dict(raw=dict(candidate=np.array([[.1, .1, .2, .2], [.3, .1, .1, .2],
        [.1, .2, .1, .2], [.2, .3, .1, 0.]]).ravel(),
        control=np.array([[0., .3, .2, .1], [.2, .3, .1, 0.],
                          [.3, .2, .1, 0.], [0., .2, .1, .3]]).ravel()),
        available=np.ones(source.size, bool))
    return arrays, forecast


class Tests(unittest.TestCase):
    def test_analytic_weight_and_counted_annual_fits(self):
        args = fixture(); built = oracle.reconstruct(*args)
        self.assertEqual([r['year'] for r in built['audit']['annual']], [2016, 2017, 2018, 2019])
        self.assertEqual([r['status'] for r in built['audit']['annual']], ['cold_start']*3+['learned'])
        row = latest(built)
        self.assertEqual(row['alpha'], .25); self.assertFalse(row['clipped'])
        self.assertEqual(row['calibration_months'], 28)
        self.assertEqual(row['latest_label_end'], '2018-12-31')
        check = oracle.check_forecasts(*args, built)
        self.assertTrue(all(check[k] for k in oracle.MODEL_FLAGS))
        self.assertEqual(check['new_model_fits'], 1); self.assertEqual(check['new_ridge_fits'], 0)
        self.assertFalse(check['computes_returns']); self.assertFalse(check['producer_modules_imported'])
        self.assertEqual(check['counts']['cached_forecast_rows'], args[4].size)

    def test_initial_partial_and_same_august_end_have_distinct_identities(self):
        args = fixture(); built = oracle.reconstruct(*args)
        self.assertEqual(latest(built)['skipped_initial_rows'], 3)
        self.assertEqual(latest(built)['months'][0]['date'], '2016-08-31')
        self.assertEqual(latest(built)['months'][0]['training_row_ids'], [0, 1, 2])
        self.assertEqual(latest(built)['months'][0]['source_row_ids'], [3, 4, 5])
        for row in built['audit']['annual']:
            self.assertNotIn('2016-08-17', [m['date'] for m in row['months']])

    def test_exact_17_18_month_threshold(self):
        for count in (17, 18):
            args = fixture(); train, tm, _, _, _ = args
            mature = [i for i, end in enumerate(tm['ends']) if end <= '2018-12-31']
            for i in mature[count:]: train['common_good'][3*i:3*i+3] = False
            row = latest(oracle.reconstruct(*args))
            self.assertEqual(row['calibration_months'], count)
            self.assertEqual(row['alpha'], .5 if count == 17 else .25)
            self.assertEqual(row['status'], 'cold_start' if count == 17 else 'learned')

    def test_month_equal_not_row_equal_loss(self):
        args = fixture(); train, tm, qm, source, eligible = args
        source['raw_candidate'][:] = 0.; source['raw_control'][:] = 1.
        train['target_log_excess'][:] = 0.; train['target_log_excess'][:3] = 1.
        eligible[1, 2] = False  # positive-loss month has two, other months have three rows.
        row = latest(oracle.reconstruct(*args))
        self.assertEqual(row['alpha'], 1./28.)
        self.assertNotEqual(row['alpha'], 2./(2+27*3))
        self.assertEqual(row['months'][0]['usable_rows'], 2)
        self.assertEqual(row['months'][0]['numerator_mean'], 1.)

    def test_zero_denominator_equal_sources_and_exact_copy(self):
        args = fixture(); args[3]['raw_control'] = args[3]['raw_candidate'].copy()
        built = oracle.reconstruct(*args); row = latest(built)
        self.assertEqual(row['status'], 'zero_denominator'); self.assertEqual(row['alpha'], .5)
        self.assertEqual(row['denominator'], 0.)
        for key in ('candidate', 'control'):
            self.assertEqual(built['raw'][key].tobytes(), args[3]['raw_candidate'].tobytes())
        self.assertEqual(built['audit']['counts']['annual_coefficient_fits'], 0)
        self.assertEqual(oracle.blend(1e308, 1e308, .5), 1e308)

    def test_clip_endpoints_and_endpoint_source_bytes(self):
        for coefficient, wanted in ((-1., 0.), (2., 1.)):
            args = fixture(); source = args[3]
            for i in range(len(args[1]['dates'])):
                p, o = source['raw_candidate'][3:6], source['raw_control'][3:6]
                args[0]['target_log_excess'][3*i:3*i+3] = p+coefficient*(o-p)
            built = oracle.reconstruct(*args); row = latest(built)
            self.assertEqual(row['alpha'], wanted); self.assertTrue(row['clipped'])
            last = [i for i, d in enumerate(args[2]['dates']) if d.startswith('2019')]
            target = source['raw_candidate'] if wanted == 0. else source['raw_control']
            for i in last:
                self.assertEqual(built['raw']['candidate'][3*i:3*i+3].tobytes(), target[3*i:3*i+3].tobytes())
        self.assertEqual(np.float64(oracle.blend(-0., 1., 0.)).tobytes(), np.float64(-0.).tobytes())
        self.assertEqual(np.float64(oracle.blend(1., -0., 1.)).tobytes(), np.float64(-0.).tobytes())

    def test_tiny_positive_denominator_has_no_floor(self):
        args = fixture(); args[3]['raw_candidate'][:] = 0.; args[3]['raw_control'][:] = 1e-100
        args[0]['target_log_excess'][:] = .25e-100
        row = latest(oracle.reconstruct(*args))
        self.assertGreater(row['denominator'], 0.); self.assertLess(row['denominator'], 1e-190)
        self.assertEqual(row['status'], 'learned'); self.assertEqual(row['alpha'], .25)

    def test_exact_triple_join_missing_mature_key_is_error(self):
        for change in ('symbol', 'end', 'offset'):
            args = fixture()
            if change == 'symbol': args[1]['symbols'][0][0] = 'OTHER'
            elif change == 'end':
                args[1]['ends'][0] = '2016-09-29'; args[0]['ends'][:3] = np.datetime64('2016-09-29')
            else: args[0]['offsets'][1] += 1
            with self.assertRaises(ValueError): oracle.reconstruct(*args)

    def test_invalid_common_labels_recorded_and_not_zero_filled(self):
        args = fixture(); train = args[0]
        train['common_good'][0] = False
        train['target_simple_excess'][1] = np.nan
        train['train_plain'][3, 0] = np.inf
        row = latest(oracle.reconstruct(*args))
        self.assertEqual(row['invalid_label_rows'], 3)
        self.assertEqual(row['small_month_rows'], 1)
        self.assertEqual(row['months'][0]['source_rows'], 3)
        self.assertEqual(row['months'][0]['targets'], [float(train['target_log_excess'][2])])
        self.assertEqual(row['months'][0]['numerator_terms'], [])
        self.assertFalse(row['months'][0]['accepted'])
        train['train_plain'] = train['train_plain'].astype(object)
        train['train_plain'][0] = 'unused because original common_good False'
        oracle.reconstruct(*args)
        train['train_plain'][4, 0] = 'nonnumeric consumed valid-base feature'
        with self.assertRaises(ValueError): oracle.reconstruct(*args)

    def test_no_local_available_or_current_membership_filter(self):
        args = fixture(); built = oracle.reconstruct(*args)
        args[3]['local_available'] = np.full(args[4].size, 'IGNORED', dtype=object)
        changed = oracle.reconstruct(*args)
        oracle.record_check(changed['audit'], built['audit'])
        for key in ('candidate', 'control'): np.testing.assert_array_equal(changed['raw'][key], built['raw'][key])
        args[4][-1] = False
        changed = oracle.reconstruct(*args)
        self.assertEqual(latest(changed)['alpha'], latest(built)['alpha'])
        self.assertTrue(changed['available'][-3:].all())

    def test_five_year_exclusive_start_and_maturity_end(self):
        args = fixture('2022-01-31'); row = latest(oracle.reconstruct(*args))
        self.assertEqual(row['lookback_exclusive_start'], '2016-12-31')
        self.assertEqual(row['months'][0]['date'], '2017-01-31')
        self.assertEqual(row['months'][-1]['date'], '2021-11-30')
        self.assertEqual(row['latest_label_end'], '2021-12-31')
        self.assertEqual(row['calibration_months'], 59)

    def test_future_and_outside_window_numeric_poison_not_read(self):
        args = fixture(); expected = oracle.reconstruct(*args)
        train = args[0]; future = train['ends'] > np.datetime64('2018-12-31')
        for key in ('common_good', 'train_plain') + tuple('target_'+k for k in oracle.TARGETS):
            train[key] = train[key].astype(object); train[key][future] = 'FUTURE DO NOT READ'
        got = oracle.reconstruct(*args)
        oracle.record_check(got['audit'], expected['audit'])
        for key in ('candidate', 'control'): np.testing.assert_array_equal(got['raw'][key], expected['raw'][key])
        empty = prefix(args, 0, physical=False)
        for key in ('common_good', 'train_plain') + tuple('target_'+k for k in oracle.TARGETS):
            empty[0][key][:] = 'EMPTY QUERY DO NOT READ'
        self.assertEqual(oracle.reconstruct(*empty)['audit']['query_rows'], 0)

    def test_current_unavailable_is_nan_and_unmatched_not_consumed(self):
        args = fixture(); args[3]['available'][3] = False
        for key in ('raw_candidate', 'raw_control', 'raw_source_log'): args[3][key][3] = np.nan
        args[1]['symbols'][0][0] = 'NO_FORECAST_NO_JOIN'
        built = oracle.reconstruct(*args)
        for key in ('candidate', 'control'): self.assertTrue(np.isnan(built['raw'][key][3]))
        args[3]['raw_control'][3] = np.inf
        with self.assertRaises(ValueError): oracle.reconstruct(*args)

    def test_finite_arithmetic_overflow_hard_failure(self):
        args = fixture(); args[3]['raw_candidate'][3] = -1e308; args[3]['raw_control'][3] = 1e308
        with self.assertRaises(FloatingPointError): oracle.reconstruct(*args)
        with self.assertRaises(FloatingPointError): oracle.blend(-1e308, 1e308, .5)
        self.assertEqual(oracle.blend(-1e308, 1e308, 1.), 1e308)

    def test_boolean_numeric_invalid_metadata_and_source_axes(self):
        for place in ('target', 'source', 'mask', 'identity', 'extra_same_month'):
            args = fixture()
            if place == 'target':
                args[0]['target_log_excess'] = args[0]['target_log_excess'].astype(object)
                args[0]['target_log_excess'][0] = True
            elif place == 'source':
                args[3]['raw_control'] = args[3]['raw_control'].astype(object); args[3]['raw_control'][0] = '0.2'
            elif place == 'mask': args[3]['available'] = args[3]['available'].astype(int)
            elif place == 'identity': args[2]['symbols'][1][1] = args[2]['symbols'][1][0]
            else: args[2]['dates'][2] = '2016-09-01'; args[2]['dates'][3] = '2016-09-30'
            with self.assertRaises(ValueError): oracle.reconstruct(*args)
        args = fixture(); args[0]['train_plain'] = args[0]['train_plain'][:, :9]
        with self.assertRaises(ValueError): oracle.reconstruct(*args)

    def test_artifact_raw_mask_audit_and_counts_corruption_rejected(self):
        args = fixture(); built = oracle.reconstruct(*args)
        for key in ('candidate', 'control'):
            bad = copy.deepcopy(built); bad['raw'][key][0] = np.nextafter(bad['raw'][key][0], np.inf)
            with self.assertRaises(ValueError): oracle.check_forecasts(*args, bad)
        bad = copy.deepcopy(built); bad['available'][0] = False
        with self.assertRaises(ValueError): oracle.check_forecasts(*args, bad)
        for key in oracle.COUNT_FIELDS:
            bad = copy.deepcopy(built); bad['audit']['counts'][key] += 1
            with self.assertRaises(ValueError): oracle.check_forecasts(*args, bad)
        bad = copy.deepcopy(built); bad['audit']['annual'][-1]['months'][0]['training_row_ids'][0] += 1
        with self.assertRaises(ValueError): oracle.check_forecasts(*args, bad)

    def test_inputs_and_original_targets_not_mutated(self):
        args = fixture(); before = copy.deepcopy(args)
        built = oracle.reconstruct(*args); oracle.check_forecasts(*args, built)
        built['raw']['control'][:] = 999.
        for pos in (0, 3):
            for key in args[pos]: np.testing.assert_array_equal(args[pos][key], before[pos][key])
        self.assertEqual(args[1:3], before[1:3]); np.testing.assert_array_equal(args[4], before[4])

    def test_rank_slot_uses_full_original_U_and_one_shared_ready_row(self):
        arrays, forecast = rank_fixture()
        ranks, deltas, ready = oracle.rank_oracle(arrays, forecast)
        proof = oracle.verify_rank_slot(arrays, forecast, ranks, deltas)
        self.assertTrue(all(proof[k] for k in oracle.RANK_FLAGS))
        for arm in oracle.NEW_RAW:
            np.testing.assert_array_equal(deltas[arm][0], 0.)
            self.assertLessEqual(np.abs(deltas[arm]).max(), .0125)
        forecast['raw']['candidate'][4] = np.nan; forecast['available'][4] = False
        rr, dd, ready = oracle.rank_oracle(arrays, forecast)
        for arm in oracle.NEW_RAW:
            self.assertFalse(ready[arm][1]); np.testing.assert_array_equal(dd[arm][1], 0.)
        bad = copy.deepcopy(dd); bad[oracle.ARMS[1]][0, 0] = 1e-16
        with self.assertRaises(ValueError): oracle.verify_rank_slot(arrays, forecast, rr, bad)

    def test_accounting_is_pinned_and_independent(self):
        self.assertTrue(callable(oracle.accounting().metrics))
        with patch.object(oracle, 'ACCOUNTING_SHA', '0'*64):
            with self.assertRaises(ValueError): oracle.accounting()

    def test_both_model_fixtures_all_prefixes_exact_audits_and_host_wiring(self):
        import monthly_forecast_blend211_model as production
        import test_monthly_forecast_blend211_model as producer_test
        import monthly_forecast_blend211_host as host
        for args in (fixture(), producer_test.fixture()):
            whole = oracle.reconstruct(*args)
            nrows, width = args[4].shape
            for n in range(nrows+1):
                for small in (prefix(args, n), host.prefix_inputs(*args, n)):
                    built = production.build_forecasts(*small)
                    proof = oracle.check_forecasts(*small, built)
                    host.require_model_audit(proof, built, expected_rows=n*width,
                        expected_years=len({d[:4] for d in small[2]['dates']}))
                    for key in ('candidate', 'control'):
                        np.testing.assert_array_equal(built['raw'][key], whole['raw'][key][:n*width])
        import test_monthly_forecast_blend211_host as host_test
        arrays, meta, forecast = host_test.fixture()
        ranks, deltas = host.rank_and_delta(host_test.SyntheticStudy(), arrays, forecast, len(meta['dates']))
        oracle.verify_rank_slot(arrays, forecast, ranks, deltas)


if __name__ == '__main__':
    unittest.main(verbosity=2)
