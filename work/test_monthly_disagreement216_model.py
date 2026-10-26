"""Synthetic-only invariants for annual own-security forecast-error reserves."""
import calendar
import copy
import json
import math
import unittest

import numpy as np

import monthly_disagreement216_model as model


def month_end(year, month):
    return '%04d-%02d-%02d' % (year, month, calendar.monthrange(year, month)[1])


def month_pair(year, month):
    after = year*12+month
    return month_end(year, month), month_end(after//12, after%12+1)


def fixture(n_history=18, symbols=('A', 'B', 'QQQ'), start=(2017, 6),
            current=((2019, 1), (2019, 6)), errors=None, initial=False, disagreements=None):
    """Deterministic caller arrays; no filesystem or real price/label inputs."""
    pairs = [month_pair((start[0]*12+start[1]-1+i)//12,
                        (start[0]*12+start[1]-1+i)%12+1) for i in range(n_history)]
    pairs += [month_pair(y, m) for y, m in current]
    if initial:
        pairs = [('2016-08-17', '2016-08-31')]+pairs
    width = len(symbols)
    query = dict(dates=[p[0] for p in pairs], ends=[p[1] for p in pairs],
                 symbols=[list(symbols) for _ in pairs])
    trainpairs = [p for p in pairs if p[0] != model.INITIAL]
    training = dict(dates=[p[0] for p in trainpairs], ends=[p[1] for p in trainpairs],
                    symbols=[list(symbols) for _ in trainpairs])
    total = len(trainpairs)*width
    train = dict(offsets=np.arange(len(trainpairs)+1, dtype=int)*width,
                 dates=np.repeat(np.array(training['dates']), width),
                 ends=np.repeat(np.array(training['ends']), width),
                 train_plain=np.zeros((total, 10)), common_good=np.ones(total, bool))
    train.update({'target_'+k: np.zeros(total) for k in model.TARGETS})
    raw = np.zeros((len(pairs), width))
    vals = list(range(-9, -9+n_history)) if errors is None else list(errors)
    assert len(vals) == n_history
    shift = int(initial)
    for i in range(n_history):
        for j in range(width):
            raw[shift+i, j] = float(vals[i]) if j != 1 else -1.
    for i in range(shift+n_history, len(pairs)):
        raw[i] = np.array([.5, -.25, .125, -.5, .75][:width])
    if initial:
        raw[0] = 999.
    diffs = np.ones_like(raw)
    historical_d = ([1. if i % 2 else -1. for i in range(n_history)]
                    if disagreements is None else list(disagreements))
    assert len(historical_d) == n_history
    for i, d in enumerate(historical_d): diffs[shift+i] = float(d)
    for i in range(shift+n_history, len(pairs)):
        diffs[i] = 1. if (i-shift-n_history) % 2 == 0 else -1.
    source = dict(raw_source_log=(raw-diffs).reshape(-1).copy(),
                  raw_local_linear=raw.reshape(-1).copy(), available=np.ones(raw.size, bool),
                  raw_control=np.full(raw.size, 9999.), raw_candidate=np.full(raw.size, -9999.),
                  local_available=np.zeros(raw.size, bool))
    use = np.ones(raw.shape, bool)
    return train, training, query, source, use


def physical_prefix(args, n):
    train, tm, qm, source, use = args
    width = use.shape[1]
    query = {k: copy.deepcopy(v[:n]) for k, v in qm.items()}
    keep = [i for i, end in enumerate(tm['ends']) if n and end <= str(int(qm['dates'][n-1][:4])-1)+'-12-31']
    offsets = np.asarray(train['offsets'])
    ids = np.array([j for i in keep for j in range(offsets[i], offsets[i+1])], dtype=int)
    training = {k: [copy.deepcopy(v[i]) for i in keep] for k, v in tm.items()}
    reduced = {k: np.asarray(v)[ids].copy() for k, v in train.items() if k != 'offsets'}
    reduced['offsets'] = np.r_[0, np.cumsum(np.array([len(tm['symbols'][i]) for i in keep], dtype=int))]
    smallsource = {k: np.asarray(v)[:n*width].copy() for k, v in source.items()}
    return reduced, training, query, smallsource, use[:n].copy()


def group(result, year=2019, symbol='A'):
    yearrow = next(row for row in result['audit']['annual'] if row['year'] == year)
    return next(row for row in yearrow['groups'] if row['symbol'] == symbol)


class ScalarTests(unittest.TestCase):
    def test_analytic_signed_means_and_empirical_shrinkage(self):
        result = model.calibration([2.]*9+[-1.]*9, [1.]*9+[-1.]*9)
        self.assertEqual(result['error_sum'], 9.)
        self.assertEqual(result['unconditional_mean'], .5)
        self.assertEqual(result['unconditional_penalty'], .5)
        self.assertEqual(result['conditional']['positive'],
                         dict(rows=9, error_sum=18., mean=1., penalty=1., empty_group_copy=False))
        self.assertEqual(result['conditional']['nonpositive'],
                         dict(rows=9, error_sum=-9., mean=0., penalty=0., empty_group_copy=False))

    def test_empty_group_copies_exact_mean_and_signed_zero_is_nonpositive(self):
        result = model.calibration([.1]*18, [0., -0.]+[-1.]*16)
        positive = result['conditional']['positive']
        self.assertEqual(positive['rows'], 0)
        self.assertEqual(positive['error_sum'], 0.)
        self.assertTrue(positive['empty_group_copy'])
        self.assertEqual(positive['mean'].hex(), result['unconditional_mean'].hex())
        self.assertEqual(result['conditional']['nonpositive']['rows'], 18)
        # Positive/nonpositive groups are symmetric, and zero is never positive.
        flipped = model.calibration([.1]*18, [1.]*18)
        self.assertTrue(flipped['conditional']['nonpositive']['empty_group_copy'])

    def test_canonical_fsum_cancellation_and_permutation(self):
        errors = [1e16, 1., -1e16]+[.1]*15
        differences = [1., 0., -1.]*6
        result = model.calibration(errors, differences)
        self.assertEqual(result['unconditional_mean'], math.fsum(errors)/18)
        for name, positive in (('positive', True), ('nonpositive', False)):
            values = [v for v,d in zip(errors,differences) if (d > 0) == positive]
            expected = math.fsum([math.fsum(values), 18*result['unconditional_mean']])/(len(values)+18)
            self.assertEqual(result['conditional'][name]['mean'], expected)
        self.assertEqual(result, model.calibration(errors[::-1], differences[::-1]))

    def test_signed_errors_not_absolute_and_nonnegative_penalties(self):
        result = model.calibration([-3.]*18, [1.]*9+[-1.]*9)
        self.assertEqual(result['unconditional_mean'], -3.)
        self.assertEqual(result['unconditional_penalty'], 0.)
        for part in result['conditional'].values():
            self.assertEqual(part['mean'], -3.)
            self.assertEqual(part['penalty'], 0.)

    def test_exact_zero_copy_and_negative_predictions(self):
        for value in (-0., .1, -.1):
            self.assertEqual(model.corrected(value, 0.).hex(), value.hex())
        self.assertEqual(model.corrected(-.1, .2), -.1-.2)
        self.assertEqual(model.corrected(.1, .2), .1-.2)
        with self.assertRaises(ValueError): model.corrected(1., -1.)

    def test_nonfinite_arithmetic_and_type_guards(self):
        invalid = [([], []), ([1.], []), ([True], [1.]), ([1.], ['1']),
                   ([math.nan], [1.]), ([1.], [math.inf])]
        for errors, differences in invalid:
            with self.assertRaises((ValueError, FloatingPointError)):
                model.calibration(errors, differences)
        with self.assertRaises(FloatingPointError): model.calibration([1e308]*18, [1.]*18)
        # Finite unconditional sum, but the explicit prior multiplication overflows.
        with self.assertRaises(FloatingPointError): model.calibration([1e308], [1.])
        with self.assertRaises(FloatingPointError): model.corrected(-1e308, 1e308)
        with self.assertRaises(FloatingPointError): model.corrected(math.inf, 0.)


class ModelTests(unittest.TestCase):
    def test_analytic_exact18_current_direction_and_per_year_reuse(self):
        args = fixture(errors=[2.]*9+[-1.]*9, disagreements=[1.]*9+[-1.]*9)
        out = model.build_forecasts(*args)
        g = group(out)
        self.assertEqual(g['own_months'], 18)
        self.assertEqual(g['status'], 'calibrated')
        self.assertEqual(g['unconditional_mean'], .5)
        self.assertEqual(g['conditional']['positive']['mean'], 1.)
        self.assertEqual(g['conditional']['nonpositive']['mean'], 0.)
        self.assertEqual(g['query_row_ids'], [54, 57])
        self.assertEqual(g['latest_label_end'], '2018-12-31')
        self.assertEqual(out['raw']['candidate'][54], -.5)
        self.assertEqual(out['raw']['candidate'][57], .5)
        self.assertEqual(out['raw']['control'][54], 0.)
        self.assertEqual(out['raw']['control'][57], 0.)
        self.assertEqual([q['disagreement_group'] for q in g['query_details']], ['positive','nonpositive'])
        self.assertEqual(group(out, symbol='B')['unconditional_penalty'], 0.)
        self.assertEqual(group(out, symbol='QQQ')['own_rows'], 18)

    def test_17_months_cold_both_exact_copy(self):
        args = fixture(n_history=17)
        args[3]['raw_local_linear'][-3] = -0.
        out = model.build_forecasts(*args)
        g = group(out)
        self.assertEqual(g['status'], 'cold_start')
        self.assertIsNone(g['error_sum'])
        self.assertIsNone(g['unconditional_mean'])
        self.assertEqual(sum(v['rows'] for v in g['conditional'].values()), 17)
        for v in g['conditional'].values():
            self.assertIsNone(v['error_sum']); self.assertIsNone(v['mean'])
            self.assertFalse(v['empty_group_copy']); self.assertEqual(v['penalty'], 0.)
        for key in ('candidate', 'control'):
            for j in g['query_row_ids']:
                self.assertEqual(out['raw'][key][j].item().hex(), args[3]['raw_local_linear'][j].item().hex())

    def test_original_U_and_source_availability_both_required(self):
        args = fixture(); args[4][0, 0] = False
        self.assertEqual(group(model.build_forecasts(*args))['own_rows'], 17)
        other = fixture(); other[3]['available'][0] = False
        other[3]['raw_local_linear'][0] = other[3]['raw_source_log'][0] = np.nan
        result = model.build_forecasts(*other)
        self.assertEqual(group(result)['own_rows'], 17)
        self.assertFalse(result['available'][0]); self.assertTrue(np.isnan(result['raw']['candidate'][0]))

    def test_current_U_and_local_available_do_not_filter_predictions(self):
        args = fixture(); baseline = model.build_forecasts(*args)
        args[4][-2:] = False; args[3]['local_available'][:] = False
        result = model.build_forecasts(*args)
        for key in baseline['raw']:
            np.testing.assert_array_equal(result['raw'][key], baseline['raw'][key])
        self.assertEqual(group(result)['own_rows'], 18)
        np.testing.assert_array_equal(result['available'], args[3]['available'])

    def test_source_uses_only_local_linear_and_pooled_raw(self):
        args = fixture(); baseline = model.build_forecasts(*args)
        args[3]['raw_control'] = np.array([object() for _ in args[3]['available']], dtype=object)
        args[3]['raw_candidate'] = args[3]['raw_control'].copy()
        result = model.build_forecasts(*args)
        self.assertEqual(result['audit'], baseline['audit'])
        for key in baseline['raw']:
            np.testing.assert_array_equal(result['raw'][key], baseline['raw'][key])
        # Changing only b changes grouping, while p and e remain unchanged.
        args[3]['raw_source_log'] = args[3]['raw_local_linear']-1.
        result = model.build_forecasts(*args)
        self.assertTrue(group(result)['conditional']['nonpositive']['empty_group_copy'])
        self.assertEqual(group(result)['unconditional_mean'], group(baseline)['unconditional_mean'])

    def test_labels_retained_not_recentered_or_reclipped(self):
        args = fixture(); args[0]['target_log_excess'][:] = 3.
        out = model.build_forecasts(*args); g = group(out)
        self.assertEqual(g['unconditional_mean'], -3.5)
        self.assertTrue(all(row['target'] == 3. for row in g['history']))
        self.assertEqual(out['raw']['candidate'][-6], .5)

    def test_invalid_common_or_any_original_target_or_plain_excluded(self):
        for key in ['common_good', 'train_plain']+['target_'+v for v in model.TARGETS]:
            args = fixture()
            if key == 'common_good': args[0][key][0] = False
            elif key == 'train_plain': args[0][key][0, 5] = np.nan
            else: args[0][key][0] = np.nan
            g = group(model.build_forecasts(*args))
            self.assertEqual(g['own_rows'], 17, key)
            self.assertEqual(g['joined_label_rows'], 18, key)
            self.assertEqual(g['invalid_label_rows'], 1, key)
            self.assertEqual(g['status'], 'cold_start', key)

    def test_no_two_peer_requirement_after_own_restriction(self):
        g = group(model.build_forecasts(*fixture(symbols=('A',))))
        self.assertEqual(g['status'], 'calibrated'); self.assertEqual(g['own_rows'], 18)

    def test_first_partial_excluded_without_requiring_its_label(self):
        args = fixture(initial=True); out = model.build_forecasts(*args); g = group(out)
        self.assertEqual(g['own_months'], 18); self.assertEqual(g['skipped_initial_rows'], 1)
        self.assertNotIn(model.INITIAL, [h['date'] for h in g['history']])
        np.testing.assert_array_equal(out['raw']['candidate'][:3], args[3]['raw_local_linear'][:3])

    def test_maturity_cutoff_inclusive_and_annual_frozen(self):
        args = fixture(n_history=19); out = model.build_forecasts(*args)
        self.assertEqual(group(out)['own_rows'], 18)
        self.assertEqual(group(out)['latest_label_end'], '2018-12-31')
        args[0]['target_log_excess'][-6:] = 1e200
        args[3]['raw_local_linear'][-3:] = 1.
        changed = model.build_forecasts(*args)
        for key in ('conditional', 'unconditional_mean', 'history', 'own_months'):
            self.assertEqual(group(changed)[key], group(out)[key])

    def test_exact_five_year_exclusive_signal_boundary(self):
        out = model.build_forecasts(*fixture(start=(2017, 12), current=((2023, 1),)))
        g = group(out, 2023)
        self.assertEqual(g['own_months'], 17)
        self.assertEqual(g['history'][0]['date'], '2018-01-31')
        self.assertNotIn('2017-12-31', [h['date'] for h in g['history']])

    def test_exact_security_ids_no_alias_or_other_security_leak(self):
        args = fixture(symbols=('A.B', 'A-B', 'QQQ'), errors=[2.]*18)
        out = model.build_forecasts(*args)
        self.assertEqual(group(out, symbol='A.B')['unconditional_penalty'], 2.)
        self.assertEqual(group(out, symbol='A-B')['unconditional_penalty'], 0.)
        args[1]['symbols'][0][0] = 'OTHER'
        with self.assertRaisesRegex(ValueError, 'Missing mature exact'): model.build_forecasts(*args)

    def test_missing_expected_key_and_wrong_label_end_fail(self):
        args = fixture(); args[1]['ends'][0] = '2017-07-30'; args[0]['ends'][:3] = '2017-07-30'
        with self.assertRaisesRegex(ValueError, 'Missing mature exact'): model.build_forecasts(*args)
        args = fixture(); args[1]['symbols'][0][0] = 'NO_MATCH'
        with self.assertRaisesRegex(ValueError, 'Missing mature exact'): model.build_forecasts(*args)

    def test_future_training_poison_ignored_and_all_physical_prefixes(self):
        args = fixture(); full = model.build_forecasts(*args)
        for n in range(len(args[2]['dates'])+1):
            small = model.build_forecasts(*physical_prefix(args, n))
            for key in full['raw']:
                np.testing.assert_array_equal(small['raw'][key], full['raw'][key][:n*3])
            np.testing.assert_array_equal(small['available'], full['available'][:n*3])
        poisoned = copy.deepcopy(args)
        for key in ('train_plain', 'common_good')+tuple('target_'+k for k in model.TARGETS):
            poisoned[0][key] = poisoned[0][key].astype(object)
            poisoned[0][key][-6:] = object()
        result = model.build_forecasts(*poisoned)
        self.assertEqual(full['audit'], result['audit'])
        for key in full['raw']:
            np.testing.assert_array_equal(full['raw'][key], result['raw'][key])

    def test_three_parameter_counts_empty_group_json_and_no_mutation(self):
        args = fixture(initial=True, disagreements=[1.]*18); before = copy.deepcopy(args)
        out = model.build_forecasts(*args); audit = out['audit']; c = audit['counts']
        self.assertEqual(set(c), set(model.COUNT_FIELDS))
        self.assertEqual(c['annual_cohorts'], len(audit['annual']))
        self.assertEqual(c['parameter_estimates'], 3*c['supported_year_security_groups'])
        self.assertEqual(c['own_history_groups'], c['supported_year_security_groups']+c['cold_start_groups'])
        self.assertEqual(c['joined_label_rows'], c['invalid_label_rows']+c['error_observations'])
        self.assertEqual(audit['disagreement_observations'], c['error_observations'])
        self.assertEqual(audit['new_model_fits'], c['parameter_estimates'])
        self.assertEqual(audit['predictive_model_fits'], c['parameter_estimates'])
        self.assertEqual(audit['new_ridge_fits'], 0); self.assertEqual(audit['new_tree_fits'], 0)
        self.assertEqual(c['supported_year_security_groups'], 3)
        self.assertEqual(c['parameter_estimates'], 9)
        self.assertTrue(group(out)['conditional']['nonpositive']['empty_group_copy'])
        json.dumps(audit, allow_nan=False)
        for pos in (0, 3):
            for key in args[pos]: np.testing.assert_array_equal(args[pos][key], before[pos][key])
        self.assertEqual(args[1], before[1]); self.assertEqual(args[2], before[2])
        np.testing.assert_array_equal(args[4], before[4])

    def test_schema_date_duplicate_offset_and_boolean_guards(self):
        mutations = [
            lambda a: a[0].__setitem__('offsets', a[0]['offsets']+1),
            lambda a: a[0]['dates'].__setitem__(0, '1999-01-01'),
            lambda a: a[1]['symbols'][0].__setitem__(1, 'A'),
            lambda a: a[2]['dates'].__setitem__(1, a[2]['dates'][0]),
            lambda a: a[2]['ends'].__setitem__(0, '2017-08-31'),
            lambda a: a[3].__setitem__('available', a[3]['available'].astype(int)),
            lambda a: a[3].__setitem__('raw_local_linear', np.zeros(2)),
            lambda a: a[3].__setitem__('raw_source_log', np.zeros(2)),
            lambda a: a[0].__setitem__('train_plain', np.zeros((60, 9))),
            lambda a: a[0].__setitem__('common_good', np.ones(len(a[0]['common_good']), int)),
            lambda a: a[3]['raw_local_linear'].__setitem__(0, np.nan),
            lambda a: a[3]['raw_source_log'].__setitem__(0, np.nan),
            lambda a: a[3].__setitem__('raw_source_log', np.full(60, '1')),
        ]
        for change in mutations:
            args = fixture(); change(args)
            with self.assertRaises((ValueError, FloatingPointError)): model.build_forecasts(*args)
        args = fixture()
        with self.assertRaises(ValueError): model.build_forecasts(*args[:4], args[4].astype(int))

    def test_historical_error_and_current_disagreement_overflow_not_missing(self):
        args = fixture(); args[3]['raw_local_linear'][0] = 1e308; args[0]['target_log_excess'][0] = -1e308
        with self.assertRaises(FloatingPointError): model.build_forecasts(*args)
        for i in (0, -1):
            args = fixture(); args[3]['raw_local_linear'][i] = 1e308; args[3]['raw_source_log'][i] = -1e308
            with self.assertRaises(FloatingPointError): model.build_forecasts(*args)

    def test_zero_disagreement_selects_nonpositive_and_zero_penalty_exact_copy(self):
        args = fixture(errors=[2.]*9+[-1.]*9, disagreements=[1.]*9+[0.]*9)
        args[3]['raw_local_linear'][-3] = -0.; args[3]['raw_source_log'][-3] = -0.
        out = model.build_forecasts(*args); g = group(out)
        self.assertEqual(g['query_details'][-1]['disagreement_group'], 'nonpositive')
        self.assertEqual(out['raw']['candidate'][-3].item().hex(), (-0.).hex())
        self.assertEqual(out['raw']['control'][-3], -.5)

    def test_permutation_equivariance_and_unsupported_current_identity(self):
        args = fixture(); out = model.build_forecasts(*args)
        changed = copy.deepcopy(args); permutation = [2, 0, 1]
        for meta in (changed[1], changed[2]):
            meta['symbols'] = [[row[j] for j in permutation] for row in meta['symbols']]
        total = len(args[0]['dates']); ids = np.arange(total).reshape(-1, 3)[:, permutation].reshape(-1)
        for key in changed[0]:
            if key != 'offsets': changed[0][key] = changed[0][key][ids].copy()
        for key in changed[3]: changed[3][key] = changed[3][key].reshape(-1, 3)[:, permutation].reshape(-1).copy()
        changed = changed[:4]+(changed[4][:, permutation].copy(),)
        permuted = model.build_forecasts(*changed)
        for key in out['raw']:
            np.testing.assert_array_equal(permuted['raw'][key].reshape(-1, 3), out['raw'][key].reshape(-1, 3)[:, permutation])
        fresh = fixture()
        for row in fresh[2]['symbols'][-2:]: row[0] = 'NEW'
        result = model.build_forecasts(*fresh); g = group(result, symbol='NEW')
        self.assertEqual(g['own_rows'], 0); self.assertEqual(g['status'], 'cold_start')
        self.assertEqual(result['raw']['candidate'][-3], .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
