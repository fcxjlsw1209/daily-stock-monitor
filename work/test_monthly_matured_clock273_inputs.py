"""Synthetic-only tests; never load TRAIN_INPUTS, a price file or a model."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import monthly_matured_clock273_inputs as subject


class Poison:
    def __getitem__(self, key):
        raise AssertionError('unused future/private body accessed')

    def __float__(self):
        raise AssertionError('unused future/private scalar accessed')


def synthetic_fixture():
    calendar = pd.bdate_range('2010-10-01', '2026-10-02')
    sessions = [str(x.date()) for x in calendar]
    last = {str(k): str(g.iloc[-1].date()) for k, g in
            pd.Series(calendar, index=calendar).groupby(calendar.to_period('M'))}
    months = [str(x) for x in np.arange('2011-01', '2026-01', dtype='datetime64[M]')]
    event_months = [str(x) for x in np.arange('2016-08', '2026-09', dtype='datetime64[M]')]
    symbols = ['S%02d' % i for i in range(49)]
    def end(m):
        return last[str(np.datetime64(m, 'M') + np.timedelta64(1, 'M'))]
    tm = dict(dates=[last[m] for m in months], ends=[end(m) for m in months],
              symbols=[symbols[:] for _ in months])
    monthly = dict(dates=['2016-08-17'] + [last[m] for m in event_months],
                   ends=['2016-08-31'] + [end(m) for m in event_months[:-1]] + ['2026-09-24'],
                   symbols=[symbols[:] for _ in range(122)], etfs=['S48'], early_months=53, train_months=88)
    n = 180 * 49
    training = dict(train_plain=np.zeros((n, 10)), event_plain=np.zeros((122 * 49, 10)),
                    train_eligible=np.ones(n, bool), train_good=np.ones(n, bool),
                    dates=np.repeat(np.array(tm['dates'], dtype='datetime64[D]'), 49),
                    ends=np.repeat(np.array(tm['ends'], dtype='datetime64[D]'), 49),
                    event_dates=np.repeat(np.array(monthly['dates'], dtype='datetime64[D]'), 49),
                    offsets=np.arange(181, dtype=np.int64) * 49,
                    eligible=np.ones((122, 49), bool), common_good=np.ones(n, bool),
                    vol63=np.full(n, .013))
    for k in subject.TARGETS:
        training['target_' + k] = np.linspace(-.25, .25, n)
    # Source contains legitimate signed zeros and distinct NaN payloads which
    # must not be normalized by appending arrays or reconstructing old labels.
    training['target_simple_excess'][0] = -0.0
    training['target_log_excess'].view(np.uint64)[1] = 0x7ff8000000000042
    ix = pd.bdate_range('2025-10-01', '2026-10-02')
    t, c = np.arange(len(ix))[:, None], np.arange(50)[None, :]
    daily = .0002 * (c + 1) + .001 * np.sin(t * .13 + c)
    price = 100.0 * np.cumprod(1.0 + daily, axis=0)
    panel = dict(adj=pd.DataFrame(price, index=ix, columns=symbols + ['SPY']))
    return training, dict(training=tm, monthly=monthly, weekly={'preserve': 'metadata'}), panel, sessions


class AppenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_fixture()
        cls.full, cls.audit = subject.append_completed(*cls.fixture, asof='2026-08-31')

    def data(self):
        return list(deepcopy(self.fixture))

    def one(self, values):
        return subject.append_completed(*values, asof='2026-02-27')

    def exact(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_full_seven_cohorts_and_original_bytes(self):
        old, meta, _, _ = self.fixture
        out, audit = self.full, self.audit
        self.assertEqual(set(out), set(subject.TRAIN_KEYS))
        self.assertEqual(out['train_plain'].shape, (9163, 10))
        self.assertEqual(out['offsets'].shape, (188,))
        for k in subject.ROW_KEYS:
            self.exact(old[k], out[k][:8820])
        for k in subject.EVENT_KEYS:
            self.exact(old[k], out[k])
        self.exact(old['offsets'], out['offsets'][:181])
        self.assertEqual(audit['counts']['appended_months'], 7)
        self.assertEqual(audit['counts']['appended_rows'], 343)
        self.assertEqual(audit['counts']['independent_scalar_comparisons'], 2401)
        self.assertEqual(audit['training_metadata']['dates'][-7:],
                         ['2026-01-30', '2026-02-27', '2026-03-31', '2026-04-30',
                          '2026-05-29', '2026-06-30', '2026-07-31'])
        self.assertEqual(audit['training_metadata']['ends'][-1], '2026-08-31')
        self.assertEqual(audit['training_metadata']['symbols'][:180], meta['training']['symbols'])
        self.assertFalse(audit['source_files_authenticated'])
        self.assertFalse(audit['portfolio_returns_computed'])
        self.assertTrue(audit['label_returns_computed'])
        self.assertFalse(audit['original_labels_recomputed'])

    def test_new_features_eligibility_are_event_snapshot_bytes(self):
        old, meta, _, _ = self.fixture
        for i, row in enumerate(self.audit['appended_months']):
            j = row['event_index']
            self.exact(self.full['train_plain'][8820 + i * 49:8820 + (i + 1) * 49],
                       old['event_plain'][j * 49:(j + 1) * 49])
            self.exact(self.full['train_eligible'][8820 + i * 49:8820 + (i + 1) * 49],
                       old['eligible'][j])
            self.assertEqual(self.audit['training_metadata']['symbols'][180 + i], meta['monthly']['symbols'][j])

    def test_all_four_targets_exact_frozen_helpers(self):
        old, meta, panel, _ = self.fixture
        ordinal, labels = subject._load_helpers()
        dates, ends = self.full['dates'][8820:], self.full['ends'][8820:]
        symbols = np.array([s for ss in self.audit['training_metadata']['symbols'][180:] for s in ss])
        eligible = self.full['train_eligible'][8820:]
        t, _ = ordinal.build_targets(panel['adj'], dates, ends, symbols, eligible)
        good = eligible & np.isfinite(t['return']) & np.isfinite(t['ordinal'])
        expected = labels.build_targets(panel['adj'], dates, ends, symbols, eligible, good,
                                        np.isfinite(self.full['train_plain'][8820:]).all(axis=1))
        self.exact(good, self.full['train_good'][8820:])
        self.exact(expected['vol63'], self.full['vol63'][8820:])
        self.exact(expected['good'], self.full['common_good'][8820:])
        for name in subject.TARGETS:
            self.exact(expected['targets'][name], self.full['target_' + name][8820:])

    def test_no_mature_extension_never_touches_panel(self):
        training, meta, _, sessions = self.data()
        for asof in ('2016-08-17', '2020-12-31', '2026-01-30', '2026-02-26'):
            out, audit = subject.append_completed(training, meta, Poison(), sessions, asof=asof)
            self.assertEqual(audit['counts']['appended_rows'], 0)
            self.assertIsNone(audit['price_window'])
            self.assertFalse(audit['label_returns_computed'])
            for k in subject.TRAIN_KEYS:
                self.exact(out[k], training[k])
                self.assertFalse(np.shares_memory(out[k], training[k]))

    def test_maturity_equal_end_and_no_later_new_labels(self):
        out, audit = self.one(self.data())
        self.assertEqual(out['train_plain'].shape, (8869, 10))
        self.assertEqual(audit['price_window']['last_session'], '2026-02-27')
        self.assertEqual(audit['appended_months'][0]['label_end'], '2026-02-27')
        for k in subject.ROW_KEYS:
            self.exact(out[k], self.full[k][:len(out[k])])

    def test_future_prices_and_unrelated_column_never_converted(self):
        values = self.data()
        adj = values[2]['adj'].astype(object)
        future = adj.index > pd.Timestamp('2026-02-27')
        adj.loc[future, :] = Poison()
        adj['UNRELATED'] = Poison()
        values[2]['adj'] = adj
        out, audit = self.one(values)
        expected, _ = self.one(self.data())
        for k in subject.TRAIN_KEYS:
            self.exact(out[k], expected[k])
        self.assertNotIn('UNRELATED', audit['price_window']['present_symbols'])

    def test_missing_eligible_interior_price_excludes_whole_month(self):
        values = self.data()
        values[2]['adj'].loc['2026-02-10', 'S01'] = np.nan
        out, audit = self.one(values)
        self.assertFalse(out['train_good'][8820:].any())
        self.assertFalse(out['common_good'][8820:].any())
        self.assertEqual(audit['ordinal']['months'][0]['status'], 'whole_month_excluded')
        for k in subject.TARGETS:
            self.assertTrue(np.isnan(out['target_' + k][8820:]).all())

    def test_missing_spy_path_excludes_whole_month(self):
        values = self.data()
        values[2]['adj'].loc['2026-02-09', 'SPY'] = 0.
        out, _ = self.one(values)
        self.assertFalse(out['train_good'][8820:].any())

    def test_missing_noneligible_price_does_not_shrink_eligible_cohort(self):
        values = self.data()
        values[0]['eligible'][114, 0] = False
        values[2]['adj'].loc['2026-02-09', 'S00'] = np.nan
        out, _ = self.one(values)
        self.assertFalse(out['train_good'][8820])
        self.assertTrue(out['train_good'][8821:].all())
        self.assertTrue(out['common_good'][8821:].all())

    def test_missing_eligible_column_excludes_whole_month(self):
        values = self.data()
        values[2]['adj'] = values[2]['adj'].drop(columns=['S11'])
        out, audit = self.one(values)
        self.assertFalse(out['train_good'][8820:].any())
        self.assertNotIn('S11', audit['price_window']['present_symbols'])

    def test_common_mask_minimum_two_and_finite_feature_policy(self):
        values = self.data()
        values[0]['eligible'][114, :] = False
        values[0]['eligible'][114, :2] = True
        values[0]['event_plain'][114 * 49, 2] = np.nan
        out, _ = self.one(values)
        self.assertEqual(int(out['train_good'][8820:].sum()), 2)
        self.assertFalse(out['common_good'][8820:].any())
        values[0]['event_plain'][114 * 49, 2] = 0.
        out, _ = self.one(values)
        self.assertEqual(int(out['common_good'][8820:].sum()), 2)
        for k in subject.TARGETS:
            self.assertEqual(float(out['target_' + k][8820:8822].sum()), 0.)

    def test_zero_vol_excludes_row_not_whole_path(self):
        values = self.data()
        values[2]['adj']['S00'] = 100.
        out, _ = self.one(values)
        self.assertTrue(out['train_good'][8820:].all())
        self.assertEqual(out['vol63'][8820], 0.)
        self.assertFalse(out['common_good'][8820])
        self.assertTrue(out['common_good'][8821:].all())

    def test_clip_then_center_and_loss_target(self):
        values = self.data()
        values[2]['adj'].loc['2026-02-27', 'S00'] *= 5.
        values[2]['adj'].loc['2026-02-27', 'S01'] *= .2
        out, audit = self.one(values)
        self.assertGreater(audit['targets']['clip_counts']['simple_excess'], 0)
        self.assertGreater(audit['targets']['clip_counts']['vol_scaled'], 0)
        self.assertNotEqual(out['target_loss_utility'][8821], out['target_simple_excess'][8821])
        self.assertTrue(audit['independent_label_oracle_passed'])

    def test_missing_official_price_session_hard_fails(self):
        values = self.data()
        values[2]['adj'] = values[2]['adj'].drop(pd.Timestamp('2026-02-09'))
        with self.assertRaisesRegex(ValueError, 'official calendar'):
            self.one(values)

    def test_missing_64_session_history_hard_fails(self):
        values = self.data()
        values[2]['adj'] = values[2]['adj'].loc['2026-01-01':]
        with self.assertRaisesRegex(ValueError, 'official calendar'):
            self.one(values)

    def test_wrong_month_end_and_truncated_label_rejected(self):
        values = self.data()
        values[1]['monthly']['ends'][114] = '2026-02-26'
        with self.assertRaisesRegex(ValueError, 'following-session'):
            self.one(values)
        values = self.data()
        values[1]['monthly']['ends'][114] = '2026-01-30'
        with self.assertRaisesRegex(ValueError, 'precede'):
            self.one(values)

    def test_bad_asof_types_and_future_rejected(self):
        for asof in ('2026-09-24', '2026-02-28', '2026-2-27', 20260227, True):
            with self.subTest(asof=asof), self.assertRaises(ValueError):
                subject.append_completed(*self.data(), asof=asof)

    def test_shared_features_eligibility_and_identity_must_match(self):
        for mode in ('features', 'eligibility', 'identity'):
            values = self.data()
            if mode == 'features':
                values[0]['event_plain'][49, 0] = -0.0
            elif mode == 'eligibility':
                values[0]['eligible'][1, 0] = False
            else:
                values[1]['monthly']['symbols'][1][0] = 'OTHER'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.one(values)

    def test_malformed_dtype_shape_offsets_and_extra_keys(self):
        for key in ('dtype', 'shape', 'offset', 'extra', 'date'):
            values = self.data()
            if key == 'dtype': values[0]['event_plain'] = values[0]['event_plain'].astype(np.float32)
            if key == 'shape': values[0]['train_plain'] = values[0]['train_plain'][:-1]
            if key == 'offset': values[0]['offsets'][3] += 1
            if key == 'extra': values[0]['new_target'] = np.zeros(8820)
            if key == 'date': values[0]['dates'][0] += np.timedelta64(1, 'D')
            with self.subTest(key=key), self.assertRaises(ValueError): self.one(values)

    def test_duplicate_identity_and_calendar_rejected(self):
        values = self.data()
        values[1]['monthly']['symbols'][114][1] = values[1]['monthly']['symbols'][114][0]
        with self.assertRaises(ValueError): self.one(values)
        values = self.data()
        values[3].insert(0, values[3][0])
        with self.assertRaises(ValueError): self.one(values)

    def test_append_twice_is_not_silently_accepted(self):
        values = self.data()
        values[0] = deepcopy(self.full)
        values[1]['training'] = deepcopy(self.audit['training_metadata'])
        with self.assertRaises(ValueError): self.one(values)

    def test_input_unchanged_and_output_does_not_alias(self):
        values = self.data()
        before = subject._snapshot(values[0])
        meta = deepcopy(values[1]); prices = values[2]['adj'].copy(deep=True)
        out, audit = self.one(values)
        self.assertEqual(before, subject._snapshot(values[0]))
        self.assertEqual(meta, values[1])
        pd.testing.assert_frame_equal(prices, values[2]['adj'], check_exact=True)
        out['train_plain'][0, 0] = 999.
        audit['training_metadata']['symbols'][0][0] = 'EDIT'
        self.assertEqual(before, subject._snapshot(values[0]))
        self.assertEqual(meta, values[1])

    def test_independent_oracle_detects_corrupted_target(self):
        ordinal, labels = subject._load_helpers()
        original = labels.build_targets
        def corrupt(*args):
            result = original(*args)
            result['targets']['simple_excess'][0] = np.nextafter(result['targets']['simple_excess'][0], np.inf)
            return result
        labels.build_targets = corrupt
        with mock.patch.object(subject, '_load_helpers', return_value=(ordinal, labels)):
            with self.assertRaisesRegex(ValueError, 'independent target_simple_excess'):
                self.one(self.data())

    def test_dependency_inventory_covers_actual_label_implementations(self):
        root = Path(subject.__file__).resolve().parents[1]
        expected = (
            root / 'outputs/monthly_core_expansion/ordinal_model.py',
            root / 'outputs/monthly_target_definition70/labels.py',
            root / 'outputs/monthly_architecture_restart/residual_model.py')
        files = subject.dependency_files()
        self.assertEqual(files, expected)
        self.assertEqual(len(files), len(set(files)))
        for path in files:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             subject.HELPER_SHA256[path.relative_to(root).as_posix()])
        ordinal, labels = subject._load_helpers()
        self.assertEqual({Path(ordinal.__file__), Path(labels.__file__), Path(ordinal._NUM.__file__)},
                         set(files))

    def test_dependency_listing_is_metadata_only_and_drift_rejected_before_import(self):
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('listing read bytes')):
            with mock.patch.object(subject.importlib.util, 'spec_from_file_location',
                                   side_effect=AssertionError('listing imported code')):
                self.assertEqual(len(subject.dependency_files()), 3)
        changed = dict(subject.HELPER_SHA256)
        changed['outputs/monthly_architecture_restart/residual_model.py'] = '0' * 64
        with mock.patch.object(subject, 'HELPER_SHA256', changed):
            with mock.patch.object(subject.importlib.util, 'spec_from_file_location') as loader:
                with self.assertRaisesRegex(ValueError, 'frozen label helper SHA mismatch'):
                    subject._load_helpers()
                loader.assert_not_called()

    def test_host_style_metadata_install_and_physical_kernel_handoff(self):
        # Exercises only array/layout preparation, never a model fit or host
        # loader. The inputs are generated by synthetic_fixture above.
        import monthly_matured_clock273_kernel as kernel
        training, combined, panel, sessions = self.data()
        combined['training']['pool_origins'] = ['synthetic_existing_provenance'] * 180
        original_meta = deepcopy(combined)
        extended, audit = subject.append_completed(training, combined, panel, sessions,
                                                   asof=combined['monthly']['dates'][-1])
        augmented = deepcopy(combined)
        augmented['training'] = audit['training_metadata']
        self.assertEqual(combined, original_meta)
        self.assertEqual(augmented['monthly'], combined['monthly'])
        self.assertEqual(augmented['weekly'], combined['weekly'])
        # Ancillary original provenance remains a historical 180-month anchor;
        # the appender only extends the three documented model identity axes.
        self.assertEqual(augmented['training']['pool_origins'], combined['training']['pool_origins'])
        E = training['eligible'].copy(); zero = np.zeros(E.shape)
        arrays = dict(baseline=zero.copy(), eligible=E, forecast_available=E.copy(),
                      raw_source_log=zero.copy(), rank_log_excess=kernel._rank(zero, E, zero),
                      model_available=E.copy(), raw_local_linear=zero.copy(),
                      raw_local_constant=zero.copy(), state_weak_narrow=np.ones(122, bool))
        with mock.patch.object(kernel, 'fit_models', side_effect=AssertionError('no model fits')):
            for n in (0, 1, 53, 88, 122):
                t, m, a, scope = kernel.physical_inputs(extended, augmented, arrays, n)
                self.assertEqual(t['event_plain'].shape, (n * 49, 10))
                self.assertEqual(a['baseline'].shape, (n, 49))
                self.assertEqual(m['monthly']['dates'], combined['monthly']['dates'][:n])
                expected = 0 if n < 2 else sum(
                    end <= combined['monthly']['dates'][n - 1]
                    for end in augmented['training']['ends'])
                self.assertEqual(scope['training_months'], expected)
                self.assertEqual(t['train_plain'].shape, (expected * 49, 10))
                self.assertEqual(m['training']['dates'], augmented['training']['dates'][:expected])
                for key in ('dates', 'ends', 'train_plain', 'common_good'):
                    self.exact(t[key], extended[key][:expected * 49])
                for key in subject.TARGETS:
                    self.exact(t['target_' + key], extended['target_' + key][:expected * 49])
            self.assertEqual(expected, 187)

    def test_ordinal_and_four_target_calls_not_used_without_new_month(self):
        values = self.data()
        with mock.patch.object(subject, '_load_helpers', side_effect=AssertionError('label arithmetic forbidden')):
            out, audit = subject.append_completed(values[0], values[1], Poison(), values[3], asof='2016-08-17')
        self.assertEqual(audit['counts']['label_builder_calls'], 0)
        self.assertEqual(audit['counts']['independent_scalar_comparisons'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
