"""Synthetic-only tests; AST-isolate frozen pure functions, never load data."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


P = Path(__file__).resolve().parent
O = P.parent
spec = importlib.util.spec_from_file_location('_budget_features_for_tests', P / 'features.py')
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def isolate(path, functions, constants=(), **dependencies):
    """Compile named definitions only; ignore all production imports/main code."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in functions or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id in constants for t in node.targets)]
    env = dict(np=np, pd=pd, **dependencies)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    if not set(functions) <= set(env):
        raise AssertionError('Missing frozen synthetic callback')
    return SimpleNamespace(**{key: env[key] for key in (*functions, *constants)})


strict = isolate(O / 'monthly_strict_relative/strict_study.py',
                 ('benchmark_map', 'relative_row'), ('SECTOR', 'FOREIGN'))
state = isolate(O / 'monthly_state_rank_research/state_rank.py', ('state_row',))
architecture = isolate(O / 'monthly_architecture_restart/study.py',
    ('rank_values', 'raw_frames', 'snapshot'), ('AUX',),
    ex=SimpleNamespace(t=SimpleNamespace(strict=strict)), old=state)
numeric = isolate(O / 'monthly_architecture_restart/residual_model.py', ('_days',))
labels = isolate(O / 'monthly_core_expansion/ordinal_model.py',
    ('_bool', '_timeline', '_ordinal', 'build_targets'), _NUM=numeric,
    ARMS=('return', 'ordinal'))


def synthetic():
    index = pd.bdate_range('2018-01-01', periods=460)
    etfs = list(strict.SECTOR.values()) + ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EEM']
    stocks = ['ASML'] + ['S%02d' % i for i in range(31)]
    symbols = stocks + etfs
    rng = np.random.default_rng(51014)
    daily = rng.normal(.0009, .003, size=(len(index), len(symbols)))
    daily[:, 1] = -.001
    prices = pd.DataFrame(100*np.exp(np.cumsum(daily, axis=0)), index=index, columns=symbols)
    phase = np.arange(len(index))[:, None]*.19 + np.arange(len(symbols))[None, :]*.37
    close = prices.copy()
    high = close*(1.02+.005*np.sin(phase))
    low = close*(.98+.004*np.cos(phase))
    volume = pd.DataFrame(1e6*(1.3+.2*np.sin(phase*.41)), index=index, columns=symbols)
    sectors = pd.DataFrame('Technology', index=index, columns=symbols)
    sectors[stocks[0]] = 'Energy'  # Frozen FOREIGN priority still maps ASML to XLK.
    sectors[stocks[2]] = 'Unknown'
    sectors[stocks[3]] = 'Healthcare'
    panel = dict(adj=prices, close=close, high=high, low=low, volume=volume, sectors=sectors)
    monthends = pd.Series(index, index=index).groupby(index.to_period('M')).last().tolist()
    monthends = [day for day in monthends if index.get_loc(day) >= 300]
    days, ends = monthends[:3], monthends[1:4]
    m = dict(dates=[str(day.date()) for day in (days[0], days[2])],
             symbols=[symbols.copy(), symbols.copy()], etfs=etfs)
    pools = [stocks[:8]+['SPY', 'QQQ', 'XLK'], symbols.copy(), stocks[:15]+etfs]
    tm = dict(dates=[str(day.date()) for day in days], ends=[str(day.date()) for day in ends], symbols=pools)
    return panel, m, tm


def prepared(panel, m, tm, **overrides):
    callbacks = dict(raw_frames=architecture.raw_frames, snapshot=architecture.snapshot,
                     label_builder=labels.build_targets)
    callbacks.update(overrides)
    return features.prepare(panel, m, tm, **callbacks)


def first_only(m, tm):
    m = copy.deepcopy(m); tm = copy.deepcopy(tm)
    m['dates'], m['symbols'] = m['dates'][:1], m['symbols'][:1]
    for key in ('dates', 'ends', 'symbols'):
        tm[key] = tm[key][:1]
    return m, tm


class FeatureContracts(unittest.TestCase):
    def setUp(self):
        self.panel, self.m, self.tm = synthetic()

    def test_exact_source_parity_shapes_dtypes_flattening_and_audit(self):
        a, audit = prepared(self.panel, self.m, self.tm)
        self.assertEqual(set(a), set(features.ARRAY_KEYS))
        offsets = np.r_[0, np.cumsum([len(ss) for ss in self.tm['symbols']])]
        np.testing.assert_array_equal(a['budget_train_offsets'], offsets)
        self.assertEqual(a['budget_train_offsets'].dtype, np.dtype('int64'))
        self.assertEqual(a['budget_train_dates'].dtype, np.dtype('datetime64[D]'))
        self.assertEqual(a['budget_train_ends'].dtype, np.dtype('datetime64[D]'))
        self.assertEqual(a['budget_event_dates'].dtype, np.dtype('datetime64[D]'))
        self.assertEqual(a['budget_train_symbols'].dtype.kind, 'U')
        self.assertEqual(a['budget_event_X'].shape, (2, 49, 4))
        for key in ('budget_train_X', 'budget_event_X', 'budget_train_target'):
            self.assertEqual(a[key].dtype, np.dtype('float64'))
        for key in ('budget_train_eligible', 'budget_event_eligible', 'budget_train_label_valid'):
            self.assertEqual(a[key].dtype, np.dtype('bool'))
        f, mom, good = architecture.raw_frames(self.panel)
        for kind, meta in (('train', self.tm), ('event', self.m)):
            for i, (day, ss) in enumerate(zip(meta['dates'], meta['symbols'])):
                row, eligible, _ = architecture.snapshot(day, ss, f, mom, good,
                    self.panel, set(self.m['etfs']), 'all')
                expected = np.column_stack([row[key] for key in features.SOURCE_KEYS])
                selector = slice(offsets[i], offsets[i+1]) if kind == 'train' else i
                np.testing.assert_array_equal(a['budget_'+kind+'_X'][selector], expected)
                np.testing.assert_array_equal(a['budget_'+kind+'_eligible'][selector], eligible)
        original, label_audit = labels.build_targets(self.panel['adj'], a['budget_train_dates'],
            a['budget_train_ends'], a['budget_train_symbols'], a['budget_train_eligible'])
        np.testing.assert_array_equal(a['budget_train_target'], original['ordinal']/2)
        np.testing.assert_array_equal(a['budget_train_label_valid'], np.isfinite(original['ordinal']))
        self.assertEqual(audit['label_audit'], label_audit)
        self.assertEqual(audit['feature_order'], ['M', 'R', 'V', 'P'])
        self.assertEqual(audit['normalization'], 'all')
        self.assertEqual(audit['training_rows'], int(offsets[-1]))
        self.assertTrue(audit['future_label_validity_never_filters_trading'])
        json.dumps(audit, allow_nan=False)

    def test_every_callback_sees_only_signal_prefix_and_same_day_reuses_frames(self):
        raw_days, snapshot_days = [], []
        def raw(panel):
            day = panel['adj'].index[-1]
            self.assertTrue(all(frame.index.max() <= day for frame in panel.values()))
            raw_days.append(day)
            return architecture.raw_frames(panel)
        def snap(day, ss, f, mom, good, panel, etfs, norm):
            self.assertEqual(norm, 'all')
            self.assertEqual(panel['adj'].index[-1], day)
            self.assertTrue(all(v.index[-1] == day for v in f.values()))
            self.assertTrue(all(v.index[-1] == day for v in mom.values()))
            self.assertEqual(good.index[-1], day)
            snapshot_days.append(day)
            return architecture.snapshot(day, ss, f, mom, good, panel, etfs, norm)
        prepared(self.panel, self.m, self.tm, raw_frames=raw, snapshot=snap)
        unique = sorted({pd.Timestamp(d) for d in self.tm['dates']+self.m['dates']})
        self.assertEqual(raw_days, unique)
        self.assertEqual(len(snapshot_days), len(self.tm['dates'])+len(self.m['dates']))

    def test_future_poison_changes_labels_only_and_unused_future_is_not_read(self):
        m, tm = first_only(self.m, self.tm)
        base, _ = prepared(self.panel, m, tm)
        altered = {key: frame.astype(object).copy() for key, frame in self.panel.items()}
        day = pd.Timestamp(m['dates'][0])
        for key in ('adj', 'close', 'high', 'low', 'volume'):
            altered[key].loc[altered[key].index > day] = 'unreadable future'
        altered['sectors'].loc[altered['sectors'].index > day] = 'future sector'
        got, audit = prepared(altered, m, tm)
        for key in ('budget_train_X', 'budget_event_X', 'budget_train_eligible', 'budget_event_eligible'):
            np.testing.assert_array_equal(got[key], base[key])
        self.assertTrue(base['budget_train_label_valid'].any())
        self.assertFalse(got['budget_train_label_valid'].any())
        self.assertEqual(audit['excluded_label_months'], 1)
        # No label or feature calls may depend on observations beyond label end.
        beyond = {key: frame.astype(object).copy() for key, frame in self.panel.items()}
        end = pd.Timestamp(tm['ends'][0])
        for frame in beyond.values():
            frame.loc[frame.index > end] = 'unused future'
        unchanged, _ = prepared(beyond, m, tm)
        for key in base:
            np.testing.assert_array_equal(unchanged[key], base[key])

    def test_latest42_change_recent_factors_but_not_momentum_or_own_sector(self):
        m, tm = first_only(self.m, self.tm)
        baseline, _ = prepared(self.panel, m, tm)
        day = pd.Timestamp(m['dates'][0]); pos = self.panel['adj'].index.get_loc(day)
        changed = {key: value.copy() for key, value in self.panel.items()}
        symbol = m['symbols'][0][0]
        recent = changed['adj'].index[pos-41:pos+1]
        changed['adj'].loc[recent, symbol] *= np.exp(np.sin(np.arange(42))*1.4)
        changed['close'].loc[recent, symbol] = changed['high'].loc[recent, symbol]
        other, _ = prepared(changed, m, tm)
        np.testing.assert_array_equal(other['budget_event_X'][..., :2], baseline['budget_event_X'][..., :2])
        self.assertFalse(np.array_equal(other['budget_event_X'][..., 2:], baseline['budget_event_X'][..., 2:]))
        self.assertEqual(other['budget_event_X'][0, 0, 3], .5)

    def test_full_pool_rank_includes_signal_ineligible_stock(self):
        m, tm = first_only(self.m, self.tm)
        day = pd.Timestamp(m['dates'][0]); pos = self.panel['adj'].index.get_loc(day)
        self.panel['adj'].iloc[pos-290, 0] = np.nan
        a, _ = prepared(self.panel, m, tm)
        self.assertFalse(a['budget_event_eligible'][0, 0])
        f, mom, good = architecture.raw_frames(self.panel)
        all_row, _, _ = architecture.snapshot(day, m['symbols'][0], f, mom, good,
            self.panel, set(m['etfs']), 'all')
        eligible_row, _, _ = architecture.snapshot(day, m['symbols'][0], f, mom, good,
            self.panel, set(m['etfs']), 'eligible')
        np.testing.assert_array_equal(a['budget_event_X'][0, :, 0], all_row['raw'])
        self.assertNotEqual(a['budget_event_X'][0, 0, 0], 0.)
        self.assertEqual(eligible_row['raw'][0], 0.)

    def test_missing_volume_price_is_neutral_and_does_not_add_eligibility_gate(self):
        m, tm = first_only(self.m, self.tm)
        day = pd.Timestamp(m['dates'][0]); symbol = m['symbols'][0][0]
        self.panel['high'].at[day, symbol] = self.panel['low'].at[day, symbol]
        a, _ = prepared(self.panel, m, tm)
        self.assertEqual(a['budget_event_X'][0, 0, 3], 0.)
        self.assertTrue(a['budget_event_eligible'][0, 0])
        self.assertTrue(np.isfinite(a['budget_event_X']).all())
        self.assertEqual(a['budget_event_X'][0, 2, 1], 0.)  # Unknown sector.
        for etf in m['etfs']:
            j = m['symbols'][0].index(etf)
            self.assertEqual(a['budget_event_X'][0, j, 1], 0.)

    def test_equal_features_keep_source_positive_tie_rank_and_zero_targets_valid(self):
        panel = self.panel
        path = 100*np.exp(np.arange(len(panel['adj']))*.001)
        for key in ('adj', 'close', 'high', 'low'):
            panel[key].iloc[:] = path[:, None]
        panel['high'] *= 1.01; panel['low'] *= .99
        panel['volume'].iloc[:] = 1e6
        a, audit = prepared(panel, self.m, self.tm)
        # Preserve the source order of operations, including its last bit.
        tied_rank = 25/49 - .5
        np.testing.assert_array_equal(a['budget_event_X'][..., 0], np.full((2, 49), tied_rank))
        self.assertAlmostEqual(tied_rank, 1/98, places=16)
        use = a['budget_train_label_valid']
        self.assertTrue(use.all())
        np.testing.assert_array_equal(a['budget_train_target'][use], 0.)
        self.assertEqual(audit['complete_label_months'], 3)

    def test_eligible_missing_future_path_excludes_entire_month_not_event_pool(self):
        m, tm = first_only(self.m, self.tm)
        base, _ = prepared(self.panel, m, tm)
        day = pd.Timestamp(tm['dates'][0]); pos = self.panel['adj'].index.get_loc(day)
        self.panel['adj'].iloc[pos+3, 0] = np.nan
        a, audit = prepared(self.panel, m, tm)
        self.assertFalse(a['budget_train_label_valid'].any())
        self.assertTrue(np.isnan(a['budget_train_target']).all())
        np.testing.assert_array_equal(a['budget_event_eligible'], base['budget_event_eligible'])
        np.testing.assert_array_equal(a['budget_event_X'], base['budget_event_X'])
        self.assertEqual(audit['training_records'][0]['label_status'], 'whole_month_excluded')

    def test_spy_invalidates_month_but_ineligible_future_missing_does_not(self):
        m, tm = first_only(self.m, self.tm)
        day = pd.Timestamp(tm['dates'][0]); pos = self.panel['adj'].index.get_loc(day)
        self.panel['adj'].iloc[pos+2, 1] = np.nan  # Declining S00 was already ineligible.
        a, _ = prepared(self.panel, m, tm)
        self.assertFalse(a['budget_train_eligible'][1])
        self.assertTrue(a['budget_train_label_valid'].any())
        self.panel['adj'].at[self.panel['adj'].index[pos+2], 'SPY'] = np.nan
        b, _ = prepared(self.panel, m, tm)
        self.assertFalse(b['budget_train_label_valid'].any())
        np.testing.assert_array_equal(a['budget_train_eligible'], b['budget_train_eligible'])

    def test_eligibility_exact295_observations_and_no_target_filtering(self):
        m = copy.deepcopy(self.m)
        m['dates'] = [str(self.panel['adj'].index[i].date()) for i in (293, 294)]
        tm = dict(dates=[], ends=[], symbols=[])
        a, audit = prepared(self.panel, m, tm)
        self.assertFalse(a['budget_event_eligible'][0].any())
        self.assertTrue(a['budget_event_eligible'][1].any())
        self.assertEqual(a['budget_train_X'].shape, (0, 4))
        self.assertEqual(audit['training_months'], 0)

    def test_reordered_pools_preserve_symbol_values_and_labels(self):
        baseline, _ = prepared(self.panel, self.m, self.tm)
        m, tm = copy.deepcopy(self.m), copy.deepcopy(self.tm)
        m['symbols'] = [ss[::-1] for ss in m['symbols']]
        tm['symbols'] = [ss[::-1] for ss in tm['symbols']]
        a, _ = prepared(self.panel, m, tm)
        np.testing.assert_array_equal(a['budget_event_X'][:, ::-1], baseline['budget_event_X'])
        np.testing.assert_array_equal(a['budget_event_eligible'][:, ::-1], baseline['budget_event_eligible'])
        for lo, hi in zip(a['budget_train_offsets'][:-1], a['budget_train_offsets'][1:]):
            for key in ('budget_train_X', 'budget_train_target', 'budget_train_eligible', 'budget_train_label_valid'):
                np.testing.assert_array_equal(a[key][lo:hi][::-1], baseline[key][lo:hi])

    def test_no_input_mutation_and_unused_cached_future_good_is_ignored(self):
        saved = {key: value.copy(deep=True) for key, value in self.panel.items()}
        m, tm = copy.deepcopy(self.m), copy.deepcopy(self.tm)
        class DoNotRead:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Future cached mask was read')
        self.tm['good'] = DoNotRead()
        a, audit = prepared(self.panel, self.m, self.tm)
        b, _ = prepared(self.panel, m, tm)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for key in self.panel:
            pd.testing.assert_frame_equal(saved[key], self.panel[key])
        self.assertEqual(self.m, m)
        self.assertEqual({k: self.tm[k] for k in tm}, tm)
        self.assertFalse(audit['cached_label_good_read'])

    def test_identity_axis_and_metadata_errors_are_explicit(self):
        cases = []
        m = copy.deepcopy(self.m); m['symbols'][0][0] = m['symbols'][0][1]; cases.append((self.panel, m, self.tm))
        m = copy.deepcopy(self.m); m['etfs'] = m['etfs'][:-1]; cases.append((self.panel, m, self.tm))
        m = copy.deepcopy(self.m); m['symbols'][0][0] = 'ABSENT'; cases.append((self.panel, m, self.tm))
        m = copy.deepcopy(self.m); m['dates'] = m['dates'][::-1]; cases.append((self.panel, m, self.tm))
        m = copy.deepcopy(self.m); m['dates'][0] += 'T01:00:00'; cases.append((self.panel, m, self.tm))
        tm = copy.deepcopy(self.tm); tm['ends'][0] = tm['dates'][0]; cases.append((self.panel, self.m, tm))
        tm = copy.deepcopy(self.tm); tm['offsets'] = [0, 1, 2, 3]; cases.append((self.panel, self.m, tm))
        tm = copy.deepcopy(self.tm); tm['symbols'][0][0] = tm['symbols'][0][1]; cases.append((self.panel, self.m, tm))
        panel = self.panel.copy(); panel['volume'] = panel['volume'].iloc[1:]; cases.append((panel, self.m, self.tm))
        panel = self.panel.copy(); panel['sectors'] = pd.concat([panel['sectors'], panel['sectors'].iloc[-1:]]); cases.append((panel, self.m, self.tm))
        for i, args in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                prepared(*args)

    def test_snapshot_cannot_silently_fix_bad_ranks_or_nonboolean_eligibility(self):
        m, tm = first_only(self.m, self.tm)
        for error in ('nan', 'range', 'shape', 'mask', 'missing'):
            def bad(*args):
                row, eligible, aux = architecture.snapshot(*args)
                row = dict(row)
                if error == 'nan': row['raw'] = np.full(len(eligible), np.nan)
                elif error == 'range': row['raw'] = np.full(len(eligible), .5000000001)
                elif error == 'shape': row['raw'] = row['raw'][:-1]
                elif error == 'mask': eligible = eligible.astype(int)
                else: del row['sector']
                return row, eligible, aux
            with self.subTest(error=error), self.assertRaises(ValueError):
                prepared(self.panel, m, tm, snapshot=bad)

    def test_target_contract_rejects_partial_masks_ineligible_targets_and_bad_audit(self):
        m, tm = first_only(self.m, self.tm)
        for error in ('partial', 'ineligible', 'infinity', 'range', 'unequal_masks', 'audit'):
            def bad(*args):
                targets, audit = labels.build_targets(*args)
                valid = np.flatnonzero(np.isfinite(targets['ordinal']))
                if error == 'partial':
                    for key in targets: targets[key][valid[0]] = np.nan
                elif error == 'ineligible':
                    for key in targets: targets[key][1] = 0.
                elif error == 'infinity': targets['ordinal'][valid[0]] = np.inf
                elif error == 'range': targets['ordinal'][valid[0]] = 1.01
                elif error == 'unequal_masks': targets['return'][valid[0]] = np.nan
                else: audit['months'][0]['status'] = 'whole_month_excluded'
                return targets, audit
            with self.subTest(error=error), self.assertRaises(ValueError):
                prepared(self.panel, m, tm, label_builder=bad)

    def test_missing_exact_label_end_is_whole_month_exclusion_not_asof_fill(self):
        m, tm = first_only(self.m, self.tm)
        tm['ends'][0] = str(pd.Timestamp(tm['ends'][0]).replace(day=1).date())
        # Pick a weekend within the required next calendar month.
        day = pd.Timestamp(tm['ends'][0])
        while day.dayofweek != 5:
            day += pd.Timedelta(days=1)
        tm['ends'][0] = str(day.date())
        a, audit = prepared(self.panel, m, tm)
        self.assertFalse(a['budget_train_label_valid'].any())
        self.assertTrue(a['budget_event_eligible'].any())
        self.assertEqual(audit['label_audit']['months'][0]['issues'][0]['kind'], 'missing_exact_price_date')


if __name__ == '__main__':
    unittest.main()
