"""Synthetic endpoint sensitivity, exact-rank precision, scoring and accounting."""
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_endpoint_median245_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest() != audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen synthetic fixture changed')
    spec = importlib.util.spec_from_file_location('endpoint245_test_prior', audit.PRIOR_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_fixture = helper()


def fixture(): return old_fixture.fixture()
def physical(args, n): return old_fixture.physical(args, n)
def selection_fixture(): return old_fixture.selection_fixture()
def selection_physical(args, n): return old_fixture.selection_physical(args, n)


def rational(pair): return Fraction(int(pair[0]), int(pair[1]))


def one_query(paths):
    days = pd.bdate_range('2015-01-01', '2017-07-03')
    date = pd.Timestamp('2017-06-30'); t = days.get_loc(date)
    names = list(paths)
    frame = pd.DataFrame(1., index=days, columns=names)
    for name, values in paths.items(): frame.loc[days[t-314:t-41], name] = values
    return {'adj': frame}, {'dates': [str(date.date())], 'symbols': [names], 'etfs': []}, [str(d.date()) for d in days]


def orders(scores, choices, meta):
    return {arm: [dict(date=meta['dates'][i], end=meta['ends'][i], signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids], scores=[float(scores[arm][i, j]) for j in ids])
        for i, ids in enumerate(choices[arm])] for arm in audit.ARMS}


def check_selection(args, n=None):
    s, c, p, r = audit.selection_oracle(*args, n=n)
    o = orders(s, c, args[1])
    receipt = audit.verify_selection(*args, o, scores=s, pre_scores=p, audit=r, n=n)
    return o, s, p, r, receipt


class MeasureTests(unittest.TestCase):
    def test_21_exact_pairs_and_odd_median(self):
        p = np.ones(273); p[252:] = np.arange(1., 22.)
        r = audit.measure_oracle(p)
        self.assertTrue(r['available'])
        self.assertEqual(rational(r['raw_median']), Fraction(10))
        self.assertEqual(rational(r['raw_mean']), Fraction(10))
        p[272] = 1025.
        r = audit.measure_oracle(p)
        self.assertEqual(rational(r['raw_median']), Fraction(10))
        self.assertEqual(rational(r['raw_mean']), Fraction(1214, 21))

    def test_not_ratio_of_price_means_or_medians(self):
        p = np.ones(273); p[:21] = np.arange(1., 22.); p[252:] = 22.
        r = audit.measure_oracle(p)
        self.assertEqual(rational(r['raw_median']), Fraction(1))
        self.assertEqual(rational(r['raw_mean']), sum((Fraction(22, k)-1 for k in range(1, 22)), Fraction())/21)
        self.assertNotEqual(rational(r['raw_mean']), Fraction(22, 11)-1)

    def test_flat_negative_zero_and_exact_cancellation_are_available(self):
        for multiplier in (1., .5, 2.):
            p = np.ones(273); p[252:] = multiplier
            r = audit.measure_oracle(p)
            self.assertTrue(r['available'])
            self.assertEqual(rational(r['raw_mean']), Fraction.from_float(multiplier)-1)
            self.assertEqual(r['raw_mean'], r['raw_median'])
        p = np.ones(273); p[252:262] = .5; p[262:272] = 1.5
        r = audit.measure_oracle(p)
        self.assertEqual(r['raw_mean'], ['0', '1']); self.assertEqual(r['raw_median'], ['0', '1'])

    def test_full_273_path_including_unused_interior_and_exact_length(self):
        for n in (252, 253, 272, 274):
            with self.assertRaises(ValueError): audit.measure_oracle(np.ones(n))
        for value, reason in ((np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price'),
                              (0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
                              (True, 'nonnumeric_adjusted_price'), ('3', 'nonnumeric_adjusted_price')):
            p = np.ones(273, dtype=object); p[100] = value
            r = audit.measure_oracle(p)
            self.assertFalse(r['available']); self.assertEqual(r['reason'], reason)
            self.assertIsNone(r['raw_median']); self.assertIsNone(r['raw_mean'])

    def test_type_sweep_and_conversion_priority(self):
        p = np.ones(273, dtype=object); p[0] = np.nan; p[-1] = 'bad'
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')
        p[-1] = 10**1000
        self.assertEqual(audit.measure_oracle(p)['reason'], 'nonnumeric_adjusted_price')

    def test_extreme_prices_no_raw_overflow_or_underflow(self):
        tiny = np.nextafter(0., 1.); huge = np.finfo(float).max
        for start, end in ((tiny, huge), (huge, tiny)):
            p = np.ones(273); p[:21] = start; p[252:] = end
            r = audit.measure_oracle(p)
            expected = Fraction.from_float(end)/Fraction.from_float(start)-1
            self.assertTrue(r['available']); self.assertEqual(rational(r['raw_median']), expected)
            self.assertEqual(r['raw_mean'], r['raw_median'])

    def test_source_scale_invariance_and_canonical_reduction(self):
        p = np.ones(273); p[:21] = np.arange(1., 22.); p[252:] = np.arange(22., 43.)
        left, right = audit.measure_oracle(p), audit.measure_oracle(p*8.)
        for field in ('raw_median', 'raw_mean'):
            self.assertEqual(left[field], right[field]); self.assertEqual(audit.pair(rational(left[field])), left[field])


class FeatureTests(unittest.TestCase):
    def test_schema_counts_and_official_21_pairs_273_off_by_one(self):
        args = fixture(); v, r = audit.independent_features(*args)
        out = audit.verify_features(*args, v, r)
        self.assertTrue(all(out[k] for k in audit.FEATURE_FLAGS))
        c = r['counts']; self.assertEqual(set(c), set(audit.COUNT_FIELDS))
        A = int(v['available'].sum())
        self.assertEqual(c['security_price_observations'], 273*A)
        self.assertEqual(c['endpoint_pairs'], 21*A); self.assertEqual(c['ranked_cells'], 2*A)
        calendar = args[2]
        for event in r['event_history']:
            self.assertEqual(set(event), set(audit.EVENT_KEYS))
            t = event['session_position']; self.assertEqual(event['source_dates'], calendar[t-314:t-41])
            self.assertEqual(len(event['source_dates']), 273)
            self.assertEqual(event['pair_dates'][0], {'start': calendar[t-314], 'end': calendar[t-62]})
            self.assertEqual(event['pair_dates'][-1], {'start': calendar[t-294], 'end': calendar[t-42]})
            for pair in event['pair_dates']: self.assertEqual(calendar.index(pair['end'])-calendar.index(pair['start']), 252)
            for cell in event['cells']: self.assertEqual(set(cell), set(audit.CELL_KEYS))

    def test_exact_tiny_mean_difference_is_not_float_tie(self):
        a = np.ones(273); a[252:] = 2.
        b = a.copy(); b[-1] = np.nextafter(2., np.inf)
        args = one_query({'A': a, 'B': b})
        v, r = audit.independent_features(*args)
        raw = [rational(c['raw_mean']) for c in r['event_history'][0]['cells']]
        self.assertNotEqual(raw[0], raw[1]); self.assertEqual(float(raw[0]), float(raw[1]))
        np.testing.assert_array_equal(v['candidate'], [[.25, .25]])
        np.testing.assert_array_equal(v['control'], [[0., .5]])

    def test_exact_near_minus_one_medians_are_not_float_ties(self):
        a = np.ones(273); a[:21] = np.finfo(float).max; a[252:] = np.nextafter(0., 1.)
        b = a.copy(); b[252:] *= 2.
        v, r = audit.independent_features(*one_query({'A': a, 'B': b}))
        raw = [rational(c['raw_median']) for c in r['event_history'][0]['cells']]
        self.assertNotEqual(raw[0], raw[1]); self.assertEqual(float(raw[0]), -1.); self.assertEqual(float(raw[1]), -1.)
        np.testing.assert_array_equal(v['candidate'], [[0., .5]])
        np.testing.assert_array_equal(v['candidate'], v['control'])

    def test_tie_singleton_empty_masks_negative_gain_valid_and_no_benchmark(self):
        args = one_query({'A': np.ones(273), 'B': np.ones(273)*2., 'ETF': np.ones(273)*4.})
        args[1]['etfs'] = ['ETF']
        v, _ = audit.independent_features(*args)
        np.testing.assert_array_equal(v['candidate'], np.full((1, 3), 2./3.-.5))
        for symbol in ('B', 'ETF'): args[0]['adj'][symbol] = np.nan
        v, _ = audit.independent_features(*args)
        self.assertEqual(v['candidate'][0, 0], .5); self.assertTrue(np.isnan(v['candidate'][0, 1:]).all())
        args[0]['adj']['A'] = np.nan
        v, r = audit.independent_features(*args)
        self.assertFalse(v['available'].any()); self.assertTrue(np.isnan(v['control']).all())
        self.assertEqual(r['counts']['endpoint_pairs'], 0)

    def test_independent_security_missing_source_and_interior(self):
        args = one_query({'A': np.ones(273), 'B': np.ones(273)})
        args[0]['adj'] = args[0]['adj'].drop(columns=['B'])
        v, r = audit.independent_features(*args)
        self.assertTrue(v['available'][0, 0]); self.assertEqual(r['event_history'][0]['cells'][1]['reason'], 'missing_symbol')
        d = r['event_history'][0]['source_dates'][100]
        args[0]['adj'].loc[d, 'A'] = np.nan
        v, r = audit.independent_features(*args); self.assertFalse(v['available'].any())
        args[0]['adj'] = args[0]['adj'].drop(index=pd.Timestamp(d))
        _, r = audit.independent_features(*args)
        self.assertEqual(r['event_history'][0]['cells'][0]['reason'], 'missing_official_session')
        self.assertEqual(r['event_history'][0]['cells'][1]['reason'], 'missing_symbol')

    def test_future_recent42_poison_physical_and_logical_prefixes(self):
        args = fixture(); before = copy.deepcopy(args); v, r = audit.independent_features(*args)
        for n in range(len(args[1]['dates'])+1):
            small = physical(args, n); vv, rr = audit.independent_features(*small)
            lv, lr = audit.independent_features(*args, limit=n)
            self.assertEqual(rr['event_history'], r['event_history'][:n]); self.assertEqual(lr['event_history'], rr['event_history'])
            if n:
                for key in audit.VALUE_KEYS:
                    audit.exact_array(vv[key], v[key][:n], key); audit.exact_array(vv[key], lv[key], key)
        poisoned = copy.deepcopy(args); t = r['event_history'][0]['session_position']
        poisoned[0]['adj'].loc[poisoned[2][t-41]:, :] = np.nan
        poisoned[1]['dates'][1] = 'bad'; poisoned[1]['symbols'][1] = None
        vv, rr = audit.independent_features(*poisoned, limit=1)
        self.assertEqual(rr['event_history'], r['event_history'][:1])
        for key in audit.VALUE_KEYS: audit.exact_array(vv[key], v[key][:1], key)
        pd.testing.assert_frame_equal(args[0]['adj'], before[0]['adj']); self.assertEqual(args[1:], before[1:])

    def test_bad_axes_short_history_and_empty_logical_physical(self):
        args = fixture()
        for field in ('dates', 'symbols'):
            bad = copy.deepcopy(args); bad[1][field][1] = bad[1][field][0] if field == 'dates' else ['A']*3
            with self.assertRaises(ValueError): audit.independent_features(*bad)
        panel, meta, calendar = one_query({'A': np.ones(273)})
        short = calendar[calendar.index(meta['dates'][0])-313:]
        v, r = audit.independent_features(panel, meta, short)
        self.assertFalse(v['available'].any()); self.assertEqual(r['reason_counts'], {'insufficient_history': 1})
        for n in (True, -1, 99):
            with self.assertRaises(ValueError): audit.independent_features(*args, limit=n)
        v, r = audit.independent_features(*args, limit=0)
        self.assertEqual(v['available'].shape, (0, 3)); self.assertEqual(r['counts']['events'], 0)
        v, r = audit.independent_features(*physical(args, 0))
        self.assertEqual(v['available'].shape, (0, 0))

    def test_complete_audit_canonical_pairs_counts_and_rank_tamper(self):
        args = fixture(); v, r = audit.independent_features(*args)
        for field in ('raw_median', 'raw_mean'):
            wrong = copy.deepcopy(r); p = wrong['event_history'][0]['cells'][0][field]
            p[:] = [str(2*int(p[0])), str(2*int(p[1]))]
            with self.assertRaises(ValueError): audit.verify_features(*args, v, wrong)
        wrong = copy.deepcopy(r); wrong['event_history'][0]['pair_dates'][0]['start'] = wrong['event_history'][0]['pair_dates'][1]['start']
        with self.assertRaises(ValueError): audit.verify_features(*args, v, wrong)
        wrong = copy.deepcopy(r); wrong['counts']['endpoint_pairs'] += 1
        with self.assertRaises(ValueError): audit.verify_features(*args, v, wrong)
        wrong = copy.deepcopy(v); wrong['candidate'][0, 0] += .001
        with self.assertRaises(ValueError): audit.verify_features(*args, wrong, r)


class SelectionTests(unittest.TestCase):
    def test_funded_hand_matches_exact_frozen_adapter_and_one_episode_each(self):
        a, m, v = selection_fixture(); o, s, p, r, check = check_selection((a, m, v))
        study = old_fixture.old_fixture.old_fixture.real_study()
        x = audit.frozen_source_oracle().old.original_helper().original_prefix(a, m, 3)
        calls = []
        for arm, key in zip(audit.ARMS, (None, 'candidate', 'control')):
            target = np.full(a['eligible'].shape, np.nan)
            if key is not None: target[1:] = v[key][1:]
            def boundary(pre, eligible, names, effects, source, policy):
                calls.append(arm); audit.exact_array(pre, p[arm], 'Independent modified pre')
                audit.exact_array(source['core'], x['core'], 'Original core')
                audit.exact_array(source['baseline'], x['baseline'], 'Original support anchor')
                return study.ep.select_boundary(pre, eligible, names, effects, source, policy)
            choices, scores = study.adapter.score_and_choose('incumbent66', .025, target, a, m, 3,
                pre_score=x['local'], effects=a['effects'], episode_policy=dict(study.POLICY_FIELDS), boundary_selector=boundary)
            audit.exact_array(scores, s[arm], arm)
            self.assertEqual(orders(s, {arm: choices for arm in audit.ARMS}, m)[arm], o[arm])
        self.assertEqual(calls, list(audit.ARMS)); self.assertTrue(all(check[k] for k in audit.SELECTION_FLAGS))
        self.assertEqual(r['counts']['original_episode_calls'], 3)

    def test_all_states_active_named_partial_only_and_valid_zero_not_missing(self):
        a, m, v = selection_fixture(); o, s, p, r, _ = check_selection((a, m, v))
        self.assertEqual([e['active'] for e in r['events']], [False, True, True])
        for arm in audit.ARMS:
            audit.exact_array(p[arm][0], p[audit.ARMS[0]][0], arm)
            audit.exact_array(s[arm][0], s[audit.ARMS[0]][0], arm)
        self.assertEqual(v['candidate'][1, 4], 0.)
        self.assertEqual(r['events'][1]['candidate_delta'][4], -.025*.375)
        v['available'][1, 4] = False; v['candidate'][1, 4] = v['control'][1, 4] = np.nan
        _, _, pp, rr, _ = check_selection((a, m, v))
        self.assertEqual(rr['events'][1]['candidate_delta'][4], 0.)
        self.assertEqual(pp[audit.ARMS[1]][1, 4], pp[audit.ARMS[0]][1, 4])

    def test_regular_row_zero_is_active_and_ineligible_sources_not_removed(self):
        a, m, v = selection_fixture(); a['eligible'][1, 4] = False
        _, _, _, r, _ = check_selection((a, m, v))
        self.assertNotEqual(r['events'][1]['candidate_delta'][4], 0.)
        a = {key: val[1:2].copy() for key, val in a.items()}
        m = {key: (val[1:2] if key in ('dates', 'ends', 'symbols') else val) for key, val in m.items()}
        v = {key: val[1:2].copy() for key, val in v.items()}
        _, _, _, r, _ = check_selection((a, m, v)); self.assertTrue(r['events'][0]['active'])

    def test_extra_calm_extension_and_conservative_remaining_support_exact(self):
        a, m, v = selection_fixture(); a['extension_available'][:] = True; a['forecast_available'][:] = True
        a['rank_log_excess'][:] = .2; a['model_available'][:] = True; a['raw_local_linear'][:] = np.arange(9)
        _, _, _, r, out = check_selection((a, m, v))
        self.assertEqual(out['calm_nonweak_extension_count'], 9)
        self.assertGreaterEqual(out['minimum_remaining_support'], .355-1e-12)
        event = r['events'][2]
        self.assertEqual(event['extension_active_indices'], list(range(9)))
        original = (.5-(.025+.01))-.05
        np.testing.assert_array_equal(event['original_remaining_support'], np.full(9, original))
        np.testing.assert_array_equal(event['candidate_remaining_support'], np.full(9, original-.025))
        self.assertLessEqual(out['maximum_budget_sum_error'], 1e-12)

    def test_missing_features_cash_two_names_and_all_input_bytes_unchanged(self):
        for count in (0, 1, 2):
            a, m, v = selection_fixture(); a['eligible'][:, count:] = False
            v['available'][:] = False; v['candidate'][:] = np.nan; v['control'][:] = np.nan
            before = copy.deepcopy(a); o, s, p, r, _ = check_selection((a, m, v))
            for arm in audit.ARMS:
                audit.exact_array(s[arm], s[audit.ARMS[0]], arm)
                self.assertEqual(len(o[arm][1]['symbols']), 2 if count == 2 else 0)
            for key in a: audit.exact_array(a[key], before[key], key)

    def test_all_short_prefixes_and_future_selection_poison(self):
        args = selection_fixture(); o, s, p, r, _ = check_selection(args)
        for n in range(4):
            small = selection_physical(args, n); oo, ss, pp, rr, _ = check_selection(small)
            logical = check_selection(args, n=n)
            audit.record_check(rr, logical[3]); self.assertEqual(rr['events'], r['events'][:n])
            for arm in audit.ARMS:
                self.assertEqual(oo[arm], o[arm][:n]); audit.exact_array(ss[arm], s[arm][:n], arm)
                audit.exact_array(pp[arm], p[arm][:n], arm)
        a, m, v = copy.deepcopy(args); a['core'][2] = np.inf; a['pre_score'][2] = np.nan
        m['dates'][2] = 'bad'; m['symbols'][2] = None; v['candidate'][2] = np.inf
        poisoned = check_selection((a, m, v), n=2)
        for arm in audit.ARMS: audit.exact_array(poisoned[1][arm], s[arm][:2], arm)

    def test_full_scores_pres_budget_audit_tamper_and_dependencies(self):
        args = selection_fixture(); o, s, p, r, _ = check_selection(args)
        for field in ('scores', 'pre_scores'):
            kwargs = dict(scores=copy.deepcopy(s), pre_scores=copy.deepcopy(p), audit=r)
            kwargs[field][audit.ARMS[1]][1, -1] += .001
            with self.assertRaises(ValueError): audit.verify_selection(*args, o, **kwargs)
        bad = copy.deepcopy(r); bad['events'][2]['candidate_remaining_support'][0] += .01
        with self.assertRaises(ValueError): audit.verify_selection(*args, o, scores=s, pre_scores=p, audit=bad)
        self.assertIn(audit.PRIOR, audit.dependency_files()); self.assertIn(audit.PRIOR_TEST, audit.dependency_files())
        tree = ast.parse(Path(audit.__file__).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn('monthly_endpoint_median245_kernel', imports)
        self.assertNotIn('monthly_endpoint_median245_host', imports)


class AccountingTests(unittest.TestCase):
    def test_independent_nine_accounts_and_fees(self):
        import test_monthly_upside_retention181_independent as f
        with patch.object(f, 'audit', audit): args = f.account_fixture()
        result = audit.audit_outcomes(*args)
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(set(result['paths']), set(audit.ARMS))
        wrong = copy.deepcopy(args); wrong[4][audit.ARMS[1]]['same_close_zero']['metrics']['fees_paid'] += 1.
        with self.assertRaises((ValueError, AssertionError)): audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixtures_complete_feature_selection_audits_and_prefixes(self):
        import monthly_endpoint_median245_kernel as producer
        import test_monthly_endpoint_median245_kernel as other
        for args in (fixture(), other.fixture()):
            v, r = producer.build_features(*args)
            audit.verify_features(*args, v, r)
            for n in range(len(args[1]['dates'])+1):
                small = physical(args, n); vv, rr = producer.build_features(*small)
                audit.verify_features(*small, vv, rr)
                self.assertEqual(rr['event_history'], r['event_history'][:n])
                lv, lr = producer.build_features(*args, limit=n)
                audit.verify_features(*args, lv, lr, limit=n)
                self.assertEqual(lr['event_history'], rr['event_history'])
                if n:
                    for key in audit.VALUE_KEYS: audit.exact_array(lv[key], vv[key], key)
        study, a, m, v = other.selection_fixture()
        for study, a, m, v in ((study, a, m, v), (old_fixture.old_fixture.old_fixture.real_study(), *selection_fixture())):
            o, s, p, r = producer.decisions(study, a, m, v, len(m['dates']))
            audit.verify_selection(a, m, v, o, scores=s, pre_scores=p, audit=r)
            for n in range(len(m['dates'])+1):
                aa, mm, vv = selection_physical((a, m, v), n)
                oo, ss, pp, rr = producer.decisions(study, aa, mm, vv, n)
                audit.verify_selection(aa, mm, vv, oo, scores=ss, pre_scores=pp, audit=rr)
                self.assertEqual(rr['events'], r['events'][:n])
                lo, ls, lp, lr = producer.decisions(study, a, m, v, n)
                audit.verify_selection(a, m, v, lo, scores=ls, pre_scores=lp, audit=lr, n=n)
                audit.record_check(lr, rr)
                for arm in audit.ARMS:
                    self.assertEqual(oo[arm], o[arm][:n]); audit.exact_array(ss[arm], s[arm][:n], arm)
                    self.assertEqual(lo[arm], oo[arm]); audit.exact_array(ls[arm], ss[arm], arm)
                    audit.exact_array(lp[arm], pp[arm], arm)


if __name__ == '__main__': unittest.main(verbosity=2)
