"""Invented exact-price participation paths, frozen-score bridge and accounting."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_up_participation244_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest() != audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen synthetic fixture changed')
    spec = importlib.util.spec_from_file_location('up244_test_prior', audit.PRIOR_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_fixture = helper()


def prices(steps):
    return np.exp2(np.r_[0., np.cumsum(steps, dtype=float)])


def fixture():
    panel, meta, calendar = old_fixture.fixture()
    panel['adj'] = panel['adj'].rename(columns={'FUND': 'SPY'})
    meta['symbols'] = [['A', 'B', 'SPY'] for _ in meta['dates']]
    meta['etfs'] = ['SPY']
    return panel, meta, calendar


def physical(args, n): return old_fixture.physical(args, n)


def selection_fixture():
    a, m, _ = old_fixture.selection_fixture()
    cw = np.asarray(a['core_weight'])
    if cw.ndim == 1: cw = cw[:, None]
    a['baseline'] = a['core'] + (1.-cw)*np.array([.125, -.25, .25, -.125, .375, -.375, .0625, -.0625, .1875])
    shape = a['eligible'].shape
    candidate = np.tile(np.linspace(-.5, .5, shape[1]), (shape[0], 1))
    return a, m, dict(available=np.ones(shape, bool), candidate=candidate, control=candidate[:, ::-1].copy())


def selection_physical(args, n):
    a, m, v = copy.deepcopy(args)
    a, m = old_fixture.old_fixture.physical((a, m), n)
    return a, m, {key: value[:n].copy() for key, value in v.items()}


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
    def test_known_integer_wins_ties_losses_and_market_flat_control(self):
        market = prices([1, 0, -1]*84)
        stock = np.ones(253)
        result = audit.measure_oracle(stock, market)
        self.assertTrue(result['available'])
        self.assertEqual((result['up_count'], result['down_count'], result['flat_count']), (84, 84, 84))
        self.assertEqual((result['wins_up'], result['ties_up'], result['losses_up']), (0, 0, 84))
        self.assertEqual((result['wins_all'], result['ties_all'], result['losses_all']), (84, 84, 84))
        self.assertEqual(result['candidate'], -.5); self.assertEqual(result['control'], 0.)

    def test_exact_ties_across_price_levels_and_self_are_valid_zero(self):
        market = prices([1, -1]*126)
        for stock in (market, market*8.):
            r = audit.measure_oracle(stock, market)
            self.assertTrue(r['available']); self.assertEqual(r['ties_up'], 126); self.assertEqual(r['ties_all'], 252)
            self.assertEqual((r['candidate'], r['control']), (0., 0.))

    def test_support_39_40_and_no_down_requirement(self):
        for ups in (0, 39, 40, 252):
            market = prices([1]*ups+[0]*(252-ups))
            r = audit.measure_oracle(market, market)
            self.assertTrue(r['source_available'] and r['benchmark_available'])
            self.assertEqual(r['available'], ups >= 40)
            self.assertEqual(r['down_count'], 0)
            self.assertEqual(r['ties_all'], 252)
            if ups < 40:
                self.assertEqual(r['reason'], 'insufficient_up_days'); self.assertIsNone(r['candidate']); self.assertIsNone(r['control'])
            else: self.assertEqual(r['candidate'], 0.)

    def test_all_up_candidate_control_exact_and_high_beta_not_alpha(self):
        market = prices([1]*252); stock = prices([2]*252)
        r = audit.measure_oracle(stock, market)
        self.assertEqual((r['candidate'], r['control']), (.5, .5))
        market = prices([1, -1]*126); stock = prices([2, -2]*126)
        r = audit.measure_oracle(stock, market)
        self.assertEqual((r['candidate'], r['control']), (.5, 0.))

    def test_fraction_comparison_avoids_rounded_ratio_ties_and_extreme_overflow(self):
        eps = 2.**-52
        self.assertEqual((1.+2*eps)/(1.+eps), (1.+eps)/1.)
        self.assertEqual(audit.relative_outcome(1.+eps, 1.+2*eps, 1., 1.+eps), -1)
        tiny, large = np.nextafter(0., 1.), np.finfo(float).max
        self.assertEqual(audit.relative_outcome(float(tiny), float(large), float(tiny), float(large)), 0)
        self.assertEqual(audit.relative_outcome(float(large), float(tiny), 1., 1.), -1)
        p = np.where(np.arange(253) % 2, large, tiny)
        result = audit.measure_oracle(p, p)
        self.assertTrue(result['available']); self.assertEqual(result['ties_all'], 252)

    def test_source_validity_is_independent_even_when_other_path_invalid(self):
        m = prices([1, -1]*126); bad = m.copy(); bad[100] = np.nan
        r = audit.measure_oracle(m, bad)
        self.assertTrue(r['source_available']); self.assertFalse(r['benchmark_available'])
        self.assertEqual(r['reason'], 'benchmark_invalid_adjusted_price')
        self.assertEqual(r['start_price'], m[0]); self.assertIsNone(r['wins_all'])
        r = audit.measure_oracle(bad, m)
        self.assertFalse(r['source_available']); self.assertTrue(r['benchmark_available'])
        self.assertEqual(r['up_count'], 126); self.assertIsNone(r['start_price'])

    def test_strict_numeric_source_and_independent_type_priority(self):
        base = prices([1, -1]*126)
        for bad, reason in ((True, 'nonnumeric_adjusted_price'), ('1', 'nonnumeric_adjusted_price'),
            (None, 'nonnumeric_adjusted_price'), (10**1000, 'nonnumeric_adjusted_price'),
            (0., 'invalid_adjusted_price'), (-1., 'invalid_adjusted_price'),
            (np.nan, 'invalid_adjusted_price'), (np.inf, 'invalid_adjusted_price')):
            p = base.astype(object); p[50] = bad
            with self.subTest(bad=str(bad)[:20]):
                r = audit.measure_oracle(p, base)
                self.assertEqual(r['source_reason'], reason); self.assertFalse(r['available'])
        p = base.astype(object); p[0] = np.nan; p[-1] = 'bad'
        self.assertEqual(audit.measure_oracle(p, base)['reason'], 'nonnumeric_adjusted_price')
        with self.assertRaises(ValueError): audit.measure_oracle(base[:-1], base)


class FeatureTests(unittest.TestCase):
    def test_full_source_counts_fields_and_neutral_spy(self):
        args = fixture(); v, r = audit.independent_features(*args)
        check = audit.verify_features(*args, v, r)
        self.assertTrue(all(check[key] for key in audit.FEATURE_FLAGS))
        self.assertEqual(check['events'], 3); self.assertEqual(check['checked_cells'], 9)
        self.assertEqual(check['security_price_observations'], 253*9)
        self.assertEqual(check['benchmark_price_observations'], 253*3)
        self.assertEqual(check['relative_comparisons'], 252*9)
        self.assertTrue(v['available'].all()); np.testing.assert_array_equal(v['candidate'][:, 2], np.zeros(3))
        for event in r['event_history']:
            self.assertEqual(set(event), set(audit.EVENT_KEYS)); self.assertEqual(len(event['source_dates']), 253)
            for cell in event['cells']: self.assertEqual(set(cell), set(audit.CELL_KEYS))

    def test_missing_spy_and_low_up_cannot_conceal_security_source_complete(self):
        p, m, c = fixture(); p['adj'] = p['adj'].drop(columns='SPY')
        v, r = audit.independent_features(p, m, c)
        self.assertFalse(v['available'].any()); self.assertEqual(r['counts']['source_available_cells'], 6)
        self.assertEqual(r['counts']['benchmark_available_events'], 0)
        self.assertEqual(r['event_history'][1]['source_available_indices'], [0, 1])
        p, m, c = fixture(); p['adj']['SPY'] = 100.
        v, r = audit.independent_features(p, m, c)
        self.assertFalse(v['available'].any()); self.assertEqual(r['counts']['source_available_cells'], 9)
        self.assertEqual(r['counts']['paired_source_cells'], 9); self.assertEqual(r['counts']['relative_comparisons'], 2268)
        self.assertEqual(r['counts']['conditional_comparisons'], 0)

    def test_invalid_security_full_interior_and_missing_official_rows(self):
        p, m, c = fixture(); t = c.index(m['dates'][1]); missing = c[t-180]
        p['adj'].loc[missing, 'A'] = np.nan
        v, r = audit.independent_features(p, m, c)
        self.assertFalse(v['available'][1, 0]); self.assertFalse(r['event_history'][1]['cells'][0]['source_available'])
        p, m, c = fixture(); p['adj'] = p['adj'].drop(pd.Timestamp(missing))
        v, r = audit.independent_features(p, m, c)
        self.assertFalse(v['available'][1].any()); self.assertEqual(r['event_history'][1]['missing_official_sessions'], [missing])
        self.assertTrue(all(cell['reason'] == 'missing_official_session' for cell in r['event_history'][1]['cells']))

    def test_no_cross_sectional_dependency_or_eligibility_mask(self):
        p, m, c = fixture(); v, r = audit.independent_features(p, m, c)
        p['adj']['B'] = np.nan
        vv, rr = audit.independent_features(p, m, c)
        for key in audit.VALUE_KEYS:
            audit.exact_array(vv[key][:, 0], v[key][:, 0], key)
            audit.exact_array(vv[key][:, 2], v[key][:, 2], key)
        self.assertEqual(rr['event_history'][1]['cells'][0], r['event_history'][1]['cells'][0])

    def test_skip42_future_numeric_poison_and_all_short_prefixes(self):
        args = fixture(); before = copy.deepcopy(args); v, r = audit.independent_features(*args)
        for n in range(4):
            small = physical(args, n); vv, rr = audit.independent_features(*small)
            logical, lr = audit.independent_features(*args, limit=n)
            if n:
                audit.record_check(rr, lr)
                for key in audit.VALUE_KEYS: audit.exact_array(vv[key], v[key][:n], key)
            else:
                self.assertEqual(vv['available'].shape, (0, 0)); self.assertEqual(logical['available'].shape, (0, 3))
            self.assertEqual(rr['event_history'], r['event_history'][:n]); audit.verify_features(*small, vv, rr)
        poisoned = copy.deepcopy(args); t = poisoned[2].index(poisoned[1]['dates'][0])
        poisoned[0]['adj'].loc[poisoned[2][t-41]:] = np.nan
        poisoned[1]['dates'][1:] = ['bad', None]; poisoned[1]['symbols'][1:] = [None, True]
        vv, rr = audit.independent_features(*poisoned, limit=1)
        for key in audit.VALUE_KEYS: audit.exact_array(vv[key], v[key][:1], key)
        self.assertEqual(rr['event_history'], r['event_history'][:1])
        pd.testing.assert_frame_equal(args[0]['adj'], before[0]['adj']); self.assertEqual(args[1:], before[1:])

    def test_metadata_errors_and_insufficient_official_history(self):
        for key in ('duplicate', 'badcalendar', 'nonmonthend', 'badlimit'):
            p, m, c = fixture()
            if key == 'duplicate': m['symbols'][1][0] = 'B'
            elif key == 'badcalendar': c[1] = c[0]
            elif key == 'nonmonthend': m['dates'][1] = '2016-08-30'
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.independent_features(p, m, c, limit=True if key == 'badlimit' else None)
        p, m, c = fixture(); m['dates'] = ['2014-01-31']; m['symbols'] = m['symbols'][:1]
        v, r = audit.independent_features(p, m, c)
        self.assertFalse(v['available'].any()); self.assertEqual(r['event_history'][0]['source_dates'], [])

    def test_probability_mask_source_flag_counts_and_integer_audit_tamper(self):
        args = fixture(); v, r = audit.independent_features(*args)
        for key in audit.VALUE_KEYS:
            bad = copy.deepcopy(v)
            if key == 'available': bad[key][1, 0] = False
            else: bad[key][1, 0] += .001
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(*args, bad, r)
        for key in ('wins_up', 'source_available', 'up_count'):
            bad = copy.deepcopy(r); cell = bad['event_history'][1]['cells'][0]
            cell[key] = False if key == 'source_available' else cell[key]+1
            with self.subTest(key=key), self.assertRaises(ValueError): audit.verify_features(*args, v, bad)
        bad = copy.deepcopy(r); bad['counts']['benchmark_price_observations'] *= 3
        with self.assertRaises(ValueError): audit.verify_features(*args, v, bad)


class SelectionTests(unittest.TestCase):
    def test_funded_hand_matches_exact_frozen_adapter_and_one_episode_each(self):
        a, m, v = selection_fixture(); o, s, p, r, check = check_selection((a, m, v))
        study = old_fixture.old_fixture.real_study()
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
        self.assertNotIn('monthly_up_participation244_kernel', imports)
        self.assertNotIn('monthly_up_participation244_host', imports)


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
        import monthly_up_participation244_kernel as producer
        import test_monthly_up_participation244_kernel as other
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
        for study, a, m, v in ((study, a, m, v), (old_fixture.old_fixture.real_study(), *selection_fixture())):
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
