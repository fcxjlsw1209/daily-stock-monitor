"""Only invented source-cache fixtures, exact algebra and independent accounting."""
import ast
import copy
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import monthly_core_conditioned243_independent as audit


def helper():
    if hashlib.sha256(audit.PRIOR_TEST.read_bytes()).hexdigest() != audit.PRIOR_TEST_SHA:
        raise ValueError('Frozen test helper changed')
    spec = importlib.util.spec_from_file_location('core_conditioned243_test_prior', audit.PRIOR_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior_fixture = helper()


def fixture():
    arrays, meta = prior_fixture.fixture()
    support = np.array([.125, -.25, .25, -.125, .375, -.375, .0625, -.0625, 0.])
    cw = np.asarray(arrays['core_weight'])
    if cw.ndim == 1:
        cw = cw[:, None]
    arrays['baseline'] = arrays['core'] + (1. - cw) * support
    return arrays, meta


def orders(scores, choices, meta):
    return {arm: [dict(date=meta['dates'][i], end=meta['ends'][i], signal_date=meta['dates'][i],
        symbols=[meta['symbols'][i][j] for j in ids], scores=[float(scores[arm][i, j]) for j in ids])
        for i, ids in enumerate(choices[arm])] for arm in audit.ARMS}


def physical(args, n):
    a, m = copy.deepcopy(args)
    return ({key: value[:n].copy() for key, value in a.items()},
            {key: (value[:n] if key in ('dates', 'ends', 'symbols') else value) for key, value in m.items()})


def real_study():
    return prior_fixture.real_study()


def check(args, n=None):
    a, m = args
    scores, choices, pre, report = audit.selection_oracle(a, m, n)
    receipt = audit.verify_selection(a, m, orders(scores, choices, m),
        scores=scores, pre_scores=pre, audit=report, n=n)
    return scores, choices, pre, report, receipt


class ProjectionTests(unittest.TestCase):
    def test_exact_beta_means_and_tied_residual_rank(self):
        q = audit.project_exact([-.25, 0., .25], [.25, 0., .25])
        self.assertEqual(q['mean_core'], ['0', '1'])
        self.assertEqual(q['mean_support'], ['1', '6'])
        self.assertEqual(q['cxx'], ['1', '8'])
        self.assertEqual(q['beta'], ['0', '1'])
        self.assertEqual(q['residual'], [['1', '12'], ['-1', '6'], ['1', '12']])
        np.testing.assert_array_equal(q['candidate_target'], [.25, -.5, .25])
        np.testing.assert_array_equal(q['control_target'], [.25, -.5, .25])

    def test_perfect_positive_and_negative_affine_relations_are_noop(self):
        x = np.array([-.5, -.25, 0., .25, .5])
        for slope in (.5, -.5):
            q = audit.project_exact(x, .125 + slope*x)
            self.assertEqual(q['branch'], 'constant_residual')
            self.assertEqual(q['beta'], [str(int(slope*2)), '2'])
            self.assertEqual(q['residual'], [['0', '1']]*5)
            for key in ('candidate_target', 'control_target'):
                np.testing.assert_array_equal(q[key], np.zeros(5))

    def test_constant_core_and_singleton_have_no_beta_or_residual(self):
        for x, s in (([.125]*3, [.25, -.25, 0.]), ([.125], [.375])):
            q = audit.project_exact(x, s)
            self.assertEqual(q['branch'], 'constant_core'); self.assertIsNone(q['beta'])
            self.assertEqual(q['residual'], [None]*len(s))
            self.assertEqual(q['residual_display'], [None]*len(s))
            np.testing.assert_array_equal(q['candidate_target'], np.zeros(len(s)))

    def test_tiny_exact_residuals_rank_even_when_display_underflows(self):
        smallest = np.nextafter(0., 1.)
        q = audit.project_exact([-.5, 0., .5], [0., smallest, 0.])
        self.assertEqual(q['branch'], 'ranked')
        self.assertNotEqual(q['residual'][0][0], '0')
        self.assertEqual(q['residual_display'][0], 0.)
        np.testing.assert_array_equal(q['candidate_target'], [-.25, .5, -.25])
        exact = [Fraction(int(a), int(b)) for a, b in q['residual']]
        self.assertEqual(sum(exact), 0)

    def test_orthogonality_is_exact_not_a_float_tolerance(self):
        x = [.1, -.2, .4, .3, -.1]
        q = audit.project_exact(x, [.2, .1, -.3, .25, -.15])
        mx = Fraction(*map(int, q['mean_core']))
        residual = [Fraction(*map(int, r)) for r in q['residual']]
        self.assertEqual(sum(residual), 0)
        self.assertEqual(sum((Fraction.from_float(v)-mx)*e for v, e in zip(x, residual)), 0)

    def test_rank_exact_uses_signed_zero_ties_and_not_float_displays(self):
        tiny = Fraction(1, 10**400)
        got = audit.rank_exact([Fraction(0), tiny, -tiny, Fraction(-0.)])
        np.testing.assert_array_equal(got, [0., .5, -.5, 0.])
        self.assertEqual(float(tiny), 0.)

    def test_projection_invalid_real_domains(self):
        for bad in (True, '0.2', None, float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit.project_exact([0., bad], [0., .25])
        with self.assertRaises(ValueError): audit.project_exact([], [])
        with self.assertRaises(ValueError): audit.project_exact([0.], [0., .25])


class SelectionTests(unittest.TestCase):
    def test_complete_audit_projection_counts_and_full_scores_receipt(self):
        a, m = fixture(); s, c, p, r, check_result = check((a, m))
        self.assertTrue(all(check_result[key] for key in audit.SELECTION_FLAGS))
        self.assertEqual(r['counts']['descriptive_ols_projections'], len(m['dates']))
        self.assertEqual(r['counts']['ranked_cells'], a['eligible'].size)
        self.assertEqual(r['counts']['original_episode_calls'], 3)
        self.assertEqual(r['counts']['new_model_fits'], 0)
        for e in r['events']:
            self.assertEqual(set(e), set(audit.EVENT_KEYS))
            for cell in e['cells']: self.assertEqual(set(cell), set(audit.CELL_KEYS))

    def test_original_pre_and_scalar_episode_match_frozen_adapter(self):
        a, m = fixture(); s, c, p, r, _ = check((a, m)); study = real_study()
        base = study.modified_pre_score(study.configs()[4], a, len(m['dates']))
        audit.exact_array(base, p[audit.ARMS[0]], 'Independent original Local67 pre')
        calls = []
        original_selector = study.ep.select_boundary
        def boundary(*args, **kwargs):
            calls.append(1)
            return original_selector(*args, **kwargs)
        for arm in audit.ARMS:
            actual_choices, actual_scores = study.adapter.score_and_choose('incumbent66', 0.,
                np.full(a['eligible'].shape, np.nan), a, m, len(m['dates']),
                pre_score=p[arm], effects=a['effects'], episode_policy=dict(study.POLICY_FIELDS),
                boundary_selector=boundary)
            audit.exact_array(actual_scores, s[arm], arm)
            self.assertEqual(actual_choices, c[arm])
        self.assertEqual(calls, [1, 1, 1])

    def test_initial_partial_describes_projection_but_all_decisions_unchanged(self):
        s, c, p, r, _ = check(fixture())
        self.assertEqual(r['events'][0]['branch'], 'ranked')
        self.assertTrue(r['events'][0]['initial_partial'])
        self.assertTrue(any(cell['candidate_target'] != 0. for cell in r['events'][0]['cells']))
        for arm in audit.ARMS:
            audit.exact_array(p[arm][0], p[audit.ARMS[0]][0], arm)
            audit.exact_array(s[arm][0], s[audit.ARMS[0]][0], arm)
            self.assertEqual(c[arm][0], c[audit.ARMS[0]][0])

    def test_regular_first_row_is_active_and_not_blanket_rowzero_noop(self):
        a, m = fixture()
        a = {key: v[1:2].copy() for key, v in a.items()}
        m = {key: (v[1:2] if key in ('dates', 'ends', 'symbols') else v) for key, v in m.items()}
        _, _, p, r, _ = check((a, m))
        self.assertFalse(r['events'][0]['initial_partial'])
        self.assertTrue(np.any(p[audit.ARMS[1]] != p[audit.ARMS[0]]))

    def test_all_members_including_ineligible_affect_projection_and_rank(self):
        a, m = fixture(); a['eligible'][:, -1] = False
        first = audit.selection_oracle(a, m)
        a['baseline'][1, -1] = a['core'][1, -1] + .48*.4
        second = audit.selection_oracle(a, m)
        self.assertNotEqual(first[3]['events'][1]['beta'], second[3]['events'][1]['beta'])
        self.assertFalse(second[3]['events'][1]['cells'][-1]['eligible'])
        self.assertNotEqual(first[3]['events'][1]['cells'][0]['residual'],
                            second[3]['events'][1]['cells'][0]['residual'])

    def test_degenerate_rows_preserve_nonzero_old_support_and_pre(self):
        a, m = fixture()
        a['state_calm10'][:] = True; a['core_weight'][:] = .5
        x = np.array([-.5, -.375, -.25, -.125, 0., .125, .25, .375, .5])
        a['core'][:] = .5*x
        a['baseline'][:] = a['core'] + .5*(.125 + .5*x)
        s, c, p, r, _ = check((a, m))
        self.assertEqual(r['counts']['constant_residual_events'], 3)
        self.assertEqual(r['counts']['candidate_nonzero_delta_cells'], 0)
        for arm in audit.ARMS:
            audit.exact_array(p[arm], p[audit.ARMS[0]], arm)
            audit.exact_array(s[arm], s[audit.ARMS[0]], arm)
        a['core'][:] = .0625
        a['baseline'][:] = a['core'] + .5*np.array([.125, -.25, .25, -.125, .375, -.375, .0625, -.0625, 0.])
        _, _, _, r, _ = check((a, m))
        self.assertEqual(r['counts']['constant_core_events'], 3)
        self.assertEqual(r['counts']['descriptive_ols_projections'], 0)
        self.assertEqual(r['counts']['constant_residual_events'], 0)

    def test_zero_target_is_valid_and_still_subtracts_nonzero_support(self):
        a, m = fixture(); a['state_calm10'][:] = True; a['core_weight'][:] = .5
        x = np.array([-.5, -.375, -.25, -.125, 0., .125, .25, .375, .5])
        a['core'][:] = .5*x
        # Centered alternating shape has an exact middle rank with nonzero S.
        raw = np.array([.125, .25, -.125, .375, .0625, -.375, .1875, -.25, 0.])
        a['baseline'][:] = a['core'] + .5*raw
        _, _, _, r, _ = check((a, m))
        cell = next(cell for cell in r['events'][1]['cells'] if cell['control_target'] == 0.)
        self.assertNotEqual(cell['support'], 0.)
        self.assertEqual(cell['control_delta'], cell['weight']*(0.-cell['support']))

    def test_calm_nonweak_extension_funding_matches_exact_original_scope(self):
        a, m = fixture(); a['extension_available'][:] = True; a['forecast_available'][:] = True
        a['rank_log_excess'][:] = .2; a['eligible'][2, 1] = False
        a['model_available'][:] = True; a['raw_local_linear'][:] = np.arange(9)
        _, _, _, r, receipt = check((a, m))
        self.assertEqual(receipt['calm_nonweak_extension_count'], 8)
        for cell in r['events'][2]['cells']:
            self.assertEqual(cell['weight'], .5-(.025+.01*float(cell['eligible'])))
        self.assertLessEqual(receipt['maximum_budget_sum_error'], 1e-12)

    def test_cash_two_names_and_singleton(self):
        for count in (0, 1, 2):
            a, m = fixture(); a['eligible'][:, count:] = False
            _, c, _, _, _ = check((a, m))
            for arm in audit.ARMS: self.assertEqual(len(c[arm][1]), 2 if count == 2 else 0)
        a, m = fixture()
        a = {key: (v[:, :1].copy() if v.ndim == 2 and v.shape[1] == 9 else v.copy()) for key, v in a.items()}
        m['symbols'] = [row[:1] for row in m['symbols']]
        _, _, _, r, _ = check((a, m))
        self.assertEqual(r['counts']['constant_core_events'], 3)

    def test_all_logical_physical_prefixes_and_input_bytes_unchanged(self):
        args = fixture(); before = copy.deepcopy(args)
        s, c, p, r, _ = check(args)
        for n in range(4):
            aa, mm = physical(args, n); ss, cc, pp, rr, _ = check((aa, mm))
            logical = check(args, n=n)
            audit.record_check(rr, logical[3]); audit.record_check(rr['events'], r['events'][:n])
            for arm in audit.ARMS:
                audit.exact_array(ss[arm], s[arm][:n], arm)
                audit.exact_array(pp[arm], p[arm][:n], arm)
                self.assertEqual(cc[arm], c[arm][:n])
        for key, value in args[0].items(): audit.exact_array(value, before[0][key], key)
        self.assertEqual(args[1], before[1])

    def test_future_numeric_and_metadata_poison_unread(self):
        a, m = fixture(); wanted = check((a, m), n=2)
        a['core'][2] = np.nan; a['baseline'][2] = np.inf
        a['effects'][2] = np.inf; a['pre_score'][2] = np.nan
        m['symbols'][2] = None; m['dates'][2] = 'bad'; m['ends'][2] = 'bad'
        got = check((a, m), n=2)
        audit.record_check(got[3], wanted[3])

    def test_source_shapes_types_bounds_and_calendar_metadata_rejected(self):
        for mode in ('support', 'core', 'cw', 'bool', 'numberbool', 'duplicate', 'badend', 'latepartial'):
            a, m = fixture()
            if mode == 'support': a['baseline'][1, 0] = 100.
            elif mode == 'core': a['core'][1, 0] = .52
            elif mode == 'cw': a['core_weight'][1, 0] = .5
            elif mode == 'bool': a['eligible'] = a['eligible'].astype(int)
            elif mode == 'numberbool': a['core'] = a['core'].astype(object); a['core'][1, 0] = True
            elif mode == 'duplicate': m['symbols'][1][0] = m['symbols'][1][1]
            elif mode == 'badend': m['ends'][1] = m['dates'][1]
            else: m['dates'][1] = '2016-08-17'
            with self.subTest(mode=mode), self.assertRaises((AssertionError, ValueError)):
                audit.selection_oracle(a, m)

    def test_any_full_score_pre_choice_or_exact_fraction_tamper_rejected(self):
        a, m = fixture(); s, c, p, r, _ = check((a, m)); oo = orders(s, c, m)
        for field in ('scores', 'pre_scores'):
            data = dict(scores=copy.deepcopy(s), pre_scores=copy.deepcopy(p), audit=r)
            data[field][audit.ARMS[1]][1, -1] += .001
            with self.assertRaises(ValueError): audit.verify_selection(a, m, oo, **data)
        for field in ('beta', 'residual', 'count', 'target', 'eligible'):
            bad = copy.deepcopy(r)
            if field == 'beta': bad['events'][1]['beta'][0] = '0'
            elif field == 'residual': bad['events'][1]['cells'][0]['residual'][1] = '1'
            elif field == 'count': bad['counts']['descriptive_ols_projections'] = 999
            elif field == 'target': bad['events'][1]['cells'][0]['candidate_target'] += .01
            else: bad['events'][1]['cells'][0]['eligible'] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify_selection(a, m, oo, scores=s, pre_scores=p, audit=bad)
        bad = copy.deepcopy(oo); bad[audit.ARMS[1]][1]['symbols'].reverse()
        with self.assertRaises(ValueError): audit.verify_selection(a, m, bad, scores=s, pre_scores=p, audit=r)

    def test_dependencies_pinned_and_no_new_producer_oracle_import(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names]
        self.assertNotIn('monthly_core_conditioned243_kernel', imports)
        self.assertNotIn('monthly_core_conditioned243_host', imports)
        self.assertIn(audit.PRIOR, audit.dependency_files())
        with patch.object(audit, 'PRIOR_SHA', 'bad'), self.assertRaises(ValueError): audit.prior()


class OutcomesTests(unittest.TestCase):
    def test_nine_paths_forty_five_slices_and_fee_tampering(self):
        import test_monthly_upside_retention181_independent as f
        with patch.object(f, 'audit', audit): args = f.account_fixture()
        result = audit.audit_outcomes(*args)
        self.assertEqual((result['checked_paths'], result['checked_slices']), (9, 45))
        self.assertEqual(set(result['paths']), set(audit.ARMS))
        for path, key in (('same_close_zero', 'fees_paid'), ('next_open_cost', 'cagr')):
            wrong = copy.deepcopy(args); wrong[4][audit.ARMS[1]][path]['metrics'][key] += 1.
            with self.assertRaises((ValueError, AssertionError)): audit.audit_outcomes(*wrong)


class BilateralTests(unittest.TestCase):
    def test_unique_both_fixtures_full_audit_and_all_prefix_cross(self):
        import monthly_core_conditioned243_kernel as producer
        from test_monthly_core_conditioned243_kernel import fixture as producer_fixture
        study, a, m = producer_fixture()
        for study, a, m in ((study, a, m), (real_study(), *fixture())):
            oo, ss, pp, rr = producer.decisions(study, a, m, len(m['dates']))
            audit.verify_selection(a, m, oo, scores=ss, pre_scores=pp, audit=rr)
            for n in range(len(m['dates'])+1):
                aa, mm = physical((a, m), n)
                o, s, p, r = producer.decisions(study, aa, mm, n)
                audit.verify_selection(aa, mm, o, scores=s, pre_scores=p, audit=r)
                audit.record_check(r['events'], rr['events'][:n])
                for arm in audit.ARMS:
                    self.assertEqual(o[arm], oo[arm][:n]); audit.exact_array(s[arm], ss[arm][:n], arm)
                    audit.exact_array(p[arm], pp[arm][:n], arm)


if __name__ == '__main__':
    unittest.main(verbosity=2)
