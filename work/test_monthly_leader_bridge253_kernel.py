"""Synthetic bridge seams and frozen endpoints; no actual research inputs."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_leader_bridge253_kernel as kernel
import test_monthly_basket_core251_kernel as previous


def fixture():
    args = previous.fixture()
    features = previous.run(args)[0]
    return args + (features,)


def refreeze(args):
    return args[:6] + (previous.run(args[:6])[0],)


def physical(args, n):
    return previous.physical(args[:6], n) + (deepcopy(args[6][:n]),)


def run(args, n=None, **kw):
    return kernel.prepare(*args, len(args[2]['dates']) if n is None else n, **kw)


def expected(args, n=None):
    n = len(args[2]['dates']) if n is None else n
    o, s, p, _ = kernel.old.decisions(args[0], args[1], args[2], args[6], n)
    return dict(orders={k: deepcopy(o[k]) for k in kernel.ENDPOINT_ARMS},
                scores={k: s[k].copy() for k in kernel.ENDPOINT_ARMS},
                pre_scores={k: p[k].copy() for k in kernel.ENDPOINT_ARMS})


def scope(args, indices):
    previous.scope(args[:6], indices)
    return refreeze(args)


def same(case, left, right):
    case.assertEqual(tuple(left), tuple(right))
    for key in left:
        case.assertEqual(left[key].shape, right[key].shape)
        case.assertEqual(left[key].dtype, right[key].dtype)
        case.assertEqual(left[key].tobytes(), right[key].tobytes())


class KernelTests(unittest.TestCase):
    def test_schema_counts_and_two_episodes(self):
        args = fixture()
        with patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep:
            o, s, p, a = run(args)
        self.assertEqual(ep.call_count, 2)
        self.assertEqual(tuple(o), kernel.ARMS)
        self.assertEqual(tuple(a['counts']), kernel.COUNT_FIELDS)
        self.assertEqual(a['schema'], kernel.SCHEMA)
        self.assertTrue(all(tuple(e) == kernel.AUDIT_EVENT_KEYS for e in a['events']))
        for key, value in dict(original_episode_calls=1, candidate_episode_calls=1,
                               diagnostic_episode_calls=0, new_factor_definitions=0,
                               new_model_fits=0, predictive_model_fits=0).items():
            self.assertEqual(a['counts'][key], value)
        json.dumps([o, a], allow_nan=False)

    def test_original_and_basket_exact_full_outputs(self):
        args = fixture(); exp = expected(args)
        o, s, p, _ = run(args, expected_endpoints=exp)
        for oldarm, newarm in zip(kernel.ENDPOINT_ARMS, (kernel.ARMS[0], kernel.ARMS[2])):
            self.assertEqual(o[newarm], exp['orders'][oldarm])
            self.assertEqual(s[newarm].tobytes(), exp['scores'][oldarm].tobytes())
            self.assertEqual(p[newarm].tobytes(), exp['pre_scores'][oldarm].tobytes())
        self.assertEqual(p[kernel.ARMS[1]].tobytes(), p[kernel.ARMS[0]].tobytes())

    def test_exact_selector_seam_only(self):
        args = fixture(); o, s, p, a = run(args)
        b = kernel.old.old._cache(args[0], args[1], args[2], 3)
        E = b['good'].copy()
        for i in (1, 2):
            E[i] = False; E[i, args[6][i]['shortlist_indices']] = True
        source = dict(baseline=b['base'], core=b['core'], core_weight=b['cw'],
                      eligible=b['good'], state_calm10=b['calm'], state_weak_narrow=b['weak'])
        choices, post = args[0].ep.select_boundary(b['pre'], E, b['meta']['symbols'],
                                                  b['effects'], source, kernel.POLICY.copy())
        self.assertEqual(o[kernel.ARMS[1]], args[0].rp.make_orders(choices, post, b['meta']))
        self.assertEqual(s[kernel.ARMS[1]].tobytes(), post.tobytes())
        self.assertEqual([e['candidate_choices'] for e in a['events']], choices)

    def test_original_E_used_for_every_pre_computation(self):
        args = fixture(); original = args[1]['eligible'].copy(); real = args[0].modified_pre_score
        seen = []
        def wrapped(spec, arrays, n):
            np.testing.assert_array_equal(arrays['eligible'], original[:n]); seen.append(n)
            return real(spec, arrays, n)
        with patch.object(args[0], 'modified_pre_score', side_effect=wrapped): run(args)
        self.assertEqual(seen, [3])

    def test_gate_order_before_candidate(self):
        args = fixture(); marks = []; exp = expected(args)
        def source(f):
            self.assertEqual(ep.call_count, 0)
            self.assertTrue(all(all(e[k] is None for k in kernel.old.CANDIDATE_KEYS) for e in f))
            marks.append('source'); return {'passed': True}
        def inc(o, s, p):
            self.assertEqual(ep.call_count, 1); self.assertEqual(final.call_count, 0)
            marks.append('inc'); return {'passed': True}
        def endpoints(o, s, p):
            self.assertEqual(ep.call_count, 1); self.assertEqual(final.call_count, 1)
            self.assertEqual(tuple(o), kernel.ENDPOINT_ARMS)
            self.assertEqual(o, exp['orders']); same(self, s, exp['scores']); same(self, p, exp['pre_scores'])
            marks.append('endpoints'); return {'passed': True}
        with patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep, \
             patch.object(kernel.old, '_finalize', wraps=kernel.old._finalize) as final:
            run(args, source_validator=source, incumbent_validator=inc,
                endpoint_validator=endpoints, expected_endpoints=exp)
        self.assertEqual(marks, ['source', 'inc', 'endpoints'])

    def test_source_gate_failure_zero_episode(self):
        args = fixture()
        with patch.object(args[0].ep, 'select_boundary', side_effect=AssertionError('episode')), \
             self.assertRaisesRegex(ValueError, 'source gate'):
            run(args, source_validator=lambda f: {'passed': False})

    def test_inc_gate_failure_no_candidate_or_old_trios(self):
        args = fixture()
        with patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep, \
             patch.object(kernel.old, '_finalize', side_effect=AssertionError('finalize')), \
             self.assertRaisesRegex(ValueError, 'incumbent gate'):
            run(args, incumbent_validator=lambda *x: {'passed': False})
        self.assertEqual(ep.call_count, 1)

    def test_endpoint_gate_failure_one_episode(self):
        args = fixture()
        with patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep, \
             self.assertRaisesRegex(ValueError, 'endpoint gate'):
            run(args, endpoint_validator=lambda *x: {'passed': False})
        self.assertEqual(ep.call_count, 1)

    def test_callback_pass_must_be_boolean_true(self):
        args = fixture()
        for kw in ('source_validator', 'incumbent_validator', 'endpoint_validator'):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                run(args, **{kw: lambda *x: {'passed': 1}})

    def test_both_endpoint_every_matrix_and_orders(self):
        args = fixture(); good = expected(args)
        for arm in kernel.ENDPOINT_ARMS:
            for key in ('orders', 'scores', 'pre_scores'):
                bad = deepcopy(good)
                if key == 'orders': bad[key][arm][1]['symbols'] = []
                else: bad[key][arm][1, 0] = np.nextafter(bad[key][arm][1, 0], np.inf)
                with self.subTest(arm=arm, key=key), patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep, \
                     self.assertRaises(ValueError): run(args, expected_endpoints=bad)
                self.assertEqual(ep.call_count, 1)

    def test_expected_envelope_axis_and_dtype_rejected(self):
        args = fixture()
        for mode in ('arm', 'axis', 'dtype'):
            bad = expected(args)
            if mode == 'arm': bad['scores'].pop(kernel.ENDPOINT_ARMS[1])
            elif mode == 'axis': bad['orders'][kernel.ENDPOINT_ARMS[0]].pop()
            else: bad['scores'][kernel.ENDPOINT_ARMS[1]] = bad['scores'][kernel.ENDPOINT_ARMS[1]].astype(np.float32)
            with self.subTest(mode=mode), self.assertRaises(ValueError): run(args, expected_endpoints=bad)

    def test_all_logical_physical_and_cached_prefixes(self):
        args = fixture(); full = run(args)
        for n in range(4):
            exp = expected(args, n)
            logical = run(args, n, expected_endpoints=exp)
            phys = run(physical(args, n), expected_endpoints=exp)
            cached = kernel.decisions(args[0], args[1], args[2], args[6], n, expected_endpoints=exp)
            self.assertEqual(logical[0], phys[0]); self.assertEqual(logical[0], cached[0])
            self.assertEqual(logical[3], phys[3]); self.assertEqual(logical[3], cached[3])
            same(self, logical[1], phys[1]); same(self, logical[2], phys[2])
            same(self, logical[1], cached[1]); same(self, logical[2], cached[2])
            self.assertEqual(logical[3]['events'], full[3]['events'][:n])
            for arm in kernel.ARMS:
                self.assertEqual(logical[0][arm], full[0][arm][:n])
                self.assertEqual(logical[1][arm].tobytes(), full[1][arm][:n].tobytes())

    def test_empty_n0_preserves_49_width(self):
        o, s, p, a = run(physical(fixture(), 0))
        self.assertEqual(o, {arm: [] for arm in kernel.ARMS})
        self.assertEqual(a['events'], [])
        self.assertTrue(all(x.shape == (0, 49) for x in s.values()))
        self.assertEqual(a['counts']['candidate_episode_calls'], 1)

    def test_partial_uses_original_domain_and_all_scores_exact(self):
        args = fixture(); args[3]['adj'] = args[3]['adj'].astype(object); args[3]['adj'].loc[:, :] = 'bad'
        o, s, p, a = run(args, 1)
        for arm in kernel.ARMS:
            self.assertEqual(o[arm], o[kernel.ARMS[0]])
            self.assertEqual(s[arm].tobytes(), s[kernel.ARMS[0]].tobytes())
            self.assertEqual(p[arm].tobytes(), p[kernel.ARMS[0]].tobytes())
        self.assertIsNone(a['events'][0]['shortlist_indices'])
        self.assertEqual(a['events'][0]['selector_eligible_indices'], a['events'][0]['original_eligible_indices'])

    def test_cash_and_two_name_rules(self):
        for count in (0, 1, 2):
            args = scope(fixture(), list(range(40, 40 + count)))
            o, _, _, a = run(args, 2)
            self.assertEqual(len(o[kernel.ARMS[1]][1]['symbols']), 2 if count == 2 else 0)
            self.assertEqual(a['events'][1]['branch'], 'two_names' if count == 2 else 'cash')

    def test_fewer_than_six_not_expanded(self):
        for count in (3, 4, 5, 6, 7):
            args = scope(fixture(), list(range(40, 40 + count))); o, _, _, a = run(args, 2)
            self.assertEqual(len(a['events'][1]['selector_eligible_indices']), min(6, count))
            self.assertEqual(len(o[kernel.ARMS[1]][1]['symbols']), 3)

    def test_rawM_exact_ties_and_symbol_order(self):
        args = fixture(); px = args[3]['adj']; px.loc[:, :] = np.asarray(px['S48'])[:, None]
        previous.previous.refresh_core(args[:6]); args = refreeze(args)
        _, _, _, a = run(args, 2)
        self.assertEqual(a['events'][1]['shortlist_indices'], list(range(6)))

    def test_nonpositive_M_not_replaced_by_seventh(self):
        args = scope(fixture(), [40, 41, 42]); c = args[4]; t = c.index(args[2]['dates'][1])
        args[3]['adj'].loc[c[t-42], 'S40'] = args[3]['adj'].loc[c[t-294], 'S40']
        previous.previous.refresh_core(args[:6]); args = refreeze(args)
        o, _, _, a = run(args, 2)
        self.assertEqual(a['events'][1]['new_eligible_indices'], [41, 42])
        self.assertEqual(set(o[kernel.ARMS[1]][1]['symbols']), {'S41', 'S42'})
        self.assertTrue(args[1]['eligible'][1, 40])

    def test_sixth_pre_name_never_enters_ceiling_five(self):
        args = scope(fixture(), list(range(40, 47)))
        short = args[6][1]['shortlist_indices']; args[1]['pre_score'][1] = -.4
        for j, value in zip(short, (.10, .09, .08, .079, .078, .077)): args[1]['pre_score'][1, j] = value
        args[1]['effects'][1] = 0.; args[1]['effects'][1, short[-1]] = 1000.
        o, s, p, a = run(args, 2)
        e = a['events'][1]
        self.assertEqual(e['pre_order'], short)
        self.assertEqual(e['boundary_candidates'], short[2:5])
        self.assertNotIn(short[-1], e['candidate_choices'])
        self.assertEqual(s[kernel.ARMS[1]][1, short[-1]], p[kernel.ARMS[1]][1, short[-1]])
        self.assertTrue(set(short[:2]) <= set(e['candidate_choices']))

    def test_bridge_top_two_are_P_order_within_S(self):
        args = scope(fixture(), list(range(40, 47))); short = args[6][1]['shortlist_indices']
        args[1]['pre_score'][1] = -.4
        for j, value in zip(reversed(short), (.2, .19, .18, .17, .16, .15)): args[1]['pre_score'][1, j] = value
        _, _, _, a = run(args, 2)
        self.assertEqual(a['events'][1]['pre_order'], list(reversed(short)))
        self.assertTrue(set(short[-2:]) <= set(a['events'][1]['candidate_choices']))

    def test_zero_variance_basket_does_not_override_bridge(self):
        args = fixture(); x = np.arange(len(args[3]['adj'])); args[3]['adj'].loc[:, :] = np.exp2(x//21)[:, None]
        previous.previous.refresh_core(args[:6]); args = refreeze(args)
        self.assertEqual(args[6][1]['branch'], 'zero_variance')
        args[1]['pre_score'][1] = -.4
        for j, value in zip(range(6), (.01, .02, .03, .04, .05, .06)): args[1]['pre_score'][1, j] = value
        _, _, _, a = run(args, 2)
        self.assertEqual(a['events'][1]['branch'], 'restricted_episode')
        self.assertTrue({4, 5} <= set(a['events'][1]['candidate_choices']))
        self.assertEqual(a['events'][1]['basket_choices'], [0, 1, 2])

    def test_frozen_source_tamper_fails_before_episode(self):
        args = fixture(); args[6][1]['cells'][0]['original_raw'] += .001
        with patch.object(args[0].ep, 'select_boundary', side_effect=AssertionError('episode')), \
             self.assertRaises(ValueError): run(args)

    def test_frozen_shortlist_tamper_fails_before_endpoint_and_candidate(self):
        args = fixture(); args[6][1]['shortlist_indices'].reverse(); seen = []
        with patch.object(args[0].ep, 'select_boundary', wraps=args[0].ep.select_boundary) as ep, \
             self.assertRaisesRegex(ValueError, 'Cached candidate'):
            run(args, endpoint_validator=lambda *x: (seen.append('endpoint') or {'passed': True}))
        self.assertEqual(seen, []); self.assertEqual(ep.call_count, 1)

    def test_missing_required_source_not_fallback(self):
        args = fixture(); day = previous.day(args, 1, 110); args[3]['adj'].loc[day, 'S48'] = np.nan
        previous.previous.refresh_core(args[:6])
        with patch.object(args[0].ep, 'select_boundary', side_effect=AssertionError('episode')), \
             self.assertRaisesRegex(ValueError, 'Required price'): run(args)

    def test_future_bodies_not_consumed_in_prefix(self):
        args = fixture(); want = run(args, 2)
        args[2]['dates'][2] = 'bad'; args[2]['ends'][2] = 'bad'; args[2]['symbols'][2] = ['bad']
        args[6][2] = {'bad': True}
        args[3]['adj'] = args[3]['adj'].astype(object); args[3]['adj'].loc['2016-09-01':, :] = 'bad'
        args[5]['clock_events'][2] = {'query_index': 2, 'bad': True}
        for key in ('core', 'baseline', 'effects'):
            args[1][key] = args[1][key].astype(object); args[1][key][2] = 'bad'
        got = run(args, 2); self.assertEqual(got[0], want[0]); self.assertEqual(got[3], want[3])
        same(self, got[1], want[1]); same(self, got[2], want[2])

    def test_future_object_bodies_never_copied_or_compared(self):
        class FuturePoison:
            def __deepcopy__(self, memo):
                raise AssertionError('Future body deepcopy')
            def __eq__(self, other):
                raise AssertionError('Future body equality')
        for n in (0, 2):
            args = fixture(); want = run(args, n)
            for i in range(n, len(args[2]['dates'])):
                for key in ('dates', 'ends', 'symbols'):
                    args[2][key][i] = FuturePoison()
                args[6][i] = FuturePoison()
            prepared = run(args, n)
            cached = kernel.decisions(args[0], args[1], args[2], args[6], n)
            for got in (prepared, cached):
                self.assertEqual(got[0], want[0]); self.assertEqual(got[3], want[3])
                same(self, got[1], want[1]); same(self, got[2], want[2])

    def test_cached_does_not_load_or_recompute_market_source(self):
        args = fixture()
        with patch.object(kernel.old, '_features', side_effect=AssertionError('source')):
            kernel.decisions(args[0], args[1], args[2], args[6], 3)

    def test_input_and_callback_isolation(self):
        args = fixture(); aa = kernel.old.old._snapshot(args[1]); mm = deepcopy(args[2]); ff = deepcopy(args[6]); rr = deepcopy(args[5]); px = args[3]['adj'].copy(deep=True)
        def source(f): f.clear(); return {'passed': True}
        def inc(o, s, p): o.clear(); s[:] = 999.; p[:] = 999.; return {'passed': True}
        def endpoint(o, s, p): o.clear(); s.clear(); p.clear(); return {'passed': True}
        got = run(args, source_validator=source, incumbent_validator=inc, endpoint_validator=endpoint)
        want = run(args); self.assertEqual(got[0], want[0]); self.assertEqual(got[3], want[3])
        self.assertEqual(aa, kernel.old.old._snapshot(args[1])); self.assertEqual(mm, args[2]); self.assertEqual(ff, args[6]); self.assertEqual(rr, args[5]); pd.testing.assert_frame_equal(px, args[3]['adj'])

    def test_invalid_n_and_width(self):
        args = fixture()
        for n in (-1, 4, True):
            with self.subTest(n=n), self.assertRaises(ValueError): run(args, n)
        args = previous.previous.fixture(); f = []
        with self.assertRaisesRegex(ValueError, '49'): kernel.decisions(args[0], args[1], args[2], f, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
