"""273 synthetic host integration. No historical arrays, prices or replay.

Temporary phase directories contain invented byte payloads. The numerical seam
uses the kernel's synthetic fixture and the frozen original selector function.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_matured_clock273_host as h
import monthly_matured_clock273_kernel as kernel
import monthly_matured_clock273_independent as independent
import monthly_selector_scope268_independent as selector_independent
from test_monthly_matured_clock273_kernel import fixture as model_fixture


def metrics():
    base = dict(training=dict(cagr=.60, sharpe=1.4, max_drawdown=-.20),
                early=dict(cagr=.50), recent=dict(cagr=.70))
    candidate = deepcopy(base); candidate['training']['cagr'] = .61
    return {h.ARMS[0]: base, h.ARMS[1]: candidate}


def pure_function(relative, name):
    path = h.ROOT / relative
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    env = dict(np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def seam_fixture():
    train, combined, arrays = model_fixture()
    meta = deepcopy(combined['monthly']); n, width = arrays['baseline'].shape
    meta.update(early_months=53, train_months=88)
    arrays.update(core=np.zeros((n, width)), core_weight=np.full((n, 1), .52),
                  state_calm10=np.zeros(n, bool), effects=np.zeros((n, width)),
                  synthetic_pre=np.tile(np.array([.50, .49, .48, .479]), (n, 1)))
    arrays['effects'][:, -1] = .5
    def ranked(raw, available, values, count):
        use = values['forecast_available'][:count] & values['eligible'][:count]
        ready = use.any(axis=1) & np.all(~use | (available[:count] &
            np.isfinite(raw['local_constant'][:count]) & np.isfinite(raw['local_linear'][:count])), axis=1)
        out = {}
        for key in ('local_constant', 'local_linear'):
            out[key] = values['rank_log_excess'][:count].copy()
            for i in np.flatnonzero(ready):
                out[key][i] = kernel._rank(raw[key][i:i+1], use[i:i+1], out[key][i:i+1])[0]
        return out, ready
    ranks, _ = ranked({key: arrays['raw_' + key] for key in ('local_constant', 'local_linear')},
                      arrays['model_available'], arrays, n)
    for key, value in ranks.items(): arrays['rank_' + key] = value
    for i in range(27 - len(arrays)): arrays['unused_' + str(i)] = np.zeros((n, width))
    assert len(arrays) == 27
    pre_calls = []
    def pre(spec, values, count):
        assert all(np.shape(v)[0] == count for v in values.values()), 'future array rows reached original pre'
        pre_calls.append(count)
        return values['synthetic_pre'].copy()
    def orders(choices, scores, m):
        return [dict(date=m['dates'][i], end=m['ends'][i],
                     symbols=[m['symbols'][i][j] for j in chosen],
                     scores=[float(scores[i, j]) for j in chosen]) for i, chosen in enumerate(choices)]
    selector = pure_function('outputs/monthly_daily_episodes/episodes.py', 'select_boundary')
    study = SimpleNamespace(configs=lambda: [None] * 4 + [dict(id=4)], modified_pre_score=pre,
        source_ranked_forecast=lambda raw, a: kernel._rank(raw, a['forecast_available'] & a['eligible'], a['rank_log_excess']),
        ranked_forecasts=ranked, ep=SimpleNamespace(select_boundary=selector),
        adapter=SimpleNamespace(_core_weight=pure_function('outputs/monthly_path70_research/incumbent_adapter.py', '_core_weight')),
        rp=SimpleNamespace(make_orders=orders), pre_calls=pre_calls)
    P = arrays['synthetic_pre'].copy()
    args = h.source.selector_inputs(study, P, arrays, meta, n)
    source = dict(baseline=args[4], core=args[5], core_weight=args[6],
                  state_calm10=args[7], state_weak_narrow=args[8])
    choices, F = selector(P, args[1], args[2], args[3], source, dict(budget=.05, ceiling=5, gap=.02))
    reference = dict(orders=orders(choices, F, meta), scores=F, pre=P)
    return study, arrays, meta, reference, train, combined


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def phase_tree(root, name, expected='a' * 64):
    folder = root / name; folder.mkdir()
    names = h.PREPARED_FILES if name == 'prepared' else h.RESULT_FILES
    for filename in sorted(names): (folder / filename).write_bytes(('invented ' + filename).encode())
    completion = dict(status='COMPLETE', plan_sha256=expected,
                      artifacts_sha256={filename: h.sha(folder / filename) for filename in names})
    write_json(folder / 'COMPLETION.json', completion)
    return folder, completion


class SelectionTests(unittest.TestCase):
    def test_fixed_two_arms_and_first88_strict_guard(self):
        out = h.select_first88(metrics())
        self.assertEqual(out, dict(primary=h.ARMS[1], admitted=True, first88_only=True, later34_used=False, cost_used=False))
        self.assertEqual(h.ARMS, ('incumbent_local67', 'monthly_pooled_clock'))

    def test_cagr_exact_equality_and_epsilon_threshold_retain_incumbent(self):
        for cagr in (.60, .60 + 1e-12):
            m = metrics(); m[h.ARMS[1]]['training']['cagr'] = cagr
            self.assertEqual(h.select_first88(m)['primary'], h.ARMS[0])
        m[h.ARMS[1]]['training']['cagr'] = np.nextafter(.60 + 1e-12, np.inf)
        self.assertTrue(h.select_first88(m)['admitted'])

    def test_all_four_guard_limits_are_inclusive(self):
        for group, field, tolerance in (('training', 'sharpe', .05), ('training', 'max_drawdown', .03),
                                        ('early', 'cagr', .025), ('recent', 'cagr', .025)):
            m = metrics(); boundary = m[h.ARMS[0]][group][field] - tolerance
            m[h.ARMS[1]][group][field] = boundary
            self.assertTrue(h.select_first88(m)['admitted'])
            m[h.ARMS[1]][group][field] = np.nextafter(boundary, -np.inf)
            self.assertFalse(h.select_first88(m)['admitted'])

    def test_extra_arm_or_later_group_rejected(self):
        m = metrics(); m['third_arm'] = deepcopy(m[h.ARMS[0]])
        with self.assertRaises(ValueError): h.select_first88(m)
        for group in ('later34', 'full122', 'next_open_zero', 'charged'):
            m = metrics(); m[h.ARMS[1]][group] = object()
            with self.assertRaises(ValueError): h.select_first88(m)

    def test_required_metrics_nonfinite_boolean_or_missing_fail(self):
        for arm in h.ARMS:
            for value in (float('nan'), float('inf'), True, np.bool_(False), '0.9'):
                m = metrics(); m[arm]['recent']['cagr'] = value
                with self.assertRaises(ValueError): h.select_first88(m)
        m = metrics(); del m[h.ARMS[1]]['early']['cagr']
        with self.assertRaises((ValueError, KeyError)): h.select_first88(m)

    def test_unneeded_metric_values_never_consumed(self):
        m = metrics()
        for arm in h.ARMS:
            for group in m[arm]: m[arm][group]['unneeded'] = object()
        self.assertTrue(h.select_first88(m)['admitted'])


class DecisionSeamTests(unittest.TestCase):
    def test_original_and_candidate_once_each_with_real_selector_oracle(self):
        s, a, m, ref, train, combined = seam_fixture()
        bundle = kernel.fit_models(train, combined, a)
        _, delta, _ = kernel.rank_and_delta(a, m, bundle)
        before = h.old.array_snapshot(a); original = s.ep.select_boundary
        with patch.object(s.ep, 'select_boundary', wraps=original) as call:
            orders, scores, pre, proofs = h.decisions(s, a, m, delta, reference=ref)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(s.pre_calls, [len(m['dates'])])
        self.assertEqual(orders[h.ARMS[0]], ref['orders'])
        h.exact(scores[h.ARMS[0]], ref['scores'], 'incumbent score')
        h.exact(pre[h.ARMS[0]], ref['pre'], 'incumbent P')
        wanted = ref['pre'].copy(); mask = delta != 0.; wanted[mask] += delta[mask]
        h.exact(pre[h.ARMS[1]], wanted, 'delta added once before episode')
        for proof in proofs.values():
            self.assertTrue(proof['passed']); self.assertEqual(proof['work']['selector_calls'], 1)
        self.assertEqual(before, h.old.array_snapshot(a))
        self.assertEqual(orders[h.ARMS[1]][0], orders[h.ARMS[0]][0])

    def test_zero_delta_is_exact_identity_including_signed_zero(self):
        s, a, m, _, _, _ = seam_fixture(); a['synthetic_pre'][0, -1] = -0.
        orders, scores, pre, _ = h.decisions(s, a, m, np.zeros(a['baseline'].shape))
        self.assertEqual(orders[h.ARMS[0]], orders[h.ARMS[1]])
        h.exact(scores[h.ARMS[0]], scores[h.ARMS[1]], 'zero scores')
        h.exact(pre[h.ARMS[0]], pre[h.ARMS[1]], 'zero P')
        self.assertTrue(np.signbit(pre[h.ARMS[1]][0, -1]))

    def test_reference_corruption_stops_before_second_episode(self):
        for field in ('orders', 'scores', 'pre'):
            s, a, m, ref, _, _ = seam_fixture()
            if field == 'orders': ref[field][1]['symbols'] = ['not-original']
            else: ref[field][1, 0] += .01
            with patch.object(s.ep, 'select_boundary', wraps=s.ep.select_boundary) as call:
                with self.assertRaises(ValueError): h.decisions(s, a, m, np.zeros(a['baseline'].shape), reference=ref)
                self.assertEqual(call.call_count, 1)

    def test_invalid_delta_before_original_pre_or_selector(self):
        s, a, m, _, _, _ = seam_fixture()
        for delta in (np.zeros((1, 1)), np.full(a['baseline'].shape, np.nan), np.full(a['baseline'].shape, np.inf)):
            with patch.object(s, 'modified_pre_score') as pre, patch.object(s.ep, 'select_boundary') as select:
                with self.assertRaises(ValueError): h.decisions(s, a, m, delta)
                pre.assert_not_called(); select.assert_not_called()

    def test_wrong_selector_choices_rejected_independently(self):
        s, a, m, _, _, _ = seam_fixture(); original = s.ep.select_boundary
        def wrong(*args):
            choices, F = original(*args); choices[1] = list(reversed(choices[1])); return choices, F
        with patch.object(s.ep, 'select_boundary', side_effect=wrong):
            with self.assertRaises(ValueError): h.decisions(s, a, m, np.zeros(a['baseline'].shape))

    def test_incomplete_selector_proof_rejected_before_second_episode(self):
        s, a, m, _, _, _ = seam_fixture()
        with patch.object(selector_independent, 'verify', return_value=dict(passed=True)):
            with patch.object(s.ep, 'select_boundary', wraps=s.ep.select_boundary) as call:
                with self.assertRaises(ValueError): h.decisions(s, a, m, np.zeros(a['baseline'].shape))
                self.assertEqual(call.call_count, 1)

    def test_verifier_mutation_cannot_change_saved_pre_after_check(self):
        s, a, m, _, _, _ = seam_fixture(); verify = selector_independent.verify
        def mutate(*args):
            proof = verify(*args); args[0][1, 0] += .01; return proof
        with patch.object(selector_independent, 'verify', side_effect=mutate):
            with self.assertRaises(ValueError): h.decisions(s, a, m, np.zeros(a['baseline'].shape))

    def test_physical_prefix_before_pre_and_future_poison(self):
        s, a, m, ref, _, _ = seam_fixture()
        for key in ('synthetic_pre', 'effects', 'baseline', 'core_weight'): a[key][2:] = np.nan
        for n in (0, 1, 2):
            aa, mm = h.physical(a, m, n)
            self.assertTrue(all(v.shape[0] == n for v in aa.values()))
            self.assertEqual(mm['early_months'], 53); self.assertEqual(mm['train_months'], 88)
            out = h.decisions(s, aa, mm, np.zeros(aa['baseline'].shape), reference=ref)
            self.assertEqual(out[0][h.ARMS[0]], ref['orders'][:n])
        self.assertEqual(s.pre_calls, [0, 1, 2])
        aa, mm = h.physical(a, m, 3)
        with self.assertRaises(ValueError): h.decisions(s, aa, mm, np.zeros(aa['baseline'].shape))

    def test_core_weight_shape_normalization_exact(self):
        results = []
        for kind in ('column', 'vector', 'matrix'):
            s, a, m, _, _, _ = seam_fixture()
            if kind == 'vector': a['core_weight'] = a['core_weight'][:, 0].copy()
            if kind == 'matrix': a['core_weight'] = np.broadcast_to(a['core_weight'], a['baseline'].shape).copy()
            results.append(h.decisions(s, a, m, np.zeros(a['baseline'].shape)))
        for out in results[1:]:
            self.assertEqual(out[0], results[0][0])
            for arm in h.ARMS: h.exact(out[1][arm], results[0][1][arm], 'core shape')

    def test_exact_array_rejects_dtype_signed_zero_and_nan_payload(self):
        pairs = [(np.array([-0.]), np.array([0.])), (np.array([1.]), np.array([1.], np.float32))]
        left = np.array([np.nan]); right = left.copy(); right.view(np.uint64)[0] ^= np.uint64(1)
        for x, y in pairs + [(left, right)]:
            with self.assertRaises(ValueError): h.exact(x, y, 'bytes')


class ModelGateTests(unittest.TestCase):
    def test_independent_control_before_candidates_and_full_proof_after(self):
        t, m, a = model_fixture(); order = []; real_record = kernel._record
        control = independent.verify_control_models; full = independent.verify_models
        def record(s, kind, *args):
            order.append(kind); return real_record(s, kind, *args)
        def control_call(tt, mm, aa, view, n=None):
            self.assertIs(tt, t); self.assertIs(mm, m); self.assertIs(aa, a)
            self.assertEqual(n, 6); self.assertIsNone(view['candidate'])
            order.append('independent_control'); return control(tt, mm, aa, view, n=n)
        def full_call(*args, **kwargs):
            order.append('independent_full'); return full(*args, **kwargs)
        with patch.object(kernel, '_record', side_effect=record), patch.object(independent, 'verify_control_models', side_effect=control_call), patch.object(independent, 'verify_models', side_effect=full_call):
            bundle, proof, controlproof = h.verified_models(t, m, a, 6)
        first_monthly = order.index('monthly')
        self.assertLess(order.index('independent_control'), first_monthly)
        self.assertEqual(order[-1], 'independent_full')
        self.assertTrue(proof['passed']); self.assertTrue(controlproof['passed'])
        self.assertEqual(bundle['audit']['control_gate'], controlproof)

    def test_control_failure_prevents_every_monthly_fit(self):
        t, m, a = model_fixture()
        with patch.object(independent, 'verify_control_models', side_effect=ValueError('synthetic independent control')):
            with patch.object(kernel, '_record', wraps=kernel._record) as record:
                with self.assertRaisesRegex(ValueError, 'synthetic independent control'): h.verified_models(t, m, a, 6)
                self.assertTrue(all(c.args[1] == 'annual' for c in record.call_args_list))

    def test_failed_full_proof_is_not_returned_as_success(self):
        t, m, a = model_fixture()
        with patch.object(independent, 'verify_models', return_value=dict(passed=False)):
            with self.assertRaises(ValueError): h.verified_models(t, m, a, 6)

    def test_physical_model_gate_scopes_empty_initial_and_formal(self):
        t, m, a = model_fixture()
        full, _, _ = h.verified_models(t, m, a, 6)
        for n in (0, 1, 4):
            pt, pm, pa, scope = kernel.physical_inputs(t, m, a, n)
            small, proof, control = h.verified_models(pt, pm, pa, n)
            self.assertEqual(proof['events'], n); self.assertEqual(control['events'], n)
            for role in ('control', 'candidate'):
                for key in ('raw', 'available', 'rank'): h.exact(small[role][key], full[role][key][:n], 'fresh model prefix')
            self.assertEqual(scope['consumed_training_rows'], len(pt['dates']))


class ArtifactTests(unittest.TestCase):
    def test_fixed_phase_inventories_and_valid_checkpoint(self):
        self.assertEqual(len(h.PREPARED_FILES), 11); self.assertEqual(len(h.RESULT_FILES), 26)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('prepared', 'results'):
                folder, completion = phase_tree(root, name)
                self.assertEqual(h.checkpoint(folder, 'a' * 64), completion)

    def test_checkpoint_changed_missing_or_omitted_member_fails(self):
        for kind in ('changed', 'missing', 'omitted', 'empty'):
            with tempfile.TemporaryDirectory() as directory:
                folder, c = phase_tree(Path(directory), 'prepared')
                name = sorted(h.PREPARED_FILES)[0]
                if kind == 'changed': (folder / name).write_bytes(b'changed')
                if kind == 'missing': (folder / name).unlink()
                if kind == 'omitted': del c['artifacts_sha256'][name]; write_json(folder / 'COMPLETION.json', c)
                if kind == 'empty': c['artifacts_sha256'] = {}; write_json(folder / 'COMPLETION.json', c)
                with self.assertRaises((ValueError, FileNotFoundError)): h.checkpoint(folder, 'a' * 64)

    def test_checkpoint_extra_member_traversal_and_symlink_fail(self):
        for kind in ('extra', 'traversal', 'symlink'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); folder, c = phase_tree(root, 'prepared')
                if kind == 'extra': (folder / 'unexpected').write_bytes(b'junk')
                if kind == 'traversal':
                    (root / 'outside').write_bytes(b'outside')
                    c['artifacts_sha256']['../outside'] = h.sha(root / 'outside')
                    write_json(folder / 'COMPLETION.json', c)
                if kind == 'symlink':
                    name = sorted(h.PREPARED_FILES)[0]; body = (folder / name).read_bytes()
                    (root / 'outside').write_bytes(body); (folder / name).unlink(); (folder / name).symlink_to(root / 'outside')
                with self.assertRaises(ValueError): h.checkpoint(folder, 'a' * 64)

    def test_checkpoint_wrong_plan_or_incomplete_status_fails(self):
        for kind in ('plan', 'status'):
            with tempfile.TemporaryDirectory() as directory:
                folder, c = phase_tree(Path(directory), 'results')
                c['plan_sha256' if kind == 'plan' else 'status'] = 'wrong'
                write_json(folder / 'COMPLETION.json', c)
                with self.assertRaises(ValueError): h.checkpoint(folder, 'a' * 64)

    def test_frozen_selection_rebind_and_recompute(self):
        for kind in ('valid', 'wrong_plan', 'changed_prepared', 'changed_training', 'forged_pick'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); prepared, _ = phase_tree(root, 'prepared')
                selected = root / 'selected'; selected.mkdir(); training = metrics()
                write_json(selected / 'TRAIN_RESULTS.json', training)
                f = dict(plan_sha256='a' * 64, prepared_sha256=h.sha(prepared / 'COMPLETION.json'),
                         training_sha256=h.sha(selected / 'TRAIN_RESULTS.json'), selection=h.select_first88(training))
                if kind == 'wrong_plan': f['plan_sha256'] = 'b' * 64
                if kind == 'forged_pick': f['selection']['primary'] = h.ARMS[0]
                write_json(selected / 'FROZEN.json', f)
                if kind == 'changed_prepared': (prepared / 'COMPLETION.json').write_bytes(b'changed')
                if kind == 'changed_training': training[h.ARMS[1]]['early']['cagr'] = -.9; write_json(selected / 'TRAIN_RESULTS.json', training)
                with patch.object(h, 'OUT', root):
                    if kind == 'valid': self.assertEqual(h.frozen('a' * 64), f)
                    else:
                        with self.assertRaises(ValueError): h.frozen('a' * 64)

    def test_plan_rejection_precedes_every_actual_loader(self):
        with patch.object(h, 'gate', side_effect=ValueError('unregistered')):
            with patch.object(h.old, 'local') as local, patch.object(h.ev, 'load_arrays') as load:
                with self.assertRaisesRegex(ValueError, 'unregistered'): h.prepare('bad')
                local.assert_not_called(); load.assert_not_called()

    def test_unfrozen_diagnose_never_reads_prices(self):
        with patch.object(h, 'prepared', return_value=({}, {})), patch.object(h, 'frozen', side_effect=ValueError('unfrozen')):
            with patch.object(h.ev, 'prices') as prices:
                with self.assertRaisesRegex(ValueError, 'unfrozen'): h.diagnose('bad')
                prices.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
