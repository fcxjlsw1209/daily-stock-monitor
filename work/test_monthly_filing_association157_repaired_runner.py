"""Synthetic integration with frozen adapter and isolated original selector.

No actual market arrays, source rows, registered PLAN or outputs are read.
Only the pure selector function is compiled from the original episode source.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import monthly_filing_association157_repaired_runner as r


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT/'outputs/monthly_path70_research/incumbent_adapter.py'
spec = importlib.util.spec_from_file_location('filing157_repair_real_adapter', ADAPTER_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
EPISODE_PATH = ROOT/'outputs/monthly_daily_episodes/episodes.py'
tree = ast.parse(EPISODE_PATH.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select_boundary')
namespace = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(EPISODE_PATH), 'exec'), namespace)
selector = namespace['select_boundary']
POLICY = dict(id=7, family='level', model_id=2, budget=.05, ceiling=5, gap=.02)


def fixture(months=122):
    """Original 27 synthetic arrays plus three hands and their common mask."""
    shape = months, 49
    symbols = ['S%02d'%j for j in range(32)] + ['ETF%02d'%j for j in range(17)]
    meta = dict(symbols=[symbols[:] for _ in range(months)], etfs=symbols[32:])
    baseline = np.tile(np.linspace(.24, .10, 49), (months, 1))
    cw = np.where(np.arange(months)%2 == 0, .52, .50)[:, None]
    support = np.tile(np.linspace(-.1, .1, 49), (months, 1))
    calm, weak = np.arange(months)%2 == 1, np.arange(months)%3 == 2
    effects = np.tile(np.linspace(.13, -.13, 49), (months, 1))
    a = dict(baseline=baseline, core=baseline-(1-cw)*support, core_weight=cw,
             eligible=np.ones(shape, bool), effects=effects,
             state_calm10=calm, state_weak_narrow=weak)
    # Opaque original cache members must remain byte-identical through panels.
    for j in range(20):
        a['original_%02d'%j] = np.full(shape, j, dtype=np.int16)
    a['eligible'][0, 3] = False
    common = np.zeros(shape, bool); common[:, :32] = True
    a['common_available'] = common
    for key, values in zip(r.HANDS, ((-.8, .8, -1., 1.), (.8, .9, 1., .75), (1., -.8, .8, -1.))):
        a[key] = np.full(shape, np.nan)
        a[key][:, :32] = np.tile(values, 8)
    pre = baseline + .025*calm[:, None]*(-support) + .05*weak[:, None]*(.05-support)
    return a, meta, pre


def expected(a, meta, pre, hand, n):
    """Separate direct score equation; no repaired/producer scoring calls."""
    before = pre[:n].copy()
    for i in range(n):
        for j in range(49):
            if a['eligible'][i,j] and np.isfinite(hand[i,j]) and hand[i,j] != 0.:
                before[i,j] += .01*hand[i,j]
    original = dict(baseline=a['baseline'][:n].copy(), core=a['core'][:n].copy(),
        core_weight=np.broadcast_to(a['core_weight'][:n], (n,49)).copy(),
        state_calm10=a['state_calm10'][:n].copy(), state_weak_narrow=a['state_weak_narrow'][:n].copy())
    return before, selector(before, a['eligible'][:n].copy(), meta['symbols'][:n],
                            a['effects'][:n].copy(), original, dict(POLICY))


class SyntheticBackend:
    def __init__(self, pre, repaired=True, capture=False):
        self.pre_values, self.calls = pre, []
        if capture:
            def captured(score, eligible, symbols, effects, original, policy):
                self.calls.append(dict(score=score.copy(), original=deepcopy(original),
                    eligible=eligible.copy(), effects=effects.copy(), policy=deepcopy(policy)))
                return selector(score, eligible, symbols, effects, original, policy)
            self.selector = captured
        else:
            self.selector = selector
        self.adapter = r.AdapterProxy(adapter, self.selector) if repaired else adapter

    def pre(self, arrays, n):
        return self.pre_values[:n].copy()

    def baseline(self, arrays, meta, n, policy):
        # The incumbent is independently unchanged by the new proxy.
        return adapter.score_and_choose('incumbent66', 0., np.zeros_like(arrays['baseline']),
            arrays, meta, n, pre_score=self.pre_values, effects=arrays['effects'],
            episode_policy=policy, boundary_selector=self.selector)

    @staticmethod
    def make_orders(choices, scores, meta):
        return [dict(symbols=[meta['symbols'][i][int(j)] for j in choice])
                for i, choice in enumerate(choices)]


class ScoringRepairTests(unittest.TestCase):
    def setUp(self):
        self.a, self.meta, self.pre = fixture()
        self.backend = SyntheticBackend(self.pre, capture=True)

    def test_unrepaired_actual_path_reproduces_bounded_rank_failure(self):
        with self.assertRaisesRegex(ValueError, 'bounded ranks'):
            r.original.panels(self.a, self.meta, SyntheticBackend(self.pre, repaired=False), POLICY)

    def test_extreme_h_added_once_before_original_episode(self):
        for name in r.HANDS:
            with self.subTest(hand=name):
                self.backend.calls.clear()
                actual = r.choose(name, self.a, self.meta, 122, self.backend, POLICY)
                before, wanted = expected(self.a, self.meta, self.pre, self.a[name], 122)
                self.assertEqual(actual[0], wanted[0])
                self.assertEqual(actual[1].tobytes(), wanted[1].tobytes())
                self.assertEqual(len(self.backend.calls), 1)
                captured = self.backend.calls[0]
                self.assertEqual(captured['score'].tobytes(), before.tobytes())
                np.testing.assert_array_equal(captured['original']['baseline'], self.a['baseline'])
                np.testing.assert_array_equal(captured['original']['core'], self.a['core'])
                np.testing.assert_array_equal(captured['effects'], self.a['effects'])
                np.testing.assert_array_equal(captured['eligible'], self.a['eligible'])
                self.assertEqual(captured['policy'], POLICY)
                self.assertEqual(captured['score'][0,3], self.pre[0,3])
                np.testing.assert_array_equal(captured['score'][:,32:], self.pre[:,32:])

    def test_actual_full_panels_preserve_27_arrays_and_123_prefixes(self):
        before = {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in self.a.items()}
        orders, scores = r.panels(self.a, self.meta, self.backend, POLICY)
        self.assertEqual(tuple(orders), r.NAMES)
        self.assertEqual(sum(map(len, orders.values())), 488)
        for name in r.HANDS:
            _, wanted = expected(self.a, self.meta, self.pre, self.a[name], 122)
            self.assertEqual(orders[name], self.backend.make_orders(*wanted, self.meta))
            self.assertEqual(scores[name].tobytes(), wanted[1].tobytes())
        baseline = self.backend.baseline(self.a, self.meta, 122, POLICY)
        self.assertEqual(orders[r.NAMES[0]], self.backend.make_orders(*baseline, self.meta))
        self.assertEqual(scores[r.NAMES[0]].tobytes(), baseline[1].tobytes())
        self.assertEqual(before, {k:(v.dtype.str,v.shape,v.tobytes()) for k,v in self.a.items()})
        self.assertEqual(len(self.a)-len(r.HANDS)-1, 27)

    def test_zero_and_missing_exact_baseline_fallback(self):
        baseline = self.backend.baseline(self.a, self.meta, 122, POLICY)
        for value in (0., np.nan):
            for name in r.HANDS:
                self.a[name][:,:32] = value
                actual = r.choose(name, self.a, self.meta, 122, self.backend, POLICY)
                self.assertEqual(actual[0], baseline[0])
                self.assertEqual(actual[1].tobytes(), baseline[1].tobytes())

    def test_small_h_exact_old_path_parity(self):
        for name in r.HANDS:
            self.a[name][:,:32] = np.clip(self.a[name][:,:32], -.5, .5)
            old = r.choose(name, self.a, self.meta, 122, SyntheticBackend(self.pre, repaired=False), POLICY)
            fixed = r.choose(name, self.a, self.meta, 122, self.backend, POLICY)
            self.assertEqual(old[0], fixed[0])
            self.assertEqual(old[1].tobytes(), fixed[1].tobytes())

    def test_future_h_is_not_inspected_for_prefix(self):
        name = r.HANDS[0]
        wanted = r.choose(name, self.a, self.meta, 36, self.backend, POLICY)
        self.a[name][36:] = np.inf
        self.pre[36:] = -1000.
        actual = r.choose(name, self.a, self.meta, 36, self.backend, POLICY)
        self.assertEqual(actual[0], wanted[0])
        self.assertEqual(actual[1].tobytes(), wanted[1].tobytes())
        with self.assertRaises(ValueError):
            r.choose(name, self.a, self.meta, 37, self.backend, POLICY)

    def test_frozen_h_and_common_mask_guards_remain(self):
        for value in (1.00001, -1.00001, np.inf):
            self.a[r.HANDS[0]][0,0] = value
            with self.assertRaises(ValueError):
                r.choose(r.HANDS[0], self.a, self.meta, 122, self.backend, POLICY)
        self.a[r.HANDS[0]][0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Identical three-feature source mask'):
            r.panels(self.a, self.meta, self.backend, POLICY)

    def test_original_two_or_cash_and_tie_rules_remain(self):
        self.a['eligible'][:4] = False
        self.a['eligible'][1,0] = True
        self.a['eligible'][2,:2] = True
        self.a['eligible'][3,:3] = True
        self.a['effects'][:4] = 0.
        self.pre[:4] = .2
        self.a[r.HANDS[0]][:4,:32] = 0.
        c, _ = r.choose(r.HANDS[0], self.a, self.meta, 4, self.backend, POLICY)
        self.assertEqual(c, [[], [], [0,1], [0,1,2]])


class ProxyScopeTests(unittest.TestCase):
    def setUp(self):
        self.a, self.meta, self.pre = fixture(4)
        self.hand = self.a[r.HANDS[0]]
        self.kw = dict(pre_score=self.pre, effects=self.a['effects'],
                       episode_policy=POLICY, boundary_selector=selector)
        self.proxy = r.AdapterProxy(adapter, selector)

    def test_exact_call_substitutes_only_factor_and_preserves_all_other_arguments(self):
        captured = {}
        def call(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return 'sentinel'
        target = SimpleNamespace(score_and_choose=call, extra='unchanged')
        proxy = r.AdapterProxy(target, selector)
        for budget in (0., 0, np.float64(0.), np.int64(0)):
            result = proxy.score_and_choose('incumbent66', budget, self.hand, self.a, self.meta, 4, **self.kw)
            self.assertEqual(result, 'sentinel')
            sent = captured['args']
            self.assertIs(sent[1], budget)
            self.assertEqual(sent[2].shape, self.hand.shape)
            self.assertEqual(sent[2].dtype, self.hand.dtype)
            self.assertFalse(sent[2].any())
            self.assertFalse(np.shares_memory(sent[2], self.hand))
            self.assertIs(sent[3], self.a); self.assertIs(sent[4], self.meta)
            for key, value in self.kw.items(): self.assertIs(captured['kwargs'][key], value)
        self.assertEqual(proxy.extra, 'unchanged')

    def test_nonzero_budget_retains_original_bound_and_small_rank_math(self):
        with self.assertRaisesRegex(ValueError, 'bounded ranks'):
            self.proxy.score_and_choose('incumbent66', .01, self.hand, self.a, self.meta, 4, **self.kw)
        hand = np.clip(self.hand, -.5, .5)
        old = adapter.score_and_choose('incumbent66', .01, hand, self.a, self.meta, 4, **self.kw)
        new = self.proxy.score_and_choose('incumbent66', .01, hand, self.a, self.meta, 4, **self.kw)
        self.assertEqual(old[0], new[0]); self.assertEqual(old[1].tobytes(), new[1].tobytes())

    def test_original65_call_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'bounded ranks'):
            self.proxy.score_and_choose('original65', 0., self.hand, self.a, self.meta, 4, **self.kw)

    def test_missing_explicit_input_or_different_selector_does_not_intercept(self):
        for key in self.kw:
            with self.subTest(key=key):
                kw = dict(self.kw); kw[key] = None
                with self.assertRaisesRegex(ValueError, 'bounded ranks'):
                    self.proxy.score_and_choose('incumbent66', 0., self.hand, self.a, self.meta, 4, **kw)
        kw = dict(self.kw, boundary_selector=lambda *a: selector(*a))
        with self.assertRaisesRegex(ValueError, 'bounded ranks'):
            self.proxy.score_and_choose('incumbent66', 0., self.hand, self.a, self.meta, 4, **kw)

    def test_boolean_or_invalid_budget_retains_original_rejection(self):
        for budget in (False, True, np.bool_(False), -.01, np.nan, np.inf):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.proxy.score_and_choose('incumbent66', budget, self.hand, self.a, self.meta, 4, **self.kw)

    def test_malformed_factor_or_prefix_retains_original_rejection(self):
        for hand, n in ((self.hand[:2],4), (self.hand*2,4), (self.hand,5), (self.hand,True)):
            with self.subTest(n=n), self.assertRaises(ValueError):
                self.proxy.score_and_choose('incumbent66', 0., hand, self.a, self.meta, n, **self.kw)

    def test_policy_support_and_prescore_original_guards_remain(self):
        for change in ('policy','support','prescore'):
            a, kw = deepcopy(self.a), dict(self.kw)
            if change == 'policy': kw['episode_policy'] = dict(POLICY, gap=.03)
            if change == 'support': a['core'][0,0] = -100.
            if change == 'prescore': kw['pre_score'] = self.pre.copy(); kw['pre_score'][0,0] = np.nan
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.proxy.score_and_choose('incumbent66', 0., self.hand, a, self.meta, 4, **kw)


class AmendmentIntegrationTests(unittest.TestCase):
    def test_actual_phase_implementations_are_inherited_unchanged(self):
        for name in ('prepare','loaded','select_first88','diagnose_later34'):
            self.assertIs(getattr(r.Runner,name), getattr(r.original.Runner,name))
        for name in ('panels','choose','corrected'):
            self.assertIs(getattr(r,name), getattr(r.original,name))

    def test_backend_wraps_frozen_adapter_without_loading_prices(self):
        def init(backend, root, runner):
            backend.adapter, backend.selector = adapter, selector
        with patch.object(r.original.Backend,'__init__',init):
            backend = r.Backend('unused','unused')
        self.assertIsInstance(backend.adapter,r.AdapterProxy)
        self.assertIs(backend.adapter._adapter,adapter)
        self.assertIs(backend.selector,selector)

    def test_amendment_check_precedes_original_initialization(self):
        events=[]
        def verify(runner): events.append(('amendment',runner.plan_sha,runner.amendment_sha))
        def init(runner, *args, **kwargs): events.append(('original',kwargs['backend_factory']))
        with patch.object(r.Runner,'_verify_amendment',verify), patch.object(r.original.Runner,'__init__',init):
            r.Runner(ROOT,'unused','plan','amendment')
        self.assertEqual(events,[('amendment','plan','amendment'),('original',r.Backend)])

    def test_rejected_amendment_prevents_original_init(self):
        with patch.object(r.Runner,'_verify_amendment',side_effect=ValueError('unregistered')), \
                patch.object(r.original.Runner,'__init__') as original_init:
            with self.assertRaisesRegex(ValueError,'unregistered'):
                r.Runner(ROOT,'unused','plan','amendment')
            original_init.assert_not_called()

    def test_amendment_helper_receives_both_frozen_identifiers(self):
        import sys
        obj=object.__new__(r.Runner);obj.root=ROOT
        obj.plan_sha,obj.amendment_sha='plan','amendment'
        calls=[]
        def verify(*args):calls.append(args);return {'verified':True}
        with patch.dict(sys.modules,monthly_filing_association157_repair=SimpleNamespace(verify_amendment=verify)):
            self.assertEqual(obj._verify_amendment(),{'verified':True})
        self.assertEqual(calls,[(ROOT,'plan','amendment')])

    def test_refresh_and_write_recheck_before_original_operation(self):
        obj=object.__new__(r.Runner)
        for method,args in (('refresh',()),('write',('FILE.json',{}))):
            events=[]
            with patch.object(r.Runner,'_verify_amendment',lambda _:events.append('amendment')), \
                    patch.object(r.original.Runner,method,lambda *a:events.append('original')):
                getattr(obj,method)(*args)
            self.assertEqual(events,['amendment','original'])
            with patch.object(r.Runner,'_verify_amendment',side_effect=ValueError('changed')), \
                    patch.object(r.original.Runner,method) as operation:
                with self.assertRaisesRegex(ValueError,'changed'):getattr(obj,method)(*args)
                operation.assert_not_called()

    def test_prepare_review_checks_old_receipt_then_amendment_receipt(self):
        import sys
        obj=object.__new__(r.Runner);obj.root=ROOT;obj.out=ROOT/'unused'
        obj.plan_sha,obj.amendment_sha='plan','amendment'
        events=[]; report={'synthetic':True}
        def verify(*args): events.append(('repair',args));return 'verified'
        helper=SimpleNamespace(verify_repair_report=verify)
        with patch.dict(sys.modules,monthly_filing_association157_repair=helper), \
                patch.object(r.original.Runner,'require_prepare_review',lambda _:events.append('original')), \
                patch.object(r.original,'read',return_value=report):
            self.assertEqual(obj.require_prepare_review(),'verified')
        self.assertEqual(events,['original',('repair',(ROOT,report,'plan','amendment'))])

    def test_prepare_review_rejects_original_failure_before_repair_receipt(self):
        obj=object.__new__(r.Runner)
        with patch.object(r.original.Runner,'require_prepare_review',side_effect=ValueError('old gate failed')), \
                patch.object(r.original,'read') as read:
            with self.assertRaisesRegex(ValueError,'old gate failed'):obj.require_prepare_review()
            read.assert_not_called()

    def test_source_inventory_adds_only_two_files_to_frozen_closure(self):
        old={'/synthetic/frozen.py':'a'*64}
        with patch.object(r.original,'source_files',return_value=old):
            pins=r.source_files(ROOT)
        expected={str(Path(r.__file__).resolve()),str(Path(__file__).resolve())}
        self.assertEqual(set(pins)-set(old),expected)
        self.assertEqual(pins['/synthetic/frozen.py'],'a'*64)
        self.assertTrue(all(len(pins[path])==64 for path in expected))


if __name__ == '__main__':
    unittest.main()
