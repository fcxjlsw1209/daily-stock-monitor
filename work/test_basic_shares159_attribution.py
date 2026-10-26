"""Synthetic prices/portfolios and isolated temporary-file phase guards only."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import basic_shares159_attribution as a


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return digest(path)


def pure_fixture():
    axis = [(date(2017, 1, 1)+timedelta(days=30*i)).isoformat() for i in range(123)]
    symbols = list(a.ISSUERS)+['S'+str(i) for i in range(45)]
    meta = dict(dates=axis[:-1], ends=axis[1:], symbols=[symbols[:] for _ in range(122)],
                etfs=['S44'], train_months=88, early_months=53)
    orders = {name: [] for name in a.NAMES}
    cells = []
    for i in range(122):
        for name, held in zip(a.NAMES, (['AAPL', 'S0'], ['AMD', 'S0'], ['MSFT', 'NVDA', 'S1'])):
            orders[name].append(dict(date=axis[i], signal_date=axis[i], end=axis[i+1],
                                     symbols=held[:], scores=[.1]*len(held)))
        for j, symbol in enumerate(symbols):
            cells.append(dict(signal_index=i, candidate_index=j, date=axis[i], symbol=symbol,
                available=j < 4 and not (symbol == 'AMD' and i == 0), original_eligible=True,
                event_id=symbol+':'+str(i//53) if j < 4 else None))
    day_index = {day: i for i, day in enumerate(axis)}
    def close_at(day, symbol):
        return 100.+day_index[day]*(symbols.index(symbol)+1)
    return meta, orders, {'cells': cells}, close_at


def gate_fixture(root, numerical=False):
    out = root/a.OUT
    out.mkdir(parents=True)
    fixed = {'prices': root/a.PRICE, 'meta': root/'outputs/monthly_local_linear70/META.json',
             'original_orders': root/'outputs/monthly_local_linear70/local_linear_ORDERS.json'}
    for p in fixed.values():
        put(p, {})
    source = root/'work/synthetic_verifier.py'
    put(source, 'synthetic source only')
    for name in a.CACHE_FILES:
        put(out/name, {})
    runtime = None
    if numerical:
        import numpy as np
        import pandas as pd
        import sys
        meta, orders, feature, prices = pure_fixture()
        put(fixed['meta'], meta)
        put(fixed['original_orders'], orders[a.NAMES[0]])
        put(out/'META.json', meta)
        put(out/'PRETRAIN_ORDERS.json', orders)
        put(out/'FEATURE_AUDIT.json', feature)
        days = meta['dates']+[meta['ends'][-1]]
        names = meta['symbols'][0]
        panel = np.array([[prices(d, s) for s in names] for d in days])
        np.savez(fixed['prices'], dates=np.array(days), symbols=np.array(names), close=panel, opening=panel)
        runtime = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__)
        for p in ('work/basic_shares159_attribution.py', 'work/test_basic_shares159_attribution.py',
                  'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
            put(root/p, 'synthetic execution evidence')
    put(out/'CACHE.json', {n: digest(out/n) for n in a.CACHE_FILES})
    for name in ('TRAIN_TRIALS.json', 'TRAIN_OPEN_TRIALS.json', 'INDEPENDENT_PREPARE.json'):
        put(out/name, {})
    inputs = {k: dict(path=str(p), sha256=digest(p)) for k, p in fixed.items()}
    pins = {str(p): digest(p) for p in [*fixed.values(), source]}
    plan = dict(round=159, status='REGISTERED', arms=list(a.NAMES), output_directory=str(out),
                inputs=inputs, hashes=pins, runtime=runtime)
    plan_sha = put(out/'PLAN.json', plan)
    selection = dict(primary=a.NAMES[0], admitted=False)
    frozen_sha = put(out/'FROZEN.json', dict(plan_sha256=plan_sha, selection=selection,
        selection_count=1, later34_diagnostic_only=True,
        hashes={n: digest(out/n) for n in a.FROZEN_FILES}))
    results_sha = put(out/'RESULTS.json', dict(plan_sha256=plan_sha, selection=selection,
        arms={n: {mode: {} for mode in ('same_close', 'next_open_zero', 'next_open_cost')} for n in a.NAMES}))
    artifacts = {str(p): digest(p) for p in out.iterdir() if p.is_file()}
    artifacts.update(pins)
    review = dict(schema='basic_shares159_independent_outcome_v1', passed=True, stage='outcome',
        producer_modules_imported=False, checks=dict.fromkeys(a.OUTCOME_CHECKS, True),
        paths_checked=9, orders_checked=366, selection_count=1, plan_sha256=plan_sha,
        frozen_sha256=frozen_sha, results_sha256=results_sha, selection=selection,
        source_hashes={str(source): digest(source)}, artifact_hashes=artifacts)
    outcome_sha = put(out/'INDEPENDENT_OUTCOME.json', review)
    return [plan_sha, frozen_sha, results_sha, outcome_sha], review


class ArithmeticTests(unittest.TestCase):
    def test_stock_sum_equals_independent_fixed_unit_portfolio_difference(self):
        start = dict(AAPL=10., AMD=25., MSFT=100.)
        end = dict(AAPL=20., AMD=20., MSFT=110.)
        result = a.month_contributions(['AAPL', 'AMD'], ['AMD', 'MSFT'], start, end, {'AAPL': 'A:1'})
        candidate_end = 50/start['AAPL']*end['AAPL']+50/start['AMD']*end['AMD']
        incumbent_end = 50/start['AMD']*end['AMD']+50/start['MSFT']*end['MSFT']
        self.assertAlmostEqual(result['sum_stock_contributions'], (candidate_end-incumbent_end)/100)
        self.assertAlmostEqual(result['issuer_contributions']['AAPL'], .5)
        self.assertAlmostEqual(result['unattributed_contribution'], -.05)
    def test_weight_drift_does_not_rebalance_midmonth(self):
        # AAPL doubles on day1; AMD doubles on day2. Equal fixed units end at2x.
        # Daily equal rebalancing would end at2.25x and is not this calculation.
        result = a.month_contributions(['AAPL', 'AMD'], [],
            {'AAPL': 100., 'AMD': 100.}, {'AAPL': 200., 'AMD': 200.}, {})
        self.assertEqual(result['candidate_gross_return'], 1.)
        self.assertNotEqual(result['candidate_gross_return'], 1.25)
    def test_same_holdings_reordered_are_exactly_zero(self):
        result = a.month_contributions(['AAPL', 'AMD'], ['AMD', 'AAPL'],
            dict(AAPL=13., AMD=21.), dict(AAPL=29., AMD=7.), {'AMD': 'D:1'})
        self.assertEqual(result['arithmetic_excess_return'], 0.)
        self.assertTrue(all(r['contribution'] == 0 for r in result['stocks']))
    def test_unequal_two_vs_three_has_nonzero_shared_stock_contribution(self):
        result = a.month_contributions(['AAPL', 'AMD'], ['AAPL', 'AMD', 'MSFT'],
            dict(AAPL=10., AMD=10., MSFT=10.), dict(AAPL=12., AMD=9., MSFT=11.), {})
        row = next(r for r in result['stocks'] if r['symbol'] == 'AAPL')
        self.assertAlmostEqual(row['contribution'], (1/2-1/3)*.2)
    def test_cash_and_empty_portfolios_are_valid(self):
        self.assertEqual(a.month_contributions([], [], {}, {}, {})['arithmetic_excess_return'], 0.)
        r = a.month_contributions([], ['AAPL', 'AMD'], {'AAPL': 1, 'AMD': 1}, {'AAPL': 2, 'AMD': 2}, {})
        self.assertEqual(r['arithmetic_excess_return'], -1.)
    def test_each_security_price_scale_leaves_contributions_unchanged(self):
        first = a.month_contributions(['AAPL', 'AMD'], [], dict(AAPL=10, AMD=100), dict(AAPL=11, AMD=90), {})
        second = a.month_contributions(['AAPL', 'AMD'], [], dict(AAPL=100, AMD=1), dict(AAPL=110, AMD=.9), {})
        self.assertAlmostEqual(first['arithmetic_excess_return'], second['arithmetic_excess_return'])
    def test_invalid_held_prices_and_holdings_fail(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.month_contributions(['AAPL', 'AMD'], [], {'AAPL': 1, 'AMD': value}, {'AAPL': 2, 'AMD': 2}, {})
        for held in (['AAPL'], ['AAPL', 'AAPL']):
            with self.assertRaises(ValueError):
                a.month_contributions(held, [], {'AAPL': 1}, {'AAPL': 2}, {})
        with self.assertRaises(KeyError):
            a.month_contributions(['AAPL', 'AMD'], [], {'AAPL': 1}, {'AAPL': 2, 'AMD': 2}, {})


class GridTests(unittest.TestCase):
    def test_full_axis_all_slices_add_and_inputs_unchanged(self):
        meta, orders, feature, prices = pure_fixture()
        before = deepcopy((meta, orders, feature))
        result = a.decompose(meta, orders, feature, prices)
        self.assertEqual((meta, orders, feature), before)
        self.assertEqual(result['orders_used'], 366)
        for arm in result['arms'].values():
            summary = arm['summaries']
            val = lambda key: summary[key]['fixed_portfolio_excess']['sum_arithmetic_monthly_excess']
            self.assertAlmostEqual(val('all122'), val('first88')+val('later34'))
            self.assertAlmostEqual(val('first88'), val('early53')+val('recent35'))
            for label, (lo, hi) in a.SLICES.items():
                s = summary[label]
                self.assertEqual(s['fixed_portfolio_excess']['months'], hi-lo)
                for group in [*s['by_issuer'].values(), *s['by_event'], s['unattributed']]:
                    self.assertAlmostEqual(group['leave_one_group_arithmetic']['sum_arithmetic_monthly_excess'],
                        val(label)-group['contribution']['sum_arithmetic_monthly_excess'])
    def test_blocked_event_is_unattributed_and_month_event_rollover_used(self):
        meta, orders, feature, prices = pure_fixture()
        result = a.decompose(meta, orders, feature, prices)['arms'][a.NAMES[1]]
        first = next(r for r in result['months'][0]['stocks'] if r['symbol'] == 'AMD')
        second = next(r for r in result['months'][1]['stocks'] if r['symbol'] == 'AMD')
        late = next(r for r in result['months'][53]['stocks'] if r['symbol'] == 'AMD')
        self.assertIsNone(first['event_id']); self.assertIsNone(first['issuer_group'])
        self.assertEqual(second['event_id'], 'AMD:0'); self.assertEqual(late['event_id'], 'AMD:1')
    def test_zero_held_contribution_still_reports_all_four_issuers(self):
        meta, orders, feature, prices = pure_fixture()
        for name in a.NAMES[1:]:
            orders[name] = deepcopy(orders[a.NAMES[0]])
        result = a.decompose(meta, orders, feature, prices)
        for arm in result['arms'].values():
            s = arm['summaries']['all122']
            self.assertEqual(set(s['by_issuer']), set(a.ISSUERS))
            self.assertEqual(s['fixed_portfolio_excess']['sum_arithmetic_monthly_excess'], 0.)
    def test_duplicate_event_issuer_and_bad_axis_rejected_before_price_access(self):
        for fault in ('event', 'axis', 'eligible', 'interval'):
            meta, orders, feature, _ = pure_fixture()
            if fault == 'event':
                feature['cells'][2]['event_id'] = feature['cells'][0]['event_id']
            elif fault == 'axis':
                feature['cells'][0]['candidate_index'] = 1
            elif fault == 'eligible':
                feature['cells'][0]['original_eligible'] = False
            else:
                orders[a.NAMES[1]][0]['end'] = '1999-01-01'
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                a.decompose(meta, orders, feature, lambda *args: self.fail('premature prices'))
    def test_no_cagr_or_strategy_counterfactual_claim(self):
        c = a.contract()
        self.assertTrue(c['no_new_portfolios'] and c['no_reselection'] and c['no_changes_to_nine_formal_paths'])
        self.assertFalse(c['causal_effect_claimed'])
        summary = a.grouped_summary([], set())
        self.assertEqual(set(summary['fixed_portfolio_excess']),
                         {'months', 'sum_arithmetic_monthly_excess', 'mean_arithmetic_monthly_excess'})


class GateTests(unittest.TestCase):
    def test_full_synthetic_publication_has_manifest_and_write_once_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); digests, _ = gate_fixture(root, numerical=True)
            result = a.run(root, *digests)
            report = json.loads(Path(result['report']).read_text())
            manifest = json.loads(Path(result['manifest']).read_text())
            self.assertEqual(report['manifest_sha256'], digest(result['manifest']))
            self.assertEqual(result['sha256'], digest(result['report']))
            self.assertEqual(manifest['independent_outcome_sha256'], digests[3])
            self.assertEqual(manifest['input_hashes'][str((root/a.PRICE).resolve())], digest(root/a.PRICE))
            self.assertEqual(report['new_strategy_paths'], 0)
            self.assertEqual(set(report['arms']), set(a.NAMES[1:]))
            with self.assertRaisesRegex(ValueError, 'Write-once'):
                a.run(root, *digests)
    def test_valid_synthetic_gate_binds_exact_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); digests, _ = gate_fixture(root)
            reader = a.Reader(root)
            plan, paths = a.phase_gate(reader, *digests)
            self.assertEqual(paths['prices'], (root/a.PRICE).resolve())
            self.assertEqual(plan['round'], 159)
    def test_unpassed_or_incomplete_outcome_never_decodes_prices(self):
        for fault in ('passed', 'stage', 'counts', 'check', 'frozen', 'selection'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); digests, review = gate_fixture(root)
                if fault == 'passed':review['passed'] = False
                elif fault == 'stage':review['stage'] = 'prepare'
                elif fault == 'counts':review['paths_checked'] = 8
                elif fault == 'check':review['checks'][a.OUTCOME_CHECKS[0]] = False
                elif fault == 'frozen':review['frozen_sha256'] = '0'*64
                else:review['selection'] = {}
                digests[3] = put(root/a.OUT/'INDEPENDENT_OUTCOME.json', review)
                with self.subTest(fault=fault), patch.object(a, 'load_prices', side_effect=AssertionError('premature numerical read')) as loader:
                    with self.assertRaises(ValueError):a.run(root, *digests)
                    loader.assert_not_called()
    def test_changed_fixed_input_or_missing_artifact_binding_rejected(self):
        for fault in ('price', 'feature_pin'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); digests, review = gate_fixture(root)
                if fault == 'price':put(root/a.PRICE, {'changed': True})
                else:
                    del review['artifact_hashes'][str(root/a.OUT/'FEATURE_AUDIT.json')]
                    digests[3] = put(root/a.OUT/'INDEPENDENT_OUTCOME.json', review)
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    a.phase_gate(a.Reader(root), *digests)
    def test_wrong_caller_hash_and_midrun_mutation_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); digests, _ = gate_fixture(root)
            with self.assertRaises(ValueError):a.phase_gate(a.Reader(root), '0'*64, *digests[1:])
            reader = a.Reader(root); a.phase_gate(reader, *digests)
            put(root/a.PRICE, {'new': 1})
            with self.assertRaisesRegex(ValueError, 'changed during'):reader.refresh()
    def test_write_once_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ATTRIBUTION.json'
            a.write_once(path, {'one': 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):a.write_once(path, {'two': 2})
            self.assertEqual(path.read_bytes(), before)
    def test_forbidden_paths(self):
        for path in ('/tmp/OneDrive/file', '/tmp/CloudStorage/file', '/tmp/stock-observer-runtime/file'):
            with self.assertRaises(ValueError):a.Reader('/tmp').path(path)


if __name__ == '__main__':
    unittest.main()
