"""Write-once evaluation of two fixed PTR169 data-sensitivity arms.

Usage (always through ./run_local.sh):
  work/ptr169_rebuild_evaluate.py test
  work/ptr169_rebuild_evaluate.py register --plan-sha256 REBUILD_PLAN_SHA
  work/ptr169_rebuild_evaluate.py evaluate --plan-sha256 EVALUATION_PLAN_SHA
--root selects a sibling rebuild directory under monthly_ptr_pool_source169;
it changes no acceptance rule. Registration verifies data, never fits a model
or computes portfolio returns. Evaluation runs the original frozen engine.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/monthly_ptr_pool_source169'
LOCAL = ROOT / 'outputs/monthly_local_linear70'
PRICE = ROOT / 'outputs/monthly_2016_extension/PRICES.npz'
ENGINE_DIR = ROOT / 'outputs/monthly_factor_search_v2'
ENGINE_HASHES = {
    'validate.py': 'b5c6bbbfa5f4db0821f6b0dacb33a43c974cceb8a150578803eef73132cd30e4',
    'search.py': '73bac455d7ca37dffd67a7abfb3f831de03184a1d4c2cfff1b3198bcdc9f7620',
    'data_inputs.py': '42086c1265faa91f208f2a5ce2fc0b2c721527e077db833615f3951a7da83bec',
}
ARMS = ('control', 'conditional')
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
ARTIFACTS = {'DATA.npz', 'META.json', 'ORDERS.json', 'TRAIN_INPUTS.npz',
             'WEEKLY_INPUTS.npz', 'MONTHLY_META.json', 'WEEKLY_META.json',
             'BASE_DATA.npz', 'AUDIT.json', 'BASE_AUDIT.json',
             'LABEL_AUDIT.json', 'MODEL_AUDIT.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        stream.write('\n')


def source_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    require(path.is_relative_to(ROOT), 'Input escapes research root: ' + str(path))
    require(path.is_file(), 'Missing pinned file: ' + str(path))
    return path


def root_path(name):
    require(re.fullmatch(r'fixed_incumbent_rebuild(?:_[a-zA-Z0-9_-]+)?', name) is not None,
            'Invalid rebuild directory name')
    path = (SOURCE / name).resolve()
    require(path.parent == SOURCE.resolve(), 'Rebuild root escape')
    return path


def verify_hashes(pins):
    require(isinstance(pins, dict) and bool(pins), 'Empty source binding')
    for name, digest in pins.items():
        require(sha(source_path(name)) == digest, 'Source digest changed: ' + name)


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)


def engine():
    # Three frozen files form this engine's full local import graph. Importing
    # them defines functions only; no source loader or CLI is invoked.
    for name, digest in ENGINE_HASHES.items():
        require(sha(ENGINE_DIR / name) == digest, 'Frozen engine changed: ' + name)
    for name in ('search', 'data_inputs'):
        existing = sys.modules.get(name)
        if existing is not None:
            require(Path(existing.__file__).resolve() == ENGINE_DIR / (name + '.py'),
                    'Unexpected engine import alias: ' + name)
    oldpath = list(sys.path)
    try:
        sys.path.insert(0, str(ENGINE_DIR))
        spec = importlib.util.spec_from_file_location('ptr169_fixed_engine', ENGINE_DIR / 'validate.py')
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
    finally:
        sys.path[:] = oldpath
    return result


def verify_artifacts(folder, source_digest, control_digest=None):
    report = read(folder / 'PARITY.json')
    require(not (folder / 'FAILURE.json').exists(), 'Failed reconstruction cannot be evaluated')
    require(report['passed'] is True and report['arm'] == folder.name, 'Reconstruction parity failed')
    require(report['source_plan_sha256'] == source_digest, 'Reconstruction plan mismatch')
    require(report['returns_computed'] is False, 'Unexpected pre-registration portfolio evaluation')
    require(set(report['artifacts_sha256']) == ARTIFACTS, 'Incomplete reconstruction attestation')
    if folder.name == 'conditional':
        require(report['original_control_required'] is True and
                report['control_parity_sha256'] == control_digest, 'Conditional control binding mismatch')
    else:
        require({'monthly', 'weekly', 'incumbent_arrays'} <= set(report['checks']),
                'Original control numerical checks missing')
    for name, digest in report['artifacts_sha256'].items():
        require(sha(folder / name) == digest, 'Reconstructed artifact changed: ' + name)
    return report


def verify_meta_orders(meta, orders, arrays):
    dates, ends, symbols = meta['dates'], meta['ends'], meta['symbols']
    require(len(dates) == len(ends) == len(symbols) == len(orders) == 122, 'Expected 122 events')
    require(dates == sorted(set(dates)) and ends[:-1] == dates[1:], 'Noncontiguous event axis')
    require(dates[0] == '2016-08-17' and ends[-1] == '2026-09-24', 'Changed full window')
    require(ends[52] == dates[53] == '2020-12-31' and
            ends[87] == dates[88] == '2023-11-30', 'Changed evaluation split')
    eligible = arrays['eligible']
    require(eligible.shape == (122, 49) and eligible.dtype.kind == 'b', 'Invalid eligibility mask')
    for i, (day, end, ss, order) in enumerate(zip(dates, ends, symbols, orders)):
        require(len(ss) == 49 and ss == sorted(set(ss)), 'Invalid pool identity')
        require(order['date'] == order['signal_date'] == day and order['end'] == end,
                'Order date or end mismatch')
        selected = order['symbols']
        require(len(selected) in (0, 2, 3) and len(set(selected)) == len(selected), 'Invalid EW3 order')
        require(len(order['scores']) == len(selected) and np.isfinite(order['scores']).all(), 'Invalid order scores')
        require(all(s in ss and eligible[i, ss.index(s)] for s in selected), 'Ineligible selection')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as file:
        return {name: file[name].copy() for name in file.files}


def equal_arrays(actual, expected):
    require(set(actual) == set(expected), 'Control array keys changed')
    for key in expected:
        x, y = actual[key], expected[key]
        require(x.shape == y.shape and x.dtype.kind == y.dtype.kind, 'Control axes/type changed: ' + key)
        if x.dtype.kind in 'fc':
            np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y), err_msg=key)
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12, equal_nan=True, err_msg=key)
        else:
            np.testing.assert_array_equal(x, y, err_msg=key)


def gates(folder, expected_sha):
    require(sha(folder / 'PLAN.json') == expected_sha, 'Wrong registered rebuild PLAN')
    plan = read(folder / 'PLAN.json')
    require(plan['schema'] == 'ptr169_fixed_incumbent_rebuild_v1', 'Wrong reconstruction schema')
    require(plan['runtime'] == runtime(), 'Pinned runtime changed')
    require(plan['fixed_candidate'] == dict(id=4, base='incumbent_scope67', mode='local_linear'),
            'Only fixed incumbent local67 is permitted')
    require(plan['new_strategy_search_trials'] == 0, 'Not a fixed data sensitivity')
    verify_hashes(plan['input_sha256'])
    records = {}
    for arm in ARMS:
        records[arm] = verify_artifacts(folder / arm, expected_sha, sha(folder / 'control/PARITY.json'))
    bound = {source_path(path) for path in plan['input_sha256']}
    reference_orders = source_path(plan.get('reference_orders', str(LOCAL / 'local_linear_ORDERS.json')))
    for path in (PRICE, LOCAL / 'DATA.npz', LOCAL / 'META.json', reference_orders):
        require(path in bound, 'Original reference not bound by rebuild PLAN: ' + str(path))
    all_meta, all_orders = {}, {}
    for arm in ARMS:
        all_meta[arm] = read(folder / arm / 'META.json')['monthly']
        all_orders[arm] = read(folder / arm / 'ORDERS.json')
        data = load_arrays(folder / arm / 'DATA.npz')
        verify_meta_orders(all_meta[arm], all_orders[arm], data)
        if arm == 'control':
            require(all_meta[arm] == read(LOCAL / 'META.json'), 'Control monthly metadata differs')
            equal_arrays(data, load_arrays(LOCAL / 'DATA.npz'))
            require(all_orders[arm] == read(reference_orders), 'Control orders are not exactly the original incumbent')
    require(all_meta['control']['dates'] == all_meta['conditional']['dates'] and
            all_meta['control']['ends'] == all_meta['conditional']['ends'] and
            all_meta['control']['etfs'] == all_meta['conditional']['etfs'], 'Arms use different evaluation axes')
    return plan, all_meta, all_orders


def register(folder, expected_sha):
    plan, _, _ = gates(folder, expected_sha)
    pins = {str(source_path(p).relative_to(ROOT)): h for p, h in plan['input_sha256'].items()}
    files = [Path(__file__), folder / 'PLAN.json']
    files += [ENGINE_DIR / name for name in ENGINE_HASHES]
    for arm in ARMS:
        files += [folder / arm / name for name in ARTIFACTS | {'PARITY.json'}]
    for path in files:
        pins[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in ENGINE_HASHES.items():
        require(sha(ENGINE_DIR / name) == digest, 'Frozen engine mismatch')
    out = folder / 'evaluation'
    out.mkdir(exist_ok=False)
    dump(out / 'PLAN.json', dict(schema='ptr169_fixed_evaluation_v1', registered_utc=stamp(),
         rebuild_plan_sha256=expected_sha, input_sha256=pins, runtime=runtime(), arms=list(ARMS),
         paths=[dict(name=n, execution=e, cost_per_side=c) for n, e, c in PATHS],
         initial_capital=100000., splits=['full122', 'early53', 'recent35', 'later34', 'first88'],
         costs='Original turnover-funded equal-weight rebalance; 5bp on each buy/sell, no terminal liquidation.',
         boundaries='Full path first, then inclusive close-to-close slices rebased at their first close, matching original cuts; no fresh account at split boundaries.',
         end_convention='Final valuation 2026-09-24; no forced sale or invented final rebalance. Next-open uses next available index session, not calendar day.',
         quantity='Original adjusted total-return units; dividends approximated by adjusted prices, not nominal broker share quantities.',
         uncertainty=plan['data_limits'], no_optimization=True, source_files_modified=False))
    print(json.dumps(dict(status='REGISTERED_NO_PORTFOLIO_RETURNS', plan_sha256=sha(out / 'PLAN.json'))), flush=True)


def prices():
    data = load_arrays(PRICE)
    require({'close', 'opening', 'dates', 'symbols'} <= set(data), 'Incomplete original price archive')
    index = pd.to_datetime(data['dates'])
    require(index.is_unique and index.is_monotonic_increasing, 'Invalid price dates')
    require(len(set(data['symbols'])) == len(data['symbols']), 'Duplicate price symbols')
    return (pd.DataFrame(data['close'], index=index, columns=data['symbols']),
            pd.DataFrame(data['opening'], index=index, columns=data['symbols']))


def split_stats(original_engine, equity, meta):
    dates, ends = meta['dates'], meta['ends']
    bounds = {'early53': (dates[0], ends[52]), 'recent35': (dates[53], ends[87]),
              'later34': (dates[88], ends[-1]), 'first88': (dates[0], ends[87])}
    result = {}
    for name, (start, end) in bounds.items():
        values = equity.loc[start:end]
        require(str(values.index[0].date()) == start and str(values.index[-1].date()) == end,
                'Missing split boundary: ' + name)
        result[name] = original_engine.stats(values / values.iloc[0] * 100000.)
    return result


def reconcile(original_engine, px, orders, equity, trades):
    """Reconstruct cash/units from exported fills; no strategy or fee solver reuse."""
    symbols = {s for order in orders for s in order['symbols']}
    display = {original_engine.display(s): s for s in symbols}
    require(len(display) == len(symbols), 'Ambiguous exported ticker alias')
    byday = {}
    for trade in trades:
        byday.setdefault(trade['date'], []).append(trade)
    cash, positions, rows, max_error = 100000., {}, [], 0.
    for day, expected in equity.items():
        daystr = str(day.date())
        for fill in byday.get(daystr, []):
            symbol = display[fill['symbol']]
            direction = 1 if fill['side'] == 'buy' else -1
            quantity = direction * fill['adjusted_units']
            positions[symbol] = positions.get(symbol, 0.) + quantity
            cash -= quantity * fill['adjusted_price'] + fill['cost']
        require(cash > -1e-5 and min(positions.values(), default=0.) > -1e-7, 'Negative cash/holding in exported ledger')
        held = {s: q for s, q in positions.items() if abs(q) > 1e-9}
        values = {s: q * float(px.at[day, s]) for s, q in held.items()}
        require(all(np.isfinite(v) for v in values.values()), 'Missing daily holding valuation')
        nav = cash + sum(values.values())
        max_error = max(max_error, abs(nav - expected))
        np.testing.assert_allclose(nav, expected, rtol=1e-10, atol=1e-5)
        rows.append(dict(date=daystr, cash=cash, adjusted_units=held, market_values=values, equity=nav))
    require(set(byday) <= {str(d.date()) for d in equity.index}, 'Fill outside equity interval')
    return rows, dict(exported_fill_ledger_matches=True, max_absolute_nav_difference=max_error,
                      terminal_liquidation=False)


def run_path(original_engine, px, opening, orders, meta, execution, cost):
    start, end = pd.Timestamp(meta['dates'][0]), pd.Timestamp(meta['ends'][-1])
    metrics, equity, trades, periods = original_engine.replay(px, opening, orders, cost=cost,
                                                           execution=execution, start=start, end=end)
    expected = [str((pd.Timestamp(o['date']) if execution == 'close' else
                    px.index[px.index.get_loc(pd.Timestamp(o['date'])) + 1]).date()) for o in orders]
    require([p['execution_date'] for p in periods] == expected, 'A registered monthly order was skipped or reordered')
    require([p['signal_date'] for p in periods] == [o['signal_date'] for o in orders], 'Signal provenance changed')
    require(str(equity.index[0].date()) == str(start.date()) and
            str(equity.index[-1].date()) == str(end.date()), 'Replay interval truncated')
    for row in metrics['annual_returns']:
        if row['year'] in (2016, 2026):
            row['partial'] = True
    ledger, accounting = reconcile(original_engine, px, orders, equity, trades)
    return dict(metrics=metrics, accounting=accounting), equity, trades, periods, ledger


def evaluate(folder, expected_sha):
    out = folder / 'evaluation'
    require(sha(out / 'PLAN.json') == expected_sha, 'Wrong evaluation PLAN')
    plan = read(out / 'PLAN.json')
    require(plan['schema'] == 'ptr169_fixed_evaluation_v1' and plan['runtime'] == runtime(), 'Evaluation contract mismatch')
    verify_hashes(plan['input_sha256'])
    source, metas, orders = gates(folder, plan['rebuild_plan_sha256'])
    original_engine = engine()
    px, opening = prices()
    dest = out / 'results'
    dest.mkdir(exist_ok=False)
    try:
        results = {}
        for arm in ARMS:
            results[arm] = {}
            dump(dest / (arm + '_ORDERS.json'), orders[arm])
            for name, execution, cost in PATHS:
                row, equity, trades, periods, ledger = run_path(original_engine, px, opening,
                                                               orders[arm], metas[arm], execution, cost)
                row['slices'] = dict(full122=row['metrics'], **split_stats(original_engine, equity, metas[arm]))
                results[arm][name] = row
                stem = arm + '_' + name
                for suffix, value in [('DAILY', [dict(date=str(d.date()), equity=float(v)) for d, v in equity.items()]),
                                      ('TRADES', trades), ('HOLDINGS', periods), ('DAILY_POSITIONS', ledger)]:
                    dump(dest / (stem + '_' + suffix + '.json'), value)
                print(arm, name, 'completed', flush=True)
        verify_hashes(plan['input_sha256'])
        differences = {}
        for name, _, _ in PATHS:
            differences[name] = {split: {key: (None if results['conditional'][name]['slices'][split][key] is None or
                                               results['control'][name]['slices'][split][key] is None else
                                               results['conditional'][name]['slices'][split][key] -
                                               results['control'][name]['slices'][split][key])
                                         for key in ('cagr', 'sharpe', 'max_drawdown', 'total_return')}
                                 for split in plan['splits']}
        dump(dest / 'RESULTS.json', dict(arms=results, conditional_minus_control=differences,
             limitations=source['data_limits'], optimized=False, promoted=False,
             records_are_simulated=True, split_boundary_convention=plan['boundaries']))
        dump(dest / 'COMPLETION.json', dict(status='COMPLETE', completed_utc=stamp(),
             evaluation_plan_sha256=expected_sha, rebuild_plan_sha256=plan['rebuild_plan_sha256'],
             paths=6, arms=2, exact_control_rechecked=True, original_engine_hashes=ENGINE_HASHES,
             source_hashes_verified_before_after=True,
             artifacts_sha256={p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()}))
        print(json.dumps(dict(status='COMPLETE', completion_sha256=sha(dest / 'COMPLETION.json'))), flush=True)
    except Exception as error:
        dump(dest / 'FAILURE.json', dict(failed_utc=stamp(), type=type(error).__name__, message=str(error),
                                       partial_outputs_not_final=True, evaluation_plan_sha256=expected_sha))
        raise


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = engine()

    def fixture(self):
        index = pd.to_datetime(['2020-01-30', '2020-01-31', '2020-02-03', '2020-02-04', '2020-02-05'])
        px = pd.DataFrame({'A': [10., 11., 12., 13., 14.], 'B': [20., 21., 22., 23., 24.]}, index=index)
        opening = px * .9
        orders = [dict(date='2020-01-30', signal_date='2020-01-30', end='2020-02-03', symbols=['A', 'B'], scores=[.3, .2]),
                  dict(date='2020-02-03', signal_date='2020-02-03', end='2020-02-05', symbols=['B', 'A'], scores=[.4, .3])]
        meta = dict(dates=['2020-01-30', '2020-02-03'], ends=['2020-02-03', '2020-02-05'])
        return px, opening, orders, meta

    def test_original_engine_three_paths_and_exported_ledger(self):
        px, opening, orders, meta = self.fixture()
        for _, execution, cost in PATHS:
            result, equity, trades, periods, ledger = run_path(self.engine, px, opening, orders, meta, execution, cost)
            self.assertEqual(len(periods), 2)
            self.assertEqual(len(ledger), len(px))
            self.assertEqual(periods[0]['execution_date'], '2020-01-30' if execution == 'close' else '2020-01-31')
            self.assertEqual(periods[1]['execution_date'], '2020-02-03' if execution == 'close' else '2020-02-04')
            self.assertTrue(all(t['date'] != '2020-02-05' for t in trades))
            self.assertEqual(set(ledger[-1]['adjusted_units']), {'A', 'B'})
            self.assertAlmostEqual(sum(t['cost'] for t in trades), result['metrics']['fees_paid'])
            if execution == 'open': self.assertEqual(equity.iloc[0], 100000.)

    def test_cost_funds_initial_entry_and_only_differential_turnover(self):
        px, opening, orders, meta = self.fixture()
        result, equity, trades, _, _ = run_path(self.engine, px, opening, orders, meta, 'close', .0005)
        first = [t for t in trades if t['date'] == orders[0]['date']]
        self.assertAlmostEqual(sum(t['dollars'] for t in first), 100000. / 1.0005, places=8)
        self.assertAlmostEqual(equity.iloc[0], 100000. / 1.0005, places=8)
        second = [t for t in trades if t['date'] == orders[1]['date']]
        self.assertEqual({t['side'] for t in second}, {'buy', 'sell'})
        self.assertLess(sum(t['dollars'] for t in second), equity.loc[orders[1]['date']] * .1)

    def test_missing_held_price_fails_and_no_fill_forward(self):
        px, opening, orders, meta = self.fixture()
        px.loc['2020-02-04', 'A'] = np.nan
        with self.assertRaises(AssertionError): run_path(self.engine, px, opening, orders, meta, 'close', 0.)

    def test_array_parity_masks_and_values(self):
        equal_arrays({'x': np.array([1., np.nan])}, {'x': np.array([1., np.nan])})
        with self.assertRaises(AssertionError): equal_arrays({'x': np.array([1., 0.])}, {'x': np.array([1., np.nan])})
        with self.assertRaises(AssertionError): equal_arrays({'x': np.array([1.01])}, {'x': np.array([1.])})

    def test_write_once_and_root_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            dump(path, {'test': True})
            with self.assertRaises(FileExistsError): dump(path, {})
        for name in ('../elsewhere', '/tmp/elsewhere', 'fixed_incumbent_rebuild/child'):
            with self.assertRaises(ValueError): root_path(name)

    def test_attestation_rejects_tamper_failed_or_unbound_control(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'control'
            folder.mkdir()
            for name in ARTIFACTS: (folder / name).write_text('synthetic fixture only')
            report = dict(passed=True, arm='control', returns_computed=False, source_plan_sha256='a' * 64,
                          checks={k: {} for k in ('monthly', 'weekly', 'incumbent_arrays')},
                          artifacts_sha256={n: sha(folder / n) for n in ARTIFACTS})
            def write_report():
                (folder / 'PARITY.json').write_text(json.dumps(report))
            write_report()
            verify_artifacts(folder, 'a' * 64)
            with self.assertRaises(ValueError): verify_artifacts(folder, 'b' * 64)
            report['passed'] = False
            write_report()
            with self.assertRaises(ValueError): verify_artifacts(folder, 'a' * 64)
            report['passed'] = True
            write_report()
            (folder / 'DATA.npz').write_text('tampered fixture')
            with self.assertRaises(ValueError): verify_artifacts(folder, 'a' * 64)

    def test_conditional_receipt_requires_exact_control_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'conditional'
            folder.mkdir()
            for name in ARTIFACTS: (folder / name).write_text('synthetic fixture only')
            report = dict(passed=True, arm='conditional', returns_computed=False, source_plan_sha256='a' * 64,
                          original_control_required=True, control_parity_sha256='c' * 64,
                          artifacts_sha256={n: sha(folder / n) for n in ARTIFACTS})
            (folder / 'PARITY.json').write_text(json.dumps(report))
            verify_artifacts(folder, 'a' * 64, 'c' * 64)
            with self.assertRaises(ValueError): verify_artifacts(folder, 'a' * 64, 'd' * 64)

    def test_slices_use_continuous_wealth_and_shared_closing_boundary(self):
        index = pd.bdate_range('2020-01-01', periods=123)
        meta = dict(dates=[str(d.date()) for d in index[:-1]], ends=[str(d.date()) for d in index[1:]])
        equity = pd.Series(np.cumprod(1. + np.linspace(.001, .003, 123)) * 100000., index=index)
        result = split_stats(self.engine, equity, meta)
        self.assertEqual(result['early53']['end'], result['recent35']['start'])
        self.assertEqual(result['first88']['end'], result['later34']['start'])
        self.assertAlmostEqual(result['recent35']['total_return'], equity.iloc[88] / equity.iloc[53] - 1.)
        self.assertAlmostEqual(result['first88']['total_return'], equity.iloc[88] / equity.iloc[0] - 1.)
        self.assertEqual(result['early53']['trading_sessions'], 54)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['test', 'register', 'evaluate'])
    parser.add_argument('--root', default='fixed_incumbent_rebuild')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.phase == 'test':
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        require(args.plan_sha256 is not None and re.fullmatch(r'[0-9a-f]{64}', args.plan_sha256) is not None,
                'Explicit expected PLAN SHA256 is required')
        (register if args.phase == 'register' else evaluate)(root_path(args.root), args.plan_sha256)
