"""Independent read-only accounting/statistics audit of completed PTR169 results.

Run through ./run_local.sh. Imports no strategy, evaluator, fee solver or metrics.
The sole real-data output is a new write-once independent_review directory.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import sys
import unittest
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/monthly_ptr_pool_source169/fixed_incumbent_rebuild'
PRICE = ROOT / 'outputs/monthly_2016_extension/PRICES.npz'
PATHS = (('same_close_zero', 'close', 0.), ('next_open_zero', 'open', 0.),
         ('next_open_cost', 'open', .0005))
ARMS = ('control', 'conditional')
INITIAL = 100000.


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, message, absolute=1e-7, relative=2e-11):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), message + ': nonfinite')
    require(abs(actual - expected) <= absolute + relative * abs(expected),
            message + ': ' + repr((actual, expected)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Invalid local source: ' + str(path))
    return path


def metrics(days, values, initial=INITIAL):
    """Scalar implementation, independent of the replay's NumPy/pandas metrics."""
    require(len(days) == len(values) and len(values) > 1, 'Invalid metric interval')
    require(all(math.isfinite(v) and v > 0 for v in values), 'Invalid metric equity')
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    require(years > 0, 'Zero metric span')
    returns = [b / a - 1. for a, b in zip(values, values[1:])]
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.
    peak, drawdown = initial, 0.
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.)
    return dict(start=days[0], end=days[-1], years=years,
                final_equity=values[-1], cagr=(values[-1] / initial) ** (1. / years) - 1.,
                sharpe=statistics.mean(returns) / stdev * math.sqrt(252.) if stdev > 0 else None,
                max_drawdown=drawdown, trading_sessions=len(days), total_return=values[-1] / initial - 1.)


def check_metrics(actual, expected, label):
    for key, value in actual.items():
        if isinstance(value, str) or value is None or key == 'trading_sessions':
            require(expected[key] == value, label + ': ' + key)
        else:
            near(value, expected[key], label + ': ' + key)


def statistics_by_split(days, values, meta):
    results = {'full122': metrics(days, values)}
    bounds = {'early53': (meta['dates'][0], meta['ends'][52]),
              'recent35': (meta['dates'][53], meta['ends'][87]),
              'later34': (meta['dates'][88], meta['ends'][-1]),
              'first88': (meta['dates'][0], meta['ends'][87])}
    for name, (lo, hi) in bounds.items():
        left, right = days.index(lo), days.index(hi)
        rebased = [value / values[left] * INITIAL for value in values[left:right + 1]]
        results[name] = metrics(days[left:right + 1], rebased)
    return results


def ticker(symbol):
    # Frozen export alias; execution continues to use the original security key.
    return 'META' if symbol == 'research_equity_FB' else symbol


def audit_path(px, opening, orders, daily, fills, periods, exported_positions, execution, cost):
    all_days = [str(day.date()) for day in px.index]
    expected_days = [d for d in all_days if orders[0]['date'] <= d <= orders[-1]['end']]
    days = [row['date'] for row in daily]
    require(days == expected_days and len(set(days)) == len(days), 'NAV trading calendar mismatch')
    require([row['date'] for row in exported_positions] == days, 'Position calendar mismatch')
    schedule = {}
    for order in orders:
        signal = order['date']
        require(order['signal_date'] == signal, 'Signal timestamp differs from registered order')
        when = signal if execution == 'close' else all_days[all_days.index(signal) + 1]
        require(when in expected_days and when not in schedule, 'Missing/duplicate execution session')
        require(len(order['symbols']) in (0, 2, 3) and len(set(order['symbols'])) == len(order['symbols']), 'Bad target size')
        schedule[when] = order
    require([row['execution_date'] for row in periods] == list(schedule), 'Period execution calendar mismatch')
    for period, (when, order) in zip(periods, schedule.items()):
        require(period['signal_date'] == order['signal_date'] and period['symbols'] == [ticker(s) for s in order['symbols']]
                and period['scores'] == order['scores'], 'Period target/rank differs from frozen order')
    aliases = {ticker(s): s for order in orders for s in order['symbols']}
    require(len(aliases) == len({s for order in orders for s in order['symbols']}), 'Ambiguous export alias')
    grouped, seen = defaultdict(list), set()
    require([f['date'] for f in fills] == sorted(f['date'] for f in fills), 'Unordered fills')
    for fill in fills:
        when = fill['date']
        require(when in schedule, 'Extra trade outside registered monthly execution')
        require(fill['side'] in ('buy', 'sell'), 'Unknown fill side')
        require(fill['symbol'] in aliases, 'Fill security absent from all targets')
        require((when, fill['symbol']) not in seen, 'Duplicate net fill/security/session')
        seen.add((when, fill['symbol']))
        for key in ('adjusted_units', 'adjusted_price', 'dollars'):
            require(math.isfinite(fill[key]) and fill[key] > 0, 'Nonpositive/nonfinite fill ' + key)
        require(math.isfinite(fill['cost']) and fill['cost'] >= 0, 'Invalid fee')
        symbol = aliases[fill['symbol']]
        reference = float((px if execution == 'close' else opening).at[pd.Timestamp(when), symbol])
        near(fill['adjusted_price'], reference, 'Fill price differs from original PRICES', absolute=1e-12, relative=1e-12)
        dollars = fill['adjusted_units'] * reference
        near(fill['dollars'], dollars, 'Fill dollars mismatch')
        near(fill['cost'], dollars * cost, 'Fill fee mismatch', absolute=1e-9)
        grouped[when].append((fill, symbol, reference, dollars))
    cash, units, values, rows = INITIAL, {}, [], []
    max_nav_error = max_cash_error = max_equal_error = 0.
    fee_total = turnover_total = 0.
    period_map = {row['execution_date']: row for row in periods}
    for i, day in enumerate(days):
        timestamp = pd.Timestamp(day)
        if day in schedule:
            chosen = schedule[day]['symbols']
            row_price = (px if execution == 'close' else opening).loc[timestamp]
            relevant = set(chosen) | {s for s, q in units.items() if abs(q) > 1e-9}
            require(all(math.isfinite(float(row_price[s])) and row_price[s] > 0 for s in relevant), 'Missing execution price')
            before = cash + math.fsum(units.get(s, 0.) * float(row_price[s]) for s in relevant)
            day_fee = 0.
            for fill, symbol, price, dollars in grouped[day]:
                require(symbol in relevant, 'Unrelated security traded')
                signed = fill['adjusted_units'] * (1. if fill['side'] == 'buy' else -1.)
                units[symbol] = units.get(symbol, 0.) + signed
                actual_fee = dollars * cost
                cash -= signed * price + actual_fee
                day_fee += actual_fee
                turnover_total += dollars
            fee_total += day_fee
            near(day_fee, period_map[day]['fee'], 'Period fee mismatch')
            require(all(q >= -1e-7 for q in units.values()), 'Net short position')
            require(cash >= -1e-5, 'Borrowed cash')
            require(all(s in chosen or abs(q) <= 1e-7 for s, q in units.items()), 'Holding outside monthly target')
            require(all(units.get(s, 0.) > 0 for s in chosen), 'Target security missing')
            invested = [units[s] * float(row_price[s]) for s in chosen]
            after = cash + math.fsum(invested)
            near(after, before - day_fee, 'Self-financing rebalance fails', absolute=1e-5)
            if chosen:
                near(cash, 0., 'Cash remains after full allocation', absolute=1e-5, relative=0.)
                target = after / len(chosen)
                for value in invested:
                    max_equal_error = max(max_equal_error, abs(value - target))
                    near(value, target, 'Post-execution target is not equal-weight', absolute=1e-5)
        market = {}
        for symbol, quantity in units.items():
            if abs(quantity) <= 1e-9:
                continue
            close = float(px.at[timestamp, symbol])
            require(math.isfinite(close) and close > 0, 'Missing held closing price')
            market[symbol] = quantity * close
        nav = cash + math.fsum(market.values())
        require(nav > 0, 'Nonpositive account NAV')
        max_nav_error = max(max_nav_error, abs(nav - daily[i]['equity']))
        near(nav, daily[i]['equity'], 'Daily NAV mismatch', absolute=1e-5)
        exported = exported_positions[i]
        max_cash_error = max(max_cash_error, abs(cash - exported['cash']))
        near(cash, exported['cash'], 'Exported cash mismatch', absolute=1e-5)
        near(nav, exported['equity'], 'Exported position NAV mismatch', absolute=1e-5)
        for symbol in set(units) | set(exported['adjusted_units']):
            near(units.get(symbol, 0.), exported['adjusted_units'].get(symbol, 0.), 'Exported units mismatch')
        for symbol in set(market) | set(exported['market_values']):
            near(market.get(symbol, 0.), exported['market_values'].get(symbol, 0.), 'Exported market value mismatch', absolute=1e-5)
        values.append(nav)
        rows.append(dict(date=day, cash=cash, equity=nav))
    return dict(days=days, values=values, daily_cash_nav=rows, fees_paid=fee_total,
                dollar_turnover=turnover_total, trade_rows=len(fills), monthly_executions=len(schedule),
                max_nav_difference=max_nav_error, max_cash_difference=max_cash_error,
                max_equal_weight_dollar_difference=max_equal_error,
                final_units={s: q for s, q in units.items() if abs(q) > 1e-9}, final_cash=cash)


def inspect(completion_sha, plan_sha):
    source = BASE / 'evaluation/results'
    plan_path, completion_path = BASE / 'evaluation/PLAN.json', source / 'COMPLETION.json'
    require(sha(completion_path) == completion_sha and sha(plan_path) == plan_sha, 'Wrong registered completion/plan')
    completion, plan = read(completion_path), read(plan_path)
    require(completion['status'] == 'COMPLETE' and completion['paths'] == 6 and completion['arms'] == 2,
            'Evaluation incomplete')
    require(completion['evaluation_plan_sha256'] == plan_sha and not (source / 'FAILURE.json').exists(), 'Failed/unbound evaluation')
    require(plan['paths'] == [dict(name=n, execution=e, cost_per_side=c) for n, e, c in PATHS], 'Changed execution contract')
    require(plan['arms'] == list(ARMS) and plan['initial_capital'] == INITIAL, 'Changed account contract')
    require(plan['splits'] == ['full122', 'early53', 'recent35', 'later34', 'first88'], 'Changed metric contract')
    runtime = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    require(plan['runtime'] == runtime, 'Different runtime')
    pins = {str(plan_path): plan_sha, str(completion_path): completion_sha, str(Path(__file__)): sha(__file__)}
    for name, digest in completion['artifacts_sha256'].items():
        require(Path(name).name == name, 'Unsafe result path')
        pins[str(source / name)] = digest
    needed = [PRICE] + [BASE / arm / 'META.json' for arm in ARMS] + [BASE / arm / 'ORDERS.json' for arm in ARMS]
    for path in needed:
        key = str(path.relative_to(ROOT))
        require(key in plan['input_sha256'], 'Audit input is not pinned')
        pins[str(path)] = plan['input_sha256'][key]
    for name, digest in pins.items():
        require(sha(local_path(name)) == digest, 'Audit input digest mismatch: ' + name)
    with np.load(PRICE, allow_pickle=False) as pack:
        index = pd.to_datetime(pack['dates'])
        symbols = pack['symbols'].astype(str)
        require(index.is_unique and index.is_monotonic_increasing and len(set(symbols)) == len(symbols), 'Invalid price axes')
        px = pd.DataFrame(pack['close'], index=index, columns=symbols)
        opening = pd.DataFrame(pack['opening'], index=index, columns=symbols)
    expected = read(source / 'RESULTS.json')
    reports, ledger_rows = {}, {}
    for arm in ARMS:
        meta = read(BASE / arm / 'META.json')['monthly']
        orders = read(source / (arm + '_ORDERS.json'))
        require(orders == read(BASE / arm / 'ORDERS.json'), 'Exported targets differ from registered targets')
        require(len(orders) == len(meta['dates']) == len(meta['ends']) == 122, 'Unexpected monthly count')
        require([o['date'] for o in orders] == meta['dates'] and [o['end'] for o in orders] == meta['ends'], 'Monthly axis mismatch')
        require(meta['ends'][:-1] == meta['dates'][1:], 'Noncontiguous month intervals')
        require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24', 'Evaluation span mismatch')
        require(meta['ends'][52] == meta['dates'][53] == '2020-12-31' and
                meta['ends'][87] == meta['dates'][88] == '2023-11-30', 'Wrong split boundaries')
        # The first 2016-08-17 initialization is intentionally midmonth.
        for order in orders[1:]:
            timestamp = pd.Timestamp(order['date'])
            last_session = index[(index.year == timestamp.year) & (index.month == timestamp.month)][-1]
            require(timestamp == last_session, 'Non-month-end recurring signal')
        reports[arm] = {}
        for name, execution, cost in PATHS:
            stem = arm + '_' + name
            data = [read(source / (stem + '_' + suffix + '.json'))
                    for suffix in ('DAILY', 'TRADES', 'HOLDINGS', 'DAILY_POSITIONS')]
            audited = audit_path(px, opening, orders, *data, execution, cost)
            wanted = expected['arms'][arm][name]
            slices = statistics_by_split(audited['days'], audited['values'], meta)
            check_metrics(slices['full122'], wanted['metrics'], stem)
            for split, actual in slices.items():
                check_metrics(actual, wanted['slices'][split], stem + '/' + split)
            for key in ('fees_paid', 'dollar_turnover', 'trade_rows'):
                near(audited[key], wanted['metrics'][key], stem + '/' + key)
            annual = []
            days, values = audited['days'], audited['values']
            for year in sorted({d[:4] for d in days}):
                ids = [i for i, day in enumerate(days) if day.startswith(year)]
                if len(ids) < 2:
                    continue
                before = values[ids[0] - 1] if ids[0] else INITIAL
                annual.append(dict(year=int(year), return_=values[ids[-1]] / before - 1., partial=int(year) in (2016, 2026)))
            require(len(annual) == len(wanted['metrics']['annual_returns']), 'Annual count mismatch')
            for actual, recorded in zip(annual, wanted['metrics']['annual_returns']):
                require(actual['year'] == recorded['year'] and actual['partial'] == recorded['partial'], 'Annual label mismatch')
                near(actual['return_'], recorded['return_'], 'Annual return mismatch')
            ledger_rows[stem] = audited.pop('daily_cash_nav')
            audited.pop('days'); audited.pop('values')
            reports[arm][name] = dict(audited, slices=slices, annual_returns=annual, passed=True)
    for name, _, _ in PATHS:
        for split in plan['splits']:
            for key in ('cagr', 'sharpe', 'max_drawdown', 'total_return'):
                value = reports['conditional'][name]['slices'][split][key] - reports['control'][name]['slices'][split][key]
                near(value, expected['conditional_minus_control'][name][split][key], 'Conditional difference mismatch')
    for name, digest in pins.items():
        require(sha(local_path(name)) == digest, 'Audit input changed during verification: ' + name)
    return dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,
                independent_of_strategy_and_original_engine=True, reused_fee_solver=False, reused_stats_function=False,
                scope='Execution accounting and metric validation only; source quality/model methodology is not certified.',
                completion_sha256=completion_sha, evaluation_plan_sha256=plan_sha, paths=reports,
                checked_paths=6, checked_slices=30, verified_input_sha256=pins), ledger_rows


class Tests(unittest.TestCase):
    def test_scalar_statistics(self):
        result = metrics(['2020-01-01', '2020-01-02', '2021-01-01'], [100., 110., 99.], initial=100.)
        self.assertAlmostEqual(result['total_return'], -.01)
        self.assertAlmostEqual(result['max_drawdown'], -.1)
        self.assertAlmostEqual(result['sharpe'], 0., places=12)
        self.assertAlmostEqual(result['cagr'], .99 ** (365.2425 / 366.) - 1.)

    def fixture(self, execution='close', cost=0.):
        dates = pd.to_datetime(['2020-01-30', '2020-01-31', '2020-02-03'])
        px = pd.DataFrame({'A': [10., 11., 12.], 'B': [20., 18., 22.]}, index=dates)
        opening = px * .9
        day = '2020-01-30' if execution == 'close' else '2020-01-31'
        row = (px if execution == 'close' else opening).loc[day]
        post = INITIAL / (1. + cost)
        units = {s: post / 2. / float(row[s]) for s in ('A', 'B')}
        fills = [dict(date=day, symbol=s, side='buy', adjusted_units=units[s], adjusted_price=float(row[s]),
                      dollars=post / 2., cost=post / 2. * cost) for s in ('A', 'B')]
        order = dict(date='2020-01-30', signal_date='2020-01-30', end='2020-02-03', symbols=['A', 'B'], scores=[.5, .4])
        periods = [dict(execution_date=day, signal_date=order['date'], symbols=['A', 'B'], scores=order['scores'], fee=post * cost)]
        daily, positions = [], []
        for timestamp in dates:
            ds = str(timestamp.date())
            active = ds >= day
            market = {s: units[s] * float(px.at[timestamp, s]) for s in units} if active else {}
            cash = 0. if active else INITIAL
            nav = cash + sum(market.values())
            daily.append(dict(date=ds, equity=nav))
            positions.append(dict(date=ds, cash=cash, adjusted_units=units if active else {}, market_values=market, equity=nav))
        return [px, opening, [order], daily, fills, periods, positions, execution, cost]

    def test_three_paths_and_fee_accounting(self):
        for _, execution, cost in PATHS:
            result = audit_path(*self.fixture(execution, cost))
            self.assertEqual(result['trade_rows'], 2)
            self.assertAlmostEqual(result['fees_paid'], INITIAL / (1. + cost) * cost)

    def test_reject_price_fee_side_and_extra_date(self):
        for key, value in [('adjusted_price', 11.), ('cost', 2.), ('side', 'hold'), ('date', '2020-02-03'), ('adjusted_units', -1.)]:
            fixture = self.fixture()
            fixture[4][0][key] = value
            with self.assertRaises(ValueError):
                audit_path(*fixture)

    def test_reject_non_equal_weight_self_consistent_ledger(self):
        fixture = self.fixture()
        for fill, scale in zip(fixture[4], (1.2, .8)):
            for key in ('adjusted_units', 'dollars', 'cost'):
                fill[key] *= scale
        with self.assertRaisesRegex(ValueError, 'equal-weight'):
            audit_path(*fixture)

    def test_reject_extra_fill_duplicate_and_missing_target(self):
        for mode in ('duplicate', 'missing'):
            fixture = self.fixture()
            if mode == 'duplicate':
                fixture[4].append(copy.deepcopy(fixture[4][0]))
            else:
                fixture[4].pop()
            with self.assertRaises(ValueError):
                audit_path(*fixture)

    def test_differential_rebalance_sells_and_buys(self):
        fixture = self.fixture()
        px, _, orders, daily, fills, periods, positions, _, _ = fixture
        orders[0]['end'] = '2020-01-31'
        orders.append(dict(date='2020-01-31', signal_date='2020-01-31', end='2020-02-03',
                           symbols=['B', 'A'], scores=[.6, .5]))
        periods.append(dict(execution_date='2020-01-31', signal_date='2020-01-31',
                            symbols=['B', 'A'], scores=[.6, .5], fee=0.))
        fills += [dict(date='2020-01-31', symbol='A', side='sell', adjusted_units=5000. / 11.,
                       adjusted_price=11., dollars=5000., cost=0.),
                  dict(date='2020-01-31', symbol='B', side='buy', adjusted_units=5000. / 18.,
                       adjusted_price=18., dollars=5000., cost=0.)]
        quantities = {'A': 50000. / 11., 'B': 50000. / 18.}
        for i in (1, 2):
            day = positions[i]['date']
            market = {s: q * float(px.at[pd.Timestamp(day), s]) for s, q in quantities.items()}
            positions[i] = dict(date=day, cash=0., adjusted_units=quantities, market_values=market, equity=sum(market.values()))
            daily[i]['equity'] = positions[i]['equity']
        result = audit_path(*fixture)
        self.assertEqual(result['trade_rows'], 4)
        self.assertAlmostEqual(result['dollar_turnover'], 110000.)

    def test_continuous_split_boundaries_rebase(self):
        dates = [str(d.date()) for d in pd.bdate_range('2020-01-01', periods=123)]
        values = [INITIAL * (1.002 ** i) for i in range(123)]
        meta = dict(dates=dates[:-1], ends=dates[1:])
        result = statistics_by_split(dates, values, meta)
        self.assertEqual(result['early53']['end'], result['recent35']['start'])
        self.assertEqual(result['first88']['end'], result['later34']['start'])
        self.assertAlmostEqual(result['recent35']['total_return'], values[88] / values[53] - 1.)
        self.assertEqual(result['first88']['trading_sessions'], 89)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('test', 'run'))
    parser.add_argument('--completion-sha256')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.phase == 'test':
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        require(args.completion_sha256 is not None and args.plan_sha256 is not None, 'Both expected digests required')
        output = BASE / 'independent_review'
        require(not output.exists(), 'Independent review is write-once')
        report, rows = inspect(args.completion_sha256, args.plan_sha256)
        output.mkdir()
        dump(output / 'REPORT.json', report)
        for name, ledger in rows.items():
            dump(output / (name + '_CASH_NAV.json'), ledger)
        dump(output / 'COMPLETION.json', dict(status='PASS', checked_paths=6, checked_slices=30,
             evaluation_completion_sha256=args.completion_sha256,
             artifacts_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
        print(json.dumps(dict(status='PASS', checked_paths=6, checked_slices=30,
                              completion_sha256=sha(output / 'COMPLETION.json'))))
