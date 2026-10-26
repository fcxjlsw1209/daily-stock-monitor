"""Read-only attribution of bound round176 exports; no feature/replay imports."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_sector_dimension176'
DEST = OUT / 'attribution'
ARMS = ('incumbent_local67', 'sector_dimension_timed', 'sector_dimension_mean')
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
SHOW_PATHS = ('same_close_zero', 'next_open_cost')
SLICES = ('full122', 'first88', 'early53', 'recent35', 'later34')
COMPARISONS = {'timed_vs_incumbent': (ARMS[1], ARMS[0]),
               'mean_vs_incumbent': (ARMS[2], ARMS[0]),
               'timed_vs_mean': (ARMS[1], ARMS[2])}


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10),
            message + ': ' + repr((actual, expected)))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(pins):
    for name, expected in pins.items():
        path = Path(name).resolve()
        require('CloudStorage' not in path.parts and
                (path.is_relative_to(ROOT) or str(path).startswith('/Library/Developer/CommandLineTools/')),
                'Unapproved input path: ' + name)
        require(path.is_file() and sha(path) == expected, 'Input hash mismatch: ' + name)


def metrics(rows, initial):
    days = [r['date'] for r in rows]
    values = [r['equity'] for r in rows]
    require(len(values) > 1 and all(math.isfinite(v) and v > 0 for v in values), 'Invalid NAV slice')
    years = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 365.2425
    returns = [b / a - 1. for a, b in zip(values, values[1:])]
    deviation = statistics.stdev(returns)
    require(years > 0 and deviation > 0, 'Degenerate metric slice')
    peak, drawdown = initial, 0.
    for value in values:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.)
    return dict(start=days[0], end=days[-1], years=years,
                cagr=(values[-1] / initial) ** (1. / years) - 1.,
                sharpe=statistics.mean(returns) / deviation * math.sqrt(252.),
                max_drawdown=drawdown, observations=len(values),
                wealth_multiple=values[-1] / initial)


def decompose(factors, reference_cagr, years):
    require(all(math.isfinite(f) and f > 0 for f in factors) and years > 0, 'Invalid factors')
    logs = [math.log(f) for f in factors]
    total = math.fsum(logs)
    difference = (1. + reference_cagr) * math.expm1(total / years)
    slope = difference / total if total else (1. + reference_cagr) / years
    contributions = [slope * value for value in logs]
    near(math.fsum(contributions), difference, 'CAGR contribution sum')
    return dict(terminal_wealth_ratio=math.exp(total),
                annual_growth_multiplier=math.exp(total / years),
                cagr_difference=difference,
                factors=factors, log_factors=logs,
                annual_growth_factors=[math.exp(v / years) for v in logs],
                cagr_contributions=contributions)


def guard(candidate, reference, gain):
    return dict(cagr=candidate['training']['cagr'] >= reference['training']['cagr'] + gain - 1e-12,
                sharpe=candidate['training']['sharpe'] >= reference['training']['sharpe'] - .05,
                drawdown=candidate['training']['max_drawdown'] >= reference['training']['max_drawdown'] - .03,
                early=candidate['early']['cagr'] >= reference['early']['cagr'] - .025,
                recent=candidate['recent']['cagr'] >= reference['recent']['cagr'] - .025)


def display(symbol):
    return {'research_equity_FB': 'META'}.get(symbol, symbol)


def markdown(report):
    names = dict(zip(ARMS, ('基准 Local67', '当期D₀×Q', '历史均值D̄×Q')))
    segments = dict(zip(SLICES, ('全期122', '训练88', '早期53', '近期35', '后段34')))
    decision = ('当期D₀×Q候选通过事先冻结的训练期门槛。' if report['selection']['admitted'] else '当期D₀×Q候选未通过事先冻结的训练期门槛，保留 Local67。')
    lines = ['# 第176轮结果归因', '',
        '区间：2016-08-17 至 2026-09-24。' + decision + '后段34期仅用于冻结后的诊断。', '',
        '只读取已通过哈希验证的结果，不重新计算策略、调参或改变观察账户。股票池是历史代理 Top32＋theme17；样本已重复使用，并非独立样本外或完整时点数据。D 是固定9行业ETF两两相关系数平方形成的相关分散度，负相关平方同样计入冗余；不是资金流。Q 是个股减其原映射行业ETF的210日相对收益强度。60个历史D窗口相互重叠，并非独立样本；仅当期D₀×Q与历史均值D̄×Q的同支持域比较隔离D的时变作用。', '',
        '下表依次为 CAGR／Sharpe／最大回撤（回撤保留负号）。同收盘路径为零成本研究假设；次开盘路径每边5bp。', '']
    for path, title in [('same_close_zero', '同收盘零成本'), ('next_open_cost', '次开盘每边5bp')]:
        lines += ['## ' + title, '', '|区间|基准 Local67|当期D₀×Q|历史均值D̄×Q|', '|---|---:|---:|---:|']
        for segment in SLICES:
            row = []
            for arm in ARMS:
                m = report['metrics'][arm][path][segment]
                row.append(f"{m['cagr']:.4%} / {m['sharpe']:.4f} / {m['max_drawdown']:.4%}")
            lines.append('|' + segments[segment] + '|' + '|'.join(row) + '|')
        lines.append('')
    lines += ['## 发生持仓集合变化的月份', '',
              '按信号日标识月份；三臂持仓按基准／当期D₀／历史均值D̄列示。META 为原始标识 research_equity_FB 的显示别名。', '',
              '|信号日 → 期末|基准|当期D₀|历史均值D̄|', '|---|---|---|---|']
    for row in report['changed_months']:
        holdings = [', '.join(display(s) for s in row['holdings'][arm]) or '现金' for arm in ARMS]
        lines.append('|' + row['signal_date'] + ' → ' + row['end'] + '|' + '|'.join(holdings) + '|')
    lines += ['', '月收益按信号日收盘至期末收盘从连续净值直接计算。次开盘路径包括旧持仓隔夜、下一开盘换仓及费用，影响可能延续到下一个持仓集合相同的月份。下表“差／贡献／倍数”均相对基准，顺序为当期D₀／历史均值D̄。', '']
    for path, title in [('same_close_zero', '同收盘月度分解'), ('next_open_cost', '次开盘5bp月度分解')]:
        lines += ['## ' + title, '', '|信号日|三臂月收益（基／当期／均值）|月收益差 pp|全期CAGR贡献 pp|财富倍数|', '|---|---:|---:|---:|---:|']
        for row in report['changed_months']:
            v = row['paths'][path]
            pairs = [v['comparisons'][key] for key in ('timed_vs_incumbent', 'mean_vs_incumbent')]
            ret = ' / '.join(f"{v['returns'][a]:.3%}" for a in ARMS)
            delta = ' / '.join(f"{x['return_difference_pp']:+.4f}" for x in pairs)
            contribution = ' / '.join(f"{x['full_cagr_contribution_pp']:+.4f}" for x in pairs)
            factor = ' / '.join(f"{x['wealth_factor']:.6f}" for x in pairs)
            lines.append(f"|{row['signal_date']}|{ret}|{delta}|{contribution}|{factor}|")
        lines.append('')
    lines += ['## 全期乘数核对', '', '|路径／比较|终值比|年化增长倍数|CAGR差 pp|', '|---|---:|---:|---:|']
    for path in SHOW_PATHS:
        for key, value in report['decompositions'][path].items():
            target, reference = COMPARISONS[key]
            title = ('同收盘' if path == 'same_close_zero' else '次开盘5bp') + '：' + names[target] + '／' + names[reference]
            lines.append(f"|{title}|{value['terminal_wealth_ratio']:.9f}|{value['annual_growth_multiplier']:.9f}|{100*value['cagr_difference']:+.5f}|")
    extra = report['unchanged_holdings_return_effects']
    if extra:
        lines += ['', '持仓集合未变但连续净值仍有差异的期间（主要是次开盘换仓的跨期影响）也已纳入总乘积：', '',
                  '|信号日|路径|当期D₀月差 pp|均值D̄月差 pp|', '|---|---|---:|---:|']
        for row in extra:
            lines.append(f"|{row['signal_date']}|{row['path']}|{row['timed_difference_pp']:+.6f}|{row['mean_difference_pp']:+.6f}|")
    lines += ['',
        '分解公式：每期财富倍数 fᵢ=(1+r目标,ᵢ)/(1+r参照,ᵢ)，终值比=∏fᵢ，年化增长倍数=(∏fᵢ)^(1/Y)。CAGR贡献按对数倍数分摊：ΔCAGR×log(fᵢ)/Σlog(f)，故贡献相加精确还原全期差；这是会计分解约定，不是额外反事实回测或唯一因果归因。', '',
        '验证：45个切片的 CAGR、Sharpe、最大回撤均从保存净值独立复算，与生产结果和独立执行回执一致；训练18个切片与冻结 TRAIN_RESULTS 一致。所有122期乘积与终值比一致。完整122期和三组两两比较保存在 REPORT.json。', '']
    return '\n'.join(lines)


def main(plan_sha, result_sha):
    for label, digest in [('PLAN', plan_sha), ('result completion', result_sha)]:
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                'An explicit valid SHA-256 is required for ' + label)
    require(not DEST.exists(), 'Attribution output is write-once')
    require(sha(OUT / 'PLAN.json') == plan_sha, 'Unexpected PLAN')
    require(sha(OUT / 'results/COMPLETION.json') == result_sha, 'Result completion not ready or changed')
    plan = read(OUT / 'PLAN.json')
    prepared = read(OUT / 'prepared/COMPLETION.json')
    completed = read(OUT / 'results/COMPLETION.json')
    frozen = read(OUT / 'selected/FROZEN.json')
    pins = dict(plan['input_sha256'])
    for path in (OUT / 'PLAN.json', OUT / 'prepared/COMPLETION.json',
                 OUT / 'results/COMPLETION.json', OUT / 'selected/FROZEN.json'):
        pins[str(path)] = sha(path)
    for directory, receipt in [('prepared', prepared), ('results', completed)]:
        require(receipt['status'] == 'COMPLETE' and receipt['plan_sha256'] == plan_sha, 'Incomplete receipt')
        for name, digest in receipt['artifacts_sha256'].items():
            require(Path(name).name == name, 'Unexpected artifact path')
            pins[str(OUT / directory / name)] = digest
    require(len(prepared['artifacts_sha256']) == 7 and len(completed['artifacts_sha256']) == 38,
            'Unexpected artifact coverage')
    for key, name in [('prepared_sha256', 'prepared/COMPLETION.json'),
                      ('orders_sha256', 'prepared/ORDERS.json'),
                      ('training_sha256', 'selected/TRAIN_RESULTS.json')]:
        require(sha(OUT / name) == frozen[key], 'Frozen binding changed: ' + name)
        pins[str(OUT / name)] = frozen[key]
    require(frozen['plan_sha256'] == plan_sha and completed['frozen_sha256'] == sha(OUT / 'selected/FROZEN.json'), 'Frozen selection changed')
    verify(pins)  # No full-period numerical analysis precedes this check.
    results = read(OUT / 'results/RESULTS.json')
    independent = read(OUT / 'results/INDEPENDENT_OUTCOME.json')
    training = read(OUT / 'selected/TRAIN_RESULTS.json')
    meta, orders = read(OUT / 'prepared/META.json'), read(OUT / 'prepared/ORDERS.json')
    require(frozen['selection'] == results['selection'], 'Result selection differs')
    require(frozen['selection']['first88_only'] is True and frozen['selection']['later34_used'] is False, 'Selection boundary changed')
    require(prepared['created_utc'] < frozen['created_utc'] < completed['created_utc'], 'Receipt sequence changed')
    require(independent['passed'] is True and independent['checked_paths'] == 9 and independent['checked_slices'] == 45
            and independent['independent_accounts'] is True and independent['independent_of_production_model_and_engine'] is True
            and independent['terminal_liquidation'] is False, 'Independent execution receipt incomplete')
    require(meta['dates'][0] == '2016-08-17' and meta['ends'][-1] == '2026-09-24'
            and len(meta['dates']) == len(meta['ends']) == 122 and set(orders) == set(ARMS), 'Unexpected period or arms')
    require(meta['ends'][:-1] == meta['dates'][1:], 'Attribution periods are not contiguous')
    for arm in ARMS:
        require(len(orders[arm]) == 122, 'Missing orders')
        for i, order in enumerate(orders[arm]):
            require(order['date'] == meta['dates'][i] and order['end'] == meta['ends'][i], 'Order/period mismatch')
    bounds = dict(full122=(meta['dates'][0], meta['ends'][-1]),
                  first88=(meta['dates'][0], meta['ends'][87]), early53=(meta['dates'][0], meta['ends'][52]),
                  recent35=(meta['dates'][53], meta['ends'][87]), later34=(meta['dates'][88], meta['ends'][-1]))
    calculated, nav = {}, {}
    for arm in ARMS:
        calculated[arm] = {}
        for path in PATHS:
            rows = read(OUT / 'results' / (arm + '_' + path + '_DAILY.json'))
            nav[arm, path] = {r['date']: r['equity'] for r in rows}
            require(len(nav[arm, path]) == len(rows) and list(nav[arm, path]) == sorted(nav[arm, path]), 'NAV order/duplicates')
            calculated[arm][path] = {}
            for segment, (start, end) in bounds.items():
                part = [r for r in rows if start <= r['date'] <= end]
                require(part[0]['date'] == start and part[-1]['date'] == end, 'Missing NAV endpoints')
                value = metrics(part, 100000. if segment == 'full122' else part[0]['equity'])
                calculated[arm][path][segment] = value
                for key in ('cagr', 'sharpe', 'max_drawdown'):
                    near(value[key], results['arms'][arm][path]['slices'][segment][key], 'Production metric ' + key)
                    near(value[key], independent['paths'][arm][path]['slices'][segment][key], 'Independent metric ' + key)
            if path in SHOW_PATHS:
                for short, full in [('training', 'first88'), ('early', 'early53'), ('recent', 'recent35')]:
                    for key in ('cagr', 'sharpe', 'max_drawdown'):
                        near(calculated[arm][path][full][key], training[arm][path][short][key], 'Frozen training ' + key)
    changes = {arm: [i for i in range(122) if set(orders[arm][i]['symbols']) != set(orders[ARMS[0]][i]['symbols'])]
               for arm in ARMS[1:]}
    require([i for i in changes[ARMS[1]] if i < 88] == frozen['selection']['changed_months'], 'Training changed-month mismatch')
    guard_details = {}
    for key, path, reference, gain in [('close_vs_incumbent', 'same_close_zero', ARMS[0], .005),
                                     ('open_vs_incumbent', 'next_open_cost', ARMS[0], 0.),
                                     ('close_vs_mean_dimension', 'same_close_zero', ARMS[2], 0.),
                                     ('open_vs_mean_dimension', 'next_open_cost', ARMS[2], 0.)]:
        details = guard(training[ARMS[1]][path], training[reference][path], gain)
        require(all(details.values()) == frozen['selection']['guards'][key], 'Selection guard mismatch')
        guard_details[key] = details
    changed_training = [i for i in changes[ARMS[1]] if i < 88]
    support = len(changed_training) >= 3 and any(i < 53 for i in changed_training) and any(i >= 53 for i in changed_training)
    require(support == frozen['selection']['guards']['changed_support'], 'Changed support guard differs')
    admitted = all(all(details.values()) for details in guard_details.values()) and support
    require(admitted == frozen['selection']['admitted'] and
            frozen['selection']['primary'] == (ARMS[1] if admitted else ARMS[0]), 'Frozen primary differs')
    guard_details['changed_support'] = dict(passed=support, changed_training=changed_training)
    changed_indices = set().union(*[set(v) for v in changes.values()])
    periods = []
    for i, (start, end) in enumerate(zip(meta['dates'], meta['ends'])):
        row = dict(index=i, signal_date=start, end=end, stage='first88' if i < 88 else 'later34',
                   holdings={a: orders[a][i]['symbols'] for a in ARMS},
                   changed_vs_incumbent={a: i in changes[a] for a in ARMS[1:]}, paths={})
        for path in SHOW_PATHS:
            growth = {a: nav[a, path][end] / nav[a, path][start] for a in ARMS}
            row['paths'][path] = dict(returns={a: growth[a] - 1. for a in ARMS}, comparisons={})
            for key, (target, reference) in COMPARISONS.items():
                row['paths'][path]['comparisons'][key] = dict(
                    wealth_factor=growth[target] / growth[reference],
                    return_difference_pp=100. * (growth[target] - growth[reference]))
        periods.append(row)
    decomposition = {}
    for path in SHOW_PATHS:
        decomposition[path] = {}
        for key, (target, reference) in COMPARISONS.items():
            near(nav[target, path][meta['dates'][0]], nav[reference, path][meta['dates'][0]], 'Initial NAV differs')
            base = calculated[reference][path]['full122']
            factors = [r['paths'][path]['comparisons'][key]['wealth_factor'] for r in periods]
            dec = decompose(factors, base['cagr'], base['years'])
            near(dec['terminal_wealth_ratio'], nav[target, path][meta['ends'][-1]] / nav[reference, path][meta['ends'][-1]], 'Terminal factorization')
            near(dec['cagr_difference'], calculated[target][path]['full122']['cagr'] - base['cagr'], 'CAGR factorization')
            for i, row in enumerate(periods):
                row['paths'][path]['comparisons'][key].update(
                    log_wealth_factor=dec['log_factors'][i], annual_growth_factor=dec['annual_growth_factors'][i],
                    full_cagr_contribution_pp=100. * dec['cagr_contributions'][i])
            decomposition[path][key] = {k: v for k, v in dec.items() if not isinstance(v, list)}
    extras = []
    for row in periods:
        if row['index'] in changed_indices:
            continue
        for path in SHOW_PATHS:
            c = row['paths'][path]['comparisons']
            if any(abs(c[k]['return_difference_pp']) > 1e-8 for k in ('timed_vs_incumbent', 'mean_vs_incumbent')):
                extras.append(dict(index=row['index'], signal_date=row['signal_date'], path=path,
                    timed_difference_pp=c['timed_vs_incumbent']['return_difference_pp'],
                    mean_difference_pp=c['mean_vs_incumbent']['return_difference_pp']))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), period=['2016-08-17', '2026-09-24'],
        verification=dict(plan_sha256=plan_sha, result_completion_sha256=result_sha,
            input_hash_count=len(pins), prepared_artifacts=7, result_artifacts=38,
            metric_slices_independently_recomputed=45, frozen_training_slices_checked=18,
            all122_period_factors_checked=True, no_strategy_or_feature_imports=True,
            no_new_strategy_runs=True, no_parameter_search=True, inputs_unchanged=True),
        selection=frozen['selection'], guard_details=guard_details, changed_indices=changes,
        metrics=calculated,
        decompositions=decomposition, changed_months=[r for r in periods if r['index'] in changed_indices],
        unchanged_holdings_return_effects=extras, all122_periods=periods,
        attribution_method=dict(monthly_return='Saved NAV(end)/NAV(signal close)-1; open execution includes old-position overnight and rebalancing fees',
            wealth_factor='(1+target_period_return)/(1+reference_period_return)',
            annual_growth_factor='wealth_factor**(1/full_period_years)',
            additive_cagr_contribution='total_CAGR_difference*log(period_wealth_factor)/sum(log(period_wealth_factor)); continuous limit when total log ratio is zero',
            interpretation='Exact accounting decomposition, not a unique causal effect or a counterfactual strategy replay'),
        limitations=['Proxy historical Top32+theme17 universe, retrospectively retrieved source prices, repeatedly reused sample; not pristine OOS/PIT',
            'Same-close zero-cost execution is a research assumption; next-open comparison charges 5bp per side',
            'D is one minus the mean squared pairwise correlation of nine fixed industry ETFs; negative correlations also count as redundancy, not fund flows',
            'Q is standardized stock-minus-own-sector relative log-return strength; XLC/XLRE may serve as original mapped Q benchmarks but are outside the fixed nine-sector D',
            'The60 historical correlation windows overlap and are not independent; only timed-vs-mean on common support isolates varying D within this Q definition',
            'Frozen candidate selection used only first88; later34 diagnostics cannot change selection'],
        input_sha256=pins, script_sha256=sha(Path(__file__)))
    content = markdown(report)
    verify(pins)
    DEST.mkdir()
    with (DEST / 'REPORT.json').open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    with (DEST / 'REPORT.md').open('x') as stream:
        stream.write(content)
    verify(pins)
    receipt = dict(status='COMPLETE', created_utc=datetime.now(timezone.utc).isoformat(),
        artifacts_sha256={n: sha(DEST / n) for n in ('REPORT.json', 'REPORT.md')},
        script_sha256=sha(Path(__file__)), verified_input_count=len(pins), inputs_unchanged=True)
    with (DEST / 'COMPLETION.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(receipt=receipt, changed_indices=changes, decompositions=decomposition,
        full_metrics={a: {p: calculated[a][p]['full122'] for p in SHOW_PATHS} for a in ARMS}), indent=2))


class DecompositionTests(unittest.TestCase):
    def test_exact_growth_identity(self):
        factors = [1.1, .8, 1.2]
        d = decompose(factors, .5, 7.)
        near(d['terminal_wealth_ratio'], math.prod(factors), 'Product')
        near(.5 + d['cagr_difference'], 1.5 * math.prod(factors) ** (1/7.) - 1., 'CAGR')
        near(sum(d['cagr_contributions']), d['cagr_difference'], 'Sum')

    def test_unchanged_and_offsetting_factors(self):
        d = decompose([1., 1.], .5, 10.)
        self.assertEqual(d['cagr_contributions'], [0., 0.])
        d = decompose([2., .5], .5, 10.)
        near(sum(d['cagr_contributions']), 0., 'Offsetting sum')

    def test_explicit_receipt_pins_required(self):
        for plan, result in ((None, None), ('a'*64, None), ('a'*64, 'Z'*64), ('short', 'b'*64)):
            with self.assertRaisesRegex(ValueError, 'explicit valid SHA-256'):
                main(plan, result)

    def test_rejects_invalid_factors(self):
        for value in (0., -1., math.inf, math.nan):
            with self.assertRaises(ValueError):
                decompose([value], .5, 10.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--result-sha256')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecompositionTests)
        require(unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(), 'Attribution tests failed')
    else:
        main(args.plan_sha256, args.result_sha256)
