"""Read-only attribution of bound round186 exports; no feature/replay imports."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/monthly_aux_reallocation186'
DEST = OUT / 'attribution'
ARMS = ('incumbent_local67', 'aux_reallocate_0025', 'aux_reallocate_0050', 'aux_reallocate_0100')
DOSES = dict(zip(ARMS[1:], (.025, .05, .10)))
PATHS = ('same_close_zero', 'next_open_zero', 'next_open_cost')
SHOW_PATHS = ('same_close_zero', 'next_open_cost')
SLICES = ('full122', 'first88', 'early53', 'recent35', 'later34')
COMPARISONS = {arm + '_vs_incumbent': (arm, ARMS[0]) for arm in ARMS[1:]}


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


def audit_selection(training, changes, frozen):
    details, passed = {}, []
    for arm in ARMS[1:]:
        changed = [i for i in changes[arm] if i < 88]
        checks = dict(close_vs_incumbent=guard(training[arm]['same_close_zero'], training[ARMS[0]]['same_close_zero'], .005),
                      open_vs_incumbent=guard(training[arm]['next_open_cost'], training[ARMS[0]]['next_open_cost'], 0.))
        support = len(changed) >= 3 and any(i < 53 for i in changed) and any(i >= 53 for i in changed)
        flags = {k: all(v.values()) for k,v in checks.items()}
        flags['changed_support'] = support
        admitted = all(flags.values())
        expected = frozen['per_candidate'][arm]
        require(expected['guards'] == flags and expected['changed_months'] == changed
                and expected['admitted'] == admitted, 'Per-candidate frozen selection mismatch')
        near(expected['training_close_cagr'], training[arm]['same_close_zero']['training']['cagr'], 'Training choice metric')
        details[arm] = dict(checks=checks, changed_support=support, changed_training=changed, admitted=admitted)
        if admitted:
            passed.append(arm)
    primary = sorted(passed, key=lambda a: (-training[a]['same_close_zero']['training']['cagr'], DOSES[a]))[0] if passed else ARMS[0]
    require(frozen['primary'] == primary and frozen['admitted'] == bool(passed), 'Frozen primary mismatch')
    require(frozen['admitted_candidates'] == passed and frozen['selected_lambda'] == DOSES.get(primary, 0.), 'Frozen dose/list mismatch')
    return details


def markdown(report):
    names = dict(zip(ARMS, ('基准 Local67', '重分配2.5%', '重分配5%', '重分配10%')))
    segments = dict(zip(SLICES, ('全期122', '训练88', '早期53', '近期35', '后段34')))
    selected = report['selection']['primary']
    lines = ['# 第186轮：波动项向原辅助权重小幅重分配', '',
        '区间：2016-08-17 至 2026-09-24。训练期选定：' + names[selected] + '。', '',
        '这是完整 Local67 的有限参数敏感性测试。三剂量提前固定，前88期筛选，后34期只诊断。全部时期已多次用于研究，并非独立样本外；历史代理 Top32＋theme17 股票池及行情数据版本的限制仍然存在。', '',
        '公式：delta=a×[(baseline−core)−(1−core_weight)×原全池波动排名]。a分别为2.5%、5%、10%，表示原非波动辅助贡献按比例增加，同时从原波动项转出对应预算，并非直接减少同等百分点的波动权重。方向是在第185轮失败后选定的新增参数搜索；字段lambda仍表示正幅度，等分时取最小幅度。原长期动量、已有模型改分和原support不变，episode只执行一次，首个部分月份不调整。原核心收益跳过最近42日；当前126日波动和63日风险分母保持原定义。月内固定单位，没有新预测模型拟合。', '',
        '下表为 CAGR／Sharpe／最大回撤。同收盘路径零成本；次开盘路径每边5bp。', '']
    for path, title in [('same_close_zero','同收盘零成本'),('next_open_cost','次开盘每边5bp')]:
        lines += ['## '+title,'','|区间|'+'|'.join(names[a] for a in ARMS)+'|','|---|'+'---:|'*len(ARMS)]
        for segment in SLICES:
            vals=[]
            for arm in ARMS:
                m=report['metrics'][arm][path][segment]
                vals.append(f"{m['cagr']:.2%}／{m['sharpe']:.3f}／{m['max_drawdown']:.2%}")
            lines.append('|'+segments[segment]+'|'+'|'.join(vals)+'|')
        lines.append('')
    lines += ['## 训练期选择及持仓变化','']
    for arm in ARMS[1:]:
        d=report['guard_details'][arm]
        lines.append(f"- {names[arm]}：通过={d['admitted']}；训练期改变{len(d['changed_training'])}个月，全期改变{len(report['changed_indices'][arm])}个月。")
    lines += ['','## 对全期年化差异影响最大的月份','','以下为保存净值的精确会计分解，不是唯一的因果解释；不据此事后更改参数。','']
    for arm in ARMS[1:]:
        comp=arm+'_vs_incumbent'
        rows=sorted(report['all122_periods'],key=lambda r:abs(r['paths']['same_close_zero']['comparisons'][comp]['full_cagr_contribution_pp']),reverse=True)[:8]
        lines += ['### '+names[arm],'','|信号日|基准持仓|候选持仓|基准月收益|候选月收益|年化差异贡献|','|---|---|---|---:|---:|---:|']
        for r in rows:
            q=r['paths']['same_close_zero']
            lines.append('|'+r['signal_date']+'|'+', '.join(display(x) for x in r['holdings'][ARMS[0]])+'|'+', '.join(display(x) for x in r['holdings'][arm])+f"|{q['returns'][ARMS[0]]:.2%}|{q['returns'][arm]:.2%}|{q['comparisons'][comp]['full_cagr_contribution_pp']:+.3f}pp|")
        lines.append('')
    lines += ['12条执行路径、60个指标切片及24个训练切片通过独立复核；122期月收益因子连乘与终值差异一致。冻结观察账户不变。','']
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
    require(set(prepared['artifacts_sha256']) == {'FEATURES.npz', 'TRANSFORMS_DELTAS.npz', 'FEATURE_AUDIT.json', 'ORDERS.json', 'META.json', 'SOURCE_PARITY.json', 'INDEPENDENT_PREPARE.json'} and len(completed['artifacts_sha256']) == 50,
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
    require(independent['passed'] is True and independent['checked_paths'] == 12 and independent['checked_slices'] == 60
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
    guard_details = audit_selection(training, changes, frozen['selection'])
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
            if any(abs(c[k]['return_difference_pp']) > 1e-8 for k in COMPARISONS):
                extras.append(dict(index=row['index'], signal_date=row['signal_date'], path=path,
                    differences_pp={k:c[k]['return_difference_pp'] for k in COMPARISONS}))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), period=['2016-08-17', '2026-09-24'],
        verification=dict(plan_sha256=plan_sha, result_completion_sha256=result_sha,
            input_hash_count=len(pins), prepared_artifacts=7, result_artifacts=50,
            metric_slices_independently_recomputed=60, frozen_training_slices_checked=24,
            all122_period_factors_checked=True, no_strategy_or_feature_imports=True,
            no_new_strategy_runs=True, no_additional_parameter_search_in_attribution=True, inputs_unchanged=True),
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
            'Fixed positive amplitudes .025/.05/.10 reallocate original volatility budget toward original nonvolatility auxiliary contributions. Reverse direction chosen after185; additional sample-reusing parameter search. Original long-core and current volatility horizons remain unchanged',
            'Stocks and ETFs share original ranking domain, originalmissingneutral0 preserved; original27arrays, existingmodel corrections, and episode/support unchanged',
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
    print(json.dumps(dict(receipt=receipt, selection=frozen['selection'], changed_counts={a:len(v) for a,v in changes.items()},
        full_metrics={a: {p: calculated[a][p]['full122'] for p in SHOW_PATHS} for a in ARMS}), indent=2))


class DecompositionTests(unittest.TestCase):
    def selection_fixture(self):
        training = {}
        changes = {a: [1, 2, 53, 90] for a in ARMS[1:]}
        frozen = dict(primary=ARMS[1], admitted=True, admitted_candidates=list(ARMS[1:]), selected_lambda=.025,
                      per_candidate={})
        for arm in ARMS:
            cagr = .50 if arm == ARMS[0] else .51
            training[arm] = {path: {era: dict(cagr=cagr, sharpe=1.8, max_drawdown=-.3)
                                  for era in ('training','early','recent')} for path in SHOW_PATHS}
            if arm != ARMS[0]:
                frozen['per_candidate'][arm] = dict(guards=dict(close_vs_incumbent=True, open_vs_incumbent=True, changed_support=True),
                    changed_months=[1,2,53], admitted=True, training_close_cagr=.51)
        return training, changes, frozen

    def test_selection_exact_tie_lower_dose(self):
        training, changes, frozen = self.selection_fixture()
        self.assertEqual(len(audit_selection(training, changes, frozen)), 3)
        frozen['primary'] = ARMS[2]; frozen['selected_lambda'] = .05
        with self.assertRaisesRegex(ValueError, 'Frozen primary'):
            audit_selection(training, changes, frozen)

    def test_selection_higher_training_cagr(self):
        training, changes, frozen = self.selection_fixture()
        training[ARMS[3]]['same_close_zero']['training']['cagr'] = .52
        frozen['per_candidate'][ARMS[3]]['training_close_cagr'] = .52
        frozen['primary'] = ARMS[3]; frozen['selected_lambda'] = .10
        audit_selection(training, changes, frozen)

    def test_selection_later_change_does_not_pass_support(self):
        training, changes, frozen = self.selection_fixture()
        changes[ARMS[1]] = [1,2,90,91]
        with self.assertRaisesRegex(ValueError, 'Per-candidate'):
            audit_selection(training, changes, frozen)

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
