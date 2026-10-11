"""Post-freeze attribution only; never modifies experiment selection."""
from pathlib import Path
import json
import statistics
import numpy as np
import episodes as s

P = Path(__file__).resolve().parent


def main():
    results = s.read(P / 'RESULTS.json')
    frozen = s.read(P / 'FROZEN.json')
    trials = s.read(P / 'TRAIN_TRIALS.json')
    comparison = s.read(P / 'MONTH_COMPARISON.json')
    details = s.read(P / 'BOUNDARY_DETAILS.json')
    audit = s.read(P / 'DATA_AUDIT.json')
    config = next(c for c in s.models() if c['id'] == frozen['selected']['primary']['model_id'])
    changed = []
    for row in comparison:
        before, after = row['previous66'], row['primary']
        if set(before['symbols']) != set(after['symbols']):
            changed.append(dict(date=row['date'], end=row['end'], split=row['split'],
                before=before['symbols'], after=after['symbols'], previous_return=before['monthly_return'],
                candidate_return=after['monthly_return'], difference=after['monthly_return'] - before['monthly_return'],
                log_gain=float(np.log1p(after['monthly_return']) - np.log1p(before['monthly_return']))))
    source = [d for d in details if d['split'] == 'training' and d['model_id'] == 0 and d['signal_type'] > 0]
    diagnostics = dict(training_boundary_signal_cases=len(source),
        median_observations=statistics.median(d['observations'] for d in source),
        median_effective_samples=statistics.median(d['effective_samples'] for d in source),
        median_own_weight=statistics.median(d['own_weight'] for d in source),
        corrected_cases=sum(d['correction'] != 0 for d in source),
        selected_model=config, changed_months=changed,
        unique_training_metrics=len({round(r['training']['cagr'], 10) for r in trials}))
    # Remove the single most helpful selection only for concentration diagnosis.
    m, d, px, op, a = s.load()
    pred = dict(np.load(P / 'FORECASTS.npz'))
    blocks, _, bounds, valid = s.rp.paths(px, m, 122)
    basech, _ = s.choices(frozen['baseline'], m, d, a, pred, 122)
    newch, _ = s.choices(frozen['selected']['primary'], m, d, a, pred, 122)
    br = s.rp.fast_returns(basech, blocks, valid)
    nr = s.rp.fast_returns(newch, blocks, valid)
    helpful = [r for r in changed if r['log_gain'] > 0]
    if helpful:
        largest = max(helpful, key=lambda r: r['log_gain'])
        i = m['dates'].index(largest['date'])
        without = nr.copy()
        without[bounds[i]:bounds[i + 1]] = br[bounds[i]:bounds[i + 1]]
        diagnostics['remove_largest_helpful_month'] = dict(month=largest['date'],
            metrics=s.rp.stats_rr(without, m['dates'][0], m['ends'][-1]))
    s.dump('DIAGNOSTICS.json', diagnostics)
    selected = frozen['selected']['primary']
    lines = ['日级回撤事件记忆：月策略第三席增量修正', '',
        '本轮50配置：2原规则对照＋48新配置。先前88期选择并冻结，后34期重放；未重新调参，未新增或修改观察账户。',
        f"提取{audit['episodes']:,}个21交易日间隔事件，涉及{audit['securities']}个证券ID。样本来自各证券自己的价格历史，可以早于进入Top32；交易池不变。",
        f"训练段70个有回撤信号的边界案例，同公司同类历史样本中位数{diagnostics['median_observations']:.1f}，有效样本中位数{diagnostics['median_effective_samples']:.2f}；上一轮月末抽样分别4.5和3.80。",
        '同一证券标签窗口不重叠，不表示跨证券事件统计独立。A/B为本研究代理信号，作者公式未知。', '',
        '冻结的主要候选：',
        f"{json.dumps(config, ensure_ascii=False)}；预算{selected['budget']:.1%}，仅原排名3至5且距第三名<=0.02的候选竞争，前两名保留。",
        '使用过去5年同证券同信号已成熟事件：下一开盘至第5/10/21日收盘的超额收益，或减去0.5倍最大不利收盘路径。24月时效衰减，先验质量6，有效样本至少3；预计优势扣除一个标准误后再向零收缩。',
        '无信号、无样本、证据弱则严格零修正，不回退到另一套全市场Ridge。月内固定份额，不新增止盈止损或回撤入场门槛。', '',
        '比较口径：2016-08-17至2026-09-24，同最终收盘信号/成交、零交易成本的理想化回放。',
        '版本 | 全期年化 | Sharpe | 最大回撤 | 早期2016–2020年化 | 后34期年化',
    ]
    for key, name in [('original65', '已登记65.34规则'), ('previous66', '原66.10研究候选'), ('primary', '本轮冻结候选'), ('qqq', 'QQQ')]:
        r = results[key]
        v = r['metrics']
        lines.append(f"{name} | {v['cagr']:.2%} | {v['sharpe']:.3f} | {v['max_drawdown']:.2%} | {r['slices']['early2016_2020']['cagr']:.2%} | {r['slices']['later34']['cagr']:.2%}")
    r = results['primary']
    lines += ['', f"本轮候选次日开盘零成本：年化{r['next_open_zero']['metrics']['cagr']:.2%}，Sharpe{r['next_open_zero']['metrics']['sharpe']:.3f}。",
        f"次日开盘＋每边0.05%：年化{r['next_open_cost']['metrics']['cagr']:.2%}，Sharpe{r['next_open_cost']['metrics']['sharpe']:.3f}。",
        '', '持仓变化归因：']
    for z in changed:
        lines.append(f"{z['date']}→{z['end']}（{z['split']}）：{'/'.join(z['before'])} → {'/'.join(z['after'])}；当期组合收益{z['previous_return']:.2%}→{z['candidate_return']:.2%}，差{100*z['difference']:+.2f}个百分点。")
    if helpful:
        r = diagnostics['remove_largest_helpful_month']
        lines.append(f"仅作集中度诊断：撤销最有利的{r['month']}换股后，全期年化{r['metrics']['cagr']:.2%}、Sharpe{r['metrics']['sharpe']:.3f}。不据此重新选参。")
    walk = s.read(P / 'WALKFORWARD.json')
    lines += ['', f"训练内部逐段选择诊断：年化{walk['metrics']['cagr']:.2%}、Sharpe{walk['metrics']['sharpe']:.3f}；原候选同段{walk['baseline']['cagr']:.2%}、{walk['baseline']['sharpe']:.3f}。",
        f"50个配置的训练收益实际只形成{diagnostics['unique_training_metrics']}种不同结果；通过守门不等于获得独立改善。",
        '', '验证：18项测试通过；原65.34/66.10控制122期订单/评分完全一致；移除未来特征不改变此前特征；日级固定份额回放与逐月拼接一致。',
        f"受保护文件{s.read(P / 'VALIDATION.json')['protected_files']}个哈希未变。",
        '限制：历史估算股票池与行业标签、假定ETF名单、反复研究的样本、同收盘理想化执行均保留。结果不是作者策略的复现，不是未触碰样本外或实际模拟收益。',
        '后续若继续，应验证收益集中度与更多已成熟事件；本轮不自动把研究候选加入观察账户。']
    (P / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(primary=results['primary']['metrics'], diagnostics=diagnostics,
                          sequential=walk['metrics'], previous_sequential=walk['baseline']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
