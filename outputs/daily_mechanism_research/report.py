# -*- coding: utf-8 -*-
from pathlib import Path
import json
import hashlib

P = Path(__file__).resolve().parent
summary = json.loads((P / 'SUMMARY.json').read_text())
diagnostics = json.loads((P / 'DIAGNOSTICS.json').read_text())
checks = json.loads((P / 'CHECKS.json').read_text())
labels = {
    'baseline': '原日策略',
    'five_day_clock': '持仓上限缩短至五日',
    'no_rank_replacement': '取消候选排名优势换股',
    'recovery_exit': '盈利且收复五日均线则退出',
    'event_ranking': '取消0.75门槛、只在回撤候选中排序',
    'conditional_rebound': '按深浅回撤学习历史反弹及路径',
    'vol_scaled_ridge': '按波动标准化Ridge预测目标',
    'medium_momentum': '动量改用三至六个月',
}
assert len(summary) == 16 and all(x['status'] == 'complete' for x in summary.values())
assert checks['baseline_reproduced'] and checks['labels_mature']
lines = [
    '日策略结构改动：第二轮有限对照实验',
    '完成：2026-10-08；范围：2019-01-02至2026-10-06。',
    '这是根据文字描述重建的研究实现，不是外部作者原代码的复现。历史数据认证未完成，不是纯样本外业绩。',
    '七个结构改动在运行前声明，每项单独对照原版，分别计买卖合计0.10%和0.30%成本，没有组合后反复择优。',
    '最近252交易日：2025-10-06至2026-10-06；排除最近252日后的区间截至2025-10-03。',
    '四大组仍等权25%；股票池、最大三只、少于两只空仓与原日版本一致。',
    '',
]
for cost in [10, 30]:
    lines.extend([
        f'买卖合计成本{cost / 100:.2f}%',
        '方案 | 全期年化 | 最大日终回撤 | 排除最近252日年化 | 最近252日收益 | 平均仓位 | 胜率',
    ])
    for kind, label in labels.items():
        v = summary[f'{kind}_cost{cost}']
        d = diagnostics[f'{kind}_cost{cost}']
        values = [v['full']['cagr'], v['full']['max_drawdown'], v['ex_recent']['cagr'],
                  v['last252']['return_'], v['mean_exposure'], d['win_rate']]
        lines.append(label + ' | ' + ' | '.join(f'{x * 100:.2f}%' for x in values))
    lines.append('')
lines += [
    '诊断',
    '原版平均每天约1.05只新入场候选，平均持仓61.25%。这确实限制资金使用，但放宽条件的实验没有解决收益问题。',
    '取消最终评分门槛后，平均候选增至9.36只、平均仓位96.53%，全期年化却降至9.89%，最近252日仅6.77%。更多交易不等于更好的反弹选择。',
    '收复五日线提前退出把胜率从55.27%提高到66.13%，但平均盈利交易从+5.17%缩至+2.37%，平均亏损为-3.21%；全期年化15.67%，三倍成本下仅7.83%。胜率上升不足以抵消盈利幅度与换手变化。',
    '原版因更优候选退出的43笔交易平均收益-0.72%，但直接删除这条退出规则，整体年化反而降至15.51%。单看某类已平仓交易的负收益，不能认定该退出规则造成亏损；必须看完整反事实组合。',
    '五日预测与十日持有的差异值得检验，但机械缩短到五日没有改善收益。条件历史样本细分、波动归一预测和较短动量窗口在本轮都未胜出。',
    '',
    '下一步研究优先级（尚未测试，不作为收益承诺）',
    '校验目标与交易路径：同时预测扣费后反弹收益、发生较大不利波动的概率，而非只拟合五日终点收益。',
    '区分市场整体回撤与个股独立下跌，测试特征交互；观察收益与风险预测的分组校准，而非只看账户总收益。',
    '检查股票和ETF是否需要分开训练／排序，并保持事前固定测试集合，避免根据最近赢家更换ETF篮子。',
    '新增研究必须使用成熟标签、时间顺序训练和成本压力测试。2019—2023与2024以后的历史已多次被查看，均不再称为未触碰的样本外。',
    '',
    '验证与账户',
    '16次执行全部完成，原版在两档成本下的订单和净值均复现；四套模型每次训练的未来标签均已成熟。',
    '两项新测试通过：未来价格变化不影响过去条件因子；取消候选换股仍保留持有期限退出。',
    '没有把任何本轮日策略候选自动加入观察账户。用户批准的月策略buffer5独立登记于outputs/monthly_buffer5_monitor，和此轮日策略研究分开。',
    '完整参数见CONFIG.json，结果见SUMMARY.json，退出归因见DIAGNOSTICS.json，训练检查见TRAINING_AUDIT.json和CHECKS.json。',
]
(P / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
(P / 'FINAL_VALIDATION.json').write_text(json.dumps({
    'completed_runs': 16, 'new_tests_passed': 2, 'baseline_reproduced_both_costs': True,
    'all_labels_mature': checks['labels_mature'], 'data_certified': False,
    'report_sha256': hashlib.sha256((P / 'REPORT.txt').read_bytes()).hexdigest(),
    'code_sha256': hashlib.sha256((P / 'run.py').read_bytes()).hexdigest(),
}, indent=2) + '\n')
print('Report and validation saved')
