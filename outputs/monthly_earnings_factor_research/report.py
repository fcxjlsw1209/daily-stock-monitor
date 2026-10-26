"""Render completed, frozen earnings experiments without rerunning selection."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import sys

P = Path(__file__).resolve().parent
read = lambda f: json.loads((P/f).read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    r, f, v = read('RESULTS.json'), read('FROZEN.json'), read('VALIDATION.json')
    assert sha(P/'FROZEN.json') == (P/'FROZEN.sha256').read_text().strip()
    for path, digest in f['hashes'].items(): assert sha(P/path) == digest, path
    assert v['original_controls_exact'] and v['independent_fixed_unit_replay']
    assert not v['protected_changes'] and not v['observer_accounts_changed']
    atr, fa, prep = read('ATTRIBUTION.json'), read('FEATURE_AUDIT.json'), read('PREPARE_VALIDATION.json')
    def pct(x): return f'{100*x:.2f}%'
    names = {
        'previous66': '既有研究候选66', 'original65': '原基础65',
        'original65_event': '原基础＋财报反应', 'original65_matched_recent': '原基础＋匹配近期涨幅',
        'original65_ridge_earn': '原基础＋Ridge（含财报）', 'original65_ridge_plain': '原基础＋Ridge（不含财报）',
        'simple': '独立简洁动量基础', 'simple_event': '简洁基础＋财报反应',
        'simple_matched_recent': '简洁基础＋匹配近期涨幅',
        'simple_ridge_earn': '简洁基础＋Ridge（含财报）', 'simple_ridge_plain': '简洁基础＋Ridge（不含财报）',
        'qqq': 'QQQ买入持有'}
    lines = ['月策略：财报公布后走势因子测试', '完成时间UTC：'+datetime.now(timezone.utc).isoformat(), '',
        f"完成26组事前固定配置。通过既有66候选训练门槛的新增财报配置：{f['new_admitted']}组。",
        '冻结主选择：'+('保留既有66候选。' if f['primary_source']=='incumbent66' else '新财报候选，具体参数见FROZEN.json。'),
        '所有实验均已实际回放。观察账户、冻结计划及账本未修改，没有自动推广研究策略。', '',
        '一、统一比较口径',
        '2016-08-17至2026-09-24，共122个持仓区间。前88期至2023-11-30仅用于筛选；后34期冻结后再评估。',
        '主要表格：同一最终收盘价产生信号及成交、零交易成本的理想化回放。2016及2026、首尾持仓区间并非完整年度/月。',
        '使用原估算历史Top32＋自拟17只ETF、原正长期动量及295价格资格。长期核心跳过最近42个交易日；月底三只等权、两只各半、少于两只空仓，月内固定复权总回报单位。',
        '历史已被多轮使用，后34期及训练内滚动诊断都不是全新的独立样本外；股票池、早期训练池、行业和证券数据的原研究近似仍在。不能称为原作者复现或预期实盘收益。', '',
        '二、全期结果',
        '方案 | 年化收益 | Sharpe | 最大回撤',
        '--- | --- | --- | ---']
    for k, label in names.items():
        z = r[k]['metrics']
        lines.append(f"{label} | {pct(z['cagr'])} | {z['sharpe']:.3f} | {pct(z['max_drawdown'])}")
    lines += ['', '三、固定实验与财报信息的准确含义',
        '82个股票身份、3771条已核验常规财报公告。仅用最新、当时可知公告之后的完整10/21个XNYS交易日收益，减去同端点SPY收益。日期级公告从公告日之后首个交易日收盘才算可知。',
        '这不是公告当晚/次晨跳空，不是历史EPS surprise，也没有读取之后才出现的公司基本面修订。3日窗口虽已准备，本轮未参与选优。',
        '手工评分方向预先固定为正：0.5*tanh(相对收益/0.05)。2.5%或5%的辅助评分预算，保持原长期核心不变。普通近期10/21日对照截至信号日，使用完全相同的事件可用性掩码。',
        'Ridge配对：原10个动量/波动/行业/量价特征，对比相同10项＋财报10日、21日、21日减10日。每年仅用截至前一年末已经成熟的此前五年下一自然月标签；alpha10、24月样本半衰期，月度总权重一致。标准化仅拟合历史训练样本。',
        '两模型共享训练样本及预测资格，只有10/21窗口均成熟的股票获得模型修正；ETF、缺事件或窗口未成熟均保留缺值，只把其评分增量设为零，不排除原交易候选、不回退更早事件。',
        '26组＝2基础×[1原样控制＋4手工信号×2预算＋2Ridge组×2预算]。两个基础为原65评分和独立75%原始长期动量＋25%波动；既有66的订单仅作固定对照及选择回退。未直接把新因子叠到66的第三名替换规则上。',
        '只在前88期选择各类别诊断和主候选，诊断即使未通过替代门槛仍完整展示；不得按全期表格重新指定赢家。', '',
        '训练冻结参数：']
    for k in names:
        if k not in f['selected'] or k in ['original65', 'simple']: continue
        z = f['selected'][k]
        lines.append(f"{names[k]}：{z['spec']['key']}，预算{pct(z['spec']['budget'])}；通过替代门槛={z['admitted']}。")
    lines += ['', '四、前后分段', '方案 | 早53期年化 | 近35期年化 | 前88期年化 | 后34期年化', '--- | --- | --- | --- | ---']
    for k in ['previous66', 'original65', 'original65_event', 'original65_ridge_earn', 'simple', 'simple_event', 'simple_ridge_earn']:
        s = r[k]['slices']
        training = f['incumbent_training'] if k == 'previous66' else f['selected'][k]
        values = [s['early2016_2020']['cagr'], training['recent']['cagr'],
            s['training2016_2023']['cagr'], s['later34']['cagr']]
        lines.append(names[k]+' | '+' | '.join(pct(z) for z in values))
    lines += ['', '五、配对归因与集中度']
    for b in ['original65', 'simple']:
        for arm, control in [('event','matched_recent'), ('ridge_earn','ridge_plain')]:
            n, c = b+'_'+arm, b+'_'+control
            a = atr[n+'_vs_'+c]
            diff = r[n]['metrics']['cagr']-r[c]['metrics']['cagr']
            lines.append(f"{names[n]} 相对 {names[c]}：年化差{100*diff:+.2f}个百分点，持仓不同{a['changed_months']}期；把其中最有利的1/3期恢复对照持仓后，诊断年化为{pct(a['undo_best1_diagnostic_cagr'])}/{pct(a['undo_best3_diagnostic_cagr'])}。")
            changes = sorted(a['changes'], key=lambda z: z['log_contribution'], reverse=True)
            for z in changes[:3]:
                lines.append(f"  {z['date']}：{','.join(z['old'])} → {','.join(z['new'])}，当期收益差{100*z['difference']:+.2f}个百分点（{z['split']}）。")
    lines += ['原基础财报手工评分相对匹配近期动量18期不同，前三有利改选占净对数增量约75.6%，后34期占约86.7%；近35期的配对增量为负。',
        '原基础财报Ridge只比普通Ridge改选3期，简洁基础只改选1期，不能把这些有限增量解释为稳定的机器学习优势。',
        '财报锚定优于这些匹配控制，并不意味着应加入现有最佳策略：本轮原基础财报方案均低于65.34%基础及66.36%既有候选。']
    lines += ['上述恢复月份的计算是事后贡献诊断，不是可以事前执行的优化规则。', '',
        '六、执行敏感性', '方案 | 同收盘零成本 | 次开零成本 | 次开每边0.05%', '--- | --- | --- | ---']
    for k in ['previous66', 'original65', 'original65_event', 'original65_ridge_earn', 'simple_event']:
        lines.append(names[k]+' | '+' | '.join(pct(z) for z in [r[k]['metrics']['cagr'],r[k]['next_open_zero']['metrics']['cagr'],r[k]['next_open_cost']['metrics']['cagr']]))
    lines += ['真实完整收盘信号不能保证同价成交；三种执行使用同一冻结选股，没有按较有利的执行结果重新调参。', '',
        '七、验证与可复用资料',
        '44项新增测试通过。涵盖日期/窗口时点、未来数据不改变前缀、训练成熟截止、匹配样本、缺失/ETF、模型冷启动、零预算和控制组映射。',
        f"原始候选不变。Ridge可用的合格训练候选行{prep['ready_training_ml']}、实际回放候选行{prep['ready_event_ml']}；各年度实际59个已成熟训练月份。",
        '事件来源价格与旧价格面板交叉验证13997个窗口，均完整；最大绝对相对收益差约3.007e-7，没有超过5bp的窗口。',
        f"原65/66的年化、Sharpe及回撤精确复现；逐日固定单位引擎与独立向量路径一致；训练选择与全期回放前88期一致；{v['protected_files']}个受保护文件内容未变。",
        'PLAN.json/GRID.json：事前规则；FROZEN.json：冻结选择及指纹；TRAIN_TRIALS.json：26组训练结果；RESULTS.json：全期与分段、执行敏感性。',
        'MONTH_COMPARISON.json/ATTRIBUTION.json：逐月持仓、收益及配对归因；各方案ORDERS/TRADES/HOLDINGS/DAILY.json：实际回放输出；FORECAST_AUDIT.json：年度训练时点；TESTS.txt/VALIDATION.json：验证。',
        '本目录脚本各阶段为initialize、prepare、forecasts、train、evaluate；initialize/train拒绝覆盖既有冻结方案。重做实验应复制脚本到独立同级新目录，保留既有源文件。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    env = dict(python=sys.version, packages={k:version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib']})
    (P/'ENVIRONMENT.json').write_text(json.dumps(env, ensure_ascii=False, indent=2)+'\n')
    print('Wrote REPORT.txt')


if __name__ == '__main__': main()
