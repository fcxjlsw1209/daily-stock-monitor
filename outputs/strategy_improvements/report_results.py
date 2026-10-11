"""Render completed isolated experiments; never rerun selection or touch ledgers."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
MONTHLY = {
    'original': '原月策略（每月完整卖出再买入）',
    'net': '只按目标仓位差额调仓',
    'buffer5': '差额调仓＋持仓在前五名则保留',
    'industry20': '行业强度及上涨广度20%',
    'residual20': '市场残差动量20%',
    'path20': '趋势路径效率20%',
    'correlation10': '相关性惩罚0.10',
    'vol20': '组合波动目标20%',
    'staged5': '一半月末入场、一半等回撤或五日到期',
}
DAILY = {
    'baseline': '原日策略',
    'confirmation': '回撤后等待止跌确认',
    'sector_context': '增加行业及广度门槛',
    'cost_margin': '预测收益增加成本余量门槛',
    'replacement25': '换股所需评分优势从0.15提高到0.25',
}


def main():
    summary = json.loads((P / 'SUMMARY.json').read_text())
    robustness = json.loads((P / 'ROBUSTNESS.json').read_text())
    checks = json.loads((P / 'CHECKS.json').read_text())
    frozen = json.loads((P / 'SOURCE_HASHES.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest() == v for k, v in frozen.items())
    assert all(v['status'] == 'complete' for v in list(summary.values()) + list(robustness.values()))
    validation = {'results': {}, 'frozen_sources_still_unchanged': True}
    paths = sorted(P.glob('*_cost*.json'))
    for path in paths:
        data = json.loads(path.read_text())
        assert data['status'] == 'complete'
        assert all(d['signal_date'] < d['session'] for d in data['decisions'])
        assert all(d['cash'] >= -1e-6 for d in data['daily'])
        assert not data['performance_publishable']
        if path.name.startswith(('monthly_', 'buffer')):
            sides = {}
            for order in data['orders']:
                sides.setdefault((order['date'], order['symbol']), set()).add(order['side'])
            assert all(len(v) == 1 for v in sides.values())
        validation['results'][path.name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sessions': len(data['daily']), 'status': 'complete',
            'signal_before_execution': True, 'nonnegative_cash': True,
            'performance_publishable': False,
        }
    assert len(validation['results']) == 30
    validation['code_hashes'] = {
        name: hashlib.sha256((P / name).read_bytes()).hexdigest()
        for name in ['research.py', 'check_buffer_stability.py', 'test_research.py', 'report_results.py']
    }
    (P / 'FINAL_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
    pct = lambda x: f'{x * 100:.2f}%'
    lines = [
        '月／日策略改进：独立回测比较',
        '完成日期：2026-10-08；行情截止：2026-10-06。',
        '结论：月策略差额调仓＋前五名保留、行业强度／广度，以及日策略成本余量门槛值得进一步观察。',
        '这些是有数据限制、反复研究过的历史结果，performance_publishable=false，不是纯样本外或真实交易业绩。',
        '',
        '范围与可比口径',
        '月策略：2010-01-04至2026-10-06，首次月末建仓2010-01-29；日策略：2019-01-02至2026-10-06。',
        '剔除最近252交易日的区间截至2025-10-03；最近252交易日为2025-10-06至2026-10-06。',
        '各方案初始10万美元、只做多、不加杠杆、最多三只；剩余现金不计息，未计个人税。',
        '股票池沿用历史Nasdaq-100与S&P100并集内、过去可用市值快照排名Top32，加17只无杠杆ETF；不可理解为已认证完整的每日真实Top32。',
        '月策略使用月末前一交易日信号、月末收盘执行；日策略用前收盘信号、次日开盘执行。',
        '本轮八个月策略改动、四个日策略改动逐项测试，未把所有改动叠加；两档往返成本均单独计算。',
        '',
    ]
    for cost in [10, 30]:
        lines += [f'月策略：买卖合计成本{cost / 100:.2f}%',
                  '方案 | 全期年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益 | 2010—2018年化 | 平均持仓比例']
        for kind, label in MONTHLY.items():
            v = summary[f'monthly_{kind}_cost{cost}']
            values = [v['full']['cagr'], v['full']['max_drawdown'], v['ex_recent']['cagr'],
                      v['last252']['return_'], v['early']['cagr'], v['mean_exposure']]
            lines.append(label + ' | ' + ' | '.join(map(pct, values)))
        lines += ['', f'日策略：买卖合计成本{cost / 100:.2f}%',
                  '方案 | 全期年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益 | 平均持仓比例']
        for kind, label in DAILY.items():
            v = summary[f'daily_{kind}_cost{cost}']
            values = [v['full']['cagr'], v['full']['max_drawdown'], v['ex_recent']['cagr'],
                      v['last252']['return_'], v['mean_exposure']]
            lines.append(label + ' | ' + ' | '.join(map(pct, values)))
        lines.append('')
    lines += ['前五名保留的邻近参数检查',
              '这是看到首轮结果后增加的稳健性检查，不是新的参数挑选轮次。原先预设值仍为五名。',
              '保留阈值 | 成本 | 全期年化 | 最大日终回撤 | 剔除最近252日年化 | 最近252日收益']
    for limit in [4, 5, 6]:
        for cost in [10, 30]:
            v = summary[f'monthly_buffer5_cost{cost}'] if limit == 5 else robustness[f'buffer{limit}_cost{cost}']
            values = [v['full']['cagr'], v['full']['max_drawdown'], v['ex_recent']['cagr'], v['last252']['return_']]
            lines.append(f'前{limit}名 | {cost / 100:.2f}% | ' + ' | '.join(map(pct, values)))
    qqq = json.loads((P.parent / 'dual_strategy_research/monthly_2010/SUMMARY.json').read_text())['qqq_buy_hold']['periods']
    lines += ['', '同期QQQ持有基准（2010起账户）',
              f"全期年化{pct(qqq['full']['cagr'])}、最大日终回撤{pct(qqq['full']['max_drawdown'])}；剔除最近252日年化{pct(qqq['excluding_last252']['cagr'])}；最近252日收益{pct(qqq['last252']['return_'])}。",
              '首日开盘买入、买入成本0.05%、未卖出不扣卖出费，复权价格近似股息再投资。',
              '', '实际改动及解释',
              '差额调仓：选股／目标权重与原版一致，只买卖现有仓位和目标仓位之差。改善属于减少执行成本，不能算作新的预测能力。',
              '前五名保留：月末仍符合资格、动量为正且排名前五的持仓优先保留；新买仅从前三名中补位，最多三只、每行业最多两只。目标等权，仅交易差额。',
              '行业版本：原月评分80%＋行业涨幅排名10%＋行业上涨广度排名10%；广度计算于当时历史指数并集，未只在Top32计算；无法映射的基金行业用并集整体广度。',
              '残差版本仅扣除市场beta影响，不是完整行业中性化；路径效率为较早210交易日对数收益和／绝对对数收益和，均跳过最近42日。',
              '相关性版本在逐一选股时扣除与已选标的的最大正相关×0.10。波动版本月末以过去63日估算等权组合年化波动，以20%为目标，最高100%仓位，月内不自动降仓。',
              '分批版本月末买入50%，其余等至少两只目标出现前一日回撤信号，或五交易日到期，按下一开盘成交；此实验不能代表所有可能的日月结合方案。',
              '日策略成本余量：原回撤、趋势和评分条件不变，增加Ridge预测五日净收益>0.30%的条件；预测目标已扣0.10%基础成本。模型未为每个新门槛重新训练。',
              '日策略成本余量平均持仓由61.25%降至42.21%，回撤改善部分来自现金更多。近期收益反而由61.46%降至38.73%，不能只看风险改善就称全面胜出。',
              '', '研究判断',
              '月策略前五保留改善了全期及剔除最近252日的年化，成本提高三倍后仍有优势，值得冻结观察。',
              '但其最近252日150.16%的收益不稳定：邻近前四／前六阈值分别为78.92%／65.71%；前四阈值最大回撤升至40.40%。近期翻倍不能作为可靠预期。',
              '行业版本早期2010—2018年化16.64%，高于原版13.16%；全期年化24.38%，但最近252日低于原版。它是有独立价值的另一个观察方向。',
              '残差动量、路径效率、相关性惩罚、等待回撤分批入场在此设定下未改善净收益；波动目标减少回撤但收益明显降低。日策略止跌确认和行业硬门槛表现更差。',
              '日策略优先保留成本余量作为低仓位候选；当前证据不支持把它自动叠加到月策略，也没有证明日月收益可以直接相加。',
              '', '核验及局限',
              '30次本轮及邻近参数执行全部完成，另复用两档成本的原月策略结果；所有信号日期早于成交，现金非负。',
              f"日基准净值／订单复现：{checks['checks']['daily_baseline_reproduced']}；差额版本月选股目标一致：{checks['checks']['net_same_monthly_targets']}。",
              f"模型{checks['training_batches']}次训练的标签均已成熟；被冻结的原研究输入和脚本哈希保持一致。",
              '8项新增测试通过：差额交易、部分卖出、股息拆股、零费率对照、开盘定仓不读取当日收盘、权重约束、分批期限、未来价格扰动不影响过去因子。既有4项双策略和12项原因子测试也通过。',
              '净额调仓可能部分减仓，平仓碎片的胜率不与原版月末整笔平仓胜率直接比较；主要按完整账户净值评估。',
              '历史成员／市值覆盖仍不完整；市值快照混合月／周／日频、滞后一日、最长沿用45日，原始披露时点未认证；行业和公司行动沿用原研究假设。',
              '严格历史准入仍为false，不能将能够完成回测等同于数据认证通过。2008—2009仍未包含，不能声称检验了金融危机。',
              '开发期和最近252日均已被反复观察，本轮属于探索性历史比较。没有自动修改六个观察账户、账本、冻结计划或执行指纹。',
              '', '文件',
              'research.py / CONFIG.json：实现与首轮参数；check_buffer_stability.py / ROBUSTNESS_CONFIG.json：后续邻近检查。',
              'SUMMARY.json / ROBUSTNESS.json：汇总；各*_cost*.json：逐日净值、决策和交易。',
              'CHECKS.json / TRAINING_AUDIT.json / FINAL_VALIDATION.json：复现、标签成熟与结果校验。',
    ]
    (P / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(f'Report generated; {len(validation["results"])} completed runs validated.')


if __name__ == '__main__':
    main()
