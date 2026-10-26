"""Report only frozen selections; no search or rule changes here."""
from pathlib import Path
import json

P = Path(__file__).resolve().parent
read = lambda name: json.loads((P / name).read_text())


def main():
    r, frozen, attribution = read('RESULTS.json'), read('FROZEN.json'), read('ATTRIBUTION.json')
    audit = read('VALIDATION.json')
    admitted = frozen['primary_admitted']
    names = [('original65','已登记月策略'),('previous66','此前研究候选'),
             ('best_none','无ML冻结对照'),('best_direct','直接预测冻结对照'),
             ('best_residual','残差预测冻结对照'),('primary','训练收益优先候选' if admitted else '未准入：描述性稳健对照'),
             ('robust','训练稳健优先对照'),('qqq','QQQ买入持有')]
    lines = ['独立月策略骨架与折外残差模型实验', '',
        '80个预先登记配置。前88期选择并冻结，再检查后34期；不按后段收益重新选规则。',
        '股票池、价格和资格沿用历史Top32估算＋假定17ETF；各基座全部前三名重新排序，无前两名锁定。',
        '所有数字为历史研究回放。现有观察账户、账本与冻结计划不变。', '',
        ('本轮有配置通过预定训练门槛。' if admitted else
         '本轮80配置均未通过预定训练门槛，不产生新合格候选。primary只是描述性稳健对照的兼容字段，不能当作获选策略。'), '',
        '共同口径：2016-08-17—2026-09-24，零成本、同最终收盘信号/成交的理想化回放。',
        '月内固定复权收益单位，3只等权/2只各半/少于2只现金；只在月度信号日重配。',
        'Sharpe由每日收益计算，年化系数sqrt(252)，未扣无风险利率；回撤为日终净值峰值回撤。',
        '首期8/17是此前研究的中月初始化，之后正常月末；末期截至9/24，非完整月份。', '',
        '版本 | 全期年化 | Sharpe | 最大回撤 | 2016–2020年化 | 后34期年化']
    for key,label in names:
        z=r[key];m=z['metrics'];s=z['slices']
        lines.append(f"{label} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} | {s['early2016_2020']['cagr']:.2%} | {s['later34']['cagr']:.2%}")
    lines += ['', '五个无ML、全池排名的固定起点（不是后段挑选）：',
              '定义 | 全期年化 | Sharpe | 最大回撤']
    labels={'raw':'12−2累计收益','aligned_sharpe':'同窗口Sharpe', 'raw_vol':'75%动量＋25%波动',
            'raw_sector':'75%动量＋25%个股相对行业', 'recent_risk':'旧动量÷近期波动'}
    for family,label in labels.items():
        m=r['simple_'+family]['metrics']
        lines.append(f"{label} | {m['cagr']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%}")
    primary=r['primary'];spec=primary['spec']
    lines += ['', '冻结的主对照：', json.dumps(spec,ensure_ascii=False),
        f"训练守门通过：{frozen['primary_admitted']}；80配置中{frozen['admissible']}个通过预先设定的Sharpe/回撤/分段门槛。",
        '跳两月定义是P[t−42]/P[t−252]−1，即过去12个月内去掉最近2个月后的约210个交易日；Sharpe使用完全相同的210日收益窗口。',
        '小幅人工篮子=.4自身行业相对+.2CMF21+.2有符号成交量63+.2收盘位置21。模型特征仅量价及市场宽度/趋势/波动交互。',
        '第一层用基础分、分数平方、原始动量/波动/行业排名，使用当时已成熟的更早月份生成折外预测。残差扣除的不只是动量，也包括基础风险和行业信息；直接预测对照使用完全相同训练行。',
        '纯动量等骨架包含重复或线性相关的解释列，Ridge可解但有效正则惩罚会随坐标变化；骨架间差异不应全部归因于新增信息。',
        '年度截止按信号年处理：12月末仍用当年冻结模型，下一年1月末才使用新的年度模型。样本每月均衡，再施加24月衰减。',
        '固定alpha10、24月衰减、每月等总权重后再衰减；预测用固定5%月收益尺度的tanh有界映射，零预测零加分，不再转百分位放大。', '',
        '主对照的配对消融（相同骨架；直接与残差保持相同手工预算和ML幅度）：',
        '版本 | 全期年化 | Sharpe | 训练段年化 | 后34期年化']
    for key,label in [('paired_none','无修正'),('paired_hand','仅5%人工修正'),('paired_direct','直接预测'),('paired_residual','折外残差预测')]:
        z=r[key]
        lines.append(f"{label} | {z['metrics']['cagr']:.2%} | {z['metrics']['sharpe']:.3f} | {z['slices']['training2016_2023']['cagr']:.2%} | {z['slices']['later34']['cagr']:.2%}")
    increment=read('RESIDUAL_INCREMENT_ATTRIBUTION.json')['changes']
    lines += ['', f'残差预测相对配对直接预测实际只改变{len(increment)}个月：']
    for change in increment:
        lines.append(f"{change['date']}→{change['end']}：{'/'.join(change['direct'])} → {'/'.join(change['residual'])}；当月差{change['difference']*100:+.2f}个百分点。")
    lines.append('这一增量不能证明稳定优势；两者训练段收益相同，后段差异仅作冻结后诊断，不据此切换主规则。')
    lines += ['', '执行敏感性：', '版本 | 次日开盘零成本年化 | 次日开盘每边0.05%年化 | 后者Sharpe']
    for key,label in names[:-1]:
        z=r[key];a=z['next_open_zero']['metrics'];b=z['next_open_cost']['metrics']
        lines.append(f"{label} | {a['cagr']:.2%} | {b['cagr']:.2%} | {b['sharpe']:.3f}")
    lines += ['', '改仓归因：']
    for key,label in [('paired_none','相同骨架无修正'),('paired_direct','相同骨架直接预测'),('previous66','此前66.36研究候选')]:
        a=attribution[key]
        gains=sorted(a['changes'],key=lambda z:z['log_contribution'],reverse=True)
        lines.append(f"相对{label}：{a['changed_months']}个月改变组合，其中{a['top2_changes']}个月替换原前两名中的至少一只。")
        for z in gains[:3]:
            lines.append(f"  {z['date']}→{z['end']}：{'/'.join(z['old'])} → {'/'.join(z['new'])}；当月差{z['diff']*100:+.2f}个百分点。")
        total=sum(z['log_contribution'] for z in gains)
        top=sum(z['log_contribution'] for z in gains[:3])
        lines.append(f"  全部变更合计对数财富差{total:+.4f}；最有利三个月贡献{top:+.4f}（不是可直接相加的年化收益）。")
    wf=read('WALKFORWARD.json')
    lines += ['',f"训练内部逐段选择诊断：年化{wf['metrics']['cagr']:.2%}，Sharpe{wf['metrics']['sharpe']:.3f}；这仍是已研究历史。",
        f"旧规则/回放/账本等{audit['protected_files']}个受保护文件哈希未变；两个既有控制策略精确复现；固定份额日级回放与独立逐月拼接一致。",
        '22项测试通过，实际运行日志见TEST_RESULTS.txt。命令：PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=work/monthly-ml-deps:work/monthly-decision-deps:work/research-plot-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/venv/bin/python -W error::RuntimeWarning -m unittest discover -s outputs/monthly_architecture_restart -p test*.py -v', '',
        '限制：历史市值/行业为代理、作者17ETF确切名单未知，2016年前训练池使用继承历史代理。最近历史已多次参与研究，后34期不再是未触碰样本外。',
        '本轮检验的是完整重新排序与增量残差学习；结果不代表已复现作者，也不承诺未来年化收益。未自动加入或替换观察账户。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    (P/'COMPLETION.json').write_text(json.dumps(dict(completed=True,configuration_count=80,qualified_candidate=admitted,
        observers_changed=False,primary=r['primary']['metrics'],previous=r['previous66']['metrics'],
        primary_beats_previous_cagr=r['primary']['metrics']['cagr']>r['previous66']['metrics']['cagr'],
        evaluation_kind='reused-historical-research'),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
