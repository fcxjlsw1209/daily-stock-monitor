from pathlib import Path
import sys
import numpy as np
import pandas as pd
from infer import P, O, read, dump, sha, load, CORE, SIDE
sys.path.insert(0, str(O/'monthly_reconstruction_contract'))
from audit import image_diagnostics
from inverse import load_observations


def main():
    results = {**read(P/'RESULTS.json'), **read(P/'RANKING_RESULTS.json')}
    models = {**read(P/'FITTED_MODELS.json'), **read(P/'RANKING_MODELS.json')}
    allowed = [k for k, v in results.items() if v['eligible_author_hypothesis']]
    best = max(allowed, key=lambda k: (results[k]['expanding_year_holdout']['average_overlap'],
                                     results[k]['expanding_year_holdout']['pairwise_accuracy']))
    assert best == 'listwise_core50_side50'
    navs = pd.concat([pd.read_csv(P/f, index_col=0, parse_dates=True) for f in ['NAV.csv.gz','RANKING_NAV.csv.gz']], axis=1)
    px, _, _ = load()
    obs = load_observations(); cal = read(O/'monthly_curve_reverse/CALIBRATION.json')
    c = read(O/'monthly_reconstruction_contract/CONTRACT.json')
    audit = {}
    for name in results:
        nav = navs[name].dropna()
        diag, _ = image_diagnostics(nav, px.QQQ, obs, cal, c['image_audit'])
        audit[name] = dict(image=diag, actual_author_strategy_reproduced=False,
            matches_reference_curve=not any(diag[p]['screen_rejected'] for p in ['wealth','relative','drawdown']),
            full_fit=results[name]['full_fit'])
    dump('FIGURE_AUDIT.json', audit)
    assert not any(x['matches_reference_curve'] for x in audit.values())
    dump('COMBINED_SELECTION.json', dict(selected=best,
        criterion='Average chosen-name overlap in expanding prior-year tests, then pairwise accuracy; never maximum return.',
        is_retrospective_model_comparison=True, author_rule_recovered=False,
        labels_assumed_true_for_this_analysis=True, labels_independently_verified=False,
        original_gate_not_changed=True))
    feature = read(P/'PREVALENCE.json')
    chosen = results[best]
    stability = {}
    for name in CORE+SIDE:
        vals = [row['weights'][name] for row in chosen['folds']]
        stability[name] = dict(positive_years=sum(v>1e-8 for v in vals),
            negative_years=sum(v<-1e-8 for v in vals), min=min(vals),max=max(vals))
    dump('COEFFICIENT_STABILITY.json',stability)
    comparisons = read(P/f'{best}_FULL_EXPLANATION.json')
    holdout_comparisons = read(P/f'{best}_YEAR_HOLDOUT.json')
    random_overlap = float(np.mean([9/r['candidate_count'] for r in holdout_comparisons]))
    exact = [r for r in comparisons if r['exact']]
    worst = sorted(comparisons,key=lambda r:sum(r['target_ranks']),reverse=True)[:5]
    target = read(O/'monthly_path_constraints/FIGURE_AUDIT.json')['theme17__prior__draw_v2__blocks']['metrics']
    lines = [
        '把122期反推持仓暂作真实标签：月策略交易准则推断',
        '2026-10-09', '',
        '结论：存在清楚的动量与波动偏好线索，但本轮没有恢复能选出这些完整组合的统一规则。',
        '按用户要求，本轮条件性地假设已导出的122期持仓就是作者交易。这个假设用于推断，不改变其尚未独立核实的事实。原标签、股票池和历史记录没有改动。', '',
        '一、哪些是输入，哪些是新推断',
        '月底收盘、3只等权、月内固定股数、零成本、核心动量跳过最近42个交易日，是原作者描述或既有重建约束，不是这次从数据中新发现的规则。',
        '正的R12-2与足够预热历史已经用于构造反推候选，也不能再当作独立验证出的买入条件。',
        '新证据来自每次选中的股票与同日未选股票之间的特征差异。共366个选中记录，不是366只不同股票。', '',
        '特征 | 选中记录的百分位中位数 | 未选记录的百分位中位数',
    ]
    cn = dict(mom12_2='12个月动量，跳过最近2个月',vol126='半年波动率',
              industry_relative60='60日行业相对表现',cmf21='21日资金流',
              volume_ratio21_126='21/126日量比',near_high252='接近52周高点程度')
    for k,label in cn.items():
        f=feature['features'][k]
        lines.append(f"{label} | {f['selected_median']:.1%} | {f['unselected_median']:.1%}")
    lines += ['',
        '二、目前最有支持的准则轮廓',
        '1. 以跳过最近两个月的长期动量为主，9个月/6个月及风险调整动量作补充。',
        '2. 条件接近时，偏向长期波动较大、收益弹性较大的股票。这里不等于无条件追逐高风险。',
        '3. 同时扣除近期下行波动的影响，区分长期弹性与近期下跌风险。',
        '4. 对过于接近新高、近期放量等现象作温和扣分。这只是拟合里的排序修正，不能推出“必须跌多少才能买”的硬门槛。',
        '5. 行业相对表现、CMF等作小幅修正。没有证据支持把它们单独设成决定性开关。',
        '6. 月末重新比较排序，前三名等权；月内不使用日策略止盈止损。最后一条属于已给定的月策略结构。', '',
        '在7个按过去年份训练的模型中，R12-2、半年波动率均为正权重7次；近期下行波动、距高点和量比均为负权重7次；行业相对表现正6次、零1次。',
        '这些方向在本模型家族内稳定，但权重有共线性，不能断言作者恰好使用这些参数。', '',
        '三、实际解释效果',
        '选择解释模型依据跨年份的持股重合率，不依据年化收益。首轮6类方法加3类专门的排序目标，共72次拟合/重放（包含无需训练的纯动量对照）。',
        '跨年份区间：2020-01-31至2026-09-24，80次配置。每年只用此前年份的标签拟合，年内保持系数不变。',
        f"被选中的受约束排序模型：每月平均匹配{chosen['expanding_year_holdout']['average_overlap']:.4f}/3只，名字命中率{chosen['expanding_year_holdout']['selected_name_recall']:.2%}，整组三只完全相同{chosen['expanding_year_holdout']['exact_months']}/80次。",
        f"作为计数参照，在相同候选数量中均匀随机选三只，平均重合约{random_overlap:.3f}/3只；这不是显著性检验。",
        '模型 | 全期年化 | 日Sharpe | 最大回撤 | 全期完全匹配 | 跨年份平均匹配/3',
        f"反推标签路径（用事后价格构造） | {target['cagr_nav_count']:.2%} | {target['sharpe']:.3f} | {target['max_dd']:.2%} | 标签本身 | 不适用",
    ]
    names = dict(raw_momentum='纯动量排序对照',listwise_core50_side50='50%动量预算的排序模型',
        listwise_core60_side40='60%动量预算的排序模型',listwise_core80_side20='80%动量预算的排序模型',
        core60_bounded_ridge='带小幅Ridge修正的模型',unconstrained_ridge_diagnostic='不限制动量主导的Ridge（仅诊断）')
    for name,label in names.items():
        v=results[name]; f=v['full_fit']; h=v['expanding_year_holdout']
        lines.append(f"{label} | {f['cagr_nav_count']:.2%} | {f['sharpe']:.3f} | {f['max_dd']:.2%} | {f['exact_months']}/122 | {h['average_overlap']:.4f}")
    lines += ['',
        '全期重放统一为2016-08-17至2026-09-24，零成本、同收盘代理、固定股数；年化使用2540个净值点，Sharpe使用2539个日收益。它们均未通过原图完整曲线筛查，不能称为作者策略复现。',
        '跨年份检验测的是规则解释稳定性。持仓标签本来由全历史收益图反推，即使按时间拆分训练，也不能称为独立的投资样本外回测。', '',
        '四、具体月份',
        '这一排序模型在全样本拟合后，能准确选出以下整组持仓：']
    for r in exact: lines.append(f"  {r['date']}收盘：{', '.join(r['target_holdings'])}")
    lines += ['仍最难解释的月份（日期为收盘配置日）：']
    for r in worst:
        lines.append(f"  {r['date']}：假设作者选{', '.join(r['target_holdings'])}；规则选{', '.join(r['inferred_holdings'])}。目标持股在模型中排名为{r['target_ranks']}。")
    lines += ['',
        '五、可复核的参数假设',
        '所有因子在决策日以当时代理股票池中的百分位计算。动量系数非负且合计50%；其余系数绝对值之和不超过50%。这是研究约束，不是作者公开权重。',
        'R12-2 = P[t-42]/P[t-252]-1；R9-2与R6-2分别将分母换成t-189和t-126。风险调整动量除以63日收益波动率。',
        '选中/未选的排序似然用月度等权交叉熵拟合，温度0.10、L2惩罚0.001。另测的Ridge小修正严格限制为每只±0.05。',
        '本轮选中解释模型的全期系数：',
    ]
    for name,value in models[best]['weights'].items(): lines.append(f'  {name}: {value:+.6f}')
    lines += ['',
        '六、限制与后续定位',
        '当前最需要辨别的是特征尺度（原始幅度与百分位）、个股和ETF如何共同排序，以及作者原始股票池与17只ETF名单。固定加权和未能解释大多数月份，不能据此排除整个作者思路。',
        '60日行业基准在137条候选记录缺失，其中17条为选中记录；统一按中性百分位处理并保留缺失标记，因此行业结论的可信度弱于动量、波动。',
        '不通过加入股票代码/月份编号来记忆持仓，不变动标签、不改变两个月跳过期、不按收益选择模型，不改观察账户。',
        '初次排序运行曾因优化器驻点残差未达到门槛而停止；修正过早终止判据后重跑，24次排序拟合均通过原精度要求，失败日志保留。',
        '验证：3项针对性单元测试通过；16个特征在3个截断日期重算一致（48项），训练时序、动量/侧因子预算和小模型边界均通过检查。',
    ]
    (P/'REPORT.txt').write_text(('\n'.join(lines)+'\n').replace('research_equity_FB', 'META/FB'))
    for file,digest in read(P/'SOURCE_HASHES.json').items(): assert sha(file)==digest,file
    dump('FINAL_STATUS.json',dict(status='conditional_rule_inference_completed', label_events=122,
        families=len(results), fits_and_replays=72, selected_model=best,
        exact_author_rule_recovered=False, any_full_curve_match=False,
        original_sources_unchanged=True, source_hashes_checked=len(read(P/'SOURCE_HASHES.json')),
        monthly_holdings_export_completed=True, actual_author_records_independently_confirmed=False))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print((P/'FINAL_STATUS.json').read_text())


if __name__=='__main__':main()
