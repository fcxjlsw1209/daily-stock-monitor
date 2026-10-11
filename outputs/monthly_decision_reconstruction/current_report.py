"""Focused early-month decision report, not a performance report."""
import numpy as np
from decisions import P,read,dump,sha,FEATURES,CORE,CN,display
from frontier import load_first_months,choices as interaction_choices


def main():
    key='raw_magnitude__largest_group__small_interaction'
    result=read(P/'FRONTIER_RESULTS.json')[key]
    fit=result['history'][1]['fit']
    rows=load_first_months(3)
    snapshots=read(P/'SNAPSHOTS.json')
    report=['月策略：从第一月开始解释选择','',
        '范围：本轮不运行全期收益回测，也不按年化、Sharpe或下一月收益选择解释。按用户假设，将既有反推持仓视为目标标签。',
        '边界：2016-08-17是初始仓位；正式从2016-08-31这一月末开始。核心问题是当时为什么选这三只、排除其他候选。',
        '当前进度：有同一组系数能解释头两个月，第三个月尚未解决。这里只是条件解释，不是已识别作者规则。',
        '该两月解释把动量理解为最大的因子组，实际正动量权重22.5%，小模型绝对权重10%。若把“动量为主”规定为至少50%，尚不能同时解释头两个月。因此不能把宽松解释无条件认作符合作者原规则。','']
    details=[]
    for k,row in enumerate(rows):
        date=row['date'];targets=[s for s,y in zip(row['symbols'],row['y']) if y]
        records={s['symbol']:s for s in snapshots if s['date']==date}
        picks,scores=interaction_choices(row,fit,'raw_magnitude',True)
        score_rank=np.argsort(-scores,kind='stable')
        rank={row['symbols'][j]:i+1 for i,j in enumerate(score_rank)}
        report.extend([f'{date}收盘决策：'+ '、'.join(map(display,targets)),
            '状态：'+('同一组两月联合拟合系数可精确选中。' if k<2 else '头两月规则不能选中目标，需要继续核对。')])
        report.append('使用上述两月共同规则的排序：'+ '、'.join(display(row['symbols'][i]) for i in score_rank[:6]))
        report.append('目标及对照 | R12-2 | R6-2 | 年化半年波动 | 60日行业相对 | CMF21 | 本规则排名')
        rivals=[row['symbols'][j] for j in score_rank if not row['y'][j]][:3]
        for s in targets+rivals:
            raw=records[s]['raw']
            ind=raw['industry_relative60']
            ind_text='缺失（中性）' if ind is None else f'{ind:+.2%}'
            report.append(f"{display(s)}{'[目标]' if s in targets else '[对照]'} | {raw['mom12_2']:+.2%} | {raw['mom6_2']:+.2%} | {raw['vol126']*np.sqrt(252):.2%} | {ind_text} | {raw['cmf21']:+.3f} | {rank[s]}")
        if k==0:
            report.extend(['逐项理解：PFE和UNH的半年动量较强，MRK的行业相对表现与资金流较强；三只不共享“CMF21为正”这一条件，不能把它写成硬性入场规则。',
                '需要解释的排除项：AMZN的年度、半年动量都高于三只目标。只按原始动量排序不能选出医疗股组合。'])
        elif k==1:
            report.extend(['逐项理解：AMZN、MSFT、META的年度动量都强，但MSFT的半年动量仅约2.79%；此前偏半年动量的规则不能直接解释这次科技股组合。',
                '时序检验：仅用第一月拟合的这类规则，第二月选择SOXX、SMH、JNJ，命中0/3。两个月共同解释是加入第二月标签后才得到的，不能称作事先预测成功。'])
        else:
            report.extend(['逐项理解：XLE和XOM分别只有约3.02%、9.02%的R12-2，低于当时SMH、SOXX。必须解释为什么能源标的能超过这些中期趋势更强的候选。',
                '时序检验：仅用前两月的共同系数，本月选择SMH、SOXX、INTC，目标命中0/3。尚未识别能自然完成此次切换的规则。'])
        report.append('')
        details.append(dict(date=date,target=targets,comparison_picks=picks,
            coefficients_inferred_through='2016-09-30',used_current_or_later_label=k<2,
            forward_decision_test=k==2,
            scores=[dict(symbol=s,selected=bool(y),score=float(sc),rank=rank[s]) for s,y,sc in zip(row['symbols'],row['y'],scores)]))
    report.extend(['本轮进一步核对及结论',
        '1. “12个月跳过2个月”与“完整12个月、整体滞后2个月”分别检查；后者未解决跨月冲突。',
        '2. 波动调整改用半年/一年窗口，分别取当前值与滞后2个月值，行业比较也核对两种时点；这些定义仍未解释第三月。',
        '3. 固定因子阈值加小幅修正可以在数学上同时拟合前三月。但最强修正惩罚高核心动量，未按中心思想收录为成功复现。',
        '4. 将阈值修正限制在波动、行业、量价上，使核心动量保持单调正向后，第三月仍冲突。',
        '5. 以上只排除了被检验的固定评分家族。作者可能定期重新训练模型；固定系数、10%修正上限、具体ETF名单和因子窗口均为研究假设，不能据此断言原作者交易不可能。','',
        '后续只推进2016-10-31这一个决策：检查具备时间依据的行业定义、资格过滤和用当时已成熟历史标签训练的小模型。每个候选仍要解释选中与排除，并回查前两个月。',
        '不会用每月独立改权重、日期/股票代码标签或未来收益拼成“已复现策略”。未改变目标交易、观察账户、既有策略规则或原始数据。','',
        '文件：months/下保留逐月因子卡；这些卡是单月检查，不代表该月已被一套共同规则复现。'])
    (P/'CURRENT_FRONTIER.txt').write_text('\n'.join(report)+'\n')
    dump('MONTH_DECISION_DETAILS.json',details)
    protected=read(P/'SOURCE_HASHES.json')
    for f,h in protected.items():assert sha(f)==h,f
    dump('VALIDATION.json',dict(full_period_return_backtest_run=False,source_files_unchanged=len(protected),
        focused_months=[r['date'] for r in rows],common_rule_explanation_months=2,
        definition_of_primary_momentum='largest factor group; not author-confirmed; core only22.5%',
        third_month_prior_rule_hits=0,all_factor_threshold_fit_rejected_as_guidance_conflict=True,
        next_unresolved_decision='2016-10-31',labels_changed=False,
        initial_prefix_feature_checks=48,source_labels_assumed_true=True))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})


if __name__=='__main__':main()
