from reconstruct import *
from fit_policy import get_rows

def main():
    rows=get_rows();trials=read(P/'DEFINITION_TRIALS.json')
    primary=[r for r in trials if r['boundary']=='top32' and r['holding_bonus']==0]
    dev=max(primary,key=lambda r:(r['development_exact'],r['development_mean']))
    allbest=max(primary,key=lambda r:(r['exact_months'],r['mean_shared']))
    dump('MAINLINE_SELECTION.json',dict(development_selected=dev,descriptive_best=allbest,
                                       exclusion='Prior-holding bonus excluded after user reminder; no daily rebound mechanisms.'))
    lookup={r['date']:r for r in rows['core17']};cap=read(P/'CAP_RANK_DIAGNOSTICS.json')
    caplookup={(r['date'],r['symbol']):r for r in cap};refs={r['date']:r for r in allbest['details']}
    longrule=read(P/'UNIFORM_WINDOW_LEADERS.json')[0]
    folder=P/'months';folder.mkdir(exist_ok=True)
    alltext=['69期月策略决策对照：用户提供模拟持仓与研究假设','',
             '目标为用户提供的模拟回放记录，非经核验的券商成交。对照选择仅是研究诊断，不是已恢复的作者交易规则。',
             '排序使用调仓日可见历史特征；Top32和17ETF均仍是代理假设。所有月内收益都未用于拟合本轮排序权重。','']
    for date,row in lookup.items():
        target=row['target'];ref=refs[date]
        top=set(ref['diagnostic_selection']);names=target+[s for s in ref['diagnostic_selection'] if s not in target]
        lines=[f"{date}收盘 → {row['end']}：目标决策核对",'',
               '提供的持仓：'+' / '.join(display(s) for s in target),
               '独立排名对照：'+' / '.join(display(s) for s in ref['diagnostic_selection']),
               f"重合{ref['shared']}/3；对照定义={allbest['core']}，只用于全历史事后说明，非独立选定规则。",'',
               '标的 | 身份 | 市值代理排名 | R12-2 | R9-2 | R6-2 | 对齐历史Sharpe | 季度波动（年化）']
        for s in names:
            caprow=caplookup.get((date,display(s)))
            cr=str(caprow['rank']) if caprow else 'ETF或非目标未列'
            if s not in row['symbols']:
                lines.append(f'{display(s)} | 目标但在代理池之外 | {cr} | 未进入当月候选表')
            else:
                x=row['raw'][s]
                fmt=lambda k:'缺失' if x[k] is None else f'{x[k]:+.2%}'
                vol='缺失' if x['vol63'] is None else f"{x['vol63']*np.sqrt(252):.2%}"
                risk='缺失' if x['arithmetic_sharpe12_2'] is None else f"{x['arithmetic_sharpe12_2']*np.sqrt(252):+.2f}"
                lines.append(f"{display(s)} | {'目标' if s in target else '竞争者'} | {cr} | {fmt('mom12_2')} | {fmt('mom9_2')} | {fmt('mom6_2')} | {risk} | {vol}")
        lines+=['','仅凭某个选中标的排序较低，不能证明作者有额外过滤；必须先区分股票池、因子定义、排名修正和模型假设。',
                '不以当月未来涨跌决定这些历史特征，不将每个月单独拟合的权重拼成收益曲线。']
        txt='\n'.join(lines)+'\n';(folder/f'{date}.txt').write_text(txt);alltext+=[txt]
    (P/'MONTHLY_DECISIONS.txt').write_text('\n'.join(alltext))
    failed_caps=[c for c in cap if c['rank'] is None or c['rank']>32]
    validation=read(P/'VALIDATION.json')
    ml=read(P/'MONTHLY_ML_SELECTION.json')['descriptive_best']
    transition=read(P/'TRANSITION_FEASIBILITY.json')
    lines=['月策略复现：以新69期账本为依据的第一轮研究','',
           '当前结论：完成候选池核对、因子定义对照、共享权重与时序检验，尚未恢复整套作者规则。',
           '重要边界：月底独立比较中期强者并等权持有；不引入日策略回撤门槛、反弹预测或月内止盈止损。',
           '老持仓加分的既有对照结果已保存，但根据用户最新提示从主线选择中排除。',
           '',
           '一、数据与候选范围',
           '复用了此前外国发行人分支的BABA、TSM、NVO、SAP、TM行情和市值代理；不按已知赢家手工改成员资格。',
           '69期中63期目标组合全部在当日代理Top32+17ETF内；其他6期的缺失者均在市值第33—35名。']
    for c in failed_caps:lines.append(f"{c['date']} {c['symbol']}：第{c['rank']}名，市值约${c['cap']/1e9:.2f}B，第32名{c['rank32_symbol']}约${c['rank32_cap']/1e9:.2f}B。")
    lines+=['将市值排名边界放宽到第32名市值的90%作为单独诊断，69期均可覆盖；这不能证明作者用了较宽股票池，不能据此改变其Top32规则。',
            '使用前一交易日池和信号时缺口为4期，已分别保留结果。估算市值数据版本、时点及完整ETF清单仍未知。',
            '',
            '二、共同规则检验',
            '检验了：跳过42交易日的6/9/12个月动量、同区间风险调整、算术Sharpe、半风险调整；波动、行业相对、CMF、量比和趋势位置。',
            '用选中与未选中标的的排序约束求解正向动量主导的共享权重；另做最多10%权重预算的市场状态及形态修正。此预算是研究假设，并非原作者已公开参数。',
            '对每个日历年以前的记录拟合后冻结，验证后续月份；不把逐月自由配权重当作规则。',
            '补测了完整12个月向前平移2个月，以及实际日历月末定义。主线所有试验均无老持仓加分。',
            f"主线本轮事后持仓命中最高的简单定义：{allbest['pool']}，{allbest['core']}，{allbest['style']}，信号延迟{allbest['lag']}交易日；完整命中{allbest['exact_months']}/69期，平均重合{allbest['mean_shared']:.2f}/3只。",
            f"只按2021—2024持仓匹配选出的定义：{dev['core']}，{dev['style']}；开发段{dev['development_exact']}/48期完整匹配，固定后2025—2026仅{dev['frozen_exact']}/21期。",
            '因此定义修正能改善部分月份，但不是完整答案；也未按收益率挑选赢家。',
            '',
            '三、可连续解释的明确片段',
            '2022-11-30至2023-05-31的7次调仓（持有期2022年12月至2023年6月），同一组权重准确选中全部目标。',
            '这一段使用core17代理池、按当日全候选的10%至90%分位线性缩放并截断的特征，正向动量权重60%、其他因子绝对权重40%。',
            '该窗口来自扫描已知历史标签，不能当作预先选定的验证期。系数分别扰动±5%的100次检查仍匹配本段全部7期，但不证明未来稳定。',
            '权重四舍五入到小数点后两位时仍命中7期；尝试减少因子后七因子版本精度变差，不能称为已经找到简洁统一规则。',
            '原始拟合权重：']
    for k,v in longrule['fit']['weights'].items():
        if abs(v)>1e-7:lines.append(f'  {k}: {v:+.6f}')
    lines+=['保持这组权重进入接下来3期，0期完整命中。第一处关键分歧是2023-06-30：目标META/MRK/NVO，对照ORCL/NVDA/NVO。',
            '另有2023年7月至2024年1月的7期共同解释，但扰动敏感、后续亦失败；不得拼接两套局部权重。',
            '',
            '四、按用户补充的精确月策略结构追加检验',
            f"市场状态修正：SPY/QQQ长趋势、科技相对强弱和市场波动状态，共{len(transition)}组受约束共享排名模型；对2022年12月至2024年1月的14期，没有一组能全部严格匹配。最优共同最小分差{max(t['margin'] for t in transition):+.6f}，尚为负。",
            '这只排除了本轮具体特征与预算假设，不证明作者的市场状态修正无效。',
            f"小幅机器学习评分：以当时已结束月份的下月相对SPY收益为标签，过去5年训练、每年冻结Ridge，评分只占10%。本轮36个固定组合中最好完整匹配{ml['exact_months']}/69期，平均{ml['mean_shared']:.2f}/3只。",
            '5年与年度训练安排是声明的月策略研究假设，不能因为作者日策略有此安排，就断言月策略也相同。模型未使用短期回撤资格、反弹质量或日内/日间退出。',
            f"另核验了{validation.get('monthly_return_model_training_cutoffs',0)}个收益预测模型的训练截止日期；未成熟或未来标签不进入拟合，相关单元测试通过。",
            '',
            '五、验证与材料',
            f"6项单元测试通过；{validation['prefix_checks']}次历史前缀特征核对通过；{validation['year_frozen_training_cutoffs']}个年度冻结训练边界检查通过；本轮引用的行情及证据文件校验和未变。",
            '全部69期目标、竞争者及特征已写入MONTHLY_DECISIONS.txt和months/；各类试验、局部冲突、系数、排名缺口均单独保存JSON。',
            '主线选择见MAINLINE_SELECTION.json；本轮未计算或挑选新的全期年化赢家，未改原六账户、月策略观察账户、账本或计划。',
            '',
            '六、下一步研究焦点',
            '优先解释2023年6月30日及其后3个月的科技切换：在同样可见数据下，是什么较小且可解释的排名修正能让META超过ORCL/NVDA，又在后续选入NVDA？',
            '核对该片段因子窗口和市场状态的具体定义，先形成能跨这个转折的单一规则，再检验此前2021/2022及之后2024/2025月份。',
            '不要再以旧图片反推的组合训练。缺2016—2020账本、完整原始候选与ETF17、精确模型定义，仍限制全期作者策略的确定复现。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    dump('STATUS.json',dict(status='research_partial_not_reconstructed',source_periods=69,primary_proxy_target_covered_periods=63,
                           mainline_best_complete_matches=allbest['exact_months'],development_selected_frozen_complete_matches=dev['frozen_exact'],
                           longest_shared_explanation_periods=7,next_three_periods_complete_matches=0,tests_passed=6,
                           source_inputs_unchanged=True,live_accounts_modified=False,monthly_ml_best_complete_matches=ml['exact_months'],market_transition_complete_fit=False,
                           next_focus='2023-06-30 META/MRK/NVO vs frozen ORCL/NVDA/NVO, then subsequent3periods; no daily rebound mechanisms'))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print('Saved report and69monthly decision dossiers. Mainline rules remain unverified.')
if __name__=='__main__':main()
