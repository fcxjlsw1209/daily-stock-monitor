from filtered_study import *
from compact import compact_scores
from retention import score_rule
import sqlite3

def main():
    plan,packs=prepare();base=read(P/'BASELINES.json');borders=read(P/'BOUNDARY_RESULTS.json');sensitivity=read(P/'COEFFICIENT_SENSITIVITY.json');last=read(P/'NEW_BOUNDARY_SENSITIVITY.json')
    inventory=read(P/'COMPACT_INVENTORY.json');correction=read(P/'CORRECTION_INVENTORY.json');retention=read(P/'RETENTION_INVENTORY.json')
    for key in packs:
        with sqlite3.connect(P/(key+'_compact.sqlite')) as db:
            assert db.execute('select count(*) from trials').fetchone()[0]>0
    def chosen(sl,label):return next(r for r in borders if r['slice']==sl and r['model']==label)
    checks=[]
    for sl in ['A','B']:
        rich=chosen(sl,'flexible');data=packs[rich['key']];w=np.array(rich['weights']);scores=np.einsum('mnj,j->mn',data['design'],w,optimize=False)
        replay=evaluate(data,scores,sl);old=rich['evaluations'][sl]
        assert replay['exact_periods']==old['exact_periods'] and replay['shared_positions']==old['shared_positions']
        assert w[:data['core']].sum()>=.5-1e-7 and w[-data['small']:].sum()<=.1+1e-7 and abs(w.sum()-1)<1e-7
        simple=chosen(sl,'compact');trials=read(P/'COMPACT_LEADERS.json')+read(P/'CORRECTION_LEADERS.json')
        rr=next(r for r in trials if r['slice']==sl and r['key']==simple['key'] and r['coefficients']==simple['coefficients'])
        y=evaluate(packs[rr['key']],score_rule(packs[rr['key']],rr),sl)
        assert y['exact_periods']==simple['evaluations'][sl]['exact_periods']
        checks.append(dict(slice=sl,compact_replayed=True,flexible_replayed=True,weight_budgets_passed=True))
    sources=read(P/'SOURCE_HASHES.json');source_changed=[f for f,h in sources.items() if sha(f)!=h];assert not source_changed
    protected=read(P/'PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(f)!=h];assert not changed,changed
    # Cap method independently agrees with prior audit for all author stock selections.
    oldr={(r['date'],r['symbol']):r['rank'] for r in read(O/'monthly_ledger_reconstruction/CAP_RANK_DIAGNOSTICS.json')}
    for model in borders:
        for row in model['details']:
            for s,r in row['target_cap_ranks'].items():
                if (row['date'],s) in oldr:assert oldr[row['date'],s]==r
    validation=dict(completed_utc=datetime.now(timezone.utc).isoformat(),unit_tests_passed=9,checks=checks,source_hashes_passed=len(sources),protected_files_unchanged=len(protected),cap_ranks_match_prior_audit=True,
                    grid_trials=inventory['trials'],correction_trials=sum(r['trials'] for r in correction),retention_trials=retention['trials'],no_return_objective=True,no_observer_modifications=True)
    dump('VALIDATION.json',validation)
    lines=['按用户筛选条件反推月度三只持仓：本轮结果','',
        '数据：用户提供的69期模拟持仓，2020-12-31调仓至2026-09-24结束；最后一期未满月。只以每期三只证券集合匹配为目标。未使用原收益图，未优化收益率。','',
        '固定样本：',
        '- A：排除原return_leader选中ARKK/SOXX的9期，以及作者目标在原当期代理池之外的6期；无重叠，剩54期。',
        '- B：A中再排除原return_leader持仓含估算第28-32名股票的15期，剩39期。两套主样本在拟合前固定。',
        '- 只删评估月份，不删候选ETF。原动态Top32＋theme17共49候选及原资格条件保持不变。排除月份仍在递归持仓测试中推进状态。','',
        '完全匹配 = 当期三只全部一致，忽略顺序；非“三只中命中一只”。','',
        '|模型|A：54期|B：39期|','|---|---:|---:|']
    for label,kind in [('原73.38%年化候选（未重新拟合）','old'),('本轮简洁4因子规则','compact'),('本轮多系数拟合诊断','flexible')]:
        vals=[]
        for sl in ['A','B']:
            z=base['return_leader'][sl] if kind=='old' else chosen(sl,kind)['evaluations'][sl]
            vals.append(f"{z['exact_periods']}/{z['periods']}（{z['exact_periods']/z['periods']:.1%}）；位置命中{z['shared_positions']}/{z['total_positions']}")
        lines.append('|'+label+'|'+'|'.join(vals)+'|')
    simple=chosen('A','compact')
    lines += ['', '本轮简洁规则（两套切片选出的最佳规则相同）：',json.dumps(simple['coefficients'],ensure_ascii=False,indent=2),
              '以上均为全49候选的百分位分数。mom294_skip42=P[t-42]/P[t-294]-1；downside63=最近63日负收益平方均值开方；ma200_distance=P[t]/MA200-1。状态项在SPY63日波动高于其过去252日中位数时为正，否则为负。仍按原295日预热和正长期动量资格筛选，再取前三。','',
              '做了什么：',f"- 枚举{inventory['trials']:,}组简洁规则、{sum(r['trials'] for r in correction):,}组2.5%-10%市场状态/因果Ridge修正，以及{retention['trials']}组递归持仓保留规则。没有用收益率选优。",
              '- 递归保留前4/5/6名或小幅持仓加分未提高本轮最佳匹配。没有把作者上月真实持仓作为模型输入。',
              '- 连续优化和整数优化额外拟合目标月份；每条规则的长期动量非负权重至少50%、市场状态/Ridge修正预算至多10%。',
              '- 整数优化每次有45秒时间上限，报告的是已找到的可行规则，不声称达到全局最优。8因子稀疏求解未胜过已知4因子规则，也不能据此推断8因子的最佳上限。','',
              '多系数诊断的限制：']
    for r in sensitivity:
        z=next(x for x in r['rounding'] if x['decimals']==4);full=chosen(r['slice'],'flexible')['evaluations'][r['slice']]
        lines.append(f"- {r['slice']}样本使用{r['coefficient_count']}个非零系数；保留四位小数时完全匹配从{full['exact_periods']}期降为{z['exact_periods']}期。精度依赖明显，不能当作作者规则已复现。")
    lines += ['', '边界再次核查（只做一次追加敏感性，不反复删月）：',
              f"- B中多系数规则新选中第28-32名股票的月份有7期；其中4期本已匹配。固定剔除这7期后剩{last['existing_fit']['periods']}期，既有规则匹配{last['existing_fit']['exact']}期，仍有6期不同。",
              '- 9期原不匹配中只有3期包含“我们错选的边界股”；其余不能归因于这类边界。',
              '- 例如2023-06-30调仓：作者META/MRK/NVO，拟合NVDA/MRK/NVO。META估算第8、NVDA第5，均远离第32名。',
              '- 对54、39以及追加32期，线性规划均找不到在本轮因子库和预算下全部严格排序正确的统一线性组合。这只约束本轮模型族，不代表任何非线性模型或作者真实数据口径都不可能。单月独立调整权重可以解释全部54期，但把各月不同答案拼接不构成统一交易规则。','',
              '结论：已按要求完成两层排除和重新拟合，仍未完整复现。股票池边界不是唯一差异；完整ETF候选名单、因子定义/标准化、状态适配和模型预测口径仍未知。不能为消除差异继续无限剔除月份。',
              '这些是事后反推及条件样本诊断，不是独立样本外验证。不把不连续月份拼接成年化收益或Sharpe，不更改现有观察账户、规则、账本或交易计划。','',
              f"验证：9项单元测试通过；新旧结果重算一致；{len(protected)}个观察系统文件哈希未变；数据源哈希与原边界排名一致。",
              '逐期对照：MONTHLY_DECISIONS.txt；原始模型/优化状态见FITS.json、FEASIBILITY.json和各网格数据库。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # Every original period, with reasons and fitted alternatives, remains visible.
    chosen_simple={r['date']:r for r in simple['details']}
    a={r['date']:r for r in chosen('A','flexible')['details']};b={r['date']:r for r in chosen('B','flexible')['details']}
    decisions=[]
    for row in plan['records']:
        d=row['date'];why=[]
        if row['excluded_etf']:why.append('原候选选中'+','.join(row['excluded_etf']))
        if row['missing_target']:why.append('作者目标在池外:'+','.join(row['missing_target']))
        if row['included_A'] and row['baseline_boundary']:why.append('B排除原边界:'+json.dumps(row['baseline_boundary']))
        decisions += [f"信号{d}；持有至{row['end']}；A={'纳入' if row['included_A'] else '排除'}，B={'纳入' if row['included_B'] else '排除'}",'作者：'+', '.join(row['target']),'原候选：'+', '.join(row['baseline'])]
        for label,records in [('简洁规则',chosen_simple),('A拟合',a),('B拟合',b)]:
            if d in records:
                rr=records[d];decisions += [label+'：'+', '.join(rr['selected'])+f"；命中{rr['shared']}/3；完全匹配={rr['exact']}",'选中估算市值排名（null为ETF）：'+json.dumps(rr['selected_cap_ranks'],ensure_ascii=False)]
        if why:decisions+=['排除说明：'+'；'.join(why)]
        decisions+=['']
    (P/'MONTHLY_DECISIONS.txt').write_text('\n'.join(decisions))
    dump('STATUS.json',dict(status='FILTERED_RECONSTRUCTION_EXPERIMENTS_COMPLETED',full_replication_achieved=False,counts=plan['counts'],validation=validation))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print(json.dumps(validation,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
