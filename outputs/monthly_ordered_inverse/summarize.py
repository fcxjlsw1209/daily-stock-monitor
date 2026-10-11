from collections import Counter
import numpy as np
from ordered_study import P,read,dump,sha

def main():
    results=read(P/'RESULTS.json');months=read(P/'MONTHS.json');state=read(P/'STATE_CONTROL.json');f=read(P/'FAMILY_FEASIBILITY.json');winner=read(P/'FROZEN.json')['winner'];industry=read(P/'INDUSTRY_CONTROL.json')
    results['state_control']=dict(metrics=state['metrics'],matching=state['matching']);months['state_control']=state['months']
    groups={};net={}
    for (name,sign,group),weight in zip(winner['spec'],winner['weights']):
        groups[group]=groups.get(group,0)+weight;net[name]=net.get(name,0)+sign*weight
    transition={}
    for key in ['previous_candidate','ordered_frozen48','state_control']:
        cc=Counter();previous=None
        for row in months[key]:
            au=set(row['author']);our=set(row['ours'])
            if previous is not None:
                for s in previous-au:cc['author_removed_still_model_top3' if s in our else 'author_removed_not_model_top3']+=1
                for s in previous&au:cc['author_retained_model_top3' if s in our else 'author_retained_not_model_top3']+=1
            previous=au
        transition[key]=dict(cc)
    summary=dict(status='research_round_complete_not_replicated',fits=256,state_followup_rules=6,
       original_order_assumption=True,primary_months=69,primary_slots=207,formula_group_weights=groups,effective_signed_weights=net,
       current_core='(P[t-42]/P[t-294]-1) / sqrt(std(daily_return[t-293:t-42])); cross-sectional percentile. The denominator is sqrt of daily-return standard deviation, not standard deviation itself.',
       formula_constraint_assumption='>=50% core and<=20% corrections are our operational interpretation of long momentum primary/small correction, not an author-disclosed parameter.',
       matching={k:v['matching'] for k,v in results.items() if v.get('matching')},
       performance={k:v['metrics'] for k,v in results.items()},feasibility=f['summary'],transitions=transition,
       data_limits=['Historical Top32 is reconstructed, not certified vintage-correct. ETF17 is assumed, not author-confirmed.',
                    '5 months have author holdings outside Top32; counted as failures, never inserted by label.',
                    'ASML industry feature had missing values; isolated XLK control reported separately. Other missing industry points remain explicitly listed in INPUTS.json.',
                    'All calendar history has been examined before. First48 selection/later21 is a reuse check, not untouched out-of-sample.',
                    'Same-close replay matches author convention; final close-derived scores cannot be assumed executable at the exact same close. Next-open control separately provided.',
                    'Local feasibility uses author holdings to choose weights separately each month and is NEVER replayed as a strategy.'])
    dump('SUMMARY.json',summary)
    lines=['按作者名单顺序反推月策略：本轮结果','',
      '结论：持仓重合有所改善，但未复现原作者。原模拟账户及观察规则未修改。',
      '范围：2020-12-31收盘建仓至2026-09-24；69期。零交易成本、月末收盘换仓、月内固定股数。',
      '名单顺序为用户批准的第1>2>3名假设，未经作者确认。核心动量跳过最近两个月；不要求短期回撤。',
      '256个受约束拟合只用前48期；规则冻结后检查后21期。另6组小幅状态调整为后续研究，仍只按前48期选定。所有历史已反复研究，不称独立样本外。','',
      '模型 | 股票重合/207 | 三只全对/69 | 次序全对/69 | 年化 | 日频Sharpe | 最大回撤']
    labels={'previous_candidate':'旧候选','ordered_frozen48':'新固定权重','state_control':'小幅市场状态修正','ordered_retrospective69':'固定结构全69期事后拟合','author':'作者持仓重播'}
    for k,title in labels.items():
        z=results[k];v=z['metrics'];a=z.get('matching');a=a['all'] if a else None
        row=[title,str(a['shared_slots']) if a else '207',str(a['set_exact']) if a else '69',str(a['order_exact']) if a else '69',f"{v['cagr']*100:.2f}%",f"{v['sharpe']:.3f}",f"{v['max_drawdown']*100:.2f}%"]
        lines.append(' | '.join(row))
    lines+=['','当前拟合规则（不是恢复出的作者公式）',summary['current_core'],
            '组权重：'+str({k:round(v*100,3) for k,v in groups.items()}),
            '有效带符号系数：'+str({k:round(v*100,3) for k,v in net.items() if abs(v)>1e-7}),
            '波动偏好为正；行业同时看近期与跳过42交易日的相对表现。少量CMF/量价/市场状态修正。加速项在入选结构中被移除。',
            '正负分量可能抵消；原始非负预算为100%，有效净系数绝对值可小于100%。',
            '状态追加试验选择SPY是否高于200日均线，限制各股票标准化分数变化不超过0.10；未按月份或ticker设特殊规则。','',
            '逐月可行性并不是可交易策略',
            f"64个可比月中，允许当月看答案后单独选因子定义/权重，有{f['summary']['feasible_in_any_family']}个月可排出三只及顺序。剩余不可行月份：{f['summary']['infeasible_dates']}。这是所测有限线性家族的结论，不能推断其他公式也不可能。",
            '当前固定规则在64个可比月中，仅44个月可被一个理论上任意取值、权重10%的标准化附加分修正到完全一致。它只是最宽松的必要边界，不是实际ML结果。',
            'ASML行业缺失控制：以XLK为代理，冻结原权重后重合129/207，完全相同13月，顺序1月；重拟合后129/207、13月、2月，仍未消除差距。',
            'ASML业务来源：https://www.asml.com/en/company/about-asml 。XLK代理是研究假设，不是作者公开定义。','',
            '退出：只核对月末名单变化，不从这些月度记录推断月内止损。',str(transition),'','限制：']+summary['data_limits']
    lines+=['','逐月：作者顺序 / 旧规则 / 新固定权重 / 状态修正 / 作者在新规则中的排名 / 局部解释结果']
    ui=[]
    for i,(r,s) in enumerate(zip(months['ordered_frozen48'],months['state_control'])):
        old=months['previous_candidate'][i];local=f['months'][i]
        lines += ['',r['date']+' → '+r['end'],'作者：'+' > '.join(r['author']),
           '旧规则：'+' > '.join(old['ours'])+'；作者排名 '+str(old['author_ranks']),
           '新固定：'+' > '.join(r['ours'])+'；作者排名 '+str(r['author_ranks']),
           '状态修正：'+' > '.join(s['ours'])+'；作者排名 '+str(s['author_ranks']),
           '池外：'+str(r['unavailable'])+'；局部可行定义数：'+str(local['feasible_specs'])]
        for c in r['candidate_contributions']:lines.append('  '+c['symbol']+' #'+str(c['rank'])+' 分数 '+str(round(c['score'],3))+' 分解 '+str({k:round(v,3) for k,v in c['contributions'].items()}))
        ui.append(dict(date=r['date'],end=r['end'],author=r['author'],old=old['ours'],fixed=r['ours'],state=s['ours'],ranks=[old['author_ranks'],r['author_ranks'],s['author_ranks']],contributions=r['candidate_contributions'],unavailable=r['unavailable']))
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    dump('UI_DATA.json',ui)
    protected=read(P/'PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(f)!=h];assert not changed,changed
    valid=read(P/'VALIDATION.json');valid.update(unit_tests_passed=11,protected_changes=changed,protected_files=len(protected),source_notes='Initial invocation needed existing PYTHONPATH. NumPy matmul warning resolved using explicit einsum and tests rerun with RuntimeWarnings treated as errors. Serialization converts NumPy scalar values.')
    dump('VALIDATION.json',valid);dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    dump('STATUS.json',dict(status=summary['status'],completed_components=['256 constrained ordered fits','first48/later21 comparison','same-close and next-open zero-cost replay','4096 local feasibility linear programs','ASML industry control','6 bounded market-state controls','11 tests'],original_accounts_unchanged=True))
    (P/'REPRODUCE.txt').write_text('Existing runtime only; no new packages installed.\nexport PYTHONPATH=work/monthly-ml-deps:work/monthly-decision-deps:work/research-plot-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\nwork/venv/bin/python -W error::RuntimeWarning -m unittest discover -s outputs/monthly_ordered_inverse -p "test_*.py" -v\nMain phases: ordered_study.py init, prepare, train, evaluate. init refuses an existing PLAN.json. Use a new experiment directory for reruns; do not overwrite frozen research. diagnose.py, industry_control.py, state_control.py add explicitly labeled diagnostics. summarize.py builds report.\n')
    print(str(P/'REPORT.txt'))

if __name__=='__main__':main()
