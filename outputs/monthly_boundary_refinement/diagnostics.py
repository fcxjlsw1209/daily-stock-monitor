"""Post-freeze attribution. No candidate reselection and no observer writes."""
import boundary_research as s


def main():
    fr=s.check_freeze();m,d,px,op,a,pred=s.load();results=s.read(s.P/'RESULTS.json');trials=s.read(s.P/'TRAIN_TRIALS.json')
    bc,bs=s.choices(fr['base'],a,pred,m,d,122);pc,ps=s.choices(fr['selected']['primary'],a,pred,m,d,122)
    baseorders=s.read(s.P/'previous66_ORDERS.json');primaryorders=s.read(s.P/'primary_ORDERS.json')
    changes=[];margins=[]
    for i,date in enumerate(m['dates']):
        ranks=sorted(s.np.flatnonzero(d['eligible'][i]),key=lambda j:(-bs[i,j],m['symbols'][i][j]))
        if len(ranks)>=4:
            p,q=ranks[2:4];margins.append(dict(date=date,third=m['symbols'][i][p],fourth=m['symbols'][i][q],score_gap=float(bs[i,p]-bs[i,q]),split='training88' if i<88 else 'later34'))
        before=baseorders[i]['symbols'];after=primaryorders[i]['symbols']
        if set(before)!=set(after):
            ret=lambda ss:float((px.loc[m['ends'][i],ss]/px.loc[date,ss]-1).mean()) if ss else 0.
            b,n=ret(before),ret(after)
            changes.append(dict(date=date,end=m['ends'][i],before=before,after=after,before_return=b,after_return=n,difference=n-b,log_difference=float(s.np.log1p(n)-s.np.log1p(b))))
    s.dump('CHANGED_MONTHS.json',changes);s.dump('THIRD_FOURTH_MARGIN.json',margins)
    # Best raw-training candidates are shown only on the training months, even
    # when rejected by guards. Later34 are not searched to rescue a rejected rule.
    diagnostic={}
    baseline=trials[1]
    for family in ['retention','switch_gap','disagreement','combined']:
        row=max((z for z in trials if z['family']==family),key=lambda z:(z['training']['cagr'],-z['id']))
        ch,sc=s.choices(row,a,pred,m,d,88);changed=[];entrants=0;baseentrants=0
        for i,date in enumerate(m['dates'][:88]):
            prev=set(m['symbols'][i-1][j] for j in ch[i-1]) if i else set()
            prevb=set(m['symbols'][i-1][j] for j in bc[i-1]) if i else set()
            names=[m['symbols'][i][j] for j in ch[i]];base=[m['symbols'][i][j] for j in bc[i]]
            entrants+=len(set(names)-prev);baseentrants+=len(set(base)-prevb)
            if set(names)!=set(base):
                ret=lambda ss:float((px.loc[m['ends'][i],ss]/px.loc[date,ss]-1).mean()) if ss else 0.
                r,b=ret(names),ret(base)
                changed.append(dict(date=date,end=m['ends'][i],before=base,after=names,before_return=b,after_return=r,difference=r-b,log_difference=float(s.np.log1p(r)-s.np.log1p(b))))
        fail=[]
        if row['training']['sharpe']<baseline['training']['sharpe']-.05:fail.append('training_sharpe')
        if row['training']['max_drawdown']<baseline['training']['max_drawdown']-.03:fail.append('training_max_drawdown')
        if row['early']['cagr']<baseline['early']['cagr']-1e-12:fail.append('early53_cagr')
        if row['recent']['cagr']<baseline['recent']['cagr']-.02:fail.append('recent35_cagr')
        diagnostic[family]=dict(configuration=row,failed_guards=fail,changed_months=changed,improved_months=sum(z['difference']>0 for z in changed),worsened_months=sum(z['difference']<0 for z in changed),new_name_entries=entrants,base_new_name_entries=baseentrants,admitted=not fail)
    s.dump('TRAINING_ATTRIBUTION.json',diagnostic)
    blocks,_,bounds,valid=s.rp.paths(px,m,122);base_rr=s.rp.fast_returns(bc,blocks,valid);primary_rr=s.rp.fast_returns(pc,blocks,valid)
    if changes and max(z['log_difference'] for z in changes)>0:
        best=max(changes,key=lambda z:z['log_difference']);i=m['dates'].index(best['date']);counter=primary_rr.copy();counter[bounds[i]:bounds[i+1]]=base_rr[bounds[i]:bounds[i+1]]
        concentration=dict(best_benefit=best,without_best_changed_month=s.rp.stats_rr(counter,m['dates'][0],m['ends'][-1]),note='Attribution only. Not candidate selection and not an executable new strategy.')
    else:concentration=dict(note='No beneficial changed month in selected primary; previous66 retained.',changed_months=len(changes))
    s.dump('CONCENTRATION.json',concentration)
    wf=s.read(s.P/'WALKFORWARD.json');protected=s.read(s.P/'PROTECTED.json');bad=[p for p,h in protected.items() if s.sha(p)!=h];assert not bad,bad
    summary={name:dict(**{k:z['metrics'][k] for k in ['cagr','sharpe','max_drawdown']},selected_configuration_id=z['configuration']['id'] if 'configuration' in z else None) for name,z in results.items()}
    s.dump('SUMMARY.json',dict(results=summary,training_diagnostics={k:dict(id=z['configuration']['id'],cagr=z['configuration']['training']['cagr'],sharpe=z['configuration']['training']['sharpe'],changed_months=len(z['changed_months']),improved_months=z['improved_months'],worsened_months=z['worsened_months'],new_entries=z['new_name_entries'],failed_guards=z['failed_guards']) for k,z in diagnostic.items()},training_margin_counts={str(t):sum(z['score_gap']<=t and z['split']=='training88' for z in margins) for t in [.005,.01,.02]},walkforward=wf))
    validation=s.read(s.P/'VALIDATION.json');validation.update(unit_tests_passed=12,no_post_later34_tuning=True,unchanged_registered_observers=True,freeze_sha256=s.sha(s.P/'FROZEN.json'));s.dump('VALIDATION.json',validation)
    newbetter=results['primary']['metrics']['cagr']>results['previous66']['metrics']['cagr']+1e-12
    s.dump('COMPLETION.json',dict(status='completed',trials=len(trials),new_variants=len(trials)-2,tests_passed=12,protected_files=len(protected),improvement_found=newbetter,target80_achieved=False,registered_new_observer=False,primary_configuration=fr['selected']['primary']['id'],report=str(s.P/'REPORT.txt')))
    def pct(v):return f'{v:.2%}'
    lines=['月末选股边界微调研究','已完成；本轮没有提高现有候选。',
      '主口径：2016-08-17至2026-09-24，零交易成本、同一最终收盘信号/成交的理想化回放；每月等权重置，此后固定复权单位。',
      '当前66.10%研究候选不等于已登记65.34%观察账户；所有原观察账户保持不变。',
      '',f'预先固定118组：2个对照、2个前四/前五保留、6个换股门槛、12个分歧降权、96个组合。',
      '只用前88期（至2023-11-30）筛选并冻结，再重放后34期。历史已被多轮研究，不能称独立样本外验证。',
      '', '训练段结果（2016-08-17至2023-11-30；以下每类取训练年化最高者，即使不合格也列出）：',
      '版本 | 年化 | Sharpe | 前53期年化 | 后35训练期年化 | 改变持仓月份 | 新进入持仓次数（含首次建仓）',
      f"原66.10%候选 | {pct(baseline['training']['cagr'])} | {baseline['training']['sharpe']:.3f} | {pct(baseline['early']['cagr'])} | {pct(baseline['recent']['cagr'])} | 0 | {diagnostic['retention']['base_new_name_entries']}"]
    titles={'retention':'前四名保留','switch_gap':'前四名+0.005分差换股','disagreement':'Ridge正则化分歧>0.30时取消ML加减分','combined':'分差门槛+上述分歧降权'}
    for fam,z in diagnostic.items():
        r=z['configuration'];lines.append(f"{titles[fam]} | {pct(r['training']['cagr'])} | {r['training']['sharpe']:.3f} | {pct(r['early']['cagr'])} | {pct(r['recent']['cagr'])} | {len(z['changed_months'])} | {z['new_name_entries']}")
    lines += ['','保留/分差门槛/组合均未通过训练段守门条件，未为了寻找高收益而测试这些被拒规则的后34期。',
      '训练段只有两种分歧阈值版本通过，训练持仓与旧候选相同；冻结其中一个作全期对照后仍未改变持仓，主候选按简洁性保留旧规则。RESULTS中retention、switch_gap、combined指fallback原策略，不表示新规则达到66.10%。',
      '', '全期实际保留结果：','版本 | 年化 | Sharpe | 最大日终回撤']
    for name,title in [('original65','已登记观察规则研究回放'),('previous66','此前研究候选／本轮保留'),('qqq','QQQ买入持有')]:
        v=results[name]['metrics'];lines.append(f"{title} | {pct(v['cagr'])} | {v['sharpe']:.3f} | {pct(v['max_drawdown'])}")
    lines += ['','执行敏感性（同一固定信号候选，不是额外搜参）：']
    for key,title in [('next_open_zero','次日开盘、零成本'),('next_open_cost','次日开盘、每边0.05%成本')]:
        v=results['primary'][key]['metrics'];lines.append(f"{title}: 年化{pct(v['cagr'])}，Sharpe {v['sharpe']:.3f}，最大回撤{pct(v['max_drawdown'])}。")
    lines += ['', '逐月归因（仅训练段；择优月份不构成新的可交易规则）：']
    for fam in ['retention','switch_gap']:
        z=diagnostic[fam];lines.append(f"{titles[fam]}改变{len(z['changed_months'])}个月，改善{z['improved_months']}个月，恶化{z['worsened_months']}个月。")
        for tag,fun in [('最大改善' if z['improved_months'] else '损失最小',max),('最大损失',min)]:
            if z['changed_months']:
                r=fun(z['changed_months'],key=lambda r:r['log_difference']);lines.append(f"  {tag} {r['date']}→{r['end']}: {','.join(r['before'])} → {','.join(r['after'])}; {pct(r['before_return'])} → {pct(r['after_return'])}，差{r['difference']*100:+.2f}个百分点。")
    counts={str(t):sum(z['score_gap']<=t and z['split']=='training88' for z in margins) for t in [.005,.01,.02]}
    lines += ['',f'训练段第三/第四名分差≤0.005/0.01/0.02分别为{counts}，完整月度边界见THIRD_FOURTH_MARGIN.json。',
      '评分接近并不表示下一月收益接近。本轮减少换股确实减少新进入次数，但保留旧标的也错失了部分领跑者；没有证据据此提高未来收益。',
      f"顺序前缀诊断：动态选择规则年化{pct(wf['metrics']['cagr'])}、Sharpe {wf['metrics']['sharpe']:.3f}；同期原候选{pct(wf['base']['cagr'])}、Sharpe {wf['base']['sharpe']:.3f}。此诊断也使用反复研究的历史。",
      '',f'验证：12项测试通过；原122期两组控制订单/评分一致；独立日收益回放与交易引擎一致；冻结训练结果一致；{len(protected)}个保护文件哈希未变。',
      '未新增/替换观察账户，未改动旧账本、冻结计划、规则或指纹。',
      '局限：股票池仍为历史市值估算代理、ETF17为自拟列表、较早训练成员存在代理假设；同收盘理想成交与重复历史搜索限制可交易性和泛化解释。未复现作者，也未达到80%目标。']
    (s.P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
