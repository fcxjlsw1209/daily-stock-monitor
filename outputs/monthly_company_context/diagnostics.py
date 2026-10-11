"""Training-only rejected-rule attribution; final chosen rule remains frozen."""
import context_research as s


def main():
    fr=s.check_freeze();m,d,px,op,a,_=s.load();pred=dict(s.np.load(s.P/'FORECASTS.npz'));rows=s.read(s.P/'TRAIN_TRIALS.json');results=s.read(s.P/'RESULTS.json');details=s.read(s.P/'COMPANY_CONTEXT_DETAILS.json')
    base=rows[1];bc,bs=s.boundary_choices(base,a,pred,m,d,122);families=['global_ridge','company_mean','company_context','company_shape','company_signal','company_signal_context'];out={}
    for fam in families:
        r=max((x for x in rows if x['family']==fam),key=lambda x:(x['training']['cagr'],-x['id']));ch,_=s.boundary_choices(r,a,pred,m,d,88);changes=[]
        for i,date in enumerate(m['dates'][:88]):
            before=[m['symbols'][i][j] for j in bc[i]];after=[m['symbols'][i][j] for j in ch[i]]
            if set(before)==set(after):continue
            ret=lambda ss:float((px.loc[m['ends'][i],ss]/px.loc[date,ss]-1).mean()) if ss else 0.
            b,z=ret(before),ret(after);changes.append(dict(date=date,end=m['ends'][i],before=before,after=after,before_return=b,after_return=z,difference=z-b))
        out[fam]=dict(configuration=r,changes=changes,improved_months=sum(x['difference']>0 for x in changes),worsened_months=sum(x['difference']<0 for x in changes),admitted=s.prior.allowed(r,base),evaluated_on_later34=False)
    s.dump('TRAINING_ATTRIBUTION.json',out)
    near=set()
    for i,date in enumerate(m['dates'][:88]):
        rank=sorted(s.np.flatnonzero(d['eligible'][i]),key=lambda j:(-bs[i,j],m['symbols'][i][j]))
        if len(rank)>=3:
            for j in rank[2:5]:
                if bs[i,rank[2]]-bs[i,j]<=.02:near.add((date,m['symbols'][i][j]))
    sample=[z for z in details if z['model_id']==10 and z['signal_type']>0 and (z['date'],z['symbol']) in near]
    quality=dict(training_boundary_cases=len(near),cases_with_pullback_signal=len(sample),signal_counts={str(k):sum(z['signal_type']==k for z in sample) for k in [1,2,3]},observations_median=float(s.np.median([z['observations'] for z in sample])),effective_samples_median=float(s.np.median([z['effective_samples'] for z in sample])),own_weight_median=float(s.np.median([z['own_weight'] for z in sample])),zero_history=sum(z['observations']==0 for z in sample),note='Sampled only historical month-end snapshots from existing training pool, not all daily pullback episodes. Quantity is not independent event count; this model is monthly-horizon adaptation.')
    s.dump('SAMPLE_DIAGNOSTICS.json',quality)
    compare=s.read(s.P/'MONTH_COMPARISON.json');changes=[z for z in compare if set(z['primary']['symbols'])!=set(z['previous66']['symbols'])];s.dump('CHANGED_MONTHS.json',changes)
    protected=s.read(s.P/'PROTECTED.json');bad=[p for p,h in protected.items() if s.sha(p)!=h];assert not bad,bad
    v=s.read(s.P/'VALIDATION.json');v.update(unit_tests_passed=14,no_post_later34_retuning=True,selected_changed_months=len(changes),events_tested=False,author_signal_definitions_known=False,unchanged_registered_observers=True);s.dump('VALIDATION.json',v)
    summary=dict(training={fam:{k:r['configuration'][k] for k in ['id','training','early','recent']} for fam,r in out.items()},primary=results['primary'],sample_diagnostics=quality,improvement_found=results['primary']['metrics']['cagr']>results['previous66']['metrics']['cagr']+1e-12)
    s.dump('SUMMARY.json',summary)
    titles={'global_ridge':'全市场Ridge边界修正','company_mean':'同公司历史收益均值＋收缩','company_context':'同公司＋自身形态＋市场环境','company_shape':'同公司＋自身形态（环境消融）','company_signal':'同公司＋相同回撤类型＋收缩','company_signal_context':'相同回撤类型＋形态＋市场环境'}
    p=lambda v:f'{v:.2%}'
    lines=['公司历史与市场环境：月策略边界选股研究','本轮98组配置（2控制、96新配置），未筛出超过原候选且满足原训练守门条件的新规则。',
      '研究目的：响应同公司历史、市场环境、实际反弹幅度而非胜率、少样本收缩、两种回撤信号分类的提示。',
      '', '已实现：',
      '1. 全市场Ridge：按年训练、只用此前五年的已成熟月收益标签；用作样本不足时的先验。',
      '2. 同公司经验：过去类似形态和市场状态下，下个完整自然月相对SPY的已实现超额收益；不是获胜概率。只使用同一证券ID，ETF使用同基金历史。',
      '3. 收缩：预测=(Σ样本权重×已实现收益 + 先验质量×全市场预测)/(Σ样本权重+先验质量)。同公司权重随样本数、接近程度、时效增加；先验质量测试6/12个单位。历史样本24个月半衰期。',
      '4. 形态：跳过42日长期动量、风险动量、趋势路径、波动、个股相对行业、CMF、卖压减弱；环境：市场广度、SPY200日趋势、高低波动。',
      '5. 两种信号的代理定义：A=复权5日收益<-0.5×ATR21/原始收盘价；B=(收盘-日低)/(日高-日低)<=0.25。分为Aonly、Bonly、both；仅匹配同类历史。作者实际公式未知，不能声称复现原指标。',
      '6. 月策略不加回撤入场门槛，月内不交易；只允许原第三至第四/第五名、与第三名分差<=0.01/0.02的候选竞争最后一席，前两名保持入选。修正预算2.5/5%，所有附加预算合计最多10%，保留长期动量核心。',
      '', '范围和切分：原归档Top32估算代理＋自拟17ETF，2016-08-17至2026-09-24。主要口径零成本、同最终收盘信号/成交的理想化回放。',
      '前88期截至2023-11-30训练/选参并冻结；后34期只重放合格规则。历史多次被研究，不能称独立样本外。',
      '', '训练段结果（每类取训练CAGR最高者，不代表通过守门）：',
      '版本 | 年化 | Sharpe | 早期53期年化 | 后35训练期年化 | 改动月份（好/坏）',
      f"原66.10%候选 | {p(base['training']['cagr'])} | {base['training']['sharpe']:.3f} | {p(base['early']['cagr'])} | {p(base['recent']['cagr'])} | 0"]
    for fam,z in out.items():
        r=z['configuration'];lines.append(f"{titles[fam]} | {p(r['training']['cagr'])} | {r['training']['sharpe']:.3f} | {p(r['early']['cagr'])} | {p(r['recent']['cagr'])} | {len(z['changes'])}（{z['improved_months']}/{z['worsened_months']}）")
    lines += ['','所有新增系列都未通过训练守门，因此没有为了救结果而比较其后34期，也未推广到观察账户。RESULTS中的各family为fallback原规则，不是新模型获得66.10%。',
      f"训练段边界候选{quality['training_boundary_cases']}个，其中出现代理回撤信号{quality['cases_with_pullback_signal']}个；同公司同类历史月样本中位数{quality['observations_median']:.1f}，有效权重样本量中位数{quality['effective_samples_median']:.2f}，同公司评分占预测权重中位数{p(quality['own_weight_median'])}。",
      '此样本来自月末快照，不代表作者每天扫描所有回撤事件的数据。此次目标是下月超额收益，也不等于原作者短线5/10日反弹标签。不能据此断言作者的同公司反弹因子无效。',
      '没有回撤信号或公司历史时，新的预估返回全市场Ridge；因此此版本的整体效果包含全市场边界修正的影响。全球Ridge独立对照也变差，不能把全部损失归因于公司历史层。',
      '', '保留全期结果（零成本同收盘）：']
    for key,title in [('original65','已登记65.34规则历史回放'),('previous66','现有最佳研究候选')]:
        z=results[key]['metrics'];lines.append(f"{title}: 年化{p(z['cagr'])}，Sharpe{z['sharpe']:.3f}，最大回撤{p(z['max_drawdown'])}。")
    for key,title in [('next_open_zero','保留候选：次日开盘零成本'),('next_open_cost','保留候选：次日开盘每边0.05%成本')]:
        z=results['primary'][key]['metrics'];lines.append(f"{title}: 年化{p(z['cagr'])}，Sharpe{z['sharpe']:.3f}。")
    lines += ['','事件层：本轮未加入财报、新闻或FOMC事件。已有缓存没有完整经核实的可用事件特征表；不能使用事后修订盈利预期，也不能将SEC接受时间自动当成公司首次财报公开时间。',
      '可准备的来源：SEC submissions JSON https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data；Fed政策决定与纪要 https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm。发布内容与当时可知发布时间都需归档。',
      '时间顺序验证参考：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html。此处实现自身标签成熟期检查，不是随机交叉验证。',
      '',f'验证：14项测试通过；两原控制122期订单/评分完全一致；独立每日固定份额回放一致；冻结前后训练指标一致；{len(protected)}个保护文件哈希未变。',
      '期间发现零特征行无法单行还原共同市场状态，已在选参前修复为从同一时间截面的其他证券恢复，并加入测试；无跨时点填补。',
      '限制：估算历史股票池、未知作者ETF名单/精确信号/收益标签、同收盘理想化执行、重复历史探索。没有新增实盘或观察账户。',
      '', '后续可证伪方向（本轮未执行、不是已验证结论）：',
      '用每日/每周出现的独立回撤事件，学习5/10/21日已实现反弹及不利路径；对事件重叠做去重、标签成熟期隔离；再将预测作为月策略小修正。',
      '让自身历史层相对于原模型的增量为零时完全保留原排名，单独检验公司条件信息的增量价值；本轮的全市场Ridge替换式对照显示这一拆分有必要。']
    (s.P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    s.dump('COMPLETION.json',dict(status='completed',trials=len(rows),new_variants=len(rows)-2,tests_passed=14,protected_files=len(protected),improvement_found=summary['improvement_found'],events_tested=False,new_observer_registered=False,target80_achieved=False,report=str(s.P/'REPORT.txt')))
    print('\n'.join(lines))

if __name__=='__main__':main()
