"""Readable audit and static scientific figure for this research iteration."""
from explore import *

def main():
    env=prepare();r=read(P/'ROUNDED_RULE.json');simple=read(P/'SIMPLIFY_SELECTED.json')
    statuses=read(P/'JOINT_RUN_STATUS.json');cross=read(P/'CROSS_STATUS.json');mono=read(P/'MONOTONE_STATUS.json')
    bounds=read(P/'BOUND_RESOLUTION.json');rolling=read(P/'ROLLING_MODEL_STATUS.json')
    timing=read(P/'TIMING_VALIDATION.json');monthly=r['months'];saved=read(O/'monthly_decision_reconstruction/SNAPSHOTS.json')
    detail=[]
    for m in monthly:
        d=m['order']['start'];row=env['rows'][d];x=row['X'][r['fit']['family']];w=np.array(r['fit']['weights']);scores=np.einsum('ij,j->i',x,w,optimize=False)
        old={v['symbol']:v['raw'] for v in saved if v['date']==d};names=[]
        for i in np.argsort(-scores,kind='stable')[:6]:
            s=row['symbols'][i]
            names.append(dict(symbol=s,score=float(scores[i]),momentum_contribution=float(x[i,2]*w[2]),
                volatility_contribution=float(x[i,FEATURES.index('vol63')]*w[FEATURES.index('vol63')]),
                volume_contribution=float(x[i,FEATURES.index('volume_ratio21_126')]*w[FEATURES.index('volume_ratio21_126')]),
                r6_2=old[s]['mom6_2'],vol63_daily=old[s]['vol63'],volume_ratio=old[s]['volume_ratio21_126']))
        detail.append(dict(date=d,top_six=names,third_fourth_margin=float(sorted(scores,reverse=True)[2]-sorted(scores,reverse=True)[3])))
    dump('SIMPLE_DECISION_DETAILS.json',detail)
    ntr=sum(len(read(f)) for pat in ['20*_TRIALS.json','JOINT_*_TRIALS.json','CROSS_*_TRIALS.json','MONOTONE_*_TRIALS.json','SIMPLIFY_TRIALS.json'] for f in P.glob(pat))
    certificates=[]
    for mode,fn in [('original','CROSS_all_four_TRIALS.json'),('monotone','MONOTONE_all_four_TRIALS.json')]:
        for t in read(P/fn):
            updated=next((b for b in bounds if b['core_hypothesis']==mode and b['kind']==t['kind'] and b['family']==t['family']),t)
            upper=-updated['mip_dual_bound'];certificates.append(dict(core=mode,kind=t['kind'],family=t['family'],margin_upper_bound=upper,strict_positive_margin_excluded=upper < -1e-6))
    dump('FAMILY_BOUND_CERTIFICATES.json',certificates)
    assert len(certificates)==18 and all(c['strict_positive_margin_excluded'] for c in certificates)
    nav=pd.Series({pd.Timestamp(z['date']):z['value'] for z in r['nav']});through_nov=nav.loc[:'2020-11-30'];ret=float(through_nov.iloc[-1]/through_nov.iloc[0]-1)
    lines=['月策略短段复现：本轮规则探索结果','',
        '结论：找到了一个仅3个因子的固定月度评分规则，可解释2020年7—11月连续短段，其中10、11月未用于这条规则的初始持仓标签拟合。12月回撤形状失配。尚未恢复作者全期策略，也未验证80.64%年化或1.867 Sharpe。',
        '这里的“可解释”仅表示没有被原有图片误差筛查排除；不是作者确认的真实交易，不是精确重建，也不是独立样本外投资业绩。原图此前已用于研究。','',
        '一、可读的简化规则',
        '评分 = 0.50 × rank(R6−2) + 0.30 × rank(63日波动率) − 0.20 × rank(21日平均成交量 / 126日平均成交量)。',
        'R6−2 = 复权价格[t−42] / 复权价格[t−126] − 1；明确跳过最近42个交易日。63日波动率为63个每日复权收益率的样本标准差。',
        'rank采用当日代理池内股票、ETF分别计算的百分位，再居中到[-0.5,0.5]，不是买入仓位。原始成交量比缺失时沿用既有中性处理。',
        '代理池为历史Nasdaq-100与S&P100并集内的估算Top32，加现有假定17只ETF。资格沿用正R12−2和253日价格预热；不得把代理池称为已核实作者池。',
        '月末以当时数据评分，选前三只等权，月内固定股数。2只合格时各半，不足2只空仓；本次所有被检验月份均有至少3只合格。零成本、月末同收盘成交为原图复盘口径，不是实际成交保证。',
        '含义：优先中期已经走强、仍有波动空间、近期成交量相对半年基准较安静的标的。没有加入日策略的回撤买入硬条件，没有新增杠杆或月内止损。',
        '精确局部拟合系数为0.500000、0.295824、−0.204176；简化成50/30/−20后，训练三个月和后续三个月的选择均与精确3因子规则一致。','',
        '二、这条规则的诊断持仓（不是获准导出的作者交易）',
        '持仓月份 | 决策收盘日 | 规则产生的三只 | 月收益 | 原图片筛查']
    for i,m in enumerate(monthly):
        o=m['order'];phase='拟合标签期' if i<3 else '冻结参数延伸'
        lines.append(f"{pd.Timestamp(o['end']).strftime('%Y-%m')} | {o['start']} | {' / '.join(o['holdings'])} | {o['period_return']:+.2%} | {'未排除' if m['check']['not_rejected'] else '失配'}；{phase}")
    lines.extend(['',f'7—11月本候选累计{ret:+.2%}。不把这5个月收益年化，也不据此外推全期Sharpe。',
        '10月同时存在另一条6因子规则产生TSLA/ARKK/SMH，也未被图片排除；3因子规则则产生TSLA/SMH/SOXX。两种解释并存，说明不能把ETF之间的替换认作作者真实交易。',
        '12月净值与相对QQQ仍未被排除，但回撤面板9个可见列中2个超过原误差范围，超过10%拒绝阈值，保留失败；没有调整图片容差使它通过。',
        '所用同一初始净值和此前峰值来自已有完整反推路径。月与月之间持续累计，不重新锚定净值或重置峰值。','',
        '三、换时间段与共同规则检查',
        '第一轮每窗只保留30条持仓路径，在2025/2026未找到规则；本轮改用混合整数优化，将每月全部保留歧义组合和同一套权重一起求解。下表组合数量是优化模型覆盖的离散搜索空间，不是全部逐条人工枚举。',
        '时间段 | 候选组合路径数 | 经局部稳健性与图片筛查的求解候选数 | 选中规则的下一月'] )
    for s in statuses:
        lines.append(f"{s['start']}—{s['end']} | {s['candidate_paths_total']:,} | {s['local_robust_count']} | {'未排除' if s.get('next_month_not_rejected') else '失配'}")
    lines.extend(['2020年8月在三个横轴微调下的近似最优集合没有共同交集；明确采用并集作为较弱候选线索。只扩大候选搜索集合，最终连续曲线的容差与拒绝标准完全不变。',
        '四段共同参数：最初有两类模型的求解上界仍容许正分差，因此将它们延长求解。最终18个预先定义的固定评分家族均有负的最大分差上界，说明它们无法在本次保留的四段候选集合中共同严格选中三只。这个结论只适用于已声明因子、预算、代理池和图片候选集，不证明作者策略不存在，也不证明其他模型不可能。',
        '2020与2025共同参数：有7个候选能同时解释两段初始三个月，但按事先简单性规则选中的固定版本在两段紧邻月份都失配。没有按延伸期收益挑另一个赢家。'])
    lines.append('额外测试了动量跨度的一致性/最强/最弱汇总，保持正向单调与跳过42日，结果如下：')
    for s in mono:lines.append(f"  {s['group']}：{s['trials']}次求解，局部通过{s['local_passes']}，限时未收敛{s['timed_out']}；下一月结果{s.get('next_months_not_rejected','无可选候选')}。")
    lines.append(f"另外测试9个历史滚动Ridge小幅修正方法：固定36个已结束月份训练、alpha10、不调参、每只修正绝对值不超过0.05。共{rolling['training_audits']}个训练截点留存审计，9个方法都未能解释全部四段；也没有任何一个完整长窗口逐月全部未被排除。这个实验没有证明作者的未知Ridge实现无效。")
    lines.extend(['',
        '四、验证与边界',
        f"本轮记录{ntr}次固定规则/优化检查，另有2次延长求解和9个滚动修正方法；它们不是独立统计检验。这是多次事后模型开发，不能把局部匹配当成泛化证明。",
        '3因子精确规则在500次相对±1%系数扰动下，三个月训练选择全部保持；0.001及两位简化也保持。此处不是持仓真实性的置信概率。',
        f"截至决策日重新截断计算：{timing['feature_date_checks']}项因子/日期检查、{timing['security_feature_cells']}个证券特征值全部一致；{timing['protected_files_unchanged']}个既有来源、研究和观察文件校验未变。",
        '混合整数算法在小型问题上与穷举LP最优值交叉验证；正向动量单调、非线性侧因子幅度、权重扰动预算及月内固定股数/跨月净值衔接有针对性测试。',
        '全部新结果留在本研究目录；未修改原六账户、冻结计划、历史账本、策略指纹，也未新增观察账户。没有运行全期收益回测。',
        '作者具体股票池、17只ETF、模型训练方式及Sharpe定义仍未知。本轮的成果是可检验、稳定、较简单的短段规则，不能宣称找到了作者本人规则。','',
        '主要文件：SIMPLE_RULE.png；ROUNDED_RULE.json；SIMPLE_DECISION_DETAILS.json；JOINT_*_TRIALS.json；CROSS_*_TRIALS.json；MONOTONE_*_TRIALS.json；TIMING_VALIDATION.json。'])
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    # Static scientific figure. Preserve the source image; show its digitized observations separately.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cal=env['cal'];initial=r['initial_nav'];peak=r['prior_peak'];start,end=pd.to_datetime(cal['dates'])
    def dates_from_x(x):return start+pd.to_timedelta((x-cal['wealth']['x0'])/cal['wealth']['width']*(end-start).days,unit='D')
    q=env['px'].loc[nav.index,'QQQ'];orig_q=env['px'].at[pd.Timestamp(cal['dates'][0]),'QQQ']
    wealth=nav/initial;relative=wealth/(q/q.iloc[0]);dd=100*(nav/np.maximum(peak,nav.cummax())-1)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for ax,panel,series,normalizer in zip(axes,['wealth','relative','dd'],[wealth,relative,dd],[initial,initial/(q.iloc[0]/orig_q),1]):
        z=env['obs'][panel];dt=dates_from_x(z.x);mask=(dt>=nav.index[0])&(dt<=nav.index[-1]);z=z[mask];dt=dt[mask]
        if panel!='dd':
            c=cal[panel];mid=np.exp((c['y0']-z.y)/c['log_scale'])/normalizer
            low=np.exp((c['y0']-z.ymax-2)/c['log_scale'])/normalizer;high=np.exp((c['y0']-z.ymin+2)/c['log_scale'])/normalizer
            ax.fill_between(dt,low,high,color='#9364a5',alpha=.18,label='Digitized reference + vertical pixel allowance')
        else:
            c=cal['dd'];mid=100*(c['y0']-z.y)/c['scale'];low=100*(c['y0']-z.ymax-4)/c['scale'];high=100*(c['y0']-z.ymin+4)/c['scale']
            ax.vlines(dt,low,high,color='#9364a5',alpha=.25,lw=6,label='Digitized reference + vertical pixel allowance')
        ax.scatter(dt,mid,color='#9364a5',s=12,label='Compressed reference pixels')
        fit=series.loc[:'2020-09-30'];extension=series.loc['2020-09-30':]
        ax.plot(fit.index,fit,color='#007d8e',lw=2,label='50 / 30 / -20 rule: fitted choices')
        ax.plot(extension.index,extension,color='#007d8e',ls='--',lw=2,label='Same fixed rule: continuation')
        ax.axvspan(pd.Timestamp('2020-11-30'),pd.Timestamp('2020-12-31'),color='#d96650',alpha=.08)
        ax.axvline(pd.Timestamp('2020-09-30'),color='#555',ls=':',lw=1)
        ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Wealth / June 30 value');axes[1].set_ylabel('Relative wealth / QQQ (rebased)');axes[2].set_ylabel('Drawdown from carried peak (%)')
    axes[0].legend(loc='upper left',fontsize=8,ncol=2)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator());axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('A simple monthly rule explains July-November 2020; December fails the drawdown screen',fontsize=13,y=.985)
    fig.text(.5,.948,'50% R6-2 momentum + 30% 63-day volatility - 20% relative volume rank | Three equal positions, monthly fixed units',ha='center',fontsize=9)
    fig.text(.5,.014,'Retrospective hypotheses, proxy universe, zero costs. Pixel test also allows +/-1 horizontal pixel; no proof of author trades or full-period Sharpe.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.938]);fig.savefig(P/'SIMPLE_RULE.png',dpi=160);plt.close(fig)
    protected=read(P/'PROTECTED_SOURCES.json')
    for f,h in protected.items():assert sha(f)==h,f
    dump('FINAL_VALIDATION.json',dict(protected_files_unchanged=len(protected),rule_optimization_checks=ntr,
        factor_prefix_checks=timing['feature_date_checks'],negative_margin_family_certificates=len(certificates),rolling_model_methods=rolling['methods'],full_period_backtest_run=False,author_strategy_admitted=False,
        original_accounts_unchanged=True,simple_rule_five_month_return=ret,all_failures_retained=True))
    dump('FINAL_CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print('REPORT and FIGURE complete; protected',len(protected),'trials',ntr,'five month return',ret)
if __name__=='__main__':main()
