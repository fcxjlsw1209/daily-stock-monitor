from validate import *

def pct(x):return f'{x*100:.2f}%'
def main():
    v=read(P/'VALIDATED_RESULTS.json');audit=read(P/'FINAL_AUDIT.json');meta=read(P/'INPUTS.json')
    inventory=[]
    for f in sorted(P.glob('*.sqlite')):
        c=sqlite3.connect(f);n=c.execute('SELECT COUNT(*) FROM trials').fetchone()[0];inventory.append(dict(file=str(f),trials=n));c.close()
    count=sum(r['trials'] for r in inventory)
    leader=v['cases'][0];spec=leader['spec'];m=meta[spec['key']];c,a,b,wc,wa,wb=spec['recipe'];g=spec['gamma'];names=m['feature_names']
    formula={names[c]:wc,names[a]:wa*(1-g/(1-wc)),names[b]:wb*(1-g/(1-wc)),spec['correction']:g}
    etf=[r['symbol'] for r in read(O/'four_factor_rebound/ETF_SOURCES.json')];etf=[s for s in etf if s not in {'DIA','XLB','XLC','XLRE','XLU','XLP'}]+['SMH','SOXX','IGV','VGT','MTUM','ARKK']
    rule=dict(id='monthly_momentum_regime_research_v1',status='Retrospective research candidate, no observer/production migration',primary_window=['2020-12-31','2026-09-24'],weights=formula,
        pool='Existing historical estimated-cap Top32 foreign-inclusive proxy plus theme17; not certified true all-US PIT Top32',etf_hypothesis=etf,
        eligibility='Require295consecutive valid positive closes and P[t-42]/P[t-252]-1>0; full49name factor rank before gate; fewer2eligible cash,2equal,otherwise3equal. Issuer dedup from cached membership. Do not inject source-ledger winners.',
        transform='Per-date per-factor ascending percentile rank(method=average)-0.5 across pool; missingneutral0. Tie on score uses lexical internal ticker; META stored as research_equity_FB.',
        definitions={'mom294_risk_recent63':'(P[t-42]/P[t-294]-1) / sample_std(last63 daily adjusted returns). A252-session momentum window ending42sessions ago.',
                     'vol126':'sample_std(last126 daily adjusted returns)',
                     'acceleration_skip42':'P[t-42]/P[t-126] - P[t-126]/P[t-210]',
                     'cmf63':'sum63(volume*(2*rawClose-High-Low)/(High-Low))/sum63(volume); CLV0when High=Low',
                     'high_market_vol__neg_cmf63':'Centered percentile(-CMF63) multiplied by +1if SPY63-session daily volatility > its own trailing252-session median, else-1'},
        signal='Last trading-day close each month, completed daily data only',executable_research_fill='Next trading-day adjusted open; preserve previous holdings through overnight gap; observed opening price proxy, no guaranteed fills',
        sizing='3equal dollar target weights at execution, fixed units between events, rebalance differences only, no leverage, no intramonth stops',
        cost='0.05%each side of actual rebalance notional, fees solved from available NAV; no exit fee for positions still held at final date; no cash interest, tax or impact model',
        labels='69user-supplied simulated holding periods are sole reconstruction target; original image is not used',
        validation=leader['runs'],
        limitations=['All periods were repeatedly searched; coefficient search is retrospective, not out-of-sample or identified author code.',
                     'Unknown real17ETFlist and estimated cap vintages remain universe limitations; missing author members remain mismatches.',
                     'Daily close ranks can differ materially from previous-day ranks; same-close fill is a reference, not an executable guarantee.',
                     'Adjusted units/prices approximate distributions/splits; are not literal unadjusted broker share counts.'],
        source_spec=spec)
    dump('CANDIDATE_RULES.json',rule)
    source=read(O/'monthly_supplied_ledger_audit/PERIODS.json');labels={r['start']:set(r['holdings']) for r in source}
    comparisons=[]
    for case in v['cases']:
        rows=read(P/(case['name']+'_close_zero_HOLDINGS.json'))
        for row in rows:
            target=labels[row['signal_date']];chosen=set(row['symbols']);comparisons.append(dict(case=case['name'],date=row['signal_date'],target=sorted(target),selected=sorted(chosen),shared=sorted(chosen&target),exact=chosen==target))
    dump('HOLDINGS_COMPARISON.json',comparisons)
    report=['月策略因子穷举与逐日复核','',
        '本轮只以用户提供的69期模拟持仓/调仓记录为复现目标。未使用原收益图、图像拟合持仓或图中绩效来打分。',
        '主区间：2020-12-31收盘至2026-09-24收盘（68个完整月＋2026年9月部分月）。初始100,000美元，均为历史研究。','',
        '实际结果：找到了超过70%历史年化的替代候选，但没有复现作者完整交易。',
        f"本轮共记录{count:,}组参数配置，包含产生相同持仓的重复配置；它们不是独立实验。全部为事后研究，不能称为样本外证明。",'',
        '主要结果（逐日净值统计；交易成本：每边0.05%，按实际差额调仓金额计）：',
        '方案 | 年化 | 日收益Sharpe（无风险利率0） | 最大日终回撤']
    for label,k in [('候选／月末同收盘零成本参考','close_zero'),('候选／月末同收盘计成本','close_cost'),('候选／月末信号、次日开盘计成本','open_cost'),('候选／提前一天信号、月末收盘计成本','prior_close_signal_cost')]:
        r=leader['runs'][k];report.append(f"{label} | {pct(r['cagr'])} | {r['sharpe']:.3f} | {pct(r['max_drawdown'])}")
    for x in v['benchmarks']:
        report.append(f"{x['symbol']}买入持有（期初收盘、只扣买入费） | {pct(x['cagr'])} | {x['sharpe']:.3f} | {pct(x['max_drawdown'])}")
    report+=['','候选规则：',
        '1. 使用现有历史市值估算Top32代理池，加下面17只ETF；不依据作者持仓反向加入股票。',
        ', '.join(etf),
        '2. 至少295个连续有效交易日预热，跳过最近42日的12个月端点动量为正；按全池横截面百分位标准化因子，再在合格标的中排名。',
        '3. 分数：52%风险调整长期动量＋21.9375%126日波动率＋18.5625%跳过近42日的动量加速＋7.5%市场波动状态×反向CMF63。',
        '长期动量为P[t-42]/P[t-294]-1，再除最近63日收益标准差。加速为P[t-42]/P[t-126]-P[t-126]/P[t-210]。',
        '状态为SPY63日波动率是否高于其过去252日中位数：高则使用反向CMF63分位，低则翻转符号。该方向由这次事后搜索得到，不是已知的作者规则。',
        '4. 月末收盘形成名单，次日开盘3只等权；只有2只合格则各半，不足2只空仓。月内固定股数，无止盈止损；月底按差额调仓。',
        '5. 没有加入杠杆或日策略的回撤买点。小幅Ridge修正也已测试，但本轮收益领先候选采用市场状态修正，未使用机器学习评分。','',
        '交易复现：',
        f"收益候选只匹配{spec['exact']}/69期完整持仓、{spec['overlap']}/207个席位。",
        f"持仓匹配最高候选为{v['cases'][2]['spec']['exact']}/69期、{v['cases'][2]['spec']['overlap']}/207个席位，次日开盘计成本年化{pct(v['cases'][2]['runs']['open_cost']['cagr'])}。",
        '原作者实际ETF全名单和历史市值快照不可得；仍有6个账本持仓落在代理Top32之外，已计作未匹配，没有删除这些月份。',
        '完整69期并排持仓见HOLDINGS_COMPARISON.json；逐笔买卖见return_leader_open_cost_TRADES.json。','',
        '稳健性与限制：',
        '提前一天形成信号后明显变差，说明它对月底截面排序较敏感。次日开盘结果保留了月末完整信号，并没有用该月月底信号买在该月月初。']
    for r in audit['extra_checks']:
        if r['case']=='return_leader':report.append(f"{r['control']}，单边费率{pct(r['cost_per_side'])}：年化{pct(r['cagr'])}，Sharpe{r['sharpe']:.3f}，回撤{pct(r['max_drawdown'])}，{r['start']}至{r['end']}。")
    n=audit['neighborhood'][0];report.append(f"最终四项权重各自±5%扰动、归一化且长期动量≥50%的{n['valid_draws']}个邻域试验：同收盘零成本年化中位{pct(n['median_cagr'])}，最低{pct(n['min_cagr'])}。这是同一历史样本的敏感度诊断，不能代替样本外检验。")
    report+=['另一按2021–2024得分优先挑出的候选，次日开盘计成本年化'+pct(v['cases'][1]['runs']['open_cost']['cagr'])+'。它来自同一事后探索集合；其2025–2026结果也不能视为严格独立样本外。',
        '“70%”是在账本整个覆盖区间的历史年化；更早区间仅作额外检查。估算股票池仍可能有市场覆盖/快照时点偏差，不能因此宣称真实可交易未来年化达到70%。','',
        '验证：10个自动化测试；370个因子截断重算检查、5个市场状态截断检查、20个缓存特征一致性检查；独立逐日持仓重算与向量化月收益累计一致；16个原观察账户/规则/账本文件校验和不变。',
        '选股使用截至当日的因子，持仓路径缺数据会使配置验证失败，不用未来缺失情况提前过滤赢家。成本扣减、初始手续费、固定股数、开盘隔夜持仓与差额调仓均有检查。',
        '未修改现有模拟账户、冻结计划、监测规则，也没有下单。','',
        '候选各自然年收益（次日开盘、计成本；2026只到9月24日）：']
    for r in leader['runs']['open_cost']['annual_returns']:report.append(f"{r['year']}{'截至9/24' if r['partial'] else ''}: {pct(r['return_'])}")
    report+=['','本目录可复算命令：','PYTHONPATH=work/monthly-decision-deps:work/research-plot-deps OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/monthly_factor_search_v2/validate.py',
        'PYTHONPATH=work/monthly-decision-deps:work/research-plot-deps OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/monthly_factor_search_v2/audit_final.py']
    (P/'REPORT.txt').write_text('\n'.join(report)+'\n')
    dump('TRIAL_INVENTORY.json',dict(total_configurations=count,files=inventory))
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    dump('STATUS.json',dict(phase='completed_finite_search_and_validation',primary_goal='Find ledger-matching monthly rule or>=70%full-ledger-window historical CAGR',alternative_numeric_threshold_passed=True,author_strategy_replicated=False,source_image_used=False,original_accounts_unchanged=True,recorded_configurations=count,top_rule='CANDIDATE_RULES.json',net_next_open_cagr=leader['runs']['open_cost']['cagr'],remaining_limitations=rule['limitations']))
    print('SAVED REPORT',count,flush=True)
if __name__=='__main__':main()
