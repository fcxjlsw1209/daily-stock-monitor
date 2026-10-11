"""Consolidated audit and static research figure, without changing any paper account."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research import P, O, REBUILD, START, END, dump, sha


def read(name):
    return json.loads((P/name).read_text())


def main():
    base=read('RESULTS.json');gates=read('GATES_RESULTS.json');foreign=read('foreign/RESULTS.json');aligned=read('ALIGNED_RESULTS.json')
    inv=read('INVERSE_SUMMARY.json');ambiguity=read('HOLDING_AMBIGUITY.json');controls=read('EXECUTION_CONTROLS.json')
    valid=read('VALIDATION.json');selection=read('SELECTION.json');clues=read('CLUE_FEATURES.json');aligned_selection=read('ALIGNED_SELECTION.json')
    registry=[]
    for source,block in [('RESULTS.json',base),('GATES_RESULTS.json',gates),('foreign/RESULTS.json',foreign),('ALIGNED_RESULTS.json',aligned)]:
        for key,v in block.items():
            category='chronological_unlevered_hypothesis'
            if key.startswith('leveraged'): category='different_embedded_leverage_diagnostic'
            if v.get('inverse_teacher'): category='future_influenced_teacher_imitation'
            registry.append(dict(source=source,key=key,category=category,result=v))
    for source,block,category in [('INVERSE_SUMMARY.json',inv,'noncausal_image_explanation'),
                                 ('EXECUTION_CONTROLS.json',controls,'execution_sensitivity'),
                                 ('INVALID_CLOCK_CONTROLS.json',read('INVALID_CLOCK_CONTROLS.json'),'invalid_synthetic_control_not_author_explanation')]:
        for key,v in block.items():
            registry.append(dict(source=source,key=key,category=category,result=v))
    counts={c:sum(r['category']==c for r in registry) for c in sorted({r['category'] for r in registry})}
    failures=[dict(source=r['source'],key=r['key'],failure=r['result']['failure']) for r in registry if 'failure' in r['result']]
    primary=[r for r in registry if r['category']=='chronological_unlevered_hypothesis' and 'failure' not in r['result']]
    best=max(primary,key=lambda r:r['result']['cagr_252'])
    bydev=min(primary,key=lambda r:r['result']['development_global_log_rmse']+2*r['result']['development_month_shape_log_rmse'])
    dump(P/'TRIAL_REGISTRY.json',dict(counts=counts,total=len(registry),failures=failures,trials=registry))
    assumption=dict(author_future_month_ranking_not_used=True,
                    basis='User explicitly states author does not apply month-end ranks to the start of the same month; use that constraint in reconstruction.',
                    original_post_unavailable=True,
                    goal='Recover a chronological monthly selection rule explaining wealth, drawdown and holdings clues, not merely produce a target CAGR.',
                    author_holdings_verified=False,author_strategy_reproduced=False,
                    remaining_unknowns=['Exact17ETFs','Author all-US floatingTop32 and market-cap vintage','Model features, targets, training window and regularization','Exact month-end selection and retention rules'],
                    original_accounts_unchanged=True)
    dump(P/'ASSUMPTIONS.json',assumption)
    protected=read('../monthly_curve_reverse/PROTECTED_BEFORE.json');changed=[f for f,h in protected.items() if sha(Path(f))!=h]
    sources=read('SOURCE_HASHES.json');source_changes=[f for f,h in sources.items() if sha(Path(f))!=h]
    foreign_sources=read('foreign/ETF_MANIFEST.json')+read('foreign/CAP_SOURCES.json')
    foreign_changes=[r['file'] for r in foreign_sources if sha(Path(r['file']))!=r['sha256']]
    assert not changed and not source_changes and not foreign_changes and not failures
    testlog=(P/'ALIGNED_TESTS.txt').read_text();assert 'Ran 3 tests' in testlog and testlog.rstrip().endswith('OK')
    assert aligned_selection['feature_prefix_max_difference']<1e-9
    completed=dict(valid,additional_factor_tests_passed=3,aligned_feature_prefix_max_difference=aligned_selection['feature_prefix_max_difference'],
                   all_recorded_runs=len(registry),run_categories=counts,failures=failures,
                   protected_files=len(protected),protected_unchanged=True,
                   foreign_source_hashes_verified=len(foreign_sources),author_holdings_verified=False,
                   agreement_months=ambiguity['all_three_agree'],zero_common_name_months=ambiguity['zero_common_names'],
                   best_unlevered_hypothesis=best['key'],best_unlevered_cagr=best['result']['cagr_252'],
                   best_development_fit=bydev['key'])
    dump(P/'COMPLETE_VALIDATION.json',completed)
    # The absence of a source post is not proof that no public source exists.
    b=best['result'];execution=controls['theme17_m252_42_lag1_cost10'];aligned_best=aligned[aligned_selection['best_full_cagr_retrospective']]
    lines=['月策略复现研究：仍未确认作者交易或恢复作者算法','2026-10-09','',
           '本轮结论：已找到三条接近原图的候选持仓路径，但它们是利用事后价格反推，不能当作作者真实交易，也不能当作可交易策略。按时间顺序计算信号的假设，尚未达到作者80.64%年化。',
           '用户已明确作者没有把月末排序用于该月开头；后续按这一条件研究。先前故意构造的无效时间对照仅保留在审计附录，不作为对作者策略的解释。','',
           '一、是否已找出每个月的股票？',
           f"完成121个月、三组不同股票池/门槛假设的连续路径拟合。三方完全同组合{ambiguity['all_three_agree']}个月；没有三方共同股票{ambiguity['zero_common_names']}个月。换ETF假设后，前两条路径只有45个月完全一致。",
           '2020-08示例：原ETF假设为AAPL/NVDA/TSLA，主题ETF假设为NFLX/NVDA/TSLA，两者都可以解释图中走势。2025-09三种假设都为ORCL/PLTR/TSLA；这属于较一致的线索，仍不是已验证持仓。',
           '图中1个横向像素约3.53个日历日，净值纵向1像素约2.56%。每天多只股票的变化被压缩到一条曲线，完整组合不能从图中唯一识别。',
           '详见CANDIDATE_HOLDINGS.txt（121个月完整候选表）、HOLDING_AMBIGUITY.json。表内是推断调仓日，不是成交凭据。','',
           '二、路径拟合与策略回放必须区分',
           '方法 | 年化 | 最大回撤 | 含义',
           '作者原图 | 80.64% | -34.46% | 2016-08-17开始；零成本，作者披露历史样本重复使用',
           f"事后候选路径，原ETF池 | {inv['core17_q0']['cagr_252']:.2%} | {inv['core17_q0']['max_dd']:.2%} | 使用未来价格解释曲线，不是策略",
           f"事后候选路径，主题ETF池 | {inv['theme17_q0']['cagr_252']:.2%} | {inv['theme17_q0']['max_dd']:.2%} | 另一组不同持仓，也能近似拟合",
           f"按当时因子排名的最高收益假设 | {b['cagr_252']:.2%} | {b['max_dd']:.2%} | {best['key']}，本轮事后最高，非独立样本外",
           f"同规则改用前一日信号、双边10bps成本 | {execution['cagr_252']:.2%} | {execution['max_dd']:.2%} | 月末收盘交易，无月内操作",
           f"新测：动量/风险窗口对齐，最高收益 | {aligned_best['cagr_252']:.2%} | {aligned_best['max_dd']:.2%} | 32个已列明假设中的最高收益，不代表被验证模型",
           '我们的完整月回放从2016-08-31至2026-09-24，比原图少最初约两周。年化按252个交易日计算。原图与候选路径最大回撤、Sharpe也尚未完全匹配。',
           '同收盘信号、同收盘成交的零成本回放用于对齐作者图示口径；实际执行需要提前形成指令。另列前一日信号和成本敏感性，不能将同收盘诊断直接视为实盘可达结果。','',
           '三、已实际完成的研究',
           f"本目录累计保存{len(registry)}次运行/对照，其中{counts['chronological_unlevered_hypothesis']}个按时间顺序的无杠杆规则回放；所有结果和失败项均在TRIAL_REGISTRY.json。不能把多次尝试后的最高值当作独立验证。",
           '包括动量长度和跳过月数、风险调整、行业相对强度、量价、前五保留；36/72个月滚动Ridge、三种成熟标签、小幅残差树；趋势门槛；行业/主题ETF；增加TSM/BABA/NVO/SAP/TM的股票池敏感性。',
           '新增32个窗口解释：动量和风险都跳过最近42个交易日，比较风险调整、下行风险、回归斜率×趋势质量、与波动空间/行业/CMF的小比例组合。本轮最高收益仍未超过原有44.73%。',
           f"14个三方一致月份有42条候选持仓；其中R12-2排前三的只有{clues['top3']}条，排前五的{clues['top5']}条，中位排名{clues['median_r12_2_rank']:.1f}。这提示纯动量前三不足以解释候选，但不能据此确定模型权重。",
           '17只ETF和全美股Top32原始定义仍未知。主池沿用指数并集内的历史市值代理Top32，扩展外国发行人也不是完整全美股池。历史市值代理披露时间尚不具备全量独立认证，因此不能称已精确恢复作者股票池。',
           '独立评估期目前不存在：这些年份已在前几轮研究中看过。按2016-2021图形误差选规则，再看2022以后，只是分段诊断。事后持仓模仿也不作为因果或样本外结果。','',
           '四、验证和保护',
           '12项账务/持仓/模型标签/连续路径测试通过；另3项新特征测试通过。实际Ridge与残差树在截断未来数据后，截点前预测差异为0；新特征截断差异也为0。',
           f"{len(protected)}个原账户与保护文件校验和未变化；{len(sources)}个核心来源及{len(foreign_sources)}个外国发行人来源校验通过。没有修改观察账户、规则或冻结计划。",
           '模块：research.py、inverse_path.py、gates_and_imitation.py、foreign_pool.py、aligned_momentum.py、identifiability.py、clue_features.py、validate.py、report.py。使用既有work/venv/bin/python，PYTHONPATH=work/research-plot-deps，OPENBLAS_NUM_THREADS=1。',
           '截图原件与提取过程保存在../monthly_curve_reverse。所有候选持仓都没有获得作者确认。','',
           '尚未完成的目标：唯一识别作者每月交易、恢复能事前选出相似持仓的月策略，以及同时解释原图收益和回撤。当前不能宣称复现成功。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    plot(base, inv, ambiguity)
    dump(P/'CODE_HASHES.json',{v.name:sha(v) for v in P.glob('*.py')})
    print(json.dumps(dict(runs=len(registry),categories=counts,best=best['key'],cagr=b['cagr_252'],goal_achieved=False,protected_unchanged=True),indent=2))


def plot(base,inv,ambiguity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    cal=read('../monthly_curve_reverse/CALIBRATION.json');obs=pd.read_csv(REBUILD/'DIGITIZED_WEALTH.csv');w=obs[obs.panel=='wealth'];axis=cal['wealth']
    nav=pd.read_csv(P/'CURVES.csv.gz',index_col=0,parse_dates=True);start=nav.index[0]
    xstart=axis['x0']+axis['width']*(start-START).days/(END-START).days
    target0=np.interp(xstart,w.x,w.log_wealth);dates=START+pd.to_timedelta((w.x-axis['x0'])/axis['width']*(END-START).days,unit='D');ok=dates>=start
    figure_wealth=np.exp(w.loc[ok,'log_wealth']-target0)
    fig,axes=plt.subplots(4,1,figsize=(13,15),gridspec_kw={'height_ratios':[1.2,1.2,1,.6]})
    purple='#9653a7';blue='#1675a9';orange='#d08736';green='#1b9b7a'
    for ax in axes[:2]:
        ax.plot(dates[ok],figure_wealth,color=purple,lw=2,label='Author monthly: digitized image (approximate)')
        ax.set_yscale('log');ax.set_ylabel('Wealth, initial = 1');ax.grid(alpha=.2)
    name='theme17__m252_42';axes[0].plot(nav.index,nav[name],color=blue,label=f"Best chronological rule of this study: {base[name]['cagr_252']:.1%} CAGR")
    execution=pd.read_csv(P/'EXECUTION_CONTROL_NAV.csv.gz',index_col=0,parse_dates=True);axes[0].plot(execution.index,execution['theme17_m252_42_lag1_cost10'],color=green,label='Previous-day signal + 10bps roundtrip: 41.9% CAGR')
    axes[0].set_title('1. Monthly selection rules: target not reproduced',loc='left',fontweight='bold');axes[0].legend(fontsize=9,loc='upper left')
    paths={}
    for tag,color in [('core17_q0',blue),('theme17_q0',orange),('theme17_q0.5',green)]:
        frame=pd.read_csv(P/f'INVERSE_{tag}_NAV.csv',index_col=0,parse_dates=True);s=frame.iloc[:,0];paths[tag]=s
        axes[1].plot(s.index,s,color=color,lw=.9,alpha=.8,label=f"Hindsight fit {tag}: {inv[tag]['cagr_252']:.1%}")
    axes[1].set_title('2. Similar curves from different candidate holdings — FUTURE PRICES USED',loc='left',fontweight='bold');axes[1].legend(fontsize=8,loc='upper left')
    mask=np.load(REBUILD/'purple_dd_mask.npy');yy,xx=np.where(mask);dt=START+pd.to_timedelta((xx-axis['x0'])/axis['width']*(END-START).days,unit='D');dd=(cal['dd']['y0']-yy)/cal['dd']['scale']
    axes[2].scatter(dt,dd,s=2,color=purple,alpha=.25,label='Author visible drawdown pixels')
    for tag,color in [('core17_q0',blue),('theme17_q0.5',green)]:
        s=paths[tag];axes[2].plot(s.index,s/s.cummax()-1,color=color,lw=.8,label=tag+' hindsight fit')
    axes[2].yaxis.set_major_formatter(PercentFormatter(1));axes[2].set_ylabel('Drawdown');axes[2].set_title('3. Drawdowns also remain different',loc='left',fontweight='bold');axes[2].legend(fontsize=8,loc='lower left');axes[2].grid(alpha=.2)
    dates=pd.to_datetime([r['month'] for r in ambiguity['rows']]);n=[len(r['intersection']) for r in ambiguity['rows']]
    axes[3].bar(dates,n,width=23,color=[green if k==3 else blue for k in n]);axes[3].set_yticks([0,1,2,3]);axes[3].set_ylabel('Common names');axes[3].set_title('4. Three fits agree on all 3 stocks in only 14 of 121 months',loc='left',fontweight='bold');axes[3].grid(axis='y',alpha=.2)
    fig.suptitle('Monthly reconstruction: candidate explanations, not verified author trades',fontsize=16,y=.987)
    fig.text(.07,.016,'2016-08-31 to 2026-09-24. Author image begins 2016-08-17. Assumed historical Top32 + 17 ETFs.\nFixed shares within months; same-close zero-cost reference except stated cost control. Reused sample, not independent OOS.\nHindsight paths cannot be traded or used as evidence that the author used future information.',fontsize=9)
    fig.tight_layout(rect=[0,.073,1,.97]);fig.savefig(P/'RECONSTRUCTION.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
