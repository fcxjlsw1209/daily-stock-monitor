"""Research delivery: keep image attribution and a selection rule separate."""
import json
import numpy as np
import pandas as pd
from study import P,O,I,read,dump,sha,pack,evaluate,load,START,END
from explain import explanations_for


def run():
    audit=read(P/'FIGURE_AUDIT.json');validation=read(P/'VERIFICATION.json');explain=read(P/'EXPLANATION.json')
    log=(P/'explain.log').read_text()
    assert log.rstrip().endswith('EXPLAIN_DONE') and 'RuntimeWarning' not in log and 'Traceback' not in log
    costs={}
    for pool in ['core17','theme17']:
        d=pack(pool);d['draw_refinement']=True
        for key in audit:
            if not key.startswith(pool+'__'):continue
            choice=np.load(P/'internal_states'/f'{key}_states.npz')['final']
            costs[key]=float(evaluate(choice,d,4.)[0][0])
    selected=min(costs,key=costs.get);a=audit[selected];m=a['metrics']
    choices=explanations_for(explain,selected)
    selected_rule=min(choices,key=lambda k:choices[k]['pairwise_loss'])
    er=explain[selected_rule]
    # Same image-and-risk objective for every final path, irrespective of optimization stage.
    dump('SELECTION.json',dict(selected_inverse_path=selected,comparison_objective='Same draw_v2 full-image loss and risk constraints, metric weight4; not largest return.',
        comparable_objectives=costs,selected_explanation_by_pairwise_loss=selected_rule,
        actual_author_holdings_recovered=False,author_algorithm_reproduced=False))
    rows=['月策略复现：从收益线拟合，转向全路径与风险特征的联合约束','2026-10-09','',
        '本轮结论：已找到更接近原图整体收益、Sharpe、最大回撤及QQQ相关性的事后归因路径；仍未找到能事前选出这些持仓的统一作者规则。没有发布合格潜在交易，没有加入观察账户。',
        '这不是新策略的80%回测：反推直接使用了原图和未来月份价格。它回答“现有价格是否可能解释原图”，不回答“当时能否选出来”。','',
        '一、为什么换这条研究路线',
        '作者月策略的中心思想和月度交易时钟保持不变。先把收益路径、相对QQQ路径、回撤发生时段、Sharpe和图中的日收益相关性0.712一起约束，再检验选股解释。',
        '旧的候选往往比原图平滑；图像线条重叠足够，并不代表复现了波动。新求解保留可见回撤竖线的极值，并防止把2026年夏季最大回撤替换成2020/2025年的一次相同幅度下跌。',
        '相邻两个月联合修改仅是求解器操作：用于同时调整峰值历史和之后的净值水平。实际假设仍是月底等权3只、月内固定股数，未增加调仓频率、杠杆、做空或组合相关性选股因子。','',
        '二、原图与最接近的完整路径',
        '范围2016-08-17—2026-09-24，零成本，同收盘代理成交；核心仍是历史代理Top32＋既有17ETF假设。',
        '指标 | 原图 | 本轮选出的事后路径',
        f"终值倍数 | 387.72 | {m['wealth']:.5f}",
        f"年化（2540净值点口径） | 80.64% | {m['cagr_nav_count']:.5%}",
        f"日Sharpe（含起始0收益假设） | 1.867 | {m['sharpe_initial_zero']:.6f}",
        f"最大回撤 | -34.46% | {m['max_dd']:.6%}",
        f"与QQQ日收益相关性 | 0.712 | {m['correlation_initial_zero']:.6f}",
        f"标准2539日收益口径年化 | 对照定义 | {m['cagr']:.5%}",
        f"标准2539日收益口径Sharpe | 对照定义 | {m['sharpe']:.6f}",
        f"路径标识：{selected}",
        '这些数字仍未全部落入原图显示数值的取整范围；不能说精确复现。图像提取误差和未知作者数据源也尚未消除。',
        f"宽松图像筛查中，净值、相对收益、回撤可见列不相交比例分别为{a['image']['wealth']['outside_fraction']:.2%}、{a['image']['relative']['outside_fraction']:.2%}、{a['image']['drawdown']['outside_fraction']:.2%}；这是排错诊断，不是统计置信度或自动准入。",'',
        '三、先核对指标定义，避免误判',
        '独立QQQ日收益计算同时符合原图终值6.78、年化20.92%、Sharpe0.958和最大回撤35.12%。',
        '但原图月策略387.72倍/80.64%和日策略2752.26倍/119.41%共同暗示2540个观测计数；QQQ暗示2539。初始0收益计入策略序列是合理的报表假设，尚无作者代码独立确认。',
        '因此并列报告标准收益条数口径和净值点口径。没有通过改定义删除失败，也没有把未验证的内部公式当成事实。','',
        '四、能否反推成统一规则',
        f"本轮实际测试{len(explain)}个解释：{len(audit)}条路径×3种核心（原始、风险调整、混合动量）。核心都跳过最近42个交易日；统一侧因子为波动、行业相对强弱、CMF、量比和距高点，跨全部月份保持同一组权重。",
        '权重通过持仓应排在非持仓之前的成对约束估计；核心权重50%，其余绝对权重之和不超过50%。这组具体比例和代理因子是可验证假设，不声称作者如此实现。',
        f"对上面的路径，成对排序误差最低的解释为{er['core_family']}；只在{er['exact_target_months']}/{er['total_months']}次事件选出完全相同的三只。",
        f"仍有{er['months_requiring_more_than_10_percent_full_score_range_correction']}/{er['total_months']}次事件，需要超过每只±0.05的事后分数调整，才可能把目标组合移到前三。这个上下调区间只是‘小修正’的操作性敏感性假设，并非作者披露参数。",
        '上述任意每只修正只是必要可行性上界，不能当作已经训练好的模型；没有将它用于生成交易或包装收益。',
        '这说明把某一条图像反推路径直接当作真实持仓，再强行拟合排序，仍然不够可靠。并不证明作者策略不存在，也不证明其他符合其中心思想的模型做不到。','',
        '五、后续最合理的方向',
        '本轮已进一步把同一套排名关系放回路径搜索，做了三次“统一排序拟合—连续路径调整”的联合求解；具体结果见下文。这比先固定一条事后交易表再要求模型硬记住它更合理，但当前实现仍未达成复现。',
        '优先检验在大涨、急跌、恢复这几个高信息月份同时成立的排名关系；仍只使用中期动量、波动、行业、量价及小幅模型修正。',
        '股票池代理和原作者17ETF清单仍未确认。发现数字相同的单月组合时，还需核对进入该月前的历史峰值；例如单月跌34.46%而月初已浮亏，会产生更深的全账户回撤，不能直接认定为作者交易。','',
        '六、验证与研究记录',
        f"{validation['full_paths']}条完整路径，保存{validation['actual_candidate_evaluations']:,}次单月/相邻月份候选比较；它们不是可交易策略。",
        f"独立按每月固定股数重建净值，最大相对差{validation['independent_nav_max_relative_error']:.3g}；收益矩算出的Sharpe、回撤和相关性与逐日重算一致。",
        f"{validation['protected_files_unchanged']}个保护文件及来源校验未变；原账户、冻结计划和规则未修改。",
        '记录包括最初缺少绘图依赖路径、输出摘要漏导入json和排序矩阵计算数值警告。均保留失败/警告日志；最后排序计算改为显式有界求和、有限值断言，重新执行通过，无RuntimeWarning。',
        '关键文件：PLAN*.json、各阶段PROGRESS/RESULT、internal_states、FIGURE_AUDIT.json、EXPLANATION.json、VERIFICATION.json、SELECTION.json、COMPARISON.png。',
        '所有中间组合保留为未准入的内部状态；没有导出为作者已确认交易或实盘建议。']
    if er['replay_metrics']:
        z=er['replay_metrics'];rows.insert(rows.index('五、后续最合理的方向')-1,
            f"这个解释规则本身重放为年化{z['cagr_252_nav_count']:.2%}、Sharpe {z['sharpe']:.3f}、回撤{z['max_dd']:.2%}，没有保留反推路径的约80%年化。")
    joint=read(P/'theme17__prior__draw_v2__blocks__joint_rule_RESULT.json')
    rows.insert(rows.index('六、验证与研究记录')-1,
        f"联合求解：需要大于±0.05分数修正的月份从{joint['trace'][0]['large_correction_months_before']}减少到{joint['trace'][-1]['large_correction_months_after']}/122；成对排序损失由{joint['trace'][0]['ranking_fit_loss']:.6f}降到{joint['trace'][-1]['ranking_fit_loss']:.6f}，但仍远未满足统一规则加小修正的解释要求。该路径仍只用于归因，没有变成新交易策略。")
    (P/'REPORT.txt').write_text('\n'.join(rows)+'\n')
    # Standard standalone research figure drawn from data; no modification to source bitmap.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    px,_,_=load();nav=pd.read_csv(P/f'{selected}_NAV.csv',index_col=0,parse_dates=True).iloc[:,0]
    q=px.loc[START:END,'QQQ'];q=q/q.iloc[0]
    obs=pd.read_csv(I/'DIGITIZED_WEALTH.csv');cal=read(I/'CALIBRATION.json')
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[2,1.2,1.2]})
    for panel,ax in [('wealth',axes[0]),('relative',axes[1])]:
        zz=obs[obs.panel==panel];dates=pd.to_datetime(zz.date)
        vals=np.exp(zz.log_wealth.to_numpy())
        if panel=='relative':
            nq=np.interp(dates.astype('int64'),q.index.astype('int64'),np.log(q))
            vals=vals/np.exp(nq)
        ax.scatter(dates,vals,s=3,c='#985e9c',alpha=.55,label='Digitized author image')
        ax.plot(nav.index,nav if panel=='wealth' else nav/q,lw=1.1,c='#007b88',label='Constrained hindsight path')
        ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].plot(q.index,q,color='#bd9732',lw=.9,label='QQQ');axes[0].legend(loc='upper left',fontsize=9)
    axes[0].set_ylabel('Wealth (log)');axes[1].set_ylabel('Wealth / QQQ (log)')
    ddobs=pd.read_csv(I/'purple_dd_pixels.csv');dt=START+pd.to_timedelta((ddobs.x-cal['wealth']['x0'])/cal['wealth']['width']*(END-START).days,unit='D')
    axes[2].vlines(dt,100*(cal['dd']['y0']-ddobs.ymax)/cal['dd']['scale'],100*(cal['dd']['y0']-ddobs.ymin)/cal['dd']['scale'],color='#985e9c',alpha=.35,lw=1)
    axes[2].plot(nav.index,100*(nav/nav.cummax()-1),c='#007b88',lw=.9)
    axes[2].set_ylabel('Drawdown (%)');axes[2].grid(alpha=.2);axes[2].set_ylim(-42,2)
    axes[2].xaxis.set_major_locator(mdates.YearLocator());axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('Monthly reconstruction: image-constrained attribution, NOT a trading strategy',fontsize=15)
    fig.text(.5,.025,f"2016-08-17 to 2026-09-24 | CAGR {m['cagr_nav_count']:.2%} | Sharpe {m['sharpe_initial_zero']:.3f} | MaxDD {m['max_dd']:.2%} | Uses future prices; no rule recovered",ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.965]);fig.savefig(P/'COMPARISON.png',dpi=150);plt.close(fig)
    dump('CODE_HASHES.json',{str(f):sha(f) for f in P.glob('*.py')})
    print(json.dumps(dict(selected=selected,metrics=m,selected_explanation=selected_rule,rule_metrics=er['replay_metrics'],author_rule_reproduced=False),indent=2))


if __name__=='__main__':run()
