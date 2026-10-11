import json
import numpy as np
import pandas as pd
from research import P,OLD,base,dump


def main():
    r=json.loads((P/'RESULTS.json').read_text());s=json.loads((P/'SELECTION.json').read_text());v=json.loads((P/'VALIDATION.json').read_text())
    conditional='core17__ridge_strong_half__strong__keep5';unconditional='core17__ridge_all_positive__strong__keep5';benchmark='core17__R12_2_baseline'
    comparisons=[('原简单动量，同一原ETF池',benchmark),('Ridge：所有正动量标的训练',unconditional),
                 ('Ridge：强势前半标的训练',conditional),('按开发段Sharpe选出的新规则',s['development_selected']),
                 ('主题ETF池的简单动量','theme17__R12_2_baseline')]
    lines=['从候选组合到可执行月策略：2026-10-09研究记录','',
           '回答：可以提炼策略，不需要先唯一恢复作者每笔交易。本轮把候选组合里的线索写成明确因子、资格条件和换仓规则，并实际回放。能形成可执行假设，不等于已经证明收益更好。',
           '候选组合仅用来产生“强者中选择、不必过度追涨”的假设；Ridge训练没有使用任何反推持仓标签，只使用在当时已经结束的历史月收益。','',
           '一、已提炼的规则',
           '股票池：现有历史Nasdaq100与S&P100并集内的市值代理Top32，加原17只无杠杆ETF；另用主题ETF组合验证敏感性。它不是已完全认证的历史全美股Top32，也不是已确认的作者ETF名单。',
           '资格：R12-2为正、在当时股票池动量排名前半、至少253个连续价格观测。月策略不要求先发生短期回撤。另比较最近63日也上涨的门槛。',
           '评分：40%R12-2＋10%R6-2＋20%条件Ridge预测排名＋10%价格不过度延伸＋10%相对行业表现＋5%CMF21＋5%63日波动率。显式动量分占50%；模型内部也含动量交互，不把此数解释成经济风险暴露上限。',
           'R12-2=P[t-42]/P[t-252]-1；R6-2=P[t-42]/P[t-126]-1。每项先做当日股票池百分位排名。',
           '价格不过度延伸：负的“价格/252日最高价”和负的“价格/200日均线-1”各占一半排名；在强势标的之间适度奖励未过度拉伸，不要求跌得越多越买。',
           '相对行业表现使用缓存sector60；ETF以SPY作相对基准。相对强度、CMF、量比缺失时取中性百分位，并在模型保留缺失指示；不因这些可选特征缺失而默默删除整只股票。',
           'Ridge：最近60个已结束月份，36个月半衰期，正则100；在历史合格强势标的中学习下一月相对SPY的对数收益，固定少量交互特征。每次拟合时所有训练收益区间均已结束。',
           '持仓与退出：只做多，三只等权；只有两只合格则各半，少于两只空仓。已有持仓仍合格且在总分前五则保留，再按排名补到三只；掉出资格或前五，于月末换出。月内固定复权价格单位，不按日排名换股，也无月内止盈止损。',
           '执行：月末前一个交易日收盘形成信号，月末收盘价代理成交；买卖各0.05%，差额调仓。这个时点留出了下达收盘指令的时间，仍是回放成交假设。','',
           '二、同口径实际回放',
           '全期2016-08-31—2026-09-24；后段为2021-12-31收盘到2026-09-24。净值已含双边合计0.10%成本。所有账户月内固定股数、无杠杆。',
           '方法 | 全期年化 | 全期最大回撤 | 2016—2021年化 | 2022年以来年化']
    for label,key in comparisons:
        z=r[key];lines.append(f"{label} | {z['cagr_252']:.2%} | {z['max_dd']:.2%} | {z['development']['cagr_252']:.2%} | {z['later_diagnostic']['cagr_252']:.2%}")
    a,b,c=r[conditional],r[unconditional],r[benchmark]
    lines+=['',f"训练范围缩到强势候选，其他规则相同，全期净年化从{b['cagr_252']:.2%}提高到{a['cagr_252']:.2%}；但回撤从{b['max_dd']:.2%}扩大到{a['max_dd']:.2%}。不能只看收益改善。",
            f"条件Ridge后段年化{a['later_diagnostic']['cagr_252']:.2%}，同期基线{c['later_diagnostic']['cagr_252']:.2%}；前段条件Ridge明显更弱，导致全期仍未超过简单动量。",
            f"全期收益最高的新假设是{s['best_full_cagr_retrospective']}，它不是仅凭开发段能选出的赢家。事先规定的开发段Sharpe选择得到{s['development_selected']}；该规则后段表现也没有胜过基线。",
            '结论：可以从共同思路提炼出规则，也找到“先筛强势、再学排序”这一可研究的改进方向；本轮没有得到足以替换现有基线、或复现作者80.64%全期年化的策略。','',
            '三、完整试验和验证',
            '共42例：两个ETF池，各5个评分体系×2个资格条件×2个留仓规则，另各有一个简单动量基线。评分体系为量价控制、适度奖励价格未延伸、按SPY趋势调整波动偏好、全体正动量训练Ridge、强势前半训练Ridge。参数与选择准则在运行前保存于CONFIG.json；没有只保留赢家。',
            '开发段按2016—2021净Sharpe选择；2022以后单独报告。这些年代此前已经被研究过，不是独立未见过的样本外测试，结果也受历史股票池代理的限制。',
            f"验证通过：{v['mature_label_fits']}次模型拟合的标签都先于下单时点结束；{v['calendar_label_rows_checked']}个历史月标签与实际月末收益核对；42例各121次调仓的日期与持仓数量通过。",
            f"截断2020-07-01之后数据并重训后，截点前预测最大差异{v['prediction_prefix_max_difference']:.3g}，缺失分布相同；数值运算警告修复前后持仓完全一致，统计最大差异{v['numerical_fix_metric_max_difference']:.3g}。",
            f"{v['protected_files_unchanged']}个原账户保护文件未变；{v['source_hashes_verified']}个数据来源校验通过。没有把新假设加入观察账户，没有自动下单。",'',
            '文件：CONFIG.json、RESULTS.json、SELECTION.json、ORDERS.json、CURVES.csv.gz、TRAINING_AUDIT.json、VALIDATION.json、COMPARISON.png。',
            '复跑：PYTHONPATH=work/research-plot-deps OPENBLAS_NUM_THREADS=1 work/venv/bin/python outputs/monthly_consensus_strategy/research.py；随后validate.py、report.py。绘图设置MPLCONFIGDIR=/private/tmp/daily-rebuild-mpl。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    plot(r,conditional,unconditional,benchmark)
    dump('CODE_HASHES.json',{p.name:base.sha(p) for p in P.glob('*.py')})
    print(json.dumps(dict(conditional_ridge=a['cagr_252'],same_pool_baseline=c['cagr_252'],cases=len(r),target_achieved=False),indent=2))


def plot(results,conditional,unconditional,benchmark):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    nav=pd.read_csv(P/'CURVES.csv.gz',index_col=0,parse_dates=True)
    labels={benchmark:'R12-2 baseline',unconditional:'Ridge trained on all positive-momentum names',conditional:'Ridge trained on stronger half'}
    colors={benchmark:'#9c7c22',unconditional:'#6c8bac',conditional:'#147971'}
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[1.6,1]})
    for key in labels:
        x=nav[key];m=results[key]
        axes[0].plot(x.index,x,color=colors[key],lw=1.4,label=f"{labels[key]}: {m['cagr_252']:.1%} CAGR")
        axes[1].plot(x.index,x/np.maximum(1,x.cummax())-1,color=colors[key],lw=.9,label=f"{labels[key]}: {m['max_dd']:.1%} max drawdown")
    for ax in axes:
        ax.axvline(pd.Timestamp('2022-01-01'),color='#555',ls='--',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left' if ax is axes[0] else 'lower left')
    axes[0].set_yscale('log');axes[0].set_ylabel('Wealth after costs, initial=1')
    axes[1].set_ylabel('Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Monthly strategy hypotheses: learn to rank within strong candidates',fontsize=14)
    axes[0].set_title('Same historical pool, 2016-08-31 to 2026-09-24. Later-period improvement does not imply full-period superiority.',fontsize=9,loc='left')
    fig.text(.075,.015,'Prior-session signals, month-end close fills, 10bps roundtrip costs; equal initial positions, fixed shares between rebalances.\nHistorical pool proxy, reused research sample, not independent OOS. No inferred author holdings used as model labels.',fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.945]);fig.savefig(P/'COMPARISON.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
