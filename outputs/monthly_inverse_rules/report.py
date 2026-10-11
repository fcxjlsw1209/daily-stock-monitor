"""Publish actual inverse-inference results, alternative holdings and limitations."""
import json
import numpy as np
import pandas as pd
from common import P,O,base,START,END,dump
from inverse import load_observations


def read(name):return json.loads((P/name).read_text())
def names(items):return '/'.join(s.replace('research_equity_','') for s in items)


def run():
    raw=read('RESULTS.json');polish=read('POLISH_RESULTS.json');sel=read('POLISH_SELECTION.json');audit=read('STRICT_MATCH_AUDIT.json');valid=read('VALIDATION.json');execution=read('EXECUTION_RESULTS.json')
    best=sel['best_full_cagr_retrospective'];tag=best.split('__lag')[0];model=read('POLISH_MODELS.json')[tag]
    strictbest=max(audit,key=lambda k:(audit[k]['summary']['all']['strict_anchor_hits'],audit[k]['summary']['all']['strict_hits']))
    assert strictbest in raw
    core=sel['selected_by_early_curve_fit']['core17'];theme=sel['selected_by_early_curve_fit']['theme17']
    metrics=[('原图月策略（作者标注）',dict(cagr_252=.8064,sharpe=1.867,max_dd=-.3446)),
             ('本轮持仓线索命中最多的固定规则（事后比较）',raw[strictbest]),
             ('只用2016—2021曲线校准并选型：原ETF池',polish[core]),
             ('只用2016—2021曲线校准并选型：主题ETF池',polish[theme]),
             ('全样本反推固定规则：本轮最高收益',polish[best]),
             ('同一全样本规则，同日信号＋0.10%双边成本',execution[tag+'__lag0_cost10']),
             ('同一全样本规则，提前一天信号＋0.10%双边成本',execution[tag+'__lag1_cost10'])]
    lines=['月度持仓候选 → 统一选股规则：反推研究结果','2026-10-09','',
        '结论：按用户指定路线，先从月份曲线枚举候选组合，再从多组候选中学习跨月份的固定评分规则。本轮最高全样本反推年化63.46%、Sharpe1.704、最大回撤34.01%；没有达到作者80.64%，也没有复现全部交易。',
        '收益接近与持仓复现是两个不同问题。该高收益规则严格匹配7/121个月的高拟合权重候选集合；22个较集中月份中匹配3个。不能把63.46%的曲线拟合结果包装成作者模型已找出。',
        '没有将某月月底才能知道的排序用在该月月初。全部实际规则在上月底选股，供下一月持有；本研究接受作者也遵守这一时间顺序的约束。','',
        '一、结果（CAGR / Sharpe / 最大回撤）',
        '本地回放2016-08-31—2026-09-24；原图2016-08-17开始，因此未猜测首个不完整月份。Sharpe使用日收益、252日年化、样本标准差、无风险利率0；作者的完整Sharpe公式未提供。',
        '方法 | 全期年化 | Sharpe | 最大回撤']
    for label,z in metrics:lines.append(f"{label} | {z['cagr_252']:.2%} | {z['sharpe']:.3f} | {z['max_dd']:.2%}")
    lines+=['',
        '同一高收益模型的执行敏感性：',
        '信号日 | 双边合计成本 | 年化 | Sharpe | 最大回撤']
    for lag,cost in [(0,0),(0,10),(1,0),(1,10)]:
        z=execution[tag+f'__lag{lag}_cost{cost}'];lines.append(f"{'月末收盘' if lag==0 else '月末前一交易日收盘'} | {cost/100:.2f}% | {z['cagr_252']:.2%} | {z['sharpe']:.3f} | {z['max_dd']:.2%}")
    lines+=['主要差别来自信号时点，非交易费。提前一天仍在月底收盘成交，并在随后月份持有；这个对照不表示作者使用未来月份排名。',
        f"全样本最高收益模型分段：2016—2021年化{polish[best]['early']['cagr_252']:.2%}、Sharpe{polish[best]['early']['sharpe']:.3f}；2022以后年化{polish[best]['later_diagnostic']['cagr_252']:.2%}、Sharpe{polish[best]['later_diagnostic']['sharpe']:.3f}。后段已参与系数反推，不能把94%的后段年化当独立验证。",'',
        '二、从持仓线索出发做了什么',
        '1. 对每个可观察月份（末月截至2026-09-24），在当时合格候选中枚举所有三只等权、月内固定股数的组合，匹配原图财富、相对QQQ和回撤三幅图；不预设作者真的持有某一组。',
        '2. 每月保留64组组合，横轴偏移−0.2/0/+0.2像素做稳定性检查。候选权重取决于图像拟合误差。原ETF池有24个候选较集中的月份，主题ETF池有22个。所谓集中度是启发式熵指标，不是统计置信概率。',
        '3. 第一阶段学习“每月选到合理备选中的任意一组”的统一线性评分；没有把多组互斥组合平均成虚构的作者持仓。保留2016—2021校准和全样本校准两类。',
        '4. 第二阶段要求同一个评分向量解释连续复利净值与回撤，不允许每月把净值重置到原图；第三阶段仅细化同一参数空间。优化损失为全局log净值RMSE＋4倍月内中心化形状RMSE＋0.5倍月度最差回撤RMSE，没有以年化或Sharpe直接作优化目标。',
        '5. 单独比较3棵深度2、合计10%评分修正的小树模型；没有改善最高收益结果，全部失败/较弱假设也保留。',
        '6. 额外统计前95%拟合权重集合中的严格命中，区分“排进64个宽泛候选”和“真的较像图”。95%只是候选权重累计阈值，不是95%真实持仓概率或置信区间。','',
        '三、可复核的月份例子（主题ETF池）',
        '月份 | 图像前三组候选 | 固定高收益规则选择 | 严格命中']
    bestrows=audit['POLISH_'+best]['months'];lookup={q['month']:q for q in bestrows}
    for month in ['2020-08','2025-09','2026-04','2026-05','2026-06']:
        z=lookup[month];lines.append(f"{month} | {'；'.join(names(c['holdings']) for c in z['candidate_groups'])} | {names(z['holdings'])} | {'是' if z['strict_fit_weight_hit'] else '否'}")
    lines+=['2026年5月尤其有用：INTC/MU在多个高拟合候选中反复出现，而高收益规则选JNJ/CAT/MU。这指出了具体待解释的排序差异；不是靠继续抬升总收益即可解决。',
        f"事后按严格锚点命中最多找到的规则为{strictbest}，在原ETF池24个集中月份中匹配10个，121个月匹配19个；该规则年化{raw[strictbest]['cagr_252']:.2%}，Sharpe{raw[strictbest]['sharpe']:.3f}。它也未复现作者。",'',
        '四、63.46%反推规则的完整定义',
        '资格：上月底历史股票池中，R12−2为正且百分位在前半；已有253个连续有效价格观测。选评分前三，只做多；只剩两只时各半，少于两只空仓。当前样本所有调仓均有至少3个候选。',
        'R12−2=P[t−42]/P[t−252]−1；R6−2=P[t−42]/P[t−126]−1；risk=R12−2/过去63日收益标准差。先按当时股票池分别做百分位排名。',
        '核心得分=25%rank(R12−2)+5%rank(R6−2)+20%rank(risk)，再加下列固定修正系数×对应中心化百分位(rank−0.5)。全部月份共用同一组系数，无按日期定制规则。',
        '可选次级特征缺失时取中性百分位。市场交互为中心化vol或near252乘SPY距200日均线的百分比（截到±20%，除以20%）。',
        '因子 | 系数']
    for feature,weight in model['coefficients'].items():lines.append(f'{feature} | {weight:.10f}')
    lines+=['特征定义在common.py及复用的raw_features中。downside本身是负的下行波动率；不要把其正系数误读为追求下跌风险。CMF等系数是图像反推拟合值，不能据此认定作者确实使用这个方向。',
        '自由修正系数绝对值和不超过0.5。部分次级因子仍包含短期动量信息，不能把核心50%解释成经济风险暴露上限。月末三只等权，之后一个月固定复权单位；没有月内止盈止损，没有每日换仓。',
        '股票池：历史Nasdaq-100与S&P100并集内的市值代理Top32＋17只ETF。股票池不是已经认证的历史全美股Top32，ETF名单也并非作者已公开确认名单。',
        '主题ETF17只：SPY,QQQ,IWM,TLT,GLD,XLE,XLF,XLI,XLK,XLV,XLY,SMH,SOXX,IGV,VGT,MTUM,ARKK。本轮未使用杠杆ETF或融资。',
        '原17只：SPY,QQQ,IWM,DIA,TLT,GLD,XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY。',
        '局部候选枚举仅为归因分析而要求整月价格可见；正式规则回放的入场资格只用当时已有价格，若持仓路径缺价格则直接报错。所有同日信号反推组合已与实际规则回放的持仓、复利终值逐月核对。','',
        '五、验证与研究状态',
        f"完成{valid['completed_replays']}个策略回放、{valid['monthly_orders_checked']}次月度事件；另有12个执行对照（其中6个与既有情形重复）。逐笔重算固定股数及差额交易费，净值最大相对误差{valid['independent_nav_max_relative_error']:.3g}；CAGR、Sharpe与回撤另行重算通过。",
        f"242个月份候选缓存校验通过；将2020-07-01以后的价格和成员数据物理截断，截点前所有输入因子和资格结果差异为{valid['feature_prefix_max_difference']:.1f}。这证明特征时序一致，不代表用整张图拟合的系数有历史可得性。",
        f"概率分区/梯度/L1投影3项数学单元测试通过。28个曲线拟合模型保留了{valid['saved_coefficient_trials']:,}组实际评估的系数向量，另有第一阶段12个模型各3次梯度拟合重启。试过的参数多，必须将全样本结果视为研究拟合，不能当成新的样本外证据。",
        '研究开发时，小树扩展的净值一致性断言曾捕获统计切片共享导致的净值修改；已修复为独立复制、重新运行全部扩展，并通过独立会计重算。失败日志保留。',
        f"{valid['protected_files_unchanged']}个原账户、冻结计划和保护文件未变；{valid['source_hashes_verified']}个原始数据来源哈希核验通过。没有把新策略加入观察账户，没有自动交易。",
        '仍缺的证据：作者准确17只ETF名单、原全市场Top32历史排名、足够分辨率的逐日净值或持仓。当前能产生可计算、可回放的候选规则；无法据此宣称该规则就是作者模型。',
        '当前目标状态：80.64%月策略未复现。保留63.46%作为全样本反推候选；更接近实际执行的同规则45.20%只作为敏感性结果，不升级正式账户。','',
        '文件：COMPARISON.png、MONTHLY_CANDIDATES.txt、STRICT_MATCH_AUDIT.json、POLISH_MODELS.json、POLISH_ORDERS.json、POLISH_CURVES.csv.gz、EXECUTION_RESULTS.json、VALIDATION.json；其余阶段配置/结果/系数档案全部保留。',
        '运行顺序：prepare.py → fit.py → coherent_fit.py → polish.py → tree_correction.py → validate.py → execution.py → report.py。统一使用已有work/venv/bin/python，PYTHONPATH=work/research-plot-deps，OPENBLAS_NUM_THREADS=1；绘图MPLCONFIGDIR=/private/tmp/daily-rebuild-mpl。']
    (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    detailed=['逐月候选与固定规则持仓对照','候选均是图像反推假设，不是已确认作者交易。权重为拟合损失转换，不是真实持仓概率。','',f'对照规则：{best}（全样本拟合）。','']
    for pool in ['core17','theme17']:
        detailed+=[pool,'月份 | 集中度 | 候选1 | 候选2 | 候选3 | 稳定候选名字 | 高收益规则（三只）']
        for m in read(f'BAGS_{pool}.json'):
            sets=[f"{names(q['holdings'])}（拟合权重{q['image_weight']:.1%}）" for q in m['candidate_triples'][:3]]
            held=names(lookup[m['month']]['holdings']) if pool=='theme17' else '见原池规则ORDERS文件'
            detailed.append(f"{m['month']} | {m['confidence']:.3f}{'*' if m['anchor'] else ''} | {' | '.join(sets)} | {names(m['stable_best_names']) or '无'} | {held}")
        detailed+=['*表示启发式集中度≥0.65，不表示已确认。','']
    (P/'MONTHLY_CANDIDATES.txt').write_text('\n'.join(detailed)+'\n')
    plot(best,core,polish,execution)
    dump('DELIVERY.json',dict(best_fullsample_rule=best,cagr=polish[best]['cagr_252'],sharpe=polish[best]['sharpe'],max_dd=polish[best]['max_dd'],
        strict_candidate_matches=audit['POLISH_'+best]['summary'],development_selected_rules=sel['selected_by_early_curve_fit'],
        author_reproduced=False,independent_oos=False,original_accounts_unchanged=True))
    dump('CODE_HASHES.json',{p.name:base.sha(p) for p in P.glob('*.py')})
    print(json.dumps(read('DELIVERY.json'),indent=2))


def plot(best,early,results,execution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    curves=pd.read_csv(P/'POLISH_CURVES.csv.gz',index_col=0,parse_dates=True);exe=pd.read_csv(P/'EXECUTION_CURVES.csv.gz',index_col=0,parse_dates=True)
    obs=load_observations();cal=json.loads((O/'monthly_curve_reverse/CALIBRATION.json').read_text());axis=cal['wealth'];first=curves.index[0]
    x0=axis['x0']+axis['width']*(first-START).days/(END-START).days;ob=obs['wealth'];ob=ob[(ob.x>=x0)&(ob.x<=axis['x0']+axis['width'])]
    dates=START+pd.to_timedelta((ob.x-axis['x0'])/axis['width']*(END-START).days,unit='D');author=np.exp(ob.log_wealth.to_numpy()-np.interp(x0,obs['wealth'].x,obs['wealth'].log_wealth))
    px,_,_=base.load();qqq=px.QQQ.reindex(curves.index);qqq=qqq/qqq.iloc[0]
    tag=best.split('__lag')[0];delayed=tag+'__lag1_cost10'
    fig,axes=plt.subplots(2,1,figsize=(12,8.6),sharex=True,gridspec_kw={'height_ratios':[1.7,1]})
    axes[0].plot(dates,author,color='#9d5cb0',lw=1.7,alpha=.8,label='Author monthly: digitized image (reported CAGR 80.64%, Sharpe 1.867)')
    specs=[(curves[best],'Full-image fixed-rule fit','#156c99',results[best]),(exe[delayed],'Same fit: prior-session signal + fees','#35856f',execution[delayed]),(curves[early],'Rule chosen using 2016-2021 only','#a76c43',results[early])]
    for nav,label,color,z in specs:
        axes[0].plot(nav.index,nav,color=color,lw=1.35,label=f"{label}: CAGR {z['cagr_252']:.1%}, Sharpe {z['sharpe']:.2f}")
        axes[1].plot(nav.index,nav/np.maximum(1,nav.cummax())-1,color=color,lw=.95,label=f"{label}: MaxDD {z['max_dd']:.1%}")
    axes[0].plot(qqq.index,qqq,color='#a49643',lw=1,label='QQQ buy-and-hold, normalized to same start')
    axes[1].axhline(-.3446,color='#9d5cb0',ls=':',lw=1,label='Author reported MaxDD -34.46%')
    for ax in axes:
        ax.axvline(pd.Timestamp('2022-01-01'),color='#555',ls='--',lw=.8);ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left' if ax is axes[0] else 'lower left')
    axes[0].set_yscale('log');axes[0].set_ylabel('Compounded wealth, initial =1');axes[1].set_ylabel('Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Monthly inverse inference: closer risk-adjusted returns, holdings still do not match',fontsize=14)
    axes[0].set_title('2016-08-31 to 2026-09-24. One scoring rule; equal entry allocations, then fixed shares. No leveraged ETFs.',fontsize=10,loc='left')
    fig.text(.075,.017,'Full-image fit uses retrospective curve labels; it is not independent out-of-sample performance.\nAuthor image starts 2016-08-17; rebased after the incomplete first month. ETF lists and historical Top32 are research proxies.',fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(P/'COMPARISON.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
