from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
m=pd.read_csv(P/'metrics.csv');diag=pd.read_csv(P/'diagnostic_summary.csv');y=pd.read_csv(P/'yearly.csv');ci=pd.read_csv(P/'paired_daily_return_intervals.csv')
names={'baseline_common':'原有R12–1加权模型','high_proximity':'加入接近一年高点','low_downside':'加入低下行波动','quiet_volume':'加入缩量回调','range_contraction':'加入振幅收窄','accumulation':'加入持续价量配合（CMF）','beta_adjusted_strength':'加入Beta调整强度','blend_all':'六个新因子一起加入'}
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def metric(k,cost=5,split='all'):
    return m[(m.strategy==k)&(m.cost_bps==cost)&(m.split==split)].iloc[0]
def percent(x):return f'{100*x:.1f}%'
main=table(['版本','交易数','净胜率','年化收益','最大回撤','平均仓位'],[[names[k],int(metric(k).closed_trades),percent(metric(k).trade_win),percent(metric(k).cagr),percent(-metric(k).max_drawdown),percent(metric(k).average_exposure)] for k in names])
splits=table(['版本','2019–2021年化','2022–至今年化','早期胜率','近期胜率'],[[names[k],percent(metric(k,split='earlier').cagr),percent(metric(k,split='recent').cagr),percent(metric(k,split='earlier').trade_win),percent(metric(k,split='recent').trade_win)] for k in names])
costs=table(['版本','5bps年化','10bps年化','20bps年化','20bps最大回撤'],[[names[k]]+[percent(metric(k,cost=c).cagr) for c in [5,10,20]]+[percent(-metric(k,cost=20).max_drawdown)] for k in names])
diagnostic=[]
for k in list(names)[1:-1]:
    r=diag[(diag.factor==k)&(diag.horizon==5)&(diag.split=='all')].iloc[0]
    diagnostic.append([names[k].removeprefix('加入'),f'{r.mean_ic:+.3f}',f'{r.mean_partial_ic:+.3f}',f'{100*r.spread:+.2f}个百分点',f'[{100*r.spread_bootstrap_low:+.2f}, {100*r.spread_bootstrap_high:+.2f}]个百分点'])
diag_table=table(['因子','原始排序相关','控制旧因子后的相关','高分减低分的5日收益差','收益差探索性95%区间'],diagnostic)
intervals=table(['版本','平均每日增量收益','探索性95%区间'],[[names[r.strategy],f'{r.mean_daily_excess*10000:+.2f}bps',f'[{r.bootstrap_low*10000:+.2f}, {r.bootstrap_high*10000:+.2f}]bps'] for r in ci.itertuples() if r.strategy in names and r.strategy!='baseline_common'])
years=table(['年份','原加权模型','加入CMF','加入缩量回调','六项一起加入'],[[str(year)+('（截至9/18）' if year==2026 else '')]+[percent(y[(y.strategy==k)&(y.cost_bps==5)&(y.year==year)].iloc[0].total_return) for k in ['baseline_common','accumulation','quiet_volume','blend_all']] for year in range(2019,2027)])
report=f'''# 新增个股因子：想法与增量测试

## 结论

六个候选因子已按预先固定的方向、窗口和权重完成测试。与偏好“胜率较高、收益适中”最接近的后续研究候选是**持续价量配合（21日CMF）**：本轮胜率略升、回撤略降。缩量回调的历史收益提升较大，但回撤更深，属于另一种取舍。两者都尚未确立可靠的增量优势。

**没有一个新增版本的平均每日增量收益，其探索性95%区间能排除零。**不能因为从六个因子中挑到一个好看的结果，就认定可以稳定获利。本轮未据此重新改权、反转弱因子的方向或继续搜索窗口。

## 先想哪些因子，为什么测试这些

| 候选思路 | 想解决的问题 | 本轮实现 |
|---|---|---|
| 接近一年高点 | 长期涨幅相同，距离历史高点是否有额外信息 | 复权收盘价÷过去252日最高复权收盘价 |
| 下行风险较低 | 避开下跌日波动过大的股票 | 63日日收益负值平方均值的平方根，取负数 |
| 回调缩量 | 相同回调幅度，成交量是否能区分回调特征 | 最近3日均量÷此前20日均量，取负数 |
| 振幅收窄 | 价格波动趋于收敛是否改善短期入场 | 最近5日平均相对振幅÷此前20日，取负数 |
| 持续价量配合 | 单根K线收盘位置较弱，连续多日是否有帮助 | 21日成交量加权的收盘位置（CMF） |
| 扣除市场影响的强度 | 去掉跟随大盘上涨的部分，是否更能识别个股优势 | 以前126日估计Beta、逐日扣除SPY影响，再汇总60日 |

“接近52周高点”有对应学术研究，但这里使用252日复权收盘高点近似，并非重现论文的组合或持有期。[George与Hwang论文](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x)

CMF把一段时间内的收盘位置按成交量加权；它是价量代理，不能识别实际机构资金流入。[Fidelity公式说明](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmf)

市场调整强度受残差动量思路启发，但本轮只扣除滞后估计的市场Beta，不扣截距、不使用多因子风险模型，因此不能当作复制某家机构的方法。[Robeco动量概述](https://www.robeco.com/en-int/glossary/quantitative-investing/momentum-factor)

还值得考虑但本轮不测试的有：盈利预期上修、财报意外及漂移、利润质量/应计项目、估值、空头与期权信息、新闻、全行业广度。这些需要历史当时可知的数据。现有数据库只有价量和分红等记录，不能拿今天的基本面或行业成员回填历史。短期策略也需要关注成本；本轮继续报告成本情景而不声称有真实成交成本数据。[交易成本研究](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies)

## 公平比较的方法

基准是上一轮的宽筛选加权模型，不是更早的简单回调策略。R12–1固定占30%，不加入R12。每个新因子占10%，其余旧因子的合计权重由70%按比例减至60%。

`新分数 = 0.30×R12–1百分位 + (6/7)×(旧分数−0.30×R12–1百分位) + 0.10×新因子百分位`

六项合并版本把这10%均分，每项约1.67%，没有事后只组合表现最好的因子。新增因子只改变排序，保留上一轮原始0.65入围门槛；不会因为新增后的分数低于0.65就取消资格。

所有模型拥有完全相同的13,844个候选股票日、427个无新候选日。共同数据检查没有减少基准候选，两个基准逐日权益相同；平均仓位均约79%，因此这次并非通过放宽筛选或明显加大仓位取得差异。换序仍会改变成交、行业占用和后续可买机会，不等同于逐笔独立加减因子。

样本2019-01-02至2026-09-18；当前美股上市市值前100家公司；起始资金10万美元。上一收盘信号，下一开盘买入，五个交易日后开盘卖出；最多同时5只、每行业2只，每笔预算20%资金，无杠杆，无信号留现金。分红、成本、期末持仓计入权益；没有增加止损、财报过滤或盘中择时。

## 组合结果

买卖合计成本5bps（0.05%），回撤按每日收盘净值计算：

{main}

CMF把胜率从约54.0%提升到54.8%，最大回撤从28.3%降到25.0%，只是约0.8个百分点的胜率差，并非高胜率策略已经建立。缩量回调年化由27.8%升至35.6%，最大回撤同时升至31.9%。把六个新因子全加进去，胜率降至53.3%，收益和回撤也未优于基准。

低下行波动的组合结果看起来稍好，但下面的单因子诊断并不支持它是更强的短期收益信号。排序、行业限制、持仓路径与风险变化会产生不同组合结果，不能只挑有利的一张表。

## 是否真的增加了不同的信息

在原宽筛选范围、综合分数门槛之前做单因子诊断，每天至少15只候选。5日标签有1007个可用日期；高低组为当日因子前20%与后20%，收益按次日开盘入场、五日后开盘退出计算，包含分红和5bps成本。

{diag_table}

“控制旧因子后的相关”是在同日股票中，对新因子和未来收益的排名分别剔除五项旧因子排名后计算的相关。它仅用于诊断，未来收益从未参与交易分数。相关并非因果关系，分组收益差也不是可复利的可交易组合。

具体看：

- **CMF**与旧的单日收盘位置相关约0.13，不是简单复制；但控制旧因子后的5日相关只有约0.006，信号仍弱。
- **缩量回调**与R12–1相关约0.02，与旧回调因子约-0.16，提供了较不同的描述；但5日高低组差的区间仍跨零。
- **市场Beta调整强度**原始表现看似正向，控制旧因子后转负；它与旧趋势稳定性的相关约0.92，重复度很高，本轮不支持叠加。
- **接近一年高点**在当前强势回调、5日持有的场景中没有表现出正向收益信息，不据此否定其他持有期或市场中的相关研究。
- **低下行波动**的5日高低组收益差为负。较低风险并不必然意味着这种短线策略的收益更高。

## 不确定性、分段与成本

下表比较每个组合相对同池基准的平均每日收益差，单位bps（1bps=0.01%），不是年化收益差：

{intervals}

区间采用20个交易日为一块、1000次重采样，固定种子20260918，以保留一定的时间相关；因子诊断保留完整交易日历中的无信号日。区间没有对多个假设做校正，也不能解决当前成分股偏差或市场阶段变化。全部区间跨零，因此不能把本轮最好结果叫作已验证的优势。

分段历史：

{splits}

这些历史段都已经用于探索，不能称为新的独立样本外测试。缩量回调也并非每年改善：2021年为负收益，而基准为正。完整年度结果如下，2026只是截至9月18日：

{years}

提高往返成本后的结果：

{costs}

在20bps成本下，CMF回撤约27.0%，缩量回调回撤约38.2%。成本是假设，未模拟真实开盘执行冲击或订单延迟。

## 当前结论与限制

若按“提高胜率、适中收益”的原始偏好安排后续研究，CMF比缩量回调更契合；缩量回调可作为偏收益的候选另行观察。当前仍不升级权重、不自动合并两者、不宣称达到高胜率。新增因子越多，越容易在同一份历史里碰到偶然赢家。

市值前100和行业映射是当前快照，缺少历史成分股、退市股及完整逐时点数据，因此存在幸存者和前视选择偏差。这轮27.8%–35.6%的年化不是未来收益预期。没有计税、ADR预提税、现金利息和分红实际到账延迟；只按日终计回撤，盘中风险可能更大。价格和成交量来自已下载Yahoo数据，未独立验证每家公司身份沿革与所有历史调整。

## 验证与复现

9项自动测试通过：六因子公式及互不重叠的比较窗口；Beta与直接OLS一致且正确滞后；R12–1权重和候选范围固定；截断未来数据不改变历史因子/信号；缺失新数据不能进入共同模型；偏相关投影代数；重采样可复现且按块连续；基准重现上一轮逐日权益；全部27个账户的现金、成交资格、五日退出、持仓/行业上限核验。原引擎还逐次核对已实现及未实现盈亏。

`protocol.md`保存查看结果前固定的规则；`research.py`实现本轮研究，依赖相邻`factor_research`和`swing_research`的已有模块与原始数据。所有候选结果均保留。

- `metrics.csv` / `yearly.csv`：组合与分段、逐年结果。
- `diagnostic_summary.csv` / `diagnostic_daily.csv.gz`：全部六因子的3/5/10日诊断。
- `correlations.csv`：新因子与旧因子及相互之间的相关。
- `paired_daily_return_intervals.csv`：组合每日增量收益和重采样区间。
- `daily_equity.csv.gz` / `trades.csv.gz` / `orders.csv.gz` / `open_positions.csv`：账户明细。
- `candidate_counts.csv`、`latest_research_scores.csv`：候选计数和最后数据日研究评分。评分没有考虑真实账户持仓，不是下单清单。
- `audit.json` / `source_hashes.json` / `validation.txt`：研究范围、来源指纹及验证记录。

运行命令（任务根目录）：

```sh
work/venv/bin/python outputs/factor_expansion/research.py
work/venv/bin/python -m unittest discover -s outputs/factor_expansion -p 'test_research.py' -v
work/venv/bin/python outputs/factor_expansion/write_report.py
```
'''
(P/'README.md').write_text(report)
