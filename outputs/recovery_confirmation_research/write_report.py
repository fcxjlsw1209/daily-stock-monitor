from pathlib import Path
import json,hashlib
import pandas as pd
P=Path(__file__).resolve().parent
m=pd.read_csv(P/'metrics.csv');y=pd.read_csv(P/'yearly.csv');ci=pd.read_csv(P/'paired_intervals.csv')
labels={'baseline':'现有基准','shadow':'现有压力影子','clv_rank':'收盘改善：10%排序','momentum_rank':'短期动量：10%排序','clv_filter':'收盘改善：硬过滤','momentum_filter':'短期转正：硬过滤'}
def label(s):
 if s in labels:return labels[s]
 return labels[s.split('_',1)[0]]+'＋'+labels[s.split('_',1)[1]]
lines=['# 回调后恢复确认：独立研究结果','',
'结论：本轮8个单项变体均未通过事先写入方案的历史筛选条件。不修改现有基准、压力影子、冻结计划或自动化，不再为通过筛选而追加参数搜索。','',
'## 范围与规则','',
'2019-01-01至2026-09-18；固定当前100家美国上市公司。股票池有幸存者偏差，历史已用于多轮探索，分时段并非未触碰样本外。所有数据截止9/18，不使用9/22起的模拟结果挑模型。以下均为历史回测，不是真实收益。','',
'每只10%前日净值、最多5只、行业最多2只；信号日收盘生成，下一交易日开盘成交。与原策略相同的5日均线反弹退出、2ATR收盘止损、最多10个交易日；不强制末日平仓。收益是包含现金的账户收益，非单只股票收益。','',
'两个信号：CLV=(2×收盘−最高−最低)/(最高−最低)，平价K线CLV=0；收盘改善分数=CLV当日减前日。短期动量=当日复权收益÷此前20日日收益样本标准差。','',
'两种用法：排序保持R12–1权重30%，其余原评分按6/7缩放并加新信号百分位10%，保留原入围条件；硬过滤分别要求CLV>0且CLV提高，或当日复权收益>0。硬过滤仍要求原来的3日回调条件成立。它们不是原文神秘指标的复刻。','']
for cost in [10,20]:
 lines += [f'## 买卖合计成本 {cost/100:.2f}%','', '|方案|年化收益|净胜率|最大回撤|平仓笔数|平均仓位|平均单笔净收益|','|---|---:|---:|---:|---:|---:|---:|']
 for r in m[(m.cost_bps==cost)&(m.split=='all')].itertuples():
  lines.append(f'|{label(r.strategy)}|{r.cagr:.2%}|{r.trade_win:.2%}|{-r.max_drawdown:.2%}|{r.closed_trades}|{r.average_exposure:.2%}|{r.mean_trade_return:.3%}|')
 lines+=['']
lines += ['## 为什么不采用','',
'1. 基准加“收盘改善”排序，年化只增加约0.21个百分点，胜率反而下降约0.44个百分点。2019–2021年化11.50%降至9.87%，而2022–截止日11.34%升至12.74%，表现依赖时段。年化算术平均日收益差的区块重采样95%描述区间为−1.87至+2.09个百分点，跨零；这不是CAGR差的置信区间，也没有多重检验校正。','',
'2. 基准加“短期动量”排序，低成本年化11.39%到11.46%，但成本提高至0.20%后7.77%到7.72%，且近期区间也略差，未通过筛选。','',
'3. 硬过滤并未稳定提高胜率或降低回撤。基准收盘改善过滤，胜率63.94%到64.77%，但年化11.39%到7.80%，回撤13.30%到15.87%。等当日涨了再买，年化降到9.10%，胜率也略降。','',
'4. 对压力影子，四个变体全期年化都低于原来的13.07%；部分胜率提升不足以补偿收益损失。候选集合内，短期动量与压力减弱的平均逐日Spearman相关约0.70，收盘改善约0.34，说明至少短期动量与已有因子相当重叠。相关不证明无增量，但结果也未显示稳定增量。','',
'## 过滤掉了什么','',
'按原基准的2546笔已平仓交易逐笔检查信号（成本0.10%）：','',
'- 收盘改善硬过滤会拒绝1658笔，其中1057笔原本盈利、601笔亏损；这些被拒交易原平均净收益仍为+0.356%。','- 短期转正硬过滤会拒绝1689笔，其中1098笔盈利、591笔亏损；原平均净收益为+0.441%。','',
'这是原组合上的条件诊断；过滤后的组合会在其他日期交易，因此不能把拒绝笔数直接当成新组合减少笔数或收益损失的完整归因。原本的回调策略可能在尚未出现明显日线恢复时已有机会；本轮结果支持保留这种可能，不能推导所有反弹确认都无效。','',
'## 分年收益（成本0.10%，2026仅截至9/18）','',
'|年份|基准|基准＋收盘排序|基准＋动量排序|压力影子|影子＋收盘排序|影子＋动量排序|','|---|---:|---:|---:|---:|---:|---:|']
for year,g in y[y.cost_bps==10].groupby('year'):
 d=g.set_index('strategy');lines.append('|'+str(year)+'|'+'|'.join(f"{d.at[n,'total_return']:.2%}" for n in ['baseline','baseline_clv_rank','baseline_momentum_rank','shadow','shadow_clv_rank','shadow_momentum_rank'])+'|')
lines += ['', '## 验证与剩余问题','',
'两个父策略在两档成本和三个时段精确复现先前结果；截断到2021年末及2024年末不改变此前信号；新增信号导致的有效候选损失为0；检查全部成交使用下一开盘价、现金与净值逐日对账、最大持仓与持有期。两账户及统一执行目录运行前后的文件哈希完全一致。','',
'本次尚未验证1小时/15分钟共振、5分钟执行或期权收益，也无法验证原作者箭头是否重绘。日线硬过滤表现差不能否定盘中等待确认；那需要可靠分钟数据、只使用已完成K线、单独固定进出场规则和足够成本假设。暂不将它混入日线开盘策略。','',
'下一步若有指标Pine Script，可审计公式、重绘和发布时间，再定义一个独立的分钟级实验；没有源码时不按截图猜参数。目前没有证据需要新增第三个模拟账户。','',
'## 可复现文件','',
'- protocol.md：看结果前冻结的8项变体、筛选标准和成本。','- research_recovery.py：复用既有回测引擎；运行 work/venv/bin/python outputs/recovery_confirmation_research/research_recovery.py。','- metrics.csv / yearly.csv / paired_intervals.csv：全部变体和描述区间。','- filter_diagnostics.csv / correlations.csv：机会成本与重复信息诊断。','- daily_equity.csv.gz / trades.csv.gz / orders.csv.gz：逐日及逐笔记录。','- screen.json / validation.json：筛选结果与原运行文件哈希。']
(P/'README.md').write_text('\n'.join(lines))
source=[P/'protocol.md',P/'research_recovery.py',P/'write_report.py',P.parent/'strategy_selection/research.py',P.parent/'factor_research/backtest_factors.py',P.parent/'swing_research/backtest_swing.py',P.parent/'stock_database/stocks.sqlite']
(P/'source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source},indent=2))
print(P/'README.md')
