from pathlib import Path
import hashlib,json
import pandas as pd
P=Path(__file__).resolve().parent;m=pd.read_csv(P/'metrics.csv');y=pd.read_csv(P/'yearly.csv')
labels={'baseline':'现有基准','shadow':'现有压力影子','residual_rank':'抗跌残差排序','gap_limit':'开盘限价近似','gap_adverse':'限价上限极端压力','hold_exit':'收复后跌回退出'}
def label(n):return labels[n] if n in labels else labels[n.split('_',1)[0]]+'＋'+labels[n.split('_',1)[1]]
lines=['# 抗跌性、跳空限制与退出规则研究','',
'结论：基准＋抗跌残差排序通过本轮预先固定的历史筛选，值得保留为研究候选；证据不足以替换现有策略。另两个方向不推荐按本轮定义实施。所有运行账户、计划、提醒和策略文件保持原样，没有启动第三套账户。','',
'## 研究口径','',
'现有当前100家公司名单，2019-01-01至2026-09-18。每只10%前日净值、最多5只、行业最多2只，次日开盘代理成交；除退出实验外，guarded10退出不变。不强制清仓期末持仓。现金收益设0。数据存在幸存者偏差，区间反复被用于探索，不是未触碰样本外。','',
'## 三个事先定义的假设','',
'**抗跌残差排序**：用此前60个交易日（不含当天）的个股与SPY收益估计beta。当天残差=个股复权收益−beta×SPY收益。SPY收益用已知日线收盘及当日股息计算总收益近似。取最近20个交易日中大盘下跌日的平均残差，至少3个有效下跌日。分数高表示相对于市场风险暴露更抗跌，不等于未来必涨。','',
'新分数=0.30×R12–1百分位+(6/7)×(父策略原分数−0.30×R12–1百分位)+0.10×抗跌残差百分位。保留原入围条件，不对新分数再施加0.65；分别与基准、压力影子比较。','',
'**跳空限制**：前一收盘固定最高买价=收盘+0.5×ATR14，仅原先选中的股票参与。开盘严格低于上限才按开盘代理价买入，等于或高于则跳过、不替补、不在盘中追单。此为事前限价开盘订单的理想化日线近似，不是先看开盘再假设同价成交。普通日线Open并非已核实的官方竞价价，也没有竞价排队、部分成交或券商零股支持数据，所以不能据此宣称实盘可执行性已验证。','',
'事先还定义了极端成交压力：成交价格按上限而非开盘计价，筛选条件相同。该上限可能高于当日实际成交价范围，这是人为保守压力边界，绝不是预期滑点或真实成交模拟；持仓路径也会随入场价改变。即使不看这个极端压力，理想化版本本身也未优于父策略。','',
'**退出对照**：首次收复5日均线后不卖，等后续收盘跌回5日线下方才在次日开盘退出；保留原2ATR止损与最长10交易日。未曾收复也不取消止损或期限。','']
for cost in [10,20]:
 lines += [f'## 买卖合计成本 {cost/100:.2f}%','', '|方案|年化收益|净胜率|最大回撤|平仓数|平均持有日|平均仓位|','|---|---:|---:|---:|---:|---:|---:|']
 for r in m[(m.cost_bps==cost)&(m.split=='all')].itertuples():
  lines.append(f'|{label(r.strategy)}|{r.cagr:.2%}|{r.trade_win:.2%}|{-r.max_drawdown:.2%}|{r.closed_trades}|{r.mean_hold:.2f}|{r.average_exposure:.2%}|')
 lines += ['']
lines += ['## 为什么只保留抗跌性作为候选','',
'基准在0.10%成本下：年化11.39%→12.17%，胜率63.94%→64.31%，回撤13.30%→12.78%；成本翻倍后年化7.77%→8.49%，回撤17.32%→15.95%。平均单笔净收益0.342%→0.359%，属于小幅改善。','',
'早期2019–2021年化11.50%→12.73%，近期2022–截止日11.34%→11.84%。8个年度/部分年度中6个更好，但2023和2026截至日更差。','',
'配对日收益的20日区块重采样（1000次，固定seed42），年化算术平均收益差约+0.70个百分点，95%描述区间−1.47至+3.01个百分点，仍跨零。这不是CAGR差置信区间，未校正多重研究/参数选择，不能宣称显著优势。','',
'对压力影子：加入该因子后年化13.07%→13.90%，但最大回撤12.04%→13.04%；近期年化13.15%→12.78%，未通过预先要求的筛选。基准＋抗跌性也没有超过原压力影子的全期年化，不能只凭单个指标宣布最优。','',
'## 另外两个方向','',
'跳空限制的理想化版本在两套父策略、两档成本下都降低全期年化，并扩大最大回撤。不能以“避免追高”直觉自动采纳；开盘跳涨也可能包含继续上涨的机会。这里没有证据支持当前0.5ATR定义，也没有追加参数寻找好结果。','',
'延后退出：基准年化11.39%→12.44%，但胜率63.94%→51.79%、回撤13.30%→16.07%。平均持有2.75→5.94天，平均仓位35.97%→40.78%；交易数2546→1313，较高成本下更有优势主要与低换手有关。平均盈利交易2.10%→4.49%，平均亏损交易−2.78%→−3.33%，是盈亏结构和资金占用的改变，不能只看年化提升。压力影子改退出后全期年化反而下降。对于用户偏好较高胜率，本轮退出方案不优先。','',
'## 抗跌性候选分年收益（0.10%成本）','', '|年份|基准|基准＋抗跌性|','|---|---:|---:|']
for year,g in y[(y.cost_bps==10)&y.strategy.isin(['baseline','baseline_residual_rank'])].groupby('year'):
 d=g.set_index('strategy');lines.append(f"|{year}{'截至9/18' if year==2026 else ''}|{d.at['baseline','total_return']:.2%}|{d.at['baseline_residual_rank','total_return']:.2%}|")
lines += ['', '## 验证与文件','',
'两父策略精确复现此前两档成本及分段结果；新因子截断到2021/2024年末时此前数据不变；新增有效数据限制未损失原候选；资金/净值/交易盈亏逐项对账；最大持仓/行业/持有期检查通过。7项额外单元测试覆盖退出状态、未来数据扰动、止损、期限与开盘限价边界。全部运行目录前后文件哈希一致。','',
'protocol.md是事先协议；research_round2.py是完整研究脚本；test_round2.py为单元测试。metrics.csv、yearly.csv、paired_intervals.csv、screen.json和validation.json保存全部结果。daily_equity.csv.gz、trades.csv.gz、orders.csv.gz、skipped_gaps.csv保存逐日和逐笔细节。','',
'运行：work/venv/bin/python outputs/regime_execution_exit_research/research_round2.py；测试：work/venv/bin/python -m unittest discover -s outputs/regime_execution_exit_research -p "test_*.py" -v。','',
'目前不自动新增账户或修改规则。保留抗跌性定义，未来独立数据再验证；不根据已经看过的历史继续调beta窗口、排名权重或ATR门槛。','',
'开盘订单机制参考：[Nasdaq开收盘竞价](https://nasdaqtrader.com/Trader.aspx?id=OpenClose)、[Nasdaq股票交易规则](https://listingcenter.nasdaq.com/rulebook/nasdaq/rules/nasdaq-equity-4)。这些资料支持订单类型的存在，不保证本研究普通Open等于实际竞价成交。']
(P/'README.md').write_text('\n'.join(lines))
source=[P/'protocol.md',P/'research_round2.py',P/'test_round2.py',P/'write_report.py',P.parent/'recovery_confirmation_research/research_recovery.py',P.parent/'stock_database/stocks.sqlite']
(P/'source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source},indent=2))
print(P/'README.md')
