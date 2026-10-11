from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
summaries={name:json.loads((P/name/'SUMMARY.json').read_text()) for name in ['','joint','diagnostics','close_target']}
qqq=json.loads((P/'diagnostics/QQQ.json').read_text());N=qqq['n_returns']
def cagr(v):return v['final_multiple']**(252/N)-1
def p(x):return f'{x*100:.2f}%'

def main():
 comparison=[]
 selections=[('原日策略候选，30日上限','diagnostics','old_daily_cap30_cost10'),('新四组等权规则，10日上限','','rules10_next_open_cost10'),('新Ridge联合排名，10日上限','joint','joint_rank10_next_open_cost10'),('原月策略，市值随价格更新','diagnostics','monthly_scaled_prior_close_cost10'),('月策略同收盘零成本诊断','diagnostics','monthly_scaled_same_close_diagnostic_cost0')]
 for label,folder,name in selections:
  v=summaries[folder][name];comparison.append(dict(label=label,cagr_252=cagr(v),max_dd=v['full']['max_drawdown'],wealth=v['final_multiple'],source=str(Path(folder)/'runs'/f'{name}.json')))
 comparison.append(dict(label='QQQ持有，零成本',cagr_252=qqq['cagr_252'],max_dd=qqq['max_dd'],wealth=qqq['wealth'],source='diagnostics/QQQ.json'))
 data=dict(period=['2016-08-17','2026-09-24'],comparison=comparison,author_screenshot=dict(daily_cagr=.1941+1,monthly_cagr=.8064,qqq_cagr=.2092,daily_wealth=2752.26,monthly_wealth=387.72,qqq_wealth=6.78,daily_max_dd=-.1884,monthly_max_dd=-.3446,qqq_max_dd=-.3512,source='User supplied screenshot; strategy curves not reproduced or independently verified'),total_runs=sum(len(x) for x in summaries.values()),cagr_convention='wealth^(252/2539)-1; matches screenshot QQQ. Underlying SUMMARY.json also retains calendar365.25 CAGR.')
 (P/'COMPARISON.json').write_text(json.dumps(data,indent=2)+'\n')
 lines=['从四个核心概念重建：82组实际回测与截图口径对照','',
 '结论：这轮独立重建没有提高原日策略，也没有复现截图的119.41%年化。没有选出新的观察策略。这个结果约束的是本次明确写下的实现，并不证明原作者的策略不可能存在或截图造假。',
 '', '对齐与差异：',
 '研究时段与截图一致：2016-08-17—2026-09-24，2540个交易日、2539个相邻收盘收益间隔。表中按252交易日年化；SUMMARY.json保留365.25日历年年化。两种口径的微小差异不代表收益改变。',
 f"QQQ复权收盘到收盘：{qqq['wealth']:.8f}倍、年化{p(qqq['cagr_252'])}、Sharpe{qqq['sharpe']:.3f}、最大回撤{p(qqq['max_dd'])}，分别对应截图6.78倍、20.92%、0.958、-35.12%。市场基准与计算方式基本对上。",
 '截图日策略119.41%是全期年化，不能用最近252天涨100%替代。日月收益也不能直接相加：组合须按实际分配资本计算。',
 '本次所有账户从10万美元现金独立启动。月策略首次月末建仓为2016-08-31；作者起始持仓未知。股息、拆股和已核实分拆沿用实际股数账务。股票池依旧是历史NDX100∪SP100内Top32、发行人去重，加本项目17只无杠杆ETF；并非已确定作者使用的全市场Top32或相同ETF名单。',
 '', '同期间表现（下一开盘日策略；月策略默认前一交易日信号、月末收盘调仓；双边成本0.10%，另标零成本者除外）：',
 '版本 | 年化(252) | 最大日终回撤 | 期末本金倍数']
 for v in comparison:lines.append(f"{v['label']} | {p(v['cagr_252'])} | {p(v['max_dd'])} | {v['wealth']:.2f}x")
 lines += ['', '表中新日策略两项按全期事后选出用于说明本轮最高水平，不是独立样本外胜者。开发段选择及全部失败/较差候选见SELECTION.json、TRIAL_REGISTRY.json。原日策略此前40%左右年化采用2019年起点，这里的约33%用2016年起点；不能混用。',
 '', '这次重新构建了什么：',
 '1. 市值快照从原来的最长45天沿用不变，改为已知快照市值×当期拆股一致收盘价/快照日或之前最近收盘价。快照最早次日可用、不用未来插值，仍最多45天。假设快照间股数不变。2539收益间隔内的2540交易日中有1146天Top32成员不同，平均每天替换0.512只、最多5只。供应商历史修订/实际披露时间未认证，缺失成员可能改变边界，因此仍不是严格PIT认证。',
 '2. 四类因素：年度风险调整动量、相似回撤已成熟的反弹历史、反弹路径与波动空间、当前回撤深度/收盘位置/量价。相似事件分为回撤深浅×收盘是否在低位四类；过去756天同类结果向样本总体收缩20个伪观察，避免少量胜例产生极端分数。',
 '3. 第一结构为四组各一份Ridge模型、输出各占25%；第二结构将四组聚合分数、组内特征和组间交互联合Ridge学习，系数可变化，不能把第二结构说成固定25%权重。另试15%残差树修正、风险单位预测、收益横截面排名标签。',
 '4. 每月用之前1008交易日滚动训练，只有完成目标持有期的标签才进入；每训练日总权重相同，504日半衰期，alpha100。训练标准化只用过去样本。',
 '5. 新买入用回撤事件模型；在手股票用另一个强势股继续持有模型。继续持有不要求每天重新满足回撤条件。新候选相对继续持有预测优势达到固定门槛才替换，且至少持有2日。普通版本最多10日、2ATR收盘止损、3ATR收盘止盈；5/15/20日等明确分支全部公开。收盘触发的止损不保证下一开盘仍在止损价。',
 '6. 最多3只、目标不足2只空仓；新仓按净值/目标只数且受现金限制，已有股数不每日强制重置。月策略保持月内固定股数。没有杠杆、卖空、隐藏加资或将月日账户各100%后把收益相加。',
 '', '逐项口径对照：',
 '项目 | 下一时点含成本年化 | 零成本/同收盘年化']
 for k,v in summaries['diagnostics'].items():lines.append(f"{k}: {p(cagr(v))}，回撤{p(v['full']['max_drawdown'])}")
 lines += ['', '同收盘说明：第一批仅改变信号与成交时钟，保留次开盘模型目标，作为机制诊断。随后close_target目录把预测标签、历史回撤结果、未来路径定义也改成当日收盘到五日后收盘，再重训四种模型，共8个额外结果。所有这些同收盘结果使用最终OHLCV后又按同一收盘成交，没有盘中可获得性/竞价提交证据，不能称为可直接实盘执行的验证。由于交易路径也改变，同收盘诊断不保证比次开盘结果高。',
 '时序对齐后的同收盘结果：']
 for k,v in summaries['close_target'].items():lines.append(f"{k}: 年化{p(cagr(v))}，回撤{p(v['full']['max_drawdown'])}")
 lines += ['', '不能从上述结果推出原作者的119%主要来自某一个问题。我们仍缺作者17只ETF名单、精确Top32来源/股类口径、回撤与路径因子的公式、模型标签/训练窗口、调仓规则、盘前或盘后信号时间及逐笔交易。这些足以产生不同策略。截图主动注明零费用、同收盘、重复使用历史样本、市值代理；这些是核验起点，不是证明其结果虚假的证据。',
 '', '数据问题与修复：',
 '宽入场分支在2020-11-17遇到PFE 1.054非整数拆股，按规则停止。Pfizer官方公告证实原股数不变、每PFE股分配0.124079股VTRS，VTRS正式交易从11月17日开始；缓存Yahoo VTRS当天开盘15.39。研究内加入分配后按首次正式开盘卖出子公司股票、扣正常卖出费的既有协议。不是把辉瑞股数乘1.054。原失败记录保留，4个剩余分支已完成。精确开盘价不是独立交易所认证。原行情和既有账户文件未改。',
 '官方条款：https://www.pfizer.com/news/press-release/press-release-detail/pfizer-completes-transaction-combine-its-upjohn-business',
 '除权交易安排：https://www.pfizer.com/news/press-release/press-release-detail/pfizer-announces-details-when-issued-and-ex-distribution',
 'Ridge正规化定义参考：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html；只参考算法定义，不代表此工具/模型保证投资收益。',
 '', '验证与边界：',
 '16项独立测试；真实数据截断到2023-12-29后，特征、等权Ridge/树预测、联合Ridge预测、指定策略订单及净值与完整历史前缀一致。逐笔成交价核对原始历史股数单位。82个完整结果重算收益与回撤，现金非负、无杠杆、最多3只；观察账户与相关配置共94个保护文件哈希检查。',
 '数据仍为available-data有条件研究，严格准入与performance_publishable仍false。截断复算证明未检测到所检验路径的未来价格依赖，不消除股票池缺失、数据修订、行业历史映射或多轮挑选造成的偏差。重写算法不让已看过的2016—2026历史重新成为独立样本外。',
 '原六账户及月策略观察账户未改，未新增实盘或模拟推荐。没有承诺100%年化。',
 '', '文件：',
 'CONFIG.json / joint/CONFIG.json / close_target/CONFIG.json：每批运行前的明确规则。',
 'SUMMARY.json及各子目录：所有结果，未隐藏落后版本。',
 'TRIAL_REGISTRY.json：82组完整结果及哈希。',
 'FINAL_VALIDATION.json / PREFIX_VALIDATION.json / TESTS.txt：验证证据。',
 'COMPARISON.json / COMPARISON.png：可读对照数据及静态图。',
 '每个runs/*.json包含逐日净值、逐笔买卖、持仓、公司行动和决策。',
 '研究入口：rebuild.py、joint.py、diagnostics.py、close_target.py；完整结果目录禁止直接覆盖。若复现请复制到独立研究目录并保留原始来源缓存；研究固定截至2026-09-24，不下载或推算未来数据。',
 '失败恢复：complete_first.py在PFE证据修复后仅续跑尚未完成的预定案例；old_control.py隔离旧研究模块重名，不改旧代码。',
 '', '本轮全部新日策略（下一开盘，双边0.10%）：']
 for folder in ['','joint']:
  for k,v in summaries[folder].items():
   if k.endswith('next_open_cost10'):lines.append(f"{k} | 年化{p(cagr(v))} | 回撤{p(v['full']['max_drawdown'])} | 平均持有{v['mean_hold']:.1f}日 | 最近252天{p(v['last252_return'])}")
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 # Small artifact-only source metadata; no strategy code or ledger state changes.
 (P/'CODE_HASHES.json').write_text(json.dumps({str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*.py')},indent=2)+'\n')
 print(json.dumps(data['comparison'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
