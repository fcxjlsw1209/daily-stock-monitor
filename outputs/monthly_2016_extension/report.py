from extend import *

def main():
 r=read(P/'RESULTS.json');f=read(P/'FROZEN.json');v=read(P/'VALIDATION.json');s=read(P/'POOL_SENSITIVITIES.json')
 names={'anchor77':'原简洁候选77.62%','industry80':'已登记行业观察80.83%','prior82':'上一轮Ridge候选82.20%','return_primary':'本轮训练选定候选','robust_control':'稳健评分对照','qqq':'QQQ买入持有'}
 lines=['月策略：2016年扩展与有限优化',f'生成时间UTC：{datetime.now(timezone.utc).isoformat()}',
 '完成：2016-08-17至2026-09-24，122次组合决策；53个新增早期时点+原69个月。',
 '主要结果均为零成本、完整收盘信号与同收盘价成交的理想化回溯。Sharpe采用日收益/日标准差×sqrt252，无风险利率0。',
 '不是新增观察账户；所有已有观察账本、规则、计划和冻结文件未改变。','',
 '一、全区间对照（年化 / Sharpe / 最大日终回撤）']
 for k,label in names.items():
  z=r[k]['metrics'];lines.append(f'{label}: {z["cagr"]:.2%} / {z["sharpe"]:.3f} / {z["max_drawdown"]:.2%}')
 lines+=['','二、不同时间段（早期2016-08-17至2020-12-31；近期2020-12-31至2026-09-24）']
 for k,label in names.items():
  a,b=r[k]['slices']['early2016_2020'],r[k]['slices']['original2021_2026'];lines.append(f'{label}: 早期{a["cagr"]:.2%}/Sharpe{a["sharpe"]:.3f}；近期{b["cagr"]:.2%}/Sharpe{b["sharpe"]:.3f}')
 lines+=['','三、本轮候选规则',
 '沿用月末最多三只等权、两只各半、少于两只空仓；月内固定复权总回报单位，无月内换仓或止盈止损。',
 '基础准入：295个连续交易日有效历史；P[t-42]/P[t-252]-1为正。ETF与股票一起排序。无短期回撤入场门槛。',
 '长期动量核心52%：P[t-42]/P[t-294]-1，除以63日收益率标准差，在当期候选中转成中心化百分位。',
 '低波动市场：126日高波动偏好21.9375%，跳过42日的动量加速18.5625%，状态CMF63为7.5%。',
 '高波动市场：SPY63日波动高于其252日滚动中位数时，将原48%支持因子按比例缩为38%，用10%行业修正替代。',
 '该10% = 2.5%个股相对自身行业ETF的P[t-42]/P[t-126]表现 + 5%行业联动稳定性 + 2.5%市场极端下跌时残差抗跌性。',
 '行业稳定性为-corr63与corr252之差的绝对值；极端下跌残差用前日可知的beta和市场分位数。ETF/未知行业对应行业相对指标为中性。',
 '本轮候选没有Ridge加分；并非每增加一层模型都会提高长周期结果。完整精确参数见FROZEN.json selected.return_primary.spec。','',
 '四、训练与验证',
 f'本轮固定候选{f["candidate_count"]}套，仅2016-08-17至2023-11-30训练收益参与选型；Sharpe不得低于行业基准0.10以上，回撤不得扩大5个百分点以上。',
 '参数先保存FROZEN.json和校验和，再统一评估2023-11-30至2026-09-24。',
 '整段历史在此前研究中多次使用，本次时间划分不使后34个月变成从未触碰的样本外。新候选不能以回溯收益当作未来预期。',
 '2016—2020是向过去扩展的诊断，随后也进入本轮训练；不要称其为独立前向验证。',
 '上一轮82.20%候选早期ML年度拟合仅用此前5年、已经成熟的21交易日超额收益标签。alpha10，近期样本半衰期12个月。2021以后的原预测原样复用。']
 for k in ['industry80','prior82','return_primary','robust_control']:
  a=r[k]['slices']['training2016_2023'];b=r[k]['slices']['later34'];lines.append(f'{names[k]}: 训练年化{a["cagr"]:.2%}/Sharpe{a["sharpe"]:.3f}；后34月年化{b["cagr"]:.2%}/Sharpe{b["sharpe"]:.3f}')
 lines+=['','五、可执行性敏感性（固定同一持仓信号，改为下一交易日开盘成交）']
 for k in ['industry80','prior82','return_primary']:
  a=r[k]['next_open_zero']['metrics'];b=r[k]['next_open_cost']['metrics'];lines.append(f'{names[k]}: 零成本次日开盘{a["cagr"]:.2%}/Sharpe{a["sharpe"]:.3f}；单边0.05%成本后{b["cagr"]:.2%}/Sharpe{b["sharpe"]:.3f}')
 lines+=['主表遵循用户零成本要求；同收盘价模型使用完整收盘特征，不能称为实盘必然能成交的价格。','',
 '六、早期股票池与证据边界',
 '早期为重建的美国上市发行人Top32代理池，非作者核准名单，也非已认证完整全市场历史排名。474条缓存市值历史配合历史上市信息，每次只取前一交易日及之前的市值观察并按价格滚动。',
 '补齐BUD、NVS、HSBC2008年以来Yahoo行情；HSBC/NVS只补早期空白，并按重叠首日复权比例拼接，未修改旧价格。重叠误差小于0.0001%。',
 'CHL按2016-04-26已公布股数20,475,482,897、每ADS5股和缓存Tiingo价格补入；PTR以2016-04-28股数183,020,977,818、每ADS100股估算。',
 'PTR全股本按H股价格估值只是代理，A/H溢价未知；SHEL历史两股类按2.1倍上界情景处理而非精确公司市值。股数向前延用、排名档案发布时间、遗漏发行人及历史行业归类仍有限制。',
 '从2020-12-31起原69个月池保持不变；这也保留了原股票池本身的已披露局限。17只ETF仍是研究自拟名单。',
 '早期候选预热缺口为0；持仓期间若缺价，回放直接失败，未利用未来完整行情筛除亏损标的。',
 '2016-08-17按当日初始化选股，其后按月末；并不声称恢复了作者在2016年7月末的真实仓位。',
 '敏感性仅改变早期外国发行人市值假设，不重新优化：']
 for label,entry in s.items():
  a=entry['controls']['return_primary'];lines.append(f'{label}: 改变早期池{entry["changed_early_pools"]}个月，新候选年化{a["metrics"]["cagr"]:.2%}，Sharpe{a["metrics"]["sharpe"]:.3f}，早期持仓改变{a["changed_early_holdings"]}个月。')
 lines+=['这些情景未覆盖全部未知排名误差，不能把范围当统计置信区间。',
 '官方来源：https://www.sec.gov/Archives/edgar/data/1117795/000119312516555034/d151371d20f.htm',
 '官方来源：https://www.sec.gov/Archives/edgar/data/1108329/000119312516561899/d264935d20f.htm',
 '股类说明：https://www.shell.com/investors/information-for-shareholders/share-information.html','',
 '七、逐年收益（2016从8/17起，2026只到9/24，其余全年；非年化）']
 for row in r['return_primary']['metrics']['annual_returns']:lines.append(f'{row["year"]}: {row["return_"]:.2%}')
 lines+=['','八、核验与文件',
 '7项单元测试全部通过：事件日期与旧特征、未来市值忽略、快照时间、年度标签成熟、不足两只空仓与缺价失败、改变未来行情不影响过去信号、冻结与观察账户保护。',
 '三个旧候选在原69个月的持仓、年化、Sharpe、回撤均精确重现。每日股数账本与独立逐月固定股数路径一致。',
 '101配置全记录TRAIN_TRIALS.json；冻结参数FROZEN.json；全结果RESULTS.json；逐月组合return_primary_ORDERS.json；成交片段return_primary_TRADES.json；日净值return_primary_DAILY.json；测试TESTS.txt。',
 '复现：设置PYTHONPATH=work/monthly-ml-deps:work/monthly-decision-deps:work/research-plot-deps，OPENBLAS_NUM_THREADS=1，OMP_NUM_THREADS=1，使用work/venv/bin/python。',
 '重新评估已冻结结果：outputs/monthly_2016_extension/optimize.py evaluate；测试：outputs/monthly_2016_extension/test_extension.py -v。',
 '训练与初始化不允许覆盖现有冻结结果；新增研究应另建版本。环境仍有现有urllib3/LibreSSL兼容警告，本次行情下载及回放均成功。','',
 '结论：本轮全区间年化比行业观察基准提高约1.48个百分点，Sharpe小幅提高；没有复现作者图示约80.64%十年年化/Sharpe1.867。早期2016—2019表现是主要差距，2020大幅上涨贡献了较多早期收益，后续应关注不同市况稳定性。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 # Record hashes for the reused forecast and every runnable artifact as audit,
 # without rewriting frozen selection or source files.
 dump('DELIVERY_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),reused_forecast_sha=sha(O/'monthly_guidance_tweaks/PREDICTIONS.npz'),files={f.name:sha(f) for f in P.iterdir() if f.is_file() and f.name!='DELIVERY_MANIFEST.json'},protected_unchanged=not v['protected_changes']))
 print('Report saved',P/'REPORT.txt')

if __name__=='__main__':main()
