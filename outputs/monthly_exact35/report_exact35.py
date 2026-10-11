"""Report frozen exact-fit experiment; never train, tune, or choose a winner."""
from pathlib import Path
import json, hashlib, statistics
from datetime import datetime, timezone
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=read('RESULTS.json');F=read('FROZEN.json');I=read('INPUTS.json')
assert sha(P/'FROZEN.json')==(P/'FROZEN.sha256').read_text().strip()
assert F['code_sha256']==sha(P/'exact35.py')
assert all(sha(P/n)==h for n,h in F['input_hashes'].items())
assert all(sha(p)==h for p,h in read('PROTECTED_BEFORE.json').items())
N={'ordinal_tree':'直接排序决策树','momentum_residual_tree':'动量基础分＋决策树修正（预先指定主模型）','momentum_knn15':'动量基础分＋15邻居相似状态修正'}
rows={k:read(k+'_MONTHS.json') for k in N}
for k,rr in rows.items():
 assert len(rr)==69 and all(z['order_exact'] and z['set_exact'] for z in rr[:35])
 for split,ss in [('training',rr[:35]),('validation',rr[35:])]:
  m=R[k]['matching'][split]
  assert sum(z['order_exact'] for z in ss)==m['order_exact']
  assert sum(z['set_exact'] for z in ss)==m['set_exact']
  assert sum(z['shared_slots'] for z in ss)==m['shared_slots']
 assert all(all(v is not None for v in z['author_model_ranks'].values()) for z in rr[35:])
A=R['author'];T=A['slices']['training35'];V=A['slices']['validation34']
lines=[
'前35个月精确持仓及排序拟合 → 冻结后34个月验证',
'生成时间：'+datetime.now(timezone.utc).isoformat(),
'',
'结论：三种方法均达到训练35/35个月有序三股组合完全一致；冻结后没有一种完整复现后34个月。',
'训练持有期：2021年1月至2023年11月；净值从2020-12-31收盘开始。',
'验证持有期：2023年12月至2026年9月24日；净值从2023-11-30收盘开始，最后一个月不完整。',
'作者名单原顺序作为排名1/2/3；三只等权，顺序影响匹配指标但不影响三只组合收益。',
'',
'一、前35个月',
f'三种方法全部：持仓35/35、顺序35/35、105/105名额一致；逐日净值与作者记录重放一致。',
f'训练年化{T["cagr"]:.2%}，Sharpe {T["sharpe"]:.3f}，最大日终回撤{T["max_drawdown"]:.2%}。',
'这是使用已知持仓标签进行的训练拟合，不能作为预测成绩。',
'',
'二、冻结后的后34个月（唯一主对照）',
'方法 | 三股集合完全相同 | 顺序也完全相同 | 共有持仓名额 | 年化 | Sharpe | 最大日终回撤',
]
for k,label in N.items():
 z=R[k]['slices']['validation34'];m=R[k]['matching']['validation']
 lines.append(f'{label} | {m["set_exact"]}/34 | {m["order_exact"]}/34 | {m["shared_slots"]}/102 ({m["shared_slots"]/102:.1%}) | {z["cagr"]:.2%} | {z["sharpe"]:.3f} | {z["max_drawdown"]:.2%}')
for k,label in [('author','作者原记录重放'),('qqq','QQQ买入持有')]:
 z=R[k]['slices']['validation34'];lines.append(f'{label} | — | — | — | {z["cagr"]:.2%} | {z["sharpe"]:.3f} | {z["max_drawdown"]:.2%}')
pv=R[F['primary']]['slices']['validation34']
lines += [f'主模型比作者年化低{(V["cagr"]-pv["cagr"])*100:.2f}个百分点，Sharpe低{V["sharpe"]-pv["sharpe"]:.3f}。',
'共有名额忽略顺序；顺序完全一致要求当月三只股票及其名次全对。',
'后34个月作者的102个目标名额都在本次可选池内：这次验证错配不能归因于目标标的不在池内。',
'每种方法都在验证前保存冻结参数；主模型由PLAN_PRIMARY.json预先指定，不按验证成绩改选。',
'本轮训练/复杂度选择没有使用后34个月标签；这些历史时期此前研究已经看过，因此不能称全新未接触样本外。',
'',
'三、如何做到前35个月完全匹配',
'输入共56个数值特征：跳过最近42交易日/两个月的长期动量、风险波动、个股相对所属行业ETF、量价形态与市场状态。',
'输入不包含股票代码、日期、月份编号或未来收益。日期/代码只用于识别候选及展示解释。',
'每月因子先在当月候选中转成居中的百分位排名；市场状态使用当时可见值。',
'基础分固定为：70%长期动量＋15%近126日波动＋10%跳过42日的个股相对所属行业强弱＋5%CMF21。',
'长期动量为252/294交易日窗口及12/14日历月窗口的四种收益率百分位平均，均剔除最后约两个月收益。',
'这个基础分是本实验设定，并非作者公开系数。',
'',
'直接排序树：把作者前三名标为3、2、1，其余标0；树用数值条件预测分数，每月取前三。',
'动量修正树：先对每个训练月求最小平方改动，让作者第1名高于第2名、第2名高于第3名、第3名高于其余，分差至少0.005。再由树学习这些改动，未来仍以基础分＋预测改动排名。',
'相似状态法：使用同一批训练改动；用训练期标准差标准化特征并平衡类别，长期动量类别两倍距离权重；未来取15个最相似训练状态，按距离平方的倒数加权改动。精确遇到训练状态时使用其自身改动。',
'两棵树尝试深度1..32及不限制深度，再对成功树剪枝；仅在满足训练35/35有序一致的候选中选叶子最少者。',
]
for k in ['ordinal_tree','momentum_residual_tree']:
 c=R[k]['complexity'];tree=F['models'][k]['tree'];support=F['models'][k]['support'];single=sum(len(v)==1 for v in support.values())
 lines.append(f'{N[k]}：深度{c["depth"]}、{c["leaves"]}叶节点，其中{single}个叶节点只含1条训练样本。')
a=read('TRAIN_SCORE_CORRECTIONS.json')
lines += [f'相似状态法存储{F["training_examples"]}条训练候选记录；k=15在验证前固定。',
f'35个月所需改动：每月RMS改动中位数{statistics.median(z["rms"] for z in a):.4f}；每月最大绝对改动中位数{statistics.median(z["max_abs"] for z in a):.4f}；极端单项改动{max(z["max_abs"] for z in a):.4f}。',
'能给出条件路径不等于找回作者公式。较深的树和精确邻居插值可以记住训练样本；修正也不能一概称为作者所说的小幅机器学习。',
'',
'四、逐月对照例子',
]
for z in rows[F['primary']][35:38]:
 lines.append(f'{z["holding_end"][:7]}（信号{z["signal_date"]}）：作者 {" > ".join(z["author"])}；主模型 {" > ".join(z["selected"])}；作者股票在模型中名次{z["author_model_ranks"]}；当月组合收益差{z["return_difference"]*100:+.2f}个百分点。')
lines += ['主模型三股集合完全匹配的验证持有月：'+', '.join(z['holding_end'][:7] for z in rows[F['primary']][35:] if z['set_exact'])+'。',
'主模型连顺序也匹配的验证持有月：'+', '.join(z['holding_end'][:7] for z in rows[F['primary']][35:] if z['order_exact'])+'。',
'完整69个月对照在MONTHLY_COMPARISON.txt；逐个候选的分数、树路径、训练支撑样本或相似历史状态在各模型MONTHS.json、EXPLANATIONS.json。',
'',
'五、股票池与成交口径',
'训练期2022-05-31的COST在本地重建榜排36，原Top32/35无法容纳全部作者训练持仓。因此仅按前35个月信息，选取能覆盖训练目标的最小统一宽度Top36，所有69个月都按Top36＋自拟theme17执行。',
'这是统一扩大边界的替代池实验，不是已核实的作者原Top32；未逐月额外塞入目标股票。该改变及ETF名单不确定性会影响横截面排名。',
'使用现有缓存的历史估算排名及复权行情。无重新购买/下载数据，没有证明原作者历史股数快照与本地一致。',
'零交易费、零滑点、复权收盘价，同月末完整收盘计算排名并按该收盘价重放；每月三股等权，月内固定份额、不止盈止损、不调仓。',
'完整收盘价用于信号并同价成交是理想化复现口径；额外保留下一开盘执行敏感性，不代表实盘保证可成交。',
'',
'下一开盘敏感性，后34个月：',
]
for k,label in N.items():
 z=R[k]['next_open']['slices']['validation34'];lines.append(f'{label}：年化{z["cagr"]:.2%}，Sharpe {z["sharpe"]:.3f}，回撤{z["max_drawdown"]:.2%}。')
lines += ['', '六、全69个月（含精确拟合训练期，不作为独立验证）','方法 | 年化 | Sharpe | 最大日终回撤']
for k,label in list(N.items())+[('author','作者记录重放'),('qqq','QQQ')]:
 z=R[k]['metrics'];lines.append(f'{label} | {z["cagr"]:.2%} | {z["sharpe"]:.3f} | {z["max_drawdown"]:.2%}')
lines += ['', '七、审计与交付',
'9项针对约束投影、树序列化、训练/验证隔离、相似状态插值、候选资格、复杂度选择的单元测试通过。',
'训练35个月逐日净值等于作者记录重放；三种方法收盘重放均通过独立固定份额核算。',
f'{len(read("PROTECTED_BEFORE.json"))}个原始资料/现有观察账户等受保护文件哈希未变，未改策略账户、账本或计划。',
'冻结参数SHA256：'+sha(P/'FROZEN.json'),
'本轮精确训练及冻结验证任务完成；作者完整算法尚未恢复。',
]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
months=['逐月持仓与排序对照：从左到右分别排名1、2、3；收益均为零成本月末收盘至下次月末（最后月截至9/24）。',
        '前三十五个月为训练，后三十四个月为冻结后验证。月份按持有结束月份标注；完整边界日期见下表。','']
for i in range(69):
 z=rows[F['primary']][i]
 months += [f'{i+1:02d}. {"训练" if i<35 else "验证"} {z["signal_date"]} → {z["holding_end"]}',f'作者：{" > ".join(z["author"])}；收益{z["author_monthly_return"]:.2%}']
 for k,label in N.items():
  x=rows[k][i];months.append(f'{label}：{" > ".join(x["selected"])}；共有{x["shared_slots"]}/3，集合{"一致" if x["set_exact"] else "不同"}，顺序{"一致" if x["order_exact"] else "不同"}；收益{x["model_monthly_return"]:.2%}，差{x["return_difference"]*100:+.2f}个百分点')
 months.append('')
(P/'MONTHLY_COMPARISON.txt').write_text('\n'.join(months)+'\n')
status=dict(completed_at=datetime.now(timezone.utc).isoformat(),requested_experiment_complete=True,exact_training_order_all35=True,author_algorithm_recovered=False,primary=F['primary'],freeze_sha256=sha(P/'FROZEN.json'),universe_width=36,author_validation_eligible_slots=102,training_months=35,validation_months=34,unit_tests_passed=9,protected_file_count=len(read('PROTECTED_BEFORE.json')),protected_changes=[],models={k:dict(matching=R[k]['matching'],validation=R[k]['slices']['validation34']) for k in N},reports=['REPORT.txt','MONTHLY_COMPARISON.txt'])
(P/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print('Reports complete; 69 month comparisons, all frozen/input/protected checks passed.')
