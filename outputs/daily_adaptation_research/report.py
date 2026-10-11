"""Full record of all phases, with explicit failures and research-selection chronology."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import adaptation as a
P=Path(__file__).resolve().parent
NAMES={'validated_control':'上一轮模型验证版','fixed15':'固定配对门槛0.15','fixed10':'固定配对门槛0.10','dispersion_gap':'按分数离散程度调整门槛','age_decay':'持仓越久门槛越低','walk_gap':'每月按过去表现选门槛（原始比较）','momentum_exception':'极强动量股放宽入场','entry_limit':'开盘买入限价＋0.5ATR','model_decay252':'Ridge近期样本加权','model_window504':'Ridge训练窗口减为504日','model_ensemble':'三个Ridge预测平均','history_decay252':'历史反弹经验近期加权','short_window':'择参窗口63／126日','long_window':'择参窗口252／504日','limit025':'开盘限价＋0.25ATR','limit075':'开盘限价＋0.75ATR','walk_plus_limit':'动态门槛＋0.5ATR限价'}

def main():
 phases=[('',24),('robustness',19),('stable_ties',4),('tie_neighbors',4)];summaries={};loaded={};hashes={}
 for folder,count in phases:
  path=P/folder;s=json.loads((path/'SUMMARY.json').read_text());assert len(s)==count;assert json.loads((path/'CHECKS.json').read_text())['complete'];summaries[folder]=s
  for key,v in s.items():
   file=path/(key+'.json');x=json.loads(file.read_text());assert x['status']=='complete' and not x['performance_publishable'];assert x['metrics']['equity']==v['ending_equity']
   assert all(q['signal_date']<q['session'] for q in x['decisions']);assert all(q['cash']>=-1e-6 and q['exposure']<=1+1e-8 for q in x['daily'])
   rel=str(file.relative_to(P));hashes[rel]=hashlib.sha256(file.read_bytes()).hexdigest();loaded[rel]=x
 protect=json.loads((P/'SOURCE_HASHES.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in protect.items())
 main_s=summaries[''];rb=summaries['robustness'];stable=summaries['stable_ties'];nb=summaries['tie_neighbors']
 comparison=[('固定0.10','fixed10_cost10.json'),('动态原始','walk_gap_cost10.json'),('动态稳定并列','stable_ties/2019_cost10.json'),('限价入场','entry_limit_cost10.json')]
 annual={};concentration={}
 for name,key in comparison:
  x=loaded[key];annual[name]={str(y):a.r.period_metrics(x['daily'],f'{y}-01-01',f'{y}-12-31')['return_'] for y in range(2019,2027)}
  frame=pd.DataFrame(x['daily']).set_index('date');z=np.log1p(frame.loc['2025-10-06':,'return']);top=z.nlargest(10)
  concentration[name]=dict(actual_return=float(np.expm1(z.sum())),zero_best5_return=float(np.expm1(z.sum()-top.iloc[:5].sum())),zero_best10_return=float(np.expm1(z.sum()-top.sum())),best5_dates=list(top.index[:5]),note='Hindsight sensitivity only, no claim this counterfactual can be traded or is an expected return.')
 (P/'ANNUAL_RETURNS.json').write_text(json.dumps(annual,ensure_ascii=False,indent=2)+'\n');(P/'CONCENTRATION.json').write_text(json.dumps(concentration,ensure_ascii=False,indent=2)+'\n')
 b=pd.DataFrame(loaded['fixed10_cost10.json']['daily']).set_index('date');x=pd.DataFrame(loaded['stable_ties/2019_cost10.json']['daily']).set_index('date');assert x.index.equals(b.index)
 delta=np.log1p(x['return'])-np.log1p(b['return']);uncertainty={}
 for period,start,end in [('full','2019-01-02','2026-10-06'),('development','2019-01-02','2023-12-31')]:
  v=delta.loc[start:end].to_numpy();rng=np.random.RandomState(811);n=len(v);starts=rng.randint(0,n,(2000,(n+19)//20));samples=v[(starts[:,:,None]+np.arange(20))%n].reshape(2000,-1)[:,:n].mean(1);ci=np.expm1(np.quantile(samples,[.025,.975])*252)
  uncertainty[period]=dict(annualized_growth_ratio_minus1=float(np.expm1(v.mean()*252)),lower95=float(ci[0]),upper95=float(ci[1]),block_sessions=20,resamples=2000,note='Descriptive paired-block interval; reused history and multiple tests not corrected. Not a CAGR percentage-point interval.')
 (P/'UNCERTAINTY.json').write_text(json.dumps(uncertainty,indent=2)+'\n')
 def table(s,kinds,cost):
  rows=[f'买卖合计成本{cost/100:.2f}%','方案 | 年化 | 最大回撤 | 剔除最近252日年化 | 最近252日收益 | 开发期年化 | 平均仓位']
  for kind in kinds:
   v=s[f'{kind}_cost{cost}'];values=[v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['last252']['return_'],v['development']['cagr'],v['mean_exposure']]
   rows.append(NAMES[kind]+' | '+' | '.join(f'{z:.2%}' for z in values))
  return rows
 lines=['日策略第五轮：因果动态门槛、近期训练和限价入场','',
 '完成日期：2026-10-08；日线研究区间2019-01-02—2026-10-06；最近252日2025-10-06—2026-10-06，排除后截至2025-10-03。',
 '第一阶段12版本×两档成本24次；追加19次敏感性/成本/空仓检验；4次数值并列处理与4次相邻窗口检查，共51次。所有结果保存，没有筛掉失败版本。',
 '股票池、因子等权和开平仓基础规则相同：历史NDX100∪SP100内可用市值Top32＋17只非杠杆ETF，四组各25%、最多3只、目标少于2只空仓；不加杠杆。初始10万美元、现金不计息、未计个人税。',
 '股票池仍是可用数据重建，严格历史认证未完成；所有结果performance_publishable=false。所有历史反复研究过，不称为未触碰样本外。',
 '', '本轮结论',
 '数值稳定的动态门槛版本全期年化25.40%、最大日终回撤27.54%、剔除最近252日年化18.62%、最近252日82.88%。固定0.10对照分别24.06%、27.54%、17.16%、82.88%。',
 '买卖合计成本增为0.30%时，稳定动态版年化20.08%，固定0.10为18.71%；最近252日两者都是74.51%。',
 '没有突破最近一年100%。最近一年动态版实际一直选择0.10，因此这段没有新增超额；从2024年空仓起步，稳定动态版年化39.21%，固定0.10为39.56%，也没有胜出。改进主要来自更早时期，不是所有市场阶段都更好。',
 '开发期标准实际选出的单项是0.5ATR限价入场：开发期年化18.54%，但全期只有17.34%，2024空仓起步年化15.28%。它没有延续开发期优势，所以未采用到任何观察账户。',
 '更短训练窗口、近期加权、预测平均和极强动量入场例外没有胜过原固定0.10对照。减少数据不等于更快且更好的适应。',
 '', '动态规则的明确实现',
 '保留三个固定参考策略：配对换股评分差门槛0.10、0.15、0.20。参考策略都遵守原四因子、成熟标签、当期研究池、前收盘信号和次日开盘成交。',
 '每月第一个信号日，只读上一个完整交易日以前的参考净值，分别算过去126和252日平均对数日收益，取两个窗口中较低者作为评分。至少252日已完成记录；不足时门槛0.15。',
 '选择历史评分最高的参考门槛，当月固定。账户继续持有自己的仓位，仅改变后续配对换股门槛；不拼接参考净值，不直接搬入参考账户持仓。',
 '数值稳定规则：两项评分差不超过1e-10（平均对数日收益单位）视为并列；上月门槛在并列组中就保留，否则选最接近0.15者，再并列选较小值。该容差是数值规则，不是为最高收益搜索出来。',
 '原始选择器曾因1e-17量级的浮点差异选不同门槛；原始结果年化24.35%完整保留。发现后追加稳定规则与测试，所得25.40%属于后续研究结果，不能包装成第一阶段事前选定策略。',
 '交易成本压力测试保持参考选择使用原0.10%费用的参考策略，比较的是同一政策下提高实际交易成本的影响，而非每档成本重新择参。',
 '入场仍要求原趋势与回撤资格及评分至少0.75；止损2ATR、止盈3ATR、最多10交易日，退出按前收盘判断、下一开盘代理成交。风险不是盘中硬止损，无法保证跳空损失。',
 '', '第一阶段全部结果']
 for cost in [10,30]:lines+=['']+table(main_s,a.KINDS,cost)
 lines+=['','第一阶段选择记录',json.dumps(json.loads((P/'SELECTION.json').read_text()),ensure_ascii=False),
 '', '原始动态门槛和限价相邻参数；看到第一阶段结果后追加，不据最近收益重选']
 for cost in [10,30]:lines+=['']+table(rb,['short_window','long_window','limit025','limit075','walk_plus_limit'],cost)
 lines+=['','稳定并列规则的相邻窗口（数值检查后再次追加，不更换126/252日默认）','窗口 | 成本 | 年化 | 最大回撤 | 最近252日']
 for label,key,source in [('63/126','w63_126',nb),('126/252','2019',stable),('252/504','w252_504',nb)]:
  for cost in [10,30]:
   v=source[f'{key}_cost{cost}'];lines.append(f'{label} | {cost/100:.2f}% | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%} | {v["last252"]["return_"]:.2%}')
 lines+=['','0.50%成本压力：原始动态版本（稳定版仅验证0.10/0.30%，不混用）','方案 | 年化 | 最大回撤 | 最近252日']
 for kind in ['fixed10','walk_gap','entry_limit']:
  v=rb[kind+'_cost50'];lines.append(NAMES[kind]+' | '+' | '.join(f'{z:.2%}' for z in [v['full']['cagr'],v['full']['max_drawdown'],v['last252']['return_']]))
 lines+=['','2024-01-02从10万美元现金重新开始；模型可读此前已成熟数据，这不是未看过的新样本外','方案 | 成本 | 年化 | 最大回撤']
 for kind in ['fixed10','walk_gap','entry_limit']:
  for cost in [10,30]:
   v=rb[f'fresh2024_{kind}_cost{cost}'];lines.append(f'{NAMES[kind]} | {cost/100:.2f}% | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%}')
 for cost in [10,30]:
  v=stable[f'2024_cost{cost}'];lines.append(f'动态稳定并列 | {cost/100:.2f}% | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%}')
 lines+=['','逐年收益；2026仅至10/6','年份 | 固定0.10 | 动态原始 | 动态稳定并列 | 限价入场']
 for year in range(2019,2027):lines.append(str(year)+' | '+' | '.join(f'{annual[k][str(year)]:.2%}' for k,_ in comparison))
 lines+=['','收益集中度：只作事后敏感性分析，不是交易建议、不是可实现替代组合','方案 | 最近252日 | 最佳5天置零 | 最佳10天置零']
 for name,x in concentration.items():lines.append(f'{name} | {x["actual_return"]:.2%} | {x["zero_best5_return"]:.2%} | {x["zero_best10_return"]:.2%}')
 lines+=['','相对固定0.10的描述性不确定性：20交易日分块、2000次固定种子抽样；未调整多轮试验。']
 for key,x in uncertainty.items():lines.append(f'{key}: 年化增长比减一{x["annualized_growth_ratio_minus1"]:.2%}，95%描述区间[{x["lower95"]:.2%},{x["upper95"]:.2%}]。这不是年化差额的百分点区间。')
 lines+=['','执行与数据限制',
 '开盘限价在前一交易日确定为收盘＋指定ATR幅度，次日开盘超过限价则取消买入；已计划卖出照常执行，不事后换选其他股票。限价取消后可能临时只剩一只或现金，下一日再按原少于两只规则处理。该版本未使用当日收盘决定是否成交；仍是限价开盘竞价代理，不保证真实成交。',
 '四大因子25%组权不变。纯学习变体只改模型训练时段/样本权重或反弹历史组；模型验证开关仍只用成熟标签。更广股票池、加杠杆或修改近期起点没有用于抬高结果。',
 '市值快照频率、已知时间、历史身份与退市/行业/公司行动仍继承原研究限制；严格认证未完成。没有补造2008—2009历史。',
 '', '来源与核验',
 '成本与适应性的研究动机参考公开原始研究摘要：https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies?aqrPDF=1；https://www.nber.org/papers/w35158。它们不验证本项目参数或收益，本轮不是其策略复制。',
 '10项测试通过：加权Ridge数值、成熟标签隔离、近期统计成熟期、月度门槛的过去信息、离散度因果性、限价拒绝/拆股单位/当日收盘隔离、近似并列和后续信息隔离。全部51组完成，三项旧对照在两档成本下逐笔订单及净值复现，信号早于执行、现金非负且无杠杆。',
 '冷启动研究没有2019—2023指标，首次补测在日志输出None时中断；修复日志后校验相同案例和保护哈希，复用完成结果继续，没有改回测计算。',
 '现有六账户和月策略观察账户的账本/计划/配置，以及既有策略和数据manifest哈希未变化，没有自动加入任何新观察账户。',
 '', '文件：CONFIG.json、robustness/CONFIG.json、stable_ties/CONFIG.json、tie_neighbors/CONFIG.json记录各阶段；SUMMARY.json和各cost结果含所有交易/净值；GAP_CHOICES.json与stable_ties/CHOICES.json记录当时择参；ANNUAL_RETURNS.json、CONCENTRATION.json、UNCERTAINTY.json及FINAL_VALIDATION.json记录诊断。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(dict(completed_runs=len(hashes),tests_passed=10,result_sha256=hashes,code_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['adaptation.py','robustness.py','stable_ties.py','tie_neighbors.py','test_adaptation.py','report.py']},protected_unchanged=True,controls_reproduced=True,labels_mature=True,data_certified=False,performance_publishable=False),indent=2)+'\n')
 print('Validated',len(hashes),'runs; report saved.');print(json.dumps(uncertainty,indent=2))

if __name__=='__main__':main()
