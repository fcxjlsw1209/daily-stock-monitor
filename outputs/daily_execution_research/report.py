"""Reconcile runs, disclose repeated research and generate a complete record."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import run_execution as e
P=Path(__file__).resolve().parent
labels={'baseline':'原日策略','validated_base':'上一轮：模型通过历史验证才启用','matched_replacement':'逐笔配对换股（分差0.15）','net_targets':'只在名单变化时差额调仓','matched_net':'配对换股＋差额调仓','no_take_profit':'取消固定止盈','trailing_profit':'移动止盈','no_progress':'四日不涨就退出','sector_cap2':'每行业最多两只','residual_pullback':'额外要求个股残差回撤','dry_pullback':'缩量回撤评分','validated_cost_margin':'验证模型＋收益成本门槛','gap10':'配对换股（分差0.10）','gap20':'配对换股（分差0.20）','matched_sector2':'配对換股＋行业上限'}

def main():
 main=json.loads((P/'SUMMARY.json').read_text());rob=json.loads((P/'robustness/SUMMARY.json').read_text());assert len(main)==24 and len(rob)==15
 assert json.loads((P/'CHECKS.json').read_text())['complete'];assert json.loads((P/'robustness/CHECKS.json').read_text())['complete']
 assert json.loads((P/'SOURCE_HASHES.json').read_text())==e.protected_hashes()
 hashes={};annual={};loaded={}
 for folder,summary in [(P,main),(P/'robustness',rob)]:
  for key,v in summary.items():
   path=folder/(key+'.json');out=json.loads(path.read_text());assert out['status']=='complete' and out['performance_publishable'] is False
   assert all(z['signal_date']<z['session'] for z in out['decisions']);assert all(z['cash']>=-1e-6 and z['exposure']<=1+1e-8 for z in out['daily']);assert abs(out['metrics']['equity']-v['ending_equity'])<1e-6
   hashes[str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest();loaded[str(path.relative_to(P))]=out
 # Paired dependent-return uncertainty. Fixed seed, circular blocks; descriptive only.
 bootstrap={};base=loaded['validated_base_cost10.json'];b=pd.DataFrame(base['daily']).set_index('date')
 for key in ['matched_replacement_cost10.json','robustness/gap10_cost10.json']:
  x=pd.DataFrame(loaded[key]['daily']).set_index('date');assert x.index.equals(b.index)
  delta=np.log1p(x['return'])-np.log1p(b['return'])
  for name,start,end in [('full','2019-01-02','2026-10-06'),('development','2019-01-02','2023-12-31')]:
   v=delta.loc[start:end].to_numpy();rng=np.random.RandomState(810);n=len(v);blocks=(n+19)//20
   starts=rng.randint(0,n,(2000,blocks));ix=(starts[:,:,None]+np.arange(20))%n;means=v[ix].reshape(2000,-1)[:,:n].mean(1)
   interval=np.expm1(np.quantile(means,[.025,.975])*252)
   bootstrap[key+'_'+name]=dict(annualized_log_growth_ratio_minus1=float(np.expm1(v.mean()*252)),lower95=float(interval[0]),upper95=float(interval[1]),block_sessions=20,resamples=2000,note='Descriptive paired-block interval on reused history; no multiple-search adjustment or proof of alpha. Not a CAGR percentage-point difference.')
 (P/'UNCERTAINTY.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
 views=[('原版','baseline_cost10.json'),('上一轮','validated_base_cost10.json'),('配对0.15','matched_replacement_cost10.json'),('配对0.10','robustness/gap10_cost10.json')]
 for label,key in views:
  out=loaded[key];annual[label]={str(y):e.r.period_metrics(out['daily'],f'{y}-01-01',f'{y}-12-31')['return_'] for y in range(2019,2027)}
 (P/'ANNUAL_RETURNS.json').write_text(json.dumps(annual,ensure_ascii=False,indent=2)+'\n')
 lines=['日策略第四轮：换股配对、持仓管理与回撤因子','',
 '研究区间：2019-01-02至2026-10-06；最近252日为2025-10-06至2026-10-06，剔除最近252日截至2025-10-03。',
 '12版本×2档成本共24次；追加15次相邻参数、行业组合、0.50%成本及2024年空仓重启检验，共39次。初始10万美元、只做多、不加杠杆，最多3只、少于2只空仓。',
 '股票池与模型输入保持一致：历史Nasdaq-100∪S&P100内可用市值Top32＋17只非杠杆ETF；四大因子各25%。数据严格历史认证尚未完成，所有结果performance_publishable=false。',
 '0.10%、0.30%、0.50%均为买卖合计交易成本。现金不计息，不计个人税，所有开盘成交是代理模型。',
 '', '结论',
 '开发期2019—2023按事前声明标准选出逐笔配对换股0.15：年化22.12%、最大回撤29.35%、剔除最近252日年化17.86%、最近252日55.43%。上一轮分别21.61%、31.13%、16.37%、64.04%。',
 '优势集中在更早阶段：开发期年化13.21%→16.31%；2024年重新以现金起步时，年化从上一轮38.58%降为33.02%。不是所有时期都改善。',
 '后续相邻参数中0.10的观察结果更高：年化24.06%、回撤27.54%、最近252日82.88%；0.20则年化18.70%、最近252日45.25%。保留0.15作为先前开发期选定的研究候选；0.10是追加检查里更好的事后结果，没有自动提升或调入观察账户。',
 '没有达到近期100%的目标。参数敏感及重复开发仍然明显，不能把单个高收益组合称为验证成功。',
 '', '为什么检查换股',
 '旧规则对每个持仓都与同一个最高分候选比较，最高分可能就是已有持仓，或不足以补足多个卖出；再叠加少于两只清仓，可能退出整个组合。它是旧规则的实际行为，所有旧结果原样保留。',
 '新规则先执行到期、止盈止损和弱排名等退出，先补空位，再从最弱持仓开始寻找独立且未持有的合格发行人；持有至少2日、新旧评分差超过0.15才配对替换，同一候选不能重复使用。无法配对便不因更优候选而卖出。执行当日缺开盘价仍会取消新买入，不能事先预知后换选其他股票。',
 '配对规则保留止损2ATR、止盈3ATR、10交易日上限、评分0.75门槛及三只上限；所有信号用前一收盘、下个开盘代理执行。',
 '上一轮历史模型开关不变：每月用已成熟预测的126/252日相关性，均为正才让Ridge占总评分12.5%；否则使用历史四因子。',
 '', '全部第一阶段结果']
 def table(summary,keys,cost):
  rows=[f'买卖成本{cost/100:.2f}%；方案 | 全期年化 | 最大回撤 | 剔除最近252日年化 | 最近252日 | 开发期年化 | 平均仓位']
  for kind in keys:
   v=summary[f'{kind}_cost{cost}'];values=[v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['last252']['return_'],v['development']['cagr'],v['mean_exposure']]
   rows.append(labels[kind]+' | '+' | '.join(f'{z:.2%}' for z in values))
  return rows
 for cost in [10,30]:lines+=['']+table(main,e.KINDS,cost)
 lines+=['','追加检验：在看到第一阶段结果后预先固定，保留全部结果']
 for cost in [10,30]:lines+=['']+table(rob,['gap10','gap20','matched_sector2'],cost)
 lines+=['','成本压力（0.50%买卖合计）','方案 | 全期年化 | 最大回撤 | 最近252日']
 for k in ['baseline','validated_base','matched_replacement']:
  v=rob[k+'_cost50'];lines.append(labels[k]+' | '+' | '.join(f'{z:.2%}' for z in [v['full']['cagr'],v['full']['max_drawdown'],v['last252']['return_']]))
 lines+=['','2024-01-02重新空仓起步：历史模型可用2024年前成熟数据；2024后历史已看过，不是新样本外','方案 | 成本 | 2024年至今年化 | 最大回撤']
 for k in ['baseline','validated_base','matched_replacement']:
  for cost in [10,30]:
   v=rob[f'fresh2024_{k}_cost{cost}'];lines.append(f'{labels[k]} | {cost/100:.2f}% | {v["full"]["cagr"]:.2%} | {v["full"]["max_drawdown"]:.2%}')
 lines+=['','逐年收益；2026仅至10/6','年份 | 原版 | 上一轮 | 配对0.15 | 配对0.10']
 for y in range(2019,2027):lines.append(str(y)+' | '+' | '.join(f'{annual[label][str(y)]:.2%}' for label,key in views))
 lines+=['','描述性不确定性：相对上一轮的日对数增长差，20交易日分块、2000次固定种子抽样；95%区间未调整多轮筛选，不是显著性背书。']
 for key,v in bootstrap.items():lines.append(f'{key}: 年化增长比减一 {v["annualized_log_growth_ratio_minus1"]:.2%}, 描述性区间[{v["lower95"]:.2%}, {v["upper95"]:.2%}]')
 lines+=['','没有胜出的方向',
 '直接扩大仓位并未改善：差额调仓的平均仓位更高、全期收益反而下降。保留盈利仓位更久（取消固定止盈、移动止盈）也弱于原规则。四日不涨退出减少回撤，但换手和成本变大，收益下降。',
 '残差回撤入场和缩量评分未带来增益。文献提供研究动机，不保证在这个大盘股＋ETF、只做多、三只集中组合中有效。残差是SPY单因子近似，没有声称复制论文中的信息分类；缩量用成交金额比例，不等于股数换手率。',
 '两项文献动机：https://www.nber.org/papers/w30917；https://www.nber.org/papers/w35840。只使用公开摘要设计方向，没有复制其模型或借用论文收益。',
 '', '验证与限制',
 '11个测试通过：未来价格不改变过去因子/决策、候选不重复配对、已持有高分股不能触发虚假换股、发行人去重、硬退出空位优先、少于两只空仓、行业入场约束、移动止盈只看入场后已知高点、净额预算不使用当天收盘。',
 '39组均完成、现金非负、无杠杆；两档成本下原版与上一轮对照逐笔订单及净值复现。标签均在预测前成熟。既有策略、输入manifest、六个观察账户及新月策略的受保护账本/计划/配置哈希一致。',
 '首次运行仅在基准结果保存后的进度打印中发现return_/return字段名错误；修正日志并校验配置和输入哈希后复用已存基准继续，没有改计算结果。',
 '净额调仓会产生部分卖出记录，其实现片段胜率不可直接与整笔平仓比较；DIAGNOSTICS字段另按入场日期和证券合并已完成持仓，但增加份额沿用原持仓时钟。',
 '所有时期均已在之前研究中查看，不能把开发期之后或最近一年称为未触碰样本外。市值快照频率、历史可得时间、成分身份/退市/公司行动/行业仍继承原数据局限；没有宣称已完成严格历史认证，也没有填入缺失的2008—2009。',
 '这次没有修改、重新训练或重写任何现有观察账户；仅在独立研究目录拟合历史研究模型并运行实验。',
 '', '文件：CONFIG.json与robustness/CONFIG.json为分阶段参数；SUMMARY.json保存全部指标；每个cost结果包含逐日净值、计划和成交；SELECTION.json保留开发期选择；UNCERTAINTY.json、ANNUAL_RETURNS.json及FINAL_VALIDATION.json为诊断与校验。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 validation=dict(completed_runs=len(hashes),tests_passed=11,result_sha256=hashes,
  code_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['run_execution.py','check_robustness.py','test_execution.py','report.py']},
  controls_reproduced=True,labels_mature=True,protected_files_unchanged=True,data_certified=False,performance_publishable=False)
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
 # Small self-contained visualization payload. Full-resolution returns stay in artifacts.
 payload={'end':'2026-10-06','cost':.001,'series':[]}
 for label,key in views:
  out=loaded[key];daily=out['daily'];equity=np.array([100000.]+[x['equity'] for x in daily]);dd=equity/np.maximum.accumulate(equity)-1
  positions=sorted(set(range(0,len(daily),5))|{len(daily)-1}|{int(np.argmin(dd))-1})
  payload['series'].append(dict(label=label,points=[[daily[i]['date'],round(daily[i]['equity']/100000,5),round(float(dd[i+1])*100,4)] for i in positions if i>=0]))
 (P/'VISUAL_DATA.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
 print('Validated39runs. SavedREPORT.txt,FINAL_VALIDATION.json,VISUAL_DATA.json')
 print(json.dumps(bootstrap,indent=2))

if __name__=='__main__':main()
