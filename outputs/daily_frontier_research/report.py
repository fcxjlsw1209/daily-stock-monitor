"""Audit every completed run and publish a clearly retrospective research record."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
import frontier as z
P=z.P
PHASES={'':56,'expansion':40,'models':24,'shape':32,'interactions':24,'continuity':26,'exits':40,'robustness':34,'neighbor_validation':12}
CANDIDATES={'baseline':'shock_control_cost10','first_hit':'exits/no_profit_cap30_cost10','trailing':'exits/trail3_cap30_cost10','stop25':'robustness/stop25_cost10','stop15':'robustness/stop15_cost10','development_pick':'exits/no_profit_cap20_gap20_cost10'}
LABELS={'baseline':'上一轮10日上限＋3ATR止盈','first_hit':'30日上限、无固定止盈、2ATR止损','trailing':'30日上限、3ATR移动退出、2ATR止损','stop25':'30日上限、无固定止盈、2.5ATR止损','stop15':'30日上限、无固定止盈、1.5ATR止损','development_pick':'末轮开发期选中：20日上限、无固定止盈、分差0.20'}
def main():
 registry={};hashes={};loaded={};checks={}
 for phase,count in PHASES.items():
  folder=P/phase;summary=json.loads((folder/'SUMMARY.json').read_text());check=json.loads((folder/'CHECKS.json').read_text())
  assert check['complete'] and len(summary)==count;checks[phase or 'initial']=check
  for key,v in summary.items():
   file=folder/(key+'.json');raw=file.read_bytes();out=json.loads(raw);relative=str(file.relative_to(P));assert out['status']=='complete' and not out['performance_publishable']
   recomputed=z.summarize(out)
   assert recomputed=={k:value for k,value in v.items() if k!='diagnostics'},relative
   assert all(d['signal_date']<d['session'] for d in out['decisions']),relative
   assert all(d['cash']>=-1e-6 and d['positions']<=3 and d['exposure']<=1+1e-8 for d in out['daily']),relative
   assert len({d['date'] for d in out['daily']})==len(out['daily'])
   for d in out['decisions']:
    pairs=d.get('replacement_pairs',[]);assert len({v['buy'] for v in pairs})==len(pairs)
   registry[relative]=v;hashes[relative]=hashlib.sha256(raw).hexdigest()
   if relative[:-5] in CANDIDATES.values():loaded[relative[:-5]]=out
 expected=json.loads((P/'SOURCE_HASHES.json').read_text());assert expected==z.protected_hashes()
 assert len(registry)==288
 metrics={name:registry[path+'.json'] for name,path in CANDIDATES.items()};annual={};diagnostics={};trades={};uncertainty={}
 baseline=pd.DataFrame(loaded[CANDIDATES['baseline']]['daily']).set_index('date')
 for name,path in CANDIDATES.items():
  out=loaded[path];annual[name]={str(y):z.r.period_metrics(out['daily'],f'{y}-01-01',f'{y}-12-31')['return_'] for y in range(2019,2027)}
  frame=pd.DataFrame(out['daily']).set_index('date');log=np.log1p(frame.loc['2025-10-06':,'return']);best=log.nlargest(10)
  td=pd.DataFrame(out['trades']);closed=td.loc[td.exit_date>='2025-10-06']
  diagnostics[name]=dict(actual=float(np.expm1(log.sum())),best5_zero=float(np.expm1(log.sum()-best.iloc[:5].sum())),best10_zero=float(np.expm1(log.sum()-best.sum())),best_days=[dict(date=date,return_=float(np.expm1(value))) for date,value in best.items()],largest_closed_pnl=closed.groupby('symbol').net_pnl.sum().sort_values(ascending=False).head(8).to_dict(),note='Best-day removal is hindsight sensitivity, not a tradable strategy. Closed-PnL contributions include some pre-window entries and omit open unrealized PnL; do not interpret as exact window attribution.')
  trades[name]=out['trades']
  if name in ['first_hit','trailing','stop25','stop15']:
   assert frame.index.equals(baseline.index)
   diff=np.log1p(frame['return'])-np.log1p(baseline['return'])
   for period,a,b in [('full','2019-01-02','2026-10-06'),('development','2019-01-02','2023-12-31')]:
    v=diff.loc[a:b].to_numpy();n=len(v);rng=np.random.RandomState(219);starts=rng.randint(0,n,(2000,(n+19)//20));sample=v[(starts[:,:,None]+np.arange(20))%n].reshape(2000,-1)[:,:n].mean(1)
    lo,hi=np.expm1(np.quantile(sample,[.025,.975])*252)
    uncertainty[name+'_'+period]=dict(annualized_growth_ratio_minus1=float(np.expm1(v.mean()*252)),lower95=float(lo),upper95=float(hi),block_sessions=20,resamples=2000,note='Descriptive only; not adjusted for many tested strategies. Not an out-of-sample confidence interval or CAGR percentage-point difference.')
 for filename,value in [('TRIAL_REGISTRY',registry),('CANDIDATE_METRICS',metrics),('ANNUAL_RETURNS',annual),('CONCENTRATION',diagnostics),('CANDIDATE_CLOSED_TRADES',trades),('UNCERTAINTY',uncertainty)]:
  (P/(filename+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 pct=lambda value:'—' if value is None else f'{value:.2%}'
 def table(keys,include_dev=True):
  lines=['方案 | 最近252日 | 全期年化 | 全期最大回撤 | 剔除最近252日年化 | 开发期年化 | 胜率 | 平均持有交易日']
  for key in keys:
   v=metrics[key];values=[v['last252']['return_'],v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['development']['cagr'],v['diagnostics']['episode_win_rate']]
   lines.append(LABELS[key]+' | '+' | '.join(pct(t) for t in values)+f" | {v['diagnostics']['mean_sessions']:.1f}")
  return lines
 qqq=json.loads((P/'robustness/QQQ.json').read_text());rb=json.loads((P/'robustness/SUMMARY.json').read_text());nv=json.loads((P/'neighbor_validation/SUMMARY.json').read_text())
 lines=['日策略继续探索：达到最近一年100%的历史目标，尚未验证稳定可复制','',
 f'生成UTC：{datetime.now(timezone.utc).isoformat()}。',
 '共288次完整模拟：242次分批探索＋34次首个目标命中后的验证＋12次邻近止损版本验证；包含控制复现、不同费用和起点，不是288个独立策略。所有失败与较差版本均保留。',
 '主回测：2019-01-02—2026-10-06；最近252交易日：2025-10-06—2026-10-06；剔除最近252日截至2025-10-03；开发期2019—2023。',
 '目标指最近252日账户净收益达到100%，并非全期年化100%，也不是未来收益预测。初始10万美元，收益以整个账户净值（含现金）计算，未加杠杆，未计个人税和现金利息。',
 '重要数据限制：历史指数并集与历史市值快照认证未完成。Top32取截至信号日可用的过去市值快照，观测日期后一天才能使用，最长沿用45天；原始快照含月度/周度/日度混合频率，历史发布时间及后续修订未知，缺失日期排除。因此不是已认证的精确每日Top32，全部结果performance_publishable=false。',
 '所有年份已经被反复查看和调参；每批声明发生在本批结果之前，但后续批次参考了前批结果。不能把开发期以后的高收益称为新获得的样本外验证。',
 '', '核心结果（买卖合计0.10%成本，开盘代理成交）']
 lines+=table(list(CANDIDATES))
 lines+=['',f"QQQ同期从现金满仓买入持有：全期年化{pct(qqq['full_cost10']['cagr'])}、最大回撤{pct(qqq['full_cost10']['max_drawdown'])}；最近252日{pct(qqq['last252_cost10']['return_'])}。基准采用复权开盘、单边0.05%买入费，无期末卖出；复权近似分红再投资，与策略的现金分红及浮动风险敞口不同。",
 '', '选择纪律与解释',
 '首个命中目标的版本为no_profit_cap30；随后看到trail3_cap30亦达到目标。验证时预先保留首个候选，不因邻近参数近期更高而换成唯一胜者。',
 '末轮按事先规定的开发期标准（开发期回撤不超过35%、0.30%成本后的开发期年化不低于控制、再取开发期最高年化）选出no_profit_cap20_gap20，其最近252日只有38.56%。这与“最近一年最高”的候选明确不同。',
 '2.5ATR止损邻居的全期年化较高；1.5ATR邻居的最近窗口收益更高，但开发期年化降至13.88%。它们是在查看首轮结果后进行的邻近检验，不是事前预测到的胜者。',
 '主要改进来自退出方式：原固定3ATR止盈和10日上限改为无固定止盈、最长30交易日；仍每天处理止损、排名和替换。首个候选平均持有19.7交易日，已经是几周的波段，不能描述成1—2天短跑。',
 '', '可复现的首个候选规则',
 '1. 池：历史Nasdaq-100∪S&P100发行人并集内，按上述过去市值快照可用值取Top32，加原17只无杠杆ETF；发行人去重。日策略这里没有行业两只上限，不能混同原六观察账户规则。',
 '2. 资格：至少253日价格预热；20日平均成交额超过2000万美元；价格、成交量、行业有效。R12-1为正、复权收盘高于100日均线；三日标准化收益z<-0.35或当日收盘位于日内下20%，并要求z>-3.5。',
 '3. 四大组各25%：长期动量、成熟历史回撤反弹、反弹路径/波动、回撤量价形态。动量组平均使用R12-1/63日波动率、252日收益/63日波动率的横截面排名。反弹组保留旧Ridge因果启用政策，模型启用时替换该组的一半；分数0.75是排序阈值，不是75%胜率。',
 '4. Ridge每月重估：1008日训练窗、至少300事件、alpha10；五日未来开盘收益标签等待六个交易日成熟。过去126/252日成熟预测相关性双正且样本充分时才启用；没有用整段未来训练一次再回填。具体公式沿用冻结输入及代码。',
 '5. 异常低开过滤：开盘/前收-1<-2×此前20日波动，且成交量>1.5×此前20日平均量，信号日及随后两天不新买该证券。仍保留已有持仓的正常退出。',
 '6. 收盘用≥0.75评分筛选，次日开盘代理成交；最多三只；目标不足两只则现金。每个新标的预算为前日总净值/目标持仓数，受实际可用现金限制；老持仓不每日重新等权，不加杠杆。',
 '7. 每日先执行强制退出：持仓30交易日；前收低于入场价减2×入场ATR14；或有效当前评分跌破0.35。无固定止盈。价格止损在收盘判断、次日开盘卖出，跳空可使实际亏损超过阈值。',
 '8. 再处理一对一替换：持有至少两交易日、合格未持有候选评分比原持仓超过动态分差，才能替换。分差从0.10/0.15/0.20中每月用截至上月底的126/252日参考策略收益选择；保留旧因果稳定平局规则，缺数据用0.15。',
 '9. 2.5ATR及1.5ATR版本只改第7条止损倍数。移动版本在曾达到+1入场ATR后，以入场以来最高收盘减3入场ATR为额外退出线。',
 '', '成本压力（同一2019起算历史净值的最近252日窗口）','方案 | 双边成本 | 全期年化 | 全期最大回撤 | 最近252日']
 for name,path in [('first_hit','exits/no_profit_cap30'),('trailing','exits/trail3_cap30'),('stop25','robustness/stop25'),('stop15','robustness/stop15')]:
  for cost in [10,30,50,100]:
   if cost in [10,30]:v=registry[f'{path}_cost{cost}.json']
   else:v=(rb if name in ['first_hit','trailing'] else nv)[({'first_hit':'no_profit_cap30','trailing':'trail3_cap30'}.get(name,name))+f'_cost{cost}']
   lines.append(f"{LABELS[name]} | {cost/100:.2f}% | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])} | {pct(v['last252']['return_'])}")
 lines+=['','现金重新起算：不继承窗口前的持仓，模型仍可使用过去已成熟资料；历史已看过，不能称新样本外','方案 | 起点 | 成本 | 起点后年化（至少一年时） | 起点后总收益 | 最近252日']
 for name,short in [('baseline','control'),('first_hit','no_profit_cap30'),('trailing','trail3_cap30'),('stop25','stop25'),('stop15','stop15')]:
  source=nv if name in ['stop25','stop15'] else rb
  for prefix,start in [('fresh2024','2024-01-02'),('fresh252','2025-10-06')]:
   for cost in [10,30]:
    v=source[f'{prefix}_{short}_cost{cost}'];lines.append(f"{LABELS[name]} | {start} | {cost/100:.2f}% | {pct(v['full']['cagr'])} | {pct(v['full']['return_'])} | {pct(v['last252']['return_'])}")
 lines+=['','主候选的邻近参数：保持原选择，不用最优邻居重写结论','邻居 | 最近252日 | 全期年化 | 开发期年化']
 for name in ['cap25','cap35','stop15','stop25','fixedgap10','fixedgap15','fixedgap20','no_shock','min_age3']:
  v=rb[name+'_cost10'];lines.append(f"{name} | {pct(v['last252']['return_'])} | {pct(v['full']['cagr'])} | {pct(v['development']['cagr'])}")
 lines+=['','集中度：最佳日置零仅为事后敏感性，不是可执行方案','方案 | 实际最近252日 | 最佳五日置零 | 最佳十日置零']
 for name,v in diagnostics.items():lines.append(f"{LABELS[name]} | {pct(v['actual'])} | {pct(v['best5_zero'])} | {pct(v['best10_zero'])}")
 lines+=['','首个候选的已平仓净利润主要来自MU等科技股；详见CONCENTRATION.json。已平仓贡献不是完整的窗口归因，也不是同额持仓的个股收益。',
 '', '逐年收益（2026仅截至10月6日）','年份 | 原对照 | 30日2ATR | 30日移动 | 30日2.5ATR | 30日1.5ATR']
 for year in range(2019,2027):lines.append(str(year)+' | '+' | '.join(pct(annual[name][str(year)]) for name in ['baseline','first_hit','trailing','stop25','stop15']))
 lines+=['','描述性不确定性：20日分块、2000次、相对原控制；未做多重筛选校正，不能据此声称显著胜出']
 for key,v in uncertainty.items():lines.append(f"{key}: 年化增长比减一{pct(v['annualized_growth_ratio_minus1'])}；95%描述区间[{pct(v['lower95'])},{pct(v['upper95'])}]。")
 lines+=['','全部试验（包含失败方向，不筛掉负收益）','文件 | 全期年化 | 最大回撤 | 开发期年化 | 最近252日']
 for file,v in registry.items():lines.append(f"{file} | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])} | {pct(v['development']['cagr']) if v['development'] else '—'} | {pct(v['last252']['return_'])}")
 lines+=['','验证与文件',
 '26项测试通过，覆盖未来价格/标签隔离、只用已收盘信号、换股最小持有、现金与持仓个数、移动退出、硬止损、单只实验和252行统计边界。多轮控制的逐笔订单与每日净值复现。288个完整输出逐一重新计算汇总核对。',
 '短窗口补测曾触发旧统计器需要253行的边界错误；原始回测已经完成，修复仅在本研究的汇总适配器，测试后复用已完成结果。错误与修复记录保存在robustness/initial_attempt.log、REPAIR_NOTE.txt。',
 '32项既有代码/行情入口/缓存及六账户与月观察账户账本、配置的哈希不变；未改观察策略，没有下实盘单，也未自动加入新观察账户。',
 '完整结果：TRIAL_REGISTRY.json；候选指标：CANDIDATE_METRICS.json；交易：CANDIDATE_CLOSED_TRADES.json和各方案JSON；逐年：ANNUAL_RETURNS.json；集中度：CONCENTRATION.json；全部校验：FINAL_VALIDATION.json。',
 '运行各批主脚本会拒绝覆盖已完成结果。report.py只重新审计并生成报告；研究不能通过旧自动监控入口运行。',
 '', '结论：当前可用历史数据里已找到超过100%的最近一年回测版本；收益依赖阶段、参数和少数大涨日，数据认证及真正未来检验仍待完成。下一步应冻结候选并收集新数据观察，继续对同一段历史追高不会创造新的独立证据。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 final=dict(complete=True,completed_runs=len(registry),unit_tests_passed=26,phase_checks=checks,protected_unchanged=True,protected_files=len(expected),
 first_target_hit='exits/no_profit_cap30_cost10',selected_candidate_not_replaced_by_neighbor=True,
 main_start=z.CONFIG['start'],end=z.CONFIG['end'],recent_start='2025-10-06',target_is_recent_total_return=True,
 data_certified=False,performance_publishable=False,all_history_reused=True,observation_accounts_unchanged=True,
 result_sha256=hashes,code_sha256={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in P.glob('*.py')})
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(final,indent=2)+'\n')
 print('VALIDATED',len(registry),'runs; protected',len(expected));print('REPORT',P/'REPORT.txt')
 print('CANDIDATE_METRICS',json.dumps({k:{'full':v['full'],'recent':v['last252'],'dev':v['development']} for k,v in metrics.items()}))
if __name__=='__main__':main()
