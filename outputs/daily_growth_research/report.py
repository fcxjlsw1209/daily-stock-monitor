"""Complete research audit: preserve negative findings and tiny improvements."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np,pandas as pd
import growth as g
z=g.z;P=g.P
PHASES={'':54,'horizons':26,'allocation':16,'validation':21}
KEYS={'control':'control25_cost10.json','crash':'crash_entry_cost10.json','extend':'strong_cap60_cost10.json','dev_management':'cap45_cost10.json','dev_horizon':'horizons/pred10_small_cost10.json','allocation_best':'allocation/net_momentum_cost10.json'}
LABELS={'control':'上一轮30日/2.5ATR研究候选','crash':'下跌且波动突增时暂停新买入','extend':'强势时允许持有60日','dev_management':'管理组开发期选中：固定45日','dev_horizon':'模型组开发期选中：10日预测、小权重','allocation_best':'差额调仓＋动量分配'}
def main():
 registry={};hashes={};loaded={};phase_checks={};selections={}
 for phase,count in PHASES.items():
  folder=P/phase;summary=json.loads((folder/'SUMMARY.json').read_text());check=json.loads((folder/'CHECKS.json').read_text());assert len(summary)==count and check['complete'];phase_checks[phase or 'management']=check
  if (folder/'SELECTION.json').exists():selections[phase or 'management']=json.loads((folder/'SELECTION.json').read_text())
  for name,v in summary.items():
   file=folder/(name+'.json');raw=file.read_bytes();out=json.loads(raw);rel=str(file.relative_to(P));assert out['status']=='complete' and not out['performance_publishable']
   assert z.summarize(out)=={k:value for k,value in v.items() if k!='diagnostics'},rel
   assert all(row['signal_date']<row['session'] for row in out['decisions'])
   assert all(row['cash']>=-1e-6 and row['positions']<=3 and row['exposure']<=1+1e-8 for row in out['daily'])
   assert len({row['date'] for row in out['daily']})==len(out['daily'])
   for row in out['decisions']:
    pairs=row.get('replacement_pairs',[]);assert len({pair['buy'] for pair in pairs})==len(pairs)
    if 'target_weights' in row:assert all(0<=w<=1 for w in row['target_weights'].values()) and sum(row['target_weights'].values())<=1+1e-10
   registry[rel]=v;hashes[rel]=hashlib.sha256(raw).hexdigest()
   if rel in KEYS.values():loaded[rel]=out
 assert len(registry)==117
 protected=json.loads((P/'SOURCE_HASHES.json').read_text());assert protected==g.protected_hashes()
 train=json.loads((P/'horizons/TRAINING_AUDIT.json').read_text());assert all(row['latest_label_session']<row['first_prediction'] for batch in train.values() for row in batch)
 annual={};concentration={};uncertainty={};baseline=pd.DataFrame(loaded[KEYS['control']]['daily']).set_index('date')
 for key,path in KEYS.items():
  out=loaded[path];frame=pd.DataFrame(out['daily']).set_index('date');annual[key]={str(y):z.r.period_metrics(out['daily'],f'{y}-01-01',f'{y}-12-31')['return_'] for y in range(2019,2027)}
  lr=np.log1p(frame.loc['2025-10-06':,'return']);best=lr.nlargest(10)
  concentration[key]=dict(actual=float(np.expm1(lr.sum())),best5_zero=float(np.expm1(lr.sum()-best.iloc[:5].sum())),best10_zero=float(np.expm1(lr.sum()-best.sum())),note='Retrospective sensitivity only, not executable performance or forecast.')
  if key in ['crash','extend']:
   assert baseline.index.equals(frame.index);diff=np.log1p(frame['return'])-np.log1p(baseline['return'])
   for label,start,end in [('full','2019-01-02','2026-10-06'),('development','2019-01-02','2023-12-31')]:
    v=diff.loc[start:end].to_numpy();n=len(v);rng=np.random.RandomState(309);starts=rng.randint(0,n,(2000,(n+19)//20));sample=v[(starts[:,:,None]+np.arange(20))%n].reshape(2000,-1)[:,:n].mean(1)
    lo,hi=np.expm1(np.quantile(sample,[.025,.975])*252)
    uncertainty[key+'_'+label]=dict(annualized_growth_ratio_minus1=float(np.expm1(v.mean()*252)),lower95=float(lo),upper95=float(hi),block_sessions=20,resamples=2000,note='Descriptive paired interval, unadjusted for repeated strategy search; not an independent out-of-sample interval or CAGR point difference.')
 for file,value in [('TRIAL_REGISTRY',registry),('ANNUAL_RETURNS',annual),('CONCENTRATION',concentration),('UNCERTAINTY',uncertainty),('SELECTIONS',selections)]: (P/(file+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 pct=lambda value:'—' if value is None else f'{value:.2%}'
 control=registry[KEYS['control']];crash=registry[KEYS['crash']];extend=registry[KEYS['extend']]
 lines=['继续提高日策略全期年化：117个完整比较结果','',f'生成UTC：{datetime.now(timezone.utc).isoformat()}。',
 '本轮96次探索结果（27组持仓/环境×2费用、13组模型周期×2费用、8组资金分配×2费用）以及21次后续验证，共117次；含重复控制、不同费用、不同起点与历史截断，不是117个独立策略。',
 '主区间2019-01-02—2026-10-06；开发期2019—2023；最近252交易日2025-10-06—2026-10-06；剔除最近窗口截至2025-10-03。初始10万美元、账户净值复利、无杠杆、现金不计息、未计个人税。',
 '历史市值和股票池认证限制与上轮相同：使用当时指数并集内可用的过去市值快照Top32＋17只无杠杆ETF；非已认证的精确每日Top32，历史发布时点/修订未知。performance_publishable=false。',
 '所有年份已多次研究，本轮不产生新的独立样本外证据。后续模型和资金分配假设是在看过前批结果后声明的，不能合并称为事先一次确定的试验。',
 '', '结论',
 f"没有发现足以替换上轮候选的明显年化提升。市场压力过滤：{pct(control['full']['cagr'])}→{pct(crash['full']['cagr'])}，仅增加{(crash['full']['cagr']-control['full']['cagr'])*100:.3f}个百分点。",
 f"本轮最高主样本年化为强势持仓延长至60日：{pct(extend['full']['cagr'])}；最大回撤由{pct(control['full']['max_drawdown'])}扩大为{pct(extend['full']['max_drawdown'])}，没有通过本轮35%回撤上限。最近一年从{pct(control['last252']['return_'])}降至{pct(extend['last252']['return_'])}。",
 '固定45日按开发期标准胜出，但全期年化只有25.72%、最近一年1.97%；十日预测小权重按模型组开发期标准胜出，但全期20.34%、最近一年−3.16%。开发期胜出没有转化为后来更好，完整结果保留。',
 '', '主要方案（双边成本0.10%）','方案 | 全期年化 | 最大回撤 | 剔除最近252日年化 | 开发期年化 | 最近252日 | 平均仓位']
 for key,path in KEYS.items():
  v=registry[path];lines.append(LABELS[key]+' | '+' | '.join(pct(x) for x in [v['full']['cagr'],v['full']['max_drawdown'],v['ex_recent']['cagr'],v['development']['cagr'],v['last252']['return_'],v['mean_exposure']]))
 lines+=['','本轮测试了什么',
 '1. 市场环境：SPY/QQQ站上200日均线才开仓、市场广度、连续跌破均线清仓、下跌且波动突增暂停开仓。简单大盘均线门槛大多损害收益。',
 '2. 持仓管理：固定45/60日；只有盈利超过1入场ATR、收盘高于20日均线且动量分数≥0.60才允许延长到45/60日；弱势缩短到15日；盈利后移动退出、回本退出；减少替换及保留单只等。',
 '3. 模型和历史反弹：预测未来10/20日收益、同步重建反弹均值及路径因子、树残差修正、小权重和多周期平均。标签必须经过11/21交易日才能使用，训练与启用检验都遵守这个延迟。没有修改旧模型缓存。',
 '4. 资金分配：换股时差额调至等权、每周/每月额外调整、偏离目标10个百分点再平衡、动量倾斜、波动率倒数、保留10%现金。仓位最大100%。均未胜过旧控制；调仓增加成本并改变收益路径。',
 '四组因子仍各25%，没有添加杠杆ETF或新股票。个别资金分配实验改变股票间权重；差额加仓保留原入场日期、更新平均成本及ATR，因此后续止损基准也会变化，不能假装只多赚了闲置现金收益。部分卖出须按完整持仓批次解释胜率，不与逐笔碎片胜率混用。',
 '', '两个微小增益版本的明确规则',
 '压力过滤：在信号日收盘，若SPY过去63交易日收益<0，且20日收益波动率>1.5×126日波动率，则该日不产生任何新买入。原持仓仍按既有退出；原最少两只目标规则保留，因此填不满两只时也可能退出。其余沿用30日、2.5ATR止损、无固定止盈的旧候选。',
 '这项过滤只改变了2022年16个交易计划日的买卖证券/原因，日期见CRASH_PLAN_DIFFERENCES.json；2024从现金重启后与原控制结果相同。它没有解决最近高收益集中于少数大涨日的问题。',
 '强势延长：每个信号日重新判断，若收盘≥入场价+1入场ATR、收盘>SMA20且动量组分数≥0.60，持仓上限为60日；否则恢复30日。硬止损和排名退出仍有效。所有卖出在次日开盘代理成交，不承诺跳空止损价。',
 '', '费用压力','方案 | 双边费用 | 全期年化 | 最大回撤 | 最近252日']
 for name,label in [('control25','control'),('crash_entry','crash'),('strong_cap60','extend')]:
  for cost in [10,30,50]:
   file=f'{name}_cost{cost}.json' if cost!=50 else f'validation/{name}_cost50.json';v=registry[file]
   lines.append(f"{LABELS[label]} | {cost/100:.2f}% | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])} | {pct(v['last252']['return_'])}")
 lines+=['','现金重新起算：不继承起点以前持仓；可用已成熟模型，仍是已经研究过的历史','起点 | 方案 | 费用 | 起点后年化 | 最大回撤']
 for start in ['2022','2024']:
  for name,label in [('control25','control'),('crash_entry','crash'),('strong_cap60','extend')]:
   for cost in [10,30]:
    v=registry[f'validation/fresh{start}_{name}_cost{cost}.json'];lines.append(f"{v['start']} | {LABELS[label]} | {cost/100:.2f}% | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])}")
 lines+=['','压力过滤邻近阈值：不据结果重新择优','短/长波动率倍数 | 全期年化（0.10%） | 最大回撤 | 开发期年化']
 for ratio,file in [('1.25','validation/crash_sigma125_cost10.json'),('1.50','crash_entry_cost10.json'),('1.75','validation/crash_sigma175_cost10.json')]:
  v=registry[file];lines.append(f"{ratio} | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])} | {pct(v['development']['cagr'])}")
 lines+=['','逐年收益；2026仅截至10月6日','年份 | 原控制 | 压力过滤 | 强势延长60日']
 for y in range(2019,2027):lines.append(str(y)+' | '+' | '.join(pct(annual[key][str(y)]) for key in ['control','crash','extend']))
 lines+=['','最近252日集中度；置零最佳日仅作事后敏感性，不是可交易组合','方案 | 实际 | 最佳五日置零 | 最佳十日置零']
 for key in ['control','crash','extend']:
  v=concentration[key];lines.append(LABELS[key]+' | '+' | '.join(pct(v[k]) for k in ['actual','best5_zero','best10_zero']))
 lines+=['','描述性不确定性：20日分块、2000次，未校正多轮筛选，不是样本外置信保证']
 for key,v in uncertainty.items():lines.append(f"{key}: 年化增长比减一{pct(v['annualized_growth_ratio_minus1'])}，95%描述区间[{pct(v['lower95'])},{pct(v['upper95'])}]。不是CAGR百分点差。")
 lines+=['','全部试验：成本、起点与截断不同，最后一栏为各次运行自己的最后252日；prefix2023不是2026的最近一年','文件 | 开始 | 结束 | 全期年化 | 最大回撤 | 开发期年化 | 最后252日']
 for file,v in registry.items():lines.append(f"{file} | {v['start']} | {v['end']} | {pct(v['full']['cagr'])} | {pct(v['full']['max_drawdown'])} | {pct(v['development']['cagr']) if v['development'] else '—'} | {pct(v['last252']['return_'])}")
 lines+=['','验证',
 '21项本目录测试通过（17项新增、4项复用）：环境/标签/特征因果性、11/21日标签隔离、硬止损、延长条件、前一收盘决定、仓位上限、风险预算、差额成交、拆股分红只记一次、会计价格单位。',
 '旧2ATR和2.5ATR控制在两档费用下逐笔订单和净值复现；所有117个输出状态完整，汇总重新计算一致；现金非负、最多三只、总风险敞口≤100%；2019—2023截断运行与长回测相同前缀订单和净值一致。',
 '37项既有研究输入/代码/缓存与六账户及月观察账户账本、配置哈希未变。未更换或新增观察账户，没有实盘单。',
 '文件：CONFIG.json及各子目录CONFIG；SELECTIONS.json；TRIAL_REGISTRY.json；ANNUAL_RETURNS.json；CONCENTRATION.json；UNCERTAINTY.json；FINAL_VALIDATION.json；各版本JSON含逐笔交易、计划和净值。',
 '', '参考研究（仅支持研究方向，不验证本项目收益）',
 'Volatility Managed Portfolios：https://www.nber.org/papers/w22208 ——动机为检查高波动时期的风险分配；这里没有复制论文因子或采用杠杆。',
 'Momentum Crashes：https://www.aqr.com/Insights/Research/Journal-Article/Momentum-Crashes ——参考动量的环境依赖；其多空动量机制与本策略仅做多不完全相同。',
 '', '本轮建议：保留上一轮30日/2.5ATR研究候选，不因0.06—0.22个百分点的事后年化差异推广新规则。继续重复筛选同一段历史不会自动提升未来证据质量；数据认证和未来观察仍未完成。']
 (P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 final=dict(complete=True,completed_runs=117,tests_passed=21,new_tests=17,phase_checks=phase_checks,controls_reproduced=True,prefix_invariant=True,
  protected_files=len(protected),protected_unchanged=True,observations_unchanged=True,data_certified=False,performance_publishable=False,
  selected_replacement=None,result_sha256=hashes,code_sha256={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in P.glob('*.py')})
 assert len(protected)==37
 (P/'FINAL_VALIDATION.json').write_text(json.dumps(final,indent=2)+'\n');print('VALIDATED',len(registry),'runs,',len(protected),'protected files');print(json.dumps(uncertainty,indent=2))
if __name__=='__main__':main()
